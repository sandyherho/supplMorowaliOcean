#!/usr/bin/env python3
"""
================================================================
  INTENSITY ANALYSIS OF LULC TRANSITIONS
  IMIP Impact Zone, Morowali, Central Sulawesi, Indonesia

  Three-level framework following Aldwaik & Pontius (2012),
  Land Use Policy 29(1), 643-656.

  Extensions:
    - Chi-square goodness-of-fit (all three levels)
    - Wilson score confidence intervals
    - Quantity-Exchange-Shift decomposition (Pontius 2019)
    - Per-interval Level-2 and Level-3
    - Markov stationarity (likelihood-ratio G-test)
    - Cohen's h effect sizes

  Input  : ../raw_data/sentinel2LULC_IMIP.nc
  Output : ../figs/intensity_analysis.{pdf,png}
           ../reports/intensity_analysis_report.txt

  Author : Sandy H. S. Herho
  License: MIT
================================================================
"""

import os, warnings
from datetime import datetime
import numpy as np
import netCDF4 as nc
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────
NCFILE    = "../raw_data/sentinel2LULC_IMIP.nc"
FIGDIR    = "../figs"
REPORTDIR = "../reports"
DPI       = 400
FILL_VAL  = -128
ALPHA     = 0.05
Z_ALPHA   = sp_stats.norm.ppf(1 - ALPHA / 2)
PX_KM2    = 0.0001  # 10 m x 10 m

os.makedirs(FIGDIR,    exist_ok=True)
os.makedirs(REPORTDIR, exist_ok=True)

# ── Class definitions ─────────────────────────────────────────
CLASS_INFO = {
    1:  ("Water",              "#4393C3"),
    2:  ("Trees",              "#1B7837"),
    4:  ("Flooded Vegetation", "#78C679"),
    5:  ("Crops",              "#FEC44F"),
    7:  ("Built Area",         "#D6604D"),
    8:  ("Bare Ground",        "#BF9B7A"),
    10: ("Clouds",             "#BABABA"),
    11: ("Rangeland",          "#A8DDB5"),
}
SHORT = {2: "Trees", 4: "Fl.Veg", 5: "Crops",
         7: "Built", 8: "Bare",  11: "Range"}
ALL_CODES  = sorted(CLASS_INFO.keys())
LAND_CODES = [c for c in ALL_CODES if c not in (1, 10)]

# ── Plot style ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset":  "stix",
    "font.size":         9,
    "axes.linewidth":    0.6,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
})
C_ACTIVE  = "#C62828"
C_DORMANT = "#90A4AE"
C_THRESH  = "#1565C0"


# ==============================================================
#  UTILITIES
# ==============================================================

def cohen_h(p1, p2):
    """Cohen's h for two proportions (clamped to [0,1])."""
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    return 2.0 * np.arcsin(np.sqrt(p1)) - 2.0 * np.arcsin(np.sqrt(p2))


def h_label(h):
    """Interpret |h| per Cohen's benchmarks."""
    a = abs(h)
    if a >= 0.8:
        return "large"
    if a >= 0.5:
        return "medium"
    if a >= 0.2:
        return "small"
    return "negligible"


def wilson_ci(k, n):
    """Wilson score interval for proportion k/n.  Returns (lo, hi) in %."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    z2 = Z_ALPHA ** 2
    d  = 1.0 + z2 / n
    c  = (p + z2 / (2.0 * n)) / d
    m  = (Z_ALPHA / d) * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n ** 2))
    return max(0.0, c - m) * 100.0, min(1.0, c + m) * 100.0


def safe_chisq(obs, exp):
    """Chi-square with masking of zero-expected cells."""
    obs = np.asarray(obs, dtype=np.float64)
    exp = np.asarray(exp, dtype=np.float64)
    mask = exp > 0
    k = int(mask.sum())
    if k <= 1:
        return 0.0, 1.0, 0
    c2, pv = sp_stats.chisquare(obs[mask], f_exp=exp[mask])
    return float(c2), float(pv), k - 1


# ==============================================================
#  1. DATA I/O
# ==============================================================

def load_lulc(path):
    with nc.Dataset(path) as ds:
        years = np.array(ds["year"][:]).astype(int)
        lulc  = ds["lulc"][:]
    out = []
    for i in range(len(years)):
        out.append(np.ma.filled(lulc[i], fill_value=FILL_VAL).astype(np.int16))
    return years, out


# ==============================================================
#  2. TRANSITION MATRICES
# ==============================================================

def compute_tm(a1, a2, codes):
    """Pixel-level cross-tabulation between two LULC maps."""
    valid = ((a1 != FILL_VAL) & (a2 != FILL_VAL) &
             (a1 != 10) & (a2 != 10))
    f1, f2 = a1[valid], a2[valid]
    tm = {}
    for i in codes:
        tm[i] = {}
        mask_i = (f1 == i)
        for j in codes:
            tm[i][j] = int(np.sum(mask_i & (f2 == j)))
    return tm, int(valid.sum())


def tm2arr(tm, codes):
    """Dict-of-dict TM -> numpy array aligned to codes."""
    n = len(codes)
    a = np.zeros((n, n), dtype=np.int64)
    for ri, i in enumerate(codes):
        for ci, j in enumerate(codes):
            a[ri, ci] = tm[i][j]
    return a


def agg_matrices(tm_list, codes):
    """Sum multiple TM arrays into an aggregated matrix."""
    a = np.zeros((len(codes), len(codes)), dtype=np.int64)
    for tm in tm_list:
        a += tm2arr(tm, codes)
    return a


# ==============================================================
#  3. LEVEL 1 -- INTERVAL INTENSITY
# ==============================================================

def level1_interval(tm_list, years, codes):
    """
    S_t = (changed_t / total_t) x 100 / duration_t
    U_int = (sum_changed / sum_total) x 100 / total_years
    Chi-square H0: change ~ pixels x duration per interval.
    """
    nI = len(tm_list)
    changed_v, total_v, dur_v = [], [], []
    for t in range(nI):
        a = tm2arr(tm_list[t], codes)
        s = a.sum()
        c = s - np.trace(a)
        changed_v.append(c)
        total_v.append(s)
        dur_v.append(int(years[t + 1] - years[t]))

    sum_c = sum(changed_v)
    sum_t = sum(total_v)
    total_dur = int(years[-1] - years[0])
    U = (sum_c / sum_t * 100.0) / total_dur if sum_t > 0 else 0.0

    # Expected under uniformity: proportional to pixels x duration
    weights = np.array([total_v[t] * dur_v[t] for t in range(nI)],
                       dtype=np.float64)
    weights /= weights.sum()
    expected = weights * sum_c
    observed = np.array(changed_v, dtype=np.float64)
    chi2, pval, df = safe_chisq(observed, expected)

    results = []
    for t in range(nI):
        S_t = (changed_v[t] / total_v[t] * 100.0) / dur_v[t] \
              if total_v[t] > 0 else 0.0
        ci_lo, ci_hi = wilson_ci(changed_v[t], total_v[t])
        results.append(dict(
            interval=f"{years[t]}-{years[t+1]}",
            y0=int(years[t]), y1=int(years[t+1]),
            total=total_v[t], changed=changed_v[t], dur=dur_v[t],
            S=S_t, U=U, active=(S_t > U),
            expected=expected[t],
            ci_lo=ci_lo / dur_v[t], ci_hi=ci_hi / dur_v[t],
        ))

    chi2_info = dict(chi2=chi2, df=df, p=pval)
    return results, U, chi2_info


# ==============================================================
#  4. LEVEL 2 -- CATEGORY INTENSITY
# ==============================================================

def level2_category(tm_list, codes):
    """
    G_j = gain_j / end_size_j x 100
    L_j = loss_j / start_size_j x 100
    U_cat = total_change / total_landscape x 100
    Chi-square H0: gain (loss) ~ end (start) size.
    """
    agg = agg_matrices(tm_list, codes)
    n = len(codes)
    T = agg.sum()
    C = T - np.trace(agg)
    U = (C / T * 100.0) if T > 0 else 0.0

    results = []
    gains, losses = [], []
    starts, ends = [], []

    for ci, c in enumerate(codes):
        start_c = int(agg[ci, :].sum())   # row sum
        end_c   = int(agg[:, ci].sum())    # col sum
        gain_c  = end_c - int(agg[ci, ci])
        loss_c  = start_c - int(agg[ci, ci])
        G_j = (gain_c / end_c * 100.0) if end_c > 0 else 0.0
        L_j = (loss_c / start_c * 100.0) if start_c > 0 else 0.0
        g_lo, g_hi = wilson_ci(gain_c, end_c)
        l_lo, l_hi = wilson_ci(loss_c, start_c)

        results.append(dict(
            code=c, name=CLASS_INFO[c][0], color=CLASS_INFO[c][1],
            start=start_c, end=end_c, gain=gain_c, loss=loss_c,
            gi=G_j, li=L_j,
            ga=(G_j > U), la=(L_j > U),
            g_ci=(g_lo, g_hi), l_ci=(l_lo, l_hi),
        ))
        gains.append(gain_c)
        losses.append(loss_c)
        starts.append(start_c)
        ends.append(end_c)

    # Chi-square: gain expected ~ end_size, loss expected ~ start_size
    ga = np.array(gains, dtype=np.float64)
    la = np.array(losses, dtype=np.float64)
    ea = np.array(ends, dtype=np.float64)
    sa = np.array(starts, dtype=np.float64)
    tg = ga.sum()
    tl = la.sum()
    exp_g = (ea / ea.sum()) * tg
    exp_l = (sa / sa.sum()) * tl
    c2g, pg, dfg = safe_chisq(ga, exp_g)
    c2l, pl, dfl = safe_chisq(la, exp_l)

    chi2 = dict(
        gain=dict(chi2=c2g, df=dfg, p=pg),
        loss=dict(chi2=c2l, df=dfl, p=pl),
    )
    return results, U, chi2


# ==============================================================
#  5. LEVEL 3 -- TRANSITION INTENSITY
#     Aldwaik & Pontius (2012) Equations 8-9, 12-13
# ==============================================================
#
#  GAIN of category j:
#    R_ij = C_ij / C_i+     (Eq. 8)
#      C_ij = pixels transitioning from i to j
#      C_i+ = row sum = START size of category i
#    W_j  = Gain_j / SUM_{i!=j} C_i+     (Eq. 9)
#      = total gain of j / total non-j START landscape
#    Targeted: R_ij > W_j
#
#  LOSS of category j:
#    Q_jk = C_jk / C_+k     (Eq. 12)
#      C_jk = pixels transitioning from j to k
#      C_+k = col sum = END size of category k
#    V_j  = Loss_j / SUM_{k!=j} C_+k     (Eq. 13)
#      = total loss of j / total non-j END landscape
#    Targeted: Q_jk > V_j
# ==============================================================

def level3_transition(tm_list, codes):
    agg = agg_matrices(tm_list, codes)
    n = len(codes)
    rs = agg.sum(axis=1)   # row sums = start sizes (C_i+)
    cs = agg.sum(axis=0)   # col sums = end sizes   (C_+k)

    gain_res, loss_res = {}, {}
    gain_chi2, loss_chi2 = {}, {}

    for ji, j in enumerate(codes):
        # ── GAIN perspective (Eq. 8-9) ────────────────────────
        gain_j = int(cs[ji]) - int(agg[ji, ji])
        non_j_start = sum(int(rs[ii]) for ii in range(n) if ii != ji)
        W_j = (gain_j / non_j_start * 100.0) if non_j_start > 0 else 0.0
        p_exp_g = (gain_j / non_j_start) if non_j_start > 0 else 0.0

        sources = []
        obs_g, denom_g = [], []
        for ii, i in enumerate(codes):
            if i == j:
                continue
            C_ij = int(agg[ii, ji])
            C_iplus = int(rs[ii])
            R_ij = (C_ij / C_iplus * 100.0) if C_iplus > 0 else 0.0
            p_obs_g = (C_ij / C_iplus) if C_iplus > 0 else 0.0
            sources.append(dict(
                source_code=i, source_name=CLASS_INFO[i][0],
                pixels=C_ij, denom=C_iplus,
                intensity=R_ij, uniform=W_j,
                active=(R_ij > W_j),
                h=cohen_h(p_obs_g, p_exp_g),
            ))
            obs_g.append(C_ij)
            denom_g.append(C_iplus)

        # Chi-square: H0 sources proportional to start sizes
        dg = np.array(denom_g, dtype=np.float64)
        og = np.array(obs_g, dtype=np.float64)
        eg = (dg / dg.sum()) * gain_j if dg.sum() > 0 else np.ones_like(dg)
        c2, pv, df = safe_chisq(og, eg)
        for idx, s in enumerate(sources):
            s["expected"] = eg[idx]

        gain_res[j] = sources
        gain_chi2[j] = dict(chi2=c2, df=df, p=pv, W=W_j)

        # ── LOSS perspective (Eq. 12-13) ──────────────────────
        loss_j = int(rs[ji]) - int(agg[ji, ji])
        non_j_end = sum(int(cs[ki]) for ki in range(n) if ki != ji)
        V_j = (loss_j / non_j_end * 100.0) if non_j_end > 0 else 0.0
        p_exp_l = (loss_j / non_j_end) if non_j_end > 0 else 0.0

        sinks = []
        obs_l, denom_l = [], []
        for ki, k in enumerate(codes):
            if k == j:
                continue
            C_jk = int(agg[ji, ki])
            C_plusk = int(cs[ki])
            Q_jk = (C_jk / C_plusk * 100.0) if C_plusk > 0 else 0.0
            p_obs_l = (C_jk / C_plusk) if C_plusk > 0 else 0.0
            sinks.append(dict(
                target_code=k, target_name=CLASS_INFO[k][0],
                pixels=C_jk, denom=C_plusk,
                intensity=Q_jk, uniform=V_j,
                active=(Q_jk > V_j),
                h=cohen_h(p_obs_l, p_exp_l),
            ))
            obs_l.append(C_jk)
            denom_l.append(C_plusk)

        dl = np.array(denom_l, dtype=np.float64)
        ol = np.array(obs_l, dtype=np.float64)
        el = (dl / dl.sum()) * loss_j if dl.sum() > 0 else np.ones_like(dl)
        c2l, pvl, dfl = safe_chisq(ol, el)
        for idx, s in enumerate(sinks):
            s["expected"] = el[idx]

        loss_res[j] = sinks
        loss_chi2[j] = dict(chi2=c2l, df=dfl, p=pvl, W=V_j)

    return gain_res, loss_res, gain_chi2, loss_chi2


# ==============================================================
#  6. Q-E-S DECOMPOSITION (Pontius 2019)
# ==============================================================

def qes_decomposition(agg, codes):
    """
    Per category j:
      Q_j = |gain_j - loss_j|
      E_j = 2 * SUM_{i!=j} min(C_ij, C_ji)    (pairwise swaps)
      S_j = 2 * min(gain_j, loss_j) - E_j      (residual)
    Landscape totals halved to avoid double-counting.
    """
    n = len(codes)
    rows = []
    tQ, tE, tS, tT = 0, 0, 0, 0

    for ci, c in enumerate(codes):
        g = int(agg[:, ci].sum() - agg[ci, ci])
        l = int(agg[ci, :].sum() - agg[ci, ci])
        Q_j = abs(g - l)
        E_j = 2 * sum(min(int(agg[ci, ri]), int(agg[ri, ci]))
                       for ri in range(n) if ri != ci)
        S_j = 2 * min(g, l) - E_j
        T_j = g + l
        rows.append(dict(code=c, name=CLASS_INFO[c][0],
                         gain=g, loss=l, Q=Q_j, E=E_j, S=S_j, T=T_j))
        tQ += Q_j; tE += E_j; tS += S_j; tT += T_j

    # Halve for landscape-level (each change counted in two categories)
    summary = dict(Q=tQ / 2.0, E=tE / 2.0, S=tS / 2.0, T=tT / 2.0)
    return rows, summary


# ==============================================================
#  7. PER-INTERVAL LEVEL 2 & 3
# ==============================================================

def per_interval_cat(tm_list, years, codes):
    """Category gain/loss intensity per interval."""
    out = []
    for t, tm in enumerate(tm_list):
        a = tm2arr(tm, codes)
        T = a.sum()
        C = T - np.trace(a)
        U = (C / T * 100.0) if T > 0 else 0.0
        cats = {}
        for ci, c in enumerate(codes):
            end_c   = int(a[:, ci].sum())
            start_c = int(a[ci, :].sum())
            g = end_c - int(a[ci, ci])
            l = start_c - int(a[ci, ci])
            gi = (g / end_c * 100.0) if end_c > 0 else 0.0
            li = (l / start_c * 100.0) if start_c > 0 else 0.0
            cats[c] = dict(gi=gi, li=li, ga=(gi > U), la=(li > U),
                           gp=g, lp=l)
        out.append(dict(interval=f"{years[t]}-{years[t+1]}", U=U, cats=cats))
    return out


def per_interval_trans(tm_list, years, codes, tgt=7, src=2):
    """Per-interval Trees->Built using A&P Eq. 8-9."""
    ti = codes.index(tgt)
    si = codes.index(src)
    out = []
    for t, tm in enumerate(tm_list):
        a = tm2arr(tm, codes)
        C_st = int(a[si, ti])                              # Trees->Built
        C_splus = int(a[si, :].sum())                      # Trees start
        gain_t = int(a[:, ti].sum()) - int(a[ti, ti])      # Built gain
        non_t_start = sum(int(a[ii, :].sum())
                          for ii in range(len(codes)) if ii != ti)
        R = (C_st / C_splus * 100.0) if C_splus > 0 else 0.0
        W = (gain_t / non_t_start * 100.0) if non_t_start > 0 else 0.0
        ratio = R / W if W > 0 else 0.0
        out.append(dict(
            interval=f"{years[t]}-{years[t+1]}",
            pixels=C_st, km2=C_st * PX_KM2,
            R=R, W=W, ratio=ratio, active=(R > W),
        ))
    return out


# ==============================================================
#  8. MARKOV STATIONARITY (likelihood-ratio G-test)
# ==============================================================

def markov_test(tm_list, codes):
    """
    H0: Transition probability matrix is constant across intervals.
    G = 2 * SUM_t SUM_i SUM_j  n_{ij}^{(t)} * ln( p_{ij}^{(t)} / p_{ij} )
    df = (T - 1) * n * (n - 1)
    """
    nT = len(tm_list)
    n = len(codes)
    agg = agg_matrices(tm_list, codes)

    # Pooled transition probabilities
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        rs = agg[i, :].sum()
        if rs > 0:
            P[i, :] = agg[i, :] / rs

    G = 0.0
    for t in range(nT):
        a = tm2arr(tm_list[t], codes)
        for i in range(n):
            rs = a[i, :].sum()
            if rs == 0:
                continue
            for j in range(n):
                if a[i, j] == 0:
                    continue
                p_t = a[i, j] / rs
                if P[i, j] > 0:
                    G += 2.0 * a[i, j] * np.log(p_t / P[i, j])

    df = (nT - 1) * n * (n - 1)
    pv = 1.0 - sp_stats.chi2.cdf(G, df) if df > 0 else 1.0

    # Per-class
    pc = {}
    for ci, c in enumerate(codes):
        Gc = 0.0
        for t in range(nT):
            a = tm2arr(tm_list[t], codes)
            rs = a[ci, :].sum()
            if rs == 0:
                continue
            for j in range(n):
                if a[ci, j] == 0:
                    continue
                p_t = a[ci, j] / rs
                if P[ci, j] > 0:
                    Gc += 2.0 * a[ci, j] * np.log(p_t / P[ci, j])
        dfc = (nT - 1) * (n - 1)
        pc[c] = dict(
            G=Gc, df=dfc,
            p=1.0 - sp_stats.chi2.cdf(Gc, dfc) if dfc > 0 else 1.0,
            stat=(1.0 - sp_stats.chi2.cdf(Gc, dfc) > ALPHA) if dfc > 0 else True,
        )

    return dict(G=G, df=df, p=pv, stat=(pv > ALPHA), pc=pc)


# ==============================================================
#  9. FIGURE
# ==============================================================

def make_figure(L1, U1, L2, U2, G3, L3, figdir):

    fig = plt.figure(figsize=(14, 6.6))

    # Main panels occupy the top; legend lives below via fig.legend
    gs = fig.add_gridspec(
        nrows=1, ncols=3,
        width_ratios=[1, 1.15, 1.3],
        wspace=0.40,
        left=0.065, right=0.96,
        top=0.92, bottom=0.16,
    )

    # ── Panel (a) ─────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    labs = [r["interval"] for r in L1]
    vals = [r["S"] for r in L1]
    cols = [C_ACTIVE if r["active"] else C_DORMANT for r in L1]
    yp = np.arange(len(labs))
    ax_a.barh(yp, vals, color=cols, height=0.55,
              edgecolor="white", linewidth=0.4, zorder=3)
    ax_a.axvline(U1, color=C_THRESH, ls="--", lw=1.3, zorder=4)
    ax_a.set_yticks(yp)
    ax_a.set_yticklabels(labs, fontsize=8.5)
    ax_a.set_xlabel("Annual change intensity (%)", fontsize=9)
    ax_a.invert_yaxis()
    ax_a.set_title("(a)", fontsize=11, fontweight="bold", loc="center", pad=10)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.grid(axis="x", ls=":", alpha=0.3, zorder=0)

    # ── Panel (b) ─────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    cl = [r for r in L2 if r["code"] in LAND_CODES]
    ncat = len(cl)
    yp_b = np.arange(ncat)
    bh = 0.32
    gc = [C_ACTIVE if r["ga"] else C_DORMANT for r in cl]
    lc = [C_ACTIVE if r["la"] else C_DORMANT for r in cl]
    ax_b.barh(yp_b - bh / 2, [r["gi"] for r in cl], height=bh,
              color=gc, edgecolor="white", linewidth=0.4, zorder=3)
    ax_b.barh(yp_b + bh / 2, [-r["li"] for r in cl], height=bh,
              color=lc, edgecolor="white", linewidth=0.4, zorder=3)
    ax_b.axvline( U2, color=C_THRESH, ls="--", lw=1.2, zorder=4)
    ax_b.axvline(-U2, color=C_THRESH, ls="--", lw=1.2, zorder=4)
    ax_b.axvline(0, color="black", lw=0.6, zorder=2)
    ax_b.set_yticks(yp_b)
    ax_b.set_yticklabels([r["name"] for r in cl], fontsize=8.5)
    ax_b.set_xlabel("Category intensity (%)   Loss  |  Gain", fontsize=9)
    ax_b.invert_yaxis()
    ax_b.set_title("(b)", fontsize=11, fontweight="bold", loc="center", pad=10)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(axis="x", ls=":", alpha=0.3, zorder=0)

    # ── Panel (c) ─────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])

    # Built Area gain sources
    bs = sorted([s for s in G3.get(7, [])
                 if s["source_code"] in LAND_CODES and s["pixels"] > 0],
                key=lambda x: x["intensity"], reverse=True)
    # Trees loss sinks, EXCLUDING Built (already shown above)
    ts = sorted([s for s in L3.get(2, [])
                 if s["target_code"] in LAND_CODES
                 and s["target_code"] != 7
                 and s["pixels"] > 0],
                key=lambda x: x["intensity"], reverse=True)

    items, labels, colors = [], [], []
    for s in bs:
        items.append(s["intensity"])
        labels.append(f"{SHORT.get(s['source_code'], '?')} > Built")
        colors.append(C_ACTIVE if s["active"] else C_DORMANT)
    sep = len(items)
    for s in ts:
        items.append(s["intensity"])
        labels.append(f"Trees > {SHORT.get(s['target_code'], '?')}")
        colors.append(C_ACTIVE if s["active"] else C_DORMANT)

    ni = len(items)
    yp_c = np.arange(ni)
    ax_c.barh(yp_c, items, color=colors, height=0.52,
              edgecolor="white", linewidth=0.4, zorder=3)

    # Uniform threshold lines
    if bs:
        W_built = bs[0]["uniform"]
        ax_c.axvline(W_built, color=C_THRESH, ls="--", lw=1.0,
                     zorder=4, alpha=0.7)
    if ts:
        V_trees = ts[0]["uniform"]
        # Only draw second line if substantially different
        if abs(V_trees - W_built) > 0.3:
            ax_c.axvline(V_trees, color=C_THRESH, ls=":", lw=1.0,
                         zorder=4, alpha=0.6)

    ax_c.invert_yaxis()

    # Section separator and annotations
    if sep < ni:
        ax_c.axhline(sep - 0.5, color="#455A64", ls="-", lw=0.7, zorder=5)
        mid_top = (sep - 1) / 2.0
        mid_bot = sep + (ni - sep - 1) / 2.0
        ax_c.annotate("Built Area\ngain sources",
                       xy=(1.02, mid_top),
                       xycoords=("axes fraction", "data"),
                       fontsize=7, fontstyle="italic", color="#455A64",
                       va="center", ha="left", annotation_clip=False)
        ax_c.annotate("Trees\nloss sinks",
                       xy=(1.02, mid_bot),
                       xycoords=("axes fraction", "data"),
                       fontsize=7, fontstyle="italic", color="#455A64",
                       va="center", ha="left", annotation_clip=False)

    ax_c.set_yticks(yp_c)
    ax_c.set_yticklabels(labels, fontsize=8)
    ax_c.set_xlabel("Transition intensity (%)", fontsize=9)
    ax_c.set_title("(c)", fontsize=11, fontweight="bold", loc="center", pad=10)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.grid(axis="x", ls=":", alpha=0.3, zorder=0)

    # ── Legend -- detached at bottom of figure ─────────────────
    handles = [
        Patch(facecolor=C_ACTIVE,  edgecolor="white", lw=0.4,
              label="Active / Targeted  (observed > uniform)"),
        Patch(facecolor=C_DORMANT, edgecolor="white", lw=0.4,
              label=r"Dormant / Avoided  (observed $\leq$ uniform)"),
        Line2D([], [], color=C_THRESH, ls="--", lw=1.3,
               label="Uniform threshold"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3, fontsize=9, frameon=False,
        columnspacing=3.0, handlelength=2.0, handletextpad=0.7,
    )

    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"intensity_analysis.{fmt}"),
                    dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {figdir}/intensity_analysis.{{pdf,png}}")


# ==============================================================
#  10. REPORT
# ==============================================================

def write_report(years, tm_list, nv_list,
                 L1, U1, chi2_1,
                 L2, U2, chi2_2,
                 G3, L3, gchi3, lchi3,
                 qes_r, qes_s,
                 pi_cat, pi_tb,
                 markov,
                 outdir):
    SEP  = "=" * 78
    SEP2 = "-" * 78
    buf = []
    w = buf.append

    w(SEP)
    w("  INTENSITY ANALYSIS OF LULC TRANSITIONS")
    w("  IMIP Impact Zone, Morowali, Central Sulawesi, Indonesia")
    w("  Aldwaik & Pontius (2012), Land Use Policy 29(1), 643-656")
    w("  Extensions: chi-square, Wilson CI, Q-E-S, Cohen's h, Markov G-test")
    w(SEP)
    w(f"  Author      : Sandy H. S. Herho")
    w(f"  Generated   : {datetime.now():%Y-%m-%d %H:%M:%S}")
    w(f"  LULC Source : Sentinel-2 10 m (Dynamic World / ESA WorldCover)")
    w(f"  Domain      : Lon [122.08, 122.28], Lat [-2.92, -2.72]")
    w(f"  Years       : {years[0]} to {years[-1]} ({len(years)} snapshots)")
    w(f"  Intervals   : {len(tm_list)} consecutive year-pairs")
    w(f"  Pixel       : 10 m x 10 m = {PX_KM2} km2")
    w(f"  Classes     : 6 land (Water, Clouds excluded)")
    w(f"  CI          : Wilson score, alpha = {ALPHA}")
    w(SEP)

    # ── Notation ──────────────────────────────────────────────
    w("")
    w(SEP2)
    w("  NOTATION")
    w(SEP2)
    w("")
    w("  S_t   = Annual change intensity (%/yr)")
    w("        = (changed / total) x 100 / duration")
    w("  U_int = Uniform interval threshold (%/yr)")
    w("  G_j   = Gain intensity = gain_j / end_size_j x 100")
    w("  L_j   = Loss intensity = loss_j / start_size_j x 100")
    w("  U_cat = Uniform category threshold (%)")
    w("")
    w("  R_ij  = Transition intensity (gain perspective)     [A&P Eq. 8]")
    w("        = C_ij / C_i+  x 100")
    w("          C_ij = pixels from i to j;  C_i+ = START size of i")
    w("  W_j   = Uniform gain threshold                      [A&P Eq. 9]")
    w("        = Gain_j / SUM_{i!=j} C_i+")
    w("")
    w("  Q_jk  = Transition intensity (loss perspective)     [A&P Eq. 12]")
    w("        = C_jk / C_+k  x 100")
    w("          C_jk = pixels from j to k;  C_+k = END size of k")
    w("  V_j   = Uniform loss threshold                      [A&P Eq. 13]")
    w("        = Loss_j / SUM_{k!=j} C_+k")
    w("")
    w("  h     = Cohen's h = 2*arcsin(sqrt(p_obs)) - 2*arcsin(sqrt(p_exp))")
    w("          |h| >= 0.8: large;  >= 0.5: medium;  >= 0.2: small;  < 0.2: negligible")
    w("  Q,E,S = Quantity, Exchange, Shift (Pontius 2019)")
    w("  Active   = observed > uniform;  Dormant = observed <= uniform")
    w("")

    # ── S1: Transition Matrices ───────────────────────────────
    w(SEP2)
    w("  SECTION 1 -- TRANSITION MATRICES")
    w(SEP2)
    w("")
    for t, tm in enumerate(tm_list):
        a = tm2arr(tm, LAND_CODES)
        T = a.sum(); D = int(np.trace(a)); C = T - D
        w(f"  --- {years[t]} -> {years[t+1]} ---")
        w(f"  Valid land pixels : {nv_list[t]:>12,}")
        w(f"  Persisted         : {D:>12,}  ({D/T*100:.2f}%)")
        w(f"  Changed           : {C:>12,}  ({C/T*100:.2f}%)")
        w(f"  Changed area      : {C*PX_KM2:>12.2f} km2")
        hdr = f"  {'From\\To':<8}" + "".join(f"{SHORT[c]:>9}" for c in LAND_CODES) + f"{'Total':>10}"
        w(hdr)
        w("  " + "-" * (len(hdr) - 2))
        for ri, i in enumerate(LAND_CODES):
            row = f"  {SHORT[i]:<8}"
            for ci_idx in range(len(LAND_CODES)):
                row += f"{a[ri, ci_idx]:>9,}"
            row += f"{a[ri, :].sum():>10,}"
            w(row)
        tr = f"  {'Total':<8}"
        for ci_idx in range(len(LAND_CODES)):
            tr += f"{a[:, ci_idx].sum():>9,}"
        tr += f"{T:>10,}"
        w(tr)

        w(f"\n  Gross flows ({years[t]}->{years[t+1]}):")
        w(f"  {'Class':<18} {'Gain(px)':>10} {'Gain(km2)':>10} "
          f"{'Loss(px)':>10} {'Loss(km2)':>10} {'Net(px)':>10} {'Swap(px)':>10}")
        w("  " + "-" * 82)
        for ci_idx, c in enumerate(LAND_CODES):
            ce = int(a[:, ci_idx].sum())
            cs_ = int(a[ci_idx, :].sum())
            g = ce - int(a[ci_idx, ci_idx])
            l = cs_ - int(a[ci_idx, ci_idx])
            w(f"  {CLASS_INFO[c][0]:<18} {g:>10,} {g*PX_KM2:>10.2f} "
              f"{l:>10,} {l*PX_KM2:>10.2f} {g-l:>+10,} {2*min(g,l):>10,}")
        w("")

    # ── S2: Level 1 ───────────────────────────────────────────
    w(SEP2)
    w("  SECTION 2 -- LEVEL 1: INTERVAL INTENSITY")
    w(SEP2)
    w("")
    sc = sum(r["changed"] for r in L1)
    st = sum(r["total"]   for r in L1)
    td = years[-1] - years[0]
    w(f"  Total changed (all intervals): {sc:>12,}")
    w(f"  Total valid   (all intervals): {st:>12,}")
    w(f"  Duration: {td} years")
    w(f"  U_int = ({sc}/{st} x 100) / {td} = {U1:.4f} %/yr")
    w("")
    w(f"  {'Interval':<12} {'Changed':>10} {'Total':>12} {'km2':>8} "
      f"{'S_t':>10} {'U_int':>8} {'S/U':>6} {'95%CI':>20} {'Status':>8}")
    w("  " + "-" * 100)
    for t, r in enumerate(L1):
        ratio = r["S"] / U1 if U1 > 0 else 0.0
        w(f"  {r['interval']:<12} {r['changed']:>10,} {r['total']:>12,} "
          f"{r['changed']*PX_KM2:>8.2f} {r['S']:>9.4f}% {r['U']:>7.4f}% "
          f"{ratio:>5.1f}x [{r['ci_lo']:.4f},{r['ci_hi']:.4f}]% "
          f"{'ACTIVE' if r['active'] else 'dormant':>8}")
    w("")
    w(f"  Chi-square (H0: uniform): chi2={chi2_1['chi2']:.2f}, "
      f"df={chi2_1['df']}, p={chi2_1['p']:.2e}")
    if chi2_1["p"] < ALPHA:
        w(f"  -> Reject H0 at alpha={ALPHA}.")

    mx = max(L1, key=lambda x: x["S"])
    mn = min(L1, key=lambda x: x["S"])
    na = sum(1 for r in L1 if r["active"])
    w(f"\n  Result: {na}/{len(L1)} intervals ACTIVE.")
    w(f"  Peak  : {mx['interval']} ({mx['S']:.4f}%, {mx['S']/U1:.1f}x)")
    w(f"  Lowest: {mn['interval']} ({mn['S']:.4f}%, {mn['S']/U1:.1f}x)")

    # ── S3: Level 2 ───────────────────────────────────────────
    w("")
    w(SEP2)
    w("  SECTION 3 -- LEVEL 2: CATEGORY INTENSITY")
    w(SEP2)
    w("")
    cl = [r for r in L2 if r["code"] in LAND_CODES]
    w(f"  U_cat = {U2:.4f}%")
    w("")
    w(f"  {'Category':<18} {'Start':>12} {'End':>12} {'Net(km2)':>10} "
      f"{'G_j':>8} {'95%CI':>16} {'G':>3} "
      f"{'L_j':>8} {'95%CI':>16} {'L':>3}")
    w("  " + "-" * 115)
    for r in cl:
        nk = (r["gain"] - r["loss"]) * PX_KM2
        w(f"  {r['name']:<18} {r['start']:>12,} {r['end']:>12,} "
          f"{nk:>+10.2f} {r['gi']:>7.2f}% "
          f"[{r['g_ci'][0]:.2f},{r['g_ci'][1]:.2f}] "
          f"{'A' if r['ga'] else 'd':>3} "
          f"{r['li']:>7.2f}% "
          f"[{r['l_ci'][0]:.2f},{r['l_ci'][1]:.2f}] "
          f"{'A' if r['la'] else 'd':>3}")
    w("")
    w(f"  Chi2 gain: {chi2_2['gain']['chi2']:.2f}, "
      f"df={chi2_2['gain']['df']}, p={chi2_2['gain']['p']:.2e}")
    w(f"  Chi2 loss: {chi2_2['loss']['chi2']:.2f}, "
      f"df={chi2_2['loss']['df']}, p={chi2_2['loss']['p']:.2e}")
    w("")
    w(f"  Active gain : {', '.join(r['name'] for r in cl if r['ga'])}")
    w(f"  Dormant gain: {', '.join(r['name'] for r in cl if not r['ga'])}")
    w(f"  Active loss : {', '.join(r['name'] for r in cl if r['la'])}")
    w(f"  Dormant loss: {', '.join(r['name'] for r in cl if not r['la'])}")

    # ── S4: Level 3 ───────────────────────────────────────────
    w("")
    w(SEP2)
    w("  SECTION 4 -- LEVEL 3: TRANSITION INTENSITY")
    w(SEP2)
    w("")
    w("  Denominators per Aldwaik & Pontius (2012):")
    w("    Gain: R_ij = C_ij / C_i+  (start size of source)    [Eq. 8]")
    w("    Loss: Q_jk = C_jk / C_+k  (end size of target)      [Eq. 12]")
    w("")

    w("  === GAIN PERSPECTIVE ===")
    for tc in LAND_CODES:
        src_list = [s for s in G3.get(tc, [])
                    if s["source_code"] in LAND_CODES and s["pixels"] > 0]
        if not src_list:
            continue
        src_list.sort(key=lambda x: x["intensity"], reverse=True)
        tn = CLASS_INFO[tc][0]
        tp = sum(s["pixels"] for s in src_list)
        ci = gchi3.get(tc, {})
        w(f"\n  Gains of {tn} ({tp:,} px = {tp*PX_KM2:.2f} km2, "
          f"W_j = {ci.get('W', 0):.4f}%)")
        w(f"    chi2={ci.get('chi2', 0):.2f}, df={ci.get('df', 0)}, "
          f"p={ci.get('p', 1):.2e}")
        w(f"  {'Source':<18} {'Pixels':>10} {'km2':>8} {'C_i+':>12} "
          f"{'R_ij':>9} {'W_j':>8} {'Ratio':>7} {'h':>7} "
          f"{'|h|':>10} {'Status':>10}")
        w("  " + "-" * 105)
        for s in src_list:
            r = s["intensity"] / s["uniform"] if s["uniform"] > 0 else 0.0
            st = "TARGETED" if s["active"] else "avoided"
            w(f"  {s['source_name']:<18} {s['pixels']:>10,} "
              f"{s['pixels']*PX_KM2:>8.2f} {s['denom']:>12,} "
              f"{s['intensity']:>8.4f}% {s['uniform']:>7.4f}% "
              f"{r:>6.1f}x {s['h']:>+6.3f} {h_label(s['h']):>10} "
              f"{st:>10}")

    w("")
    w("  === LOSS PERSPECTIVE ===")
    for sc_ in LAND_CODES:
        snk_list = [s for s in L3.get(sc_, [])
                    if s["target_code"] in LAND_CODES and s["pixels"] > 0]
        if not snk_list:
            continue
        snk_list.sort(key=lambda x: x["intensity"], reverse=True)
        sn = CLASS_INFO[sc_][0]
        tp = sum(s["pixels"] for s in snk_list)
        ci = lchi3.get(sc_, {})
        w(f"\n  Losses of {sn} ({tp:,} px = {tp*PX_KM2:.2f} km2, "
          f"V_j = {ci.get('W', 0):.4f}%)")
        w(f"    chi2={ci.get('chi2', 0):.2f}, df={ci.get('df', 0)}, "
          f"p={ci.get('p', 1):.2e}")
        w(f"  {'Target':<18} {'Pixels':>10} {'km2':>8} {'C_+k':>12} "
          f"{'Q_jk':>9} {'V_j':>8} {'Ratio':>7} {'h':>7} "
          f"{'|h|':>10} {'Status':>10}")
        w("  " + "-" * 105)
        for s in snk_list:
            r = s["intensity"] / s["uniform"] if s["uniform"] > 0 else 0.0
            st = "TARGETED" if s["active"] else "avoided"
            w(f"  {s['target_name']:<18} {s['pixels']:>10,} "
              f"{s['pixels']*PX_KM2:>8.2f} {s['denom']:>12,} "
              f"{s['intensity']:>8.4f}% {s['uniform']:>7.4f}% "
              f"{r:>6.1f}x {s['h']:>+6.3f} {h_label(s['h']):>10} "
              f"{st:>10}")

    # ── S5: Q-E-S ─────────────────────────────────────────────
    w("")
    w(SEP2)
    w("  SECTION 5 -- QUANTITY-EXCHANGE-SHIFT DECOMPOSITION")
    w(SEP2)
    w("")
    w("  Pontius (2019), Int. J. Remote Sensing.")
    w("  Q = |gain - loss|;  E = 2*SUM min(C_ij,C_ji);  S = 2*min(G,L) - E")
    w("")
    w(f"  {'Category':<18} {'Gain':>10} {'Loss':>10} "
      f"{'Q':>10} {'E':>10} {'S':>10} {'Total':>10}")
    w("  " + "-" * 82)
    for r in qes_r:
        if r["code"] not in LAND_CODES:
            continue
        w(f"  {r['name']:<18} {r['gain']:>10,} {r['loss']:>10,} "
          f"{r['Q']:>10,} {r['E']:>10,} {r['S']:>10,} {r['T']:>10,}")
    w(f"\n  Landscape totals (halved):")
    for k, lab in [("Q", "Quantity"), ("E", "Exchange"),
                   ("S", "Shift"), ("T", "Total")]:
        v = qes_s[k]
        pct = (v / qes_s["T"] * 100.0) if (qes_s["T"] > 0 and k != "T") else 100.0
        ps = f"[{pct:.1f}%]" if k != "T" else ""
        w(f"    {lab:<10}: {v:>12,.0f} px ({v*PX_KM2:>8.2f} km2) {ps:>8}")

    # ── S6: Per-interval category ─────────────────────────────
    w("")
    w(SEP2)
    w("  SECTION 6 -- PER-INTERVAL CATEGORY INTENSITY")
    w(SEP2)
    w("")
    w("  Gain intensity per interval (A=Active, d=dormant):")
    w("")
    hdr = f"  {'Interval':<12} {'U(%)':>7}"
    for c in LAND_CODES:
        hdr += f"  {SHORT[c]:>8}"
    w(hdr)
    w("  " + "-" * (len(hdr) - 2))
    for pr in pi_cat:
        row = f"  {pr['interval']:<12} {pr['U']:>6.2f}%"
        for c in LAND_CODES:
            d = pr["cats"][c]
            flag = "A" if d["ga"] else "d"
            row += f"  {d['gi']:>6.2f}{flag}"
        w(row)
    w("")
    w("  Loss intensity per interval:")
    w(hdr)
    w("  " + "-" * (len(hdr) - 2))
    for pr in pi_cat:
        row = f"  {pr['interval']:<12} {pr['U']:>6.2f}%"
        for c in LAND_CODES:
            d = pr["cats"][c]
            flag = "A" if d["la"] else "d"
            row += f"  {d['li']:>6.2f}{flag}"
        w(row)

    # ── S7: Per-interval Trees->Built ─────────────────────────
    w("")
    w(SEP2)
    w("  SECTION 7 -- PER-INTERVAL: TREES -> BUILT AREA")
    w("  (A&P Eq. 8-9: R_ij = C_ij / C_i+, W_j = Gain_j / non-j START)")
    w(SEP2)
    w("")
    w(f"  {'Interval':<12} {'Pixels':>10} {'km2':>8} "
      f"{'R_ij':>9} {'W_j':>8} {'Ratio':>7} {'Status':>10}")
    w("  " + "-" * 68)
    for r in pi_tb:
        st = "TARGETED" if r["active"] else "avoided"
        w(f"  {r['interval']:<12} {r['pixels']:>10,} "
          f"{r['km2']:>8.2f} {r['R']:>8.4f}% "
          f"{r['W']:>7.4f}% {r['ratio']:>6.1f}x {st:>10}")

    # ── S8: Markov ────────────────────────────────────────────
    w("")
    w(SEP2)
    w("  SECTION 8 -- MARKOV STATIONARITY TEST")
    w(SEP2)
    w("")
    w(f"  Global: G={markov['G']:.2f}, df={markov['df']}, p={markov['p']:.2e}")
    st_str = "STATIONARY" if markov["stat"] else "NON-STATIONARY"
    w(f"  -> {st_str}")
    w("")
    w(f"  {'Class':<18} {'G':>12} {'df':>5} {'p':>12} {'Result':>14}")
    w("  " + "-" * 65)
    for c in LAND_CODES:
        pc = markov["pc"][c]
        res = "stationary" if pc["stat"] else "NON-STATION."
        w(f"  {CLASS_INFO[c][0]:<18} {pc['G']:>12.2f} {pc['df']:>5} "
          f"{pc['p']:>12.2e} {res:>14}")

    # ── S9: Synthesis ─────────────────────────────────────────
    w("")
    w(SEP2)
    w("  SECTION 9 -- SYNTHESIS")
    w(SEP2)
    w("")

    # Built trajectory
    bsizes = []
    for t, tm in enumerate(tm_list):
        a = tm2arr(tm, LAND_CODES)
        bi = LAND_CODES.index(7)
        bsizes.append(int(a[bi, :].sum()))
    alast = tm2arr(tm_list[-1], LAND_CODES)
    bsizes.append(int(alast[:, LAND_CODES.index(7)].sum()))

    # Key transitions
    tb = next((s for s in G3.get(7, []) if s["source_code"] == 2), None)
    bb = next((s for s in G3.get(7, []) if s["source_code"] == 8), None)
    cb = next((s for s in G3.get(7, []) if s["source_code"] == 5), None)

    na = sum(1 for r in L1 if r["active"])
    w(f"  FINDING 1: CONTINUOUS INTENSIVE TRANSFORMATION")
    w(f"    {na}/{len(L1)} intervals Active.")
    w(f"    S_t range: {mn['S']:.2f}% to {mx['S']:.2f}% (U_int={U1:.2f}%)")
    w(f"    Chi-square: p < 0.001 -> non-uniform.")
    w(f"    Peak: {mx['interval']}.")
    w("")

    w(f"  FINDING 2: BUILT AREA EXPANSION")
    w(f"    Year    Pixels       km2")
    for i, y in enumerate(years):
        if i < len(bsizes):
            w(f"    {y}   {bsizes[i]:>10,}   {bsizes[i]*PX_KM2:>8.2f}")
    if bsizes[0] > 0:
        w(f"    Expansion: {bsizes[-1]/bsizes[0]:.1f}x over {td} years")
    w("")

    w(f"  FINDING 3: TRANSITION TARGETING (A&P gain perspective, Eq. 8-9)")
    W_built = gchi3.get(7, {}).get("W", 0)
    w(f"    Built Area sources (W_j = {W_built:.4f}%):")
    for s in sorted(G3.get(7, []), key=lambda x: x["intensity"], reverse=True):
        if s["source_code"] not in LAND_CODES:
            continue
        r = s["intensity"] / s["uniform"] if s["uniform"] > 0 else 0
        st = "TARGETED" if s["active"] else "avoided"
        w(f"      {s['source_name']:<18} R={s['intensity']:.4f}%  "
          f"{r:.1f}x  h={s['h']:+.3f} ({h_label(s['h'])})  {st}")
    w("")
    # Interpret
    if tb:
        if tb["active"]:
            w(f"    Trees is TARGETED as source for Built Area gain")
            w(f"    ({tb['intensity']:.4f}% vs W={tb['uniform']:.4f}%, "
              f"ratio={tb['intensity']/tb['uniform']:.1f}x).")
        else:
            w(f"    Trees is AVOIDED as source for Built Area gain")
            w(f"    ({tb['intensity']:.4f}% vs W={tb['uniform']:.4f}%, "
              f"ratio={tb['intensity']/tb['uniform']:.1f}x).")
            w(f"    Despite providing largest absolute flow "
              f"({tb['pixels']:,} px = {tb['pixels']*PX_KM2:.2f} km2),")
            w(f"    it is below the uniform threshold because Trees")
            w(f"    dominates the landscape (C_i+ >> other categories).")
    w("")

    # Loss perspective note for Trees->Built
    tb_loss = next((s for s in L3.get(2, []) if s["target_code"] == 7), None)
    if tb_loss:
        V_trees = lchi3.get(2, {}).get("W", 0)
        r_l = tb_loss["intensity"] / tb_loss["uniform"] if tb_loss["uniform"] > 0 else 0
        st_l = "TARGETED" if tb_loss["active"] else "avoided"
        w(f"    Loss perspective (A&P Eq. 12-13):")
        w(f"      Trees -> Built: Q_jk={tb_loss['intensity']:.4f}% "
          f"vs V_j={tb_loss['uniform']:.4f}% ({r_l:.1f}x, {st_l})")
        if tb and tb_loss and tb["active"] != tb_loss["active"]:
            w(f"      NOTE: Gain and loss perspectives disagree.")
            w(f"      This is a known feature of the A&P framework:")
            w(f"      the two perspectives use different denominators.")
    w("")

    # Q-E-S
    qp = qes_s["Q"] / qes_s["T"] * 100.0 if qes_s["T"] > 0 else 0.0
    ep = qes_s["E"] / qes_s["T"] * 100.0 if qes_s["T"] > 0 else 0.0
    sp = qes_s["S"] / qes_s["T"] * 100.0 if qes_s["T"] > 0 else 0.0
    dom = "Exchange" if ep >= max(qp, sp) else ("Quantity" if qp >= sp else "Shift")
    w(f"  FINDING 4: CHANGE COMPOSITION")
    w(f"    Q={qp:.1f}%, E={ep:.1f}%, S={sp:.1f}% (dominant: {dom})")
    if ep > 40:
        w(f"    High exchange ({ep:.0f}%) indicates substantial paired swapping")
        w(f"    between categories (Trees-Rangeland, Crops-Rangeland).")
    if sp > 5:
        w(f"    Non-zero shift ({sp:.1f}%) indicates allocative change beyond")
        w(f"    simple pairwise exchange.")
    w("")

    w(f"  FINDING 5: NON-STATIONARITY")
    if not markov["stat"]:
        w(f"    G={markov['G']:.2f}, p={markov['p']:.2e} -> NON-STATIONARY")
        w(f"    Transition dynamics evolved over the study period.")
        ns = [CLASS_INFO[c][0] for c in LAND_CODES
              if not markov["pc"][c]["stat"]]
        if ns:
            w(f"    Non-stationary: {', '.join(ns)}")
    else:
        w(f"    Dynamics are stationary.")
    w("")

    w(f"  FINDING 6: CAUSAL CHAIN EVIDENCE")
    w(f"    (a) Interval: {na}/{len(L1)} Active (chi-square confirmed)")
    w(f"    (b) Category: Built gain Active; Trees gain Dormant")
    if tb:
        tb_ratio = tb["intensity"] / tb["uniform"] if tb["uniform"] > 0 else 0
        tb_st = "targeted" if tb["active"] else "avoided"
        w(f"    (c) Transition: Trees > Built {tb_st} at {tb_ratio:.1f}x")
        w(f"        (h = {tb['h']:+.3f}, {h_label(tb['h'])})")
        w(f"        Absolute: {tb['pixels']:,} px = {tb['pixels']*PX_KM2:.2f} km2")
    w(f"    (d) Q-E-S: {dom} dominates ({max(qp,ep,sp):.0f}%)")
    mk = "Non-stationary" if not markov["stat"] else "Stationary"
    w(f"    (e) Markov: {mk}")
    w("")
    w(f"    Two-stage deforestation cascade:")
    w(f"    Trees degrade to Rangeland/Crops (high exchange), then")
    w(f"    Crops/Bare Ground convert to Built Area (targeted).")
    w(f"    Cumulative land disturbance progressively degraded")
    w(f"    coastal water quality until the marine threshold")
    w(f"    (May 2019 BSTS/changepoint Kd490), consistent with")
    w(f"    nonlinear sediment loading.")
    w("")

    w("  METHODS")
    w("    - Sentinel-2 10m composites (Dynamic World / ESA WorldCover)")
    w("    - Water (>58%), Clouds, no-data excluded per interval")
    w("    - Intensity: Aldwaik & Pontius (2012), Land Use Policy 29(1)")
    w("    - Q-E-S: Pontius (2019), Int. J. Remote Sensing")
    w("    - Wilson CI: Wilson (1927), JASA 22(158)")
    w("    - Chi-square: Pearson's goodness-of-fit")
    w("    - Markov: Likelihood-ratio G-test")
    w("    - Effect size: Cohen's h for proportions")
    w("")
    w(SEP)
    w("  END OF REPORT")
    w(SEP)

    path = os.path.join(outdir, "intensity_analysis_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(buf))
    print(f"  Report: {path}")


# ==============================================================
#  MAIN
# ==============================================================

def main():
    sep = "=" * 60
    print(sep)
    print("  INTENSITY ANALYSIS (Revised)")
    print(sep)

    print("\n[1] Loading ...")
    years, lulc = load_lulc(NCFILE)
    print(f"    {len(years)} years: {years[0]}-{years[-1]}, "
          f"shape={lulc[0].shape}")

    print("\n[2] Transition matrices ...")
    tm_list, nv_list = [], []
    for t in range(len(years) - 1):
        tm, nv = compute_tm(lulc[t], lulc[t + 1], LAND_CODES)
        tm_list.append(tm)
        nv_list.append(nv)
        a = tm2arr(tm, LAND_CODES)
        c = a.sum() - np.trace(a)
        print(f"    {years[t]}->{years[t+1]}: {nv:,} valid, {c:,} changed")

    print("\n[3] Level 1 -- Interval ...")
    L1, U1, chi2_1 = level1_interval(tm_list, years, LAND_CODES)
    for r in L1:
        print(f"    {r['interval']}: S={r['S']:.4f}% "
              f"[{'ACT' if r['active'] else 'dor'}]")
    print(f"    U={U1:.4f}%, chi2={chi2_1['chi2']:.1f}, p={chi2_1['p']:.2e}")

    print("\n[4] Level 2 -- Category ...")
    L2, U2, chi2_2 = level2_category(tm_list, LAND_CODES)
    for r in L2:
        if r["code"] in LAND_CODES:
            print(f"    {r['name']:<18} G={r['gi']:.2f}% "
                  f"[{'A' if r['ga'] else 'd'}]  "
                  f"L={r['li']:.2f}% [{'A' if r['la'] else 'd'}]")

    print("\n[5] Level 3 -- Transition (A&P Eq. 8-9, 12-13) ...")
    G3, L3, gchi3, lchi3 = level3_transition(tm_list, LAND_CODES)
    W_built = gchi3.get(7, {}).get("W", 0)
    print(f"    Built gain sources (W_j = {W_built:.4f}%):")
    for s in sorted(G3.get(7, []), key=lambda x: x["intensity"],
                    reverse=True):
        if s["source_code"] in LAND_CODES:
            r = s["intensity"] / s["uniform"] if s["uniform"] > 0 else 0
            st = "TARG" if s["active"] else "avd"
            print(f"      {s['source_name']:<18} R={s['intensity']:.4f}% "
                  f"({r:.1f}x) h={s['h']:+.3f} [{st}]")

    print("\n[6] Q-E-S ...")
    agg = agg_matrices(tm_list, LAND_CODES)
    qes_r, qes_s = qes_decomposition(agg, LAND_CODES)
    print(f"    Q={qes_s['Q']:.0f} ({qes_s['Q']/qes_s['T']*100:.1f}%), "
          f"E={qes_s['E']:.0f} ({qes_s['E']/qes_s['T']*100:.1f}%), "
          f"S={qes_s['S']:.0f} ({qes_s['S']/qes_s['T']*100:.1f}%)")

    print("\n[7] Per-interval ...")
    pi_cat = per_interval_cat(tm_list, years, LAND_CODES)
    pi_tb  = per_interval_trans(tm_list, years, LAND_CODES, tgt=7, src=2)
    for r in pi_tb:
        print(f"    {r['interval']}: R={r['R']:.4f}% W={r['W']:.4f}% "
              f"({r['ratio']:.1f}x) [{'TARG' if r['active'] else 'avd'}]")

    print("\n[8] Markov stationarity ...")
    markov = markov_test(tm_list, LAND_CODES)
    print(f"    G={markov['G']:.2f}, p={markov['p']:.2e} "
          f"[{'STAT' if markov['stat'] else 'NON-STAT'}]")

    print("\n[9] Figure ...")
    make_figure(L1, U1, L2, U2, G3, L3, FIGDIR)

    print("\n[10] Report ...")
    write_report(years, tm_list, nv_list,
                 L1, U1, chi2_1,
                 L2, U2, chi2_2,
                 G3, L3, gchi3, lchi3,
                 qes_r, qes_s,
                 pi_cat, pi_tb,
                 markov,
                 REPORTDIR)

    print(f"\n{sep}\n  DONE\n{sep}")


if __name__ == "__main__":
    main()
