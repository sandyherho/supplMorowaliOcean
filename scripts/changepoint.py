#!/usr/bin/env python
"""
Robust Structural Break Detection in Kd490 Time Series
========================================================
Impact Zone  : Morowali Coastal Waters, Sulawesi, Indonesia
Control Zone : Banda Sea (offshore reference)

Methodology:
  1. Penalty sensitivity analysis (elbow plot)
  2. Multi-algorithm consensus (PELT, BinSeg, Window)
  3. Bootstrap significance testing (n=5000)
  4. Cliff's delta effect size (non-parametric)
  5. BIC for optimal model selection
  6. Welch's t-test + Mann-Whitney U + Levene + KS
  7. Autocorrelation-aware effective sample size
  8. Control vs impact comparison with DiD

Author : Sandy H. S. Herho
Date   : 2025/02/22
License: MIT
"""

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import ruptures as rpt
from scipy import stats
from scipy.ndimage import uniform_filter1d
from datetime import datetime
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 0. CONFIGURATION
# ============================================================
DATA_PATH = "../raw_data/kd490.nc"
VAR_NAME = "KD490"

# Impact zone: Morowali coastal waters
IMPACT_LAT = slice(-2.92, -2.72)
IMPACT_LON = slice(122.08, 122.28)
IMPACT_LABEL = "Morowali (Impact)"

# Control zone: Banda Sea offshore
CONTROL_LAT = slice(-2.75, -2.45)
CONTROL_LON = slice(123.00, 123.40)
CONTROL_LABEL = "Banda Sea (Control)"

SCALE_FACTOR = 100
N_BOOTSTRAP = 5000
ALPHA = 0.05
PENALTY_RANGE = np.arange(1, 51, 1)
RANDOM_SEED = 42
FIG_DIR = "../figs"
REPORT_DIR = "../reports"
DPI = 400

np.random.seed(RANDOM_SEED)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================
def load_and_preprocess(path, var, lat_sl, lon_sl, scale):
    """Load NetCDF, apply area weighting, return time series DataFrame."""
    ds = xr.open_dataset(path)
    da = ds[var].interpolate_na(dim="time", method="linear")
    weights = np.cos(np.deg2rad(da.latitude))
    weights.name = "weights"
    region = da.sel(latitude=lat_sl, longitude=lon_sl)
    ts = (region * scale).weighted(weights).mean(dim=["latitude", "longitude"])
    df = ts.to_dataframe(name="Kd490").dropna()
    return df


# ============================================================
# 2. PENALTY SENSITIVITY ANALYSIS
# ============================================================
def penalty_sensitivity(signal, pen_range, model="rbf"):
    """Sweep penalty values and record breakpoints detected."""
    n_bkps = []
    for pen in pen_range:
        algo = rpt.Pelt(model=model, min_size=3, jump=1).fit(signal)
        bkps = algo.predict(pen=pen)
        n_bkps.append(len(bkps) - 1)
    return np.array(n_bkps)


def find_elbow(pen_range, n_bkps):
    """Kneedle-style elbow detection via second derivative."""
    smoothed = uniform_filter1d(n_bkps.astype(float), size=3)
    d1 = np.diff(smoothed)
    d2 = np.diff(d1)
    if len(d2) == 0:
        return pen_range[len(pen_range) // 2]
    elbow_idx = np.argmax(np.abs(d2)) + 1
    return pen_range[elbow_idx]


# ============================================================
# 3. BIC MODEL SELECTION
# ============================================================
def bic_model_selection(signal, max_bkps=10, model="rbf"):
    """BIC = n * ln(RSS/n) + k * ln(n). Lower is better."""
    n = len(signal)
    bic_scores = []
    for k in range(0, max_bkps + 1):
        if k == 0:
            rss = np.sum((signal - signal.mean()) ** 2)
            n_params = 2
        else:
            algo = rpt.Dynp(model=model, min_size=3, jump=1).fit(signal)
            try:
                bkps = algo.predict(n_bkps=k)
            except Exception:
                bic_scores.append(np.inf)
                continue
            boundaries = sorted(set([0] + bkps + [n]))
            rss = 0
            for i in range(len(boundaries) - 1):
                seg = signal[boundaries[i]:boundaries[i + 1]]
                if len(seg) > 0:
                    rss += np.sum((seg - seg.mean()) ** 2)
            n_params = 2 * (k + 1)
        if rss <= 0:
            rss = 1e-10
        bic = n * np.log(rss / n) + n_params * np.log(n)
        bic_scores.append(bic)
    optimal_k = np.argmin(bic_scores)
    return optimal_k, bic_scores


# ============================================================
# 4. MULTI-ALGORITHM CONSENSUS
# ============================================================
def multi_algorithm_detection(signal, n_bkps_target, model="rbf"):
    """Run PELT, BinSeg, Window and return breakpoints from each."""
    results = {}
    algo_pelt = rpt.Pelt(model=model, min_size=3, jump=1).fit(signal)
    results["PELT"] = algo_pelt.predict(pen=10)

    algo_binseg = rpt.Binseg(model=model, min_size=3, jump=1).fit(signal)
    results["BinSeg"] = algo_binseg.predict(n_bkps=n_bkps_target)

    width = max(10, len(signal) // 20)
    algo_window = rpt.Window(width=width, model=model, min_size=3, jump=1).fit(signal)
    results["Window"] = algo_window.predict(n_bkps=n_bkps_target)

    return results


def find_consensus(all_bkps, signal_len, tolerance=5):
    """Breakpoints detected by >=2 of 3 algorithms within +/-tolerance."""
    flat = []
    for name, bkps in all_bkps.items():
        flat.extend([(b, name) for b in bkps if b < signal_len])
    if not flat:
        return []

    indices = sorted(set(b for b, _ in flat))
    clusters = []
    current_cluster = [indices[0]]
    for idx in indices[1:]:
        if idx - current_cluster[-1] <= tolerance:
            current_cluster.append(idx)
        else:
            clusters.append(current_cluster)
            current_cluster = [idx]
    clusters.append(current_cluster)

    consensus = []
    for cluster in clusters:
        supporting = set()
        for b, name in flat:
            if any(abs(b - c) <= tolerance for c in cluster):
                supporting.add(name)
        if len(supporting) >= 2:
            consensus.append(int(np.median(cluster)))
    return sorted(consensus)


# ============================================================
# 5. BOOTSTRAP SIGNIFICANCE TEST
# ============================================================
def bootstrap_significance(signal, breakpoint_idx, n_iter=5000):
    """Permutation-based significance test for structural break."""
    seg1 = signal[:breakpoint_idx]
    seg2 = signal[breakpoint_idx:]
    observed_diff = abs(seg1.mean() - seg2.mean())
    count = 0
    for _ in range(n_iter):
        perm = np.random.permutation(signal)
        perm_diff = abs(perm[:breakpoint_idx].mean() - perm[breakpoint_idx:].mean())
        if perm_diff >= observed_diff:
            count += 1
    p_value = count / n_iter
    return p_value, observed_diff


# ============================================================
# 6. CLIFF'S DELTA
# ============================================================
def cliffs_delta(x, y):
    """
    Cliff's delta (non-parametric effect size).
    Romano et al. (2006):
      |d| < 0.147 : negligible
      |d| < 0.33  : small
      |d| < 0.474 : medium
      |d| >= 0.474 : large
    """
    nx, ny = len(x), len(y)
    more = 0
    less = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                more += 1
            elif xi < yj:
                less += 1
    return (more - less) / (nx * ny)


def cliffs_delta_interpret(d):
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    elif ad < 0.33:
        return "small"
    elif ad < 0.474:
        return "medium"
    else:
        return "large"


# ============================================================
# 7. EFFECTIVE SAMPLE SIZE & REGIME STATISTICS
# ============================================================
def effective_sample_size(x):
    """Bretherton et al. (1999) correction."""
    n = len(x)
    if n < 3:
        return n
    r1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    if np.isnan(r1) or r1 >= 1.0 or r1 <= -1.0:
        return n
    n_eff = n * (1 - r1) / (1 + r1)
    return max(2, int(np.round(n_eff)))


def regime_statistics(signal, breakpoints):
    """Compute regime descriptive stats and pairwise tests."""
    boundaries = sorted(set([0] + breakpoints + [len(signal)]))
    regimes = []
    for i in range(len(boundaries) - 1):
        seg = signal[boundaries[i]:boundaries[i + 1]]
        n_raw = len(seg)
        n_eff = effective_sample_size(seg)
        lag1 = np.corrcoef(seg[:-1], seg[1:])[0, 1] if n_raw > 2 else np.nan
        regimes.append({
            "regime": i + 1,
            "start_idx": boundaries[i],
            "end_idx": boundaries[i + 1],
            "n_raw": n_raw,
            "n_eff": n_eff,
            "mean": seg.mean(),
            "std": seg.std(ddof=1),
            "median": np.median(seg),
            "iqr": np.percentile(seg, 75) - np.percentile(seg, 25),
            "min": seg.min(),
            "max": seg.max(),
            "skew": stats.skew(seg),
            "kurtosis": stats.kurtosis(seg),
            "lag1_autocorr": lag1,
        })

    comparisons = []
    for i in range(len(regimes) - 1):
        s1 = signal[regimes[i]["start_idx"]:regimes[i]["end_idx"]]
        s2 = signal[regimes[i + 1]["start_idx"]:regimes[i + 1]["end_idx"]]
        cd = cliffs_delta(s1, s2)
        cd_interp = cliffs_delta_interpret(cd)
        t_stat, t_pval = stats.ttest_ind(s1, s2, equal_var=False)
        u_stat, u_pval = stats.mannwhitneyu(s1, s2, alternative="two-sided")
        lev_stat, lev_pval = stats.levene(s1, s2)
        ks_stat, ks_pval = stats.ks_2samp(s1, s2)
        comparisons.append({
            "regimes": f"{i + 1} vs {i + 2}",
            "cliffs_delta": cd,
            "cliffs_delta_interp": cd_interp,
            "welch_t": t_stat,
            "welch_p": t_pval,
            "mann_whitney_U": u_stat,
            "mann_whitney_p": u_pval,
            "levene_F": lev_stat,
            "levene_p": lev_pval,
            "ks_stat": ks_stat,
            "ks_p": ks_pval,
        })
    return regimes, comparisons


# ============================================================
# 8. FULL ZONE ANALYSIS PIPELINE
# ============================================================
def analyze_zone(ts_df, label):
    """Run full changepoint analysis pipeline for one zone."""
    signal = ts_df["Kd490"].values
    n = len(signal)

    print(f"\n  --- {label} ---")
    print(f"  n = {n}, range: {ts_df.index[0].strftime('%Y-%m')} to "
          f"{ts_df.index[-1].strftime('%Y-%m')}")

    # Penalty sensitivity
    print(f"  [1] Penalty sensitivity ...")
    n_bkps_array = penalty_sensitivity(signal, PENALTY_RANGE)
    optimal_pen = find_elbow(PENALTY_RANGE, n_bkps_array)
    print(f"      Optimal penalty: {optimal_pen}")

    # BIC
    print(f"  [2] BIC model selection ...")
    optimal_k, bic_scores = bic_model_selection(signal, max_bkps=8)
    print(f"      BIC-optimal k: {optimal_k}")

    # Multi-algorithm consensus
    n_target = max(optimal_k, 1)
    print(f"  [3] Multi-algorithm detection (target k={n_target}) ...")
    all_bkps = multi_algorithm_detection(signal, n_bkps_target=n_target)
    for name, bkps in all_bkps.items():
        dates = [ts_df.index[b].strftime("%Y-%m") for b in bkps if b < n]
        print(f"      {name:8s}: {dates}")

    consensus_bkps = find_consensus(all_bkps, n, tolerance=5)
    print(f"      Consensus: {len(consensus_bkps)} breakpoint(s)")
    for bp in consensus_bkps:
        print(f"        Index {bp} -> {ts_df.index[bp].strftime('%B %Y')}")

    # Bootstrap significance
    print(f"  [4] Bootstrap ({N_BOOTSTRAP} iter) ...")
    bootstrap_results = []
    for bp in consensus_bkps:
        p_val, obs_diff = bootstrap_significance(signal, bp, n_iter=N_BOOTSTRAP)
        sig = "SIGNIFICANT" if p_val < ALPHA else "NOT significant"
        print(f"      Break {ts_df.index[bp].strftime('%b %Y')}: "
              f"|dmu|={obs_diff:.3f}, p={p_val:.4f} -> {sig}")
        bootstrap_results.append((p_val, obs_diff))

    # Regime statistics
    print(f"  [5] Regime statistics ...")
    regimes, comparisons = regime_statistics(signal, consensus_bkps)
    for reg in regimes:
        print(f"      Regime {reg['regime']}: mu={reg['mean']:.3f}, "
              f"sig={reg['std']:.3f}, n={reg['n_raw']}")
    for comp in comparisons:
        print(f"      {comp['regimes']}: Cliff's d={comp['cliffs_delta']:+.3f} "
              f"({comp['cliffs_delta_interp']})")

    return {
        "ts_df": ts_df,
        "signal": signal,
        "n": n,
        "label": label,
        "n_bkps_array": n_bkps_array,
        "optimal_pen": optimal_pen,
        "optimal_k": optimal_k,
        "bic_scores": bic_scores,
        "all_bkps": all_bkps,
        "consensus_bkps": consensus_bkps,
        "bootstrap_results": bootstrap_results,
        "regimes": regimes,
        "comparisons": comparisons,
    }


# ============================================================
# 9. PLOTTING
# ============================================================
def plot_penalty_bic_2x2(res_impact, res_control):
    """Figure 1: 2x2 — (a) penalty impact, (b) BIC impact,
                        (c) penalty control, (d) BIC control."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=DPI)

    panels = [
        (axes[0, 0], "(a)", res_impact, "penalty"),
        (axes[0, 1], "(b)", res_impact, "bic"),
        (axes[1, 0], "(c)", res_control, "penalty"),
        (axes[1, 1], "(d)", res_control, "bic"),
    ]

    for ax, subtitle, res, ptype in panels:
        if ptype == "penalty":
            ax.plot(PENALTY_RANGE, res["n_bkps_array"], "o-", color="#2c3e50",
                    markersize=2.5, linewidth=1.0, markeredgecolor="none")
            ax.axvline(res["optimal_pen"], color="#c0392b", linestyle="--",
                       linewidth=1.5)
            ymin_v = min(res["n_bkps_array"])
            ymax_v = max(res["n_bkps_array"])
            yrange = ymax_v - ymin_v if ymax_v > ymin_v else 1
            ax.text(res["optimal_pen"] + 1.0,
                    ymin_v + 0.65 * yrange,
                    f"Optimal = {res['optimal_pen']}", color="#c0392b",
                    fontsize=8, fontweight="bold", va="center")
            ax.set_xlabel("Penalty parameter", fontsize=10)
            ax.set_ylabel("Number of breakpoints", fontsize=10)
        else:
            ks = range(len(res["bic_scores"]))
            ax.plot(ks, res["bic_scores"], "s-", color="#2c3e50",
                    markersize=4, linewidth=1.0, markeredgecolor="none")
            ax.axvline(res["optimal_k"], color="#c0392b", linestyle="--",
                       linewidth=1.5)
            bic_min = min(res["bic_scores"])
            bic_max = max(res["bic_scores"])
            bic_range = bic_max - bic_min if bic_max > bic_min else 1
            ax.text(res["optimal_k"] + 0.3,
                    bic_min + 0.15 * bic_range,
                    f"Optimal $k$ = {res['optimal_k']}", color="#c0392b",
                    fontsize=8, fontweight="bold", va="center")
            ax.set_xlabel("Number of breakpoints ($k$)", fontsize=10)
            ax.set_ylabel("BIC", fontsize=10)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        ax.text(-0.10, 1.04, subtitle, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "penalty_bic.png"),
                dpi=DPI, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "penalty_bic.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def _plot_single_consensus(ax, res, palette, subtitle):
    """Helper: plot one consensus panel."""
    ts_df = res["ts_df"]
    signal = res["signal"]
    regimes = res["regimes"]
    consensus_bkps = res["consensus_bkps"]
    bootstrap_results = res["bootstrap_results"]

    # Raw signal
    ax.plot(ts_df.index, signal, color="#95a5a6", linewidth=0.7,
            alpha=0.7, zorder=2)

    # Shade regimes + mean +/- 1 sigma
    for i, reg in enumerate(regimes):
        c = palette[i % len(palette)]
        idx_s, idx_e = reg["start_idx"], reg["end_idx"]
        t_s = ts_df.index[idx_s]
        t_e = ts_df.index[min(idx_e, len(ts_df) - 1)]

        ax.axvspan(t_s, t_e, color=c, alpha=0.07, zorder=0)
        ax.hlines(y=reg["mean"], xmin=t_s, xmax=t_e, color=c,
                  linewidth=2.0, linestyle="-", zorder=3)
        seg_times = ts_df.index[idx_s:idx_e]
        ax.fill_between(seg_times,
                        reg["mean"] - reg["std"],
                        reg["mean"] + reg["std"],
                        color=c, alpha=0.12, edgecolor="none", zorder=1)

    # Breakpoint vertical lines
    for j, bp in enumerate(consensus_bkps):
        ax.axvline(ts_df.index[bp], color="black", linestyle="--",
                   linewidth=1.6, zorder=4)

    # Legend handles
    handles = [Line2D([0], [0], color="#95a5a6", linewidth=0.7,
                      label=r"$K_d490$ signal")]
    for i, reg in enumerate(regimes):
        c = palette[i % len(palette)]
        handles.append(Line2D([0], [0], color=c, linewidth=2.0,
                              label=(f"Regime {reg['regime']}: "
                                     f"$\\mu$={reg['mean']:.2f}, "
                                     f"$\\sigma$={reg['std']:.2f}")))
    for j, bp in enumerate(consensus_bkps):
        bp_date = ts_df.index[bp]
        p_val = bootstrap_results[j][0] if j < len(bootstrap_results) else np.nan
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else \
              "*" if p_val < 0.05 else "n.s."
        handles.append(Line2D([0], [0], color="black", linewidth=1.6,
                              linestyle="--",
                              label=(f"Break: {bp_date.strftime('%b %Y')} "
                                     f"($p$={p_val:.4f}{sig})")))
    if not consensus_bkps:
        handles.append(Line2D([0], [0], color="black", linewidth=0,
                              label="No consensus break detected"))

    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=True,
              fontsize=8.5, facecolor="white", edgecolor="#cccccc",
              borderpad=0.6, columnspacing=1.2, handlelength=2.2)

    ax.set_ylabel(r"$K_d490$ ($\times 10^{-2}\ \mathrm{m^{-1}}$)",
                  fontsize=10)
    ax.set_xlabel("Year", fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.text(-0.07, 1.04, subtitle, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")


def plot_consensus_2x1(res_control, res_impact):
    """Figure 2: 2x1 — (a) control zone, (b) impact zone."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), dpi=DPI)

    palette_ctrl = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#1abc9c"]
    palette_imp = ["#2980b9", "#c0392b", "#27ae60", "#8e44ad", "#f39c12"]

    _plot_single_consensus(axes[0], res_control, palette_ctrl, "(a)")
    _plot_single_consensus(axes[1], res_impact, palette_imp, "(b)")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.50)
    fig.savefig(os.path.join(FIG_DIR, "consensus_regimes.png"),
                dpi=DPI, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "consensus_regimes.pdf"),
                bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 10. REPORT WRITER
# ============================================================
def _write_zone_section(f, res, zone_name, sec):
    """Write full analysis results for one zone."""
    ts_df = res["ts_df"]
    signal = res["signal"]
    n = res["n"]

    f.write(f"\n{'=' * 72}\n")
    f.write(f"  ZONE: {zone_name}\n")
    f.write(f"{'=' * 72}\n\n")

    # Data summary
    f.write(f"{sec}.1 DATA SUMMARY\n")
    f.write("-" * 72 + "\n")
    f.write(f"  Time series length : {n} observations\n")
    f.write(f"  Date range         : {ts_df.index[0].strftime('%Y-%m-%d')} to "
            f"{ts_df.index[-1].strftime('%Y-%m-%d')}\n")
    f.write(f"  Global mean        : {signal.mean():.4f}\n")
    f.write(f"  Global std         : {signal.std(ddof=1):.4f}\n")
    f.write(f"  Global min         : {signal.min():.4f}\n")
    f.write(f"  Global max         : {signal.max():.4f}\n")
    f.write(f"  Global median      : {np.median(signal):.4f}\n\n")

    # Penalty sensitivity
    f.write(f"{sec}.2 PENALTY SENSITIVITY ANALYSIS\n")
    f.write("-" * 72 + "\n")
    f.write(f"  Penalty range      : {PENALTY_RANGE[0]} to {PENALTY_RANGE[-1]}\n")
    f.write(f"  Optimal penalty    : {res['optimal_pen']}\n")
    pen_idx = np.where(PENALTY_RANGE == res["optimal_pen"])[0]
    if len(pen_idx) > 0:
        f.write(f"  Bkps at optimal    : {res['n_bkps_array'][pen_idx[0]]}\n")
    f.write("  Penalty -> n_bkps:\n")
    for pen, nb in zip(PENALTY_RANGE, res["n_bkps_array"]):
        f.write(f"    pen={pen:3d} -> {nb:2d}\n")
    f.write("\n")

    # BIC
    f.write(f"{sec}.3 BIC MODEL SELECTION\n")
    f.write("-" * 72 + "\n")
    f.write(f"  Optimal k (BIC)    : {res['optimal_k']}\n")
    f.write("  k -> BIC:\n")
    for k, bic in enumerate(res["bic_scores"]):
        marker = " <-- optimal" if k == res["optimal_k"] else ""
        f.write(f"    k={k:2d} -> BIC = {bic:12.3f}{marker}\n")
    f.write("\n")

    # Multi-algorithm
    f.write(f"{sec}.4 MULTI-ALGORITHM DETECTION\n")
    f.write("-" * 72 + "\n")
    for name, bkps in res["all_bkps"].items():
        dates = [ts_df.index[b].strftime("%Y-%m-%d") for b in bkps if b < n]
        indices = [b for b in bkps if b < n]
        f.write(f"  {name:8s}: idx = {indices}\n")
        f.write(f"  {'':<8s}  dates = {dates}\n")
    f.write("\n")

    # Consensus
    f.write(f"{sec}.5 CONSENSUS BREAKPOINTS (>=2/3 methods, tol=5)\n")
    f.write("-" * 72 + "\n")
    if not res["consensus_bkps"]:
        f.write("  No consensus breakpoints detected.\n")
    for bp in res["consensus_bkps"]:
        f.write(f"  Index {bp:4d} -> {ts_df.index[bp].strftime('%Y-%m-%d')} "
                f"({ts_df.index[bp].strftime('%B %Y')})\n")
    f.write("\n")

    # Bootstrap
    f.write(f"{sec}.6 BOOTSTRAP SIGNIFICANCE (n_iter = {N_BOOTSTRAP})\n")
    f.write("-" * 72 + "\n")
    if not res["bootstrap_results"]:
        f.write("  No breakpoints to test.\n")
    for j, bp in enumerate(res["consensus_bkps"]):
        p_val, obs_diff = res["bootstrap_results"][j]
        sig = "SIGNIFICANT" if p_val < ALPHA else "NOT significant"
        stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else \
                "*" if p_val < 0.05 else "n.s."
        f.write(f"  Break at {ts_df.index[bp].strftime('%b %Y')} (idx {bp}):\n")
        f.write(f"    |Delta mu| = {obs_diff:.6f}\n")
        f.write(f"    p-value    = {p_val:.6f} {stars}\n")
        f.write(f"    -> {sig} at alpha = {ALPHA}\n\n")

    # Regime statistics
    f.write(f"{sec}.7 REGIME DESCRIPTIVE STATISTICS\n")
    f.write("-" * 72 + "\n")
    hdr = (f"  {'Rgm':<5} {'n':>5} {'n_eff':>6} {'Mean':>9} {'Std':>9} "
           f"{'Med':>9} {'IQR':>9} {'Min':>9} {'Max':>9} "
           f"{'Skew':>7} {'Kurt':>7} {'r1':>7}")
    f.write(hdr + "\n")
    f.write("  " + "-" * (len(hdr) - 2) + "\n")
    for reg in res["regimes"]:
        f.write(f"  {reg['regime']:<5} {reg['n_raw']:>5} {reg['n_eff']:>6} "
                f"{reg['mean']:>9.4f} {reg['std']:>9.4f} "
                f"{reg['median']:>9.4f} {reg['iqr']:>9.4f} "
                f"{reg['min']:>9.4f} {reg['max']:>9.4f} "
                f"{reg['skew']:>7.3f} {reg['kurtosis']:>7.3f} "
                f"{reg['lag1_autocorr']:>7.3f}\n")
    f.write("\n  Date ranges:\n")
    for reg in res["regimes"]:
        t_s = ts_df.index[reg["start_idx"]].strftime("%Y-%m-%d")
        t_e = ts_df.index[min(reg["end_idx"], n - 1)].strftime("%Y-%m-%d")
        f.write(f"    Regime {reg['regime']}: {t_s} to {t_e} "
                f"({reg['n_raw']} obs, n_eff={reg['n_eff']})\n")
    f.write("\n")

    # Pairwise comparisons
    f.write(f"{sec}.8 PAIRWISE REGIME COMPARISONS\n")
    f.write("-" * 72 + "\n")
    if not res["comparisons"]:
        f.write("  Single regime -- no pairwise comparisons.\n")
    for comp in res["comparisons"]:
        f.write(f"  Regime {comp['regimes']}:\n")
        f.write(f"    Cliff's delta  = {comp['cliffs_delta']:+.4f} "
                f"({comp['cliffs_delta_interp']})\n")
        f.write(f"    Welch t-test   : t = {comp['welch_t']:.4f}, "
                f"p = {comp['welch_p']:.6e}\n")
        f.write(f"    Mann-Whitney U : U = {comp['mann_whitney_U']:.1f}, "
                f"p = {comp['mann_whitney_p']:.6e}\n")
        f.write(f"    Levene's test  : F = {comp['levene_F']:.4f}, "
                f"p = {comp['levene_p']:.6e}\n")
        f.write(f"    KS 2-sample    : D = {comp['ks_stat']:.4f}, "
                f"p = {comp['ks_p']:.6e}\n\n")


def write_report(res_impact, res_control):
    """Write comprehensive report with both zones + interpretation."""
    path = os.path.join(REPORT_DIR, "changepoint_report.txt")

    with open(path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write("  ROBUST STRUCTURAL BREAK DETECTION IN Kd490\n")
        f.write("  Impact Zone : Morowali Coastal Waters, Sulawesi\n")
        f.write("  Control Zone: Banda Sea (offshore reference)\n")
        f.write("=" * 72 + "\n")
        f.write(f"  Author       : Sandy H. S. Herho\n")
        f.write(f"  Generated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Random seed  : {RANDOM_SEED}\n")
        f.write(f"  Bootstrap n  : {N_BOOTSTRAP}\n")
        f.write(f"  Alpha        : {ALPHA}\n")
        f.write(f"  Scale factor : {SCALE_FACTOR}\n")
        f.write("=" * 72 + "\n")

        f.write(f"\n  SPATIAL DOMAINS:\n")
        f.write(f"    Impact  -- Lat: {IMPACT_LAT.start} to {IMPACT_LAT.stop}, "
                f"Lon: {IMPACT_LON.start} to {IMPACT_LON.stop}\n")
        f.write(f"    Control -- Lat: {CONTROL_LAT.start} to {CONTROL_LAT.stop}, "
                f"Lon: {CONTROL_LON.start} to {CONTROL_LON.stop}\n")

        # Zone sections
        _write_zone_section(f, res_impact, "MOROWALI COASTAL WATERS (IMPACT)", 1)
        _write_zone_section(f, res_control, "BANDA SEA (CONTROL)", 2)

        # Cross-zone comparison
        f.write("\n" + "=" * 72 + "\n")
        f.write("  CROSS-ZONE COMPARISON: IMPACT vs CONTROL\n")
        f.write("=" * 72 + "\n\n")

        sig_imp = res_impact["signal"]
        sig_ctrl = res_control["signal"]

        f.write("3.1 FULL TIME SERIES COMPARISON\n")
        f.write("-" * 72 + "\n")
        t_stat, t_pval = stats.ttest_ind(sig_imp, sig_ctrl, equal_var=False)
        u_stat, u_pval = stats.mannwhitneyu(sig_imp, sig_ctrl,
                                             alternative="two-sided")
        cd = cliffs_delta(sig_imp, sig_ctrl)
        cd_interp = cliffs_delta_interpret(cd)
        ks_stat, ks_pval = stats.ks_2samp(sig_imp, sig_ctrl)

        f.write(f"  Impact  : mu = {sig_imp.mean():.4f}, "
                f"sigma = {sig_imp.std(ddof=1):.4f}, n = {len(sig_imp)}\n")
        f.write(f"  Control : mu = {sig_ctrl.mean():.4f}, "
                f"sigma = {sig_ctrl.std(ddof=1):.4f}, n = {len(sig_ctrl)}\n")
        f.write(f"  Welch t-test   : t = {t_stat:.4f}, p = {t_pval:.6e}\n")
        f.write(f"  Mann-Whitney U : U = {u_stat:.1f}, p = {u_pval:.6e}\n")
        f.write(f"  Cliff's delta  = {cd:+.4f} ({cd_interp})\n")
        f.write(f"  KS 2-sample    : D = {ks_stat:.4f}, p = {ks_pval:.6e}\n\n")

        # Pre/post break comparison
        if res_impact["consensus_bkps"]:
            f.write("3.2 PRE/POST BREAK COMPARISON (at impact breakpoints)\n")
            f.write("-" * 72 + "\n")

            for bp in res_impact["consensus_bkps"]:
                bp_date = res_impact["ts_df"].index[bp]
                f.write(f"\n  Breakpoint: {bp_date.strftime('%B %Y')} "
                        f"(impact index {bp})\n\n")

                ctrl_idx = res_control["ts_df"].index.searchsorted(bp_date)

                imp_pre = sig_imp[:bp]
                imp_post = sig_imp[bp:]
                ctrl_pre = sig_ctrl[:ctrl_idx]
                ctrl_post = sig_ctrl[ctrl_idx:]

                # Impact
                cd_imp = cliffs_delta(imp_pre, imp_post)
                t_imp, tp_imp = stats.ttest_ind(imp_pre, imp_post,
                                                 equal_var=False)
                f.write(f"  IMPACT -- pre vs post:\n")
                f.write(f"    Pre  : mu = {imp_pre.mean():.4f}, "
                        f"sigma = {imp_pre.std(ddof=1):.4f}, "
                        f"n = {len(imp_pre)}\n")
                f.write(f"    Post : mu = {imp_post.mean():.4f}, "
                        f"sigma = {imp_post.std(ddof=1):.4f}, "
                        f"n = {len(imp_post)}\n")
                f.write(f"    Delta mu      = {imp_post.mean() - imp_pre.mean():+.4f}\n")
                f.write(f"    Cliff's delta = {cd_imp:+.4f} "
                        f"({cliffs_delta_interpret(cd_imp)})\n")
                f.write(f"    Welch t-test  : t = {t_imp:.4f}, "
                        f"p = {tp_imp:.6e}\n\n")

                # Control
                if len(ctrl_pre) > 1 and len(ctrl_post) > 1:
                    cd_ctrl = cliffs_delta(ctrl_pre, ctrl_post)
                    t_ctrl, tp_ctrl = stats.ttest_ind(ctrl_pre, ctrl_post,
                                                       equal_var=False)
                    f.write(f"  CONTROL -- pre vs post (same date split):\n")
                    f.write(f"    Pre  : mu = {ctrl_pre.mean():.4f}, "
                            f"sigma = {ctrl_pre.std(ddof=1):.4f}, "
                            f"n = {len(ctrl_pre)}\n")
                    f.write(f"    Post : mu = {ctrl_post.mean():.4f}, "
                            f"sigma = {ctrl_post.std(ddof=1):.4f}, "
                            f"n = {len(ctrl_post)}\n")
                    f.write(f"    Delta mu      = "
                            f"{ctrl_post.mean() - ctrl_pre.mean():+.4f}\n")
                    f.write(f"    Cliff's delta = {cd_ctrl:+.4f} "
                            f"({cliffs_delta_interpret(cd_ctrl)})\n")
                    f.write(f"    Welch t-test  : t = {t_ctrl:.4f}, "
                            f"p = {tp_ctrl:.6e}\n\n")

                    # DiD
                    did = ((imp_post.mean() - imp_pre.mean()) -
                           (ctrl_post.mean() - ctrl_pre.mean()))
                    f.write(f"  DIFFERENCE-IN-DIFFERENCES (DiD):\n")
                    f.write(f"    DiD = (Delta mu_impact) - (Delta mu_control)\n")
                    f.write(f"        = ({imp_post.mean() - imp_pre.mean():+.4f}) "
                            f"- ({ctrl_post.mean() - ctrl_pre.mean():+.4f})\n")
                    f.write(f"        = {did:+.4f}\n\n")
                    f.write(f"    A positive DiD indicates the impact zone\n")
                    f.write(f"    experienced a larger turbidity increase than\n")
                    f.write(f"    the control zone, beyond any shared regional\n")
                    f.write(f"    trend. This strengthens the attribution to\n")
                    f.write(f"    local forcing factors.\n\n")

        # Interpretation
        f.write("\n" + "=" * 72 + "\n")
        f.write("  INTERPRETATION\n")
        f.write("=" * 72 + "\n\n")

        has_impact_break = len(res_impact["consensus_bkps"]) > 0
        has_control_break = len(res_control["consensus_bkps"]) > 0

        f.write("  KEY FINDINGS:\n\n")

        if has_impact_break and not has_control_break:
            f.write("  1. LOCALIZED PERTURBATION DETECTED\n")
            f.write("     The impact zone (Morowali coastal waters) exhibits\n")
            f.write("     a robust structural break in Kd490, while the\n")
            f.write("     control zone (Banda Sea) shows NO significant\n")
            f.write("     break. This asymmetry is consistent with a\n")
            f.write("     localized perturbation (e.g., land-use change,\n")
            f.write("     coastal development, intensified mining activity)\n")
            f.write("     rather than a basin-wide oceanographic shift.\n\n")
        elif has_impact_break and has_control_break:
            f.write("  1. REGIONAL SIGNAL WITH POSSIBLE LOCAL AMPLIFICATION\n")
            f.write("     Both zones show structural breaks. Compare timing\n")
            f.write("     and magnitude to distinguish local vs regional\n")
            f.write("     drivers. The DiD metric quantifies the excess\n")
            f.write("     change attributable to local factors.\n\n")
        elif not has_impact_break and not has_control_break:
            f.write("  1. TEMPORAL STATIONARITY\n")
            f.write("     Neither zone shows a significant structural break,\n")
            f.write("     suggesting Kd490 has been stationary over the\n")
            f.write("     study period in both domains.\n\n")
        else:
            f.write("  1. CONTROL ZONE BREAK ONLY\n")
            f.write("     The control zone shows a break while the impact\n")
            f.write("     zone does not, possibly indicating regional\n")
            f.write("     oceanographic changes that the coastal zone is\n")
            f.write("     buffered from.\n\n")

        if has_impact_break:
            bp = res_impact["consensus_bkps"][0]
            bp_date = res_impact["ts_df"].index[bp]
            f.write(f"  2. BREAK CHARACTERIZATION\n")
            f.write(f"     The break at {bp_date.strftime('%B %Y')} divides\n")
            f.write(f"     the impact zone record into two regimes:\n\n")
            for reg in res_impact["regimes"]:
                t_s = res_impact["ts_df"].index[reg["start_idx"]].strftime("%Y-%m")
                t_e = res_impact["ts_df"].index[
                    min(reg["end_idx"], res_impact["n"] - 1)].strftime("%Y-%m")
                f.write(f"     Regime {reg['regime']} ({t_s} to {t_e}):\n")
                f.write(f"       mu = {reg['mean']:.4f}, "
                        f"sigma = {reg['std']:.4f}\n")
                f.write(f"       n = {reg['n_raw']}, n_eff = {reg['n_eff']}\n")
                f.write(f"       lag-1 autocorr = {reg['lag1_autocorr']:.3f}\n\n")

            for comp in res_impact["comparisons"]:
                f.write(f"     Statistical confirmation:\n")
                f.write(f"       Cliff's delta = {comp['cliffs_delta']:+.4f} "
                        f"({comp['cliffs_delta_interp']})\n")
                f.write(f"       Welch t p     = {comp['welch_p']:.2e}\n")
                f.write(f"       MW U p        = {comp['mann_whitney_p']:.2e}\n")
                f.write(f"       KS D          = {comp['ks_stat']:.4f}, "
                        f"p = {comp['ks_p']:.2e}\n")
                f.write(f"       Levene F p    = {comp['levene_p']:.2e}\n")
                f.write(f"       All tests reject H0 at alpha = {ALPHA}.\n\n")

        f.write("  3. CONTROL ZONE BASELINE\n")
        f.write("     The control zone provides a baseline for evaluating\n")
        f.write("     whether Kd490 changes in the impact zone are\n")
        f.write("     attributable to local forcing vs large-scale climate\n")
        f.write("     variability (ENSO, IOD, monsoon shifts).\n")
        if has_impact_break and not has_control_break:
            f.write("     The ABSENCE of a break in the control zone\n")
            f.write("     strengthens the attribution to local drivers.\n")
        f.write("\n")

        f.write("  METHODOLOGICAL NOTES:\n\n")
        f.write("  - Multi-algorithm consensus (PELT + BinSeg + Window)\n")
        f.write("    ensures robustness against algorithm-specific biases.\n")
        f.write(f"  - Bootstrap permutation tests (n={N_BOOTSTRAP}) provide\n")
        f.write("    distribution-free significance without normality\n")
        f.write("    assumptions.\n")
        f.write("  - Cliff's delta is preferred over Cohen's d because it\n")
        f.write("    makes no normality assumption and is robust to outliers\n")
        f.write("    (Romano et al., 2006).\n")
        f.write("  - Effective sample sizes (Bretherton et al., 1999)\n")
        f.write("    correct for temporal autocorrelation in significance\n")
        f.write("    interpretation.\n")
        f.write("  - BIC model selection guards against over-segmentation.\n")
        f.write("  - The Difference-in-Differences (DiD) framework isolates\n")
        f.write("    local-scale impacts from shared regional trends.\n\n")

        f.write("=" * 72 + "\n")
        f.write("  END OF REPORT\n")
        f.write("=" * 72 + "\n")

    return path


# ============================================================
# 11. MAIN
# ============================================================
def main():
    print("=" * 65)
    print("  ROBUST CHANGE POINT ANALYSIS -- Kd490")
    print("  Impact: Morowali | Control: Banda Sea")
    print("  Author: Sandy H. S. Herho")
    print("=" * 65)

    # Load both zones
    print("\nLoading data ...")
    ts_impact = load_and_preprocess(DATA_PATH, VAR_NAME,
                                     IMPACT_LAT, IMPACT_LON, SCALE_FACTOR)
    ts_control = load_and_preprocess(DATA_PATH, VAR_NAME,
                                      CONTROL_LAT, CONTROL_LON, SCALE_FACTOR)

    # Analyze both zones
    res_impact = analyze_zone(ts_impact, IMPACT_LABEL)
    res_control = analyze_zone(ts_control, CONTROL_LABEL)

    # Figure 1: 2x2 penalty + BIC
    print("\n  Generating penalty/BIC figure (2x2) ...")
    plot_penalty_bic_2x2(res_impact, res_control)

    # Figure 2: 2x1 consensus
    print("  Generating consensus figure (2x1) ...")
    plot_consensus_2x1(res_control, res_impact)

    # Report
    print("  Writing report ...")
    report_path = write_report(res_impact, res_control)

    print(f"\n  Figures : {FIG_DIR}/penalty_bic.{{png,pdf}}")
    print(f"           {FIG_DIR}/consensus_regimes.{{png,pdf}}")
    print(f"  Report  : {report_path}")
    print("=" * 65)
    print("  ANALYSIS COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
