#!/usr/bin/env python3
"""
=================================================================
  BAYESIAN STRUCTURAL TIME SERIES — CausalImpact
  Impact Zone Kd490  |  Intervention: May 2019
  Author: Sandy H. S. Herho
=================================================================
"""

import os, textwrap, warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ── config ──────────────────────────────────────────────────────
DPI          = 400
INTERVENTION = "2019-05-01"
DATADIR      = "../processed_data"
FIGDIR       = "../figs"
REPORTDIR    = "../reports"

matplotlib.rc("font", family="serif", size=9)
matplotlib.rc("axes", linewidth=0.5)
matplotlib.rc("xtick", direction="in", top=True)
matplotlib.rc("ytick", direction="in", right=True)

UNIT = r"[$\times\,10^{-2}$ m$^{-1}$]"


# ════════════════════════════════════════════════════════════════
#  1. DATA
# ════════════════════════════════════════════════════════════════

def load_and_merge(datadir):
    impact = pd.read_csv(
        os.path.join(datadir, "Impact_Zone_Kd490_Temp_Sal.csv"),
        parse_dates=["Time"]).set_index("Time").sort_index()
    control = pd.read_csv(
        os.path.join(datadir, "Control_Zone_Kd490_Temp_Sal.csv"),
        parse_dates=["Time"]).set_index("Time").sort_index()
    df = pd.DataFrame({
        "y":            impact["Kd490"],
        "X_Kd490_ctrl": control["Kd490"],
        "X_Temp_ctrl":  control["Temperature"],
        "X_Sal_ctrl":   control["Salinity"],
    })
    df.index.name = "date"
    df = df.asfreq("MS").dropna()
    return df


# ════════════════════════════════════════════════════════════════
#  2. BSTS
# ════════════════════════════════════════════════════════════════

def fit_bsts(df, pre_end, post_start, post_end):
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    pre    = df.loc[:pre_end]
    post   = df.loc[post_start:post_end]
    full   = df.loc[:post_end]
    y_col  = "y"
    x_cols = [c for c in df.columns if c != y_col]
    n_post = len(post)

    mod = UnobservedComponents(
        pre[y_col], level="local linear trend", seasonal=12,
        exog=pre[x_cols],
        stochastic_level=True, stochastic_trend=True,
        stochastic_seasonal=True)
    res = mod.fit(disp=False, maxiter=1000)

    pred_pre = res.get_prediction()
    pm_pre   = pred_pre.predicted_mean
    ci_pre   = pred_pre.conf_int(alpha=0.05)
    pm_pre.index   = pre.index
    ci_pre.index   = pre.index
    ci_pre.columns = ["lower", "upper"]

    fcast   = res.get_forecast(steps=n_post, exog=post[x_cols])
    pm_post = fcast.predicted_mean
    ci_post = fcast.conf_int(alpha=0.05)
    pm_post.index   = post.index
    ci_post.index   = post.index
    ci_post.columns = ["lower", "upper"]

    pm  = pd.concat([pm_pre, pm_post])
    cil = pd.concat([ci_pre["lower"], ci_post["lower"]])
    ciu = pd.concat([ci_pre["upper"], ci_post["upper"]])

    actual       = full[y_col]
    point_effect = actual - pm
    cumul_effect = point_effect.loc[post_start:].cumsum()

    pe = point_effect.loc[post_start:post_end]
    pp = pm_post
    pa = actual.loc[post_start:post_end]
    pl = ci_post["lower"]
    pu = ci_post["upper"]

    avg_eff = pe.mean()
    se = (pu - pl).mean() / (2 * 1.96)
    p_val = (2 * (1 - sp_stats.norm.cdf(abs(avg_eff) / se))
             if se > 0 else np.nan)

    return dict(
        actual=actual, predicted=pm,
        pred_lower=cil, pred_upper=ciu,
        point_effect=point_effect, cumul_effect=cumul_effect,
        post_avg_eff=avg_eff, post_cum_eff=pe.sum(),
        post_avg_actual=pa.mean(), post_avg_pred=pp.mean(),
        post_avg_lower=pl.mean(), post_avg_upper=pu.mean(),
        rel_eff=(avg_eff / pp.mean() * 100
                 if pp.mean() != 0 else np.nan),
        p_value=p_val,
        pre_resid=(actual.loc[:pre_end] - pm_pre).dropna(),
        pre_end=pre_end, post_start=post_start, post_end=post_end)


def fit_bsts_quick(df, pre_end, post_start, post_end):
    try:
        r = fit_bsts(df, pre_end, post_start, post_end)
        return r["post_avg_eff"], r["p_value"]
    except Exception:
        return np.nan, np.nan


# ════════════════════════════════════════════════════════════════
#  3. FIGURE  — single 3-panel, no title, labels centered
# ════════════════════════════════════════════════════════════════

def plot_main(result, figdir):
    fig, axes = plt.subplots(
        3, 1, figsize=(7.5, 8.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2]})
    fig.subplots_adjust(
        hspace=0.08, left=0.13, right=0.97,
        top=0.98, bottom=0.12)
    intv = pd.Timestamp(INTERVENTION)

    # Trim Kalman burn-in (13 months)
    t0   = result["actual"].index[13]
    act  = result["actual"].loc[t0:]
    pred = result["predicted"].loc[t0:]
    plo  = result["pred_lower"].loc[t0:]
    phi  = result["pred_upper"].loc[t0:]
    pe   = result["point_effect"].loc[t0:]

    # ── (a) Observed vs counterfactual ──
    ax = axes[0]
    h1, = ax.plot(act, "k-", lw=0.9,
                  label=r"Observed $K_d$(490) — impact zone")
    h2, = ax.plot(pred, color="#1f77b4", lw=0.9,
                  label="Counterfactual prediction")
    h3 = ax.fill_between(plo.index, plo, phi,
                         color="#1f77b4", alpha=0.18,
                         label="95% CI")
    ax.axvline(intv, color="red", ls="--", lw=0.8, alpha=0.75)
    ax.set_ylabel(r"$K_d$(490)  " + UNIT)
    ax.text(0.5, 0.95, "(a)", transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="center", va="top")

    # ── (b) Pointwise effect ──
    ax = axes[1]
    ax.fill_between(pe.index, plo - pred, phi - pred,
                    color="#1f77b4", alpha=0.18)
    ax.plot(pe, "k-", lw=0.7)
    ax.axhline(0, color="gray", ls=":", lw=0.5)
    ax.axvline(intv, color="red", ls="--", lw=0.8, alpha=0.75)
    ax.set_ylabel("Pointwise effect  " + UNIT)
    ax.text(0.5, 0.95, "(b)", transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="center", va="top")

    # ── (c) Cumulative effect ──
    ax = axes[2]
    ce = result["cumul_effect"]
    ax.plot(ce, "k-", lw=0.8)
    ax.fill_between(ce.index, 0, ce, where=ce > 0,
                    color="#d62728", alpha=0.2, interpolate=True)
    ax.fill_between(ce.index, 0, ce, where=ce < 0,
                    color="#1f77b4", alpha=0.2, interpolate=True)
    ax.axhline(0, color="gray", ls=":", lw=0.5)
    ax.axvline(intv, color="red", ls="--", lw=0.8, alpha=0.75)
    ax.set_ylabel("Cumulative effect  " + UNIT)
    ax.set_xlabel("Date")
    ax.text(0.5, 0.95, "(c)", transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="center", va="top")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Legend outside bottom ──
    fig.legend(
        handles=[h1, h2, h3],
        loc="lower center", ncol=3, fontsize=8,
        frameon=True, framealpha=0.95, edgecolor="gray",
        bbox_to_anchor=(0.55, 0.005))

    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"bsts_causalimpact.{fmt}"),
                    dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
#  4. REPORT  (all results + robustness in text)
# ════════════════════════════════════════════════════════════════

def write_report(result, rank_p, placebo_effects, sensitivity,
                 sw_p, ljung_p, reportdir):
    L = []
    w = L.append
    w("=" * 70)
    w("  BAYESIAN STRUCTURAL TIME SERIES — CausalImpact")
    w("  Impact Zone Kd490  |  Intervention: May 2019")
    w("  Author: Sandy H. S. Herho")
    w(f"  Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    w("=" * 70)

    w("\n── 1  DATA ────────────────────────────────────────")
    w("  Pre-period  : 1998-01 -> 2019-04  (256 months)")
    w("  Post-period : 2019-05 -> 2024-12  (68 months)")
    w("  Response    : Kd490 in impact zone")
    w("  Covariates  : Kd490_ctrl, SST_ctrl, SSS_ctrl")
    w("  Units       : x10^-2 m^-1")

    w("\n── 2  CAUSAL IMPACT ───────────────────────────────")
    w(f"  Post avg. observed  : {result['post_avg_actual']:.4f}"
      f"  x10^-2 m^-1")
    w(f"  Post avg. predicted : {result['post_avg_pred']:.4f}"
      f"  x10^-2 m^-1")
    w(f"      95% CI          : [{result['post_avg_lower']:.4f},"
      f" {result['post_avg_upper']:.4f}]")
    w(f"  Avg. causal effect  : {result['post_avg_eff']:+.4f}"
      f"  x10^-2 m^-1")
    w(f"  Relative effect     : {result['rel_eff']:+.1f}%")
    w(f"  Cumulative effect   : {result['post_cum_eff']:+.4f}"
      f"  x10^-2 m^-1")
    w(f"  Two-sided p-value   : {result['p_value']:.4f}")

    if result["p_value"] < 0.05:
        d = "increase" if result["post_avg_eff"] > 0 else "decrease"
        w(f"\n  *** STATISTICALLY SIGNIFICANT (p < 0.05) ***")
        w(f"  The intervention is associated with a significant {d}")
        w(f"  in Kd490 in the impact zone.")
    else:
        w("\n  NOT significant at alpha = 0.05.")

    w("\n── 3  PHYSICAL INTERPRETATION ─────────────────────")
    w(textwrap.fill(
        "Kd490 (diffuse attenuation coefficient at 490 nm) is a "
        "satellite-derived proxy for upper-ocean turbidity.  Higher "
        "Kd490 indicates increased light attenuation due to suspended "
        "sediments, CDOM, or phytoplankton biomass.  Values are in "
        "units of x10^-2 m^-1.",
        width=68, initial_indent="  ", subsequent_indent="  "))
    w("")
    w(textwrap.fill(
        "The BSTS model uses control-zone covariates (Kd490, SST, "
        "SSS) to absorb basin-wide natural forcing: ENSO, IOD, "
        "monsoonal cycles, MJO, and inter-annual SST variability.  "
        "By constructing a synthetic counterfactual for the impact "
        "zone, any RESIDUAL divergence after May 2019 must arise "
        "from a LOCAL perturbation rather than large-scale climate "
        "variability.",
        width=68, initial_indent="  ", subsequent_indent="  "))
    w("")

    avg_phys  = result["post_avg_eff"] * 1e-2
    pred_phys = result["post_avg_pred"] * 1e-2
    obs_phys  = result["post_avg_actual"] * 1e-2
    w(textwrap.fill(
        f"The average causal effect of "
        f"{result['post_avg_eff']:+.4f} (x10^-2 m^-1) corresponds "
        f"to {avg_phys:+.6f} m^-1 in absolute terms.  This "
        f"translates to a {result['rel_eff']:+.1f}% relative "
        f"increase in Kd490 over the counterfactual baseline.  "
        f"Such an increase implies a reduction in euphotic depth "
        f"(Zeu ~ 4.6/Kd) from approximately "
        f"{4.6 / pred_phys:.1f} m (predicted) to "
        f"{4.6 / obs_phys:.1f} m (observed).",
        width=68, initial_indent="  ", subsequent_indent="  "))
    w("")

    if result["post_avg_eff"] > 0 and result["p_value"] < 0.05:
        w(textwrap.fill(
            "CONCLUSION: The positive and significant increase in "
            "Kd490 indicates the impact-zone water became "
            "substantially more turbid after the intervention.  "
            "Since basin-wide natural drivers are accounted for by "
            "the control covariates, the most parsimonious "
            "explanation is a LOCAL ANTHROPOGENIC perturbation — "
            "e.g. enhanced sediment/effluent discharge from mining, "
            "smelting, coastal construction, or land-use change in "
            "the catchment.",
            width=68, initial_indent="  -> ",
            subsequent_indent="     "))
    elif result["p_value"] >= 0.05:
        w(textwrap.fill(
            "CONCLUSION: The change is statistically "
            "indistinguishable from natural variability.  No "
            "anthropogenic signal can be confidently attributed.",
            width=68, initial_indent="  -> ",
            subsequent_indent="     "))

    # ── 4. Robustness ──
    w("\n── 4  ROBUSTNESS CHECKS ───────────────────────────")

    # 4a Placebo
    w("\n  4a  Placebo / falsification test")
    vp = [e for e in placebo_effects if not np.isnan(e)]
    w(f"      Number of placebos   : {len(vp)}")
    w(f"      True effect          : {result['post_avg_eff']:+.4f}")
    w(f"      Placebo mean         : {np.mean(vp):+.4f}")
    w(f"      Placebo std          : {np.std(vp):.4f}")
    w(f"      Placebo range        : [{np.min(vp):+.4f},"
      f" {np.max(vp):+.4f}]")
    w(f"      Rank-based p-value   : {rank_p:.4f}")
    w(textwrap.fill(
        "The rank-based p-value is the fraction of placebo effects "
        "whose absolute value equals or exceeds the absolute true "
        "effect.  A small value (< 0.05) confirms the true effect "
        "is anomalous relative to natural pre-period fluctuations.",
        width=68, initial_indent="      ",
        subsequent_indent="      "))
    if rank_p < 0.10:
        w("      -> PASSED: true effect is extreme vs. placebos.")
    else:
        w("      -> CAUTION: true effect within placebo range.")

    # 4b Sensitivity
    w("\n  4b  Leave-one-out covariate sensitivity")
    w(f"      {'Configuration':26s}  {'Effect':>10s}"
      f"  {'p-value':>10s}")
    w(f"      {'-' * 26}  {'-' * 10}  {'-' * 10}")
    for s in sensitivity:
        w(f"      {s['label']:26s}  {s['eff']:+10.4f}"
          f"  {s['p']:10.4f}")
    same_sign = all(
        np.sign(s["eff"]) == np.sign(result["post_avg_eff"])
        for s in sensitivity if not np.isnan(s["eff"]))
    all_sig = all(
        s["p"] < 0.05
        for s in sensitivity if not np.isnan(s["p"]))
    w("")
    w(textwrap.fill(
        "If the sign and significance of the effect remain stable "
        "across all leave-one-out configurations, the result is "
        "robust to covariate selection.",
        width=68, initial_indent="      ",
        subsequent_indent="      "))
    if same_sign and all_sig:
        w("      -> ROBUST: sign and significance stable.")
    elif same_sign:
        w("      -> PARTIALLY ROBUST: sign stable, significance"
          " varies.")
    else:
        w("      -> CAUTION: effect direction changes with"
          " removal.")

    # 4c Diagnostics
    w("\n  4c  Pre-period model diagnostics")
    resid = result["pre_resid"].dropna().values
    sw_stat = sp_stats.shapiro(resid[:5000])[0]
    w(f"      Shapiro-Wilk normality test")
    w(f"        statistic  : {sw_stat:.4f}")
    w(f"        p-value    : {sw_p:.4f}")
    if sw_p > 0.05:
        w("        -> Residuals consistent with normality.")
    else:
        w("        -> Residuals depart from normality; CIs")
        w("           interpreted with caution (common for")
        w("           environmental time series).")
    w(f"      Ljung-Box(12) autocorrelation test")
    w(f"        p-value    : {ljung_p:.4f}")
    if ljung_p > 0.05:
        w("        -> No significant residual autocorrelation.")
    else:
        w("        -> Residual autocorrelation present; model")
        w("           may underfit some seasonal structure.")

    # ── 5. Overall ──
    w("\n── 5  OVERALL ASSESSMENT ──────────────────────────")
    ev = []
    if result["p_value"] < 0.05: ev.append("main_sig")
    if rank_p < 0.10:            ev.append("placebo_pass")
    if same_sign:                ev.append("sensitivity_robust")

    if len(ev) == 3:
        w("  STRONG EVIDENCE of an anthropogenic effect:")
        w("    [x] Statistically significant causal impact"
          " (p < 0.05)")
        w("    [x] Placebo test confirms effect is anomalous")
        w("    [x] Sensitivity analysis confirms robustness")
    elif len(ev) >= 2:
        w(f"  MODERATE EVIDENCE ({len(ev)}/3 robustness checks"
          f" passed):")
        w(f"    {'[x]' if 'main_sig' in ev else '[ ]'}"
          f" Significant causal impact")
        w(f"    {'[x]' if 'placebo_pass' in ev else '[ ]'}"
          f" Placebo test passed")
        w(f"    {'[x]' if 'sensitivity_robust' in ev else '[ ]'}"
          f" Sensitivity robust")
    else:
        w("  WEAK / INSUFFICIENT EVIDENCE:")
        w("    The observed change cannot be robustly")
        w("    distinguished from natural variability.")
    w("\n" + "=" * 70)

    txt = "\n".join(L)
    with open(os.path.join(reportdir, "bsts_report.txt"), "w") as f:
        f.write(txt)
    print(txt)


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(REPORTDIR, exist_ok=True)

    print("=" * 65)
    print("  BAYESIAN STRUCTURAL TIME SERIES — CausalImpact")
    print("  Impact Zone Kd490 | Intervention: May 2019")
    print("=" * 65)

    print("\n[1] Loading data ...")
    df = load_and_merge(DATADIR)
    print(f"    {len(df)} obs: {df.index[0]:%Y-%m} -> "
          f"{df.index[-1]:%Y-%m}")

    intv = pd.Timestamp(INTERVENTION)
    pre_end    = (intv - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    post_start = intv.strftime("%Y-%m-%d")
    post_end   = df.index[-1].strftime("%Y-%m-%d")

    print("\n[2] Fitting BSTS model ...")
    result = fit_bsts(df, pre_end, post_start, post_end)
    print(f"    Avg. effect: {result['post_avg_eff']:+.4f}"
          f" x10^-2 m^-1 ({result['rel_eff']:+.1f}%)")
    print(f"    p-value    : {result['p_value']:.4f}")

    print("\n[3] Plotting figure ...")
    plot_main(result, FIGDIR)
    print(f"    -> {FIGDIR}/bsts_causalimpact.pdf")
    print(f"    -> {FIGDIR}/bsts_causalimpact.png")

    N_PLAC = 40
    print(f"\n[4] Placebo tests (n={N_PLAC}) ...")
    pre_dates = df.loc[:pre_end].index
    cands = list(range(36, len(pre_dates) - 12))
    rng = np.random.default_rng(42)
    sel = sorted(rng.choice(cands,
                            size=min(N_PLAC, len(cands)),
                            replace=False))
    plac_effs = []
    for i, idx in enumerate(sel):
        fi  = pre_dates[idx]
        fpe = (fi - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        fps = fi.strftime("%Y-%m-%d")
        fpe2 = min(fi + pd.DateOffset(months=11),
                   pd.Timestamp(pre_end)).strftime("%Y-%m-%d")
        e, p = fit_bsts_quick(df, fpe, fps, fpe2)
        plac_effs.append(e)
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(sel)}] {fi:%Y-%m}"
                  f"  eff={e:+.4f}")
    valid = [e for e in plac_effs if not np.isnan(e)]
    rank_p = np.mean([abs(e) >= abs(result["post_avg_eff"])
                      for e in valid])
    print(f"    Rank-based p = {rank_p:.4f}")

    print("\n[5] Leave-one-out sensitivity ...")
    covs = ["X_Kd490_ctrl", "X_Temp_ctrl", "X_Sal_ctrl"]
    sens = []
    for label, keep in (
        [("All covariates", covs)] +
        [(f"Drop {c}", [x for x in covs if x != c])
         for c in covs]
    ):
        e, p = fit_bsts_quick(df[["y"] + keep],
                              pre_end, post_start, post_end)
        sens.append(dict(label=label, eff=e, p=p))
        print(f"    {label:26s}  eff={e:+.4f}  p={p:.4f}")

    print("\n[6] Diagnostics ...")
    resid = result["pre_resid"].dropna().values
    _, sw_p = sp_stats.shapiro(resid[:5000])
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = acorr_ljungbox(resid, lags=[12], return_df=True)
    ljung_p = lb["lb_pvalue"].values[0]
    print(f"    Shapiro-Wilk  p = {sw_p:.4f}")
    print(f"    Ljung-Box(12) p = {ljung_p:.4f}")

    print("\n[7] Writing report ...")
    write_report(result, rank_p, plac_effs, sens,
                 sw_p, ljung_p, REPORTDIR)
    print(f"    -> {REPORTDIR}/bsts_report.txt")
    print(f"\n    Done.")


if __name__ == "__main__":
    main()
