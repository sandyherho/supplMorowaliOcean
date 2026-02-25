#!/usr/bin/env python3
"""
=================================================================
  DIFFERENCE-IN-DIFFERENCES WITH COVARIATES
  Impact vs Control Zone Kd490 | Intervention: May 2019
  Author: Sandy H. S. Herho
=================================================================

Panel regression:
  Kd490_it = α + β₁D_i + β₂P_t + β₃(D_i×P_t)
             + γ₁T_it + γ₂S_it + Σδ_m 1[month(t)=m] + ε_it

  β₃ = causal estimate (excess Kd490 in Impact Zone post-2019
       controlling for T, S, and seasonality)

Inference:
  Primary   — Newey-West HAC standard errors
  Robustness — Block bootstrap (10,000 iterations)

Diagnostics:
  1. Parallel pre-trends (event-study specification)
  2. Placebo treatment dates
  3. Covariate stability
  4. Leave-one-year-out sensitivity
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from scipy import stats

# ── Configuration ──────────────────────────────────────────────
INTERVENTION = "2019-05"
ALPHA = 0.05

# --- DATA PATHS (adjust to your directory structure) -----------
IMPACT_KD490 = "../processed_data/Impact_Zone_Kd490_Temp_Sal.csv"
CONTROL_KD490 = "../processed_data/Control_Zone_Kd490_Temp_Sal.csv"
IMPACT_TS = "../processed_data/Impact_Zone_Kd490_Temp_Sal.csv"
CONTROL_TS = "../processed_data/Control_Zone_Kd490_Temp_Sal.csv"

# Output
FIG_OUT = "../figs/fig_did.png"
FIG_DPI = 400


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING & PANEL CONSTRUCTION
# ═══════════════════════════════════════════════════════════════
def load_panel():
    """Build a stacked panel: Impact + Control, long format.

    Returns DataFrame with columns:
      date, zone, D (1=impact), P (1=post), Kd490, Temp, Sal, month
    """
    # Load all CSVs
    df_imp_kd = pd.read_csv(IMPACT_KD490, index_col=0, parse_dates=True)
    df_ctrl_kd = pd.read_csv(CONTROL_KD490, index_col=0, parse_dates=True)
    df_imp_ts = pd.read_csv(IMPACT_TS, index_col=0, parse_dates=True)
    df_ctrl_ts = pd.read_csv(CONTROL_TS, index_col=0, parse_dates=True)

    # Build Impact panel
    impact = pd.DataFrame({
        "date": df_imp_kd.index,
        "zone": "impact",
        "D": 1,
        "Kd490": df_imp_kd.iloc[:, 0].values,
        "Temp": df_imp_ts["Temperature"].values,
        "Sal": df_imp_ts["Salinity"].values,
    })

    # Build Control panel
    control = pd.DataFrame({
        "date": df_ctrl_kd.index,
        "zone": "control",
        "D": 0,
        "Kd490": df_ctrl_kd.iloc[:, 0].values,
        "Temp": df_ctrl_ts["Temperature"].values,
        "Sal": df_ctrl_ts["Salinity"].values,
    })

    # Stack
    panel = pd.concat([impact, control], ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["P"] = (panel["date"] >= pd.Timestamp(INTERVENTION)).astype(int)
    panel["DP"] = panel["D"] * panel["P"]  # interaction = β₃
    panel["month"] = panel["date"].dt.month
    panel["year"] = panel["date"].dt.year

    # Drop NaN rows
    panel = panel.dropna(subset=["Kd490", "Temp", "Sal"])

    print(f"  Panel: {len(panel)} obs "
          f"({panel['D'].sum()} impact, {(1-panel['D']).sum()} control)")
    print(f"  Date range: {panel['date'].min().strftime('%Y-%m')} "
          f"→ {panel['date'].max().strftime('%Y-%m')}")
    return panel


# ═══════════════════════════════════════════════════════════════
#  CORE DID REGRESSION
# ═══════════════════════════════════════════════════════════════
def run_did(panel, include_temp=True, include_sal=True,
            include_month_fe=True, label="Full"):
    """Run DiD regression with Newey-West HAC standard errors.

    Kd490 = α + β₁D + β₂P + β₃(D×P) + γ₁T + γ₂S + Σδ_m + ε

    Returns:
      OLS result object, β₃ estimate, HAC std error, t-stat, p-value
    """
    # Build design matrix
    cols = ["D", "P", "DP"]
    if include_temp:
        cols.append("Temp")
    if include_sal:
        cols.append("Sal")

    X = panel[cols].copy()

    # Monthly fixed effects (drop January = reference)
    if include_month_fe:
        month_dummies = pd.get_dummies(panel["month"], prefix="m",
                                       drop_first=True, dtype=float)
        X = pd.concat([X, month_dummies], axis=1)

    X = sm.add_constant(X)
    y = panel["Kd490"]

    # OLS with Newey-West HAC standard errors
    # Bandwidth = automatic (Newey-West 1994 rule)
    n_obs = len(y)
    max_lag = int(np.floor(4 * (n_obs / 100) ** (2 / 9)))
    model = OLS(y, X).fit(cov_type="HAC",
                          cov_kwds={"maxlags": max_lag})

    beta3 = model.params["DP"]
    se3 = model.bse["DP"]
    t3 = model.tvalues["DP"]
    p3 = model.pvalues["DP"]
    ci_lo = model.conf_int().loc["DP", 0]
    ci_hi = model.conf_int().loc["DP", 1]

    return {
        "model": model,
        "label": label,
        "beta3": beta3,
        "se": se3,
        "t": t3,
        "p": p3,
        "ci": (ci_lo, ci_hi),
        "nobs": n_obs,
        "r2": model.rsquared,
    }


# ═══════════════════════════════════════════════════════════════
#  BLOCK BOOTSTRAP
# ═══════════════════════════════════════════════════════════════
def block_bootstrap(panel, n_boot=10000, seed=42):
    """Bootstrap β₃ by resampling zone-level time series blocks.

    Bertrand, Duflo & Mullainathan (2004) approach.
    """
    rng = np.random.default_rng(seed)

    impact = panel[panel["D"] == 1].sort_values("date").reset_index(drop=True)
    control = panel[panel["D"] == 0].sort_values("date").reset_index(drop=True)

    n_t = len(impact)  # number of time periods per zone
    boot_betas = np.zeros(n_boot)

    for b in range(n_boot):
        # Resample time indices (block = entire time series for each zone)
        idx = rng.choice(n_t, size=n_t, replace=True)
        boot_impact = impact.iloc[idx].copy()
        boot_control = control.iloc[idx].copy()
        boot_panel = pd.concat([boot_impact, boot_control],
                               ignore_index=True)

        # Refit minimal model
        cols = ["D", "P", "DP", "Temp", "Sal"]
        X = sm.add_constant(boot_panel[cols])
        y = boot_panel["Kd490"]
        try:
            res = OLS(y, X).fit(disp=False)
            boot_betas[b] = res.params["DP"]
        except Exception:
            boot_betas[b] = np.nan

    boot_betas = boot_betas[~np.isnan(boot_betas)]
    ci_lo = np.percentile(boot_betas, 2.5)
    ci_hi = np.percentile(boot_betas, 97.5)
    se_boot = np.std(boot_betas)

    return {
        "betas": boot_betas,
        "mean": np.mean(boot_betas),
        "se": se_boot,
        "ci": (ci_lo, ci_hi),
    }


# ═══════════════════════════════════════════════════════════════
#  DIAGNOSTIC 1: PARALLEL PRE-TRENDS (EVENT STUDY)
# ═══════════════════════════════════════════════════════════════
def event_study(panel):
    """Event-study specification on pre+post period.

    Kd490 = α + β₁D + Σ φ_τ (D × 1[year=τ]) + γ₁T + γ₂S + ε

    Omits the year before intervention as reference.
    Returns φ_τ coefficients with CIs.
    """
    intervention_year = pd.Timestamp(INTERVENTION).year  # 2019
    ref_year = intervention_year - 1  # 2018 = reference

    years = sorted(panel["year"].unique())
    panel = panel.copy()

    # Create year × treatment interactions (omit reference year)
    interaction_cols = []
    for yr in years:
        if yr == ref_year:
            continue
        col = f"D_x_{yr}"
        panel[col] = (panel["D"] * (panel["year"] == yr)).astype(float)
        interaction_cols.append(col)

    # Design matrix
    X = panel[["D", "Temp", "Sal"] + interaction_cols].copy()
    X = sm.add_constant(X)
    y = panel["Kd490"]

    n_obs = len(y)
    max_lag = int(np.floor(4 * (n_obs / 100) ** (2 / 9)))
    model = OLS(y, X).fit(cov_type="HAC",
                          cov_kwds={"maxlags": max_lag})

    # Extract φ_τ
    results = []
    for yr in years:
        if yr == ref_year:
            results.append({"year": yr, "coef": 0.0,
                            "ci_lo": 0.0, "ci_hi": 0.0,
                            "is_ref": True})
            continue
        col = f"D_x_{yr}"
        coef = model.params[col]
        ci = model.conf_int().loc[col]
        results.append({"year": yr, "coef": coef,
                        "ci_lo": ci[0], "ci_hi": ci[1],
                        "is_ref": False})

    df_es = pd.DataFrame(results)

    # Joint F-test on pre-period interactions
    pre_cols = [f"D_x_{yr}" for yr in years
                if yr < intervention_year and yr != ref_year]
    if pre_cols:
        r_matrix = np.zeros((len(pre_cols), len(model.params)))
        for i, col in enumerate(pre_cols):
            r_matrix[i, list(model.params.index).index(col)] = 1
        f_test = model.f_test(r_matrix)
        f_stat = float(f_test.fvalue)
        f_pval = float(f_test.pvalue)
    else:
        f_stat, f_pval = np.nan, np.nan

    print(f"    Pre-trend F-test: F={f_stat:.3f}, p={f_pval:.4f}")
    if f_pval > 0.05:
        print("    → Pre-trends NOT jointly significant (parallel trends hold)")
    else:
        print("    → WARNING: Pre-trends jointly significant")

    return df_es, f_stat, f_pval


# ═══════════════════════════════════════════════════════════════
#  DIAGNOSTIC 2: PLACEBO TREATMENT DATES
# ═══════════════════════════════════════════════════════════════
def placebo_tests(panel):
    """Re-run DiD with fake intervention dates."""
    placebo_dates = ["2005-01", "2008-01", "2012-01", "2015-01"]
    real_date = INTERVENTION
    all_dates = placebo_dates + [real_date]

    results = []
    for date in all_dates:
        panel_tmp = panel.copy()
        panel_tmp["P"] = (panel_tmp["date"] >= pd.Timestamp(date)).astype(int)
        panel_tmp["DP"] = panel_tmp["D"] * panel_tmp["P"]

        res = run_did(panel_tmp, label=date)
        results.append({
            "date": date,
            "beta3": res["beta3"],
            "se": res["se"],
            "ci_lo": res["ci"][0],
            "ci_hi": res["ci"][1],
            "p": res["p"],
            "is_real": (date == real_date),
        })

    df_placebo = pd.DataFrame(results)
    return df_placebo


# ═══════════════════════════════════════════════════════════════
#  DIAGNOSTIC 3: COVARIATE STABILITY
# ═══════════════════════════════════════════════════════════════
def covariate_stability(panel):
    """Run DiD under different covariate specifications."""
    specs = [
        {"include_temp": True,  "include_sal": True,
         "include_month_fe": True,  "label": "Full"},
        {"include_temp": False, "include_sal": True,
         "include_month_fe": True,  "label": "No Temp"},
        {"include_temp": True,  "include_sal": False,
         "include_month_fe": True,  "label": "No Sal"},
        {"include_temp": True,  "include_sal": True,
         "include_month_fe": False, "label": "No Month FE"},
    ]

    results = []
    for spec in specs:
        res = run_did(panel, **spec)
        results.append({
            "spec": res["label"],
            "beta3": res["beta3"],
            "se": res["se"],
            "ci_lo": res["ci"][0],
            "ci_hi": res["ci"][1],
            "p": res["p"],
            "r2": res["r2"],
        })

    df_stab = pd.DataFrame(results)
    return df_stab


# ═══════════════════════════════════════════════════════════════
#  DIAGNOSTIC 4: LEAVE-ONE-YEAR-OUT
# ═══════════════════════════════════════════════════════════════
def leave_one_year_out(panel):
    """Drop each year, re-estimate β₃."""
    years = sorted(panel["year"].unique())
    results = []

    for yr in years:
        panel_tmp = panel[panel["year"] != yr].copy()
        res = run_did(panel_tmp, label=f"Drop {yr}")
        results.append({
            "dropped_year": yr,
            "beta3": res["beta3"],
            "se": res["se"],
            "ci_lo": res["ci"][0],
            "ci_hi": res["ci"][1],
        })

    df_loyo = pd.DataFrame(results)
    return df_loyo


# ═══════════════════════════════════════════════════════════════
#  FIGURE 2: DID DIAGNOSTIC 2×2 PANEL
# ═══════════════════════════════════════════════════════════════
def make_figure(df_es, df_placebo, df_stab, df_loyo,
                main_result, output_path):
    """Publication-quality 2×2 diagnostic panel.

    (a) Event study (pre-trends)
    (b) Placebo tests
    (c) Covariate stability
    (d) Leave-one-year-out
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    c_pt = "#1a1a2e"
    c_sig = "#e63946"
    c_ns = "#2a9d8f"
    c_ref = "#888888"

    intervention_year = pd.Timestamp(INTERVENTION).year

    # ── (a) Event Study ───────────────────────────────────────
    ax = axes[0, 0]
    for _, row in df_es.iterrows():
        color = c_sig if row["year"] >= intervention_year else c_pt
        if row["is_ref"]:
            ax.scatter(row["year"], 0, color=c_ref, marker="D",
                       s=60, zorder=5, label="Reference year")
        else:
            ax.errorbar(row["year"], row["coef"],
                        yerr=[[row["coef"] - row["ci_lo"]],
                              [row["ci_hi"] - row["coef"]]],
                        fmt="o", color=color, capsize=3,
                        markersize=5, linewidth=1.5, zorder=4)
    ax.axhline(0, color=c_ref, linestyle="-", linewidth=0.8, zorder=1)
    ax.axvline(intervention_year - 0.5, color=c_ns, linestyle=":",
               linewidth=1.5)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel(r"$\hat{\phi}_\tau$ (DiD coeff.)", fontsize=9)
    ax.set_title("(a) Event Study — Pre-Trend Validation",
                 fontsize=10, fontweight="bold", loc="left")

    # ── (b) Placebo Tests ─────────────────────────────────────
    ax = axes[0, 1]
    for i, row in df_placebo.iterrows():
        color = c_sig if row["is_real"] else c_pt
        marker = "D" if row["is_real"] else "o"
        ax.errorbar(i, row["beta3"],
                    yerr=[[row["beta3"] - row["ci_lo"]],
                          [row["ci_hi"] - row["beta3"]]],
                    fmt=marker, color=color, capsize=3,
                    markersize=6, linewidth=1.5, zorder=4)
    ax.axhline(0, color=c_ref, linestyle="-", linewidth=0.8, zorder=1)
    ax.set_xticks(range(len(df_placebo)))
    ax.set_xticklabels(df_placebo["date"], rotation=30, fontsize=8)
    ax.set_ylabel(r"$\hat{\beta}_3$", fontsize=9)
    ax.set_title("(b) Placebo Treatment Dates",
                 fontsize=10, fontweight="bold", loc="left")

    # ── (c) Covariate Stability ───────────────────────────────
    ax = axes[1, 0]
    x_pos = range(len(df_stab))
    for i, row in df_stab.iterrows():
        ax.errorbar(i, row["beta3"],
                    yerr=[[row["beta3"] - row["ci_lo"]],
                          [row["ci_hi"] - row["beta3"]]],
                    fmt="s", color=c_pt, capsize=3,
                    markersize=6, linewidth=1.5, zorder=4)
    ax.axhline(main_result["beta3"], color=c_sig, linestyle="--",
               linewidth=0.8, alpha=0.7, label=f"Full β₃={main_result['beta3']:.4f}")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(df_stab["spec"], fontsize=8)
    ax.set_ylabel(r"$\hat{\beta}_3$", fontsize=9)
    ax.set_title("(c) Covariate Stability",
                 fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=7, loc="best")

    # ── (d) Leave-One-Year-Out ────────────────────────────────
    ax = axes[1, 1]
    ax.plot(df_loyo["dropped_year"], df_loyo["beta3"],
            "o-", color=c_pt, markersize=4, linewidth=1.0, zorder=4)
    ax.axhline(main_result["beta3"], color=c_sig, linestyle="--",
               linewidth=0.8, alpha=0.7)
    ax.fill_between(df_loyo["dropped_year"],
                    main_result["beta3"] - main_result["se"],
                    main_result["beta3"] + main_result["se"],
                    color=c_sig, alpha=0.1,
                    label=f"Full β₃ ± 1 SE")
    ax.set_xlabel("Dropped Year", fontsize=9)
    ax.set_ylabel(r"$\hat{\beta}_3$", fontsize=9)
    ax.set_title("(d) Leave-One-Year-Out Sensitivity",
                 fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=7, loc="best")

    for ax in axes.flat:
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved → {output_path}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  DIFFERENCE-IN-DIFFERENCES WITH COVARIATES")
    print("  Impact vs Control Zone Kd490 | Intervention: May 2019")
    print("  Author: Sandy H. S. Herho")
    print("=" * 65)

    # 1 — Load data
    print("\n[1] Building panel dataset ...")
    panel = load_panel()

    # 2 — Main DiD regression
    print("\n[2] Main DiD regression (Newey-West HAC) ...")
    main_res = run_did(panel, label="Full")
    m = main_res["model"]
    print(m.summary())
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  β₃ (causal estimate) = {main_res['beta3']:.6f}")
    print(f"  │  HAC Std. Error       = {main_res['se']:.6f}")
    print(f"  │  t-statistic          = {main_res['t']:.3f}")
    print(f"  │  p-value              = {main_res['p']:.6f}")
    print(f"  │  95% CI               = [{main_res['ci'][0]:.6f}, "
          f"{main_res['ci'][1]:.6f}]")
    print(f"  │  R²                   = {main_res['r2']:.4f}")
    print(f"  │  N                    = {main_res['nobs']}")
    print(f"  └─────────────────────────────────────────┘")

    # 3 — Block bootstrap
    print("\n[3] Block bootstrap (10,000 iterations) ...")
    boot = block_bootstrap(panel, n_boot=10000)
    print(f"    Bootstrap β₃ mean:    {boot['mean']:.6f}")
    print(f"    Bootstrap SE:         {boot['se']:.6f}")
    print(f"    Bootstrap 95% CI:     [{boot['ci'][0]:.6f}, "
          f"{boot['ci'][1]:.6f}]")

    # 4 — Event study (pre-trends)
    print("\n[4] Event study — parallel pre-trends ...")
    df_es, f_stat, f_pval = event_study(panel)
    print(df_es.to_string(index=False))

    # 5 — Placebo tests
    print("\n[5] Placebo treatment dates ...")
    df_placebo = placebo_tests(panel)
    print(df_placebo.to_string(index=False))

    # 6 — Covariate stability
    print("\n[6] Covariate stability ...")
    df_stab = covariate_stability(panel)
    print(df_stab.to_string(index=False))

    spread = df_stab["beta3"].max() - df_stab["beta3"].min()
    print(f"\n    β₃ spread across specs: {spread:.6f}")
    if spread / abs(main_res["beta3"]) < 0.15:
        print("    → β₃ is STABLE across specifications")

    # 7 — Leave-one-year-out
    print("\n[7] Leave-one-year-out sensitivity ...")
    df_loyo = leave_one_year_out(panel)
    loyo_spread = df_loyo["beta3"].max() - df_loyo["beta3"].min()
    print(f"    β₃ range: [{df_loyo['beta3'].min():.6f}, "
          f"{df_loyo['beta3'].max():.6f}]")
    print(f"    Spread: {loyo_spread:.6f}")

    # 8 — Figure
    print("\n[8] Generating Figure 2 ...")
    import os
    os.makedirs(os.path.dirname(FIG_OUT) if os.path.dirname(FIG_OUT) else ".",
                exist_ok=True)
    make_figure(df_es, df_placebo, df_stab, df_loyo, main_res, FIG_OUT)

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
