#!/usr/bin/env python3
"""
=================================================================
  BAYESIAN STRUCTURAL TIME SERIES — CausalImpact
  Impact Zone Kd490 | Intervention: May 2019
  Author: Sandy H. S. Herho
=================================================================

Brodersen et al. (2015) Bayesian state-space model:
  Observation:  y_t = Z_t'α_t + β'X_t + ε_t
  State:        α_{t+1} = T_t α_t + R_t η_t

Covariates (Control Zone only — unaffected by treatment):
  X1 = Control Zone Kd490
  X2 = Control Zone Temperature
  X3 = Control Zone Salinity

Intervention date: May 2019
Pre-period:  1998-01 → 2019-04  (256 months)
Post-period: 2019-05 → 2024-12  ( 68 months)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from causalimpact import CausalImpact
from scipy import stats

# ── Configuration ──────────────────────────────────────────────
INTERVENTION = "2019-05"
PRE_END = "2019-04"
ALPHA = 0.05

# --- DATA PATHS (adjust to your directory structure) -----------
# Option A: single processed CSV with columns:
#   [y, X_Kd490_ctrl, X_Temp_ctrl, X_Sal_ctrl]
PROCESSED_CSV = None  # e.g., "data/processed/bsts_input.csv"

# Option B: separate raw CSVs
IMPACT_KD490  = "../processed_data/Impact_Zone_Kd490_Raw.csv"
CONTROL_KD490 = "../processed_dataControl_Zone_Kd490_Raw.csv"
CONTROL_TS    = "../processed_data/Control_Zone_Kd490_Temp_Sal.csv"

# Output
FIG_OUT = "../figs/fig_causalimpact.png"
FIG_DPI = 400


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_data():
    """Load and merge data into CausalImpact input format.

    Returns DataFrame with DatetimeIndex and columns:
      y              — Impact Zone Kd490 (response)
      X_Kd490_ctrl   — Control Zone Kd490
      X_Temp_ctrl    — Control Zone Temperature
      X_Sal_ctrl     — Control Zone Salinity
    """
    if PROCESSED_CSV is not None:
        data = pd.read_csv(PROCESSED_CSV, index_col=0, parse_dates=True)
        data.index = pd.DatetimeIndex(data.index, freq="MS")
        print(f"  Loaded {len(data)} obs from processed CSV: "
              f"{data.index[0].strftime('%Y-%m')} to "
              f"{data.index[-1].strftime('%Y-%m')}")
        print(f"  Columns: {list(data.columns)}")
        return data

    # Load separate CSVs
    df_imp = pd.read_csv(IMPACT_KD490, index_col=0, parse_dates=True)
    df_ctrl = pd.read_csv(CONTROL_KD490, index_col=0, parse_dates=True)
    df_ctrl_ts = pd.read_csv(CONTROL_TS, index_col=0, parse_dates=True)

    # Build the CausalImpact input DataFrame
    # First column MUST be the response (y)
    data = pd.DataFrame(index=df_imp.index)
    data["y"] = df_imp.iloc[:, 0]
    data["X_Kd490_ctrl"] = df_ctrl.iloc[:, 0]
    data["X_Temp_ctrl"] = df_ctrl_ts["Temperature"]
    data["X_Sal_ctrl"] = df_ctrl_ts["Salinity"]

    # Ensure DatetimeIndex with monthly freq
    data.index = pd.DatetimeIndex(data.index, freq="MS")
    data = data.dropna()

    print(f"  Loaded {len(data)} obs from raw CSVs: "
          f"{data.index[0].strftime('%Y-%m')} to "
          f"{data.index[-1].strftime('%Y-%m')}")
    print(f"  Columns: {list(data.columns)}")
    return data


# ═══════════════════════════════════════════════════════════════
#  CAUSAL IMPACT MODEL
# ═══════════════════════════════════════════════════════════════
def run_causal_impact(data):
    """Fit Bayesian structural time-series via CausalImpact.

    Key: nseasons goes inside model_args, NOT as a direct kwarg.
    """
    pre_period = [data.index[0], pd.Timestamp(PRE_END)]
    post_period = [pd.Timestamp(INTERVENTION), data.index[-1]]

    ci = CausalImpact(
        data,
        pre_period=pre_period,
        post_period=post_period,
        model_args={
            "nseasons": 12,          # annual seasonality (12 months)
            "season_duration": 1,     # each season = 1 obs (month)
            "prior_level_sd": 0.01,   # diffuse local level prior
            "standardize_data": True,
            "niter": 1000,
        },
        alpha=ALPHA,
    )
    return ci, pre_period, post_period


# ═══════════════════════════════════════════════════════════════
#  DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════
def run_diagnostics(ci, data, pre_period):
    """Pre-period fit diagnostics."""
    inf = ci.inferences
    pre_mask = inf.index <= pre_period[1]

    # Pre-period residuals
    resid_pre = (inf.loc[pre_mask, "response"]
                 - inf.loc[pre_mask, "point_pred"])

    # Ljung-Box test for residual autocorrelation
    lb_stat, lb_p = stats.boxcox_llf  # placeholder
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_result = acorr_ljungbox(resid_pre.dropna(), lags=[12], return_df=True)
    lb_stat = lb_result["lb_stat"].values[0]
    lb_p = lb_result["lb_pvalue"].values[0]

    # Pre-period R²
    ss_res = np.sum(resid_pre.dropna() ** 2)
    y_pre = inf.loc[pre_mask, "response"].dropna()
    ss_tot = np.sum((y_pre - y_pre.mean()) ** 2)
    r2_pre = 1 - ss_res / ss_tot

    # Pre-period MAPE
    mape_pre = np.mean(np.abs(resid_pre.dropna() / y_pre)) * 100

    return {
        "r2_pre": r2_pre,
        "mape_pre": mape_pre,
        "ljung_box_stat": lb_stat,
        "ljung_box_p": lb_p,
        "resid_pre": resid_pre,
    }


# ═══════════════════════════════════════════════════════════════
#  PRIOR SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════
def prior_sensitivity(data):
    """Re-run CausalImpact with different prior_level_sd values.

    If cumulative effect is invariant across priors → robust.
    """
    pre_period = [data.index[0], pd.Timestamp(PRE_END)]
    post_period = [pd.Timestamp(INTERVENTION), data.index[-1]]

    priors = [0.001, 0.01, 0.05, 0.1]
    results = {}

    for sd in priors:
        ci_tmp = CausalImpact(
            data,
            pre_period=pre_period,
            post_period=post_period,
            model_args={
                "nseasons": 12,
                "season_duration": 1,
                "prior_level_sd": sd,
                "standardize_data": True,
                "niter": 1000,
            },
            alpha=ALPHA,
        )
        inf = ci_tmp.inferences
        post_mask = inf.index >= pd.Timestamp(INTERVENTION)
        cum_eff = inf.loc[post_mask, "cum_effect"].iloc[-1]
        results[sd] = cum_eff
        print(f"    prior_level_sd={sd:.3f}  →  cumulative effect = {cum_eff:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════
#  FIGURE 1: CAUSAL IMPACT 3-PANEL
# ═══════════════════════════════════════════════════════════════
def make_figure(ci, pre_period, post_period, diagnostics, output_path):
    """Publication-quality 3×1 panel figure.

    (a) Observed vs Counterfactual
    (b) Pointwise Causal Effect
    (c) Cumulative Causal Effect
    """
    inf = ci.inferences
    idx = inf.index
    intervention_dt = pd.Timestamp(INTERVENTION)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                             gridspec_kw={"hspace": 0.12})

    # ── Color scheme ──
    c_obs = "#1a1a2e"
    c_pred = "#e63946"
    c_band = "#e63946"
    c_zero = "#888888"
    c_interv = "#2a9d8f"

    # ── Panel (a): Observed vs Counterfactual ──────────────────
    ax = axes[0]
    ax.plot(idx, inf["response"], color=c_obs, linewidth=1.0,
            label="Observed", zorder=3)
    ax.plot(idx, inf["point_pred"], color=c_pred, linewidth=1.0,
            linestyle="--", label="Counterfactual", zorder=3)
    ax.fill_between(idx,
                    inf["point_pred_lower"],
                    inf["point_pred_upper"],
                    color=c_band, alpha=0.15, label="95% CI", zorder=2)
    ax.axvline(intervention_dt, color=c_interv, linestyle=":",
               linewidth=1.5, label="Intervention (May 2019)")
    ax.set_ylabel(r"Kd490 ($\times 10^{-3}$ m$^{-1}$)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.set_title("(a) Observed vs. Counterfactual", fontsize=11,
                 fontweight="bold", loc="left")

    # Annotate pre-period R²
    ax.text(0.98, 0.95, f"Pre-period R² = {diagnostics['r2_pre']:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.8))

    # ── Panel (b): Pointwise Effect ───────────────────────────
    ax = axes[1]
    ax.plot(idx, inf["point_effect"], color=c_obs, linewidth=1.0,
            zorder=3)
    ax.fill_between(idx,
                    inf["point_effect_lower"],
                    inf["point_effect_upper"],
                    color=c_band, alpha=0.15, zorder=2)
    ax.axhline(0, color=c_zero, linestyle="-", linewidth=0.8, zorder=1)
    ax.axvline(intervention_dt, color=c_interv, linestyle=":",
               linewidth=1.5)
    ax.set_ylabel("Pointwise Effect", fontsize=10)
    ax.set_title("(b) Pointwise Causal Effect", fontsize=11,
                 fontweight="bold", loc="left")

    # ── Panel (c): Cumulative Effect ──────────────────────────
    ax = axes[2]
    ax.plot(idx, inf["cum_effect"], color=c_obs, linewidth=1.0,
            zorder=3)
    ax.fill_between(idx,
                    inf["cum_effect_lower"],
                    inf["cum_effect_upper"],
                    color=c_band, alpha=0.15, zorder=2)
    ax.axhline(0, color=c_zero, linestyle="-", linewidth=0.8, zorder=1)
    ax.axvline(intervention_dt, color=c_interv, linestyle=":",
               linewidth=1.5)
    ax.set_ylabel("Cumulative Effect", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_title("(c) Cumulative Causal Effect", fontsize=11,
                 fontweight="bold", loc="left")

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="both", labelsize=9)
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
    print("  BAYESIAN STRUCTURAL TIME SERIES — CausalImpact")
    print("  Impact Zone Kd490 | Intervention: May 2019")
    print("  Author: Sandy H. S. Herho")
    print("=" * 65)

    # 1 — Load data
    print("[1] Loading data ...")
    data = load_data()

    # 2 — Fit model
    print("[2] Fitting BSTS model ...")
    ci, pre_period, post_period = run_causal_impact(data)
    print("    Model fitted successfully.")

    # 3 — Summary
    print("[3] CausalImpact Summary:")
    print(ci.summary())
    print()
    print(ci.summary("report"))

    # 4 — Diagnostics
    print("\n[4] Pre-period diagnostics ...")
    diag = run_diagnostics(ci, data, pre_period)
    print(f"    Pre-period R²:           {diag['r2_pre']:.4f}")
    print(f"    Pre-period MAPE:         {diag['mape_pre']:.2f}%")
    print(f"    Ljung-Box(12) statistic: {diag['ljung_box_stat']:.2f}")
    print(f"    Ljung-Box(12) p-value:   {diag['ljung_box_p']:.4f}")
    if diag["ljung_box_p"] > 0.05:
        print("    → No significant residual autocorrelation (good)")
    else:
        print("    → Residual autocorrelation detected (check model)")

    # 5 — Post-period effect statistics
    print("\n[5] Post-period causal effect ...")
    inf = ci.inferences
    post_mask = inf.index >= pd.Timestamp(INTERVENTION)
    post_inf = inf.loc[post_mask]

    avg_effect = post_inf["point_effect"].mean()
    cum_effect = post_inf["cum_effect"].iloc[-1]
    cum_lower = post_inf["cum_effect_lower"].iloc[-1]
    cum_upper = post_inf["cum_effect_upper"].iloc[-1]

    # Relative effect
    avg_pred = post_inf["point_pred"].mean()
    rel_effect = (avg_effect / avg_pred) * 100 if avg_pred != 0 else np.nan

    print(f"    Avg. pointwise effect:   {avg_effect:.4f}")
    print(f"    Relative effect:         {rel_effect:.1f}%")
    print(f"    Cumulative effect:       {cum_effect:.4f}")
    print(f"    Cumulative 95% CI:       [{cum_lower:.4f}, {cum_upper:.4f}]")

    # Bayesian one-sided tail-area probability
    # (proportion of post-period where effect CI excludes zero)
    sig_months = (post_inf["point_effect_lower"] > 0).sum()
    total_months = len(post_inf)
    print(f"    Months with CI > 0:      {sig_months}/{total_months}")

    # 6 — Prior sensitivity
    print("\n[6] Prior sensitivity analysis ...")
    sens = prior_sensitivity(data)
    values = list(sens.values())
    spread = max(values) - min(values)
    print(f"    Spread across priors: {spread:.4f}")
    if spread / abs(np.mean(values)) < 0.1:
        print("    → Cumulative effect is ROBUST to prior specification")
    else:
        print("    → Some sensitivity to prior (inspect carefully)")

    # 7 — Figure
    print("\n[7] Generating Figure 1 ...")
    import os
    os.makedirs(os.path.dirname(FIG_OUT) if os.path.dirname(FIG_OUT) else ".",
                exist_ok=True)
    make_figure(ci, pre_period, post_period, diag, FIG_OUT)

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
