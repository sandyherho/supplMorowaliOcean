#!/usr/bin/env python
"""
Morowali Monthly Climatology (Annual Cycle) - Entire Area
Computes robust monthly statistics (median, IQR, bootstrap 95% CI)
for Kd490, Temperature, and Salinity over the Entire Area.

Outputs:
  - 3x1 publication-quality figure (PNG 400 dpi + PDF) in ../figs
  - Detailed statistics report (TXT) in ../reports

Author : Sandy H. S. Herho
Date   : 2025/02/22
License: MIT
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

PROCESSED_DIR = "../processed_data"
FIGS_DIR = "../figs"
REPORTS_DIR = "../reports"

CSV_PATH = os.path.join(PROCESSED_DIR, "Entire_Area_Kd490_Temp_Sal.csv")

VARIABLES = {
    "Kd490": {
        "label": r"$K_d490$ [$\times 10^{-2}\ \mathrm{m^{-1}}$]",
        "color": "#D50000",
        "fill": "#D50000",
        "panel": "(a)",
    },
    "Temperature": {
        "label": r"Temperature [$\degree$C]",
        "color": "#2962FF",
        "fill": "#2962FF",
        "panel": "(b)",
    },
    "Salinity": {
        "label": "Salinity [PSU]",
        "color": "#2E7D32",
        "fill": "#2E7D32",
        "panel": "(c)",
    },
}

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

N_BOOT = 10000
CI_LEVEL = 0.95
RNG_SEED = 42


# ============================================================
# Robust Statistics
# ============================================================

def bootstrap_median_ci(data, n_boot=N_BOOT, ci=CI_LEVEL, rng=None):
    """Compute bootstrap confidence interval for the median."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    n = len(data)
    boot_medians = np.array([
        np.median(rng.choice(data, size=n, replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    ci_lo = np.percentile(boot_medians, alpha * 100)
    ci_hi = np.percentile(boot_medians, (1 - alpha) * 100)
    return ci_lo, ci_hi


def compute_monthly_robust_stats(df, var):
    """Compute robust monthly statistics for a single variable."""
    rng = np.random.default_rng(RNG_SEED)
    records = []
    grouped = df.groupby(df.index.month)[var]

    for month in range(1, 13):
        vals = grouped.get_group(month).dropna().values
        n = len(vals)
        med = np.median(vals)
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)
        mad = np.median(np.abs(vals - med))
        skew = pd.Series(vals).skew()
        kurt = pd.Series(vals).kurtosis()
        ci_lo, ci_hi = bootstrap_median_ci(vals, rng=rng)

        records.append({
            "month": month,
            "n": n,
            "mean": mean,
            "std": std,
            "median": med,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "mad": mad,
            "skewness": skew,
            "kurtosis": kurt,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        })

    return pd.DataFrame(records)


# ============================================================
# Report
# ============================================================

def write_report(all_stats):
    """Write monthly climatology statistics to a text report."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "Monthly_Climatology_Robust_Stats.txt")

    with open(report_path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write(" MONTHLY CLIMATOLOGY: ROBUST DESCRIPTIVE STATISTICS\n")
        f.write(" Entire Area (Morowali Coastal Waters)\n")
        f.write(" Bootstrap Median 95% CI  |  IQR  |  MAD  |  Skewness  |  Kurtosis\n")
        f.write(f" Bootstrap iterations: {N_BOOT} | Seed: {RNG_SEED}\n")
        f.write(" Author: Sandy H. S. Herho | Date: 2025/02/22\n")
        f.write("=" * 72 + "\n\n")

        for var, cfg in VARIABLES.items():
            stats_df = all_stats[var]

            f.write(f"--- {var} [{cfg['label'].split('[')[-1] if '[' in cfg['label'] else ''}  ---\n\n")

            header = (
                f"{'Mon':>3s}  {'N':>4s}  {'Mean':>9s}  {'Std':>9s}  "
                f"{'Median':>9s}  {'Q1':>9s}  {'Q3':>9s}  {'IQR':>9s}  "
                f"{'MAD':>9s}  {'CI_lo':>9s}  {'CI_hi':>9s}  "
                f"{'Skew':>8s}  {'Kurt':>8s}"
            )
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            for _, row in stats_df.iterrows():
                line = (
                    f"{MONTH_LABELS[int(row['month'])-1]:>3s}  "
                    f"{int(row['n']):4d}  "
                    f"{row['mean']:9.4f}  {row['std']:9.4f}  "
                    f"{row['median']:9.4f}  {row['q1']:9.4f}  "
                    f"{row['q3']:9.4f}  {row['iqr']:9.4f}  "
                    f"{row['mad']:9.4f}  {row['ci_lo']:9.4f}  "
                    f"{row['ci_hi']:9.4f}  "
                    f"{row['skewness']:8.4f}  {row['kurtosis']:8.4f}"
                )
                f.write(line + "\n")

            f.write("\n" + "-" * 72 + "\n\n")

    print(f"Report saved: {report_path}")


# ============================================================
# Plotting
# ============================================================

def plot_climatology(all_stats):
    """Generate 3x1 publication-quality annual cycle figure."""
    os.makedirs(FIGS_DIR, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), dpi=400)
    months = np.arange(1, 13)

    for ax, (var, cfg) in zip(axes, VARIABLES.items()):
        stats_df = all_stats[var]
        color = cfg["color"]

        med = stats_df["median"].values
        q1 = stats_df["q1"].values
        q3 = stats_df["q3"].values
        ci_lo = stats_df["ci_lo"].values
        ci_hi = stats_df["ci_hi"].values

        # IQR shading
        ax.fill_between(months, q1, q3, alpha=0.15, color=color, label="IQR (Q1\u2013Q3)")

        # Bootstrap 95% CI as error bars on median
        yerr_lo = med - ci_lo
        yerr_hi = ci_hi - med
        ax.errorbar(months, med, yerr=[yerr_lo, yerr_hi],
                    fmt="o-", color=color, markersize=5, linewidth=1.8,
                    capsize=3.5, capthick=1.2, elinewidth=1.0,
                    label="Median \u00b1 95% CI")

        # Y-axis
        ax.set_ylabel(cfg["label"], fontsize=12)

        # X-axis
        ax.set_xticks(months)
        ax.set_xticklabels(MONTH_LABELS, fontsize=10)
        ax.set_xlim(0.5, 12.5)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Panel subtitle centered above each subplot
        ax.set_title(cfg["panel"], fontsize=12, fontweight="bold", pad=8)

        # Remove per-panel legends
        ax.legend().remove()

    # Bottom x-label
    axes[-1].set_xlabel("Month", fontsize=12)

    # Color-neutral shared legend at the bottom
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="0.3", linewidth=1.8, markersize=5,
               markerfacecolor="0.3", label="Median \u00b1 95% CI"),
        Patch(facecolor="0.5", alpha=0.25, edgecolor="0.5", label="IQR (Q1\u2013Q3)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=True,
               fontsize=10, facecolor="white", edgecolor="gray",
               borderpad=0.8)

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    pdf_path = os.path.join(FIGS_DIR, "Monthly_Climatology_Entire_Area.pdf")
    png_path = os.path.join(FIGS_DIR, "Monthly_Climatology_Entire_Area.png")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"Figures saved: {pdf_path}")
    print(f"              {png_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("Starting Monthly Climatology Analysis (Entire Area)...\n")

    print(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH, index_col="Time", parse_dates=True)

    all_stats = {}
    for var in VARIABLES:
        print(f"  Computing robust stats: {var}")
        all_stats[var] = compute_monthly_robust_stats(df, var)

    write_report(all_stats)
    plot_climatology(all_stats)

    print("\nMonthly Climatology Analysis finished successfully!")


if __name__ == "__main__":
    main()
