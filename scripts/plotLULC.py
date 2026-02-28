#!/usr/bin/env python3
"""
================================================================
  Sentinel-2 Annual LULC — Publication Map  (2 × 4 panel)
  Author : Sandy H. S. Herho
  License: MIT

  Input  : ../raw_data/sentinel2LULC_IMIP.nc
  Output : ../figs/lulc_map_imip.{pdf,png}        (400 dpi)
           ../reports/lulc_percentage_report.txt
================================================================
"""

import os
from datetime import datetime
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.colorbar import ColorbarBase
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────
NCFILE     = "../raw_data/sentinel2LULC_IMIP.nc"
FIGDIR     = "../figs"
REPORTDIR  = "../reports"
OUT_PDF    = os.path.join(FIGDIR,    "lulc_map_imip.pdf")
OUT_PNG    = os.path.join(FIGDIR,    "lulc_map_imip.png")
OUT_REPORT = os.path.join(REPORTDIR, "lulc_percentage_report.txt")
DPI        = 400

os.makedirs(FIGDIR,    exist_ok=True)
os.makedirs(REPORTDIR, exist_ok=True)

# ── global style ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family":         "serif",
    "font.serif":          ["DejaVu Serif", "Times New Roman", "Georgia"],
    "axes.linewidth":      0.7,
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "xtick.major.width":   0.6,
    "ytick.major.width":   0.6,
    "xtick.major.size":    3.0,
    "ytick.major.size":    3.0,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
})

BG_DARK      = "#1C2833"
BG_OCEAN     = "#BFD7EA"
CLOUD_CODE   = 10
PANEL_LABELS = list("abcdefgh")

# ── LULC class table ──────────────────────────────────────────
CLASSES = [
    (1,  "Water",              "#4393C3"),
    (2,  "Trees",              "#1B7837"),
    (4,  "Flooded Vegetation", "#78C679"),
    (5,  "Crops",              "#FEC44F"),
    (7,  "Built Area",         "#D6604D"),
    (8,  "Bare Ground",        "#BF9B7A"),
    (10, "Clouds",             "#BABABA"),
    (11, "Rangeland",          "#A8DDB5"),
]
CODES  = np.array([c[0] for c in CLASSES], dtype=np.int8)
LABELS = [c[1] for c in CLASSES]
HEXCOL = [c[2] for c in CLASSES]
N      = len(CLASSES)

CODE_TO_IDX = {int(c): i for i, c in enumerate(CODES)}
CMAP = mcolors.ListedColormap(HEXCOL, name="lulc", N=N)
NORM = mcolors.Normalize(vmin=-0.5, vmax=N - 0.5)


def remap(arr):
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    for code, idx in CODE_TO_IDX.items():
        out[arr == code] = float(idx)
    return out


def add_scalebar(ax, lon0, lat0, length_deg, label):
    h      = (lat0 - float(ax.get_ylim()[0])) * 0.018
    kw     = dict(transform=ax.transData, zorder=7, solid_capstyle="butt")
    stroke = [pe.withStroke(linewidth=2.8, foreground=BG_DARK)]
    ax.plot([lon0, lon0 + length_deg], [lat0, lat0],
            color="white", lw=2.0, path_effects=stroke, **kw)
    for x in (lon0, lon0 + length_deg):
        ax.plot([x, x], [lat0 - h * 0.4, lat0 + h * 0.4],
                color="white", lw=1.5, path_effects=stroke, **kw)
    ax.text(lon0 + length_deg / 2, lat0 + h * 0.55, label,
            ha="center", va="bottom", fontsize=4.8,
            fontweight="bold", color="white", zorder=7,
            path_effects=[pe.withStroke(linewidth=2.0,
                                        foreground=BG_DARK)])


def add_north_arrow(ax, x=0.928, y=0.845):
    """Pure black north arrow — no shading, no background, no path effects."""
    ax.annotate("", xy=(x, y + 0.068), xytext=(x, y),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black",
                                lw=1.0, mutation_scale=8), zorder=7)
    ax.text(x, y + 0.085, "N", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.0,
            fontweight="bold", color="black", zorder=7)


# ══════════════════════════════════════════════════════════════
#  PERCENTAGE REPORT
# ══════════════════════════════════════════════════════════════

def write_report(lulc, years, out_path):
    SEP  = "=" * 72
    SEP2 = "-" * 72
    L    = []
    w    = L.append

    w(SEP)
    w("  Sentinel-2 Annual LULC — Class Percentage Report")
    w("  IMIP Impact Zone, Morowali, Central Sulawesi, Indonesia")
    w("  Spatial extent : Lon [122.08, 122.28]  Lat [-2.92, -2.72]")
    w(f"  Generated      : {datetime.now():%Y-%m-%d %H:%M:%S}")
    w(SEP)
    w("")
    w("  Notes:")
    w("  - 'Grid pixels'     : total number of raster cells in the domain.")
    w("  - 'Valid pixels'    : grid pixels with a recognised LULC code")
    w("                        (excludes fill / no-data = -128).")
    w("  - 'Non-cloud pixels': valid pixels whose class is NOT Clouds (10).")
    w("  - Percentages are computed relative to total valid pixels.")
    w("  - Non-cloud % is relative to total valid pixels.")
    w("  - Native pixel resolution: 10 m × 10 m = 100 m² = 0.0001 km².")

    # ── pre-scan grid dimensions from first year ──────────────
    raw0       = np.ma.filled(lulc[0], fill_value=-128).astype(np.int16)
    total_grid = int(raw0.size)
    grid_rows  = lulc.shape[1]
    grid_cols  = lulc.shape[2]

    w("")
    w(f"  Grid dimensions    : {grid_rows} rows × {grid_cols} cols"
      f"  =  {total_grid:,} total grid pixels")

    # ── per-year detail ───────────────────────────────────────
    for i, yr in enumerate(years):
        raw       = np.ma.filled(lulc[i], fill_value=-128).astype(np.int16)
        n_grid    = int(raw.size)                          # all pixels
        n_fill    = int(np.sum(raw == -128))               # no-data
        n_valid   = n_grid - n_fill                        # coded pixels
        n_cloud   = int(np.sum(raw == CLOUD_CODE))         # cloud pixels
        n_nocloud = n_valid - n_cloud                      # non-cloud valid
        area_valid   = n_valid   * 1e-4
        area_nocloud = n_nocloud * 1e-4
        pct_cloud    = n_cloud   / n_valid * 100 if n_valid > 0 else 0.0
        pct_nocloud  = n_nocloud / n_valid * 100 if n_valid > 0 else 0.0

        w("")
        w(SEP2)
        w(f"  YEAR {yr}   Panel ({PANEL_LABELS[i]})")
        w(SEP2)
        w(f"  Total grid pixels   : {n_grid:>14,}  "
          f"({grid_rows} rows × {grid_cols} cols)")
        w(f"  No-data (fill=-128) : {n_fill:>14,}")
        w(f"  Valid pixels        : {n_valid:>14,}  "
          f"({area_valid:.4f} km²)")
        w(f"  Cloud pixels (cl=10): {n_cloud:>14,}  "
          f"({pct_cloud:.3f}% of valid)")
        w(f"  Non-cloud pixels    : {n_nocloud:>14,}  "
          f"({pct_nocloud:.3f}% of valid  |  {area_nocloud:.4f} km²)")
        w("")
        w(f"  {'Class':<22} {'Code':>5} {'Pixels':>12} "
          f"{'% valid':>10} {'% non-cld':>11} {'Area (km²)':>13}")
        w(f"  {'-'*22} {'-'*5} {'-'*12} {'-'*10} {'-'*11} {'-'*13}")

        class_rows = {}
        for code, label, _ in CLASSES:
            cnt      = int(np.sum(raw == code))
            pct_v    = cnt / n_valid   * 100 if n_valid   > 0 else 0.0
            pct_nc   = cnt / n_nocloud * 100 if n_nocloud > 0 and code != CLOUD_CODE else float("nan")
            area     = cnt * 1e-4
            class_rows[label] = (cnt, pct_v, area)
            pct_nc_str = f"{pct_nc:>10.3f}%" if not (pct_nc != pct_nc) else f"{'—':>10}"  # nan check
            w(f"  {label:<22} {code:>5} {cnt:>12,} "
              f"{pct_v:>9.3f}% {pct_nc_str} {area:>13.4f}")

        w(f"  {'No-data (fill)':<22} {'—':>5} {n_fill:>12,} "
          f"{'—':>10} {'—':>11} {'—':>13}")
        w(f"  {'TOTAL GRID PIXELS':<22} {'—':>5} {n_grid:>12,} "
          f"{'—':>10} {'—':>11} {n_grid*1e-4:>13.4f}")

        dom = max(class_rows, key=lambda k: class_rows[k][0])
        w("")
        w(f"  Dominant class (by valid pixel count) : "
          f"{dom}  ({class_rows[dom][1]:.3f}%)")
        w(f"  Cloud cover fraction                  : "
          f"{pct_cloud:.3f}%")
        w(f"  Effective (non-cloud) coverage        : "
          f"{pct_nocloud:.3f}%")

    # ── cross-year summary — % of valid pixels ────────────────
    w("")
    w(SEP)
    w("  CROSS-YEAR SUMMARY  —  Class percentage (% of valid pixels) by year")
    w(SEP)
    w("")
    yr_hdr = "".join(f"{y:>9}" for y in years)
    w(f"  {'Class':<22} {yr_hdr}")
    w(f"  {'-'*22}" + "-" * 9 * len(years))

    for code, label, _ in CLASSES:
        row = f"  {label:<22}"
        for i in range(len(years)):
            raw    = np.ma.filled(lulc[i], fill_value=-128).astype(np.int16)
            n_valid = int(np.sum(raw != -128))
            cnt    = int(np.sum(raw == code))
            pct    = cnt / n_valid * 100 if n_valid > 0 else 0.0
            row   += f"{pct:>8.3f}%"
        w(row)

    # ── cross-year non-cloud pixel count ─────────────────────
    w("")
    w(SEP2)
    w("  CROSS-YEAR GRID & CLOUD SUMMARY")
    w(SEP2)
    w(f"  {'Year':<7} {'Grid px':>12} {'Valid px':>12} "
      f"{'Cloud px':>12} {'Cloud%':>9} {'Non-cloud px':>14} "
      f"{'Non-cld%':>10} {'Non-cld km²':>13}")
    w(f"  {'-'*7} {'-'*12} {'-'*12} {'-'*12} {'-'*9} {'-'*14} "
      f"{'-'*10} {'-'*13}")

    for i, yr in enumerate(years):
        raw      = np.ma.filled(lulc[i], fill_value=-128).astype(np.int16)
        n_grid   = int(raw.size)
        n_valid  = int(np.sum(raw != -128))
        n_cloud  = int(np.sum(raw == CLOUD_CODE))
        n_nc     = n_valid - n_cloud
        pct_cl   = n_cloud / n_valid * 100 if n_valid > 0 else 0.0
        pct_nc   = n_nc    / n_valid * 100 if n_valid > 0 else 0.0
        w(f"  {yr:<7} {n_grid:>12,} {n_valid:>12,} "
          f"{n_cloud:>12,} {pct_cl:>8.3f}% {n_nc:>14,} "
          f"{pct_nc:>9.3f}% {n_nc*1e-4:>13.4f}")

    # ── built-area trend ──────────────────────────────────────
    w("")
    w(SEP2)
    w("  BUILT AREA TREND  (key IMIP indicator)")
    w(SEP2)
    built = []
    for i, yr in enumerate(years):
        raw    = np.ma.filled(lulc[i], fill_value=-128).astype(np.int16)
        n_v    = int(np.sum(raw != -128))
        cnt    = int(np.sum(raw == 7))
        pct    = cnt / n_v * 100 if n_v > 0 else 0.0
        built.append(pct)
        w(f"  {yr}: {cnt:>12,} px   {pct:>8.4f}%")

    delta = built[-1] - built[0]
    w("")
    w(f"  Change 2017 → {years[-1]} : {delta:>+8.4f} percentage points")
    w("  " + ("INCREASING — consistent with IMIP expansion."
               if delta > 0 else
               "DECREASING — interpret with caution."
               if delta < 0 else
               "No net change detected."))

    w("")
    w(SEP)
    w("  END OF REPORT")
    w(SEP)

    with open(out_path, "w") as f:
        f.write("\n".join(L))
    print(f"  Report → {out_path}")


# ══════════════════════════════════════════════════════════════
#  FIGURE
#  GridSpec rows:
#    0  — top map panels        (height 1)
#    1  — bottom map panels     (height 1)
#    2  — colorbar title row    (height 0.04)
#    3  — colorbar              (height 0.055)
#  Panel labels (a)–(h) rendered INSIDE each map at top-centre.
# ══════════════════════════════════════════════════════════════

print(f"Reading {NCFILE} …")
with nc.Dataset(NCFILE) as ds:
    lats  = np.array(ds["lat"][:])
    lons  = np.array(ds["lon"][:])
    years = np.array(ds["year"][:]).astype(int)
    lulc  = ds["lulc"][:]

assert len(years) == 8, f"Expected 8 years, got {len(years)}"

LON0, LON1 = float(lons.min()), float(lons.max())
LAT0, LAT1 = float(lats.min()), float(lats.max())
DLON, DLAT = LON1 - LON0, LAT1 - LAT0

xticks = np.round(np.linspace(LON0, LON1, 5), 3)
yticks = np.round(np.linspace(LAT0, LAT1, 4), 3)

def fmt_lon(v): return f"{v:.2f}°E"
def fmt_lat(v): return f"{abs(v):.2f}°S" if v < 0 else f"{v:.2f}°N"

fig = plt.figure(figsize=(18.0, 12.2), dpi=DPI)
fig.patch.set_facecolor("white")

# Row 0 : top panel labels (a)–(d)   — outside above top maps
# Row 1 : top map panels
# Row 2 : bottom panel labels (e)–(h) — outside above bottom maps
# Row 3 : bottom map panels
# Row 4 : colorbar title
# Row 5 : colorbar
gs = GridSpec(
    nrows=6, ncols=4, figure=fig,
    left=0.055, right=0.978,
    top=0.970,  bottom=0.048,
    hspace=0.0, wspace=0.045,
    height_ratios=[0.038, 1, 0.038, 1, 0.032, 0.055],
)

print("Rendering 8 panels …")
for i, yr in enumerate(years):
    map_row = 1 if i < 4 else 3
    col     = i % 4
    ax      = fig.add_subplot(gs[map_row, col])

    ax.set_facecolor(BG_OCEAN)
    raw = np.ma.filled(lulc[i], fill_value=-128).astype(np.float32)
    ax.pcolormesh(lons, lats, remap(raw),
                  cmap=CMAP, norm=NORM,
                  shading="nearest", rasterized=True, zorder=1)

    ax.set_xlim(LON0, LON1)
    ax.set_ylim(LAT0, LAT1)
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_edgecolor(BG_DARK)
        spine.set_linewidth(0.7)

    ax.grid(True, linewidth=0.18, color="#7F8C8D",
            alpha=0.45, linestyle=":", zorder=0)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    if map_row == 3:   # bottom row — x labels
        ax.set_xticklabels(
            [fmt_lon(v) for v in xticks],
            fontsize=5.8, fontweight="bold",
            color=BG_DARK, rotation=30, ha="right")
    else:
        ax.set_xticklabels([])

    if col == 0:       # left column — y labels
        ax.set_yticklabels(
            [fmt_lat(v) for v in yticks],
            fontsize=5.8, fontweight="bold", color=BG_DARK)
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis="both", color=BG_DARK,
                   length=3.0, width=0.55, zorder=5)

    # scale bar
    lat_centre_deg = (LAT0 + LAT1) / 2.0
    sb_len_deg = 0.04
    sb_km = sb_len_deg * abs(np.cos(np.radians(lat_centre_deg))) * 111.32
    sb_label = f"~{sb_km:.1f} km"
    add_scalebar(ax,
                 lon0=LON0 + DLON * 0.04,
                 lat0=LAT0 + DLAT * 0.055,
                 length_deg=sb_len_deg, label=sb_label)

    # north arrow — pure black, no shading
    add_north_arrow(ax)

    # year badge — bottom-centre
    ax.text(0.50, 0.032, str(yr),
            transform=ax.transAxes,
            fontsize=11.0, fontweight="bold",
            color="white", ha="center", va="bottom", zorder=6,
            bbox=dict(boxstyle="round,pad=0.26",
                      facecolor=BG_DARK, edgecolor="none", alpha=0.90))

# ── panel labels: outside above each map, centred ────────────
print("Adding panel labels …")
for col in range(4):
    # top row labels (a)–(d) in gs row 0
    lax = fig.add_subplot(gs[0, col])
    lax.set_axis_off()
    lax.text(0.5, 0.05, f"({PANEL_LABELS[col]})",
             transform=lax.transAxes,
             ha="center", va="bottom",
             fontsize=12, fontweight="bold",
             color=BG_DARK, fontfamily="serif")
    # bottom row labels (e)–(h) in gs row 2
    lax2 = fig.add_subplot(gs[2, col])
    lax2.set_axis_off()
    lax2.text(0.5, 0.05, f"({PANEL_LABELS[col + 4]})",
              transform=lax2.transAxes,
              ha="center", va="bottom",
              fontsize=12, fontweight="bold",
              color=BG_DARK, fontfamily="serif")

# ── colorbar title row ────────────────────────────────────────
print("Adding colorbar …")
title_ax = fig.add_subplot(gs[4, :])
title_ax.set_axis_off()
title_ax.text(0.5, 0.5, "Land Use / Land Cover Class",
              transform=title_ax.transAxes,
              ha="center", va="center",
              fontsize=11, fontweight="bold",
              color=BG_DARK, fontfamily="serif")

# ── colorbar ─────────────────────────────────────────────────
cbar_ax = fig.add_subplot(gs[5, :])
cb = ColorbarBase(cbar_ax, cmap=CMAP, norm=NORM,
                  orientation="horizontal",
                  ticks=np.arange(N), spacing="proportional")
cb.set_ticklabels(LABELS)
cb.ax.tick_params(labelsize=7.8, length=2.8, width=0.55,
                  color=BG_DARK, labelcolor=BG_DARK,
                  bottom=True, top=False)
for lbl in cb.ax.xaxis.get_ticklabels():
    lbl.set_ha("center")
    lbl.set_fontfamily("serif")
    lbl.set_fontweight("bold")
cb.outline.set_linewidth(0.55)
cb.outline.set_edgecolor(BG_DARK)

# ── save ─────────────────────────────────────────────────────
for path in (OUT_PDF, OUT_PNG):
    print(f"Saving {path} …")
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
plt.close(fig)

# ── report ───────────────────────────────────────────────────
print("Writing percentage report …")
write_report(lulc, years, OUT_REPORT)
print("\nDone.")
