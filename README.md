# Supplementary Material for "Causal Attribution of Coastal Water Clarity Degradation to Nickel Processing Expansion at the Indonesia Morowali Industrial Park, Sulawesi"

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11%2B-8CAAE6.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557c.svg)](https://matplotlib.org/)
[![xarray](https://img.shields.io/badge/xarray-2023.6%2B-E88D2A.svg)](https://xarray.dev/)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.14%2B-4C72B0.svg)](https://www.statsmodels.org/)
[![ruptures](https://img.shields.io/badge/ruptures-1.1%2B-FF6F00.svg)](https://centre-borelli.github.io/ruptures-docs/)
[![netCDF4](https://img.shields.io/badge/netCDF4-1.6%2B-1E88E5.svg)](https://unidata.github.io/netcdf4-python/)
[![PyGMT](https://img.shields.io/badge/PyGMT-0.10%2B-44AA99.svg)](https://www.pygmt.org/)
[![GlobColour](https://img.shields.io/badge/Data-GlobColour%20L3-0B3D91.svg)](https://hermes.acri.fr/)
[![Sentinel-2](https://img.shields.io/badge/Data-Sentinel--2%2010m-003399.svg)](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
[![GLORYS12v1](https://img.shields.io/badge/Data-GLORYS12v1-009FE3.svg)](https://marine.copernicus.eu/)

## Authors

- **Sandy H. S. Herho* — Department of Earth and Planetary Sciences, University of California, Riverside, CA, USA; School of Systems Science and Industrial Engineering, State University of New York, Binghamton, NY, USA; Center for Agrarian Studies, Bandung Institute of Technology, Indonesia
- **Alfita P. Handayani** — Center for Agrarian Studies; Spatial Systems and Cadaster Research Group, Bandung Institute of Technology, Indonesia
- **Iwan P. Anwar** — Applied and Environmental Oceanography Research Group, Bandung Institute of Technology, Indonesia
- **Faruq Khadami** — Applied and Environmental Oceanography Research Group, Bandung Institute of Technology, Indonesia
- **Karina A. Sujatmiko** — Applied and Environmental Oceanography Research Group, Bandung Institute of Technology, Indonesia
- **Doandy Y. Wibisono** — Department of Civil and Environmental Engineering, Colorado School of Mines, Golden, CO, USA; Brierley Associates, Englewood, CO, USA
- **Rusmawan Suwarman** — Atmospheric Science Research Group, Bandung Institute of Technology, Indonesia
- **Dasapta E. Irawan** — Applied Geology Research Group, Bandung Institute of Technology, Indonesia

\*Corresponding author: sh001@ucr.edu

## Overview

This repository accompanies Herho et al. (202x). We apply Bayesian structural time-series (BSTS) causal inference to 27 years (1998–2024) of satellite-derived diffuse attenuation coefficient at 490 nm, $K_d$(490), to test whether industrial expansion at the Indonesia Morowali Industrial Park (IMIP)—the world's largest integrated nickel processing complex—has causally degraded nearshore water clarity in the adjacent Tolo Bay, Central Sulawesi. A Banda Sea control zone absorbs basin-scale climate variability (ENSO, IOD, monsoons), isolating the local anthropogenic signal. Sentinel-2 land cover analysis independently corroborates the marine findings.

## Repository Structure

```
.
├── LICENSE
├── README.md
├── raw_data/                        # Not tracked (see Data Access)
│   ├── kd490.nc                     # GlobColour Kd(490) multi-sensor L3 monthly, 4 km
│   ├── salinity_temp.nc             # GLORYS12v1 SST & SSS reanalysis
│   └── sentinel2LULC_IMIP.nc        # Esri 10 m annual LULC (2017–2024)
├── processed_data/                  # Area-weighted monthly CSVs (provided)
│   ├── Impact_Zone_Kd490_Temp_Sal.csv
│   ├── Control_Zone_Kd490_Temp_Sal.csv
│   └── Entire_Area_Kd490_Temp_Sal.csv
├── scripts/                         # Analysis pipeline (run from scripts/)
│   ├── export_csv.py                # Extract & merge zone-level time series
│   ├── map.py                       # PyGMT context map + bathymetry report
│   ├── time_series_plot.py          # Raw Kd(490) with policy timeline markers
│   ├── climatology.py               # Monthly climatology (bootstrap median CI)
│   ├── trend_analysis.py            # Theil-Sen / Kendall trend estimation
│   ├── changepoint.py               # Multi-algorithm structural break detection
│   ├── bsts.py                      # BSTS causal impact + robustness checks
│   ├── plotLULC.py                  # Sentinel-2 LULC 2×4 panel map
│   └── intensityLULC.py             # Intensity Analysis of LULC transitions
├── figs/                            # Generated figures (PDF + PNG, 400 dpi)
└── reports/                         # Machine-generated statistical reports
```

## Methods Summary

| Step | Script | Method | Key Output |
|:-----|:-------|:-------|:-----------|
| 1 | `export_csv.py` | Cosine-latitude area weighting; zonal extraction | Zone-level CSVs |
| 2 | `map.py` | PyGMT rendering of SRTM15+V2 bathymetry | Study area map (Fig. 1) |
| 3 | `time_series_plot.py` | Raw time series visualization | $K_d$(490) with policy markers (Fig. 5) |
| 4 | `climatology.py` | Bootstrap median 95% CI ($B = 10{,}000$) | Annual cycle (Fig. 4) |
| 5 | `trend_analysis.py` | Theil-Sen estimator; Kendall's $\tau$ | Trend significance per epoch |
| 6 | `changepoint.py` | PELT + BinSeg + Window consensus; bootstrap permutation ($n = 5{,}000$); Cliff's $\delta$; DiD | Breakpoint detection (Figs. 6–7) |
| 7 | `bsts.py` | Unobserved Components Model with control-zone covariates; placebo rank test ($N = 40$); leave-one-out sensitivity | Causal impact (Fig. 8) |
| 8 | `plotLULC.py` | Esri 10 m Sentinel-2 LULC composites | LULC maps (Fig. 2) |
| 9 | `intensityLULC.py` | Aldwaik & Pontius (2012) three-level Intensity Analysis; QES decomposition; Markov $G$-test | Intensity analysis (Fig. 3) |

## Key Results

- **Consensus breakpoint** at May 2019 in the impact zone ($p < 0.001$, Cliff's $\delta = -0.81$); **no break** in the control zone.
- **BSTS causal effect:** $\bar{\delta} = +0.676 \times 10^{-2}$ m$^{-1}$ (+14.4%, $p = 0.012$); placebo rank $p = 0.000$; leave-one-out robust.
- **Euphotic zone shoaling:** $\Delta Z_\text{eu} = -12.3$ m (from ~98 m to ~85 m).
- **LULC:** Built area expanded 3.8× (12.3 → 46.2 km²); tree cover declined 5.0 percentage points; two-stage deforestation cascade with exchange-dominated change (55%).

## Data Access

| Dataset | Source | Resolution |
|:--------|:-------|:-----------|
| $K_d$(490) | [CMEMS GlobColour](https://marine.copernicus.eu/) | 4 km, monthly |
| SST & SSS | [GLORYS12v1](https://marine.copernicus.eu/) | 1/12°, monthly |
| LULC | [Esri Land Cover](https://livingatlas.arcgis.com/landcoverexplorer) | 10 m, annual |
| Bathymetry | [SRTM15+V2](https://topex.ucsd.edu/WWW_html/srtm15_plus.html) | 15 arc-sec |

Place downloaded files in `raw_data/` as `kd490.nc`, `salinity_temp.nc`, and `sentinel2LULC_IMIP.nc`. The `processed_data/` CSVs are provided for reproducing Steps 4–7 without the raw NetCDF files.

## Installation

```bash
# Conda (recommended)
conda create -n morowali python=3.11
conda activate morowali
conda install numpy pandas scipy matplotlib xarray netCDF4 statsmodels
conda install -c conda-forge pygmt ruptures

# Or pip
pip install numpy pandas scipy matplotlib xarray netCDF4 statsmodels ruptures pygmt
```

PyGMT requires [GMT ≥ 6.4](https://www.generic-mapping-tools.org/download/).

## Reproducing the Analysis

```bash
cd scripts/

python export_csv.py          # Step 1: zone-level CSV extraction
python map.py                 # Step 2: geospatial context map
python time_series_plot.py    # Step 3: raw time series plot
python climatology.py         # Step 4: monthly climatology
python trend_analysis.py      # Step 5: trend estimation
python changepoint.py         # Step 6: structural break detection
python bsts.py                # Step 7: BSTS causal impact
python plotLULC.py            # Step 8: LULC maps
python intensityLULC.py       # Step 9: intensity analysis
```

Steps 4–7 can run independently using the provided `processed_data/` CSVs. Steps 1–3 and 8–9 require the raw NetCDF files.

## Citation

```bibtex
@article{herho2026causal,
  title     = {{Causal Attribution of Coastal Water Clarity Degradation to
               Nickel Processing Expansion at the Indonesia Morowali
               Industrial Park}, Sulawesi}},
  author    = {Herho, Sandy H. S. and Handayani, Alfita P. and Anwar, Iwan P.
               and Khadami, Faruq and Sujatmiko, Karina A. and Wibisono,
               Doandy Y. and Suwarman, Rusmawan and Irawan, Dasapta E.},
  journal   = {xxxx},
  year      = {202x}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

Copyright © 2026 Center for Agrarian Studies, Bandung Institute of Technology (ITB)
