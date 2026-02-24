#!/usr/bin/env python
"""
Morowali Coastal Optical Analysis - Pure Statistical Engine
Calculates robust non-parametric descriptive statistics, spatial geometries, 
and trend estimations for Kd490 attenuation without formal hypothesis testing.

Author : Sandy H. S. Herho
Date   : 2026/02/22
License: MIT
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

class RobustStatisticalAnalyzer:
    def __init__(self, netcdf_path):
        self.raw_data_path = netcdf_path
        self.reports_dir = "../reports"
        
        # Policy Timelines
        self.smelter_start = pd.to_datetime("2015-04-01")
        self.export_ban = pd.to_datetime("2020-01-01")
        
        # Spatial Bounding Boxes
        self.bounds = {
            "Entire_Area": {"lat": slice(-3.80, -1.80), "lon": slice(121.30, 123.80)},
            "Impact_Zone": {"lat": slice(-2.92, -2.72), "lon": slice(122.08, 122.28)},
            "Control_Zone": {"lat": slice(-2.75, -2.45), "lon": slice(123.00, 123.40)}
        }
        
        self.df_combined = None
        self.spatial_stats = {}
        self._setup_directories()

    def _setup_directories(self):
        os.makedirs(self.reports_dir, exist_ok=True)
        print(f"Directory verified: {self.reports_dir}")

    def load_and_process_data(self):
        """Loads NetCDF, extracts exact grid geometries, and calculates weighted spatial means."""
        print(f"Loading dataset: {self.raw_data_path}")
        ds = xr.open_dataset(self.raw_data_path)
        kd490_clean = ds["KD490"].interpolate_na(dim="time", method="linear")
        
        weights = np.cos(np.deg2rad(kd490_clean.latitude))
        weights.name = "weights"
        
        # 1. Extract Spatial Grid Shapes and calculate physical areas
        for name, bbox in self.bounds.items():
            sliced = kd490_clean.sel(latitude=bbox["lat"], longitude=bbox["lon"])
            n_lat = len(sliced.latitude)
            n_lon = len(sliced.longitude)
            total_cells = n_lat * n_lon
            
            # Count valid (non-NaN) cells using temporal mean to collapse time dim
            temporal_mean = sliced.mean(dim="time")
            valid_cells = int(np.isfinite(temporal_mean.values).sum())
            nan_cells = total_cells - valid_cells
            
            # Approx 4km x 4km = 16 sq km per pixel (only valid cells)
            area_km2 = valid_cells * 16 
            
            self.spatial_stats[name] = {
                "bbox": bbox,
                "n_lat": n_lat,
                "n_lon": n_lon,
                "total_cells": total_cells,
                "valid_cells": valid_cells,
                "nan_cells": nan_cells,
                "area_km2": area_km2,
                "slice": sliced
            }

        # 2. Calculate Area-Weighted Means (Scaled by 100)
        ts_impact = (self.spatial_stats["Impact_Zone"]["slice"] * 100).weighted(weights).mean(dim=["latitude", "longitude"]).to_dataframe(name="Kd490").dropna()
        ts_control = (self.spatial_stats["Control_Zone"]["slice"] * 100).weighted(weights).mean(dim=["latitude", "longitude"]).to_dataframe(name="Kd490").dropna()
        
        self.df_combined = pd.DataFrame({
            "Impact_Zone": ts_impact["Kd490"],
            "Control_Zone": ts_control["Kd490"]
        })

    def _calculate_mad(self, series):
        """Calculates the robust Median Absolute Deviation (MAD)."""
        median = series.median()
        return np.median(np.abs(series - median))

    def _get_trend_stats(self, series):
        """Gold Standard Non-Parametric Trend Detection."""
        x = np.arange(len(series))
        y = series.values
        res = stats.theilslopes(y, x, 0.95)
        tau, p_value = stats.kendalltau(x, y)
        
        sig = "Highly Significant (p < 0.01)" if p_value < 0.01 else "Significant (p < 0.05)" if p_value < 0.05 else "Not Significant"
        
        return {
            "sen_slope": res[0],
            "tau": tau,
            "p_value": p_value,
            "significance": sig
        }

    def generate_report(self):
        """Compiles the highly detailed robust statistical report."""
        report_path = os.path.join(self.reports_dir, "Comprehensive_Robust_Statistics_Report.txt")
        df = self.df_combined
        
        periods = {
            "ENTIRE PERIOD (1997 - Present)": df,
            "PRE-SMELTER BASELINE (Before Apr 2015)": df[df.index < self.smelter_start],
            "INITIAL OPERATIONS (Apr 2015 - Dec 2019)": df[(df.index >= self.smelter_start) & (df.index < self.export_ban)],
            "POST-BAN HYPER-EXPANSION (Jan 2020 - Present)": df[df.index >= self.export_ban]
        }

        with open(report_path, "w") as f:
            f.write("=================================================================\n")
            f.write(" COMPREHENSIVE ROBUST STATISTICAL REPORT: Kd490 ATTENUATION\n")
            f.write(" Optical values scaled by 10^-2 m^-1\n")
            f.write(" Author: Sandy H. S. Herho | Date: 2026/02/22\n")
            f.write("=================================================================\n\n")
            
            # --- SECTION 1: SPATIAL DOMAIN GEOMETRY ---
            f.write("[ SECTION 1: SPATIAL DOMAIN GRID CELL REPORT (4km Resol.) ]\n")
            f.write("-" * 65 + "\n")
            
            labels = ["ENTIRE AREA (Tolo Bay & Banda Sea)", "IMPACT ZONE (IMIP Coastline & Drift)", "CONTROL ZONE (Banda Sea Baseline)"]
            keys = ["Entire_Area", "Impact_Zone", "Control_Zone"]
            
            for i, (label, key) in enumerate(zip(labels, keys)):
                s = self.spatial_stats[key]
                lat_str = f"[{s['bbox']['lat'].start:.2f} to {s['bbox']['lat'].stop:.2f}]"
                lon_str = f"[{s['bbox']['lon'].start:.2f} to {s['bbox']['lon'].stop:.2f}]"
                
                f.write(f"{i+1}. {label}\n")
                f.write(f"   Coordinates  : Lat {lat_str}, Lon {lon_str}\n")
                f.write(f"   Grid Shape   : {s['n_lat']} Latitudes x {s['n_lon']} Longitudes\n")
                f.write(f"   Total Cells  : {s['total_cells']:,} pixels\n")
                f.write(f"   Valid Cells  : {s['valid_cells']:,} pixels (NaN removed: {s['nan_cells']:,})\n")
                f.write(f"   Physical Area: ~{s['area_km2']:,} km^2 (valid cells only)\n\n")

            # --- SECTION 2: ROBUST DESCRIPTIVE STATISTICS ---
            f.write("[ SECTION 2: ROBUST NON-PARAMETRIC DESCRIPTIVE STATISTICS ]\n")
            f.write("-" * 65 + "\n")
            
            for period_name, data in periods.items():
                f.write(f"--- {period_name} (n={len(data)} months) ---\n")
                for zone in ["Impact_Zone", "Control_Zone"]:
                    series = data[zone]
                    f.write(f"  > {zone}:\n")
                    f.write(f"      Median (Central Tendency) : {series.median():.4f}\n")
                    f.write(f"      MAD (Robust Dispersion)   : {self._calculate_mad(series):.4f}\n")
                    f.write(f"      Min / Max                 : {series.min():.4f} / {series.max():.4f}\n")
                    f.write(f"      Skewness (Asymmetry)      : {series.skew():.4f}\n")
                    f.write(f"      Kurtosis (Tail Extremes)  : {series.kurtosis():.4f}\n")
                f.write("\n")

            # --- SECTION 3: NON-PARAMETRIC TREND ANALYSIS ---
            f.write("[ SECTION 3: TIME-SERIES TREND ESTIMATION (Theil-Sen & Kendall) ]\n")
            f.write("-" * 65 + "\n")
            f.write("Methodology: Theil-Sen estimator derives median slope magnitude, immune to \n")
            f.write("up to 29% outliers. Kendall's Tau evaluates monotonic significance.\n\n")
            
            for period_name, data in periods.items():
                f.write(f"--- {period_name} ---\n")
                for zone in ["Impact_Zone", "Control_Zone"]:
                    t = self._get_trend_stats(data[zone])
                    f.write(f"  > {zone}:\n")
                    f.write(f"      Sen's Slope (Mag/Month): {t['sen_slope']:>8.5f}\n")
                    f.write(f"      Kendall's Tau          : {t['tau']:>8.5f}\n")
                    f.write(f"      P-Value                : {t['p_value']:>8.5e} [{t['significance']}]\n")
                f.write("\n")
            
            # --- SECTION 4: AUTOMATED INTERPRETATION ---
            f.write("[ SECTION 4: SCIENTIFIC INTERPRETATION ]\n")
            f.write("-" * 65 + "\n")
            f.write(f"1. Spatial Context: The Impact Zone contains {self.spatial_stats['Impact_Zone']['valid_cells']} valid pixels \n")
            f.write(f"   (out of {self.spatial_stats['Impact_Zone']['total_cells']} total, {self.spatial_stats['Impact_Zone']['nan_cells']} NaN removed), representing \n")
            f.write(f"   an ocean integration area of ~{self.spatial_stats['Impact_Zone']['area_km2']} km^2 off IMIP.\n")
            f.write(f"   The Control Zone provides a stable ~{self.spatial_stats['Control_Zone']['area_km2']} km^2 pelagic baseline \n")
            f.write(f"   ({self.spatial_stats['Control_Zone']['valid_cells']} valid pixels, {self.spatial_stats['Control_Zone']['nan_cells']} NaN removed).\n\n")
            
            trend_entire = self._get_trend_stats(periods["ENTIRE PERIOD (1997 - Present)"]["Impact_Zone"])
            if trend_entire["p_value"] < 0.05:
                f.write(f"2. Decadal Trend: Over the entire studied epoch, the Impact Zone exhibits a \n")
                f.write(f"   {trend_entire['significance'].lower()} upward monotonic trend (\u03C4 = {trend_entire['tau']:.3f}), \n")
                f.write(f"   indicating persistent, long-term ecosystem alteration.\n\n")
                
            pre_med = periods["PRE-SMELTER BASELINE (Before Apr 2015)"]["Impact_Zone"].median()
            post_med = periods["POST-BAN HYPER-EXPANSION (Jan 2020 - Present)"]["Impact_Zone"].median()
            shift = ((post_med - pre_med) / pre_med) * 100

            f.write(f"3. Policy Impact (Descriptive): The median Kd490 attenuation in the Impact Zone \n")
            f.write(f"   shifted from a natural baseline of {pre_med:.2f} to {post_med:.2f} during the Post-Ban \n")
            f.write(f"   Hyper-Expansion era. This represents a descriptive increase of {shift:.1f}% \n")
            f.write(f"   in central tendency, coinciding with accelerated industrial processing activity.\n")

        print(f"Report fully generated and saved to: {report_path}")

    def run(self):
        print("Starting Robust Statistical Engine...")
        self.load_and_process_data()
        self.generate_report()
        print("Done.")

if __name__ == "__main__":
    analyzer = RobustStatisticalAnalyzer("../raw_data/kd490.nc")
    analyzer.run()
