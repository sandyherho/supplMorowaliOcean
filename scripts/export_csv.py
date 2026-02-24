#!/usr/bin/env python
"""
Morowali Coastal Combined Export
Merges Kd490, Temperature (thetao), and Salinity (so) into 3 zone CSVs
and generates comprehensive descriptive statistics reports per zone.

Author : Sandy H. S. Herho
Date   : 2025/02/22
License: MIT
"""

import os
import numpy as np
import pandas as pd
import xarray as xr


class MorowaliCombinedExporter:
    def __init__(self, kd490_path, sal_temp_path):
        """Initialize with paths to both NetCDF files."""
        self.kd490_path = kd490_path
        self.sal_temp_path = sal_temp_path

        # Output Directories
        self.processed_dir = "../processed_data"
        self.reports_dir = "../reports"

        # Timelines for policy changes
        self.smelter_start = pd.to_datetime("2015-04-01")
        self.export_ban = pd.to_datetime("2020-01-01")

        # Zone definitions (same as original pipeline)
        self.zones = {
            "Impact_Zone": {
                "latitude": slice(-2.92, -2.72),
                "longitude": slice(122.08, 122.28)
            },
            "Control_Zone": {
                "latitude": slice(-2.75, -2.45),
                "longitude": slice(123.00, 123.40)
            },
            "Entire_Area": {
                "latitude": slice(-2.92, -2.45),
                "longitude": slice(122.08, 123.40)
            }
        }

        # Store processed dataframes per zone
        self.zone_data = {}

        self._setup_directories()

    def _setup_directories(self):
        """Create necessary output directories if they don't exist."""
        for directory in [self.processed_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)
            print(f"Directory verified/created: {directory}")

    def _weighted_mean(self, da, lat_slice, lon_slice):
        """Compute area-weighted spatial mean for a given zone."""
        subset = da.sel(latitude=lat_slice, longitude=lon_slice)
        weights = np.cos(np.deg2rad(subset.latitude))
        weights.name = "weights"
        return subset.weighted(weights).mean(dim=["latitude", "longitude"])

    def load_and_process_data(self):
        """Load both datasets and build combined DataFrames per zone."""
        # --- Load Kd490 ---
        print(f"\nLoading Kd490 from: {self.kd490_path}...")
        ds_kd = xr.open_dataset(self.kd490_path)
        kd490 = ds_kd["KD490"].interpolate_na(dim="time", method="linear") * 100  # scale

        # --- Load Salinity & Temperature ---
        print(f"Loading Salinity/Temperature from: {self.sal_temp_path}...")
        ds_st = xr.open_dataset(self.sal_temp_path)

        # Select surface level (first depth) if depth dimension exists
        salinity = ds_st["so"]
        temperature = ds_st["thetao"]
        if "depth" in salinity.dims:
            salinity = salinity.isel(depth=0)
            temperature = temperature.isel(depth=0)
            print("Selected surface depth level for salinity and temperature.")

        salinity = salinity.interpolate_na(dim="time", method="linear")
        temperature = temperature.interpolate_na(dim="time", method="linear")

        # --- Process each zone ---
        for zone_name, bounds in self.zones.items():
            lat_sl = bounds["latitude"]
            lon_sl = bounds["longitude"]

            ts_kd = self._weighted_mean(kd490, lat_sl, lon_sl).to_dataframe(name="Kd490").dropna()
            ts_sal = self._weighted_mean(salinity, lat_sl, lon_sl).to_dataframe(name="Salinity").dropna()
            ts_temp = self._weighted_mean(temperature, lat_sl, lon_sl).to_dataframe(name="Temperature").dropna()

            # Merge on time index (inner join keeps only overlapping months)
            df = pd.concat([ts_kd["Kd490"], ts_temp["Temperature"], ts_sal["Salinity"]], axis=1, join="inner")
            self.zone_data[zone_name] = df

        print("Data processing complete.")

    def export_csv(self):
        """Export combined CSVs per zone."""
        for zone_name, df in self.zone_data.items():
            out_path = os.path.join(self.processed_dir, f"{zone_name}_Kd490_Temp_Sal.csv")
            df.to_csv(out_path, index_label="Time")
            print(f"Exported: {out_path}  ({len(df)} months)")

    def generate_descriptive_stats(self):
        """Generate comprehensive descriptive statistics reports per zone."""
        for zone_name, df in self.zone_data.items():
            report_path = os.path.join(self.reports_dir, f"{zone_name}_Descriptive_Stats_Report.txt")

            periods = {
                "ENTIRE PERIOD (1997 - Present)": df,
                "PERIOD 1: Pre-Smelter Baseline (Before Apr 2015)": df[df.index < self.smelter_start],
                "PERIOD 2: Initial Operations (Apr 2015 - Dec 2019)": df[(df.index >= self.smelter_start) & (df.index < self.export_ban)],
                "PERIOD 3: Post-Export Ban Hyper-Expansion (Jan 2020 - Present)": df[df.index >= self.export_ban]
            }

            with open(report_path, "w") as f:
                f.write("=========================================================\n")
                f.write(f" DESCRIPTIVE STATISTICS: {zone_name.replace('_', ' ').upper()}\n")
                f.write(" Kd490 (x10^-2 m^-1) | Temperature (°C) | Salinity (PSU)\n")
                f.write(" Author: Sandy H. S. Herho | Date: 2025/02/22\n")
                f.write("=========================================================\n\n")

                for period_name, data in periods.items():
                    f.write(f"--- {period_name} ---\n")
                    f.write(f"Data Points (Months): {len(data)}\n\n")

                    if len(data) == 0:
                        f.write("No data available for this period.\n")
                        f.write("\n" + "-"*57 + "\n\n")
                        continue

                    # Extract summary stats and add Skewness/Kurtosis
                    stats = data.describe().T[['mean', 'std', 'min', '50%', 'max']]
                    stats.rename(columns={'50%': 'median'}, inplace=True)
                    stats['skewness'] = data.skew()
                    stats['kurtosis'] = data.kurtosis()

                    # Format text perfectly
                    f.write(stats.to_string(float_format="{:.4f}".format))
                    f.write("\n\n" + "-"*57 + "\n\n")

            print(f"Report saved: {report_path}")

    def run(self):
        """Execute the full export pipeline."""
        print("Starting Morowali Combined Export Pipeline...")
        self.load_and_process_data()
        self.export_csv()
        self.generate_descriptive_stats()
        print("\nCombined export pipeline finished successfully!")


if __name__ == "__main__":
    INPUT_KD490 = "../raw_data/kd490.nc"
    INPUT_SAL_TEMP = "../raw_data/salinity_temp.nc"

    exporter = MorowaliCombinedExporter(
        kd490_path=INPUT_KD490,
        sal_temp_path=INPUT_SAL_TEMP
    )
    exporter.run()
