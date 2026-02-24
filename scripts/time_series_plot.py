#!/usr/bin/env python
"""
Morowali Coastal Optical Analysis Pipeline - Time Series Plot
Generates publication-quality Kd490 time series with policy timeline markers.

Author : Sandy H. S. Herho
Date   : 2025/02/22
License: MIT
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class MorowaliTimeSeriesPlotter:
    def __init__(self, netcdf_path):
        """Initialize the plotter with file paths and setup directories."""
        self.raw_data_path = netcdf_path

        # Output Directories
        self.figs_dir = "../figs"

        # Timelines for policy changes
        self.smelter_start = pd.to_datetime("2015-04-01")
        self.export_ban = pd.to_datetime("2020-01-01")

        # Initialize placeholders for data
        self.df_combined = None

        self._setup_directories()

    def _setup_directories(self):
        """Create necessary output directories if they don't exist."""
        for directory in [self.figs_dir]:
            os.makedirs(directory, exist_ok=True)
            print(f"Directory verified/created: {directory}")

    def load_and_process_data(self):
        """Load NetCDF, apply area weighting, and extract spatial means."""
        print(f"\nLoading NetCDF from: {self.raw_data_path}...")
        ds = xr.open_dataset(self.raw_data_path)

        # Interpolate missing values
        kd490_clean = ds["KD490"].interpolate_na(dim="time", method="linear")

        # Mathematically Rigorous Area Weighting
        weights = np.cos(np.deg2rad(kd490_clean.latitude))
        weights.name = "weights"

        # Expanded Spatial Slices
        impact_zone = kd490_clean.sel(latitude=slice(-2.92, -2.72), longitude=slice(122.08, 122.28))
        control_zone = kd490_clean.sel(latitude=slice(-2.75, -2.45), longitude=slice(123.00, 123.40))

        # Calculate Area-Weighted Spatial Mean & Scale Units (x100)
        # We are keeping the RAW data as requested (no rolling windows)
        ts_impact = (impact_zone * 100).weighted(weights).mean(dim=["latitude", "longitude"]).to_dataframe(name="Kd490").dropna()
        ts_control = (control_zone * 100).weighted(weights).mean(dim=["latitude", "longitude"]).to_dataframe(name="Kd490").dropna()

        # Merge into a single dataframe for easy stats extraction
        self.df_combined = pd.DataFrame({
            "Impact_Zone": ts_impact["Kd490"],
            "Control_Zone": ts_control["Kd490"]
        })
        print("Data processing complete.")

    def plot_timeseries(self):
        """Generate and save the high-resolution tight-bound publication plot."""
        plt.style.use('seaborn-v0_8-whitegrid')

        # Create figure without a title
        fig, ax = plt.subplots(figsize=(11, 5), dpi=400)

        # Publication Color Scheme
        color_ctrl = "#2962FF" # Royal Blue
        color_imp = "#D50000"  # Crimson Red

        # Plot RAW data
        ax.plot(self.df_combined.index, self.df_combined["Control_Zone"],
                label="Banda Sea (Control)", color=color_ctrl, linewidth=1.5, alpha=0.9)

        ax.plot(self.df_combined.index, self.df_combined["Impact_Zone"],
                label="IMIP Coast (Impact)", color=color_imp, linewidth=1.8, alpha=0.9)

        # Policy Timeline Breaks
        ax.axvline(self.smelter_start, color="#FF6D00", linestyle="--", linewidth=2.5,
                   label="1st Smelter (Apr 2015)")
        ax.axvline(self.export_ban, color="#6200EA", linestyle="-.", linewidth=2.5,
                   label="Export Ban (Jan 2020)")

        # Formatting
        ax.set_ylabel(r"$K_d 490$ [$\times 10^{-2}\ m^{-1}$]", fontsize=13, fontweight='medium')
        ax.set_xlabel(r"Time [months]", fontsize=13, fontweight='medium')

        # TIGHT BOUNDS (xlim and ylim)
        ax.set_xlim(self.df_combined.index.min(), self.df_combined.index.max())

        y_max = self.df_combined.max().max()
        y_min = self.df_combined.min().min()
        ax.set_ylim(y_min - (0.05 * y_max), y_max + (0.05 * y_max)) # 5% padding for aesthetics

        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Short, meaningful legend strictly at the bottom
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4,
                  frameon=True, shadow=True, fontsize=11, facecolor='white',
                  borderpad=0.8, edgecolor='black')

        # Adjust layout to prevent bottom legend clipping
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.20)

        # Save in High-Resolution PDF and PNG
        pdf_path = os.path.join(self.figs_dir, "Morowali_Raw_Policy_Kd490.pdf")
        png_path = os.path.join(self.figs_dir, "Morowali_Raw_Policy_Kd490.png")

        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.savefig(png_path, dpi=400, bbox_inches='tight')

        print(f"Plots saved successfully to: {self.figs_dir}")
        plt.close()

    def run(self):
        """Execute the plotting pipeline."""
        print("Starting Morowali Time Series Plotting Pipeline...")
        self.load_and_process_data()
        self.plot_timeseries()
        print("\nPlotting pipeline finished successfully!")


if __name__ == "__main__":
    # Define the path to your raw dataset
    INPUT_NETCDF = "../raw_data/kd490.nc"

    # Initialize and run the plotter
    plotter = MorowaliTimeSeriesPlotter(netcdf_path=INPUT_NETCDF)
    plotter.run()
