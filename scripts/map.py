#!/usr/bin/env python
"""
Morowali Coastal and Regional Geospatial Context Map
Generates a high-resolution 2-panel map (Indonesia & Tolo Bay) using PyGMT.
Extracts and reports bathymetry and elevation statistics.

Author : Sandy H. S. Herho
Date   : 2026/02/22
License: MIT
"""

import os
import sys
import numpy as np
import pygmt

class MorowaliMapGenerator:
    def __init__(self):
        # 1. Output Directories
        self.figs_dir = "../figs"
        self.reports_dir = "../reports"
        self._setup_directories()

        # 2. Coordinates & Bounding Boxes
        self.reg_indo = [90.0, 145.0, -13.0, 8.0]               # Panel A: Indonesia General (expanded)
        self.reg_study = [121.30, 123.80, -3.80, -1.80]         # Panel B: Tolo Bay & Banda Sea
        
        # Exact IMIP Location
        self.imip_lon = 122.16
        self.imip_lat = -2.82
        
        # Polygon Coordinates for Expanded Zones
        # Impact Zone (Red Box)
        self.box_impact_x = [122.08, 122.28, 122.28, 122.08, 122.08]
        self.box_impact_y = [-2.92, -2.92, -2.72, -2.72, -2.92]
        
        # Control Zone (Blue Box)
        self.box_control_x = [123.00, 123.40, 123.40, 123.00, 123.00]
        self.box_control_y = [-2.75, -2.75, -2.45, -2.45, -2.75]

        # 3. Data Grids Placeholder
        self.grid_study = None

    def _setup_directories(self):
        """Create necessary output directories if they don't exist."""
        for directory in [self.figs_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)
            print(f"Directory verified: {directory}")

    def load_grids_safely(self):
        """Loads grids naturally with error catching for corrupted network cache."""
        try:
            print("Downloading/Loading 15 arc-second high-res grid for the study area...")
            self.grid_study = pygmt.datasets.load_earth_relief(
                resolution="15s", 
                region=self.reg_study
            )
        except ValueError:
            print("\n[CRITICAL ERROR] Corrupted GMT Cache Detected.")
            print("A previous network timeout left a corrupted, 0-byte file on your machine.")
            print("Please open your terminal and run the following command to fix this:\n")
            print("    rm -rf ~/.gmt/server/earth/earth_relief\n")
            print("Then run this script again.")
            sys.exit(1)

    def generate_bathymetry_report(self):
        """Extract high-resolution bathymetry data and generate a statistical report."""
        values = self.grid_study.values
        ocean_mask = values < 0
        land_mask = values >= 0
        
        bathymetry = values[ocean_mask]
        elevation = values[land_mask]

        report_path = os.path.join(self.reports_dir, "Bathymetry_Topography_Report.txt")
        
        with open(report_path, "w") as f:
            f.write("=================================================================\n")
            f.write(" GEOMORPHOLOGICAL REPORT: STUDY AREA (Tolo Bay & Banda Sea)\n")
            f.write(" Coordinate Bounding Box : Lon [121.30 to 123.80], Lat [-3.80 to -1.80]\n")
            f.write(" Dataset Resolution      : 15 arc-seconds (~450 meters)\n")
            f.write(" Author: Sandy H. S. Herho | Date: 2026/02/22\n")
            f.write("=================================================================\n\n")
            
            f.write("[ MARINE BATHYMETRY (OCEAN) ]\n")
            f.write(f"Total Ocean Pixels  : {len(bathymetry):,}\n")
            f.write(f"Maximum Depth       : {np.min(bathymetry):.2f} meters\n")
            f.write(f"Average Depth (Mean): {np.mean(bathymetry):.2f} meters\n")
            f.write(f"Median Depth        : {np.median(bathymetry):.2f} meters\n\n")
            
            f.write("[ TERRESTRIAL TOPOGRAPHY (LAND) ]\n")
            f.write(f"Total Land Pixels   : {len(elevation):,}\n")
            f.write(f"Maximum Elevation   : {np.max(elevation):.2f} meters\n")
            f.write(f"Average Elevation   : {np.mean(elevation):.2f} meters\n\n")
            
            f.write("[ TARGETED IMPACT ZONE (IMIP) ]\n")
            f.write("Longitude: 122.08 to 122.28 | Latitude: -2.92 to -2.72\n")
            f.write("Note: This coastline features a rapid transition from steep \n")
            f.write("lateritic hills to a sharp, deep continental shelf, heavily \n")
            f.write("influencing the advection of suspended particulate matter.\n")
            f.write("=================================================================\n")
            
        print(f"Bathymetry Report successfully saved to: {report_path}")

    def generate_maps(self):
        """Generate the publication-ready 2-panel PyGMT plot."""
        print("Generating geospatial figures...")
        
        fig = pygmt.Figure()
        
        # Global color palette for Earth relief
        pygmt.makecpt(cmap="geo", series=[-6000, 3000, 100], continuous=True)

        # 2x1 subplot structure. No autolabel — manual bold centered labels inside panels.
        with fig.subplot(nrows=2, ncols=1, figsize=("16c", "22c"), margins=["0.5c", "0.8c"]):
            
            # ---------------------------------------------------------
            # PANEL A: INDONESIA GENERAL
            # ---------------------------------------------------------
            with fig.set_panel(panel=0):
                fig.basemap(region=self.reg_indo, projection="M?", frame=["WSne", "xa10f5", "ya5f1"])
                
                try:
                    grid_indo = pygmt.datasets.load_earth_relief(resolution="10m", region=self.reg_indo)
                except ValueError:
                    print("\n[CRITICAL ERROR] Corrupted GMT Cache Detected during Panel A generation.")
                    print("Please run: rm -rf ~/.gmt/server/earth/earth_relief")
                    sys.exit(1)
                    
                fig.grdimage(grid=grid_indo, cmap=True, shading=True)

                
                # Plot the Study Area Box (Black) for macro context
                study_x = [self.reg_study[0], self.reg_study[1], self.reg_study[1], self.reg_study[0], self.reg_study[0]]
                study_y = [self.reg_study[2], self.reg_study[2], self.reg_study[3], self.reg_study[3], self.reg_study[2]]
                fig.plot(x=study_x, y=study_y, pen="1.5p,black,-")
                
                # Plot exact IMIP location as a Red Dot
                fig.plot(x=self.imip_lon, y=self.imip_lat, style="c0.35c", fill="red", pen="0.5p,black")
                
                # Bold centered (a) label above the panel
                fig.text(position="TC", text="(a)", font="14p,Helvetica-Bold,black", justify="BC", offset="0c/0.3c", no_clip=True)

            # ---------------------------------------------------------
            # PANEL B: TOLO BAY & BANDA SEA (Study Area)
            # ---------------------------------------------------------
            with fig.set_panel(panel=1):
                fig.basemap(region=self.reg_study, projection="M?", frame=["WSne", "xa1f0.5", "ya0.5f0.25"])
                
                fig.grdimage(grid=self.grid_study, cmap=True, shading=True)

                
                # Plot the Control Zone (Blue Box)
                fig.plot(x=self.box_control_x, y=self.box_control_y, pen="2p,blue")
                fig.text(x=123.20, y=-2.40, text="Control Zone", font="11p,Helvetica-Bold,blue", justify="BC")
                
                # Plot the Impact Zone (Red Box)
                fig.plot(x=self.box_impact_x, y=self.box_impact_y, pen="2p,red")
                fig.text(x=122.18, y=-2.95, text="Impact Zone", font="11p,Helvetica-Bold,red", justify="TC")
                
                # Plot exact IMIP location as a Red Dot inside the Impact Zone
                fig.plot(x=self.imip_lon, y=self.imip_lat, style="c0.35c", fill="red", pen="0.5p,black")
                fig.text(x=121.95, y=-2.82, text="IMIP", font="11p,Helvetica-Bold,black", justify="RM", fill="white@30")
                
                # Bold centered (b) label above the panel
                fig.text(position="TC", text="(b)", font="14p,Helvetica-Bold,black", justify="BC", offset="0c/0.3c", no_clip=True)

        # ---------------------------------------------------------
        # GLOBAL COLORBAR
        # ---------------------------------------------------------
        # Positioned well outside the right edge of figures
        with pygmt.config(
            FONT_LABEL="13p,Helvetica-Bold",
            FONT_ANNOT_PRIMARY="10p,Helvetica",
            MAP_LABEL_OFFSET="10p"
        ):
            fig.colorbar(
                cmap=True, 
                position="JRM+jMC+w16c/0.6c+o3.0c/0c", 
                frame=["x+lElevation / Bathymetry", "y+lmeters"]
            )

        # ---------------------------------------------------------
        # SAVE FIGURES
        # ---------------------------------------------------------
        pdf_path = os.path.join(self.figs_dir, "Study_Area_Map.pdf")
        png_path = os.path.join(self.figs_dir, "Study_Area_Map.png")
        
        fig.savefig(pdf_path)
        fig.savefig(png_path, dpi=400)
        
        print(f"High-resolution maps saved successfully to: {self.figs_dir}")

    def run(self):
        """Execute the mapping and reporting pipeline."""
        print("Starting PyGMT Geospatial Mapping Pipeline...")
        self.load_grids_safely()
        self.generate_bathymetry_report()
        self.generate_maps()
        print("\nPipeline execution finished successfully!")

if __name__ == "__main__":
    generator = MorowaliMapGenerator()
    generator.run()
