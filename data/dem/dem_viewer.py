import matplotlib.pyplot as plt
import rasterio
import numpy as np


def display_dem(tif_path):
    # Read the DEM file
    with rasterio.open(tif_path) as src:
        dem = src.read(1)  # Read the first band

        # Create a figure
        plt.figure(figsize=(10, 8))

        # Display the DEM with a colorbar
        im = plt.imshow(dem, cmap="terrain")
        plt.colorbar(im, label="Elevation (m)")

        # Add title and labels
        plt.title("Digital Elevation Model")
        plt.xlabel("Column")
        plt.ylabel("Row")

        # Show the plot
        plt.show()


if __name__ == "__main__":
    from pathlib import Path

    here = Path(__file__).resolve().parent

    dem_file = here / "taree.tif"
    display_dem(dem_file)
