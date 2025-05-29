import os
import elevation
from src.utils import PATH

# NSW ELVIS for 1-5m resolution


def download_dem(bounds, output_path="dem.tif", product="SRTM1"):
    os.makedirs(PATH.SRC / "cache", exist_ok=True)
    elevation.clip(bounds=bounds, output=str(output_path), product=product)
    print(f"DEM saved to {output_path}")


if __name__ == "__main__":
    import rasterio

    # Bounding box for Taree, NSW (urban area): (min_lon, min_lat, max_lon, max_lat)
    bbox = (152.39938, -31.93833, 152.51154, -31.84501)
    download_dem(bbox, output_path=PATH.DEM / "taree.tif")

    with rasterio.open(PATH.DEM / "taree.tif") as src:
        import matplotlib.pyplot as plt

        plt.imshow(src.read(1), cmap="terrain")
        plt.colorbar()
        plt.show()
