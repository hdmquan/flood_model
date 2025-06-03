import hashlib
import json
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
import torch

from src.data.fetch.dem import download_dem
from src.data.fetch.landuse import fetch_landuse
from src.utils import PATH


class TerrainData:
    def __init__(self, bbox: Tuple[float, float, float, float], aoi_name: str = None):
        """
        Initialize TerrainData with bounding box coordinates
        Args:
            bbox: Tuple of (min_lon, min_lat, max_lon, max_lat)
            aoi_name: Name of the area of interest (e.g., 'taree')
        """
        self.bbox = bbox
        self.bbox_dict = {
            "xmin": bbox[0],
            "ymin": bbox[1],
            "xmax": bbox[2],
            "ymax": bbox[3],
        }

        if aoi_name is None:
            self.aoi_name = (
                f"area_{abs(hashlib.sha256(str(bbox).encode()).hexdigest())[:8]}"
            )
        else:
            self.aoi_name = aoi_name

        # Initialize data holders
        self.dem_data: Optional[np.ndarray] = None
        self.dem_transform = None
        self.landuse_data: Optional[gpd.GeoDataFrame] = None

        # Setup cache paths
        self._setup_cache_paths()

    def _setup_cache_paths(self):
        """Setup cache directory structure and paths"""
        # Ensure cache directories exist
        PATH.DEM.mkdir(parents=True, exist_ok=True)
        PATH.LANDUSE.mkdir(parents=True, exist_ok=True)

        # Set cache paths
        self.dem_cache = PATH.DEM / f"{self.aoi_name}.tif"
        self.landuse_cache = PATH.LANDUSE / f"{self.aoi_name}.geojson"

        # Cache metadata
        self.dem_meta_cache = PATH.DEM / f"{self.aoi_name}_meta.json"
        self.landuse_meta_cache = PATH.LANDUSE / f"{self.aoi_name}_meta.json"

    def _cache_is_valid(self, meta_path: Path, data_path: Path) -> bool:
        """
        Check if cached data is valid by comparing bbox
        """
        if not (meta_path.exists() and data_path.exists()):
            return False

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            return meta["bbox"] == self.bbox
        except:
            return False

    def fetch_dem(self, force_download: bool = False) -> np.ndarray:
        """
        Fetch DEM data and cache it
        Args:
            force_download: If True, force new download even if cache exists
        Returns:
            numpy array of DEM data
        """
        if not force_download and self._cache_is_valid(
            self.dem_meta_cache, self.dem_cache
        ):
            # Load from cache
            with rasterio.open(self.dem_cache) as src:
                self.dem_data = src.read(1)
                self.dem_transform = src.transform
        else:
            # Download and cache
            download_dem(self.bbox, output_path=str(self.dem_cache))
            with rasterio.open(self.dem_cache) as src:
                self.dem_data = src.read(1)
                self.dem_transform = src.transform

            # Save metadata
            meta = {
                "bbox": self.bbox,
                "timestamp": str(Path(self.dem_cache).stat().st_mtime),
                "shape": self.dem_data.shape,
            }
            with open(self.dem_meta_cache, "w") as f:
                json.dump(meta, f)

        return self.dem_data

    def fetch_landuse(self, force_download: bool = False) -> gpd.GeoDataFrame:
        """
        Fetch landuse data including roads and water bodies
        Args:
            force_download: If True, force new download even if cache exists
        Returns:
            GeoDataFrame containing landuse features
        """
        if not force_download and self._cache_is_valid(
            self.landuse_meta_cache, self.landuse_cache
        ):
            # Load from cache
            self.landuse_data = gpd.read_file(self.landuse_cache)
        else:
            # Download and cache
            self.landuse_data = fetch_landuse(self.bbox_dict)

            # Cache the data
            if not self.landuse_data.empty:
                self.landuse_data.to_file(self.landuse_cache, driver="GeoJSON")

                # Save metadata
                meta = {
                    "bbox": self.bbox,
                    "timestamp": str(Path(self.landuse_cache).stat().st_mtime),
                    "feature_count": len(self.landuse_data),
                }
                with open(self.landuse_meta_cache, "w") as f:
                    json.dump(meta, f)

        return self.landuse_data

    @classmethod
    def from_aoi_name(cls, aoi_name: str) -> "TerrainData":
        """
        Create TerrainData instance from AOI name defined in aoi.yaml
        Args:
            aoi_name: Name of the area of interest (e.g., 'taree')
        Returns:
            TerrainData instance
        """
        with open(PATH.SRC / "aoi.yaml", "r") as f:
            aoi_config = yaml.safe_load(f)

        if aoi_name not in aoi_config:
            raise ValueError(f"AOI '{aoi_name}' not found in aoi.yaml")

        bbox = aoi_config[aoi_name]["bbox"]
        return cls(bbox=tuple(bbox), aoi_name=aoi_name)

    def to_tensor(self, device: str = "cuda") -> Dict[str, torch.Tensor]:
        """
        Convert data to PyTorch tensors for ML model input
        Args:
            device: PyTorch device to put tensors on
        Returns:
            Dictionary of tensors containing DEM and landuse data
        """
        if self.dem_data is None:
            self.fetch_dem()
        if self.landuse_data is None:
            self.fetch_landuse()

        # Normalize DEM data
        dem_tensor = torch.from_numpy(self.dem_data).float()
        dem_tensor = (dem_tensor - dem_tensor.mean()) / dem_tensor.std()

        # Create binary masks for landuse features
        height, width = self.dem_data.shape
        road_mask = torch.zeros((height, width), dtype=torch.float32)
        water_mask = torch.zeros((height, width), dtype=torch.float32)

        # Convert vector data to raster masks
        if not self.landuse_data.empty:
            roads = self.landuse_data[self.landuse_data["feature_type"] == "road"]
            water = self.landuse_data[self.landuse_data["feature_type"] == "water"]

            # Here you might want to add rasterization logic for the vector data
            # This is a placeholder - you'll need to implement proper vector-to-raster conversion
            # based on your specific needs

        return {
            "dem": dem_tensor.to(device),
            "roads": road_mask.to(device),
            "water": water_mask.to(device),
        }

    def visualize(self) -> None:
        """
        Visualize the terrain data with matplotlib
        """
        import matplotlib.pyplot as plt

        if self.dem_data is None:
            self.fetch_dem()
        if self.landuse_data is None:
            self.fetch_landuse()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Plot DEM
        im1 = ax1.imshow(self.dem_data, cmap="terrain")
        ax1.set_title("Digital Elevation Model")
        plt.colorbar(im1, ax=ax1)

        # Plot landuse
        self.landuse_data.plot(ax=ax2, column="feature_type", legend=True)
        ax2.set_title("Landuse Features")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    taree = TerrainData.from_aoi_name("taree")
    taree.visualize()
