import hashlib
import json
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio import features
from scipy import ndimage
from scipy.ndimage import gaussian_filter, uniform_filter
from whitebox import WhiteboxTools
from skimage.transform import resize
from shapely.geometry import box


from src.data.fetch.dem import download_dem
from src.data.fetch.landuse import fetch_landuse
from src.utils import PATH


class TerrainData:
    def __init__(self, bbox: Tuple[float, float, float, float], aoi_name: str = None):
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

        self.dem_data: Optional[np.ndarray] = None
        self.dem_transform = None
        self.landuse_data: Optional[gpd.GeoDataFrame] = None
        self.flow_acc: Optional[np.ndarray] = None
        self.slope: Optional[np.ndarray] = None

        self._setup_cache_paths()

    def _setup_cache_paths(self):
        PATH.DEM.mkdir(parents=True, exist_ok=True)
        PATH.LANDUSE.mkdir(parents=True, exist_ok=True)

        self.dem_cache = PATH.DEM / f"{self.aoi_name}.tif"
        self.landuse_cache = PATH.LANDUSE / f"{self.aoi_name}.geojson"
        self.footprint_cache = PATH.FOOTPRINT / f"{self.aoi_name}.geojson"

        self.dem_meta_cache = PATH.DEM / f"{self.aoi_name}_meta.json"
        self.landuse_meta_cache = PATH.LANDUSE / f"{self.aoi_name}_meta.json"
        self.footprint_meta_cache = PATH.FOOTPRINT / f"{self.aoi_name}_meta.json"

    def _cache_is_valid(self, meta_path: Path, data_path: Path) -> bool:
        if not (meta_path.exists() and data_path.exists()):
            return False
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            return tuple(meta["bbox"]) == self.bbox
        except:
            return False

    def fetch_dem(self, force_download: bool = False) -> np.ndarray:
        if not force_download and self._cache_is_valid(
            self.dem_meta_cache, self.dem_cache
        ):
            with rasterio.open(self.dem_cache) as src:
                self.dem_data = src.read(1)
                self.dem_transform = src.transform
        else:
            download_dem(self.bbox, output_path=str(self.dem_cache))
            with rasterio.open(self.dem_cache) as src:
                self.dem_data = src.read(1)
                self.dem_transform = src.transform

            meta = {
                "bbox": self.bbox,
                "timestamp": str(Path(self.dem_cache).stat().st_mtime),
                "shape": self.dem_data.shape,
            }
            with open(self.dem_meta_cache, "w") as f:
                json.dump(meta, f)

        return self.dem_data

    def fetch_landuse(self, force_download: bool = False) -> gpd.GeoDataFrame:
        if not force_download and self._cache_is_valid(
            self.landuse_meta_cache, self.landuse_cache
        ):
            self.landuse_data = gpd.read_file(self.landuse_cache)
        else:
            self.landuse_data = fetch_landuse(self.bbox_dict)
            if not self.landuse_data.empty:
                self.landuse_data.to_file(self.landuse_cache, driver="GeoJSON")
                meta = {
                    "bbox": self.bbox,
                    "timestamp": str(Path(self.landuse_cache).stat().st_mtime),
                    "feature_count": len(self.landuse_data),
                }
                with open(self.landuse_meta_cache, "w") as f:
                    json.dump(meta, f)

        return self.landuse_data

    def fetch_footprint(self) -> gpd.GeoDataFrame:
        # TODO: Index for faster slicing
        if not self._cache_is_valid(self.footprint_meta_cache, self.footprint_cache):
            self.footprint_data = self._fetch_footprint()
            self.footprint_data.to_file(self.footprint_cache, driver="GeoJSON")
            meta = {
                "bbox": self.bbox,
                "timestamp": str(Path(self.footprint_cache).stat().st_mtime),
                "feature_count": len(self.footprint_data),
            }
            with open(self.footprint_meta_cache, "w") as f:
                json.dump(meta, f)
        else:
            self.footprint_data = gpd.read_file(self.footprint_cache)

        return self.footprint_data

    def _fetch_footprint(self) -> gpd.GeoDataFrame:
        path = PATH.FOOTPRINT / "source.geojson"

        gdf = gpd.read_file(path)
        bbox_polygon = box(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3])
        gdf = gdf[gdf.geometry.within(bbox_polygon)]

        return gdf

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

    def compute_flow_metrics(self):
        """
        Compute flow accumulation and slope using scipy/numpy instead of richdem
        """
        if self.dem_data is None:
            self.fetch_dem()

        # Fill depressions using a simple minimum filter approach
        dem_filled = self.dem_data.copy()
        dem_filled = ndimage.minimum_filter(dem_filled, size=3)

        # Calculate slope using gradient
        dy, dx = np.gradient(dem_filled)
        self.slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

        # Compute flow accumulation using a simple D8 method
        flow_dir = np.zeros_like(dem_filled)
        height, width = dem_filled.shape

        # Simple flow accumulation calculation
        self.flow_acc = np.zeros_like(dem_filled, dtype="float32")

        # Smooth the DEM slightly to reduce noise
        dem_smooth = gaussian_filter(dem_filled, sigma=1)

        # Calculate flow accumulation using steepest descent
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                window = dem_smooth[i - 1 : i + 2, j - 1 : j + 2]
                center = dem_smooth[i, j]
                if np.any(window < center):
                    # Water flows to lowest neighbor
                    self.flow_acc[i, j] = np.sum(window < center)

        # Normalize flow accumulation
        self.flow_acc = self.flow_acc / np.max(self.flow_acc)

    def rasterize_landuse(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.dem_data is None:
            self.fetch_dem()
        if self.landuse_data is None:
            self.fetch_landuse()

        shape = self.dem_data.shape
        transform = self.dem_transform

        road_mask = np.zeros(shape, dtype=np.uint8)
        water_mask = np.zeros(shape, dtype=np.uint8)

        if not self.landuse_data.empty:
            roads = self.landuse_data[self.landuse_data["feature_type"] == "road"]
            water = self.landuse_data[self.landuse_data["feature_type"] == "water"]

            if not roads.empty:
                road_mask = features.rasterize(
                    [(geom, 1) for geom in roads.geometry],
                    out_shape=shape,
                    transform=transform,
                    fill=0,
                    all_touched=True,
                    dtype=np.uint8,
                )

            if not water.empty:
                water_mask = features.rasterize(
                    [(geom, 1) for geom in water.geometry],
                    out_shape=shape,
                    transform=transform,
                    fill=0,
                    all_touched=True,
                    dtype=np.uint8,
                )

        return road_mask, water_mask

    def generate_susceptibility_map(self) -> np.ndarray:
        if self.flow_acc is None or self.slope is None:
            self.compute_flow_metrics()

        flow_norm = self.flow_acc / np.nanmax(self.flow_acc)
        slope_norm = self.slope / np.nanmax(self.slope)

        # Optional weighting
        fsi = 0.6 * flow_norm + 0.4 * (1 - slope_norm)

        return np.clip(fsi, 0, 1)

    def compute_hand(self, force: bool = False) -> np.ndarray:
        """
        Compute Height Above Nearest Drainage (HAND) using WhiteboxTools.
        Caches intermediate results and only recomputes if forced or missing.
        """
        hand_dir = PATH.RESULTS / "hand"
        hand_dir.mkdir(parents=True, exist_ok=True)

        hand_raster = hand_dir / f"{self.aoi_name}_hand.tif"
        filled_dem = hand_dir / f"{self.aoi_name}_filled.tif"
        flow_acc = hand_dir / f"{self.aoi_name}_flow_acc.tif"
        streams = hand_dir / f"{self.aoi_name}_streams.tif"

        if hand_raster.exists() and not force:
            with rasterio.open(hand_raster) as src:
                hand = src.read(1)
                self.hand = hand
                self.hand_transform = src.transform
                return hand

        if self.dem_data is None:
            self.fetch_dem()

        # Save DEM to GeoTIFF for Whitebox
        dem_temp_path = hand_dir / f"{self.aoi_name}_dem_input.tif"
        with rasterio.open(
            dem_temp_path,
            "w",
            driver="GTiff",
            height=self.dem_data.shape[0],
            width=self.dem_data.shape[1],
            count=1,
            dtype=self.dem_data.dtype,
            crs="EPSG:4326",
            transform=self.dem_transform,
        ) as dst:
            dst.write(self.dem_data, 1)

        # Set up WhiteboxTools
        wbt = WhiteboxTools()
        wbt.work_dir = str(hand_dir)

        # Run DEM conditioning and stream extraction
        wbt.fill_depressions(str(dem_temp_path), str(filled_dem))
        wbt.d8_flow_accumulation(str(filled_dem), str(flow_acc), out_type="cells")
        wbt.extract_streams(str(flow_acc), str(streams), threshold=50)

        # Clean up stream raster: remove nodata and force binary
        with rasterio.open(streams, "r+") as src:
            data = src.read(1)
            data[data == -32768] = 0  # Remove nodata
            data[data > 0] = 1  # Binary mask: stream = 1
            src.write(data, 1)

        # Compute HAND
        result = wbt.run_tool(
            "elevation_above_stream",
            [f"--dem={filled_dem}", f"--streams={streams}", f"--output={hand_raster}"],
        )

        if not hand_raster.exists():
            raise FileNotFoundError(
                f"HAND raster was not created: {hand_raster}\nWhitebox output:\n{result}"
            )

        with rasterio.open(hand_raster) as src:
            hand = src.read(1)
            self.hand = hand
            self.hand_transform = src.transform

        return hand

    def to_tensor(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        if self.dem_data is None:
            self.fetch_dem()
        if self.landuse_data is None:
            self.fetch_landuse()
        if self.flow_acc is None or self.slope is None:
            self.compute_flow_metrics()

        road_mask, water_mask = self.rasterize_landuse()
        susceptibility = self.generate_susceptibility_map()

        dem_tensor = torch.tensor(self.dem_data).float()
        dem_tensor = (dem_tensor - dem_tensor.mean()) / dem_tensor.std()

        if not hasattr(self, "hand"):
            self.compute_hand()

        return {
            "dem": dem_tensor.to(device),
            "flow_acc": torch.tensor(self.flow_acc).float().to(device),
            "slope": torch.tensor(self.slope).float().to(device),
            "roads": torch.tensor(road_mask).float().to(device),
            "water": torch.tensor(water_mask).float().to(device),
            "susceptibility": torch.tensor(susceptibility).float().to(device),
            "hand": torch.tensor(self.hand).float().to(device),
        }

    def visualize(self) -> None:
        """
        Visualize the terrain data with matplotlib in a smooth, presentation-ready style.
        """
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter
        import numpy as np

        if self.dem_data is None:
            self.fetch_dem()
        if self.landuse_data is None:
            self.fetch_landuse()
        if self.flow_acc is None or self.slope is None:
            self.compute_flow_metrics()
        if not hasattr(self, "hand"):
            self.compute_hand()
        if not hasattr(self, "footprint"):
            self.fetch_footprint()

        road_mask, water_mask = self.rasterize_landuse()
        overlay = road_mask + 2 * water_mask

        fig, axs = plt.subplots(2, 2, figsize=(20, 12))

        # DEM
        im0 = axs[0, 0].imshow(self.dem_data, cmap="terrain", interpolation="bilinear")
        axs[0, 0].set_title("DEM", fontsize=14)
        fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

        # Flow Accumulation
        im1 = axs[0, 1].imshow(self.flow_acc, cmap="Blues_r", interpolation="bilinear")
        axs[0, 1].set_title("Flow Accumulation", fontsize=14)
        fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

        # HAND
        # im2 = axs[0, 2].imshow(smooth_hand, cmap="viridis", interpolation="bilinear")
        # axs[0, 2].set_title("HAND (Height Above Drainage)", fontsize=14)
        # fig.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

        # FSI
        smoothed_fsi = uniform_filter(
            self.generate_susceptibility_map().astype("float32"), size=5
        )

        im3 = axs[1, 0].imshow(
            smoothed_fsi,
            cmap="YlOrRd_r",
            interpolation="bilinear",
        )
        axs[1, 0].set_title("Flood Susceptibility Index (FSI)", fontsize=14)
        fig.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)

        # Roads & Water + Building Footprint
        axs[1, 1].imshow(overlay, cmap="gray", interpolation="nearest")
        if not hasattr(self, "footprint"):
            self.footprint = self.fetch_footprint()

        self.footprint.plot(
            ax=axs[1, 1], facecolor="none", edgecolor="red", linewidth=0.5
        )
        axs[1, 1].set_title("Roads, Water & Building Footprints", fontsize=14)

        # Clean axes
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout(pad=2)
        plt.show()


class FloodDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for terrain data to be used with FloodNet.
    Currently supports only 1 sample (full AOI), but built like a PyTorch dataset.
    """

    def __init__(
        self,
        aoi_name: str,
        device: str = "cpu",
        rainfall_pattern: Optional[str] = "center",
    ):
        self.data = TerrainData.from_aoi_name(aoi_name)
        self.tensors = self.data.to_tensor(device=device)
        self.device = device
        self.rainfall = self._generate_rainfall(pattern=rainfall_pattern)

    def _generate_rainfall(self, pattern: str = "center") -> torch.Tensor:
        dem = self.tensors["dem"]
        rainfall = torch.zeros_like(dem).unsqueeze(0)  # Shape: (1, H, W)

        if pattern == "center":
            H, W = dem.shape
            rainfall[:, H // 3 : 2 * H // 3, W // 3 : 2 * W // 3] = 1.0
        elif pattern == "random":
            rainfall.uniform_(0, 1)
        # Add more patterns if needed

        return rainfall.to(self.device).unsqueeze(0)  # Shape: (1, 1, H, W)

    def __len__(self):
        return 1  # Single AOI

    def __getitem__(self, idx):
        print(self.tensors["dem"].shape)
        print(self.tensors["flow_acc"].shape)
        print(self.tensors["slope"].shape)
        print(self.tensors["roads"].shape)
        print(self.tensors["water"].shape)
        print(self.tensors["susceptibility"].shape)

        x = torch.stack(
            [
                self.tensors["dem"],
                self.tensors["flow_acc"],
                self.tensors["slope"],
                self.tensors["roads"],
                self.tensors["water"],
                self.tensors["susceptibility"],
            ],
            dim=0,
        ).unsqueeze(
            0
        )  # Shape: (1, 6, H, W)

        dem = self.tensors["dem"].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        rainfall = self.rainfall  # Already (1, 1, H, W)

        return x, dem, rainfall


if __name__ == "__main__":
    taree = TerrainData.from_aoi_name("taree")
    taree.visualize()

    dataset = FloodDataset(aoi_name="taree")
    print(dataset[0])
