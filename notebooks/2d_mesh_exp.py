# %% Imports and Setup
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
from shapely.geometry import box
import contextily as cx
from matplotlib.collections import PolyCollection
from pyproj import Transformer
from src.data.dataclass import TerrainData
from rasterio.windows import from_bounds

# %% Load DEM and clip to AOI
terrain = TerrainData.from_aoi_name("taree")
terrain.fetch_dem()

bbox = terrain.bbox  # [xmin, ymin, xmax, ymax]

with rasterio.open(terrain.dem_cache) as src:
    window = from_bounds(*bbox, src.transform)
    dem = src.read(1, window=window)
    transform = src.window_transform(window)

# %% Compute slope and sample adaptively
dy, dx = np.gradient(dem)
slope = np.sqrt(dx**2 + dy**2)

threshold = 1.0
mask = slope > threshold
sample_points = np.argwhere(mask)

rows, cols = sample_points[:, 0], sample_points[:, 1]
xs, ys = rasterio.transform.xy(transform, rows, cols)
x = np.array(xs)
y = np.array(ys)
points_xy = np.stack([x, y], axis=1)
points_xy += np.random.normal(scale=2, size=points_xy.shape)

# %% Generate Delaunay triangulation
tri = Delaunay(points_xy)
triangles = tri.simplices
centroids = points_xy[triangles].mean(axis=1)

# %% Interpolate elevation at centroids
row_inds = ((centroids[:, 1] - transform.f) / transform.e).astype(int)
col_inds = ((centroids[:, 0] - transform.c) / transform.a).astype(int)
row_inds = np.clip(row_inds, 0, dem.shape[0] - 1)
col_inds = np.clip(col_inds, 0, dem.shape[1] - 1)
elev = dem[row_inds, col_inds]

# %% Simulate flood depth
rain = 10
infiltration = 3
slope_factor = ((elev.max() - elev) / (elev.max() - elev.min() + 1e-6)) ** 2
h = np.maximum(rain - infiltration, 0) * slope_factor
h_smooth = gaussian_filter(h, sigma=1)

# %% Plot triangles filled by flood depth
polys = [points_xy[tri] for tri in triangles]
depth_norm = (h_smooth - h_smooth.min()) / (h_smooth.max() - h_smooth.min())

coll = PolyCollection(
    polys, array=depth_norm, cmap="Blues", edgecolor="k", linewidth=0.1
)

fig, ax = plt.subplots(figsize=(10, 10))
ax.add_collection(coll)
ax.autoscale()
ax.set_aspect("equal")
plt.title("Flood Simulation - Depth Colored Triangles")
plt.axis("off")
plt.tight_layout()
plt.show()

# %% Overlay projected triangles on basemap
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
xv_proj, yv_proj = transformer.transform(points_xy[:, 0], points_xy[:, 1])
points_proj = np.stack([xv_proj, yv_proj], axis=1)
tri_polys = [points_proj[tri] for tri in triangles]

fig, ax = plt.subplots(figsize=(10, 10))
coll = PolyCollection(
    tri_polys, array=depth_norm, cmap="Blues", edgecolor="none", alpha=0.9
)
ax.add_collection(coll)
ax.autoscale()
ax.set_aspect("equal")
cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.CartoDB.Positron)
ax.set_title("Flood Simulation Overlay (Projected and Accurate)")
ax.axis("off")
plt.tight_layout()
plt.show()

# %% Raw triangle mesh (for debug)
fig, ax = plt.subplots(figsize=(10, 10))
ax.triplot(points_xy[:, 0], points_xy[:, 1], triangles, lw=0.3, color="gray")
ax.set_aspect("equal")
ax.set_title("Raw AOI Triangle Mesh (No Projection)")
plt.tight_layout()
plt.show()

# %%
