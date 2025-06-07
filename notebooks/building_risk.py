# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from rasterio.transform import rowcol
from rasterio import features
from shapely.geometry import box
from src.data import TerrainData

# %% Load terrain data
aoi_name = "taree"
terrain = TerrainData.from_aoi_name(aoi_name)
terrain.fetch_dem()
terrain.fetch_landuse()
terrain.compute_flow_metrics()
terrain.compute_hand()

# %% Load building footprints and FSI map
buildings = terrain.fetch_footprint()
fsi = terrain.generate_susceptibility_map()
transform = terrain.dem_transform
shape = fsi.shape

# %% Preview all building footprints
fig, ax = plt.subplots(figsize=(12, 10))

buildings.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.3)
ax.set_title("All Building Footprints in AOI", fontsize=14)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.show()

# %%

# %% Risk thresholds and containers
low_thresh = 0.3
moderate_thresh = 0.6

risky_geoms = []
risky_labels = []

# %% Evaluate flood risk for each building
for _, row in buildings.iterrows():
    try:
        mask = features.geometry_mask(
            [row.geometry], transform=transform, invert=True, out_shape=shape
        )
        fsi_values = fsi[mask]
        if fsi_values.size == 0:
            continue
        mean_fsi = np.nanmean(fsi_values)

        if mean_fsi > moderate_thresh:
            risky_labels.append("High")
        elif mean_fsi > low_thresh:
            risky_labels.append("Moderate")
        else:
            risky_labels.append("Low")

        risky_geoms.append(row.geometry)
    except Exception as e:
        print(f"Skipping building due to error: {e}")
        continue

# %% Create GeoDataFrame with risk categories
risk_gdf = gpd.GeoDataFrame(
    {"risk": risky_labels, "geometry": risky_geoms}, crs=buildings.crs
)

# %% Zoomed-in view of buildings at risk with vector roads/waterways
high_risk_subset = risk_gdf[risk_gdf.risk == "High"].head(2)
zoom_bounds = high_risk_subset.total_bounds  # (minx, miny, maxx, maxy)

# Fetch landuse as vector
landuse = terrain.fetch_landuse()
roads = landuse[landuse["feature_type"] == "road"]
water = landuse[landuse["feature_type"] == "water"]

# Spatially filter roads and water to the zoom window
minx, miny, maxx, maxy = zoom_bounds
zoom_box = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=landuse.crs)
roads_clip = gpd.overlay(roads, zoom_box, how="intersection")
water_clip = gpd.overlay(water, zoom_box, how="intersection")

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot clipped roads and waterways
roads_clip.plot(ax=ax, color="dimgray", linewidth=0.5, label="Roads")
water_clip.plot(ax=ax, color="deepskyblue", linewidth=0.8, label="Waterways")

# Plot buildings by flood risk
for risk_level, color in [("Low", "green"), ("Moderate", "orange"), ("High", "red")]:
    risk_gdf[risk_gdf.risk == risk_level].plot(
        ax=ax,
        facecolor=color,
        edgecolor="black",
        linewidth=0.3,
        label=risk_level,
    )

# Final plot formatting
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.set_title("Flood Risk + Roads and Waterways", fontsize=14)
ax.set_xticks([])
ax.set_yticks([])
ax.legend()
plt.tight_layout()
plt.show()


# %%
