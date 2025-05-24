# %%
import geopandas as gpd
import matplotlib.pyplot as plt
from urllib.parse import urlencode
import pandas as pd


def download(layer_number, bbox, verbose=True, plot=True):
    """
    Downloads all features from a NSW FeatureServer layer using paging.

    Parameters:
    - layer_number (int): The layer number from the FeatureServer (e.g. 6 for roads).
    - bbox (dict): A dictionary with 'xmin', 'ymin', 'xmax', 'ymax' keys.
    - verbose (bool): Print progress messages.
    - plot (bool): Plot the final GeoDataFrame.

    Returns:
    - GeoDataFrame containing all features within the bounding box.
    """
    base_url = f"https://portal.spatial.nsw.gov.au/server/rest/services/NSW_Land_Parcel_Property_Theme/FeatureServer/{layer_number}/query"

    base_params = {
        "where": "1=1",
        "geometry": f"{bbox['xmin']},{bbox['ymin']},{bbox['xmax']},{bbox['ymax']}",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": "4326",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }

    all_gdfs = []
    offset = 0
    limit = 100

    if verbose:
        print(f"Downloading from layer {layer_number} in pages...")

    while True:
        params = base_params.copy()
        params.update({"resultOffset": offset, "resultRecordCount": limit})
        full_url = f"{base_url}?{urlencode(params)}"

        try:
            gdf = gpd.read_file(full_url)
            if gdf.empty:
                break
            all_gdfs.append(gdf)
            if verbose:
                print(f"Fetched {len(gdf)} features at offset {offset}")
            offset += limit
        except Exception as e:
            print(f"Error at offset {offset}: {e}")
            break

    if all_gdfs:
        full_gdf = pd.concat(all_gdfs, ignore_index=True)
        if verbose:
            print(f"Total features downloaded: {len(full_gdf)}")
    else:
        full_gdf = gpd.GeoDataFrame()
        print("No data downloaded.")

    if plot and not full_gdf.empty:
        full_gdf.plot(figsize=(10, 10), edgecolor="black", linewidth=0.8, color="blue")
        plt.title(f"Layer {layer_number} - Feature Plot")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True)
        plt.show()

    return full_gdf


# %%
lismore_bbox = {"xmin": 153.25, "ymin": -28.85, "xmax": 153.30, "ymax": -28.80}
# %%
roads = download(5, lismore_bbox)
# %%
water = download(3, lismore_bbox)

# %%
# Add type column to each GeoDataFrame
roads["feature_type"] = "road"
water["feature_type"] = "water"

# Merge the two GeoDataFrames
combined = pd.concat([roads, water], ignore_index=True)

# Define a manual color mapping
color_map = {"road": "gray", "water": "blue"}

# Plot
fig, ax = plt.subplots(figsize=(10, 10))

for feature_type, color in color_map.items():
    subset = combined[combined["feature_type"] == feature_type]
    subset.plot(
        ax=ax, color=color, edgecolor="black", linewidth=0.8, label=feature_type
    )

# Formatting
plt.title("Roads and Water in Lismore")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.legend()
plt.show()

# %%
