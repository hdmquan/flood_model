import geopandas as gpd
import matplotlib.pyplot as plt
from urllib.parse import urlencode
import pandas as pd


def download(layer_number, bbox, verbose=True, plot=True):

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


def fetch_landuse(bbox, verbose=True):

    roads = download(5, bbox)
    water = download(3, bbox)

    roads["feature_type"] = "road"
    water["feature_type"] = "water"

    combined = pd.concat([roads, water], ignore_index=True)

    return combined


if __name__ == "__main__":
    import yaml
    from src.utils import PATH

    with open(PATH.SRC / "aoi.yaml", "r") as f:
        aoi = yaml.safe_load(f)

    bbox = aoi["taree"]["bbox"]

    bbox = {
        "xmin": bbox[0],
        "ymin": bbox[1],
        "xmax": bbox[2],
        "ymax": bbox[3],
    }

    print(bbox)

    # Fetch landuse data
    landuse_data = fetch_landuse(bbox, verbose=True)

    # Create and save plot with transparent background
    if not landuse_data.empty:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        landuse_data.plot(ax=ax, edgecolor="black", linewidth=0.8, color="blue")

        # Turn off everything
        ax.set_axis_off()

        # Save plot with transparent background
        plt.savefig("landuse_plot.png", transparent=True, bbox_inches="tight", dpi=300)
        plt.close()
