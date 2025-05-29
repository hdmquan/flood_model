import os
import yaml
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sentinelhub import (
    CRS,
    BBox,
    bbox_to_dimensions,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    MosaickingOrder,
    SHConfig,
)

from src.utils import PATH

# Load environment variables
load_dotenv(PATH.SRC / ".env")

config = SHConfig()
config.sh_client_id = os.getenv("SENTINEL_HUB_CLIENT_ID")
config.sh_client_secret = os.getenv("SENTINEL_HUB_CLIENT_SECRET")


def get_available_images(bbox, start_date, end_date, max_cloud_cover=20):
    from sentinelhub import SentinelHubCatalog

    catalog = SentinelHubCatalog(config=config)
    search_iterator = catalog.search(
        DataCollection.SENTINEL2_L2A,
        bbox=bbox,
        time=(start_date, end_date),
        filter=f"eo:cloud_cover <= {max_cloud_cover}",
    )
    results = list(search_iterator)
    print(f"Found {len(results)} available scenes")
    return results


def retrieve_image(coordinates, start_date, end_date, resolution=10):
    bbox = BBox(bbox=coordinates, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution)
    print(f"Image size at {resolution}m: {size} pixels")

    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: [{ bands: ["B02", "B03", "B04"], units: "DN" }],
            output: { bands: 3, sampleType: SampleType.FLOAT32 }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
    """

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(start_date, end_date),
                mosaicking_order=MosaickingOrder.LEAST_CC,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )

    images = request.get_data()
    if not images or images[0].max() == 0:
        print("Image is empty or contains no valid data.")
        return None

    return images[0]


def normalize_image(img):
    img = img.astype(np.float32)
    for i in range(img.shape[2]):
        band_max = np.nanmax(img[:, :, i])
        if band_max > 0:
            # To avoid outliers
            p2 = np.percentile(img, 2)
            p98 = np.percentile(img, 98)
            img = np.clip((img - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
            # img[:, :, i] = img[:, :, i] / band_max * 255

    return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    # Load bounding box from YAML
    with open(PATH.SRC / "aoi.yaml", "r") as f:
        data = yaml.safe_load(f)
    bbox = data["taree"]["bbox"]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    bbox_obj = BBox(bbox=bbox, crs=CRS.WGS84)
    available = get_available_images(bbox_obj, start_str, end_str)
    if not available:
        print("No valid images found for this time range and location.")
        exit()

    img = retrieve_image(bbox, start_str, end_str)
    if img is not None:
        print(f"Image shape: {img.shape}")
        print(f"Min: {img.min()}, Max: {img.max()}, Mean: {img.mean()}")

        img = normalize_image(img)

        print(f"Min: {img.min()}, Max: {img.max()}, Mean: {img.mean()}")
        im = Image.fromarray(img)
        im.save("sentinel_retrieve.png")
        print("Image saved to sentinel_retrieve.png")
