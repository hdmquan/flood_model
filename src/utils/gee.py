import os
import re
from pathlib import Path
from typing import List, Union, Tuple, Optional, Literal
from datetime import datetime

import ee
from dotenv import load_dotenv
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from src.common import setup_logger, PATH
from src.data.metadata.constants import AUSTRALIA_BBOX

logger = setup_logger()

load_dotenv()
ee.Initialize(project=os.getenv("GOOGLE_PROJECT_ID"))

DYNAMIC_WORLD_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
EXPORT_SCALE = 10
EXPORT_CRS = "EPSG:4326"
MAX_PIXELS = 1e13


def create_export_task(image: ee.Image, year: int, folder: str) -> ee.batch.Task:
    filename = f"dw_label_{year}"

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=filename,
        folder=folder,
        fileNamePrefix=filename,
        region=AUSTRALIA_BBOX,
        scale=EXPORT_SCALE,
        crs=EXPORT_CRS,
        maxPixels=MAX_PIXELS,
    )
    return task


def get_dynamic_world_label_image(year: int) -> ee.Image:
    start = f"{year}-01-01"
    end = f"{year}-12-31"

    collection = (
        ee.ImageCollection(DYNAMIC_WORLD_COLLECTION)
        .filterDate(start, end)
        .filterBounds(ee.Geometry.Rectangle(AUSTRALIA_BBOX))
        .select("label")
    )

    # Take the most frequent class per pixel across the year
    return collection.mode().clip(ee.Geometry.Rectangle(AUSTRALIA_BBOX))


def export_annual_dynamic_world(years: List[int], folder: str):
    for year in years:
        image = get_dynamic_world_label_image(year)
        task = create_export_task(image, year, folder)
        task.start()
        logger.info(f"Export started for year {year}")


def authenticate_drive() -> GoogleDrive:
    gauth = GoogleAuth()
    gauth.LoadClientConfigFile(str(PATH.CRED / "google_oauth_client_secret.json"))
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)


def download_exports_from_drive(folder_name: str, local_path: Path):
    local_path.mkdir(parents=True, exist_ok=True)
    drive = authenticate_drive()

    logger.info(
        f"Searching for exported Dynamic World files in Drive folder '{folder_name}'..."
    )

    file_list = drive.ListFile(
        {"q": f"'{folder_name}' in parents and trashed=false"}
    ).GetList()

    dw_files = [f for f in file_list if re.match(r"dw_label_\d{4}\.tif", f["titltqe"])]

    if not dw_files:
        logger.warning("No matching Dynamic World .tif files found.")
        return

    for file in dw_files:
        file_title = file["title"]
        dest_path = local_path / file_title

        if dest_path.exists():
            logger.info(f"Skipped (already exists): {file_title}")
            continue

        logger.info(f"Downloading {file_title} → {dest_path}")
        file.GetContentFile(str(dest_path))

    logger.success(f"All Dynamic World GeoTIFFs downloaded to {local_path}")


def download_gee_images(
    time_range: Tuple[datetime, datetime],
    local_path: Path,
    collection_name: str,
    band_name: Optional[str] = None,
    bbox: List[float] = AUSTRALIA_BBOX,
    mode: Literal["best", "first", "all"] = "best",
    file_prefix: str = "gee_image",
    cloud_cover_prop: str = "CLOUD_COVER",
    max_cloud_cover: float = 20.0,
    scale: int = EXPORT_SCALE,
    crs: str = EXPORT_CRS,
    max_pixels: float = MAX_PIXELS,
) -> List[Path]:
    """
    Download Earth Engine images for a specific time range and collection.

    Args:
        time_range: Tuple of (start_date, end_date) as datetime objects
        local_path: Path to store downloaded images
        collection_name: Name of the GEE collection (e.g., "LANDSAT/LC08/C02/T1_TOA")
        band_name: Band or list of bands to select (None means all bands)
        bbox: Bounding box coordinates [west, south, east, north]
        mode: Download mode - 'best' (lowest cloud cover), 'first', or 'all'
        file_prefix: Prefix for saved files
        cloud_cover_prop: Property name for cloud cover in the collection
        max_cloud_cover: Maximum acceptable cloud cover percentage
        scale: Resolution in meters per pixel
        crs: Coordinate reference system
        max_pixels: Maximum pixels to export

    Returns:
        List of paths to downloaded files
    """
    local_path.mkdir(parents=True, exist_ok=True)
    drive_folder = os.getenv("GOOGLE_DRIVE_FOLDER_NAME")

    if not drive_folder:
        raise ValueError("GOOGLE_DRIVE_FOLDER_NAME environment variable not set")

    # Format dates for GEE
    start_date = time_range[0].strftime("%Y-%m-%d")
    end_date = time_range[1].strftime("%Y-%m-%d")

    # Create the geometry
    geometry = ee.Geometry.Rectangle(bbox)

    # Initialize the image collection
    collection = (
        ee.ImageCollection(collection_name)
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
    )

    # Apply band selection if specified
    if band_name:
        collection = collection.select(band_name)

    # Apply cloud filtering
    if cloud_cover_prop:
        collection = collection.filter(ee.Filter.lte(cloud_cover_prop, max_cloud_cover))

    # Get the count of images
    count = collection.size().getInfo()
    logger.info(
        f"Found {count} images in collection {collection_name} for the specified time range"
    )

    if count == 0:
        logger.warning("No images found matching criteria")
        return []

    # Timestamp for this export batch to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    downloaded_files = []

    if mode == "best":
        # Sort by cloud cover and get the best image
        if cloud_cover_prop:
            best_image = collection.sort(cloud_cover_prop).first()
        else:
            best_image = collection.first()

        # Create a unique filename
        filename = f"{file_prefix}_{timestamp}_best"

        # Export the image
        task = ee.batch.Export.image.toDrive(
            image=best_image,
            description=filename,
            folder=drive_folder,
            fileNamePrefix=filename,
            region=bbox,
            scale=scale,
            crs=crs,
            maxPixels=max_pixels,
        )

        task.start()
        logger.info(f"Export started for best image: {filename}")

        # Wait for the task to complete
        logger.info("Waiting for export to complete...")
        while task.active():
            import time

            time.sleep(10)

        # Download the image from Google Drive
        drive = authenticate_drive()

        file_list = drive.ListFile(
            {
                "q": f"'{drive_folder}' in parents and title contains '{filename}' and trashed=false"
            }
        ).GetList()

        if file_list:
            file = file_list[0]
            file_title = file["title"]
            dest_path = local_path / file_title

            logger.info(f"Downloading {file_title} → {dest_path}")
            file.GetContentFile(str(dest_path))
            downloaded_files.append(dest_path)

    elif mode == "first":
        # Get the first image
        first_image = collection.first()

        # Create a unique filename
        filename = f"{file_prefix}_{timestamp}_first"

        # Export the image
        task = ee.batch.Export.image.toDrive(
            image=first_image,
            description=filename,
            folder=drive_folder,
            fileNamePrefix=filename,
            region=bbox,
            scale=scale,
            crs=crs,
            maxPixels=max_pixels,
        )

        task.start()
        logger.info(f"Export started for first image: {filename}")

        # Wait for the task to complete
        logger.info("Waiting for export to complete...")
        while task.active():
            import time

            time.sleep(10)

        # Download the image from Google Drive
        drive = authenticate_drive()

        file_list = drive.ListFile(
            {
                "q": f"'{drive_folder}' in parents and title contains '{filename}' and trashed=false"
            }
        ).GetList()

        if file_list:
            file = file_list[0]
            file_title = file["title"]
            dest_path = local_path / file_title

            logger.info(f"Downloading {file_title} → {dest_path}")
            file.GetContentFile(str(dest_path))
            downloaded_files.append(dest_path)

    elif mode == "all":
        # Get all images
        image_list = collection.toList(collection.size())
        image_count = image_list.size().getInfo()

        for i in range(image_count):
            image = ee.Image(image_list.get(i))

            # Create a unique filename
            filename = f"{file_prefix}_{timestamp}_{i+1}"

            # Export the image
            task = ee.batch.Export.image.toDrive(
                image=image,
                description=filename,
                folder=drive_folder,
                fileNamePrefix=filename,
                region=bbox,
                scale=scale,
                crs=crs,
                maxPixels=max_pixels,
            )

            task.start()
            logger.info(f"Export started for image {i+1}/{image_count}: {filename}")

            # Wait for the task to complete
            logger.info(
                f"Waiting for export of image {i+1}/{image_count} to complete..."
            )
            while task.active():
                import time

                time.sleep(10)

            # Download the image from Google Drive
            drive = authenticate_drive()

            file_list = drive.ListFile(
                {
                    "q": f"'{drive_folder}' in parents and title contains '{filename}' and trashed=false"
                }
            ).GetList()

            if file_list:
                file = file_list[0]
                file_title = file["title"]
                dest_path = local_path / file_title

                logger.info(f"Downloading {file_title} → {dest_path}")
                file.GetContentFile(str(dest_path))
                downloaded_files.append(dest_path)

    # Clean up after downloading
    logger.success(f"Downloaded {len(downloaded_files)} images to {local_path}")

    return downloaded_files


if __name__ == "__main__":

    folder = os.getenv("GOOGLE_DRIVE_FOLDER_NAME")
    drive_url = os.getenv("GOOGLE_DRIVE_URL")

    logger.info("Starting Dynamic World export tasks to Google Drive...")

    export_annual_dynamic_world(list(range(2014, 2024)), folder)

    logger.success("All export tasks launched.")

    download_exports_from_drive(folder, PATH.R_DATA / "deforestation")

    logger.warning("These will pile up in Google Drive. Go to:")
    logger.warning(str(drive_url))
    logger.warning("and delete manually when done. Sorry. Google is Google.")
