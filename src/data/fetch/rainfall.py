import ee
import geemap
import datetime


def fetch_rainfall_series(bbox, start_date, end_date, output_csv="rainfall.csv"):
    """
    Downloads GPM-IMERG rainfall time series (mm/hr) for a region
    bbox: (minLon, minLat, maxLon, maxLat)
    """
    ee.Initialize()
    region = ee.Geometry.Rectangle(list(bbox))

    dataset = (
        ee.ImageCollection("NASA/GPM_L3/IMERG_V06")
        .filterDate(start_date, end_date)
        .select("precipitationCal")
        .map(lambda img: img.set("system:time_start", img.date().millis()))
    )

    def extract_time_series(image):
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=10000, maxPixels=1e9
        )
        return ee.Feature(
            None,
            {
                "datetime": image.date().format("YYYY-MM-dd'T'HH:mm:ss"),
                "rain_mmhr": stats.get("precipitationCal"),
            },
        )

    rainfall_fc = ee.FeatureCollection(dataset.map(extract_time_series))
    geemap.ee_export_vector(rainfall_fc, filename=output_csv)
    print(f"Rainfall time series saved to {output_csv}")
