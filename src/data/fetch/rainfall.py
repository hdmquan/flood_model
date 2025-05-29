import os
import ee
import geemap
import datetime
import dotenv

dotenv.load_dotenv()


def fetch_rainfall_series(bbox, start_date, end_date, output_csv="rainfall.csv"):
    ee.Initialize(project=os.getenv("GEE_PROJECT"))
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
    print(f"Rainfall time series saved to {str(output_csv)}")


if __name__ == "__main__":
    import yaml
    import pandas as pd
    import matplotlib.pyplot as plt
    from src.utils import PATH

    print(os.getenv("GEE_PROJECT"))

    with open(PATH.SRC / "aoi.yaml", "r") as f:
        aoi = yaml.load(f, Loader=yaml.FullLoader)

    fetch_rainfall_series(
        aoi["taree"]["bbox"], "2024-01-01", "2024-2-31", PATH.RAINFALL / "taree.csv"
    )

    with open(PATH.RAINFALL / "taree.csv", "r") as f:
        df = pd.read_csv(f)

    plt.plot(df["datetime"], df["rain_mmhr"])
    plt.show()
