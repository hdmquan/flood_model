import requests
import datetime
import json
import os

SENTINEL_HUB_CLIENT_ID = os.getenv("SENTINEL_HUB_CLIENT_ID")
SENTINEL_HUB_CLIENT_SECRET = os.getenv("SENTINEL_HUB_CLIENT_SECRET")


def get_oauth_token():
    url = "https://services.sentinel-hub.com/oauth/token"
    response = requests.post(
        url,
        data={
            "grant_type": "client_credentials",
            "client_id": SENTINEL_HUB_CLIENT_ID,
            "client_secret": SENTINEL_HUB_CLIENT_SECRET,
        },
    )
    response.raise_for_status()
    return response.json()["access_token"]


def fetch_sentinel_image(bbox, date, size_px=(512, 512), output_file="image.tif"):
    """
    bbox = [minLon, minLat, maxLon, maxLat]
    date = "2022-03-01"
    """
    token = get_oauth_token()
    url = "https://services.sentinel-hub.com/api/v1/process"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{date}T00:00:00Z",
                            "to": f"{date}T23:59:59Z",
                        }
                    },
                }
            ],
        },
        "output": {
            "width": size_px[0],
            "height": size_px[1],
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B03", "B02"],
    output: { bands: 3 }
  };
}

function evaluatePixel(sample) {
  return [sample.B04, sample.B03, sample.B02];
}
        """,
    }

    r = requests.post(url, headers=headers, data=json.dumps(payload))
    r.raise_for_status()

    with open(output_file, "wb") as f:
        f.write(r.content)

    print(f"Saved image to {output_file}")
