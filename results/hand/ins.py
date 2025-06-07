import rasterio
import numpy as np

from pathlib import Path

HERE = Path(__file__).resolve().parent

with rasterio.open(HERE / "taree_streams.tif") as src:
    stream_data = src.read(1)
    print("Unique stream values:", np.unique(stream_data))
    print("Has NaNs:", np.isnan(stream_data).any())
    print("Has -32768:", (stream_data == -32768).any())
