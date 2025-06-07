from pathlib import Path


class PATH:
    ROOT = Path(__file__).parent.parent.parent
    CRED = ROOT / "cred"
    DATA = ROOT / "data"
    RESULTS = ROOT / "results"

    DEM = DATA / "dem"
    RAINFALL = DATA / "rainfall"
    LANDUSE = DATA / "landuse"
    FOOTPRINT = DATA / "footprint"

    MODELS = ROOT / "models"
    NOTEBOOKS = ROOT / "notebooks"
    SRC = ROOT / "src"
    UTILS = ROOT / "utils"
