from pathlib import Path


class PATH:
    ROOT = Path(__file__).parent.parent.parent
    CRED = ROOT / "cred"
    DATA = ROOT / "data"
    DEM = DATA / "dem"
    RAINFALL = DATA / "rainfall"
    MODELS = ROOT / "models"
    NOTEBOOKS = ROOT / "notebooks"
    SRC = ROOT / "src"
    UTILS = ROOT / "utils"
