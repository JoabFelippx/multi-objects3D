from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent

CALIB_PATH_RAW = Path(os.environ.get("CALIB_PATH_RAW", PROJECT_ROOT / "data" / "calibration"))

IMAGES_PATH_RAW = Path(os.environ.get("IMAGES_PATH_RAW", PROJECT_ROOT / "data" / "images"))

ANNOTATIONS_PATH = Path(os.environ.get("ANNOTATIONS_PATH", PROJECT_ROOT / "annotations"))

