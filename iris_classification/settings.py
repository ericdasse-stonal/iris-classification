import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_PATH = Path(__file__).parent.parent

RAW_DATA_DIR = os.environ.get("RAW_DATA_PATH", REPO_PATH / "data/raw/iris.data")
PREPARED_DATA_DIR = os.environ.get("PREPARED_DATA_DIR", REPO_PATH / "data/prepared")

IRIS_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "species"]
MODEL_DIR = os.environ.get("MODEL_PATH", REPO_PATH / "model")
