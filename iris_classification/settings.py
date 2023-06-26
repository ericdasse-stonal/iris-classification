import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_PATH = Path(__file__).parent.parent

RAW_DATA_PATH = os.environ.get("RAW_DATA_PATH", REPO_PATH / "data/raw/iris.data")
PREPARED_DATA_PATH = os.environ.get(
    "PREPARED_DATA_PATH", REPO_PATH / "data/prepared/prepared_iris.csv"
)
