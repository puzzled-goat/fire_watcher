import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

IMAGE_DIR_PATH: Path = Path(os.getenv("IMAGE_DIR_PATH"))
