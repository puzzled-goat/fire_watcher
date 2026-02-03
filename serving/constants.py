import os

import numpy as np

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

BLACK_MEAN_THRESHOLD = 3
BLACK_PIXEL_RATIO = 0.99
BLACK_PIXEL_THRESHOLD = 10

ORB_N_FEATURES = 2000
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS = 8

MIN_KEYPOINTS = 50
MIN_MATCHES = 30
RANSAC_REPROJ_THRESHOLD = 3.0
MIN_INLIERS = 15
MAX_TRANSLATION_PX = 25

POLY_SPACE = "pixel"
POLY_SMOOTHING_ALPHA = 0.9

FIREPLACE_POLY_INIT = np.array(
    [[270, 205], [373, 217], [370, 270], [270, 255]], dtype=np.float32
)

MODEL_PATH = "models/random_forest_v1.1.joblib"

CLASS_LABELS = ["large flames", "black", "medium flames", "ember", "small flames"]

DEBUG_SAVE_FRAMES = True
DEBUG_OUTPUT_DIR = "tmp/fireplace_debug"
LATEST_FRAME_PATH = "tmp/fireplace_latest.jpg"
LATEST_FRAME_LOG = "tmp/fireplace_latest.log"

N_BANDS = 3
HOT_PIXEL_V_THRESHOLD = 200
OUTPUT_SIZE = 256

STREAM_URL = os.getenv("STREAM_URL")

POLYGON_PATH = "tmp/fireplace_polygon.json"

IMG_FOLDER_PATH = os.getenv("IMG_FOLDER_PATH")
CONFIG_PATH = "tmp/scheduler_config.json"
