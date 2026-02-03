import json
import os

from dotenv import load_dotenv

load_dotenv()
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from constants import (
    BLACK_MEAN_THRESHOLD,
    BLACK_PIXEL_RATIO,
    BLACK_PIXEL_THRESHOLD,
    FIREPLACE_POLY_INIT,
    HOT_PIXEL_V_THRESHOLD,
    LATEST_FRAME_LOG,
    LATEST_FRAME_PATH,
    MAX_TRANSLATION_PX,
    MIN_INLIERS,
    MIN_KEYPOINTS,
    MIN_MATCHES,
    MODEL_PATH,
    N_BANDS,
    OUTPUT_SIZE,
    POLY_SMOOTHING_ALPHA,
    POLYGON_PATH,
    RANSAC_REPROJ_THRESHOLD,
    STREAM_URL,
)
from scheduler_service import CaptureScheduler
from schemas import CaptureConfig
from utils_image import *


class PolygonUpdate(BaseModel):
    polygon: List[List[float]]


os.makedirs("tmp", exist_ok=True)
logger.add(LATEST_FRAME_LOG, rotation="1 MB", retention=5)

app = FastAPI(title="Fireplace Flame Detector")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/tmp", StaticFiles(directory="tmp"), name="tmp")

HOME_HTML_PATH = Path("static/home.html")
UPDATE_POLYGONE_HTML_PATH = Path("static/update.html")


@app.on_event("startup")
def startup_event():
    app.state.fireplace = init_fireplace_state(FIREPLACE_POLY_INIT.copy())
    app.state.clf = joblib.load(MODEL_PATH)

    capture_image_from_stream(STREAM_URL, LATEST_FRAME_PATH)

    img = cv2.imread(LATEST_FRAME_PATH)
    if img is None:
        return JSONResponse({"error": "Failed to read screenshot"}, status_code=500)

    # Load saved polygon if exists
    if Path(POLYGON_PATH).exists():
        import json

        with open(POLYGON_PATH) as f:
            poly = json.load(f)
            if len(poly) == 4:
                app.state.fireplace["last_polygon"] = np.array(poly, dtype=np.float32)

    # Scheduler
    app.state.capture_scheduler = CaptureScheduler()
    app.state.capture_scheduler.start()


@app.on_event("shutdown")
def shutdown_event():
    app.state.capture_scheduler.shutdown()


@app.get("/")
def root():
    return RedirectResponse(url="/home")


@app.get("/home")
def serve_home():
    return FileResponse(HOME_HTML_PATH)


@app.get("/update")
def serve_update():
    return FileResponse(UPDATE_POLYGONE_HTML_PATH)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file.file.seek(0)
    img = load_image_from_upload(file)
    if img is None:
        return JSONResponse({"error": "invalid image"}, status_code=400)

    state = app.state.fireplace

    result, updated_state = predict_from_image(
        img,
        clf=app.state.clf,
        # orb=app.state.orb,
        last_polygon=state["last_polygon"],
        ref_gray=state["ref_gray"],
        ref_kp=state["ref_kp"],
        ref_des=state["ref_des"],
        black_mean_threshold=BLACK_MEAN_THRESHOLD,
        black_pixel_threshold=BLACK_PIXEL_THRESHOLD,
        black_pixel_ratio=BLACK_PIXEL_RATIO,
        min_keypoints=MIN_KEYPOINTS,
        min_matches=MIN_MATCHES,
        ransac_reproj_threshold=RANSAC_REPROJ_THRESHOLD,
        min_inliers=MIN_INLIERS,
        max_translation_px=MAX_TRANSLATION_PX,
        poly_smoothing_alpha=POLY_SMOOTHING_ALPHA,
        output_size=OUTPUT_SIZE,
        n_bands=N_BANDS,
        hot_pixel_v_threshold=HOT_PIXEL_V_THRESHOLD,
    )

    app.state.fireplace.update(updated_state)
    return result


@app.get("/trigger_predict")
def trigger_predict():
    capture_image_from_stream(STREAM_URL, LATEST_FRAME_PATH)

    img = cv2.imread(LATEST_FRAME_PATH)
    if img is None:
        return JSONResponse({"error": "Failed to read screenshot"}, status_code=500)

    state = app.state.fireplace
    result, updated_state = predict_from_image(
        img,
        clf=app.state.clf,
        # orb=app.state.orb,
        last_polygon=state["last_polygon"],
        ref_gray=state["ref_gray"],
        ref_kp=state["ref_kp"],
        ref_des=state["ref_des"],
        black_mean_threshold=BLACK_MEAN_THRESHOLD,
        black_pixel_threshold=BLACK_PIXEL_THRESHOLD,
        black_pixel_ratio=BLACK_PIXEL_RATIO,
        min_keypoints=MIN_KEYPOINTS,
        min_matches=MIN_MATCHES,
        ransac_reproj_threshold=RANSAC_REPROJ_THRESHOLD,
        min_inliers=MIN_INLIERS,
        max_translation_px=MAX_TRANSLATION_PX,
        poly_smoothing_alpha=POLY_SMOOTHING_ALPHA,
        output_size=OUTPUT_SIZE,
        n_bands=N_BANDS,
        hot_pixel_v_threshold=HOT_PIXEL_V_THRESHOLD,
    )

    app.state.fireplace.update(updated_state)
    return result


@app.post("/update_polygon")
def update_polygon(update: PolygonUpdate):
    polygon = update.polygon
    if len(polygon) != 4:
        return {"error": "Must provide 4 points"}

    new_poly = np.array(polygon, dtype=np.float32)
    app.state.fireplace["last_polygon"] = new_poly

    # Save to disk
    with open(POLYGON_PATH, "w") as f:
        json.dump(polygon, f)

    return {"success": True}


@app.get("/scheduler", response_class=HTMLResponse)
def serve_scheduler():
    return FileResponse("static/scheduler.html")


def get_next_scheduler_jobs(n: int = 5):
    scheduler = app.state.capture_scheduler
    config = scheduler._config

    if not scheduler or not config or not config.active:
        return []

    jobs = scheduler._scheduler.get_jobs()
    if not jobs:
        return []

    job = jobs[0]

    if job.next_run_time is None:
        return []

    # Safe interval (minimum 1 second)
    seconds = max(1, int(3600 / config.images_per_hour))
    interval = timedelta(seconds=seconds)

    next_jobs = []
    t = job.next_run_time

    # Ensure we start inside the configured window
    if t < config.scheduler_start_datetime:
        delta = config.scheduler_start_datetime - t
        steps = (delta // interval) + 1
        t = t + steps * interval

    while len(next_jobs) < n and t <= config.scheduler_end_datetime:
        next_jobs.append(t.isoformat())
        t += interval

    return next_jobs


@app.post("/scheduler/config")
async def configure_scheduler(
    images_per_hour: int = Form(...),
    scheduler_start_datetime: str = Form(...),
    scheduler_end_datetime: str = Form(...),
    toggle_active: str = Form("true"),
):
    # Parse ISO datetime from the form
    start_dt = datetime.fromisoformat(scheduler_start_datetime).replace(
        tzinfo=timezone.utc
    )
    end_dt = datetime.fromisoformat(scheduler_end_datetime).replace(tzinfo=timezone.utc)

    config = CaptureConfig(
        active=toggle_active.lower() == "true",
        images_per_hour=images_per_hour,
        scheduler_start_datetime=start_dt,
        scheduler_end_datetime=end_dt,
    )

    scheduler = app.state.capture_scheduler
    scheduler.update_config(config)

    next_jobs = get_next_scheduler_jobs(5)

    return {
        "success": True,
        "active": config.active,
        "next_jobs": next_jobs,
    }


@app.get("/scheduler/config")
def get_scheduler_config():
    scheduler = app.state.capture_scheduler
    config = scheduler._config
    next_jobs = get_next_scheduler_jobs(5)

    return {
        "active": config.active if config else False,
        "scheduler_start_datetime": config.scheduler_start_datetime.date().isoformat(),
        "scheduler_end_datetime": config.scheduler_end_datetime.date().isoformat(),
        "images_per_hour": config.images_per_hour if config else None,
        "next_jobs": next_jobs,
    }
