import json
import os
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from constants import CONFIG_PATH, IMG_FOLDER_PATH, STREAM_URL
from schemas import CaptureConfig
from utils_image import capture_image_from_stream


class CaptureScheduler:
    """
    Manages a background scheduler that periodically captures images from a video
    stream based on a persisted capture configuration.
    """

    def __init__(self):
        self._scheduler = BackgroundScheduler()
        self._config: CaptureConfig | None = None

    def start(self):
        self._config = self._load_config()
        self._scheduler.start()
        self._apply_config()

    def shutdown(self):
        self._scheduler.shutdown(wait=False)

    def update_config(self, config: CaptureConfig):
        self._config = config
        self._save_config(config)
        self._apply_config()

    def _load_config(self) -> CaptureConfig:
        if not os.path.exists(CONFIG_PATH):
            return CaptureConfig(
                active=False,
                images_per_hour=12,
                scheduler_start_datetime=datetime.now(timezone.utc),
                scheduler_end_datetime=datetime.now(timezone.utc),
            )

        with open(CONFIG_PATH) as f:
            return CaptureConfig(**json.load(f))

    def _save_config(self, config: CaptureConfig):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(config.model_dump(), f, default=str)

    def _apply_config(self):
        self._scheduler.remove_all_jobs()

        if not self._config or not self._config.active:
            return

        interval = int(3600 / self._config.images_per_hour)

        self._scheduler.add_job(
            self._capture_job,
            trigger=IntervalTrigger(seconds=interval),
            id="firewatcher_capture",
            replace_existing=True,
            max_instances=1,
        )

    def _capture_job(self):
        cfg = self._config
        if not cfg:
            return

        now = datetime.now(timezone.utc)

        if cfg.scheduler_end_datetime < now:
            return

        os.makedirs(IMG_FOLDER_PATH, exist_ok=True)
        ts = now.strftime("%Y-%m-%d_%H-%M-%S")
        out = os.path.join(IMG_FOLDER_PATH, f"screenshot_{ts}.jpg")

        capture_image_from_stream(STREAM_URL, out)
