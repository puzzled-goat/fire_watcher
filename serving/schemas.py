from datetime import datetime

from pydantic import BaseModel, Field


class CaptureConfig(BaseModel):
    """schemas used for scheduler"""

    active: bool = False
    images_per_hour: int = Field(gt=0, default=12)

    scheduler_start_datetime: datetime
    scheduler_end_datetime: datetime
