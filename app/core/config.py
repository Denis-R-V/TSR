from __future__ import annotations
from dataclasses import dataclass
import os
from dotenv import load_dotenv

@dataclass(frozen=True)
class Config:
    bot_token: str
    db_path: str

    default_detector_thr: float
    default_classifier_thr: float
    default_multiplication_thr: float

    device_id: int
    models_path: str
    tele_images_path: str
    detector_file: str
    classifier_file: str

def load_config() -> Config:
    load_dotenv()

    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN is required")

    return Config(
        bot_token=token,
        db_path=os.getenv("BOT_DB_PATH", "bot_settings.sqlite3"),
        default_detector_thr=float(os.getenv("DEFAULT_DETECTOR_THR", "0.9")),
        default_classifier_thr=float(os.getenv("DEFAULT_CLASSIFIER_THR", "0.9")),
        default_multiplication_thr=float(os.getenv("DEFAULT_MULTIPLICATION_THR", "0.85")),
        device_id=int(os.getenv("DEVICE_ID", "0")),
        models_path=os.getenv("MODELS_PATH", "models"),
        tele_images_path=os.getenv("TELE_IMAGES_PATH", "images/telebot_images"),
        detector_file=os.getenv("DETECTOR_FILE", ""),
        classifier_file=os.getenv("CLASSIFIER_FILE", ""),
    )
