from __future__ import annotations

import asyncio

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

from app.core.config import load_config
from app.core.logging import setup_logging
from app.storage.db import init_db
from app.storage.repo import SettingsRepo
from app.services.inference.service import InferenceService
from app.bot.router import build_router


async def main() -> None:
    setup_logging()
    cfg = load_config()

    await init_db(cfg.db_path)

    bot = Bot(token=cfg.bot_token)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(build_router())

    settings_repo = SettingsRepo(
        db_path=cfg.db_path,
        default_detector_thr=cfg.default_detector_thr,
        default_classifier_thr=cfg.default_classifier_thr,
    )

    inference_service = InferenceService(
        device_id=cfg.device_id,
        models_path=cfg.models_path,
        detector_file=cfg.detector_file,
        classifier_file=cfg.classifier_file,
        default_detector_thr=cfg.default_detector_thr,
        default_classifier_thr=cfg.default_classifier_thr,
        default_multiplication_thr=cfg.default_multiplication_thr,
    )

    await dp.start_polling(
        bot,
        cfg=cfg,
        settings_repo=settings_repo,
        inference_service=inference_service,
    )


if __name__ == "__main__":
    asyncio.run(main())
