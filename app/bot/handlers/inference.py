from __future__ import annotations
from aiogram import Router, F
from aiogram.types import Message, BufferedInputFile

from app.storage.repo import SettingsRepo
from app.services.inference.service import InferenceService

router = Router()

def extract_image_file_id(message: Message) -> str | None:
    if message.photo:
        return message.photo[-1].file_id
    if message.document:
        mt = (message.document.mime_type or "")
        if mt.startswith("image/"):
            return message.document.file_id
        name = (message.document.file_name or "").lower()
        if name.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
            return message.document.file_id
    return None

@router.message(F.photo | F.document)
async def on_image(message: Message, settings_repo: SettingsRepo, inference_service: InferenceService) -> None:
    file_id = extract_image_file_id(message)
    if not file_id:
        await message.answer("Пришли изображение (photo) или файл-изображение (document).")
        return

    eff = await settings_repo.get_effective(message.from_user.id)

    tg_file = await message.bot.get_file(file_id)
    stream = await message.bot.download_file(tg_file.file_path)
    image_bytes = stream.read()

    out = await inference_service.predict(
        image_bytes=image_bytes,
        det_thr=eff.detector_thr,
        cls_thr=eff.classifier_thr,
        debug=eff.debug,
    )

    await message.answer_photo(
        BufferedInputFile(out.image_bytes, filename="result.png"),
        caption=(out.debug_text[:900] if (eff.debug and out.debug_text) else None),
    )

    if out.labels_text:
        await message.answer(out.labels_text)
