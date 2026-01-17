from aiogram import Router, F
from aiogram.types import Message

router = Router()

@router.message(F.text.in_({"/start", "/help"}))
async def start(message: Message) -> None:
    await message.answer(
        "Пришли изображение (фото или файл-изображение).\n"
        "Настройки: /settings\n"
        "Threshold вводится числом 0..1 (например 0.90)."
    )
