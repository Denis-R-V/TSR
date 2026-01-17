from aiogram import Router
from app.bot.handlers.start import router as start_router
from app.bot.handlers.settings import router as settings_router
from app.bot.handlers.inference import router as inference_router

def build_router() -> Router:
    r = Router()
    r.include_router(start_router)
    r.include_router(settings_router)
    r.include_router(inference_router)
    return r
