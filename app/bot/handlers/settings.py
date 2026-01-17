from __future__ import annotations
from aiogram import Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.fsm.context import FSMContext

from app.core.config import Config
from app.storage.repo import SettingsRepo
from app.utils.validators import parse_thr
from app.bot.states import SettingsFSM
from app.bot.keyboards.settings import settings_keyboard

router = Router()

def fmt(eff, cfg: Config) -> str:
    def line(name: str, eff_v: float, def_v: float, ov: float | None) -> str:
        ov_s = "—" if ov is None else f"{ov:.3f}"
        return f"{name}: {eff_v:.3f} (default {def_v:.3f}, override {ov_s})"
    return "\n".join([
        line("detector_thr", eff.detector_thr, cfg.default_detector_thr, eff.detector_thr_override),
        line("classifier_thr", eff.classifier_thr, cfg.default_classifier_thr, eff.classifier_thr_override),
        f"debug: {'ON' if eff.debug else 'OFF'}",
    ])

@router.message(F.text == "/settings")
async def settings(message: Message, cfg: Config, settings_repo: SettingsRepo) -> None:
    eff = await settings_repo.get_effective(message.from_user.id)
    await message.answer(fmt(eff, cfg), reply_markup=settings_keyboard())

@router.callback_query(F.data.startswith("settings:"))
async def settings_cb(cq: CallbackQuery, state: FSMContext, cfg: Config, settings_repo: SettingsRepo) -> None:
    action = cq.data.split(":", 1)[1]
    uid = cq.from_user.id

    if action == "det":
        await state.set_state(SettingsFSM.det)
        await cq.message.answer("Введи detector_thr (0..1):")
    elif action == "cls":
        await state.set_state(SettingsFSM.cls)
        await cq.message.answer("Введи classifier_thr (0..1):")
    elif action == "both":
        await state.set_state(SettingsFSM.both_det)
        await cq.message.answer("Введи detector_thr (0..1):")
    elif action == "reset":
        await settings_repo.reset(uid)
        eff = await settings_repo.get_effective(uid)
        await cq.message.answer("Сброшено.\n" + fmt(eff, cfg), reply_markup=settings_keyboard())
    elif action == "debug":
        await settings_repo.toggle_debug(uid)
        eff = await settings_repo.get_effective(uid)
        await cq.message.answer(fmt(eff, cfg), reply_markup=settings_keyboard())

    await cq.answer()

@router.message(SettingsFSM.det)
async def set_det(message: Message, state: FSMContext, cfg: Config, settings_repo: SettingsRepo) -> None:
    try:
        v = parse_thr(message.text)
    except Exception:
        await message.answer("Нужно число 0..1. Пример: 0.90")
        return
    await settings_repo.set_detector_thr(message.from_user.id, v)
    await state.clear()
    eff = await settings_repo.get_effective(message.from_user.id)
    await message.answer("Готово.\n" + fmt(eff, cfg), reply_markup=settings_keyboard())

@router.message(SettingsFSM.cls)
async def set_cls(message: Message, state: FSMContext, cfg: Config, settings_repo: SettingsRepo) -> None:
    try:
        v = parse_thr(message.text)
    except Exception:
        await message.answer("Нужно число 0..1. Пример: 0.90")
        return
    await settings_repo.set_classifier_thr(message.from_user.id, v)
    await state.clear()
    eff = await settings_repo.get_effective(message.from_user.id)
    await message.answer("Готово.\n" + fmt(eff, cfg), reply_markup=settings_keyboard())

@router.message(SettingsFSM.both_det)
async def both_det(message: Message, state: FSMContext) -> None:
    try:
        v = parse_thr(message.text)
    except Exception:
        await message.answer("Нужно число 0..1. Пример: 0.90")
        return
    await state.update_data(det=v)
    await state.set_state(SettingsFSM.both_cls)
    await message.answer("Теперь введи classifier_thr (0..1):")

@router.message(SettingsFSM.both_cls)
async def both_cls(message: Message, state: FSMContext, cfg: Config, settings_repo: SettingsRepo) -> None:
    try:
        v2 = parse_thr(message.text)
    except Exception:
        await message.answer("Нужно число 0..1. Пример: 0.90")
        return
    data = await state.get_data()
    v1 = float(data["det"])
    await settings_repo.set_both(message.from_user.id, v1, v2)
    await state.clear()
    eff = await settings_repo.get_effective(message.from_user.id)
    await message.answer("Готово.\n" + fmt(eff, cfg), reply_markup=settings_keyboard())
