from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

def settings_keyboard() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="Изменить detector_thr", callback_data="settings:det")
    kb.button(text="Изменить classifier_thr", callback_data="settings:cls")
    kb.button(text="Изменить оба", callback_data="settings:both")
    kb.button(text="Сбросить к дефолту", callback_data="settings:reset")
    kb.button(text="Debug ON/OFF", callback_data="settings:debug")
    kb.adjust(2, 1, 2)
    return kb.as_markup()
