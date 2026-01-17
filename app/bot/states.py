from aiogram.fsm.state import State, StatesGroup

class SettingsFSM(StatesGroup):
    det = State()
    cls = State()
    both_det = State()
    both_cls = State()
