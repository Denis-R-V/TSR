from __future__ import annotations
from dataclasses import dataclass
import aiosqlite

@dataclass(frozen=True)
class EffectiveSettings:
    user_id: int
    detector_thr: float
    classifier_thr: float
    debug: bool
    detector_thr_override: float | None
    classifier_thr_override: float | None

class SettingsRepo:
    def __init__(self, db_path: str, default_detector_thr: float, default_classifier_thr: float):
        self.db_path = db_path
        self.default_detector_thr = default_detector_thr
        self.default_classifier_thr = default_classifier_thr

    async def _ensure(self, user_id: int) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO user_settings(user_id, detector_thr_override, classifier_thr_override, debug) "
                "VALUES (?, NULL, NULL, 0)",
                (user_id,),
            )
            await db.commit()

    async def get_effective(self, user_id: int) -> EffectiveSettings:
        await self._ensure(user_id)
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT detector_thr_override, classifier_thr_override, debug FROM user_settings WHERE user_id=?",
                (user_id,),
            )
            det_ov, cls_ov, dbg = await cur.fetchone()

        det = float(det_ov) if det_ov is not None else self.default_detector_thr
        cls = float(cls_ov) if cls_ov is not None else self.default_classifier_thr

        return EffectiveSettings(
            user_id=user_id,
            detector_thr=det,
            classifier_thr=cls,
            debug=bool(dbg),
            detector_thr_override=float(det_ov) if det_ov is not None else None,
            classifier_thr_override=float(cls_ov) if cls_ov is not None else None,
        )

    async def set_detector_thr(self, user_id: int, value: float) -> None:
        await self._ensure(user_id)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("UPDATE user_settings SET detector_thr_override=? WHERE user_id=?", (value, user_id))
            await db.commit()

    async def set_classifier_thr(self, user_id: int, value: float) -> None:
        await self._ensure(user_id)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("UPDATE user_settings SET classifier_thr_override=? WHERE user_id=?", (value, user_id))
            await db.commit()

    async def set_both(self, user_id: int, det: float, cls: float) -> None:
        await self._ensure(user_id)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE user_settings SET detector_thr_override=?, classifier_thr_override=? WHERE user_id=?",
                (det, cls, user_id),
            )
            await db.commit()

    async def reset(self, user_id: int) -> None:
        await self._ensure(user_id)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE user_settings SET detector_thr_override=NULL, classifier_thr_override=NULL, debug=0 WHERE user_id=?",
                (user_id,),
            )
            await db.commit()

    async def toggle_debug(self, user_id: int) -> bool:
        await self._ensure(user_id)
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute("SELECT debug FROM user_settings WHERE user_id=?", (user_id,))
            (dbg,) = await cur.fetchone()
            new_val = 0 if dbg else 1
            await db.execute("UPDATE user_settings SET debug=? WHERE user_id=?", (new_val, user_id))
            await db.commit()
        return bool(new_val)
