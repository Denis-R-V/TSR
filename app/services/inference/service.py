from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass

import torch
from PIL import Image

from src.execute import Builder


@dataclass(frozen=True)
class InferenceOut:
    image_bytes: bytes
    labels_text: str
    debug_text: str | None


class InferenceService:
    """
    Builder мутирует внутренние поля thresholds/debug_mode на время predict_single.
    Чтобы исключить гонки при параллельных апдейтах — один общий lock (1 инференс за раз).
    """

    def __init__(
        self,
        device_id: int,
        models_path: str,
        detector_file: str,
        classifier_file: str,
        default_detector_thr: float,
        default_classifier_thr: float,
        default_multiplication_thr: float,
    ):
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.device = device

        self._mul_thr = float(default_multiplication_thr)

        self.model = Builder(
            device=device,
            detector_path=f"{models_path}/{detector_file}",
            classifier_path=f"{models_path}/{classifier_file}",
            detector_threshold=default_detector_thr,
            classifier_threshold=default_classifier_thr,
            multiplication_threshold=self._mul_thr,
            debug_mode=False,
        )

        self._lock = asyncio.Lock()

    @staticmethod
    def _pil_to_png_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _predict_sync(self, image_bytes: bytes, det_thr: float, cls_thr: float, debug: bool) -> InferenceOut:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        img_pred, description = self.model.predict_single_visualized(
            img,
            display_img=False,
            save_path=None,
            detector_threshold=det_thr,
            classifier_threshold=cls_thr,
            multiplication_threshold=self._mul_thr,  # фиксированный, не в UI
            debug_mode=debug,
        )

        out_bytes = self._pil_to_png_bytes(img_pred)
        labels_text = "\n".join(description) if description else ""

        dbg = None
        if debug:
            dbg = (
                f"detector_thr={det_thr:.3f}\n"
                f"classifier_thr={cls_thr:.3f}\n"
                f"multiplication_thr={self._mul_thr:.3f}\n"
                f"labels_count={len(description) if description else 0}"
            )

        return InferenceOut(image_bytes=out_bytes, labels_text=labels_text, debug_text=dbg)

    async def predict(self, image_bytes: bytes, det_thr: float, cls_thr: float, debug: bool) -> InferenceOut:
        async with self._lock:
            return await asyncio.to_thread(self._predict_sync, image_bytes, det_thr, cls_thr, debug)
