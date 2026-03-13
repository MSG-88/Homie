from __future__ import annotations
import logging
from homie_core.screen_reader.pii_filter import PIIFilter

logger = logging.getLogger(__name__)

try:
    import mss
    import mss.tools
except ImportError:
    mss = None  # type: ignore[assignment]


class OCRReader:
    def __init__(self, pii_filter: PIIFilter):
        self._pii_filter = pii_filter

    def capture_screen(self) -> bytes | None:
        if mss is None:
            return None
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # primary monitor
                shot = sct.grab(monitor)
                return mss.tools.to_png(shot.rgb, shot.size)
        except Exception:
            logger.debug("Screen capture failed", exc_info=True)
            return None

    def extract_text(self, image_bytes: bytes) -> str | None:
        """Extract text via Windows OCR or Tesseract. Returns PII-filtered text."""
        raw = self._ocr_windows(image_bytes) or self._ocr_tesseract(image_bytes)
        if raw:
            return self._apply_pii_filter(raw)
        return None

    def _ocr_windows(self, image_bytes: bytes) -> str | None:
        """Try Windows.Media.Ocr via WinRT."""
        try:
            import wrt_ocr  # placeholder — actual WinRT binding
            return None  # TODO: implement WinRT OCR path
        except ImportError:
            return None

    def _ocr_tesseract(self, image_bytes: bytes) -> str | None:
        """Fallback to Tesseract if available."""
        try:
            from PIL import Image
            import pytesseract
            import io
            img = Image.open(io.BytesIO(image_bytes))
            return pytesseract.image_to_string(img)
        except ImportError:
            return None
        except Exception:
            logger.debug("Tesseract OCR failed", exc_info=True)
            return None

    def _apply_pii_filter(self, text: str) -> str:
        return self._pii_filter.filter(text)
