from __future__ import annotations
import base64
import logging
import requests
from io import BytesIO

logger = logging.getLogger(__name__)

_PROMPT = (
    "Describe what the user is doing at a high level. "
    "Do not extract specific text, names, or personal data. "
    "Keep the response to one or two sentences."
)


class VisualAnalyzer:
    def __init__(self, engine: str = "cloud", api_base_url: str = "",
                 api_key: str = "", model: str = ""):
        self._engine = engine
        self._api_base_url = api_base_url.rstrip("/")
        self._api_key = api_key
        self._model = model

    def analyze(self, image_bytes: bytes) -> str | None:
        resized = self._resize(image_bytes, max_height=720)
        if self._engine == "cloud":
            return self._analyze_cloud(resized)
        return self._analyze_local(resized)

    def _resize(self, image_bytes: bytes, max_height: int = 720) -> bytes:
        try:
            from PIL import Image
            img = Image.open(BytesIO(image_bytes))
            if img.height > max_height:
                ratio = max_height / img.height
                new_size = (int(img.width * ratio), max_height)
                img = img.resize(new_size, Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            return image_bytes
        except Exception:
            logger.debug("Image resize failed, using original bytes", exc_info=True)
            return image_bytes

    def _analyze_cloud(self, image_bytes: bytes) -> str | None:
        b64 = base64.b64encode(image_bytes).decode()
        try:
            resp = requests.post(
                f"{self._api_base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    }],
                    "max_tokens": 150,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            logger.debug("Cloud visual analysis failed", exc_info=True)
            return None

    def _analyze_local(self, image_bytes: bytes) -> str | None:
        # Placeholder for local multimodal model analysis
        logger.debug("Local visual analysis not yet implemented")
        return None
