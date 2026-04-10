"""Homie plugin for on-device image classification using MobileNetV3-Small.

Uses the timm/mobilenetv3_small_100.lamb_in1k model (ONNX-exported or PyTorch)
to tag images locally with zero network calls. Designed for mobile and
resource-constrained devices where Homie runs as a local assistant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ImageNet-1k class count
_NUM_CLASSES = 1000
_DEFAULT_TOP_K = 5
_DEFAULT_INPUT_SIZE = (224, 224)
_MODEL_ID = "timm/mobilenetv3_small_100.lamb_in1k"


@dataclass
class TagResult:
    """Single classification result."""
    label: str
    score: float
    class_index: int


@dataclass
class ImageTagOutput:
    """Full tagging output for one image."""
    path: str
    tags: List[TagResult]
    model_id: str = _MODEL_ID


@dataclass
class MobileNetImageTaggerConfig:
    """Plugin configuration."""
    model_dir: Optional[str] = None
    top_k: int = _DEFAULT_TOP_K
    confidence_threshold: float = 0.1
    device: str = "cpu"
    use_onnx: bool = False


class MobileNetImageTagger:
    """On-device image tagger powered by MobileNetV3-Small.

    Runs entirely locally using either PyTorch or ONNX Runtime.
    No network calls are made after initial model download.
    """

    def __init__(self, config: Optional[MobileNetImageTaggerConfig] = None) -> None:
        self.config = config or MobileNetImageTaggerConfig()
        self._model: Any = None
        self._transform: Any = None
        self._labels: List[str] = []
        self._active = False

    # â”€â”€ Lifecycle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def activate(self) -> None:
        """Load model and preprocessing pipeline. Call once at startup."""
        if self._active:
            logger.debug("MobileNetImageTagger already active")
            return

        if self.config.use_onnx:
            self._load_onnx()
        else:
            self._load_pytorch()

        self._active = True
        logger.info(
            "MobileNetImageTagger activated (backend=%s, device=%s)",
            "onnx" if self.config.use_onnx else "pytorch",
            self.config.device,
        )

    def deactivate(self) -> None:
        """Release model resources."""
        self._model = None
        self._transform = None
        self._labels = []
        self._active = False
        logger.info("MobileNetImageTagger deactivated")

    # â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def tag_image(self, image_path: str | Path) -> ImageTagOutput:
        """Classify a single image and return top-k tags.

        Args:
            image_path: Path to a local image file (JPEG, PNG, etc.).

        Returns:
            ImageTagOutput with ranked tags above the confidence threshold.

        Raises:
            RuntimeError: If the plugin has not been activated.
            FileNotFoundError: If the image does not exist.
        """
        if not self._active:
            raise RuntimeError("Plugin not activated. Call activate() first.")

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        scores = self._infer(path)
        tags = self._scores_to_tags(scores)
        return ImageTagOutput(path=str(path), tags=tags)

    def tag_batch(self, image_paths: List[str | Path]) -> List[ImageTagOutput]:
        """Classify multiple images sequentially."""
        return [self.tag_image(p) for p in image_paths]

    # â”€â”€ PyTorch backend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _load_pytorch(self) -> None:
        try:
            import timm
            import torch
            from timm.data import resolve_data_config, create_transform
        except ImportError as exc:
            raise ImportError(
                "PyTorch backend requires 'timm' and 'torch'. "
                "Install with: pip install timm torch torchvision"
            ) from exc

        model_name = "mobilenetv3_small_100.lamb_in1k"
        cache_dir = self.config.model_dir

        self._model = timm.create_model(
            model_name,
            pretrained=True,
            cache_dir=cache_dir,
        )
        self._model.to(self.config.device)
        self._model.eval()

        data_cfg = resolve_data_config(self._model.pretrained_cfg)
        self._transform = create_transform(**data_cfg)
        self._labels = self._model.pretrained_cfg.get("label_names", [])
        if not self._labels:
            self._labels = self._load_imagenet_labels()

    def _infer_pytorch(self, image_path: Path) -> List[float]:
        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        tensor = self._transform(img).unsqueeze(0).to(self.config.device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.nn.functional.softmax(logits, dim=-1)

        return probs.squeeze().cpu().tolist()

    # â”€â”€ ONNX backend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _load_onnx(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNX backend requires 'onnxruntime'. "
                "Install with: pip install onnxruntime"
            ) from exc

        model_dir = Path(self.config.model_dir or Path.home() / ".homie" / "models")
        onnx_path = model_dir / "mobilenetv3_small_100.onnx"

        if not onnx_path.is_file():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}. "
                f"Export from PyTorch first or download the ONNX file."
            )

        self._model = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._labels = self._load_imagenet_labels()

    def _infer_onnx(self, image_path: Path) -> List[float]:
        import numpy as np
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        img = img.resize(_DEFAULT_INPUT_SIZE, Image.BILINEAR)

        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW

        input_name = self._model.get_inputs()[0].name
        outputs = self._model.run(None, {input_name: arr})
        logits = outputs[0].squeeze()

        # softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        return probs.tolist()

    # â”€â”€ Shared helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _infer(self, image_path: Path) -> List[float]:
        if self.config.use_onnx:
            return self._infer_onnx(image_path)
        return self._infer_pytorch(image_path)

    def _scores_to_tags(self, scores: List[float]) -> List[TagResult]:
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        tags: List[TagResult] = []
        for idx, score in indexed[: self.config.top_k]:
            if score < self.config.confidence_threshold:
                break
            label = self._labels[idx] if idx < len(self._labels) else f"class_{idx}"
            tags.append(TagResult(label=label, score=round(score, 4), class_index=idx))
        return tags

    @staticmethod
    def _load_imagenet_labels() -> List[str]:
        """Return ImageNet-1k labels from timm or fall back to index-based names."""
        try:
            from timm.data import ImageNetInfo
            info = ImageNetInfo()
            return [info.label_name(i) for i in range(_NUM_CLASSES)]
        except Exception:  # noqa: BLE001
            return [f"class_{i}" for i in range(_NUM_CLASSES)]


# â”€â”€ Module-level convenience (matches Homie plugin contract) â”€â”€â”€â”€â”€â”€â”€â”€â”€

_instance: Optional[MobileNetImageTagger] = None


def register(config: Optional[Dict[str, Any]] = None) -> MobileNetImageTagger:
    """Register and return the plugin singleton.

    Args:
        config: Optional dict with keys matching MobileNetImageTaggerConfig fields.

    Returns:
        The activated MobileNetImageTagger instance.
    """
    global _instance
    cfg = MobileNetImageTaggerConfig(**(config or {}))
    _instance = MobileNetImageTagger(cfg)
    _instance.activate()
    return _instance


def activate(config: Optional[Dict[str, Any]] = None) -> MobileNetImageTagger:
    """Alias for register()."""
    return register(config)


def deactivate() -> None:
    """Deactivate and release the plugin singleton."""
    global _instance
    if _instance is not None:
        _instance.deactivate()
        _instance = None
