"""YOLO-based face detector.

Reference implementation for collaborators adding new detectors. Mirrors the
behaviour of the YOLO branch in the original face_detection.py:
- Six selectable weight files, downloaded from the project's GitHub Releases
  on first use.
- Device selection: MPS on Apple Silicon, CUDA if available, else CPU.
- Returns one Detection per bounding box.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import List, Optional

import numpy as np

from .base import Detection, VisionDetector, register_detector

logger = logging.getLogger(__name__)

# NOTE: torch and ultralytics are intentionally NOT imported at module load time.
# They are the single largest contributor to app startup latency (multi-second
# cold import), and importing them here would block the Python subprocess's
# "ready" signal — and therefore the app's loading screen — on every launch.
# Instead we import them lazily inside load()/_select_device(), i.e. only when a
# detection actually runs. Availability is probed cheaply via importlib.util so
# the model list can be populated without paying the import cost.


_RELEASE_BASE = (
    "https://github.com/cardiff-babylab/tinyexplorer-detection-app"
    "/releases/download/v1.0.0-models"
)

_FACE_WEIGHT_URLS = {
    "yolov8n-face.pt": f"{_RELEASE_BASE}/yolov8n-face.pt",
    "yolov8m-face.pt": f"{_RELEASE_BASE}/yolov8m-face.pt",
    "yolov8l-face.pt": f"{_RELEASE_BASE}/yolov8l-face.pt",
    "yolov11m-face.pt": f"{_RELEASE_BASE}/yolov11m-face.pt",
    "yolov11l-face.pt": f"{_RELEASE_BASE}/yolov11l-face.pt",
    "yolov12l-face.pt": f"{_RELEASE_BASE}/yolov12l-face.pt",
}


@register_detector("face_yolo")
class FaceYoloDetector(VisionDetector):
    """YOLO face detector. Variant names are .pt filenames."""

    name = "face"
    variants = list(_FACE_WEIGHT_URLS.keys())

    def __init__(self, progress_callback=None) -> None:
        super().__init__(progress_callback)
        self.model = None
        self.weights_path: Optional[str] = None

    def load(self, weights_dir: str, variant: Optional[str] = None) -> bool:
        # Heavy imports happen here (lazily), not at module import time.
        try:
            from ultralytics import YOLO
        except ImportError as e:
            self._emit(f"YOLO/Ultralytics is not available in this environment: {e}")
            return False

        variant = variant or self.variants[0]
        if variant not in _FACE_WEIGHT_URLS:
            self._emit(f"Unknown YOLO face weight: {variant}")
            return False

        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, variant)

        if not os.path.exists(weights_path):
            if not self._download_weights(variant, weights_path):
                return False
        else:
            self._emit(f"Found existing model: {variant}")

        device = self._select_device()
        self._emit(f"Loading YOLO model: {weights_path}")
        self.model = YOLO(weights_path)
        try:
            self.model = self.model.to(device)
            self._emit(f"Using device: {device}")
        except Exception as e:
            self._emit(f"Device {device} failed, falling back to CPU: {e}")
            self.model = self.model.to("cpu")

        self._loaded_variant = variant
        self.weights_path = weights_path
        self._emit(f"YOLO model loaded successfully: {weights_path}")
        return True

    def detect_image(self, image: np.ndarray, confidence_threshold: float) -> List[Detection]:
        if self.model is None:
            raise RuntimeError("FaceYoloDetector.detect_image called before load()")

        results = self.model.predict(
            source=image,
            conf=confidence_threshold,
            save=False,
            save_txt=False,
            save_conf=False,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            width = float(x2 - x1)
            height = float(y2 - y1)
            x_center = float(x1 + width / 2)
            y_center = float(y1 + height / 2)
            detections.append(
                Detection(
                    label=self.name,
                    confidence=float(confidences[i]),
                    bbox=(x_center, y_center, width, height),
                )
            )
        return detections

    def _download_weights(self, variant: str, dest_path: str) -> bool:
        url = _FACE_WEIGHT_URLS[variant]
        self._emit(f"Downloading {variant} from GitHub...")
        try:
            import requests  # lazy: only needed when actually downloading YOLO weights

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            written = 0
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    written += len(chunk)
                    if total > 0:
                        pct = (written / total) * 100
                        self._emit(f"Downloading {variant}: {pct:.1f}%")
            self._emit(f"Downloaded {variant} successfully to {dest_path}")
            return True
        except Exception as e:
            self._emit(f"Error downloading {variant}: {e}")
            try:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
            except OSError:
                pass
            return False

    @staticmethod
    def _select_device() -> str:
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"


def is_available() -> bool:
    """Whether YOLO/Ultralytics is importable. Used by the orchestrator's
    legacy ``get_available_models``.

    Uses ``importlib.util.find_spec`` so we can report availability WITHOUT
    actually importing torch/ultralytics (which would defeat the lazy-load and
    slow startup). find_spec only locates the module; it does not execute it."""
    return (
        importlib.util.find_spec("ultralytics") is not None
        and importlib.util.find_spec("torch") is not None
    )
