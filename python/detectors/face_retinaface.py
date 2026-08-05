"""RetinaFace-based face detector.

The retina-face package downloads its own weights on first use, so ``load``
is a no-op apart from confirming the import worked. We pass numpy arrays
straight through (avoids the original code's path-based call which doesn't
work for video frames extracted in-memory).
"""

from __future__ import annotations

import importlib.util
import logging
from typing import List, Optional

import numpy as np

from .base import Detection, VisionDetector, register_detector

logger = logging.getLogger(__name__)

# NOTE: retina-face (and the TensorFlow it pulls in) is intentionally NOT imported
# at module load time — TensorFlow's cold import is a multi-second startup cost that
# would block the subprocess "ready" signal on every launch. It is imported lazily
# inside load()/detect_image() instead, and availability is probed cheaply via
# importlib.util (which does not execute the module).


@register_detector("face_retinaface")
class FaceRetinaFaceDetector(VisionDetector):
    """RetinaFace face detector. Single variant; the library manages weights."""

    name = "face"
    variants = ["RetinaFace"]

    def load(self, weights_dir: str, variant: Optional[str] = None) -> bool:
        # Heavy import (retina-face + TensorFlow) happens here lazily, not at
        # module load time.
        try:
            from retinaface import RetinaFace  # type: ignore  # noqa: F401
        except ImportError:
            self._emit("RetinaFace is not available in this environment")
            self._emit(
                "Ensure 'retina-face' and 'tensorflow' are installed in the active Python environment"
            )
            return False
        self._emit("RetinaFace will download models on first use if needed")
        self._loaded_variant = variant or self.variants[0]
        self._emit("RetinaFace model ready")
        return True

    def detect_image(self, image: np.ndarray, confidence_threshold: float) -> List[Detection]:
        try:
            from retinaface import RetinaFace  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "FaceRetinaFaceDetector.detect_image called but retina-face is not installed"
            ) from e

        face_detections = RetinaFace.detect_faces(image, threshold=confidence_threshold)
        detections: List[Detection] = []
        if not face_detections or not isinstance(face_detections, dict):
            return detections

        for _, detection in face_detections.items():
            facial_area = detection["facial_area"]
            confidence = float(detection["score"])
            x1, y1, x2, y2 = facial_area
            width = float(x2 - x1)
            height = float(y2 - y1)
            # Original code stored (x1, y1) as the legacy "x"/"y" rather than
            # the box centre. Preserve that to keep CSV output schema stable.
            detections.append(
                Detection(
                    label=self.name,
                    confidence=confidence,
                    bbox=(float(x1), float(y1), width, height),
                )
            )
        return detections


def is_available() -> bool:
    """Whether retina-face is importable, probed cheaply via importlib.util so we
    do NOT import TensorFlow just to answer this."""
    return importlib.util.find_spec("retinaface") is not None
