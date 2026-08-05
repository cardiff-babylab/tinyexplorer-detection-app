"""Detector interface and registry.

Stage 1 of the multi-modality refactor. Defines:
- ``Detection`` — single canonical output schema for any detector.
- ``VisionDetector`` / ``AudioDetector`` — base classes a contributor subclasses.
- ``register_detector`` — decorator that registers a class in DETECTOR_FACTORY.

Contract for collaborators (see docs/contributing-detectors.md):
1. Subclass VisionDetector or AudioDetector.
2. Set ``name`` (output label) and ``variants`` (list of selectable weight names).
3. Implement ``load(weights_dir, variant)`` and ``detect_image`` / ``detect_audio``.
4. Decorate the class with ``@register_detector("unique_key")``.
5. Drop the file in python/detectors/. Auto-import does the rest.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Iterator, List, Optional, Tuple, Type

import numpy as np

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str], None]]

DETECTOR_FACTORY: Dict[str, Type["BaseDetector"]] = {}


@dataclass(frozen=True)
class Detection:
    """Canonical detection record.

    ``bbox`` is for vision detectors (xywh in pixels, top-left origin).
    ``time_range`` is for audio detectors (start, end in seconds).
    ``extra`` carries detector-specific fields (landmarks, frame_idx, etc.)
    without expanding the schema.
    """

    label: str
    confidence: float
    bbox: Optional[Tuple[float, float, float, float]] = None
    time_range: Optional[Tuple[float, float]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_legacy_face_dict(self, image_path: str) -> Dict[str, Any]:
        """Render in the legacy {x, y, width, height, confidence, image_path} format
        used by the existing IPC and CSV exporter. Used only by the orchestrator
        shim; new code should consume Detection directly."""
        if self.bbox is None:
            raise ValueError("Detection has no bbox; cannot render as legacy face dict")
        x, y, w, h = self.bbox
        out: Dict[str, Any] = {
            "x": float(x),
            "y": float(y),
            "width": float(w),
            "height": float(h),
            "confidence": float(self.confidence),
            "image_path": image_path,
        }
        out.update(self.extra)
        return out


class BaseDetector(ABC):
    """Common bits for vision and audio detectors."""

    # ``name`` is the per-detection output label (also used as Detection.label).
    # ``mode`` is the modality/UI grouping key ("face", "hand", "speech", ...);
    # the Select Mode widget filters the model dropdown on it. It defaults to
    # ``name`` (they coincide for the current detectors) but is a distinct field
    # so a detector can label detections differently from its mode if needed.
    name: ClassVar[str] = ""
    mode: ClassVar[str] = ""
    variants: ClassVar[List[str]] = []

    def __init__(self, progress_callback: ProgressCallback = None) -> None:
        self.progress_callback = progress_callback
        self._loaded_variant: Optional[str] = None

    def _emit(self, message: str) -> None:
        if self.progress_callback is not None:
            try:
                self.progress_callback(message)
            except Exception as e:
                logger.warning("progress_callback raised: %s", e)

    @abstractmethod
    def load(self, weights_dir: str, variant: Optional[str] = None) -> bool:
        """Materialise/load weights for ``variant``.

        ``weights_dir`` is a writable cache directory (the detector decides what
        to put there). Idempotent — re-calling with the same variant is a no-op.
        Returns True on success.
        """

    @property
    def loaded_variant(self) -> Optional[str]:
        return self._loaded_variant


class VisionDetector(BaseDetector):
    """Per-frame image detector (face, hand, pose, ...)."""

    @abstractmethod
    def detect_image(self, image: np.ndarray, confidence_threshold: float) -> List[Detection]:
        """Run inference on a single BGR image and return detections."""

    def detect_video(
        self,
        video_path: str,
        confidence_threshold: float,
        frame_interval_s: float = 1.0,
    ) -> Iterator[Tuple[int, float, List[Detection]]]:
        """Default: sample one frame every ``frame_interval_s`` seconds and call
        ``detect_image``. Yields (frame_index, timestamp_s, detections).
        Detectors needing temporal context can override.
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, int(round(fps * frame_interval_s))) if fps > 0 else 1
            for frame_idx in range(0, frame_count, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = frame_idx / fps if fps > 0 else 0.0
                yield frame_idx, timestamp, self.detect_image(frame, confidence_threshold)
        finally:
            cap.release()


class AudioDetector(BaseDetector):
    """Detector that consumes a waveform (e.g. speech activity, speaker change)."""

    sample_rate: ClassVar[int] = 16000

    @abstractmethod
    def detect_audio(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        confidence_threshold: float,
    ) -> List[Detection]:
        """Run inference on a 1-D float32 waveform."""

    def detect_file(self, file_path: str, confidence_threshold: float) -> List[Detection]:
        """Default: decode the file with ffmpeg/soundfile and call detect_audio.
        Override if your library can read files directly more efficiently.
        """
        waveform, sr = _decode_audio(file_path, self.sample_rate)
        return self.detect_audio(waveform, sr, confidence_threshold)


def _decode_audio(file_path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    """Lazy audio decode helper. Imported only when an AudioDetector is used,
    so a face-only deployment doesn't pay for soundfile/librosa."""
    try:
        import soundfile as sf  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "AudioDetector.detect_file requires the 'soundfile' package; "
            "install it or override detect_file in your subclass."
        ) from e
    waveform, sr = sf.read(file_path, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sr != target_sr:
        try:
            import librosa  # type: ignore
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except ImportError:
            logger.warning(
                "Sample rate %d != target %d but librosa not available; "
                "passing original waveform through.", sr, target_sr,
            )
    return waveform, sr


def register_detector(key: str) -> Callable[[Type[BaseDetector]], Type[BaseDetector]]:
    """Class decorator: add to DETECTOR_FACTORY under ``key``."""

    def decorator(cls: Type[BaseDetector]) -> Type[BaseDetector]:
        if key in DETECTOR_FACTORY:
            logger.warning("Detector key %r overwritten by %s", key, cls.__name__)
        DETECTOR_FACTORY[key] = cls
        return cls

    return decorator


__all__ = [
    "Detection",
    "BaseDetector",
    "VisionDetector",
    "AudioDetector",
    "register_detector",
    "DETECTOR_FACTORY",
    "ProgressCallback",
]
