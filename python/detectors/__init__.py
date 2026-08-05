"""Detector registry for TinyExplorer.

Importing this package auto-discovers every sibling ``*.py`` file (skipping
those starting with ``_`` and ``base.py`` itself) so each detector's
``@register_detector`` decorator fires. Drop a new detector file in this
directory and it becomes available without touching any other code.
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
from typing import Dict, List, Optional, Type

from .base import (
    AudioDetector,
    BaseDetector,
    DETECTOR_FACTORY,
    Detection,
    ProgressCallback,
    VisionDetector,
    register_detector,
)

logger = logging.getLogger(__name__)


def _auto_import_detectors() -> None:
    """Import every detector module in this package so registrations fire."""
    package_dir = os.path.dirname(__file__)
    for _, mod_name, is_pkg in pkgutil.iter_modules([package_dir]):
        if is_pkg:
            continue
        if mod_name.startswith("_") or mod_name == "base":
            continue
        try:
            importlib.import_module(f"{__name__}.{mod_name}")
        except ImportError as e:
            # A detector may rely on an optional dependency that isn't
            # installed in this environment. Skip it; the factory just won't
            # have it. This matches the conditional-import behaviour of the
            # original face_detection.py.
            logger.info("Detector %s skipped (optional deps missing): %s", mod_name, e)
        except Exception as e:
            logger.error("Detector %s failed to import: %s", mod_name, e)


def DetectorFactory(key: str) -> Optional[Type[BaseDetector]]:
    """Look up a detector class by registry key. Returns None if unknown."""
    return DETECTOR_FACTORY.get(key)


def list_detectors() -> Dict[str, Dict[str, object]]:
    """Return ``{key: {"name": ..., "mode": ..., "variants": [...], "kind": ...}}``
    for every registered detector. ``mode`` is the modality the UI groups/filters
    on (falls back to ``name`` when a detector didn't set it explicitly). Used by
    the IPC layer to populate the UI."""
    out: Dict[str, Dict[str, object]] = {}
    for key, cls in DETECTOR_FACTORY.items():
        if issubclass(cls, VisionDetector):
            kind = "vision"
        elif issubclass(cls, AudioDetector):
            kind = "audio"
        else:
            kind = "unknown"
        out[key] = {
            "name": cls.name,
            "mode": cls.mode or cls.name,
            "variants": list(cls.variants),
            "kind": kind,
        }
    return out


def detector_for_variant(variant: str) -> Optional[Type[BaseDetector]]:
    """Find the detector class that owns a given variant string.

    Supports the legacy IPC where the UI sends a model filename
    (e.g. "yolov8n-face.pt" or "RetinaFace") rather than a detector key.
    Returns the first match; raises if the same variant is claimed by two
    detectors.
    """
    matches = [cls for cls in DETECTOR_FACTORY.values() if variant in cls.variants]
    if not matches:
        return None
    if len(matches) > 1:
        names = [cls.__name__ for cls in matches]
        raise RuntimeError(
            f"Variant {variant!r} is claimed by multiple detectors: {names}"
        )
    return matches[0]


_auto_import_detectors()


__all__ = [
    "AudioDetector",
    "BaseDetector",
    "DETECTOR_FACTORY",
    "Detection",
    "DetectorFactory",
    "ProgressCallback",
    "VisionDetector",
    "detector_for_variant",
    "list_detectors",
    "register_detector",
]
