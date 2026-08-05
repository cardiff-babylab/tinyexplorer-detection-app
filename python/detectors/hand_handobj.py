"""HandObject hand detector (100DOH / TinyExplorer-tuned Faster R-CNN res101).

Registers two selectable checkpoints:

* ``HandObject-Tuned``   – TinyExplorer-tuned, includes own/other ownership head.
* ``HandObject-Baseline``– original 100DOH baseline (no ownership).

The ~361 MB checkpoints are **not** bundled; they are downloaded on first use to
the app's model-cache directory (mirroring how the YOLO/RetinaFace weights are
fetched on demand). Set ``HANDOBJ_WEIGHTS_DIR`` to a folder containing the
``.pth`` files to use local copies instead of downloading (used for dev/testing).

The heavy PyTorch model + native extension are imported lazily inside ``load``
so that importing this module in a face-only environment still registers the
detector (it just reports itself unavailable).
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .base import Detection, VisionDetector, register_detector

logger = logging.getLogger(__name__)

# Cheap availability probe: is torch importable at all in this environment?
_TORCH_AVAILABLE = False
try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except Exception as e:  # pragma: no cover - env dependent
    print(f"HandObject detector: torch not available ({e})", file=sys.stderr)


# Base URL the checkpoints are downloaded from on first use. Points at the
# project's own GitHub Release so the weights are self-hosted (mirrors the YOLO
# face weights in face_yolo.py). Override with the HANDOBJ_WEIGHTS_BASE_URL env
# var for local mirrors or testing.
_DEFAULT_WEIGHTS_BASE_URL = (
    "https://github.com/cardiff-babylab/tinyexplorer-detection-app/"
    "releases/download/handobj-weights-v1"
)


@dataclass(frozen=True)
class _VariantSpec:
    filename: str
    has_ownership: bool
    sha256: Optional[str] = None  # if set, downloaded weights are verified against it


# variant name -> checkpoint spec. sha256 values are of the assets hosted on the
# handobj-weights-v1 GitHub Release; downloads are verified against them.
_VARIANTS: Dict[str, _VariantSpec] = {
    "HandObject-Tuned": _VariantSpec(
        filename="faster_rcnn_1_15_2739.pth",
        has_ownership=True,
        sha256="56c4aa5dc973eb28b9d07266fcb0aaff8f96f9e9ca9afe7c3b85034513093d13",
    ),
    "HandObject-Baseline": _VariantSpec(
        filename="faster_rcnn_1_8_132028.pth",
        has_ownership=False,
        sha256="3444e3b992417cb6adfcd71539ca8c92602c21a3e2fdfff4628dbdbdf8a2dd03",
    ),
}


@register_detector("hand_handobj")
class HandObjectDetector(VisionDetector):
    """Hand detector reporting bbox, contact state, side and (Tuned) ownership."""

    name = "hand"
    variants = ["HandObject-Tuned", "HandObject-Baseline"]

    def __init__(self, progress_callback=None) -> None:
        super().__init__(progress_callback)
        self._model = None
        self._has_ownership = False

    # ---- weight resolution -------------------------------------------------

    def _weights_base_url(self) -> str:
        return os.environ.get("HANDOBJ_WEIGHTS_BASE_URL", _DEFAULT_WEIGHTS_BASE_URL)

    def _ensure_weights(self, weights_dir: str, spec: _VariantSpec) -> Optional[str]:
        """Return a path to the checkpoint, fetching it on first use if needed."""
        cache_dir = os.path.join(weights_dir, "handobj")
        os.makedirs(cache_dir, exist_ok=True)
        target = os.path.join(cache_dir, spec.filename)

        if os.path.exists(target) and os.path.getsize(target) > 0:
            return target

        # Dev/testing / manual-placement escape hatch: use a local copy.
        local_dir = os.environ.get("HANDOBJ_WEIGHTS_DIR")
        if local_dir:
            local_path = os.path.join(local_dir, spec.filename)
            if os.path.exists(local_path):
                self._emit(f"📁 Using local HandObject weights: {local_path}")
                return local_path

        url = f"{self._weights_base_url()}/{spec.filename}"
        self._emit(f"⏳ Downloading {self._loaded_variant or spec.filename}: 0%")
        try:
            self._download_with_progress(url, target)
        except Exception as e:
            self._emit(f"❌ Failed to download HandObject weights from {url}: {e}")
            # Clean up a partial file so a retry starts fresh.
            if os.path.exists(target):
                try:
                    os.remove(target)
                except OSError:
                    pass
            return None

        if spec.sha256 and not self._verify_sha256(target, spec.sha256):
            self._emit("❌ HandObject weights failed checksum verification")
            try:
                os.remove(target)
            except OSError:
                pass
            return None
        return target

    def _download_with_progress(self, url: str, target: str) -> None:
        tmp = target + ".part"
        with urllib.request.urlopen(url) as resp:  # nosec - fixed release host
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            last_pct = -1
            chunk = 1024 * 256
            with open(tmp, "wb") as fh:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    fh.write(buf)
                    downloaded += len(buf)
                    if total > 0:
                        pct = int(downloaded * 100 / total)
                        if pct != last_pct and pct % 2 == 0:
                            last_pct = pct
                            self._emit(
                                f"⏳ Downloading {self._loaded_variant}: {pct}%"
                            )
        shutil.move(tmp, target)
        self._emit(f"✅ Downloaded HandObject weights to {target}")

    @staticmethod
    def _verify_sha256(path: str, expected: str) -> bool:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest() == expected

    # ---- detector lifecycle ------------------------------------------------

    def load(self, weights_dir: str, variant: Optional[str] = None) -> bool:
        if not _TORCH_AVAILABLE:
            self._emit(
                "HandObject detector unavailable: PyTorch is not installed in this "
                "environment"
            )
            return False

        variant = variant or self.variants[0]
        spec = _VARIANTS.get(variant)
        if spec is None:
            self._emit(f"❌ Unknown HandObject variant: {variant}")
            return False

        self._loaded_variant = variant
        weights_path = self._ensure_weights(weights_dir, spec)
        if not weights_path:
            return False

        try:
            from handobj.inference import HandObjectModel
        except Exception as e:
            self._emit(f"❌ HandObject native extension failed to load: {e}")
            return False

        try:
            self._emit(f"⏳ Loading HandObject model ({variant})...")
            self._model = HandObjectModel(
                weights_path, has_ownership=spec.has_ownership
            )
            self._has_ownership = spec.has_ownership
            self._emit(f"✅ HandObject model ready ({variant})")
            return True
        except Exception as e:
            self._emit(f"❌ Error loading HandObject model: {e}")
            self._model = None
            return False

    def detect_image(
        self, image: np.ndarray, confidence_threshold: float
    ) -> List[Detection]:
        if self._model is None:
            raise RuntimeError("HandObjectDetector.detect_image called before load()")

        hands = self._model.detect(image, thresh_hand=confidence_threshold)
        detections: List[Detection] = []
        for hand_id, h in enumerate(hands):
            width = h.x2 - h.x1
            height = h.y2 - h.y1
            detections.append(
                Detection(
                    label=self.name,
                    confidence=h.score,
                    bbox=(h.x1, h.y1, width, height),
                    extra={
                        "hand_id": hand_id,
                        "state": h.state,
                        "state_label": h.state_label,
                        "side": h.side,
                        "owner": h.owner,
                    },
                )
            )
        return detections


def is_available() -> bool:
    """True if this environment can actually run the HandObject detector."""
    if not _TORCH_AVAILABLE:
        return False
    try:
        from handobj import inference  # noqa: F401

        return True
    except Exception:
        return False
