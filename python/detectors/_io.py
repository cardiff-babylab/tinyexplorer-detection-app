"""Cross-platform image I/O helpers.

OpenCV's imread/imwrite fail on Windows paths containing spaces or unicode.
These wrappers fall back to numpy + imdecode/imencode for those cases.
"""

import os
from typing import Optional

import cv2
import numpy as np


def safe_imread(file_path: str) -> Optional[np.ndarray]:
    """Read an image, falling back to numpy decode for unicode/space paths."""
    try:
        img = cv2.imread(file_path)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        if data.size > 0:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        pass
    return None


def safe_imwrite(file_path: str, image: np.ndarray) -> bool:
    """Write an image, falling back to numpy encode for unicode/space paths."""
    try:
        if cv2.imwrite(file_path, image):
            return True
    except Exception:
        pass
    try:
        ext = os.path.splitext(file_path)[1] or ".jpg"
        ok, encoded = cv2.imencode(ext, image)
        if ok:
            encoded.tofile(file_path)
            return True
    except Exception:
        return False
    return False
