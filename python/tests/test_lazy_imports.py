"""Regression tests locking in the lazy-load of the heavy ML stacks.

The app's Python subprocess emits its "ready" signal (which dismisses the
loading screen) only after its imports finish. Historically the detector
modules imported torch/ultralytics (YOLO) and tensorflow/retina-face at module
top level, so "ready" was gated behind a multi-second cold import on every
launch. These tests assert that importing the detector registry and the
orchestrator does NOT drag torch/TensorFlow into ``sys.modules`` — if anyone
re-introduces a top-level heavy import, these fail.

The tests only need the light deps (numpy, opencv, requests). torch/TF do not
need to be installed for the guard to be meaningful: when they ARE installed,
an accidental eager import would show up in ``sys.modules`` here; when they are
NOT installed, the lazy code path is exercised and must still succeed.
"""

import os
import subprocess
import sys
import textwrap

import pytest

# Make the bundled ``python/`` modules importable regardless of where pytest is
# invoked from (tests live in ``python/tests/``).
PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

_HEAVY = ("torch", "tensorflow", "ultralytics", "retinaface")


def _heavy_loaded():
    """Heavy modules currently present in this interpreter's sys.modules."""
    return [m for m in _HEAVY if m in sys.modules]


def test_importing_detectors_is_torch_free():
    """Importing the detectors package must not import torch/TensorFlow."""
    pytest.importorskip("numpy")
    pytest.importorskip("requests")

    # Anything already imported by the test session shouldn't mask a regression:
    # assert specifically that importing the package doesn't ADD a heavy module.
    before = set(_heavy_loaded())
    import detectors  # noqa: F401

    added = set(_heavy_loaded()) - before
    assert not added, f"Importing detectors eagerly imported heavy modules: {added}"


def test_detector_registry_lists_both_backends():
    """Both detectors register regardless of which ML backend is installed."""
    pytest.importorskip("numpy")
    pytest.importorskip("requests")

    from detectors import list_detectors

    registry = list_detectors()
    assert "face_yolo" in registry
    assert "face_retinaface" in registry
    # variants are static metadata (no torch needed to enumerate them)
    assert registry["face_yolo"]["variants"], "YOLO should advertise weight variants"


def test_is_available_does_not_import_heavy():
    """is_available() must answer via find_spec, never by importing the stack."""
    pytest.importorskip("numpy")
    pytest.importorskip("requests")

    from detectors import face_yolo, face_retinaface

    before = set(_heavy_loaded())
    yolo_ok = face_yolo.is_available()
    retina_ok = face_retinaface.is_available()

    assert isinstance(yolo_ok, bool)
    assert isinstance(retina_ok, bool)
    added = set(_heavy_loaded()) - before
    assert not added, f"is_available() eagerly imported heavy modules: {added}"


def test_importing_face_detection_is_torch_free():
    """Importing the orchestrator (which the subprocess does on startup) is
    the actual ready-path import — it must stay torch/TF-free."""
    pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    before = set(_heavy_loaded())
    import face_detection  # noqa: F401

    added = set(_heavy_loaded()) - before
    assert not added, f"Importing face_detection eagerly imported heavy modules: {added}"


def test_get_available_models_is_torch_free():
    """Populating the model dropdown must not import torch/TensorFlow."""
    pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    from face_detection import FaceDetectionProcessor

    processor = FaceDetectionProcessor()
    before = set(_heavy_loaded())
    models = processor.get_available_models()

    assert isinstance(models, (list, tuple))
    added = set(_heavy_loaded()) - before
    assert not added, f"get_available_models() eagerly imported heavy modules: {added}"


def test_startup_path_torch_free_even_when_heavy_installed(tmp_path):
    """The strongest guard: even when torch/TF/ultralytics/retina-face ARE
    importable, walking the whole startup/ready path must import none of them.

    We can't assume the real heavy stacks are installed in the test env, so we
    fabricate importable stub modules and run the import path in a *fresh*
    subprocess (to avoid this test's own sys.modules affecting others). If any
    stub gets imported, the lazy-load has regressed.
    """
    pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    stub_dir = tmp_path / "fakemods"
    stub_dir.mkdir()
    for name in ("torch", "ultralytics", "tensorflow", "retinaface"):
        (stub_dir / f"{name}.py").write_text(f"IMPORTED_{name} = True\n")

    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(stub_dir)!r})   # heavy stubs importable (simulate installed)
        sys.path.insert(0, {PYTHON_DIR!r})

        import detectors
        import face_detection
        from face_detection import FaceDetectionProcessor

        FaceDetectionProcessor().get_available_models()
        detectors.list_detectors()
        detectors.face_yolo.is_available()
        detectors.face_retinaface.is_available()

        heavy = [n for n in ("torch", "ultralytics", "tensorflow", "retinaface")
                 if n in sys.modules]
        if heavy:
            print("LEAKED:" + ",".join(heavy))
            sys.exit(1)
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "Startup path imported heavy modules when they were installed:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout
