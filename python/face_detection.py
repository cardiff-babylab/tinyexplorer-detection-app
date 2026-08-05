"""Detection orchestrator (folder walk, video sampling, CSV export, callbacks).

Inference itself is delegated to the registered detectors in
``python/detectors/``. This file no longer contains backend-specific logic
- to add a new detector, drop a file in ``detectors/`` and it appears in the UI.

The public surface (``FaceDetectionProcessor`` constructor + methods) is held
stable for the existing ``subprocess_api*.py`` IPC modules.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import cv2

from detectors import (
    DetectorFactory,
    Detection,
    detector_for_variant,
    list_detectors,
)
from detectors._io import safe_imread, safe_imwrite
from detectors.face_retinaface import is_available as _retinaface_available
from detectors.face_yolo import is_available as _yolo_available


_STATUS_SYMBOLS = {
    "info": "ℹ️",
    "success": "✅",
    "error": "❌",
    "warning": "⚠️",
    "processing": "⏳",
    "folder": "📁",
    "image": "🖼️",
    "video": "🎬",
    "detection": "🔍",
    "face": "👤",
    "complete": "🏁",
}

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")


class FaceDetectionProcessor:
    """Folder/video processing pipeline. Now backend-agnostic."""

    def __init__(
        self,
        progress_callback: Optional[Callable[[str], None]] = None,
        completion_callback: Optional[Callable[[Dict], None]] = None,
    ) -> None:
        self.progress_callback = progress_callback
        self.completion_callback = completion_callback
        self.is_processing = False
        self.results: List[Dict] = []
        self.current_model_path: Optional[str] = None
        self.status_symbols = _STATUS_SYMBOLS
        self.detector = None
        self._model_dir: Optional[str] = None

    # ---- Model directory ---------------------------------------------------

    def _get_model_dir(self) -> str:
        """Return a writable directory for cached weights."""
        if self._model_dir is not None:
            return self._model_dir
        try:
            base_dir = os.environ.get("FACE_MODEL_DIR")
            if not base_dir:
                data_home = os.environ.get(
                    "XDG_DATA_HOME",
                    os.path.join(os.path.expanduser("~"), ".local", "share"),
                )
                base_dir = os.path.join(data_home, "TinyExplorerDetection", "models")
            os.makedirs(base_dir, exist_ok=True)
            self._model_dir = base_dir
            self._emit(f"{_STATUS_SYMBOLS['folder']} Using model directory: {base_dir}")
            return base_dir
        except Exception:
            self._model_dir = os.getcwd()
            return self._model_dir

    # ---- Detector lifecycle ------------------------------------------------

    def load_model(self, model_path: str = "yolov8n-face.pt") -> bool:
        """Resolve ``model_path`` to a registered detector and load its weights."""
        try:
            detector_cls = detector_for_variant(model_path)
            if detector_cls is None:
                self._emit(
                    f"{_STATUS_SYMBOLS['error']} No detector registered for variant: {model_path}"
                )
                return False

            self.detector = detector_cls(progress_callback=self.progress_callback)
            self.current_model_path = model_path
            ok = self.detector.load(self._get_model_dir(), variant=model_path)
            if not ok:
                self.detector = None
            return ok
        except Exception as e:
            self._emit(f"{_STATUS_SYMBOLS['error']} Error loading model: {e}")
            return False

    # ---- Single-image inference --------------------------------------------

    def process_image(
        self,
        image_path: str,
        confidence_threshold: float = 0.5,
        save_results: bool = False,
        result_folder: Optional[str] = None,
    ) -> List[Dict]:
        """Run the loaded detector on one image; optionally save annotated copy."""
        if self.detector is None:
            self._emit(
                f"{_STATUS_SYMBOLS['error']} process_image called before load_model succeeded"
            )
            return []

        image = safe_imread(image_path)
        if image is None:
            self._emit(
                f"{_STATUS_SYMBOLS['error']} Could not load image: {image_path}"
            )
            return []

        try:
            self._emit(
                f"{_STATUS_SYMBOLS['processing']} Running inference on {os.path.basename(image_path)}..."
            )
            detections = self.detector.detect_image(image, confidence_threshold)
        except Exception as e:
            self._emit(
                f"{_STATUS_SYMBOLS['error']} Error processing {os.path.basename(image_path)}: {e}"
            )
            return []

        legacy = [d.to_legacy_face_dict(image_path) for d in detections]

        if save_results and result_folder:
            self._save_annotated_image(image, legacy, image_path, result_folder)

        if legacy:
            self._emit(
                f"{_STATUS_SYMBOLS['face']} Found {len(legacy)} face(s) in {os.path.basename(image_path)}"
            )
        else:
            self._emit(
                f"{_STATUS_SYMBOLS['complete']} No faces detected in {os.path.basename(image_path)}"
            )
        return legacy

    def _draw_detections(self, image, detections: List[Dict]):
        """Return a copy of ``image`` with each detection's bbox + confidence drawn.

        Pure helper used by both image and video pipelines so they render boxes
        identically. The legacy face dict stores YOLO-style centre-xy or
        RetinaFace-style top-left; we normalise to top-left here.
        """
        result_img = image.copy()
        uses_centre = self._detector_uses_centre_xy()
        for det in detections:
            x = det["x"]
            y = det["y"]
            w = det["width"]
            h = det["height"]
            if uses_centre:
                x = x - w / 2
                y = y - h / 2
            cv2.rectangle(
                result_img,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                result_img,
                f"{det['confidence']:.2f}",
                (int(x), int(y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        return result_img

    def _save_annotated_image(
        self,
        image,
        detections: List[Dict],
        image_path: str,
        result_folder: str,
    ) -> None:
        """Draw bboxes onto the image and save into ``result_folder/results/``."""
        try:
            annotated = self._draw_detections(image, detections)
            out_dir = os.path.join(result_folder, "results")
            os.makedirs(out_dir, exist_ok=True)
            safe_imwrite(os.path.join(out_dir, os.path.basename(image_path)), annotated)
        except Exception as e:
            self._emit(
                f"{_STATUS_SYMBOLS['error']} Error saving result image: {e}"
            )

    def _detector_uses_centre_xy(self) -> bool:
        """The legacy YOLO branch stored bbox as (x_centre, y_centre, w, h);
        RetinaFace stored (x1, y1, w, h). Detect which one is loaded so the
        drawing helper interprets them consistently."""
        from detectors.face_yolo import FaceYoloDetector

        return isinstance(self.detector, FaceYoloDetector)

    # ---- Folder / batch ----------------------------------------------------

    def process_folder(
        self,
        folder_path: str,
        confidence_threshold: float = 0.5,
        model_name: str = "yolov8n-face.pt",
        save_results: bool = False,
        results_folder: Optional[str] = None,
    ) -> None:
        self.is_processing = True
        self.results = []
        self.current_confidence = confidence_threshold
        folder_path = os.path.normpath(folder_path)
        if results_folder:
            results_folder = os.path.normpath(results_folder)

        if self.completion_callback:
            self.completion_callback({
                "status": "processing_started",
                "folder_path": folder_path,
                "model": model_name,
                "confidence": confidence_threshold,
            })
            time.sleep(0.1)

        try:
            result_folder = self._prepare_result_folder(save_results, results_folder)

            self._emit(f"{_STATUS_SYMBOLS['info']} Loading model: {model_name}...")
            if not self.load_model(model_name):
                return

            image_files, video_files = self._collect_files(folder_path)
            total_files = len(image_files) + len(video_files)
            if total_files == 0:
                self._emit(
                    f"{_STATUS_SYMBOLS['warning']} No image or video files found in the specified location"
                )
                return

            self._emit(
                f"{_STATUS_SYMBOLS['folder']} Found {len(image_files)} images and {len(video_files)} videos to process"
            )

            self._process_image_batch(
                image_files, confidence_threshold, save_results, result_folder
            )
            self._process_video_batch(video_files, confidence_threshold, result_folder)

            if save_results and result_folder and self.results:
                csv_path = os.path.join(result_folder, "detection_results.csv")
                self.export_results_to_csv(self.results, csv_path)
                summary_csv_path = os.path.join(result_folder, "summary.csv")
                self.export_summary_to_csv(
                    folder_path, image_files, video_files, result_folder, summary_csv_path
                )

            self._emit(
                f"{_STATUS_SYMBOLS['complete']} Processing complete. Found {len(self.results)} face detections across {total_files} files"
            )
            if save_results and result_folder:
                self._emit(f"{_STATUS_SYMBOLS['folder']} Results saved to: {result_folder}")

            if self.completion_callback:
                self.completion_callback({
                    "status": "completed",
                    "results_count": len(self.results),
                    "total_files": total_files,
                    "results_folder": result_folder if save_results else None,
                })
        except Exception as e:
            self._emit(f"{_STATUS_SYMBOLS['error']} Error during processing: {e}")
            if self.completion_callback:
                self.completion_callback({
                    "status": "error",
                    "error": str(e),
                    "results_count": len(self.results),
                })
        finally:
            self.is_processing = False
            if self.completion_callback:
                self.completion_callback({
                    "status": "finished",
                    "is_processing": self.is_processing,
                    "results_count": len(self.results),
                })

    def _prepare_result_folder(
        self,
        save_results: bool,
        results_folder: Optional[str],
    ) -> Optional[str]:
        if not save_results:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if results_folder:
            result_folder = os.path.join(results_folder, f"detection_results_{timestamp}")
            self._emit(f"{_STATUS_SYMBOLS['folder']} Creating results folder: {result_folder}")
        else:
            result_folder = os.path.join(os.getcwd(), f"detection_results_{timestamp}")
            self._emit(
                f"{_STATUS_SYMBOLS['folder']} Creating default results folder: {result_folder}"
            )
        os.makedirs(result_folder, exist_ok=True)
        os.makedirs(os.path.join(result_folder, "results"), exist_ok=True)
        return result_folder

    def _collect_files(self, folder_path: str):
        image_files: List[str] = []
        video_files: List[str] = []
        if os.path.isfile(folder_path):
            lower = folder_path.lower()
            if lower.endswith(_IMAGE_EXTENSIONS):
                image_files.append(folder_path)
            elif lower.endswith(_VIDEO_EXTENSIONS):
                video_files.append(folder_path)
        else:
            for root, _, files in os.walk(folder_path):
                for fname in files:
                    p = os.path.join(root, fname)
                    lower = fname.lower()
                    if lower.endswith(_IMAGE_EXTENSIONS):
                        image_files.append(p)
                    elif lower.endswith(_VIDEO_EXTENSIONS):
                        video_files.append(p)
        return image_files, video_files

    def _process_image_batch(
        self,
        image_files: List[str],
        confidence_threshold: float,
        save_results: bool,
        result_folder: Optional[str],
    ) -> None:
        for i, image_path in enumerate(image_files):
            if not self.is_processing:
                self._emit(f"{_STATUS_SYMBOLS['warning']} Processing stopped by user")
                break
            self._emit(
                f"{_STATUS_SYMBOLS['image']} Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}"
            )
            try:
                detections = self.process_image(
                    image_path, confidence_threshold, save_results, result_folder
                )
                if detections:
                    self.results.extend(detections)
                progress_percent = ((i + 1) / len(image_files)) * 100
                self._emit(
                    f"{_STATUS_SYMBOLS['complete']} Image {i+1}/{len(image_files)} complete ({progress_percent:.1f}%)"
                )
                if self.completion_callback:
                    self.completion_callback({
                        "status": "image_completed",
                        "image_index": i + 1,
                        "total_images": len(image_files),
                        "progress_percent": progress_percent,
                        "detections_in_image": len(detections),
                        "total_detections": len(self.results),
                        "image_path": os.path.basename(image_path),
                    })
                time.sleep(0.05)
            except Exception as e:
                self._emit(
                    f"{_STATUS_SYMBOLS['error']} Failed to process {os.path.basename(image_path)}: {e}"
                )
                continue

    def _process_video_batch(
        self,
        video_files: List[str],
        confidence_threshold: float,
        result_folder: Optional[str],
    ) -> None:
        for i, video_path in enumerate(video_files):
            if not self.is_processing:
                break
            self._emit(
                f"{_STATUS_SYMBOLS['video']} Processing video {i+1}/{len(video_files)}: {os.path.basename(video_path)}"
            )
            if result_folder:
                video_detections = self.process_video(
                    video_path, confidence_threshold, result_folder
                )
                self.results.extend(video_detections)

    # ---- Video sampling ----------------------------------------------------

    def process_video(
        self,
        video_path: str,
        confidence_threshold: float = 0.5,
        result_folder: Optional[str] = None,
    ) -> List[Dict]:
        if self.detector is None:
            self._emit(
                f"{_STATUS_SYMBOLS['error']} process_video called before load_model succeeded"
            )
            return []
        try:
            self._emit(f"{_STATUS_SYMBOLS['video']} Processing video: {os.path.basename(video_path)}")
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frames_to_skip = max(1, int(fps)) if fps > 0 else 1
            total_sampled = max(1, frame_count // frames_to_skip)

            frames_with_faces = 0
            processed_frames = 0
            all_detections: List[Dict] = []

            for frame_idx in range(0, frame_count, frames_to_skip):
                if not self.is_processing:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    break
                processed_frames += 1
                try:
                    raw_dets: List[Detection] = self.detector.detect_image(
                        frame, confidence_threshold
                    )
                except Exception as e:
                    self._emit(
                        f"{_STATUS_SYMBOLS['error']} Error processing frame {frame_idx}: {e}"
                    )
                    raw_dets = []

                detections = [d.to_legacy_face_dict(video_path) for d in raw_dets]
                timestamp = frame_idx / fps if fps > 0 else 0.0
                if detections:
                    frames_with_faces += 1
                    for det in detections:
                        det["frame_idx"] = frame_idx
                        det["timestamp"] = timestamp
                    all_detections.extend(detections)

                    if result_folder:
                        try:
                            annotated = self._draw_detections(frame, detections)
                            out_dir = os.path.join(result_folder, "results")
                            os.makedirs(out_dir, exist_ok=True)
                            video_stem = os.path.splitext(os.path.basename(video_path))[0]
                            out_name = f"{video_stem}_frame_{frame_idx:06d}.jpg"
                            safe_imwrite(os.path.join(out_dir, out_name), annotated)
                        except Exception as e:
                            self._emit(
                                f"{_STATUS_SYMBOLS['error']} Error saving annotated frame {frame_idx} of {os.path.basename(video_path)}: {e}"
                            )
                else:
                    # Record the sample so the CSV can show "frame processed, no faces".
                    all_detections.append({
                        "image_path": video_path,
                        "frame_idx": frame_idx,
                        "timestamp": timestamp,
                        "x": None,
                        "y": None,
                        "width": None,
                        "height": None,
                        "confidence": None,
                        "_no_face": True,
                    })

                if self.completion_callback:
                    self.completion_callback({
                        "status": "frame_completed",
                        "frame_index": frame_idx,
                        "processed_frames": processed_frames,
                        "total_frames": total_sampled,
                        "progress_percent": (processed_frames / total_sampled) * 100,
                        "detections_in_frame": len(detections),
                        "total_detections": len(all_detections),
                        "timestamp": frame_idx / fps if fps > 0 else 0.0,
                        "video_path": os.path.basename(video_path),
                    })

            cap.release()
            face_pct = (frames_with_faces / processed_frames) * 100 if processed_frames else 0.0
            self._emit(
                f"{_STATUS_SYMBOLS['complete']} Video processing complete. "
                f"{frames_with_faces}/{processed_frames} frames with faces ({face_pct:.1f}%)"
            )
            if result_folder and frames_with_faces:
                self._emit(
                    f"{_STATUS_SYMBOLS['folder']} Saved {frames_with_faces} annotated "
                    f"frame(s) to {os.path.join(result_folder, 'results')}"
                )
            return all_detections
        except Exception as e:
            self._emit(f"{_STATUS_SYMBOLS['error']} Error processing video: {e}")
            return []

    # ---- Lifecycle / accessors --------------------------------------------

    def stop_processing(self) -> None:
        self.is_processing = False
        self._emit("Processing stopped by user")

    def get_results(self) -> List[Dict]:
        return self.results

    # ---- CSV export --------------------------------------------------------

    def export_results_to_csv(self, results: List[Dict], output_path: str) -> bool:
        """Write detection results as one row per (image | sampled video frame).

        Schema: ``id, frame_idx, filename, face_detected, face_count, face_N_x,
        face_N_y, face_N_width, face_N_height, face_N_confidence`` (N grows up
        to the largest face count seen). ``frame_idx`` is empty for image rows;
        for video frames with no faces it is the frame index but per-face
        columns stay empty.
        """
        try:
            if not results:
                self._emit(f"{_STATUS_SYMBOLS['warning']} No results to export")
                return False

            # Group by (image_path, frame_idx). frame_idx is None for images,
            # which keeps each image as its own row and each video frame as a
            # separate row even though they share the video's path.
            groups: Dict[Tuple[str, Optional[int]], List[Dict]] = {}
            order: List[Tuple[str, Optional[int]]] = []
            for det in results:
                key = (det["image_path"], det.get("frame_idx"))
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                if not det.get("_no_face"):
                    groups[key].append(det)

            max_faces = max((len(v) for v in groups.values()), default=0)
            headers = ["id", "frame_idx", "filename", "face_detected", "face_count"]
            for i in range(max_faces):
                headers.extend([
                    f"face_{i+1}_x",
                    f"face_{i+1}_y",
                    f"face_{i+1}_width",
                    f"face_{i+1}_height",
                    f"face_{i+1}_confidence",
                ])

            rows = []
            for row_id, key in enumerate(order, start=1):
                path, frame_idx = key
                dets = groups[key]
                row = [
                    row_id,
                    "" if frame_idx is None else frame_idx,
                    os.path.basename(path),
                    1 if dets else 0,
                    len(dets),
                ]
                for det in dets:
                    row.extend([det["x"], det["y"], det["width"], det["height"], det["confidence"]])
                while len(row) < len(headers):
                    row.append("")
                rows.append(row)

            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
            self._emit(f"{_STATUS_SYMBOLS['success']} Results exported to {output_path}")
            return True
        except Exception as e:
            self._emit(f"{_STATUS_SYMBOLS['error']} Error exporting results: {e}")
            return False

    def export_summary_to_csv(
        self,
        folder_path: str,
        image_files: List[str],
        video_files: List[str],
        result_folder: str,
        output_path: str,
    ) -> bool:
        """Write a per-file summary so each input file is auditable on its own.

        Schema: ``path, type, total_processed_frames, total_duration,
        processed_frames_with_faces, face_percentage, model,
        confidence_threshold``. One row per image (frames=1, duration=N/A) and
        one row per video (frames = sampled frames, duration = seconds).
        """
        try:
            headers = [
                "path",
                "type",
                "total_processed_frames",
                "total_duration",
                "processed_frames_with_faces",
                "face_percentage",
                "model",
                "confidence_threshold",
            ]

            # Group results by source path so each row can be answered without
            # rescanning self.results.
            by_path: Dict[str, List[Dict]] = {}
            for det in self.results:
                by_path.setdefault(det["image_path"], []).append(det)

            model_label = self.current_model_path or "Unknown"
            conf_label = getattr(self, "current_confidence", "Unknown")
            rows = []

            for img_path in image_files:
                dets = by_path.get(img_path, [])
                has_faces = any(not d.get("_no_face") for d in dets)
                rows.append([
                    img_path,
                    "image",
                    1,
                    "N/A",
                    1 if has_faces else 0,
                    100.0 if has_faces else 0.0,
                    model_label,
                    conf_label,
                ])

            for video_file in video_files:
                dets = by_path.get(video_file, [])
                sampled_frames = len({
                    d["frame_idx"] for d in dets if "frame_idx" in d
                })
                frames_with_faces = len({
                    d["frame_idx"] for d in dets
                    if "frame_idx" in d and not d.get("_no_face")
                })
                duration = 0.0
                try:
                    cap = cv2.VideoCapture(video_file)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                    if fps > 0:
                        duration = frame_count / fps
                    cap.release()
                except Exception as e:
                    self._emit(
                        f"{_STATUS_SYMBOLS['warning']} Could not get video info for {video_file}: {e}"
                    )
                pct = (frames_with_faces / sampled_frames * 100) if sampled_frames else 0.0
                rows.append([
                    video_file,
                    "video",
                    sampled_frames,
                    duration,
                    frames_with_faces,
                    pct,
                    model_label,
                    conf_label,
                ])

            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
            self._emit(f"{_STATUS_SYMBOLS['success']} Summary exported to {output_path}")
            return True
        except Exception as e:
            self._emit(f"{_STATUS_SYMBOLS['error']} Error exporting summary: {e}")
            return False

    # ---- Available-model listing ------------------------------------------

    def get_available_models(self) -> List[str]:
        """Return the same list of selectable model names the UI used pre-refactor.

        The list is filtered by which detector environments are reachable
        (bundled venv, conda env, or active interpreter)."""
        env_flags = _detect_environments()

        models: List[str] = []
        if env_flags["any_yolo"] or env_flags["any_retinaface"]:
            print(
                f"Detected model environments - YOLO: {env_flags['any_yolo']}, "
                f"RetinaFace: {env_flags['any_retinaface']}",
                file=sys.stderr,
            )
            for key, info in list_detectors().items():
                if info["kind"] != "vision":
                    continue
                if "yolo" in key and env_flags["any_yolo"]:
                    models.extend(info["variants"])  # type: ignore[arg-type]
                elif "retinaface" in key and env_flags["any_retinaface"]:
                    models.extend(info["variants"])  # type: ignore[arg-type]
        else:
            for key, info in list_detectors().items():
                if info["kind"] != "vision":
                    continue
                if "yolo" in key and _yolo_available():
                    models.extend(info["variants"])  # type: ignore[arg-type]
                elif "retinaface" in key and _retinaface_available():
                    models.extend(info["variants"])  # type: ignore[arg-type]
        return models

    # ---- Helpers -----------------------------------------------------------

    def _emit(self, message: str) -> None:
        if self.progress_callback:
            try:
                self.progress_callback(message)
            except Exception as e:
                print(f"progress_callback raised: {e}", file=sys.stderr)


def _detect_environments() -> Dict[str, bool]:
    """Discover whether the bundled or conda envs for each backend exist.

    Mirrors the original ``get_available_models`` heuristics so the UI keeps
    working for users running against split-env installations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    yolo_env_path = os.path.join(parent_dir, "yolo-env")
    retinaface_env_path = os.path.join(parent_dir, "retinaface-env")

    has_yolo_env = os.path.exists(yolo_env_path)
    has_retinaface_env = os.path.exists(retinaface_env_path)

    conda_env_dirs: List[str] = []
    try:
        if os.name == "nt":
            candidates = [
                os.path.join(os.environ.get("USERPROFILE", ""), "miniconda3", "envs"),
                os.path.join(os.environ.get("USERPROFILE", ""), "anaconda3", "envs"),
                r"C:\\Miniconda3\\envs",
                r"C:\\ProgramData\\Miniconda3\\envs",
                r"C:\\Anaconda3\\envs",
            ]
        else:
            candidates = [
                os.path.join(os.environ.get("HOME", ""), "miniconda3", "envs"),
                os.path.join(os.environ.get("HOME", ""), "anaconda3", "envs"),
                "/opt/homebrew/miniconda3/envs",
                "/opt/homebrew/anaconda3/envs",
            ]
        conda_env_dirs = [d for d in candidates if d and os.path.exists(d)]
    except Exception:
        pass

    has_yolo_conda = any(
        os.path.exists(os.path.join(d, "electron-python-yolo")) for d in conda_env_dirs
    )
    has_retinaface_conda = any(
        os.path.exists(os.path.join(d, "electron-python-retinaface")) for d in conda_env_dirs
    )
    return {
        "any_yolo": has_yolo_env or has_yolo_conda,
        "any_retinaface": has_retinaface_env or has_retinaface_conda,
        "yolo_bundled": has_yolo_env,
        "retinaface_bundled": has_retinaface_env,
        "yolo_conda": has_yolo_conda,
        "retinaface_conda": has_retinaface_conda,
    }
