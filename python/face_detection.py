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
from detectors.hand_handobj import is_available as _handobj_available


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
    "hand": "✋",
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

        noun = self._modality_noun()
        icon = _STATUS_SYMBOLS.get(noun, _STATUS_SYMBOLS["face"])
        if legacy:
            self._emit(
                f"{icon} Found {len(legacy)} {noun}(s) in {os.path.basename(image_path)}"
            )
        else:
            self._emit(
                f"{_STATUS_SYMBOLS['complete']} No {noun}s detected in {os.path.basename(image_path)}"
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
            is_hand = "state_label" in det or det.get("owner") is not None
            if is_hand:
                color, label = self._hand_box_style(det)
            else:
                color, label = (0, 255, 0), f"{det['confidence']:.2f}"
            cv2.rectangle(
                result_img,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                color,
                2,
            )
            cv2.putText(
                result_img,
                label,
                (int(x), int(y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        return result_img

    @staticmethod
    def _hand_box_style(det: Dict) -> Tuple[Tuple[int, int, int], str]:
        """Return (BGR color, label) for a hand detection.

        Colour encodes ownership (cyan=own, pink=other, white=unknown); the
        label carries side, contact state and confidence.
        """
        owner = det.get("owner", "unknown")
        color = {
            "own": (255, 255, 0),      # cyan (BGR)
            "other": (203, 192, 255),  # pink (BGR)
        }.get(owner, (255, 255, 255))  # white for baseline / unknown
        side = str(det.get("side", "")).capitalize()
        state = det.get("state_label", "")
        conf = det.get("confidence")
        conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else ""
        owner_str = "" if owner == "unknown" else f" {owner}"
        label = f"{side} {state}{owner_str} {conf_str}".strip()
        return color, label

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
                if self._active_modality() == "hand":
                    csv_path = os.path.join(result_folder, "hand_detections.csv")
                    dataset = os.path.basename(os.path.normpath(folder_path))
                    self.export_hand_results_to_csv(self.results, csv_path, dataset=dataset)
                    summary_csv_path = os.path.join(result_folder, "hand_summary.csv")
                    self.export_hand_summary_to_csv(
                        image_files, video_files, summary_csv_path
                    )
                else:
                    csv_path = os.path.join(result_folder, "detection_results.csv")
                    self.export_results_to_csv(self.results, csv_path)
                    summary_csv_path = os.path.join(result_folder, "summary.csv")
                    self.export_summary_to_csv(
                        folder_path, image_files, video_files, result_folder, summary_csv_path
                    )

            detection_count = sum(
                1 for d in self.results if not d.get("_no_face") and d.get("x") is not None
            )
            self._emit(
                f"{_STATUS_SYMBOLS['complete']} Processing complete. Found {detection_count} "
                f"{self._modality_noun()} detections across {total_files} files"
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

    # ---- Hand-specific CSV export -----------------------------------------

    def export_hand_results_to_csv(
        self, results: List[Dict], output_path: str, dataset: str = ""
    ) -> bool:
        """Write one row per detected hand.

        Schema mirrors the 100DOH prototype export — ``dataset, filename,
        hand_id, hand_x1, hand_y1, hand_x2, hand_y2, Hand_confidence, state,
        Hand_side, Owner_label, paired_obj_id, obj_x1, obj_y1, obj_x2, obj_y2,
        obj_score`` — plus app convenience columns: ``frame_idx`` (empty for
        image inputs), ``state_raw`` and ``state_label``.

        ``state`` is the contact state AFTER hand-object pairing correction (this
        is what matches the reference); ``state_raw`` is the model's uncorrected
        argmax. The ``obj_*`` cells are blank and ``paired_obj_id`` is ``-1`` for
        hands not paired with any object.
        """
        try:
            headers = [
                "dataset", "filename", "hand_id",
                "hand_x1", "hand_y1", "hand_x2", "hand_y2", "Hand_confidence",
                "state", "Hand_side", "Owner_label",
                "paired_obj_id", "obj_x1", "obj_y1", "obj_x2", "obj_y2", "obj_score",
                "frame_idx", "state_raw", "state_label",
            ]
            def _blank(v):
                # unpaired hands carry None for the obj_* fields -> empty cell
                return "" if v is None else v

            rows = []
            for det in results:
                if det.get("_no_face") or det.get("x") is None:
                    continue
                x, y = det["x"], det["y"]
                w, h = det["width"], det["height"]
                frame_idx = det.get("frame_idx")
                rows.append([
                    dataset,
                    os.path.basename(det["image_path"]),
                    det.get("hand_id", ""),
                    x, y, x + w, y + h,
                    det["confidence"],
                    det.get("state", ""),
                    det.get("side", ""),
                    det.get("owner", ""),
                    det.get("paired_obj_id", -1),
                    _blank(det.get("obj_x1")), _blank(det.get("obj_y1")),
                    _blank(det.get("obj_x2")), _blank(det.get("obj_y2")),
                    _blank(det.get("obj_score")),
                    "" if frame_idx is None else frame_idx,
                    det.get("state_raw", ""),
                    det.get("state_label", ""),
                ])

            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
            self._emit(f"{_STATUS_SYMBOLS['success']} Hand detections exported to {output_path}")
            return True
        except Exception as e:
            self._emit(f"{_STATUS_SYMBOLS['error']} Error exporting hand results: {e}")
            return False

    def export_hand_summary_to_csv(
        self,
        image_files: List[str],
        video_files: List[str],
        output_path: str,
    ) -> bool:
        """Write one row per image (and per sampled video frame) with hand counts.

        Schema: ``filename, frame_idx, img_w, img_h, n_hands, n_own, n_other,
        n_state0_none, n_state1_self, n_state2_other, n_state3_portable,
        n_state4_furniture``.
        """
        try:
            headers = [
                "filename", "frame_idx", "img_w", "img_h", "n_hands",
                "n_own", "n_other",
                "n_state0_none", "n_state1_self", "n_state2_other",
                "n_state3_portable", "n_state4_furniture",
            ]

            groups: Dict[Tuple[str, Optional[int]], List[Dict]] = {}
            for det in self.results:
                if det.get("_no_face") or det.get("x") is None:
                    continue
                key = (det["image_path"], det.get("frame_idx"))
                groups.setdefault(key, []).append(det)

            def _summ(filename: str, frame_idx, img_w, img_h, dets: List[Dict]):
                state_counts = {i: 0 for i in range(5)}
                n_own = n_other = 0
                for d in dets:
                    st = d.get("state")
                    if isinstance(st, int) and st in state_counts:
                        state_counts[st] += 1
                    owner = d.get("owner")
                    if owner == "own":
                        n_own += 1
                    elif owner == "other":
                        n_other += 1
                return [
                    filename,
                    "" if frame_idx is None else frame_idx,
                    img_w, img_h, len(dets), n_own, n_other,
                    state_counts[0], state_counts[1], state_counts[2],
                    state_counts[3], state_counts[4],
                ]

            rows = []
            for img_path in image_files:
                dets = groups.pop((img_path, None), [])
                w, h = self._image_dims(img_path)
                rows.append(_summ(os.path.basename(img_path), None, w, h, dets))

            # Any remaining groups are video frames (path, frame_idx).
            for (path, frame_idx), dets in groups.items():
                rows.append(_summ(os.path.basename(path), frame_idx, "", "", dets))

            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
            self._emit(f"{_STATUS_SYMBOLS['success']} Hand summary exported to {output_path}")
            return True
        except Exception as e:
            self._emit(f"{_STATUS_SYMBOLS['error']} Error exporting hand summary: {e}")
            return False

    def _image_dims(self, path: str) -> Tuple[object, object]:
        """Return (width, height) of an image without fully decoding when possible."""
        try:
            from PIL import Image

            with Image.open(path) as im:
                return im.width, im.height
        except Exception:
            try:
                img = cv2.imread(path)
                if img is not None:
                    return img.shape[1], img.shape[0]
            except Exception:
                pass
        return "", ""

    def _active_modality(self) -> str:
        """Modality of the loaded detector ('face' | 'hand' | ...); defaults to 'face'."""
        name = getattr(self.detector, "name", None)
        return name or "face"

    def _modality_noun(self) -> str:
        return self._active_modality()

    # ---- Available-model listing ------------------------------------------

    def _available_variants_with_mode(self) -> List[Tuple[str, str]]:
        """Return ``[(variant, mode), ...]`` for every selectable model, filtered
        by which detector environments are reachable (bundled venv, conda env, or
        active interpreter). Single source of truth for the two public accessors
        below so the flat list and the variant->mode map never disagree."""
        env_flags = _detect_environments()
        any_bundled = (
            env_flags["any_yolo"]
            or env_flags["any_retinaface"]
            or env_flags["any_hand"]
        )
        if any_bundled:
            print(
                f"Detected model environments - YOLO: {env_flags['any_yolo']}, "
                f"RetinaFace: {env_flags['any_retinaface']}, "
                f"Hand: {env_flags['any_hand']}",
                file=sys.stderr,
            )

        def _reachable(key: str) -> bool:
            if "yolo" in key:
                return env_flags["any_yolo"] if any_bundled else _yolo_available()
            if "retinaface" in key:
                return env_flags["any_retinaface"] if any_bundled else _retinaface_available()
            if "hand" in key:
                return env_flags["any_hand"] if any_bundled else _handobj_available()
            return False

        out: List[Tuple[str, str]] = []
        for key, info in list_detectors().items():
            if info["kind"] != "vision" or not _reachable(key):
                continue
            mode = str(info.get("mode") or info["name"])
            for variant in info["variants"]:  # type: ignore[union-attr]
                out.append((variant, mode))
        return out

    def get_available_models(self) -> List[str]:
        """Return the flat list of selectable model names the UI used pre-refactor."""
        return [variant for variant, _ in self._available_variants_with_mode()]

    def get_available_model_modes(self) -> Dict[str, str]:
        """Return ``{variant: mode}`` for every selectable model, so the UI can
        filter the model dropdown by the selected Mode without a second call."""
        return {variant: mode for variant, mode in self._available_variants_with_mode()}

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
    hand_env_path = os.path.join(parent_dir, "hand-env")

    has_yolo_env = os.path.exists(yolo_env_path)
    has_retinaface_env = os.path.exists(retinaface_env_path)
    has_hand_env = os.path.exists(hand_env_path)

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
    has_hand_conda = any(
        os.path.exists(os.path.join(d, "electron-python-hand")) for d in conda_env_dirs
    )
    return {
        "any_yolo": has_yolo_env or has_yolo_conda,
        "any_retinaface": has_retinaface_env or has_retinaface_conda,
        "any_hand": has_hand_env or has_hand_conda,
        "yolo_bundled": has_yolo_env,
        "retinaface_bundled": has_retinaface_env,
        "hand_bundled": has_hand_env,
        "yolo_conda": has_yolo_conda,
        "retinaface_conda": has_retinaface_conda,
        "hand_conda": has_hand_conda,
    }
