"""Local audio transcription pipeline.

The implementation deliberately imports Whisper libraries only when a job is
started.  This keeps the face/hand application startup fast and allows the
vision environments to remain usable when the optional audio environment is
not installed.
"""
from __future__ import annotations

import csv
import os
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple


AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".mp4", ".mov", ".mkv")
TRANSCRIPTION_VARIANTS = ["Whisper (OpenAI)", "Faster Whisper", "WhisperX"]


class TranscriptionProcessor:
    def __init__(self, progress_callback: Optional[Callable[[str], None]] = None,
                 completion_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.progress_callback = progress_callback
        self.completion_callback = completion_callback
        self.is_processing = False
        self.results: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._model: Any = None
        self._model_variant: Optional[str] = None

    def _emit(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)

    def stop_processing(self) -> None:
        self._stop.set()

    @staticmethod
    def _files(source: str) -> List[str]:
        if os.path.isfile(source):
            return [source] if source.lower().endswith(AUDIO_EXTENSIONS) else []
        found: List[str] = []
        for root, _, names in os.walk(source):
            found.extend(os.path.join(root, n) for n in names
                         if n.lower().endswith(AUDIO_EXTENSIONS))
        return sorted(found)

    def _load_model(self, variant: str) -> None:
        self._emit("🎤 Loading transcription model (first use may download model weights)...")
        if variant == "Whisper (OpenAI)":
            import whisper
            self._model = whisper.load_model(os.environ.get("TINYEXPLORER_WHISPER_MODEL", "base"))
        elif variant == "Faster Whisper":
            from faster_whisper import WhisperModel
            model_name = os.environ.get("TINYEXPLORER_WHISPER_MODEL", "base")
            device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1") else "cpu"
            compute = "float16" if device == "cuda" else "int8"
            self._model = WhisperModel(model_name, device=device, compute_type=compute)
        elif variant == "WhisperX":
            import whisperx
            device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1") else "cpu"
            self._model = whisperx.load_model(os.environ.get("TINYEXPLORER_WHISPER_MODEL", "base"), device=device,
                                              compute_type="float16" if device == "cuda" else "int8")
        else:
            raise ValueError("Unknown transcription model: %s" % variant)
        self._model_variant = variant

    def _transcribe(self, path: str, variant: str) -> Tuple[List[Dict[str, Any]], str]:
        if self._model_variant != variant or self._model is None:
            self._load_model(variant)
        if variant == "Faster Whisper":
            segments, info = self._model.transcribe(path, word_timestamps=True)
            return [self._segment_dict(s.start, s.end, s.text, getattr(s, "words", None)) for s in segments], getattr(info, "language", "unknown")
        result = self._model.transcribe(path, word_timestamps=True, fp16=False)
        return [self._segment_dict(s.get("start"), s.get("end"), s.get("text", ""), s.get("words"))
                for s in result.get("segments", [])], result.get("language", "unknown")

    @staticmethod
    def _segment_dict(start: Any, end: Any, text: Any, words: Any) -> Dict[str, Any]:
        out_words = []
        for word in words or []:
            if isinstance(word, dict):
                out_words.append({"word": word.get("word"), "start": word.get("start"), "end": word.get("end"),
                                  "probability": word.get("probability", word.get("prob"))})
            else:
                out_words.append({"word": getattr(word, "word", ""), "start": getattr(word, "start", None),
                                  "end": getattr(word, "end", None), "probability": getattr(word, "probability", None)})
        return {"start": float(start or 0), "end": float(end or 0), "text": str(text or "").strip(), "words": out_words}

    def process(self, source: str, variant: str, results_folder: str) -> None:
        self.is_processing = True
        self._stop.clear()
        self.results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = os.path.join(results_folder, "transcription_results_%s" % timestamp)
        os.makedirs(output, exist_ok=True)
        files = self._files(source)
        shared_headers = [
            "id", "frame_idx", "filename", "mode", "start", "end",
            "label", "confidence", "model", "text", "language",
        ]
        shared_rows: List[List[Any]] = []
        summary_rows: List[List[Any]] = []
        try:
            if not files:
                raise ValueError("No supported audio or video files found")
            self._emit("🎤 Found %d audio/video file(s)" % len(files))
            for index, path in enumerate(files):
                if self._stop.is_set():
                    break
                self._emit("🎤 Transcribing %d/%d: %s" % (index + 1, len(files), os.path.basename(path)))
                segments, language = self._transcribe(path, variant)
                stem = os.path.splitext(os.path.basename(path))[0]
                csv_path = os.path.join(output, stem + "_transcript.csv")
                txt_path = os.path.join(output, stem + "_transcript.txt")
                with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    # Keep the first columns compatible with the detection
                    # exporters (id/frame_idx/filename), then add the
                    # speech-specific time and text fields.  Empty values are
                    # intentional for frame_idx/confidence: speech is
                    # timestamped rather than frame- or box-based.
                    writer.writerow(shared_headers)
                    file_rows = []
                    for segment_id, segment in enumerate(segments, start=1):
                        if segment["text"]:
                            row = [
                                segment_id, "", os.path.basename(path), "speech",
                                segment["start"], segment["end"], "speech", "", variant,
                                segment["text"], language,
                            ]
                            writer.writerow(row)
                            file_rows.append(row)
                            shared_rows.append(row)
                            self.results.append(dict(segment, audio_path=path, language=language, model=variant))
                with open(txt_path, "w", encoding="utf-8") as handle:
                    for segment in segments:
                        if segment["text"]:
                            handle.write("[%0.2f-%0.2f] %s\n" % (segment["start"], segment["end"], segment["text"]))
                summary_rows.append([
                    path,
                    "audio",
                    len(file_rows),
                    max((row[5] for row in file_rows), default=0.0),
                    language,
                    variant,
                ])
                self.completion_callback({"status": "audio_completed", "progress_percent": (index + 1) / len(files) * 100,
                                          "audio_index": index + 1, "total_audio": len(files), "audio_path": path}) if self.completion_callback else None
            with open(os.path.join(output, "detections.csv"), "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(shared_headers)
                writer.writerows(shared_rows)
            with open(os.path.join(output, "summary.csv"), "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["path", "type", "segments", "duration", "language", "model"])
                writer.writerows(summary_rows)
            self._emit("✅ Transcription complete. Results saved to: %s" % output)
            if self.completion_callback:
                self.completion_callback({"status": "completed", "results_folder": output, "results_count": len(self.results)})
        except ImportError as exc:
            message = "Audio transcription dependencies are not installed. Install python/requirements-whisper.txt (%s)" % exc
            self._emit("❌ " + message)
            if self.completion_callback: self.completion_callback({"status": "error", "error": message})
        except Exception as exc:
            self._emit("❌ Transcription failed: %s" % exc)
            if self.completion_callback: self.completion_callback({"status": "error", "error": str(exc)})
        finally:
            self.is_processing = False
            if self.completion_callback: self.completion_callback({"status": "finished", "results_count": len(self.results)})
