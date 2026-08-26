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
        self._model_size: Optional[str] = None
        # WhisperX extras, cached across files: (language, model, metadata) and
        # the pyannote diarization pipeline.
        self._align_cache: Optional[Tuple[str, Any, Any]] = None
        self._diarizer: Any = None

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

    def _emit_download_progress(self, name: str, downloaded: int, total: int,
                                state: Dict[str, int]) -> None:
        """Emit '⏳ Downloading <name>: NN%' lines (one per percent). The
        renderer coalesces consecutive lines for the same name into a single
        live-updating row, so this reads as a progress bar in the UI. When the
        size is unknown, fall back to a line every 25 MB so long downloads
        still visibly move."""
        if total:
            pct = int(downloaded * 100 / total)
            if pct != state.get("last"):
                state["last"] = pct
                self._emit("⏳ Downloading %s: %d%% (%d/%d MB)"
                           % (name, pct, downloaded // 1048576, total // 1048576))
        elif downloaded > 0:
            bucket = downloaded // (25 * 1048576)
            if bucket != state.get("last_bucket"):
                state["last_bucket"] = bucket
                self._emit("⏳ Downloading %s: %d MB so far (total size unknown)"
                           % (name, downloaded // 1048576))

    def _ensure_openai_whisper_weights(self, model_name: str) -> None:
        """Pre-download the OpenAI Whisper checkpoint with progress reporting.

        whisper.load_model() downloads silently (tqdm on stderr); fetching the
        file into whisper's cache first gives the UI real percentages. If the
        cached file is corrupt, load_model's own sha256 check re-downloads it.
        """
        import urllib.request
        import whisper
        url = getattr(whisper, "_MODELS", {}).get(model_name)
        if not url:
            return
        root = (os.environ.get("TINYEXPLORER_WHISPER_CACHE")
                or os.path.join(os.path.expanduser("~"), ".cache", "whisper"))
        target = os.path.join(root, os.path.basename(url))
        if os.path.exists(target):
            return
        os.makedirs(root, exist_ok=True)
        state: Dict[str, int] = {}
        tmp = target + ".part"
        with urllib.request.urlopen(url) as source, open(tmp, "wb") as out:
            total = int(source.headers.get("Content-Length") or 0)
            downloaded = 0
            while True:
                chunk = source.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                self._emit_download_progress("Whisper %s" % model_name, downloaded, total, state)
        os.replace(tmp, target)

    def _hf_download_progress(self, fallback_label: str):
        """Context manager: while active, huggingface_hub downloads (faster-
        whisper / whisperx / pyannote weights) report progress through _emit
        instead of a terminal tqdm bar. Hub >= 0.36 builds its bars via
        file_download._get_progress_bar_context for BOTH the plain-http and
        xet transports, so that is the primary patch point; the module-level
        tqdm swap covers older hubs. Best-effort — if the hub internals
        change again, downloads simply stay silent as before."""
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            try:
                import huggingface_hub.file_download as hf_fd
            except Exception:
                yield
                return
            processor = self

            class _EmitBar:
                """Minimal tqdm stand-in: counts bytes, emits app progress."""

                def __init__(self, desc: Any, total: Any, initial: Any = 0):
                    self.desc = str(desc) if desc else fallback_label
                    self.total = int(total or 0)
                    self.n = int(initial or 0)
                    self._state: Dict[str, int] = {}

                def update(self, n: int = 1) -> None:
                    self.n += int(n or 0)
                    processor._emit_download_progress(self.desc, self.n, self.total, self._state)

                def __enter__(self) -> "_EmitBar":
                    return self

                def __exit__(self, *args: Any) -> bool:
                    return False

                def close(self) -> None: pass
                def refresh(self) -> None: pass
                def set_description(self, *args: Any, **kwargs: Any) -> None: pass

            originals: Dict[str, Any] = {}

            if hasattr(hf_fd, "_get_progress_bar_context"):
                originals["_get_progress_bar_context"] = hf_fd._get_progress_bar_context

                def _emitting_context(*args: Any, **kwargs: Any) -> Any:
                    existing = kwargs.get("_tqdm_bar")
                    if isinstance(existing, _EmitBar):
                        # The hub creates the bar before the response arrives
                        # (total unknown) and passes it back here with the real
                        # size once headers are in — adopt it.
                        if kwargs.get("total"):
                            existing.total = int(kwargs["total"])
                        return existing
                    if existing is not None:
                        return originals["_get_progress_bar_context"](*args, **kwargs)
                    return _EmitBar(kwargs.get("desc", ""), kwargs.get("total"), kwargs.get("initial", 0))

                hf_fd._get_progress_bar_context = _emitting_context

            if hasattr(hf_fd, "tqdm"):
                base = hf_fd.tqdm
                originals["tqdm"] = base

                class _EmitTqdm(base):  # type: ignore[misc, valid-type]
                    def update(self, n: int = 1):
                        result = super().update(n)
                        try:
                            if self.total:
                                name = getattr(self, "desc", "") or fallback_label
                                if not hasattr(self, "_emit_state"):
                                    self._emit_state = {}
                                processor._emit_download_progress(name, self.n, self.total, self._emit_state)
                        except Exception:
                            pass
                        return result

                    def display(self, *args: Any, **kwargs: Any) -> None:
                        pass  # keep the packaged app's stderr clean

                hf_fd.tqdm = _EmitTqdm

            try:
                yield
            finally:
                for attr, value in originals.items():
                    setattr(hf_fd, attr, value)

        return _cm()

    @staticmethod
    def _device() -> str:
        return "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1") else "cpu"

    @staticmethod
    def _hf_token() -> str:
        return os.environ.get("TINYEXPLORER_HF_TOKEN") or os.environ.get("HF_TOKEN") or ""

    def _load_model(self, variant: str, size: Optional[str] = None) -> None:
        # Explicit UI choice wins; the env var covers headless use; "base"
        # keeps old callers working.
        model_name = size or os.environ.get("TINYEXPLORER_WHISPER_MODEL", "base")
        self._emit("🎤 Loading transcription model '%s' (first use may download model weights)..." % model_name)
        if variant == "Whisper (OpenAI)":
            import whisper
            self._ensure_openai_whisper_weights(model_name)
            self._model = whisper.load_model(model_name)
        elif variant == "Faster Whisper":
            from faster_whisper import WhisperModel
            device = self._device()
            compute = "float16" if device == "cuda" else "int8"
            with self._hf_download_progress("Faster Whisper %s" % model_name):
                self._model = WhisperModel(model_name, device=device, compute_type=compute)
        elif variant == "WhisperX":
            import whisperx
            device = self._device()
            with self._hf_download_progress("WhisperX %s" % model_name):
                self._model = whisperx.load_model(model_name, device=device,
                                                  compute_type="float16" if device == "cuda" else "int8")
        else:
            raise ValueError("Unknown transcription model: %s" % variant)
        if not self._hf_token():
            self._emit("ℹ️ Set a Hugging Face token (🔑 in the app, or TINYEXPLORER_HF_TOKEN) to "
                       "enable speaker diarization; exporting without speaker labels.")
        self._model_variant = variant
        self._model_size = size

    @staticmethod
    def _load_audio(path: str) -> Any:
        """Decode audio to the 16 kHz mono float32 array Whisper models expect.

        openai-whisper's and whisperx's own load_audio() spawn an external
        ffmpeg binary, which the packaged app does not ship. PyAV (already a
        faster-whisper dependency) links the ffmpeg libraries directly, so
        decoding works in-process everywhere.
        """
        import av
        import numpy as np

        chunks: List[Any] = []
        with av.open(path) as container:
            if not container.streams.audio:
                raise ValueError("No audio stream found in %s" % path)
            stream = container.streams.audio[0]
            resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
            for frame in container.decode(stream):
                chunks.extend(r.to_ndarray() for r in resampler.resample(frame))
            chunks.extend(r.to_ndarray() for r in resampler.resample(None))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate([c.reshape(-1) for c in chunks])
        return audio.astype(np.float32) / 32768.0

    def _transcribe(self, path: str, variant: str, size: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str]:
        if self._model_variant != variant or self._model_size != size or self._model is None:
            self._load_model(variant, size)
        self._emit("🎤 Running speech recognition (may take a while for long recordings)...")
        if variant == "Faster Whisper":
            raw_segments, info = self._model.transcribe(path, word_timestamps=True)
            segments = [self._segment_dict(s.start, s.end, s.text, getattr(s, "words", None),
                                           confidence=getattr(s, "avg_logprob", None))
                        for s in raw_segments]
            if self._hf_token():  # skip the audio decode when diarization is off
                segments = self._assign_speakers(segments, self._load_audio(path))
            return segments, getattr(info, "language", "unknown")
        if variant == "WhisperX":
            # whisperx.load_model() returns a FasterWhisperPipeline whose
            # transcribe() takes neither word_timestamps nor fp16. Word-level
            # times come from the separate align() stage, speaker labels from
            # the diarization stage — both are best-effort extras.
            audio = self._load_audio(path)
            result = self._model.transcribe(
                audio, batch_size=int(os.environ.get("TINYEXPLORER_WHISPERX_BATCH", "8")))
            language = result.get("language", "unknown")
            segments = self._whisperx_enrich(result.get("segments", []), audio, language)
            return [self._segment_dict(s.get("start"), s.get("end"), s.get("text", ""), s.get("words"),
                                       s.get("speaker"))
                    for s in segments], language
        audio = self._load_audio(path)
        result = self._model.transcribe(audio, word_timestamps=True, fp16=False)
        segments = [self._segment_dict(s.get("start"), s.get("end"), s.get("text", ""), s.get("words"),
                                       s.get("speaker"), s.get("avg_logprob"))
                    for s in result.get("segments", [])]
        return self._assign_speakers(segments, audio), result.get("language", "unknown")

    def _whisperx_enrich(self, segments: List[Dict[str, Any]], audio: Any, language: str) -> List[Dict[str, Any]]:
        """Best-effort word alignment + speaker diarization for WhisperX.

        Either stage failing (missing alignment model for the language, no
        Hugging Face token for the gated pyannote weights, offline, ...) must
        degrade to the plain segment output rather than fail the run.
        """
        import whisperx
        device = self._device()
        try:
            if self._align_cache is None or self._align_cache[0] != language:
                with self._hf_download_progress("alignment model (%s)" % language):
                    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
                self._align_cache = (language, model_a, metadata)
            _, model_a, metadata = self._align_cache
            self._emit("📐 Aligning word-level timestamps...")
            segments = whisperx.align(segments, model_a, metadata, audio, device).get("segments", segments)
        except Exception as exc:
            self._emit("⚠️ Word alignment unavailable (%s); exporting utterance-level output only." % exc)
        return self._assign_speakers(segments, audio)

    def _assign_speakers(self, segments: List[Dict[str, Any]], audio: Any) -> List[Dict[str, Any]]:
        """Best-effort speaker diarization for any backend's segments.

        Diarization is not a Whisper capability: it runs the separate gated
        pyannote pipeline (via whisperx.diarize) and tags segments/words by
        timestamp overlap, so it works the same for every backend. Without a
        Hugging Face token, or on any failure, the segments come back
        unchanged and the speaker columns stay empty.
        """
        if not self._hf_token() or not segments:
            return segments
        device = self._device()
        try:
            import inspect
            from whisperx.diarize import DiarizationPipeline, assign_word_speakers
            if self._diarizer is None:
                # whisperx >= 3.8 renamed the auth kwarg from use_auth_token
                # to token; support both.
                params = inspect.signature(DiarizationPipeline.__init__).parameters
                token_kwarg = "token" if "token" in params else "use_auth_token"
                # HF accounts are gated per model. Try, in order: an explicit
                # override, the whisperx default (community-1 on 3.8+), and the
                # older 3.1 pipeline many existing accounts are approved for.
                candidates: List[Optional[str]] = [None, "pyannote/speaker-diarization-3.1"]
                override = os.environ.get("TINYEXPLORER_DIARIZATION_MODEL")
                if override:
                    candidates.insert(0, override)
                self._emit("🗣️ Loading speaker diarization model (first use downloads weights)...")
                last_error: Optional[Exception] = None
                for model_name in candidates:
                    kwargs: Dict[str, Any] = {token_kwarg: self._hf_token(), "device": device}
                    if model_name:
                        kwargs["model_name"] = model_name
                    try:
                        with self._hf_download_progress("speaker diarization model"):
                            self._diarizer = DiarizationPipeline(**kwargs)
                        break
                    except Exception as exc:
                        last_error = exc
                        self._emit("⚠️ Diarization model %s not accessible; trying the next option..."
                                   % (model_name or "(whisperx default)"))
                if self._diarizer is None and last_error is not None:
                    raise last_error
            self._emit("🗣️ Identifying speakers (can take several minutes on CPU)...")
            diarization = self._diarizer(audio)
            segments = assign_word_speakers(diarization, {"segments": segments}).get("segments", segments)
        except Exception as exc:
            self._emit("⚠️ Speaker diarization unavailable (%s); speaker column left empty. "
                       "If this is a gated-repo error, request access to the model on "
                       "huggingface.co with the account that issued your token." % exc)
        return segments

    @staticmethod
    def _segment_dict(start: Any, end: Any, text: Any, words: Any, speaker: Any = None,
                      confidence: Any = None) -> Dict[str, Any]:
        out_words = []
        for word in words or []:
            if isinstance(word, dict):
                # whisperx aligned words carry "score" instead of "probability".
                out_words.append({"word": word.get("word"), "start": word.get("start"), "end": word.get("end"),
                                  "probability": word.get("probability", word.get("prob", word.get("score"))),
                                  "speaker": word.get("speaker")})
            else:
                out_words.append({"word": getattr(word, "word", ""), "start": getattr(word, "start", None),
                                  "end": getattr(word, "end", None), "probability": getattr(word, "probability", None),
                                  "speaker": getattr(word, "speaker", None)})
        if confidence is None:
            # Backends without a native segment confidence (whisperx's batched
            # pipeline): fall back to the mean word probability/score.
            scores = [w["probability"] for w in out_words if w.get("probability") is not None]
            confidence = sum(scores) / len(scores) if scores else None
        return {"start": float(start or 0), "end": float(end or 0), "text": str(text or "").strip(),
                "words": out_words, "speaker": str(speaker) if speaker else "",
                "confidence": None if confidence is None else float(confidence)}

    def process(self, source: str, variant: str, results_folder: str,
                size: Optional[str] = None) -> None:
        self.is_processing = True
        self._stop.clear()
        self.results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = os.path.join(results_folder, "transcription_results_%s" % timestamp)
        os.makedirs(output, exist_ok=True)
        files = self._files(source)
        shared_headers = [
            "id", "frame_idx", "filename", "mode", "start", "end",
            "label", "confidence", "model", "text", "language", "speaker",
        ]
        word_headers = [
            "filename", "word", "start", "end", "speaker", "word_score",
            "segment_start", "segment_end", "segment_text",
        ]
        shared_rows: List[List[Any]] = []
        shared_word_rows: List[List[Any]] = []
        summary_rows: List[List[Any]] = []
        try:
            if not files:
                raise ValueError("No supported audio or video files found")
            self._emit("🎤 Found %d audio/video file(s)" % len(files))
            for index, path in enumerate(files):
                if self._stop.is_set():
                    break
                self._emit("🎤 Transcribing %d/%d: %s" % (index + 1, len(files), os.path.basename(path)))
                segments, language = self._transcribe(path, variant, size)
                stem = os.path.splitext(os.path.basename(path))[0]
                csv_path = os.path.join(output, stem + "_transcript.csv")
                txt_path = os.path.join(output, stem + "_transcript.txt")
                with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    # Keep the first columns compatible with the detection
                    # exporters (id/frame_idx/filename), then add the
                    # speech-specific time and text fields.  frame_idx stays
                    # empty: speech is timestamped rather than frame-based.
                    # confidence is each model's own measure — avg_logprob for
                    # the Whisper backends, mean aligned word score for
                    # WhisperX — so values are not comparable across models.
                    writer.writerow(shared_headers)
                    file_rows = []
                    for segment_id, segment in enumerate(segments, start=1):
                        if segment["text"]:
                            confidence = segment.get("confidence")
                            row = [
                                segment_id, "", os.path.basename(path), "speech",
                                segment["start"], segment["end"], "speech",
                                "" if confidence is None else round(float(confidence), 3),
                                variant,
                                segment["text"], language, segment.get("speaker", ""),
                            ]
                            writer.writerow(row)
                            file_rows.append(row)
                            shared_rows.append(row)
                            self.results.append(dict(segment, audio_path=path, language=language, model=variant))
                word_rows = [
                    [os.path.basename(path), word.get("word"), word.get("start"), word.get("end"),
                     word.get("speaker") or segment.get("speaker", ""),
                     round(float(word["probability"]), 3) if word.get("probability") is not None else "",
                     segment["start"], segment["end"], segment["text"]]
                    for segment in segments if segment["text"]
                    for word in segment.get("words") or []
                ]
                if word_rows:
                    with open(os.path.join(output, stem + "_words.csv"), "w", newline="", encoding="utf-8") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(word_headers)
                        writer.writerows(word_rows)
                    shared_word_rows.extend(word_rows)
                with open(txt_path, "w", encoding="utf-8") as handle:
                    for segment in segments:
                        if segment["text"]:
                            prefix = ("%s: " % segment["speaker"]) if segment.get("speaker") else ""
                            handle.write("[%0.2f-%0.2f] %s%s\n" % (segment["start"], segment["end"],
                                                                   prefix, segment["text"]))
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
            # Word-level counterpart of detections.csv: all files' word rows in
            # one CSV, so researchers don't have to merge the per-file word
            # CSVs themselves. Skipped entirely when no backend produced word
            # timings, matching the per-file behaviour.
            if shared_word_rows:
                with open(os.path.join(output, "detections_words.csv"), "w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(word_headers)
                    writer.writerows(shared_word_rows)
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
