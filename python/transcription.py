"""Local audio transcription pipeline.

The implementation deliberately imports Whisper libraries only when a job is
started.  This keeps the face/hand application startup fast and allows the
vision environments to remain usable when the optional audio environment is
not installed.
"""
from __future__ import annotations

import csv
import importlib
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple


AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".mp4", ".mov", ".mkv")
TRANSCRIPTION_VARIANTS = ["Whisper (OpenAI)", "Faster Whisper", "WhisperX"]


class ProcessingStopped(Exception):
    """Raised from inside a backend's progress hook when the user hits stop,
    so a stop can interrupt a file mid-transcription instead of only taking
    effect between files."""


class TranscriptionProcessor:
    # Interval between "still working" status lines while a file is being
    # transcribed. Class-level so tests can shrink it.
    WORK_HEARTBEAT_SECONDS = 30.0
    # Interval between "still loading" lines during model load, and how long
    # without any download/phase activity before the loading heartbeat stops
    # reassuring and points at the network instead.
    LOAD_HEARTBEAT_SECONDS = 20.0
    LOAD_STALL_SECONDS = 120.0

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
        # Position within the current batch, for intra-file progress events.
        self._file_index = 0
        self._file_total = 1
        self._last_percent = -1.0
        self._loading_phase = "starting speech runtime"
        self._loading_started = 0.0
        # Last time the load visibly moved (phase change or download bytes);
        # read by the loading heartbeat's stall escalation.
        self._load_activity_at = 0.0
        # Per-file work phase for the heartbeat: recognition loops report a
        # fraction (and get stall detection); decode/alignment/diarization
        # phases have no fraction and get plain elapsed-time lines.
        self._work_phase = "running speech recognition"
        self._work_tracks_fraction = True
        self._current_fraction = 0.0

    def _emit(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)

    def _set_loading_phase(self, phase: str) -> None:
        """Record and display the precise model-loading stage."""
        self._loading_phase = phase
        self._load_activity_at = time.monotonic()
        elapsed = time.monotonic() - self._loading_started if self._loading_started else 0.0
        self._emit("⏳ %s (%.1f s elapsed)..." % (phase, elapsed))

    def _check_stop(self) -> None:
        if self._stop.is_set():
            raise ProcessingStopped()

    def _emit_file_progress(self, fraction: float) -> None:
        """Move the UI progress bar while a single file transcribes.

        The renderer already maps 'audio_completed' progress_percent onto the
        bar, so intra-file updates reuse that event with a fractional
        position. Throttled to whole-percent steps to keep stdout light."""
        fraction = min(max(fraction, 0.0), 1.0)
        self._current_fraction = fraction  # read by the work heartbeat
        if not self.completion_callback or not self._file_total:
            return
        percent = (self._file_index + fraction) / self._file_total * 100
        if percent - self._last_percent >= 1.0:
            self._last_percent = percent
            self.completion_callback({"status": "audio_completed", "progress_percent": percent,
                                      "audio_index": self._file_index + 1,
                                      "total_audio": self._file_total})

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
        self._load_activity_at = time.monotonic()  # bytes are flowing
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

    def _whisper_progress(self):
        """Context manager: routes openai-whisper's internal tqdm bar (one
        update per 30 s window) into _emit_file_progress, and turns a pending
        stop into ProcessingStopped mid-file. whisper/transcribe.py resolves
        the bar as the module-global `tqdm.tqdm`, so swapping that module
        attribute is contained to whisper. No-op if the internals change."""
        import types as _types
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            try:
                # ``whisper.__init__`` exports a function named ``transcribe``.
                # A dotted import can therefore bind that function instead of
                # the module whose global tqdm object we need to replace.
                wt = importlib.import_module("whisper.transcribe")
                original = wt.tqdm
            except Exception:
                yield
                return
            processor = self

            class _EmitBar:
                def __init__(self, total: Any = None, **kwargs: Any):
                    self.total = int(total or 0)
                    self.n = 0

                def __enter__(self) -> "_EmitBar":
                    return self

                def __exit__(self, *args: Any) -> bool:
                    return False

                def update(self, n: int = 1) -> None:
                    processor._check_stop()
                    self.n += int(n or 0)
                    if self.total:
                        processor._emit_file_progress(self.n / self.total)

                def close(self) -> None: pass
                def refresh(self) -> None: pass
                def set_description(self, *args: Any, **kwargs: Any) -> None: pass

            wt.tqdm = _types.SimpleNamespace(tqdm=_EmitBar)
            try:
                yield
            finally:
                wt.tqdm = original

        return _cm()

    def _network_timeout(self):
        """Context manager: apply a default socket timeout while model
        weights may be fetched. whisperx's and whisper's weight fetches pass
        no timeout at all, so on a firewalled/proxied lab network a dropped
        connection hangs the load forever (2026-09-02 WhisperX 40-minute
        report); with a default timeout it becomes a clear error instead.
        Slow-but-flowing downloads are unaffected — the timeout applies per
        socket operation, not to the whole transfer."""
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            import socket
            try:
                seconds = float(os.environ.get("TINYEXPLORER_NETWORK_TIMEOUT", "60"))
            except ValueError:
                seconds = 60.0
            previous = socket.getdefaulttimeout()
            if seconds > 0:
                socket.setdefaulttimeout(seconds)
            try:
                yield
            finally:
                socket.setdefaulttimeout(previous)

        return _cm()

    def _set_work_phase(self, phase: str, tracks_fraction: bool) -> None:
        """Label the current per-file stage for the work heartbeat."""
        self._work_phase = phase
        self._work_tracks_fraction = tracks_fraction

    def _work_heartbeat(self):
        """Context manager: emit a status line every WORK_HEARTBEAT_SECONDS
        while a file is worked on, covering the phases that otherwise print
        nothing between 'Running speech recognition...' and the results
        (recognition compute, audio decode, alignment, diarization — the
        2026-09-02 'tiny model stalled' report). Recognition phases report
        percent done; four consecutive beats without movement add an explicit
        may-be-stuck warning so a real hang is distinguishable from slow
        progress. The renderer coalesces consecutive '⏳ Still' lines into a
        single live-updating row."""
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            stop = threading.Event()
            started = time.monotonic()
            processor = self

            def _beat() -> None:
                last_fraction = processor._current_fraction
                stalled_beats = 0
                while not stop.wait(processor.WORK_HEARTBEAT_SECONDS):
                    elapsed_min = (time.monotonic() - started) / 60.0
                    phase = processor._work_phase
                    if not processor._work_tracks_fraction:
                        stalled_beats = 0
                        last_fraction = processor._current_fraction
                        processor._emit("⏳ Still %s (%.1f min elapsed)..." % (phase, elapsed_min))
                        continue
                    fraction = processor._current_fraction
                    if fraction > last_fraction:
                        last_fraction = fraction
                        stalled_beats = 0
                        processor._emit("⏳ Still %s — %d%% of this file done (%.1f min elapsed)..."
                                        % (phase, int(fraction * 100), elapsed_min))
                    else:
                        stalled_beats += 1
                        warning = ("" if stalled_beats < 4 else
                                   " If this keeps repeating, the job may be stuck — "
                                   "press Stop and use 'Copy log for bug report'.")
                        processor._emit("⏳ Still %s — %d%% of this file done, no change for "
                                        "%.0f s (%.1f min elapsed)...%s"
                                        % (phase, int(fraction * 100),
                                           stalled_beats * processor.WORK_HEARTBEAT_SECONDS,
                                           elapsed_min, warning))

            threading.Thread(target=_beat, daemon=True).start()
            try:
                yield
            finally:
                stop.set()

        return _cm()

    def _decode_audio(self, path: str) -> Any:
        """_load_audio plus heartbeat phase bookkeeping: PyAV-decoding a long
        recording in one go is otherwise a silent stretch users read as a
        hang."""
        previous = (self._work_phase, self._work_tracks_fraction)
        self._set_work_phase("decoding the audio track", False)
        self._emit("🔉 Decoding audio track...")
        try:
            return self._load_audio(path)
        finally:
            self._set_work_phase(*previous)

    def _emit_runtime_info(self, variant: str) -> None:
        """One-line library/threading summary for the progress log, and thus
        for the copy-log bug report. The 2026-09-03 field hang came down to
        torch CPU inference stalling on a hybrid P/E-core machine, and no
        report carried the thread count or OMP settings needed to see that.
        Reads only libraries the backend import already loaded (whisper and
        whisperx pull in torch, faster_whisper pulls in ctranslate2) — a
        diagnostics line must never trigger a heavy import of its own."""
        import platform
        parts = ["Python %s" % platform.python_version()]
        try:
            if variant == "Faster Whisper":
                ct2 = sys.modules.get("ctranslate2")
                if ct2 is not None:
                    parts.append("ctranslate2 %s" % getattr(ct2, "__version__", "unknown"))
            else:
                torch = sys.modules.get("torch")
                if torch is not None:
                    parts.append("torch %s" % getattr(torch, "__version__", "unknown"))
                    parts.append("%d torch CPU threads" % torch.get_num_threads())
        except Exception:
            pass
        overrides = ["%s=%s" % (key, os.environ[key]) for key in sorted(os.environ)
                     if key.startswith(("OMP_", "KMP_", "MKL_"))
                     or key == "TINYEXPLORER_TORCH_THREADS"]
        parts.append("thread env: " + (", ".join(overrides) if overrides else "defaults"))
        self._emit("🧵 Speech runtime: %s" % "; ".join(parts))

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
        self._loading_started = time.monotonic()
        self._loading_phase = "starting speech runtime"
        self._load_activity_at = time.monotonic()
        # Importing the speech libraries alone takes ~45 s even on a fast
        # machine (torch + pyannote + lightning for WhisperX), and antivirus
        # cold-scans can stretch that to minutes of silence that users read
        # as a hang. Heartbeat until the load returns — but once nothing has
        # visibly moved for LOAD_STALL_SECONDS, stop reassuring and point at
        # the likely cause (2026-09-02 lab report: WhisperX sat in the load
        # phase for 40+ minutes on a filtered network while we kept printing
        # "the app is not stuck").
        heartbeat_stop = threading.Event()

        def _heartbeat() -> None:
            waited = 0.0
            while not heartbeat_stop.wait(self.LOAD_HEARTBEAT_SECONDS):
                waited += self.LOAD_HEARTBEAT_SECONDS
                quiet_for = time.monotonic() - self._load_activity_at
                if quiet_for >= self.LOAD_STALL_SECONDS:
                    self._emit("⏳ Still loading %s (%.0f s; %s) — no download or loading "
                               "activity for %.0f s. A firewall or proxy may be blocking "
                               "model downloads (huggingface.co / github.com). If this keeps "
                               "repeating, press Stop, check the network, and use "
                               "'Copy log for bug report'."
                               % (variant, waited, self._loading_phase, quiet_for))
                else:
                    self._emit("⏳ Still loading %s (%.0f s; %s) — first load is slow while the speech "
                               "libraries are read and scanned; the app is not stuck..."
                               % (variant, waited, self._loading_phase))

        threading.Thread(target=_heartbeat, daemon=True).start()
        try:
            with self._network_timeout():
                self._load_model_impl(variant, model_name)
        finally:
            heartbeat_stop.set()
        elapsed = time.monotonic() - self._loading_started
        self._emit("✅ %s model ready (%.1f s)." % (variant, elapsed))
        if not self._hf_token():
            self._emit("ℹ️ Set a Hugging Face token (🔑 in the app, or TINYEXPLORER_HF_TOKEN) to "
                       "enable speaker diarization; exporting without speaker labels.")
        self._model_variant = variant
        self._model_size = size

    def _load_model_impl(self, variant: str, model_name: str) -> None:
        if variant == "Whisper (OpenAI)":
            self._set_loading_phase("Importing OpenAI Whisper")
            import whisper
            self._emit_runtime_info(variant)
            self._set_loading_phase("Checking OpenAI Whisper model weights")
            self._ensure_openai_whisper_weights(model_name)
            self._set_loading_phase("Loading OpenAI Whisper checkpoint into memory")
            self._model = whisper.load_model(model_name)
        elif variant == "Faster Whisper":
            self._set_loading_phase("Importing Faster Whisper")
            from faster_whisper import WhisperModel
            self._emit_runtime_info(variant)
            device = self._device()
            compute = "float16" if device == "cuda" else "int8"
            self._set_loading_phase("Loading Faster Whisper model")
            with self._hf_download_progress("Faster Whisper %s" % model_name):
                self._model = WhisperModel(model_name, device=device, compute_type=compute)
        elif variant == "WhisperX":
            self._set_loading_phase("Importing WhisperX and speech libraries")
            import whisperx
            self._emit_runtime_info(variant)
            device = self._device()
            self._set_loading_phase("Loading WhisperX model and voice-activity pipeline")
            with self._hf_download_progress("WhisperX %s" % model_name):
                self._model = whisperx.load_model(model_name, device=device,
                                                  compute_type="float16" if device == "cuda" else "int8")
        else:
            raise ValueError("Unknown transcription model: %s" % variant)

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
        self._check_stop()
        self._emit("🎤 Running speech recognition (may take a while for long recordings)...")
        self._current_fraction = 0.0
        self._set_work_phase("running speech recognition", True)
        with self._work_heartbeat():
            return self._transcribe_impl(path, variant)

    def _transcribe_impl(self, path: str, variant: str) -> Tuple[List[Dict[str, Any]], str]:
        if variant == "Faster Whisper":
            # transcribe() returns a lazy generator: consuming it segment by
            # segment lets the UI bar track s.end/duration and lets a stop
            # interrupt mid-file instead of after the whole recording.
            raw_segments, info = self._model.transcribe(path, word_timestamps=True)
            duration = float(getattr(info, "duration", 0) or 0)
            segments = []
            for s in raw_segments:
                self._check_stop()
                segments.append(self._segment_dict(s.start, s.end, s.text, getattr(s, "words", None),
                                                   confidence=getattr(s, "avg_logprob", None)))
                if duration:
                    self._emit_file_progress(float(s.end or 0) / duration)
            if self._hf_token():  # skip the audio decode when diarization is off
                segments = self._assign_speakers(segments, self._decode_audio(path))
            return segments, getattr(info, "language", "unknown")
        if variant == "WhisperX":
            # whisperx.load_model() returns a FasterWhisperPipeline whose
            # transcribe() takes neither word_timestamps nor fp16 but does
            # accept a percent progress_callback (3.8+). Word-level times come
            # from the separate align() stage, speaker labels from the
            # diarization stage — both are best-effort extras.
            audio = self._decode_audio(path)

            def _wx_progress(percent: Any) -> None:
                self._check_stop()
                self._emit_file_progress(float(percent or 0) / 100.0)

            result = self._model.transcribe(
                audio, batch_size=int(os.environ.get("TINYEXPLORER_WHISPERX_BATCH", "8")),
                progress_callback=_wx_progress)
            language = result.get("language", "unknown")
            segments = self._whisperx_enrich(result.get("segments", []), audio, language)
            return [self._segment_dict(s.get("start"), s.get("end"), s.get("text", ""), s.get("words"),
                                       s.get("speaker"))
                    for s in segments], language
        audio = self._decode_audio(path)
        with self._whisper_progress():
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
            self._set_work_phase("aligning word-level timestamps", False)
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
        if not self._hf_token() or not segments or self._stop.is_set():
            return segments
        device = self._device()
        # pyannote exposes no progress hook, so minutes of silence are normal
        # here: name the phase so the heartbeat never calls it stuck.
        self._set_work_phase("preparing speaker diarization", False)
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
                with self._network_timeout():
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
            self._set_work_phase("identifying speakers", False)
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
            "model", "segment_start", "segment_end", "segment_text",
        ]
        shared_rows: List[List[Any]] = []
        shared_word_rows: List[List[Any]] = []
        summary_rows: List[List[Any]] = []
        try:
            if not files:
                raise ValueError("No supported audio or video files found")
            if self.completion_callback:
                self.completion_callback({"status": "processing_started", "total_audio": len(files)})
            self._emit("🎤 Found %d audio/video file(s)" % len(files))
            for index, path in enumerate(files):
                if self._stop.is_set():
                    break
                self._file_index, self._file_total, self._last_percent = index, len(files), -1.0
                self._emit("🎤 Transcribing %d/%d: %s" % (index + 1, len(files), os.path.basename(path)))
                try:
                    segments, language = self._transcribe(path, variant, size)
                except ProcessingStopped:
                    self._emit("⏹️ Processing stopped by user")
                    break
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
                     variant, segment["start"], segment["end"], segment["text"]]
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
