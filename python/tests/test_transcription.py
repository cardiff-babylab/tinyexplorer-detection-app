import csv
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from transcription import TRANSCRIPTION_VARIANTS, ProcessingStopped, TranscriptionProcessor


class FakeTranscriptionProcessor(TranscriptionProcessor):
    def _transcribe(self, path, variant, size=None):
        return [{
            "start": 0.0, "end": 1.25, "text": "hello world", "speaker": "SPEAKER_00",
            "confidence": -0.42,
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.5, "probability": 0.91, "speaker": "SPEAKER_00"},
                # No word-level speaker: the export must fall back to the segment's.
                {"word": "world", "start": 0.6, "end": 1.25, "probability": 0.874999, "speaker": None},
            ],
        }], "en"


class TranscriptionTests(unittest.TestCase):
    def test_backends_are_exposed(self):
        self.assertEqual(TRANSCRIPTION_VARIANTS, ["Whisper (OpenAI)", "Faster Whisper", "WhisperX"])

    def test_writes_timestamped_csv_and_text(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "sample.wav"
            source.write_bytes(b"not decoded by fake backend")
            output = Path(temp) / "results"
            processor = FakeTranscriptionProcessor()
            processor.process(str(source), "Faster Whisper", str(output))
            result_dirs = list(output.glob("transcription_results_*"))
            self.assertEqual(len(result_dirs), 1)
            with (result_dirs[0] / "sample_transcript.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["text"], "hello world")
            self.assertEqual(
                rows[0].keys(),
                {
                    "id", "frame_idx", "filename", "mode", "start", "end",
                    "label", "confidence", "model", "text", "language", "speaker",
                },
            )
            self.assertEqual(rows[0]["filename"], "sample.wav")
            self.assertEqual(rows[0]["mode"], "speech")
            self.assertEqual(rows[0]["label"], "speech")
            self.assertEqual(rows[0]["model"], "Faster Whisper")
            self.assertEqual(rows[0]["speaker"], "SPEAKER_00")
            # The segment's own confidence value must be preserved, not blanked.
            self.assertEqual(rows[0]["confidence"], "-0.42")
            self.assertTrue((result_dirs[0] / "detections.csv").exists())
            self.assertTrue((result_dirs[0] / "summary.csv").exists())
            with (result_dirs[0] / "summary.csv").open(newline="", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(summary_rows[0]["segments"], "1")
            self.assertEqual(summary_rows[0]["type"], "audio")
            self.assertIn("[0.00-1.25] SPEAKER_00: hello world",
                          (result_dirs[0] / "sample_transcript.txt").read_text())

    def test_writes_word_level_csv_with_speaker_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "sample.wav"
            source.write_bytes(b"not decoded by fake backend")
            output = Path(temp) / "results"
            processor = FakeTranscriptionProcessor()
            processor.process(str(source), "WhisperX", str(output))
            result_dir = next(output.glob("transcription_results_*"))
            with (result_dir / "sample_words.csv").open(newline="", encoding="utf-8") as handle:
                words = list(csv.DictReader(handle))
        self.assertEqual(
            list(words[0].keys()),
            ["filename", "word", "start", "end", "speaker", "word_score",
             "model", "segment_start", "segment_end", "segment_text"],
        )
        self.assertEqual(words[0]["filename"], "sample.wav")
        self.assertEqual(words[0]["word"], "hello")
        self.assertEqual(words[0]["model"], "WhisperX")
        self.assertEqual(words[0]["speaker"], "SPEAKER_00")
        self.assertEqual(words[0]["word_score"], "0.91")
        # Word without its own speaker inherits the segment speaker.
        self.assertEqual(words[1]["speaker"], "SPEAKER_00")
        self.assertEqual(words[1]["word_score"], "0.875")
        self.assertEqual(words[1]["segment_text"], "hello world")

    def test_no_words_csv_when_backend_has_no_word_output(self):
        class UtteranceOnlyProcessor(TranscriptionProcessor):
            def _transcribe(self, path, variant, size=None):
                return [{"start": 0.0, "end": 1.0, "text": "hi", "words": [], "speaker": ""}], "en"

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "sample.wav"
            source.write_bytes(b"fake")
            output = Path(temp) / "results"
            UtteranceOnlyProcessor().process(str(source), "WhisperX", str(output))
            result_dir = next(output.glob("transcription_results_*"))
            self.assertFalse((result_dir / "sample_words.csv").exists())
            self.assertIn("[0.00-1.00] hi", (result_dir / "sample_transcript.txt").read_text())


def _fake_whisper_module():
    mod = types.ModuleType("whisper")

    class _Model:
        # openai-whisper: transcribe(audio, **decode_options) — permissive kwargs.
        # Real segments always carry avg_logprob; words carry probability when
        # word_timestamps=True.
        def transcribe(self, audio, word_timestamps=False, fp16=True, **decode_options):
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi",
                                  "avg_logprob": -0.25,
                                  "words": [{"word": "hi", "start": 0.0, "end": 1.0,
                                             "probability": 0.88}]}],
                    "language": "en"}

    mod.load_model = lambda name: _Model()
    return mod


def _fake_faster_whisper_module():
    mod = types.ModuleType("faster_whisper")

    class _Word:
        # faster-whisper Word namedtuple fields.
        word, start, end, probability = "hi", 0.0, 1.0, 0.88

    class _Segment:
        start, end, text = 0.0, 1.0, "hi"
        avg_logprob = -0.31
        words = [_Word()]

    class _Info:
        language = "en"
        duration = 1.0

    class WhisperModel:
        def __init__(self, name, device="cpu", compute_type="int8"):
            pass

        def transcribe(self, audio, word_timestamps=False, beam_size=5,
                       language=None, task="transcribe"):
            return iter([_Segment()]), _Info()

    mod.WhisperModel = WhisperModel
    return mod


def _fake_whisperx_with_diarization():
    """Fake whisperx incl. align + diarize with the REAL 3.8.6 signatures:
    DiarizationPipeline takes `token=`, not `use_auth_token=`."""
    mod = _fake_whisperx_module()
    mod.load_align_model = lambda language_code, device: ("align-model", {"lang": language_code})
    mod.align = lambda segments, model_a, metadata, audio, device: {
        "segments": [dict(s, words=[{"word": "hi", "start": 0.0, "end": 1.0, "score": 0.9}])
                     for s in segments]}

    diarize_mod = types.ModuleType("whisperx.diarize")

    class DiarizationPipeline:
        def __init__(self, model_name=None, token=None, device="cpu", cache_dir=None):
            if not token:
                raise ValueError("token required")

        def __call__(self, audio, num_speakers=None, min_speakers=None, max_speakers=None):
            return "diarize-df"

    def assign_word_speakers(diarize_df, result):
        return {"segments": [dict(s, speaker="SPEAKER_00",
                                  words=[dict(w, speaker="SPEAKER_00") for w in s.get("words", [])])
                             for s in result["segments"]]}

    diarize_mod.DiarizationPipeline = DiarizationPipeline
    diarize_mod.assign_word_speakers = assign_word_speakers
    mod.diarize = diarize_mod
    return mod, diarize_mod


def _fake_whisperx_module():
    mod = types.ModuleType("whisperx")

    class _Pipeline:
        # Mirrors whisperx.asr.FasterWhisperPipeline.transcribe (3.8.6): it
        # takes NO word_timestamps/fp16 kwargs, so passing them raises
        # TypeError just like the real library. progress_callback is the real
        # 3.8+ per-segment percent hook the app relies on.
        def transcribe(self, audio, batch_size=None, num_workers=0, language=None,
                       task=None, chunk_size=30, print_progress=False,
                       combined_progress=False, verbose=False, progress_callback=None):
            if progress_callback is not None:
                progress_callback(100.0)
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
                    "language": "en"}

    mod.load_model = lambda name, device="cpu", compute_type="int8": _Pipeline()
    mod.load_audio = lambda path: [0.0] * 16000
    return mod


class BackendCallSignatureTests(unittest.TestCase):
    """Run the real _transcribe against fakes that enforce each library's
    transcribe() signature, so kwarg mismatches fail here instead of at
    runtime inside the packaged app."""

    def _run(self, variant, modules, processor=None):
        processor = processor or TranscriptionProcessor()
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(TranscriptionProcessor, "_load_audio",
                                  staticmethod(lambda path: [0.0] * 16000)):
            segments, language = processor._transcribe("sample.wav", variant)
        self.assertEqual(language, "en")
        self.assertEqual(segments[0]["text"], "hi")
        return processor

    def test_whisper_openai_call_matches_backend_signature(self):
        self._run("Whisper (OpenAI)", {"whisper": _fake_whisper_module()})

    def test_faster_whisper_call_matches_backend_signature(self):
        self._run("Faster Whisper", {"faster_whisper": _fake_faster_whisper_module()})

    def test_whisperx_call_matches_backend_signature(self):
        self._run("WhisperX", {"whisperx": _fake_whisperx_module()})

    def test_whisperx_diarization_uses_current_token_kwarg(self):
        mod, diarize_mod = _fake_whisperx_with_diarization()
        processor = TranscriptionProcessor()
        with mock.patch.dict(sys.modules, {"whisperx": mod, "whisperx.diarize": diarize_mod}), \
                mock.patch.dict(os.environ, {"TINYEXPLORER_HF_TOKEN": "hf_test"}), \
                mock.patch.object(TranscriptionProcessor, "_load_audio",
                                  staticmethod(lambda path: [0.0] * 16000)):
            segments, language = processor._transcribe("sample.wav", "WhisperX")
        self.assertEqual(language, "en")
        self.assertEqual(segments[0]["speaker"], "SPEAKER_00")
        self.assertEqual(segments[0]["words"][0]["speaker"], "SPEAKER_00")
        self.assertEqual(segments[0]["words"][0]["probability"], 0.9)

    def test_whisperx_diarization_falls_back_to_legacy_gated_model(self):
        mod, diarize_mod = _fake_whisperx_with_diarization()
        original_pipeline = diarize_mod.DiarizationPipeline

        class _GatedDefaultPipeline(original_pipeline):
            def __init__(self, model_name=None, token=None, device="cpu", cache_dir=None):
                if model_name is None:
                    raise RuntimeError("403 Client Error: gated repo")
                super().__init__(model_name=model_name, token=token, device=device, cache_dir=cache_dir)

        diarize_mod.DiarizationPipeline = _GatedDefaultPipeline
        processor = TranscriptionProcessor()
        with mock.patch.dict(sys.modules, {"whisperx": mod, "whisperx.diarize": diarize_mod}), \
                mock.patch.dict(os.environ, {"TINYEXPLORER_HF_TOKEN": "hf_test"}), \
                mock.patch.object(TranscriptionProcessor, "_load_audio",
                                  staticmethod(lambda path: [0.0] * 16000)):
            segments, _ = processor._transcribe("sample.wav", "WhisperX")
        # Default model 403s; the pyannote 3.1 fallback still yields speakers.
        self.assertEqual(segments[0]["speaker"], "SPEAKER_00")

    def test_model_size_is_passed_to_backend_and_triggers_reload(self):
        loaded_sizes = []
        mod = types.ModuleType("whisper")

        class _Model:
            def transcribe(self, audio, word_timestamps=False, fp16=True, **decode_options):
                return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi", "words": []}],
                        "language": "en"}

        def load_model(name):
            loaded_sizes.append(name)
            return _Model()

        mod.load_model = load_model
        processor = TranscriptionProcessor()
        with mock.patch.dict(sys.modules, {"whisper": mod}), \
                mock.patch.object(TranscriptionProcessor, "_load_audio",
                                  staticmethod(lambda path: [0.0] * 16000)):
            processor._transcribe("sample.wav", "Whisper (OpenAI)", "large-v2")
            processor._transcribe("sample.wav", "Whisper (OpenAI)", "large-v2")  # cached, no reload
            processor._transcribe("sample.wav", "Whisper (OpenAI)", "small")     # size change reloads
        self.assertEqual(loaded_sizes, ["large-v2", "small"])

    def test_switching_models_in_one_session_reloads_backend(self):
        # A processor is reused across runs; picking a different model for a
        # subsequent run must load the new backend and transcribe cleanly.
        modules = {
            "whisper": _fake_whisper_module(),
            "faster_whisper": _fake_faster_whisper_module(),
            "whisperx": _fake_whisperx_module(),
        }
        processor = self._run("Faster Whisper", modules)
        self._run("Whisper (OpenAI)", modules, processor=processor)
        self.assertEqual(processor._model_variant, "Whisper (OpenAI)")
        self._run("WhisperX", modules, processor=processor)
        self.assertEqual(processor._model_variant, "WhisperX")


class EndToEndCsvExportTests(unittest.TestCase):
    """Run the real pipeline (process -> _transcribe -> CSV export) against
    each fake backend and check the researcher-facing CSV contract: each
    model's own confidence values are preserved, word-level speaker labels
    survive to the word CSVs, and a merged word-level detections file is
    written next to detections.csv."""

    def _process(self, variant, modules, env=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        source = Path(temp.name) / "clips"
        source.mkdir()
        for name in ("a.wav", "b.wav"):
            (source / name).write_bytes(b"not decoded by fake backend")
        output = Path(temp.name) / "results"
        processor = TranscriptionProcessor()
        # Default to NO token so the no-diarization tests stay deterministic
        # even on machines with a global HF_TOKEN; tests opt in via env.
        env_vars = {"TINYEXPLORER_HF_TOKEN": "", "HF_TOKEN": ""}
        env_vars.update(env or {})
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.dict(os.environ, env_vars), \
                mock.patch.object(TranscriptionProcessor, "_load_audio",
                                  staticmethod(lambda path: [0.0] * 16000)):
            processor.process(str(source), variant, str(output))
        return next(output.glob("transcription_results_*"))

    @staticmethod
    def _read(path):
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_whisper_openai_exports_confidence_and_merged_words(self):
        result_dir = self._process("Whisper (OpenAI)", {"whisper": _fake_whisper_module()})
        transcript = self._read(result_dir / "a_transcript.csv")
        # openai-whisper's segment confidence variable is avg_logprob.
        self.assertEqual(transcript[0]["confidence"], "-0.25")
        detections = self._read(result_dir / "detections.csv")
        self.assertEqual([row["confidence"] for row in detections], ["-0.25", "-0.25"])
        words = self._read(result_dir / "a_words.csv")
        self.assertEqual(words[0]["filename"], "a.wav")
        self.assertEqual(words[0]["word_score"], "0.88")
        merged = self._read(result_dir / "detections_words.csv")
        self.assertEqual([row["filename"] for row in merged], ["a.wav", "b.wav"])
        self.assertEqual([row["model"] for row in merged],
                         ["Whisper (OpenAI)", "Whisper (OpenAI)"])
        self.assertEqual(list(merged[0].keys()), list(words[0].keys()))

    def test_faster_whisper_exports_confidence_and_merged_words(self):
        result_dir = self._process("Faster Whisper",
                                   {"faster_whisper": _fake_faster_whisper_module()})
        transcript = self._read(result_dir / "a_transcript.csv")
        # faster-whisper's segment confidence variable is avg_logprob.
        self.assertEqual(transcript[0]["confidence"], "-0.31")
        merged = self._read(result_dir / "detections_words.csv")
        self.assertEqual([(row["filename"], row["word"], row["word_score"]) for row in merged],
                         [("a.wav", "hi", "0.88"), ("b.wav", "hi", "0.88")])

    def test_whisperx_exports_word_scores_speakers_and_merged_words(self):
        mod, diarize_mod = _fake_whisperx_with_diarization()
        result_dir = self._process(
            "WhisperX", {"whisperx": mod, "whisperx.diarize": diarize_mod},
            env={"TINYEXPLORER_HF_TOKEN": "hf_test"})
        transcript = self._read(result_dir / "a_transcript.csv")
        # whisperx's batched pipeline has no native segment confidence; the
        # export falls back to the mean aligned word score.
        self.assertEqual(transcript[0]["confidence"], "0.9")
        self.assertEqual(transcript[0]["speaker"], "SPEAKER_00")
        words = self._read(result_dir / "a_words.csv")
        self.assertEqual(words[0]["speaker"], "SPEAKER_00")
        merged = self._read(result_dir / "detections_words.csv")
        self.assertEqual([row["speaker"] for row in merged], ["SPEAKER_00", "SPEAKER_00"])

    def test_no_merged_words_csv_when_backend_has_no_word_output(self):
        # Plain whisperx fake: no align model available, so no word output at
        # all — the merged word file must not appear as an empty husk.
        result_dir = self._process("WhisperX", {"whisperx": _fake_whisperx_module()})
        self.assertFalse((result_dir / "detections_words.csv").exists())

    # --- Diarization is offered for every backend, with and without token ---

    def _diarization_modules(self, backend_name, backend_module):
        whisperx_mod, diarize_mod = _fake_whisperx_with_diarization()
        return {backend_name: backend_module, "whisperx": whisperx_mod,
                "whisperx.diarize": diarize_mod}

    def test_whisper_openai_diarizes_when_token_present(self):
        result_dir = self._process(
            "Whisper (OpenAI)",
            self._diarization_modules("whisper", _fake_whisper_module()),
            env={"TINYEXPLORER_HF_TOKEN": "hf_test"})
        transcript = self._read(result_dir / "a_transcript.csv")
        self.assertEqual(transcript[0]["speaker"], "SPEAKER_00")
        words = self._read(result_dir / "a_words.csv")
        self.assertEqual(words[0]["speaker"], "SPEAKER_00")

    def test_faster_whisper_diarizes_when_token_present(self):
        result_dir = self._process(
            "Faster Whisper",
            self._diarization_modules("faster_whisper", _fake_faster_whisper_module()),
            env={"TINYEXPLORER_HF_TOKEN": "hf_test"})
        transcript = self._read(result_dir / "a_transcript.csv")
        self.assertEqual(transcript[0]["speaker"], "SPEAKER_00")
        merged = self._read(result_dir / "detections_words.csv")
        self.assertEqual([row["speaker"] for row in merged], ["SPEAKER_00", "SPEAKER_00"])

    def test_faster_whisper_without_token_completes_with_empty_speakers(self):
        # The user declined / never configured a token: the run must complete
        # normally, just without speaker labels.
        result_dir = self._process("Faster Whisper",
                                   {"faster_whisper": _fake_faster_whisper_module()})
        transcript = self._read(result_dir / "a_transcript.csv")
        self.assertEqual(transcript[0]["text"], "hi")
        self.assertEqual(transcript[0]["speaker"], "")
        words = self._read(result_dir / "a_words.csv")
        self.assertEqual(words[0]["speaker"], "")

    def test_diarization_failure_degrades_to_empty_speakers(self):
        # Token present but the gated pipeline is unusable (no access,
        # offline...): the transcription itself must still be exported.
        modules = self._diarization_modules("faster_whisper", _fake_faster_whisper_module())

        class _BrokenPipeline:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("403 Client Error: gated repo")

        modules["whisperx.diarize"].DiarizationPipeline = _BrokenPipeline
        result_dir = self._process("Faster Whisper", modules,
                                   env={"TINYEXPLORER_HF_TOKEN": "hf_test"})
        transcript = self._read(result_dir / "a_transcript.csv")
        self.assertEqual(transcript[0]["text"], "hi")
        self.assertEqual(transcript[0]["speaker"], "")
        self.assertTrue((result_dir / "detections_words.csv").exists())


def _fake_hub_modules():
    """A fake huggingface_hub whose progress factory raises unless the app's
    download hook has replaced it — any hub download outside the hook fails."""
    hf_pkg = types.ModuleType("huggingface_hub")
    hf_fd = types.ModuleType("huggingface_hub.file_download")

    def _unpatched_context(*args, **kwargs):
        raise AssertionError("hub download ran outside the progress hook")

    hf_fd._get_progress_bar_context = _unpatched_context
    hf_pkg.file_download = hf_fd
    return hf_pkg, hf_fd, _unpatched_context


def _simulate_hub_download(filename, total):
    """What a backend does internally while loading weights: build a bar via
    the (possibly patched) factory and push byte updates through it."""
    import huggingface_hub.file_download as fd
    with fd._get_progress_bar_context(desc=filename, log_level=0, total=total,
                                      name="huggingface_hub.http_get") as bar:
        bar.update(total // 2)
        bar.update(total - total // 2)


class DownloadProgressTests(unittest.TestCase):
    def test_hf_hook_covers_progress_bar_context_hub_036(self):
        """huggingface_hub >= 0.36 (incl. the xet transport) builds bars via
        file_download._get_progress_bar_context — the hook must intercept it."""
        hf_pkg, hf_fd, _real_context = _fake_hub_modules()
        messages = []
        processor = TranscriptionProcessor(progress_callback=messages.append)
        with mock.patch.dict(sys.modules, {"huggingface_hub": hf_pkg,
                                           "huggingface_hub.file_download": hf_fd}):
            with processor._hf_download_progress("test model"):
                # Mirror hub 0.36's real flow: the bar is created before the
                # response (total unknown), then handed back via _tqdm_bar
                # together with the real size once headers arrive.
                with hf_fd._get_progress_bar_context(
                        desc="model.bin", log_level=0, total=None,
                        name="huggingface_hub.http_get") as outer:
                    with hf_fd._get_progress_bar_context(
                            desc="model.bin", log_level=0, total=4 * 1048576,
                            name="huggingface_hub.http_get", _tqdm_bar=outer) as bar:
                        bar.update(2 * 1048576)
                        bar.update(2 * 1048576)
            # Restored afterwards.
            self.assertIs(hf_fd._get_progress_bar_context, _real_context)
        downloads = [m for m in messages if m.startswith("⏳ Downloading model.bin:")]
        self.assertTrue(downloads)
        self.assertIn("50%", downloads[0])
        self.assertIn("100%", downloads[-1])

    def test_faster_whisper_load_runs_inside_progress_hook(self):
        hf_pkg, hf_fd, _ = _fake_hub_modules()
        mod = types.ModuleType("faster_whisper")

        class WhisperModel:
            def __init__(self, name, device="cpu", compute_type="int8"):
                _simulate_hub_download("model.bin", 4 * 1048576)

        mod.WhisperModel = WhisperModel
        messages = []
        processor = TranscriptionProcessor(progress_callback=messages.append)
        with mock.patch.dict(sys.modules, {"faster_whisper": mod,
                                           "huggingface_hub": hf_pkg,
                                           "huggingface_hub.file_download": hf_fd}):
            processor._load_model("Faster Whisper", "tiny")
        # Would raise inside the fake hub if _load_model dropped the hook.
        self.assertTrue(any(m.startswith("⏳ Downloading model.bin: 100%") for m in messages))

    def test_whisperx_load_runs_inside_progress_hook(self):
        hf_pkg, hf_fd, _ = _fake_hub_modules()
        mod = _fake_whisperx_module()
        original_load = mod.load_model

        def load_model(name, device="cpu", compute_type="int8"):
            _simulate_hub_download("model.safetensors", 8 * 1048576)
            return original_load(name, device=device, compute_type=compute_type)

        mod.load_model = load_model
        messages = []
        processor = TranscriptionProcessor(progress_callback=messages.append)
        with mock.patch.dict(sys.modules, {"whisperx": mod,
                                           "huggingface_hub": hf_pkg,
                                           "huggingface_hub.file_download": hf_fd}):
            processor._load_model("WhisperX", "tiny")
        self.assertTrue(any(m.startswith("⏳ Downloading model.safetensors: 100%") for m in messages))

    def test_openai_weight_predownload_reports_progress(self):
        import io
        import urllib.request

        payload = b"x" * (2 * 1048576 + 512)

        class _FakeResponse(io.BytesIO):
            def __init__(self, data):
                super().__init__(data)
                self.headers = {"Content-Length": str(len(data))}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        mod = types.ModuleType("whisper")
        mod._MODELS = {"base": "https://example.invalid/base.pt"}
        messages = []
        with tempfile.TemporaryDirectory() as temp:
            processor = TranscriptionProcessor(progress_callback=messages.append)
            with mock.patch.dict(sys.modules, {"whisper": mod}), \
                    mock.patch.dict(os.environ, {"TINYEXPLORER_WHISPER_CACHE": temp}), \
                    mock.patch.object(urllib.request, "urlopen", lambda url: _FakeResponse(payload)):
                processor._ensure_openai_whisper_weights("base")
                downloads = [m for m in messages if m.startswith("⏳ Downloading Whisper base:")]
                self.assertTrue(downloads)
                self.assertIn("100%", downloads[-1])
                self.assertEqual((Path(temp) / "base.pt").read_bytes(), payload)

                # Second call must hit the cache, not the network.
                def _fail(url):
                    raise AssertionError("network hit despite cached weights")
                with mock.patch.object(urllib.request, "urlopen", _fail):
                    processor._ensure_openai_whisper_weights("base")


class LoadAudioTests(unittest.TestCase):
    """In-process decoding via PyAV — the packaged app has no ffmpeg binary."""

    def test_decodes_wav_without_ffmpeg_binary(self):
        try:
            import av  # noqa: F401
        except ImportError:
            self.skipTest("PyAV not installed in this environment")
        import numpy as np
        import wave

        with tempfile.TemporaryDirectory() as temp:
            wav_path = Path(temp) / "tone.wav"
            rate = 8000
            samples = (np.sin(2 * np.pi * 440 * np.arange(rate) / rate) * 20000).astype("<i2")
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(rate)
                handle.writeframes(samples.tobytes())
            with mock.patch.dict(os.environ, {"PATH": ""}):
                audio = TranscriptionProcessor._load_audio(str(wav_path))
        self.assertEqual(audio.dtype, np.float32)
        # Resampled 8 kHz -> 16 kHz, so roughly twice the samples of one second.
        self.assertGreater(len(audio), rate * 1.5)
        self.assertLessEqual(float(np.abs(audio).max()), 1.0)


class ProgressAndStopTests(unittest.TestCase):
    """Intra-file progress events and mid-file stop (2026-09-01 Windows
    feedback: the bar sat at 0% for the whole file and stop only applied
    between files)."""

    def _events_for(self, variant, modules):
        events = []
        processor = TranscriptionProcessor(completion_callback=events.append)
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(TranscriptionProcessor, "_load_audio",
                                  staticmethod(lambda path: [0.0] * 16000)):
            processor._transcribe("sample.wav", variant)
        return [e for e in events if e.get("status") == "audio_completed"]

    def test_faster_whisper_emits_intra_file_progress(self):
        events = self._events_for("Faster Whisper",
                                  {"faster_whisper": _fake_faster_whisper_module()})
        # Fake: one segment ending at 1.0 of a 1.0 s file -> 100%.
        self.assertTrue(events)
        self.assertEqual(events[-1]["progress_percent"], 100.0)

    def test_whisperx_emits_intra_file_progress(self):
        events = self._events_for("WhisperX", {"whisperx": _fake_whisperx_module()})
        self.assertTrue(events)
        self.assertEqual(events[-1]["progress_percent"], 100.0)

    def test_pending_stop_interrupts_mid_file(self):
        processor = TranscriptionProcessor()
        processor.stop_processing()
        with mock.patch.dict(sys.modules, {"faster_whisper": _fake_faster_whisper_module()}):
            with self.assertRaises(ProcessingStopped):
                processor._transcribe("sample.wav", "Faster Whisper")

    def test_stopped_run_still_writes_merged_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in"
            source.mkdir()
            (source / "a.wav").write_bytes(b"")
            (source / "b.wav").write_bytes(b"")
            processor = FakeTranscriptionProcessor()
            original = FakeTranscriptionProcessor._transcribe

            def stop_after_first(self_, path, variant, size=None):
                # First file transcribes normally, then the user hits stop:
                # the second file must raise instead of transcribing.
                if self_._file_index == 0:
                    return original(self_, path, variant, size)
                raise ProcessingStopped()

            with mock.patch.object(FakeTranscriptionProcessor, "_transcribe", stop_after_first):
                processor.process(str(source), "Faster Whisper", tmp)
            result_dir = next(Path(tmp).glob("transcription_results_*"))
            merged = (result_dir / "detections.csv").read_text()
            self.assertIn("a.wav", merged)
            self.assertNotIn("b.wav", merged)


if __name__ == "__main__":
    unittest.main()
