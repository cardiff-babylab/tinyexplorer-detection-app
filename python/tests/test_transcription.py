import csv
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from transcription import TRANSCRIPTION_VARIANTS, TranscriptionProcessor


class FakeTranscriptionProcessor(TranscriptionProcessor):
    def _transcribe(self, path, variant):
        return [{"start": 0.0, "end": 1.25, "text": "hello world", "words": []}], "en"


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
                    "label", "confidence", "model", "text", "language",
                },
            )
            self.assertEqual(rows[0]["filename"], "sample.wav")
            self.assertEqual(rows[0]["mode"], "speech")
            self.assertEqual(rows[0]["label"], "speech")
            self.assertEqual(rows[0]["model"], "Faster Whisper")
            self.assertTrue((result_dirs[0] / "detections.csv").exists())
            self.assertTrue((result_dirs[0] / "summary.csv").exists())
            with (result_dirs[0] / "summary.csv").open(newline="", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(summary_rows[0]["segments"], "1")
            self.assertEqual(summary_rows[0]["type"], "audio")
            self.assertIn("[0.00-1.25] hello world", (result_dirs[0] / "sample_transcript.txt").read_text())


def _fake_whisper_module():
    mod = types.ModuleType("whisper")

    class _Model:
        # openai-whisper: transcribe(audio, **decode_options) — permissive kwargs.
        def transcribe(self, audio, word_timestamps=False, fp16=True, **decode_options):
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi", "words": []}],
                    "language": "en"}

    mod.load_model = lambda name: _Model()
    return mod


def _fake_faster_whisper_module():
    mod = types.ModuleType("faster_whisper")

    class _Segment:
        start, end, text, words = 0.0, 1.0, "hi", []

    class _Info:
        language = "en"

    class WhisperModel:
        def __init__(self, name, device="cpu", compute_type="int8"):
            pass

        def transcribe(self, audio, word_timestamps=False, beam_size=5,
                       language=None, task="transcribe"):
            return iter([_Segment()]), _Info()

    mod.WhisperModel = WhisperModel
    return mod


def _fake_whisperx_module():
    mod = types.ModuleType("whisperx")

    class _Pipeline:
        # Mirrors whisperx.asr.FasterWhisperPipeline.transcribe: it takes NO
        # word_timestamps/fp16 kwargs, so passing them raises TypeError just
        # like the real library.
        def transcribe(self, audio, batch_size=None, num_workers=0, language=None,
                       task=None, chunk_size=30, print_progress=False,
                       combined_progress=False):
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


if __name__ == "__main__":
    unittest.main()
