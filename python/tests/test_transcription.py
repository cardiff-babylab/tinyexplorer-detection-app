import csv
import tempfile
import unittest
from pathlib import Path

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
            self.assertIn("[0.00-1.25] hello world", (result_dirs[0] / "sample_transcript.txt").read_text())


if __name__ == "__main__":
    unittest.main()
