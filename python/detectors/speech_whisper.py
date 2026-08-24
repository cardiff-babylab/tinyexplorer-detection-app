"""Registry entry for the optional Whisper transcription backends."""
from .base import AudioDetector, Detection, register_detector


@register_detector("speech_whisper")
class WhisperTranscriptionDetector(AudioDetector):
    name = "speech"
    mode = "speech"
    variants = ["Whisper (OpenAI)", "Faster Whisper", "WhisperX"]

    def load(self, weights_dir: str, variant=None) -> bool:
        self._loaded_variant = variant or self.variants[0]
        return True

    def detect_audio(self, waveform, sample_rate, confidence_threshold):
        raise NotImplementedError("Speech uses the transcription batch pipeline")
