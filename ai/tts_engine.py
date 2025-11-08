"""
Text-to-Speech Engine Abstraction Layer
Supports multiple TTS providers: Piper, Coqui, eSpeak, Festival
"""

import asyncio
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class BaseTTSEngine(ABC):
    """Abstract base class for TTS engines"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_rate = config.get('sample_rate', 22050)

    @abstractmethod
    async def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize speech from text

        Args:
            text: Text to convert to speech

        Returns:
            Audio data as numpy array (int16 format)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the TTS engine is available"""
        pass

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to int16 range"""
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            # Normalize float audio to -1.0 to 1.0 range
            audio = np.clip(audio, -1.0, 1.0)
            # Convert to int16
            audio = (audio * 32767).astype(np.int16)
        return audio


class PiperTTSEngine(BaseTTSEngine):
    """Piper TTS engine (recommended - fast, local, high quality)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.model_path = config.get('model_path', './models/piper/en_US-lessac-medium.onnx')
        self.config_path = config.get('config_path', self.model_path + '.json')
        self.speaker_id = config.get('speaker_id', 0)
        self.length_scale = config.get('length_scale', 1.0)  # Speed
        self.noise_scale = config.get('noise_scale', 0.667)
        self.noise_w = config.get('noise_w', 0.8)

        # Check if model exists
        if not os.path.exists(self.model_path):
            logger.warning(f"Piper model not found at {self.model_path}")
            logger.info("Download models from: https://github.com/rhasspy/piper/releases")

        try:
            # Import piper_tts if using Python wrapper
            # For now, we'll use command-line piper which is more reliable
            self.use_cli = True
            logger.info(f"Piper TTS initialized with model {self.model_path}")
        except ImportError:
            logger.warning("piper-tts not installed, will try CLI fallback")
            self.use_cli = True

    def is_available(self) -> bool:
        """Check if Piper binary or model is available"""
        # Check if piper binary exists
        result = subprocess.run(['which', 'piper'], capture_output=True)
        has_binary = result.returncode == 0

        # Check if model exists
        has_model = os.path.exists(self.model_path)

        if not has_binary:
            logger.warning("Piper binary not found. Install from: https://github.com/rhasspy/piper")

        return has_binary and has_model

    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize speech using Piper"""
        try:
            # Use CLI piper for better compatibility
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                output_path = tmp_file.name

            # Run piper command
            cmd = [
                'piper',
                '--model', self.model_path,
                '--output_file', output_path,
                '--length_scale', str(self.length_scale),
                '--noise_scale', str(self.noise_scale),
                '--noise_w', str(self.noise_w),
            ]

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            process = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    input=text.encode('utf-8'),
                    capture_output=True,
                    timeout=30
                )
            )

            if process.returncode != 0:
                logger.error(f"Piper failed: {process.stderr.decode()}")
                raise RuntimeError("Piper synthesis failed")

            # Load the generated audio
            import soundfile as sf
            audio, sr = sf.read(output_path)

            # Clean up temp file
            os.unlink(output_path)

            # Convert to expected format
            audio = self._normalize_audio(audio)

            # Resample if needed
            if sr != self.sample_rate:
                import librosa
                audio = librosa.resample(
                    audio.astype(np.float32) / 32768.0,
                    orig_sr=sr,
                    target_sr=self.sample_rate
                )
                audio = self._normalize_audio(audio)

            return audio

        except Exception as e:
            logger.error(f"Piper synthesis failed: {e}")
            raise


class CoquiTTSEngine(BaseTTSEngine):
    """Coqui TTS engine (more voices, slower)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.model_name = config.get('model_name', 'tts_models/en/ljspeech/tacotron2-DDC')
        self.vocoder_name = config.get('vocoder_name', None)

        try:
            from TTS.api import TTS
            self.tts = TTS(model_name=self.model_name, vocoder_name=self.vocoder_name)
            logger.info(f"Coqui TTS initialized with {self.model_name}")
        except ImportError:
            logger.error("TTS package not installed. Install with: pip install TTS")
            raise

    def is_available(self) -> bool:
        """Check if Coqui is initialized"""
        return hasattr(self, 'tts')

    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize speech using Coqui TTS"""
        try:
            # Coqui is CPU-intensive, run in executor
            loop = asyncio.get_event_loop()

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                output_path = tmp_file.name

            # Generate audio
            await loop.run_in_executor(
                None,
                lambda: self.tts.tts_to_file(text=text, file_path=output_path)
            )

            # Load audio
            import soundfile as sf
            audio, sr = sf.read(output_path)

            # Clean up
            os.unlink(output_path)

            # Convert and resample
            audio = self._normalize_audio(audio)

            if sr != self.sample_rate:
                import librosa
                audio = librosa.resample(
                    audio.astype(np.float32) / 32768.0,
                    orig_sr=sr,
                    target_sr=self.sample_rate
                )
                audio = self._normalize_audio(audio)

            return audio

        except Exception as e:
            logger.error(f"Coqui synthesis failed: {e}")
            raise


class ESpeakEngine(BaseTTSEngine):
    """eSpeak TTS engine (lightweight, robotic but fast)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.voice = config.get('voice', 'en-us')
        self.speed = config.get('speed', 175)  # words per minute
        self.pitch = config.get('pitch', 50)  # 0-99

    def is_available(self) -> bool:
        """Check if espeak is installed"""
        result = subprocess.run(['which', 'espeak'], capture_output=True)
        return result.returncode == 0

    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize speech using eSpeak"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                output_path = tmp_file.name

            # Run espeak command
            cmd = [
                'espeak',
                '-v', self.voice,
                '-s', str(self.speed),
                '-p', str(self.pitch),
                '-w', output_path,
                text
            ]

            loop = asyncio.get_event_loop()
            process = await loop.run_in_executor(
                None,
                lambda: subprocess.run(cmd, capture_output=True, timeout=30)
            )

            if process.returncode != 0:
                logger.error(f"eSpeak failed: {process.stderr.decode()}")
                raise RuntimeError("eSpeak synthesis failed")

            # Load audio
            import soundfile as sf
            audio, sr = sf.read(output_path)

            # Clean up
            os.unlink(output_path)

            # Convert and resample
            audio = self._normalize_audio(audio)

            if sr != self.sample_rate:
                import librosa
                audio = librosa.resample(
                    audio.astype(np.float32) / 32768.0,
                    orig_sr=sr,
                    target_sr=self.sample_rate
                )
                audio = self._normalize_audio(audio)

            return audio

        except Exception as e:
            logger.error(f"eSpeak synthesis failed: {e}")
            raise


class FestivalEngine(BaseTTSEngine):
    """Festival TTS engine (classic, Unix-style)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.voice = config.get('voice', 'cmu_us_slt_arctic_hts')

    def is_available(self) -> bool:
        """Check if festival is installed"""
        result = subprocess.run(['which', 'festival'], capture_output=True)
        return result.returncode == 0

    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize speech using Festival"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                output_path = tmp_file.name

            # Create festival script
            festival_script = f'(voice_{self.voice})\n(utt.save.wave (SayText "{text}") "{output_path}")\n'

            loop = asyncio.get_event_loop()
            process = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ['festival'],
                    input=festival_script.encode('utf-8'),
                    capture_output=True,
                    timeout=30
                )
            )

            if not os.path.exists(output_path):
                raise RuntimeError("Festival failed to generate audio")

            # Load audio
            import soundfile as sf
            audio, sr = sf.read(output_path)

            # Clean up
            os.unlink(output_path)

            # Convert and resample
            audio = self._normalize_audio(audio)

            if sr != self.sample_rate:
                import librosa
                audio = librosa.resample(
                    audio.astype(np.float32) / 32768.0,
                    orig_sr=sr,
                    target_sr=self.sample_rate
                )
                audio = self._normalize_audio(audio)

            return audio

        except Exception as e:
            logger.error(f"Festival synthesis failed: {e}")
            raise


class TTSEngineFactory:
    """Factory to create appropriate TTS engine based on configuration"""

    _engines = {
        'piper': PiperTTSEngine,
        'coqui': CoquiTTSEngine,
        'espeak': ESpeakEngine,
        'festival': FestivalEngine,
    }

    @staticmethod
    def create_engine(tts_config: Dict[str, Any]) -> BaseTTSEngine:
        """
        Create a TTS engine based on configuration

        Args:
            tts_config: TTS configuration from ai_config.yaml

        Returns:
            Initialized TTS engine
        """
        provider = tts_config.get('provider', 'piper').lower()

        if provider not in TTSEngineFactory._engines:
            raise ValueError(f"Unknown TTS provider: {provider}")

        # Get provider-specific config
        provider_config = tts_config.get('providers', {}).get(provider, {})

        # Add global audio config
        provider_config['sample_rate'] = tts_config.get('sample_rate', 22050)

        engine_class = TTSEngineFactory._engines[provider]
        engine = engine_class(provider_config)

        # Verify availability
        if not engine.is_available():
            logger.warning(f"{provider} TTS engine created but may not be available")

        return engine


# Convenience function
def create_tts_engine(config: Dict[str, Any]) -> BaseTTSEngine:
    """Create TTS engine from configuration"""
    # Merge TTS and audio config
    tts_config = config.get('tts', {})
    audio_config = config.get('audio', {})
    tts_config['sample_rate'] = audio_config.get('sample_rate', 22050)

    return TTSEngineFactory.create_engine(tts_config)
