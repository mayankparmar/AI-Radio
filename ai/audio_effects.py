"""
Audio Effects Pipeline
Applies various audio effects to generated speech
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AudioEffectsProcessor:
    """Process audio with various effects"""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def apply_effects(self, audio: np.ndarray, effects_config: Dict[str, Any]) -> np.ndarray:
        """
        Apply configured effects to audio

        Args:
            audio: Input audio as numpy array (int16 or float32)
            effects_config: Dictionary of effect settings from channel config

        Returns:
            Processed audio (same dtype as input)
        """
        if len(audio) == 0:
            return audio

        # Convert to float32 for processing
        original_dtype = audio.dtype
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)

        # Apply effects in order
        if effects_config.get('pitch_shift_semitones', 0) != 0:
            audio_float = self.pitch_shift(
                audio_float,
                effects_config['pitch_shift_semitones']
            )

        if effects_config.get('speed_factor', 1.0) != 1.0:
            audio_float = self.time_stretch(
                audio_float,
                effects_config['speed_factor']
            )

        if effects_config.get('reverb_amount', 0) > 0:
            audio_float = self.add_reverb(
                audio_float,
                effects_config['reverb_amount']
            )

        if effects_config.get('echo_delay_ms', 0) > 0:
            audio_float = self.add_echo(
                audio_float,
                effects_config['echo_delay_ms']
            )

        if effects_config.get('chorus_enabled', False):
            audio_float = self.add_chorus(audio_float)

        if effects_config.get('tremolo_rate_hz', 0) > 0:
            audio_float = self.add_tremolo(
                audio_float,
                effects_config.get('tremolo_rate_hz', 3.0),
                effects_config.get('tremolo_depth', 0.3)
            )

        if effects_config.get('low_pass_filter_hz'):
            audio_float = self.low_pass_filter(
                audio_float,
                effects_config['low_pass_filter_hz']
            )

        if effects_config.get('multi_voice_layers', 0) > 1:
            audio_float = self.multi_voice_layer(
                audio_float,
                effects_config.get('multi_voice_layers', 3),
                effects_config.get('layer_pitch_offsets', [0, 2, -2])
            )

        # Convert back to original dtype
        if original_dtype == np.int16:
            audio_float = np.clip(audio_float, -1.0, 1.0)
            return (audio_float * 32767).astype(np.int16)
        else:
            return audio_float

    def pitch_shift(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        """Shift pitch by semitones using librosa"""
        try:
            import librosa
            return librosa.effects.pitch_shift(
                audio,
                sr=self.sample_rate,
                n_steps=semitones
            )
        except ImportError:
            logger.warning("librosa not available, skipping pitch shift")
            return audio

    def time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """
        Time stretch audio (change speed without changing pitch)
        rate < 1.0: slower, rate > 1.0: faster
        """
        try:
            import librosa
            # librosa's rate is inverse (rate=2.0 means 2x slower)
            return librosa.effects.time_stretch(audio, rate=1.0/rate)
        except ImportError:
            logger.warning("librosa not available, skipping time stretch")
            return audio

    def add_reverb(self, audio: np.ndarray, amount: float = 0.3) -> np.ndarray:
        """
        Add reverb effect (simple algorithmic reverb)

        Args:
            audio: Input audio
            amount: Reverb amount (0.0 to 1.0)
        """
        # Simple comb filter reverb
        delay_samples = [
            int(0.0297 * self.sample_rate),  # ~30ms
            int(0.0371 * self.sample_rate),  # ~37ms
            int(0.0411 * self.sample_rate),  # ~41ms
            int(0.0437 * self.sample_rate),  # ~44ms
        ]

        reverb = np.zeros_like(audio)

        for delay in delay_samples:
            # Create delayed version
            delayed = np.zeros_like(audio)
            delayed[delay:] = audio[:-delay]

            # Apply decay
            decay = 0.5 * amount
            reverb += delayed * decay

        # Mix with original
        return audio * (1.0 - amount * 0.5) + reverb * amount

    def add_echo(self, audio: np.ndarray, delay_ms: int, decay: float = 0.4) -> np.ndarray:
        """Add echo effect"""
        delay_samples = int(delay_ms * self.sample_rate / 1000)

        echo = np.zeros_like(audio)
        if delay_samples < len(audio):
            echo[delay_samples:] = audio[:-delay_samples] * decay

        return audio + echo

    def add_chorus(self, audio: np.ndarray, voices: int = 3) -> np.ndarray:
        """
        Add chorus effect (layered slightly detuned voices)
        """
        chorus_audio = audio.copy()

        # Add detuned copies
        detune_cents = [-10, 10, -15, 15]  # Slight detuning

        for i in range(voices - 1):
            if i < len(detune_cents):
                semitones = detune_cents[i] / 100.0  # Convert cents to semitones
                try:
                    import librosa
                    detuned = librosa.effects.pitch_shift(
                        audio,
                        sr=self.sample_rate,
                        n_steps=semitones
                    )
                    # Slight delay for depth
                    delay_samples = int(0.015 * self.sample_rate * (i + 1))
                    delayed = np.zeros_like(detuned)
                    if delay_samples < len(detuned):
                        delayed[delay_samples:] = detuned[:-delay_samples]
                    chorus_audio += delayed * 0.5
                except ImportError:
                    # Fallback: just add slight delays without pitch shift
                    delay_samples = int(0.020 * self.sample_rate * (i + 1))
                    delayed = np.zeros_like(audio)
                    if delay_samples < len(audio):
                        delayed[delay_samples:] = audio[:-delay_samples]
                    chorus_audio += delayed * 0.3

        # Normalize
        return chorus_audio / (voices * 0.5)

    def add_tremolo(self, audio: np.ndarray, rate_hz: float = 3.0, depth: float = 0.3) -> np.ndarray:
        """
        Add tremolo effect (amplitude modulation)

        Args:
            rate_hz: Modulation frequency in Hz
            depth: Modulation depth (0.0 to 1.0)
        """
        # Create modulation signal
        t = np.arange(len(audio)) / self.sample_rate
        modulation = 1.0 - depth * (0.5 + 0.5 * np.sin(2 * np.pi * rate_hz * t))

        return audio * modulation

    def low_pass_filter(self, audio: np.ndarray, cutoff_hz: float) -> np.ndarray:
        """
        Apply low-pass filter (removes high frequencies)

        Args:
            cutoff_hz: Cutoff frequency in Hz
        """
        try:
            from scipy import signal

            # Design butterworth low-pass filter
            nyquist = self.sample_rate / 2
            normalized_cutoff = cutoff_hz / nyquist

            if normalized_cutoff >= 1.0:
                return audio  # No filtering needed

            b, a = signal.butter(4, normalized_cutoff, btype='low')
            return signal.filtfilt(b, a, audio)

        except ImportError:
            logger.warning("scipy not available, skipping low-pass filter")
            return audio

    def multi_voice_layer(self, audio: np.ndarray, num_layers: int = 3,
                          pitch_offsets: list = [0, 2, -2]) -> np.ndarray:
        """
        Layer multiple pitch-shifted copies for ethereal effect

        Args:
            num_layers: Number of voice layers
            pitch_offsets: Semitone offsets for each layer
        """
        try:
            import librosa

            layered = np.zeros_like(audio)

            for i in range(min(num_layers, len(pitch_offsets))):
                if pitch_offsets[i] == 0:
                    layered += audio
                else:
                    shifted = librosa.effects.pitch_shift(
                        audio,
                        sr=self.sample_rate,
                        n_steps=pitch_offsets[i]
                    )
                    layered += shifted

            # Normalize
            return layered / num_layers

        except ImportError:
            logger.warning("librosa not available, skipping multi-voice layering")
            return audio

    def add_noise(self, audio: np.ndarray, noise_type: str = 'white',
                  mix_level: float = 0.1) -> np.ndarray:
        """
        Add noise to audio

        Args:
            noise_type: 'white', 'pink', 'brown'
            mix_level: Noise amplitude (0.0 to 1.0)
        """
        if noise_type == 'white':
            noise = np.random.normal(0, mix_level, len(audio))
        elif noise_type == 'pink':
            # Simple pink noise approximation
            white = np.random.normal(0, 1, len(audio))
            try:
                from scipy import signal
                # Low-pass filter to approximate pink noise
                b, a = signal.butter(1, 0.1)
                noise = signal.filtfilt(b, a, white) * mix_level
            except ImportError:
                noise = white * mix_level  # Fallback to white
        elif noise_type == 'brown':
            # Brown noise (cumulative sum of white noise)
            white = np.random.normal(0, 0.01, len(audio))
            noise = np.cumsum(white)
            noise = noise / np.max(np.abs(noise)) * mix_level
        else:
            noise = np.zeros_like(audio)

        return audio + noise.astype(audio.dtype)

    def bit_crush(self, audio: np.ndarray, bits: int = 8) -> np.ndarray:
        """
        Reduce bit depth for lo-fi effect

        Args:
            bits: Target bit depth (1-16)
        """
        # Quantize audio
        levels = 2 ** bits
        quantized = np.round(audio * levels) / levels

        return quantized

    def ring_modulate(self, audio: np.ndarray, freq_hz: float = 440) -> np.ndarray:
        """
        Ring modulation effect (multiply by sine wave)

        Args:
            freq_hz: Modulation frequency
        """
        t = np.arange(len(audio)) / self.sample_rate
        modulator = np.sin(2 * np.pi * freq_hz * t)

        return audio * modulator


def load_background_audio(file_path: str, sample_rate: int, duration: float) -> Optional[np.ndarray]:
    """
    Load background audio file

    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        duration: Desired duration in seconds

    Returns:
        Audio data or None if file doesn't exist
    """
    import os
    if not os.path.exists(file_path):
        logger.warning(f"Background audio not found: {file_path}")
        return None

    try:
        import soundfile as sf
        import librosa

        audio, sr = sf.read(file_path)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

        # Loop or trim to desired duration
        target_samples = int(duration * sample_rate)
        if len(audio) < target_samples:
            # Loop audio
            repeats = int(np.ceil(target_samples / len(audio)))
            audio = np.tile(audio, repeats)[:target_samples]
        else:
            audio = audio[:target_samples]

        return audio

    except Exception as e:
        logger.error(f"Failed to load background audio: {e}")
        return None


def mix_audio(foreground: np.ndarray, background: np.ndarray, bg_volume: float = 0.1) -> np.ndarray:
    """
    Mix foreground and background audio

    Args:
        foreground: Main audio
        background: Background audio
        bg_volume: Background volume level (0.0 to 1.0)

    Returns:
        Mixed audio
    """
    # Ensure same length
    if background is None or len(background) == 0:
        return foreground

    min_len = min(len(foreground), len(background))
    mixed = foreground[:min_len] + (background[:min_len] * bg_volume)

    # If foreground is longer, append the rest
    if len(foreground) > min_len:
        mixed = np.concatenate([mixed, foreground[min_len:]])

    return mixed
