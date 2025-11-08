"""
Sonification Engine for Channel X-NULL
Converts non-linguistic data (math, symbols, hex) into audio
"""

import numpy as np
import re
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class SonificationEngine:
    """Convert symbolic data to audio"""

    def __init__(self, config: Dict[str, Any], sample_rate: int = 22050):
        self.config = config
        self.sample_rate = sample_rate
        self.method = config.get('method', 'data_to_frequency')

    def sonify(self, text: str) -> np.ndarray:
        """
        Convert text containing symbolic data to audio

        Args:
            text: Output from LLM containing math, hex, symbols

        Returns:
            Audio data as numpy array
        """
        method = self.config.get('method', 'data_to_frequency')

        if method == 'data_to_frequency':
            return self.data_to_frequency(text)
        elif method == 'morse_beep':
            return self.morse_beep(text)
        elif method == 'extreme_tts':
            return self.extreme_tts_processing(text)
        elif method == 'hybrid':
            return self.hybrid_method(text)
        else:
            logger.warning(f"Unknown sonification method: {method}, using data_to_frequency")
            return self.data_to_frequency(text)

    def data_to_frequency(self, text: str) -> np.ndarray:
        """
        Convert data to frequency mapping

        Maps different character types to different frequencies:
        - Digits -> base freq + digit*step
        - Letters -> ASCII value mapped
        - Symbols -> special frequencies
        - Hex values -> parsed and mapped
        """
        config = self.config.get('data_mapping', {})
        digit_base = config.get('digit_freq_base', 200)
        digit_step = config.get('digit_freq_step', 80)
        letter_base = config.get('letter_freq_base', 300)
        symbol_base = config.get('symbol_freq_base', 500)
        tone_duration = config.get('tone_duration_ms', 100) / 1000.0

        audio_segments = []

        # Parse different data types
        i = 0
        while i < len(text):
            char = text[i]

            if char.isdigit():
                # Digit -> frequency
                freq = digit_base + (int(char) * digit_step)
                audio_segments.append(self._generate_tone(freq, tone_duration))

            elif char.isalpha():
                # Letter -> ASCII mapped frequency
                freq = letter_base + (ord(char.upper()) - ord('A')) * 15
                audio_segments.append(self._generate_tone(freq, tone_duration * 0.8))

            elif char in '+-*/=<>':
                # Math operators -> specific frequencies
                operator_freqs = {
                    '+': 440, '-': 330, '*': 550,
                    '/': 220, '=': 880, '<': 350, '>': 450
                }
                if char in operator_freqs:
                    audio_segments.append(
                        self._generate_tone(operator_freqs[char], tone_duration * 1.2)
                    )

            elif char in '()[]{}':
                # Brackets -> chirps
                freq = symbol_base + (ord(char) * 5)
                audio_segments.append(
                    self._generate_chirp(freq, freq + 200, tone_duration * 0.5)
                )

            elif char in '∴∵∞∫∂∇∑∏⊕⊗⊙◊◈◉●○◐◑◒◓⚛⚡⚠⚙':
                # Special symbols -> harmonic tones
                freq = 300 + (ord(char) % 500)
                audio_segments.append(
                    self._generate_harmonic_tone(freq, tone_duration, harmonics=3)
                )

            # Check for hex pattern (0xAB)
            if i < len(text) - 3 and text[i:i+2] == '0x':
                hex_match = re.match(r'0x([0-9A-Fa-f]{2})', text[i:i+4])
                if hex_match:
                    hex_value = int(hex_match.group(1), 16)
                    # Map hex value (0-255) to frequency range
                    freq = 200 + (hex_value * 3)
                    audio_segments.append(self._generate_tone(freq, tone_duration))
                    i += 3  # Skip ahead

            # Add tiny gap between sounds
            audio_segments.append(np.zeros(int(0.01 * self.sample_rate)))

            i += 1

        # Concatenate all segments
        if not audio_segments:
            # Empty, generate carrier wave
            return self._generate_tone(440, 1.0)

        audio = np.concatenate(audio_segments)

        # Apply carrier wave and noise if configured
        audio = self._apply_carrier_and_noise(audio)

        # Apply additional effects
        audio = self._apply_xnull_effects(audio)

        return audio

    def morse_beep(self, text: str) -> np.ndarray:
        """
        Convert text to morse-like beep patterns

        Different character types get different pitches
        """
        audio_segments = []

        for char in text:
            if char.isdigit():
                # Digits -> low beep
                audio_segments.append(self._generate_tone(400, 0.1))
            elif char.isalpha():
                # Letters -> mid beep
                audio_segments.append(self._generate_tone(600, 0.08))
            elif char in '+-*/=':
                # Operators -> high beep
                audio_segments.append(self._generate_tone(800, 0.12))
            elif char in '()[]{}':
                # Brackets -> chirp
                audio_segments.append(self._generate_chirp(500, 700, 0.06))
            else:
                # Space/other -> silence
                audio_segments.append(np.zeros(int(0.05 * self.sample_rate)))

            # Gap between beeps
            audio_segments.append(np.zeros(int(0.03 * self.sample_rate)))

        audio = np.concatenate(audio_segments) if audio_segments else np.zeros(int(self.sample_rate))

        # Apply carrier and effects
        audio = self._apply_carrier_and_noise(audio)
        audio = self._apply_xnull_effects(audio)

        return audio

    def extreme_tts_processing(self, text: str) -> np.ndarray:
        """
        Generate TTS then heavily process it to sound alien

        This method would require TTS integration
        For now, return data_to_frequency as fallback
        """
        logger.warning("extreme_tts not fully implemented, using data_to_frequency")
        return self.data_to_frequency(text)

    def hybrid_method(self, text: str) -> np.ndarray:
        """
        Combine multiple sonification methods
        """
        # Use different methods for different parts
        data_audio = self.data_to_frequency(text[:len(text)//2])
        morse_audio = self.morse_beep(text[len(text)//2:])

        # Crossfade
        return np.concatenate([data_audio, morse_audio])

    def _generate_tone(self, frequency: float, duration: float, amplitude: float = 0.3) -> np.ndarray:
        """Generate pure sine wave tone"""
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, False)

        # Sine wave
        tone = amplitude * np.sin(2 * np.pi * frequency * t)

        # Apply envelope to avoid clicks
        envelope = np.ones_like(tone)
        fade_samples = min(100, samples // 10)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        return (tone * envelope).astype(np.float32)

    def _generate_harmonic_tone(self, frequency: float, duration: float,
                                harmonics: int = 3, amplitude: float = 0.3) -> np.ndarray:
        """Generate tone with harmonics"""
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, False)

        tone = np.zeros(samples, dtype=np.float32)

        # Add harmonics
        for i in range(1, harmonics + 1):
            harmonic_amp = amplitude / i
            tone += harmonic_amp * np.sin(2 * np.pi * frequency * i * t)

        # Normalize
        tone = tone / np.max(np.abs(tone)) * amplitude

        # Envelope
        envelope = np.ones_like(tone)
        fade_samples = min(100, samples // 10)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        return tone * envelope

    def _generate_chirp(self, f0: float, f1: float, duration: float, amplitude: float = 0.3) -> np.ndarray:
        """Generate frequency sweep (chirp)"""
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, False)

        # Linear frequency sweep
        freq = np.linspace(f0, f1, samples)
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate

        chirp = amplitude * np.sin(phase)

        return chirp.astype(np.float32)

    def _apply_carrier_and_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add carrier wave and noise"""
        carrier_config = self.config.get('carrier_wave', {})
        noise_config = self.config.get('noise', {})

        # Add carrier wave
        if carrier_config.get('enabled', True):
            carrier_freq = carrier_config.get('frequency_hz', 1420.406)
            carrier_amp = carrier_config.get('amplitude', 0.2)

            t = np.arange(len(audio)) / self.sample_rate
            carrier = carrier_amp * np.sin(2 * np.pi * carrier_freq * t)

            audio = audio + carrier.astype(np.float32)

        # Add noise
        if noise_config.get('enabled', True):
            noise_type = noise_config.get('type', 'pink')
            noise_level = noise_config.get('mix_level', 0.4)

            noise = self._generate_noise(len(audio), noise_type, noise_level)
            audio = audio + noise

        return audio

    def _generate_noise(self, length: int, noise_type: str = 'white', amplitude: float = 0.1) -> np.ndarray:
        """Generate noise"""
        if noise_type == 'white':
            return np.random.normal(0, amplitude, length).astype(np.float32)

        elif noise_type == 'pink':
            # Simple pink noise approximation
            white = np.random.normal(0, 1, length)
            # Apply 1/f filter approximation
            from scipy import signal
            b, a = signal.butter(1, 0.1)
            pink = signal.filtfilt(b, a, white)
            return (pink / np.max(np.abs(pink)) * amplitude).astype(np.float32)

        elif noise_type == 'brown':
            white = np.random.normal(0, 0.01, length)
            brown = np.cumsum(white)
            return (brown / np.max(np.abs(brown)) * amplitude).astype(np.float32)

        else:
            return np.zeros(length, dtype=np.float32)

    def _apply_xnull_effects(self, audio: np.ndarray) -> np.ndarray:
        """Apply X-NULL specific effects"""
        effects_config = self.config.get('effects', {})

        # Bit crush
        if effects_config.get('bit_crush_bits', 16) < 16:
            bits = effects_config['bit_crush_bits']
            levels = 2 ** bits
            audio = np.round(audio * levels) / levels

        # Ring modulation
        if effects_config.get('ring_modulation_hz', 0) > 0:
            freq = effects_config['ring_modulation_hz']
            t = np.arange(len(audio)) / self.sample_rate
            modulator = np.sin(2 * np.pi * freq * t)
            audio = audio * modulator

        # Random reversals
        if effects_config.get('random_reversals', False):
            chunk_size = int(0.1 * self.sample_rate)  # 100ms chunks
            for i in range(0, len(audio) - chunk_size, chunk_size):
                if np.random.random() < 0.3:  # 30% chance
                    audio[i:i+chunk_size] = audio[i:i+chunk_size][::-1]

        # Glitch effect
        if effects_config.get('glitch_probability', 0) > 0:
            glitch_prob = effects_config['glitch_probability']
            for i in range(0, len(audio), int(0.05 * self.sample_rate)):
                if np.random.random() < glitch_prob:
                    glitch_len = np.random.randint(100, 1000)
                    if i + glitch_len < len(audio):
                        # Random glitch: repeat, reverse, or zero
                        glitch_type = np.random.choice(['repeat', 'reverse', 'zero', 'noise'])
                        if glitch_type == 'repeat':
                            audio[i:i+glitch_len] = audio[i]
                        elif glitch_type == 'reverse':
                            audio[i:i+glitch_len] = audio[i:i+glitch_len][::-1]
                        elif glitch_type == 'zero':
                            audio[i:i+glitch_len] = 0
                        elif glitch_type == 'noise':
                            audio[i:i+glitch_len] = np.random.normal(0, 0.5, glitch_len)

        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val * 0.9

        return audio


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text"""
    # Match integers, floats, scientific notation
    pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'
    matches = re.findall(pattern, text)
    return [float(m) for m in matches if m]


def extract_hex_values(text: str) -> List[int]:
    """Extract hexadecimal values from text"""
    pattern = r'0x([0-9A-Fa-f]+)'
    matches = re.findall(pattern, text)
    return [int(m, 16) for m in matches]


def parse_coordinates(text: str) -> List[tuple]:
    """Extract coordinate pairs from text"""
    # Match patterns like [RA: 14h 39m 36s, DEC: -60° 50' 02"]
    pattern = r'\[?RA:\s*(\d+h\s*\d+m\s*\d+s).*?DEC:\s*([-+]?\d+°\s*\d+\'\s*\d+")\]?'
    matches = re.findall(pattern, text)
    return matches
