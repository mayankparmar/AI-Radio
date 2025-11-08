"""
AI Talk Show Channel
Generates AI-powered radio talk shows for different alternate realities
"""

import asyncio
import os
import sys
import yaml
import numpy as np
from typing import Optional, Dict, Any, List
import logging
from queue import Queue, Empty
import sounddevice as sd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_channel import BaseChannel
from ai.llm_client import create_llm_client, BaseLLMClient
from ai.tts_engine import create_tts_engine, BaseTTSEngine
from ai.audio_effects import AudioEffectsProcessor, mix_audio
from ai.sonification import SonificationEngine

logger = logging.getLogger(__name__)


class AITalkShowChannel(BaseChannel):
    """
    AI-generated talk show channel with alternate reality themes

    This channel uses LLMs to generate content and TTS to convert to speech.
    The encoder dial (0-99) selects different alternate reality channels.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Load AI-specific configuration
        self.ai_config = self._load_ai_config()

        # Current encoder position (0-99 scale)
        self.encoder_position = 0

        # Current reality channel
        self.current_reality: Optional[Dict[str, Any]] = None

        # Initialize AI components
        try:
            self.llm_client: BaseLLMClient = create_llm_client(self.ai_config)
            self.tts_engine: BaseTTSEngine = create_tts_engine(self.ai_config)
            self.audio_processor = AudioEffectsProcessor(
                sample_rate=self.ai_config.get('audio', {}).get('sample_rate', 22050)
            )

            logger.info("AI Talk Show channel initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI components: {e}")
            logger.info("Please check ai_config.yaml and ensure all dependencies are installed")
            raise

        # Audio segment buffer (pre-generate segments)
        self.segment_buffer: Queue = Queue(maxsize=3)

        # Generation settings
        self.gen_config = self.ai_config.get('generation', {})
        self.segment_length = self.gen_config.get('segment_length_seconds', 30)
        self.buffer_segments = self.gen_config.get('buffer_segments', 2)

        # Currently playing audio
        self.current_audio: Optional[np.ndarray] = None
        self.audio_position = 0

        # Background task for content generation
        self.generation_task: Optional[asyncio.Task] = None
        self.generation_lock = asyncio.Lock()

        # History for anti-repetition
        self.topic_history: List[str] = []

        # Set initial reality based on encoder position
        self._update_current_reality()

    def _load_ai_config(self) -> Dict[str, Any]:
        """Load AI configuration from ai_config.yaml"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'ai_config.yaml'
        )

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"AI config not found: {config_path}")

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _update_current_reality(self):
        """Update current reality based on encoder position"""
        channels = self.ai_config.get('channels', [])

        for channel in channels:
            if not channel.get('enabled', True):
                continue

            pos_range = channel.get('encoder_position_range', [0, 99])
            if pos_range[0] <= self.encoder_position <= pos_range[1]:
                if self.current_reality != channel:
                    logger.info(f"Switching to reality: {channel['name']}")
                    self.current_reality = channel

                    # Clear buffer when changing realities
                    self._clear_buffer()

                return

        # Default to first enabled channel
        for channel in channels:
            if channel.get('enabled', True):
                self.current_reality = channel
                return

    def _clear_buffer(self):
        """Clear the segment buffer"""
        while not self.segment_buffer.empty():
            try:
                self.segment_buffer.get_nowait()
            except Empty:
                break

        # Reset playback position
        self.current_audio = None
        self.audio_position = 0

    async def on_encoder_A_input(self, value: int):
        """
        Encoder A: Tuning dial (changes reality channel)

        Args:
            value: Encoder change value (positive = clockwise, negative = counter-clockwise)
        """
        # Update position
        self.encoder_position = max(0, min(99, self.encoder_position + value))

        logger.info(f"[{self.name}] Encoder position: {self.encoder_position}")

        # Check if reality changed
        old_reality = self.current_reality
        self._update_current_reality()

        if old_reality != self.current_reality and self.current_reality:
            logger.info(f"[{self.name}] Now tuned to: {self.current_reality['name']}")

    async def on_encoder_B_input(self, value: int):
        """
        Encoder B: Volume control (handled by system audio)

        For now, this is a placeholder. Volume could be controlled via PulseAudio
        similar to the analogue radio channel.
        """
        logger.info(f"[{self.name}] Volume control not yet implemented")

    async def play(self):
        """
        Main playback loop

        This is called repeatedly while the channel is RUNNING.
        It manages audio playback and content generation.
        """
        if not self.current_reality:
            logger.warning("[AI Radio] No reality selected")
            return

        # Start background generation if not running
        if self.generation_task is None or self.generation_task.done():
            self.generation_task = asyncio.create_task(self._generation_loop())

        # Play buffered audio
        await self._play_buffered_audio()

    async def _generation_loop(self):
        """
        Background loop to generate audio segments

        Keeps the buffer filled with pre-generated content
        """
        while self.state == self.RUNNING:
            try:
                # Check if buffer needs filling
                if self.segment_buffer.qsize() < self.buffer_segments:
                    async with self.generation_lock:
                        await self._generate_segment()

                # Wait before checking again
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Generation loop error: {e}", exc_info=True)
                await asyncio.sleep(5)  # Wait before retrying

    async def _generate_segment(self):
        """Generate a single audio segment (LLM → TTS → Effects)"""
        if not self.current_reality:
            return

        try:
            logger.info(f"[{self.name}] Generating segment for {self.current_reality['name']}")

            # Load prompt template
            prompt = self._load_prompt_template()

            # Generate text from LLM
            logger.debug(f"[{self.name}] Calling LLM...")
            text = await self.llm_client.generate(
                prompt,
                temperature=self.gen_config.get('temperature', 0.85),
                max_tokens=self.gen_config.get('max_tokens', 600)
            )

            logger.info(f"[{self.name}] Generated text: {text[:100]}...")

            # Check if this is X-NULL channel (sonification mode)
            if self.current_reality.get('audio_mode') == 'sonification':
                audio = await self._sonify_text(text)
            else:
                # Normal TTS pipeline
                audio = await self._text_to_speech(text)

            # Add to buffer
            if not self.segment_buffer.full():
                self.segment_buffer.put(audio)
                logger.info(f"[{self.name}] Segment added to buffer (size: {self.segment_buffer.qsize()})")

        except Exception as e:
            logger.error(f"Failed to generate segment: {e}", exc_info=True)

    def _load_prompt_template(self) -> str:
        """Load and enhance prompt template for current reality"""
        template_file = self.current_reality.get('prompt_template_file')

        if not template_file:
            logger.warning("No prompt template specified")
            return "Generate a 30-second radio segment."

        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            template_file
        )

        if not os.path.exists(template_path):
            logger.warning(f"Prompt template not found: {template_path}")
            return "Generate a 30-second radio segment."

        with open(template_path, 'r') as f:
            prompt = f.read()

        # Enhance prompt with dynamic context
        prompt = self._enhance_prompt(prompt)

        return prompt

    def _enhance_prompt(self, base_prompt: str) -> str:
        """Add dynamic context to prevent repetition"""
        enhancements = []

        enhance_config = self.ai_config.get('prompt_enhancement', {})

        # Add timestamp
        if enhance_config.get('include_timestamp', True):
            from datetime import datetime
            enhancements.append(f"\nContext: Current time is {datetime.now().strftime('%H:%M')}")

        # Anti-repetition
        if enhance_config.get('anti_repetition', {}).get('enabled', True):
            if self.topic_history:
                recent_topics = ', '.join(self.topic_history[-5:])
                enhancements.append(
                    f"\nIMPORTANT: Do NOT discuss these recent topics: {recent_topics}"
                )
                enhancements.append("Choose a completely different topic or angle.")

        return base_prompt + '\n'.join(enhancements)

    async def _text_to_speech(self, text: str) -> np.ndarray:
        """Convert text to speech with effects"""
        # Generate TTS
        logger.debug(f"[{self.name}] Synthesizing speech...")
        audio = await self.tts_engine.synthesize(text)

        # Apply voice effects
        voice_config = self.current_reality.get('voice_config', {})
        if voice_config:
            logger.debug(f"[{self.name}] Applying audio effects...")
            audio = self.audio_processor.apply_effects(audio, voice_config)

        # Mix with background audio if configured
        bg_config = self.current_reality.get('background_audio', {})
        if bg_config.get('enabled', False):
            bg_file = bg_config.get('file')
            if bg_file:
                bg_file_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    bg_file
                )
                # Note: Background loading would need implementation
                # For now, skip background audio

        return audio

    async def _sonify_text(self, text: str) -> np.ndarray:
        """Convert symbolic text to audio (for X-NULL channel)"""
        sonification_config = self.current_reality.get('sonification', {})

        logger.debug(f"[{self.name}] Sonifying data...")
        sonifier = SonificationEngine(
            sonification_config,
            sample_rate=self.ai_config.get('audio', {}).get('sample_rate', 22050)
        )

        audio = sonifier.sonify(text)

        return audio

    async def _play_buffered_audio(self):
        """Play audio from buffer using sounddevice"""
        # Get next segment if we don't have one
        if self.current_audio is None or self.audio_position >= len(self.current_audio):
            if not self.segment_buffer.empty():
                try:
                    self.current_audio = self.segment_buffer.get_nowait()
                    self.audio_position = 0
                    logger.info(f"[{self.name}] Playing new segment")
                except Empty:
                    # Buffer empty, wait for generation
                    logger.debug(f"[{self.name}] Waiting for content generation...")
                    await asyncio.sleep(1)
                    return
            else:
                # No audio available
                logger.debug(f"[{self.name}] No audio in buffer")
                await asyncio.sleep(1)
                return

        # Play chunk of audio
        chunk_size = int(0.5 * self.ai_config.get('audio', {}).get('sample_rate', 22050))
        chunk_end = min(self.audio_position + chunk_size, len(self.current_audio))

        if chunk_end > self.audio_position:
            chunk = self.current_audio[self.audio_position:chunk_end]

            # Play via sounddevice
            sd.play(
                chunk,
                samplerate=self.ai_config.get('audio', {}).get('sample_rate', 22050),
                blocking=True
            )

            self.audio_position = chunk_end

        # Small delay
        await asyncio.sleep(0.1)

    async def stop(self):
        """Stop the channel and clean up"""
        logger.info(f"[{self.name}] Stopping AI Talk Show channel")

        # Cancel generation task
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()
            try:
                await self.generation_task
            except asyncio.CancelledError:
                pass

        # Stop audio playback
        sd.stop()

        # Clear buffer
        self._clear_buffer()


# Example configuration for config.yaml
"""
channels:
  - name: AIRadio
    class: channels.ai_talkshow.AITalkShowChannel
    button: R
    auto_start: false
    requires_internet: true  # If using online LLMs
"""
