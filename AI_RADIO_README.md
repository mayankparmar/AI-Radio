# AI Radio - Alternate Reality Talk Shows

AI-powered radio channel that generates talk shows from alternate realities using LLMs and text-to-speech.

## Overview

The AI Radio channel uses the tuning dial (Encoder A) to switch between different "alternate reality" radio stations. Each position on the dial (0-99) corresponds to a different reality with unique characteristics:

1. **DinoTalk FM (0-19)**: Morning show from a world where dinosaurs evolved intelligence
2. **Quantum Uncertainty Hour (20-39)**: Host exists in quantum superposition across timelines
3. **Mycelium Network News (40-59)**: Fungal consciousness broadcasting slow thoughts
4. **Nebula Broadcasts (60-79)**: Post-biological plasma entities from Andromeda
5. **Channel X-NULL (80-99)**: Non-linguistic data transmission (bizarre/experimental)

## Architecture

```
AITalkShowChannel
├── LLM Client (generates content)
│   ├── Ollama (local)
│   ├── OpenAI (cloud)
│   ├── Anthropic (cloud)
│   └── LlamaCPP (local)
├── TTS Engine (text-to-speech)
│   ├── Piper (recommended)
│   ├── Coqui TTS
│   ├── eSpeak
│   └── Festival
├── Audio Effects
│   ├── Pitch shifting
│   ├── Reverb
│   ├── Time stretching
│   ├── Chorus/tremolo
│   └── Multi-voice layering
└── Sonification (for X-NULL)
    ├── Data-to-frequency mapping
    ├── Morse-like beeps
    └── Extreme processing
```

## Installation

### 1. Install Python Dependencies

```bash
cd /home/user/AI-Radio
pip install -r requirements.txt
```

### 2. Install System Dependencies

#### For Piper TTS (Recommended)

Download Piper binary and models:

```bash
# Download Piper
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz
tar -xzf piper_amd64.tar.gz
sudo mv piper /usr/local/bin/

# Create model directory
mkdir -p models/piper

# Download a voice model (example: US English)
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx \
  -O models/piper/en_US-lessac-medium.onnx

wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json \
  -O models/piper/en_US-lessac-medium.onnx.json
```

More voices available at: https://github.com/rhasspy/piper/blob/master/VOICES.md

#### For Ollama (Local LLM - Recommended)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (llama3.2 3B is fast and good)
ollama pull llama3.2:3b

# Verify it's running
ollama list
```

#### Alternative: eSpeak (Lightweight TTS)

```bash
sudo apt-get install espeak
```

### 3. Configure LLM Provider

Edit `ai_config.yaml`:

**For local Ollama (recommended):**
```yaml
llm:
  provider: "ollama"
  providers:
    ollama:
      base_url: "http://localhost:11434"
      model: "llama3.2:3b"
```

**For OpenAI (requires API key):**
```yaml
llm:
  provider: "openai"
  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      model: "gpt-4"
```

Set environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**For Anthropic Claude (requires API key):**
```yaml
llm:
  provider: "anthropic"
  providers:
    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
      model: "claude-3-5-sonnet-20241022"
```

Set environment variable:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 4. Configure TTS Provider

Edit `ai_config.yaml`:

**For Piper (recommended):**
```yaml
tts:
  provider: "piper"
  providers:
    piper:
      model_path: "./models/piper/en_US-lessac-medium.onnx"
```

**For eSpeak (fallback):**
```yaml
tts:
  provider: "espeak"
```

### 5. Add to Main Configuration

Edit `config.yaml` and add the AI Radio channel:

```yaml
channels:
  # ... existing channels ...

  - name: AIRadio
    class: channels.ai_talkshow.AITalkShowChannel
    button: R  # Press 'R' to activate
    auto_start: false
    requires_internet: false  # Set true if using online LLMs
```

## Usage

### Starting the Radio

```bash
cd /home/user/AI-Radio
python main.py
```

### Controls

- **Press 'R'**: Activate AI Radio channel
- **Press '+'**: Turn dial clockwise (higher reality number)
- **Press '-'**: Turn dial counter-clockwise (lower reality number)
- **Press 'Q'**: Quit

### Encoder Positions

| Position | Reality | Description |
|----------|---------|-------------|
| 0-19 | DinoTalk FM | Dinosaurs discuss modern dino-society |
| 20-39 | Quantum Hour | Host collapses between quantum timelines |
| 40-59 | Mycelium News | Slow fungal consciousness broadcasts |
| 60-79 | Nebula Broadcasts | Post-biological plasma entities |
| 80-99 | Channel X-NULL | Non-linguistic data/beeps/bizarre |

## Customization

### Creating New Realities

1. **Create a prompt template** in `prompts/your_reality.txt`:

```
You are hosting a radio show from [describe your reality].

STYLE GUIDELINES:
- Keep it 30 seconds (75-90 words)
- [Specific instructions]

GENERATE SEGMENT NOW:
```

2. **Add to `ai_config.yaml`**:

```yaml
channels:
  - name: "Your Reality"
    enabled: true
    encoder_position_range: [40, 59]  # Choose range
    prompt_template_file: "prompts/your_reality.txt"
    voice_config:
      pitch_shift_semitones: 0
      reverb_amount: 0.3
      speed_factor: 1.0
```

### Adjusting Voice Effects

Each reality can have custom voice effects:

```yaml
voice_config:
  pitch_shift_semitones: -5    # Negative = deeper, positive = higher
  reverb_amount: 0.3           # 0.0 to 1.0
  echo_delay_ms: 300           # Milliseconds
  speed_factor: 0.9            # <1 slower, >1 faster
  chorus_enabled: true         # Multi-voice effect
  tremolo_rate_hz: 3.0         # Voice wavering speed
  tremolo_depth: 0.3           # Wavering intensity
  low_pass_filter_hz: 2000     # Muffle high frequencies
```

### Tuning Generation Settings

In `ai_config.yaml`:

```yaml
generation:
  segment_length_seconds: 30   # Length of each segment
  buffer_segments: 2           # Pre-generate this many segments
  temperature: 0.85            # LLM creativity (0.0-2.0)
  max_tokens: 600              # Maximum response length
```

## Channel X-NULL: The Bizarre Channel

This channel uses **sonification** instead of normal TTS. It converts the LLM's output (mathematical expressions, hex data, symbols) directly into audio.

### Sonification Methods

Edit `ai_config.yaml` under Channel X-NULL:

```yaml
sonification:
  method: "data_to_frequency"  # Options below
```

**Available methods:**

1. **data_to_frequency**: Maps different character types to frequencies
   - Digits → tones at digit-specific frequencies
   - Letters → ASCII-mapped frequencies
   - Math operators → specific beeps
   - Hex values → frequency sweeps
   - Symbols → harmonic tones

2. **morse_beep**: Converts to morse-like beep patterns
   - Different pitches for different character types

3. **hybrid**: Combines multiple methods

### X-NULL Effects

```yaml
sonification:
  carrier_wave:
    enabled: true
    frequency_hz: 1420.406  # Hydrogen line (sci-fi number station)
    amplitude: 0.2

  noise:
    enabled: true
    type: "pink"          # white, pink, brown
    mix_level: 0.4

  effects:
    bit_crush_bits: 8              # Reduce to 8-bit quality
    ring_modulation_hz: 440        # Ring mod frequency
    random_reversals: true         # Randomly reverse segments
    glitch_probability: 0.15       # 15% glitch chance
```

## Troubleshooting

### "LLM generation failed"

**Ollama not running:**
```bash
# Start Ollama service
ollama serve

# In another terminal, verify
ollama list
```

**Model not downloaded:**
```bash
ollama pull llama3.2:3b
```

### "Piper synthesis failed"

**Piper not installed:**
```bash
which piper
# If nothing, reinstall Piper (see installation section)
```

**Model not found:**
```bash
ls -l models/piper/
# Should show .onnx and .onnx.json files
# If missing, re-download (see installation section)
```

**Fallback to eSpeak:**

Edit `ai_config.yaml`:
```yaml
tts:
  provider: "espeak"  # Robotic but works
```

### "No audio output"

**Check sounddevice:**
```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

**Verify ALSA/PulseAudio:**
```bash
aplay -l  # List playback devices
pactl list sinks  # PulseAudio sinks
```

### "Generation is slow"

**Use faster LLM:**
- Ollama with llama3.2:3b (recommended)
- Smaller models are faster

**Reduce max_tokens:**
```yaml
generation:
  max_tokens: 300  # Shorter responses
```

**Use faster TTS:**
```yaml
tts:
  provider: "espeak"  # Much faster than Piper
```

### "Content is repetitive"

**Enable anti-repetition:**
```yaml
prompt_enhancement:
  anti_repetition:
    enabled: true
    avoid_recent_count: 10  # Increase this
```

**Increase temperature:**
```yaml
generation:
  temperature: 1.0  # More creative (was 0.85)
```

## Performance Tips

### For Raspberry Pi

1. **Use local models:**
   - Ollama with llama3.2:1b (smallest, fastest)
   - eSpeak for TTS (very light)

2. **Reduce buffer:**
```yaml
generation:
  buffer_segments: 1  # Less memory usage
```

3. **Lower sample rate:**
```yaml
audio:
  sample_rate: 16000  # Lower quality but faster
```

### For Desktop

1. **Use better models:**
   - Ollama with llama3:8b or llama3:70b
   - Piper with high-quality voices

2. **Pre-buffer more:**
```yaml
generation:
  buffer_segments: 3  # Smoother playback
```

## File Structure

```
AI-Radio/
├── ai_config.yaml              # AI-specific configuration
├── config.yaml                 # Main radio configuration
├── requirements.txt            # Python dependencies
├── prompts/                    # Prompt templates
│   ├── dino_talk.txt
│   ├── quantum.txt
│   ├── mycelium.txt
│   ├── nebula.txt
│   └── xnull.txt
├── ai/                         # AI modules
│   ├── llm_client.py          # LLM abstraction
│   ├── tts_engine.py          # TTS abstraction
│   ├── audio_effects.py       # Audio processing
│   └── sonification.py        # X-NULL sonification
├── channels/
│   └── ai_talkshow.py         # Main AI channel class
└── models/                     # TTS models (create this)
    └── piper/
        ├── en_US-lessac-medium.onnx
        └── en_US-lessac-medium.onnx.json
```

## Advanced: Custom LLM Provider

To add a new LLM provider, edit `ai/llm_client.py`:

```python
class MyCustomClient(BaseLLMClient):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Your initialization

    async def generate(self, prompt: str, **kwargs) -> str:
        # Your generation logic
        return "generated text"

    def is_available(self) -> bool:
        return True

# Register in factory
LLMClientFactory._clients['mycustom'] = MyCustomClient
```

Then in `ai_config.yaml`:
```yaml
llm:
  provider: "mycustom"
  providers:
    mycustom:
      # Your config
```

## Credits

- **LLM**: Ollama, OpenAI, Anthropic
- **TTS**: Piper (Rhasspy), Coqui TTS, eSpeak
- **Audio**: librosa, pedalboard, sounddevice
- **Framework**: AI-Radio by mayankparmar

## License

MIT License (same as main project)
