# AI-Radio

Vintage Radio Python Framework with AI-Generated Talk Shows from Alternate Realities

## Overview

AI-Radio is a modular Python framework for creating a vintage radio experience on Raspberry Pi hardware. It supports multiple "channels" that can be switched using buttons or a rotary encoder, just like tuning a real vintage radio.

**New in this version**: AI-powered talk show channels that generate unique content from alternate realities using Large Language Models (LLMs) and Text-to-Speech (TTS).

## Features

### Traditional Channels
- **Internet Radio**: Stream online radio stations
- **Analogue Radio**: Internet radio with pseudo-analogue tuning and volume control
- **Local MP3 Player**: Play local audio files
- **Morse Code Generator**: Generate and play morse code

### AI Talk Show Channels (NEW)
Generate live talk shows from **5 alternate realities**, selectable via tuning dial (0-99):

1. **DinoTalk FM (0-19)** 🦖 - Morning show from a world where dinosaurs evolved intelligence
2. **Quantum Uncertainty Hour (20-39)** ⚛️ - Host exists in quantum superposition across timelines
3. **Mycelium Network News (40-59)** 🍄 - Fungal consciousness broadcasting slow thoughts
4. **Nebula Broadcasts (60-79)** 🌌 - Post-biological plasma entities from Andromeda
5. **Channel X-NULL (80-99)** ⚠️ - Non-linguistic data transmission (bizarre experimental channel)

## Architecture

The framework uses a channel-based architecture where each radio mode is a discrete channel:

```
main.py → Supervisor → Channels
                    ├─ Input Handlers (Keyboard, GPIO Encoders)
                    └─ Channel Instances:
                        ├─ InternetRadioChannel
                        ├─ LocalMP3Channel
                        ├─ MorseCode
                        └─ AITalkShowChannel (NEW)
                            ├─ LLM Client (Ollama, OpenAI, Anthropic)
                            ├─ TTS Engine (Piper, Coqui, eSpeak)
                            ├─ Audio Effects Pipeline
                            └─ Sonification Engine
```

## Requirements

### System Requirements
- **Python**: 3.11 or greater
- **OS**: Linux (tested on Raspberry Pi 4 and Pi 5)
- **Hardware** (optional): Rotary encoders for physical dial control

### Dependencies
- **FFMPEG**: For audio playback
- **PulseAudio**: For analogue radio channel mixing (optional)
- **Python packages**: See `requirements.txt`

### AI Radio Additional Requirements
- **LLM Provider** (choose one):
  - Ollama (local, recommended for Raspberry Pi)
  - OpenAI API (cloud)
  - Anthropic API (cloud)
  - LlamaCPP (local)

- **TTS Engine** (choose one):
  - Piper (local, recommended - high quality)
  - Coqui TTS (local, more voices but slower)
  - eSpeak (local, lightweight but robotic)
  - Festival (local, classic Unix TTS)

## Installation

### Quick Start (AI Radio)

```bash
# Clone the repository
git clone https://github.com/mayankparmar/AI-Radio.git
cd AI-Radio

# Run the automated setup script
./setup_ai_radio.sh
```

The setup script will:
- Install Python dependencies
- Check for and optionally install Ollama (local LLM)
- Check for and optionally install Piper TTS
- Download voice models
- Verify configuration

### Manual Installation

1. **Set up Python virtual environment**:
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFMPEG**:
   ```bash
   sudo apt-get install ffmpeg
   ```

4. **Install Ollama (for local AI)**:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull llama3.2:3b
   ```

5. **Install Piper TTS**:
   ```bash
   # Download Piper
   wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz
   tar -xzf piper_amd64.tar.gz
   sudo mv piper/piper /usr/local/bin/

   # Download voice model
   mkdir -p models/piper
   wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx \
     -O models/piper/en_US-lessac-medium.onnx
   wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json \
     -O models/piper/en_US-lessac-medium.onnx.json
   ```

## Configuration

### Main Configuration (`config.yaml`)

Defines available channels and hardware mapping:

```yaml
channels:
  - name: AnalogueRadio
    class: channels.internet_radio_analogue.InternetRadioChannel
    button: A

  - name: LocalTunes
    class: channels.local_mp3_player.LocalMP3Channel
    button: B

  - name: AIRadio
    class: channels.ai_talkshow.AITalkShowChannel
    button: R
    requires_internet: false  # Set true if using cloud LLMs
```

### AI Configuration (`ai_config.yaml`)

Controls AI-specific settings:

```yaml
llm:
  provider: "ollama"  # or "openai", "anthropic", "llamacpp"

tts:
  provider: "piper"   # or "coqui", "espeak", "festival"

channels:
  - name: "DinoTalk FM"
    encoder_position_range: [0, 19]
    voice_config:
      pitch_shift_semitones: -5
      reverb_amount: 0.3
```

For detailed AI configuration, see [AI_RADIO_README.md](AI_RADIO_README.md)

## Usage

### Starting the Radio

```bash
cd AI-Radio
source myenv/bin/activate  # Activate virtual environment
python main.py
```

### Keyboard Controls

| Key | Function |
|-----|----------|
| `A` | Switch to Analogue Radio channel |
| `B` | Switch to Local Tunes channel |
| `C` | Switch to Classic Radio channel |
| `M` | Switch to Morse Code channel |
| `R` | Switch to AI Radio channel |
| `+` | Encoder A clockwise (tune up / next track) |
| `-` | Encoder A counter-clockwise (tune down) |
| `.` | Encoder B clockwise (volume up) |
| `,` | Encoder B counter-clockwise (volume down) |
| `Q` | Quit application |

### Using AI Radio

1. **Start the radio**: `python main.py`
2. **Press 'R'**: Activate AI Radio channel
3. **Press '+' or '-'**: Tune the dial (0-99 position)
4. **Listen**: Each position range corresponds to a different reality:
   - 0-19: DinoTalk FM
   - 20-39: Quantum Uncertainty Hour
   - 40-59: Mycelium Network News
   - 60-79: Nebula Broadcasts
   - 80-99: Channel X-NULL (bizarre/experimental)

The AI will generate unique talk show content for each reality, with distinct voice characteristics and themes.

### Hardware GPIO Encoders

If using physical rotary encoders, configure pin mappings in `config.yaml`:

```yaml
pin_mapping:
  enc_A_clk: 17  # GPIO17 (Physical Pin 11)
  enc_A_dir: 18  # GPIO18 (Physical Pin 12)
  enc_A_sw:  27  # GPIO27 (Physical Pin 13)
```

## Project Structure

```
AI-Radio/
├── main.py                    # Entry point
├── supervisor.py              # Core orchestrator
├── base_channel.py           # Abstract channel base class
├── config.yaml               # Main configuration
├── ai_config.yaml            # AI-specific configuration
├── requirements.txt          # Python dependencies
│
├── channels/                 # Channel implementations
│   ├── internet_radio.py
│   ├── internet_radio_analogue.py
│   ├── local_mp3_player.py
│   ├── morse_code.py
│   └── ai_talkshow.py       # AI talk show channel (NEW)
│
├── ai/                       # AI modules (NEW)
│   ├── llm_client.py        # LLM abstraction layer
│   ├── tts_engine.py        # TTS abstraction layer
│   ├── audio_effects.py     # Audio effects pipeline
│   └── sonification.py      # Data-to-audio conversion
│
├── prompts/                  # AI prompt templates (NEW)
│   ├── dino_talk.txt
│   ├── quantum.txt
│   ├── mycelium.txt
│   ├── nebula.txt
│   └── xnull.txt
│
├── input/                    # Input handlers
│   ├── keyboard_handler.py
│   └── encoder_handler.py
│
└── models/                   # TTS models (create this)
    └── piper/
```

## Creating Custom Channels

To add a new channel, create a class that extends `BaseChannel`:

```python
from base_channel import BaseChannel

class MyCustomChannel(BaseChannel):
    def __init__(self, config):
        super().__init__(config)
        # Your initialization

    async def play(self):
        # Called repeatedly while channel is active
        pass

    async def stop(self):
        # Cleanup when channel stops
        pass

    async def on_encoder_A_input(self, value: int):
        # Handle encoder A rotation
        pass
```

Then add to `config.yaml`:

```yaml
channels:
  - name: MyChannel
    class: channels.my_custom_channel.MyCustomChannel
    button: X
```

## Creating Custom AI Realities

To add your own alternate reality channel:

1. **Create a prompt template** in `prompts/my_reality.txt`:
   ```
   You are hosting a radio show from [your reality description].

   Generate a 30-second segment (75-90 words).
   ```

2. **Add to `ai_config.yaml`**:
   ```yaml
   channels:
     - name: "My Reality"
       encoder_position_range: [40, 59]
       prompt_template_file: "prompts/my_reality.txt"
       voice_config:
         pitch_shift_semitones: -3
         reverb_amount: 0.5
   ```

## AI Radio Documentation

For comprehensive AI Radio documentation including:
- Detailed installation steps
- LLM provider setup (Ollama, OpenAI, Anthropic)
- TTS engine configuration
- Voice effects customization
- Channel X-NULL sonification details
- Troubleshooting guide
- Performance optimization

See: **[AI_RADIO_README.md](AI_RADIO_README.md)**

## Troubleshooting

### Common Issues

**"No module named 'yaml'"**
```bash
source myenv/bin/activate
pip install PyYAML
```

**"FFMPEG not found"**
```bash
sudo apt-get install ffmpeg
```

**"AI Radio generation failed"**
- Check that Ollama is running: `ollama serve`
- Verify model is downloaded: `ollama list`
- Check `ai_config.yaml` LLM provider settings

**"Piper synthesis failed"**
- Verify Piper is installed: `which piper`
- Check model path in `ai_config.yaml`
- Fallback to eSpeak: Set `tts.provider: "espeak"` in `ai_config.yaml`

**No audio output**
- Check audio devices: `aplay -l`
- Test with: `speaker-test -t wav`
- Verify PulseAudio: `pactl list sinks`

### Performance Tips

**For Raspberry Pi:**
- Use Ollama with smaller models (llama3.2:1b)
- Use eSpeak instead of Piper for faster TTS
- Reduce buffer size in `ai_config.yaml`
- Lower sample rate to 16000 Hz

**For Desktop:**
- Use larger LLM models for better quality
- Use Piper TTS for natural voices
- Increase buffer size for smoother playback

## Examples

### Example 1: Simple Usage
```bash
python main.py
# Press 'R' for AI Radio
# Press '+' repeatedly to tune through realities
```

### Example 2: Using with Cloud LLM
```bash
# Set API key
export OPENAI_API_KEY="your-key-here"

# Edit ai_config.yaml
# Change: llm.provider to "openai"

# Run
python main.py
```

### Example 3: Quick eSpeak Fallback
```bash
# Edit ai_config.yaml
# Change: tts.provider to "espeak"

# Run (no model download needed)
python main.py
```

## Features in Detail

### Analogue Radio Channel
- Pseudo-analogue tuning between internet stations
- Volume control via encoder B
- White noise during tuning transitions
- Smooth blending between stations

### AI Talk Show Channel
- Real-time content generation using LLMs
- 5 distinct alternate realities
- Voice effects per reality (pitch, reverb, speed)
- Anti-repetition system
- Pre-buffered segments for smooth playback
- Special sonification mode for Channel X-NULL

### Channel X-NULL (The Bizarre Channel)
Instead of normal speech, this channel:
- Generates mathematical/symbolic data via LLM
- Converts data to audio frequencies
- Adds carrier waves and noise
- Creates alien-sounding transmissions
- Sounds like: number stations + modem + alien data

## Contributing

This is a framework designed for experimentation and extension. Feel free to:
- Create new channel types
- Add new AI realities
- Implement new audio effects
- Add new input methods

## Technical Notes

- **Async/await**: All channels use async architecture for non-blocking operation
- **Dynamic loading**: Channels are loaded dynamically via importlib
- **Modular design**: Easy to add new LLM or TTS providers
- **YAML configuration**: No code changes needed for most customization
- **Buffer management**: Pre-generates AI content to prevent gaps

## License

MIT License

Copyright (c) 2025 mayankparmar

## Credits

- **Original Framework**: mayankparmar
- **AI Integration**: Claude (Anthropic)
- **LLM Support**: Ollama, OpenAI, Anthropic
- **TTS Support**: Piper (Rhasspy), Coqui TTS, eSpeak
- **Audio Libraries**: librosa, pedalboard, sounddevice

## Links

- **Detailed AI Documentation**: [AI_RADIO_README.md](AI_RADIO_README.md)
- **Piper TTS**: https://github.com/rhasspy/piper
- **Ollama**: https://ollama.ai
- **Voice Models**: https://github.com/rhasspy/piper/blob/master/VOICES.md

## Version History

- **v2.0** (2025): Added AI talk show channels with 5 alternate realities
- **v1.0** (2025): Initial release with internet radio, local playback, morse code
