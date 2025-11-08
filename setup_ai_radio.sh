#!/bin/bash

# AI Radio Setup Script
# Helps with installation of dependencies and models

set -e  # Exit on error

echo "======================================"
echo "AI Radio Setup Script"
echo "======================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${YELLOW}Warning: This script is designed for Linux. Some steps may fail on other systems.${NC}"
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "Step 1: Installing Python dependencies..."
echo "========================================="

if command_exists pip3; then
    pip3 install -r requirements.txt
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
else
    echo -e "${RED}✗ pip3 not found. Please install Python 3 and pip first.${NC}"
    exit 1
fi

echo ""
echo "Step 2: Checking for Ollama (Local LLM)..."
echo "=========================================="

if command_exists ollama; then
    echo -e "${GREEN}✓ Ollama is installed${NC}"

    # Check if ollama is running
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama is running${NC}"
    else
        echo -e "${YELLOW}⚠ Ollama is installed but not running${NC}"
        echo "  Start it with: ollama serve"
    fi

    # Check if model is downloaded
    if ollama list | grep -q "llama3.2"; then
        echo -e "${GREEN}✓ llama3.2 model found${NC}"
    else
        echo -e "${YELLOW}⚠ llama3.2 model not found${NC}"
        read -p "Download llama3.2:3b model? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ollama pull llama3.2:3b
            echo -e "${GREEN}✓ Model downloaded${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠ Ollama not found${NC}"
    read -p "Install Ollama? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
        echo -e "${GREEN}✓ Ollama installed${NC}"
        echo "  Starting Ollama..."
        ollama serve &
        sleep 3
        echo "  Downloading llama3.2:3b model..."
        ollama pull llama3.2:3b
    else
        echo -e "${YELLOW}⚠ Skipping Ollama installation. You'll need an online LLM (OpenAI/Anthropic)${NC}"
    fi
fi

echo ""
echo "Step 3: Checking for TTS (Text-to-Speech)..."
echo "============================================="

# Check for Piper
if command_exists piper; then
    echo -e "${GREEN}✓ Piper is installed${NC}"

    # Check for model
    if [ -f "models/piper/en_US-lessac-medium.onnx" ]; then
        echo -e "${GREEN}✓ Piper voice model found${NC}"
    else
        echo -e "${YELLOW}⚠ Piper voice model not found${NC}"
        read -p "Download Piper voice model? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            mkdir -p models/piper
            echo "Downloading en_US-lessac-medium voice..."
            wget -q --show-progress \
                https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx \
                -O models/piper/en_US-lessac-medium.onnx
            wget -q --show-progress \
                https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json \
                -O models/piper/en_US-lessac-medium.onnx.json
            echo -e "${GREEN}✓ Voice model downloaded${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠ Piper not found${NC}"
    read -p "Install Piper? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Downloading Piper..."

        # Detect architecture
        ARCH=$(uname -m)
        if [ "$ARCH" = "x86_64" ]; then
            PIPER_ARCH="amd64"
        elif [ "$ARCH" = "aarch64" ]; then
            PIPER_ARCH="arm64"
        else
            echo -e "${RED}✗ Unsupported architecture: $ARCH${NC}"
            echo "  Please install Piper manually from: https://github.com/rhasspy/piper/releases"
            PIPER_ARCH=""
        fi

        if [ -n "$PIPER_ARCH" ]; then
            wget -q --show-progress \
                https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_${PIPER_ARCH}.tar.gz
            tar -xzf piper_${PIPER_ARCH}.tar.gz
            sudo mv piper/piper /usr/local/bin/
            rm -rf piper piper_${PIPER_ARCH}.tar.gz
            echo -e "${GREEN}✓ Piper installed${NC}"

            # Download voice model
            mkdir -p models/piper
            echo "Downloading voice model..."
            wget -q --show-progress \
                https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx \
                -O models/piper/en_US-lessac-medium.onnx
            wget -q --show-progress \
                https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json \
                -O models/piper/en_US-lessac-medium.onnx.json
            echo -e "${GREEN}✓ Voice model downloaded${NC}"
        fi
    else
        # Check for eSpeak fallback
        if command_exists espeak; then
            echo -e "${GREEN}✓ eSpeak available as fallback TTS${NC}"
            echo "  Note: eSpeak sounds robotic but works without models"
            echo "  To use it, edit ai_config.yaml and set tts.provider to 'espeak'"
        else
            echo -e "${YELLOW}⚠ Installing eSpeak as fallback TTS...${NC}"
            sudo apt-get update && sudo apt-get install -y espeak
            echo -e "${GREEN}✓ eSpeak installed${NC}"
            echo "  Edit ai_config.yaml and set tts.provider to 'espeak'"
        fi
    fi
fi

echo ""
echo "Step 4: Checking audio system..."
echo "================================="

if command_exists aplay; then
    echo -e "${GREEN}✓ ALSA audio found${NC}"
else
    echo -e "${RED}✗ ALSA not found${NC}"
    echo "  Install with: sudo apt-get install alsa-utils"
fi

if command_exists pactl; then
    echo -e "${GREEN}✓ PulseAudio found${NC}"
else
    echo -e "${YELLOW}⚠ PulseAudio not found (optional)${NC}"
fi

echo ""
echo "Step 5: Configuration check..."
echo "==============================="

if [ -f "ai_config.yaml" ]; then
    echo -e "${GREEN}✓ ai_config.yaml exists${NC}"
else
    echo -e "${RED}✗ ai_config.yaml not found${NC}"
fi

if [ -f "config.yaml" ]; then
    echo -e "${GREEN}✓ config.yaml exists${NC}"
else
    echo -e "${RED}✗ config.yaml not found${NC}"
fi

# Check prompt templates
PROMPTS_OK=true
for prompt in prompts/dino_talk.txt prompts/quantum.txt prompts/mycelium.txt prompts/nebula.txt prompts/xnull.txt; do
    if [ ! -f "$prompt" ]; then
        echo -e "${RED}✗ Missing: $prompt${NC}"
        PROMPTS_OK=false
    fi
done

if [ "$PROMPTS_OK" = true ]; then
    echo -e "${GREEN}✓ All prompt templates found${NC}"
fi

echo ""
echo "======================================"
echo "Setup Summary"
echo "======================================"
echo ""

# Determine recommended config
if command_exists ollama && ollama list | grep -q "llama3.2"; then
    echo -e "${GREEN}LLM: Ollama with llama3.2 (local, recommended)${NC}"
    LLM_OK=true
else
    echo -e "${YELLOW}LLM: Not configured. Options:${NC}"
    echo "  1. Use Ollama (local, free) - run: ollama serve && ollama pull llama3.2:3b"
    echo "  2. Use OpenAI (cloud, requires API key)"
    echo "  3. Use Anthropic Claude (cloud, requires API key)"
    LLM_OK=false
fi

if command_exists piper && [ -f "models/piper/en_US-lessac-medium.onnx" ]; then
    echo -e "${GREEN}TTS: Piper (local, high quality, recommended)${NC}"
    TTS_OK=true
elif command_exists espeak; then
    echo -e "${YELLOW}TTS: eSpeak (local, robotic sound)${NC}"
    echo "  Edit ai_config.yaml and set tts.provider to 'espeak'"
    TTS_OK=true
else
    echo -e "${RED}TTS: Not configured${NC}"
    TTS_OK=false
fi

echo ""

if [ "$LLM_OK" = true ] && [ "$TTS_OK" = true ]; then
    echo -e "${GREEN}✓ AI Radio is ready to use!${NC}"
    echo ""
    echo "To start:"
    echo "  python main.py"
    echo ""
    echo "Then press 'R' to activate AI Radio channel"
    echo "Use '+' and '-' to tune between realities (0-99)"
else
    echo -e "${YELLOW}⚠ AI Radio is partially configured${NC}"
    echo ""
    echo "Next steps:"
    if [ "$LLM_OK" = false ]; then
        echo "  1. Set up an LLM (see AI_RADIO_README.md)"
    fi
    if [ "$TTS_OK" = false ]; then
        echo "  2. Set up TTS (see AI_RADIO_README.md)"
    fi
    echo "  3. Review ai_config.yaml and adjust settings"
    echo "  4. Run: python main.py"
fi

echo ""
echo "For detailed documentation, see: AI_RADIO_README.md"
echo "======================================"
