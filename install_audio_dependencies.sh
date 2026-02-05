#!/usr/bin/env bash
# Install audio processing dependencies (macOS / Linux)
# Run: ./install_audio_dependencies.sh

set -e

echo "====================================="
echo "Audio Processing Dependencies Setup"
echo "====================================="
echo ""

if [ -z "$VIRTUAL_ENV" ]; then
    echo "WARNING: No virtual environment detected!"
    echo "Activate your venv first: source .venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! "$REPLY" =~ ^[yY]$ ]]; then
        echo "Installation cancelled."
        exit 1
    fi
fi

echo "Step 1: Installing Python packages..."
pip install SpeechRecognition pydub

echo ""
echo "Step 2: Checking for FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo "✓ FFmpeg is already installed."
    ffmpeg -version | head -1
else
    echo "✗ FFmpeg not found. Install with: brew install ffmpeg  (macOS) or sudo apt-get install ffmpeg  (Linux)"
fi

echo ""
echo "Next: streamlit run localragdemo.py → open 🎤 Audio Processing"
