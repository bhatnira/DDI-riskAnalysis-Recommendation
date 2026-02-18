#!/bin/bash
# DDI Risk Analysis - Full Setup Script
# Installs Python dependencies and Ollama for local LLM support

set -e

echo "========================================"
echo "  DDI Risk Analysis - Setup Script"
echo "========================================"
echo ""

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    OS="windows"
fi

echo "Detected OS: $OS"
echo ""

# Step 1: Python virtual environment
echo "[1/4] Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  Created virtual environment"
else
    echo "  Virtual environment already exists"
fi

# Activate venv
source .venv/bin/activate
echo "  Activated virtual environment"

# Step 2: Install Python dependencies
echo ""
echo "[2/4] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  Python dependencies installed"

# Step 3: Install Ollama
echo ""
echo "[3/4] Installing Ollama (local LLM)..."

install_ollama() {
    if command -v ollama &> /dev/null; then
        echo "  Ollama is already installed"
        return 0
    fi
    
    case $OS in
        linux)
            echo "  Installing Ollama for Linux..."
            curl -fsSL https://ollama.com/install.sh | sh
            ;;
        macos)
            if command -v brew &> /dev/null; then
                echo "  Installing Ollama via Homebrew..."
                brew install ollama
            else
                echo "  Installing Ollama for macOS..."
                curl -fsSL https://ollama.com/install.sh | sh
            fi
            ;;
        windows)
            echo "  For Windows, please download Ollama from: https://ollama.com/download"
            echo "  After installation, run: ollama pull llama3"
            return 1
            ;;
        *)
            echo "  Unknown OS. Please install Ollama manually from: https://ollama.com"
            return 1
            ;;
    esac
}

if install_ollama; then
    echo "  Ollama installed successfully"
else
    echo "  Ollama installation requires manual steps (see above)"
fi

# Step 4: Download LLM model
echo ""
echo "[4/4] Downloading LLM model..."

download_model() {
    if ! command -v ollama &> /dev/null; then
        echo "  Skipping model download (Ollama not installed)"
        return 1
    fi
    
    # Check if model exists
    if ollama list 2>/dev/null | grep -q "llama3"; then
        echo "  llama3 model already downloaded"
        return 0
    fi
    
    echo "  Downloading llama3 model (this may take a few minutes)..."
    echo "  Model size: ~4.7GB"
    
    # Start ollama service if not running
    if [[ "$OS" == "linux" ]]; then
        # Check if systemd service exists
        if systemctl is-active --quiet ollama 2>/dev/null; then
            echo "  Ollama service is running"
        else
            echo "  Starting Ollama service..."
            ollama serve &>/dev/null &
            sleep 3
        fi
    fi
    
    ollama pull llama3
    echo "  llama3 model downloaded"
}

download_model || true

# Summary
echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "To start the DDI Assistant GUI:"
echo "  source .venv/bin/activate"
echo "  python ddi_chat_gui.py"
echo ""
echo "Then open: http://localhost:8080"
echo ""

if command -v ollama &> /dev/null; then
    echo "Ollama Status: Installed"
    if ollama list 2>/dev/null | grep -q "llama3"; then
        echo "LLM Model: llama3 ready"
    else
        echo "LLM Model: Not downloaded. Run: ollama pull llama3"
    fi
else
    echo "Ollama Status: Not installed (chat will use template responses)"
    echo "To install later: curl -fsSL https://ollama.com/install.sh | sh"
fi

echo ""
