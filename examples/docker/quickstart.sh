#!/bin/bash
# TensorRT-LLM Quick Start for Jetson

set -e

echo "🚀 TensorRT-LLM Quick Start"
echo "============================"
echo

# Check docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Install it first:"
    echo "   https://docs.docker.com/engine/install/"
    exit 1
fi

# Check docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "⚠️  docker-compose not found, installing..."
    pip install docker-compose || sudo apt install docker-compose
fi

# Check NVIDIA Docker
echo "🔍 Checking NVIDIA GPU support..."
if docker run --rm --gpus all nvidia/cuda:12.4.1-base nvidia-smi &>/dev/null ; then
    echo "✅ NVIDIA GPU support OK"
else
    echo "❌ NVIDIA GPU support not working"
    echo "   Install nvidia-docker2 and restart Docker daemon"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create directories
mkdir -p models engines scripts

# Build image (pulls NGC TensorRT-LLM)
echo
echo "📦 Building TensorRT-LLM container (may take a few minutes)..."
docker-compose build

# Start container
echo
echo "🐳 Starting TensorRT-LLM container..."
docker-compose up -d

sleep 2

# Verify container
if docker exec tensorrt-llm nvidia-smi &>/dev/null; then
    echo "✅ Container running with GPU access"
else
    echo "⚠️  Container started but GPU access failed"
fi

echo
echo "✅ Setup complete!"
echo
echo "🎯 Next Steps:"
echo
echo "1. Enter container:"
echo "   docker exec -it tensorrt-llm bash"
echo
echo "2. Download a model (inside container):"
echo "   huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF --local-dir /workspace/models/qwen3"
echo
echo "3. Convert & run (inside container):"
echo "   ./scripts/convert-and-run.sh qwen3"
echo
echo "Or one-liner from outside:"
echo "   docker exec -it tensorrt-llm bash -c './scripts/convert-and-run.sh qwen3'"
echo
echo "📚 See README.md for details"
echo
echo "Stop: docker-compose down"
echo
