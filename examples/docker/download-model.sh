#!/bin/bash
# Helper script to download models from HuggingFace for TensorRT-LLM

set -e

MODELS_DIR="models"
mkdir -p "$MODELS_DIR"

download_hf_model() {
    local repo=$1
    local name=$2
    
    echo "📥 Downloading $name from HuggingFace..."
    echo "   Repo: $repo"
    
    if ! command -v huggingface-cli &> /dev/null; then
        echo "Installing huggingface-hub..."
        pip install -U huggingface-hub
    fi
    
    huggingface-cli download "$repo" --local-dir "$MODELS_DIR/$name"
    
    echo "✅ Downloaded to $MODELS_DIR/$name"
}

case "${1:-help}" in
    qwen3-coder)
        download_hf_model "unsloth/Qwen3-Coder-Next-GGUF" "qwen3-coder"
        ;;
    qwen-7b)
        download_hf_model "Qwen/Qwen2.5-Coder-7B-Instruct" "qwen-7b"
        ;;
    llama3-8b)
        download_hf_model "meta-llama/Llama-3.1-8B-Instruct" "llama3-8b"
        ;;
    mistral-7b)
        download_hf_model "mistralai/Mistral-7B-Instruct-v0.3" "mistral-7b"
        ;;
    *)
        cat << EOF
📥 TensorRT-LLM Model Downloader

Usage: ./download-model.sh <model>

Available models:
  qwen3-coder    - Qwen3 Coder Next (recommended for code)
  qwen-7b        - Qwen 2.5 7B Instruct (general purpose)
  llama3-8b      - Meta Llama 3.1 8B Instruct
  mistral-7b     - Mistral 7B Instruct v0.3

Or manually:
  huggingface-cli download <repo> --local-dir models/<name>

Examples:
  ./download-model.sh qwen3-coder
  ./download-model.sh llama3-8b

After downloading, convert to TensorRT engine:
  docker exec -it tensorrt-llm bash
  ./scripts/convert-and-run.sh <model-name>

EOF
        ;;
esac
