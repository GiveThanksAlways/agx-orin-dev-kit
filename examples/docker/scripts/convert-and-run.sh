#!/bin/bash
# Quick conversion and inference for TensorRT-LLM
# Usage: ./convert-and-run.sh model-name [prompt]

set -e

MODEL_NAME=${1:-qwen3}
PROMPT=${2:-"Write a hello world program in Python"}

MODEL_DIR="/workspace/models/$MODEL_NAME"
ENGINE_DIR="/workspace/engines/$MODEL_NAME"

echo "🔥 TensorRT-LLM Quick Start"
echo "============================="
echo "Model: $MODEL_NAME"
echo

# Check model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Model not found: $MODEL_DIR"
    echo
    echo "Download a model first:"
    echo "  huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF --local-dir $MODEL_DIR"
    exit 1
fi

# Convert to TensorRT engine if not exists
if [ ! -d "$ENGINE_DIR" ]; then
    echo "⚙️  Building TensorRT engine (this takes 5-10 minutes, one-time setup)..."
    echo
    
    trtllm-build \
        --checkpoint_dir "$MODEL_DIR" \
        --output_dir "$ENGINE_DIR" \
        --gemm_plugin auto \
        --max_batch_size 8 \
        --max_input_len 2048 \
        --max_output_len 512 \
        --max_num_tokens 4096 \
        --dtype float16
    
    echo
    echo "✅ Engine built successfully!"
else
    echo "✅ Using existing engine: $ENGINE_DIR"
fi

echo
echo "🚀 Running inference..."
echo

python /workspace/run-model.py \
    --engine "$ENGINE_DIR" \
    --prompt "$PROMPT"

echo
echo "✅ Done! For interactive mode, run:"
echo "   python /workspace/run-model.py --engine $ENGINE_DIR --interactive"
