#!/bin/bash
# Run MLC LLM in Docker on Jetson Orin AGX
# Uses dustynv's JetPack 6 MLC container
#
# PREREQUISITE: Docker + NVIDIA Container Toolkit must be enabled:
#   sudo nixos-rebuild switch --flake /home/agent/jetpack-nixos/examples/nixos#nixos-docker-bench
#
# Usage:
#   ./run-mlc-docker.sh [mode] [model]
#
# Modes:
#   serve   — Start OpenAI-compatible API server (default)
#   chat    — Interactive chat session
#   bench   — Run decode benchmark inside container
#
# Examples:
#   ./run-mlc-docker.sh                                                              # Serve LLaMA 1B
#   ./run-mlc-docker.sh serve HF://mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC        # Serve 3B
#   ./run-mlc-docker.sh chat  HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC        # Interactive
#   ./run-mlc-docker.sh bench HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC        # Benchmark

set -euo pipefail

MODE="${1:-serve}"
MODEL="${2:-HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC}"
PORT="${3:-8001}"
CONTAINER_NAME="mlc-llm-orin"
MLC_CACHE="${HOME}/.cache/mlc_llm"
HF_CACHE="${HOME}/.cache/huggingface"

echo "=== MLC LLM Docker Launcher (Jetson Orin AGX) ==="
echo "Mode: $MODE"
echo "Model: $MODEL"
echo "Port: $PORT"
echo ""

# Pre-flight checks
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found."
    echo "Enable Docker in NixOS:"
    echo "  sudo nixos-rebuild switch --flake /home/agent/jetpack-nixos/examples/nixos#nixos-docker-bench"
    exit 1
fi

if ! docker info &>/dev/null 2>&1; then
    echo "ERROR: Docker daemon not running or no permission."
    echo "  sudo systemctl start docker"
    echo "  # or: sudo usermod -aG docker \$USER && newgrp docker"
    exit 1
fi

# Ensure cache dirs exist
mkdir -p "$MLC_CACHE" "$HF_CACHE"

# Build or pull image
if ! docker image inspect mlc-jetson:latest >/dev/null 2>&1; then
    if [ -f Dockerfile.jetson ]; then
        echo "Building MLC LLM Jetson Docker image from Dockerfile.jetson..."
        docker build -t mlc-jetson:latest -f Dockerfile.jetson .
    else
        echo "Pulling pre-built Jetson MLC image..."
        # dustynv's JetPack 6 MLC containers
        for TAG in "dustynv/mlc:r36.4.0" "dustynv/mlc:0.1.dev-r36.4.0" "dustynv/mlc:latest"; do
            echo "  Trying $TAG ..."
            if docker pull "$TAG" 2>/dev/null; then
                docker tag "$TAG" mlc-jetson:latest
                echo "  Tagged as mlc-jetson:latest"
                break
            fi
        done

        if ! docker image inspect mlc-jetson:latest >/dev/null 2>&1; then
            echo "ERROR: Could not pull any pre-built MLC image."
            echo "Create Dockerfile.jetson or manually: docker pull <image>"
            exit 1
        fi
    fi
fi

# Stop existing container
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

COMMON_DOCKER_ARGS=(
    --runtime nvidia
    -e NVIDIA_VISIBLE_DEVICES=all
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
    -v "$MLC_CACHE:/root/.cache/mlc_llm"
    -v "$HF_CACHE:/root/.cache/huggingface"
    --shm-size 8g
)

case "$MODE" in
    serve)
        echo "Starting MLC LLM server on port $PORT..."
        docker run -d \
            --name "$CONTAINER_NAME" \
            "${COMMON_DOCKER_ARGS[@]}" \
            -p "$PORT:8000" \
            mlc-jetson:latest \
            python3 -m mlc_llm.serve \
                --model "$MODEL" \
                --mode server \
                --host 0.0.0.0 \
                --port 8000

        echo "Waiting for server..."
        for i in $(seq 1 90); do
            if curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
                echo ""
                echo "MLC LLM server ready at http://localhost:$PORT"
                echo ""
                echo "Test:  curl http://localhost:$PORT/v1/chat/completions \\"
                echo "         -H 'Content-Type: application/json' \\"
                echo "         -d '{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
                echo ""
                echo "Bench: python3 bench_mlc_llm.py --server http://localhost:$PORT --model $MODEL"
                echo "Logs:  docker logs -f $CONTAINER_NAME"
                echo "Stop:  docker stop $CONTAINER_NAME"
                exit 0
            fi
            sleep 2
            printf "\r  Waiting... (%d/90)" "$i"
        done
        echo ""
        echo "ERROR: Server did not become ready in 180s"
        echo "Check logs: docker logs $CONTAINER_NAME"
        exit 1
        ;;

    chat)
        echo "Starting interactive MLC LLM chat..."
        docker run -it --rm \
            --name "$CONTAINER_NAME" \
            "${COMMON_DOCKER_ARGS[@]}" \
            mlc-jetson:latest \
            python3 -m mlc_llm chat "$MODEL"
        ;;

    bench)
        echo "Running MLC LLM benchmark inside container..."
        docker run --rm \
            --name "${CONTAINER_NAME}-bench" \
            "${COMMON_DOCKER_ARGS[@]}" \
            mlc-jetson:latest \
            python3 -m mlc_llm bench "$MODEL" \
                --generate-length 128 \
                --prompt "Hello, how are you today?"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [serve|chat|bench] [model] [port]"
        exit 1
        ;;
esac
