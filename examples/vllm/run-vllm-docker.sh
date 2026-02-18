#!/bin/bash
# Run vLLM in Docker on Jetson Orin AGX
# Uses NVIDIA L4T / dustynv base images with JetPack 6 CUDA 12.6
#
# PREREQUISITE: Docker + NVIDIA Container Toolkit must be enabled:
#   sudo nixos-rebuild switch --flake /home/agent/jetpack-nixos/examples/nixos#nixos-docker-bench
#
# Usage:
#   ./run-vllm-docker.sh [model] [port]
#
# Examples:
#   ./run-vllm-docker.sh                                     # Default: Llama 3.2 1B
#   ./run-vllm-docker.sh meta-llama/Llama-3.2-3B-Instruct    # 3B model
#   ./run-vllm-docker.sh Qwen/Qwen3-0.6B 8001                # Custom port

set -euo pipefail

MODEL="${1:-meta-llama/Llama-3.2-1B-Instruct}"
PORT="${2:-8000}"
CONTAINER_NAME="vllm-orin"
HF_CACHE="${HOME}/.cache/huggingface"

echo "=== vLLM Docker Launcher (Jetson Orin AGX) ==="
echo "Model: $MODEL"
echo "Port: $PORT"
echo "HF cache: $HF_CACHE"
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

# Ensure HF cache dir exists
mkdir -p "$HF_CACHE"

# Build or pull image
if ! docker image inspect vllm-jetson:latest >/dev/null 2>&1; then
    if [ -f Dockerfile.jetson ]; then
        echo "Building vLLM Jetson Docker image from Dockerfile.jetson..."
        docker build -t vllm-jetson:latest -f Dockerfile.jetson .
    else
        echo "Pulling pre-built Jetson vLLM image..."
        # dustynv's JetPack 6 images — try multiple tags
        for TAG in "dustynv/vllm:r36.4.0" "dustynv/vllm:0.6.6-r36.4.0" "dustynv/vllm:latest"; do
            echo "  Trying $TAG ..."
            if docker pull "$TAG" 2>/dev/null; then
                docker tag "$TAG" vllm-jetson:latest
                echo "  Tagged as vllm-jetson:latest"
                break
            fi
        done

        if ! docker image inspect vllm-jetson:latest >/dev/null 2>&1; then
            echo "ERROR: Could not pull any pre-built image."
            echo "Create Dockerfile.jetson or manually: docker pull <image>"
            exit 1
        fi
    fi
fi

# Stop existing container
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo ""
echo "Starting vLLM server..."
echo "  Model: $MODEL"
echo "  Port: localhost:$PORT"
echo ""

# Run with CDI-based GPU access (jetpack-nixos nvidia-container-toolkit)
# --runtime nvidia works when nvidia-container-toolkit is enabled
docker run -d \
    --name "$CONTAINER_NAME" \
    --runtime nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -p "$PORT:8000" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    --shm-size 8g \
    vllm-jetson:latest \
    --model "$MODEL" \
    --max-model-len 2048 \
    --dtype half \
    --gpu-memory-utilization 0.8 \
    --enforce-eager

echo ""
echo "Container started: $CONTAINER_NAME"
echo "Waiting for server to be ready..."

# Wait for health check
for i in $(seq 1 90); do
    if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo ""
        echo "vLLM server is ready at http://localhost:$PORT"
        echo ""
        echo "Test:"
        echo "  curl http://localhost:$PORT/v1/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"model\": \"$MODEL\", \"prompt\": \"Hello\", \"max_tokens\": 10}'"
        echo ""
        echo "Benchmark:"
        echo "  python3 bench_vllm.py --server http://localhost:$PORT --model $MODEL"
        echo ""
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
