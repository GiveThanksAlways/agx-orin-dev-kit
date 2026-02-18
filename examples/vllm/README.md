# vLLM on Jetson Orin AGX

High-throughput LLM inference engine with PagedAttention, continuous batching,
and OpenAI-compatible API.

## Prerequisites

Docker + NVIDIA Container Toolkit must be enabled in NixOS:

```bash
sudo nixos-rebuild switch --flake /home/agent/jetpack-nixos/examples/nixos#nixos-docker-bench
```

This activates:
- Docker daemon with overlay2 storage
- NVIDIA Container Toolkit (CDI mode, Jetson L4T CSV mounts)
- `docker` group for user-level access (re-login required)

## Quick Start

```bash
cd /home/agent/jetpack-nixos/examples/vllm

# Enter dev shell (provides python3, curl, jq for benchmarking)
nix develop

# Start vLLM server (pulls dustynv's JetPack 6 container if needed)
./run-vllm-docker.sh

# Or with a specific model + port:
./run-vllm-docker.sh meta-llama/Llama-3.2-3B-Instruct 8000

# Benchmark against running server
python3 bench_vllm.py --server http://localhost:8000 --model meta-llama/Llama-3.2-1B-Instruct

# Stop
docker stop vllm-orin
```

## Why Docker?

vLLM cannot be installed natively on NixOS aarch64 because:
1. **No PyPI wheel**: vLLM only ships x86_64 wheels on PyPI
2. **Complex C++ build**: Requires xgrammar, flashinfer, custom CUDA kernels
3. **Dependency hell**: NixOS's isolated environment breaks many pip-native builds

Docker provides a reliable path with near-zero GPU overhead on Jetson
(shared memory, direct CUDA passthrough via `--runtime nvidia`).

## Container Images

| Image | Size | Status |
|-------|------|--------|
| `dustynv/vllm:r36.4.0` | ~8 GB | Recommended for JetPack 6 |
| Build from `Dockerfile.jetson` | ~6 GB | Fallback (uses l4t-pytorch base) |

## Benchmarking

```bash
# Default (LLaMA 1B, 25 tokens)
python3 bench_vllm.py

# Custom model and token count
python3 bench_vllm.py --model meta-llama/Llama-3.2-3B-Instruct --num-tokens 50
```

The benchmark measures TTFT, streaming decode throughput, and total throughput.

## File Layout

```
vllm/
├── flake.nix              # Dev shell (python3, curl, jq)
├── run-vllm-docker.sh     # Docker launcher script
├── Dockerfile.jetson      # Fallback container build
├── bench_vllm.py          # Benchmark client (OpenAI API)
└── README.md              # This file
```
