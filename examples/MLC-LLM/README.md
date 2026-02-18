# MLC LLM on Jetson Orin AGX

TVM-based LLM inference engine with optimized CUDA kernels via Apache TVM.

## Prerequisites

Docker + NVIDIA Container Toolkit must be enabled in NixOS:

```bash
sudo nixos-rebuild switch --flake /home/agent/jetpack-nixos/examples/nixos#nixos-docker-bench
```

## Quick Start

```bash
cd /home/agent/jetpack-nixos/examples/MLC-LLM

# Enter dev shell
nix develop

# Start MLC LLM server (pulls dustynv's JetPack 6 container if needed)
./run-mlc-docker.sh serve

# Or interactive chat:
./run-mlc-docker.sh chat HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC

# Benchmark inside container:
./run-mlc-docker.sh bench

# Benchmark against server (from host):
python3 bench_mlc_llm.py --server http://localhost:8001

# Stop
docker stop mlc-llm-orin
```

## Why Docker?

MLC LLM cannot be installed natively on NixOS aarch64 because:
1. **No PyPI wheel**: MLC only ships x86_64 + macOS pre-built wheels
2. **TVM C++ build**: Requires LLVM, TVM runtime, custom CUDA codegen
3. **NixOS isolation**: Breaks the complex native build toolchain

## Container Images

| Image | Status |
|-------|--------|
| `dustynv/mlc:r36.4.0` | Recommended for JetPack 6 |
| Build from `Dockerfile.jetson` | Fallback (l4t-pytorch base + pip) |

## Modes

| Mode | Description |
|------|-------------|
| `serve` | Start OpenAI-compatible API server on port 8001 |
| `chat` | Interactive terminal chat |
| `bench` | Run decode benchmark inside container |

## Available Models

MLC LLM uses its own quantized model format (q4f16_1, q4f32_1, etc.):

| Model | Size | Quant |
|-------|------|-------|
| `HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC` | ~0.8 GB | q4f16 |
| `HF://mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC` | ~2.0 GB | q4f16 |
| `HF://mlc-ai/Llama-3.1-8B-Instruct-q4f16_1-MLC` | ~5.0 GB | q4f16 |

## File Layout

```
MLC-LLM/
├── flake.nix              # Dev shell (python3, curl, jq)
├── run-mlc-docker.sh      # Docker launcher (serve/chat/bench)
├── Dockerfile.jetson      # Fallback container build
├── bench_mlc_llm.py       # Benchmark client (server + native modes)
└── README.md              # This file
```
