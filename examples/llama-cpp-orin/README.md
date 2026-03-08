# llama.cpp on Jetson Orin AGX

Minimal flake that builds llama.cpp from the upstream repo with CUDA and
OpenSSL support for Orin AGX.

## Fastest path with a local GGUF

If `/nix/store` already contains `llama-cpp-cuda`, the simplest workflow is:

```bash
cd /home/agent/agx-orin-dev-kit/examples/llama-cpp-orin
nix develop
llama-server --model $HOME/.cache/tinygrad/downloads/qwen3-1.7b/Qwen3-1.7B-Q4_K_M.gguf --host 0.0.0.0 --port 5000 --ctx-size 4096 --threads 8 --batch-size 512 --ubatch-size 512 --n-gpu-layers auto --flash-attn on --mmap
```

The dev shell prints the exact commands to run for the local models it finds.

Defaults:

- Recommended for now: `~/.cache/tinygrad/downloads/qwen3-1.7b/Qwen3-1.7B-Q4_K_M.gguf`
- Binds to `0.0.0.0:5000`
- Uses `--n-gpu-layers auto --flash-attn on --mmap`

Override settings with environment variables:

```bash
MODEL=/models/Qwen3-1.7B-Q4_K_M.gguf
llama-server --model "$MODEL" --host 0.0.0.0 --port 5000 --ctx-size 8192 --threads 10 --batch-size 512 --ubatch-size 512 --n-gpu-layers auto --flash-attn on --mmap
```

Quick check:

```bash
curl http://127.0.0.1:5000/v1/models
```

## Quick start

```bash
nix develop            # enter the dev shell
llama-cli -hf unsloth/Qwen3-Coder-Next-GGUF:Q5_K_XL --gpu-layers 999
```

## API server

```bash
llama-server \
  -hf unsloth/Qwen3-Coder-Next-GGUF:Q5_K_XL \
  --gpu-layers 999 --host 0.0.0.0 --port 5000
```

OpenAI-compatible endpoint at `http://<host>:5000/v1`.

## Model quants

| Quant | Size | Command |
|---|---|---|
| Q4_K_M | ~35 GB | `llama-cli -hf unsloth/Qwen3-Coder-Next-GGUF:Q4_K_M -ngl 999` |
| Q5_K_M | ~45 GB | `llama-cli -hf unsloth/Qwen3-Coder-Next-GGUF:Q5_K_M -ngl 999` |
| Q5_K_XL | ~57 GB | `llama-cli -hf unsloth/Qwen3-Coder-Next-GGUF:Q5_K_XL -ngl 999` |

## Prerequisite

The host NixOS system must have jetpack-nixos configured.
See `examples/nixos/` for the base system flake.

## Other useful commands

```bash
llama-bench   # benchmark
llama-quantize  # requantize models
```
