# llama.cpp on Jetson Orin AGX

Minimal flake that builds llama.cpp from the upstream repo with CUDA and
OpenSSL support for Orin AGX.

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
