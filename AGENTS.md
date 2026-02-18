# AGENTS.md

We are working with an Nvidia Jetson dev kit Orin AGX 64GB


## More info

- I put the `configuration.nix` and the `hardware-configuration.nix` there for quick experimentation

```bash
# example build
sudo nixos-rebuild switch --flake /home/agent/jetpack-nixos/examples/nixos#nixos-telemetry --show-trace

```

Each subfolder is a self-contained flake. Pick the one that fits your use case.

## Layout

| Folder                        | What it does                                                                                                                        |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `nixos/`                      | Base NixOS system config with composable modules (performance, llama.cpp, TabbyAPI, telemetry). Pick a configuration at build time. |
| `llama-cpp-orin/`             | Minimal flake -- builds llama.cpp from upstream with CUDA/OpenSSL for Orin. Dev shell only.                                         |
| `llama-cpp-orin-nix-overlay/` | Full-featured overlay with wrapper scripts (`qwen3-coder`, `qwen3-server`, `llama-benchmark`). Reusable via `overlays.default`.     |
| `LLM/`                        | LLM's                                                                                                                               |
| `qwen3-tts-12hz-1.7b-customvoice/` | Dev shell to run `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` on Jetson Orin AGX.                                                  |
| `tinygrad/`                   | Tinygrad dev shell with CUDA + NV backend. Clone-and-go for LLM examples (GPT-2, LLaMA) on Orin AGX.                                |
| `telemetry-viewer/`           | SSH tunnel script to view Grafana dashboards from your PC. One command, then open Chrome.                                           |

## Quick reference

```bash
# NixOS system build (pick a config)
nixos-rebuild switch --flake ./nixos#nixos
nixos-rebuild switch --flake ./nixos#nixos-perf
nixos-rebuild switch --flake ./nixos#nixos-llama-cpp

# llama.cpp dev shell (simple)
cd llama-cpp-orin && nix develop

# llama.cpp dev shell (overlay + wrappers)
cd llama-cpp-orin-nix-overlay && nix develop

# vLLM dev shell
cd vLLM && nix develop

# Qwen3-TTS CustomVoice dev shell
cd qwen3-tts-12hz-1.7b-customvoice && nix develop

# tinygrad dev shell (CUDA + NV backend, includes pytest)
cd examples/tinygrad && nix develop
# Then run tests from inside the shell:
#   cd tinygrad && NV=1 python3 -m pytest test/test_ops.py -v --tb=short

# view telemetry dashboards from your PC (user named agent)
cd telemetry-viewer && ./connect-telemetry.sh <jetson-ip> <ssh-user>
