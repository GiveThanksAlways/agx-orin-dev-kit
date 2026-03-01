# AGENTS.md

Nvidia Jetson AGX Orin 64 GB running NixOS (jetpack-nixos).

## Primary demo

`examples/presentation/demo_learned_inertial_odometry_flow.py` — walks through the full tinygrad NV pipeline (lazy UOp DAG → JIT capture → HCQGraph replay) using the Cioffi TCN (~250K params, learned inertial odometry for drone racing).

### VS Code debugger (direnv)

The `.envrc` at the repo root activates the nix flake via direnv. Toggle it for debugging:

```
Ctrl+Shift+P → direnv: Allow    # activate nix env (before F5)
Ctrl+Shift+P → direnv: Deny     # deactivate when done
```

Launch configs are in `.vscode/launch.json`. Pick **"Demo: Cioffi TCN flow (NV=1 DEBUG=2)"** and hit F5.

Breakpoints:
- `external/tinygrad/tinygrad/engine/realize.py:212` — `ei.run()` fires a fused kernel
- `external/tinygrad/tinygrad/runtime/ops_nv.py:127` — `dev.gpu_mmio[0x90 // 4]` MMIO doorbell wakes the GPU

### Run from terminal

```bash
cd examples/learned-inertial-odometry && nix develop
cd ../presentation
NV=1 DEBUG=2 python3 demo_learned_inertial_odometry_flow.py
```

## Key flakes

Each subfolder under `examples/` is a self-contained flake.

| Flake | Purpose | Notes |
|-------|---------|-------|
| `examples/learned-inertial-odometry/` | Cioffi TCN benchmarks + EKF pipeline | **Primary.** Follows control-loop's nixpkgs so PyTorch is pre-built. |
| `examples/control-loop/` | PyTorch CUDA built from source via nixpkgs | Upstream `nixos-25.11`. Other flakes follow this to avoid rebuild. |
| `examples/presentation/` | Presentation benchmarks (tinygrad NV vs TensorRT) | Demo script lives here. |
| `examples/tinygrad/` | Standalone tinygrad dev shell (CUDA + NV) | For tinygrad tests (`NV=1 pytest`). |
| `examples/nixos/` | NixOS system configuration | `sudo nixos-rebuild switch --flake ./examples/nixos#nixos` |

## Source layout

| Path | What |
|------|------|
| `external/tinygrad/` | Tinygrad source (git submodule, built from source — not a package) |
| `external/learned_inertial_model_odometry/` | Cioffi et al. original repo (git submodule) |
| `examples/learned-inertial-odometry/cioffi_tcn.py` | TCN model builder (tinygrad + PyTorch + ONNX export) |
| `examples/learned-inertial-odometry/cioffi_ekf.py` | Their ImuMSCKF (numba JIT) |
| `examples/control-loop/hot_path/` | C hot-path shared objects for real-time benchmarks |
| `.envrc` | direnv config — `use flake ./examples/learned-inertial-odometry` |
| `.env.nix` | Env vars captured from `nix develop` (CUDA_PATH, LD_LIBRARY_PATH, etc.) |
| `.vscode/launch.json` | 5 debug configs for the demo, benchmarks, and current-file |

## SSH access

```
Host: Orin-AGX-NixOS
User: agent
IP:   192.168.8.162
```

VIZ server (tinygrad profiler): `ssh -L 3000:localhost:3000 agent@192.168.8.162`

## NixOS rebuild

```bash
sudo nixos-rebuild switch --flake /home/agent/agx-orin-dev-kit/examples/nixos#nixos --show-trace
```

## random notes

```bash
# highlight the GPU usage during vscode debugging live demo
tegrastats | grep --color=always -E 'GR3D_FREQ [1-9][0-9]*%|$'
```
