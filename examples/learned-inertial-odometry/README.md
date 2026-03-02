# Cioffi TCN Benchmark — Learned Inertial Odometry

Benchmark the exact TCN from **Cioffi et al., "Learned Inertial Odometry for Autonomous Drone Racing"** (IEEE RA-L 2023, [arXiv:2210.15287](https://arxiv.org/abs/2210.15287)) across five inference backends on **Jetson AGX Orin 64GB**.

## Model

| Property | Value |
|---|---|
| Architecture | Temporal Convolutional Network (TCN) |
| Blocks | 7 TemporalBlocks, causal dilated Conv1d → GELU → residual → ReLU |
| Channels | `[6 → 64 → 64 → 64 → 64 → 128 → 128 → 128]` |
| Kernel size | 2, dilations `[1, 2, 4, 8, 16, 32, 64]` |
| Input | `(1, 6, 200)` — 3 gyro + 3 thrust, 200 timesteps |
| Output | `(1, 3)` — Δp displacement (x, y, z) |
| Parameters | ~250K |
| Source | [uzh-rpg/learned_inertial_model_odometry](https://github.com/uzh-rpg/learned_inertial_model_odometry) (our fork in `external/learned_inertial_model_odometry/`) |

## Backends

| Backend | Description |
|---|---|
| **tinygrad NV=1** | Raw Tegra ioctls, direct `memmove` via unified memory, TinyJit + JITBEAM |
| **tinygrad CUDA=1** | Standard CUDA runtime path via tinygrad |
| **C Hot Path** | Same GPU kernels as NV=1, replayed from C via raw MMIO doorbell — zero Python, zero ioctls |
| **TensorRT** | NVIDIA's optimized inference engine (ONNX → TRT engine) |
| **PyTorch** | Eager CUDA + CUDA Graphs (requires CUDA torch from control-loop flake) |

## Quick Start

```bash
# From repo root
cd examples/learned-inertial-odometry
nix develop

# ── TCN-only benchmarks (all backends) ──
NV=1 JITBEAM=2 python3 bench_cioffi_tcn.py

# Individual backends
NV=1 JITBEAM=2 python3 bench_cioffi_tcn.py --backend nv
NV=1 JITBEAM=2 python3 bench_cioffi_tcn.py --backend hotpath
CUDA=1 python3 bench_cioffi_tcn.py --backend cuda
python3 bench_cioffi_tcn.py --backend trt
python3 bench_cioffi_tcn.py --backend pytorch

# ── End-to-end pipeline (Fig. 2: IMU Prop → Buffer → TCN → EKF Update) ──
NV=1 JITBEAM=2 python3 bench_e2e_pipeline.py                    # Simulated IMU data
NV=1 JITBEAM=2 python3 bench_e2e_pipeline.py --data real         # Real Blackbird dataset
NV=1 JITBEAM=2 python3 bench_e2e_pipeline.py --backend hotpath   # C Hot Path
python3 bench_e2e_pipeline.py --backend trt                       # TensorRT

# Options
NV=1 JITBEAM=2 python3 bench_cioffi_tcn.py --iters 1000        # Quick run
NV=1 JITBEAM=2 python3 bench_cioffi_tcn.py --precision fp32    # FP32 comparison
NV=1 JITBEAM=2 python3 bench_cioffi_tcn.py --save results.json # Save results
```

## Prerequisites

### C Hot Path

The C hot path reuses the shared library from `examples/control-loop/hot_path/`. Build it first:

```bash
cd ../control-loop/hot_path
make     # produces hot_path.so
```

The benchmark auto-discovers `hot_path.so` via the `HOT_PATH_DIR` env var (set by the flake shell hook).

### PyTorch CUDA

This flake includes a CPU-only PyTorch wheel (for ONNX export). For PyTorch CUDA benchmarks, use the control-loop flake which has torch built from source with CUDA:

```bash
cd ../control-loop && nix develop
python3 ../learned-inertial-odometry/bench_cioffi_tcn.py --backend pytorch
```

### tinygrad

The tinygrad submodule at `external/tinygrad/` is auto-discovered by the shell hook. If not found:

```bash
git submodule update --init --recursive
```

## Files

| File | Purpose |
|---|---|
| `cioffi_tcn.py` | TCN model in PyTorch + tinygrad, shared weight generation, ONNX export |
| `cioffi_ekf.py` | Lightweight IMU-MSCKF matching the paper's EKF (SO(3) propagation + Kalman update) |
| `bench_cioffi_tcn.py` | TCN-only benchmark — runs all backends, prints comparison table |
| `bench_e2e_pipeline.py` | End-to-end pipeline benchmark (Fig. 2): IMU Prop → Buffer → TCN → EKF Update |
| `flake.nix` | Nix dev shell with torch-bin, TensorRT, CUDA, clang, h5py |

## How It Works

1. **Shared weights**: `generate_weights()` creates deterministic FP32 weights (seed=42), ensuring all backends compute the same function.

2. **Input pool**: `generate_input_pool()` creates simulated IMU data (gyro + thrust) as FP16 tensors.

3. **tinygrad NV=1**: Builds the TCN in tinygrad, warms up TinyJit (which captures the HCQGraph and runs BEAM search), then benchmarks with direct `memmove` to unified memory buffers.

4. **C Hot Path**: After TinyJit warmup, exports the HCQGraph internals (GPU buffer addresses, GPFifo ring, command queue, patch map) via `export_graph.py`. The C code replays the exact same GPU commands by writing to the GPFifo ring buffer and poking the MMIO doorbell — zero Python overhead, zero ioctls in the hot loop.

5. **TensorRT**: Exports to ONNX via PyTorch, builds a TRT engine, runs inference via `cudaMemcpyAsync` → `execute_async_v3` → `cudaStreamSynchronize`.

6. **PyTorch**: Standard eager forward pass + CUDA Graphs for graph-captured replay.

## End-to-End Pipeline (Fig. 2)

The `bench_e2e_pipeline.py` reproduces the complete system from the paper:

```text
IMU data (gyro ω_b, accel a_b, thrust T_b)
  │
  ├──▶ IMU Propagation (5× per update @ 100 Hz)
  │      R_new = R @ exp((ω - b_g) × dt)
  │      v_new = v + R(a - b_a)dt + g×dt
  │      p_new = p + v×dt + ½R(a - b_a)dt² + ½g×dt²
  │      + 15×15 covariance propagation
  │
  ├──▶ Ring Buffer → (1, 6, 200) tensor
  │
  ├──▶ TCN Inference → Δp (3-DoF displacement)
  │
  └──▶ EKF Update
         innovation = Δp_measured - (p_clone_end - p_clone_begin)
         K = Σ Hᵀ (H Σ Hᵀ + R)⁻¹
         δX = K × innovation
         Apply correction to R, v, p, b_g, b_a
```

Per update cycle at 20 Hz (50,000 µs budget):

- **IMU propagation**: 5 steps of SO(3) integration + Jacobian + covariance
- **TCN inference**: the neural network forward pass (the bottleneck we're benchmarking)
- **EKF update**: Kalman gain computation + state correction + marginalization

### Data Sources

- **Simulated** (`--data sim`): Gentle hovering flight with realistic IMU noise
- **Real Blackbird** (`--data real`): Actual drone flight data from the paper's test split (included in the repo at `datasets/Blackbird/`)
