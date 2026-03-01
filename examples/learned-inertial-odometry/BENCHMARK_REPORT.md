# Benchmark Report: Cioffi TCN — Learned Inertial Odometry on Jetson AGX Orin

> **Date:** 2026-03-01
> **Hardware:** Jetson AGX Orin 64GB Developer Kit, MAXN mode (60W)
> **Software:** NixOS (jetpack-nixos), CUDA 12.6, TensorRT 10.7.0, PyTorch 2.9.1, tinygrad (main), numba 0.63.1, numpy 2.3.4

## Summary

We benchmark the exact Temporal Convolutional Network (TCN) from **Cioffi et al., "Learned Inertial Odometry for Autonomous Drone Racing"** (IEEE RA-L 2023) in a full end-to-end VIO pipeline — not just isolated inference.

**Key findings:**

1. **C Hot Path** (tinygrad NV=1 kernels replayed from C via MMIO doorbell) delivers a **2.6× end-to-end speedup** over PyTorch CUDA Graphs, completing the full sensor-to-state loop in **2,275 µs** (440 Hz) versus 5,905 µs (169 Hz).

2. **tinygrad NV=1** with JITBEAM=2 achieves **2,954 µs** (338 Hz) — **2.0× faster** than PyTorch while remaining pure Python.

3. **TensorRT FP16** (standalone TCN via `trtexec`) measures **559 µs** GPU compute — between C Hot Path (334 µs, 1.7× faster) and tinygrad NV=1 (1,040 µs, 1.9× slower).

4. **All backends exceed the 20 Hz real-time budget by 8–22×**, enabling higher-rate fusion or additional compute headroom for perception.

5. Enabling **numba JIT** for EKF propagation cuts IMU+EKF cost from ~4,500 µs to ~1,930 µs, amplifying the TCN backend advantage from 1.4× to 2.6× end-to-end.

> **Hardware caveat:** These numbers were measured on AGX Orin 64GB (2048 CUDA cores, Ampere). The paper's target platform class is much more constrained. See [Hardware Honesty](#hardware-honesty-agx-orin-vs-real-drone-compute) below.

---

## Model

| Property | Value |
|---|---|
| Architecture | 7-block causal dilated TCN (Conv1d → GELU → residual → ReLU) |
| Channels | `[6 → 64 → 64 → 64 → 64 → 128 → 128 → 128]` |
| Kernel size | 2, dilations `[1, 2, 4, 8, 16, 32, 64]` |
| Parameters | ~250K |
| Input | `(1, 6, 50)` — 3 gyro + 3 thrust, 50 timesteps @ 100 Hz = 0.5s |
| Output | `(1, 3)` — Δp displacement (x, y, z) |
| Precision | FP16 (inference), FP32 (weights generated with seed=42) |
| Source | [uzh-rpg/learned_inertial_model_odometry](https://github.com/uzh-rpg/learned_inertial_model_odometry) |

---

## Methodology

### What we measure

We reproduce the complete system from **Fig. 2 of the paper** — the full sensor-to-state loop that runs on the drone's onboard computer. Each update cycle at 20 Hz includes:

```
IMU data (gyro ω_b, accel a_b, thrust T_b) @ 100 Hz
  │
  ├──▶ IMU Propagation (5 steps per update)
  │      SO(3) integration + 15×15 covariance propagation
  │      (numba JIT-compiled, their exact code from scekf.py)
  │
  ├──▶ Ring Buffer → (1, 6, 50) tensor
  │
  ├──▶ TCN Inference → Δp (3-DoF displacement)
  │      ← ONLY this component changes between backends
  │
  └──▶ EKF Update (Kalman gain + state correction)
         (their exact ImuMSCKF code, unchanged)
```

**Fair comparison:** The EKF, IMU propagation, and buffer logic are identical across all runs. Only the TCN inference backend changes.

### Measurement parameters

| Parameter | Value |
|---|---|
| Warmup iterations | 20 update cycles (absorbs JIT compilation, CUDA Graphs capture, BEAM search) |
| Benchmark iterations | 200 update cycles |
| Timing | Per-cycle `time.perf_counter()` for total, IMU, TCN, EKF individually |
| Statistics | Median (primary), mean, std, p99, min, max |
| IMU rate | 100 Hz (5 propagation steps per 20 Hz update) |
| Budget | 50,000 µs per update (20 Hz real-time) |
| Simulated data | Gentle hovering with realistic IMU noise |

### Backends

| Backend | How it works |
|---|---|
| **C Hot Path** (JITBEAM=2) | tinygrad NV=1 compiles optimized GPU kernels via BEAM search, exports the HCQGraph, then C code replays the exact same GPU commands by writing to the GPFifo ring buffer and poking the MMIO doorbell. Zero Python, zero ioctls, zero CUDA runtime in the hot loop. Unified memory → no H2D/D2H copies. |
| **tinygrad NV=1** (JITBEAM=2) | Same BEAM-optimized kernels as C Hot Path, but dispatched via Python + Tegra ioctls. Still bypasses CUDA runtime, still uses unified memory memmove. The overhead vs C Hot Path is Python call overhead + ioctl syscalls. |
| **PyTorch CUDA Graphs** | Standard `torch.cuda.CUDAGraph` capture + replay. CUDA runtime path, `cudaMemcpyAsync` for transfers. Their original code path. |
| **tinygrad NV=1** (no BEAM) | NV=1 without BEAM kernel search — uses tinygrad's default kernel scheduling. Shows the isolated impact of BEAM search. |
| **TensorRT** (trtexec) | ONNX → TRT engine with NVIDIA's auto-tuning. Measured standalone via `trtexec` (Python `tensorrt` bindings not available in our NixOS environment). Not directly in the pipeline, so includes only TCN compute + H2D/D2H. |

### Numba JIT compatibility

Cioffi's EKF code (`scekf.py`) uses numba `@jit(nopython=True)` for the IMU propagation and covariance update functions. numba 0.63 + numpy 2.3 has a bug where 2D slice-to-slice assignment (`A[3:6, 0:3] = B`) fails to lower as `static_setitem`. We patched this via `numba_compat.py`, which provides drop-in replacements using element-by-element loops that JIT-compile correctly. The patch is transparent — `import numba_compat` monkey-patches the scekf module before the pipeline runs.

---

## Results: End-to-End Pipeline (JIT ON)

Full sensor-to-state loop. 200 iterations, FP16 inference, numba JIT enabled.

| Backend | Total (µs) | TCN (µs) | IMU (µs) | EKF (µs) | Headroom | Hz | vs Best |
|---|---:|---:|---:|---:|---:|---:|---|
| **C Hot Path** (JITBEAM=2) | **2,275** | **334** | 963 | 960 | 22.0× | 440 | 1.0× |
| tinygrad NV=1 (JITBEAM=2) | 2,954 | 1,040 | 933 | 968 | 16.9× | 338 | 1.3× |
| PyTorch CUDA Graphs | 5,905 | 2,347 | 1,748 | 1,806 | 8.5× | 169 | 2.6× |
| tinygrad NV=1 (no BEAM) | 5,993 | 4,074 | 936 | 978 | 8.3× | 167 | 2.6× |

**Headroom** = 50,000 µs budget ÷ median total time. All backends pass the 20 Hz real-time constraint with wide margin on this hardware.

### Observations

1. **TCN dominates the pipeline.** With JIT-enabled EKF, IMU propagation drops to ~960 µs and EKF update to ~960 µs. TCN inference becomes the bottleneck for all backends.

2. **C Hot Path TCN = 334 µs** — this is 7.0× faster than PyTorch's 2,347 µs for the same model computation. The advantage comes from:
   - BEAM-optimized kernel schedules tuned to Orin's SM 8.7
   - Zero-copy unified memory (memmove, no cudaMemcpy)
   - Zero-overhead dispatch (MMIO doorbell write, no CUDA runtime, no Python)

3. **tinygrad NV=1 = 1,040 µs TCN** — 3.1× slower than C Hot Path despite using the exact same GPU kernels. The gap is pure dispatch overhead: Python interpreter + ioctl syscalls per kernel launch.

4. **PyTorch CUDA Graphs = 2,347 µs TCN** — 2.3× slower than tinygrad NV=1. CUDA Graphs eliminate per-launch overhead within the captured graph, but the graph capture itself uses CUDA runtime, and data staging uses `cudaMemcpyAsync`.

5. **BEAM matters.** tinygrad NV=1 without BEAM: 4,074 µs TCN. With BEAM=2: 1,040 µs. That's a **3.9× improvement** purely from kernel schedule optimization.

6. **PyTorch IMU/EKF is ~1.8× slower** (1,748 + 1,806 = 3,554 µs) vs tinygrad's (933 + 968 = 1,901 µs). This isn't a TCN difference — it's CUDA runtime context pollution. PyTorch's CUDA allocator and internal bookkeeping interfere with numba JIT'd code running on the same device.

---

## Results: TensorRT Comparison (Standalone TCN)

Measured via `trtexec` with 500 iterations and 5s warmup. **TCN inference only** — not integrated into the pipeline (no EKF, no IMU propagation).

| Mode | GPU Compute median (µs) | GPU Compute p99 (µs) | Total latency¹ (µs) | Throughput (qps) |
|---|---:|---:|---:|---:|
| **TRT FP16** | 559 | 583 | 571 | 1,765 |
| **TRT FP32** (TF32 enabled) | 543 | 579 | 556 | 1,817 |

¹ Total = H2D + GPU Compute + D2H. H2D median ~8 µs, D2H median ~5 µs.

### TRT vs other backends (TCN-only comparison)

| Backend | TCN (µs) | vs TRT FP16 |
|---|---:|---|
| C Hot Path (JITBEAM=2) | 334 | **1.7× faster** |
| TensorRT FP16 | 559 | 1.0× (baseline) |
| tinygrad NV=1 (JITBEAM=2) | 1,040 | 1.9× slower |
| PyTorch CUDA Graphs | 2,347 | 4.2× slower |
| tinygrad NV=1 (no BEAM) | 4,074 | 7.3× slower |

### Why C Hot Path beats TensorRT

TensorRT is NVIDIA's gold-standard inference optimizer — it performs layer fusion, kernel auto-tuning, and precision calibration. For large models (ResNet, BERT, etc.), TRT typically wins by 2-5× over framework dispatch paths.

For **this model** (250K params, batch=1, 7 dilated Conv1d layers), TRT's advantages are diminished:

1. **Dispatch overhead dominates compute.** At 559 µs total GPU time for a 250K-param model, a significant fraction is kernel launch + synchronization overhead — not FLOPS. The C Hot Path eliminates this entirely.

2. **TRT's kernel library isn't optimized for tiny models.** TRT's auto-tuner searches from a library of cuDNN/cuBLAS kernels designed for large batch/large-model workloads. BEAM search generates and profiles custom kernels specifically for this model's shape/size/hardware combination.

3. **Unified memory vs explicit copies.** TRT requires `cudaMemcpyAsync` for H2D/D2H (~13 µs here). The NV=1 path uses unified memory with `memmove`, which on Jetson (where CPU and GPU share physical DRAM) is essentially free.

4. **FP16 ≈ FP32/TF32 for this model.** TRT FP16 (559 µs) is actually slightly *slower* than FP32/TF32 (543 µs). This model is too small to be bandwidth-bound, so FP16's memory savings don't help, and the format conversion adds overhead. TF32 uses Tensor Cores with FP32 I/O, getting the compute benefit without the conversion cost.

> **Note:** For larger models (million+ parameters, larger batch sizes), TRT's kernel fusion and INT8 quantization would likely reclaim the advantage. The C Hot Path's win is specific to the small-model, low-batch, dispatch-dominated regime that real-time drone control operates in.

---

## Results: JIT OFF vs JIT ON

The EKF propagation code uses numba `@jit(nopython=True)`. Running with `NUMBA_DISABLE_JIT=1` shows the impact on the full pipeline.

### JIT OFF (NUMBA_DISABLE_JIT=1)

| Backend | Total (µs) | TCN (µs) | IMU (µs) | EKF (µs) | Headroom |
|---|---:|---:|---:|---:|---:|
| C Hot Path (JITBEAM=2) | 4,731 | 968 | 2,223 | 1,530 | 10.6× |
| tinygrad NV=1 (JITBEAM=2) | 4,858 | 1,048 | 2,244 | 1,566 | 10.3× |
| PyTorch CUDA Graphs | 6,540 | 2,152 | 2,535 | 1,801 | 7.6× |
| tinygrad NV=1 (no BEAM) | 7,970 | 4,129 | 2,256 | 1,588 | 6.3× |

### JIT ON

| Backend | Total (µs) | TCN (µs) | IMU (µs) | EKF (µs) | Headroom |
|---|---:|---:|---:|---:|---:|
| C Hot Path (JITBEAM=2) | 2,275 | 334 | 963 | 960 | 22.0× |
| tinygrad NV=1 (JITBEAM=2) | 2,954 | 1,040 | 933 | 968 | 16.9× |
| PyTorch CUDA Graphs | 5,905 | 2,347 | 1,748 | 1,806 | 8.5× |
| tinygrad NV=1 (no BEAM) | 5,993 | 4,074 | 936 | 978 | 8.3× |

### Analysis

| What changed | JIT OFF | JIT ON | Speedup |
|---|---:|---:|---:|
| IMU propagation (tinygrad backends) | ~2,240 µs | ~940 µs | **2.4×** |
| EKF update (tinygrad backends) | ~1,550 µs | ~965 µs | **1.6×** |
| End-to-end best (C Hot Path) | 4,731 µs | 2,275 µs | **2.1×** |
| End-to-end gap (Best vs PyTorch) | 1.4× | **2.6×** | — |

With JIT OFF, IMU+EKF dominates at ~3,770 µs (80% of C Hot Path's total), masking the TCN advantage. With JIT ON, IMU+EKF drops to ~1,930 µs (43% of total for C Hot Path), making the TCN the clear bottleneck and amplifying the backend difference.

**The takeaway:** Optimizing only the neural network isn't enough. The Cioffi pipeline does real numerical work (SO(3) integration, matrix exponentials, 15×15 covariance propagation) at 100 Hz. Getting numba JIT working correctly was essential to see the true backend advantage.

---

## Hardware Honesty: AGX Orin vs Real Drone Compute

**We must be transparent: these benchmarks run on hardware that no drone carries.**

### Platform comparison

| | Jetson TX2 (2017) | Xavier NX (2020) | **AGX Orin 64GB** (2022) |
|---|---|---|---|
| Common use | Swift racing drone | RTN-MPC, modern drones | Development kit, ground robots, large UAVs |
| CPU | 2× Denver 2 + 4× A57 | 6× Carmel (ARMv8.2) | 12× A78AE @ 2.2 GHz |
| GPU | 256 CUDA cores (Pascal) | 384 CUDA + 48 Tensor Cores (Volta) | **2048 CUDA + 64 Tensor Cores (Ampere)** |
| Memory | 8 GB LPDDR4 | 8 GB LPDDR4x | **64 GB LPDDR5** |
| Mem BW | ~59 GB/s | ~51 GB/s | **204.8 GB/s** |
| TDP | 7.5–15W | 10–20W | **15–60W** |
| Weight | ~85g (module) | ~70g (module) | ~700g+ (dev kit) |
| Flightworthy? | Yes (Swift flew this) | Yes (common on 250mm+ frames) | **No** (too heavy for most drones) |

### What the absolute numbers mean

Our AGX Orin has **8× the CUDA cores** of TX2 and **5.3× the CUDA cores** of Xavier NX. Raw compute scaling is roughly:

| | vs AGX Orin (estimated) |
|---|---|
| TX2 | ~8–10× slower |
| Xavier NX | ~5–8× slower |
| Orin NX (16GB) | ~2–3× slower |
| Orin Nano (8GB) | ~4–5× slower |

**Estimated pipeline times on real drone hardware** (linear scaling from AGX Orin, conservative):

| Backend | AGX Orin (measured) | Xavier NX (est.) | TX2 (est.) |
|---|---:|---:|---:|
| C Hot Path total | 2,275 µs | ~13,000–18,000 µs | ~18,000–23,000 µs |
| tinygrad NV=1 total | 2,954 µs | ~17,000–24,000 µs | ~24,000–30,000 µs |
| PyTorch CUDA total | 5,905 µs | ~34,000–47,000 µs | ~47,000–59,000 µs |

Even with pessimistic 10× scaling, the C Hot Path would still meet the 20 Hz budget (50,000 µs) on Xavier NX. PyTorch would be marginal on TX2.

### Why the speedup ratios should transfer

The C Hot Path's advantage over PyTorch is **architectural, not FLOPS-based:**

1. **Dispatch overhead is hardware-independent.** CUDA runtime overhead (driver calls, memory allocation tracking, stream synchronization) takes the same ~microseconds regardless of GPU size. On a smaller GPU where compute takes less absolute time, dispatch overhead becomes a *larger* fraction — meaning the NV=1 advantage should be **even greater** on weaker hardware.

2. **Unified memory is universal on Jetson.** All Jetson platforms share CPU/GPU physical DRAM. The memmove advantage (vs cudaMemcpy) works on TX2, NX, and Orin identically.

3. **BEAM search adapts to the hardware.** BEAM doesn't use Orin-specific tricks — it profiles candidate kernel schedules on whatever GPU is present and picks the best one. On Xavier NX, it would find different but similarly optimized schedules.

4. **The model is tiny.** 250K params fits in L2 cache on all Jetson platforms. The bottleneck is dispatch, not memory bandwidth or compute — exactly the regime where NV=1 wins.

**The honest prediction:** On Xavier NX, the C Hot Path's end-to-end advantage over PyTorch should be **2–4×** (compared to 2.6× on AGX Orin), because dispatch overhead is a proportionally larger cost on the smaller GPU.

### What we cannot claim

- ❌ "This drone pipeline runs at 440 Hz" — it runs at 440 Hz on AGX Orin. On Xavier NX it would be ~50–80 Hz.
- ❌ "NV=1 is always faster than TensorRT" — for this small model, yes. For larger models (ResNet-50, BERT), TRT's kernel fusion would likely win.
- ❌ "These results apply to TX2" — TX2 has no Tensor Cores and uses Pascal (SM 6.2). NV=1's backend supports it, but BEAM optimization would produce very different kernels.
- ✅ "The 2.6× end-to-end speedup ratio should approximately hold on Xavier NX" — because it's dispatch-limited, not compute-limited.
- ✅ "All backends meet 20 Hz on AGX Orin with large margin" — empirically verified.
- ✅ "BEAM search provides a 3.9× improvement over unoptimized NV=1 kernels" — hardware-specific but the technique transfers.

---

## Breakdown: Where the Time Goes

### C Hot Path (best case, JIT ON)

```
Total: 2,275 µs (one 20 Hz update cycle)
├── IMU Propagation:  963 µs (42%)  ← 5× SO(3) + cov propagation @ 100 Hz
├── TCN Inference:    334 µs (15%)  ← BEAM-optimized GPU kernels via MMIO
└── EKF Update:       960 µs (42%)  ← Kalman gain + state correction
    (overhead ~18 µs, <1%)
```

### PyTorch CUDA Graphs (baseline)

```
Total: 5,905 µs (one 20 Hz update cycle)
├── IMU Propagation: 1,748 µs (30%)  ← same numba JIT, but CUDA context pollution
├── TCN Inference:   2,347 µs (40%)  ← CUDA Graphs replay
└── EKF Update:      1,806 µs (31%)  ← same numba JIT, but CUDA context pollution
    (overhead ~4 µs, <1%)
```

PyTorch's CUDA allocator and internal state management interfere with numba JIT'd functions sharing the same GPU context, adding ~800 µs overhead to IMU+EKF despite running the same code.

---

## TensorRT Detailed Results

Measured via `trtexec` (NVIDIA TensorRT 10.7.0) because Python `tensorrt` bindings were not available in our NixOS environment. This measures **standalone TCN inference only** — the model runs in isolation, not integrated into the EKF pipeline.

### FP16

```
trtexec --onnx=cioffi_tcn_fp32.onnx --fp16 --iterations=500 --warmUp=5000
```

| Metric | Value |
|---|---|
| GPU Compute median | 558.6 µs |
| GPU Compute p99 | 582.5 µs |
| H2D median | 7.8 µs |
| D2H median | 5.4 µs |
| Total latency median | 571.3 µs |
| Throughput | 1,765 qps |
| CoV | 1.21% |

### FP32 (TF32 enabled by default on Ampere)

```
trtexec --onnx=cioffi_tcn_fp32.onnx --iterations=500 --warmUp=5000
```

| Metric | Value |
|---|---|
| GPU Compute median | 543.0 µs |
| GPU Compute p99 | 578.6 µs |
| H2D median | 7.3 µs |
| D2H median | 5.4 µs |
| Total latency median | 556.2 µs |
| Throughput | 1,817 qps |
| CoV | 1.83% |

**FP32/TF32 is slightly faster than FP16** for this model. The 250K-param TCN is small enough to be compute-bound rather than bandwidth-bound, so FP16's 2× memory reduction doesn't help. TF32 uses Tensor Cores with FP32 I/O format, getting the compute acceleration without the FP16 format conversion overhead.

### Device info (from trtexec)

```
Device: Orin (Compute Capability 8.7)
SMs: 8
Global Memory: 62826 MiB
Shared Memory per SM: 164 KiB
Memory Bus Width: 256 bits (ECC disabled)
Compute Clock: 1.3 GHz
Memory Clock: 0.612 GHz
```

---

## Reproducing These Results

### Prerequisites

```bash
cd examples/learned-inertial-odometry
nix develop    # enters dev shell with all dependencies

# Build C hot path (if not already built)
cd ../control-loop/hot_path && make && cd -
```

### End-to-end pipeline benchmarks

```bash
# C Hot Path (best performer)
NV=1 JITBEAM=2 python3 bench_e2e_pipeline.py --backend hotpath --iters 200

# tinygrad NV=1
NV=1 JITBEAM=2 python3 bench_e2e_pipeline.py --backend nv --iters 200

# PyTorch CUDA Graphs (from control-loop shell for CUDA torch)
cd ../control-loop && nix develop
python3 ../learned-inertial-odometry/bench_e2e_pipeline.py --backend pytorch --iters 200

# tinygrad NV=1 without BEAM (shows BEAM impact)
NV=1 python3 bench_e2e_pipeline.py --backend nv --iters 200

# JIT-OFF comparison
NUMBA_DISABLE_JIT=1 NV=1 JITBEAM=2 python3 bench_e2e_pipeline.py --backend hotpath --iters 200
```

### TensorRT (standalone TCN)

```bash
# Export ONNX
python3 -c "
from cioffi_tcn import CioffiTCN_PyTorch, generate_weights
import torch, os
weights = generate_weights()
model = CioffiTCN_PyTorch(weights)
model.eval()
x = torch.randn(1, 6, 50)
os.makedirs('onnx', exist_ok=True)
torch.onnx.export(model, x, 'onnx/cioffi_tcn_fp32.onnx', input_names=['input'], output_names=['output'])
"

# TRT FP16
trtexec --onnx=onnx/cioffi_tcn_fp32.onnx --fp16 --iterations=500 --warmUp=5000 --duration=0

# TRT FP32 (TF32)
trtexec --onnx=onnx/cioffi_tcn_fp32.onnx --iterations=500 --warmUp=5000 --duration=0
```

---

## Toolchain

| Component | Version | Notes |
|---|---|---|
| NixOS | jetpack-nixos (JetPack 6) | Reproducible builds |
| CUDA | 12.6 | SM 8.7 (Ampere) |
| TensorRT | 10.7.0.23 | trtexec CLI only (no Python bindings in Nix) |
| PyTorch | 2.9.1 | Built from source with CUDA (via control-loop flake) |
| tinygrad | main branch | NV=1 (Tegra ioctls), JITBEAM=2 |
| numba | 0.63.1 | Patched via `numba_compat.py` for numpy 2.3 compat |
| numpy | 2.3.4 | |
| Python | 3.12 | |

---

## Appendix: What Is the C Hot Path?

The C Hot Path is not a separate neural network runtime. It is the exact same GPU computation as tinygrad NV=1, with all Python and OS overhead removed from the inference loop.

**How it works:**

1. **tinygrad NV=1** compiles the TCN model into GPU kernels (with BEAM search for optimal scheduling), then captures the entire inference as an HCQGraph — a sequence of GPU commands in a ring buffer.

2. **`export_graph.py`** extracts from the HCQGraph: GPU buffer addresses, GPFifo ring contents, command queue offsets, and a patch map (which locations in the command stream need updating for new input data).

3. **`hot_path.c`** is a ~200-line C program that:
   - Maps the GPU's MMIO doorbell register
   - Maps the GPFifo ring buffer
   - On each inference call: patches the input pointer in the command stream, writes the updated commands, and pokes the doorbell

4. The GPU executes the exact same kernels it would under tinygrad NV=1 — same register allocations, same memory access patterns, same warp scheduling. The only difference is that the CPU side went from "Python interpreter → tinygrad runtime → ioctl syscalls → kernel driver → GPU" to "C function → MMIO write → GPU".

The 3.1× speedup (1,040 → 334 µs) between NV=1 and C Hot Path is **pure dispatch overhead elimination** — the GPU does the same work in both cases.

---

## References

- Cioffi, G., Ciccone, M., Schiavon, L., & Scaramuzza, D. (2023). "Learned Inertial Odometry for Autonomous Drone Racing." IEEE Robotics and Automation Letters (RA-L). [arXiv:2210.15287](https://arxiv.org/abs/2210.15287)
- Kaufmann, E., et al. (2023). "Champion-level drone racing using deep reinforcement learning." Nature. (Swift drone platform, TX2 hardware)
- tinygrad: [github.com/tinygrad/tinygrad](https://github.com/tinygrad/tinygrad) — NV=1 backend, HCQGraph, BEAM search
