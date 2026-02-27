# Presentation Benchmarks: tinygrad NV=1 vs TensorRT

**Hardware:** Jetson AGX Orin 64GB, JetPack 6 / CUDA 12.6  
**Goal:** Find where tinygrad NV=1 wins, where TensorRT wins, and what that means for real-world robotics.

## The Question

tinygrad NV=1 bypasses the CUDA runtime entirely on Tegra — talking directly to the GPU via raw nvgpu/nvmap ioctls. We've shown it beats PyTorch CUDA Graphs by 1.85x for small MLP control loops. But **can it beat TensorRT?** TensorRT is NVIDIA's own inference optimizer with cuDNN/cuBLAS kernels and layer fusion.

This benchmark answers: **for which architectures and model sizes does NV=1's dispatch speed advantage outweigh TensorRT's kernel quality advantage?**

## Architectures Tested

### MLP (Multi-Layer Perceptron)
Standard for learned drone/robot controllers. 7 sizes from 5K to 2.1M params.

| Config | Params | Use Case |
|--------|--------|----------|
| 12→64→64→4 | ~5K | PID replacement, rate gyro filter |
| 12→128→128→4 | ~18K | Learned hover controller |
| 12→256→256→256→4 | ~135K | Full attitude policy |
| 12→512→512→4 | ~270K | Visual-inertial navigation |
| 12→512→512→512→4 | ~530K | GPU/NEON crossover zone |
| 12→1024→1024→4 | ~1.1M | Path planner, obstacle avoidance |
| 12→1024→1024→1024→4 | ~2.1M | Large policy, multi-agent |

### 1D-CNN (Temporal Convolution)
For extracting features from IMU time-series (16-sample window @ 1kHz = 16ms).

| Config | Params | Use Case |
|--------|--------|----------|
| Conv(32,k3)→Conv(64,k3)→FC(64)→4 | ~30K | IMU denoising |
| Conv(64,k3)→Conv(128,k3)→Conv(128,k3)→FC(128)→4 | ~150K | Temporal features for agile flight |
| Conv(128,k3)→Conv(256,k3)→Conv(256,k3)→FC(256)→FC(128)→4 | ~500K | E2E state estimation |

### Hybrid CNN+MLP
CNN processes temporal IMU window, MLP processes current state vector, fused for action output. This is the architecture pattern from modern end-to-end drone controllers.

| Config | Params | Use Case |
|--------|--------|----------|
| CNN(32,64) + MLP(128,64)→4 | ~50K | Lightweight sensor fusion |
| CNN(64,128) + MLP(256,128)→4 | ~200K | Agile flight |
| CNN(128,256,256) + MLP(512,256,128)→4 | ~700K | Full autonomy |

## Quick Start

```bash
# From repo root:
cd examples/presentation && nix develop

# Run all benchmarks (takes ~15-30 min with JITBEAM=2)
NV=1 JITBEAM=2 python3 bench_models.py

# Run just MLPs (fastest, ~5 min)
NV=1 JITBEAM=2 python3 bench_models.py --arch mlp

# Fewer iterations for quick sanity check
NV=1 JITBEAM=2 python3 bench_models.py --iters 1000

# tinygrad only (skip TensorRT)
NV=1 JITBEAM=2 python3 bench_models.py --skip-tensorrt
```

## What We Measure

For each model × framework, we measure **full inference round-trip**:
1. **H2D transfer** — copy input from CPU to GPU (memmove vs cudaMemcpy)
2. **GPU dispatch + compute** — kernel execution
3. **D2H transfer** — copy output back
4. **Sync** — wait for completion

This is the realistic control-loop measurement. For a drone flying at 4 kHz, every microsecond of this round-trip matters.

### tinygrad NV=1 (Approach C — direct memory)
- `ctypes.memmove` to `cpu_view()` for H2D (<1 µs for 24 bytes)
- `@TinyJit` with `JITBEAM=2` for kernel optimization
- HCQGraph replay for dispatch
- `dev.synchronize()` for completion
- `ctypes.memmove` for D2H (<1 µs for 8 bytes)

### TensorRT FP16
- ONNX model → TensorRT engine build (FP16 mode, layer fusion enabled)
- `cudaMemcpyAsync` H2D 
- `execute_async_v3` (optimized cuDNN/cuBLAS kernels)
- `cudaMemcpyAsync` D2H
- `cudaStreamSynchronize`

## Expected Results (Hypothesis)

Based on our previous benchmarks:

- **Small models (<100K params):** NV=1 wins. Dispatch overhead dominates GPU compute time. NV=1's zero-CUDA-runtime path saves ~100-200 µs per iteration. TensorRT's kernel optimization doesn't help when the kernel takes 5 µs.

- **Medium models (100K-500K params):** Crossover zone. GPU compute time starts to matter. TensorRT's fused kernels may close the gap.

- **Large models (>500K params):** TensorRT likely wins. cuDNN GEMM kernels and layer fusion dominate total time. NV=1's dispatch savings become proportionally insignificant.

- **CNN/Hybrid architectures:** TensorRT may have an extra advantage here — it fuses Conv+ReLU layers and uses specialized cuDNN convolution kernels. tinygrad's BEAM-optimized PTX may not match cuDNN for convolutions.

## Honest Caveats

1. **JITBEAM matters.** tinygrad's kernel quality depends on beam search. JITBEAM=2 is reasonable but JITBEAM=8 might produce better kernels (at the cost of 10x longer compilation).

2. **TensorRT builds an optimized engine.** The first run is slow (engine compilation). Subsequent runs use the cached engine. We measure only post-compilation inference.

3. **Unified memory is Tegra-only.** The `cpu_view()` trick that makes NV=1 fast for H2D/D2H only works on Jetson (shared DRAM). On desktop GPUs, both frameworks would use DMA.

4. **Batch size 1.** Real-time control loops are always batch-1. This is NV=1's sweet spot. TensorRT is optimized for larger batches too, but batch-1 is what drones need.

## Files

| File | Purpose |
|------|---------|
| `bench_models.py` | Main runner — orchestrates all benchmarks |
| `bench_tinygrad.py` | tinygrad NV=1 benchmark (Approach C direct memory) |
| `bench_tensorrt.py` | TensorRT benchmark via Python API + ONNX |
| `models.py` | Architecture definitions + weight generation |
| `flake.nix` | Nix dev shell with tinygrad + TensorRT + ONNX |
| `results.json` | Raw benchmark results (generated) |
| `weights/` | Exported model weights (generated) |
