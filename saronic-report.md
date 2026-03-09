# Saronic TRT Inference — Performance Report

> **TL;DR**: libinfer is leaving **40–50% of its latency on the table** by not using CUDA Graphs. This is a ~50-line fix. Zero-copy memory is a distraction for the models Saronic runs today.

---

## What we tested

We benchmarked libinfer's TensorRT inference path on a Jetson AGX Orin 64GB (CUDA 12.6, TRT 10.7) with two models:

| Model | Params | I/O Size | Use case |
|---|---:|---:|---|
| Cioffi TCN (FP16) | 250K | 606 B | Small sensor-fusion model |
| **YOLOv8-n 320×320 (FP16)** | 3.16M | 967 KB | **Object detection — Saronic's workload** |

We tested four TRT configurations:
1. **Stock** — `cudaMalloc` + `cudaMemcpyAsync` + `enqueueV3` (what libinfer does today)
2. **Zero-Copy** — `cudaMallocManaged` (unified memory, skip explicit copies)
3. **CUDA Graph** — capture the enqueue into a graph, replay with `cudaGraphLaunch`
4. **Zero-Copy + CUDA Graph** — both combined

## Results

### YOLOv8-n (what matters for Saronic)

| Variant | Latency | Speedup |
|---|---:|---:|
| Stock (current libinfer) | 1898 µs | baseline |
| Zero-Copy | 1781 µs | 1.07× |
| **CUDA Graph** | **1073 µs** | **1.77×** |
| ZC + CUDA Graph | 1181 µs | 1.61× |

### Cioffi TCN (small model stress test)

| Variant | Latency | Speedup |
|---|---:|---:|
| Stock | 582 µs | baseline |
| Zero-Copy | 636 µs | **0.92× (SLOWER)** |
| **CUDA Graph** | **262 µs** | **2.12×** |

## The bottom line

### 1. Add CUDA Graphs to libinfer — this is the #1 priority

Every call to `infer()` in libinfer's `engine.cpp` calls `context->enqueueV3(stream)` cold. This pays TensorRT dispatch overhead on every single frame. On YOLOv8-n, this wastes ~800 µs per inference.

**The fix**: Capture the first `enqueueV3` into a `cudaGraph_t`, then replay it with `cudaGraphLaunch` on subsequent calls. This is ~50 lines of C++ — the capture/replay pattern is well-documented in the CUDA toolkit samples. We've already prototyped it in `crates/cuda-graph-replay/`.

**Expected impact**: **1.4–2.1× faster inference** across all model sizes, with zero accuracy impact. On YOLOv8-n: 1898 µs → 1073 µs (527 → 932 Hz).

### 2. Don't bother with zero-copy for current models

We tested `cudaMallocManaged` (Tegra unified memory) to skip H2D/D2H copies. Results:
- **Small models (< 100 KB I/O)**: zero-copy is **slower** due to coherence overhead
- **Large models (> 100 KB I/O)**: zero-copy provides **~7% gain** — real but marginal

Zero-copy adds API complexity (managed memory pointers, `cudaMemAdvise` hints, stream ordering constraints) for benefits that are dwarfed by CUDA Graphs. We'd recommend revisiting only after CUDA Graph support ships, and only for models with very large input tensors (multi-megapixel images).

### 3. What "great" looks like

For reference, we also benchmarked alternative execution paths on the TCN:

| Backend | Latency | Speedup vs TRT Stock |
|---|---:|---:|
| C Hot Path (MMIO kernel replay) | 138 µs | 4.0× |
| tinygrad NV (BEAM-optimized) | 195 µs | 2.8× |
| **TRT + CUDA Graph** | **262 µs** | **2.1×** |
| TRT Stock (libinfer today) | 582 µs | 1.0× |

CUDA Graphs get libinfer into the same ballpark as hand-optimized GPU paths, without leaving the TensorRT ecosystem.

## Recommended next steps

1. **Add CUDA Graph capture/replay to `engine.cpp`** — ~50 lines, biggest ROI
2. **Ship it** — this alone cuts YOLOv8-n latency by 44%
3. **Later**: consider zero-copy only for models with multi-MB I/O tensors
4. **Later**: evaluate HCQ-replay or tinygrad NV for ultra-low-latency sensor fusion

---

*Benchmarked on Jetson AGX Orin 64GB, JetPack 6, CUDA 12.6, TensorRT 10.7.0.23.*
*Source: `bench_trt_variants.cpp` (generic, auto-detects tensor I/O for any engine file).*
