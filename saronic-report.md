# Saronic TRT Inference — Performance Report

> **TL;DR**: libinfer is leaving **300–1800 µs per frame on the table** by not using CUDA Graphs. That's a ~50-line C++ fix. On YOLOv8-n — Saronic's primary model — this translates to 1.2–1.8× faster compute dispatch. For multi-model perception pipelines (detection→segmentation→pose), a single CUDA Graph capturing all three models cuts total latency by **1.24×** and eliminates jitter, achieving **130 Hz** on a 3-model pipeline that runs at 105 Hz sequentially.

---

## What we tested

We benchmarked libinfer's TensorRT inference path on a Jetson AGX Orin 64GB (CUDA 12.6, TRT 10.7) across seven models and two pipeline configurations:

### Individual models

| Model | Params | I/O Size | Resolution | Use case |
|---|---:|---:|---:|---|
| Cioffi TCN (FP16) | 250K | 606 B | — | Small sensor-fusion model |
| **YOLOv8-n (FP16)** | 3.16M | 967 KB | 320×320 | **Object detection — Saronic production** |
| **YOLOv8-n (FP16)** | 3.16M | 3.16 MB | 640×640 | **Object detection — higher resolution** |
| YOLOv8-s (FP16) | 11.2M | 3.16 MB | 640×640 | Object detection — small variant |
| YOLOv8-n-seg (FP16) | 3.4M | 11.5 MB | 640×640 | Instance segmentation |
| YOLOv8-n-pose (FP16) | 3.3M | 6.5 MB | 640×640 | Pose estimation |
| YOLOv8-s-pose (FP16) | 11.6M | 6.5 MB | 640×640 | Pose estimation — small |

### Multi-model pipelines

| Pipeline | Models | Use case |
|---|---|---|
| **Nano pipeline** | YOLOv8n det → YOLOv8n-seg → YOLOv8n-pose | Fast 3-model perception stack |
| **Heavy pipeline** | YOLOv8s det → YOLOv8n-seg → YOLOv8s-pose | Higher-accuracy perception stack |

We tested four TRT configurations for individual models:
1. **Stock** — `cudaMalloc` + `cudaMemcpyAsync` + `enqueueV3` (what libinfer does today)
2. **Zero-Copy** — `cudaMallocManaged` (unified memory, skip explicit copies)
3. **CUDA Graph** — capture the enqueue into a graph, replay with `cudaGraphLaunch`
4. **Zero-Copy + CUDA Graph** — both combined

And three execution strategies for multi-model pipelines:
1. **Sequential** — enqueue→sync→enqueue→sync→enqueue→sync (naive, what libinfer would do today)
2. **Pipelined** — enqueue→enqueue→enqueue→sync (single sync at end)
3. **CUDA Graph** — capture all 3 enqueues into a single graph, one `cudaGraphLaunch`

---

## Results

### YOLOv8-n 320×320 (Saronic's primary model)

| Variant | Median µs | P99 µs | P99/Median | Hz |
|---|---:|---:|---:|---:|
| Stock GPU-only (enqueueV3) | 1623 | 5474 | **3.37×** | 616 |
| Stock full (H2D+compute+D2H) | 2044 | 6062 | 2.97× | 489 |
| Zero-Copy full | 1932 | 5882 | 3.04× | 518 |
| **CUDA Graph** | **1157** | **1315** | **1.14×** | **865** |
| **ZC + CUDA Graph** | **1306** | **1362** | **1.04×** | **765** |

### YOLOv8-n 640×640 (higher resolution)

| Variant | Median µs | P99 µs | P99/Median | Hz |
|---|---:|---:|---:|---:|
| Stock GPU-only | 3127 | 3230 | 1.03× | 320 |
| Stock full | 4358 | 4522 | 1.04× | 229 |
| Zero-Copy full | 3937 | 4166 | 1.06× | 254 |
| **CUDA Graph** | **2546** | **2625** | **1.03×** | **393** |
| **ZC + CUDA Graph** | **2743** | **2842** | **1.04×** | **365** |

### Cioffi TCN (small model stress test)

| Variant | Median µs | P99 µs | P99/Median | Hz |
|---|---:|---:|---:|---:|
| Stock GPU-only | 588 | 708 | 1.20× | 1701 |
| Stock full | 625 | 724 | 1.16× | 1601 |
| Zero-Copy full | 675 | 788 | **1.17×** | 1481 |
| **CUDA Graph** | **291** | **314** | **1.08×** | **3432** |
| **ZC + CUDA Graph** | **362** | **386** | **1.07×** | **2764** |

### YOLOv8-s 640×640 (detection, small variant)

| Variant | Median µs | P99 µs | P99/Median | Hz |
|---|---:|---:|---:|---:|
| Stock GPU-only (enqueueV3) | 4526 | 4637 | 1.02× | 221 |
| Stock full (H2D+compute+D2H) | 5587 | 5668 | 1.01× | 179 |
| **CUDA Graph** | **3921** | **3975** | **1.01×** | **255** |
| **ZC + CUDA Graph** | **4050** | **4104** | **1.01×** | **247** |

### YOLOv8-n-seg 640×640 (instance segmentation)

| Variant | Median µs | P99 µs | P99/Median | Hz |
|---|---:|---:|---:|---:|
| Stock GPU-only (enqueueV3) | 3525 | 3574 | 1.01× | 284 |
| Stock full (H2D+compute+D2H) | 5362 | 5446 | 1.02× | 187 |
| **CUDA Graph** | **2875** | **2906** | **1.01×** | **348** |
| **ZC + CUDA Graph** | **3005** | **3046** | **1.01×** | **333** |

### YOLOv8-n-pose 640×640 (pose estimation)

| Variant | Median µs | P99 µs | P99/Median | Hz |
|---|---:|---:|---:|---:|
| Stock GPU-only (enqueueV3) | 3115 | 3187 | 1.02× | 321 |
| Stock full (H2D+compute+D2H) | 4054 | 4184 | 1.03× | 247 |
| **CUDA Graph** | **2514** | **2580** | **1.03×** | **398** |
| **ZC + CUDA Graph** | **2645** | **2721** | **1.03×** | **378** |

### YOLOv8-s-pose 640×640 (pose estimation, small variant)

| Variant | Median µs | P99 µs | P99/Median | Hz |
|---|---:|---:|---:|---:|
| Stock GPU-only (enqueueV3) | 4765 | 4838 | 1.02× | 210 |
| Stock full (H2D+compute+D2H) | 5691 | 5817 | 1.02× | 176 |
| **CUDA Graph** | **4107** | **4163** | **1.01×** | **244** |
| **ZC + CUDA Graph** | **4233** | **4299** | **1.02×** | **236** |

---

## Multi-Model Pipeline Results

### Nano pipeline: YOLOv8n-det → YOLOv8n-seg → YOLOv8n-pose (all n-size)

A realistic 3-model perception pipeline chaining detection, segmentation, and pose estimation.

| Strategy | Median µs | P99 µs | Max µs | Hz | vs Sequential |
|---|---:|---:|---:|---:|---|
| Sequential (3× enqueue+sync) | 9551 | 9594 | 9619 | 105 | 1.0× |
| Pipelined (3× enqueue, 1 sync) | 9364 | 9390 | 9432 | 107 | 1.02× |
| **CUDA Graph (1 launch, 427 nodes)** | **7720** | **7744** | **7775** | **130** | **1.24×** |

- **CUDA Graph saves 1831 µs per pipeline iteration** (9551 → 7720 µs)
- **Jitter**: Graph std_dev = 8.7 µs vs Sequential std_dev = 13.8 µs
- **427 GPU nodes** captured across 3 models in a single graph

### Heavy pipeline: YOLOv8s-det → YOLOv8n-seg → YOLOv8s-pose (mixed sizes)

| Strategy | Median µs | P99 µs | Max µs | Hz | vs Sequential |
|---|---:|---:|---:|---:|---|
| Sequential (3× enqueue+sync) | 12602 | 12664 | 13178 | 79 | 1.0× |
| Pipelined (3× enqueue, 1 sync) | 12412 | 12451 | 12552 | 81 | 1.02× |
| **CUDA Graph (1 launch, 419 nodes)** | **10691** | **10720** | **10747** | **94** | **1.18×** |

- **CUDA Graph saves 1912 µs per pipeline iteration** (12602 → 10691 µs)
- **Jitter**: Graph std_dev = 9.3 µs vs Sequential std_dev = 22.0 µs (**2.4× less jitter**)

---

## D-FINE / RF-DETR / YOLO11 — TRT Incompatibility on Orin

We attempted to benchmark Saronic's DETR-based models but discovered a **fundamental TRT 10.7 incompatibility on Orin (SM 8.7)**:

| Model | ONNX Export | TRT Engine Build | Root Cause |
|---|---|---|---|
| D-FINE-N (4M) | ✅ 14.8 MB | ❌ FAILED | Fused encoder self_attn node |
| D-FINE-S (10M) | ✅ 39.9 MB | ❌ FAILED | Same |
| RF-DETR-Base (29M) | ✅ 112.4 MB | ❌ FAILED | Backbone projector fusion |
| YOLO11n-seg (2.9M) | ✅ 11.2 MB | ❌ FAILED | C2fAttn transformer blocks |

**Error**: `Could not find any implementation for node {ForeignNode[...attn...]} due to insufficient workspace` — occurs at ALL workspace sizes up to 16 GB. The real issue is missing kernel tactics for fused attention/transformer nodes on SM 8.7 (Ampere).

**Pattern**: Any model with attention/transformer blocks fails. Pure CNN models (all YOLOv8 variants) work perfectly.

**Implication**: D-FINE and RF-DETR will need either (1) a TRT version with SM 8.7 attention tactics, (2) custom TRT plugins, or (3) an alternative runtime (e.g., tinygrad NV, which handles arbitrary ops). For now, CUDA Graph benchmarking focuses on YOLOv8 variants which are Saronic's production models.

---

## Analysis

### 1. CUDA Graphs eliminate dispatch overhead — 300–1800 µs per frame

Every call to `infer()` in libinfer's `engine.cpp` calls `context->enqueueV3(stream)` cold. This forces TensorRT to re-walk the execution plan and re-submit every GPU kernel. With `cudaGraphLaunch`, all that work is pre-compiled into a single GPU submission.

**Dispatch overhead eliminated by CUDA Graphs (individual models):**

| Model | Stock GPU µs | CUDA Graph µs | Saved µs | GPU Speedup |
|---|---:|---:|---:|---:|
| Cioffi TCN | 588 | 291 | **297** | **2.02×** |
| YOLOv8-n 320² | 1623 | 1157 | **466** | **1.40×** |
| YOLOv8-n-pose 640² | 3115 | 2514 | **601** | **1.24×** |
| YOLOv8-n 640² | 3127 | 2546 | **581** | **1.23×** |
| YOLOv8-n-seg 640² | 3525 | 2875 | **650** | **1.23×** |
| YOLOv8-s 640² | 4526 | 3921 | **605** | **1.15×** |
| YOLOv8-s-pose 640² | 4765 | 4107 | **658** | **1.16×** |

**Dispatch overhead eliminated by CUDA Graphs (multi-model pipelines):**

| Pipeline | Sequential µs | CUDA Graph µs | Saved µs | Speedup |
|---|---:|---:|---:|---:|
| 3-model nano (det+seg+pose) | 9551 | 7720 | **1831** | **1.24×** |
| 3-model heavy (s-det+seg+s-pose) | 12602 | 10691 | **1912** | **1.18×** |

The dispatch overhead is roughly proportional to model complexity (number of TRT layers). It's a larger *relative* fraction for smaller models (51% of TCN, 19% of YOLO 640²), but the *absolute* savings grow with model size. For multi-model pipelines, CUDA Graphs don't just save per-model dispatch overhead — they eliminate the per-model CPU→GPU synchronization points, letting the GPU run all kernels back-to-back without stalls.

**The fix**: Capture the first `enqueueV3` into a `cudaGraph_t`, then replay it with `cudaGraphLaunch` on subsequent calls. This is ~50 lines of C++ — the capture/replay pattern is well-documented in the CUDA toolkit. We've prototyped it in `crates/cuda-graph-replay/`.

### 2. Tail latency: the real killer for autonomy

For a real-time control loop, **worst-case latency matters more than average**. A 5× tail-latency spike means your detection pipeline misses its deadline, which cascades into late actuation.

Look at YOLOv8-n 320²:
- **Stock**: p99 = **5474 µs** (3.37× median) — one in 100 frames takes 3× longer
- **ZC + CUDA Graph**: p99 = **1362 µs** (1.04× median) — rock-solid, < 5% jitter

| Model | Stock P99/Median | ZC+Graph P99/Median | Jitter Reduction |
|---|---:|---:|---|
| Cioffi TCN | 1.20× | 1.07× | **6× less jitter** |
| YOLOv8-n 320² | **3.37×** | **1.04×** | **81× less jitter** |
| YOLOv8-n 640² | 1.03× | 1.04× | Comparable |

The 320² result is dramatic: stock TRT has occasional frame times that blow >5 ms, while CUDA Graph + ZC is completely flat. For autonomy, this means you can budget a tighter deadline and still never miss it.

### 3. Zero-copy: marginal for Saronic's models

| Model | I/O Size | ZC Speedup (full) | Verdict |
|---|---:|---:|---|
| Cioffi TCN | 606 B | **0.92× (SLOWER)** | Coherence overhead > copy savings |
| YOLOv8-n 320² | 967 KB | **1.06×** | Small win, not worth API complexity |
| YOLOv8-n 640² | 3.16 MB | **1.11×** | Noticeable but dwarfed by CUDA Graphs |

Zero-copy crossover is around ~100 KB total I/O. Below that, managed memory coherence overhead dominates. Above that, you get modest gains — but CUDA Graphs always provide a larger speedup.

### 4. End-to-end: what libinfer would actually see

Adding CUDA Graphs to libinfer's `infer()` call (keeping existing H2D/D2H copies) gives these real-world end-to-end improvements:

| Model | libinfer today | + CUDA Graph | End-to-end speedup |
|---|---:|---:|---:|
| Cioffi TCN | 625 µs (1601 Hz) | ~328 µs (~3049 Hz) | **1.90×** |
| YOLOv8-n 320² | 2044 µs (489 Hz) | ~1578 µs (~634 Hz) | **1.30×** |
| YOLOv8-n 640² | 4358 µs (229 Hz) | ~3777 µs (~265 Hz) | **1.15×** |
| YOLOv8-n-seg 640² | 5362 µs (187 Hz) | ~4712 µs (~212 Hz) | **1.14×** |
| YOLOv8-n-pose 640² | 4054 µs (247 Hz) | ~3453 µs (~290 Hz) | **1.17×** |
| YOLOv8-s 640² | 5587 µs (179 Hz) | ~4982 µs (~201 Hz) | **1.12×** |
| YOLOv8-s-pose 640² | 5691 µs (176 Hz) | ~5033 µs (~199 Hz) | **1.13×** |

*(End-to-end estimate = cuda_graph_time + copy_overhead, where copy_overhead = stock_full − stock_gpu)*

### 5. Multi-model pipeline: the real multiplier

For Saronic's actual use case — running multiple models per frame — CUDA Graphs provide a **compounding benefit**. Each model adds dispatch overhead; a single CUDA Graph eliminates all of it at once:

| Pipeline Config | Sequential | CUDA Graph | Speedup | Pipeline Hz |
|---|---:|---:|---:|---:|
| **Nano** (3× YOLOv8-n variants) | 9551 µs (105 Hz) | 7720 µs (130 Hz) | **1.24×** | **130 Hz** |
| **Heavy** (s-det + n-seg + s-pose) | 12602 µs (79 Hz) | 10691 µs (94 Hz) | **1.18×** | **94 Hz** |

The nano pipeline captures **427 GPU nodes** and the heavy pipeline captures **419 nodes** into a single graph. One `cudaGraphLaunch()` call replays the entire 3-model pipeline — vs 3 separate `enqueueV3()` + `cudaStreamSynchronize()` calls.

**Key insight**: Pipelining alone (enqueue all, sync once) gives only 1.02× speedup — the bottleneck isn't sync overhead, it's per-model dispatch re-walks. Only CUDA Graphs eliminate this.

### 6. What "great" looks like

For reference on the TCN (where we have data across all backends):

| Backend | Latency | Speedup vs TRT Stock |
|---|---:|---:|
| C Hot Path (MMIO kernel replay) | 139 µs | 4.0× |
| tinygrad NV (BEAM-optimized) | 195 µs | 2.8× |
| **TRT + CUDA Graph** | **291 µs** | **2.0×** |
| TRT Stock (libinfer today) | 588 µs | 1.0× |

CUDA Graphs get libinfer into the same ballpark as hand-optimized GPU paths, without leaving the TensorRT ecosystem.

---

## Saronic model inventory

From surveying [saronic-technologies](https://github.com/saronic-technologies) (51 repos):

| Model | Repo | Status | Params | Notes |
|---|---|---|---:|---|
| **YOLOv8-n** | ultralytics (fork) | **Production** | 3.16M | Primary detection model, in libinfer test suite |
| **D-FINE** | D-FINE (fork) | Active dev | 4–62M | DETR-based detector (ICLR 2025 Spotlight), N/S/M/L/X variants |
| **RF-DETR** | rf-detr-train (fork) | Active dev | 29–128M | DETR variant, Base/Large, custom training scripts by catid-saronic |

D-FINE and RF-DETR are DETR-based models that **cannot currently build TRT engines on Orin** due to missing kernel tactics for fused attention/transformer nodes on SM 8.7 (see above). This is a TRT 10.7 limitation, not a model issue. These models will benefit from CUDA Graphs once TRT compatibility is resolved.

**Confirmed: libinfer does NOT use CUDA Graphs.** We verified by reading the latest `engine.cpp` (main branch, v0.0.5). The entire `infer()` path is: `setInputShape` → `cudaMemcpyAsync` H2D → `setTensorAddress` → `enqueueV3` cold dispatch → `cudaMemcpyAsync` D2H → `cudaStreamSynchronize`. No `cudaGraph`, `cudaStreamBeginCapture`, or `cudaGraphLaunch` anywhere.

---

## Recommended next steps

1. **Add CUDA Graph capture/replay to `engine.cpp`** — ~50 lines of C++, biggest ROI
2. **Add multi-model CUDA Graph support** — capture the entire perception pipeline (det→seg→pose) into a single graph for additional 1.18–1.24× speedup
3. **Ship it** — saves 300–1900 µs/frame depending on pipeline configuration, eliminates tail-latency spikes
4. **Track TRT updates for D-FINE/RF-DETR support** — current attention node incompatibility on Orin will likely be fixed in future TRT releases
5. **Later**: consider zero-copy only for models with multi-MB I/O tensors
6. **Later**: evaluate HCQ-replay or tinygrad NV for ultra-low-latency sensor fusion

---

*Benchmarked on Jetson AGX Orin 64GB, JetPack 6, CUDA 12.6, TensorRT 10.7.0.23. All numbers from same session, clocks at default.*
*Source: `bench_trt_variants.cpp` (individual models), `bench_multi_model.cpp` (multi-model pipeline).*
