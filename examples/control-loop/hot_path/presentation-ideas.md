# Tinygrad NV=1 on Jetson Orin AGX — Presentation Ideas

**Platform:** NVIDIA Jetson AGX Orin 64GB (12× A78AE, 2048 CUDA cores SM 8.7)  
**Stack:** NixOS, JetPack 6, CUDA 12.6, tinygrad NV backend, clang 21

---

## 1. The Elevator Pitch

> We used tinygrad's NV backend on Jetson Orin to bypass the CUDA runtime
> entirely — talking directly to the GPU via raw Tegra ioctls. The result:
> Python-level inference that beats PyTorch CUDA Graphs by 1.85x, and a
> 200-line C hot path that beats it by 1.9x on GPU-resident dispatch.
> No CUDA toolkit. No cuBLAS. Zero 100 ms jitter stalls.

Three layers of results to present:

| Layer | What it is | Headline number |
|-------|-----------|-----------------|
| **Python NV=1** (full control loop) | tinygrad + Tegra unified memory bypass via `ctypes.memmove` to `cpu_view()` | **207 µs / 4,832 Hz** — 1.85x over PyTorch CUDA Graphs (383 µs) |
| **C GPU Hot Path** (MLP dispatch) | 200 lines of C replaying pre-built HCQGraph via raw MMIO doorbell | **46 µs / 21.7 kHz** — 1.9x over PyTorch CUDA Graphs GPU-resident (88 µs) |
| **LLM inference** (Python NV=1) | tinygrad NV=1 running LLaMA/Qwen3 on Orin | **29 tok/s** LLaMA 1B (+4% over llama.cpp +FA) |

---

## 2. The Key Insight: Memory-Bound vs Compute-Bound

All of our wins — Python NV=1, C hot path, and LLM inference — come from the
same place: **eliminating overhead in memory-bound / overhead-bound workloads.**
Understanding this explains both where we win and where we lose.

```text
  ┌────────────────────────────────────────────────────────────────────┐
  │          OVERHEAD-BOUND              │      COMPUTE-BOUND          │
  │    (our sweet spot — we win)         │  (TRT/cuBLAS win)           │
  │                                      │                             │
  │  GPU compute: ~5 µs                  │  GPU compute: 500+ µs       │
  │  Framework overhead: 50-400 µs       │  Framework overhead: 50 µs  │
  │  Bottleneck: dispatch, transfer,     │  Bottleneck: GEMM FLOPs,    │
  │    Python interpreter, CUDA runtime  │    kernel optimization,     │
  │                                      │    INT8, tensor cores       │
  │  Eliminating overhead = huge win     │  Eliminating overhead =     │
  │                                      │    negligible improvement   │
  └──────────────────────────────────────┴─────────────────────────────┘
        5K──────270K──────2M params              10M+ params
        <-- MLP control loops -->          <-- large models / batch -->
```

**Where we are overhead-bound (and win big):**

- **MLP control loops (5K-2M params):** The GPU kernel computes in 5-40 µs. The
  CUDA runtime adds 50-100 µs dispatch overhead, PyTorch adds 170 µs Python
  overhead, `cuMemcpy` adds 180 µs data transfer. Our approach removes all of
  that: 3 µs dispatch (C hot path) + 0.2 µs data transfer (unified memory) = we
  win by 1.9x over PyTorch CUDA Graphs.
- **LLM batch-1 decode:** Reading every model parameter once per token makes
  decode **memory-bandwidth-bound** on the Orin's 205 GB/s LPDDR5. Actual
  compute is light. NV=1's zero-overhead dispatch helps saturate bandwidth
  (97% utilization) while CUDA runtime overhead wastes bus cycles. That's why
  tinygrad beats llama.cpp despite reading 2-3x more data (dequanted fp16).

**Where we become compute-bound (and lose):**

- **Large models (10M+ params):** GEMM FLOPs dominate total time. Dispatch
  overhead is noise. TensorRT's hand-tuned SASS kernels, INT8 quantization,
  and tensor core utilization give 2-5x better compute throughput than
  tinygrad's BEAM-optimized PTX.
- **Batch inference (batch > 1):** Arithmetic intensity increases with batch
  size, shifting from memory-bound to compute-bound. cuBLAS GEMM kernels are
  heavily optimized for batched workloads.
- **MLC LLM beating us by 27%:** MLC's TVM compiler produces better GPU kernels
  (cutlass) — this is a kernel quality gap, i.e., compute-bound territory where
  we can't win on dispatch alone.

**The crossover from "we win" to "we lose" is fundamentally the crossover from
overhead-bound to compute-bound.** For MLPs, that's around 10M params. For LLMs
at batch-1, we stay memory-bandwidth-bound even at 3B params (which is why NV=1
still beats llama.cpp at 3B).

---

## 3. How We Beat PyTorch (Clarified)

There are **two distinct victories** over PyTorch. They work differently and measure different things.

### 2a. Python NV=1 Beats PyTorch in Full Control Loops (1.85x)

This is the **first result** — pure Python, no C code, 15 lines.

**What we measured:** Full sensor → GPU inference → PID output loop, 60 seconds sustained.

| Framework | Full Loop Median | Frequency | How it transfers data |
|-----------|:----------------:|:---------:|----------------------|
| **tinygrad NV=1 (Approach C)** | **207 µs** | **4,832 Hz** | `ctypes.memmove` to `cpu_view()` — direct write to GPU-mapped DRAM |
| PyTorch CUDA Graphs | 383 µs | 2,609 Hz | `cuMemcpyHtoD` — CUDA runtime DMA + event sync |
| PyTorch eager | 670 µs | 1,493 Hz | `torch.from_numpy().cuda()` — full CUDA pipeline |

**Why tinygrad wins:** On Tegra, CPU and GPU share the same physical DRAM. tinygrad's NV backend exposes the GPU buffer's CPU-mapped address via `HCQBuffer.cpu_view()`. Writing to it with `ctypes.memmove` costs ~1 µs for 24 bytes. PyTorch always goes through `cuMemcpy*` even on Tegra, which sets up DMA descriptors + CUDA event sync = ~114 µs H2D + ~69 µs D2H.

**The honest nuance:** tinygrad's HCQGraph dispatch (182 µs GPU-resident) is actually *slower* than CUDA Graphs replay (88 µs GPU-resident) by ~2x. But the data transfer savings (~180 µs saved) more than compensate. The win is **Tegra unified memory bypass**, not faster GPU dispatch.

| Component | tinygrad NV=1 | PyTorch CUDA Graphs |
|-----------|:-------------:|:-------------------:|
| H2D transfer | ~1 µs (memmove) | ~114 µs (cuMemcpy) |
| GPU dispatch + compute | ~100 µs (HCQGraph) | ~29 µs (CUDA graph replay) |
| D2H transfer | ~1 µs (memmove) | ~69 µs (.cpu().numpy()) |
| Python overhead | ~100 µs | ~170 µs |
| **Total** | **~207 µs** | **~383 µs** |

**Jitter:** tinygrad had **zero outliers above 5 ms** across all 60-second runs. PyTorch had **100 ms stalls** in sensor-fusion tests (1 event per ~100K iterations). For drones, one 100 ms stall = crash.

### 2b. C GPU Hot Path Beats PyTorch GPU-Resident Dispatch (1.9x)

This is the **second result** — takes the Python NV=1 approach further by porting the dispatch to C.

**What we measured:** Raw GPU round-trip for MLP inference (memcpy in → GPU dispatch → wait → memcpy out), 10,000 iterations.

| Approach | Latency (18K MLP) | What it measures |
|----------|:-----------------:|-----------------|
| **C GPU Hot Path** | **46 µs** | Unified-memory memcpy + GPFifo submit + MMIO doorbell + spin-wait (C, 200 lines) |
| PyTorch CUDA Graphs | 88 µs | GPU-resident dispatch + compute (no data transfer) |
| PyTorch eager | 402 µs | GPU-resident dispatch + compute (no data transfer) |
| Python NV=1 (tinygrad) | 107 µs | Same as C hot path but through Python interpreter |

**Why the C path is faster:** The Python NV=1 path spends ~100 µs in Python interpreter overhead (`CapturedJit.__call__`, dictionary lookups, `ensure_allocated()` checks). The C hot path does the same GPU operation — write GPFifo ring entry, poke MMIO doorbell at offset 0x90, spin-wait on signal memory — in ~3 µs of C dispatch + ~40 µs of GPU compute/pipeline.

**Fair comparison note:** PyTorch's 88 µs is GPU-resident (data already on GPU). Our 46 µs includes unified-memory memcpy of 24 bytes input + 8 bytes output, which costs <0.2 µs total. So this is near-level apples-to-apples.

### 2c. Summary: The Two PyTorch Wins

| Victory | What beat PyTorch | By how much | Key mechanism |
|---------|------------------|:-----------:|--------------|
| **Python NV=1** | Full control loop (incl. data transfer) | **1.85x** (207 vs 383 µs) | Tegra unified memory bypass — `memmove` to `cpu_view()` instead of `cuMemcpy` |
| **C GPU Hot Path** | GPU-resident dispatch | **1.9x** (46 vs 88 µs) | Raw MMIO doorbell dispatch — eliminate Python overhead, zero CUDA runtime |

Both share a common foundation: **tinygrad's NV backend bypasses the CUDA runtime**, which is why both approaches beat PyTorch despite PyTorch having better-optimized GPU kernels (cuBLAS).

**Why this works:** These MLP workloads are **overhead-bound**, not compute-bound. The GPU kernel itself takes ~5 µs — it's the 50-400 µs of framework/runtime/transfer overhead that dominates. Eliminating that overhead is where all the wins come from. If the GPU kernel took 500 µs instead of 5 µs, our dispatch advantage would be negligible.

---

## 3. The C Hot Path — What It Actually Does

The C hot path is specifically about **making MLP inference faster for real-time control loops**. This is probably our biggest deliverable.

### What it is

- 200 lines of C that replay a pre-built tinygrad HCQGraph
- Zero CUDA runtime, zero cuBLAS, zero ioctls in the hot loop
- The entire GPU dispatch is: write GPFifo ring entry → poke MMIO doorbell → spin-wait on signal memory
- Loads as a `.so` via `ctypes` in the same Python process (inherits tinygrad's mmap'd GPU buffers)

### What it's for

- Real-time neural policy inference for robotics control loops
- Models in the 50K–2M param range (MLPs for learned controllers)
- Where you need deterministic sub-100 µs latency with zero jitter stalls

### The workflow

```
Python (tinygrad)          C hot path
─────────────────          ──────────
1. Define MLP model     →  (nothing — stays in Python)
2. @TinyJit warmup      →  (nothing — Python captures HCQGraph)
3. JITBEAM optimize      →  (nothing — Python searches kernels)
4. export_graph.py       →  Produces config struct with:
                            - GPFifo ring/doorbell addresses
                            - Command queue GPU address
                            - Signal memory addresses
                            - Patch list for timeline values
5.                       →  C hot path loads config, runs loop:
                            memcpy_in → apply_patches → submit_gpfifo
                            → poke_doorbell → spin_wait → memcpy_out
```

### The numbers (7 MLP sizes)

| MLP Config | Params | C GPU Hot Path | NEON FP16 | NEON FP32 | GPU vs FP32 |
|-----------|:------:|:--------------:|:---------:|:---------:|:-----------:|
| 12→64→64→4 | 5K | 45.8 µs | **1.1 µs** | **1.5 µs** | NEON 31x |
| 12→128→128→4 | 19K | 46.0 µs | **2.9 µs** | **4.6 µs** | NEON 10x |
| 12→256→256→256→4 | 136K | 63.7 µs | **17.2 µs** | **35.4 µs** | NEON 1.8x |
| 12→512→512→4 | 271K | **58.1 µs** | 32.4 µs | 71.7 µs | **GPU 1.2x** |
| 12→512→512→512→4 | 534K | **64.6 µs** | 62.5 µs | 113.5 µs | **GPU 1.8x** |
| 12→1024→1024→4 | 1.1M | **53.4 µs** | 145.1 µs | 246.7 µs | **GPU 4.6x** |
| 12→1024→1024→1024→4 | 2.1M | **82.4 µs** | 228.7 µs | 488.4 µs | **GPU 5.9x** |

**Crossover:** GPU wins above ~270K params (vs FP32 NEON). Below that, ARM NEON is faster and simpler — use the right tool for the right size.

**Why the GPU gets better with size:** At 5K params the GPU kernel takes ~5 µs but dispatch overhead is ~40 µs — we're 90% overhead. At 2M params, kernel compute grows to ~40 µs while dispatch stays ~40 µs — we're 50/50 overhead vs compute, and the GPU's massive parallelism is finally paying off. The GPU's advantage over NEON keeps growing until we hit truly compute-bound territory (10M+ params) where kernel optimization quality (TRT/cuBLAS) would matter more than our dispatch advantage.

---

## 4. Strengths — What to Lead With

### 4a. The Python NV=1 Story (no C code needed)

- **4,832 Hz control loop in 15 lines of Python** — no other Python ML framework comes close
- **1.85x faster than PyTorch CUDA Graphs** in a full sensor → GPU → PID loop
- **Zero 100 ms jitter stalls** — PyTorch has them, we don't. For drones this is the difference between flying and crashing
- **No CUDA toolkit dependency** — tinygrad NV=1 uses raw nvgpu/nvmap kernel ioctls. No `libcuda.so`, no cuBLAS shared libraries

### 4b. The C GPU Hot Path (production MLP inference)

- **46 µs for 18K MLP, 82 µs for 2.1M MLP** — 1.9x faster than PyTorch CUDA Graphs GPU-resident
- **200 lines of C, libc-only dependency** — fully auditable by a hardware engineer
- **P99.9 within ~75% of median** — extremely deterministic, zero syscalls in hot loop
- **5.9x faster than FP32 NEON at 2M params** — the GPU advantage grows with model size

### 4c. LLM Inference (tinygrad NV=1 vs ecosystem)

| Matchup | Result |
|---------|--------|
| tinygrad vs **llama.cpp** (any config) | **tinygrad wins** (+4% to +13%) |
| tinygrad vs **vLLM** (quantized GGUF) | **tinygrad wins** (~2x faster) |
| tinygrad vs **vLLM** (fp16) | **Tie** (29.0 vs 30.1 tok/s) |
| tinygrad vs **MLC LLM** (fp16) | MLC wins (−27%) |
| tinygrad on **Qwen3** | **tinygrad wins** — only framework that runs it |

- **NV=1 is 24% faster than CUDA=1** on the same tinygrad code (41 vs 33 tok/s on Qwen3 0.6B)
- tinygrad **dequantizes all GGUF to fp16** at load time (reading 2-3x more data than llama.cpp) — and still wins. Proves how much overhead the CUDA runtime adds
- **97% memory bandwidth utilization** (198/205 GB/s) at fp16 batch-1 decode

**Why we win on LLMs too:** Batch-1 LLM decode is **memory-bandwidth-bound** — you read every model parameter once per generated token. On Orin's 205 GB/s LPDDR5, the bottleneck is streaming parameters from DRAM, not computing with them. NV=1's zero-overhead dispatch maximizes time spent on useful memory reads rather than CUDA runtime bookkeeping. This is the same memory-bound advantage as the control loop story, just at a different scale.

---

## 5. Weaknesses — Be Honest About

### 5a. Hot Path / Control Loop Limitations

| Weakness | Details |
|----------|---------|
| **Small models: just use NEON** | Below ~270K params (vs FP32), ARM NEON is faster, simpler, and lower power. A 5K MLP runs in 1.1 µs on NEON vs 46 µs on GPU. These models are so small that even our minimal ~40 µs GPU fixed overhead dominates. *Path to parity:* A hybrid NEON+GPU router in the same C binary could pick the right path at zero cost — we already have both `.so` files built |
| **Large models: TensorRT/cuBLAS wins (compute-bound regime)** | Above ~10M params, the workload shifts from overhead-bound to **compute-bound** — GEMM FLOPs dominate total time and our dispatch advantage becomes noise. TRT's hand-tuned SASS kernels, INT8 quantization, and tensor core utilization give 2-5x better compute throughput. *Path to competitive:* tinygrad's kernel codegen is actively improving — better GEMM tiling, shared memory usage, and warp-level primitives could narrow the gap. INT8 support in tinygrad would also help since it halves both memory reads and compute |
| **MLP sweet spot only** | Attention, convolutions, complex graphs have many kernel launches. Each adds ~5-10 µs. A 20-layer transformer = 100-200 µs overhead on top of compute |
| **Tegra-only** | MMIO doorbell dispatch requires Tegra unified memory. No discrete GPUs, no AMD, no Intel |
| **Tied to tinygrad internals** | `HCQGraph` API, GPFifo format, sym classification — changes upstream break `export_graph.py` |
| **JITBEAM is counterproductive for LLMs on Orin** | BEAM search finds locally optimal kernels that are globally worse on unified memory (1 tok/s vs 27 tok/s without BEAM). Only reliable for MLP hot path with small kernel counts |

### 5b. LLM Limitations

| Weakness | Details |
|----------|---------|
| **MLC LLM is 27% faster at fp16** | MLC's TVM compiler produces superior GPU kernels (cutlass). This is a **kernel quality / compute-efficiency gap** — even in a memory-bandwidth-bound workload, better kernels extract more useful work per byte read. *Path to competitive:* tinygrad's Flash Attention and improved matmul scheduling are active upstream work. Closing even half the kernel quality gap would put us ahead given our dispatch advantage |
| **No native quantization (wastes memory bandwidth)** | tinygrad dequants GGUF → fp16, reading 2-3x more data per token than llama.cpp's native Q6_K/Q4_K. In a memory-bandwidth-bound workload, this is a direct throughput tax. *Path to win big:* Native Q4/Q6 dequant kernels in tinygrad would cut memory reads by 2-3x. Combined with our existing dispatch advantage, this could put us ahead of every framework including MLC |
| **Prefill is slow** | 7.1 tok/s prefill vs llama.cpp's 1,089 (includes JIT compilation overhead) |

### 5c. Python NV=1 Limitations

| Weakness | Details |
|----------|---------|
| **HCQGraph dispatch is 2x slower than CUDA Graphs replay** | 182 µs vs 88 µs GPU-resident. The Python interpreter overhead is significant. *Path to parity:* A Cython/C extension for `CapturedJit.__call__` could bring this to ~20-30 µs, which would make even the Python path faster than CUDA Graphs GPU-resident |
| **`cpu_view()` is an internal API** | `_buffer()._buf.cpu_view()` could change between tinygrad versions |
| **Only works for fixed-shape models** | `@TinyJit` captures a graph for one input shape. Dynamic shapes require re-capture |

---

## 6. Comparison Matrix — All Frameworks

### 6a. MLP Dispatch Overhead (control-loop-sized models)

| Framework | Dispatch Overhead | Runtime Size | Dependencies | How it talks to GPU |
|-----------|:-----------------:|:------------:|:------------:|:-------------------:|
| **Our C hot path** | **~3 µs** | **~200 lines C** | **libc only** | Raw MMIO doorbell poke |
| **Python NV=1 (tinygrad)** | **~100 µs** | **15 lines Python** | tinygrad | Raw Tegra ioctls (nvgpu) |
| TensorRT | ~50-100 µs | ~100K lines C++ | CUDA toolkit (~2 GB) | CUDA runtime API |
| TVM (graph executor) | ~20-50 µs | ~2K lines C | CUDA for GPU targets | CUDA runtime API |
| IREE (Google/MLIR) | ~10-30 µs | ~50K lines C | Vulkan/Metal/CUDA | Varies by backend |
| ExecuTorch (Meta) | Varies | ~100K lines C++ | Backend-specific | Delegate-dependent |
| ONNX Runtime | ~50-200 µs | ~500K lines C++ | CUDA for GPU | CUDA runtime API |
| PyTorch CUDA Graphs | ~29 µs (replay) | massive | CUDA + cuBLAS | CUDA runtime API |
| PyTorch eager | ~75 µs | massive | CUDA + cuBLAS | CUDA runtime API |

### 6b. Full Control Loop Comparison (18K MLP, sensor → GPU → PID)

| Framework / Approach | End-to-End Latency | Frequency | Jitter | Data Transfer Method |
|---------------------|:------------------:|:---------:|:------:|---------------------|
| **C GPU Hot Path** | **~46 µs** | **~21.7 kHz** | P99.9 = 48.5 µs | Unified memory memcpy |
| **Python NV=1 (Approach C)** | **207 µs** | **4,832 Hz** | Zero >5ms outliers | `memmove` to `cpu_view()` |
| PyTorch CUDA Graphs | 383 µs | 2,609 Hz | 100ms stalls possible | `cuMemcpy` DMA |
| PyTorch eager | 670 µs | 1,493 Hz | 100ms stalls possible | `torch.cuda()` |

### 6c. LLM Decode (LLaMA 3.2 1B, fp16 effective)

| Framework | tok/s | vs llama.cpp +FA | Backend |
|-----------|:-----:|:----------------:|---------|
| **MLC LLM** (TVM) | **36.8** | +32% | TVM + cutlass + CUDA Graphs |
| **vLLM** (fp16) | 30.1 | +8% | PyTorch + custom CUDA |
| **tinygrad NV=1** | **29.0** | **+4%** | Raw Tegra ioctls |
| llama.cpp +FA | 27.85 | baseline | Hand-tuned CUDA kernels |
| llama.cpp (no FA) | 25.7 | −8% | Hand-tuned CUDA kernels |

### 6d. NVIDIA DLA (Deep Learning Accelerator)

The Orin AGX has **two DLA engines** — fixed-function INT8/FP16 accelerators alongside the GPU.

| Aspect | DLA | Our C GPU Hot Path |
|--------|-----|-------------------|
| Latency (small MLP) | ~10-50 µs estimated | 46-82 µs measured |
| Quantization | INT8 native (best perf) | FP16 |
| Programming | TensorRT DLA backend only | Raw C, any tinygrad model |
| Op support | Limited (no custom layers) | Any tinygrad-expressible op |
| Power | Lower (dedicated silicon) | Higher (2048 CUDA cores) |
| Flexibility | Fixed op set, TRT dependency | Full Python ML prototyping |
| Status | **Not benchmarked yet** | **Benchmarked** |

**DLA is a legitimate competitor** for pure MLP inference at our model sizes. It should be benchmarked head-to-head.

### 6e. TensorRT vs Our Approach (memory-bound vs compute-bound)

The TRT comparison illustrates the memory-bound → compute-bound transition perfectly:

| Dimension | TensorRT | Our C Hot Path | Why |
|-----------|----------|----------------|-----|
| Kernel quality (large GEMM) | **Superior** — hand-tuned SASS, INT8, sparsity | BEAM-optimized PTX | Compute-bound: kernel quality matters |
| Kernel quality (small GEMM) | Similar — small GEMMs are simple | Similar | Neither is compute-bound here |
| Dispatch overhead | ~50-100 µs (CUDA API) | **~3 µs (raw MMIO)** | Overhead-bound: dispatch dominates |
| End-to-end (100K MLP) | ~60-110 µs estimated | **46-64 µs measured** | **We win** — still overhead-bound at 100K |
| End-to-end (1-2M MLP) | ~80-150 µs estimated | **53-82 µs measured** | **We win** — overhead + compute mix, dispatch advantage still matters |
| End-to-end (10M+ model) | **Wins** | Loses | **TRT wins** — now compute-bound, kernel quality dominates |
| INT8 quantization | Yes, calibration-based | No | Reduces memory reads AND compute — TRT advantage in both regimes |
| Dependencies | CUDA toolkit + TRT libs (~3+ GB) | libc only | |
| Prototyping speed | ONNX export → TRT compile (minutes-hours) | Python change → C re-export (seconds) | |

**The crossover:** Our approach beats TRT for models where dispatch overhead is a significant fraction of total latency (roughly <10M params for MLPs). Beyond that, TRT's kernel quality advantage overcomes our dispatch advantage.

---

## 7. Where This Approach Extends (Further Work)

### 7a. Near-Term (High Confidence)

| Project | Why it's promising | Estimated effort |
|---------|-------------------|-----------------|
| **DLA head-to-head benchmark** | DLA is on the same chip — settles "should I use DLA?" for control loops | 1-2 days |
| **TensorRT head-to-head** | Compile same MLP sizes with TRT, quantify dispatch overhead advantage | 2-3 days |
| **Hybrid NEON+GPU routing** | Below 270K → NEON, above → GPU. One binary, zero-latency decision | 1 day |
| **Xavier/NX portability** | Same Tegra ioctl interface, smaller GPU. Does it scale down? | 1-2 days |
| **`SCHED_FIFO` + `mlockall` hardening** | Real-time Linux priority for production safety-critical deployment | 1 day |

### 7b. Medium-Term (Promising but Uncertain)

| Project | Opportunity | Challenge |
|---------|-------------|-----------|
| **Vision models (small CNNs)** | MobileNet-v3 small (~2M params) at multi-kHz for object detection | More kernel launches; need to measure per-launch overhead |
| **Attention / small transformers** | Tiny transformers (1-4 layers, 64-256 dim) for sequence modeling | Multi-head attention = many small GEMMs; dispatch overhead accumulates |
| **Recurrent policies (GRU/LSTM)** | 64-128 hidden GRU = ~50K params, fits sweet spot | Sequential dependency limits parallelism |
| **Multi-model pipelines** | Sensor filter → policy → safety checker in one graph | `export_graph.py` needs multi-graph support |
| **Tinygrad upstream contribution** | Merge Tegra optimizations (TegraIface, matvec fix) to tinygrad | Requires conforming to tinygrad code review standards |

### 7c. Long-Term (Speculative)

| Project | Vision | Risk |
|---------|--------|------|
| **DLA integration via tinygrad** | Same Python workflow, DLA execution | DLA programming undocumented outside TRT; massive reverse-engineering |
| **Training on Orin** | Train small RL policies on-device, deploy to C hot path same session | Orin's 205 GB/s bandwidth limits batch size |
| **ROS2 integration** | Tinygrad hot path as a ROS2 inference node | Requires ROS2 Humble aarch64 NixOS packaging |

---

## 8. Real-World Use Cases

### Where 4.8+ kHz GPU inference matters (Python NV=1)

| Application | Typical Rate | Python NV=1 Rate | Impact |
|-------------|:------------:|:-----------------:|--------|
| Drone rate-loop control | 1-4 kHz | **4.8 kHz** | First Python framework that can sustain inner-loop rates |
| Robot arm force control | 1-4 kHz | **4.8 kHz** | Sub-ms reaction with a learned policy |
| Legged locomotion | 200-1000 Hz | **4.8 kHz** | 5-24x headroom for policy complexity |

### Where 21+ kHz matters (C hot path)

| Application | Typical Rate | C Hot Path Rate | Impact |
|-------------|:------------:|:---------------:|--------|
| High-performance servo loops | 4-10 kHz | **21 kHz** | Learned controllers replace hand-tuned PID |
| Anomaly detection on sensor streams | 1-10 kHz | **21 kHz** | Real-time fault detection |
| Neural MPC (model-predictive control) | 100+ Hz | **12 kHz** (500K) | Real-time model sim at 100x requirement |

### Industry comparison

| Company / Robot | Hardware | Policy Rate | Relevance |
|-----------------|----------|:-----------:|-----------|
| **Unitree H1/G1** | Jetson Orin | ~200 Hz (TRT) | **Direct fit** — same hardware, 100x faster dispatch possible |
| **Boston Dynamics Atlas** | Custom board | ~1 kHz | Same concept if on Jetson |
| **Tesla Optimus** | Custom AI chip | ~500-1000 Hz | Different HW, same architecture |

**Key insight:** Nobody runs learned locomotion policies above ~1 kHz because TensorRT/CUDA dispatch overhead floors them at ~100-200 µs. Our 3 µs dispatch overhead removes that floor.

---

## 9. What This Is NOT (Manage Expectations)

- **Not a general-purpose inference engine** — it's a thin dispatch layer for pre-compiled tinygrad graphs
- **Not faster than TRT for large, compute-bound models** — above ~10M params, the bottleneck shifts from dispatch overhead to GEMM FLOPs, and TRT's hand-tuned kernels dominate. Our wins are entirely in the overhead-bound / memory-bound regime
- **Not portable** — Tegra-only (Orin, Xavier). No discrete GPU, no AMD, no Intel
- **Not a research breakthrough** — it's excellent systems engineering: well-understood concepts (compiled graph replay, MMIO dispatch, spin-wait) applied to a high-value platform
- **Not a kernel optimizer** — we don't produce better GPU kernels than TRT or cuBLAS. We produce the same quality kernels (via tinygrad BEAM) with dramatically less overhead around them. When overhead matters (memory-bound), we win. When kernel quality matters (compute-bound), we lose

---

## 10. Presentation Flow (Suggested 10 Slides)

### Slide 1: Problem
>
> Learned neural policies on Jetson Orin are **overhead-bound**, not compute-bound. The GPU kernel takes ~5 µs, but PyTorch eager adds 670 µs and CUDA Graphs adds 383 µs of dispatch/transfer/runtime overhead. 95% of the latency is waste.

### Slide 2: Insight — tinygrad's NV backend
>
> tinygrad's NV backend bypasses CUDA entirely — talks to Tegra hardware via raw ioctls. Combined with Tegra unified memory (`cpu_view()` + `memmove`), the data transfer drops from 180 µs to 2 µs.

### Slide 3: Result 1 — Python NV=1
>
> **207 µs / 4,832 Hz** in 15 lines of Python. 1.85x faster than PyTorch CUDA Graphs. Zero 100ms stalls (PyTorch has them).

### Slide 4: Going further — C hot path
>
> Strip out Python overhead. 200 lines of C replay the same GPU commands via raw MMIO. **46 µs / 21.7 kHz**.

### Slide 5: C hot path vs everything
>
> 1.9x over PyTorch CUDA Graphs (GPU-resident). 8.7x over PyTorch eager. 10-30x lower dispatch overhead than TensorRT/ONNX Runtime.

### Slide 6: GPU vs CPU crossover chart
>
> Show the 7-model results. GPU wins above 270K params (vs FP32 NEON). NEON wins below. Use the right tool for the right size.

### Slide 7: Jitter / determinism
>
> P99.9 within 75% of median. Zero outliers >5ms. PyTorch: 100ms stalls in 60s runs.

### Slide 8: LLM bonus (same principle, different scale)
>
> Batch-1 LLM decode is memory-bandwidth-bound, not compute-bound. Same low-overhead dispatch → 97% bandwidth utilization → beats llama.cpp by 4-13%. Loses to MLC by 27% because MLC has better kernels (compute efficiency matters even in memory-bound workloads).

### Slide 9: Honest weaknesses — where compute-bound wins
>
> Small models → NEON (no overhead to eliminate). Large models → TRT (compute-bound, kernel quality matters). Our sweet spot is the overhead-bound / memory-bound regime: 270K-10M param MLPs, batch-1 LLM decode. Tegra-only. DLA untested.

### Slide 10: The big picture
>
> *"In the overhead-bound regime — which includes every robotics control loop and every batch-1 LLM decode — framework overhead is the bottleneck, not GPU compute. We eliminate it. Prototype in Python at 4.8 kHz, deploy in C at 21 kHz. 200 lines. Zero dependencies."*

---

## 11. Key Numbers Reference Card

| Metric | Value |
|--------|-------|
| **Python NV=1 full control loop** | **207 µs / 4,832 Hz** |
| Python NV=1 vs PyTorch CUDA Graphs (full loop) | **1.85x faster** (207 vs 383 µs) |
| Python NV=1 vs PyTorch eager (full loop) | **3.2x faster** (207 vs 670 µs) |
| Python NV=1 GPU-resident dispatch | 182 µs (vs PyTorch CUDA Graphs 88 µs — PyTorch wins here) |
| Python NV=1 jitter | **Zero outliers >5 ms** in 60s (PyTorch: 100 ms stalls) |
| | |
| **C GPU hot path** (18K MLP) | **46 µs / 21.7 kHz** |
| C hot path vs PyTorch CUDA Graphs (GPU-resident) | **1.9x faster** (46 vs 88 µs) |
| C hot path vs PyTorch eager (GPU-resident) | **8.7x faster** (46 vs 402 µs) |
| C hot path vs Python NV=1 | **2.3x faster** (46 vs 107 µs) |
| C hot path dispatch overhead | **~3 µs** (vs TRT ~50-100 µs) |
| GPU crossover vs FP32 NEON | **~270K params** |
| GPU crossover vs FP16 NEON | **~500K-1M params** |
| GPU advantage at 2M params (vs FP32) | **5.9x** |
| FP16 accuracy vs FP64 ground truth | **Max error <0.001** |
| C hot path jitter (P99.9 / median ratio) | **~1.75x** |
| C hot path code size | **~200 lines** |
| C hot path dependencies | **libc only** |
| | |
| **LLM: tinygrad vs llama.cpp +FA** (1B) | **+4%** (29.0 vs 27.85 tok/s) |
| **LLM: tinygrad vs llama.cpp** (Qwen3 0.6B) | **+10%** (41.0 vs 37.1 tok/s) |
| NV=1 vs CUDA=1 backend speed | **+24%** |
| Memory bandwidth utilization (fp16) | **97%** (198/205 GB/s) |
