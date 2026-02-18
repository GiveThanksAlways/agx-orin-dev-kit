# Learning: How We Made tinygrad Beat llama.cpp on LLM Inference

**TL;DR:** tinygrad's LLM decoder was running at 3.85 tok/s on Jetson Orin — 7× slower than llama.cpp's 25.62 tok/s. The matvec pattern-match heuristic in the tinygrad compiler was silently failing on every single matmul kernel in the LLM, causing them to use 16 threads with 1.5 GB/s bandwidth instead of 128 threads with 49 GB/s. We fixed the pattern match, tuned the thread count, and added BEAM search. Final result: **36.71 tok/s — 43% faster than llama.cpp**.

**This document is a teaching walkthrough.** It follows the actual chronological order of the work, explains the methodology, concepts, mistakes, and debugging. Written so you can teach this to others or reproduce it yourself.

---

## Table of Contents

1. [The Setup: What Are We Working With?](#1-the-setup-what-are-we-working-with)
2. [The Target: What Is llama.cpp Doing?](#2-the-target-what-is-llamacpp-doing)
3. [Step 1: Establish a Clean Baseline](#3-step-1-establish-a-clean-baseline)
4. [Step 2: Profile — Where Does the Time Go?](#4-step-2-profile-where-does-the-time-go)
5. [Step 3: Understand the Compiler Pipeline](#5-step-3-understand-the-compiler-pipeline)
6. [Step 4: Find the Root Cause](#6-step-4-find-the-root-cause)
7. [Step 5: Implement the Fix](#7-step-5-implement-the-fix)
8. [Step 6: Tune the Parameters](#8-step-6-tune-the-parameters)
9. [Step 7: BEAM Search — Let the Compiler Explore](#9-step-7-beam-search-let-the-compiler-explore)
10. [Key Concepts](#10-key-concepts)
11. [Reproduction Commands](#11-reproduction-commands)
12. [Remaining Opportunities](#12-remaining-opportunities)
13. [Summary of Numbers](#13-summary-of-numbers)

---

## 1. The Setup: What Are We Working With?

### The Hardware

**Jetson Orin AGX 64GB** — NVIDIA's most powerful edge AI computer.

| Spec | Value | Why It Matters |
|------|-------|----------------|
| GPU | Ampere iGPU, SM 8.7, 2048 CUDA cores | Same architecture as RTX 3000 series, but integrated |
| Memory | 64 GB LPDDR5, ~102 GB/s effective | Shared between CPU and GPU — no PCIe bottleneck |
| Tensor cores | Yes, fp16/bf16 WMMA | Fast matrix multiply, but only at large batch sizes |
| Cache | 128 KB L1 per SM, 4 MB L2 | Small compared to weight matrices |

### The Software Stack

**tinygrad** is a minimalist deep learning framework (~10K lines of Python) that compiles tensor operations into GPU kernels. It has two NVIDIA backend paths:

```
tinygrad kernel compilation pipeline:

  Tensor ops (matmul, add, softmax...)
       ↓
  Scheduler (fuses ops, decides kernel boundaries)
       ↓
  Codegen (turns fused ops into a kernel IR)
       ↓
  Heuristic / BEAM search (optimizes thread layout, tiling, upcasting)
       ↓
  PTX Renderer (generates NVIDIA assembly)
       ↓
  CUDA Compiler (nvJitLink → CUBIN machine code)
       ↓
  Backend dispatch:
    CUDA=1 → cuLaunchKernel() via libcuda.so
    NV=1   → Direct GPFIFO/QMD via /dev/nvgpu ioctl (no CUDA driver)
```

The **NV backend** bypasses the CUDA driver entirely, talking directly to the GPU kernel driver. This gives lower dispatch latency (~1.2ms p99 vs 2.6ms for CUDA) but uses the exact same compiled kernels.

### The Model

**LLaMA 3.2 1B Instruct, Q6_K quantization** — a 1-billion parameter language model stored in GGUF format with ~6.5-bit quantization.

- File size: ~967 MB (quantized)
- In tinygrad: weights are **dequantized to fp16** at load time, expanding to ~2.5 GB in memory
- llama.cpp: reads quantized weights directly, dequantizes inside the matmul kernel

This difference — 2.5 GB vs 0.97 GB per token — is important. We'll come back to it.

---

## 2. The Target: What Is llama.cpp Doing?

Before optimizing, we need to understand our competition.

### llama.cpp Benchmark

```bash
cd /home/agent/jetpack-nixos/examples/llama-cpp-orin
nix develop -c llama-bench -m ~/.cache/tinygrad/llama3-1b-instruct/Llama-3.2-1B-Instruct-Q6_K.gguf -p 128 -n 128
```

Results:
- **Decode (tg128):** 25.62 tok/s (39ms per token)
- **Prefill (pp128):** 1089 tok/s
- **With flash attention (FA=1):** 27.82 tok/s decode, 1392 tok/s prefill

### Why batch=1 decode is memory-bandwidth-bound

At batch=1, generating one token requires reading **every weight** in the model once, but doing very little compute per weight. The math:

```
Arithmetic intensity = FLOPs per token / Bytes read per token

For a 2048×N matmul at batch=1:
  FLOPs = 2 × 2048 × N  (multiply-accumulate)
  Bytes = 2 × 2048 × N  (fp16 weights, 2 bytes each)
  
  Arithmetic intensity = 2 × 2048 × N / (2 × 2048 × N) = 1.0 FLOP/byte
```

The Orin's GPU can do ~22 TFLOPS (fp16 tensor cores) but can only read ~102 GB/s. With 1 FLOP/byte arithmetic intensity, we hit the memory wall at:

```
Max throughput = 102 GB/s / (2.5 GB per token) = 40.8 tok/s  (fp16 weights)
Max throughput = 102 GB/s / (0.97 GB per token) = 105 tok/s   (Q6_K weights)
```

llama.cpp achieves 25.62/105 = **24% of theoretical peak**. That's normal for small models — overhead from non-matmul operations, cache effects, and kernel launch takes the rest.

**Key insight:** At batch=1, the GPU's compute power is irrelevant. The only thing that matters is how fast you can read memory. Every byte saved = direct speedup.

---

## 3. Step 1: Establish a Clean Baseline

Before optimizing, you need a reliable number to measure against. Two critical mistakes to avoid:

### Mistake 1: DEBUG overhead

tinygrad's `DEBUG=2` flag prints per-kernel timing info. This is invaluable for profiling, but it **synchronizes the GPU after every kernel** to read timing values. This adds ~64% overhead:

| Condition | tok/s | ms/tok |
|-----------|------:|-------:|
| Clean (no DEBUG) | 3.85 | 260 |
| DEBUG=2 | ~2.3 | ~430 |

Always measure final performance without DEBUG flags.

### Mistake 2: Warmup tokens

The first 2-3 tokens are slow because tinygrad JIT-compiles kernels on the first run. By token 3, the JIT has captured the execution pattern and replays it from cache. Skip the first 5+ tokens when measuring.

### Clean baseline command

```bash
cd /home/agent/jetpack-nixos/examples/tinygrad
nix develop -c bash -c 'cd tinygrad && NV=1 python3 -c "
from tinygrad.apps.llm import Transformer, models
from tinygrad import Tensor
from tinygrad.helpers import fetch
import time
gguf = fetch(models[\"llama3.2:1b\"], \"Llama-3.2-1B-Instruct-Q6_K.gguf\",
             subdir=\"llama3-1b-instruct\")
model, kv = Transformer.from_gguf(Tensor(gguf))
tokens = [128000, 9906]
times = []
for i, tok in enumerate(model.generate(tokens)):
    times.append(time.time())
    if i >= 17: break
# Skip first 5 tokens (JIT warmup)
dts = [times[i]-times[i-1] for i in range(5, len(times))]
avg = sum(dts)/len(dts)
print(f\"Steady-state: {avg*1000:.1f}ms = {1/avg:.2f} tok/s\")
"'
```

**Our baseline: 3.85 tok/s (260ms per token)**

That's 6.7× slower than llama.cpp. Something is very wrong.

---

## 4. Step 2: Profile — Where Does the Time Go?

### Using DEBUG=2 for kernel profiling

```bash
NV=1 DEBUG=2 python3 -c "..." 2>&1 | tee /tmp/nv_debug2.txt
```

Each kernel appears as a line like:
```
*** NV   123 r_2048_16_128   arg 4 mem 12.15 GB tm 10400.00us/1234.56ms (0 GFLOPS 1|1 GB/s) ['linear']
```

Breakdown:
- `r_2048_16_128` — kernel name (r=reduce, dimensions after `_` are the shape)
- `tm 10400.00us` — this kernel took 10.4ms
- `1|1 GB/s` — achieved memory bandwidth (before and after a separator)
- `['linear']` — which tinygrad operations this kernel implements

### What the profiling revealed (BEFORE the fix)

For a single decode token, 485 kernels run in ~542ms:

| Kernel Pattern | Count | Total Time | % | BW (GB/s) | What |
|---------------|------:|-----------:|----:|----------:|------|
| `r_2048_16_128` | 60 | 152.8ms | 28.2% | **1.5** | Matmul (Q/K/V/O projections) |
| `r_2048_16_128n1` | 32 | 110.6ms | 20.4% | **1.7** | Matmul (FFN gate/up/down) |
| `r_2_8_64_16_128n1` | 14 | 76.0ms | 14.0% | **2.0** | Matmul (fused with RoPE) |
| attention + softmax | 75 | 24.5ms | 4.5% | various | Small kernels, OK |
| RMSNorm | 45 | 5.2ms | 1.0% | ~0 | Tiny reductions, OK |

**The obvious problem:** every matmul kernel achieves only **1.5-2.0 GB/s** memory bandwidth. The Orin can do 102 GB/s. We're using **1.5% of available bandwidth**.

For comparison, a well-optimized matvec kernel should get 40-60 GB/s on this hardware.

---

## 5. Step 3: Understand the Compiler Pipeline

To understand why the kernels are so slow, we need to understand how tinygrad compiles them.

### The kernel optimization pipeline

When tinygrad creates a reduce kernel (like a matmul), it goes through `hand_coded_optimizations()` in `tinygrad/codegen/opt/heuristic.py`. This function decides:

1. **Should we use tensor cores?** (For matmul at batch=1: no — arithmetic intensity is too low)
2. **Is this a matvec?** ← THE CRITICAL DECISION
3. **If not matvec, use generic GROUPTOP** (cooperative reduction with default thread count)

### What "matvec" optimization does

A matrix-vector multiply `y = W @ x` where W is (M, K) and x is (K,) can be parallelized as:

```
For each output row m (0..M-1):
    y[m] = sum over k of W[m,k] * x[k]
```

The **matvec heuristic** applies three transformations:

1. **GROUP** — Split the K (reduction) dimension across `MV_THREADS_PER_ROW` threads. Each thread handles K/TPR elements. Then they cooperatively reduce using shared memory.

2. **LOCAL** — Pack `MV_BLOCKSIZE` output rows into one thread block. Helps with scheduling.

3. **UPCAST** — Each thread computes `MV_ROWS_PER_THREAD` rows at once (instruction-level parallelism).

With defaults (TPR=8, BS=4, RPT=4): 8 × 4 = 32 threads per block, each computing 4 rows.
With our optimal (TPR=32, BS=4, RPT=4): 32 × 4 = 128 threads per block.

### What happens WITHOUT matvec (GROUPTOP fallback)

Without matvec, the kernel falls through to `GROUPTOP(16)`, which:
- Creates 16 threads doing a cooperative reduction
- But these threads access memory with strides of 262,144 elements apart
- This means **zero memory coalescing** — each thread reads from a completely different cache line
- Modern GPUs are designed for 32 threads (one warp) reading 32 consecutive values

The result: 1.5 GB/s instead of 49 GB/s. **A 33× bandwidth difference.**

### The matvec pattern match

The heuristic recognizes a matmul by looking at the UOp (micro-operation) tree:

```python
# Expected pattern in heuristic.py:
# REDUCE(op=ADD,
#   MUL(
#     INDEX(buf0, idx0),    ← one buffer (weights)
#     INDEX(buf1, idx1)     ← other buffer (input)
#   )
# )
```

It checks:
1. Is there a REDUCE with ADD?
2. Is the reduce input a MUL of two INDEX nodes?
3. Does one INDEX iterate over the reduce dimension?

If all checks pass → apply matvec optimizations.
If any check fails → fall through to GROUPTOP.

---

## 6. Step 4: Find the Root Cause

### The hypothesis

The matmul kernels are getting 1.5 GB/s instead of 49 GB/s. The matvec heuristic is supposed to handle exactly these kernels. So either:
- (a) The heuristic is matching but applying bad parameters, or
- (b) The heuristic is **not matching at all**

### Adding diagnostic prints

We added a `MV_TRACE` environment variable to print what the heuristic sees:

```python
if mulop.op is Ops.MUL and len(mulop.src) == 2:
    if os.environ.get("MV_TRACE"):
        print(f"MV_TRACE: mulop.src[0].op={mulop.src[0].op} mulop.src[1].op={mulop.src[1].op}")
```

### What we found

**Standalone matmul (for sanity check):**
```
MV_TRACE: mulop.op=Ops.MUL src=[Ops.INDEX, Ops.INDEX]  ← MATCH ✓
→ Applies matvec → 0.47ms, 47 GB/s
```

**LLM matmul kernels:**
```
MV_TRACE: mulop.op=Ops.CAST src=[Ops.MUL]               ← NO MATCH ✗
MV_TRACE: mulop.op=Ops.MUL src=[Ops.MUL, Ops.CAST]      ← NO MATCH ✗
```

**The matvec heuristic was NEVER triggering for any LLM kernel.** Two reasons:

### Reason 1: CAST wrapping (fp16 → fp32 accumulation)

When you multiply two fp16 values, tinygrad accumulates the result in fp32 to avoid precision loss. This inserts a CAST node:

```
Before (what heuristic expects):     After (what actually happens):
REDUCE(ADD,                           REDUCE(ADD,
  MUL(                                  CAST(fp32,        ← added
    INDEX(W),                             MUL(
    INDEX(x)                                INDEX(W),
  )                                         INDEX(x)
)                                         )
                                        )
                                      )
```

The heuristic checks `k.reduceop.src[0]` — in the first case it's a MUL, in the second case it's a **CAST**. The CAST doesn't match `Ops.MUL`, so the matvec code-path is skipped.

### Reason 2: Fused operations (RMSNorm + matmul)

tinygrad's scheduler is smart enough to fuse RMSNorm with the following matmul. This means the UOp tree for a linear layer looks like:

```
REDUCE(ADD,
  MUL(                     ← outer MUL
    MUL(                   ← inner MUL (RMSNorm: x * scale)
      INDEX(x),
      INDEX(norm_weight)
    ),
    INDEX(W)               ← matmul weight
  )
)
```

The heuristic checks `MUL(INDEX, INDEX)` — but what it finds is `MUL(MUL(...), INDEX)`. The inner `MUL` is not an `INDEX`, so it doesn't match.

### The impact

**Every single matmul kernel in the 16-layer transformer was falling through to GROUPTOP(16).** That's 60+ matmul launches per token, each running with 16 threads and 1.5 GB/s bandwidth instead of 128 threads and 49 GB/s.

---

## 7. Step 5: Implement the Fix

Two changes to `hand_coded_optimizations()` in `tinygrad/codegen/opt/heuristic.py`:

### Change 1: Unwrap CAST

```python
mulop = k.reduceop.src[0]
# NEW: unwrap CAST (e.g. fp16 inputs accumulated in fp32)
if mulop.op is Ops.CAST: mulop = mulop.src[0]
```

This handles `REDUCE(CAST(MUL(INDEX, INDEX)))` by peeling off the CAST to expose the MUL underneath.

### Change 2: Recursive INDEX finder

Instead of checking `MUL(INDEX, INDEX)`, we recursively search through MUL/CAST chains to find INDEX nodes:

```python
def _find_indices(u, depth=0):
    """Find INDEX nodes through MUL/CAST chains, max depth 3."""
    if u.op is Ops.INDEX: return [u]
    if depth > 3: return []
    ret = []
    for s in u.src:
        ret.extend(_find_indices(s, depth+1))
    return ret

indices = _find_indices(mulop) if mulop.op is Ops.MUL else []
if len(indices) >= 2:
    idx0, idx1 = indices[0].src[1].get_idx(), indices[1].src[1].get_idx()
    # ... rest of matvec logic
```

This handles:
- `MUL(INDEX, INDEX)` — simple matmul (depth 1)
- `CAST(MUL(INDEX, INDEX))` — fp16 accumulating as fp32 (CAST + depth 1)
- `MUL(MUL(INDEX, INDEX_norm), INDEX_W)` — fused RMSNorm+matmul (depth 2)
- `MUL(CAST(MUL(...)), INDEX)` — weird combos (depth 2-3)

Depth limit of 3 prevents runaway searching on deeply nested ops that aren't matmuls.

### Verification: standalone matmul

```
Before fix:  r_2048_16_128   → 10.4ms, 1.5 GB/s, 16 threads
After fix:   r_128_8_4_4_256 → 0.47ms, 47 GB/s, 128 threads
Speedup: 22×
```

The kernel name change tells the story:
- `r_2048_16_128`: GROUPTOP(16) — 16 threads reducing 2048 elements
- `r_128_8_4_4_256`: matvec — 128 outputs × 8 GROUP × 4 LOCAL × 4 UPCAST × 256 reduce/thread

### Committed as `2439279b1`

---

## 8. Step 6: Tune the Parameters

The matvec heuristic has three tunable parameters:

| Parameter | Default | What It Controls |
|-----------|--------:|------------------|
| `MV_THREADS_PER_ROW` | 8 | How many threads cooperate on one output row's reduction |
| `MV_BLOCKSIZE` | 4 | Output rows per thread block |
| `MV_ROWS_PER_THREAD` | 4 | Output rows computed per thread (UPCAST/ILP) |

### Why MV_THREADS_PER_ROW=32 is optimal

A GPU **warp** is 32 threads that execute in lockstep. The GROUP operation uses shared memory for cooperative reduction. When TPR=8, only 8 of the 32 warp threads participate — the other 24 are idle during the reduction phase.

At TPR=32, the full warp participates. Memory reads are maximally coalesced (32 consecutive fp16 values = 64 bytes = one cache line). The shared memory reduction uses **warp shuffle** instructions instead of explicit shared mem loads/stores.

### Full parameter sweep

```bash
# Run with different MV params:
NV=1 MV_THREADS_PER_ROW=X MV_BLOCKSIZE=Y MV_ROWS_PER_THREAD=Z \
  python3 -c "... benchmark script ..."
```

| TPR | BS | RPT | tok/s | ms/tok | Notes |
|----:|---:|----:|------:|-------:|-------|
| 8 | 4 | 4 | 18.34 | 54.5 | Defaults — already 4.8× faster than before fix |
| 32 | 2 | 4 | 29.03 | 34.4 | |
| **32** | **4** | **4** | **29.90** | **33.4** | **Optimal for heuristic** |
| 32 | 4 | 2 | 28.20 | 35.5 | Less ILP |
| 32 | 4 | 8 | 22.10 | 45.2 | Too much upcast, register pressure |
| 32 | 8 | 4 | 24.24 | 41.3 | Block too large |
| 64 | 4 | 4 | 25.93 | 38.6 | 2 warps — diminishing returns |

### What the optimal config means physically

With TPR=32, BS=4, RPT=4:
- **32 threads** cooperate on each output row's dot product (one warp)
- **4 thread blocks** share an SM (128 threads total)
- Each thread computes **4 output rows** via loop unrolling (ILP)
- Each thread reads `K/32` weight elements → 32 threads together read `K` consecutive elements → perfect coalescing

---

## 9. Step 7: BEAM Search — Let the Compiler Explore

The hand-coded heuristic is a starting point. BEAM search systematically explores alternative kernel configurations and benchmarks each one.

### How BEAM search works

```
                  ┌─────────────┐
                  │ Initial     │ ← hand_coded_optimizations() applies
                  │ kernel opts │   GROUP, LOCAL, UPCAST based on heuristics
                  └─────┬───────┘
                        │
                  ┌─────▼───────┐
                  │ BEAM search │ ← Tries adding/changing opts:
                  │ explores    │   different UPCAST amounts,
                  │ alternatives│   different LOCAL sizes,
                  │             │   UNROLL amounts, etc.
                  └─────┬───────┘
                        │
              ┌─────────┼─────────┐
              │         │         │
           ┌──▼──┐  ┌──▼──┐  ┌──▼──┐
           │var1 │  │var2 │  │var3 │  ← Each variant is compiled
           │bench│  │bench│  │bench│    and benchmarked on the GPU
           └──┬──┘  └──┬──┘  └──┬──┘
              │         │         │
              └─────────┼─────────┘
                        │
                  ┌─────▼───────┐
                  │ Keep top-K  │ ← BEAM=K controls how many
                  │ variants    │   survive each round
                  └─────────────┘
```

### JITBEAM vs BEAM

- `BEAM=N`: Every kernel goes through BEAM search at compile time. Very slow.
- `JITBEAM=N`: Only beams during JIT capture (second inference run). First run uses heuristics, then the JIT captures the kernel sequence, applies BEAM to each kernel, and replays the optimized versions. **Much faster startup.**

### Results

```bash
# JITBEAM=2 (148s compile time, then optimized)
NV=1 MV_THREADS_PER_ROW=32 JITBEAM=2 IGNORE_BEAM_CACHE=1 python3 ...
# → 34.21 tok/s (29.2ms per token)

# JITBEAM=4 (229s compile time)
NV=1 MV_THREADS_PER_ROW=32 JITBEAM=4 IGNORE_BEAM_CACHE=1 python3 ...
# → 36.71 tok/s (27.2ms per token)
```

| Config | NV=1 tok/s | CUDA=1 tok/s | NV advantage |
|--------|----------:|------------:|:-------------|
| Heuristic only (MV_TPR=32) | 29.90 | 29.97 | Tied |
| JITBEAM=2 | 34.21 | — | — |
| JITBEAM=4 | **36.71** | 31.83 | **+15%** |

### Why BEAM helps NV more than CUDA

BEAM search finds better kernel thread layouts, memory access patterns, and unroll factors. The improved kernels are shorter and have fewer memory stalls.

On CUDA, kernel launch overhead is ~2.6ms (p99) — the time between submitting a kernel and the GPU starting it. This overhead partially masks kernel execution time, so a 10% faster kernel only gives ~6% wall-clock improvement.

On NV, kernel launch overhead is ~1.2ms (p99). The GPU pipeline is tighter, so better kernels translate more directly to faster wall-clock time. That's why BEAM gives NV a +23% boost vs only +6% for CUDA.

This is the first time NV=1 has conclusively beaten CUDA=1 on end-to-end LLM inference, confirming the NV backend's lower-overhead architecture pays off when the kernels are good enough.

---

## 10. Key Concepts

### Memory bandwidth is everything at batch=1

At batch=1, every token reads the entire weight matrix. The GPU's compute capacity is irrelevant — what matters is how fast you can stream data from memory to the ALUs. This is called being **memory-bandwidth-bound**.

```
Effective bandwidth = bytes_read / kernel_time

Good matvec kernel:  49 GB/s  (48% of 102 GB/s peak)
Bad kernel:           1.5 GB/s (1.5% of peak)
llama.cpp:           ~25 GB/s  (24% of peak, but reads less data)
```

### Coalesced memory access

When 32 threads in a warp access 32 consecutive memory addresses, the GPU combines them into **one** cache-line read. This is coalesced access — up to 128 bytes in one transaction.

When threads access non-consecutive addresses (e.g., stride of 262,144 elements), each thread generates a separate memory transaction. This is **uncoalesced** — 32× more transactions for the same data.

The GROUPTOP(16) fallback created uncoalesced access. The matvec GROUP(32) optimization creates perfectly coalesced access.

### Warp divergence and occupancy

A warp is 32 threads. If your GROUP reduction uses only 8 threads (TPR=8), the other 24 threads in the warp sit idle during the reduction phase = 75% wasted execution slots.

With TPR=32 (one full warp), no threads are wasted. The kernel achieves higher **occupancy** — more useful work per clock cycle.

### The UOp tree and pattern matching

tinygrad represents computations as trees of micro-operations (UOps). The compiler's heuristic matches patterns in this tree to decide which optimizations to apply. When the pattern doesn't account for all the ways a matmul can appear (with CAST wrappers, fused operations, etc.), the heuristic silently fails and falls back to a generic (slow) strategy.

**Lesson:** When a kernel is unexpectedly slow, check if the optimization heuristic is actually triggering, not just that the optimization exists.

### JIT GRAPH batching

tinygrad's JIT captures the sequence of kernel launches and replays them as a **kernel graph** — a pre-built command buffer submitted to the GPU in one shot. The graph batches multiple kernels into single GPU submissions (e.g., 32+64+128+30 = 254 kernels in 4 batches), virtually eliminating inter-kernel dispatch overhead.

Without JIT: each of the 254 kernels requires a Python → GPU round-trip.
With JIT GRAPH: 4 submissions total, GPU pipeline stays full.

---

## 11. Reproduction Commands

All from `/home/agent/jetpack-nixos/examples/tinygrad`:

```bash
# Baseline (before matvec fix — revert to commit 6977530dc)
nix develop -c bash -c 'cd tinygrad && NV=1 python3 -c "... benchmark ..."'
# → 3.85 tok/s

# After matvec fix, default params (commit 2439279b1)
nix develop -c bash -c 'cd tinygrad && NV=1 MV_THREADS_PER_ROW=32 python3 -c "..."'
# → 29.90 tok/s

# With BEAM search
nix develop -c bash -c 'cd tinygrad && NV=1 MV_THREADS_PER_ROW=32 JITBEAM=4 IGNORE_BEAM_CACHE=1 python3 -c "..."'
# → 36.71 tok/s (first run: 229s compile, then 27.2ms/tok)

# llama.cpp comparison
cd ../llama-cpp-orin && nix develop -c llama-bench \
  -m ~/.cache/tinygrad/llama3-1b-instruct/Llama-3.2-1B-Instruct-Q6_K.gguf \
  -p 128 -n 128
# → 25.62 tok/s

# Kernel profiling (use for analysis, not for perf measurement)
nix develop -c bash -c 'cd tinygrad && NV=1 MV_THREADS_PER_ROW=32 DEBUG=2 python3 -c "..." 2>&1 | tee /tmp/trace.txt'
```

---

## 12. Remaining Opportunities

### On-the-fly Q6_K dequant (potential 2-2.5× more)

Currently, tinygrad expands Q6_K weights to fp16 at model load time. Each token reads ~2.5 GB from memory. If the dequant were fused into the matmul kernel (like llama.cpp does), only ~0.97 GB would be read — a **2.5× reduction** in memory traffic.

This is the single biggest remaining optimization, worth potentially 2-2.5× more throughput. But it's hard — it requires the tinygrad scheduler to fuse the complex Q6_K bit-manipulation graph into the matmul kernel.

### Better default MV_THREADS_PER_ROW

The default of 8 is wrong for any GPU with warp size 32. Changing to 32 would give all NVIDIA users the 7.6× speedup out of the box, without requiring an env var.

### Batch > 1

At batch=1, NV ties or slightly beats CUDA on the same kernels, with BEAM giving NV a 15% edge. At batch≥8, the arithmetic intensity rises and NV's 26-50% matmul advantage should compound.

---

## 13. Summary of Numbers

### The Journey

```
3.85 tok/s  ─── Baseline (matvec heuristic broken)
     │
     │  ×4.8  (matvec fix with default TPR=8)
     ▼
18.34 tok/s
     │
     │  ×1.63  (tune MV_THREADS_PER_ROW to 32)
     ▼
29.90 tok/s  ─── Beats llama.cpp (25.62) by 17%
     │
     │  ×1.14  (JITBEAM=2)
     ▼
34.21 tok/s
     │
     │  ×1.07  (JITBEAM=4)
     ▼
36.71 tok/s  ─── Beats llama.cpp by 43%
              ─── Beats CUDA=1 by 15%
              ─── 9.5× total improvement
```

### Final Scoreboard

| Engine | tok/s | vs llama.cpp | Config |
|--------|------:|:-------------|--------|
| llama.cpp (no FA) | 25.62 | baseline | Hand-tuned C++ CUDA kernels |
| llama.cpp (FA=1) | 27.82 | 108.6% | + flash attention |
| **tinygrad NV=1** | **36.71** | **143%** | General-purpose Python compiler, no handwritten kernels |
| tinygrad CUDA=1 | 31.83 | 124% | Same compiler, CUDA dispatch |

**A general-purpose tensor compiler, written in Python, running through a homegrown GPU driver interface, is 43% faster than a hand-optimized C++ inference engine.** The fix: two lines of pattern matching + parameter tuning + autotuning.

---

## Appendix: File Locations

| File | What | Relevant Section |
|------|------|-----------------|
| `tinygrad/codegen/opt/heuristic.py` | Kernel optimization heuristics | Lines 65-85 (matvec section) |
| `tinygrad/runtime/ops_nv.py` | NV/Tegra backend (driver interface) | TegraIface class (~L780) |
| `tinygrad/apps/llm.py` | LLM inference app | `from_gguf()`, `generate()` |
| `tinygrad/nn/state.py` | GGUF loading / Q6_K dequant | Lines ~325-342 |
| `tinygrad/codegen/opt/postrange.py` | BEAM search entry point | `apply_opts()` |
| `tinygrad/engine/jit.py` | JIT and JITBEAM mechanism | TinyJit class |
| `beat-llama-cpp-results.md` | All benchmark results | Full file |
| `nv-optimization-plan.md` | Optimization roadmap | Full file |
