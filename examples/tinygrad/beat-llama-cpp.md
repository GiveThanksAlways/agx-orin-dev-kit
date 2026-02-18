# Beat llama.cpp: A Tinygrad Roadmap for Jetson Orin AGX 64GB

**Date:** 2026-02-11  
**Target:** Beat or match llama.cpp decode performance (~25-28 tok/s) on LLaMA 3.2 1B Q6_K  
**Philosophy:** No handwritten kernels. Lean fully on tinygrad's compiler — scheduler, codegen, BEAM search, and the graph rewrite framework. Every improvement must be a general-purpose tinygrad improvement that benefits all models and backends.

**Current scores (LLaMA 3.2 1B Q6_K, Jetson Orin AGX 64GB):**

| Engine | Decode tok/s | Prefill (pp512) tok/s | Notes |
|--------|-------------|----------------------|-------|
| llama.cpp (no FA) | 25.61 | 1090 | Hand-tuned CUDA kernels |
| llama.cpp (FA=1) | 27.82 | 1392 | + flash attention |
| tinygrad CUDA=1 | 3.28 | N/A | Dequant to fp16, memory-bound |
| tinygrad NV=1 | CRASH | CRASH | MISALIGNED_ADDR in dequant kernel |

**The gap: 8× slower.** But the gap is not fundamental — it comes from tinygrad materializing dequantized weights instead of fusing dequant into matmul. This document is the prioritized plan to close it.

---

## Table of Contents

1. [Why tinygrad is 8× slower](#1-why-tinygrad-is-8-slower)
2. [The 6-phase plan](#2-the-6-phase-plan)
3. [Phase 0: Unblock NV=1 (Fix the crash)](#phase-0-unblock-nv1)
4. [Phase 1: Measure properly (baselines)](#phase-1-measure-properly)
5. [Phase 2: Keep weights quantized (the 2.6× win)](#phase-2-keep-weights-quantized)
6. [Phase 3: BEAM search tuning (the 1.3-1.5× win)](#phase-3-beam-search-tuning)
7. [Phase 4: Softmax and attention fusion (the 1.1-1.2× win)](#phase-4-softmax-and-attention-fusion)
8. [Phase 5: Memory system (the 1.1× win)](#phase-5-memory-system)
9. [Phase 6: Stretch goals](#phase-6-stretch-goals)
10. [Code locations quick reference](#code-locations)
11. [Test commands](#test-commands)
12. [Appendix: What llama.cpp does that we don't (yet)](#appendix)

---

## 1. Why tinygrad is 8× slower

### The arithmetic is simple

At batch=1, LLM decode is **memory-bandwidth-bound**. Each token requires reading all model weights once to produce one output token. The speed limit is:

```
tok/s = memory_bandwidth (GB/s) / bytes_read_per_token (GB)
```

**llama.cpp reads 0.967 GB/token** (quantized Q6_K weights, ~6.5 bits/param, 1B params).  
**tinygrad reads ~2.5 GB/token** (dequantized to fp16: 2 bytes/param, 1B params + overhead from dequant ops).

At Orin's ~102 GB/s effective memory bandwidth:
- llama.cpp theoretical max: 102 / 0.967 ≈ 105 tok/s → achieves 26 tok/s (~25% efficiency, normal for small models)
- tinygrad theoretical max: 102 / 2.5 ≈ 41 tok/s → achieves 3.3 tok/s (~8% efficiency, something else is also wrong)

### The three problems (in order of impact)

1. **Weight materialization (2.6×):** tinygrad dequantizes Q6_K → fp32 → fp16 and stores the result in memory. Every token re-reads 2.5 GB of fp16 weights. llama.cpp reads 0.967 GB of quantized weights and dequantizes inside the matmul kernel.

2. **Suboptimal kernel tuning (1.3-1.5×):** tinygrad's hand-coded heuristics pick reasonable but not optimal tile sizes, local memory usage, and unroll factors. The BEAM search can do better but isn't used by default.

3. **Extra kernel launches and missing fusions (1.1-1.2×):** Softmax is 3 kernels instead of 1. Multiple small ops (RMSNorm, RoPE, residual adds) that could be fused aren't. Each kernel launch has overhead.

Close all three and we reach: 3.3 × 2.6 × 1.4 × 1.15 ≈ 13.8 tok/s minimum, with potential to go higher if bandwidth efficiency improves (it should — fewer kernels = less launch overhead, better cache utilization).

---

## 2. The 6-Phase Plan

| Phase | What | Expected speedup | Difficulty | Dependencies |
|-------|------|-------------------|------------|--------------|
| 0 | Fix NV=1 crash on quantized models | Unblocks everything | Medium | None |
| 1 | Establish proper baselines | Measurement only | Easy | Phase 0 |
| 2 | Keep weights quantized in compute | 2-3× | Hard | Phase 1 |
| 3 | BEAM search kernel tuning | 1.3-1.5× | Medium | Phase 1 |
| 4 | Softmax and attention fusion | 1.1-1.2× | Medium-Hard | Phase 1 |
| 5 | Memory system optimizations | 1.05-1.1× | Easy | Phase 0 |
| 6 | Stretch goals (batch>1, speculative) | Varies | Hard | All above |

Phases 2-5 are **independent** and can be worked on in parallel after Phase 1 is done. Phase 2 is the single biggest win.

---

## Phase 0: Unblock NV=1

### Problem

Running any GGUF quantized model with `NV=1` crashes with `MISALIGNED_ADDR` GPU exception. The crash occurs during weight loading when the dequantization kernel runs.

### Root cause (verified)

The PTX renderer (before our fix) emitted vectorized loads like `ld.global.v4.u32` for data originating from GGUF blocks. Q6_K blocks are 210 bytes each — not a power of 2. When the scheduler reshapes the raw bytes into `(-1, 210)` and slices at non-aligned offsets (e.g., `blocks[:,128:192]`), the resulting load addresses aren't aligned to the vector width (16 bytes for v4.u32).

On desktop CUDA, the driver installs a trap handler that decomposes misaligned vector loads into scalar loads transparently. On Tegra (nvgpu kernel driver), **there is no trap handler** — the GPU's SM raises a machine check exception and the kernel is killed.

### Fix (ALREADY DONE in ptx.py)

We already patched `ptx.py` to decompose vector global loads/stores into scalar `ld.volatile` / `st.volatile` ops on the NV device. The `.volatile` modifier prevents ptxas/nvJitLink from merging them back into vector SASS instructions.

**Key functions added:**
- `_nv_decompose_load()` — at `tinygrad/renderer/ptx.py` line 94
- `_nv_decompose_store()` — at `tinygrad/renderer/ptx.py` line 109
- Integration into `string_rewrite` patterns — at `tinygrad/renderer/ptx.py` lines 115-145

### Validation needed

```bash
# Test that NV=1 can now load and run a quantized model
cd /home/agent/jetpack-nixos/examples/tinygrad
nix develop -c bash -c 'cd tinygrad && NV=1 HALF=1 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 5'
```

**Success criteria:** No crash. Gets a tok/s number (even if slow).

**If it still crashes:** The `.volatile` modifier may not be sufficient — ptxas may still merge loads. In that case:
1. Check the SASS output: `NV=1 DEBUG=6 python3 ...` and look for `LDG.E.128` (merged) vs `LDG.E.32` (scalar)
2. If merged: add explicit `bar.sync` between loads (heavyweight) or use `.cs` cache hint suffix
3. Alternative: pad the Q6_K block size to 256 bytes (wastes 22% memory but guarantees alignment)

### Time estimate: 1-2 hours (verification + potential fixup)

---

## Phase 1: Measure Properly

Before optimizing, we need clean baseline numbers for both `NV=1` and `CUDA=1` on multiple configurations.

### Benchmarks to run

```bash
# All from: cd /home/agent/jetpack-nixos/examples/tinygrad && nix develop -c bash -c '...'
# Pre-download the model once:
#   cd tinygrad && python3 -c "from tinygrad import Tensor; Tensor.from_url('https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf')"

# 1. tinygrad CUDA=1 baseline (should already work, ~3.3 tok/s)
CUDA=1 HALF=1 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 20

# 2. tinygrad NV=1 baseline (should now work after Phase 0)
NV=1 HALF=1 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 20

# 3. tinygrad with BEAM=2 (auto-tune kernel parameters)
NV=1 HALF=1 BEAM=2 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 5

# 4. tinygrad with JITBEAM=2 (beam search only in JIT, faster startup)
NV=1 HALF=1 JITBEAM=2 IGNORE_BEAM_CACHE=1 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 20

# 5. llama.cpp comparison (already done: 25.61 / 27.82 tok/s)
# See tests/results_llama3_llamacpp.log and tests/results_llama3_llamacpp_fa.log

# 6. Profile kernel-level breakdown
NV=1 HALF=1 DEBUG=2 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 3 2>&1 | grep -E "ran |ms"
```

### What to record

For each run, capture:
- **Decode tok/s** (steady state, after warmup)
- **Memory bandwidth** (GB/s from GlobalCounters)
- **Param bandwidth** (GB/s — how fast we're reading the model weights)
- **Number of kernels per token** (from DEBUG=2 output)
- **Time spent in dequant vs matmul vs attention** (from DEBUG=2 kernel timings)

### Kernel profiling

The DEBUG=2 output shows each kernel's execution time. For a single decode token, we want to see:
- How many kernels fire
- Which ones are slow (likely: dequantize, matmul for each of 16 transformer blocks)
- Total kernel time vs Python overhead

```bash
# Detailed per-kernel profiling for one token
NV=1 HALF=1 DEBUG=4 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 1 2>&1 | tail -200
```

### Time estimate: 2-3 hours

---

## Phase 2: Keep Weights Quantized (THE BIG WIN)

This is where the 2-3× speedup lives. It's also the hardest phase because it touches the scheduler, which is the most complex part of tinygrad.

### Current flow (wasteful)

```
GGUF file on disk (967 MB quantized)
    | ggml_data_to_tensor()        [lazy: produces computation graph]
    | .cast('float16')             [lazy: adds CAST node]
    | .contiguous()                [forces materialization into a flat fp16 buffer]
    | .realize()                   [executes: writes 2.5 GB of fp16 weights to memory]
    v
    -> matmul reads 2.5 GB of fp16 per token
```

### Ideal flow (what llama.cpp does)

```
GGUF file on disk (967 MB quantized)
    | keep in memory as raw quantized bytes (967 MB)
    v
    -> matmul kernel reads 967 MB quantized + dequantizes in registers -> accumulate
```

### The tinygrad way to achieve this

**The scheduler already has the machinery.** The key insight is that `ggml_data_to_tensor` returns a **lazy** Tensor — the dequant ops are nodes in the computation graph, not executed yet. If we don't call `.contiguous().realize()`, the dequant stays fused with whatever reads the weights next (i.e., the matmul in `nn.Linear.__call__`).

But the current code at `tinygrad/apps/llm.py` line 213-214 forces materialization:

```python
# NOTE: without this contiguous, it unpacks the weights from the model every time.
# we shouldn't need this, but for now it's faster
for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
```

The comment says "we shouldn't need this" — the tinygrad devs know this is wrong. The issue is that without `.contiguous()`, each forward pass re-executes the dequant computation graph from the raw GGUF tensor. This is slower because:

1. **The dequant graph is complex** — Q6_K has bit manipulation, shifts, masks, casts — and the scheduler may not fuse it efficiently with the matmul
2. **The raw GGUF tensor lives on disk** — every token would re-read from disk

### Strategy: Two sub-tasks

#### Task 2a: Load quantized weights into GPU memory (not disk)

The GGUF tensor starts on the default device but the raw bytes backing the quantized weights may not be efficiently laid out. We need to:

1. After `gguf_load()`, copy the raw quantized bytes to a contiguous GPU buffer (one `realize()` for the raw data)
2. Keep the lazy dequant graph pointing at the GPU-resident raw bytes
3. Remove the `.contiguous()` call so the dequant stays lazy and fuses with matmul

```python
# Proposed change in llm.py from_gguf():
nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)
# DON'T do .contiguous() -- let dequant stay lazy!
# DO realize the underlying raw quantized buffers so they're in GPU memory:
if realize: Tensor.realize(*params)
```

But this alone won't work — the scheduler needs to actually fuse the dequant into the matmul kernel.

#### Task 2b: Make the scheduler fuse dequant + matmul

This is the core challenge. When the matmul `A @ B` runs where B is a lazy dequant graph, the scheduler currently creates two kernels:
1. Kernel 1: dequant -> write fp16 output buffer
2. Kernel 2: matmul reading fp16 buffer

We need it to create one kernel:
1. Kernel 1: matmul that reads quantized bytes and dequants inline

**How the scheduler decides to materialize (create kernel boundaries):**

The scheduling logic is in `tinygrad/schedule/indexing.py`. The `run_rangeify()` function (line 200+) walks the computation graph and decides where to place kernel boundaries. It materializes (creates a new buffer) when:

- A node is used by multiple consumers (data reuse)
- A REDUCE_AXIS ends and new ranges begin
- PCONTIG heuristics decide fusion is too aggressive (buffer limit, mismatched ranges)

For dequant + matmul fusion:
- The dequant output is used by exactly ONE consumer (the matmul)
- The ranges should be compatible (both iterate over the weight dimensions)
- But the dequant has complex indexing (slicing at non-uniform offsets within 210-byte blocks) that may confuse the scheduler

**Concrete investigation steps:**

```bash
# 1. See the current schedule (what kernels are generated)
NV=1 HALF=1 DEBUG=3 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 1 2>&1 | head -500

# 2. Check if removing .contiguous() changes the schedule
# (Temporarily edit llm.py to comment out line 214)
NV=1 HALF=1 DEBUG=3 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 1 2>&1 | head -500

# 3. Use DEBUG_RANGEIFY to see the scheduling decisions
NV=1 HALF=1 DEBUG_RANGEIFY=1 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 1 2>&1 | head -1000

# 4. Try with PCONTIG (enables more aggressive fusion)
NV=1 HALF=1 PCONTIG=2 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 1 2>&1 | head -500
```

**If the scheduler does NOT fuse dequant+matmul (most likely):**

The fix is to modify the scheduling heuristics so that element-wise ops feeding into a single REDUCE_AXIS consumer with compatible ranges are not materialized. This is exactly what `PCONTIG` is designed to do, but it's marked as broken on PTX.

The relevant code is at `tinygrad/schedule/indexing.py` line 217:
```python
if all_all_same or (PCONTIG and all_same(local_rngs)):
```

And at `tinygrad/schedule/indexing.py` line 234:
```python
if not (PCONTIG > 1) or any(any(rr.arg > e.arg for e in ending_ranges[x]) for rr in r.ranges):
```

**The key question for the scheduler is:** can we inline the Q6_K dequant logic (bit shifts, masks, casts on 210-byte blocks) into a matmul kernel without blowing up register usage or making the kernel too large?

llama.cpp proves the answer is yes — their fused dequant+matmul kernels work fine on Ampere. The question is whether tinygrad's compiler can discover the same kernel structure automatically.

**Fallback approach if scheduler fusion is too hard:**

Create a `QuantizedLinear` layer that keeps weights as raw quantized bytes and does the dequant inside `__call__`:

```python
class QuantizedLinear:
  def __init__(self, raw_bytes: Tensor, n_elements: int, ggml_type: int, out_features: int):
    self.raw = raw_bytes  # stays quantized in memory
    self.n_elements = n_elements
    self.ggml_type = ggml_type
    self.shape = (out_features, n_elements // out_features)

  def __call__(self, x: Tensor) -> Tensor:
    # dequant is part of the computation graph, will be fused with matmul by scheduler
    w = ggml_data_to_tensor(self.raw, self.n_elements, self.ggml_type).reshape(self.shape)
    return x @ w.T
```

This is still "tinygrad philosophy" — no handwritten kernels, just expressing the computation differently so the scheduler can optimize it better.

### Expected speedup: 2-3×

Reading 0.967 GB instead of 2.5 GB → theoretical 2.6×. Minus some overhead for the inline dequant arithmetic, net ~2-2.5×.

### Time estimate: 1-3 weeks (this is the hard one)

---

## Phase 3: BEAM Search Kernel Tuning

### Problem

tinygrad's `hand_coded_optimizations` in `tinygrad/codegen/opt/heuristic.py` uses fixed heuristics to set tile sizes, local memory, and upcast factors. These work reasonably but aren't optimal for Orin's specific hardware (SM 8.7, 128KB L1, 2048 KB shared mem, 204 GB/s LPDDR5).

### Solution: BEAM search

BEAM search (`BEAM=N` env var) systematically explores the optimization space and benchmarks each variant. It's tinygrad's equivalent of autotuning.

From the CI benchmark config (`.github/workflows/benchmark.yml`), tinygrad's own team uses:
```bash
NV=1 HALF=1 JITBEAM=2 IGNORE_BEAM_CACHE=1 python3 examples/gpt2.py ...
```

### How it works

1. **`BEAM=N`** sets beam width during compile. Every kernel goes through BEAM search, including the first run.
2. **`JITBEAM=N`** uses BEAM only during JIT capture (second run of the function). First run uses default heuristics, JIT capture run uses BEAM. This is faster because prefill can start immediately.
3. The BEAM search tries different combinations of `OptOps.TC` (tensor cores), `OptOps.UPCAST`, `OptOps.LOCAL`, `OptOps.UNROLL`, `OptOps.GROUPTOP`, etc.
4. Results are cached to disk (`CACHELEVEL=2` by default) so subsequent runs don't re-search.

### How JITBEAM works (from engine/jit.py):

```python
# During JIT capture (2nd call), BEAM is set to JITBEAM value:
with Context(BEAM=getenv("JITBEAM", BEAM.value), NO_MEMORY_PLANNER=int(self.prune)):
    capturing.append(self)
```

### What to do

```bash
# Step 1: Baseline without BEAM
NV=1 HALF=1 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 20

# Step 2: With JITBEAM=2
NV=1 HALF=1 JITBEAM=2 IGNORE_BEAM_CACHE=1 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 20

# Step 3: With JITBEAM=4 (wider search, slower startup)
NV=1 HALF=1 JITBEAM=4 IGNORE_BEAM_CACHE=1 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 20

# Step 4: Use BEAM_PADTO to allow non-multiple-of-TC-dim shapes
NV=1 HALF=1 JITBEAM=2 BEAM_PADTO=1 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 20
```

### Tuning the matvec heuristic

For batch=1 decode, every linear layer is a matrix-vector multiply (matvec). The matvec heuristic in `tinygrad/codegen/opt/heuristic.py` (lines 65-80) uses env vars:

```bash
# Default matvec params
MV_BLOCKSIZE=4
MV_THREADS_PER_ROW=8
MV_ROWS_PER_THREAD=4

# Try different configs
NV=1 HALF=1 MV_BLOCKSIZE=8 MV_THREADS_PER_ROW=16 MV_ROWS_PER_THREAD=8 \
  python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 20
```

### Expected speedup: 1.3-1.5×

Based on tinygrad CI results where BEAM improves GPT-2 by 20-50% over hand-coded.

### Time estimate: 1-2 days (mostly benchmarking)

---

## Phase 4: Softmax and Attention Fusion

### Current state

**Softmax is 3 kernels:** The test at `test/null/test_schedule.py:938` confirms: `check_schedule(t, 3) # TODO: 1?` — the tinygrad devs know this should be 1 kernel but isn't yet.

The 3 kernels are:
1. Compute max (reduce)
2. Compute exp(x - max) and sum (reduce)
3. Divide by sum (element-wise)

Each kernel reads/writes the full attention matrix, adding memory traffic and launch overhead.

**Flash attention exists but isn't available on NV:**
- `FLASH_ATTENTION=1` env var gates `extra/thunder/tiny/fa.py` in `tinygrad/tensor.py` line 3592
- The Thunder TK flash attention kernel is 340 lines, uses bfloat16, WMMA ops
- Tests are AMD-only: `@skipIf(Device.DEFAULT not in ["AMD"])` in test_tk.py
- It uses bfloat16 throughout — Orin SM 8.7 supports bf16 tensor cores but fp16 is more commonly used

**PCONTIG/RANGEIFY fusion (the compiler-driven approach) exists but is broken on PTX:**
- `test_rangeify.py:9`: `@skipIf(PTXRenderer, "broken in LVP and PTX")`
- `test_softmax_fusion.py`: Multiple `@skip("needs RANGEIFY>1")`

### Strategy

#### 4a: Try FLASH_ATTENTION=1 (quick test)

Even though it's AMD-targeted, the Thunder TK kernel framework supports NVIDIA conceptually (`WARP_THREADS = 32` for non-AMD in `extra/thunder/tiny/tk/__init__.py`). Try it and see what happens:

```bash
NV=1 HALF=1 FLASH_ATTENTION=1 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 5
```

If it crashes (likely due to bf16 on a model using fp16), this confirms we need an fp16 adaptation.

#### 4b: Fix softmax fusion in the scheduler (the right way)

The softmax fusion is a scheduler problem in `tinygrad/schedule/indexing.py`. The RANGEIFY system needs to recognize that max-reduce → exp-subtract → sum-reduce → divide can all share the same ranges over the softmax dimension.

This is tracked by the tinygrad team (the `TODO: 1?` comment). If upstream fixes it, we get it for free. If not, we can investigate:

```bash
# See what RANGEIFY produces
NV=1 HALF=1 RANGEIFY=2 DEBUG_RANGEIFY=1 python3 -m tinygrad.apps.llm \
  --model llama3.2:1b --benchmark 1 2>&1 | head -500
```

#### 4c: RMSNorm and residual fusion

Each transformer block has:
```
x -> RMSNorm -> attention -> residual add -> RMSNorm -> FFN -> residual add
```

The RMSNorm and residual add are element-wise and should be fused with their consumers. Check if they are:

```bash
NV=1 HALF=1 DEBUG=2 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 1 2>&1 | grep "ran " | wc -l
```

Count the number of kernels per token. For LLaMA 3.2 1B with 16 blocks, the minimum is roughly:
- 16 × (QKV_matmul + KV_cache + attn_score + softmax + attn_out + FFN_gate + FFN_up + FFN_down) ≈ 128 kernels
- Ideal with fusion: ~64-80 kernels
- If it's much more than 128, things aren't fusing properly

### Expected speedup: 1.1-1.2×

Softmax fusion alone saves 2/3 of attention memory traffic for that stage. But attention is only ~10-15% of total time at batch=1, so net impact is 1.1-1.2×.

### Time estimate: 3-7 days

---

## Phase 5: Memory System

### 5a: Huge pages for TegraIface

The NVKIface (desktop) path at `tinygrad/runtime/ops_nv.py` line 494 uses 2MB huge pages for allocations >=8MB:
```python
page_size = (2 << 20) if size >= (8 << 20) else (4 << 10)
```

The TegraIface at `tinygrad/runtime/ops_nv.py` line 1166 always uses `mmap.PAGESIZE` (4KB) for the GPU page table:
```python
page_size = mmap.PAGESIZE  # GPU MMU page size -- always 4KB on Tegra ga10b
```

However, `alloc_align` does handle 2MB alignment already. The issue is the GPU's SMMU TLB — with 4KB pages, large buffers create many TLB entries, increasing TLB miss rate. On Tegra, the nvgpu GPU page table doesn't support big pages (ga10b has `big_page_size=0`), but the SMMU (ARM System MMU) that sits between the GPU and DRAM does support 2MB huge pages if the physical memory is 2MB-aligned.

**What to try:**
```python
# In TegraIface.alloc():
# Use mmap.MAP_HUGETLB for large allocations to request 2MB SMMU pages
import ctypes
MAP_HUGETLB = 0x40000  # Linux: force huge pages
if size >= (8 << 20) and not uncached and not host:
    addr = libc_so.mmap(ct.c_void_p(gpu_va), size,
                        mmap.PROT_READ | mmap.PROT_WRITE,
                        mmap.MAP_SHARED | MAP_FIXED | MAP_HUGETLB,
                        dmabuf_fd, 0)
```

**Validation:**
```bash
# Check if huge pages are available
cat /proc/meminfo | grep Huge
# Reserve some: echo 128 > /proc/sys/vm/nr_hugepages

NV=1 HALF=1 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 20
# Compare with baseline
```

### 5b: Kernel argument caching

The NV compute queue creates a new QMD (Queue Meta Data) for every kernel launch. For JIT-compiled inference where the same kernels repeat every token, the QMD can be pre-built and reused, reducing CPU-side setup.

This is already partially handled by TinyJit, which caches the program and replays the queue. But the QMD still gets rebuilt each time. Check if this matters:

```bash
# Compare JIT=0 vs JIT=1 to see JIT overhead reduction
NV=1 HALF=1 JIT=0 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 5
NV=1 HALF=1 JIT=1 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 5
```

### Expected speedup: 1.05-1.1×

### Time estimate: 1-2 days

---

## Phase 6: Stretch Goals

These are harder and less certain but could provide additional wins.

### 6a: Batch size > 1 (if applicable)

At batch=1, decode is purely memory-bandwidth-bound (FLOP/byte ≈ 0.5). At batch=8, the arithmetic intensity increases to ~4 FLOP/byte, which is closer to compute-bound territory where tensor cores shine.

If the use case supports batching (e.g., serving multiple users), this could unlock NV=1's matmul advantage.

### 6b: Speculative decoding

Generate N candidate tokens with a small "draft" model, then verify them in parallel with the full model. This turns N decode steps into 1 prefill step, which is much more GPU-efficient.

tinygrad can implement this at the application level without any compiler changes.

### 6c: KV cache quantization

The KV cache grows with context length. At 4096 context with LLaMA 1B, the KV cache is:
- 2 × 1 × 8 × 4096 × 64 × 2 = 8.4 MB (fp16)

This is small for LLaMA 1B but becomes significant at larger context lengths or larger models. Quantizing KV cache to int8 would halve the attention memory traffic.

### 6d: Continuous batching for server mode

The `llm.py` server mode (`--serve`) currently handles one request at a time. Continuous batching would allow multiple requests to share GPU compute.

---

## Code Locations

### Critical files to modify (relative to `tinygrad/tinygrad/`)

| File | What it does | Relevant to |
|------|-------------|-------------|
| `renderer/ptx.py` | PTX assembly generation | Phase 0 (misaligned loads) |
| `runtime/ops_nv.py` | TegraIface — nvgpu driver interface | Phase 0, 5 |
| `apps/llm.py` | LLM app (Transformer, GGUF loading, inference) | Phase 1, 2 |
| `nn/state.py` | `ggml_data_to_tensor`, `load_state_dict` | Phase 2 |
| `schedule/indexing.py` | Kernel boundary decisions (rangeify) | Phase 2, 4 |
| `schedule/rangeify.py` | Range propagation, PCONTIG, fusion | Phase 2, 4 |
| `codegen/opt/heuristic.py` | Hand-coded kernel optimizations | Phase 3 |
| `codegen/opt/search.py` | BEAM search implementation | Phase 3 |
| `codegen/opt/tc.py` | Tensor core definitions (cuda_sm80) | Phase 3 |
| `codegen/__init__.py` | `get_program`, `apply_opts` — compilation pipeline | Phase 3, 4 |
| `codegen/opt/postrange.py` | `apply_opts` — chooses BEAM vs hand_coded | Phase 3 |
| `tensor.py` | `scaled_dot_product_attention`, FLASH_ATTENTION gate | Phase 4 |
| `engine/jit.py` | TinyJit, JITBEAM mechanism | Phase 3 |

### GGUF Q6_K dequant path (nn/state.py, line 339)

```python
# Q6_K: 256 elements per 210-byte block
if ggml_type == 14:
    xl, xh = q_to_uint8(blocks[:,:128].reshape((-1, 2, 64)), 4), \
             q_to_uint8(blocks[:,128:192].reshape((-1, 2, 32)), 2).lshift(4)
    scales = blocks[:,192:208].bitcast(dtypes.int8).unsqueeze(-1).expand((-1, 16, 16)).reshape((-1, 256))
    d = blocks[:,-2:].bitcast(dtypes.float16).cast(dtypes.float32).expand((-1, 256))
    return d * (xl.bitwise_or(xh).bitcast(dtypes.int8) - 32).flatten(-2) * scales
```

Note the slices at offsets 0, 128, 192, 208 within 210-byte blocks — these cause the misalignment.

### Key environment variables

| Variable | Default | What it does |
|----------|---------|-------------|
| `NV=1` | 0 | Use NV/Tegra backend (TegraIface) |
| `CUDA=1` | 0 | Use CUDA driver API backend |
| `HALF=1` | 1 (in llm.py) | Cast weights to fp16 |
| `BEAM=N` | 0 | Beam search width for ALL kernels |
| `JITBEAM=N` | BEAM | Beam search width during JIT capture only |
| `JIT=1` | 1 | Enable TinyJit (kernel caching) |
| `DEBUG=N` | 0 | Verbosity (2=kernels, 3=schedule, 4=codegen, 6=PTX) |
| `FLASH_ATTENTION=1` | 0 | Use Thunder TK flash attention kernel |
| `PCONTIG=N` | 0 | Enable aggressive buffer fusion in scheduler |
| `RANGEIFY=N` | 0 | Enable experimental range analysis features |
| `DEBUG_RANGEIFY=1` | 0 | Print range analysis decisions |
| `MV_BLOCKSIZE=N` | 4 | Matvec: block size |
| `MV_THREADS_PER_ROW=N` | 8 | Matvec: threads per row |
| `MV_ROWS_PER_THREAD=N` | 4 | Matvec: rows per thread |
| `TC_OPT=N` | 0 (2 during BEAM) | Tensor core padding mode |
| `IGNORE_BEAM_CACHE=1` | 0 | Force re-run BEAM search ignoring cache |
| `SPLIT_REDUCEOP=1` | 1 | Split large reduce ops across kernels |

---

## Test Commands

All commands assume you're in `/home/agent/jetpack-nixos/examples/tinygrad` and use `nix develop -c bash -c '...'` to enter the environment.

### Quick validation

```bash
# Correctness: run core tensor ops
nix develop -c bash -c 'cd tinygrad && NV=1 python3 -m pytest test/test_ops.py -x -q --tb=short 2>&1 | tail -5'

# Correctness: run a few tokens
nix develop -c bash -c 'cd tinygrad && NV=1 HALF=1 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 3'
```

### Performance benchmarking

```bash
# Decode performance (main metric)
nix develop -c bash -c 'cd tinygrad && NV=1 HALF=1 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 20'

# With BEAM tuning
nix develop -c bash -c 'cd tinygrad && NV=1 HALF=1 JITBEAM=2 IGNORE_BEAM_CACHE=1 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 20'

# Kernel profiling
nix develop -c bash -c 'cd tinygrad && NV=1 HALF=1 DEBUG=2 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 1'
```

### llama.cpp comparison

```bash
# Build llama.cpp
cd /home/agent/jetpack-nixos/examples/llama-cpp-orin && nix build

# Run benchmark
./result/bin/llama-bench \
  -m ~/.cache/tinygrad/downloads/*/Llama-3.2-1B-Instruct-Q6_K.gguf \
  -p 512 -n 128 -ngl 999 -r 3
```

### System build

```bash
sudo nixos-rebuild switch --flake /home/agent/jetpack-nixos/examples/nixos#nixos-telemetry --show-trace
```

---

## Appendix: What llama.cpp Does That We Don't (Yet)

Understanding the enemy:

| Feature | llama.cpp | tinygrad (current) | tinygrad (goal) |
|---------|-----------|-------------------|-----------------|
| Fused dequant+matmul | Yes, hand-tuned CUDA kernels read Q6_K natively | No — dequant to fp16, store, then matmul | Scheduler fuses dequant into matmul |
| Flash attention | Yes, optional, +28% prefill | Available but AMD-only / broken on PTX | PCONTIG/RANGEIFY or adapted Thunder kernel |
| Tensor cores for matmul | Yes, uses WMMA | Yes, uses WMMA (via TC opt) | Already there, just needs better tiling via BEAM |
| Kernel fusion | Yes — RMSNorm+rope fused, gate+up fused | Some fusion via scheduler | Better heuristics or BEAM |
| KV cache | Contiguous, paged | Contiguous (cache_kv tensor) | Already comparable |
| Memory bandwidth util | ~25% of theoretical | ~8% of theoretical | ~20-25% (matching llama.cpp) |
| Quantized KV cache | Optional (Q8_0) | fp16 only | Future |
| Weight format | Reads GGUF natively | Dequants GGUF to fp16 | Reads GGUF natively (Phase 2) |

The single most impactful difference is **fused dequant+matmul**. Everything else is incremental. If we solve Phase 2, we should be within striking distance of llama.cpp.

---

## Success Metric

**Goal: >=20 tok/s on LLaMA 3.2 1B Q6_K with `NV=1`**

That would be within 25% of llama.cpp (no FA), which is remarkable for a general-purpose compiler with no hand-written kernels.

**Stretch goal: >=25 tok/s** — matching llama.cpp.

**Ultimate goal: >=28 tok/s** — beating llama.cpp with flash attention. This requires fused dequant+matmul + BEAM-tuned kernels + some attention fusion.

---

*Good Luck and God Speed.*
