# NV Backend Optimization Plan — Jetson Orin AGX 64GB

**Goal:** Make `NV=1` (TegraIface/HCQ) decisively faster than `CUDA=1` on real LLM inference, and unblock quantized model support.

**Device:** Jetson Orin AGX 64GB, JetPack 6, L4T r36.4.4, ga10b iGPU, SM 8.7, 64 GB LPDDR5 (~204 GB/s theoretical)

---

## Current State (Updated 2026-02-12)

### ✅ MISSION ACCOMPLISHED: NV=1 Beats llama.cpp by 43%

| Config | tok/s | ms/tok | vs llama.cpp |
| ------ | -----:| ------:| :----------- |
| llama.cpp (no FA) | 25.62 | 39.0 | baseline |
| llama.cpp (FA=1) | 27.82 | 36.0 | 108.6% |
| tinygrad NV=1 (before matvec fix) | 3.85 | 260.0 | 15% |
| tinygrad NV=1 + matvec fix (heuristic, MV_TPR=32) | 29.90 | 33.4 | **117%** |
| tinygrad NV=1 + matvec + JITBEAM=2 | 34.21 | 29.2 | **134%** |
| **tinygrad NV=1 + matvec + JITBEAM=4** | **36.71** | **27.2** | **143%** |
| tinygrad CUDA=1 + matvec fix (heuristic) | 29.97 | 33.4 | 117% |
| tinygrad CUDA=1 + matvec + JITBEAM=4 | 31.83 | 31.4 | 124% |

### NV=1 Now Beats CUDA=1 on LLM Decode

With BEAM search, NV=1 is **15% faster than CUDA=1** (36.71 vs 31.83). BEAM helps NV more (+23%) than CUDA (+6%) because NV's lower dispatch overhead means the GPU pipeline stays fuller when kernel configs are optimized.

### Where NV=1 Already Wins (Micro-Benchmarks, from Phase E)

| Benchmark                  | NV=1       | CUDA=1     | NV Advantage    | Why NV Wins                                                         |
| -------------------------- | ---------- | ---------- | --------------- | ------------------------------------------------------------------- |
| **fp16 matmul 2048×2048**  | 932 GFLOPS | 622 GFLOPS | **+50%**        | Direct QMD dispatch skips CUDA driver overhead; same tensor core HW |
| **fp16 matmul 1024×1024**  | 388 GFLOPS | 307 GFLOPS | **+26%**        | Lower launch overhead is proportionally larger for smaller problems |
| **Element-wise 10M (add)** | 47% faster | baseline   | **+47%**        | Near-zero launch overhead via direct GPFIFO/QMD                     |
| **Copyout (D→H) 16MB**     | 6.35 GB/s  | 2.58 GB/s  | **+146%**       | Direct memmove on unified memory                                     |
| **Kernel launch p99**      | 1199 µs    | 2637 µs    | **2.2× better** | More deterministic — no CUDA driver jitter                          |

---

## Priority Roadmap

### P0 — ✅ DONE: Fix Misaligned Access Crash (Unblocks Quantized Models)

**Status:** Fixed in commit `6977530dc`, tagged `p0-va-window-fix`. See `P0-VA-window-fix-summary.md` for full write-up.

**Actual Root Cause:** NOT the dequant alignment issue originally hypothesized. The real bug was that the nvgpu kernel VA allocator placed user data buffers at GPU VA addresses overlapping with the `shared_mem_window` (`0xFE00000000`) and `local_mem_window` (`0xFD00000000`). When the GPU accessed these buffers, it intercepted the access as a window operation instead of a global memory access, causing `MISALIGNED_ADDR` faults.

**Fix:** Reserve 1GB VA ranges at both window addresses using `NVGPU_AS_IOCTL_ALLOC_SPACE` after creating the GPU address space, preventing the allocator from placing user buffers there.

**Results:** LLaMA 3.2 1B Q6_K inference works at 1.38 tok/s (NV=1). GPT-2 fp16: ~31 tok/s. test_ops: 409/409 passed.

<details><summary>Original (incorrect) root cause analysis</summary>

~~**Impact:** Currently NV=1 **cannot run any quantized model at all** — LLaMA Q6_K crashes with GPU SM exception.~~

~~**Root Cause:** Q6_K uses 210-byte blocks. The dequantization code in `nn/state.py` slices these blocks at odd offsets:~~

```python
# nn/state.py line ~340 (ggml_type == 14, Q6_K)
blocks = t[:(n//256)*210].reshape((-1, 210))  # 210-byte rows
xl, xh = q_to_uint8(blocks[:,:128].reshape((-1, 2, 64)), 4), \
         q_to_uint8(blocks[:,128:192].reshape((-1, 2, 32)), 2).lshift(4)
scales = blocks[:,192:208].bitcast(dtypes.int8)
d = blocks[:,-2:].bitcast(dtypes.float16)
```

When `blocks` starts at an aligned address, block `i` starts at `base + i*210`. For `i=1`, that's `base+210` — **not aligned to any power of 2**. The PTX renderer then emits vectorized loads like `ld.global.v4.u32` which require 16-byte alignment. CUDA driver silently handles misaligned vectorized loads via trap-and-emulate; the NV/Tegra backend enforces strict hardware alignment and crashes.

**dmesg evidence:**

```
nvgpu: sm machine check err. gpc_id(0), tpc_id(0)
hww_warp_esr 0x10 (MISALIGNED_ADDR)
hww_warp_esr_pc 0xff55a2a830
```

**Three Fix Options:**

| Option              | Approach                                                         | Pros                                                     | Cons                                        |
| ------------------- | ---------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------- |
| **A (Recommended)** | Force scalar loads in PTX renderer when stride is not power-of-2 | Minimal code change, correct by construction             | Slightly slower dequant (~5-10%)            |
| **B**               | Pad Q6_K blocks to 256 bytes before dequant                      | Aligned access, vectorized loads work                    | Uses 22% more memory, changes `nn/state.py` |
| **C**               | Alignment check in tinygrad scheduler/linearizer                 | Prevents vectorized loads when alignment can't be proven | Complex, may miss cases                     |

**Code locations:**

- PTX vectorized load emission: `tinygrad/renderer/ptx.py` lines ~107-119 (the `ld.{mem_type}.v{count}` patterns)
- Q6_K dequant: `tinygrad/nn/state.py` lines ~325-342
- Q4_K dequant: `tinygrad/nn/state.py` lines ~333-339 (144-byte blocks, same class of problem)

**Verification:** After fix, run:

```bash
cd /home/agent/jetpack-nixos/examples/tinygrad
nix develop -c bash -c 'cd tinygrad && NV=1 python3 examples/llama3.py --size 1B --no_api --benchmark --timing 2>&1 | tee tests/results_llama3_nv_fixed.log'
```

</details>

---

### P1 — ✅ DONE: Enable Huge Pages in TegraIface (Allocation Alignment)

**Status:** Implemented in commit `a50a60925`. Uses 2MB `alloc_align` for allocations ≥ 8MB (nvmap physical contiguity + SMMU TLB improvement). Note: the GPU page table `page_size` remains 4KB on ga10b (big_page_size=0), but the nvmap allocation alignment is increased to improve physical contiguity.

~~**Impact:** Reduce TLB misses for large allocations, improving memory bandwidth.~~

**Original analysis (kept for reference):** `TegraIface.alloc()` hardcodes `page_size = mmap.PAGESIZE` (4 KB) for all allocations. The desktop `NVKIface.alloc()` uses 2 MB huge pages for allocations ≥ 8 MB:

```python
# NVKIface.alloc() (ops_nv.py ~L494) — does this:
page_size = (2 << 20) if size >= (8 << 20) else mmap.PAGESIZE

# TegraIface.alloc() (ops_nv.py ~L1166) — always does:
page_size = mmap.PAGESIZE  # 4 KB, always
```

For a 1B parameter model with ~4.9 GB of weight tensors, that's **1.2 million 4KB pages** vs **2,450 huge pages**. The TLB on SM 8.7 has ~32 entries per SM — with 4KB pages, nearly every weight access is a TLB miss.

**Fix:** Mirror NVKIface's logic in TegraIface:

```python
# In TegraIface.alloc():
page_size = (2 << 20) if size >= (8 << 20) else mmap.PAGESIZE
```

Also need to verify that `nvgpu_as_map_buffer_ex` and `nvmap` support 2MB pages on Tegra — check `_NVGPU_AS_MAP_BUFFER_FLAGS_LARGE_PAGES` flag.

**Code location:** `tinygrad/runtime/ops_nv.py` line ~1166, `TegraIface.alloc()`

**Expected impact:** 10-30% improvement on memory-bandwidth-bound workloads (which includes all batch=1 LLM decode).

---

### P2 — ✅ DONE: Run GPT-2 with HALF=1 (fp16)

**Status:** Benchmarked. Both backends tied at ~31 tok/s (~32 ms/token). Confirms GPT-2 124M is purely **memory-bandwidth-bound** even at fp16 batch=1 — the 50% matmul micro-benchmark advantage does not translate because matmul is not the bottleneck.

**Results:**

| Backend       | ms/token |
| ------------- | -------- |
| NV=1 HALF=1   | 32.05    |
| CUDA=1 HALF=1 | 32.35    |

**Conclusion:** To see NV=1's matmul advantage in end-to-end inference, we need either batch > 1 (P4) or a much larger model where compute dominates.

---

### P2.5 — ✅ DONE: Matvec Heuristic Fix (THE BREAKTHROUGH — 7.6× speedup)

**Status:** Fixed in commit `2439279b1`. This was the single biggest win — **7.6× speedup** (3.85 → 29.90 tok/s). Combined with JITBEAM=4, achieves **36.71 tok/s** (9.5× total improvement).

**Root Cause:** The matvec heuristic in `tinygrad/codegen/opt/heuristic.py` was **never triggering** for LLM matmul kernels. It expected the pattern `REDUCE(MUL(INDEX, INDEX))` but encountered:

1. **fp16→fp32 accumulation wraps in CAST:** `REDUCE(CAST(MUL(INDEX, INDEX)))`
2. **Fused RMSNorm+matmul has nested MULs:** `REDUCE(MUL(MUL(x, norm), weight))`

Without matvec, all matmul kernels fell through to the generic `GROUPTOP(16)` heuristic:
- Only **16 threads per block** (half a warp!)
- Non-coalesced memory access (threads 262,144 elements apart)
- No vectorization, scalar half loads
- **1.5 GB/s** of the ~100+ GB/s available — 1.5% utilization

**Fix (two changes to `hand_coded_optimizations()`):**

1. **CAST unwrap:** `if mulop.op is Ops.CAST: mulop = mulop.src[0]`
2. **Recursive INDEX finder:** Instead of rigid `MUL(INDEX, INDEX)`, recursively find INDEX nodes through MUL/CAST chains up to depth 3. Handles fused RMSNorm→matmul.

**After fix with `MV_THREADS_PER_ROW=32`:**
- 128 threads per block (32 GROUP × 4 LOCAL)
- Coalesced memory access via GROUP reduction
- 4-way UPCAST for ILP
- **43–52 GB/s** bandwidth — 42-51% utilization

**MV Parameter Tuning Results:**

| MV_TPR | BS | RPT | tok/s | notes |
|--------|-----|-----|-------|-------|
| 8 | 4 | 4 | 18.34 | defaults |
| **32** | **4** | **4** | **29.90** | **best heuristic** |
| 32 | 2 | 4 | 29.03 | |
| 32 | 4 | 2 | 28.20 | |
| 64 | 4 | 4 | 25.93 | |

**Code location:** `tinygrad/codegen/opt/heuristic.py` lines 65-85 (the matvec section)

---

### P3 — ✅ DONE: Kernel Optimization with BEAM Search

**Status:** Benchmarked. JITBEAM=4 gives an additional 23% speedup on NV=1 on top of the matvec fix.

**Results:**

| Config | NV=1 tok/s | CUDA=1 tok/s | NV Advantage |
| ------ | ----------:| -----------:| :----------- |
| Heuristic only (MV_TPR=32) | 29.90 | 29.97 | Tied |
| JITBEAM=2 | 34.21 | — | — |
| JITBEAM=4 | 36.71 | 31.83 | **+15%** |

**Key finding:** BEAM benefits NV=1 much more than CUDA=1 (+23% vs +6%). This confirms that NV's lower dispatch overhead amplifies optimized kernel configs.

---

### P4 — MEDIUM: Batch > 1 Inference (Where Compute Matters)

**Impact:** NV=1 now beats CUDA=1 at batch=1 by 15%. At batch ≥ 8, NV's matmul advantage (26-50%) should compound further.

**Status:** Not tested. Now that batch=1 is solved, this is the natural next step for server workloads.

---

### P5 — LOW: Advanced TegraIface Optimizations

These are smaller optimizations identified in the codebase:

| #   | Optimization                          | Code Location                                     | Expected Impact                                                                                                                                           |
| --- | ------------------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Remove `_tegra_signal` workaround** | `ops_nv.py` L148-149, L183-193, L1329             | Enables QMD chaining for lower dispatch overhead. Requires fixing the QMD reuse race at the scheduler level instead of the current pushbuffer workaround. |
| 2   | **GPFIFO entry count tuning**         | `NVDevice._new_gpu_fifo()` — `tegra_entries=1024` | Try 2048/4096 entries to reduce ring wrap overhead                                                                                                        |
| 3   | **Doorbell write coalescing**         | `NVCommandQueue._submit_to_gpfifo()`              | Batch N GPFIFO entries, single doorbell write                                                                                                             |
| 4   | **QMD field tuning**                  | `NVComputeQueue.exec()`                           | Compare QMD fields against CUDA's QMD for same kernel                                                                                                     |
| 5   | **VA recycling**                      | `TegraIface` — `_alloc_gpu_vaddr` not implemented | Prevent VA space exhaustion in long-running workloads                                                                                                     |

---

## Kernel-Level Profile (Post Matvec Fix)

Per-token decode at 29.9 tok/s (heuristic, MV_TPR=32). 230 kernels, 16 transformer layers:

| Kernel | Count | Total (µs) | % | BW (GB/s) | Role |
|--------|------:|----------:|----:|----------:|------|
| `r_512_32_4_4_64` | 30 | 20,560 | 37.4% | 49 | gate/up proj (2048→8192) |
| `r_128_32_4_4_256` | 15 | 10,924 | 19.9% | 46 | down proj (8192→2048) |
| `r_8016_32_4_4_64` | 1 | 10,110 | 18.4% | 52 | lm_head (2048→128256) |
| `r_128_32_4_4_64` | 30 | 5,931 | 10.8% | 43 | Q/O proj (2048→2048) |
| attention/RMSNorm/other | 124 | 7,394 | 13.5% | various | softmax, norms, RoPE, KV |

The lm_head alone reads ~500MB of fp16 weights per token — 20% of total weight data for one kernel.

## Remaining Optimization Opportunities

Now that tinygrad beats llama.cpp, these are the next frontiers:

### HIGH: On-the-fly Q6_K Dequant (potential 2-2.5× further speedup)

Weights are currently expanded to fp16 in memory. Each token reads ~2.5 GB. If dequant were fused into the matmul kernel (like llama.cpp), only ~0.97 GB would need to be read per token. This is the single biggest remaining opportunity.

**Challenge:** Requires the tinygrad scheduler to fuse the complex Q6_K dequant graph (bit shifts, masks, casts on 210-byte blocks) into the matmul kernel. The `.contiguous()` call in `llm.py` line 214 currently forces materialization.

### MEDIUM: Better Default MV_THREADS_PER_ROW

The default `MV_THREADS_PER_ROW=8` is suboptimal for warp-size-32 GPUs. Changing to 32 for NVIDIA devices would improve out-of-the-box performance from 18.3 to 29.9 tok/s without requiring env var overrides.

### MEDIUM: Batch > 1 / Prefill Performance

Not benchmarked yet. NV's matmul advantage should compound at higher batch sizes.

### LOW: Softmax Fusion

Softmax currently executes as 3 separate kernels. Fusing into 1 would reduce memory traffic for the attention stage. Impact is small (~3% of total time at batch=1).

---

## Success Criteria

| Milestone                                  | What It Proves                     | Priority      | Status                       |
| ------------------------------------------ | ---------------------------------- | ------------- | ---------------------------- |
| NV=1 runs LLaMA Q6_K without crash         | P0 fix works                       | **Must have** | ✅ Done                       |
| NV=1 beats CUDA=1 on GPT-2 HALF=1          | fp16 advantage translates          | High          | ❌ Tied (mem-bw bound)        |
| NV=1 beats CUDA=1 on LLaMA Q6_K decode     | Quantized model perf win           | High          | ✅ 36.71 vs 31.83 (+15% with BEAM) |
| NV=1 beats CUDA=1 at batch=8+              | Compute-bound advantage translates | Medium        | Not tested                   |
| NV=1 shows better p99 in sustained serving | Latency advantage translates       | Medium        | Not tested                   |
| **Beat llama.cpp on LLaMA 1B Q6_K**        | tinygrad competitive with C++      | **Stretch**   | ✅ **36.71 vs 25.62 (+43%)** |

---

## File References

| File                                       | What                                           | Key Lines                                                             |
| ------------------------------------------ | ---------------------------------------------- | --------------------------------------------------------------------- |
| `tinygrad/tinygrad/runtime/ops_nv.py`      | NV backend (TegraIface, NVDevice, NVAllocator) | L1155-1220 (alloc), L148-193 (tegra_signal), L343-362 (direct memcpy) |
| `tinygrad/tinygrad/renderer/ptx.py`        | PTX code generation                            | L107-119 (vectorized load/store emission)                             |
| `tinygrad/tinygrad/nn/state.py`            | GGUF loading / dequantization                  | L325-342 (Q6_K/Q4_K block layout)                                     |
| `tinygrad/tinygrad/runtime/support/hcq.py` | HCQ framework (signals, queues, kernargs)      | L361 (kernargs_size), L382-383 (BumpAllocator)                        |
| `tests/benchmark_nv_vs_cuda.py`            | All micro-benchmarks                           | Full file                                                             |
| `robust-testing-and-performance.md`        | Master tracking doc                            | Phase E section (~L498)                                               |
