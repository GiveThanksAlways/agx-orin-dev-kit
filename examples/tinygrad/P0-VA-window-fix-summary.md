# P0 Fix: GPU VA Window Collision — Root Cause & Resolution

**Date:** 2026-02-11
**Commit:** `6977530dc` on branch `nv-agx-orin-dev-kit`
**Tag:** `p0-va-window-fix`

---

## The Problem

`NV=1` LLaMA 3.2 1B Q6_K inference crashed with `hww_global_esr 4` (MISALIGNED_ADDR) on Jetson Orin AGX 64GB. Model **loading** worked (after prior fixes), but the first `model.generate()` call — specifically a simple `float4` zero-fill kernel `E_1048576_32_4` — caused a GPU hang every time.

## The Investigation

The crash was systematically narrowed down through a multi-session investigation:

| Hypothesis | Test | Result |
|-----------|------|--------|
| Race condition between kernels | Forced `synchronize()` after every dispatch | **Still crashed** — not a race |
| BumpAllocator wrapping | Computed cmdq/kernargs usage vs capacity | **Math disproves** — no wrap occurs |
| Kernel code bug | Isolated the crashing kernel, ran standalone | **Works fine** — context-dependent |
| CUDA vs NV backend | Ran same inference with `CUDA=1` | **Works** — NV-specific |
| Crashing kernel identity | Monkeypatched `NVRTCCompiler.compile` to capture source | **Captured** — trivial float4 zero-fill |

The crashing kernel source was a simple zero-fill:
```cuda
extern "C" __global__ void __launch_bounds__(32)
E_1048576_32_4(float* data0_134217728) {
  int gidx0 = blockIdx.x;
  int lidx0 = threadIdx.x;
  *((float4*)((data0_134217728+((gidx0<<7)+(lidx0<<2))))) = make_float4(0.0f,...);
}
```

## The Root Cause

Instrumenting `NVComputeQueue.exec()` to read the kernel's constant buffer revealed:

```
data0 ptr = 0xFE00200000
```

And the GPU's `shared_mem_window` is set to `0xFE00000000`.

**The data buffer was allocated at a VA address that falls inside the GPU's shared memory window.** When the kernel accesses `0xFE00200000`, the GPU hardware intercepts it as a shared memory access instead of a global memory read/write, causing the fault.

### Why It Happens

On Tegra's 40-bit GPU VA space (`[0x200000, 0xFFFFE00000]`), the NV backend configures two virtual windows:

- `local_mem_window  = 0xFD00000000` — GPU redirects accesses here to per-thread local memory
- `shared_mem_window = 0xFE00000000` — GPU redirects accesses here to per-CTA shared memory

The nvgpu kernel's VA allocator (`MAP_BUFFER_EX` with `flags=0`) assigns GPU VAs from the top of the address space downward. For large workloads like LLaMA 1B (~3.8 GB of model buffers + ~500 MB of intermediate buffers), enough VA space is consumed that the allocator places a 512 MB buffer at `0xFE00200000` — right inside the shared memory window.

The desktop NV backend avoids this by placing windows at `0x729300000000` / `0x729400000000` — addresses in the 43-bit+ range far above any user allocation. Tegra's 40-bit space has no such luxury.

## The Fix

After creating the GPU address space, reserve 1 GB VA ranges at both window addresses using `NVGPU_AS_IOCTL_ALLOC_SPACE` with `FIXED_OFFSET`:

```python
# In TegraIface.rm_alloc(), after ALLOC_AS:
for window_va in [0xFD00000000, 0xFE00000000]:
    rsv = _nvgpu_as_alloc_space_args()
    rsv.pages = 0x40000000 // mmap.PAGESIZE  # 1GB
    rsv.page_size = mmap.PAGESIZE
    rsv.flags = 0x1  # NVGPU_AS_ALLOC_SPACE_FLAGS_FIXED_OFFSET
    rsv.offset = window_va
    _tegra_ioctl(self._as_fd, _NVGPU_AS_IOCTL_ALLOC_SPACE, rsv)
```

This tells the kernel VA allocator those ranges are taken, preventing it from assigning user buffers there.

## Code Changes

**`tinygrad/runtime/ops_nv.py`** (3 changes):
1. Added `_nvgpu_as_alloc_space_args` ctypes structure
2. Added `_NVGPU_AS_IOCTL_ALLOC_SPACE` ioctl constant
3. Added VA reservation loop in `rm_alloc(NV01_MEMORY_VIRTUAL)` after AS creation

**`tinygrad/runtime/support/hcq.py`** (cleanup):
- Removed debug tracing code from previous debugging sessions

## Verification

| Test | Result |
|------|--------|
| LLaMA 3.2 1B Q6_K inference (NV=1) | **WORKS** — 1.38 tok/s, correct output (tok0=11, tok1=358, tok2=2846, tok3=3411) |
| GPT-2 124M fp16 (NV=1) | **WORKS** — 32.05 ms/tok (~31 tok/s) |
| GPT-2 124M fp16 (CUDA=1 baseline) | 32.35 ms/tok (~31 tok/s) |
| `dmesg` GPU faults | **Zero** |
| `test_ops.py` (NV=1) | **409/409 passed** |

## Performance Summary

| Workload | NV=1 | CUDA=1 | Notes |
|----------|------|--------|-------|
| LLaMA 1B Q6_K decode | 1.38 tok/s | 3.28 tok/s | NV slower — dequant kernels unoptimized |
| GPT-2 fp16 decode | ~31 tok/s | ~31 tok/s | Tied — both memory-bandwidth-bound |

## What's Left

The NV backend now **works correctly** for both fp16 and quantized models. Remaining optimization opportunities:

1. **Dequant kernel optimization** — NV=1 is 2.4x slower than CUDA=1 on Q6_K. The dequant kernels are complex and may benefit from BEAM search or hand-tuning.
2. **Batch > 1 inference** — NV=1's 50% matmul advantage only shows at batch ≥ 8+ where inference becomes compute-bound.
3. **Beat llama.cpp** — llama.cpp on this device runs LLaMA 1B Q6_K at ~25-40 tok/s. tinygrad (either backend) is far behind, suggesting the bottleneck is in the ML framework / kernel codegen, not the GPU driver interface.
