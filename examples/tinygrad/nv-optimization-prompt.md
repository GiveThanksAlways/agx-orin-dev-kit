# NV Backend Optimization — Handoff Prompt for Next Agent

You are continuing work on the tinygrad NV backend (TegraIface/HCQ) on a **Jetson Orin AGX 64GB** (JetPack 6, L4T r36.4.4, ga10b iGPU, SM 8.7, CUDA 12.6, 64 GB LPDDR5). The NV backend bypasses the CUDA driver API and talks directly to `/dev/nvgpu/igpu0/ctrl` and `/dev/nvmap` kernel interfaces.

## What's Already Done

A previous agent completed P0, P1, and P2. Read these for full context:

- **`P0-VA-window-fix-summary.md`** — Detailed write-up of the GPU VA window collision bug and fix
- **`nv-optimization-plan.md`** — The full roadmap with P0/P1/P2 marked ✅ DONE

### Summary of completed work

| Task | Status | Result |
|------|--------|--------|
| **P0: Fix crash** | ✅ Done | GPU VA window collision — nvgpu kernel placed user buffers at `shared_mem_window` (0xFE00000000). Fixed by reserving VA ranges via `ALLOC_SPACE`. LLaMA Q6_K inference now works. |
| **P1: Alloc alignment** | ✅ Done | 2MB nvmap alignment for allocations ≥ 8MB. Improves physical contiguity / SMMU TLB. |
| **P2: GPT-2 fp16 benchmark** | ✅ Done | NV=1 ~32ms/tok, CUDA=1 ~32ms/tok — **tied**. Both memory-bandwidth-bound at batch=1. |
| **`_ensure_has_local_memory` sync** | ✅ Done | Added `synchronize()` before `_realloc` of shader_local_mem to prevent use-after-free. |
| **`_copyin`/`_copyout` memmove** | ✅ Done (in HEAD) | Tegra uses direct memmove instead of DMA staging for host↔device copies. |
| **test_ops.py** | ✅ 409/409 passed | No regressions. |

### Current performance

| Workload | NV=1 | CUDA=1 | Gap |
|----------|------|--------|-----|
| LLaMA 3.2 1B Q6_K | 1.38 tok/s | 3.28 tok/s | **NV 2.4× slower** |
| GPT-2 124M fp16 | ~31 tok/s | ~31 tok/s | Tied |
| GPT-2 124M fp32 | 38.3 tok/s | 38.8 tok/s | Tied |
| fp16 matmul 2048² | 932 GFLOPS | 622 GFLOPS | **NV +50%** |

**Key insight:** NV=1 has 50% better matmul throughput, but at batch=1 LLM decode is entirely memory-bandwidth-bound (~0.5 FLOP/byte arithmetic intensity), so the matmul advantage is irrelevant. Both backends hit the same ~29 GB/s effective bandwidth wall.

## Your Mission: Beat llama.cpp

The ultimate goal is to make tinygrad's NV backend competitive with llama.cpp on this device. llama.cpp runs LLaMA 1B Q6_K at roughly **25-40 tok/s** on Orin AGX 64GB. tinygrad is currently at **1.38 tok/s** (NV=1) or **3.28 tok/s** (CUDA=1) — a **10-20× gap**.

This gap is NOT in the GPU driver interface (NV=1's micro-benchmarks are excellent). It's in:

1. **Kernel codegen quality** — tinygrad generates many small kernels vs llama.cpp's hand-fused CUDA kernels
2. **Dequantization inefficiency** — Q6_K dequant on tinygrad is extremely slow
3. **Scheduling overhead** — hundreds of kernel dispatches per token vs few large fused kernels
4. **Lack of custom quantized matmul kernels** — llama.cpp has hand-optimized Q6_K×fp16 GEMV

## Remaining Tasks (Priority Order)

### Task 1: Profile the bottleneck (FIRST — do this before optimizing anything)

Before writing any code, understand WHERE the time goes. Run a profiled LLaMA inference:

```bash
cd /home/agent/jetpack-nixos/examples/tinygrad
nix develop -c bash -c 'NV=1 DEBUG=2 python3 -c "
from tinygrad import Tensor, Device
from tinygrad.apps.llm import Transformer, models
from tinygrad.helpers import fetch, Timing
gguf = fetch(models[\"llama3.2:1b\"], \"Llama-3.2-1B-Instruct-Q6_K.gguf\", subdir=\"llama3-1b-instruct\")
model, kv = Transformer.from_gguf(Tensor(gguf))
tokens = [128000, 9906]
for i, tok in enumerate(model.generate(tokens)):
    if i >= 2: break
" 2>&1 | head -500'
```

Look for:
- How many kernels per token?
- Which kernels take the most time?
- Are there kernels that should be fused but aren't?
- How much time is dequant vs matmul vs attention vs other?

Also try `PROFILE=1` to get GPU-side kernel timing.

### Task 2: BEAM search for better kernels (P3 from roadmap)

```bash
# Try different BEAM levels
NV=1 BEAM=2 python3 tinygrad/examples/gpt2.py --model_size gpt2 --count 20 --temperature 0 --timing
NV=1 BEAM=4 python3 tinygrad/examples/gpt2.py --model_size gpt2 --count 20 --temperature 0 --timing
NV=1 JITBEAM=2 python3 tinygrad/examples/gpt2.py --model_size gpt2 --count 20 --temperature 0 --timing
```

BEAM search explores alternative kernel schedules. It might find significantly better schedules for Orin SM 8.7.

### Task 3: Optimize LLaMA Q6_K dequant throughput

NV=1 is 2.4× slower than CUDA=1 on Q6_K — the dequant code is the bottleneck. Investigate:

1. How does CUDA=1 handle the same dequant? Does it fuse operations differently?
2. Can tinygrad's scheduler fuse the dequant + matmul into fewer kernels?
3. Would a custom quantized GEMV kernel (like llama.cpp uses) help? The `extra/` dir might have examples.
4. Try `NOOPT=1` vs default to see if the optimizer is hurting Q6_K codegen.

### Task 4: Batch > 1 inference (P4)

This is where NV=1's 50% matmul advantage should shine:

```bash
# Try prompt processing (prefill) which is compute-bound
NV=1 python3 -c "
from tinygrad import Tensor, Device
from tinygrad.apps.llm import Transformer, models
from tinygrad.helpers import fetch, Timing
gguf = fetch(models['llama3.2:1b'], 'Llama-3.2-1B-Instruct-Q6_K.gguf', subdir='llama3-1b-instruct')
model, kv = Transformer.from_gguf(Tensor(gguf))
# Process a long prompt (compute-bound, not memory-bound)
tokens = list(range(128))  # 128-token prompt
with Timing('prefill: '):
    model(Tensor([tokens]))
Device['NV'].synchronize()
"
```

### Task 5: Compare with llama.cpp directly

Install and benchmark llama.cpp on the same model to establish a concrete target:

```bash
# Use the llama.cpp dev shell in this repo
cd /home/agent/jetpack-nixos/examples/llama-cpp-orin-nix-overlay
nix develop
# Or build from the flake:
# llama-benchmark, qwen3-server, etc. are available

# Benchmark with same model
llama-cli -m /home/agent/.cache/tinygrad/downloads/llama3-1b-instruct/Llama-3.2-1B-Instruct-Q6_K.gguf \
  -p "What is the answer to life" -n 20 --n-gpu-layers 99
```

### Task 6: Advanced TegraIface optimizations (P5)

If kernel codegen is the bottleneck (not dispatch), these won't help much. But if profiling shows dispatch overhead:

1. **Remove `_tegra_signal` workaround** — enables QMD chaining, reduces per-kernel overhead
2. **GPFIFO tuning** — try 2048/4096 entries
3. **Doorbell coalescing** — batch multiple submits, single doorbell write

## Environment

```bash
cd /home/agent/jetpack-nixos/examples/tinygrad
nix develop  # Enters shell with CUDA 12.6, Python 3.13
cd tinygrad   # The tinygrad source tree
```

- `NV=1` — NV/Tegra backend (direct nvgpu/nvmap)
- `CUDA=1` — Standard CUDA backend (cuLaunchKernel)
- Only set one at a time
- Always `rm -f /home/agent/.cache/tinygrad/cache.db` when changing compiler/codegen
- Check `sudo dmesg | tail -20` after runs for GPU errors

## Key Files

| File | What |
|------|------|
| `tinygrad/runtime/ops_nv.py` | NV backend — TegraIface (~L780), NVDevice (~L1300+), NVAllocator (~L344), GPFifo submission |
| `tinygrad/runtime/support/hcq.py` | HCQ framework — queues, signals, kernargs, `HCQProgram.__call__` dispatch loop |
| `tinygrad/renderer/ptx.py` | PTX code generation (vectorized loads, register allocation) |
| `tinygrad/nn/state.py` | GGUF model loading, Q6_K/Q4_K dequantization |
| `tinygrad/apps/llm.py` | Transformer/LLaMA model implementation |
| `P0-VA-window-fix-summary.md` | Write-up of the VA window collision fix |
| `nv-optimization-plan.md` | Full roadmap with benchmark data |

## Important Constraints

- **The tinygrad repo is upstream code** — keep patches minimal and clean for eventual contribution
- **The `_tegra_signal` workaround is required for correctness** — don't remove it without fixing the underlying QMD reuse race
- **The `_ensure_has_local_memory` synchronize is required** — prevents use-after-free of shader_local_mem during realloc
- **The VA window reservation is required** — without it, large workloads crash
- **ga10b does NOT support big pages** (`big_page_size=0`, `available_big_page_sizes=0x0`) — GPU page table is always 4KB. The 2MB alloc_align only improves nvmap physical contiguity, not GPU MMU page size.

## Build & Test

```bash
# Quick sanity test
NV=1 python3 -c 'from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())'

# Full test suite (409 tests, ~6 min)
NV=1 python3 -m pytest tinygrad/test/test_ops.py -x -q --tb=short

# LLaMA inference
NV=1 python3 -c "
from tinygrad import Tensor
from tinygrad.apps.llm import Transformer, models
from tinygrad.helpers import fetch
gguf = fetch(models['llama3.2:1b'], 'Llama-3.2-1B-Instruct-Q6_K.gguf', subdir='llama3-1b-instruct')
model, kv = Transformer.from_gguf(Tensor(gguf))
for i, tok in enumerate(model.generate([128000, 9906])):
    print(f'tok{i}={tok}')
    if i >= 5: break
"

# NixOS system build (if needed)
sudo nixos-rebuild switch --flake /home/agent/jetpack-nixos/examples/nixos#nixos-telemetry --show-trace
```
