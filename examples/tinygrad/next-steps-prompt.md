# Prompt: Tinygrad NV Backend — Next Steps

**Date:** 2026-02-12  
**Device:** Jetson Orin AGX 64GB, JetPack 6, L4T r36.4.4, SM 8.7, CUDA 12.6, 64 GB LPDDR5  
**tinygrad branch:** `nv-agx-orin-dev-kit` at commit `2439279b1`  
**Workspace:** `/home/agent/jetpack-nixos/examples/tinygrad`

---

## Context: Where We Are

We made tinygrad's NV backend beat llama.cpp on LLaMA 3.2 1B Q6_K decode:

| Config | tok/s | vs llama.cpp |
|--------|------:|:-------------|
| llama.cpp (no FA) | 25.62 | baseline |
| tinygrad NV=1 + matvec fix + MV_TPR=32 | 29.90 | 117% |
| tinygrad NV=1 + matvec fix + JITBEAM=4 | 36.71 | **143%** |
| tinygrad CUDA=1 + JITBEAM=4 | 31.83 | 124% |

The key fix was in `tinygrad/codegen/opt/heuristic.py` — the matvec pattern match was silently failing on every LLM matmul kernel due to CAST/MUL wrapping. Details in `Learning-Beat-Llama-Cpp.md` and `beat-llama-cpp-results.md`.

### What our changes are (relative to upstream tinygrad)

All changes are on branch `nv-agx-orin-dev-kit`. Only 4 files are modified from upstream:

1. **`tinygrad/codegen/opt/heuristic.py`** — Matvec heuristic: CAST unwrap + recursive INDEX finder (commit `2439279b1`)
2. **`tinygrad/runtime/ops_nv.py`** — TegraIface: VA window reservation fix (P0), 2MB alloc_align for large buffers (P1)
3. **`tinygrad/renderer/ptx.py`** — Decompose vectorized global loads/stores into scalar ops for NV backend (fixes MISALIGNED_ADDR on Tegra)
4. **`tinygrad/runtime/support/compiler_cuda.py`** — CUDA_INCLUDE_PATH for NixOS

---

## Task 1: Run the Test Suite (Regression Check)

**Priority: HIGH — do this first before any new changes.**

Our heuristic.py change affects ALL kernels, not just LLM matmuls. We need to verify nothing is broken.

### Commands

```bash
cd /home/agent/jetpack-nixos/examples/tinygrad
nix develop -c bash -c 'cd tinygrad && NV=1 python3 -m pytest test/test_ops.py -x -v --tb=short 2>&1 | tail -30'
```

Run these test suites in order of importance:

| Test | Command | What it covers | Expected |
|------|---------|---------------|----------|
| **Core ops** | `NV=1 python3 -m pytest test/test_ops.py -x -v --tb=short` | 409 tensor operations | All pass (verified before) |
| **Schedule** | `NV=1 python3 -m pytest test/test_schedule.py -x -v --tb=short` | Kernel fusion, scheduling | Should pass |
| **GGUF loading** | `NV=1 python3 -m pytest test/unit/test_gguf.py -x -v --tb=short` | Quantized weight loading | Should pass |
| **LLM server** | `NV=1 python3 -m pytest test/unit/test_llm_server.py -x -v --tb=short` | Transformer generate() | Should pass |
| **HCQ device** | `NV=1 python3 -m pytest test/device/test_hcq.py -x -v --tb=short` | NV/HCQ backend specifics | Should pass |
| **Linearizer** | `NV=1 python3 -m pytest test/test_linearizer.py -x -v --tb=short` | Kernel codegen | May have slow tests |
| **JIT** | `NV=1 python3 -m pytest test/test_jit.py -x -v --tb=short` | JIT compilation | Should pass |

If any test fails, investigate whether it's a pre-existing failure (try reverting to `6977530dc` and re-running) or caused by our changes.

Also run with CUDA to compare:
```bash
nix develop -c bash -c 'cd tinygrad && CUDA=1 python3 -m pytest test/test_ops.py -x -v --tb=short'
```

### Environment note

Tests must run inside `nix develop -c bash -c '...'` because the flake provides CUDA toolkit, Python, and correct LD_LIBRARY_PATH. The `nix develop` shell also sets `PYTHONPATH` to include the local tinygrad checkout.

---

## Task 2: Batch > 1 Benchmarking (NV vs CUDA vs llama.cpp)

**Priority: MEDIUM — extends our benchmarking to the compute-bound regime.**

At batch=1, inference is memory-bandwidth-bound and NV/CUDA perform similarly on heuristic-only configs. At batch≥4, arithmetic intensity rises and NV's 26-50% matmul advantage (from micro-benchmarks) should start to show.

### What "batch > 1" means for LLM decode

There are two ways to increase batch:
1. **Prefill** — processing multiple prompt tokens at once. `Transformer.__call__` already supports `T > 1` via the `start_pos` mechanism. Measure this by passing a longer prompt.
2. **Multi-request batching** — processing multiple independent prompts simultaneously. This requires modifying `generate()` to handle `B > 1`. The model architecture supports it (shapes are `(B, T, D)`) but `generate()` hardcodes `B=1`.

### Measuring prefill throughput

Prefill is the easier benchmark — just use a longer prompt:

```bash
# tinygrad prefill benchmark (different prompt lengths)
nix develop -c bash -c 'cd tinygrad && NV=1 MV_THREADS_PER_ROW=32 python3 -c "
from tinygrad.apps.llm import Transformer, models
from tinygrad import Tensor
from tinygrad.helpers import fetch
import time

gguf = fetch(models[\"llama3.2:1b\"], \"Llama-3.2-1B-Instruct-Q6_K.gguf\", subdir=\"llama3-1b-instruct\")
model, kv = Transformer.from_gguf(Tensor(gguf))

# Prefill with 128 tokens then measure decode
tokens = [128000] + [9906]*127  # 128-token prompt
t0 = time.time()
for i, tok in enumerate(model.generate(tokens)):
    if i == 0:
        prefill_time = time.time() - t0
        print(f\"Prefill (128 tokens): {prefill_time*1000:.0f}ms = {128/prefill_time:.0f} tok/s\")
    if i >= 10: break
"'
```

```bash
# llama.cpp comparison (different prompt lengths)
cd ../llama-cpp-orin && nix develop -c llama-bench \
  -m ~/.cache/tinygrad/llama3-1b-instruct/Llama-3.2-1B-Instruct-Q6_K.gguf \
  -p 32,128,512 -n 128
```

### Multi-request batching (harder)

`Transformer.generate()` at `tinygrad/apps/llm.py` needs modification to support `B > 1`. The KV cache is already shaped for it but `generate()` creates input tensors with `B=1`. A minimal change:

```python
# Current (B=1):
cur_tok = Tensor([tokens[start_pos:]], ...)
# Would need (B>1):
cur_tok = Tensor([[tok_seq[start_pos:] for tok_seq in batch_tokens]], ...)
```

This is non-trivial because:
- Each sequence in the batch may have different lengths → need padding
- KV cache needs `B` dimension
- Token sampling happens per-sequence

**Suggestion:** Start with prefill benchmarking (easy), defer multi-request batching unless there's a clear need.

---

## Task 3: Benchmark Another Model (Qwen3 0.6B or 1.7B)

**Priority: MEDIUM — validates our fix is general, not LLaMA-specific.**

tinygrad supports these models that also run on llama.cpp (both use GGUF format):

| Model | tinygrad key | Size | Quant | Good candidate? |
|-------|-------------|------|-------|----------------|
| Qwen3 0.6B | `qwen3:0.6b` | ~640 MB | Q8_0 | ✅ Tiny, fast to test |
| Qwen3 1.7B | `qwen3:1.7b` | ~1 GB | Q4_K_M | ✅ Similar to LLaMA 1B |
| LLaMA 3.2 3B | `llama3.2:3b` | ~2.5 GB | Q6_K | ⚠️ Larger, slower BEAM |
| OLMoE 1B-7B | `olmoe` | ~4 GB | Q4_K_M | ⚠️ MoE architecture, interesting but different |

**Recommended:** Start with `qwen3:0.6b` (tiny, fast iteration) and `qwen3:1.7b` (closer to our LLaMA 1B benchmark).

### tinygrad benchmark

```bash
nix develop -c bash -c 'cd tinygrad && NV=1 MV_THREADS_PER_ROW=32 python3 -c "
from tinygrad.apps.llm import Transformer, models
from tinygrad import Tensor
from tinygrad.helpers import fetch
import time

for model_name in [\"qwen3:0.6b\", \"qwen3:1.7b\"]:
    url, filename = models[model_name].rsplit(\"/\", 1)
    gguf = fetch(url+\"/\"+filename, filename, subdir=model_name.replace(\":\", \"-\"))
    model, kv = Transformer.from_gguf(Tensor(gguf))
    tokens = [151644, 8948]  # Qwen3 BOS + system
    times = []
    for i, tok in enumerate(model.generate(tokens)):
        times.append(time.time())
        if i >= 15: break
    dts = [times[i]-times[i-1] for i in range(5, len(times))]
    avg = sum(dts)/len(dts)
    print(f\"{model_name}: {avg*1000:.1f}ms = {1/avg:.2f} tok/s\")
    del model, kv  # free memory
"'
```

### llama.cpp benchmark

llama.cpp can run any GGUF model. Download the same models and run llama-bench:

```bash
# Download Qwen3 0.6B Q8_0 (same file tinygrad fetches)
cd ../llama-cpp-orin
# The model URL is in tinygrad/apps/llm.py models dict
# After downloading:
nix develop -c llama-bench -m /path/to/Qwen3-0.6B-Q8_0.gguf -p 128 -n 128
```

**Note:** llama.cpp needs to support the model's architecture. Qwen3 uses the same transformer architecture so llama.cpp handles it natively.

### What to record

For each model × backend × BEAM config:
- Decode tok/s (steady-state, skip first 5 tokens)
- Prefill tok/s (if measured)
- Memory usage (from `/proc/self/status` VmRSS or `GlobalCounters`)

---

## Task 4: Default MV_THREADS_PER_ROW for NVIDIA

**Priority: LOW — quality-of-life improvement.**

Currently users must set `MV_THREADS_PER_ROW=32` via env var. The default of 8 is wrong for any GPU with warp size 32. Consider changing the default conditionally:

```python
# In heuristic.py, the current line:
MV_THREADS_PER_ROW = getenv("MV_THREADS_PER_ROW", 8)

# Could become:
# Detect if we're on an NVIDIA GPU (NV or CUDA backend) and default to warp size
_default_tpr = 32 if k.ren.device in ("NV", "CUDA") else 8
MV_THREADS_PER_ROW = getenv("MV_THREADS_PER_ROW", _default_tpr)
```

**Caution:** This changes behavior for all NVIDIA users. Run the full test suite with `MV_THREADS_PER_ROW=32` first to ensure no regressions. Some kernels might be slower with TPR=32 if they have small reduction dimensions.

This could also be submitted upstream as a PR since it's a general improvement.

---

## Task 5: On-the-fly Q6_K Dequant (THE BIG REMAINING WIN)

**Priority: RESEARCH — this is the hardest task but has the most potential.**

Currently tinygrad reads 2.5 GB of fp16 weights per token. llama.cpp reads 0.97 GB of Q6_K weights and dequantizes inline. Fusing dequant into the matmul kernel would give a theoretical 2.5× bandwidth reduction.

### Current flow

```python
# tinygrad/apps/llm.py line ~214
for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
#                                                              ^^^^^^^^^^
# This .contiguous() forces the lazy dequant graph to materialize as fp16
```

### Investigation steps

1. **Comment out `.contiguous()` and see what happens:**
   ```python
   # for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
   ```
   This will cause the dequant to re-run every token from the raw GGUF bytes. It will likely be slower because the raw data is on disk, but it tells you if the scheduler CAN fuse the dequant.

2. **Load raw quantized bytes into GPU memory, keep dequant lazy:**
   ```python
   # Load the raw bytes as a realized tensor on GPU
   # Keep the dequant ops lazy so they fuse with matmul
   ```

3. **Check the schedule with `DEBUG=3`:** Does the scheduler create one kernel (fused dequant+matmul) or two (separate dequant, then matmul)?

4. **Look at PCONTIG/RANGEIFY:** These are experimental tinygrad features for aggressive fusion. Try:
   ```bash
   NV=1 PCONTIG=2 DEBUG=3 python3 -c "..." 2>&1 | head -500
   ```

This is a multi-day research project. The code locations are:
- `tinygrad/nn/state.py` lines ~306-355: `ggml_data_to_tensor()` — the dequant graph
- `tinygrad/apps/llm.py` line ~214: the `.contiguous()` that forces materialization
- `tinygrad/schedule/indexing.py`: kernel boundary decisions

---

## Reference: Key Numbers

| Metric | Value |
|--------|-------|
| LLaMA 1B Q6_K decode (NV=1, JITBEAM=4) | 36.71 tok/s |
| LLaMA 1B Q6_K decode (NV=1, heuristic) | 29.90 tok/s |
| LLaMA 1B Q6_K decode (CUDA=1, JITBEAM=4) | 31.83 tok/s |
| llama.cpp decode (no FA) | 25.62 tok/s |
| llama.cpp prefill (pp128) | 1089 tok/s |
| Memory BW effective (LPDDR5) | ~102 GB/s |
| Matvec kernel BW (optimized) | 43-52 GB/s |
| Kernels per decode token | 230 (heuristic), 254 (JIT captured) |
| Weight data read per token (fp16) | ~2.5 GB |
| Weight data read per token (Q6_K) | ~0.97 GB |

## Reference: Key Files

| File | What |
|------|------|
| `tinygrad/codegen/opt/heuristic.py` | **OUR FIX** — matvec pattern match |
| `tinygrad/runtime/ops_nv.py` | **OUR FIX** — TegraIface VA + alloc |
| `tinygrad/renderer/ptx.py` | **OUR FIX** — scalar load decomposition |
| `tinygrad/apps/llm.py` | LLM app — models dict, Transformer, generate() |
| `tinygrad/nn/state.py` | GGUF loading, Q6_K dequant |
| `tinygrad/codegen/opt/postrange.py` | BEAM search entry point |
| `tinygrad/engine/jit.py` | TinyJit, JITBEAM |
| `beat-llama-cpp-results.md` | All benchmark results |
| `nv-optimization-plan.md` | Full optimization roadmap |
| `Learning-Beat-Llama-Cpp.md` | Teaching doc — the full story |

## Reference: How to Build / Enter Shell

```bash
# Enter tinygrad dev shell
cd /home/agent/jetpack-nixos/examples/tinygrad
nix develop

# Inside the shell:
cd tinygrad
NV=1 python3 -m pytest test/test_ops.py -x -v --tb=short   # run tests
NV=1 MV_THREADS_PER_ROW=32 python3 -c "..."                 # benchmark
NV=1 MV_THREADS_PER_ROW=32 JITBEAM=4 python3 -c "..."       # with BEAM

# Build llama.cpp
cd /home/agent/jetpack-nixos/examples/llama-cpp-orin
nix develop -c llama-bench -m <model.gguf> -p 128 -n 128

# NixOS system build (if needed)
sudo nixos-rebuild switch --flake /home/agent/jetpack-nixos/examples/nixos#nixos-telemetry --show-trace
```

## Reference: AGENTS.md

The workspace root has `AGENTS.md` with the full layout, build commands, and folder descriptions. Read it first.
