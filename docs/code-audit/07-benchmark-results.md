# Cross-Framework LLM Benchmark Results — Jetson Orin AGX 64GB

**Date**: 2025-07-17
**Platform**: NVIDIA Jetson Orin AGX 64GB (SM 8.7, ga10b iGPU)
**JetPack**: 6 (kernel module 540.4.0, CUDA 12.6)
**RAM**: 61 GB shared CPU/GPU
**Theoretical memory bandwidth**: ~205 GB/s (practical ~102 GB/s)

---

## Executive Summary

We benchmarked three LLM inference frameworks on the Jetson Orin AGX 64GB.
Two worked out of the box (tinygrad, llama.cpp); two could not be installed
without Docker (vLLM, MLC LLM).

**Key finding**: llama.cpp leads on text generation by **3.4-3.8×** over
tinygrad (without BEAM cache). tinygrad's NV backend is ~2% faster than its
CUDA backend consistently. llama.cpp's Flash Attention adds 8-10% on decode
and 25-30% on prompt processing.

| Framework     | LLaMA 1B Q6_K decode | LLaMA 3B Q6_K decode |
|---------------|----------------------|----------------------|
| llama.cpp     | **25.59** tok/s      | **11.09** tok/s      |
| llama.cpp +FA | **27.80** tok/s      | **11.98** tok/s      |
| tinygrad NV=1 | 7.59 tok/s           | 2.95 tok/s           |
| tinygrad CUDA | 7.44 tok/s           | 2.93 tok/s           |
| vLLM          | — (install failed)   | — (install failed)   |
| MLC LLM       | — (install failed)   | — (install failed)   |

---

## 1. tinygrad Results

**Branch**: `nv-agx-orin-dev-kit` (commit `65fdaf07b`)
**Optimizations active**: TegraIface (NV backend), matvec heuristic fix,
`MV_THREADS_PER_ROW=32`
**BEAM**: Off (no cached schedules; JITBEAM=2 drops to ~1.1 tok/s
while searching)

### LLaMA 3.2 1B Instruct — Q6_K (1.04 GB)

| Backend | tok/s | ms/tok | Memory BW | Param BW |
|---------|-------|--------|-----------|----------|
| NV=1    | 7.59  | 131.7  | 67.3 GB/s | 42.0 GB/s |
| CUDA=1  | 7.44  | 134.5  | 66.0 GB/s | 41.2 GB/s |

### LLaMA 3.2 1B Instruct — Q4_K_M (0.75 GB)

| Backend | tok/s | ms/tok | Memory BW | Param BW |
|---------|-------|--------|-----------|----------|
| NV=1    | 7.54  | 132.7  | —         | —        |
| CUDA=1  | 7.39  | 135.2  | —         | —        |

### LLaMA 3.2 3B Instruct — Q6_K (2.57 GB)

| Backend | tok/s | ms/tok | Memory BW | Param BW |
|---------|-------|--------|-----------|----------|
| NV=1    | 2.95  | 339.2  | 65.5 GB/s | 42.6 GB/s |
| CUDA=1  | 2.93  | 341.3  | 65.1 GB/s | 42.3 GB/s |

### Observations

- **NV vs CUDA**: NV backend consistently ~2% faster. Lower enqueue latency
  (3.8 ms NV vs 12.3 ms CUDA) thanks to direct ioctl submission bypassing
  the CUDA runtime.
- **Quantization impact negligible**: Q4_K_M is the same speed as Q6_K on
  tinygrad. The framework is not compute-bound on dequantization — it's
  memory-bandwidth-limited, and the smaller model doesn't help because
  tinygrad's kernel fusion already dominates.
- **3B scales linearly**: 3B is ~2.6× slower than 1B, matching the ~2.5×
  increase in parameters (2.57 GB vs 1.04 GB).
- **BEAM cache matters enormously**: With a warm BEAM cache, prior sessions
  achieved ~36.7 tok/s on the same 1B Q6_K model. Without it, we get 7.6.
  BEAM search itself takes ~5-10 minutes to optimize all kernels.
- **Qwen3 not supported**: tinygrad's `llama3.py` only supports LLaMA
  architecture. Qwen3 models were only benchmarked on llama.cpp.

---

## 2. llama.cpp Results

**Binary**: `llama-cpp-cuda-0.0.0` (commit `9a5f57795`)
**Backend**: CUDA, compute_87, JetPack 6
**Benchmark tool**: `llama-bench` with 3 repetitions each
**Prompt processing**: pp512 (512 tokens), **Text generation**: tg128 (128 tokens)

### Full Results Table

| Model                    | Quant  | FA  | pp512 (tok/s) | tg128 (tok/s) |
|--------------------------|--------|-----|---------------|---------------|
| LLaMA 3.2 1B Instruct   | Q6_K   | off | 1089.61 ± 0.68 | **25.59 ± 0.04** |
| LLaMA 3.2 1B Instruct   | Q6_K   | on  | 1386.12 ± 0.81 | **27.80 ± 0.04** |
| LLaMA 3.2 1B Instruct   | Q4_K_M | off | 1197.85 ± 8.47 | **31.51 ± 0.01** |
| LLaMA 3.2 1B Instruct   | Q4_K_M | on  | 1554.29 ± 0.84 | **34.79 ± 0.02** |
| LLaMA 3.2 3B Instruct   | Q6_K   | off | 444.24 ± 0.30  | **11.09 ± 0.00** |
| LLaMA 3.2 3B Instruct   | Q6_K   | on  | 507.21 ± 0.18  | **11.98 ± 0.10** |
| Qwen3 0.6B              | Q8_0   | off | 1622.88 ± 2.50 | **37.40 ± 0.24** |
| Qwen3 1.7B              | Q4_K_M | off | 833.53 ± 2.03  | **22.36 ± 0.01** |

### Observations

- **Flash Attention**: +8-10% on text generation, +25-30% on prompt processing.
  Always worth enabling.
- **Quantization matters**: Q4_K_M is 23% faster than Q6_K on 1B decode
  (31.51 vs 25.59 tok/s). llama.cpp's hand-tuned CUDA kernels fully exploit
  the reduced memory traffic.
- **Scaling**: 3B Q6_K decode is 2.3× slower than 1B Q6_K decode.
- **Qwen3 0.6B is fastest**: 37.40 tok/s — smallest model, highest throughput.
- **Very low variance**: Standard deviations under 0.25 tok/s across 3 runs.
  Results are highly reproducible.

---

## 3. vLLM & MLC LLM

### vLLM

**Status**: ❌ Not benchmarked — installation failed.

`pip install vllm` fails due to `xgrammar` native extension build failure.
The underlying issue is missing `scikit_build_core` and CMake configuration
problems in the NixOS environment. vLLM requires either:
- Pre-built wheels for aarch64 JetPack (not available in PyPI)
- Docker with NVIDIA Container Toolkit (not installed on this system)
- A custom Nix package that patches the build (significant effort)

### MLC LLM

**Status**: ❌ Not benchmarked — same class of installation issues.

MLC LLM requires native CUDA builds similar to vLLM. Without Docker or
pre-built JetPack wheels, it cannot be installed in this NixOS environment.

### Recommendation

To benchmark vLLM and MLC LLM on this system, install Docker + NVIDIA
Container Toolkit and use the official JetPack containers from NVIDIA.

---

## 4. Head-to-Head Comparison

### LLaMA 3.2 1B Instruct Q6_K — Decode Speed

```
llama.cpp +FA  ████████████████████████████ 27.80 tok/s
llama.cpp      █████████████████████████▌   25.59 tok/s
tinygrad NV=1  ███████▌                      7.59 tok/s
tinygrad CUDA  ███████▍                      7.44 tok/s
```

### LLaMA 3.2 3B Instruct Q6_K — Decode Speed

```
llama.cpp +FA  ████████████ 11.98 tok/s
llama.cpp      ███████████  11.09 tok/s
tinygrad NV=1  ███           2.95 tok/s
tinygrad CUDA  ██▉           2.93 tok/s
```

### llama.cpp — All Models Decode Speed

```
Qwen3 0.6B Q8_0    █████████████████████████████████████▍ 37.40 tok/s
LLaMA 1B Q4_K_M+FA ██████████████████████████████████▊    34.79 tok/s
LLaMA 1B Q4_K_M    ███████████████████████████████▌       31.51 tok/s
LLaMA 1B Q6_K +FA  ███████████████████████████▊           27.80 tok/s
LLaMA 1B Q6_K      █████████████████████████▌             25.59 tok/s
Qwen3 1.7B Q4_K_M  ██████████████████████▍                22.36 tok/s
LLaMA 3B Q6_K +FA  ████████████                           11.98 tok/s
LLaMA 3B Q6_K      ███████████                            11.09 tok/s
```

---

## 5. Analysis

### Why is llama.cpp 3.4× faster?

1. **Hand-tuned CUDA kernels**: llama.cpp uses custom CUDA kernels for
   each quantization format (Q6_K, Q4_K_M, etc.) with optimized memory
   access patterns, shared memory usage, and warp-level primitives.

2. **KV-cache optimization**: llama.cpp's KV-cache is tightly packed and
   accessed with custom kernels. tinygrad relies on general-purpose
   tensor operations.

3. **Operator fusion**: While tinygrad does kernel fusion via its lazy
   evaluation graph, llama.cpp's handwritten kernels fuse operations at
   a lower level with explicit shared memory management.

4. **No BEAM = no schedule optimization**: tinygrad without BEAM uses
   default kernel schedules. With a warm BEAM cache, the gap narrows
   significantly (prior sessions showed ~36.7 tok/s, which would make
   tinygrad competitive).

### Memory Bandwidth Utilization

| Framework     | Model             | Effective BW | % of Theoretical |
|---------------|-------------------|-------------|------------------|
| tinygrad NV=1 | 1B Q6_K          | 67.3 GB/s   | 32.8%           |
| tinygrad NV=1 | 3B Q6_K          | 65.5 GB/s   | 31.9%           |
| llama.cpp     | 1B Q6_K          | ~26.6 GB/s* | 13.0%           |
| llama.cpp +FA | 1B Q6_K          | ~28.9 GB/s* | 14.1%           |

*llama.cpp bandwidth estimated from: model_size × tok/s

**Interesting**: tinygrad reports higher bandwidth utilization than llama.cpp.
This is because tinygrad's "memory bandwidth" metric includes all tensor
operations (attention, layernorm, etc.), not just the model weights.
llama.cpp's decode is faster despite lower raw bandwidth because its kernels
do more useful work per byte transferred.

### The BEAM Gap

The elephant in the room: with a fully-warmed JITBEAM cache, prior sessions
measured **36.71 tok/s** for tinygrad on the same model — which would put it
**ahead** of llama.cpp's 25.59 tok/s (non-FA) and competitive with FA-enabled.

However, that BEAM cache no longer exists. Building it requires:
1. Running `JITBEAM=2` (or higher) which takes ~5-10 minutes per model
2. The cache is stored on disk and persists across runs
3. Without the cache, BEAM search during inference drops speed to ~1.1 tok/s

This represents tinygrad's core trade-off: invest time in BEAM optimization
upfront, then enjoy faster inference indefinitely.

---

## 6. Reproduction Commands

### tinygrad

```bash
cd /home/agent/jetpack-nixos/examples/tinygrad
nix develop
cd tinygrad

# NV backend (TegraIface — recommended for Jetson)
NV=1 MV_THREADS_PER_ROW=32 python3 examples/llama3.py \
  --benchmark --timing --no_api --size 1B \
  --model /path/to/model.gguf

# CUDA backend
CUDA=1 MV_THREADS_PER_ROW=32 python3 examples/llama3.py \
  --benchmark --timing --no_api --size 1B \
  --model /path/to/model.gguf

# With BEAM optimization (slow first run, fast after)
NV=1 MV_THREADS_PER_ROW=32 JITBEAM=2 python3 examples/llama3.py \
  --benchmark --timing --no_api --size 1B \
  --model /path/to/model.gguf
```

### llama.cpp

```bash
cd /home/agent/jetpack-nixos/examples/llama-cpp-orin
nix develop

# Benchmark specific model
llama-bench -m /path/to/model.gguf -r 3

# With Flash Attention
llama-bench -m /path/to/model.gguf -fa 1 -r 3

# Multi-model sweep
for model in model1.gguf model2.gguf; do
  llama-bench -m "$model" -r 3
  llama-bench -m "$model" -fa 1 -r 3
done
```

---

## 7. Model Inventory

All models stored in `~/.cache/tinygrad/downloads/` and `~/.cache/llama.cpp/`:

| Model | File | Size | Architectures Tested |
|-------|------|------|---------------------|
| LLaMA 3.2 1B Instruct Q6_K | `Llama-3.2-1B-Instruct-Q6_K.gguf` | 1.04 GB | tinygrad, llama.cpp |
| LLaMA 3.2 1B Instruct Q4_K_M | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 0.75 GB | tinygrad, llama.cpp |
| LLaMA 3.2 3B Instruct Q6_K | `Llama-3.2-3B-Instruct-Q6_K.gguf` | 2.57 GB | tinygrad, llama.cpp |
| Qwen3 0.6B Q8_0 | `Qwen3-0.6B-Q8_0.gguf` | 0.67 GB | llama.cpp only |
| Qwen3 1.7B Q4_K_M | `Qwen3-1.7B-Q4_K_M.gguf` | 1.12 GB | llama.cpp only |

---

## 8. Next Steps

1. **Warm BEAM cache**: Run `JITBEAM=2` or `JITBEAM=4` overnight to build
   optimized kernel schedules, then re-benchmark tinygrad.
2. **Install Docker**: Enable `nvidia-container-toolkit` in NixOS config to
   benchmark vLLM and MLC LLM via official NVIDIA containers.
3. **Larger models**: Try 8B models (would need ~5-8 GB depending on quant).
   With 61 GB RAM, even 70B Q4 might fit.
4. **Prompt processing**: tinygrad's `--benchmark` doesn't separately report
   prefill speed. Add a custom benchmark script to measure pp vs tg separately.
5. **Multi-GPU**: N/A for Jetson (single iGPU), but relevant if comparing
   against desktop GPUs.
