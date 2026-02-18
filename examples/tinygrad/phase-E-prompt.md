# Phase E: Big-Picture LLM Benchmarks — Agent Prompt

## Your Task

You are running benchmarks on an **NVIDIA Jetson Orin AGX 64GB** to compare real-world LLM inference speed across three backends:
1. **tinygrad NV=1** — custom Tegra/HCQ backend (direct nvgpu ioctls, no CUDA runtime)
2. **tinygrad CUDA=1** — standard CUDA backend (cuLaunchKernel, CUDA driver API)
3. **llama.cpp** — C++ inference engine with CUDA support

We already proved NV=1 beats CUDA=1 on micro-benchmarks (matmul 26-50% faster fp16, 32% faster fp32 at medium sizes, 2.2× better kernel launch p99). **Now we need to know: does NV=1 actually generate tokens faster for real models?**

## System Info

- **Device:** Jetson Orin AGX 64GB, JetPack 6, L4T r36.4.4
- **GPU:** ga10b iGPU, Ampere SM 8.7, unified memory (CPU+GPU share RAM)
- **CUDA:** 12.6
- **tinygrad:** v0.12.0 at `/home/agent/jetpack-nixos/examples/tinygrad/tinygrad/`
- **llama.cpp:** Available via nix flake at `/home/agent/jetpack-nixos/examples/llama-cpp-orin-nix-overlay/`
- **Working doc:** `/home/agent/jetpack-nixos/examples/tinygrad/robust-testing-and-performance.md` (Phase E section at bottom of Phase C)

## Step-by-Step Instructions

### Step 0: Enter the tinygrad dev shell

```bash
cd /home/agent/jetpack-nixos/examples/tinygrad && nix develop
cd tinygrad
```

All tinygrad commands must be run from inside this dev shell. It provides Python 3.13, all CUDA libs, and the correct `CUDA_INCLUDE_PATH`.

### Step 1: GPT-2 124M — NV=1 vs CUDA=1 (Simplest test)

GPT-2 is small (124M params, ~500MB) and downloads automatically. This is the quick sanity check.

```bash
# NV=1 — generate 50 tokens with timing
NV=1 python3 examples/gpt2.py --model_size gpt2 --count 50 --temperature 0 --timing 2>&1 | tee ../tests/results_gpt2_nv.log

# CUDA=1 — same
CUDA=1 python3 examples/gpt2.py --model_size gpt2 --count 50 --temperature 0 --timing 2>&1 | tee ../tests/results_gpt2_cuda.log
```

**What to record:**
- Per-token timing from `--timing` output (each token prints elapsed time)
- Total wall-clock time for all 50 tokens
- First token time vs subsequent token times (TTFT vs decode)
- The `--timing` flag prints "ran model in X.XX ms" per token — capture these

**Important:** Run each backend **twice** and take the second run (first run downloads + compiles PTX, second is warm cache).

### Step 2: LLaMA 3.2 1B — NV=1 vs CUDA=1

This is the real test. LLaMA 3.2 1B with Q6_K quantization is ~1.1GB and fits easily in memory.

```bash
# Download model first (only needed once)
NV=1 python3 examples/llama3.py --size 1B --no_api --benchmark 2>&1 | tee ../tests/results_llama3_nv.log

# CUDA=1
CUDA=1 python3 examples/llama3.py --size 1B --no_api --benchmark 2>&1 | tee ../tests/results_llama3_cuda.log
```

The `--benchmark` flag runs a standardized generation benchmark. If it doesn't output tok/s directly, use `--no_api --timing` instead and time manually:

```bash
NV=1 python3 examples/llama3.py --size 1B --no_api --timing 2>&1 | tee ../tests/results_llama3_nv.log
# Then type a prompt like "Explain quantum computing in simple terms" and let it generate ~50 tokens
```

**What to record:**
- Tokens per second (decode speed)
- Time to first token
- Total generation time

### Step 3: llama.cpp — LLaMA 3.2 1B (Three-Way Comparison)

Switch to the llama.cpp dev shell for this:

```bash
cd /home/agent/jetpack-nixos/examples/llama-cpp-orin-nix-overlay && nix develop

# llama.cpp with the same model (auto-downloads from HF)
llama-cli \
  -hf bartowski/Llama-3.2-1B-Instruct-GGUF:Q6_K \
  -ngl 999 \
  --prompt "What is the answer to life, the universe, and everything?" \
  -n 50 --temp 0 \
  2>&1 | tee ../tinygrad/tests/results_llama3_llamacpp.log
```

If `llama-cli` is not available, try `llama-bench` or just build directly:
```bash
# Alternative: use llama-bench for standardized output
llama-bench \
  -hf bartowski/Llama-3.2-1B-Instruct-GGUF:Q6_K \
  -ngl 999 \
  -t 12 \
  2>&1 | tee ../tinygrad/tests/results_llama3_llamacpp_bench.log
```

llama.cpp prints stats at the end like:
```
llama_perf_context_print:        load time =   XXX.XX ms
llama_perf_context_print: prompt eval time =   XXX.XX ms / N tokens (XX.XX ms per token, XX.XX tokens per second)
llama_perf_context_print:        eval time =   XXX.XX ms / N tokens (XX.XX ms per token, XX.XX tokens per second)
```

**What to record:**
- `prompt eval` tokens/sec (= prefill speed)
- `eval` tokens/sec (= decode speed)
- Load time

### Step 4: Record Results

Fill in the Phase E results table in `robust-testing-and-performance.md`:

| Backend | Model | Prefill tok/s | Decode tok/s | TTFT (ms) | Notes |
|---------|-------|---------------|--------------|-----------|-------|
| NV=1 | GPT-2 124M | ??? | ??? | ??? | |
| CUDA=1 | GPT-2 124M | ??? | ??? | ??? | |
| NV=1 | LLaMA 3.2 1B Q6_K | ??? | ??? | ??? | |
| CUDA=1 | LLaMA 3.2 1B Q6_K | ??? | ??? | ??? | |
| llama.cpp | LLaMA 3.2 1B Q6_K | ??? | ??? | ??? | |

### Step 5: Analysis

Write a brief analysis section answering:
1. **Does NV=1 beat CUDA=1 on real models?** (Our micro-benchmarks say it should, especially for fp16.)
2. **How does tinygrad compare to llama.cpp?** (llama.cpp is heavily optimized C++ — expect it to be faster, but by how much?)
3. **Where is the bottleneck?** If NV=1 doesn't show the expected speedup, is it the model code, the JIT, quantization overhead, or memory bandwidth?

### Step 6: Commit

```bash
cd /home/agent/jetpack-nixos
git add examples/tinygrad/robust-testing-and-performance.md examples/tinygrad/tests/results_*.log
git commit -m "Phase E: big-picture LLM benchmarks (GPT-2, LLaMA 3.2 1B, llama.cpp)"
```

## Troubleshooting

- **Model download fails:** Models download to `~/.cache/tinygrad/`. If disk is full, clear old caches.
- **OOM:** LLaMA 1B Q6_K is ~1.1GB — should fit easily in 64GB. If OOM, try `MAX_CONTEXT=512`.
- **llama.cpp not found:** Make sure you're in the llama-cpp-orin-nix-overlay dev shell, not the tinygrad shell.
- **Segfault on long generation:** Known kernargs bump allocator issue. Keep generation to ≤100 tokens.
- **CUDA_INCLUDE_PATH:** Already set in the tinygrad flake. If fp16 errors occur, verify `echo $CUDA_INCLUDE_PATH` shows a path containing `cuda_fp16.h`.
- **tinygrad's llama3.py download:** The first run of `llama3.py --size 1B` auto-downloads the GGUF from HuggingFace. Let it finish before timing.

## What Success Looks Like

- GPT-2: NV=1 should be **similar or faster** than CUDA=1 (based on our micro-benchmarks)
- LLaMA 1B: NV=1 should show **measurable advantage** over CUDA=1, especially during decode
- llama.cpp will likely be **faster than both** tinygrad backends (it's a mature, hand-optimized C++ engine) — but the gap tells us how much optimization headroom remains in tinygrad's Tegra path
- **The big question:** Is the NV=1 micro-benchmark advantage (26-50% fp16, 32% fp32) visible at the model level, or does it get lost in other overhead?
