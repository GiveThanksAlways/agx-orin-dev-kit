# Production Hot Path Benchmark Results

**Platform:** Jetson AGX Orin 64GB, JetPack 6, NixOS, CUDA 12.6  
**Tinygrad:** NV=1 backend, JITBEAM=2  
**Iterations:** 10,000 per test

## Summary

| MLP Config | Params | Python NV=1 | C GPU Hot Path | C NEON FP16 | GPU Speedup | GPU vs NEON |
|-----------|--------|-------------|----------------|-------------|-------------|-------------|
| 12→64→64→4 | 5,252 | 105.3 µs / 9,493 Hz | **45.7 µs / 21,869 Hz** | 1.1 µs / 919,118 Hz | **2.3x** | NEON 42x faster |
| 12→128→128→4 | 18,692 | 106.6 µs / 9,379 Hz | **46.1 µs / 21,702 Hz** | 2.9 µs / 347,222 Hz | **2.3x** | NEON 16x faster |
| 12→256→256→256→4 | 135,940 | 125.4 µs / 7,976 Hz | **64.3 µs / 15,548 Hz** | 16.3 µs / 61,520 Hz | **1.9x** | NEON 4x faster |

**All correctness checks pass** — C GPU output matches Python NV=1 within tolerance.

## Key Insights

### C GPU Hot Path (45–64 µs)
- Eliminates ~60 µs of Python overhead per iteration
- 4 patches per iteration (1 kickoff + 1 tl_wait + 1 tl_signal + 1 const)
- Single GPFifo ring write + MMIO doorbell poke replays entire HCQ graph
- Zero syscalls, zero ioctls, zero CUDA — pure memory-mapped I/O

### NEON FP16 MLP (1–16 µs)
- For small/medium MLPs, ARM NEON dominates the GPU
- GPU launch overhead (~45 µs minimum) makes CPU faster for < ~200K parameter MLPs
- 4×8 unrolled FMLA loop with float16x8_t achieves near peak NEON throughput
- ReLU integrated into the forward pass (branchless via fused comparison)

### Crossover Analysis
- GPU advantage grows with model size (more compute to amortize launch latency)
- At 135K params: GPU is still 4x slower than NEON due to launch overhead
- Estimated crossover: ~500K–1M parameters (where GPU compute >> launch cost)
- For robotic control loops (typically < 50K params): **NEON is the clear winner**

## Architecture

```
Python NV=1 (105 µs):
  TinyJit.__call__   →  HCQGraph.__call__  →  _apply_var_vals  →  _submit_to_gpfifo  →  GPU
  ^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^       ^^^^^^^^^^^^^^^^^^
  Python dispatch         Python graph           Python sym eval     Ring write + doorbell
  (~60 µs overhead)       (~10 µs)               (~15 µs)            (~5 µs)

C GPU Hot Path (45 µs):
  memcpy_in  →  apply_patches  →  submit_gpfifo  →  kick  →  GPU  →  memcpy_out
  (<1 µs)       (<1 µs)           (<1 µs)           (<1 µs)   (~40 µs)   (<1 µs)

C NEON MLP (1–16 µs):
  memcpy_in  →  NEON FP16 forward pass  →  memcpy_out
  (<1 µs)       (1–16 µs)                   (<1 µs)
```

## Files

| File | Purpose |
|------|---------|
| `hot_path.h` | C header: config struct, patch types, API |
| `hot_path.c` | C GPU dispatch: GPFifo submit, signal wait, patch apply |
| `neon_mlp.h` | NEON MLP header: struct, init/forward/benchmark API |
| `neon_mlp.c` | ARM NEON FP16 MLP: 4×8 FMLA unrolled forward pass |
| `export_graph.py` | Extract HCQGraph internals → C config struct |
| `bench_hot_path.py` | Benchmark driver: 3 sizes × 3 approaches |
| `Makefile` | Build both .so files with clang |

## Build & Run

```bash
cd examples/tinygrad && nix develop
cd ../../examples/control-loop/hot_path

# Build
CC=clang make -j2

# Run benchmark
NV=1 JITBEAM=2 python3 bench_hot_path.py
```
