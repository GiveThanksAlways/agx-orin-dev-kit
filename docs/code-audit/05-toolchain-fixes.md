# Toolchain Fixes: BEAM Search, PTX, CUDA Paths

Three small but essential fixes that make tinygrad's toolchain work on NixOS +
Jetson. Without these, you can't compile kernels or use BEAM optimization.

## Table of Contents

1. [BEAM Search: Fork vs Spawn](#beam-search-fork-vs-spawn)
2. [PTX: Vector Load Formatting](#ptx-vector-load-formatting)
3. [CUDA Include Path for NixOS](#cuda-include-path-for-nixos)

---

## BEAM Search: Fork vs Spawn

**File**: `tinygrad/codegen/opt/search.py` (+3 / -1 lines)

### What is BEAM Search?

BEAM search is tinygrad's kernel optimization strategy. For each kernel, it
tries multiple optimization schedules (different combinations of LOCAL, UPCAST,
GROUP, UNROLL) and benchmarks each one on real hardware, keeping the best.

```
JITBEAM=2 means: try the top-2 options at each step of the optimization

Step 1: Try GROUP=8, GROUP=16        → keep best 2
Step 2: Try LOCAL=4, LOCAL=8 (×2)    → keep best 2
Step 3: Try UPCAST=2, UPCAST=4 (×2)  → keep best 2
...
Final: Use the single best schedule, cache it to disk
```

This runs in a **multiprocessing pool** — each candidate schedule is benchmarked
in a separate worker process to avoid GPU state leaks.

### The Problem: Spawn on NixOS

Python's multiprocessing has two modes:
- **spawn**: Creates a fresh Python process by re-importing the main module
- **fork**: Creates a child by forking the current process (copy-on-write)

The old code used `spawn`:

```python
beam_pool = multiprocessing.get_context("spawn").Pool(workers, ...)
```

On NixOS inside `nix develop`, the Python process was started from stdin (piped
nix expression), not a regular `.py` file. When `spawn` tries to re-import the
main module, it looks for `__main__.__spec__`, which points to `<stdin>` — not
a file that exists on disk:

```
FileNotFoundError: [Errno 2] No such file or directory: '<stdin>'
```

### The Fix

```python
import sys
_mp_ctx = "fork" if sys.platform == "linux" else "spawn"
beam_pool = multiprocessing.get_context(_mp_ctx).Pool(workers, ...)
```

On Linux, use `fork`. No re-import needed — the child process inherits the
parent's memory (including loaded modules, GPU state, compiled kernels).

On macOS/Windows, keep `spawn` (macOS deprecated `fork` in multiprocessing,
Windows doesn't support it).

### Why Fork is Safe Here

`fork` in Python has a bad reputation because it can deadlock if the parent
held a lock at fork time. But tinygrad's BEAM workers:
1. Don't hold locks at fork point (pool is created at first BEAM invocation)
2. Only call compute functions (no file I/O in the hot path)
3. Communicate through the pool protocol (pickle over pipes)

The GPU driver (`nvgpu.ko`) is fork-safe on Linux because the child inherits
the parent's file descriptors. GPU memory allocations and channels are
per-process state that transfers correctly.

---

## PTX: Vector Load Formatting

**File**: `tinygrad/renderer/ptx.py` (+2 / -1 lines)

### What Changed

```python
# OLD: one long line
(UPat(Ops.LOAD, ...),
    lambda ctx, x, loc, buf: f"ld.{mem_type(buf)}.v{x.dtype.count}..." \
        if x.dtype.count > 1 else f"ld.{mem_type(buf)}.{ctx.mem_types[x.dtype]}...")

# NEW: line break after lambda header
(UPat(Ops.LOAD, ...),
    lambda ctx, x, loc, buf:
    f"ld.{mem_type(buf)}.v{x.dtype.count}..." \
        if x.dtype.count > 1 else f"ld.{mem_type(buf)}.{ctx.mem_types[x.dtype]}...")
```

This is a **formatting-only change** — no functional difference. The line was
too long, making the ternary condition hard to read when debugging PTX generation.

### What's PTX?

**PTX (Parallel Thread Execution)** is NVIDIA's intermediate assembly language.
Tinygrad generates PTX from its IR, then NVRTC compiles PTX to native CUBIN
(binary GPU code).

This particular pattern matches vector loads — loading multiple values at once
from global memory:

```ptx
// Scalar load (count=1):
ld.global.f32 %f0, [%r0+0];

// Vector load (count=4):
ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%r0+0];
```

Vector loads are important for memory bandwidth — reading 4 floats in one
instruction uses the memory bus more efficiently than 4 separate loads.

---

## CUDA Include Path for NixOS

**File**: `tinygrad/runtime/support/compiler_cuda.py` (+2 lines)

### The Problem

NVRTC (NVIDIA Runtime Compilation) needs CUDA header files (like
`cuda_fp16.h`) to compile kernels using half-precision types. It searches
standard paths:

```python
self.compile_options += [
    "-I/usr/local/cuda/include",
    "-I/usr/include",
    "-I/opt/cuda/include"
]
```

On NixOS, CUDA lives in the Nix store at a non-standard path like:

```
/nix/store/abc123-cudatoolkit-12.6/include/
```

None of the standard search paths find it.

### The Fix

```python
CUDA_INCLUDE_PATH = getenv("CUDA_INCLUDE_PATH", "")

# In NVRTCCompiler.__init__:
self.compile_options += [f"-I{CUDA_PATH}/include"] if CUDA_PATH else [...]
if CUDA_INCLUDE_PATH:
    self.compile_options += [f"-I{CUDA_INCLUDE_PATH}"]
```

A new environment variable `CUDA_INCLUDE_PATH` lets the user (or the Nix
flake) point NVRTC at the right directory.

### How Our Flake Uses It

In the tinygrad dev shell flake:

```nix
shellHook = ''
  export CUDA_INCLUDE_PATH="${cudaPackages.cuda_cudart}/include"
'';
```

This resolves to something like:

```bash
CUDA_INCLUDE_PATH=/nix/store/abc123-cuda_cudart-12.6.77/include
```

And NVRTC can now find `cuda_fp16.h`, `cuda_bf16.h`, etc.

### Why Not Just Use CUDA_PATH?

`CUDA_PATH` expects a full CUDA toolkit directory with `/include`, `/lib64`,
etc. On NixOS, the toolkit is split across multiple store paths (one per
component: cudart, nvcc, nvrtc, etc.). `CUDA_INCLUDE_PATH` lets you point
directly to the include directory without requiring a monolithic install.
