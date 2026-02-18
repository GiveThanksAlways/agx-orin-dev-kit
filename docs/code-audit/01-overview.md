# Code Audit: NV Backend Optimization Campaign

## Overview

This document series walks through every change made to bring **tinygrad's NV
backend** to the **Jetson Orin AGX 64GB**, achieving a **7.6× speedup** over the
unoptimized baseline and **143% of llama.cpp's performance** on LLaMA 1B Q6_K
inference (36.71 tok/s vs 25.7 tok/s).

## What Changed

The work spans **10 commits** in the tinygrad submodule and **~20 commits** in
the parent jetpack-nixos repo. The core changes are in 5 tinygrad source files:

| File | Lines Changed | What It Does |
|---|---|---|
| `ops_nv.py` | +766 | **TegraIface**: Full nvgpu/nvmap backend for Jetson |
| `heuristic.py` | +33 / -22 | **Matvec fix**: Enable matvec for fp16 CAST/MUL chains |
| `search.py` | +3 / -1 | **BEAM fix**: Use fork context on Linux |
| `ptx.py` | +2 / -1 | **PTX fix**: Decompose vector loads for alignment |
| `compiler_cuda.py` | +2 | **NixOS fix**: CUDA include path for nix-develop |

## Reading Order

1. **[01-overview.md](01-overview.md)** — This file. Architecture context + reading guide.
2. **[02-tegra-iface.md](02-tegra-iface.md)** — The big one. TegraIface implementation (nvgpu/nvmap).
3. **[03-heuristic-matvec.md](03-heuristic-matvec.md)** — The 7.6× speedup. Matvec pattern matching.
4. **[04-runtime-fixes.md](04-runtime-fixes.md)** — QMD race, VA collisions, copyout optimization.
5. **[05-toolchain-fixes.md](05-toolchain-fixes.md)** — BEAM search, PTX, NixOS compiler paths.
6. **[06-parent-repo.md](06-parent-repo.md)** — NixOS configs, benchmarks, frameworks, projects.

## Background: How GPUs Work on Jetson

### The Hardware

The Jetson Orin AGX 64GB has an **integrated GPU** (iGPU) — the `ga10b`. Unlike
discrete GPUs (dGPUs) that have their own VRAM connected via PCIe, the iGPU
shares the SoC's **64GB LPDDR5** memory with the CPU. This is called **unified
memory**.

```
┌─────────────────────────────────────────────┐
│                Orin SoC                      │
│  ┌──────────────┐   ┌──────────────┐        │
│  │  ARM Cortex  │   │   ga10b GPU  │        │
│  │  A78AE × 12  │   │  SM 8.7      │        │
│  │  (CPU)       │   │  2048 cores  │        │
│  └──────┬───────┘   └──────┬───────┘        │
│         │                  │                 │
│         └────────┬─────────┘                 │
│                  │                           │
│          ┌───────▼───────┐                   │
│          │  SMMU (IOMMU) │ ← IO-coherent     │
│          └───────┬───────┘                   │
│                  │                           │
│          ┌───────▼───────┐                   │
│          │   64GB LPDDR5  │ ← ~204 GB/s peak │
│          │   (unified)    │   ~102 GB/s eff.  │
│          └───────────────┘                   │
└─────────────────────────────────────────────┘
```

Key implications:
- **No PCIe transfer**: CPU and GPU see the same physical memory
- **IO-coherent SMMU**: After GPU writes, CPU can read directly (no flush needed)
- **Memory-bandwidth-bound**: LLM decode reads all model weights per token
- **40-bit VA space**: GPU virtual addresses are 40 bits (1 TB), not 48 bits

### The Software Stack

NVIDIA's Jetson uses a different kernel driver than desktop GPUs:

| Desktop (dGPU) | Jetson (iGPU) |
|---|---|
| `nvidia.ko` (proprietary) | `nvgpu.ko` (open-source, in-tree) |
| NVIDIA RM (Resource Manager) | nvgpu ioctl interface |
| PCIe BAR access | MMIO via `/dev/nvgpu/igpu0/ctrl` |
| Separate VRAM | Unified RAM via `/dev/nvmap` |
| UVM (unified virtual memory) | Native unified memory |

Tinygrad's existing NV backend (`NVKIface` and `PCIIface`) talks to the desktop
RM driver. **TegraIface** is our new backend that talks to `nvgpu.ko` instead.

### The HCQ Execution Model

Tinygrad uses **HCQ (Hardware Command Queue)** for GPU execution:

1. **Compile**: Python tensor operations → PTX assembly → CUBIN binary
2. **Prepare**: Allocate buffers, set up kernel arguments (QMD)
3. **Submit**: Write commands to the GPU's **GPFIFO** ring buffer
4. **Execute**: GPU reads GPFIFO, executes compute kernels
5. **Signal**: GPU writes completion signal, CPU polls for it

The GPFIFO (GPU FIFO) is a circular buffer in GPU-visible memory. Each entry
points to a **pushbuffer** (a list of GPU commands). The CPU writes entries and
rings a **doorbell** (an MMIO register) to wake the GPU.

```
CPU                          GPU
 │                            │
 │ Write pushbuffer commands  │
 │ Write GPFIFO entry         │
 │ Ring doorbell (MMIO)       │
 │───────────────────────────►│
 │                            │ Read GPFIFO entry
 │                            │ Read pushbuffer
 │                            │ Execute compute kernel
 │                            │ Write signal value
 │◄───────────────────────────│
 │ Poll signal (memory read)  │
 │ Done!                      │
```

## The Optimization Journey

### Phase 0: P0 — VA Window Fix (commit `6977530dc`)
The GPU has special memory "windows" at fixed virtual addresses (0xFD..., 0xFE...)
for shared/local memory. Without reserving these VA ranges, user buffer allocations
could land on top of them → instant GPU fault.

### Phase 1: P1 — TegraIface (commit `a2c290df5` → `cd0746003`)
Implement the nvgpu/nvmap ioctl interface from scratch. ~700 lines of ctypes
structs and ioctl calls. This replaces the desktop PCI/NVK driver path.

### Phase 2: P2 — QMD Race Fix (commit `cd0746003`)
On Tegra, the CPU can submit the next kernel before the GPU finishes reading
the current one's metadata (QMD). Force pushbuffer-based signal release.

### Phase 3: P3 — Matvec Heuristic (commit `2439279b1`)
The heuristic optimizer didn't recognize fp16 matmul patterns (which have CAST
operations in the computation graph). Fix: unwrap CAST/MUL chains recursively.
This was the **7.6× speedup** — from 4.8 to 36.71 tok/s.

### Phase 4: P4 — Toolchain (commits `060970219`, `c7ee61870`)
BEAM search multiprocessing crashed in nix-develop. Fix: use fork instead of
spawn. Also: CUDA include path for NixOS's non-standard layout.
