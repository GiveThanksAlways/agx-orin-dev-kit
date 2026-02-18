# Learning Phase 3: Command Submission on Jetson Orin AGX

**Date:** 2025-01-XX  
**Hardware:** Jetson Orin AGX 64GB, ga10b iGPU (Ampere SM 8.7)  
**Previous:** [Learning-Phase2.md](Learning-Phase2.md), [Learning-Phase1.md](Learning-Phase1.md)

## Executive Summary

Phase 3 proved that we can push GPU commands through the GPFIFO and execute a compute shader on the Jetson Orin's ga10b GPU entirely from userspace Python — no CUDA runtime, no driver abstractions. We:

1. Mapped the userd and GPFIFO buffers to CPU (mmap on DMA-BUF fds)
2. Discovered the **doorbell mechanism**: write work_submit_token to offset 0x90 in the ctrl fd mmap
3. Compiled CUDA C → CUBIN SASS via libnvrtc.so
4. Built a QMD V03 struct and push buffer from scratch
5. Dispatched a 32-thread compute kernel and verified output from CPU

All 15 tests pass (7 Phase 1 + 4 Phase 2 + 4 Phase 3).

---

## The Command Submission Pipeline

### Architecture Overview

```
CPU (Python)                              GPU (ga10b Ampere)
─────────────                             ─────────────────
1. Allocate buffers (nvmap)
2. Map buffers to GPU VA (AS)
3. Build push buffer (methods)
4. Build QMD (compute descriptor)
5. Write GPFIFO entry ──────────────────> GPFIFO ring buffer
6. Update GPPut in userd ────────────────> userd control regs
7. Write token to doorbell ──────────────> usermode MMIO regs
                                           │
                                           ▼
                                          GPU reads GPFIFO entry
                                          GPU fetches push buffer
                                          GPU executes methods
                                          GPU launches QMD
                                          GPU runs shader
                                          GPU writes semaphore
                                           │
8. Poll semaphore <──────────────────────── completion
9. Read output buffer
```

### Step-by-Step Data Flow

1. **Push buffer** = array of GPU "methods" (register writes): SET_OBJECT, SET_SHADER_WINDOWS, SEND_PCAS, etc.
2. **GPFIFO entry** (8 bytes) points to push buffer GPU VA + length
3. **GPPut** in userd tells GPU "I added entries up to index N"
4. **Doorbell** at ctrl fd mmap offset 0x90 wakes the GPU scheduler
5. GPU processes push buffer methods, encounters SEND_PCAS_A → reads QMD
6. QMD tells GPU: shader address, grid/block dims, registers, constant buffer
7. GPU executes shader, then signals completion via semaphore in memory

---

## The Doorbell: The Hardest Part

### What We Expected
Based on the kernel returning `work_submit_token=511`, we assumed updating GPPut in the userd buffer would be sufficient (GPU polls userd). This was **wrong**.

### What Actually Works
The doorbell is a **physical MMIO register** mapped via `mmap()` on the ctrl fd (`/dev/nvgpu/igpu0/ctrl`):

```python
# Map the usermode register page (kernel's gk20a_ctrl_dev_mmap)
doorbell_mm = mmap.mmap(ctrl_fd, 0x1000, mmap.MAP_SHARED,
                         mmap.PROT_READ | mmap.PROT_WRITE, 0)

# Ring doorbell — write token at offset 0x90 (NV_USERMODE_NOTIFY_CHANNEL_PENDING)
struct.pack_into('<I', doorbell_mm, 0x90, work_submit_token)
```

### How We Found This
1. GPPut-only updates did nothing (GPU never processed entries)
2. SUBMIT_GPFIFO ioctl with 0 entries → ENOTTY (blocked in usermode submit mode)
3. Searched nvgpu kernel source → found `gk20a_ctrl_dev_mmap()` mapping `g->usermode_regs_bus_addr`
4. Offset 0x90 matches desktop NV's `gpu_mmio[0x90 // 4]` in tinygrad
5. Writing token=511 at offset 0x90 → immediate GPU response (~1ms)

### Key Insight
This is the **same** mechanism as desktop NV:
- Desktop: `gpu_mmio` = BAR0 usermode region → write token at 0x90
- Jetson: ctrl fd mmap = `usermode_regs_bus_addr` → write token at 0x90

The only difference is how you get the CPU mapping. On desktop it's via the PCI BAR; on Jetson it's via the ctrl device's mmap handler.

---

## GPFIFO and Push Buffer Format

### GPFIFO Entry (8 bytes, little-endian u64)
```python
gpfifo_entry = (pushbuf_gpu_va & ~3) | (pushbuf_len_words << 42) | (1 << 41)
```
- Bits [40:2]: GPU VA of push buffer >> 2
- Bit 41: PRIV flag (always set for compute)
- Bits [52:42]: length in 32-bit words

### Push Buffer Method Header (4 bytes)
```python
header = (typ << 28) | (count << 16) | (subchannel << 13) | (method >> 2)
```
- `typ=2` = increasing method (auto-increment register)
- `subchannel`: 0 = GPFIFO/host, 1 = compute, 4 = DMA copy
- `method >> 2`: target register divided by 4
- `count`: number of data words following

### Example: Launching a QMD
```python
pb = PushBufferBuilder()
pb.nvm(1, 0x0000, compute_class)           # SET_OBJECT
pb.nvm(1, 0x02a0, hi17, lo32)              # SET_SHADER_SHARED_MEMORY_WINDOW
pb.nvm(1, 0x07b0, hi17, lo32)              # SET_SHADER_LOCAL_MEMORY_WINDOW
pb.nvm(1, 0x0790, 0, 0)                    # SET_SHADER_LOCAL_MEMORY (null)
pb.nvm(1, 0x02e4, 0, 0, 0x100)             # SET_SHADER_LOCAL_MEMORY_NON_THROTTLED
pb.nvm(1, 0x021c, 0x1011)                  # INVALIDATE_SHADER_CACHES
pb.nvm(1, 0x02b4, qmd_gpu_va >> 8)         # SEND_PCAS_A
pb.nvm(1, 0x02c0, 9)                       # SEND_SIGNALING_PCAS2_B (PREFETCH_SCHEDULE)
pb.nvm(0, 0x005c, sem_lo, sem_hi, val_lo, val_hi, sem_flags)  # SEM release
```

---

## QMD V03 (Queue Meta Data)

The QMD is 256 bytes (64 dwords). It tells the GPU everything about the compute dispatch.

### Critical Fields
| Field | Bits | Our Value | Notes |
|-------|------|-----------|-------|
| qmd_major_version | 583:580 | 3 | Must be 3 for V03 |
| qmd_version | 579:576 | 0 | Subversion. NOT 3! |
| qmd_group_id | 133:128 | 0x3f | Standard |
| program_address_lower | 1567:1536 | .text VA lo32 | Points to SASS code |
| program_address_upper | 1584:1568 | .text VA hi17 | |
| register_count_v | 656:648 | 16 | From CUBIN ELF |
| shared_memory_size | 561:544 | 0x400 | Min 1024, round to 128 |
| cta_raster_width | 415:384 | 1 | Grid X |
| cta_thread_dimension0 | 607:592 | 32 | Block X (one warp) |
| sass_version | 1663:1656 | 0x87 | SM 8.7 |
| barrier_count | 767:763 | 1 | At least 1 |
| sm_global_caching_enable | 134:134 | 1 | |
| api_visible_call_limit | 378:378 | 1 | NO_CHECK |
| sampler_index | 382:382 | 1 | VIA_HEADER_INDEX |
| cwd_membar_type | 369:368 | 1 | L1_SYSMEMBAR |

### Constant Buffer 0
Constant buffer 0 carries kernel arguments and runtime data:
- `cbuf_0[6:8]` (bytes 24-31): `shared_mem_window` GPU VA
- `cbuf_0[8:10]` (bytes 32-39): `local_mem_window` GPU VA
- `cbuf_0[10:12]` (bytes 40-47): `0xfffdc0` (constant)
- Offset 0x160: kernel parameters (pointers, scalars)

### Shared Memory Windows
On Jetson with 40-bit VA, we use addresses within range:
```python
shared_mem_window = 0xfe00000000  # fits in 40-bit
local_mem_window  = 0xfd00000000  # fits in 40-bit
```
Desktop tinygrad uses `0x729400000000` (48-bit), which is too large for Jetson.

---

## Shader Compilation

### Using libnvrtc.so
We compile CUDA C (not PTX) directly to CUBIN using `libnvrtc.so`:
```python
cubin = compile_ptx_to_cubin(kernel_source, arch="sm_87")
```
This calls `nvrtcCreateProgram` → `nvrtcCompileProgram` → `nvrtcGetCUBIN`.

### Test Kernel
```c
extern "C" __global__ void test_kernel(float *out) {
    int tid = threadIdx.x;
    out[tid] = (float)(tid * tid + 1);
}
```
Expected output for 32 threads: `[1.0, 2.0, 5.0, 10.0, 17.0, 26.0, ...]`

### CUBIN Structure
- Full ELF with sections: .text.test_kernel, .nv.info, .nv.info.test_kernel, etc.
- .text section = SASS machine code (640 bytes for our kernel)
- EIATTR in .nv.info contains: register count, stack frame size, etc.
- Our kernel: 16 registers, 0 shared memory, 640B SASS

### Program Address
The `program_address` in QMD points to the `.text` section **within the ELF** — not the start of the CUBIN. We upload the entire CUBIN ELF and add the .text offset:
```python
prog_addr = shader_buf.gpu_va + cubin_info['text_offset']
```

---

## What Went Wrong (Debugging Log)

### Bug 1: mmap.flush() EINVAL
**Symptom:** `OSError: [Errno 22] Invalid argument` on `mmap.flush()`  
**Cause:** `mmap.flush()` calls `msync()`, which returns EINVAL on DMA-BUF file descriptors  
**Fix:** Remove all flush() calls. IO_COHERENCE means writes are immediately visible.

### Bug 2: Doorbell Not Triggering GPU
**Symptom:** GPPut updated but GPGet stays at 0, semaphore never written  
**Cause:** Updating GPPut in userd is necessary but not sufficient. A doorbell write is required.  
**Fix:** mmap the ctrl fd and write `work_submit_token` to offset 0x90

### Bug 3: SKED Exception (GPU Scheduler Error)
**Symptom:** `sked exception: esr 0x01000000` in dmesg, semaphore never written  
**Root Causes (3 bugs at once):**
1. `NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI` was set to 0x021c (that's the WITH WFI version). Correct NO_WFI = 0x1698. Using WITH WFI (0x021c) actually works fine.
2. `qmd_version=3` instead of `qmd_version=0`. V03 = major=3, subversion=0.
3. Missing `SET_SHADER_SHARED_MEMORY_WINDOW_A` and `SET_SHADER_LOCAL_MEMORY_WINDOW_A` methods in push buffer. The GPU requires these before any compute dispatch.

**Fix:** Corrected all three issues → compute dispatch works.

---

## Performance Observations

- Doorbell to GPU completion: ~1.1ms for both semaphore-only and compute+semaphore
- nvrtc compilation: fast (sub-second for trivial kernel)
- The 1ms latency likely includes GPU power-up or scheduling overhead

---

## Compare: tinygrad Desktop NV vs Jetson nvgpu

| Aspect | Desktop NV (tinygrad) | Jetson nvgpu (our code) |
|--------|----------------------|------------------------|
| Driver | nvidia.ko (proprietary) | nvgpu.ko (open source) |
| Doorbell mmap | PCI BAR0 via /dev/nvidia-uvm or fd_ctl | ctrl fd mmap |
| Doorbell offset | 0x90 | 0x90 (same!) |
| Doorbell value | `gpfifo.token` | `work_submit_token` |
| GPU VA bits | 48 (or more) | 40 |
| GPFIFO entry format | Same | Same |
| Push buffer format | Same | Same |
| QMD version | V03 (Ampere) | V03 (same) |
| Shared mem window | 0x729400000000 | 0xfe00000000 (fits 40-bit) |
| SUBMIT_GPFIFO ioctl | Not used (usermode) | Not used (ENOTTY in usermode) |
| mmap.flush() | Works (BAR mapping) | EINVAL (DMA-BUF) |

---

## Code Structure (test_nvgpu.py Phase 3 additions)

### New Classes
- **QMDBuilder**: 256-byte QMD V03 struct builder with bit-level field write
- **PushBufferBuilder**: GPU push buffer builder with `nvm()` method

### New Functions
- `compile_ptx_to_cubin()`: CUDA C → CUBIN via libnvrtc.so
- `parse_cubin_elf()`: Extract .text, register count, shared mem from CUBIN ELF
- `submit_pushbuf()`: Write GPFIFO entry + GPPut + doorbell

### Test Functions
- `test_mmap_userd_gpfifo()` (Test 12): Map userd/GPFIFO/doorbell
- `test_gpfifo_semaphore_release()` (Test 13): Prove GPFIFO submission works
- `test_nvrtc_compile()` (Test 14): Compile kernel to CUBIN
- `test_compute_dispatch()` (Test 15): Full QMD dispatch with verification

---

## What's Next (Phase 4)

Phase 3 proved we can dispatch compute from userspace. Next steps toward a tinygrad backend:

1. **DMA copy operations** — use subchannel 4 / DMA copy class for memory transfers
2. **Multiple kernel dispatches** — ring buffer management, GPFIFO wrap-around
3. **Syncpoint-based completion** — use syncpoint_id=17 instead of semaphore polling
4. **Performance optimization** — batch multiple kernels, minimize doorbell writes
5. **tinygrad integration** — implement `TegraIface` backend class for tinygrad's runtime
