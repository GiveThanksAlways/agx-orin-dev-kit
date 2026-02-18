# Phase 3: Command Submission

**Status:** COMPLETE ✓  
**Parent doc:** [nv-attempt.md](nv-attempt.md)  
**Previous phase:** [phase2.md](phase2.md)  
**Learning context:** [Learning-Phase3.md](Learning-Phase3.md)

## Goal

Push actual GPU commands via GPFIFO and execute a compute shader. Prove we can dispatch work and wait for completion.

## Results

All 4 Phase 3 tests pass (15/15 total):

| Test | Description | Result |
|------|-------------|--------|
| 12 | mmap userd + GPFIFO buffers | PASS |
| 13 | GPFIFO semaphore release (doorbell verified) | PASS (1.1ms) |
| 14 | nvrtc compiles CUDA C → CUBIN for SM 8.7 | PASS (2984B CUBIN) |
| 15 | Full compute dispatch via QMD | PASS (1.1ms, 32 values verified) |

## Key Discoveries

### Doorbell Mechanism (Critical Finding)
- **NOT** GPPut polling — GPU does NOT detect GPPut updates on its own
- Doorbell is via **mmap of ctrl fd** (`/dev/nvgpu/igpu0/ctrl`)
- The kernel's `gk20a_ctrl_dev_mmap()` maps `g->usermode_regs_bus_addr` (physical IO registers)
- Write `work_submit_token` (511) to offset **0x90** (`NV_USERMODE_NOTIFY_CHANNEL_PENDING`)
- Same offset 0x90 as desktop NV (`gpu_mmio[0x90 // 4]` in tinygrad)
- Latency: ~1ms from doorbell write to GPU completion

### GPFIFO Entry Format
```
gpfifo_entry = (pushbuf_gpu_va & ~3) | (pushbuf_len_words << 42) | (1 << 41)
```
- Bits [40:2] = GPU VA >> 2
- Bit 41 = PRIV flag (always set)
- Bits [52:42] = length in 32-bit words

### Push Buffer Format
```
header = (typ << 28) | (count << 16) | (subchannel << 13) | (method >> 2)
```
- typ=2 = increasing method (auto-increment register address)
- subchannel 0 = GPFIFO/channel class (semaphores)
- subchannel 1 = compute class (0xc7c0)
- subchannel 4 = DMA copy class (0xc7b5)

### QMD V03 (256 bytes)
- `qmd_major_version=3`, `qmd_version=0` (NOT 3!)
- `program_address_lower/upper` = GPU VA of .text section in CUBIN ELF
- `register_count_v` from CUBIN ELF EIATTR
- `shared_memory_size` minimum 0x400 (1024), rounded to 128
- `sass_version=0x87` for SM 8.7
- Constant buffer 0: kernel args at offset 0x160, shared/local mem windows at u32[6:12]

### Compute Dispatch Sequence (Push Buffer)
1. `SET_OBJECT` (subchannel 1, compute class 0xc7c0)
2. `SET_SHADER_SHARED_MEMORY_WINDOW_A` (0x02a0) — **REQUIRED**
3. `SET_SHADER_LOCAL_MEMORY_WINDOW_A` (0x07b0) — **REQUIRED**
4. `SET_SHADER_LOCAL_MEMORY_A` (0x0790) — set to 0 if no local mem
5. `SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A` (0x02e4) — with limit 0x100
6. `INVALIDATE_SHADER_CACHES` (0x021c) — NOT 0x1698 (NO_WFI variant)
7. `SEND_PCAS_A` (0x02b4) — QMD GPU VA >> 8
8. `SEND_SIGNALING_PCAS2_B` (0x02c0) — action=9 (PREFETCH_SCHEDULE)
9. Semaphore release on subchannel 0 for completion detection

### Shader Compilation (nvrtc)
- `libnvrtc.so` compiles CUDA C to CUBIN (not PTX)
- Target: `--gpu-architecture=sm_87`
- Output: ELF with `.text.kernel_name` section containing SASS
- Our test kernel: 2984B CUBIN, 640B SASS, 16 registers

### IO_COHERENCE
- `mmap.flush()` → EINVAL on DMA-BUF mmaps (msync not supported)
- No cache flushes needed — writes are immediately visible to GPU
- Confirmed: CPU writes to output buffer visible; GPU writes to semaphore visible

## Bugs Found & Fixed

1. **`NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI = 0x021c`** — Wrong. 0x021c is WITH WFI. NO_WFI = 0x1698. Using the WFI version (0x021c) works fine for our case.
2. **`qmd_version=3`** — Wrong. For V03_00, `qmd_version` must be 0 (subversion).
3. **Missing shader memory windows** — Must set `SET_SHADER_SHARED_MEMORY_WINDOW_A` and `SET_SHADER_LOCAL_MEMORY_WINDOW_A` before compute dispatch, even for simple kernels.
4. **Missing cbuf_0 fields** — shared_mem_window and local_mem_window at u32[6:12] are needed.
5. **`mmap.flush()` EINVAL** — DMA-BUF mmaps don't support msync. IO_COHERENCE means no flush needed.

## Key Info From Phase 1

- CUDA uses usermode submit (no SUBMIT_GPFIFO ioctl)
- Work submit token (doorbell): 511
- Syncpoint ID: 17, max=38000, GPU VA=0xffffe10000
- GPFIFO: 1024 entries x 8 bytes = 8192 bytes
- Userd: 4096 bytes (GPGet@0x88, GPPut@0x8C)
- Compute class: 0xc7c0 (Ampere compute)
- GPFIFO class: 0xc76f
- DMA copy class: 0xc7b5
