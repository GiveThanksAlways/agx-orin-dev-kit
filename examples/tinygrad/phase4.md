# Phase 4: TegraIface Integration

**Status:** COMPLETE ✅  
**Parent doc:** [nv-attempt.md](nv-attempt.md)  
**Previous phase:** [phase3.md](phase3.md)  
**Learning context:** [Learning-Phase1.md](Learning-Phase1.md), [Learning-Phase4.md](Learning-Phase4.md)

## Goal

Build `TegraIface` class that plugs into tinygrad's NV runtime (`ops_nv.py`), enabling `NV=1` on Jetson Orin.

## Result

**Working.** All target operations verified on Orin AGX 64GB:

```bash
NV=1 python3 -c "from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())"
# [1 2 3]

NV=1 python3 -c "from tinygrad import Tensor; print((Tensor([1,2,3]) + Tensor([4,5,6])).numpy())"
# [5 7 9]

# Matrix multiply: [[19. 22.] [43. 50.]]
# 1024x1024 matmul: ~33 ms/iter, ~65 GFLOPS
# Conv2d: correct output shape (1, 16, 62, 62)
```

## Architecture

TegraIface replaces the NVK/PCI driver layer while keeping tinygrad's GPU programming layer (QMD, shaders, push buffers) unchanged:

```
tinygrad HCQ (GPU programming)
    │
    ├── NVKIface (desktop open kernel RM)
    ├── PCIIface (desktop proprietary PCI)
    └── TegraIface (Jetson nvgpu+nvmap)  ← NEW
```

### Key Implementation Details

| Component | Desktop (NVK/PCI) | Tegra |
|---|---|---|
| Memory alloc | UVM / nvidia-uvm | nvmap CREATE→ALLOC→GET_FD→MAP_BUFFER_EX→mmap |
| Address space | RM VASPACE | nvgpu ALLOC_AS (40-bit, UNIFIED_VA) |
| Channel group | RM KEPLER_CHANNEL_GROUP_A | nvgpu OPEN_TSG |
| Subcontext | RM FERMI_CONTEXT_SHARE_A | nvgpu TSG CREATE_SUBCONTEXT |
| Channel setup | RM gpfifo alloc | nvgpu OPEN_CHANNEL→BIND→TSG_BIND→WDT→SETUP_BIND |
| Object alloc | RM rm_alloc | nvgpu ALLOC_OBJ_CTX |
| Command submit | Write GPPut + doorbell | Same (usermode submit via USERD + doorbell at ctrl mmap +0x90) |
| Doorbell | BAR0 mmap | ctrl fd mmap at offset 0x90 |

### Critical Kernel Constraint (Root Cause of Crashes)

Found in `nvgpu/os/linux/linux-channel.c` line 518-524:

1. **`gpfifo_dmabuf_offset` MUST be 0** — non-zero returns `-EINVAL` ("TODO - not yet supported")
2. **`userd_dmabuf_offset` MUST be 0** — same constraint
3. **Kernel maps the ENTIRE dmabuf** into GPU VA via `nvgpu_gmmu_map` — a 3MB dmabuf consumes 3MB of GPU VA, which fragments the channel's VA space and causes `tu104_gr_init_commit_rtv_cb` WARNING + system freeze during ALLOC_OBJ_CTX

**Solution: per-channel small dedicated buffers:**
- Each channel gets its OWN gpfifo ring dmabuf (entries*8 bytes, e.g., 8KB for 1024 entries)
- Each channel gets its OWN userd dmabuf (4KB)
- Both at offset 0 in their respective dmabufs
- MAP_FIXED overlays at correct positions within the gpfifo_area CPU mmap so tinygrad's cpu_view() works unchanged

### Files Modified

- **`tinygrad/tinygrad/runtime/ops_nv.py`**: TegraIface class (~400 lines) + NVDevice Tegra paths
- **`tests/tegra_helpers.py`**: Shared test helpers (ctypes structs, ioctl codes, channel setup)
- **`tests/test_two_channels.py`**: Validates compute + DMA channels with ALLOC_OBJ_CTX
- **`tests/test_gpu_submit.py`**: Validates full GPU command submission with semaphore release

## Tasks

- [x] Study NVKIface and PCIIface interface contracts
- [x] Implement TegraIface class with nvgpu/nvmap backend
- [x] Replace UVM memory ops with nvmap allocator
- [x] Replace RM channel creation with nvgpu TSG+channel pipeline
- [x] Replace RM GPFIFO submit with usermode submit
- [x] Add detection: check for /dev/nvgpu/igpu0/ctrl in _select_iface()
- [x] Debug crash: 3MB gpfifo buffer → system freeze at ALLOC_OBJ_CTX
- [x] Root cause: kernel maps entire dmabuf + non-zero offset not supported
- [x] Fix: per-channel small buffers with MAP_FIXED overlays
- [x] Test: vector add
- [x] Test: matrix multiply
- [x] Test: conv2d
- [x] Test: 1024x1024 matmul (~65 GFLOPS)
- [ ] Test: GPT-2 end-to-end
- [ ] Compare outputs with CUDA backend for correctness

## Crash Investigation Timeline

1. Initial TegraIface with 3MB gpfifo_area → ALLOC_OBJ_CTX → system freeze
2. DMA channel with gpfifo_dmabuf_offset=0x100000 → EINVAL (caught by kernel)
3. Created test_channel_variants.py: proved 8KB buffer PASSES, 3MB buffer CRASHES
4. Read kernel source (linux-channel.c): found both constraints
5. Created tests/test_two_channels.py: validated two channels with small buffers
6. Created tests/test_gpu_submit.py: validated GPU write 0xDEADBEEF42 to semaphore
7. Fixed TegraIface: per-channel small buffers + MAP_FIXED overlays
8. NV=1 tinygrad works end-to-end ✅
