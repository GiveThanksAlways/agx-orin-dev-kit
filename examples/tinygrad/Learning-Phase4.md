# Learning-Phase4.md — TegraIface Integration

## Summary

Phase 4 integrated all knowledge from Phases 1-3 into a working `TegraIface` class in tinygrad's `ops_nv.py`. The class translates tinygrad's HCQ (Hardware Command Queue) API from NVIDIA's desktop RM/NVK driver model to Jetson's nvgpu+nvmap driver model. After fixing a critical kernel constraint around GPFIFO buffer sizing, `NV=1` tinygrad works end-to-end on Orin AGX 64GB.

## Key Lesson: One Buffer Size Brought Down the Whole System

The most important learning from Phase 4 was a single root-cause bug: **passing a 3MB dmabuf to SETUP_BIND causes a system-level freeze at ALLOC_OBJ_CTX**.

The nvgpu kernel code in `linux-channel.c` has two undocumented constraints:
1. `gpfifo_dmabuf_offset` and `userd_dmabuf_offset` must both be 0 (non-zero returns EINVAL)
2. The kernel maps the **entire** dmabuf into the channel's GPU VA space via `nvgpu_gmmu_map()`

When tinygrad's 3MB gpfifo_area dmabuf was passed to SETUP_BIND, the kernel mapped all 3MB into GPU VA. This consumed enough of the channel's internal VA space that the subsequent GR context allocation in ALLOC_OBJ_CTX couldn't find room, triggering `tu104_gr_init_commit_rtv_cb` kernel WARNING followed by CE engine halt and system freeze.

The fix: create per-channel small dedicated nvmap buffers (entries*8 bytes for gpfifo ring, 4KB for userd) and MAP_FIXED overlay them at the correct positions within the original gpfifo_area CPU mmap.

## What TegraIface Translates

tinygrad's NV backend uses NVIDIA's Resource Manager (RM) API. TegraIface translates each RM concept to its nvgpu equivalent:

### rm_alloc translations
| RM class | nvgpu equivalent | Notes |
|---|---|---|
| NV01_DEVICE_0 | NOP | Hierarchy container only |
| NV20_SUBDEVICE_0 | NOP | Hierarchy container only |
| NV01_MEMORY_VIRTUAL | ALLOC_AS | Creates 40-bit unified VA address space |
| FERMI_VASPACE_A | NOP | AS already created |
| KEPLER_CHANNEL_GROUP_A | OPEN_TSG | Thread scheduling group |
| FERMI_CONTEXT_SHARE_A | CREATE_SUBCONTEXT | Async compute subcontext |
| AMPERE_CHANNEL_GPFIFO_A | OPEN_CHANNEL + SETUP_BIND | Full channel pipeline (6 ioctls) |
| AMPERE_COMPUTE_B | ALLOC_OBJ_CTX | Binds GR engine to channel |
| AMPERE_DMA_COPY_B | ALLOC_OBJ_CTX | Binds CE engine to channel |
| GT200_DEBUGGER | NOP (stub) | Not needed on Tegra |

### rm_control translations
| RM control | nvgpu equivalent | Notes |
|---|---|---|
| NV2080_CTRL_CMD_PERF_BOOST | devfreq sysfs write | Best-effort max frequency |
| NVA06C_CTRL_CMD_GPFIFO_SCHEDULE | NOP | Auto-scheduled on nvgpu |
| NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN | Return saved token | From SETUP_BIND output |
| NV2080_CTRL_CMD_GR_GET_INFO | GET_CHARACTERISTICS fields | Maps GR info indices to chars |
| NV0080_CTRL_CMD_GPU_GET_CLASSLIST | Return known classes | compute, gpfifo, dma, usermode |
| NV2080_CTRL_CMD_FB_FLUSH_GPU_CACHE | NOP | IO coherent on Orin |

### Memory allocation
Desktop: UVM (nvidia-uvm kernel module) or NVK mmap
Tegra: nvmap CREATE → ALLOC → GET_FD → MAP_BUFFER_EX → mmap

### Command submission
Identical on both paths once the channel is set up:
1. Write GPFIFO entry to ring buffer (push buffer address + length)
2. Update GPPut in USERD (AmpereAControlGPFifo)
3. Memory barrier
4. Write work_submit_token to doorbell (ctrl mmap + 0x90 on Tegra, BAR0 on desktop)

## Channel Setup Pipeline

The most complex part of TegraIface is the channel setup, which requires 6 ioctls in exact order:

```
OPEN_CHANNEL → AS_BIND → TSG_BIND → WDT_DISABLE → SETUP_BIND → ALLOC_OBJ_CTX
```

Each step must succeed before the next. SETUP_BIND is the critical one that creates the kernel-side GPFIFO ring mapping and returns the work_submit_token for doorbell.

### SETUP_BIND requirements (discovered the hard way)
- `gpfifo_dmabuf_fd`: must point to a dedicated small buffer (exactly entries*8 bytes)
- `gpfifo_dmabuf_offset`: MUST be 0
- `userd_dmabuf_fd`: must be a separate 4KB buffer (not the gpfifo buffer)
- `userd_dmabuf_offset`: MUST be 0
- `flags`: USERMODE_SUPPORT (0x2) | DETERMINISTIC (0x8) = 0xa
- Output: `work_submit_token`, `gpfifo_gpu_va`, `userd_gpu_va`

## MAP_FIXED Overlay Technique

tinygrad expects a single contiguous gpfifo_area buffer that holds both the ring buffer and USERD for all channels. But nvgpu requires separate small dmabufs per channel.

Solution: allocate gpfifo_area as a normal nvmap buffer (for CPU-side addressing), then MAP_FIXED overlay per-channel dmabuf pages at the correct offsets:

```
gpfifo_area CPU mmap (64KB):
┌─────────────────┬────────────┬─────────────────┬────────────┬──────┐
│  Ring (8KB)     │ USERD (4KB)│  Ring (8KB)     │ USERD (4KB)│ pad  │
│  MAP_FIXED      │ MAP_FIXED  │  MAP_FIXED      │ MAP_FIXED  │      │
│  Ch0 ring dmabuf│ Ch0 userd  │  Ch1 ring dmabuf│ Ch1 userd  │      │
└─────────────────┴────────────┴─────────────────┴────────────┴──────┘
offset: 0          8K           12K               20K          24K
```

Each MAP_FIXED replaces the original nvmap pages with the per-channel dmabuf pages. CPU reads/writes go to the same physical pages the GPU sees via SETUP_BIND.

## GPU Constants for Orin AGX (ga10b)

From GET_CHARACTERISTICS:
- `arch`: 0x0170 (Ampere)
- `impl`: 0x0b (ga10b)
- `sm_arch_sm_version`: 0x807 (SM 8.7)
- `compute_class`: 0xc7c0 (AMPERE_COMPUTE_B)
- `gpfifo_class`: 0xc76f
- `dma_copy_class`: 0xc7b5 (AMPERE_DMA_COPY_B)
- `num_gpc`: 1
- `num_tpc_per_gpc`: 8
- `max_warps_per_sm`: 48

Derived values:
- 40-bit GPU VA space (max 0xFFFFFFFFFF)
- `shared_mem_window`: 0xFE00000000 (40-bit range, avoids collision)
- `local_mem_window`: 0xFD00000000

## Debugging Techniques That Worked

1. **A/B variant testing** (`test_channel_variants.py`): Systematically varied one parameter at a time to isolate the crash variable (buffer size)
2. **Kernel source reading**: Reading the actual nvgpu kernel code (`linux-channel.c`) revealed the exact constraints that no documentation mentioned
3. **Incremental test harness** (`tests/` directory): Built standalone tests for each GPU pipeline stage (AS → channels → submit) before integrating into tinygrad
4. **MAP_FIXED technique**: Used `mmap(MAP_FIXED)` to overlay different dmabuf pages at specific CPU addresses, bridging tinygrad's single-buffer expectation with nvgpu's per-channel requirement
5. **UART/serial monitoring**: Connected UART to see kernel panics that couldn't be captured in dmesg post-reboot

## Performance Notes

- 1024x1024 matmul: ~33 ms, ~65 GFLOPS on ga10b iGPU
- This is with tinygrad's generic Ampere code, not Orin-optimized
- CUDA backend via cuDNN would be faster but NV backend gives direct GPU control
- IO coherence means no cache flush overhead on Orin

## What's Not Working Yet

- GPT-2 inference (not tested yet, may need local memory / shader features)
- Video decode path (viddec_class is None on nvgpu)
- PMA profiling (disabled on Tegra: `self.pma_enabled = PMA.value > 0 and PROFILE >= 1 and not self.is_tegra()`)

## Upstream Considerations

To upstream TegraIface to tinygrad:
1. The code is self-contained — it's an additional `Iface` class alongside NVKIface and PCIIface
2. The ctypes struct definitions should be auto-generated from nvgpu headers (currently hand-written)
3. Detection is clean: `os.path.exists("/dev/nvgpu/igpu0/ctrl")`
4. Entries reduction (1024 vs 65536) only affects Tegra path
5. Test coverage: should add `NV=1` CI on Jetson hardware
