# IOCTL ASCII Diagrams -- Tinygrad NV=1 on Jetson Orin AGX

Presentation-ready reference for the TegraIface reverse-engineering project.

Hardware: NVIDIA Jetson Orin AGX 64GB, ga10b iGPU (Ampere SM 8.7)
Software: JetPack 6 (L4T r36.4.4), Kernel 5.15.148, CUDA 12.6, tinygrad v0.12.0

---

## Table of Contents

1. [Big Picture Summary (TL;DR of Phases 1-4)](#1-big-picture-summary)
2. [Tegra vs Desktop Driver Stacks](#2-tegra-vs-desktop-driver-stacks)
3. [Full IOCTL Pipeline -- How NV=1 Works on Tegra](#3-full-ioctl-pipeline)
4. [Anatomy of SETUP_BIND (the key ioctl)](#4-anatomy-of-setup_bind)
5. [Key Takeaways Slide](#5-key-takeaways-slide)
6. [FAQ / Anticipated Engineering Questions](#6-faq--anticipated-engineering-questions)
7. [C Hot Path -- Production GPU Dispatch Loop](#7-c-hot-path)
8. [Full Pipeline -- Tensor Ops to GPU Warps](#8-full-pipeline----tensor-ops-to-gpu-warps)
9. [Compact Pipeline (one-page)](#9-compact-pipeline-one-page)

---

## 1. Big Picture Summary

### The Problem

tinygrad's NV backend (`NV=1`) lets you program NVIDIA GPUs without CUDA, using
direct ioctl calls to the kernel driver. On desktop Linux, it has two interfaces:

- **NVKIface** -- talks to nvidia.ko via RM (Resource Manager) ioctls + nvidia-uvm.ko
- **PCIIface** -- maps PCI BAR registers directly

Neither works on the Jetson Orin AGX 64GB:

```text
NVKIface failure:
  /dev/nvidia-uvm        --> "Module not found" (nvidia-uvm is JetPack 7 / Thor only)
  NV01_DEVICE_0 rm_alloc --> NV_ERR_INVALID_CLASS (34) -- Tegra nvidia.ko is display-only

PCIIface failure:
  GPU is on platform bus @ 0x17000000, NOT on PCIe
  No PCI BARs to map
```

### The Solution

We built a third interface, **TegraIface**, by reverse-engineering the Jetson-specific
kernel drivers (nvgpu.ko + nvmap.ko) through a combination of:

- Extracting and reading the L4T BSP kernel source (public_sources.tbz2)
- Stracing CUDA programs to observe the exact ioctl call sequence
- Brute-force parameter testing against the kernel validation code
- Reading nvgpu kernel source (C) directly to find undocumented constraints

### Phase-by-Phase Summary

```text
Phase 1: Reverse-Engineer nvgpu IOCTLs                           7/7  tests pass
---------------------------------------------------------------------------
  - Downloaded L4T BSP kernel sources, found ioctl headers for nvgpu + nvmap
  - Straced CUDA: 1793 total ioctls, 39 unique codes, 5 device fds
  - Key discovery: ZERO SUBMIT_GPFIFO calls in the trace -- CUDA uses
    usermode submit (writes GPFIFO entries + rings doorbell, no syscall)
  - Built Python ctypes structs for every ioctl struct (alignment traps!)
  - Decoded initialization sequence from strace ordering
  - Solved ALLOC_AS: VA range must be PDE-aligned (2MB boundary for ga10b)
  - Solved SETUP_BIND: flags must include both USERMODE_SUPPORT and
    DETERMINISTIC (0x0A) -- kernel rejects one without the other
  - Solved channel bind order: AS_BIND must come before TSG_BIND
  - Result: compute class 0xc7c0 allocated, work_submit_token=511

Phase 2: Memory Management via nvmap                            11/11 tests pass
---------------------------------------------------------------------------
  - mmap works on the dmabuf fd (from NVMAP_IOC_GET_FD), NOT /dev/nvmap
  - IO_COHERENCE is real: CPU writes are immediately visible to GPU with
    no explicit cache operations needed (confirmed via dual-mmap test)
  - Fixed nvmap_alloc_handle struct: 5th field is numa_nid (s32), not
    kind (u8) -- Phase 1 worked by coincidence (both produce zero bytes)
  - Cacheability benchmarks (1MB buffer):
      INNER_CACHEABLE: 2737 MB/s write, 11826 MB/s read (best overall)
      WRITE_COMBINE:   2917 MB/s write,   691 MB/s read
      UNCACHEABLE:      688 MB/s write,    75 MB/s read (avoid!)
  - INNER_CACHEABLE gives 17x faster reads than WRITE_COMBINE
  - Built TegraAllocator class: alloc/mmap/gpu_map/free with cleanup
  - Tested 4KB through 64MB allocations, GPU VA assigned top-down

Phase 3: Command Submission                                     15/15 tests pass
---------------------------------------------------------------------------
  - Discovered doorbell mechanism: mmap ctrl fd, write token to offset 0x90
    (same offset 0x90 as desktop NV BAR0 -- hardware convergence point)
  - GPPut updates alone do NOT wake the GPU -- doorbell is mandatory
  - Built GPFIFO entry format: (gpu_va & ~3) | (len_words << 42) | (1<<41)
  - Built push buffer method headers: (typ<<28)|(count<<16)|(subch<<13)|(meth>>2)
  - Built QMD V03 (256 bytes) from scratch: qmd_major_version=3, qmd_version=0
  - Compiled CUDA C to CUBIN via libnvrtc.so (sm_87 target)
  - Dispatched 32-thread compute kernel, verified output from CPU
  - Latency: ~1.1ms doorbell-to-completion
  - Fixed 3 bugs at once for SKED exception: wrong cache invalidate register,
    wrong qmd_version field, missing shader memory windows in push buffer

Phase 4: TegraIface Integration                                 NV=1 working
---------------------------------------------------------------------------
  - Built ~400-line TegraIface class in tinygrad ops_nv.py
  - Translates 10 RM rm_alloc classes to nvgpu equivalents (or NOPs)
  - Translates 6 RM rm_control calls to nvgpu equivalents
  - Root cause of system freeze: passing 3MB dmabuf to SETUP_BIND causes
    kernel to map entire buffer into channel GPU VA, exhausting VA space
    so ALLOC_OBJ_CTX cannot allocate GR context --> CE engine halt --> freeze
  - Fix: per-channel small dedicated buffers (8KB ring + 4KB userd)
    with MAP_FIXED overlays into tinygrad's expected contiguous mmap
  - Verified: vector add, matrix multiply, conv2d, 1024x1024 matmul (~65 GFLOPS)
```

### End-to-End Data Flow

```text
 NV=1 python3 -c "print((Tensor([1,2,3]) + Tensor([4,5,6])).numpy())"
      |
      v
 tinygrad runtime (ops_nv.py)
      |
      |  _select_iface() detects /dev/nvgpu/igpu0/ctrl --> TegraIface
      |
      v
 TegraIface.__init__()
      |
      |  1. open /dev/nvmap            nvmap_fd
      |  2. open /dev/nvgpu/igpu0/ctrl ctrl_fd
      |  3. GET_CHARACTERISTICS        arch=0x170, SM 8.7, compute=0xc7c0
      |  4. ALLOC_AS                   40-bit VA space (0x200000..0xFFFFE00000)
      |  5. OPEN_TSG + OPEN_CHANNEL    scheduling group + command channel
      |  6. SETUP_BIND                 usermode submit enabled, token=511
      |  7. ALLOC_OBJ_CTX              compute engine + DMA engine bound
      |  8. mmap ctrl fd               doorbell page at offset 0x90
      |
      v
 TegraIface.alloc() -- for each tensor buffer
      |
      |  nvmap CREATE --> ALLOC (INNER_CACHEABLE) --> GET_FD --> mmap (CPU)
      |  MAP_BUFFER_EX (GPU VA)
      |
      v
 tinygrad compiles kernel (libnvrtc.so, sm_87) --> CUBIN ELF
      |
      v
 HCQ builds push buffer + QMD, writes GPFIFO entry
      |
      |  GPFIFO[gp_put] = pushbuf_gpu_va | (len << 42) | PRIV
      |  USERD[0x8C]    = gp_put + 1                (update GPPut)
      |  doorbell[0x90] = 511                       (wake GPU)
      |
      v
 GPU reads GPFIFO --> fetches push buffer --> processes methods --> runs shader
      |
      v
 GPU writes semaphore (IO coherent -- CPU sees it immediately)
      |
      v
 CPU reads output buffer --> [5 7 9]
```

---

## 2. Tegra vs Desktop Driver Stacks

```text
 +----------------------------------------------------------------------------------+
 |   DESKTOP NVIDIA (dGPU)                          JETSON ORIN (iGPU)              |
 |   ====================                           =================               |
 |                                                                                  |
 |   tinygrad NV=1                                  tinygrad NV=1                   |
 |       |                                              |                           |
 |       v                                              v                           |
 |   NVKIface / PCIIface                           TegraIface                       |
 |       |                                              |                           |
 |   +---+--------------------+                    +----+----------------------+    |
 |   |  nvidia.ko             |                    |  nvgpu.ko                 |    |
 |   |  (full RM API)         |                    |  (compute driver)         |    |
 |   |  /dev/nvidiactl        |                    |  /dev/nvgpu/igpu0/*       |    |
 |   +------------------------+                    +---------------------------+    |
 |   |  nvidia-uvm.ko         |                    |  nvmap.ko                 |    |
 |   |  (memory mgmt)         |                    |  (memory allocator)       |    |
 |   |  /dev/nvidia-uvm       |                    |  /dev/nvmap               |    |
 |   +--------+---------------+                    +----------+----------------+    |
 |            |                                               |                     |
 |       +----+----+                                   +------+-----+               |
 |       |   GPU   |                                   |   GPU      |               |
 |       |  (PCIe) |                                   |   (SoC)    |               |
 |       |  VRAM   |                                   |  Unified   |               |
 |       +---------+                                   |  DRAM 64GB |               |
 |                                                     +------------+               |
 |                                                                                  |
 |   nvidia-uvm.ko   -->  MISSING on Orin (JetPack 6)                               |
 |   NV01_DEVICE_0   -->  NV_ERR_INVALID_CLASS (34)                                 |
 |   /dev/nvidia-uvm -->  "Module not found"                                        |
 |                                                                                  |
 |   ==========================================================                     |
 |   Both end up doing THE SAME THING at the hardware level:                        |
 |     Write GPFIFO entry --> Update GPPut --> Ring doorbell @ offset 0x90          |
 +----------------------------------------------------------------------------------+
```

### Device File Comparison

```text
  Desktop NVIDIA                          Jetson Orin (JetPack 6)
  ============================            ====================================
  /dev/nvidiactl    (RM API)              /dev/nvgpu/igpu0/ctrl  (GPU control)
  /dev/nvidia0      (GPU device)             + returns AS fd     (Magic 'A')
  /dev/nvidia-uvm   (UVM memory)             + returns TSG fd    (Magic 'T')
                                             + returns chan fd   (Magic 'H')
                                             + /dev/nvmap        (memory ioctls)
                                             + returns dmabuf fd (standard Linux)

  Ioctl magic bytes:                      Ioctl magic bytes:
  'F' = nvidiactl frontend                'G' = ctrl-gpu        (nvgpu-ctrl.h)
  0x2A = nvidia-uvm                       'N' = nvmap           (nvmap.h)
                                          'A' = address space   (nvgpu-as.h)
                                          'H' = channel         (nvgpu.h)
                                          'T' = TSG             (nvgpu.h)
```

### Memory Architecture

```text
  Desktop GPU (discrete):                 Jetson Orin (unified):

  +-------+   PCIe   +-------+           +---------+   +---------+
  |  CPU  |<-------->|  GPU  |           |   CPU   |   |   GPU   |
  |  RAM  |          | VRAM  |           | (Cortex |   | (ga10b) |
  | (DDR) |          |(GDDR) |           |  A78AE) |   |         |
  +-------+          +-------+           +----+----+   +----+----+
  Host mem            Device mem               |             |
  (system)            (on-GPU)                 +------+------+
                                                      |
  cudaMemcpy() =                               +------+------+
    DMA transfer                               | LPDDR5 64GB |
    across PCIe                                | (shared)    |
    (slow, explicit)                           +-------------+

                                               CPU mmap(dmabuf_fd) = same physical pages
                                               GPU MAP_BUFFER_EX   = same physical pages
                                               IO_COHERENCE = no cache flushes needed
                                               cudaMemcpy() = no-op (already shared!)
```

---

## 3. Full IOCTL Pipeline

```text
 +----------------------------------------------------------------------+
 |                  THE IOCTL PIPELINE  (reverse-engineered)            |
 |                  =======================================             |
 |                                                                      |
 |  USER SPACE (Python / tinygrad)         KERNEL (nvgpu.ko + nvmap.ko) |
 |  ==========================            ============================  |
 |                                                                      |
 |  (1) GPU DISCOVERY                                                   |
 |  -----------------                                                   |
 |  open("/dev/nvgpu/igpu0/ctrl") ------> ctrl_fd                       |
 |  open("/dev/nvmap")            ------> nvmap_fd                      |
 |                                                                      |
 |  ioctl(ctrl_fd, GET_CHARACTERISTICS)    +---------------------------- +|
 |       |                            --->| arch=0x0170 (Ampere)       ||
 |       |                                | SM 8.7, compute=0xc7c0     ||
 |       +--- "What GPU do I have?" <---- | gpfifo=0xc76f, dma=0xc7b5  ||
 |                                        +----------------------------+|
 |                                                                      |
 |  (2) MEMORY ALLOCATION                                               |
 |  ----------------------                                              |
 |  ioctl(nvmap_fd, CREATE, {size})  -----> handle (e.g. 0x80000d4c)    |
 |  ioctl(nvmap_fd, ALLOC,  {heap,  -----> physical pages committed     |
 |         flags=INNER_CACHEABLE})          (IOVMM heap, IO coherent)   |
 |  ioctl(nvmap_fd, GET_FD, {handle})-----> dmabuf_fd                   |
 |        |                                                             |
 |        v                                                             |
 |  mmap(dmabuf_fd, size) -------------------> CPU virtual address      |
 |  +---------------------------------------------------------------+   |
 |  |  KEY INSIGHT: mmap the dmabuf fd, NOT /dev/nvmap              |   |
 |  |  Unified memory = CPU and GPU see same physical pages         |   |
 |  |  IO_COHERENCE = no cache flushes needed                       |   |
 |  +---------------------------------------------------------------+   |
 |                                                                      |
 |  (3) GPU ADDRESS SPACE                                               |
 |  ---------------------                                               |
 |  ioctl(ctrl_fd, ALLOC_AS,          +-------------------------------+ |
 |    {UNIFIED_VA,                --->| as_fd (Magic 'A')             | |
 |     start=0x200000,                | 40-bit VA: 0x200000           | |
 |     end=0xFFFFE00000})             |         to 0xFFFFE00000       | |
 |                                    +-------------------------------+ |
 |  NOTE: start/end MUST be PDE-aligned (2MB = 2^21 for ga10b)          |
 |                                                                      |
 |  ioctl(as_fd, MAP_BUFFER_EX,  -----> GPU VA (top-down allocation)    |
 |    {dmabuf_fd, flags, page_size})    Maps dmabuf into GPU VA space   |
 |                                                                      |
 |  (4) CHANNEL SETUP  (6 ioctls, strict order)                         |
 |  ------------------------------------------------                    |
 |                                                                      |
 |  ioctl(ctrl, OPEN_TSG)         -----> tsg_fd     (scheduling group)  |
 |  ioctl(tsg, CREATE_SUBCONTEXT) -----> veid       (virtual engine)    |
 |  ioctl(ctrl, OPEN_CHANNEL)     -----> chan_fd    (command channel)   |
 |  ioctl(as_fd, BIND_CHANNEL)    -----> bind channel to addr space     |
 |  ioctl(tsg_fd, BIND_CHANNEL)   -----> bind channel to TSG            |
 |  ioctl(chan_fd, DISABLE_WDT)   -----> prevent timeout during init    |
 |                                                                      |
 |  IMPORTANT: AS_BIND must come BEFORE TSG_BIND (EINVAL otherwise)     |
 |                                                                      |
 |  +=================================================================+ |
 |  || SETUP_BIND  --  THE KEY IOCTL  (reverse-engineered)           || |
 |  ||                                                               || |
 |  ||  ioctl(chan_fd, SETUP_BIND, {                                 || |
 |  ||      gpfifo_dmabuf_fd,    <-- small dedicated buf (entries*8) || |
 |  ||      gpfifo_dmabuf_offset = 0,  <-- MUST be 0 (kernel check)  || |
 |  ||      userd_dmabuf_fd,     <-- separate 4KB buffer             || |
 |  ||      userd_dmabuf_offset  = 0,  <-- MUST be 0                 || |
 |  ||      num_gpfifo_entries = 1024,                               || |
 |  ||      flags = USERMODE_SUPPORT | DETERMINISTIC  (0x0A)         || |
 |  ||  })                                                           || |
 |  ||       |                                                       || |
 |  ||       v  RETURNS:                                             || |
 |  ||       work_submit_token = 511   <-- doorbell value            || |
 |  ||       gpfifo_gpu_va             <-- GPU sees the ring here    || |
 |  ||       userd_gpu_va              <-- GPU reads GPPut here      || |
 |  ||                                                               || |
 |  ||  Enables USERMODE SUBMIT -- no kernel round-trips             || |
 |  ||  for every GPU command. CUDA uses this, not SUBMIT_GPFIFO     || |
 |  +=================================================================+ |
 |                                                                      |
 |  ioctl(chan_fd, ALLOC_OBJ_CTX,  --> bind compute engine (0xc7c0)     |
 |    {class=AMPERE_COMPUTE_B})        bind DMA engine     (0xc7b5)     |
 |                                                                      |
 |  (5) COMMAND SUBMISSION  (zero ioctls -- pure usermode)              |
 |  --------------------------------------------------                  |
 |                                                                      |
 |  +----------------------------------------------------------------+  |
 |  |                                                                |  |
 |  |  CPU writes push buffer    GPU VA + method headers + QMD       |  |
 |  |       |                                                        |  |
 |  |       v                                                        |  |
 |  |  Write GPFIFO entry        (pushbuf_va | len<<42 | PRIV)       |  |
 |  |  to ring buffer            at gpfifo_area[gp_put * 8]          |  |
 |  |       |                                                        |  |
 |  |       v                                                        |  |
 |  |  Update GPPut              userd[0x8C] = gp_put + 1            |  |
 |  |       |                    (tells GPU: new work at index N)    |  |
 |  |       v                                                        |  |
 |  |  +-----------------------------------------------------------+ |  |
 |  |  |  RING DOORBELL                                            | |  |
 |  |  |  mmap(ctrl_fd, 0x1000) --> doorbell_mm                    | |  |
 |  |  |  doorbell_mm[0x90] = work_submit_token (511)              | |  |
 |  |  |                                                           | |  |
 |  |  |  Same offset 0x90 as desktop NV (BAR0 usermode page)     |  |  |
 |  |  |  Wakes GPU scheduler -> processes GPFIFO -> runs shader   | |  |
 |  |  +-----------------------------------------------------------+ |  |
 |  |       |                                                        |  |
 |  |       v                                                        |  |
 |  |  Poll semaphore            GPU writes completion value         |  |
 |  |  Read output buffer        (IO coherent, visible instantly)    |  |
 |  |                                                                |  |
 |  +----------------------------------------------------------------+  |
 |                                                                      |
 +----------------------------------------------------------------------+
```

---

## 4. Anatomy of SETUP_BIND

This is the single most important ioctl. It enables usermode command submission,
which means zero syscall overhead per GPU dispatch.

```text
  USER SPACE                              KERNEL  (nvgpu.ko)
  ==========                              ==================

  struct {                                linux-channel.c
    u32 num_gpfifo_entries = 1024;           |
    u32 num_inflight_jobs  = 0;              |
    u32 flags = 0x0A; ---------------------->+-- check USERMODE_SUPPORT (0x2)
         |  USERMODE (0x2)                   +-- check DETERMINISTIC    (0x8)
         |  DETERMINISTIC (0x8)              |   (both REQUIRED -- EINVAL if missing)
    s32 gpfifo_dmabuf_fd; ----------------->+-- import dmabuf
    u32 gpfifo_dmabuf_offset = 0; --------->+-- MUST be 0
    s32 userd_dmabuf_fd; ------------------>+-- import dmabuf  |  "TODO - not yet
    u32 userd_dmabuf_offset = 0; ---------->+-- MUST be 0     |   supported"
    s32 work_submit_token; <----------------+-- = channel->userd_iova >> 12
    u64 gpfifo_gpu_va; <-------------------+-- nvgpu_gmmu_map(entire dmabuf)
    u64 userd_gpu_va; <--------------------+-- nvgpu_gmmu_map(userd dmabuf)
  }                                          |
                                             v
                               +------------------------------+
                               |  CRITICAL: kernel maps the   |
                               |  ENTIRE dmabuf into GPU VA   |
                               |                              |
                               |  3MB dmabuf = 3MB of VA gone |
                               |  --> ALLOC_OBJ_CTX can't fit |
                               |  --> GR context alloc fails  |
                               |  --> CE engine halt           |
                               |  --> SYSTEM FREEZE            |
                               |                              |
                               |  FIX: use tiny dedicated     |
                               |  buffers per channel         |
                               |  (8KB ring + 4KB userd)      |
                               +------------------------------+
```

### MAP_FIXED Overlay (how we bridge tinygrad's expectations)

tinygrad expects one contiguous CPU buffer holding all channels' ring + userd data.
nvgpu requires separate small dmabufs per channel with offset=0.
We bridge this with MAP_FIXED:

```text
  tinygrad's view: one big gpfifo_area mmap (64KB)
  +------------------+------------+------------------+------------+------+
  |  Ring (8KB)      | USERD (4KB)|  Ring (8KB)      | USERD (4KB)| pad  |
  |  Channel 0       | Channel 0  |  Channel 1       | Channel 1  |      |
  +------------------+------------+------------------+------------+------+
  offset: 0           8K           12K                20K          24K

  Underneath (what the kernel sees):
  +------------------+            +------------------+
  | ch0_ring dmabuf  |            | ch1_ring dmabuf  |
  | (8KB, offset=0)  |            | (8KB, offset=0)  |
  +------------------+            +------------------+
           +------------+                  +------------+
           | ch0_userd  |                  | ch1_userd  |
           | (4KB)      |                  | (4KB)      |
           +------------+                  +------------+

  MAP_FIXED replaces pages in the big mmap with per-channel dmabuf pages.
  CPU reads/writes go to the same physical pages the GPU sees via SETUP_BIND.
```

---

## 5. Key Takeaways Slide

```text
+------------------------------------------------------------------+
|  WHAT WAS REVERSE-ENGINEERED                                     |
|  ===========================                                     |
|                                                                  |
|  39 ioctls decoded across 5 device file descriptors              |
|  Source: kernel headers + strace of CUDA + nvgpu source code     |
|                                                                  |
|  Key ioctls:                                                     |
|  o GET_CHARACTERISTICS  -- discover GPU (arch, SM, classes)      |
|  o NVMAP CREATE/ALLOC   -- allocate unified memory (no VRAM)     |
|  o MAP_BUFFER_EX        -- give GPU a virtual address for buffer |
|  o SETUP_BIND           -- enable usermode submit                |
|  o ALLOC_OBJ_CTX        -- bind compute/DMA engines to channel   |
|                                                                  |
|  ZERO ioctls for command submission -- pure MMIO doorbell        |
|                                                                  |
|  +-------------------------------------------------------------+ |
|  |  Desktop: nvidia.ko --> RM API --> UVM --> PCIe --> VRAM    | |
|  |  Tegra:   nvgpu.ko + nvmap.ko --> unified DRAM --> doorbell | |
|  |                                                             | |
|  |  Same usermode submit at the end:                           | |
|  |    GPFIFO entry --> GPPut --> doorbell[0x90] --> GPU runs   | |
|  +-------------------------------------------------------------+ |
+------------------------------------------------------------------+
```

### Why SETUP_BIND is the key ioctl

1. It enables **zero-ioctl command submission** (usermode doorbell)
2. CUDA uses it too (discovered via strace -- zero SUBMIT_GPFIFO calls in the trace)
3. It has undocumented kernel constraints (offset must be 0, entire dmabuf mapped)
    that caused system freezes until we read the kernel source
4. The doorbell offset (0x90) is identical to desktop NV -- the hardware convergence point

---

## 6. FAQ / Anticipated Engineering Questions

### Q: Why not just use CUDA on Jetson?

A: CUDA works fine on Jetson. The point is that tinygrad's NV backend bypasses
CUDA entirely for lower overhead and more control. On desktop this already works
via NVKIface/PCIIface. We are extending this to Jetson so the same `NV=1` flag
works everywhere NVIDIA hardware exists.

### Q: What is the actual performance difference?

A: 1024x1024 matmul runs at ~65 GFLOPS on NV=1 vs higher on CUDA+cuDNN. The
NV backend uses tinygrad's generic Ampere codegen, not Orin-optimized kernels.
The value is in the architecture (no CUDA dependency) rather than peak throughput.

### Q: How did you know the doorbell is at offset 0x90?

A: Three converging sources:

- tinygrad desktop NV code writes to `gpu_mmio[0x90 // 4]`
- The nvgpu kernel function `gk20a_ctrl_dev_mmap()` maps `g->usermode_regs_bus_addr`
- NVIDIA open-gpu-kernel-modules defines NV_USERMODE_NOTIFY_CHANNEL_PENDING = 0x90
We confirmed by writing the work_submit_token there and seeing immediate GPU response.

### Q: What is IO_COHERENCE and why does it matter?

A: Jetson Orin has hardware IO coherence between CPU and GPU. When the CPU writes
to a shared buffer, the GPU sees the updated data without any explicit cache flush
or invalidate operations. This is reported by GET_CHARACTERISTICS flag bit 20.
Desktop GPUs do NOT have this because they have separate VRAM across PCIe.
Practically, this means:

- No NVMAP_IOC_CACHE calls needed
- mmap.flush() is irrelevant (and actually returns EINVAL on DMA-BUF fds)
- INNER_CACHEABLE gives 17x faster CPU reads with no coherence penalty

### Q: What is the IOVMM heap?

A: The Jetson memory allocator (nvmap) supports multiple heaps. The IOVMM heap
(bit 30) allocates from the general-purpose DRAM pool via IOMMU virtual memory.
The SYSMEM heap (bit 31) returns ENOMEM. VPR and FSI are security-related heaps.
For compute workloads, IOVMM is the correct choice.

### Q: What are the 40-bit GPU VA space limits?

A: ga10b supports 40-bit GPU virtual addresses (max 0xFFFFFFFFFF = ~1TB).
Desktop Ampere uses 48-bit or larger. This affects shared_mem_window and
local_mem_window addresses -- tinygrad desktop uses 0x729400000000 (too large
for 40-bit). We use 0xFE00000000 and 0xFD00000000 instead.

### Q: Why did a 3MB buffer crash the system?

A: nvgpu's SETUP_BIND calls `nvgpu_gmmu_map()` on the ENTIRE dmabuf. A 3MB
buffer consumes 3MB of the channel's internal GPU VA space. When ALLOC_OBJ_CTX
subsequently tries to allocate the GR (graphics) context, it cannot find enough
contiguous VA space. This triggers a kernel WARNING in `tu104_gr_init_commit_rtv_cb`,
followed by CE (copy engine) halt, which freezes the entire system. The fix is
to use the smallest possible buffers: 8KB for the GPFIFO ring (1024 entries x 8 bytes)
and 4KB for USERD.

### Q: How is usermode submit different from kernel-mode submit?

A: Kernel-mode: each GPU dispatch requires `ioctl(SUBMIT_GPFIFO)`, which means
a syscall (context switch to kernel, validation, copy GPFIFO entry, MMIO write,
context switch back). Usermode: CPU writes the GPFIFO entry directly to mmapped
memory and pokes the doorbell register (also mmapped). Zero syscalls. This is
how both CUDA and tinygrad submit work on production systems.

### Q: What is QMD and why V03?

A: QMD (Queue Meta Data) is a 256-byte descriptor that tells the GPU everything
about a compute dispatch: shader address, grid/block dimensions, register count,
shared memory size, constant buffers. V03 is the Ampere version (qmd_major_version=3,
qmd_version=0). ga10b is Ampere architecture so it uses V03. The QMD format is
identical to desktop Ampere GPUs.

### Q: What structs did you get wrong and how did you find out?

A: Main ones:

- nvmap_alloc_handle: 5th field is numa_nid (s32), not kind (u8). Worked by
    accident because both produce 4 zero bytes. Found by reading the kernel header.
- OPEN_CHANNEL: is a 4-byte union (single s32), not 16 bytes. Wrong size caused
    ENOTTY. Found by checking kernel header more carefully.
- CREATE_SUBCONTEXT: correct layout is (u32, s32, u32, u32), not (u64, u64).
    Found by trial-and-error + kernel source reading.
The methodology: always verify ctypes.sizeof() matches the kernel struct size.

### Q: Could this approach work on other Jetson platforms?

A: The nvgpu + nvmap driver model is shared across all Jetson platforms (TX2, Xavier,
Orin). The specific GPU class numbers, SM versions, and QMD versions differ:

- Xavier: gv11b, Volta, SM 7.2, QMD V02
- Orin: ga10b, Ampere, SM 8.7, QMD V03
- Thor (JetPack 7): will use nvidia-uvm like desktop (different driver model)
The TegraIface architecture should work on Xavier/Orin with class number changes.

### Q: What is the nvgpu object model?

A: nvgpu uses file-descriptor-based objects. Each ioctl that creates a resource
returns a new fd with its own ioctl magic byte:

```text
  ctrl fd (Magic 'G')  -->  ALLOC_AS    -->  as_fd   (Magic 'A')
                        -->  OPEN_TSG    -->  tsg_fd  (Magic 'T')
                        -->  OPEN_CHANNEL --> chan_fd  (Magic 'H')

  nvmap fd (Magic 'N') -->  CREATE/GET_FD -> dmabuf_fd (standard Linux DMA-BUF)
```

This is fundamentally different from desktop nvidia.ko which uses a single
/dev/nvidiactl fd with a handle-based object hierarchy managed by the RM API.

---

## 7. C Hot Path

Production GPU dispatch loop in ~200 lines of C. Python/tinygrad builds the
JIT graph once, exports addresses into a config struct, then C takes over.
Zero ioctls, zero CUDA, zero syscalls in the loop.

### 3 Bullet Summary

1. **Pre-built replay**: tinygrad compiles the ML model into a GPU command
   queue once; C just patches variable values and replays it each iteration
2. **One MMIO write per dispatch**: the entire GPU submit is a single
   volatile write to the doorbell register at offset 0x90 (no syscalls)
3. **46 us end-to-end** for an 18K-param MLP (memcpy in + GPU + memcpy out),
   1.9x faster than PyTorch CUDA Graphs on the same Orin hardware

### Diagram

```text
  Python (one-time setup)                C hot_path.so (every iteration)
  =======================                ===============================

  tinygrad JIT compile                   hot_path_run_iteration(cfg, sensor, action)
       |                                      |
       v                                      v
  HCQGraph export:                   +-----------------------------------+
    - cmdq GPU addr                  |  1. memcpy sensor -> input buf    |  ~0.1 us
    - gpfifo ring addr               |     (unified memory, no DMA)      |
    - userd addr                     |                                   |
    - doorbell mmio addr             |  2. Patch cmdq variables          |  ~0.1 us
    - patch list                     |     (kickoff, timeline values)    |
    - signal addrs                   |                                   |
       |                             |  3. Write GPFIFO entry to ring    |  ~0.5 us
       v                             |     Update GPPut                  |
  hot_path_config_t ---- ctypes ---> |     dmb sy  (ARM barrier)         |
                                     |     doorbell[0x90] = token  <-- THE MMIO WRITE
                                     |                                   |
                                     |  4. Set KICK signal               |  ~0.1 us
                                     |     (unblocks GPU wait)           |
                                     |                                   |
                                     |  5. Spin-wait on timeline signal  |  ~5-25 us
                                     |     (GPU writes on completion,    |
                                     |      IO coherent = instant)       |
                                     |                                   |
                                     |  6. memcpy output buf -> action   |  ~0.1 us
                                     +-----------------------------------+
                                            Total: ~46 us (18K MLP)
```

### How It Comes Together

1. tinygrad JIT-compiles your ML model into a fixed GPU command queue (once)
2. Python exports all raw addresses (ring buffer, doorbell, signals) into a C struct
3. The C .so is loaded via ctypes -- Python calls C directly, no interpreter in the loop
4. Each iteration: C patches semaphore values, writes one GPFIFO entry, pokes the doorbell
5. "Hot path" = the latency-critical loop is pure C + MMIO, not Python + syscalls

### What the C Loop Actually Does (per iteration)

1. **memcpy in**: copies sensor data from a numpy array into the GPU input buffer
   (unified memory -- same physical DRAM, no DMA transfer, just a pointer copy)
2. **patch**: overwrites specific addresses in the pre-built command queue with
   new semaphore/timeline values so the GPU knows which iteration this is.
   "Patch" = the command queue is frozen from the JIT, but a few fields change
   each time (kick signal address, timeline counter). C writes those directly.
3. **doorbell + GPFIFO**: writes one 8-byte GPFIFO entry to the ring buffer,
   updates GPPut, issues an ARM memory barrier (`dmb sy`), then pokes
   doorbell[0x90] -- one volatile MMIO write that wakes the GPU
4. **spin-wait**: polls a timeline semaphore in shared memory until the GPU
   writes a completion value (IO coherent = CPU sees it with zero latency)
5. **memcpy out**: copies the GPU output buffer into the action numpy array

**What is "the graph"?**  When tinygrad JIT-compiles your model, it records every
GPU command (shader launch, memory copy, barrier) into an HCQGraph -- a fixed
sequence of push-buffer entries + QMD descriptors stored in GPU memory. The C
hot path "replays" this graph by patching a few variable fields (semaphore values)
and re-submitting the same command queue. The GPU re-executes the same shaders
on new input data. No recompilation, no driver calls, no Python interpreter.

### How ctypes Cuts Python Out of the Loop

Python is NOT cut out entirely -- it still runs setup and kicks off the loop.
What ctypes does is let Python call a compiled C function as if it were a
Python function, but the actual work runs as native machine code with zero
Python overhead. Here is exactly how it works in bench_hot_path.py:

```text
Step 1: Load the shared library  (bench_hot_path.py line 289)

    lib = ctypes.CDLL("hot_path.so")      # dlopen() under the hood
                                          # lib now has C functions as attributes

Step 2: Declare the C function signatures  (lines 290-296)

    lib.hot_path_init.argtypes = [c_void_p]
    lib.hot_path_benchmark.argtypes = [
        c_void_p, c_void_p, c_void_p,      # config*, sensor*, action*
        c_uint32, POINTER(c_uint64)]        # n_iters, times_ns[]

Step 3: Build config struct with exported GPU addresses  (line 298)

    cfg_struct = _build_config_struct(cfg_dict)
    lib.hot_path_init(byref(cfg_struct))    # precompute GPFIFO entry

Step 4: Call into C -- this is where Python "disappears"  (lines 322-328)

    lib.hot_path_benchmark(
        byref(cfg_struct),                  # pointer to config
        sensor_np.ctypes.data,              # pointer to numpy input
        action_np.ctypes.data,              # pointer to numpy output
        N_ITERS,                            # 10000
        times)                              # C writes timing here

    # What happens inside this ONE call:
    #   C runs a tight loop 10000 times --
    #     for each: patch -> GPFIFO write -> doorbell -> spin-wait
    #   Python interpreter is BLOCKED and IDLE the entire time.
    #   All 10000 GPU dispatches = pure C, zero Python frames.
```

So "no interpreter in the loop" means: Python makes ONE function call into C,
and C runs the entire dispatch loop (memcpy, patch, doorbell, wait) thousands
of times before returning. Python only runs before (setup) and after (read
results). The 46 us per iteration is all C + GPU -- no Python overhead.

---

## 8. Full Pipeline -- Tensor Ops to GPU Warps

End-to-end flow from Python tensor operations to shader execution on GPU SMs.
Covers compilation, scheduling, command building, and hardware dispatch.

```text
 +----------------------------------------------------------------------+
 |  FULL PIPELINE: TENSOR OPS --> GPU WARPS                             |
 |  ==========================================                          |
 |                                                                      |
 |  (1) FRONTEND -- what the user writes                                |
 |  ------------------------------------                                |
 |                                                                      |
 |  Tensor([1,2,3]) + Tensor([4,5,6])                                   |
 |       |                                                              |
 |       v                                                              |
 |  Lazy tensor graph  (each op is a node, not executed yet)            |
 |       |                                                              |
 |       v                                                              |
 |  UOp DAG-AST  (unified micro-op representation)                      |
 |       |        Algebraic rewrites, constant folding,                 |
 |       |        shape inference, dtype promotion                      |
 |       v                                                              |
 |                                                                      |
 |  (2) SCHEDULER -- fuse ops into GPU kernels                          |
 |  ----------------------------------------------                      |
 |                                                                      |
 |  Scheduler groups UOps into fused ExecItems                          |
 |       |   Each ExecItem = one GPU kernel launch                      |
 |       |   Adjacent elementwise ops fused into single kernel          |
 |       |   Reduces memory round-trips (no intermediate buffers)       |
 |       v                                                              |
 |                                                                      |
 |  (3) COMPILER -- per ExecItem, generate GPU binary                   |
 |  -------------------------------------------------                   |
 |                                                                      |
 |  +---------------------------------------------------------------+   |
 |  |  Linearizer                                                   |   |
 |  |    Converts fused UOps into a linear instruction sequence     |   |
 |  |    BEAM search tries multiple loop orderings + unroll factors |   |
 |  |    Picks the fastest variant (autotuned on first run)         |   |
 |  |       |                                                       |   |
 |  |       v                                                       |   |
 |  |  Renderer (NV backend)                                        |   |
 |  |    UOp sequence --> PTX assembly source                       |   |
 |  |    Inserts memory ops, math ops, control flow, barriers       |   |
 |  |       |                                                       |   |
 |  |       v                                                       |   |
 |  |  Compiler (libnvrtc.so, target sm_87)                         |   |
 |  |    PTX source --> CUBIN ELF binary                            |   |
 |  |    (same compiler CUDA uses, just called directly)            |   |
 |  |       |                                                       |   |
 |  |       v                                                       |   |
 |  |  HCQProgram  (cached: binary + register count + shared mem)   |   |
 |  |    Stored in GPU memory, reused across invocations            |   |
 |  +---------------------------------------------------------------+   |
 |       |                                                              |
 |       v                                                              |
 |                                                                      |
 |  (4) COMMAND BUILDING -- assemble GPU instructions                   |
 |  -----------------------------------------------------               |
 |                                                                      |
 |  HCQGraph builds a static HWQueue command sequence:                  |
 |                                                                      |
 |    +---+     +----------------------------+     +--------+           |
 |    |WAI|---->| EXEC                       |---->|SIGNAL  |           |
 |    |T  |     | HCQProgram (binary addr)   |     |(sema   |           |
 |    |   |     | QMD v03 (256 bytes):       |     | write) |           |
 |    |sem|     |   program_address          |     |        |           |
 |    |aph|     |   grid_dim (X, Y, Z)       |     |timeline|           |
 |    |ore|     |   block_dim (threads)      |     |++      |           |
 |    |   |     |   register_count           |     |        |           |
 |    |   |     |   shared_mem_size          |     |        |           |
 |    |   |     |   constant_buf[0] (args)   |     |        |           |
 |    +---+     +----------------------------+     +--------+           |
 |                                                                      |
 |    [WAIT --> EXEC --> SIGNAL]  repeated per kernel in the graph      |
 |                                                                      |
 |    These are push buffer methods (GPU register writes)               |
 |    written to a pre-allocated command queue in GPU-visible memory    |
 |                                                                      |
 |  (5) SUBMISSION -- zero syscalls (usermode path)                     |
 |  -----------------------------------------------                     |
 |                                                                      |
 |  +---------------------------------------------------------------+   |
 |  |                                                               |   |
 |  |  Write GPFIFO entry to ring buffer:                           |   |
 |  |    gpfifo[gp_put] = cmdq_gpu_va | (len << 42) | PRIV          |   |
 |  |                                                               |   |
 |  |  Update GPPut:                                                |   |
 |  |    userd[0x8C] = gp_put + 1                                   |   |
 |  |                                                               |   |
 |  |  ARM memory barrier:                                          |   |
 |  |    dmb sy   (ensure all writes visible before doorbell)       |   |
 |  |                                                               |   |
 |  |  Ring doorbell:                                               |   |
 |  |    doorbell[0x90] = work_submit_token (511)                   |   |
 |  |    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                  |   |
 |  |    ONE volatile MMIO write -- this is the only thing that     |   |
 |  |    actually triggers GPU execution. Everything above is       |   |
 |  |    just writing to shared memory.                             |   |
 |  |                                                               |   |
 |  +---------------------------------------------------------------+   |
 |       |                                                              |
 |       v                                                              |
 |                                                                      |
 |  (6) GPU HARDWARE EXECUTION                                          |
 |  ------------------------------                                      |
 |                                                                      |
 |  GPU Host Interface                                                  |
 |    |  reads GPFIFO ring at gpfifo_gpu_va                             |
 |    |  fetches push buffer (command queue) from GPU memory            |
 |    v                                                                 |
 |  Command Processor (PBDMA)                                           |
 |    |  decodes method headers                                         |
 |    |  processes WAIT (stall until semaphore value met)               |
 |    |  processes EXEC: reads QMD from constant_buf address            |
 |    v                                                                 |
 |  GPC (Graphics Processing Cluster)                                   |
 |    |  loads CUBIN binary from program_address into instruction cache |
 |    |  allocates registers (from QMD register_count)                  |
 |    |  allocates shared memory (from QMD shared_mem_size)             |
 |    v                                                                 |
 |  SM Dispatch (ga10b: 16 SMs, 128 CUDA cores each = 2048 total)       |
 |    |  grid_dim blocks distributed across SMs                         |
 |    |  each block split into 32-thread warps                          |
 |    |  warps execute in lock-step (SIMT)                              |
 |    v                                                                 |
 |  Memory Access                                                       |
 |    |  loads/stores go to LPDDR5 64GB (unified, IO coherent)          |
 |    |  L2 cache (4MB) shared across all SMs                           |
 |    |  no PCIe, no VRAM copy -- direct DRAM access                    |
 |    v                                                                 |
 |  Completion                                                          |
 |    |  processes SIGNAL: writes timeline value to semaphore address   |
 |    |  IO coherent --> CPU sees it immediately (no cache flush)       |
 |    v                                                                 |
 |  CPU polls semaphore --> reads output buffer --> done                |
 |                                                                      |
 +----------------------------------------------------------------------+
```

---

## 9. Compact Pipeline (one-page)

All the reverse-engineered ioctls (nvgpu + nvmap) run once at startup to allocate
memory, create the GPU channel, and enable usermode submit. After that, the entire
steady-state dispatch is pure memory writes + one MMIO doorbell poke — zero syscalls.

```text
  ── STARTUP (ioctls, once) ───────────────────────────────────────────

  Python NV=1 opens /dev/nvgpu/igpu0/ctrl  +  /dev/nvmap
       |
       v
  ~20 reverse-engineered ioctls across 5 fd types
       |  nvgpu + nvmap: discover GPU, allocate memory, create channels,
       |  build address space, enable usermode submit — all at startup
       v
  Channel ready — VA space, doorbell, ring buffers all mapped

  ── COMPILE (Python, once per model) ────────────────────────────────

  Tensor ops
       |
       v
  UOp DAG-AST  (lazy graph of micro-ops)
       |
       v
  Scheduler  (fuse adjacent ops into kernels)
       |
       v
  Linearizer + BEAM search  (pick fastest kernel variant)
       |
       v
  Renderer  (UOps --> PTX assembly)
       |
       v
  Compiler  (libnvrtc.so, sm_87 --> CUBIN ELF)
       |
       v
  HCQProgram  (compiled CUBIN binary stored in GPU memory)

  ── DISPATCH (pure memory writes + MMIO, every iteration) ──────────

  HWQueue builds push buffer in GPU memory:
       |  [WAIT semaphore → EXEC (QMD points to CUBIN) → SIGNAL semaphore]
       |  repeated per kernel in the graph
       |  (with @TinyJit, pre-recorded as HCQGraph for zero-overhead replay)
       |
       v
  Three levels of pointers, all in the same 64 GB unified memory:
       |  GPFIFO entry  →  push buffer  →  QMD (256B)  →  CUBIN binary
       |  (ring slot)      (method hdrs)   (grid, regs)   (shader code)
       |
       v
  GPFIFO entry --> GPPut --> doorbell[0x90]  (ONE MMIO write, zero syscalls)
       |
       v
  GPU Host Interface --> PBDMA --> GPC --> (2048 CUDA Cores + 64 Tensor Cores)
       |
       v
  Warps execute on unified LPDDR5 64GB  (IO coherent, no PCIe)
       |
       v
  Semaphore written --> CPU reads result
```

Same GPU pipeline as Section 9, but Python stops after export. C replays the
push buffer directly — no interpreter, no ioctls, no syscalls in the hot loop.

```text
  ── STARTUP (ioctls, once) ───────────────────────────────────────────

  Python NV=1 opens /dev/nvgpu/igpu0/ctrl  +  /dev/nvmap
       |
       v
  ~20 reverse-engineered ioctls across 5 fd types  (see Section 9a)
       |  nvgpu + nvmap: discover GPU, allocate memory, create channels,
       |  build address space, enable usermode submit — all at startup
       v
  Channel ready — VA space, doorbell, ring buffers all mapped

  ── COMPILE + JIT (Python, once per model) ──────────────────────────

  Tensor ops
       |
       v
  UOp DAG-AST  (lazy graph of micro-ops)
       |
       v
  Scheduler  (fuse adjacent ops into kernels)
       |
       v
  Linearizer + BEAM search  (pick fastest kernel variant)
       |
       v
  Renderer  (UOps --> PTX assembly)
       |
       v
  Compiler  (libnvrtc.so, sm_87 --> CUBIN ELF binary in GPU memory)
       |
       v
  HWQueue builds push buffer in GPU memory:
       |  [WAIT semaphore → EXEC (QMD points to CUBIN) → SIGNAL semaphore]
       |  repeated per kernel in the graph
       |
       v
  @TinyJit warmup  (captures HCQGraph -- frozen snapshot of push buffer)
       |  the push buffer is now a fixed byte sequence in GPU memory
       |  only 3 values change per iteration: kickoff, timeline_wait, timeline_signal
       |
       v
  export_graph.py  (walks the HCQGraph, extracts raw pointers for C):
       |  input/output buffer addrs    (where sensor data / actions live)
       |  GPFIFO ring + gpput addrs    (where to write the ring entry)
       |  doorbell MMIO addr           (the register that wakes the GPU)
       |  push buffer GPU VA + length  (what the GPFIFO entry points to)
       |  patch map: list of [addr, var_type, mask] for each variable slot
       |    (the CPU addrs of the 3 words inside the push buffer that change)
       v
  hot_path_init()  (precompute the 8-byte GPFIFO entry, hand config to C)

  ══ Python exits. C owns everything below. No ioctls, no syscalls. ══

  ── HOT LOOP (C, every iteration) ───────────────────────────────────

  sensor_data --memcpy--> input_buf_addr  (unified LPDDR5, ~0.1 µs)
       |
       v
  apply_patches()  (~0.1 µs)
       |  The push buffer is mostly static (same shaders, same grid dims).
       |  But 3 semaphore counters must increment each iteration so the GPU
       |  knows which dispatch this is (kickoff) and which completion to
       |  wait for / signal (timeline_wait, timeline_signal).
       |  C writes these 3 values directly into the push buffer words
       |  at the addresses from the patch map. Volatile uint32 writes.
       |
       v
  submit_gpfifo()  (~0.5 µs)
       |  Write one 8-byte GPFIFO ring entry:
       |    ring[put] = (push_buffer_GPU_VA | length | flags)
       |  GPFIFO → push buffer → QMD → CUBIN  (three levels of pointers,
       |    all in the same 64 GB unified memory)
       |  Update GPPut index
       |  dmb sy                (ARM barrier -- CPU writes visible to GPU)
       |  mmio[0x90/4] = token  (ONE MMIO doorbell write -- wakes the GPU)
       |
       v
  Reset queue signals + write KICK signal  (~0.1 µs)
       |
       v
  GPU Host Interface --> PBDMA --> GPC --> (2048 CUDA Cores + 64 Tensor Cores)
       |  PBDMA follows the pointer chain:
       |    reads GPFIFO entry → fetches push buffer → reads QMD → launches CUBIN
       |
       v
  Warps execute on unified LPDDR5 64GB  (IO coherent, no PCIe)
       |
       v
  Semaphore written  (GPU writes timeline value to completion address)
       |
       v
  wait_signal()  (C spin-loop: atomic_load + ARM yield, IO coherent, ~5-25 µs)
       |
       v
  output_buf_addr --memcpy--> action_output  (~0.1 µs)
       |
       v
  ──loop──>  next iteration  (no re-compile, no re-schedule, no Python at all)
```

---

## 9a. Reverse-Engineered Ioctls (Reference)

Complete list of ioctls that tinygrad NV=1 calls on Jetson Orin AGX during startup.
Every ioctl runs once (or once per channel / per buffer). Zero ioctls in the hot
path — after setup, the entire dispatch loop is pure MMIO + shared-memory writes.

Source: `tegradev.py` (TegraIface) and `ops_nv.py` (NVDevice.__init__ + _new_gpu_fifo)

```text
  ── IOCTL REFERENCE (call order during NVDevice.__init__) ────────────

  fd type         ioctl                       purpose
  ─────────────────────────────────────────────────────────────────────

  (1) OPEN DEVICES  [TegraIface.__init__]

      -              open("/dev/nvmap")          → nvmap_fd
      -              open("/dev/nvgpu/igpu0/ctrl") → ctrl_fd

  (2) DISCOVER GPU  [TegraIface.__init__]

      ctrl_fd  'G'   GET_CHARACTERISTICS (nr 5)  arch, SM 8.7, class IDs,
                                                  VA bits, max GPFIFO entries

  (3) BUILD GPU ADDRESS SPACE  [rm_alloc → NV01_MEMORY_VIRTUAL]

      ctrl_fd  'G'   ALLOC_AS (nr 8)             create 40-bit GPU VA → as_fd
      as_fd    'A'   AS_ALLOC_SP (nr 6) ×2       reserve VA for shader windows
                                                  (shared_mem @ 0xFE0000_0000,
                                                   local_mem  @ 0xFD0000_0000)

  (4) ALLOCATE MEMORY  [TegraIface.alloc, repeated per buffer]
      (gpfifo_area, notifiers ×2, ring ×2, userd ×2, cmdq_page,
       signal page, kernargs buf, 32 copy bufs = ~40 buffers total)

      nvmap_fd 'N'   NVMAP_CREATE (nr 0)         create nvmap handle
      nvmap_fd 'N'   NVMAP_ALLOC (nr 3)          back handle with IOVMM pages
      nvmap_fd 'N'   NVMAP_GET_FD (nr 15)        export handle → dmabuf fd
      as_fd    'A'   AS_MAP_BUFFER_EX (nr 7)     map dmabuf into GPU VA
      -              mmap(dmabuf_fd)              CPU access to buffer

  (5) CREATE THREAD SCHEDULING GROUP  [rm_alloc → KEPLER_CHANNEL_GROUP_A]

      ctrl_fd  'G'   OPEN_TSG (nr 9)             → tsg_fd

  (6) CREATE ASYNC SUBCONTEXT  [rm_alloc → FERMI_CONTEXT_SHARE_A]

      tsg_fd   'T'   CREATE_SUBCONTEXT (nr 18)   async compute VEID on TSG

  (7) CREATE CHANNELS ×2  [_new_gpu_fifo → rm_alloc → gpfifo_class]
      (compute channel + DMA channel, each gets the full sequence below)

      ctrl_fd  'G'   OPEN_CHANNEL (nr 11)        → ch_fd
      as_fd    'A'   AS_BIND_CH (nr 1)           bind channel to address space
      tsg_fd   'T'   TSG_BIND_CH (nr 11)         bind channel into TSG
      ch_fd    'H'   CH_WDT (nr 119)             disable watchdog timer
      nvmap_fd 'N'   NVMAP_CREATE+ALLOC+GET_FD   ring buffer (GPFIFO entries)
      nvmap_fd 'N'   NVMAP_CREATE+ALLOC+GET_FD   userd (GPPut register page)
      ch_fd    'H'   SETUP_BIND (nr 128)         enable usermode submit → token
      -              mmap MAP_FIXED ×2            overlay ring + userd onto
                                                  gpfifo_area CPU mapping

  (8) BIND ENGINE OBJECTS ×2  [rm_alloc → compute_class / dma_class]

      ch_fd    'H'   ALLOC_OBJ_CTX (nr 108)      compute engine (class 0xc7c0)
      ch_fd    'H'   ALLOC_OBJ_CTX (nr 108)      DMA copy engine (class 0xc7b5)

  (9) MAP DOORBELL  [setup_usermode]

      -              mmap(ctrl_fd, offset=0)      doorbell register page
                                                  (write offset 0x90 = GPU poke)

  ─────────────────────────────────────────────────────────────────────

  5 fd types (Magic byte = ioctl number-space):
    ctrl_fd  (Magic 'G')   /dev/nvgpu/igpu0/ctrl     GPU control
    nvmap_fd (Magic 'N')   /dev/nvmap                 memory allocator
    as_fd    (Magic 'A')   from ALLOC_AS              GPU address space
    tsg_fd   (Magic 'T')   from OPEN_TSG              thread scheduling group
    ch_fd    (Magic 'H')   from OPEN_CHANNEL          GPU channel (compute/DMA)

  15 unique ioctl types, ~20 structural calls + N × (3 nvmap + 1 map) per buffer
  After startup: pure MMIO doorbell + shared-memory writes (zero syscalls)

  NOT ioctls on Tegra (handled differently):
    PERF_BOOST              → writes sysfs /sys/class/devfreq/.../min_freq
    GET_WORK_SUBMIT_TOKEN   → reads cached value from SETUP_BIND result
    GPFIFO_SCHEDULE         → no-op (SETUP_BIND flags handle this)
    GR_GET_INFO             → reads cached GET_CHARACTERISTICS data
```

---

## 9b. Reverse-Engineered Ioctls (Compact)

Same ioctls as Section 9a, condensed to one screen. The goal: open two devices,
get one ioctl to tell us what GPU we have, build a virtual address space, allocate
buffers, create a channel, and flip on usermode submit so we never need ioctls again.

```text
  WHAT                      IOCTL / CALL                     fd   Magic
  ────────────────────────────────────────────────────────────────────────
  open kernel interfaces    open /dev/nvmap                  →N
                            open /dev/nvgpu/igpu0/ctrl       →G

  discover GPU              GET_CHARACTERISTICS (nr 5)       G    'G'
                              → arch, SM version, class IDs, VA bits

  build GPU address space   ALLOC_AS (nr 8)                  G    'G'  →A
                            AS_ALLOC_SP (nr 6) ×2            A    'A'
                              → 40-bit VA space + 2 shader windows

  allocate buffers (×N)     NVMAP_CREATE (nr 0)              N    'N'
                            NVMAP_ALLOC (nr 3)               N    'N'
                            NVMAP_GET_FD (nr 15)             N    'N'
                            AS_MAP_BUFFER_EX (nr 7)          A    'A'
                            mmap(dmabuf_fd)                  -
                              → unified LPDDR5 buf with GPU VA + CPU ptr

  create scheduling group   OPEN_TSG (nr 9)                  G    'G'  →T
                            CREATE_SUBCONTEXT (nr 18)        T    'T'
                              → thread scheduling group + async VEID

  create channel (×2)       OPEN_CHANNEL (nr 11)             G    'G'  →H
                            AS_BIND_CH (nr 1)                A    'A'
                            TSG_BIND_CH (nr 11)              T    'T'
                            CH_WDT (nr 119)                  H    'H'
                            NVMAP_CREATE+ALLOC+GET_FD ×2     N    'N'
                            SETUP_BIND (nr 128)              H    'H'
                            mmap MAP_FIXED ×2                -
                              → channel with ring buf, userd, submit token

  bind engines (×2)         ALLOC_OBJ_CTX (nr 108)           H    'H'
                              → compute 0xc7c0 + DMA 0xc7b5

  map doorbell              mmap(ctrl_fd)                    -
                              → MMIO page, write offset 0x90 to poke GPU
  ────────────────────────────────────────────────────────────────────────
  15 ioctl types  ·  5 fd types (G N A T H)  ·  ~20 structural + N per buf
  After this: zero syscalls — just shared-memory writes + one MMIO doorbell
```

---

## 10. Compact Pipeline with C Hot Path (one-page)

The C hot path eliminates **all Python overhead** from the steady-state dispatch
loop. Python + ~20 reverse-engineered ioctls (see Section 9a) run once at startup
to discover the GPU, allocate unified memory, create the address space, create
channels, and enable usermode submit. Then Python compiles the model, JIT-warms it, and captures the
HCQGraph. `export_graph.py` extracts every raw address (GPU buffers, GPFIFO ring,
doorbell MMIO, push buffer, signal semaphores, and a patch map) into a
`hot_path_config_t` struct. From that point on, **C owns the entire hot loop** —
no Python interpreter, no ctypes marshalling, no GC pauses, no `__call__` overhead,
no ioctls per iteration.

The GPU pointer chain: **GPFIFO → push buffer → QMD → CUBIN**. Three levels of
pointers, all in the same 64 GB unified memory. The GPFIFO ring entry points to
the push buffer. The push buffer contains method headers that point to QMD
descriptors (256 bytes each: grid dims, register count, shared mem). Each QMD
points to a compiled CUBIN shader binary. C replays this entire chain by writing
one 8-byte ring entry and poking the doorbell.

What C does on each iteration:
1. `memcpy` sensor data into the GPU input buffer (unified memory, ~0.1 µs)
2. Patch 3 semaphore counters in the push buffer (kickoff + timeline values, ~0.1 µs)
3. Write one GPFIFO ring entry + poke the doorbell MMIO register (~0.5 µs)
4. Reset queue signals + write the KICK signal to unblock the GPU (~0.1 µs)
5. Spin-wait on the completion semaphore (ARM `yield` loop, ~5-25 µs)
6. `memcpy` action output from the GPU output buffer (~0.1 µs)

Total CPU-side overhead: **< 1 µs** (vs ~1700 µs through Python).

```text
  ┌─────────────────────────────────────────────────────────────────────┐
  │  SETUP PHASE (Python, runs once)                                    │
  │                                                                     │
  │  Tensor ops                                                         │
  │       |                                                             │
  │       v                                                             │
  │  UOp DAG-AST  (lazy graph of micro-ops)                             │
  │       |                                                             │
  │       v                                                             │
  │  Scheduler  (fuse adjacent ops into kernels)                        │
  │       |                                                             │
  │       v                                                             │
  │  Linearizer + BEAM search  (pick fastest kernel variant)            │
  │       |                                                             │
  │       v                                                             │
  │  Renderer  (UOps --> PTX assembly)                                  │
  │       |                                                             │
  │       v                                                             │
  │  Compiler  (libnvrtc.so, sm_87 --> CUBIN ELF)                       │
  │       |                                                             │
  │       v                                                             │
  │  HCQProgram  (compiled CUBIN + launch metadata, in GPU memory)      │
  │       |                                                             │
  │       v                                                             │
  │  @TinyJit warmup  (captures HCQGraph: full command queue snapshot)  │
  │       |                                                             │
  │       v                                                             │
  │  export_graph.py  (extracts raw addrs into hot_path_config_t):      │
  │       ├── input/output buffer CPU addrs  (unified memory)           │
  │       ├── GPFifo ring addr, gpput addr, doorbell MMIO addr          │
  │       ├── command queue GPU VA + length                             │
  │       ├── timeline signal addr, KICK signal addr                    │
  │       └── patch map: [addr, var_type, mask] for each variable slot  │
  │                                                                     │
  └──────────────────────────┬──────────────────────────────────────────┘
                             │  config struct handed to C via ctypes
                             v
  ┌─────────────────────────────────────────────────────────────────────┐
  │  HOT LOOP (C, no Python interpreter, no syscalls, no ioctls)        │
  │                                                                     │
  │  sensor_data ──memcpy──> input_buf_addr  (unified LPDDR5, ~0.1 µs) │
  │       |                                                             │
  │       v                                                             │
  │  apply_patches()  (~0.1 µs)                                        │
  │       |   Push buffer is mostly static. Only 3 semaphore counters   │
  │       |   change per iteration (kickoff, timeline_wait, timeline_   │
  │       |   signal). C writes them directly at the patch map addrs.   │
  │       v                                                             │
  │  submit_gpfifo()  (~0.5 µs)                                        │
  │       |   ring[put] = GPFIFO entry (points to push buffer GPU VA)   │
  │       |   GPFIFO → push buffer → QMD → CUBIN (pointer chain)        │
  │       |   gpput[0]  = (put+1) % entries                             │
  │       |   dmb sy                     (ARM barrier: CPU→GPU visible) │
  │       |   mmio[0x90/4] = token       (ONE doorbell write)           │
  │       v                                                             │
  │  reset queue signals + write KICK signal  (~0.1 µs)                 │
  │       |                                                             │
  │       v                                                             │
  │  ┌───────────────────────────────────────────────────────────────┐  │
  │  │  GPU executes (triggered by doorbell)                         │  │
  │  │  PBDMA reads push buffer → GPC dispatches warps               │  │
  │  │  2048 CUDA Cores + 64 Tensor Cores process kernels            │  │
  │  │  Results written to output buf (unified LPDDR5, IO coherent)  │  │
  │  │  SIGNAL: writes timeline value to semaphore                   │  │
  │  └───────────────────────────────────────────────────────────────┘  │
  │       |                                                             │
  │       v                                                             │
  │  wait_signal()  (spin on semaphore with ARM yield, IO coherent)     │
  │       |                                                             │
  │       v                                                             │
  │  output_buf_addr ──memcpy──> action_output  (~0.1 µs)              │
  │       |                                                             │
  │       v                                                             │
  │  return cycle_time_ns  ──loop──>  next iteration                    │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘

  Key: Python overhead eliminated
  ─────────────────────────────────────────────────────────────
  ✗ No Python interpreter in hot loop
  ✗ No ctypes marshalling per iteration (C loops directly)
  ✗ No GC pauses or reference counting
  ✗ No __call__ / schedule / linearize per iteration
  ✗ No ioctls, no CUDA runtime, no driver calls
  ─────────────────────────────────────────────────────────────
  ✓ Ioctls run once at startup (nvmap + nvgpu), pure MMIO after that
  ✓ Same GPU commands as tinygrad NV=1 HCQGraph replay
  ✓ GPFIFO → push buffer → QMD → CUBIN (pointer chain in unified memory)
  ✓ C does: memcpy → patch 3 words → ring write → doorbell poke → spin-wait
  ✓ Total CPU dispatch overhead: < 1 µs per iteration
```

---

## 11. NixOS: Declarative Hardware Enablement

The kernel already has every driver we need (`inv_mpu6050`, `i2c-dev`, `uvcvideo`).
The hard part is the **wiring**: loading modules, setting permissions, binding
devices, creating groups, installing tools. Ubuntu makes you do it by hand.
NixOS makes it declarative.

### The Ubuntu Way (6 steps, 6 files, 0 rollback)

```bash
# 1. Load kernel modules (and figure out which ones you need)
sudo modprobe i2c-dev
sudo modprobe inv-mpu6050-i2c
sudo modprobe industrialio

# 2. Persist across reboots
echo "i2c-dev"         | sudo tee    /etc/modules-load.d/i2c.conf
echo "inv-mpu6050-i2c" | sudo tee -a /etc/modules-load.d/i2c.conf
echo "industrialio"    | sudo tee -a /etc/modules-load.d/i2c.conf

# 3. Hand-craft udev rules
cat <<'EOF' | sudo tee /etc/udev/rules.d/99-imu.rules
SUBSYSTEM=="iio",         GROUP="video", MODE="0660"
SUBSYSTEM=="i2c-dev",     GROUP="i2c",   MODE="0660"
SUBSYSTEM=="video4linux", GROUP="video", MODE="0660"
EOF
sudo udevadm control --reload

# 4. Create groups, add users
sudo groupadd i2c
sudo usermod -aG i2c,video agent
sudo usermod -aG i2c,video spencer
# (log out, log back in)

# 5. Write a systemd unit to bind the IMU at boot
cat <<'EOF' | sudo tee /etc/systemd/system/mpu9250-bind.service
[Unit]
Description=Bind MPU-9250 to inv_mpu6050
After=systemd-modules-load.service
[Service]
Type=oneshot
RemainAfterExit=true
ExecStart=/bin/sh -c 'echo "inv_mpu6050 0x68" > /sys/bus/i2c/devices/i2c-7/new_device'
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable mpu9250-bind

# 6. Install userspace tools
sudo apt install i2c-tools v4l-utils

# Files touched: /etc/modules-load.d/i2c.conf
#                /etc/udev/rules.d/99-imu.rules
#                /etc/systemd/system/mpu9250-bind.service
#                /etc/group, /etc/passwd, apt state
# Second board? Do it all again. From memory.
# apt upgrade breaks it? Good luck rolling back.
```

### The NixOS Way (1 file, 1 command, instant rollback)

```nix
# sensors.nix — the ENTIRE sensor stack
services.orin-sensors = {
  enable      = true;          # ← one switch
  imuI2cBus   = 7;
  imuI2cAddr  = "0x68";
  cameraUsers = [ "agent" "spencer" ];
};
```

```bash
sudo nixos-rebuild switch --flake .#nixos-sensors
# ✓ Kernel modules loaded   (i2c-dev, inv-mpu6050-i2c, industrialio, uvcvideo)
# ✓ udev rules applied      (IIO, I2C, V4L2 permissions)
# ✓ Groups created           (i2c)
# ✓ Users added to groups    (agent, spencer → video + i2c)
# ✓ MPU-9250 bound at boot   (systemd oneshot)
# ✓ Tools installed           (i2c-tools, v4l-utils)
```

Broke something? Roll back in 10 seconds:

```bash
sudo nixos-rebuild switch --rollback
```

### Verify (3 commands)

```bash
i2cdetect -y 7                              # 0x68 = MPU-9250 responding
cat /sys/bus/iio/devices/iio:device*/name    # inv_mpu6050 + ak8963
v4l2-ctl --list-devices                      # USB stereo camera
```

### Composability — stack modules like LEGO

```nix
# flake.nix — same pattern, different stacks
nixosConfigurations = {
  nixos-sensors     = base ++ [ performance  sensors       ];
  nixos-llama-cpp   = base ++ [ performance  llama-cpp     ];
  nixos-docker-bench = base ++ [ performance  docker-nvidia ];
  nixos-telemetry   = base ++ [ telemetry                  ];
};
# Every config:   sudo nixos-rebuild switch --flake .#<name>
# Every rollback: sudo nixos-rebuild switch --rollback
```


