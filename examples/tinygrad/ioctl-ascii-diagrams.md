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

```text
  NV=1 python3 -c "print((Tensor([1,2,3]) + Tensor([4,5,6])).numpy())"

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
  HCQProgram  (cached binary + metadata, NV=1 below)
       |
       v
  HCQGraph  (WAIT --> EXEC+QMD --> SIGNAL, per kernel)
       |
       v
  Push buffer written to GPU memory  (method headers + QMD)
       |
       v
  GPFIFO entry --> GPPut --> doorbell[0x90]  (ONE MMIO write)
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
