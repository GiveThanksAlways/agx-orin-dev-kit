# TegraIface: The nvgpu/nvmap Backend

This is the largest change — **766 lines added** to `ops_nv.py`. It implements
a complete GPU command submission backend for Jetson's `nvgpu.ko` kernel driver,
replacing the desktop-only `NVKIface`/`PCIIface` paths.

## Table of Contents

1. [Why a New Backend?](#why-a-new-backend)
2. [The ioctl Interface](#the-ioctl-interface)
3. [ctypes Structs](#ctypes-structs)
4. [Memory Allocation Pipeline](#memory-allocation-pipeline)
5. [Channel & TSG Setup](#channel--tsg-setup)
6. [VA Space Management](#va-space-management)
7. [RM Translation Layer](#rm-translation-layer)
8. [Doorbell & Usermode](#doorbell--usermode)

---

## Why a New Backend?

Tinygrad's NV backend talks to NVIDIA GPUs by issuing **ioctl** system calls to
kernel device files. On desktop Linux, the driver is `nvidia.ko` (proprietary),
and the API uses NVIDIA's **Resource Manager (RM)** — a complex object-oriented
interface with `rm_alloc()`, `rm_control()`, `rm_free()`.

On Jetson, the driver is `nvgpu.ko` (open-source, mainline-adjacent). It has a
**completely different** ioctl interface:

```
Desktop (nvidia.ko)                 Jetson (nvgpu.ko)
─────────────────                   ────────────────
/dev/nvidia0                        /dev/nvgpu/igpu0/ctrl
/dev/nvidia-uvm                     /dev/nvmap
NV_ESC_RM_ALLOC                     NVGPU_GPU_IOCTL_*
NV_ESC_RM_CONTROL                   NVGPU_AS_IOCTL_*
RM object hierarchy                 Flat ioctl per operation
PCIe BAR for MMIO                   Memory-mapped ctrl fd
```

**TegraIface** translates tinygrad's RM-style calls into nvgpu equivalents,
so the rest of tinygrad's NV backend (queue submission, program loading, etc.)
works unchanged.

---

## The ioctl Interface

### What's an ioctl?

An ioctl ("input/output control") is a Linux system call for device-specific
operations that don't fit into read/write:

```python
import fcntl
result = fcntl.ioctl(fd, ioctl_number, data_buffer)
```

The `ioctl_number` encodes:
- **Direction** (bits 30-31): read, write, both, or neither
- **Size** (bits 16-29): size of the data struct
- **Type** (bits 8-15): driver identifier character (e.g., 'G' for GPU)
- **Number** (bits 0-7): command index within the driver

### Our ioctl Number Constructors

```python
# Linux ioctl direction bits (aarch64)
_IOC_NONE  = 0; _IOC_WRITE = 1; _IOC_READ  = 2

def _tegra_IOC(d, t, nr, size):
    return (d << 30) | (size << 16) | (ord(t) << 8) | nr

def _tegra_IO(t, nr):    return _tegra_IOC(_IOC_NONE, t, nr, 0)
def _tegra_IOR(t, nr, sz):  return _tegra_IOC(_IOC_READ, t, nr, sz)
def _tegra_IOW(t, nr, sz):  return _tegra_IOC(_IOC_WRITE, t, nr, sz)
def _tegra_IOWR(t, nr, sz): return _tegra_IOC(_IOC_READ | _IOC_WRITE, t, nr, sz)
```

**Example**: `_NVGPU_GPU_IOCTL_GET_CHARACTERISTICS` uses type `'G'` (GPU),
number 5, direction read+write, size = `sizeof(_nvgpu_gpu_get_characteristics)`.

```python
_NVGPU_GPU_IOCTL_GET_CHARACTERISTICS = _tegra_IOWR('G', 5, 16)
# = (3 << 30) | (16 << 16) | (ord('G') << 8) | 5
# = 0xC010_4705
```

The type characters map to subsystems:
- `'G'` = GPU control (`/dev/nvgpu/igpu0/ctrl`)
- `'N'` = nvmap (memory management, `/dev/nvmap`)
- `'A'` = Address space (GPU virtual memory)
- `'H'` = channel (Host1X legacy name)
- `'T'` = TSG (Timeslice Group)

---

## ctypes Structs

Each ioctl passes a C struct as its data buffer. We define these in Python
using `ctypes.Structure`:

### GPU Characteristics (most important one)

```python
class _nvgpu_gpu_characteristics(ctypes.Structure):
    _fields_ = [
        ("arch", ctypes.c_uint32),        # 0x170 for Ampere/ga10b
        ("impl", ctypes.c_uint32),        # implementation revision
        ("rev", ctypes.c_uint32),
        ("num_gpc", ctypes.c_uint32),     # Graphics Processing Clusters
        ...
        ("compute_class", ctypes.c_uint32),  # 0xc7c0 = AMPERE_COMPUTE_B
        ("gpfifo_class", ctypes.c_uint32),   # 0xc76f
        ("dma_copy_class", ctypes.c_uint32), # 0xc7b5 = AMPERE_DMA_COPY_B
        ...
        ("sm_arch_sm_version", ctypes.c_uint32),  # 0x807 = SM 8.7
        ("gpu_va_bit_count", ctypes.c_uint8),     # 40 (not 48!)
        ...
    ]
```

This struct tells us everything about the GPU: architecture, SM version, number
of compute units, and most importantly the **class numbers** for compute/DMA.

**Why SM version matters**: SM 8.7 = ga10b. The first digit (8) is the major
architecture (Ampere), 7 is the minor variant. Tinygrad uses this to select
the correct PTX ISA version and instruction set.

### Memory Management Structs

```python
class _nvmap_create_handle(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint32), ("handle", ctypes.c_uint32)]
    # Double duty: GET_FD returns dmabuf fd in the "size" field

class _nvmap_alloc_handle(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),      # from CREATE
        ("heap_mask", ctypes.c_uint32),    # NVMAP_HEAP_IOVMM = SMMU-managed
        ("flags", ctypes.c_uint32),        # cache policy + tag
        ("align", ctypes.c_uint32),        # alignment requirement
        ("numa_nid", ctypes.c_int32),      # NUMA node (0 on Orin)
    ]
```

And for GPU mapping:

```python
class _nvgpu_as_map_buffer_ex_args(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),        # 0 = let kernel pick VA
        ("compr_kind", ctypes.c_int16),    # -1 = no compression
        ("incompr_kind", ctypes.c_int16),  # 0
        ("dmabuf_fd", ctypes.c_uint32),    # from GET_FD
        ("page_size", ctypes.c_uint32),    # 4096 on ga10b
        ("buffer_offset", ctypes.c_uint64),
        ("mapping_size", ctypes.c_uint64), # 0 = whole buffer
        ("offset", ctypes.c_uint64),       # output: GPU VA
    ]
```

---

## Memory Allocation Pipeline

This is the core of TegraIface — how we allocate GPU-visible memory. It's a
5-step pipeline:

```
 CREATE → ALLOC → GET_FD → MAP_BUFFER_EX → mmap
   │        │        │          │              │
   │        │        │          │              └─ CPU virtual address
   │        │        │          └─ GPU virtual address
   │        │        └─ dmabuf file descriptor
   │        └─ Physical pages allocated
   └─ Handle created (kernel tracking)
```

### Step 1: nvmap CREATE

```python
create = _nvmap_create_handle()
create.size = size
_tegra_ioctl(self._nvmap_fd, _NVMAP_IOC_CREATE, create)
handle = create.handle  # kernel returns a handle
```

This just creates a handle — no physical memory yet.

### Step 2: nvmap ALLOC

```python
alloc_args = _nvmap_alloc_handle()
alloc_args.handle = handle
alloc_args.heap_mask = _NVMAP_HEAP_IOVMM     # SMMU-managed memory
alloc_args.flags = (_NVMAP_TAG_TINYGRAD << 16) | cache_flags
alloc_args.align = alloc_align
_tegra_ioctl(self._nvmap_fd, _NVMAP_IOC_ALLOC, alloc_args)
```

Now the kernel allocates physical pages. Key choices:

- **`_NVMAP_HEAP_IOVMM`** (bit 30): Memory managed by the SMMU (IOMMU). On
  Orin, all GPU memory goes through the SMMU for address translation.

- **`_NVMAP_TAG_TINYGRAD`** (0x0900): A tag in bits [31:16] of flags. This
  identifies the subsystem to the kernel, silencing `nvmap_alloc_handle WARNING`
  messages in dmesg.

- **Cache flags**: Two options:
  - `WRITE_COMBINE` (1): For GPU-only buffers (GPFIFO, USERD). CPU writes are
    coalesced but reads are slow.
  - `INNER_CACHEABLE` (2): For compute buffers. CPU can read/write at cache
    speed. IO-coherent SMMU ensures GPU sees the latest data.

### Step 3: GET_FD (dmabuf)

```python
get_fd = _nvmap_create_handle()
get_fd.handle = handle
_tegra_ioctl(self._nvmap_fd, _NVMAP_IOC_GET_FD, get_fd)
dmabuf_fd = get_fd.size  # fd reused in the .size field (quirk)
```

This converts the nvmap handle to a **dmabuf** file descriptor. A dmabuf
(DMA buffer) is a Linux kernel abstraction for sharing memory between drivers.
The nvgpu GPU driver accepts dmabuf fds for mapping into GPU VA space.

### Step 4: MAP_BUFFER_EX (GPU VA)

```python
map_args = _nvgpu_as_map_buffer_ex_args()
map_args.flags = 0              # kernel picks VA
map_args.compr_kind = -1        # no compression
map_args.dmabuf_fd = dmabuf_fd
map_args.page_size = 4096       # always 4KB on ga10b
_tegra_ioctl(self._as_fd, _NVGPU_AS_IOCTL_MAP_BUFFER_EX, map_args)
gpu_va = map_args.offset        # kernel returns GPU VA
```

Now the buffer has a **GPU virtual address**. The GPU's MMU page table maps
this VA to the physical pages allocated in step 2.

### Step 5: mmap (CPU VA)

```python
addr = libc_so.mmap(
    ct.c_void_p(gpu_va),        # hint: MAP_FIXED at GPU VA
    size,
    mmap.PROT_READ | mmap.PROT_WRITE,
    mmap.MAP_SHARED | MAP_FIXED,
    dmabuf_fd,                   # map the same dmabuf
    0
)
```

**This is the magic of unified memory**: We `mmap` the dmabuf fd at the *same*
virtual address as the GPU VA using `MAP_FIXED`. This means:

```
va_addr = 0x100200000  (both GPU VA and CPU pointer)

GPU: Load from 0x100200000 → physical page X (via GPU MMU)
CPU: Load from 0x100200000 → physical page X (via SMMU + CPU MMU)
```

The same `va_addr` value works as both a CPU pointer (for `ctypes.memmove`) and
a GPU address (for kernel launch arguments). This is why tinygrad's `HCQBuffer`
puts the VA in `va_addr` and uses it for everything.

**Fallback**: If `MAP_FIXED` at `gpu_va` fails (address already taken), we
fall back to a kernel-chosen address. This means GPU VA ≠ CPU VA, but
tinygrad handles this via the `view` property.

### Alignment Strategy

```python
# Small/host buffers: 4KB aligned
alloc_align = mmap.PAGESIZE
# Large device buffers (≥8MB): 2MB aligned for SMMU TLB efficiency
if size >= (8 << 20): alloc_align = 2 << 20
```

The SMMU translates GPU virtual addresses to physical addresses using page
tables, with a TLB (Translation Lookaside Buffer) cache. 2MB alignment
for large buffers means fewer TLB entries needed → fewer TLB misses →
better memory bandwidth.

---

## Channel & TSG Setup

### What's a Channel?

A GPU **channel** is like a CPU thread — it has its own command stream (GPFIFO)
and can execute GPU kernels independently. On NVIDIA GPUs, a channel consists of:

1. **GPFIFO**: Ring buffer of pointers to pushbuffers (GPU command lists)
2. **USERD**: User-mode doorbell region (to wake the GPU)
3. **Pushbuffer**: Actual GPU commands (compute launches, memory copies, etc.)

### What's a TSG?

A **TSG (Timeslice Group)** is a scheduling unit that groups channels together.
The GPU scheduler switches between TSGs, giving each group a timeslice. For
tinygrad, we put all channels in one TSG:

```
TSG
├── Compute Channel (kernel launches)
└── DMA Channel (memory copies)
```

### The Setup Sequence

On desktop (NVKIface), channel setup uses `rm_alloc()` with RM class objects.
On Tegra, we map each RM class to the equivalent nvgpu ioctl:

```
RM Class                    → nvgpu ioctl
────────────────────          ──────────────
NV01_DEVICE_0               → (NOP — no equivalent)
NV01_MEMORY_VIRTUAL         → ALLOC_AS (create address space)
KEPLER_CHANNEL_GROUP_A      → OPEN_TSG
FERMI_CONTEXT_SHARE_A       → TSG.CREATE_SUBCONTEXT
AMPERE_CHANNEL_GPFIFO_A     → OPEN_CHANNEL + AS_BIND + TSG_BIND + SETUP_BIND
AMPERE_COMPUTE_B            → CHANNEL.ALLOC_OBJ_CTX
AMPERE_DMA_COPY_B           → CHANNEL.ALLOC_OBJ_CTX
```

### GPFIFO Channel Setup (the complex one)

When tinygrad calls `rm_alloc(gpfifo_class)`, TegraIface does:

```python
# 1. Open a raw channel
ch_args = _nvgpu_gpu_open_channel_args()
_tegra_ioctl(self._ctrl_fd, _NVGPU_GPU_IOCTL_OPEN_CHANNEL, ch_args)
ch_fd = ch_args.channel_fd

# 2. Bind channel to address space (must be before TSG bind!)
as_bind = _nvgpu_as_bind_channel_args()
as_bind.channel_fd = ch_fd
_tegra_ioctl(self._as_fd, _NVGPU_AS_IOCTL_BIND_CHANNEL, as_bind)

# 3. Bind channel to TSG with subcontext
tsg_bind = _nvgpu_tsg_bind_channel_ex_args()
tsg_bind.channel_fd = ch_fd
tsg_bind.subcontext_id = self._subctx_veid
_tegra_ioctl(self._tsg_fd, _NVGPU_TSG_IOCTL_BIND_CHANNEL_EX, tsg_bind)

# 4. Disable watchdog (prevents timeout for long-running kernels)
wdt = _nvgpu_channel_wdt_args()
wdt.wdt_status = 1  # disable
_tegra_ioctl(ch_fd, _NVGPU_IOCTL_CHANNEL_WDT, wdt)

# 5. SETUP_BIND: finalize with GPFIFO ring buffer + USERD doorbell
setup = _nvgpu_channel_setup_bind_args()
setup.num_gpfifo_entries = 1024
setup.gpfifo_dmabuf_fd = gpfifo_ring_dmabuf_fd
setup.userd_dmabuf_fd = userd_dmabuf_fd
setup.flags = USERMODE_SUPPORT | DETERMINISTIC
_tegra_ioctl(ch_fd, _NVGPU_IOCTL_CHANNEL_SETUP_BIND, setup)
```

### The Small-Buffer Constraint

This was a hard-won lesson. The nvgpu kernel's `SETUP_BIND` handler
(`linux-channel.c`) has three strict constraints:

1. `gpfifo_dmabuf_offset` **MUST** be 0 (non-zero → `-EINVAL`)
2. `userd_dmabuf_offset` **MUST** be 0 (non-zero → `-EINVAL`)
3. The kernel maps the **ENTIRE** dmabuf into GPU VA

Tinygrad's NVDevice allocates a large `gpfifo_area` (3MB on desktop) containing
both the GPFIFO ring and USERD region for all channels. But on Tegra, passing
the full 3MB dmabuf to SETUP_BIND would:
- Map 3MB into GPU VA for each channel (redundant)
- Fragment the 40-bit VA space
- Crash `ALLOC_OBJ_CTX` (GR context allocation fails with fragmented VA)

**Solution**: Allocate **small dedicated buffers** per channel:

```python
# 8KB for GPFIFO ring (1024 entries × 8 bytes)
gpfifo_ring_size = gpfifo_entries * 8  # 8192

# 4KB for USERD (doorbell region)
USERD_SIZE = 4096
```

Then use `MAP_FIXED` overlays to replace the corresponding pages in
`gpfifo_area`'s mmap with the small per-channel dmabufs:

```python
# Overlay gpfifo ring: replace gpfifo_area pages with per-channel ring
libc_so.mmap(gpfifo_area_cpu + offset, ring_size,
             PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_FIXED,
             gpfifo_ring_dmabuf_fd, 0)
```

This way, tinygrad's `cpu_view()` reads from `gpfifo_area`'s mmap (unchanged API),
but the physical backing is now the small per-channel dmabufs that match what the
GPU sees.

### NVDevice Changes

The gpfifo sizing is adjusted for Tegra:

```python
# Desktop: 3MB gpfifo_area, 64K entries per channel
self.gpfifo_area = self.iface.alloc(0x300000, ...)  # desktop

# Tegra: 64KB gpfifo_area, 1024 entries per channel
self.gpfifo_area = self.iface.alloc(0x10000 if self.is_tegra() else 0x300000, ...)

if self.is_tegra():
    tegra_entries = 1024
    self.compute_gpfifo = self._new_gpu_fifo(... offset=0, entries=1024 ...)
    self.dma_gpfifo = self._new_gpu_fifo(... offset=1024*8 + 0x1000, entries=1024 ...)
```

Why 1024 entries? Each GPFIFO entry points to a pushbuffer that can contain
multiple GPU commands. 1024 entries is plenty for tinygrad's usage pattern
(one kernel at a time, sequential submission).

---

## VA Space Management

### The 40-bit Problem

Desktop GPUs have 48-bit VA (256 TB). Jetson's ga10b has **40-bit VA** (1 TB).
This is much smaller, and special GPU hardware memory windows eat into it:

```
0x0000000000 ┌──────────────────┐
             │  User buffers     │ ← normal allocations
             │  (via MAP_BUFFER) │
0xFD00000000 ├──────────────────┤
             │ local_mem_window  │ ← hardware-intercepted
             │ (1GB reserved)    │
0xFE00000000 ├──────────────────┤
             │ shared_mem_window │ ← hardware-intercepted
             │ (1GB reserved)    │
0xFFFFFFFFFF └──────────────────┘
```

### Window Addresses

The GPU has **memory windows** at fixed VA ranges. When a shader accesses
`shared` memory (fast on-chip SRAM) or `local` memory (per-thread scratch),
the hardware intercepts loads/stores to these VA ranges and redirects them
to physical SRAM or spill RAM.

Desktop puts windows at `0x729400000000` and `0x729300000000` (well above
any normal allocation in 48-bit VA space). On Tegra, we use:

```python
if self.is_tegra():
    self.shared_mem_window = 0xFE00000000  # near top of 40-bit space
    self.local_mem_window  = 0xFD00000000
else:
    self.shared_mem_window = 0x729400000000
    self.local_mem_window  = 0x729300000000
```

### The VA Collision Fix (P0)

Without explicit reservation, the kernel's VA allocator can place user buffers
at these addresses, causing a **GPU fault** when the hardware tries to intercept
the access but finds a normal buffer instead. We fix this during address space
creation (`NV01_MEMORY_VIRTUAL`):

```python
# Reserve 1GB at each window address
for window_va in [0xFD00000000, 0xFE00000000]:
    rsv = _nvgpu_as_alloc_space_args()
    rsv.pages = 0x40000000 // mmap.PAGESIZE  # 1GB / 4KB = 262144 pages
    rsv.page_size = mmap.PAGESIZE
    rsv.flags = 0x1  # FIXED_OFFSET
    rsv.offset = window_va
    _tegra_ioctl(self._as_fd, _NVGPU_AS_IOCTL_ALLOC_SPACE, rsv)
```

This "wastes" 2GB of the 1TB VA space, but prevents random crashes during
LLaMA inference (which allocates many large buffers that can push the VA
allocator into the window regions).

---

## RM Translation Layer

### rm_alloc() Dispatch

The `rm_alloc()` method is a big switch statement mapping RM class numbers to
nvgpu operations:

| RM Class | nvgpu Action |
|---|---|
| `NV01_DEVICE_0` | NOP (hierarchy container) |
| `NV20_SUBDEVICE_0` | NOP (hierarchy container) |
| `NV01_MEMORY_VIRTUAL` | `ALLOC_AS` + reserve window VAs |
| `FERMI_VASPACE_A` | NOP (AS already created) |
| `KEPLER_CHANNEL_GROUP_A` | `OPEN_TSG` |
| `FERMI_CONTEXT_SHARE_A` | `TSG.CREATE_SUBCONTEXT` |
| `AMPERE_CHANNEL_GPFIFO_A` | Full channel setup (6 ioctls) |
| `AMPERE_COMPUTE_B` | `CHANNEL.ALLOC_OBJ_CTX` |
| `AMPERE_DMA_COPY_B` | `CHANNEL.ALLOC_OBJ_CTX` |
| `GT200_DEBUGGER` | NOP (skip — no Tegra debug support) |
| Everything else | Stub + warning |

### rm_control() Dispatch

Similarly, `rm_control()` translates control commands:

| RM Control | Tegra Action |
|---|---|
| `NV2080_CTRL_CMD_PERF_BOOST` | Write max freq to devfreq sysfs |
| `NVA06C_CTRL_CMD_GPFIFO_SCHEDULE` | NOP (auto-scheduled) |
| `NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN` | Return saved token |
| `NV2080_CTRL_CMD_GR_GET_INFO` | Map from characteristics |
| `NV0080_CTRL_CMD_GPU_GET_CLASSLIST` | Return known classes |
| `NV2080_CTRL_CMD_GPU_GET_GID_INFO` | Synthetic UUID ("J" for Jetson) |
| `NV2080_CTRL_CMD_FB_FLUSH_GPU_CACHE` | NOP (IO-coherent) |

**PERF_BOOST** is fun — on desktop, this talks to the RM driver to boost GPU
clocks. On Tegra, we just write to the Linux devfreq sysfs:

```python
max_freq = open("/sys/class/devfreq/17000000.gpu/max_freq").read()
open("/sys/class/devfreq/17000000.gpu/min_freq", "w").write(max_freq)
```

This pins the GPU to max frequency, which matters for benchmarking (default
governor would downclock during pauses between kernel launches).

---

## Doorbell & Usermode

The CPU tells the GPU "you have new work" by writing to a **doorbell** register.
On desktop, this is a PCIe BAR (Base Address Register) — a memory-mapped region
of the GPU's physical registers.

On Tegra, the doorbell is accessed by mmapping the GPU control fd:

```python
def setup_usermode(self):
    addr = libc_so.mmap(None, 0x10000,
                        PROT_READ | PROT_WRITE, MAP_SHARED,
                        self._ctrl_fd,  # /dev/nvgpu/igpu0/ctrl
                        0)
    return 0, MMIOInterface(addr, 0x10000, fmt='I')
```

The doorbell write location is at offset 0x90 from this mmap base. When
tinygrad writes the GPFIFO pointer and rings this doorbell, the GPU wakes up
and starts processing commands.

### The Submission Hot Path

```
CPU:                                              GPU:
1. Write pushbuffer (compute dispatch)
2. Write GPFIFO entry (ptr + length)
3. Update GPPUT register
4. Memory barrier (DSB on ARM)
5. Write doorbell (MMIO at offset 0x90)  ──→  GPU wakes, reads GPFIFO
                                               Reads pushbuffer
                                               Launches kernel
                                               Writes signal ──→  6. CPU polls signal
```

This is the critical path for latency. On Tegra, the MMIO doorbell is *very*
fast (nanoseconds, not microseconds like PCIe round-trips), which is actually
what causes the QMD race condition discussed in the next document.
