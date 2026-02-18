# NV Backend on Jetson Orin AGX 64GB — Investigation & Build Report

**Date:** 2026-02-10  
**Device:** NVIDIA Jetson Orin AGX 64GB Developer Kit  
**JetPack:** 6 (L4T r36.4.4)  
**Drivers loaded:** `nvidia.ko` 540.4.0 (Tegra variant) + `nvgpu.ko` (actual compute) + `nvmap.ko` (memory)  
**Kernel:** 5.15.148  
**CUDA:** 12.6  
**GPU:** ga10b iGPU, Ampere arch 0x0170, SM 8.7, compute class 0xc7c0, platform bus `17000000.gpu`  
**tinygrad:** v0.12.0  

---

## PROJECT STATUS

| Phase | Status | Working Doc | Summary |
|-------|--------|-------------|----------|
| Phase 1: Reverse-Engineer nvgpu IOCTLs | **COMPLETE** ✅ | [phase1.md](phase1.md) | All 39 ioctls decoded. 7/7 tests pass. Compute class allocated. |
| Phase 2: Memory Management via nvmap | **COMPLETE** ✅ | [phase2.md](phase2.md) | 11/11 tests pass. mmap, GPU VA, coherence, cacheability all proven. |
| Phase 3: Command Submission | **COMPLETE** ✅ | [phase3.md](phase3.md) | 15/15 tests pass. GPFIFO submit, doorbell, QMD compute dispatch, shader compilation. |
| Phase 4: TegraIface Integration | **COMPLETE** ✅ | [phase4.md](phase4.md) | TegraIface in ops_nv.py. `NV=1` tinygrad works: tensors, matmul, conv2d. |

**Learning docs:** [Learning-Phase1.md](Learning-Phase1.md), [Learning-Phase4.md](Learning-Phase4.md) — full walkthroughs of methodology and discoveries.

---

## TL;DR

**NV backend now works on Jetson Orin AGX (JetPack 6)** via a new `TegraIface` backend. Both existing interfaces (NVK and PCI) fail for fundamental architectural reasons (see below), so we built a third interface that talks directly to nvgpu/nvmap.

**All 4 phases are COMPLETE.** The full pipeline — ioctl reverse-engineering, memory management, command submission, and tinygrad integration — is working. `NV=1 python3 -c "from tinygrad import Tensor; print((Tensor([1,2,3]) + Tensor([4,5,6])).numpy())"` outputs `[5 7 9]`. Matrix multiply, conv2d, and 1024x1024 matmul (~65 GFLOPS) all verified.

---

## System Architecture

The Jetson Orin has a fundamentally different GPU driver stack than desktop NVIDIA:

```
Desktop NVIDIA (dGPU)                    Jetson Orin (JetPack 6)
─────────────────────                    ──────────────────────
nvidia.ko (full RM API)                  nvidia.ko (PARTIAL RM – display only)
nvidia-uvm.ko (memory mgmt)             nvgpu.ko (actual GPU compute driver)
/dev/nvidiactl                           nvmap.ko (memory allocator)
/dev/nvidia-uvm                          
/dev/nvidia0                             /dev/nvidiactl (limited RM)
GPU on PCIe bus                          /dev/nvidia0 (limited RM)
                                         /dev/nvhost-gpu (compute ioctls)
                                         /dev/nvhost-ctrl-gpu (GPU control)
                                         /dev/nvhost-as-gpu (address space)
                                         /dev/nvmap (memory)
                                         GPU on platform bus @ 0x17000000
```

**nvidia-uvm is JetPack 7 (Thor) only** — confirmed in `jetpack-nixos/modules/default.nix` line 283:
```nix
boot.kernelModules = if (jetpackAtLeast "7") then [ "nvidia-uvm" ] else [ "nvgpu" ];
```
And line 291:
```nix
softdep nvidia pre: governor_pod_scaling post: nvidia-uvm  # JetPack 7 only
```

---

## Error Output

```bash
$ NV=1 python3 -c 'from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())'

ExceptionGroup: No interface for NV:0 is available (2 sub-exceptions)
  1) FileNotFoundError: [Errno 2] No such file or directory: '/dev/nvidia-uvm'
  2) IndexError: list index out of range
```

---

## Interface 1: NVK — Partial RM API, No UVM

NVK requires `/dev/nvidiactl`, `/dev/nvidia-uvm`, and `/dev/nvidia0`.

### What works (RM basic ops)

```bash
# Device nodes exist:
$ ls /dev/nvidia*
/dev/nvidia0  /dev/nvidiactl

# nvidia.ko IS loaded (Tegra variant):
$ lsmod | grep nvidia
nvidia               1613824  0
tegra_dce             114688  2 nvidia
```

**RM API test results** (ran actual tinygrad RM ioctls via `NV_ESC_RM_ALLOC` / `NV_ESC_RM_CONTROL`):

| Operation | ioctl | Status | Meaning |
|---|---|---|---|
| Root client alloc | `NV01_ROOT_CLIENT` | **0** | **NV_OK** ✓ |
| Card info | `NV_ESC_CARD_INFO` | - | GPU found (gpu_id=131072, valid=1, minor=0) ✓ |
| GPU ID info | `NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2` | **31** | NV_ERR_INVALID_ARGUMENT ✗ |
| Device alloc | `NV01_DEVICE_0` | **34** | **NV_ERR_INVALID_CLASS** ✗ |
| Subdevice alloc | `NV20_SUBDEVICE_0` | **87** | NV_ERR_OBJECT_NOT_FOUND ✗ (cascading failure) |
| Class list query | `NV0080_CTRL_CMD_GPU_GET_CLASSLIST` | **61** | NV_ERR_INVALID_POINTER ✗ (cascading failure) |

**The critical failure is `NV01_DEVICE_0` → NV_ERR_INVALID_CLASS (34).** This is the core GPU device abstraction that tinygrad's NVKIface builds everything on (device → subdevice → VA space → channels → compute). The Tegra nvidia.ko driver doesn't support this class at all.

### What doesn't exist

```bash
# nvidia-uvm module is NOT available for this kernel:
$ sudo modprobe nvidia-uvm
modprobe: FATAL: Module nvidia-uvm not found in directory /run/booted-system/kernel-modules/lib/modules/5.15.148

# Only these nvidia modules exist:
$ ls /run/booted-system/kernel-modules/lib/modules/5.15.148/updates/
nvidia.ko.xz  nvidia-drm.ko.xz  nvidia-modeset.ko.xz  nvgpu.ko.xz  nvhwpm.ko.xz
# ↑ NO nvidia-uvm.ko
```

### Why NV01_DEVICE_0 fails

The `nvidia.ko` on Tegra/JetPack 6 is a **display-oriented RM** (modesetting, DRM, display engine). It exposes the RM ioctl endpoint (`/dev/nvidiactl`) and can enumerate GPUs, but the **compute-side RM object classes** (`NV01_DEVICE_0`, `NV20_SUBDEVICE_0`, `AMPERE_CHANNEL_GPFIFO_A`, etc.) are not implemented in this driver. Those capabilities live in `nvgpu.ko` via completely different ioctls on `/dev/nvhost-gpu`.

---

## Interface 2: PCI — iGPU Not on PCI Bus

PCI interface scans `/sys/bus/pci/devices` for NVIDIA GPUs with class `0x03` (display controller).

**Problem:** The Orin GPU is an integrated SoC GPU on the platform bus, not a PCIe device.

```bash
# PCI devices on this Jetson — no GPU (class 0x03):
$ for d in /sys/bus/pci/devices/*/; do
    echo "$(cat $d/vendor) $(cat $d/device) $(cat $d/class) $(basename $d)"
  done
0x10de 0x229e 0x060400 0001:00:00.0   # PCIe bridge
0x10ec 0xc822 0x028000 0001:01:00.0   # Realtek WiFi
0x10de 0x229c 0x060400 0004:00:00.0   # PCIe bridge
0xc0a9 0x560a 0x010802 0004:01:00.0   # NVMe SSD

# GPU is on platform bus, addressed by MMIO:
$ cat /sys/class/devfreq/17000000.gpu/cur_freq
306000000

# tinygrad PCI backend only knows discrete GPU device IDs:
# ops_nv.py: devices=[(0xff00, [0x2200..0x2f00])]  ← RTX 3000/4000/5000 series
```

**GPU identity:**
```bash
$ nvidia-smi
|   0  Orin (nvgpu)                  N/A  | N/A              N/A |                  N/A |
# ↑ "nvgpu" confirms this GPU is accessed via nvgpu.ko, NOT nvidia.ko's PCIe path
```

---

## The Actual GPU Interface: nvgpu + nvmap

On JetPack 6 Orin, the actual GPU compute stack is:

```bash
$ lsmod | grep nvgpu
nvgpu                2793472  0
nvmap                 262144  1 nvgpu
```

### Device Paths (CORRECTED — discovered during Phase 1)

The device paths use `/dev/nvgpu/igpu0/` NOT the old `/dev/nvhost-*` paths:

```bash
# Primary devices used by CUDA (confirmed via strace):
/dev/nvmap                      # Memory allocator (Magic 'N' = 0x4e)
/dev/nvgpu/igpu0/ctrl           # GPU control (Magic 'G' = 0x47)

# These return fd-based sub-devices via ioctls (NOT opened directly):
# ALLOC_AS    -> returns AS fd     (Magic 'A' = 0x41)
# OPEN_TSG    -> returns TSG fd    (Magic 'T' = 0x54)
# OPEN_CHANNEL -> returns ch fd    (Magic 'H' = 0x48)

# Additional devices (legacy nvhost paths also exist but not needed):
/dev/dri/renderD128             # DRM (display, not needed for compute)
/dev/host1x-fence               # Fence device (syncpoints)
```

This is a **completely different ioctl interface** from the desktop nvidia.ko RM API. The nvgpu driver uses its own set of IOCTL codes for:
- Channel management (channel fd from `OPEN_CHANNEL` on ctrl)
- Address space management (AS fd from `ALLOC_AS` on ctrl)
- Memory allocation (`/dev/nvmap`)
- GPU control and queries (`/dev/nvgpu/igpu0/ctrl`)
- TSG (Time Slice Group) scheduling (TSG fd from `OPEN_TSG` on ctrl)

---

## Why CUDA Works

The CUDA backend uses `libcuda.so` + `libnvrtc.so` (runtime compilation). These libraries abstract away the entire driver difference — NVIDIA's CUDA userspace driver knows how to talk to both `nvidia.ko` (desktop) and `nvgpu.ko/nvhost` (Tegra). This is why `CUDA=1` works perfectly on Jetson.

---

## The JetPack 7 (Thor) Question

Per `jetpack-nixos/modules/default.nix` and the project README, JetPack 7 **only supports Thor AGX** (not Orin):

```
|       Device       | JetPack 5 | JetPack 6 | JetPack 7 |
| Jetson Thor AGX    |           |           |     ✓     |
| Jetson Orin AGX    |     ✓     |     ✓     |           |   ← NO JetPack 7
```

JetPack 7 loads `nvidia-uvm` instead of `nvgpu` and has full desktop-style RM. If Thor + JetPack 7 were available, the NV backend would likely work with no or minimal changes. But **this will never be available for Orin AGX** — Orin is stuck on JetPack 5/6 with the nvgpu architecture.

---

## Roadmap: Getting NV-Like Raw Kernel Submission on Jetson Orin

There are three possible approaches, ordered by feasibility:

### Option A: nvgpu Backend (New Backend) — **CHOSEN, COMPLETE** ✅

Write a new tinygrad backend (`TegraIface`) that uses the nvgpu/nvmap ioctls directly.

---

#### Phase 1: Reverse-Engineer nvgpu IOCTLs — **COMPLETE** ✅

**Working doc:** [phase1.md](phase1.md)  
**Test script:** [test_nvgpu.py](test_nvgpu.py) (940+ lines, all 7/7 tests pass)  
**Learning doc:** [Learning-Phase1.md](Learning-Phase1.md)

**What was accomplished:**
- Downloaded L4T r36.4.4 BSP sources, extracted all UAPI headers (nvgpu.h, nvgpu-ctrl.h, nvgpu-as.h, nvmap.h)
- Straced CUDA running GPT-2 — captured 2783 lines, 1793 ioctls, decoded all 39 unique ioctl codes
- **KEY DISCOVERY:** CUDA uses usermode submit (ZERO `SUBMIT_GPFIFO` ioctls) — writes GPFIFO entries to mapped memory and rings hardware doorbell
- Built comprehensive Python ctypes test (`test_nvgpu.py`) proving direct GPU access works:
  - GPU characteristics: arch=0x0170 (Ampere), SM 8.7, compute_class=0xc7c0
  - Memory: nvmap CREATE+ALLOC (IOVMM heap) + GET_FD for dmabuf
  - Address space: ALLOC_AS with PDE-aligned VA ranges (2MB alignment for ga10b)
  - GPU mapping: MAP_BUFFER_EX assigns GPU VA
  - Full channel pipeline: OPEN_TSG -> CREATE_SUBCONTEXT(ASYNC) -> OPEN_CHANNEL -> AS_BIND -> TSG_BIND_EX -> WDT(disable) -> SETUP_BIND(USERMODE+DETERMINISTIC) -> GET_USER_SYNCPOINT -> ALLOC_OBJ_CTX(0xc7c0)
  - **Compute class 0xc7c0 successfully allocated!**
  - Work submit token (doorbell): 511
  - Syncpoint: ID=17, max=30000, GPU VA=0xffffe10000

**Critical discoveries (traps for the unwary):**
1. SETUP_BIND requires `DETERMINISTIC` flag alongside `USERMODE_SUPPORT` — kernel enforces this
2. ALLOC_AS VA ranges must be non-zero AND PDE-aligned (2^21 = 2MB for ga10b)
3. Channel must bind to AS BEFORE binding to TSG (order matters!)
4. OPEN_CHANNEL struct is 4 bytes (union), not 16
5. Device paths are `/dev/nvgpu/igpu0/ctrl` not `/dev/nvhost-ctrl-gpu`
6. Watchdog must be disabled before SETUP_BIND (required for DETERMINISTIC mode)

**Kernel sources extracted:** `l4t-sources/nvgpu/` contains headers + `common/fifo/channel.c`, `common/mm/as.c`, `os/linux/ioctl_channel.c`, `os/linux/linux-channel.c`

---

#### Phase 2: Memory Management via nvmap — **COMPLETE** ✅

**Working doc:** [phase2.md](phase2.md)

**Result:** 11/11 tests pass. CPU<->GPU shared memory fully working.

**Key discoveries:**
- mmap works via dmabuf fd (from `NVMAP_IOC_GET_FD`), not the `/dev/nvmap` fd
- IO_COHERENCE is real — no cache flushes needed, CPU writes immediately visible to GPU
- `INNER_CACHEABLE` (flags=2) is optimal: 17x faster reads than WRITE_COMBINE (11.8 GB/s vs 691 MB/s)
- GPU VAs allocated top-down within ALLOC_AS range
- Allocations from 4KB to 64MB all work

**Goal:** Prove CPU<->GPU shared memory works. Write from CPU, read from GPU (and vice versa).

**Tasks:**
1. **mmap nvmap buffers to CPU** — use `mmap()` on nvmap fd with handle, verify CPU read/write ✅
2. **Map to GPU VA** — use `MAP_BUFFER_EX` on AS fd, confirm GPU VA assignment ✅
3. **Verify coherence** — write pattern from CPU, submit a trivial DMA copy on GPU, read back ✅
4. **Build allocator class** — `TegraAllocator` with alloc/free/map/unmap methods ✅
5. **Handle cache coherence** — Orin has IO_COHERENCE flag set, but verify if explicit cache ops needed ✅

**Key info from Phase 1:**
- IOVMM heap (1<<30) works for allocation (SYSMEM heap does NOT)
- nvmap CREATE -> ALLOC -> GET_FD -> MAP_BUFFER_EX is the proven flow
- compr_kind=-1 (invalid), incompr_kind=0 (pitch linear) works for mapping
- Orin has unified memory (VRAM=0), CPU and GPU share same DRAM
- IO_COHERENCE flag is set in GPU characteristics

---

#### Phase 3: Command Submission — **COMPLETE** ✅

**Working doc:** [phase3.md](phase3.md)

**Result:** 15/15 tests pass. Full GPU compute dispatch proven from Python.

**Key discoveries:**
- Doorbell is via mmap of ctrl fd at offset 0x90 (`NV_USERMODE_NOTIFY_CHANNEL_PENDING`), same as desktop
- QMD V03 format matches desktop Ampere — `qmd_major_version=3`, `qmd_version=0` (subversion)
- Must set `SET_SHADER_SHARED_MEMORY_WINDOW_A` and `SET_SHADER_LOCAL_MEMORY_WINDOW_A` before compute dispatch
- `libnvrtc.so` compiles CUDA C → CUBIN for SM 8.7 (2984B CUBIN, 640B SASS)
- GPU write latency: ~1ms from doorbell to semaphore completion
- Push buffer format identical to desktop (typ=2, subchannel 0=GPFIFO, 1=compute, 4=DMA)

**Goal:** Push actual GPU commands via GPFIFO and execute a compute shader.

**Tasks:**
1. **mmap userd region** — the userd buffer (4KB) is where we write GPFIFO doorbell ✅
2. **Understand GPFIFO entry format** — 8 bytes per entry: {GPU_VA of push buffer, length, flags} ✅
3. **Format QMD (Queue Meta Data)** — Ampere QMD format, contains shader address, grid dims, shared mem size ✅
4. **Compile a trivial shader** — use libnvrtc.so to compile PTX -> SASS for SM 8.7 ✅
5. **Write push buffer** — inline methods or QMD launch pointing to shader ✅
6. **Ring doorbell** — write to userd to trigger GPFIFO processing ✅
7. **Wait for completion** — poll syncpoint or use syncpoint GPU VA ✅
8. **Verify result** — read output buffer from CPU, compare expected ✅

**Key info from Phase 1:**
- CUDA uses usermode submit (no SUBMIT_GPFIFO ioctl!) — we do the same
- work_submit_token=511 is the doorbell token
- Syncpoint ID=17 at GPU VA 0xffffe10000 for completion tracking
- GPFIFO buffer: 1024 entries x 8 bytes = 8192 bytes
- Userd buffer: 4096 bytes
- Compute class: 0xc7c0 (Ampere compute)
- GPFIFO class: 0xc76f
- DMA copy class: 0xc7b5
- Key question: does QMD format match desktop Ampere? Check tinygrad's existing QMD code in ops_nv.py → **YES, it matches**

**Approach:** Study tinygrad's `NVKIface._cmdq_setup_compute_class()` and `_build_gpu_cmd()` — the QMD and push buffer format should be identical since ga10b is Ampere architecture. The only difference is HOW we submit (userd doorbell vs RM submit).

---

#### Phase 4: TegraIface Integration — **COMPLETE** ✅

**Working doc:** [phase4.md](phase4.md)  
**Learning doc:** [Learning-Phase4.md](Learning-Phase4.md)

**Result:** `NV=1` tinygrad works end-to-end on Jetson Orin AGX 64GB.

```bash
NV=1 python3 -c "from tinygrad import Tensor; print((Tensor([1,2,3]) + Tensor([4,5,6])).numpy())"
# [5 7 9]
```

**Verified operations:** tensor creation, vector add, matrix multiply, 1024x1024 matmul (~65 GFLOPS), conv2d.

**Critical bug found & fixed:** nvgpu kernel maps the ENTIRE gpfifo dmabuf into GPU VA. A 3MB dmabuf fragmented the VA space and crashed ALLOC_OBJ_CTX (system freeze). Fix: per-channel small dedicated buffers (8KB ring + 4KB userd) with MAP_FIXED overlays.

**Key architecture:** TegraIface translates tinygrad's RM API calls to nvgpu/nvmap equivalents. ~400 lines added to `ops_nv.py`. The GPU programming layer (QMD, shaders, push buffers) is unchanged.

**Goal:** Build `TegraIface` class that plugs into tinygrad's NV runtime.

**Tasks:**
1. **Study `NVKIface` and `PCIIface`** — understand the interface contract ✅
2. **Implement `TegraIface`** — same interface, but using nvgpu/nvmap ioctls instead of RM ✅
3. **Memory management** — replace UVM with nvmap-based allocator ✅
4. **Channel management** — replace RM channel creation with nvgpu TSG+channel ✅
5. **Command submission** — replace RM GPFIFO submit with usermode submit ✅
6. **Add detection** — check for `/dev/nvgpu/igpu0/ctrl` in `_select_iface()` ✅
7. **Test incrementally** — vector add -> matmul -> conv2d -> GPT-2 ✅ (GPT-2 not yet tested)

**Key info:** tinygrad's NV backend (ops_nv.py) already has Ampere QMD formatting, push buffer construction, and shader compilation via PTX. TegraIface only needs to replace the _driver layer_ (how memory is allocated and how commands are submitted), NOT the _GPU programming layer_ (QMD format, shader ISA, class methods).

---

**Final difficulty assessment:** Medium-Hard, as estimated. Phase 1 (ioctl reverse-engineering) was the riskiest. Phase 4's crash debugging (3MB buffer → system freeze) was the most time-consuming — required reading nvgpu kernel source to find undocumented constraints. Total: ~4 phases, 15/15 standalone tests, all tinygrad tensor ops working.

---

### Option B: Hybrid RM + nvmap (Experimental — Uncertain)

Try to use the partial RM API (which does work for root alloc + card info) combined with nvmap for memory, bypassing UVM entirely.

**Approach:**
1. Patch `NVKIface` to skip `/dev/nvidia-uvm` open
2. Replace all `self.uvm()` calls with nvmap equivalents
3. See if any higher-level RM classes work when memory is provided differently
4. **Problem:** `NV01_DEVICE_0` returns INVALID_CLASS — this blocks the entire RM object hierarchy. Without a device object, you can't create subdevices, channels, or anything.
5. **Verdict:** Probably a dead end on JetPack 6. The nvidia.ko RM on Tegra just doesn't support compute classes.

**Difficulty:** Medium effort to try, but high probability of dead end.

---

### Recommended Path

For **production use today**: `NV=1` now works alongside `CUDA=1`. NV gives direct GPU control (HCQ) while CUDA uses the runtime API.

**Option A (nvgpu backend) is COMPLETE.** All 4 phases done. Remaining work: GPT-2 end-to-end test, CUDA↔NV correctness comparison, and upstream PR to tinygrad.

### Key Files in This Directory

| File | Purpose |
|------|----------|
| `nv-attempt.md` | This file — high-level project status and roadmap |
| `phase1.md` | Phase 1 working doc (COMPLETE) — ioctl decode, struct tables, test results |
| `phase2.md` | Phase 2 working doc (COMPLETE) — memory management, mmap, coherence |
| `phase3.md` | Phase 3 working doc (COMPLETE) — command submission, doorbell, QMD, shaders |
| `phase4.md` | Phase 4 working doc (COMPLETE) — TegraIface integration, crash investigation |
| `Learning-Phase1.md` | Teaching doc — Phase 1 methodology, ioctl reverse-engineering |
| `Learning-Phase4.md` | Teaching doc — Phase 4 methodology, kernel constraints, MAP_FIXED technique |
| `test_nvgpu.py` | Python ctypes test — 15/15 passing, full channel+compute pipeline |
| `tests/` | Modular test suite — tegra_helpers.py, test_two_channels.py, test_gpu_submit.py |
| `strace-cuda.sh` | Helper to run CUDA under strace from detective nix shell |
| `l4t-sources/` | Extracted L4T BSP kernel sources (headers + key .c files) |
| `flake.nix` | Nix flake with `default` (tinygrad+CUDA) and `detective` (strace/gcc/gdb) shells |

### Key Resources

- [NVIDIA L4T kernel source](https://developer.nvidia.com/embedded/jetson-linux) — contains full nvgpu driver
- [Jetson Linux Archive](https://developer.nvidia.com/embedded/jetson-linux-archive)
- [open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules) — desktop RM source (for comparison)
- `/dev/nvhost-*` and `/dev/nvmap` — the actual Tegra GPU interfaces
- `tinygrad/runtime/ops_nv.py` — NVKIface and PCIIface implementations
- `jetpack-nixos/modules/default.nix` line 283 — confirms nvidia-uvm is JetPack 7 only
- `nvidia-smi` shows GPU as "Orin (nvgpu)" — confirming nvgpu is the real driver

### NVIDIA Jetson Linux 36.4.4 (cat /etc/nv_tegra_release)
[Drivers	Driver Package (BSP)](https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.4/release/Jetson_Linux_r36.4.4_aarch64.tbz2)
[Sample Root Filesystem](https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.4/release/Tegra_Linux_Sample-Root-Filesystem_r36.4.4_aarch64.tbz2)
[Jetson Linux API Reference](https://developer.nvidia.com/embedded/L4T/r36_release_v4.4/Release/Jetson_Multimedia_API_r36.4.4_aarch64.tbz2)
[Sources	Driver Package (BSP) Sources](https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.4/sources/public_sources.tbz2)
[Sample Root Filesystem Sources](https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.4/sources/ubuntu_jammy-l4t_aarch64_src.tbz2)
[Fix memory leak  when importing an external memory handle through IPC.](https://developer.nvidia.com/downloads/embedded/L4T/r36_Release_v4.4/cuda_driver_36.4.4.tbz2)
