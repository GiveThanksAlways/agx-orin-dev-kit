# Phase 1: Reverse-Engineer nvgpu IOCTLs — Working Document

**Goal:** Map the complete ioctl interface for `/dev/nvgpu/igpu0/ctrl`, `/dev/nvmap`, and associated fd-based devices so we can build a tinygrad `TegraIface`.

## TODO Checklist

- [x] Download & extract L4T BSP kernel sources (public_sources.tbz2)
- [x] Find nvgpu ioctl header files (struct definitions, ioctl codes)
- [x] Find nvmap ioctl header files
- [x] Strace a CUDA program to capture real ioctl sequence
- [x] Decode strace output — map ioctl numbers to names
- [x] Document the ioctl flow: init → alloc memory → create channel → submit work
- [x] Write Python ctypes structs for key ioctls
- [x] Test basic ioctls from Python (GPU characteristics, memory alloc)
- [x] Test ALLOC_AS, MAP_BUFFER_EX, TSG, channel, SETUP_BIND
- [x] **ALL 7/7 TESTS PASS — compute class allocated!**
- [ ] Build minimal `TegraIface` prototype

## Key Finding: CUDA Uses Usermode Submit on Jetson!

**There are ZERO `SUBMIT_GPFIFO` ioctls (H:107) in the strace.** CUDA uses usermode submit
on Jetson — it writes GPFIFO entries directly to mapped memory and rings the hardware
doorbell through MMIO, just like tinygrad does on desktop NV via `NVKIface`. This is
enabled by `NVGPU_IOCTL_CHANNEL_SETUP_BIND` (H:128) with `USERMODE_SUPPORT` flag.

## Test Results: 7/7 PASS

```
Test 1: GET_CHARACTERISTICS      PASS  — arch=0x0170 (Ampere), compute=0xc7c0, SM 8.7
Test 2: ZCULL_GET_CTX_SIZE       PASS  — 164352 bytes
Test 3: NVMAP GET_AVAILABLE_HEAPS PASS — VPR + FSI available (IOVMM works for alloc)
Test 4: NVMAP CREATE + ALLOC     PASS  — IOVMM heap, dmabuf fd obtained
Test 5: ALLOC_AS                 PASS  — VA range 0x200000 - 0xFFFFE00000, UNIFIED_VA
Test 6: MAP_BUFFER_EX            PASS  — GPU VA 0xffffa00000
Test 7: FULL CHANNEL SETUP       PASS  — TSG->subctx->channel->AS_bind->TSG_bind->WDT->SETUP_BIND->syncpoint->ALLOC_OBJ_CTX
```

### Channel setup outputs:
- **Work submit token (doorbell):** 511
- **Syncpoint ID:** 17, max=30000, GPU VA=0xffffe10000
- **Compute class:** 0xc7c0 allocated successfully

## Critical Discoveries During Testing

### 1. SETUP_BIND requires DETERMINISTIC flag
The kernel (`channel.c:1531`) enforces:
```c
if ((args->flags & USERMODE_SUPPORT) != 0U &&
    (args->flags & SUPPORT_DETERMINISTIC) == 0U) {
    nvgpu_err(g, "need deterministic for usermode submit");
    return -EINVAL;
}
```
**Fix:** flags must be `USERMODE_SUPPORT | DETERMINISTIC` = `(1<<3) | (1<<1)` = 0x0A

### 2. ALLOC_AS requires PDE-aligned VA ranges
`va_range_start` and `va_range_end` MUST be non-zero AND aligned to PDE size (2^21 = 2MB for ga10b).
Setting them to 0 causes EINVAL. Working params:
- `big_page_size=0, flags=UNIFIED_VA(2), start=0x200000, end=0xFFFFE00000, split=0`

### 3. Channel binding order matters
CUDA binds channel to AS (`A:1`) **BEFORE** binding to TSG (`T:11`). Reversing this causes EINVAL.

### 4. OPEN_CHANNEL is a 4-byte union
Not 16 bytes — it's a union of `{in: runlist_id}` and `{out: channel_fd}`, just one `s32`.

### 5. CREATE_SUBCONTEXT struct
Must be: `type(u32)=ASYNC(1), as_fd(s32), veid(u32, out), reserved(u32)`. Total 16 bytes.

### 6. GPU_MAP_RESOURCES_SUPPORT not available
GPU flags bit 57 (SUPPORT_GPU_MMIO) is NOT set on ga10b. GPFIFO/userd/MMIO GPU VAs
are not returned to userspace but are mapped internally by the kernel.

## Device Files Used by CUDA

```
fd=3  /dev/nvmap              — Memory allocator (nvmap)         Magic: 'N' (0x4e)
fd=4  /dev/nvgpu/igpu0/ctrl   — GPU control (ctrl-gpu)           Magic: 'G' (0x47)
fd=5  /dev/dri/renderD128     — DRM device                       (DRM ioctls)
fd=6  /dev/host1x-fence       -> then reused as AS fd            Magic: 'X'->'A' (0x41)
fd=7  /dev/dri/renderD128     — DRM (second open)
fd=8  (TSG fd from OPEN_TSG)                                     Magic: 'T' (0x54)
fd=9+ (channel fds from OPEN_CHANNEL)                            Magic: 'H' (0x48)
```

## Complete IOCTL Decode

### Magic -> Device Mapping

| Magic | Char | Device | Header |
|-------|------|--------|--------|
| 0x47 | 'G' | /dev/nvgpu/igpu0/ctrl | nvgpu-ctrl.h |
| 0x4e | 'N' | /dev/nvmap | nvmap.h |
| 0x41 | 'A' | address space (from ALLOC_AS) | nvgpu-as.h |
| 0x48 | 'H' | channel (from OPEN_CHANNEL) | nvgpu.h |
| 0x54 | 'T' | TSG (from OPEN_TSG) | nvgpu.h |
| 0x58 | 'X' | /dev/host1x-fence | upstream kernel |

### Ctrl-GPU Ioctls (Magic 'G' = 0x47)

| Nr | Hex | Name | Count | Notes |
|----|-----|------|-------|-------|
| 1 | 0x01 | ZCULL_GET_CTX_SIZE | 2 | ZCull context size |
| 2 | 0x02 | ZCULL_GET_INFO | 2 | ZCull geometry info |
| 5 | 0x05 | GET_CHARACTERISTICS | 2 | **KEY: returns arch, SM version, compute_class, gpfifo_class** |
| 8 | 0x08 | ALLOC_AS | 1 | Allocate address space, returns AS fd |
| 9 | 0x09 | OPEN_TSG | 3 | Create TSG, returns TSG fd |
| 10 | 0x0a | GET_TPC_MASKS | 2 | TPC topology |
| 11 | 0x0b | OPEN_CHANNEL | 16 | Create channel, returns channel fd |
| 19 | 0x13 | VSMS_MAPPING | 2 | Virtual SM mapping |
| 26 | 0x1a | GET_ENGINE_INFO | 2 | Engine list (GR, CE, etc.) |
| 28 | 0x1c | CLK_GET_RANGE | 4 | Clock frequency range |
| 29 | 0x1d | CLK_GET_VF_POINTS | 2 | Voltage-frequency points |
| 38 | 0x26 | GET_FBP_L2_MASKS | 2 | L2 cache topology |
| 40 | 0x28 | SET_DETERMINISTIC_OPTS | 1 | Determinism control |
| 41 | 0x29 | REGISTER_BUFFER | 171 | Register nvmap buffer with GPU |
| 43 | 0x2b | GET_GPC_LOCAL_TO_PHYSICAL_MAP | 2 | GPC mapping |
| 44 | 0x2c | GET_GPC_LOCAL_TO_LOGICAL_MAP | 2 | GPC mapping |

### nvmap Ioctls (Magic 'N' = 0x4e)

| Nr | Hex | Name | Count | Notes |
|----|-----|------|-------|-------|
| 0 | 0x00 | CREATE | 171 | **HOT PATH: create handle** |
| 3 | 0x03 | ALLOC | 171 | **HOT PATH: allocate physical memory** |
| 15 | 0x0f | GET_FD | 553 | **HOT PATH: get dmabuf fd** |
| 25 | 0x19 | GET_AVAILABLE_HEAPS | 1 | Discover memory heaps |
| 105 | 0x69 | QUERY_HEAP_PARAMS | 1 | Heap parameters |

### Address Space Ioctls (Magic 'A' = 0x41)

| Nr | Hex | Name | Count | Notes |
|----|-----|------|-------|-------|
| 1 | 0x01 | BIND_CHANNEL | 16 | Bind channel to AS |
| 6 | 0x06 | ALLOC_SPACE | 4 | Pre-allocate VA regions |
| 7 | 0x07 | MAP_BUFFER_EX | 138 | **HOT PATH: map buffer into GPU VA** |
| 8 | 0x08 | GET_VA_REGIONS | 2 | Query VA region layout |
| 12 | 0x0c | GET_SYNC_RO_MAP | 1 | Syncpoint read-only mapping |

### Channel Ioctls (Magic 'H' = 0x48)

| Nr | Hex | Name | Count | Notes |
|----|-----|------|-------|-------|
| 14 | 0x0e | SET_TIMEOUT_EX | 1 | Failed ENOTTY (wrong fd) |
| 108 | 0x6c | ALLOC_OBJ_CTX | 16 | **Allocate compute class** (class_num!) |
| 111 | 0x6f | SET_ERROR_NOTIFIER | 16 | Error notification setup |
| 119 | 0x77 | WDT | 16 | Watchdog timer config |
| 122 | 0x7a | SET_PREEMPTION_MODE | 1 | Preemption mode |
| 126 | 0x7e | GET_USER_SYNCPOINT | 16 | Syncpoint for completion |
| 128 | 0x80 | SETUP_BIND | 16 | **KEY: GPFIFO+userd+usermode submit** |

### TSG Ioctls (Magic 'T' = 0x54)

| Nr | Hex | Name | Count | Notes |
|----|-----|------|-------|-------|
| 7 | 0x07 | EVENT_ID_CTRL | 4 | Event notification |
| 9 | 0x09 | SET_TIMESLICE | 1 | Scheduling timeslice |
| 11 | 0x0b | BIND_CHANNEL_EX | 16 | Bind channel to TSG |
| 13 | 0x0d | SET_L2_MAX_WAYS_EVICT_LAST | 1 | L2 cache policy |
| 18 | 0x12 | CREATE_SUBCONTEXT | 1 | Create subcontext in TSG |

## CUDA Init Sequence (Decoded)

### Phase 1: Discovery
```
1. open(/dev/nvmap) -> fd=3
2. N:25 GET_AVAILABLE_HEAPS          — discover heaps
3. DRM probe (renderD128 x 2)
4. open(/dev/nvgpu/igpu0/ctrl) -> fd=4
5. G:5  GET_CHARACTERISTICS          — GPU arch, SM ver, compute_class, flags
6. G:10 GET_TPC_MASKS                — TPC topology
7-15. Various discovery ioctls (GPC maps, L2 masks, clocks, etc.)
```

### Phase 2: Memory System Setup
```
16. N:105 QUERY_HEAP_PARAMS          — detailed heap info
17. open(/dev/host1x-fence)          — fence device
18. G:8   ALLOC_AS                   — create address space -> AS fd=6
   (big_page_size=0, flags=UNIFIED_VA, PDE-aligned VA range)
19. A:12  GET_SYNC_RO_MAP            — syncpoint map for GPU
20. A:8   GET_VA_REGIONS (x2)        — understand VA layout
21. A:6   ALLOC_SPACE (x3)           — pre-reserve VA regions
```

### Phase 3: Buffer Allocation Loop (171 iterations!)
```
For each buffer:
  N:0  NVMAP_CREATE               — create handle (size -> handle)
  N:3  NVMAP_ALLOC                — back with physical memory (IOVMM heap)
  N:15 NVMAP_GET_FD               — get dmabuf fd
  G:41 REGISTER_BUFFER            — register with GPU
  N:15 NVMAP_GET_FD               — get another fd
  A:7  MAP_BUFFER_EX              — map into GPU VA space
```

### Phase 4: TSG + Channel Setup (per-channel)
```
G:9   OPEN_TSG                   — create TSG -> fd=8
T:18  CREATE_SUBCONTEXT          — type=ASYNC, as_fd -> VEID=1

N:0+3+15 + G:41  (alloc+register error notifier buffer)

G:11  OPEN_CHANNEL               — create channel -> fd=9
A:1   BIND_CHANNEL               — bind channel to AS (BEFORE TSG bind!)
T:11  BIND_CHANNEL_EX            — bind channel to TSG with VEID
H:119 WDT                       — disable watchdog (required for DETERMINISTIC)

N:0+3+15 + G:41  (alloc+register GPFIFO buffer, 1024 entries x 8 bytes)
N:0+3+15 + G:41  (alloc+register userd buffer, 4096 bytes)
N:15 x 2         (get dmabuf fds for SETUP_BIND)

H:128 SETUP_BIND                — **GPFIFO + userd + usermode submit!**
  flags = USERMODE_SUPPORT | DETERMINISTIC = (1<<3)|(1<<1)
  -> returns work_submit_token (doorbell)

H:126 GET_USER_SYNCPOINT        — syncpoint for completion tracking
  -> returns syncpoint_id, syncpoint_max, gpu_va

H:108 ALLOC_OBJ_CTX             — **allocate compute class 0xc7c0!**
H:111 SET_ERROR_NOTIFIER        — error notification

T:7   EVENT_ID_CTRL              — event setup
H:122 SET_PREEMPTION_MODE        — preemption config
T:9   SET_TIMESLICE              — scheduling timeslice
T:13  SET_L2_MAX_WAYS_EVICT_LAST — L2 cache policy
```

### Steady State: No SUBMIT_GPFIFO ioctls!
CUDA writes GPFIFO entries directly to mapped memory (from SETUP_BIND)
and rings the doorbell via MMIO. Zero ioctl overhead for submissions.

## Verified Python ctypes Structs (test_nvgpu.py)

All structs verified by running ioctls successfully against the kernel:

| Struct | Size | Key Fields |
|--------|------|------------|
| `nvgpu_gpu_characteristics` | 328 | arch, compute_class, flags, sm_arch_sm_version |
| `nvgpu_gpu_get_characteristics` | 16 | buf_size, buf_addr |
| `nvmap_create_handle` | 8 | size, handle |
| `nvmap_alloc_handle` | 20 | handle, heap_mask, flags, align, kind (pack=1) |
| `nvgpu_alloc_as_args` | 64 | big_page_size, as_fd, flags, va_range_start/end/split, padding[6] |
| `nvgpu_as_map_buffer_ex_args` | 40 | flags, compr_kind, incompr_kind, dmabuf_fd, offset |
| `nvgpu_gpu_open_tsg_args` | 24 | tsg_fd, flags, token |
| `nvgpu_tsg_create_subcontext_args` | 16 | type(ASYNC=1), as_fd, veid(out) |
| `nvgpu_gpu_open_channel_args` | 4 | channel_fd (union, just s32) |
| `nvgpu_tsg_bind_channel_ex_args` | 24 | channel_fd, subcontext_id, reserved[16] |
| `nvgpu_as_bind_channel_args` | 4 | channel_fd |
| `nvgpu_channel_wdt_args` | 8 | wdt_status(1=disable), timeout_ms |
| `nvgpu_channel_setup_bind_args` | 104 | gpfifo/userd dmabuf fds, flags, outputs: token/gpu_vas |
| `nvgpu_get_user_syncpoint_args` | 16 | gpu_va(u64), syncpoint_id(u32), syncpoint_max(u32) |
| `nvgpu_alloc_obj_ctx_args` | 16 | class_num, flags, obj_id |

## GPU Characteristics (ga10b — Jetson Orin AGX 64GB)

| Property | Value |
|----------|-------|
| Architecture | 0x0170 (Ampere) |
| Implementation | 0x000b (ga10b) |
| SM arch version | 0x0807 (SM 8.7) |
| Compute class | 0xc7c0 |
| GPFIFO class | 0xc76f |
| DMA copy class | 0xc7b5 |
| GPC count | 1 |
| TPC per GPC | 4 (= 4 SMs total) |
| GPU VA bits | 40 (1 TB address space) |
| L2 cache | 4 MB |
| VRAM | 0 (unified memory) |
| Big page size | 0 (not supported) |
| Max GPFIFO entries | 268435456 (2^28) |
| Max freq | 1300 MHz |
| PDE alignment | 2^21 = 2 MB |
| Key flags | USERMODE_SUBMIT, COMPUTE, DETERMINISTIC, TSG_SUBCONTEXTS, IO_COHERENCE |

## Next Steps

1. **Build minimal `TegraIface` prototype** — TegraDevice + TegraAllocator + TegraCompiler
   - Memory: nvmap CREATE+ALLOC+GET_FD -> MAP_BUFFER_EX
   - Channel: TSG -> subctx -> channel -> AS_bind -> TSG_bind -> WDT -> SETUP_BIND
   - Compute: ALLOC_OBJ_CTX(0xc7c0) -> push methods via GPFIFO -> doorbell
2. **Map usermode submit registers** — mmap the userd and doorbell regions
3. **Push a simple compute shader** — inline a NOP or memcopy shader
4. **Syncpoint-based completion** — wait for work_submit_token
