# Phase 2: Memory Management — Working Document

**Status:** COMPLETE — 11/11 tests pass  
**Parent doc:** [nv-attempt.md](nv-attempt.md)  
**Previous phase:** [phase1.md](phase1.md) (COMPLETE)  
**Learning context:** [Learning-Phase1.md](Learning-Phase1.md)

## Test Results: 11/11 PASS

```
Phase 1 (tests 1-7):   PASS — all existing tests still pass
Test  8: MMAP READ/WRITE        PASS — sequential bytes, u32 patterns, full fills
Test  9: MEMORY COHERENCE       PASS — dual mmap, interleaved, cross-page, NVMAP_IOC_R/W
Test 10: MULTI-SIZE GPU MAP     PASS — 4KB, 64KB, 1MB, 16MB, 64MB all work
Test 11: CACHEABLE FLAGS        PASS — all 4 flag modes work (WC, CACHEABLE, UC, IC)
```

## Tasks

- [x] mmap nvmap buffers to CPU address space
- [x] Write test patterns from CPU, read back to verify CPU access
- [x] Map same buffer into GPU VA via MAP_BUFFER_EX
- [x] Verify coherence (dual mmap + NVMAP_IOC_WRITE/READ)
- [x] Build TegraAllocator helper class (alloc/free/map/unmap)
- [x] Test with various buffer sizes (4KB, 64KB, 1MB, 16MB, 64MB)
- [x] Understand cache coherence requirements (IO_COHERENCE = no flushes needed)
- [x] Compare cacheability flags (WC vs CACHEABLE vs INNER_CACHEABLE)

## Key Discoveries

### 1. mmap works via dmabuf fd (NOT nvmap device fd)

The mmap target is the dmabuf fd returned by `NVMAP_IOC_GET_FD`, not the `/dev/nvmap` fd.
Standard Linux DMA-BUF mmap:
```python
mm = mmap.mmap(dmabuf_fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, 0)
```
No special ioctls needed for mmap — the DMA-BUF subsystem handles it.

### 2. IO_COHERENCE is real — no cache flushes needed

The Orin's GPU characteristics report `SUPPORT_IO_COHERENCE` (flag bit 20). Empirically confirmed:
- Write through mmap #1, immediately read through mmap #2 of the same buffer → data matches
- Byte-level interleaved writes between two mappings → all correct
- Cross-page boundary writes (offset 4094-4098) → correct
- NVMAP_IOC_WRITE → visible immediately through mmap → correct

**No explicit cache operations (`NVMAP_IOC_CACHE`) are needed.**

### 3. NVMAP_IOC_WRITE/READ ioctls work

Alternative data-path ioctls (`_IOW('N', 6, ...)` and `_IOW('N', 7, ...)`) using `nvmap_rw_handle` (56 bytes):

```python
class nvmap_rw_handle(ctypes.Structure):
    _fields_ = [
        ("addr",        c_uint64),   # user pointer
        ("handle",      c_uint32),   # nvmap handle
        ("_pad",        c_uint32),   # alignment padding
        ("offset",      c_uint64),   # offset into handle memory
        ("elem_size",   c_uint64),   # individual atom size
        ("hmem_stride", c_uint64),   # stride in handle memory
        ("user_stride", c_uint64),   # stride in user memory
        ("count",       c_uint64),   # number of atoms
    ]
```

**mmap is preferred** for tinygrad (zero ioctl overhead, direct pointer access).

### 4. nvmap_alloc_handle struct corrected

Phase 1 used `kind` (u8) + padding. The actual kernel struct has `numa_nid` (s32):
```c
struct nvmap_alloc_handle {
    __u32 handle, heap_mask, flags, align;
    __s32 numa_nid;     // NUMA node id (0 for Orin)
};
```
Phase 1 worked because `kind=0` + 3 zero pad bytes == `numa_nid=0`. Fixed for correctness.

### 5. NVMAP_IOC_FREE handle overflow

nvmap handles have bit 31 set (e.g. `0x80000d4c`), exceeding Python's signed int range.
`NVMAP_IOC_FREE` is `_IO('N', 4)` — passes the handle directly.
Fix: convert u32 → s32:
```python
signed_handle = handle if handle < 0x80000000 else handle - 0x100000000
fcntl.ioctl(nvmap_fd, NVMAP_IOC_FREE, signed_handle)
```

### 6. Cacheability flag performance (1MB buffer)

| Flag | Value | Write | Read | Notes |
|------|-------|-------|------|-------|
| UNCACHEABLE | 0 | 688 MB/s | 75 MB/s | 18x slower reads — avoid |
| WRITE_COMBINE | 1 | 2917 MB/s | 691 MB/s | Good writes, moderate reads |
| **INNER_CACHEABLE** | **2** | **2737 MB/s** | **11826 MB/s** | **Best overall — cached reads!** |
| CACHEABLE | 5 | 2862 MB/s | 656 MB/s | Similar to WC (surprising) |

**Recommendation for tinygrad: Use `INNER_CACHEABLE` (flags=2).**
- Write perf identical to WC (~2.7-2.9 GB/s)
- Read perf **17x faster** than WC (11.8 GB/s vs 691 MB/s)
- IO_COHERENCE means no explicit flushes needed even with caching

### 7. Multi-size allocation timing

| Size | Alloc | mmap | GPU map | Total |
|------|-------|------|---------|-------|
| 4 KB | 0.0ms | 0.0ms | 0.1ms | 0.1ms |
| 64 KB | 0.0ms | 0.0ms | 0.1ms | 0.1ms |
| 1 MB | 0.0ms | 0.0ms | 0.1ms | 0.1ms |
| 16 MB | 0.1ms | 0.2ms | 0.9ms | 0.9ms |
| 64 MB | 0.1ms | 0.6ms | 3.1ms | 3.1ms |

GPU VA mapping time scales with size (page table population).

### 8. GPU VAs are allocated top-down

```
4KB  → 0xffffa01000
64KB → 0xffffa10000
1MB  → 0xffffb00000
16MB → 0xfffe200000
64MB → 0xfff8200000
```
Within our ALLOC_AS range `0x200000 - 0xFFFFE00000`. Top-down matches CUDA behavior.

## TegraAllocator API

```python
class TegraAllocator:
    def __init__(self, nvmap_fd, as_fd=None)
    def alloc(self, size, heap=IOVMM, flags=WC, align=4096) -> TegraBuffer
    def mmap_buffer(self, buf) -> mmap.mmap
    def gpu_map(self, buf, page_size=4096) -> int  # returns GPU VA
    def gpu_unmap(self, buf)
    def free(self, buf)
    def free_all(self)

class TegraBuffer:
    size, handle, dmabuf_fd, cpu_addr (mmap), gpu_va, flags
```

## Memory Model (Jetson Orin AGX 64GB)

```
                  ┌─────────────────────────────┐
                  │     Unified DRAM (64GB)      │
                  │   (shared CPU + GPU memory)  │
                  └──────────┬──────────────────┘
                             │
              ┌──────────────┼──────────────────┐
              │              │                  │
         ┌────▼────┐   ┌────▼────┐        ┌────▼────┐
         │  CPU    │   │  GPU    │        │  GPU    │
         │  mmap   │   │  VA     │        │  VA     │
         │ (dmabuf)│   │ (MAP_   │        │ (MAP_   │
         │         │   │ BUFFER_ │        │ BUFFER_ │
         │ R/W via │   │ EX)     │        │ EX)     │
         │ mm[off] │   │         │        │         │
         └─────────┘   └─────────┘        └─────────┘
         
         IO_COHERENCE: CPU writes visible to GPU immediately
         No NVMAP_IOC_CACHE needed!
```

## Next Steps (Phase 3: Compute Dispatch)

1. **mmap GPFIFO + userd buffers** — use TegraAllocator to map the channel's GPFIFO ring
2. **Push GPU methods via GPFIFO** — write NVC6C0 compute methods
3. **Ring the doorbell** — mmap the usermode MMIO region, write work_submit_token
4. **Syncpoint-based completion** — wait for compute to finish
5. **Simple compute test** — memcpy or memset via DMA copy class (0xc7b5)
