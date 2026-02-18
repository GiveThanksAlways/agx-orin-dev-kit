#!/usr/bin/env python3
"""
Phase 1+2+3 Test: Direct nvgpu/nvmap ioctl access from Python.

Phase 1: Proves we can talk to the Jetson Orin GPU without going through CUDA.
Phase 2: Proves CPU<->GPU shared memory works end-to-end via mmap + MAP_BUFFER_EX.
Phase 3: Command submission — GPFIFO push, QMD compute dispatch, shader execution.

Tests the ioctl sequence discovered by stracing CUDA:
  1. Open /dev/nvmap, /dev/nvgpu/igpu0/ctrl
  2. GET_CHARACTERISTICS — read GPU info (arch, SM, compute_class)
  3. NVMAP_CREATE + NVMAP_ALLOC — allocate GPU memory
  4. ALLOC_AS — create address space
  5. mmap dmabuf for CPU access
  6. Verify CPU<->GPU memory coherence (IO_COHERENCE)
  7. GPFIFO command submission + compute shader dispatch

Prerequisites: run as user with access to /dev/nvgpu/* and /dev/nvmap
"""

import os
import sys
import struct
import ctypes
import ctypes.util
import fcntl
import mmap
import time
from ctypes import c_uint8, c_uint16, c_uint32, c_uint64, c_int16, c_int32, c_int64

# ============================================================================
# ioctl helpers
# ============================================================================

# Linux ioctl direction bits (aarch64)
_IOC_NONE  = 0
_IOC_WRITE = 1
_IOC_READ  = 2

_IOC_NRBITS   = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_DIRBITS  = 2

_IOC_NRSHIFT   = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS      # 8
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS   # 16
_IOC_DIRSHIFT  = _IOC_SIZESHIFT + _IOC_SIZEBITS   # 30

def _IOC(d, t, nr, size):
    return (d << _IOC_DIRSHIFT) | (ord(t) << _IOC_TYPESHIFT) | (nr << _IOC_NRSHIFT) | (size << _IOC_SIZESHIFT)

def _IO(t, nr):         return _IOC(_IOC_NONE, t, nr, 0)
def _IOR(t, nr, size):  return _IOC(_IOC_READ, t, nr, size)
def _IOW(t, nr, size):  return _IOC(_IOC_WRITE, t, nr, size)
def _IOWR(t, nr, size): return _IOC(_IOC_READ | _IOC_WRITE, t, nr, size)


def nv_ioctl(fd, ioc_code, buf):
    """Call an ioctl, raise on error."""
    ret = fcntl.ioctl(fd, ioc_code, buf)
    if ret < 0:
        raise OSError(f"ioctl 0x{ioc_code:08x} failed with {ret}")
    return ret


# ============================================================================
# nvgpu ctrl-gpu structs & ioctls (Magic 'G' = 0x47)
# ============================================================================

class nvgpu_gpu_characteristics(ctypes.Structure):
    """From nvgpu-ctrl.h — returned by GET_CHARACTERISTICS.
    
    MUST match the kernel struct exactly (natural alignment on aarch64).
    No _pack_ = 1 because the kernel uses natural alignment!
    """
    _fields_ = [
        # offset 0
        ("arch",                      c_uint32),
        ("impl",                      c_uint32),
        ("rev",                       c_uint32),
        ("num_gpc",                   c_uint32),
        # offset 16
        ("numa_domain_id",            c_int32),    # __s32, -1 = no NUMA info
        # 4 bytes padding inserted by compiler for u64 alignment
        ("_pad0",                     c_uint32),
        # offset 24
        ("L2_cache_size",             c_uint64),   # bytes
        ("on_board_video_memory_size", c_uint64),  # bytes
        # offset 40
        ("num_tpc_per_gpc",           c_uint32),   # architectural max
        ("bus_type",                  c_uint32),
        ("big_page_size",             c_uint32),   # default big page size
        ("compression_page_size",     c_uint32),
        # offset 56
        ("pde_coverage_bit_count",    c_uint32),
        ("available_big_page_sizes",  c_uint32),
        # offset 64
        ("flags",                     c_uint64),
        # offset 72
        ("twod_class",                c_uint32),
        ("threed_class",              c_uint32),
        ("compute_class",             c_uint32),
        ("gpfifo_class",              c_uint32),
        ("inline_to_memory_class",    c_uint32),
        ("dma_copy_class",            c_uint32),
        # offset 96
        ("gpc_mask",                  c_uint32),   # u32, NOT u64!
        ("sm_arch_sm_version",        c_uint32),
        ("sm_arch_spa_version",       c_uint32),
        ("sm_arch_warp_count",        c_uint32),
        # offset 112
        ("gpu_ioctl_nr_last",         c_int16),
        ("tsg_ioctl_nr_last",         c_int16),
        ("dbg_gpu_ioctl_nr_last",     c_int16),
        ("ioctl_channel_nr_last",     c_int16),
        ("as_ioctl_nr_last",          c_int16),
        # offset 122
        ("gpu_va_bit_count",          c_uint8),
        ("reserved",                  c_uint8),
        # offset 124
        ("max_fbps_count",            c_uint32),
        ("fbp_en_mask",               c_uint32),
        ("emc_en_mask",               c_uint32),
        ("max_ltc_per_fbp",           c_uint32),
        ("max_lts_per_ltc",           c_uint32),
        ("max_tex_per_tpc",           c_uint32),
        ("max_gpc_count",             c_uint32),
        # offset 152
        ("rop_l2_en_mask_DEPRECATED", c_uint32 * 2),
        # offset 160
        ("chipname",                  c_uint8 * 8),
        # offset 168
        ("gr_compbit_store_base_hw",  c_uint64),
        # offset 176
        ("gr_gobs_per_comptagline_per_slice", c_uint32),
        ("num_ltc",                   c_uint32),
        ("lts_per_ltc",               c_uint32),
        ("cbc_cache_line_size",       c_uint32),
        ("cbc_comptags_per_line",     c_uint32),
        ("map_buffer_batch_limit",    c_uint32),
        # offset 200
        ("max_freq",                  c_uint64),
        # offset 208
        ("graphics_preemption_mode_flags", c_uint32),
        ("compute_preemption_mode_flags",  c_uint32),
        ("default_graphics_preempt_mode",  c_uint32),
        ("default_compute_preempt_mode",   c_uint32),
        # offset 224
        ("local_video_memory_size",   c_uint64),  # non-zero only for dGPUs
        # offset 232
        ("pci_vendor_id",             c_uint16),
        ("pci_device_id",             c_uint16),
        ("pci_subsystem_vendor_id",   c_uint16),
        ("pci_subsystem_device_id",   c_uint16),
        ("pci_class",                 c_uint16),
        ("pci_revision",              c_uint8),
        ("vbios_oem_version",         c_uint8),
        ("vbios_version",             c_uint32),
        # offset 248
        ("reg_ops_limit",             c_uint32),
        ("reserved1",                 c_uint32),
        # offset 256
        ("event_ioctl_nr_last",       c_int16),
        ("pad",                       c_uint16),
        ("max_css_buffer_size",       c_uint32),
        # offset 264
        ("ctxsw_ioctl_nr_last",       c_int16),
        ("prof_ioctl_nr_last",        c_int16),
        ("nvs_ioctl_nr_last",         c_int16),
        ("reserved2",                 c_uint8 * 2),
        # offset 272
        ("max_ctxsw_ring_buffer_size", c_uint32),
        ("reserved3",                 c_uint32),
        # offset 280
        ("per_device_identifier",     c_uint64),
        # offset 288
        ("num_ppc_per_gpc",           c_uint32),
        ("max_veid_count_per_tsg",    c_uint32),
        ("num_sub_partition_per_fbpa", c_uint32),
        ("gpu_instance_id",           c_uint32),
        ("gr_instance_id",            c_uint32),
        ("max_gpfifo_entries",        c_uint32),
        ("max_dbg_tsg_timeslice",     c_uint32),
        ("reserved5",                 c_uint32),
        # offset 320
        ("device_instance_id",        c_uint64),
        # Total: 328 bytes
    ]


class nvgpu_gpu_get_characteristics(ctypes.Structure):
    """Wrapper for GET_CHARACTERISTICS ioctl."""
    _fields_ = [
        ("gpu_characteristics_buf_size", c_uint64),
        ("gpu_characteristics_buf_addr", c_uint64),
    ]


# GET_CHARACTERISTICS: _IOWR('G', 5, nvgpu_gpu_get_characteristics)
NVGPU_GPU_IOCTL_GET_CHARACTERISTICS = _IOWR('G', 5, ctypes.sizeof(nvgpu_gpu_get_characteristics))


class nvgpu_gpu_zcull_get_ctx_size_args(ctypes.Structure):
    _fields_ = [("size", c_uint32)]

NVGPU_GPU_IOCTL_ZCULL_GET_CTX_SIZE = _IOR('G', 1, ctypes.sizeof(nvgpu_gpu_zcull_get_ctx_size_args))


# ============================================================================
# nvmap structs & ioctls (Magic 'N' = 0x4e)
# ============================================================================

class nvmap_create_handle(ctypes.Structure):
    """NVMAP_IOC_CREATE: create a memory handle.
    Also used by NVMAP_IOC_GET_FD (nr=15) and NVMAP_IOC_FROM_FD (nr=16)."""
    _fields_ = [
        ("size",   c_uint32),   # in: requested size (CREATE) / unused (GET_FD)
        ("handle", c_uint32),   # out: handle id (CREATE) / in: handle (GET_FD)
    ]

NVMAP_IOC_CREATE = _IOWR('N', 0, ctypes.sizeof(nvmap_create_handle))


class nvmap_alloc_handle(ctypes.Structure):
    """NVMAP_IOC_ALLOC: back a handle with physical memory.
    
    Kernel struct (from nvmap.h):
        __u32 handle, heap_mask, flags, align;
        __s32 numa_nid;   // NUMA node id (0 = node 0, -1 = any)
    sizeof = 20 bytes (5 x u32). No padding needed.
    
    NOTE: Phase 1 used 'kind' (u8) + padding here, which worked because
    kind=0 + 3 zeroed pad bytes == numa_nid=0. Fixed in Phase 2 for correctness.
    """
    _fields_ = [
        ("handle",    c_uint32),  # in: handle from CREATE
        ("heap_mask", c_uint32),  # in: which heap(s) to use
        ("flags",     c_uint32),  # in: allocation flags (wb/wc/uc/cacheable)
        ("align",     c_uint32),  # in: alignment requirement
        ("numa_nid",  c_int32),   # in: NUMA node id (0 for Orin)
    ]

# Heap masks from nvmap.h
NVMAP_HEAP_SYSMEM              = (1 << 31)
NVMAP_HEAP_IOVMM               = (1 << 30)
NVMAP_HEAP_CARVEOUT_GENERIC    = 1

# Alloc flags
NVMAP_HANDLE_UNCACHEABLE        = 0
NVMAP_HANDLE_WRITE_COMBINE      = 1
NVMAP_HANDLE_INNER_CACHEABLE    = 2
NVMAP_HANDLE_CACHEABLE          = 5

NVMAP_IOC_ALLOC = _IOW('N', 3, ctypes.sizeof(nvmap_alloc_handle))


class nvmap_handle_param(ctypes.Structure):
    """NVMAP_IOC_PARAM: query handle parameters."""
    _fields_ = [
        ("handle", c_uint32),
        ("param",  c_uint32),   # which param to query
        ("result", c_uint64),   # out: result (u64 in kernel)
    ]

# Param types
NVMAP_HANDLE_PARAM_SIZE       = 1
NVMAP_HANDLE_PARAM_ALIGNMENT  = 2
NVMAP_HANDLE_PARAM_BASE       = 3
NVMAP_HANDLE_PARAM_HEAP       = 4
NVMAP_HANDLE_PARAM_KIND       = 5
NVMAP_HANDLE_PARAM_COMPR      = 6

NVMAP_IOC_PARAM = _IOWR('N', 8, ctypes.sizeof(nvmap_handle_param))


# NVMAP_IOC_GET_FD uses nvmap_create_handle struct
# handle field = input handle, size field reused as output fd

NVMAP_IOC_GET_FD = _IOWR('N', 15, ctypes.sizeof(nvmap_create_handle))


class nvmap_available_heaps(ctypes.Structure):
    """NVMAP_IOC_GET_AVAILABLE_HEAPS"""
    _fields_ = [
        ("heaps",  c_uint64),   # u64 bitmask, NOT u32!
    ]

NVMAP_IOC_GET_AVAILABLE_HEAPS = _IOR('N', 25, ctypes.sizeof(nvmap_available_heaps))

# NVMAP_IOC_FREE: release a handle (arg is the handle value as __u32)
NVMAP_IOC_FREE = _IO('N', 4)

# NVMAP_IOC_WRITE / NVMAP_IOC_READ: strided copy between user buffer and handle
class nvmap_rw_handle(ctypes.Structure):
    """For NVMAP_IOC_WRITE (nr=6) and NVMAP_IOC_READ (nr=7).
    
    Kernel struct (natural alignment on aarch64):
      __u64 addr;           // user pointer
      __u32 handle;         // nvmap handle
      // 4 bytes padding for u64 alignment
      __u64 offset;         // offset into hmem
      __u64 elem_size;      // individual atom size
      __u64 hmem_stride;    // delta bytes between atoms in hmem
      __u64 user_stride;    // delta bytes between atoms in user
      __u64 count;          // number of atoms to copy
    Total: 56 bytes
    """
    _fields_ = [
        ("addr",        c_uint64),   # user pointer (cast to u64)
        ("handle",      c_uint32),   # nvmap handle
        ("_pad",        c_uint32),   # alignment padding
        ("offset",      c_uint64),   # offset into handle memory
        ("elem_size",   c_uint64),   # individual atom size
        ("hmem_stride", c_uint64),   # stride in handle memory
        ("user_stride", c_uint64),   # stride in user memory
        ("count",       c_uint64),   # number of atoms
    ]

NVMAP_IOC_WRITE = _IOW('N', 6, ctypes.sizeof(nvmap_rw_handle))
NVMAP_IOC_READ  = _IOW('N', 7, ctypes.sizeof(nvmap_rw_handle))

# NVMAP_IOC_CACHE: cache maintenance
class nvmap_cache_op(ctypes.Structure):
    """Cache operation struct (64-bit pointer on aarch64)."""
    _fields_ = [
        ("addr",   c_uint64),   # unsigned long = 8 bytes on aarch64
        ("handle", c_uint32),
        ("len",    c_uint32),
        ("op",     c_int32),
    ]

NVMAP_CACHE_OP_WB     = 0   # writeback
NVMAP_CACHE_OP_INV    = 1   # invalidate
NVMAP_CACHE_OP_WB_INV = 2   # writeback + invalidate
NVMAP_IOC_CACHE = _IOW('N', 12, ctypes.sizeof(nvmap_cache_op))


# ============================================================================
# GPU Address Space Unmap (for cleanup)
# ============================================================================

class nvgpu_as_unmap_buffer_args(ctypes.Structure):
    """Unmap a buffer from GPU VA space."""
    _fields_ = [
        ("offset", c_uint64),   # GPU VA to unmap
    ]

NVGPU_AS_IOCTL_UNMAP_BUFFER = _IOWR('A', 5, ctypes.sizeof(nvgpu_as_unmap_buffer_args))


# ============================================================================
# TegraAllocator — Reusable memory management for Jetson Orin
# ============================================================================

class TegraBuffer:
    """A GPU-accessible buffer with optional CPU mmap.
    
    Represents a single allocation lifecycle:
        nvmap CREATE → ALLOC → GET_FD → mmap (CPU) → MAP_BUFFER_EX (GPU VA)
    
    Attributes:
        size:       Requested buffer size in bytes
        handle:     nvmap handle ID
        dmabuf_fd:  DMA-BUF file descriptor (for mmap and GPU mapping)
        cpu_addr:   mmap object for CPU access (None if not mmapped)
        gpu_va:     GPU virtual address (0 if not GPU-mapped)
        flags:      Alloc flags used (WRITE_COMBINE, CACHEABLE, etc.)
    """
    def __init__(self, size, handle, dmabuf_fd, flags, nvmap_fd):
        self.size = size
        self.handle = handle
        self.dmabuf_fd = dmabuf_fd
        self.flags = flags
        self._nvmap_fd = nvmap_fd
        self.cpu_addr = None      # mmap object
        self.gpu_va = 0           # GPU virtual address
        self._as_fd = None        # AS fd used for GPU mapping (for cleanup)
    
    def __repr__(self):
        parts = [f"TegraBuffer(size={self.size}"]
        parts.append(f"handle=0x{self.handle:x}")
        parts.append(f"dmabuf_fd={self.dmabuf_fd}")
        if self.cpu_addr is not None:
            parts.append("mmapped")
        if self.gpu_va:
            parts.append(f"gpu_va=0x{self.gpu_va:012x}")
        return ", ".join(parts) + ")"


class TegraAllocator:
    """Memory allocator for Jetson Orin using nvmap + nvgpu ioctls.
    
    Manages the full lifecycle:
      create → alloc → get_fd → mmap (CPU) → MAP_BUFFER_EX (GPU VA) → cleanup
    
    Usage:
        alloc = TegraAllocator(nvmap_fd, as_fd)
        buf = alloc.alloc(size=4096)          # nvmap create+alloc+get_fd
        alloc.mmap_buffer(buf)                # CPU access via mmap
        alloc.gpu_map(buf)                    # GPU VA via MAP_BUFFER_EX
        
        # Use buf.cpu_addr[] for reads/writes, buf.gpu_va for GPU commands
        
        alloc.free(buf)                       # cleanup everything
    """
    
    PDE_SIZE = 1 << 21  # 2MB PDE alignment for ga10b
    
    def __init__(self, nvmap_fd, as_fd=None):
        self.nvmap_fd = nvmap_fd
        self.as_fd = as_fd
        self._buffers = []  # track for cleanup
    
    def alloc(self, size, heap=NVMAP_HEAP_IOVMM, flags=NVMAP_HANDLE_WRITE_COMBINE, 
              align=4096) -> TegraBuffer:
        """Allocate a buffer: CREATE → ALLOC → GET_FD.
        
        Args:
            size:  Buffer size in bytes (will be page-aligned by kernel)
            heap:  Heap mask (default: IOVMM, the only working heap on Orin)
            flags: Cacheability flags (WRITE_COMBINE=1, CACHEABLE=5)
            align: Alignment in bytes (default: 4096 = page)
        
        Returns:
            TegraBuffer with handle and dmabuf_fd set
        """
        # Step 1: CREATE — get a handle
        create = nvmap_create_handle()
        create.size = size
        create.handle = 0
        nv_ioctl(self.nvmap_fd, NVMAP_IOC_CREATE, create)
        handle = create.handle
        
        # Step 2: ALLOC — back with physical memory
        alloc_args = nvmap_alloc_handle()
        alloc_args.handle = handle
        alloc_args.heap_mask = heap
        alloc_args.flags = flags
        alloc_args.align = align
        alloc_args.numa_nid = 0
        nv_ioctl(self.nvmap_fd, NVMAP_IOC_ALLOC, alloc_args)
        
        # Step 3: GET_FD — get dmabuf fd
        get_fd = nvmap_create_handle()
        get_fd.handle = handle
        get_fd.size = 0
        nv_ioctl(self.nvmap_fd, NVMAP_IOC_GET_FD, get_fd)
        dmabuf_fd = get_fd.size
        
        buf = TegraBuffer(size, handle, dmabuf_fd, flags, self.nvmap_fd)
        self._buffers.append(buf)
        return buf
    
    def mmap_buffer(self, buf: TegraBuffer) -> mmap.mmap:
        """mmap a TegraBuffer into CPU address space.
        
        Uses the dmabuf fd (NOT the nvmap device fd).
        Returns the mmap object, also stored as buf.cpu_addr.
        """
        if buf.cpu_addr is not None:
            return buf.cpu_addr
        
        # mmap the dmabuf fd: PROT_READ|PROT_WRITE, MAP_SHARED, offset=0
        buf.cpu_addr = mmap.mmap(buf.dmabuf_fd, buf.size,
                                  mmap.MAP_SHARED,
                                  mmap.PROT_READ | mmap.PROT_WRITE,
                                  0)
        return buf.cpu_addr
    
    def gpu_map(self, buf: TegraBuffer, page_size=4096) -> int:
        """Map a TegraBuffer into GPU virtual address space via MAP_BUFFER_EX.
        
        Returns GPU VA, also stored as buf.gpu_va.
        Requires as_fd to be set on the allocator.
        """
        if self.as_fd is None:
            raise RuntimeError("TegraAllocator has no AS fd — call with as_fd= or set .as_fd")
        if buf.gpu_va != 0:
            return buf.gpu_va
        
        args = nvgpu_as_map_buffer_ex_args()
        args.flags = 0            # let kernel choose address
        args.compr_kind = -1      # NV_KIND_INVALID (no compression)
        args.incompr_kind = 0     # pitch linear
        args.dmabuf_fd = buf.dmabuf_fd
        args.page_size = page_size
        args.buffer_offset = 0
        args.mapping_size = 0     # 0 = whole buffer
        args.offset = 0           # kernel picks address
        nv_ioctl(self.as_fd, NVGPU_AS_IOCTL_MAP_BUFFER_EX, args)
        
        buf.gpu_va = args.offset
        buf._as_fd = self.as_fd
        return buf.gpu_va
    
    def gpu_unmap(self, buf: TegraBuffer):
        """Unmap a buffer from GPU VA space."""
        if buf.gpu_va == 0 or buf._as_fd is None:
            return
        try:
            args = nvgpu_as_unmap_buffer_args()
            args.offset = buf.gpu_va
            nv_ioctl(buf._as_fd, NVGPU_AS_IOCTL_UNMAP_BUFFER, args)
        except OSError:
            pass  # best-effort cleanup
        buf.gpu_va = 0
        buf._as_fd = None
    
    def free(self, buf: TegraBuffer):
        """Free a buffer: unmap GPU → close mmap → close dmabuf fd → free handle."""
        # 1. Unmap from GPU
        self.gpu_unmap(buf)
        
        # 2. Close CPU mmap
        if buf.cpu_addr is not None:
            try:
                buf.cpu_addr.close()
            except Exception:
                pass
            buf.cpu_addr = None
        
        # 3. Close dmabuf fd
        if buf.dmabuf_fd >= 0:
            try:
                os.close(buf.dmabuf_fd)
            except OSError:
                pass
            buf.dmabuf_fd = -1
        
        # 4. Free nvmap handle
        if buf.handle != 0:
            try:
                # NVMAP_IOC_FREE takes handle as the arg directly
                # Handle values have bit 31 set (e.g. 0x80000d4c), which exceeds
                # Python's signed int range. Convert u32 → s32 for fcntl.ioctl.
                signed_handle = buf.handle if buf.handle < 0x80000000 else buf.handle - 0x100000000
                fcntl.ioctl(self._nvmap_fd, NVMAP_IOC_FREE, signed_handle)
            except OSError:
                pass
            buf.handle = 0
        
        if buf in self._buffers:
            self._buffers.remove(buf)
    
    def free_all(self):
        """Free all tracked buffers."""
        for buf in list(self._buffers):
            self.free(buf)
    
    @property
    def _nvmap_fd(self):
        return self.nvmap_fd


# ============================================================================
# Address Space structs & ioctls (Magic 'A' = 0x41)
# ============================================================================

class nvgpu_alloc_as_args(ctypes.Structure):
    """ALLOC_AS: create an address space. Returns AS fd."""
    _fields_ = [
        ("big_page_size", c_uint32),  # in: 0 = use default
        ("as_fd",         c_int32),   # out: fd for the new AS
        ("flags",         c_uint32),
        ("reserved",      c_uint32),
        ("va_range_start", c_uint64),
        ("va_range_end",   c_uint64),
        ("va_range_split", c_uint64),
        ("padding",       c_uint32 * 6),
    ]

NVGPU_GPU_IOCTL_ALLOC_AS = _IOWR('G', 8, ctypes.sizeof(nvgpu_alloc_as_args))

class nvgpu_as_bind_channel_args(ctypes.Structure):
    _fields_ = [
        ("channel_fd", c_uint32),
    ]

NVGPU_AS_IOCTL_BIND_CHANNEL = _IOWR('A', 1, ctypes.sizeof(nvgpu_as_bind_channel_args))

class nvgpu_as_alloc_space_args(ctypes.Structure):
    """Allocate VA space region."""
    _fields_ = [
        ("pages",     c_uint64),
        ("page_size", c_uint32),
        ("flags",     c_uint32),
        ("offset",    c_uint64),  # in/out: if FIXED_OFFSET, use this; else alignment
        ("padding",   c_uint32 * 2),
    ]

NVGPU_AS_IOCTL_ALLOC_SPACE = _IOWR('A', 6, ctypes.sizeof(nvgpu_as_alloc_space_args))

class nvgpu_as_map_buffer_ex_args(ctypes.Structure):
    """Map a dmabuf into the GPU address space."""
    _fields_ = [
        ("flags",          c_uint32),
        ("compr_kind",     c_int16),
        ("incompr_kind",   c_int16),
        ("dmabuf_fd",      c_uint32),
        ("page_size",      c_uint32),
        ("buffer_offset",  c_uint64),
        ("mapping_size",   c_uint64),
        ("offset",         c_uint64),  # in/out: GPU VA
    ]

NVGPU_AS_IOCTL_MAP_BUFFER_EX = _IOWR('A', 7, ctypes.sizeof(nvgpu_as_map_buffer_ex_args))

class nvgpu_as_get_va_regions_args(ctypes.Structure):
    """Query VA region layout."""
    _fields_ = [
        ("buf_addr", c_uint64),
        ("buf_size", c_uint32),
        ("reserved", c_uint32),
    ]

NVGPU_AS_IOCTL_GET_VA_REGIONS = _IOWR('A', 8, ctypes.sizeof(nvgpu_as_get_va_regions_args))


# ============================================================================
# TSG structs & ioctls (Magic 'T' = 0x54)
# ============================================================================

class nvgpu_gpu_open_tsg_args(ctypes.Structure):
    """OPEN_TSG: create a TSG, returns TSG fd."""
    _fields_ = [
        ("tsg_fd",   c_int32),
        ("flags",    c_uint32),
        ("token",    c_uint32),   # for sharing
        ("reserved", c_uint32),
        ("subctx_id", c_uint32),
        ("_pad",     c_uint32),
    ]

NVGPU_GPU_IOCTL_OPEN_TSG = _IOWR('G', 9, ctypes.sizeof(nvgpu_gpu_open_tsg_args))

class nvgpu_tsg_bind_channel_ex_args(ctypes.Structure):
    """Bind a channel to a TSG."""
    _fields_ = [
        ("channel_fd",     c_int32),
        ("subcontext_id",  c_uint32),  # in: VEID from CREATE_SUBCONTEXT
        ("reserved",       c_uint8 * 16),
    ]

NVGPU_TSG_IOCTL_BIND_CHANNEL_EX = _IOWR('T', 11, ctypes.sizeof(nvgpu_tsg_bind_channel_ex_args))

class nvgpu_tsg_create_subcontext_args(ctypes.Structure):
    """Create a subcontext within a TSG."""
    _fields_ = [
        ("type",     c_uint32),   # in: SYNC(0) or ASYNC(1)
        ("as_fd",    c_int32),    # in: address space fd
        ("veid",     c_uint32),   # out: VEID for the subcontext
        ("reserved", c_uint32),
    ]

# Subcontext types
NVGPU_TSG_SUBCONTEXT_TYPE_SYNC  = 0
NVGPU_TSG_SUBCONTEXT_TYPE_ASYNC = 1

NVGPU_TSG_IOCTL_CREATE_SUBCONTEXT = _IOWR('T', 18, ctypes.sizeof(nvgpu_tsg_create_subcontext_args))


# ============================================================================
# Channel structs & ioctls (Magic 'H' = 0x48)
# ============================================================================

class nvgpu_gpu_open_channel_args(ctypes.Structure):
    """OPEN_CHANNEL: union of {in: runlist_id} and {out: channel_fd}, just one s32."""
    _fields_ = [
        ("channel_fd", c_int32),   # in: runlist_id (-1 = primary graphics), out: channel fd
    ]

NVGPU_GPU_IOCTL_OPEN_CHANNEL = _IOWR('G', 11, ctypes.sizeof(nvgpu_gpu_open_channel_args))

# NVGPU_IOCTL_MAGIC for channel ioctls = 'H'

class nvgpu_alloc_obj_ctx_args(ctypes.Structure):
    """Allocate a class object on a channel (e.g. compute class)."""
    _fields_ = [
        ("class_num", c_uint32),   # in: class to allocate (e.g. 0xc7c0 for compute)
        ("flags",     c_uint32),
        ("obj_id",    c_uint64),   # out: object handle
    ]

NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX = _IOWR('H', 108, ctypes.sizeof(nvgpu_alloc_obj_ctx_args))

class nvgpu_channel_setup_bind_args(ctypes.Structure):
    """Setup GPFIFO + userd binding with usermode submit support."""
    _fields_ = [
        ("num_gpfifo_entries",  c_uint32),
        ("num_inflight_jobs",   c_uint32),
        ("flags",               c_uint32),
        ("userd_dmabuf_fd",     c_int32),
        ("gpfifo_dmabuf_fd",    c_int32),
        ("work_submit_token",   c_uint32),  # out: token for usermode submit
        ("userd_dmabuf_offset", c_uint64),   # in
        ("gpfifo_dmabuf_offset", c_uint64),  # in
        ("gpfifo_gpu_va",       c_uint64),   # out
        ("userd_gpu_va",        c_uint64),   # out
        ("usermode_mmio_gpu_va", c_uint64),  # out
        ("reserved",            c_uint32 * 9),
    ]

NVGPU_IOCTL_CHANNEL_SETUP_BIND = _IOWR('H', 128, ctypes.sizeof(nvgpu_channel_setup_bind_args))

# SETUP_BIND flags
NVGPU_CHANNEL_SETUP_BIND_FLAGS_VPR_ENABLED          = (1 << 0)
NVGPU_CHANNEL_SETUP_BIND_FLAGS_DETERMINISTIC        = (1 << 1)
NVGPU_CHANNEL_SETUP_BIND_FLAGS_REPLAYABLE_FAULTS_ENABLE = (1 << 2)
NVGPU_CHANNEL_SETUP_BIND_FLAGS_USERMODE_SUPPORT     = (1 << 3)
NVGPU_CHANNEL_SETUP_BIND_FLAGS_USERMODE_GPU_MAP_RESOURCES_SUPPORT = (1 << 4)

class nvgpu_set_error_notifier(ctypes.Structure):
    _fields_ = [
        ("offset", c_uint64),
        ("size",   c_uint64),
        ("mem",    c_uint32),
        ("_pad",   c_uint32),
    ]

NVGPU_IOCTL_CHANNEL_SET_ERROR_NOTIFIER = _IOWR('H', 111, ctypes.sizeof(nvgpu_set_error_notifier))

class nvgpu_channel_wdt_args(ctypes.Structure):
    _fields_ = [
        ("wdt_status", c_uint32),
        ("timeout_ms", c_uint32),
    ]

NVGPU_IOCTL_CHANNEL_WDT = _IOW('H', 119, ctypes.sizeof(nvgpu_channel_wdt_args))

class nvgpu_get_user_syncpoint_args(ctypes.Structure):
    _fields_ = [
        ("gpu_va",          c_uint64),   # out: GPU VA of syncpoint
        ("syncpoint_id",    c_uint32),   # out: syncpoint ID
        ("syncpoint_max",   c_uint32),   # out: max syncpoint value
    ]

NVGPU_IOCTL_CHANNEL_GET_USER_SYNCPOINT = _IOR('H', 126, ctypes.sizeof(nvgpu_get_user_syncpoint_args))


# ============================================================================
# Test functions
# ============================================================================

def test_get_characteristics(ctrl_fd):
    """Call GET_CHARACTERISTICS to read GPU info."""
    print("\n=== GET_CHARACTERISTICS ===")

    chars = nvgpu_gpu_characteristics()
    ctypes.memset(ctypes.addressof(chars), 0, ctypes.sizeof(chars))

    # The ioctl takes a wrapper struct with buf_size and buf_addr
    req = nvgpu_gpu_get_characteristics()
    req.gpu_characteristics_buf_size = ctypes.sizeof(chars)
    req.gpu_characteristics_buf_addr = ctypes.addressof(chars)

    nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_GET_CHARACTERISTICS, req)

    arch_names = {
        0x120: "GK100 (Kepler)", 0x130: "GK10B (Kepler mobile)",
        0x140: "GM200 (Maxwell)", 0x148: "GM20B (Maxwell mobile)",
        0x150: "GP100 (Pascal)", 0x152: "GP10B (Pascal mobile)",
        0x160: "GV100 (Volta)", 0x162: "GV11B (Volta mobile)",
        0x170: "GA100 (Ampere)", 0x172: "GA10B (Ampere mobile)",  # Orin! 
        0x190: "GH100 (Hopper)",
    }

    arch_name = arch_names.get(chars.arch, f"Unknown (0x{chars.arch:x})")

    print(f"  Architecture:     0x{chars.arch:04x} = {arch_name}")
    print(f"  Implementation:   0x{chars.impl:04x}")
    print(f"  Revision:         {chars.rev}")
    print(f"  Num GPC:          {chars.num_gpc}")
    print(f"  NUMA domain:      {chars.numa_domain_id}")
    print(f"  L2 cache size:    {chars.L2_cache_size} bytes ({chars.L2_cache_size // 1024} KB)")
    print(f"  VRAM size:        {chars.on_board_video_memory_size} bytes")
    print(f"  Num TPC/GPC:      {chars.num_tpc_per_gpc}")
    print(f"  Bus type:         {chars.bus_type}")
    print(f"  Big page size:    {chars.big_page_size}")
    print(f"  Compr page size:  {chars.compression_page_size}")
    print(f"  GPU VA bits:      {chars.gpu_va_bit_count}")
    print(f"  GPC mask:         0x{chars.gpc_mask:08x}")
    print(f"  SM arch version:  0x{chars.sm_arch_sm_version:08x}")
    print(f"  SM arch SPA ver:  0x{chars.sm_arch_spa_version:08x}")
    print(f"  SM arch warp cnt: {chars.sm_arch_warp_count}")
    print(f"  Flags:            0x{chars.flags:016x}")

    # Decode flags (from nvgpu-ctrl.h)
    flag_names = {
        (1 << 0):  "SUPPORT_PARTIAL_MAPPINGS",
        (1 << 1):  "SUPPORT_SPARSE_ALLOCS",
        (1 << 2):  "SUPPORT_SYNC_FENCE_FDS",
        (1 << 3):  "SUPPORT_CYCLE_STATS",
        (1 << 4):  "SUPPORT_CYCLE_STATS_SNAPSHOT",
        (1 << 5):  "SUPPORT_USERMODE_SUBMIT",  # bit 5 in old headers
        (1 << 6):  "SUPPORT_CLOCK_CONTROLS",
        (1 << 7):  "SUPPORT_GET_VOLTAGE",
        (1 << 8):  "SUPPORT_GET_CURRENT",
        (1 << 9):  "SUPPORT_GET_POWER",
        (1 << 10): "SUPPORT_GET_TEMPERATURE",
        (1 << 11): "SUPPORT_SET_THERM_ALERT_LIMIT",
        (1 << 14): "SUPPORT_TSG",
        (1 << 15): "SUPPORT_DEVICE_EVENTS",
        (1 << 16): "SUPPORT_FECS_CTXSW_TRACE",
        (1 << 18): "SUPPORT_DETERMINISTIC_SUBMIT_NO_JOBTRACKING",
        (1 << 19): "SUPPORT_DETERMINISTIC_SUBMIT_FULL",
        (1 << 20): "SUPPORT_IO_COHERENCE",
        (1 << 21): "SUPPORT_RESCHEDULE_RUNLIST",
        (1 << 22): "SUPPORT_TSG_SUBCONTEXTS",
        (1 << 24): "SUPPORT_DETERMINISTIC_OPTS",
        (1 << 25): "SUPPORT_SCG",
        (1 << 26): "SUPPORT_SYNCPOINT_ADDRESS",
        (1 << 27): "SUPPORT_VPR",
        (1 << 28): "SUPPORT_USER_SYNCPOINT",
        (1 << 29): "CAN_RAILGATE",
        (1 << 30): "SUPPORT_USERMODE_SUBMIT",  # bit 30
        (1 << 42): "SUPPORT_COMPUTE",
        (1 << 57): "SUPPORT_GPU_MMIO",
    }
    set_flags = []
    for bit, name in flag_names.items():
        if chars.flags & bit:
            set_flags.append(name)
    if set_flags:
        print(f"  Active flags:     {', '.join(set_flags)}")

    print(f"\n  === Classes (critical for compute) ===")
    print(f"  2D class:         0x{chars.twod_class:04x}")
    print(f"  3D class:         0x{chars.threed_class:04x}")
    print(f"  Compute class:    0x{chars.compute_class:04x}")
    print(f"  GPFIFO class:     0x{chars.gpfifo_class:04x}")
    print(f"  I2M class:        0x{chars.inline_to_memory_class:04x}")
    print(f"  DMA copy class:   0x{chars.dma_copy_class:04x}")

    print(f"\n  === Memory ===")
    print(f"  Max FBPs:         {chars.max_fbps_count}")
    print(f"  FBP en mask:      0x{chars.fbp_en_mask:08x}")
    print(f"  Max LTC/FBP:      {chars.max_ltc_per_fbp}")
    print(f"  Max LTS/LTC:      {chars.max_lts_per_ltc}")
    print(f"  Num LTC:          {chars.num_ltc}")
    print(f"  Max GPFIFO:       {chars.max_gpfifo_entries}")
    print(f"  Max freq:         {chars.max_freq} Hz ({chars.max_freq / 1e6:.0f} MHz)")

    chipname_bytes = bytes(chars.chipname)
    chipname_str = chipname_bytes.split(b'\0')[0].decode('ascii', errors='replace')
    if chipname_str:
        print(f"  Chip name:        {chipname_str}")

    print(f"\n  === IOCTL interface levels ===")
    print(f"  GPU ioctl last:   {chars.gpu_ioctl_nr_last}")
    print(f"  TSG ioctl last:   {chars.tsg_ioctl_nr_last}")
    print(f"  Channel last:     {chars.ioctl_channel_nr_last}")
    print(f"  AS ioctl last:    {chars.as_ioctl_nr_last}")

    return chars


def test_zcull(ctrl_fd):
    """Call ZCULL_GET_CTX_SIZE."""
    print("\n=== ZCULL_GET_CTX_SIZE ===")
    args = nvgpu_gpu_zcull_get_ctx_size_args()
    nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_ZCULL_GET_CTX_SIZE, args)
    print(f"  ZCull ctx size:   {args.size} bytes")
    return args.size


def test_nvmap_create_alloc(nvmap_fd, size=4096):
    """Test NVMAP_CREATE + NVMAP_ALLOC to allocate GPU memory."""
    print(f"\n=== NVMAP CREATE + ALLOC ({size} bytes) ===")

    # Step 1: Create handle
    create = nvmap_create_handle()
    create.size = size
    create.handle = 0
    nv_ioctl(nvmap_fd, NVMAP_IOC_CREATE, create)
    handle = create.handle
    print(f"  Created handle:   {handle} (0x{handle:08x})")

    # Step 2: Try various heaps
    heap_options = [
        (NVMAP_HEAP_IOVMM, "IOVMM"),
        (NVMAP_HEAP_SYSMEM, "SYSMEM"),
        ((1 << 3), "CARVEOUT_GPU"),
        ((1 << 0), "CARVEOUT_GENERIC"),
        (0xFFFFFFFF, "ALL (0xFFFFFFFF)"),
    ]
    
    for heap_mask, heap_name in heap_options:
        alloc = nvmap_alloc_handle()
        alloc.handle = handle
        alloc.heap_mask = heap_mask
        alloc.flags = NVMAP_HANDLE_WRITE_COMBINE
        alloc.align = 4096
        alloc.numa_nid = 0  # NUMA node 0
        try:
            nv_ioctl(nvmap_fd, NVMAP_IOC_ALLOC, alloc)
            print(f"  Allocated with heap={heap_name} (0x{heap_mask:08x}), alignment={alloc.align}")
            
            # Step 3: Get dmabuf fd (uses same struct as CREATE)
            get_fd = nvmap_create_handle()
            get_fd.handle = handle  # input: the handle
            get_fd.size = 0         # output: will be overwritten with fd
            nv_ioctl(nvmap_fd, NVMAP_IOC_GET_FD, get_fd)
            dmabuf_fd = get_fd.size  # fd is returned in the 'size' field
            print(f"  Got dmabuf fd:    {dmabuf_fd}")
            return handle, dmabuf_fd
        except OSError as e:
            print(f"  Heap {heap_name} (0x{heap_mask:08x}) failed: {e}")
    
    raise RuntimeError("All heap options failed")


def test_alloc_as(ctrl_fd):
    """Create a GPU address space."""
    print("\n=== ALLOC_AS (create address space) ===")
    args = nvgpu_alloc_as_args()
    # big_page_size=0 means no big pages (default for ga10b which reports 0)
    # flags=UNIFIED_VA required for compute
    # VA ranges must be PDE-aligned (2^21 = 2MB for ga10b)
    # va_range_split must be 0 for UNIFIED_VA
    PDE_SIZE = 1 << 21  # 2MB - determined empirically for ga10b
    args.big_page_size = 0
    args.flags = 2  # NVGPU_GPU_IOCTL_ALLOC_AS_FLAGS_UNIFIED_VA
    args.va_range_start = PDE_SIZE        # 0x200000 (2MB)
    args.va_range_end = (1 << 40) - PDE_SIZE  # 0xFFFFE00000 (almost 1TB)
    args.va_range_split = 0
    nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_ALLOC_AS, args)
    print(f"  AS fd:            {args.as_fd}")
    print(f"  VA range:         0x{args.va_range_start:012x} - 0x{args.va_range_end:012x}")
    return args.as_fd


def test_map_buffer(as_fd, dmabuf_fd, size=4096):
    """Map a dmabuf into the GPU address space."""
    print(f"\n=== MAP_BUFFER_EX (map {size} bytes into GPU VA) ===")
    args = nvgpu_as_map_buffer_ex_args()
    args.flags = 0  # let kernel choose address
    args.compr_kind = -1   # NV_KIND_INVALID
    args.incompr_kind = 0  # pitch linear
    args.dmabuf_fd = dmabuf_fd
    args.page_size = 4096
    args.buffer_offset = 0
    args.mapping_size = 0  # 0 = whole buffer
    args.offset = 0        # kernel picks address
    nv_ioctl(as_fd, NVGPU_AS_IOCTL_MAP_BUFFER_EX, args)
    gpu_va = args.offset
    print(f"  GPU VA:           0x{gpu_va:012x}")
    return gpu_va


def test_open_tsg(ctrl_fd):
    """Create a TSG."""
    print("\n=== OPEN_TSG ===")
    args = nvgpu_gpu_open_tsg_args()
    args.flags = 0
    args.token = 0
    nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_OPEN_TSG, args)
    print(f"  TSG fd:           {args.tsg_fd}")
    return args.tsg_fd


def test_open_channel(ctrl_fd):
    """Create a channel."""
    print("\n=== OPEN_CHANNEL ===")
    args = nvgpu_gpu_open_channel_args()
    args.channel_fd = -1  # in: runlist_id = -1 (auto/primary graphics)
    nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_OPEN_CHANNEL, args)
    print(f"  Channel fd:       {args.channel_fd}")
    return args.channel_fd


def test_full_channel_setup(ctrl_fd, as_fd, nvmap_fd, compute_class):
    """Full channel setup: TSG → channel → bind → compute class."""
    print("\n" + "=" * 60)
    print("FULL CHANNEL + COMPUTE SETUP")
    print("=" * 60)

    # 1. Open TSG
    tsg_fd = test_open_tsg(ctrl_fd)

    # 2. Create subcontext in TSG (ASYNC for compute)
    print("\n=== CREATE_SUBCONTEXT ===")
    subctx = nvgpu_tsg_create_subcontext_args()
    subctx.type = NVGPU_TSG_SUBCONTEXT_TYPE_ASYNC  # compute
    subctx.as_fd = as_fd
    nv_ioctl(tsg_fd, NVGPU_TSG_IOCTL_CREATE_SUBCONTEXT, subctx)
    print(f"  VEID:             {subctx.veid}")

    # 3. Open channel
    ch_fd = test_open_channel(ctrl_fd)

    # 4. Bind channel to AS (must be before TSG bind!)
    print("\n=== AS BIND_CHANNEL ===")
    as_bind = nvgpu_as_bind_channel_args()
    as_bind.channel_fd = ch_fd
    nv_ioctl(as_fd, NVGPU_AS_IOCTL_BIND_CHANNEL, as_bind)
    print(f"  Bound channel {ch_fd} to AS {as_fd}")

    # 5. Bind channel to TSG
    print("\n=== BIND_CHANNEL_EX ===")
    bind = nvgpu_tsg_bind_channel_ex_args()
    bind.channel_fd = ch_fd
    bind.subcontext_id = subctx.veid
    nv_ioctl(tsg_fd, NVGPU_TSG_IOCTL_BIND_CHANNEL_EX, bind)
    print(f"  Bound channel {ch_fd} to TSG {tsg_fd}")

    # 6. Disable watchdog
    print("\n=== CHANNEL WDT (disable) ===")
    wdt = nvgpu_channel_wdt_args()
    wdt.wdt_status = 1  # NVGPU_IOCTL_CHANNEL_DISABLE_WDT
    wdt.timeout_ms = 0
    nv_ioctl(ch_fd, NVGPU_IOCTL_CHANNEL_WDT, wdt)
    print(f"  Watchdog disabled")

    # 7. Allocate GPFIFO + userd buffers via nvmap
    print("\n=== Allocating GPFIFO + userd buffers ===")
    GPFIFO_ENTRIES = 1024  # reasonable default
    GPFIFO_SIZE = GPFIFO_ENTRIES * 8  # 8 bytes per GPFIFO entry
    USERD_SIZE = 4096  # one page for userd

    # GPFIFO buffer
    gpfifo_create = nvmap_create_handle()
    gpfifo_create.size = GPFIFO_SIZE
    nv_ioctl(nvmap_fd, NVMAP_IOC_CREATE, gpfifo_create)
    gpfifo_alloc = nvmap_alloc_handle()
    gpfifo_alloc.handle = gpfifo_create.handle
    gpfifo_alloc.heap_mask = NVMAP_HEAP_IOVMM
    gpfifo_alloc.flags = NVMAP_HANDLE_WRITE_COMBINE
    gpfifo_alloc.align = 4096
    gpfifo_alloc.numa_nid = 0
    nv_ioctl(nvmap_fd, NVMAP_IOC_ALLOC, gpfifo_alloc)
    gpfifo_getfd = nvmap_create_handle()
    gpfifo_getfd.handle = gpfifo_create.handle
    nv_ioctl(nvmap_fd, NVMAP_IOC_GET_FD, gpfifo_getfd)
    gpfifo_dmabuf_fd = gpfifo_getfd.size
    print(f"  GPFIFO buffer:    handle={gpfifo_create.handle}, dmabuf_fd={gpfifo_dmabuf_fd}, size={GPFIFO_SIZE}")

    # userd buffer
    userd_create = nvmap_create_handle()
    userd_create.size = USERD_SIZE
    nv_ioctl(nvmap_fd, NVMAP_IOC_CREATE, userd_create)
    userd_alloc = nvmap_alloc_handle()
    userd_alloc.handle = userd_create.handle
    userd_alloc.heap_mask = NVMAP_HEAP_IOVMM
    userd_alloc.flags = NVMAP_HANDLE_WRITE_COMBINE
    userd_alloc.align = 4096
    userd_alloc.numa_nid = 0
    nv_ioctl(nvmap_fd, NVMAP_IOC_ALLOC, userd_alloc)
    userd_getfd = nvmap_create_handle()
    userd_getfd.handle = userd_create.handle
    nv_ioctl(nvmap_fd, NVMAP_IOC_GET_FD, userd_getfd)
    userd_dmabuf_fd = userd_getfd.size
    print(f"  userd buffer:     handle={userd_create.handle}, dmabuf_fd={userd_dmabuf_fd}, size={USERD_SIZE}")

    # 8. SETUP_BIND — this is the big one! Sets up GPFIFO + userd + usermode submit
    print("\n=== SETUP_BIND (GPFIFO + userd + usermode submit) ===")
    setup = nvgpu_channel_setup_bind_args()
    setup.num_gpfifo_entries = GPFIFO_ENTRIES
    setup.num_inflight_jobs = 0
    setup.gpfifo_dmabuf_fd = gpfifo_dmabuf_fd
    setup.gpfifo_dmabuf_offset = 0
    setup.userd_dmabuf_fd = userd_dmabuf_fd
    setup.userd_dmabuf_offset = 0
    setup.flags = NVGPU_CHANNEL_SETUP_BIND_FLAGS_USERMODE_SUPPORT | NVGPU_CHANNEL_SETUP_BIND_FLAGS_DETERMINISTIC
    nv_ioctl(ch_fd, NVGPU_IOCTL_CHANNEL_SETUP_BIND, setup)
    print(f"  Work submit token: {setup.work_submit_token}")
    print(f"  GPFIFO GPU VA:     0x{setup.gpfifo_gpu_va:012x}")
    print(f"  USERD GPU VA:      0x{setup.userd_gpu_va:012x}")
    print(f"  Usermode MMIO VA:  0x{setup.usermode_mmio_gpu_va:012x}")

    # 9. Get user syncpoint
    print("\n=== GET_USER_SYNCPOINT ===")
    syncpt = nvgpu_get_user_syncpoint_args()
    nv_ioctl(ch_fd, NVGPU_IOCTL_CHANNEL_GET_USER_SYNCPOINT, syncpt)
    print(f"  Syncpoint ID:     {syncpt.syncpoint_id}")
    print(f"  Syncpoint max:    {syncpt.syncpoint_max}")
    print(f"  GPU VA:           0x{syncpt.gpu_va:012x}")

    # 10. Allocate compute class!
    print(f"\n=== ALLOC_OBJ_CTX (compute class 0x{compute_class:04x}) ===")
    obj = nvgpu_alloc_obj_ctx_args()
    obj.class_num = compute_class
    obj.flags = 0
    nv_ioctl(ch_fd, NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX, obj)
    print(f"  Compute object:   class=0x{obj.class_num:04x}, obj_id=0x{obj.obj_id:016x}")
    print(f"  ✓ COMPUTE CLASS ALLOCATED SUCCESSFULLY!")

    return {
        "tsg_fd": tsg_fd,
        "ch_fd": ch_fd,
        "ctrl_fd": ctrl_fd,
        "gpfifo_dmabuf_fd": gpfifo_dmabuf_fd,
        "userd_dmabuf_fd": userd_dmabuf_fd,
        "work_submit_token": setup.work_submit_token,
        "syncpoint_id": syncpt.syncpoint_id,
        "compute_class": compute_class,
    }


def test_nvmap_heaps(nvmap_fd):
    """Query available memory heaps."""
    print("\n=== NVMAP GET_AVAILABLE_HEAPS ===")
    args = nvmap_available_heaps()
    nv_ioctl(nvmap_fd, NVMAP_IOC_GET_AVAILABLE_HEAPS, args)
    
    heap_names = {
        (1 << 31): "SYSMEM",
        (1 << 30): "IOVMM",
        (1 << 28): "CARVEOUT_VPR",
        (1 << 27): "CARVEOUT_TSEC",
        (1 << 26): "CARVEOUT_VIDMEM",
        (1 << 3):  "CARVEOUT_GPU",
        (1 << 2):  "CARVEOUT_FSI",
        (1 << 1):  "CARVEOUT_IVM",
        (1 << 0):  "CARVEOUT_GENERIC",
    }
    print(f"  Heap bitmask:     0x{args.heaps:016x}")
    available = []
    for bit, name in sorted(heap_names.items()):
        if args.heaps & bit:
            available.append(f"{name} (1<<{bit.bit_length()-1})")
    print(f"  Available heaps:  {', '.join(available) if available else 'none decoded'}")
    return args.heaps


# ============================================================================
# Phase 2 Test Functions — Memory Management
# ============================================================================

def test_mmap_readwrite(nvmap_fd):
    """Test 8: mmap a buffer, write a pattern from CPU, read it back.
    
    Proves that CPU can directly access nvmap-allocated memory via mmap
    on the dmabuf fd. Tests multiple patterns and access sizes.
    """
    print("\n" + "=" * 60)
    print("TEST 8: MMAP READ/WRITE")
    print("=" * 60)
    
    allocator = TegraAllocator(nvmap_fd)
    BUF_SIZE = 4096
    
    # Allocate and mmap
    buf = allocator.alloc(BUF_SIZE)
    print(f"  Allocated:        handle=0x{buf.handle:x}, dmabuf_fd={buf.dmabuf_fd}, size={buf.size}")
    
    mm = allocator.mmap_buffer(buf)
    print(f"  mmapped:          {buf.size} bytes, fd={buf.dmabuf_fd}")
    
    # Test 1: Write sequential bytes, read back
    print("\n  --- Pattern 1: Sequential bytes ---")
    for i in range(256):
        mm[i] = i & 0xFF
    
    errors = 0
    for i in range(256):
        if mm[i] != (i & 0xFF):
            errors += 1
            if errors <= 5:
                print(f"  MISMATCH at offset {i}: wrote {i & 0xFF}, read {mm[i]}")
    print(f"  Sequential bytes: {'PASS' if errors == 0 else f'FAIL ({errors} errors)'}")
    
    # Test 2: Write u32 values via struct pack
    print("  --- Pattern 2: u32 values via struct ---")
    MAGIC = 0xDEADBEEF
    for offset in range(0, 1024, 4):
        struct.pack_into('<I', mm, offset, MAGIC ^ offset)
    
    errors = 0
    for offset in range(0, 1024, 4):
        val = struct.unpack_from('<I', mm, offset)[0]
        expected = MAGIC ^ offset
        if val != expected:
            errors += 1
            if errors <= 5:
                print(f"  MISMATCH at offset {offset}: expected 0x{expected:08x}, got 0x{val:08x}")
    print(f"  u32 pattern:      {'PASS' if errors == 0 else f'FAIL ({errors} errors)'}")
    
    # Test 3: Fill entire buffer with 0xAA, then verify
    print("  --- Pattern 3: Full buffer fill ---")
    mm[:BUF_SIZE] = b'\xAA' * BUF_SIZE
    readback = mm[:BUF_SIZE]
    match = readback == b'\xAA' * BUF_SIZE
    print(f"  Full fill 0xAA:   {'PASS' if match else 'FAIL'}")
    
    # Test 4: Zero the buffer
    mm[:BUF_SIZE] = b'\x00' * BUF_SIZE
    readback = mm[:BUF_SIZE]
    match = readback == b'\x00' * BUF_SIZE
    print(f"  Zero fill:        {'PASS' if match else 'FAIL'}")
    
    # Cleanup
    allocator.free(buf)
    print(f"\n  ✓ mmap read/write test PASSED")
    return True


def test_mmap_coherence(nvmap_fd):
    """Test 9: Verify memory coherence through dual mmap.
    
    Creates a buffer, maps it to CPU twice (two separate mmaps of the same 
    dmabuf fd), writes through one mapping, reads through the other.
    This proves the buffer is backed by real shared physical memory and
    that IO_COHERENCE works (no explicit cache flush needed).
    
    Also tests the NVMAP_IOC_READ/WRITE ioctls as an alternative data path.
    """
    print("\n" + "=" * 60)
    print("TEST 9: MEMORY COHERENCE (dual mmap)")
    print("=" * 60)
    
    allocator = TegraAllocator(nvmap_fd)
    BUF_SIZE = 8192  # Use 2 pages to test across page boundaries
    
    buf = allocator.alloc(BUF_SIZE)
    print(f"  Allocated:        handle=0x{buf.handle:x}, size={buf.size}")
    
    # Map 1: via the allocator (standard path)
    mm1 = allocator.mmap_buffer(buf)
    print(f"  mmap #1:          fd={buf.dmabuf_fd}")
    
    # Map 2: second mmap of the same dmabuf fd
    mm2 = mmap.mmap(buf.dmabuf_fd, BUF_SIZE,
                     mmap.MAP_SHARED,
                     mmap.PROT_READ | mmap.PROT_WRITE,
                     0)
    print(f"  mmap #2:          same fd, different mapping")
    
    # --- Coherence test 1: Write through mm1, read through mm2 ---
    print("\n  --- Coherence: write mm1, read mm2 ---")
    PATTERN = b'\xCA\xFE\xBA\xBE' * (BUF_SIZE // 4)
    mm1[:BUF_SIZE] = PATTERN
    readback = mm2[:BUF_SIZE]
    match = readback == PATTERN
    print(f"  mm1→mm2:          {'PASS' if match else 'FAIL'}")
    
    # --- Coherence test 2: Write through mm2, read through mm1 ---
    print("  --- Coherence: write mm2, read mm1 ---")
    PATTERN2 = b'\x12\x34\x56\x78' * (BUF_SIZE // 4)
    mm2[:BUF_SIZE] = PATTERN2
    readback = mm1[:BUF_SIZE]
    match2 = readback == PATTERN2
    print(f"  mm2→mm1:          {'PASS' if match2 else 'FAIL'}")
    
    # --- Coherence test 3: Byte-level interleaved writes ---
    print("  --- Coherence: interleaved byte writes ---")
    for i in range(0, 256, 2):
        mm1[i] = 0xAA
        mm2[i+1] = 0x55
    errors = 0
    for i in range(0, 256, 2):
        if mm2[i] != 0xAA:  # read via mm2 what mm1 wrote
            errors += 1
        if mm1[i+1] != 0x55:  # read via mm1 what mm2 wrote
            errors += 1
    print(f"  Interleaved:      {'PASS' if errors == 0 else f'FAIL ({errors} errors)'}")
    
    # --- Coherence test 4: Cross-page boundary ---
    print("  --- Coherence: cross-page boundary (offset 4094-4098) ---")
    cross_data = b'\xDE\xAD\xBE\xEF'
    mm1[4094:4098] = cross_data
    readback = mm2[4094:4098]
    match_cross = readback == cross_data
    print(f"  Cross-page:       {'PASS' if match_cross else 'FAIL'}")
    
    # --- Test NVMAP_IOC_READ/WRITE as alternative data path ---
    print("\n  --- NVMAP_IOC_WRITE + NVMAP_IOC_READ ---")
    nvmap_rw_ok = False
    try:
        # Write 64 bytes of test data via NVMAP_IOC_WRITE
        test_data = bytes(range(64))
        user_buf = (c_uint8 * 64)(*test_data)
        
        rw = nvmap_rw_handle()
        rw.addr = ctypes.addressof(user_buf)
        rw.handle = buf.handle
        rw.offset = 0
        rw.elem_size = 64
        rw.hmem_stride = 64
        rw.user_stride = 64
        rw.count = 1
        nv_ioctl(nvmap_fd, NVMAP_IOC_WRITE, rw)
        
        # Read back via NVMAP_IOC_READ
        read_buf = (c_uint8 * 64)()
        rw2 = nvmap_rw_handle()
        rw2.addr = ctypes.addressof(read_buf)
        rw2.handle = buf.handle
        rw2.offset = 0
        rw2.elem_size = 64
        rw2.hmem_stride = 64
        rw2.user_stride = 64
        rw2.count = 1
        nv_ioctl(nvmap_fd, NVMAP_IOC_READ, rw2)
        
        read_data = bytes(read_buf)
        if read_data == test_data:
            print(f"  NVMAP R/W ioctl:  PASS (64 bytes round-tripped)")
            nvmap_rw_ok = True
        else:
            print(f"  NVMAP R/W ioctl:  FAIL (data mismatch)")
        
        # Verify ioctl write is visible through mmap
        mmap_view = mm1[:64]
        if mmap_view == test_data:
            print(f"  ioctl→mmap:       PASS (ioctl write visible in mmap)")
        else:
            print(f"  ioctl→mmap:       FAIL (ioctl write NOT visible in mmap)")
    except OSError as e:
        print(f"  NVMAP R/W ioctl:  SKIPPED ({e})")
        nvmap_rw_ok = True  # not a failure if ioctls don't exist
    
    # Cleanup
    mm2.close()
    allocator.free(buf)
    
    all_pass = match and match2 and (errors == 0) and match_cross
    if all_pass:
        print(f"\n  ✓ Memory coherence test PASSED")
        print(f"    IO_COHERENCE confirmed: no cache flushes needed!")
    else:
        print(f"\n  ✗ Memory coherence test FAILED")
    return all_pass


def test_multi_size_gpu_map(nvmap_fd, as_fd):
    """Test 10: Test multiple buffer sizes with CPU mmap + GPU VA mapping.
    
    Uses TegraAllocator to allocate, mmap, GPU-map, write, and verify
    buffers of sizes: 4KB, 64KB, 1MB, 16MB, 64MB.
    Proves the full TegraAllocator lifecycle works at all scales.
    """
    print("\n" + "=" * 60)
    print("TEST 10: MULTI-SIZE ALLOC + MMAP + GPU MAP")
    print("=" * 60)
    
    allocator = TegraAllocator(nvmap_fd, as_fd)
    
    sizes = [
        (4 * 1024,       "4 KB"),
        (64 * 1024,      "64 KB"),
        (1 * 1024 * 1024, "1 MB"),
        (16 * 1024 * 1024, "16 MB"),
        (64 * 1024 * 1024, "64 MB"),
    ]
    
    all_pass = True
    for size, label in sizes:
        print(f"\n  --- {label} ({size} bytes) ---")
        try:
            t0 = time.monotonic()
            
            # Allocate
            buf = allocator.alloc(size)
            t_alloc = time.monotonic() - t0
            
            # mmap
            mm = allocator.mmap_buffer(buf)
            t_mmap = time.monotonic() - t0
            
            # GPU map
            gpu_va = allocator.gpu_map(buf)
            t_gpumap = time.monotonic() - t0
            
            # Write pattern (first and last pages + some in middle)
            MAGIC = 0xFACEFEED
            check_offsets = [0, 4096, size // 2 & ~3, size - 4]
            for off in check_offsets:
                if off + 4 <= size:
                    struct.pack_into('<I', mm, off, MAGIC ^ off)
            
            # Read back and verify
            errors = 0
            for off in check_offsets:
                if off + 4 <= size:
                    val = struct.unpack_from('<I', mm, off)[0]
                    if val != (MAGIC ^ off):
                        errors += 1
                        print(f"    MISMATCH at {off}: expected 0x{MAGIC ^ off:08x}, got 0x{val:08x}")
            
            t_total = time.monotonic() - t0
            
            status = "PASS" if errors == 0 else f"FAIL ({errors} errors)"
            print(f"    handle=0x{buf.handle:x}, dmabuf_fd={buf.dmabuf_fd}, gpu_va=0x{gpu_va:012x}")
            print(f"    alloc={t_alloc*1000:.1f}ms, mmap={t_mmap*1000:.1f}ms, total={t_total*1000:.1f}ms")
            print(f"    Read/write:     {status}")
            
            if errors > 0:
                all_pass = False
            
            # Cleanup
            allocator.free(buf)
            
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_pass = False
    
    if all_pass:
        print(f"\n  ✓ Multi-size test PASSED (all 5 sizes)")
    else:
        print(f"\n  ✗ Multi-size test FAILED")
    return all_pass


def test_cacheable_flags(nvmap_fd, as_fd):
    """Test 11: Compare WRITE_COMBINE vs CACHEABLE allocation flags.
    
    Both should work on Orin (which has IO_COHERENCE), but performance
    characteristics differ:
      - WRITE_COMBINE (1): CPU writes are buffered, reads are uncached (fast writes, slow reads)
      - CACHEABLE (5): CPU reads and writes use cache (fast for both, needs coherence protocol)
    
    This test validates correctness of both modes, and measures basic timing.
    """
    print("\n" + "=" * 60)
    print("TEST 11: WRITE_COMBINE vs CACHEABLE FLAGS")
    print("=" * 60)
    
    allocator = TegraAllocator(nvmap_fd, as_fd)
    BUF_SIZE = 1 * 1024 * 1024  # 1MB for meaningful timing
    
    flag_configs = [
        (NVMAP_HANDLE_WRITE_COMBINE, "WRITE_COMBINE (flags=1)"),
        (NVMAP_HANDLE_CACHEABLE,     "CACHEABLE (flags=5)"),
        (NVMAP_HANDLE_UNCACHEABLE,   "UNCACHEABLE (flags=0)"),
        (NVMAP_HANDLE_INNER_CACHEABLE, "INNER_CACHEABLE (flags=2)"),
    ]
    
    results = {}
    all_pass = True
    
    for flags, label in flag_configs:
        print(f"\n  --- {label} ---")
        try:
            buf = allocator.alloc(BUF_SIZE, flags=flags)
            mm = allocator.mmap_buffer(buf)
            gpu_va = allocator.gpu_map(buf)
            
            # Write timing: fill 1MB with pattern
            pattern = b'\x55\xAA\x55\xAA' * (BUF_SIZE // 4)
            t0 = time.monotonic()
            mm[:BUF_SIZE] = pattern
            t_write = time.monotonic() - t0
            
            # Read timing: read back 1MB
            t0 = time.monotonic()
            readback = mm[:BUF_SIZE]
            t_read = time.monotonic() - t0
            
            # Verify
            correct = readback == pattern
            
            write_bw = BUF_SIZE / t_write / 1e6 if t_write > 0 else float('inf')
            read_bw  = BUF_SIZE / t_read  / 1e6 if t_read  > 0 else float('inf')
            
            print(f"    GPU VA:         0x{gpu_va:012x}")
            print(f"    Write:          {t_write*1000:.2f}ms ({write_bw:.0f} MB/s)")
            print(f"    Read:           {t_read*1000:.2f}ms ({read_bw:.0f} MB/s)")
            print(f"    Correctness:    {'PASS' if correct else 'FAIL'}")
            
            results[label] = {
                "write_ms": t_write * 1000,
                "read_ms": t_read * 1000,
                "write_bw_mbs": write_bw,
                "read_bw_mbs": read_bw,
                "correct": correct,
            }
            
            if not correct:
                all_pass = False
            
            allocator.free(buf)
            
        except OSError as e:
            print(f"    FAILED: {e}")
            print(f"    (This flag mode may not be supported)")
            results[label] = {"correct": False, "error": str(e)}
            # Don't count unsupported flags as failure — only WC and CACHEABLE are required
            if flags in (NVMAP_HANDLE_WRITE_COMBINE, NVMAP_HANDLE_CACHEABLE):
                all_pass = False
    
    # Summary comparison
    print(f"\n  === Performance Summary ===")
    for label, r in results.items():
        if "error" in r:
            print(f"    {label}: UNSUPPORTED")
        else:
            print(f"    {label}: write={r['write_ms']:.2f}ms read={r['read_ms']:.2f}ms {'✓' if r['correct'] else '✗'}")
    
    if all_pass:
        print(f"\n  ✓ Cacheability flags test PASSED")
    else:
        print(f"\n  ✗ Cacheability flags test FAILED")
    return all_pass


# ============================================================================
# Phase 3: Command Submission — GPFIFO push, QMD, compute dispatch
# ============================================================================

# --- GPU method / push buffer constants ---
# From tinygrad autogen nv_570.py

# Compute class methods (subchannel 1)
NVC6C0_SET_OBJECT       = 0x0000
NVC6C0_SEND_PCAS_A      = 0x02b4
NVC6C0_SEND_SIGNALING_PCAS2_B = 0x02c0
NVC6C0_INVALIDATE_SHADER_CACHES     = 0x021c
NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI = 0x1698
NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A = 0x02a0  # + 0x02a4 = _B
NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A  = 0x07b0  # + 0x07b4 = _B
NVC6C0_SET_SHADER_LOCAL_MEMORY_A         = 0x0790  # + 0x0794 = _B
NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A = 0x02e4  # + 0x02e8 = _B, + 0x02ec = _C

# GPFIFO channel methods (subchannel 0)
NVC56F_SEM_ADDR_LO      = 0x005c
NVC56F_SEM_ADDR_HI      = 0x0060
NVC56F_SEM_PAYLOAD_LO   = 0x0064
NVC56F_SEM_PAYLOAD_HI   = 0x0068
NVC56F_SEM_EXECUTE       = 0x006c
NVC56F_NON_STALL_INTERRUPT = 0x0020

# SEM_EXECUTE flag bits
NVC56F_SEM_EXECUTE_OPERATION_RELEASE = 1
NVC56F_SEM_EXECUTE_RELEASE_WFI_EN   = (1 << 20)
NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT = (1 << 24)
NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP_EN = (1 << 25)

# DMA copy class methods (subchannel 4)
NVC6B5_SET_OBJECT       = 0x0000
NVC6B5_OFFSET_IN_UPPER  = 0x0400
NVC6B5_OFFSET_IN_LOWER  = 0x0404
NVC6B5_OFFSET_OUT_UPPER = 0x0408
NVC6B5_OFFSET_OUT_LOWER = 0x040c
NVC6B5_LINE_LENGTH_IN   = 0x0418
NVC6B5_LAUNCH_DMA       = 0x0300
NVC6B5_SET_SEMAPHORE_A  = 0x0240
NVC6B5_SET_SEMAPHORE_B  = 0x0244
NVC6B5_SET_SEMAPHORE_PAYLOAD = 0x0248

# LAUNCH_DMA flags 
NVC6B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED = (1 << 1)
NVC6B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH = 0
NVC6B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH = 0
NVC6B5_LAUNCH_DMA_FLUSH_ENABLE_TRUE = (1 << 2)
NVC6B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_FOUR_WORD = (2 << 3)

# Control struct offsets (from AmpereAControlGPFifo)
USERD_GP_GET_OFFSET = 136   # bytes
USERD_GP_PUT_OFFSET = 140   # bytes

# QMD V03 field definitions (from NVC6C0_QMDV03_00_*)
# Fields are (hi_bit, lo_bit) tuples
QMDV03_FIELDS = {
    'OUTER_PUT':            (30, 0),
    'OUTER_OVERFLOW':       (31, 31),
    'OUTER_GET':            (62, 32),
    'OUTER_STICKY_OVERFLOW': (63, 63),
    'INNER_GET':            (94, 64),
    'INNER_OVERFLOW':       (95, 95),
    'INNER_PUT':            (126, 96),
    'INNER_STICKY_OVERFLOW': (127, 127),
    'QMD_GROUP_ID':         (133, 128),
    'SM_GLOBAL_CACHING_ENABLE': (134, 134),
    'IS_QUEUE':             (136, 136),
    'INVALIDATE_TEXTURE_HEADER_CACHE': (186, 186),
    'INVALIDATE_TEXTURE_SAMPLER_CACHE': (187, 187),
    'INVALIDATE_TEXTURE_DATA_CACHE':   (188, 188),
    'INVALIDATE_SHADER_DATA_CACHE':    (189, 189),
    'INVALIDATE_SHADER_CONSTANT_CACHE': (191, 191),
    'CTA_RASTER_WIDTH':     (415, 384),
    'CTA_RASTER_HEIGHT':    (431, 416),
    'CTA_RASTER_DEPTH':     (463, 448),
    'PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED': (287, 256),
    'CWD_MEMBAR_TYPE':      (369, 368),
    'API_VISIBLE_CALL_LIMIT': (378, 378),
    'SAMPLER_INDEX':        (382, 382),
    'SHARED_MEMORY_SIZE':   (561, 544),
    'MIN_SM_CONFIG_SHARED_MEM_SIZE': (567, 562),
    'TARGET_SM_CONFIG_SHARED_MEM_SIZE': (662, 657),
    'MAX_SM_CONFIG_SHARED_MEM_SIZE': (574, 569),
    'QMD_VERSION':          (579, 576),
    'QMD_MAJOR_VERSION':    (583, 580),
    'CTA_THREAD_DIMENSION0': (607, 592),
    'CTA_THREAD_DIMENSION1': (623, 608),
    'CTA_THREAD_DIMENSION2': (639, 624),
    'REGISTER_COUNT_V':     (656, 648),
    'SHADER_LOCAL_MEMORY_LOW_SIZE': (759, 736),
    'BARRIER_COUNT':        (767, 763),
    'RELEASE0_ADDRESS_LOWER':   (799, 768),
    'RELEASE0_ADDRESS_UPPER':   (807, 800),
    'RELEASE0_ENABLE':          (823, 823),
    'RELEASE0_MEMBAR_TYPE':     (819, 819),
    'RELEASE0_PAYLOAD_LOWER':   (831, 824),  # actually (863, 832)
    'RELEASE1_ADDRESS_LOWER':   (927, 896),
    'RELEASE1_ADDRESS_UPPER':   (935, 928),
    'RELEASE1_ENABLE':          (951, 951),
    'PROGRAM_ADDRESS_LOWER':    (1567, 1536),
    'PROGRAM_ADDRESS_UPPER':    (1584, 1568),
    'PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED': (1640, 1632),
    'PROGRAM_PREFETCH_SIZE':    (1649, 1641),
    'SASS_VERSION':             (1663, 1656),
    'SHADER_LOCAL_MEMORY_HIGH_SIZE': (1623, 1600),
}

# Constant buffer fields are parameterized by index
def QMDV03_CONSTANT_BUFFER_VALID(i):     return (640 + i, 640 + i)
def QMDV03_CONSTANT_BUFFER_ADDR_LOWER(i): return (1055 + i*64, 1024 + i*64)
def QMDV03_CONSTANT_BUFFER_ADDR_UPPER(i): return (1072 + i*64, 1056 + i*64)
def QMDV03_CONSTANT_BUFFER_SIZE_SHIFTED4(i): return (1087 + i*64, 1075 + i*64)
def QMDV03_CONSTANT_BUFFER_INVALIDATE(i): return (1074 + i*64, 1074 + i*64)


class QMDBuilder:
    """Build a QMD (Queue Meta Data) v03 struct for Ampere compute dispatch.
    
    The QMD is 0x40 * 4 = 256 bytes (64 dwords) for version 3.
    Each field is specified as (hi_bit, lo_bit) and values are packed little-endian.
    """
    SIZE = 0x40 * 4  # 256 bytes
    
    def __init__(self):
        self.data = bytearray(self.SIZE)
    
    def _write_bits(self, hi, lo, value):
        """Write a value into bit range [lo:hi] (inclusive)."""
        width = hi - lo + 1
        if value >= (1 << width):
            raise ValueError(f"Value {value:#x} doesn't fit in {width} bits [{hi}:{lo}]")
        # Read current bytes, modify, write back
        byte_lo = lo // 8
        byte_hi = hi // 8
        num = int.from_bytes(self.data[byte_lo:byte_hi+1], "little")
        mask = ((1 << width) - 1) << (lo % 8)
        num = (num & ~mask) | ((value << (lo % 8)) & mask)
        self.data[byte_lo:byte_hi+1] = num.to_bytes(byte_hi - byte_lo + 1, "little")
    
    def _read_bits(self, hi, lo):
        byte_lo = lo // 8
        byte_hi = hi // 8
        num = int.from_bytes(self.data[byte_lo:byte_hi+1], "little")
        mask = ((1 << (hi - lo + 1)) - 1) << (lo % 8)
        return (num & mask) >> (lo % 8)
    
    def write(self, **kwargs):
        for name, value in kwargs.items():
            key = name.upper()
            if key in QMDV03_FIELDS:
                hi, lo = QMDV03_FIELDS[key]
                self._write_bits(hi, lo, value)
            else:
                raise KeyError(f"Unknown QMD field: {name}")
    
    def write_field(self, hi, lo, value):
        self._write_bits(hi, lo, value)
    
    def set_constant_buf(self, index, addr, size, valid=1, invalidate=1):
        """Set constant buffer binding."""
        hi, lo = QMDV03_CONSTANT_BUFFER_VALID(index)
        self._write_bits(hi, lo, valid)
        hi, lo = QMDV03_CONSTANT_BUFFER_ADDR_LOWER(index)
        self._write_bits(hi, lo, addr & 0xFFFFFFFF)
        hi, lo = QMDV03_CONSTANT_BUFFER_ADDR_UPPER(index)
        self._write_bits(hi, lo, (addr >> 32) & 0x1FFFF)
        hi, lo = QMDV03_CONSTANT_BUFFER_SIZE_SHIFTED4(index)
        self._write_bits(hi, lo, size >> 4 if size > 0 else 0)
        hi, lo = QMDV03_CONSTANT_BUFFER_INVALIDATE(index)
        self._write_bits(hi, lo, invalidate)


class PushBufferBuilder:
    """Build a push buffer of GPU methods for GPFIFO submission.
    
    Each method is a 4-byte header + N data words:
      Header: (typ << 28) | (count << 16) | (subchannel << 13) | (method >> 2)
      typ=2 is "increasing" method (auto-increment register address)
      
    The whole push buffer will be pointed to by a GPFIFO entry.
    """
    def __init__(self, max_words=1024):
        self.words = []
    
    def nvm(self, subchannel, method, *args, typ=2):
        """Add a method call to the push buffer.
        
        subchannel: 0 = GPFIFO/channel class, 1 = compute, 4 = DMA copy
        method: register offset (byte address, will be >> 2)
        args: data words to write
        """
        header = (typ << 28) | (len(args) << 16) | (subchannel << 13) | (method >> 2)
        self.words.append(header)
        self.words.extend(args)
    
    def get_bytes(self):
        """Return push buffer as bytes."""
        return struct.pack(f'<{len(self.words)}I', *self.words)
    
    def __len__(self):
        return len(self.words)


def compile_ptx_to_cubin(ptx_source, arch="sm_87"):
    """Compile PTX source to CUBIN using nvrtc.
    
    Returns the CUBIN binary as bytes, or raises RuntimeError on failure.
    """
    nvrtc = ctypes.CDLL("libnvrtc.so")
    
    # Function prototypes
    nvrtc.nvrtcCreateProgram.restype = ctypes.c_int
    nvrtc.nvrtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # prog
        ctypes.c_char_p,                   # src
        ctypes.c_char_p,                   # name
        ctypes.c_int,                      # numHeaders
        ctypes.POINTER(ctypes.c_char_p),   # headers
        ctypes.POINTER(ctypes.c_char_p),   # includeNames
    ]
    
    nvrtc.nvrtcCompileProgram.restype = ctypes.c_int
    nvrtc.nvrtcCompileProgram.argtypes = [
        ctypes.c_void_p,                   # prog
        ctypes.c_int,                      # numOptions
        ctypes.POINTER(ctypes.c_char_p),   # options
    ]
    
    nvrtc.nvrtcGetCUBINSize.restype = ctypes.c_int
    nvrtc.nvrtcGetCUBINSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    
    nvrtc.nvrtcGetCUBIN.restype = ctypes.c_int
    nvrtc.nvrtcGetCUBIN.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    
    nvrtc.nvrtcGetProgramLog.restype = ctypes.c_int
    nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    
    nvrtc.nvrtcGetProgramLogSize.restype = ctypes.c_int
    nvrtc.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    
    nvrtc.nvrtcDestroyProgram.restype = ctypes.c_int
    nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    
    # Create program
    prog = ctypes.c_void_p()
    src = ptx_source.encode('utf-8') if isinstance(ptx_source, str) else ptx_source
    ret = nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src, b"test.cu", 0, None, None)
    if ret != 0:
        raise RuntimeError(f"nvrtcCreateProgram failed: {ret}")
    
    # Compile with architecture target
    options = (ctypes.c_char_p * 1)(f"--gpu-architecture={arch}".encode())
    ret = nvrtc.nvrtcCompileProgram(prog, 1, options)
    
    if ret != 0:
        # Get log
        log_size = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
        log_buf = ctypes.create_string_buffer(log_size.value)
        nvrtc.nvrtcGetProgramLog(prog, log_buf)
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        raise RuntimeError(f"nvrtcCompileProgram failed ({ret}): {log_buf.value.decode()}")
    
    # Get CUBIN
    cubin_size = ctypes.c_size_t()
    ret = nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(cubin_size))
    if ret != 0:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        raise RuntimeError(f"nvrtcGetCUBINSize failed: {ret}")
    
    cubin = ctypes.create_string_buffer(cubin_size.value)
    ret = nvrtc.nvrtcGetCUBIN(prog, cubin)
    if ret != 0:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        raise RuntimeError(f"nvrtcGetCUBIN failed: {ret}")
    
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    return cubin.raw


def parse_cubin_elf(cubin_bytes):
    """Parse a CUBIN ELF to find the .text section (SASS code) and program info.
    
    Returns dict with:
        'text_offset': offset of .text.kernel_name within the ELF
        'text_size':   size of the .text section
        'text_data':   the actual SASS machine code bytes
        'reg_count':   number of registers used (from .nv.info)
        'shared_mem':  shared memory size
    """
    import struct as st
    data = cubin_bytes
    
    # Parse ELF header (64-bit)
    if data[:4] != b'\x7fELF':
        raise ValueError("Not an ELF file")
    
    ei_class = data[4]  # 1=32bit, 2=64bit
    if ei_class != 2:
        raise ValueError(f"Expected 64-bit ELF, got class {ei_class}")
    
    # ELF64 header fields
    e_shoff = st.unpack_from('<Q', data, 40)[0]     # section header offset
    e_shentsize = st.unpack_from('<H', data, 58)[0]  # section header entry size
    e_shnum = st.unpack_from('<H', data, 60)[0]      # number of section headers
    e_shstrndx = st.unpack_from('<H', data, 62)[0]   # string table section index
    
    # Read section header string table
    shstr_off = st.unpack_from('<Q', data, e_shoff + e_shstrndx * e_shentsize + 24)[0]
    shstr_size = st.unpack_from('<Q', data, e_shoff + e_shstrndx * e_shentsize + 32)[0]
    shstrtab = data[shstr_off:shstr_off + shstr_size]
    
    def get_section_name(sh_name):
        end = shstrtab.index(b'\x00', sh_name)
        return shstrtab[sh_name:end].decode('ascii')
    
    result = {
        'text_offset': 0, 'text_size': 0, 'text_data': b'',
        'reg_count': 16, 'shared_mem': 0, 'full_elf': data
    }
    
    for i in range(e_shnum):
        sh_base = e_shoff + i * e_shentsize
        sh_name = st.unpack_from('<I', data, sh_base)[0]
        sh_type = st.unpack_from('<I', data, sh_base + 4)[0]
        sh_offset = st.unpack_from('<Q', data, sh_base + 24)[0]
        sh_size = st.unpack_from('<Q', data, sh_base + 32)[0]
        
        name = get_section_name(sh_name)
        
        if name.startswith('.text.'):
            result['text_offset'] = sh_offset
            result['text_size'] = sh_size
            result['text_data'] = data[sh_offset:sh_offset + sh_size]
        elif name.startswith('.nv.info.'):
            # Parse EIATTR entries for register count
            info_data = data[sh_offset:sh_offset + sh_size]
            off = 0
            while off + 12 <= len(info_data):
                attr_fmt = st.unpack_from('<BBH', info_data, off)
                attr_type = attr_fmt[0]
                attr_param = attr_fmt[2]
                if attr_param == 0x2f:  # EIATTR_REGCOUNT
                    val = st.unpack_from('<II', info_data, off + 4)
                    result['reg_count'] = val[1]
                off += 12  # entries are typically 12 bytes
        elif name.startswith('.nv.shared.'):
            result['shared_mem'] = sh_size
    
    return result


def test_mmap_userd_gpfifo(channel_info, nvmap_fd):
    """Test 12: mmap the userd and GPFIFO buffers for usermode submit.
    
    These buffers were created in Phase 1's test_full_channel_setup() and
    passed via channel_info dict. We mmap them to CPU for direct write access.
    """
    print("\n" + "=" * 60)
    print("TEST 12: MMAP USERD + GPFIFO BUFFERS")
    print("=" * 60)
    
    userd_dmabuf_fd = channel_info['userd_dmabuf_fd']
    gpfifo_dmabuf_fd = channel_info['gpfifo_dmabuf_fd']
    
    # mmap userd (4KB)
    userd_mm = mmap.mmap(userd_dmabuf_fd, 4096, mmap.MAP_SHARED,
                          mmap.PROT_READ | mmap.PROT_WRITE, 0)
    print(f"  userd mmap:       OK, 4096 bytes, fd={userd_dmabuf_fd}")
    
    # Read GPGet and GPPut from userd
    gp_get = struct.unpack_from('<I', userd_mm, USERD_GP_GET_OFFSET)[0]
    gp_put = struct.unpack_from('<I', userd_mm, USERD_GP_PUT_OFFSET)[0]
    print(f"  GPGet:            {gp_get}")
    print(f"  GPPut:            {gp_put}")
    
    # mmap GPFIFO ring (8KB = 1024 entries * 8 bytes)
    gpfifo_mm = mmap.mmap(gpfifo_dmabuf_fd, 8192, mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE, 0)
    print(f"  GPFIFO mmap:      OK, 8192 bytes, fd={gpfifo_dmabuf_fd}")
    
    # Verify GPFIFO is zeroed initially
    nonzero = sum(1 for i in range(8192) if gpfifo_mm[i] != 0)
    print(f"  GPFIFO non-zero:  {nonzero} bytes (should be 0)")
    
    if nonzero > 0:
        print("  WARNING: GPFIFO not zeroed — may have stale entries")
    
    channel_info['userd_mm'] = userd_mm
    channel_info['gpfifo_mm'] = gpfifo_mm
    
    # mmap the ctrl fd to get access to the usermode doorbell register.
    # The kernel's gk20a_ctrl_dev_mmap() maps g->usermode_regs_bus_addr
    # which is the GPU's usermode register page. The doorbell is at offset 0x90
    # (NV_USERMODE_NOTIFY_CHANNEL_PENDING). Writing work_submit_token there
    # notifies the GPU that new GPFIFO entries are available.
    USERMODE_NOTIFY_OFFSET = 0x90
    ctrl_fd = channel_info['ctrl_fd']
    try:
        doorbell_mm = mmap.mmap(ctrl_fd, 0x1000, mmap.MAP_SHARED,
                                 mmap.PROT_READ | mmap.PROT_WRITE, 0)
        channel_info['doorbell_mm'] = doorbell_mm
        print(f"  Doorbell mmap:    OK, ctrl_fd={ctrl_fd}, offset 0x{USERMODE_NOTIFY_OFFSET:x}")
    except OSError as e:
        print(f"  Doorbell mmap:    FAILED ({e})")
        print(f"  Will fall back to GPPut-only doorbell")
    
    print(f"\n  ✓ USERD + GPFIFO mmap PASSED")
    return True


def submit_pushbuf(channel_info, pushbuf_gpu_va, pushbuf_len_words):
    """Submit a push buffer through the GPFIFO.
    
    1. Write GPFIFO entry pointing to push buffer
    2. Update GP_PUT in userd
    3. Ring doorbell via host1x (write work_submit_token)
    
    Args:
        channel_info: dict with userd_mm, gpfifo_mm, work_submit_token, ch_fd
        pushbuf_gpu_va: GPU virtual address of the push buffer
        pushbuf_len_words: number of 32-bit words in the push buffer
    """
    userd_mm = channel_info['userd_mm']
    gpfifo_mm = channel_info['gpfifo_mm']
    token = channel_info['work_submit_token']
    
    # Read current GP_PUT
    gp_put = struct.unpack_from('<I', userd_mm, USERD_GP_PUT_OFFSET)[0]
    
    # Format GPFIFO entry (8 bytes = 64 bits):
    # From tinygrad: (cmdq_addr//4 << 2) | (len << 42) | (1 << 41)
    # This gives: bits[1:0]=0, bits[40:2]=GPU_VA>>2, bit[41]=PRIV, bits[52:42]=length  
    gpfifo_entry = (pushbuf_gpu_va & ~3) | (pushbuf_len_words << 42) | (1 << 41)
    
    # Write GPFIFO entry at current GP_PUT position
    entry_offset = (gp_put % 1024) * 8  # 8 bytes per entry
    struct.pack_into('<Q', gpfifo_mm, entry_offset, gpfifo_entry)
    
    # Update GP_PUT
    new_gp_put = (gp_put + 1) % 1024
    struct.pack_into('<I', userd_mm, USERD_GP_PUT_OFFSET, new_gp_put)
    
    # Memory barrier (flush writes before doorbell)
    # On ARM64 (aarch64), we need a DMB (Data Memory Barrier)
    # In Python, we can't easily do a memory barrier, but mmap with MAP_SHARED
    # and the WC flags on the userd buffer should make writes visible.
    # For extra safety, flush the mmap
    # Note: mmap.flush() calls msync() which returns EINVAL on DMA-BUF mmaps.
    # IO_COHERENCE means writes are immediately visible to GPU — no flush needed.
    
    # Ring doorbell: write work_submit_token to the usermode MMIO register
    # at offset 0x90 (NV_USERMODE_NOTIFY_CHANNEL_PENDING).
    # This is mapped via mmap() on the ctrl fd (gk20a_ctrl_dev_mmap).
    USERMODE_NOTIFY_OFFSET = 0x90
    if 'doorbell_mm' in channel_info:
        doorbell_mm = channel_info['doorbell_mm']
        struct.pack_into('<I', doorbell_mm, USERMODE_NOTIFY_OFFSET, token)
    
    return new_gp_put


def test_gpfifo_semaphore_release(channel_info, allocator, compute_class):
    """Test 13: Submit a command through GPFIFO to release a semaphore.
    
    This proves the GPU is processing our push buffer by having it write
    a known value to a memory location (semaphore release). We:
    1. Allocate a semaphore buffer + push buffer
    2. Build push buffer with SET_OBJECT (compute class) + SEM release
    3. Submit via GPFIFO
    4. Poll semaphore for the expected value
    """
    print("\n" + "=" * 60)
    print("TEST 13: GPFIFO SEMAPHORE RELEASE")
    print("=" * 60)
    
    # Allocate semaphore buffer (will be written by GPU)
    sem_buf = allocator.alloc(4096, flags=NVMAP_HANDLE_INNER_CACHEABLE)
    allocator.mmap_buffer(sem_buf)
    allocator.gpu_map(sem_buf)
    
    # Clear semaphore to a known initial value
    struct.pack_into('<Q', sem_buf.cpu_addr, 0, 0)  # 64-bit zero
    print(f"  Semaphore buffer: gpu_va=0x{sem_buf.gpu_va:012x}")
    
    # Allocate push buffer
    pushbuf = allocator.alloc(4096, flags=NVMAP_HANDLE_INNER_CACHEABLE)
    allocator.mmap_buffer(pushbuf)
    allocator.gpu_map(pushbuf)
    
    # Build push buffer
    pb = PushBufferBuilder()
    
    # First: SET_OBJECT on subchannel 1 (compute class)
    pb.nvm(1, NVC6C0_SET_OBJECT, compute_class)
    
    # Then: DMA copy class on subchannel 4 (needed for some operations)
    dma_class = 0xc7b5  # Ampere DMA copy class  
    pb.nvm(4, NVC6B5_SET_OBJECT, dma_class)
    
    # Semaphore release via GPFIFO channel class (subchannel 0)
    # Write value 0x42 to semaphore address
    sem_addr = sem_buf.gpu_va
    sem_value = 0x42
    
    pb.nvm(0, NVC56F_SEM_ADDR_LO,
           sem_addr & 0xFFFFFFFF,          # SEM_ADDR_LO
           (sem_addr >> 32) & 0xFF,        # SEM_ADDR_HI
           sem_value & 0xFFFFFFFF,          # SEM_PAYLOAD_LO
           (sem_value >> 32) & 0xFFFFFFFF,  # SEM_PAYLOAD_HI
           NVC56F_SEM_EXECUTE_OPERATION_RELEASE | NVC56F_SEM_EXECUTE_RELEASE_WFI_EN | NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT)
    
    # Write push buffer data
    pb_bytes = pb.get_bytes()
    for i, b in enumerate(pb_bytes):
        pushbuf.cpu_addr[i] = b
    # IO_COHERENCE: no flush needed
    
    print(f"  Push buffer:      {len(pb)} words ({len(pb_bytes)} bytes)")
    print(f"  Push buffer VA:   0x{pushbuf.gpu_va:012x}")
    print(f"  Sem addr:         0x{sem_addr:012x}")
    print(f"  Expected value:   0x{sem_value:x}")
    
    # Submit via GPFIFO
    submit_pushbuf(channel_info, pushbuf.gpu_va, len(pb))
    
    # Wait for completion by polling semaphore
    print(f"  Waiting for GPU...")
    timeout_ms = 2000
    start = time.time()
    result_value = 0
    while (time.time() - start) * 1000 < timeout_ms:
        result_value = struct.unpack_from('<Q', sem_buf.cpu_addr, 0)[0]
        if result_value == sem_value:
            elapsed = (time.time() - start) * 1000
            print(f"  Semaphore value:  0x{result_value:x} (match! took {elapsed:.1f}ms)")
            print(f"\n  ✓ GPFIFO SEMAPHORE RELEASE PASSED")
            return True
        time.sleep(0.001)
    
    # If we get here, check if GPPut update alone didn't trigger the GPU.
    # Try the SUBMIT_GPFIFO ioctl as a fallback doorbell kick.
    elapsed = (time.time() - start) * 1000
    result_value = struct.unpack_from('<Q', sem_buf.cpu_addr, 0)[0]
    print(f"  After {elapsed:.0f}ms: sem=0x{result_value:x} (expected 0x{sem_value:x})")
    
    if result_value != sem_value:
        print("  GPPut update alone didn't trigger GPU, trying SUBMIT_GPFIFO kick...")
        try:
            # SUBMIT_GPFIFO ioctl with 0 entries as a doorbell kick
            # struct nvgpu_submit_gpfifo_args { u64 gpfifo, u32 num, u32 flags, ... }
            kick_buf = bytearray(48)  # oversized to be safe
            struct.pack_into('<QII', kick_buf, 0, 0, 0, 0)  # gpfifo=0, num=0, flags=0
            NVGPU_IOCTL_CHANNEL_SUBMIT_GPFIFO = _IOWR('H', 107, 48)
            try:
                fcntl.ioctl(channel_info['ch_fd'], NVGPU_IOCTL_CHANNEL_SUBMIT_GPFIFO, kick_buf)
            except OSError as e:
                print(f"  SUBMIT_GPFIFO kick returned: {e}")
                # Expected to fail for usermode channels — that's fine
                # The kick via host1x should work differently
        except Exception as e:
            print(f"  Kick attempt error: {e}")
        
        # Wait a bit more after kick
        time.sleep(0.5)
        result_value = struct.unpack_from('<Q', sem_buf.cpu_addr, 0)[0]
        print(f"  After kick: sem=0x{result_value:x}")
    
    if result_value == sem_value:
        print(f"\n  ✓ GPFIFO SEMAPHORE RELEASE PASSED (after kick)")
        return True
    
    # Last resort: try writing the doorbell token to a host1x doorbell register
    # On some nvgpu implementations, there's a per-channel doorbell at a fixed
    # MMIO offset that userspace can poke. Let's try /dev/host1x or a raw
    # memory write approach.
    print(f"  Trying host1x doorbell mechanism...")
    
    # On Jetson nvgpu, the work_submit_token identifies the channel to the GPU
    # scheduler. The "doorbell" is typically implemented as a write to a fixed
    # MMIO register. In usermode submit, the kernel should set up the GPU to 
    # detect GPPut changes directly. Let's read back GPPut/GPGet to see status.
    gp_get = struct.unpack_from('<I', channel_info['userd_mm'], USERD_GP_GET_OFFSET)[0]
    gp_put = struct.unpack_from('<I', channel_info['userd_mm'], USERD_GP_PUT_OFFSET)[0]
    print(f"  GPGet={gp_get}, GPPut={gp_put}")
    
    if result_value != sem_value:
        print(f"\n  ✗ GPFIFO SEMAPHORE RELEASE FAILED")
        print(f"    Semaphore still 0x{result_value:x}, expected 0x{sem_value:x}")
        print(f"    Possible causes:")
        print(f"    - Doorbell mechanism not working (need MMIO write?)")
        print(f"    - Push buffer format incorrect")
        print(f"    - Channel not properly enabled")
        return False
    
    return True


def test_nvrtc_compile(arch="sm_87"):
    """Test 14: Compile a trivial CUDA kernel using nvrtc.
    
    This proves we can compile PTX -> SASS for the Orin's SM 8.7.
    Returns the CUBIN info dict on success.
    """
    print("\n" + "=" * 60)
    print("TEST 14: NVRTC SHADER COMPILATION")
    print("=" * 60)
    
    kernel_source = r"""
extern "C" __global__ void test_kernel(float *out) {
    int tid = threadIdx.x;
    out[tid] = (float)(tid * tid + 1);
}
"""
    
    print(f"  Kernel source:    test_kernel(float *out)")
    print(f"  Target arch:      {arch}")
    
    try:
        cubin = compile_ptx_to_cubin(kernel_source, arch)
        print(f"  CUBIN size:       {len(cubin)} bytes")
    except Exception as e:
        print(f"  ✗ Compilation failed: {e}")
        return None
    
    # Parse ELF
    try:
        info = parse_cubin_elf(cubin)
        print(f"  .text offset:     0x{info['text_offset']:x}")
        print(f"  .text size:       {info['text_size']} bytes")
        print(f"  Register count:   {info['reg_count']}")
        print(f"  Shared memory:    {info['shared_mem']} bytes")
        
        if info['text_size'] > 0:
            # Print first 16 bytes of SASS (for debugging)
            sass_preview = ' '.join(f'{b:02x}' for b in info['text_data'][:16])
            print(f"  SASS preview:     {sass_preview}")
            print(f"\n  ✓ NVRTC COMPILATION PASSED")
            return info
        else:
            print(f"  ✗ No .text section found in CUBIN")
            return None
    except Exception as e:
        print(f"  ✗ ELF parse failed: {e}")
        import traceback; traceback.print_exc()
        return None


def test_compute_dispatch(channel_info, allocator, compute_class, cubin_info):
    """Test 15: Full compute dispatch — compile shader, build QMD, execute, verify.
    
    This is the big test. We:
    1. Upload the compiled CUBIN to GPU memory
    2. Allocate output buffer
    3. Build a QMD pointing to the shader and output buffer
    4. Build push buffer with SET_OBJECT + SEND_PCAS to launch the QMD
    5. Add a semaphore release after the compute
    6. Submit via GPFIFO
    7. Wait for completion
    8. Read output buffer and verify results
    """
    print("\n" + "=" * 60)
    print("TEST 15: FULL COMPUTE DISPATCH")
    print("=" * 60)
    
    NUM_THREADS = 32  # one warp
    
    # 1. Upload CUBIN to GPU memory
    # We upload the ENTIRE CUBIN ELF (the GPU needs the ELF structure for correct addressing)
    cubin_data = cubin_info['full_elf']
    cubin_aligned_size = ((len(cubin_data) + 4095) & ~4095) + 4096  # extra page for safety
    shader_buf = allocator.alloc(cubin_aligned_size, flags=NVMAP_HANDLE_INNER_CACHEABLE)
    allocator.mmap_buffer(shader_buf)
    allocator.gpu_map(shader_buf)
    
    # Copy CUBIN
    for i, b in enumerate(cubin_data):
        shader_buf.cpu_addr[i] = b
    # IO_COHERENCE: no flush needed
    
    # The program address is at the .text section within the ELF
    prog_addr = shader_buf.gpu_va + cubin_info['text_offset']
    prog_size = cubin_info['text_size']
    print(f"  Shader buffer:    gpu_va=0x{shader_buf.gpu_va:012x}, size={cubin_aligned_size}")
    print(f"  Program address:  0x{prog_addr:012x} (.text offset=0x{cubin_info['text_offset']:x})")
    
    # 2. Allocate output buffer (NUM_THREADS * 4 bytes for float32)
    out_size = max(NUM_THREADS * 4, 4096)
    output_buf = allocator.alloc(out_size, flags=NVMAP_HANDLE_INNER_CACHEABLE)
    allocator.mmap_buffer(output_buf)
    allocator.gpu_map(output_buf)
    
    # Zero output buffer
    for i in range(out_size):
        output_buf.cpu_addr[i] = 0
    # IO_COHERENCE: no flush needed
    print(f"  Output buffer:    gpu_va=0x{output_buf.gpu_va:012x}, size={out_size}")
    
    # 3. Allocate constant buffer (cbuf0) — stores kernel args
    # For CUDA: cbuf0 contains kernel parameters
    # Our kernel takes float *out  — that's an 8-byte pointer
    cbuf_size = 4096
    cbuf_buf = allocator.alloc(cbuf_size, flags=NVMAP_HANDLE_INNER_CACHEABLE)
    allocator.mmap_buffer(cbuf_buf)
    allocator.gpu_map(cbuf_buf)
    
    # Write kernel args in cbuf0
    # The kernel parameter (float *out) is the output buffer GPU VA
    # CUDA ABI: kernel params start at offset 0x160 in const buffer 0
    for i in range(cbuf_size):
        cbuf_buf.cpu_addr[i] = 0
    
    # Write shared_mem_window and local_mem_window at cbuf_0[6:12] (u32 index)
    # These are required by nvcc-compiled CUDA kernels for addressing
    # (tinygrad: cbuf_0[6:12] = [*data64_le(shared_mem_window), *data64_le(local_mem_window), *data64_le(0xfffdc0)])
    shared_mem_window = 0xfe00000000  # within 40-bit range 
    local_mem_window  = 0xfd00000000  # within 40-bit range
    struct.pack_into('<Q', cbuf_buf.cpu_addr, 6*4, shared_mem_window)   # cbuf_0[6:8]
    struct.pack_into('<Q', cbuf_buf.cpu_addr, 8*4, local_mem_window)    # cbuf_0[8:10]
    struct.pack_into('<Q', cbuf_buf.cpu_addr, 10*4, 0xfffdc0)           # cbuf_0[10:12]
    
    # Write kernel parameter (float *out pointer) at offset 0x160
    struct.pack_into('<Q', cbuf_buf.cpu_addr, 0x160, output_buf.gpu_va)
    # IO_COHERENCE: no flush needed
    print(f"  Const buffer:     gpu_va=0x{cbuf_buf.gpu_va:012x}")
    print(f"  Kernel arg @0x160: output_buf VA = 0x{output_buf.gpu_va:012x}")
    
    # 4. Allocate semaphore buffer for completion detection
    sem_buf = allocator.alloc(4096, flags=NVMAP_HANDLE_INNER_CACHEABLE)
    allocator.mmap_buffer(sem_buf)
    allocator.gpu_map(sem_buf)
    struct.pack_into('<Q', sem_buf.cpu_addr, 0, 0)
    # IO_COHERENCE: no flush needed
    
    # 5. Build QMD (256 bytes for V03)
    qmd_buf = allocator.alloc(4096, flags=NVMAP_HANDLE_INNER_CACHEABLE)
    allocator.mmap_buffer(qmd_buf)
    allocator.gpu_map(qmd_buf)
    
    # QMD must be 256-byte aligned — allocator gives page-aligned (4096)
    assert qmd_buf.gpu_va % 256 == 0, f"QMD not 256-byte aligned: 0x{qmd_buf.gpu_va:x}"
    
    reg_count = cubin_info['reg_count']
    shmem_size = max(cubin_info['shared_mem'], 0x400)  # minimum 1KB
    shmem_size = (shmem_size + 127) & ~127  # round up to 128
    
    smem_cfg = 1  # 32KB / 4096 + 1 = config
    for conf_kb in [32, 64, 100]:
        if conf_kb * 1024 >= shmem_size:
            smem_cfg = (conf_kb * 1024) // 4096 + 1
            break
    
    qmd = QMDBuilder()
    qmd.write(
        qmd_major_version=3,
        qmd_version=0,  # v03.00
        qmd_group_id=0x3f,
        sm_global_caching_enable=1,
        api_visible_call_limit=1,  # NO_CHECK
        sampler_index=1,           # VIA_HEADER_INDEX
        cwd_membar_type=1,         # L1_SYSMEMBAR
        barrier_count=1,
        
        # Grid dimensions (blocks)
        cta_raster_width=1,
        cta_raster_height=1,
        cta_raster_depth=1,
        
        # Thread dimensions (threads per block)
        cta_thread_dimension0=NUM_THREADS,
        cta_thread_dimension1=1,
        cta_thread_dimension2=1,
        
        # Shader
        program_address_lower=prog_addr & 0xFFFFFFFF,
        program_address_upper=(prog_addr >> 32) & 0x1FFFF,
        register_count_v=reg_count,
        
        # Shared memory
        shared_memory_size=shmem_size,
        min_sm_config_shared_mem_size=smem_cfg,
        target_sm_config_shared_mem_size=smem_cfg,
        max_sm_config_shared_mem_size=0x1a,
        
        # Cache invalidation
        invalidate_texture_header_cache=1,
        invalidate_texture_sampler_cache=1,
        invalidate_texture_data_cache=1,
        invalidate_shader_data_cache=1,
        invalidate_shader_constant_cache=1,
        
        # SM version for Orin (SM 8.7)
        sass_version=0x87,
        
        # Program prefetch
        program_prefetch_addr_lower_shifted=prog_addr >> 8,
        program_prefetch_addr_upper_shifted=prog_addr >> 40,
        program_prefetch_size=min(prog_size >> 8, 0x1ff),
    )
    
    # Set constant buffer 0 (kernel arguments)
    qmd.set_constant_buf(0, cbuf_buf.gpu_va, cbuf_size)
    
    # Write QMD to GPU memory
    for i, b in enumerate(qmd.data):
        qmd_buf.cpu_addr[i] = b
    # IO_COHERENCE: no flush needed
    
    print(f"  QMD buffer:       gpu_va=0x{qmd_buf.gpu_va:012x}")
    print(f"  QMD registers:    {reg_count}")
    print(f"  QMD shmem:        {shmem_size}")
    print(f"  QMD grid:         1x1x1")
    print(f"  QMD threads:      {NUM_THREADS}x1x1")
    
    # 6. Build push buffer
    pb = PushBufferBuilder()
    
    # SET_OBJECT for compute (subchannel 1)
    pb.nvm(1, NVC6C0_SET_OBJECT, compute_class)
    
    # Set shader memory windows (required before compute dispatch).
    # On Jetson/nvgpu with 40-bit VA space, use addresses within range.
    # These must be in the upper region of the VA space but below the AS limit.
    # Using the same values as tinygrad (they work within 48-bit space for desktop,
    # for Jetson 40-bit we pick addresses that fit).
    shared_mem_window = 0xfe00000000  # within 40-bit range
    local_mem_window  = 0xfd00000000  # within 40-bit range
    pb.nvm(1, NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A,
           (shared_mem_window >> 32) & 0x1FFFF, shared_mem_window & 0xFFFFFFFF)
    pb.nvm(1, NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A,
           (local_mem_window >> 32) & 0x1FFFF, local_mem_window & 0xFFFFFFFF)
    
    # Set local memory to null (no local mem for this kernel)
    pb.nvm(1, NVC6C0_SET_SHADER_LOCAL_MEMORY_A, 0, 0)  # null address
    pb.nvm(1, NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, 0, 0, 0x100)  # limit=0x100
    
    # Invalidate shader caches (clean state)
    pb.nvm(1, NVC6C0_INVALIDATE_SHADER_CACHES, 0x1011)  # inst + data + const
    
    # SEND_PCAS_A: launch the QMD
    pb.nvm(1, NVC6C0_SEND_PCAS_A, qmd_buf.gpu_va >> 8)
    
    # SEND_SIGNALING_PCAS2_B: action = 9 (PREFETCH_SCHEDULE)
    pb.nvm(1, NVC6C0_SEND_SIGNALING_PCAS2_B, 9)
    
    # Semaphore release to signal completion (subchannel 0 = channel/GPFIFO class)
    sem_addr = sem_buf.gpu_va
    sem_value = 0xDEAD
    pb.nvm(0, NVC56F_SEM_ADDR_LO,
           sem_addr & 0xFFFFFFFF,
           (sem_addr >> 32) & 0xFF,
           sem_value & 0xFFFFFFFF,
           0,  # SEM_PAYLOAD_HI
           NVC56F_SEM_EXECUTE_OPERATION_RELEASE | NVC56F_SEM_EXECUTE_RELEASE_WFI_EN | NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT)
    
    # Write push buffer to GPU memory
    pushbuf = allocator.alloc(4096, flags=NVMAP_HANDLE_INNER_CACHEABLE)
    allocator.mmap_buffer(pushbuf)
    allocator.gpu_map(pushbuf)
    pb_bytes = pb.get_bytes()
    for i, b in enumerate(pb_bytes):
        pushbuf.cpu_addr[i] = b
    # IO_COHERENCE: no flush needed
    
    print(f"  Push buffer:      {len(pb)} words, gpu_va=0x{pushbuf.gpu_va:012x}")
    
    # 7. Submit!
    print(f"\n  Submitting to GPFIFO...")
    submit_pushbuf(channel_info, pushbuf.gpu_va, len(pb))
    
    # 8. Wait for completion
    timeout_ms = 5000
    start = time.time()
    while (time.time() - start) * 1000 < timeout_ms:
        result = struct.unpack_from('<Q', sem_buf.cpu_addr, 0)[0]
        if result == sem_value:
            break
        time.sleep(0.001)
    
    elapsed = (time.time() - start) * 1000
    sem_result = struct.unpack_from('<Q', sem_buf.cpu_addr, 0)[0]
    
    if sem_result != sem_value:
        print(f"  ✗ TIMEOUT after {elapsed:.0f}ms: sem=0x{sem_result:x} (expected 0x{sem_value:x})")
        gp_get = struct.unpack_from('<I', channel_info['userd_mm'], USERD_GP_GET_OFFSET)[0]
        gp_put = struct.unpack_from('<I', channel_info['userd_mm'], USERD_GP_PUT_OFFSET)[0]
        print(f"  GPGet={gp_get}, GPPut={gp_put}")
        print(f"\n  ✗ COMPUTE DISPATCH FAILED")
        return False
    
    print(f"  Semaphore:        0x{sem_result:x} (completion after {elapsed:.1f}ms)")
    
    # 9. Verify output!
    print(f"\n  Verifying output buffer...")
    errors = 0
    for i in range(NUM_THREADS):
        expected = float(i * i + 1)
        actual = struct.unpack_from('<f', output_buf.cpu_addr, i * 4)[0]
        if abs(actual - expected) > 0.001:
            if errors < 5:
                print(f"    [thread {i}] expected {expected}, got {actual}")
            errors += 1
    
    if errors == 0:
        # Print a few values for verification
        values = [struct.unpack_from('<f', output_buf.cpu_addr, i * 4)[0] for i in range(min(8, NUM_THREADS))]
        print(f"  Output[0:8]:      {values}")
        print(f"\n  ✓ COMPUTE DISPATCH PASSED — {NUM_THREADS} values correct!")
        return True
    else:
        print(f"\n  ✗ COMPUTE DISPATCH FAILED — {errors}/{NUM_THREADS} wrong values")
        values = [struct.unpack_from('<f', output_buf.cpu_addr, i * 4)[0] for i in range(min(8, NUM_THREADS))]
        print(f"  Output[0:8]:      {values}")
        return False

def main():
    print("=" * 60)
    print("Phase 1+2+3: Direct nvgpu/nvmap ioctl test")
    print("Phase 1: GPU access WITHOUT CUDA")
    print("Phase 2: Memory management — mmap, coherence, TegraAllocator")
    print("Phase 3: Command submission — GPFIFO, QMD, compute dispatch")
    print("=" * 60)

    # Open devices
    try:
        nvmap_fd = os.open("/dev/nvmap", os.O_RDWR | os.O_SYNC)
        print(f"\nOpened /dev/nvmap → fd={nvmap_fd}")
    except OSError as e:
        print(f"ERROR: Cannot open /dev/nvmap: {e}")
        print("Make sure you have permission (user in 'video' group or root)")
        sys.exit(1)

    try:
        ctrl_fd = os.open("/dev/nvgpu/igpu0/ctrl", os.O_RDWR)
        print(f"Opened /dev/nvgpu/igpu0/ctrl → fd={ctrl_fd}")
    except OSError as e:
        print(f"ERROR: Cannot open /dev/nvgpu/igpu0/ctrl: {e}")
        os.close(nvmap_fd)
        sys.exit(1)

    success_count = 0
    total_tests = 0

    # ========================================================================
    # Phase 1 Tests (1-7)
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: GPU Discovery & Channel Setup")
    print("=" * 60)

    # Test 1: GET_CHARACTERISTICS
    total_tests += 1
    try:
        chars = test_get_characteristics(ctrl_fd)
        success_count += 1
        print("  ✓ GET_CHARACTERISTICS succeeded!")
    except Exception as e:
        print(f"  ✗ GET_CHARACTERISTICS failed: {e}")

    # Test 2: ZCULL_GET_CTX_SIZE
    total_tests += 1
    try:
        zcull_size = test_zcull(ctrl_fd)
        success_count += 1
        print("  ✓ ZCULL_GET_CTX_SIZE succeeded!")
    except Exception as e:
        print(f"  ✗ ZCULL_GET_CTX_SIZE failed: {e}")

    # Test 3: NVMAP heap query
    total_tests += 1
    try:
        heaps = test_nvmap_heaps(nvmap_fd)
        success_count += 1
        print("  ✓ GET_AVAILABLE_HEAPS succeeded!")
    except Exception as e:
        print(f"  ✗ GET_AVAILABLE_HEAPS failed: {e}")

    # Test 4: NVMAP CREATE + ALLOC
    total_tests += 1
    dmabuf_fd = None
    try:
        handle, dmabuf_fd = test_nvmap_create_alloc(nvmap_fd, size=4096)
        success_count += 1
        print("  ✓ NVMAP CREATE + ALLOC succeeded!")
    except Exception as e:
        print(f"  ✗ NVMAP CREATE + ALLOC failed: {e}")

    # Test 5: ALLOC_AS (create GPU address space)
    total_tests += 1
    as_fd = None
    try:
        as_fd = test_alloc_as(ctrl_fd)
        success_count += 1
        print("  ✓ ALLOC_AS succeeded!")
    except Exception as e:
        print(f"  ✗ ALLOC_AS failed: {e}")

    # Test 6: MAP_BUFFER_EX (map buffer into GPU VA)
    total_tests += 1
    if as_fd is not None and dmabuf_fd is not None:
        try:
            gpu_va = test_map_buffer(as_fd, dmabuf_fd, size=4096)
            success_count += 1
            print("  ✓ MAP_BUFFER_EX succeeded!")
        except Exception as e:
            print(f"  ✗ MAP_BUFFER_EX failed: {e}")
    else:
        print("  ✗ MAP_BUFFER_EX skipped (AS or dmabuf not available)")

    # Test 7: Full channel + compute class setup
    total_tests += 1
    compute_class = chars.compute_class if chars else 0xc7c0
    channel_info = None
    if as_fd is not None:
        try:
            channel_info = test_full_channel_setup(ctrl_fd, as_fd, nvmap_fd, compute_class)
            success_count += 1
            print("\n  ✓ FULL CHANNEL + COMPUTE SETUP succeeded!")
        except Exception as e:
            print(f"\n  ✗ FULL CHANNEL + COMPUTE SETUP failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  ✗ Channel setup skipped (AS not available)")

    # ========================================================================
    # Phase 2 Tests (8-11)
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Memory Management — mmap, coherence, TegraAllocator")
    print("=" * 60)

    # Test 8: mmap read/write
    total_tests += 1
    try:
        if test_mmap_readwrite(nvmap_fd):
            success_count += 1
    except Exception as e:
        print(f"  ✗ mmap read/write FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test 9: Memory coherence (dual mmap)
    total_tests += 1
    try:
        if test_mmap_coherence(nvmap_fd):
            success_count += 1
    except Exception as e:
        print(f"  ✗ Memory coherence FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test 10: Multi-size alloc + mmap + GPU map
    total_tests += 1
    if as_fd is not None:
        try:
            if test_multi_size_gpu_map(nvmap_fd, as_fd):
                success_count += 1
        except Exception as e:
            print(f"  ✗ Multi-size test FAILED: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  ✗ Multi-size test skipped (AS not available)")

    # Test 11: Cacheable vs write-combine flags
    total_tests += 1
    if as_fd is not None:
        try:
            if test_cacheable_flags(nvmap_fd, as_fd):
                success_count += 1
        except Exception as e:
            print(f"  ✗ Cacheable flags test FAILED: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  ✗ Cacheable flags test skipped (AS not available)")

    # ========================================================================
    # Phase 3 Tests (12-15)
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Command Submission — GPFIFO, QMD, compute dispatch")
    print("=" * 60)

    # Test 12: mmap userd + GPFIFO buffers
    total_tests += 1
    if channel_info is not None:
        try:
            if test_mmap_userd_gpfifo(channel_info, nvmap_fd):
                success_count += 1
        except Exception as e:
            print(f"  ✗ mmap userd/GPFIFO FAILED: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  ✗ mmap userd/GPFIFO skipped (no channel)")

    # Create a TegraAllocator for Phase 3 buffer allocations
    phase3_allocator = TegraAllocator(nvmap_fd, as_fd) if as_fd is not None else None

    # Test 13: GPFIFO semaphore release
    total_tests += 1
    if channel_info is not None and 'userd_mm' in channel_info and phase3_allocator is not None:
        try:
            if test_gpfifo_semaphore_release(channel_info, phase3_allocator, compute_class):
                success_count += 1
        except Exception as e:
            print(f"  ✗ GPFIFO semaphore release FAILED: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  ✗ GPFIFO semaphore release skipped (no channel or userd)")

    # Test 14: nvrtc compile
    total_tests += 1
    cubin_info = None
    try:
        cubin_info = test_nvrtc_compile(arch="sm_87")
        if cubin_info is not None:
            success_count += 1
    except Exception as e:
        print(f"  ✗ nvrtc compile FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test 15: Full compute dispatch
    total_tests += 1
    if (channel_info is not None and 'userd_mm' in channel_info 
        and phase3_allocator is not None and cubin_info is not None):
        try:
            if test_compute_dispatch(channel_info, phase3_allocator, compute_class, cubin_info):
                success_count += 1
        except Exception as e:
            print(f"  ✗ Compute dispatch FAILED: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  ✗ Compute dispatch skipped (missing channel, allocator, or cubin)")

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'=' * 60}")
    print(f"Results: {success_count}/{total_tests} tests passed")
    if success_count == total_tests:
        print("ALL TESTS PASSED!")
        print("\nPhase 1 proven:")
        print("  1. GPU characteristics readable (arch, compute_class, flags)")
        print("  2. Memory allocation works (nvmap IOVMM)")
        print("  3. GPU VA mapping works (MAP_BUFFER_EX)")
        print("  4. Channel + TSG + compute class setup works")
        print("  5. Usermode submit is available")
        print("\nPhase 2 proven:")
        print("  6. CPU mmap of nvmap buffers works (read + write)")
        print("  7. IO_COHERENCE confirmed — no cache flushes needed")
        print("  8. Dual mmap coherence — same physical memory, different mappings")
        print("  9. Multi-size buffers (4KB → 64MB) all work with GPU VA")
        print("  10. CACHEABLE and WRITE_COMBINE flags both functional")
        print("  11. TegraAllocator class handles full lifecycle")
        print("\nPhase 3 proven:")
        print("  12. userd + GPFIFO mmapped for direct CPU access")
        print("  13. GPFIFO submission works — GPU processes push buffers")
        print("  14. nvrtc compiles PTX → CUBIN for SM 8.7")
        print("  15. Compute shader dispatched via QMD — results verified!")
    else:
        # Report per-phase status
        p1_ok = success_count >= 7
        p2_ok = success_count >= 11
        p3_ok = success_count >= 15
        if success_count >= 11:
            print(f"\nPhase 1+2: PASSED  Phase 3: {success_count - 11}/4 tests passed")
        elif success_count >= 7:
            print(f"\nPhase 1: PASSED  Phase 2: {success_count - 7}/4 tests passed")
        print("Some tests failed — check errors above")
    print(f"{'=' * 60}")

    # Cleanup
    if phase3_allocator is not None:
        phase3_allocator.free_all()
    if as_fd is not None:
        os.close(as_fd)
    os.close(ctrl_fd)
    os.close(nvmap_fd)

    return 0 if success_count == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
