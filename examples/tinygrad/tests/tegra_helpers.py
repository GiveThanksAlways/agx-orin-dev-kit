#!/usr/bin/env python3
"""
Shared helpers for Tegra GPU tests.

Provides:
  - ctypes struct definitions for nvgpu/nvmap ioctls
  - ioctl codes
  - Constants
  - Logger class that writes to both stdout and a log file
  - Helper functions for common operations
"""

import os, sys, ctypes, fcntl, mmap, struct, time, datetime, traceback
from ctypes import c_uint8, c_uint16, c_int16, c_uint32, c_int32, c_uint64, c_int64

# ===========================================================================
# Logging
# ===========================================================================

class Logger:
    """Writes to both stdout (unbuffered) and a log file (fsynced after every line)."""

    def __init__(self, path: str):
        self.path = path
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        self._start = time.monotonic()
        self.log(f"=== Log started at {datetime.datetime.now().isoformat()} ===")

    def log(self, msg: str = ""):
        elapsed = time.monotonic() - self._start
        line = f"[{elapsed:8.3f}] {msg}\n"
        sys.stdout.write(line)
        sys.stdout.flush()
        os.write(self.fd, line.encode())
        os.fsync(self.fd)

    def close(self):
        os.close(self.fd)

# ===========================================================================
# ioctl direction bits (aarch64 Linux)
# ===========================================================================

_IOC_NONE  = 0
_IOC_WRITE = 1
_IOC_READ  = 2

def _IOC(d, t, nr, size):
    return (d << 30) | (size << 16) | (ord(t) << 8) | nr

def _IO(t, nr):
    return _IOC(_IOC_NONE, t, nr, 0)

def _IOR(t, nr, sz):
    return _IOC(_IOC_READ, t, nr, sz)

def _IOW(t, nr, sz):
    return _IOC(_IOC_WRITE, t, nr, sz)

def _IOWR(t, nr, sz):
    return _IOC(_IOC_READ | _IOC_WRITE, t, nr, sz)

# ===========================================================================
# ctypes struct definitions
# ===========================================================================

class nvgpu_gpu_characteristics(ctypes.Structure):
    _fields_ = [
        ("arch", c_uint32), ("impl", c_uint32), ("rev", c_uint32), ("num_gpc", c_uint32),
        ("numa_domain_id", c_int32), ("_pad0", c_uint32),
        ("L2_cache_size", c_uint64), ("on_board_video_memory_size", c_uint64),
        ("num_tpc_per_gpc", c_uint32), ("bus_type", c_uint32), ("big_page_size", c_uint32),
        ("compression_page_size", c_uint32), ("pde_coverage_bit_count", c_uint32),
        ("available_big_page_sizes", c_uint32),
        ("flags", c_uint64),
        ("twod_class", c_uint32), ("threed_class", c_uint32), ("compute_class", c_uint32),
        ("gpfifo_class", c_uint32), ("inline_to_memory_class", c_uint32), ("dma_copy_class", c_uint32),
        ("gpc_mask", c_uint32), ("sm_arch_sm_version", c_uint32), ("sm_arch_spa_version", c_uint32),
        ("sm_arch_warp_count", c_uint32),
        ("gpu_ioctl_nr_last", c_int16), ("tsg_ioctl_nr_last", c_int16), ("dbg_gpu_ioctl_nr_last", c_int16),
        ("ioctl_channel_nr_last", c_int16), ("as_ioctl_nr_last", c_int16),
        ("gpu_va_bit_count", c_uint8), ("reserved", c_uint8),
        ("max_fbps_count", c_uint32), ("fbp_en_mask", c_uint32), ("emc_en_mask", c_uint32),
        ("max_ltc_per_fbp", c_uint32), ("max_lts_per_ltc", c_uint32), ("max_tex_per_tpc", c_uint32),
        ("max_gpc_count", c_uint32),
        ("rop_l2_en_mask_DEPRECATED", c_uint32 * 2),
        ("chipname", c_uint8 * 8),
        ("gr_compbit_store_base_hw", c_uint64),
        ("gr_gobs_per_comptagline_per_slice", c_uint32), ("num_ltc", c_uint32),
        ("lts_per_ltc", c_uint32), ("cbc_cache_line_size", c_uint32),
        ("cbc_comptags_per_line", c_uint32), ("map_buffer_batch_limit", c_uint32),
        ("max_freq", c_uint64),
        ("graphics_preemption_mode_flags", c_uint32), ("compute_preemption_mode_flags", c_uint32),
        ("default_graphics_preempt_mode", c_uint32), ("default_compute_preempt_mode", c_uint32),
        ("local_video_memory_size", c_uint64),
        ("pci_vendor_id", c_uint16), ("pci_device_id", c_uint16), ("pci_subsystem_vendor_id", c_uint16),
        ("pci_subsystem_device_id", c_uint16), ("pci_class", c_uint16), ("pci_revision", c_uint8),
        ("vbios_oem_version", c_uint8), ("vbios_version", c_uint32),
        ("reg_ops_limit", c_uint32), ("reserved1", c_uint32),
        ("event_ioctl_nr_last", c_int16), ("pad", c_uint16), ("max_css_buffer_size", c_uint32),
        ("ctxsw_ioctl_nr_last", c_int16), ("prof_ioctl_nr_last", c_int16),
        ("nvs_ioctl_nr_last", c_int16), ("reserved2", c_uint8 * 2),
        ("max_ctxsw_ring_buffer_size", c_uint32), ("reserved3", c_uint32),
        ("per_device_identifier", c_uint64),
        ("num_ppc_per_gpc", c_uint32), ("max_veid_count_per_tsg", c_uint32),
        ("num_sub_partition_per_fbpa", c_uint32), ("gpu_instance_id", c_uint32),
        ("gr_instance_id", c_uint32), ("max_gpfifo_entries", c_uint32),
        ("max_dbg_tsg_timeslice", c_uint32), ("reserved5", c_uint32),
        ("device_instance_id", c_uint64),
    ]

class nvgpu_gpu_get_characteristics(ctypes.Structure):
    _fields_ = [("gpu_characteristics_buf_size", c_uint64), ("gpu_characteristics_buf_addr", c_uint64)]

class nvmap_create_handle(ctypes.Structure):
    _fields_ = [("size", c_uint32), ("handle", c_uint32)]

class nvmap_alloc_handle(ctypes.Structure):
    _fields_ = [("handle", c_uint32), ("heap_mask", c_uint32), ("flags", c_uint32),
                ("align", c_uint32), ("numa_nid", c_int32)]

class nvgpu_alloc_as_args(ctypes.Structure):
    _fields_ = [("big_page_size", c_uint32), ("as_fd", c_int32), ("flags", c_uint32),
                ("reserved", c_uint32), ("va_range_start", c_uint64), ("va_range_end", c_uint64),
                ("va_range_split", c_uint64), ("padding", c_uint32 * 6)]

class nvgpu_as_bind_channel_args(ctypes.Structure):
    _fields_ = [("channel_fd", c_uint32)]

class nvgpu_as_map_buffer_ex_args(ctypes.Structure):
    _fields_ = [("flags", c_uint32), ("compr_kind", c_int16), ("incompr_kind", c_int16),
                ("dmabuf_fd", c_uint32), ("page_size", c_uint32), ("buffer_offset", c_uint64),
                ("mapping_size", c_uint64), ("offset", c_uint64)]

class nvgpu_as_unmap_buffer_args(ctypes.Structure):
    _fields_ = [("offset", c_uint64)]

class nvgpu_gpu_open_tsg_args(ctypes.Structure):
    _fields_ = [("tsg_fd", c_int32), ("flags", c_uint32), ("token", c_uint32),
                ("reserved", c_uint32), ("subctx_id", c_uint32), ("_pad", c_uint32)]

class nvgpu_tsg_bind_channel_ex_args(ctypes.Structure):
    _fields_ = [("channel_fd", c_int32), ("subcontext_id", c_uint32), ("reserved", c_uint8 * 16)]

class nvgpu_tsg_create_subcontext_args(ctypes.Structure):
    _fields_ = [("type", c_uint32), ("as_fd", c_int32), ("veid", c_uint32), ("reserved", c_uint32)]

class nvgpu_gpu_open_channel_args(ctypes.Structure):
    _fields_ = [("channel_fd", c_int32)]

class nvgpu_alloc_obj_ctx_args(ctypes.Structure):
    _fields_ = [("class_num", c_uint32), ("flags", c_uint32), ("obj_id", c_uint64)]

class nvgpu_channel_setup_bind_args(ctypes.Structure):
    _fields_ = [("num_gpfifo_entries", c_uint32), ("num_inflight_jobs", c_uint32),
                ("flags", c_uint32), ("userd_dmabuf_fd", c_int32), ("gpfifo_dmabuf_fd", c_int32),
                ("work_submit_token", c_uint32), ("userd_dmabuf_offset", c_uint64),
                ("gpfifo_dmabuf_offset", c_uint64), ("gpfifo_gpu_va", c_uint64),
                ("userd_gpu_va", c_uint64), ("usermode_mmio_gpu_va", c_uint64),
                ("reserved", c_uint32 * 9)]

class nvgpu_channel_wdt_args(ctypes.Structure):
    _fields_ = [("wdt_status", c_uint32), ("timeout_ms", c_uint32)]

class nvgpu_get_user_syncpoint_args(ctypes.Structure):
    _fields_ = [("gpu_va", c_uint64), ("syncpoint_id", c_uint32), ("syncpoint_max", c_uint32)]

# ===========================================================================
# ioctl codes
# ===========================================================================

NVGPU_GPU_IOCTL_GET_CHARACTERISTICS  = _IOWR('G', 5, ctypes.sizeof(nvgpu_gpu_get_characteristics))
NVGPU_GPU_IOCTL_ALLOC_AS             = _IOWR('G', 8, ctypes.sizeof(nvgpu_alloc_as_args))
NVGPU_GPU_IOCTL_OPEN_TSG             = _IOWR('G', 9, ctypes.sizeof(nvgpu_gpu_open_tsg_args))
NVGPU_GPU_IOCTL_OPEN_CHANNEL         = _IOWR('G', 11, ctypes.sizeof(nvgpu_gpu_open_channel_args))
NVMAP_IOC_CREATE                     = _IOWR('N', 0, ctypes.sizeof(nvmap_create_handle))
NVMAP_IOC_ALLOC                      = _IOW('N', 3, ctypes.sizeof(nvmap_alloc_handle))
NVMAP_IOC_GET_FD                     = _IOWR('N', 15, ctypes.sizeof(nvmap_create_handle))
NVMAP_IOC_FREE                       = _IO('N', 4)
NVGPU_AS_IOCTL_BIND_CHANNEL          = _IOWR('A', 1, ctypes.sizeof(nvgpu_as_bind_channel_args))
NVGPU_AS_IOCTL_MAP_BUFFER_EX         = _IOWR('A', 7, ctypes.sizeof(nvgpu_as_map_buffer_ex_args))
NVGPU_AS_IOCTL_UNMAP_BUFFER          = _IOWR('A', 5, ctypes.sizeof(nvgpu_as_unmap_buffer_args))
NVGPU_TSG_IOCTL_BIND_CHANNEL_EX      = _IOWR('T', 11, ctypes.sizeof(nvgpu_tsg_bind_channel_ex_args))
NVGPU_TSG_IOCTL_CREATE_SUBCONTEXT    = _IOWR('T', 18, ctypes.sizeof(nvgpu_tsg_create_subcontext_args))
NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX    = _IOWR('H', 108, ctypes.sizeof(nvgpu_alloc_obj_ctx_args))
NVGPU_IOCTL_CHANNEL_SETUP_BIND       = _IOWR('H', 128, ctypes.sizeof(nvgpu_channel_setup_bind_args))
NVGPU_IOCTL_CHANNEL_WDT              = _IOW('H', 119, ctypes.sizeof(nvgpu_channel_wdt_args))
NVGPU_IOCTL_CHANNEL_GET_USER_SYNCPOINT = _IOR('H', 126, ctypes.sizeof(nvgpu_get_user_syncpoint_args))

# ===========================================================================
# Constants
# ===========================================================================

NVMAP_HEAP_IOVMM          = (1 << 30)
NVMAP_HANDLE_WRITE_COMBINE = 1
NVMAP_HANDLE_INNER_CACHEABLE = 2

NVGPU_SETUP_BIND_FLAGS_USERMODE_SUPPORT = (1 << 3)
NVGPU_SETUP_BIND_FLAGS_DETERMINISTIC    = (1 << 1)

# Subcontext types
NVGPU_TSG_SUBCONTEXT_TYPE_ASYNC = 1

# ===========================================================================
# Helper functions
# ===========================================================================

def nv_ioctl(fd, code, buf):
    """Call ioctl, raise OSError on failure."""
    ret = fcntl.ioctl(fd, code, buf)
    if ret < 0:
        raise OSError(f"ioctl 0x{code:08x} failed with {ret}")
    return ret


def nvmap_create_alloc_getfd(nvmap_fd, size, flags=NVMAP_HANDLE_WRITE_COMBINE, align=4096):
    """Create + Alloc + Get FD for an nvmap buffer.  Returns (handle, dmabuf_fd)."""
    c = nvmap_create_handle()
    c.size = size
    nv_ioctl(nvmap_fd, NVMAP_IOC_CREATE, c)
    a = nvmap_alloc_handle()
    a.handle = c.handle
    a.heap_mask = NVMAP_HEAP_IOVMM
    a.flags = flags
    a.align = align
    a.numa_nid = 0
    nv_ioctl(nvmap_fd, NVMAP_IOC_ALLOC, a)
    g = nvmap_create_handle()
    g.handle = c.handle
    nv_ioctl(nvmap_fd, NVMAP_IOC_GET_FD, g)
    return c.handle, g.size   # g.size carries the dmabuf fd


def gpu_va_map(as_fd, dmabuf_fd, size, page_size=4096):
    """Map an nvmap buffer into GPU VA space. Returns gpu_va."""
    m = nvgpu_as_map_buffer_ex_args()
    m.flags = 0
    m.compr_kind = -1
    m.incompr_kind = 0
    m.dmabuf_fd = dmabuf_fd
    m.page_size = page_size
    m.buffer_offset = 0
    m.mapping_size = 0
    m.offset = 0
    nv_ioctl(as_fd, NVGPU_AS_IOCTL_MAP_BUFFER_EX, m)
    return m.offset


def cpu_mmap(dmabuf_fd, size):
    """mmap a dmabuf to CPU. Returns address (int)."""
    ct = ctypes.CDLL("libc.so.6", use_errno=True)
    ct.mmap.restype = ctypes.c_void_p
    ct.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    addr = ct.mmap(None, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, dmabuf_fd, 0)
    if addr is None or addr == ctypes.c_void_p(-1).value or addr == 0xffffffffffffffff:
        raise RuntimeError(f"mmap failed for fd={dmabuf_fd} size={size} errno={ctypes.get_errno()}")
    return addr


def cpu_mmap_fixed(dmabuf_fd, size, target_addr):
    """MAP_FIXED mmap of a dmabuf at a specific CPU address. Returns address (int)."""
    MAP_FIXED = 0x10
    ct = ctypes.CDLL("libc.so.6", use_errno=True)
    ct.mmap.restype = ctypes.c_void_p
    ct.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    addr = ct.mmap(ctypes.c_void_p(target_addr), size, mmap.PROT_READ | mmap.PROT_WRITE,
                   mmap.MAP_SHARED | MAP_FIXED, dmabuf_fd, 0)
    if addr is None or addr == ctypes.c_void_p(-1).value or addr == 0xffffffffffffffff:
        raise RuntimeError(f"MAP_FIXED mmap failed for fd={dmabuf_fd} target=0x{target_addr:x} errno={ctypes.get_errno()}")
    return addr


def get_gpu_characteristics(ctrl_fd):
    """Get GPU characteristics. Returns nvgpu_gpu_characteristics struct."""
    chars = nvgpu_gpu_characteristics()
    req = nvgpu_gpu_get_characteristics()
    req.gpu_characteristics_buf_size = ctypes.sizeof(chars)
    req.gpu_characteristics_buf_addr = ctypes.addressof(chars)
    nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_GET_CHARACTERISTICS, req)
    return chars


def alloc_address_space(ctrl_fd):
    """Allocate a GPU address space. Returns as_fd."""
    PDE_SIZE = 1 << 21
    args = nvgpu_alloc_as_args()
    args.big_page_size = 0
    args.flags = 2  # UNIFIED_VA
    args.va_range_start = PDE_SIZE
    args.va_range_end = (1 << 40) - PDE_SIZE
    nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_ALLOC_AS, args)
    return args.as_fd


def open_tsg(ctrl_fd):
    """Open a TSG. Returns tsg_fd."""
    args = nvgpu_gpu_open_tsg_args()
    nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_OPEN_TSG, args)
    return args.tsg_fd


def create_subcontext(tsg_fd, as_fd):
    """Create an async subcontext in the TSG. Returns veid."""
    args = nvgpu_tsg_create_subcontext_args()
    args.type = NVGPU_TSG_SUBCONTEXT_TYPE_ASYNC
    args.as_fd = as_fd
    nv_ioctl(tsg_fd, NVGPU_TSG_IOCTL_CREATE_SUBCONTEXT, args)
    return args.veid


def open_channel(ctrl_fd):
    """Open a GPU channel. Returns ch_fd."""
    args = nvgpu_gpu_open_channel_args()
    args.channel_fd = -1
    nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_OPEN_CHANNEL, args)
    return args.channel_fd


def bind_channel_to_as(as_fd, ch_fd):
    """Bind a channel to an address space."""
    args = nvgpu_as_bind_channel_args()
    args.channel_fd = ch_fd
    nv_ioctl(as_fd, NVGPU_AS_IOCTL_BIND_CHANNEL, args)


def bind_channel_to_tsg(tsg_fd, ch_fd, veid=0):
    """Bind a channel to a TSG with subcontext."""
    args = nvgpu_tsg_bind_channel_ex_args()
    args.channel_fd = ch_fd
    args.subcontext_id = veid
    nv_ioctl(tsg_fd, NVGPU_TSG_IOCTL_BIND_CHANNEL_EX, args)


def disable_wdt(ch_fd):
    """Disable watchdog timer on a channel."""
    args = nvgpu_channel_wdt_args()
    args.wdt_status = 1
    nv_ioctl(ch_fd, NVGPU_IOCTL_CHANNEL_WDT, args)


def setup_bind(ch_fd, gpfifo_dmabuf_fd, userd_dmabuf_fd, entries):
    """Run SETUP_BIND with USERMODE_SUPPORT. Returns (token, gpfifo_gpu_va, userd_gpu_va)."""
    args = nvgpu_channel_setup_bind_args()
    args.num_gpfifo_entries = entries
    args.num_inflight_jobs = 0
    args.gpfifo_dmabuf_fd = gpfifo_dmabuf_fd
    args.gpfifo_dmabuf_offset = 0
    args.userd_dmabuf_fd = userd_dmabuf_fd
    args.userd_dmabuf_offset = 0
    args.flags = NVGPU_SETUP_BIND_FLAGS_USERMODE_SUPPORT | NVGPU_SETUP_BIND_FLAGS_DETERMINISTIC
    nv_ioctl(ch_fd, NVGPU_IOCTL_CHANNEL_SETUP_BIND, args)
    return args.work_submit_token, args.gpfifo_gpu_va, args.userd_gpu_va


def alloc_obj_ctx(ch_fd, class_num):
    """Allocate an object context (compute/DMA class) on a channel. Returns obj_id."""
    args = nvgpu_alloc_obj_ctx_args()
    args.class_num = class_num
    args.flags = 0
    nv_ioctl(ch_fd, NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX, args)
    return args.obj_id


def create_full_channel(log, ctrl_fd, as_fd, tsg_fd, veid, nvmap_fd, entries, label="channel"):
    """
    Create a complete channel: OPEN → AS_BIND → TSG_BIND → WDT → SETUP_BIND.
    
    GPFIFO buffer is sized exactly to entries*8 (tight fit, as kernel requires).
    USERD buffer is 4KB (separate, as kernel requires).
    
    Returns dict with ch_fd, token, gpfifo_*, userd_*, dmabuf fds.
    """
    gpfifo_size = entries * 8

    log.log(f"  [{label}] Alloc gpfifo buffer: {gpfifo_size} bytes ({entries} entries)")
    gpfifo_handle, gpfifo_dmabuf = nvmap_create_alloc_getfd(nvmap_fd, gpfifo_size)
    log.log(f"  [{label}] gpfifo: handle={gpfifo_handle}, dmabuf={gpfifo_dmabuf}")

    log.log(f"  [{label}] Alloc userd buffer: 4096 bytes")
    userd_handle, userd_dmabuf = nvmap_create_alloc_getfd(nvmap_fd, 4096)
    log.log(f"  [{label}] userd: handle={userd_handle}, dmabuf={userd_dmabuf}")

    log.log(f"  [{label}] OPEN_CHANNEL...")
    ch_fd = open_channel(ctrl_fd)
    log.log(f"  [{label}] ch_fd={ch_fd}")

    log.log(f"  [{label}] AS_BIND...")
    bind_channel_to_as(as_fd, ch_fd)

    log.log(f"  [{label}] TSG_BIND (veid={veid})...")
    bind_channel_to_tsg(tsg_fd, ch_fd, veid)

    log.log(f"  [{label}] WDT disable...")
    disable_wdt(ch_fd)

    log.log(f"  [{label}] SETUP_BIND (entries={entries})...")
    token, gpfifo_gpu_va, userd_gpu_va = setup_bind(ch_fd, gpfifo_dmabuf, userd_dmabuf, entries)
    log.log(f"  [{label}] token={token}, gpfifo_gpu_va=0x{gpfifo_gpu_va:012x}, userd_gpu_va=0x{userd_gpu_va:012x}")

    return {
        "ch_fd": ch_fd, "token": token,
        "gpfifo_handle": gpfifo_handle, "gpfifo_dmabuf": gpfifo_dmabuf,
        "gpfifo_gpu_va": gpfifo_gpu_va, "gpfifo_size": gpfifo_size,
        "userd_handle": userd_handle, "userd_dmabuf": userd_dmabuf,
        "userd_gpu_va": userd_gpu_va,
    }
