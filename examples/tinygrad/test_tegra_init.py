#!/usr/bin/env python3
"""
Standalone test: mirrors TegraIface's NVDevice.__init__ sequence step-by-step.
Tests each phase independently to find the exact failure point.

Run: python3 -u test_tegra_init.py
"""
import os, sys, struct, ctypes, fcntl, mmap, time, signal

# ============================================================================
# File logger — survives crashes by fsyncing after every line
# ============================================================================
LOG_PATH = "/home/agent/jetpack-nixos/examples/tinygrad/test_tegra_init.log"
_log_fd = None

def log(msg):
    """Print to stdout AND append to log file with immediate fsync."""
    global _log_fd
    print(msg, flush=True)
    if _log_fd is None:
        _log_fd = os.open(LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(_log_fd, (msg + "\n").encode())
    os.fsync(_log_fd)

# Timeout handler to prevent system freeze
def timeout_handler(signum, frame):
    log("\n\n*** TIMEOUT! System would have hung here ***")
    sys.exit(1)
signal.signal(signal.SIGALRM, timeout_handler)

# ============================================================================
# ioctl helpers (same as test_nvgpu.py)
# ============================================================================
_IOC_NONE=0; _IOC_WRITE=1; _IOC_READ=2
def _IOC(d,t,nr,sz): return (d<<30)|(sz<<16)|(ord(t)<<8)|nr
def _IO(t,nr): return _IOC(_IOC_NONE,t,nr,0)
def _IOR(t,nr,sz): return _IOC(_IOC_READ,t,nr,sz)
def _IOW(t,nr,sz): return _IOC(_IOC_WRITE,t,nr,sz)
def _IOWR(t,nr,sz): return _IOC(_IOC_READ|_IOC_WRITE,t,nr,sz)

def nv_ioctl(fd, ioc, buf):
    ret = fcntl.ioctl(fd, ioc, buf)
    if ret < 0: raise OSError(f"ioctl 0x{ioc:08x} failed: {ret}")
    return ret

# ============================================================================
# Structs
# ============================================================================
class nvgpu_gpu_characteristics(ctypes.Structure):
    _fields_ = [
        ("arch",ctypes.c_uint32),("impl",ctypes.c_uint32),("rev",ctypes.c_uint32),("num_gpc",ctypes.c_uint32),
        ("numa_domain_id",ctypes.c_int32),("_pad0",ctypes.c_uint32),
        ("L2_cache_size",ctypes.c_uint64),("on_board_video_memory_size",ctypes.c_uint64),
        ("num_tpc_per_gpc",ctypes.c_uint32),("bus_type",ctypes.c_uint32),("big_page_size",ctypes.c_uint32),
        ("compression_page_size",ctypes.c_uint32),("pde_coverage_bit_count",ctypes.c_uint32),
        ("available_big_page_sizes",ctypes.c_uint32),("flags",ctypes.c_uint64),
        ("twod_class",ctypes.c_uint32),("threed_class",ctypes.c_uint32),("compute_class",ctypes.c_uint32),
        ("gpfifo_class",ctypes.c_uint32),("inline_to_memory_class",ctypes.c_uint32),("dma_copy_class",ctypes.c_uint32),
        ("gpc_mask",ctypes.c_uint32),("sm_arch_sm_version",ctypes.c_uint32),("sm_arch_spa_version",ctypes.c_uint32),
        ("sm_arch_warp_count",ctypes.c_uint32),
        ("gpu_ioctl_nr_last",ctypes.c_int16),("tsg_ioctl_nr_last",ctypes.c_int16),("dbg_gpu_ioctl_nr_last",ctypes.c_int16),
        ("ioctl_channel_nr_last",ctypes.c_int16),("as_ioctl_nr_last",ctypes.c_int16),
        ("gpu_va_bit_count",ctypes.c_uint8),("reserved",ctypes.c_uint8),
        ("max_fbps_count",ctypes.c_uint32),("fbp_en_mask",ctypes.c_uint32),("emc_en_mask",ctypes.c_uint32),
        ("max_ltc_per_fbp",ctypes.c_uint32),("max_lts_per_ltc",ctypes.c_uint32),("max_tex_per_tpc",ctypes.c_uint32),
        ("max_gpc_count",ctypes.c_uint32),
        ("rop_l2_en_mask_DEPRECATED",ctypes.c_uint32*2),("chipname",ctypes.c_uint8*8),
        ("gr_compbit_store_base_hw",ctypes.c_uint64),
        ("gr_gobs_per_comptagline_per_slice",ctypes.c_uint32),("num_ltc",ctypes.c_uint32),
        ("lts_per_ltc",ctypes.c_uint32),("cbc_cache_line_size",ctypes.c_uint32),
        ("cbc_comptags_per_line",ctypes.c_uint32),("map_buffer_batch_limit",ctypes.c_uint32),
        ("max_freq",ctypes.c_uint64),
        ("graphics_preemption_mode_flags",ctypes.c_uint32),("compute_preemption_mode_flags",ctypes.c_uint32),
        ("default_graphics_preempt_mode",ctypes.c_uint32),("default_compute_preempt_mode",ctypes.c_uint32),
        ("local_video_memory_size",ctypes.c_uint64),
        ("pci_vendor_id",ctypes.c_uint16),("pci_device_id",ctypes.c_uint16),("pci_subsystem_vendor_id",ctypes.c_uint16),
        ("pci_subsystem_device_id",ctypes.c_uint16),("pci_class",ctypes.c_uint16),("pci_revision",ctypes.c_uint8),
        ("vbios_oem_version",ctypes.c_uint8),("vbios_version",ctypes.c_uint32),
        ("reg_ops_limit",ctypes.c_uint32),("reserved1",ctypes.c_uint32),
        ("event_ioctl_nr_last",ctypes.c_int16),("pad",ctypes.c_uint16),("max_css_buffer_size",ctypes.c_uint32),
        ("ctxsw_ioctl_nr_last",ctypes.c_int16),("prof_ioctl_nr_last",ctypes.c_int16),
        ("nvs_ioctl_nr_last",ctypes.c_int16),("reserved2",ctypes.c_uint8*2),
        ("max_ctxsw_ring_buffer_size",ctypes.c_uint32),("reserved3",ctypes.c_uint32),
        ("per_device_identifier",ctypes.c_uint64),
        ("num_ppc_per_gpc",ctypes.c_uint32),("max_veid_count_per_tsg",ctypes.c_uint32),
        ("num_sub_partition_per_fbpa",ctypes.c_uint32),("gpu_instance_id",ctypes.c_uint32),
        ("gr_instance_id",ctypes.c_uint32),("max_gpfifo_entries",ctypes.c_uint32),
        ("max_dbg_tsg_timeslice",ctypes.c_uint32),("reserved5",ctypes.c_uint32),
        ("device_instance_id",ctypes.c_uint64),
    ]

class nvgpu_gpu_get_characteristics(ctypes.Structure):
    _fields_ = [("gpu_characteristics_buf_size",ctypes.c_uint64),("gpu_characteristics_buf_addr",ctypes.c_uint64)]

class nvmap_create_handle(ctypes.Structure):
    _fields_ = [("size",ctypes.c_uint32),("handle",ctypes.c_uint32)]

class nvmap_alloc_handle(ctypes.Structure):
    _fields_ = [("handle",ctypes.c_uint32),("heap_mask",ctypes.c_uint32),("flags",ctypes.c_uint32),
                ("align",ctypes.c_uint32),("numa_nid",ctypes.c_int32)]

class nvgpu_alloc_as_args(ctypes.Structure):
    _fields_ = [("big_page_size",ctypes.c_uint32),("as_fd",ctypes.c_int32),("flags",ctypes.c_uint32),
                ("reserved",ctypes.c_uint32),("va_range_start",ctypes.c_uint64),("va_range_end",ctypes.c_uint64),
                ("va_range_split",ctypes.c_uint64),("padding",ctypes.c_uint32*6)]

class nvgpu_as_bind_channel_args(ctypes.Structure):
    _fields_ = [("channel_fd",ctypes.c_uint32)]

class nvgpu_as_map_buffer_ex_args(ctypes.Structure):
    _fields_ = [("flags",ctypes.c_uint32),("compr_kind",ctypes.c_int16),("incompr_kind",ctypes.c_int16),
                ("dmabuf_fd",ctypes.c_uint32),("page_size",ctypes.c_uint32),("buffer_offset",ctypes.c_uint64),
                ("mapping_size",ctypes.c_uint64),("offset",ctypes.c_uint64)]

class nvgpu_gpu_open_tsg_args(ctypes.Structure):
    _fields_ = [("tsg_fd",ctypes.c_int32),("flags",ctypes.c_uint32),("token",ctypes.c_uint32),
                ("reserved",ctypes.c_uint32),("subctx_id",ctypes.c_uint32),("_pad",ctypes.c_uint32)]

class nvgpu_tsg_bind_channel_ex_args(ctypes.Structure):
    _fields_ = [("channel_fd",ctypes.c_int32),("subcontext_id",ctypes.c_uint32),("reserved",ctypes.c_uint8*16)]

class nvgpu_tsg_create_subcontext_args(ctypes.Structure):
    _fields_ = [("type",ctypes.c_uint32),("as_fd",ctypes.c_int32),("veid",ctypes.c_uint32),("reserved",ctypes.c_uint32)]

class nvgpu_gpu_open_channel_args(ctypes.Structure):
    _fields_ = [("channel_fd",ctypes.c_int32)]

class nvgpu_alloc_obj_ctx_args(ctypes.Structure):
    _fields_ = [("class_num",ctypes.c_uint32),("flags",ctypes.c_uint32),("obj_id",ctypes.c_uint64)]

class nvgpu_channel_setup_bind_args(ctypes.Structure):
    _fields_ = [("num_gpfifo_entries",ctypes.c_uint32),("num_inflight_jobs",ctypes.c_uint32),
                ("flags",ctypes.c_uint32),("userd_dmabuf_fd",ctypes.c_int32),("gpfifo_dmabuf_fd",ctypes.c_int32),
                ("work_submit_token",ctypes.c_uint32),("userd_dmabuf_offset",ctypes.c_uint64),
                ("gpfifo_dmabuf_offset",ctypes.c_uint64),("gpfifo_gpu_va",ctypes.c_uint64),
                ("userd_gpu_va",ctypes.c_uint64),("usermode_mmio_gpu_va",ctypes.c_uint64),
                ("reserved",ctypes.c_uint32*9)]

class nvgpu_channel_wdt_args(ctypes.Structure):
    _fields_ = [("wdt_status",ctypes.c_uint32),("timeout_ms",ctypes.c_uint32)]

# ioctl codes
NVGPU_GPU_IOCTL_GET_CHARACTERISTICS = _IOWR('G',5,ctypes.sizeof(nvgpu_gpu_get_characteristics))
NVGPU_GPU_IOCTL_ALLOC_AS = _IOWR('G',8,ctypes.sizeof(nvgpu_alloc_as_args))
NVGPU_GPU_IOCTL_OPEN_TSG = _IOWR('G',9,ctypes.sizeof(nvgpu_gpu_open_tsg_args))
NVGPU_GPU_IOCTL_OPEN_CHANNEL = _IOWR('G',11,ctypes.sizeof(nvgpu_gpu_open_channel_args))
NVMAP_IOC_CREATE = _IOWR('N',0,ctypes.sizeof(nvmap_create_handle))
NVMAP_IOC_ALLOC = _IOW('N',3,ctypes.sizeof(nvmap_alloc_handle))
NVMAP_IOC_GET_FD = _IOWR('N',15,ctypes.sizeof(nvmap_create_handle))
NVMAP_IOC_FREE = _IO('N',4)
NVGPU_AS_IOCTL_BIND_CHANNEL = _IOWR('A',1,ctypes.sizeof(nvgpu_as_bind_channel_args))
NVGPU_AS_IOCTL_MAP_BUFFER_EX = _IOWR('A',7,ctypes.sizeof(nvgpu_as_map_buffer_ex_args))
NVGPU_TSG_IOCTL_BIND_CHANNEL_EX = _IOWR('T',11,ctypes.sizeof(nvgpu_tsg_bind_channel_ex_args))
NVGPU_TSG_IOCTL_CREATE_SUBCONTEXT = _IOWR('T',18,ctypes.sizeof(nvgpu_tsg_create_subcontext_args))
NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX = _IOWR('H',108,ctypes.sizeof(nvgpu_alloc_obj_ctx_args))
NVGPU_IOCTL_CHANNEL_SETUP_BIND = _IOWR('H',128,ctypes.sizeof(nvgpu_channel_setup_bind_args))
NVGPU_IOCTL_CHANNEL_WDT = _IOW('H',119,ctypes.sizeof(nvgpu_channel_wdt_args))

NVMAP_HEAP_IOVMM = 1<<30
NVMAP_HANDLE_WRITE_COMBINE = 1
NVMAP_HANDLE_INNER_CACHEABLE = 2
NVGPU_SETUP_BIND_FLAGS_USERMODE_SUPPORT = 1<<3
NVGPU_SETUP_BIND_FLAGS_DETERMINISTIC = 1<<1

def nvmap_alloc_buf(nvmap_fd, size, flags=NVMAP_HANDLE_WRITE_COMBINE):
    """Helper: CREATE + ALLOC + GET_FD → returns (handle, dmabuf_fd)"""
    c = nvmap_create_handle(); c.size = size
    nv_ioctl(nvmap_fd, NVMAP_IOC_CREATE, c)
    a = nvmap_alloc_handle()
    a.handle = c.handle; a.heap_mask = NVMAP_HEAP_IOVMM; a.flags = flags; a.align = 4096
    nv_ioctl(nvmap_fd, NVMAP_IOC_ALLOC, a)
    g = nvmap_create_handle(); g.handle = c.handle
    nv_ioctl(nvmap_fd, NVMAP_IOC_GET_FD, g)
    return c.handle, g.size  # g.size = dmabuf fd

def gpu_map(as_fd, dmabuf_fd, size):
    """Helper: MAP_BUFFER_EX → returns GPU VA"""
    m = nvgpu_as_map_buffer_ex_args()
    m.flags = 0; m.compr_kind = -1; m.incompr_kind = 0
    m.dmabuf_fd = dmabuf_fd; m.page_size = 4096; m.buffer_offset = 0; m.mapping_size = 0; m.offset = 0
    nv_ioctl(as_fd, NVGPU_AS_IOCTL_MAP_BUFFER_EX, m)
    return m.offset

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
MAP_FIXED = 0x10

def do_mmap(addr, size, fd, offset=0, fixed=False):
    flags = mmap.MAP_SHARED | (MAP_FIXED if fixed else 0)
    r = libc.mmap(ctypes.c_void_p(addr), size, mmap.PROT_READ|mmap.PROT_WRITE, flags, fd, offset)
    if r == ctypes.c_void_p(-1).value or r == 0xffffffffffffffff:
        raise RuntimeError(f"mmap failed: errno={ctypes.get_errno()}")
    return r

# ============================================================================
# Main test — mirrors NVDevice.__init__ with TegraIface
# ============================================================================
def main():
    ENTRIES = 1024  # Match test_nvgpu.py (tinygrad uses 0x10000)
    passed = 0
    total = 0

    # ---- Step 1: Open device nodes ----
    total += 1
    log("Step 1: Open device nodes...")
    signal.alarm(5)
    try:
        nvmap_fd = os.open("/dev/nvmap", os.O_RDWR | os.O_SYNC)
        ctrl_fd = os.open("/dev/nvgpu/igpu0/ctrl", os.O_RDWR)
        log(f"  OK: nvmap_fd={nvmap_fd}, ctrl_fd={ctrl_fd}")
        passed += 1
    except Exception as e:
        log(f"  FAIL: {e}")
        return 1
    signal.alarm(0)

    # ---- Step 2: GET_CHARACTERISTICS ----
    total += 1
    log("\nStep 2: GET_CHARACTERISTICS...")
    signal.alarm(5)
    try:
        chars = nvgpu_gpu_characteristics()
        req = nvgpu_gpu_get_characteristics()
        req.gpu_characteristics_buf_size = ctypes.sizeof(chars)
        req.gpu_characteristics_buf_addr = ctypes.addressof(chars)
        nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_GET_CHARACTERISTICS, req)
        log(f"  arch=0x{chars.arch:04x} compute=0x{chars.compute_class:04x} gpfifo=0x{chars.gpfifo_class:04x} dma=0x{chars.dma_copy_class:04x}")
        log(f"  sm_version=0x{chars.sm_arch_sm_version:x} va_bits={chars.gpu_va_bit_count}")
        passed += 1
    except Exception as e:
        log(f"  FAIL: {e}")
        return 1
    signal.alarm(0)

    compute_class = chars.compute_class
    dma_class = chars.dma_copy_class

    # ---- Step 3: ALLOC_AS ----
    total += 1
    log("\nStep 3: ALLOC_AS...")
    signal.alarm(5)
    try:
        PDE = 1<<21
        as_args = nvgpu_alloc_as_args()
        as_args.big_page_size = 0; as_args.flags = 2  # UNIFIED_VA
        as_args.va_range_start = PDE; as_args.va_range_end = (1<<40)-PDE; as_args.va_range_split = 0
        nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_ALLOC_AS, as_args)
        as_fd = as_args.as_fd
        log(f"  OK: as_fd={as_fd}")
        passed += 1
    except Exception as e:
        log(f"  FAIL: {e}")
        return 1
    signal.alarm(0)

    # ---- Step 4: Allocate gpfifo_area (3MB, like tinygrad) ----
    total += 1
    log("\nStep 4: Allocate gpfifo_area (3MB)...")
    signal.alarm(10)
    try:
        gpfifo_area_size = 0x300000
        gpfifo_handle, gpfifo_dmabuf = nvmap_alloc_buf(nvmap_fd, gpfifo_area_size, NVMAP_HANDLE_INNER_CACHEABLE)
        gpfifo_gpu_va = gpu_map(as_fd, gpfifo_dmabuf, gpfifo_area_size)
        gpfifo_cpu = do_mmap(None, gpfifo_area_size, gpfifo_dmabuf)
        log(f"  OK: handle={gpfifo_handle} dmabuf={gpfifo_dmabuf} gpu_va=0x{gpfifo_gpu_va:010x} cpu=0x{gpfifo_cpu:x}")
        passed += 1
    except Exception as e:
        log(f"  FAIL: {e}")
        return 1
    signal.alarm(0)

    # ---- Step 5: Allocate notifier (4KB, tinygrad uses 48MB) ----
    total += 1
    log("\nStep 5: Allocate notifier (4KB)...")
    signal.alarm(10)
    try:
        notifier_handle, notifier_dmabuf = nvmap_alloc_buf(nvmap_fd, 4096, NVMAP_HANDLE_WRITE_COMBINE)
        notifier_gpu_va = gpu_map(as_fd, notifier_dmabuf, 4096)
        log(f"  OK: handle={notifier_handle} dmabuf={notifier_dmabuf} gpu_va=0x{notifier_gpu_va:010x}")
        passed += 1
    except Exception as e:
        log(f"  FAIL: {e}")
        return 1
    signal.alarm(0)

    # ---- Step 6: mmap ctrl fd for doorbell ----
    total += 1
    log("\nStep 6: mmap ctrl fd for doorbell...")
    signal.alarm(5)
    try:
        doorbell_addr = do_mmap(None, 0x10000, ctrl_fd)
        log(f"  OK: doorbell base=0x{doorbell_addr:x}")
        passed += 1
    except Exception as e:
        log(f"  FAIL: {e}")
        return 1
    signal.alarm(0)

    # ---- Step 7: OPEN_TSG ----
    total += 1
    log("\nStep 7: OPEN_TSG...")
    signal.alarm(5)
    try:
        tsg = nvgpu_gpu_open_tsg_args()
        nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_OPEN_TSG, tsg)
        tsg_fd = tsg.tsg_fd
        log(f"  OK: tsg_fd={tsg_fd}")
        passed += 1
    except Exception as e:
        log(f"  FAIL: {e}")
        return 1
    signal.alarm(0)

    # ---- Step 8: CREATE_SUBCONTEXT ----
    total += 1
    log("\nStep 8: CREATE_SUBCONTEXT...")
    signal.alarm(5)
    try:
        subctx = nvgpu_tsg_create_subcontext_args()
        subctx.type = 1; subctx.as_fd = as_fd
        nv_ioctl(tsg_fd, NVGPU_TSG_IOCTL_CREATE_SUBCONTEXT, subctx)
        log(f"  OK: veid={subctx.veid}")
        passed += 1
    except Exception as e:
        log(f"  FAIL: {e}")
        return 1
    signal.alarm(0)

    # ---- Step 9: Channel 1 (compute) — OPEN + AS_BIND + TSG_BIND + WDT + SETUP_BIND ----
    total += 1
    log(f"\nStep 9: Create compute channel (entries={ENTRIES})...")
    signal.alarm(10)
    try:
        # OPEN_CHANNEL
        ch = nvgpu_gpu_open_channel_args(); ch.channel_fd = -1
        nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_OPEN_CHANNEL, ch)
        ch1_fd = ch.channel_fd
        log(f"  OPEN_CHANNEL: ch_fd={ch1_fd}")

        # AS_BIND
        ab = nvgpu_as_bind_channel_args(); ab.channel_fd = ch1_fd
        nv_ioctl(as_fd, NVGPU_AS_IOCTL_BIND_CHANNEL, ab)
        log("  AS_BIND: OK")

        # TSG_BIND
        tb = nvgpu_tsg_bind_channel_ex_args(); tb.channel_fd = ch1_fd; tb.subcontext_id = subctx.veid
        nv_ioctl(tsg_fd, NVGPU_TSG_IOCTL_BIND_CHANNEL_EX, tb)
        log("  TSG_BIND: OK")

        # WDT disable
        wdt = nvgpu_channel_wdt_args(); wdt.wdt_status = 1
        nv_ioctl(ch1_fd, NVGPU_IOCTL_CHANNEL_WDT, wdt)
        log("  WDT disable: OK")

        # Separate userd buffer
        userd1_handle, userd1_dmabuf = nvmap_alloc_buf(nvmap_fd, 4096, NVMAP_HANDLE_WRITE_COMBINE)
        log(f"  Userd alloc: handle={userd1_handle} dmabuf={userd1_dmabuf}")

        # SETUP_BIND
        sb = nvgpu_channel_setup_bind_args()
        sb.num_gpfifo_entries = ENTRIES
        sb.num_inflight_jobs = 0
        sb.gpfifo_dmabuf_fd = gpfifo_dmabuf
        sb.gpfifo_dmabuf_offset = 0  # compute at offset 0
        sb.userd_dmabuf_fd = userd1_dmabuf
        sb.userd_dmabuf_offset = 0
        sb.flags = NVGPU_SETUP_BIND_FLAGS_USERMODE_SUPPORT | NVGPU_SETUP_BIND_FLAGS_DETERMINISTIC
        nv_ioctl(ch1_fd, NVGPU_IOCTL_CHANNEL_SETUP_BIND, sb)
        token1 = sb.work_submit_token
        log(f"  SETUP_BIND: OK, token={token1}, gpfifo_gpu_va=0x{sb.gpfifo_gpu_va:x}, userd_gpu_va=0x{sb.userd_gpu_va:x}")

        # MAP_FIXED userd overlay  
        userd_offset = ENTRIES * 8  # userd region within gpfifo_area
        userd_cpu = do_mmap(gpfifo_cpu + userd_offset, 4096, userd1_dmabuf, fixed=True)
        log(f"  Userd overlay: 0x{userd_cpu:x} (target was 0x{gpfifo_cpu + userd_offset:x})")

        passed += 1
    except Exception as e:
        log(f"  FAIL: {e}")
        import traceback; log(traceback.format_exc())
        return 1
    signal.alarm(0)

    # ---- Step 10: ALLOC_OBJ_CTX for compute class ----
    total += 1
    log(f"\nStep 10: ALLOC_OBJ_CTX compute class 0x{compute_class:04x}...")
    log(f"  >>> ABOUT TO CALL ALLOC_OBJ_CTX ioctl on ch_fd={ch1_fd}")
    signal.alarm(15)  # This is the suspected crash point
    try:
        obj = nvgpu_alloc_obj_ctx_args(); obj.class_num = compute_class
        log(f"  >>> calling ioctl NOW...")
        nv_ioctl(ch1_fd, NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX, obj)
        log(f"  OK: obj_id={obj.obj_id}")
        passed += 1
    except Exception as e:
        log(f"  FAIL: {e}")
        import traceback; log(traceback.format_exc())
    signal.alarm(0)

    # ---- Step 11: Channel 2 (DMA) — same flow ----
    total += 1
    log(f"\nStep 11: Create DMA channel (entries={ENTRIES})...")
    signal.alarm(10)
    try:
        ch2 = nvgpu_gpu_open_channel_args(); ch2.channel_fd = -1
        log("  11a: OPEN_CHANNEL...")
        nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_OPEN_CHANNEL, ch2)
        ch2_fd = ch2.channel_fd
        log(f"  11a: OPEN_CHANNEL: ch_fd={ch2_fd}")

        ab2 = nvgpu_as_bind_channel_args(); ab2.channel_fd = ch2_fd
        log("  11b: AS_BIND...")
        nv_ioctl(as_fd, NVGPU_AS_IOCTL_BIND_CHANNEL, ab2)
        log("  11b: AS_BIND: OK")

        tb2 = nvgpu_tsg_bind_channel_ex_args(); tb2.channel_fd = ch2_fd; tb2.subcontext_id = subctx.veid
        log("  11c: TSG_BIND...")
        nv_ioctl(tsg_fd, NVGPU_TSG_IOCTL_BIND_CHANNEL_EX, tb2)
        log("  11c: TSG_BIND: OK")

        wdt2 = nvgpu_channel_wdt_args(); wdt2.wdt_status = 1
        log("  11d: WDT disable...")
        nv_ioctl(ch2_fd, NVGPU_IOCTL_CHANNEL_WDT, wdt2)
        log("  11d: WDT disable: OK")

        # Allocate SEPARATE gpfifo buffer for DMA channel (NOT shared with compute)
        dma_gpfifo_size = ENTRIES * 8 + 4096  # ring + userd
        log(f"  11e: Alloc separate DMA gpfifo ({dma_gpfifo_size} bytes)...")
        dma_gpfifo_handle, dma_gpfifo_dmabuf = nvmap_alloc_buf(nvmap_fd, dma_gpfifo_size, NVMAP_HANDLE_INNER_CACHEABLE)
        dma_gpfifo_gpu_va = gpu_map(as_fd, dma_gpfifo_dmabuf, dma_gpfifo_size)
        dma_gpfifo_cpu = do_mmap(None, dma_gpfifo_size, dma_gpfifo_dmabuf)
        log(f"  11e: DMA gpfifo: gpu_va=0x{dma_gpfifo_gpu_va:010x} cpu=0x{dma_gpfifo_cpu:x}")

        log("  11f: Alloc userd2...")
        userd2_handle, userd2_dmabuf = nvmap_alloc_buf(nvmap_fd, 4096, NVMAP_HANDLE_WRITE_COMBINE)
        log(f"  11f: Userd alloc: handle={userd2_handle} dmabuf={userd2_dmabuf}")

        sb2 = nvgpu_channel_setup_bind_args()
        sb2.num_gpfifo_entries = ENTRIES
        sb2.num_inflight_jobs = 0
        sb2.gpfifo_dmabuf_fd = dma_gpfifo_dmabuf  # SEPARATE dmabuf
        sb2.gpfifo_dmabuf_offset = 0  # offset 0 in its own buffer
        sb2.userd_dmabuf_fd = userd2_dmabuf
        sb2.userd_dmabuf_offset = 0
        sb2.flags = NVGPU_SETUP_BIND_FLAGS_USERMODE_SUPPORT | NVGPU_SETUP_BIND_FLAGS_DETERMINISTIC
        log(f"  11g: SETUP_BIND (gpfifo_fd={dma_gpfifo_dmabuf}, userd_fd={userd2_dmabuf})...")
        nv_ioctl(ch2_fd, NVGPU_IOCTL_CHANNEL_SETUP_BIND, sb2)
        token2 = sb2.work_submit_token
        log(f"  11g: SETUP_BIND: OK, token={token2}")

        # MAP_FIXED userd2 overlay into dma gpfifo area
        log("  11h: MAP_FIXED userd2...")
        userd2_cpu = do_mmap(dma_gpfifo_cpu + ENTRIES*8, 4096, userd2_dmabuf, fixed=True)
        log(f"  11h: Userd overlay: 0x{userd2_cpu:x}")

        passed += 1
    except Exception as e:
        log(f"  FAIL: {e}")
        import traceback; log(traceback.format_exc())
    signal.alarm(0)

    # ---- Step 12: ALLOC_OBJ_CTX for DMA class ----
    total += 1
    log(f"\nStep 12: ALLOC_OBJ_CTX DMA class 0x{dma_class:04x}...")
    signal.alarm(15)
    try:
        obj2 = nvgpu_alloc_obj_ctx_args(); obj2.class_num = dma_class
        log("  12: calling ioctl...")
        nv_ioctl(ch2_fd, NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX, obj2)
        log(f"  OK: obj_id={obj2.obj_id}")
        passed += 1
    except Exception as e:
        log(f"  FAIL: {e}")
        import traceback; log(traceback.format_exc())
    signal.alarm(0)

    # ---- Step 13: Allocate cmdq_page (2MB) ----
    total += 1
    log("\nStep 13: Allocate cmdq_page (2MB)...")
    signal.alarm(10)
    try:
        cmdq_handle, cmdq_dmabuf = nvmap_alloc_buf(nvmap_fd, 0x200000, NVMAP_HANDLE_INNER_CACHEABLE)
        cmdq_gpu_va = gpu_map(as_fd, cmdq_dmabuf, 0x200000)
        cmdq_cpu = do_mmap(None, 0x200000, cmdq_dmabuf)
        log(f"  OK: gpu_va=0x{cmdq_gpu_va:010x} cpu=0x{cmdq_cpu:x}")
        passed += 1
    except Exception as e:
        log(f"  FAIL: {e}")
    signal.alarm(0)

    # ---- Step 14: Submit a simple semaphore release (like test_nvgpu.py Test 13) ----
    total += 1
    log("\nStep 14: Submit semaphore release via compute GPFIFO...")
    signal.alarm(10)
    try:
        # Allocate a semaphore buffer
        sem_handle, sem_dmabuf = nvmap_alloc_buf(nvmap_fd, 4096, NVMAP_HANDLE_WRITE_COMBINE)
        sem_gpu_va = gpu_map(as_fd, sem_dmabuf, 4096)
        sem_cpu = do_mmap(None, 4096, sem_dmabuf)
        log(f"  Semaphore: gpu_va=0x{sem_gpu_va:010x} cpu=0x{sem_cpu:x}")

        # Clear semaphore
        struct.pack_into('<Q', (sem_mv:=ctypes.string_at(sem_cpu, 4096)), 0, 0)  # can't use this
        # Use ctypes directly
        ctypes.memset(sem_cpu, 0, 64)

        # Build push buffer: SET_OBJECT(compute) + SEM release
        # subchannel 1 = compute, method NVC6C0_SET_OBJECT = 0x0000
        NVC56F_SEM_ADDR_LO = 0x5a0
        NVC56F_SEM_EXECUTE = 0x5b0
        
        # Push buffer: semaphore release on subchannel 0 (channel class)
        pb = []
        # SEM_ADDR_LO (method 0x5a0, subchannel 0, 4 words)
        pb.append((2 << 28) | (4 << 16) | (0 << 13) | (NVC56F_SEM_ADDR_LO >> 2))
        pb.append(sem_gpu_va & 0xffffffff)  # SEM_ADDR_LO
        pb.append(sem_gpu_va >> 32)         # SEM_ADDR_HI
        pb.append(0xDEAD)                   # SEM_PAYLOAD_LO
        pb.append(0)                        # SEM_PAYLOAD_HI
        # SEM_EXECUTE
        pb.append((2 << 28) | (1 << 16) | (0 << 13) | (NVC56F_SEM_EXECUTE >> 2))
        # operation=release(1), release_wfi=en(1<<20), payload_size=64bit(1<<24), release_timestamp=en(1<<25)
        pb.append(1 | (1<<20) | (1<<24) | (1<<25))

        # Write push buffer to cmdq memory
        pb_offset = 0  # write at start of cmdq
        for i, word in enumerate(pb):
            struct.pack_into('<I', ctypes.string_at(cmdq_cpu, 0x200000), pb_offset + i*4, word)
        # Actually use ctypes pointer
        pb_arr = (ctypes.c_uint32 * len(pb))(*pb)
        ctypes.memmove(cmdq_cpu + pb_offset, pb_arr, len(pb)*4)
        pb_gpu_va = cmdq_gpu_va + pb_offset
        log(f"  Push buffer: {len(pb)} words at gpu_va=0x{pb_gpu_va:010x}")

        # Write GPFIFO entry
        gpfifo_entry = (pb_gpu_va & ~3) | (len(pb) << 42) | (1 << 41)
        gp_put_ptr = gpfifo_cpu  # compute ring at offset 0
        struct.pack_into('<Q', ctypes.string_at(gpfifo_cpu, 8), 0, gpfifo_entry)
        # Actually use ctypes
        ctypes.c_uint64.from_address(gpfifo_cpu).value = gpfifo_entry
        log(f"  GPFIFO entry: 0x{gpfifo_entry:016x}")

        # Write GPPut = 1
        GP_PUT_OFFSET = 0x8C  # offset within userd (AmpereAControlGPFifo.GPPut)
        userd_base = gpfifo_cpu + ENTRIES * 8  # userd overlay address
        ctypes.c_uint32.from_address(userd_base + GP_PUT_OFFSET).value = 1
        log(f"  GPPut written: 1 at 0x{userd_base + GP_PUT_OFFSET:x}")

        # Memory barrier
        try:
            libatomic = ctypes.CDLL("libatomic.so.1")
            libatomic.atomic_thread_fence(5)  # SEQ_CST
        except: pass

        # Ring doorbell
        ctypes.c_uint32.from_address(doorbell_addr + 0x90).value = token1
        log(f"  Doorbell: wrote token {token1} at offset 0x90")

        # Wait for semaphore
        t0 = time.monotonic()
        while time.monotonic() - t0 < 5.0:
            val = ctypes.c_uint32.from_address(sem_cpu).value
            if val == 0xDEAD:
                elapsed = (time.monotonic() - t0) * 1000
                log(f"  SEMAPHORE RELEASED! val=0x{val:x} ({elapsed:.1f}ms)")
                passed += 1
                break
            time.sleep(0.001)
        else:
            val = ctypes.c_uint32.from_address(sem_cpu).value
            log(f"  TIMEOUT after 5s! sem val=0x{val:x} (expected 0xDEAD)")
            # Read GPGet to see if GPU advanced
            gp_get = ctypes.c_uint32.from_address(userd_base + 0x88).value
            gp_put = ctypes.c_uint32.from_address(userd_base + 0x8C).value
            log(f"  GPGet={gp_get} GPPut={gp_put}")
    except Exception as e:
        log(f"  FAIL: {e}")
        import traceback; log(traceback.format_exc())
    signal.alarm(0)

    log(f"\n{'='*60}")
    log(f"RESULTS: {passed}/{total} steps passed")
    log(f"{'='*60}")
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
