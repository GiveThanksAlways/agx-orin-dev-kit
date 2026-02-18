#!/usr/bin/env python3
"""
A/B test: find exactly which condition causes ALLOC_OBJ_CTX to hang.

test_nvgpu.py's channel setup WORKS. This script starts from that EXACT
working sequence and changes ONE variable at a time to isolate the crash.

Variant A: EXACT copy of test_nvgpu.py's working channel setup (baseline)
Variant B: Use INNER_CACHEABLE instead of WRITE_COMBINE for GPFIFO buffer
Variant C: MAP_BUFFER_EX the GPFIFO dmabuf before SETUP_BIND
Variant D: Use a large (3MB) GPFIFO buffer instead of 8KB
Variant E: Combination (large + mapped + cacheable) = tinygrad's approach

Run: python3 -u test_channel_variants.py [A|B|C|D|E|all]
"""
import os, sys, struct, ctypes, fcntl, mmap, time, signal

# ============================================================================
# File logger
# ============================================================================
LOG_PATH = "/home/agent/jetpack-nixos/examples/tinygrad/test_channel_variants.log"
_log_fd = None
def log(msg):
    global _log_fd
    print(msg, flush=True)
    if _log_fd is None:
        _log_fd = os.open(LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(_log_fd, (msg + "\n").encode())
    os.fsync(_log_fd)

def timeout_handler(signum, frame):
    log("\n*** TIMEOUT — ioctl hung! ***")
    os._exit(1)
signal.signal(signal.SIGALRM, timeout_handler)

# ============================================================================
# ioctl helpers
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

# ============================================================================
# Structs (minimal — only what we need)
# ============================================================================
class nvgpu_gpu_characteristics(ctypes.Structure):
    _fields_ = [
        ("arch", ctypes.c_uint32), ("impl", ctypes.c_uint32), ("rev", ctypes.c_uint32),
        ("num_gpc", ctypes.c_uint32), ("numa_domain_id", ctypes.c_int32), ("_pad0", ctypes.c_uint32),
        ("L2_cache_size", ctypes.c_uint64), ("on_board_video_memory_size", ctypes.c_uint64),
        ("num_tpc_per_gpc", ctypes.c_uint32), ("bus_type", ctypes.c_uint32),
        ("big_page_size", ctypes.c_uint32), ("compression_page_size", ctypes.c_uint32),
        ("pde_coverage_bit_count", ctypes.c_uint32), ("available_big_page_sizes", ctypes.c_uint32),
        ("flags", ctypes.c_uint64),
        ("twod_class", ctypes.c_uint32), ("threed_class", ctypes.c_uint32),
        ("compute_class", ctypes.c_uint32), ("gpfifo_class", ctypes.c_uint32),
        ("inline_to_memory_class", ctypes.c_uint32), ("dma_copy_class", ctypes.c_uint32),
        ("gpc_mask", ctypes.c_uint32), ("sm_arch_sm_version", ctypes.c_uint32),
        ("sm_arch_spa_version", ctypes.c_uint32), ("sm_arch_warp_count", ctypes.c_uint32),
        ("gpu_ioctl_nr_last", ctypes.c_int16), ("tsg_ioctl_nr_last", ctypes.c_int16),
        ("dbg_gpu_ioctl_nr_last", ctypes.c_int16), ("ioctl_channel_nr_last", ctypes.c_int16),
        ("as_ioctl_nr_last", ctypes.c_int16),
        ("gpu_va_bit_count", ctypes.c_uint8), ("reserved", ctypes.c_uint8),
        ("max_fbps_count", ctypes.c_uint32), ("fbp_en_mask", ctypes.c_uint32),
        ("emc_en_mask", ctypes.c_uint32), ("max_ltc_per_fbp", ctypes.c_uint32),
        ("max_lts_per_ltc", ctypes.c_uint32), ("max_tex_per_tpc", ctypes.c_uint32),
        ("max_gpc_count", ctypes.c_uint32),
        ("rop_l2_en_mask_DEPRECATED", ctypes.c_uint32 * 2),
        ("chipname", ctypes.c_uint8 * 8),
        ("gr_compbit_store_base_hw", ctypes.c_uint64),
        ("gr_gobs_per_comptagline_per_slice", ctypes.c_uint32), ("num_ltc", ctypes.c_uint32),
        ("lts_per_ltc", ctypes.c_uint32), ("cbc_cache_line_size", ctypes.c_uint32),
        ("cbc_comptags_per_line", ctypes.c_uint32), ("map_buffer_batch_limit", ctypes.c_uint32),
        ("max_freq", ctypes.c_uint64),
        ("graphics_preemption_mode_flags", ctypes.c_uint32), ("compute_preemption_mode_flags", ctypes.c_uint32),
        ("default_graphics_preempt_mode", ctypes.c_uint32), ("default_compute_preempt_mode", ctypes.c_uint32),
        ("local_video_memory_size", ctypes.c_uint64),
        ("pci_vendor_id", ctypes.c_uint16), ("pci_device_id", ctypes.c_uint16),
        ("pci_subsystem_vendor_id", ctypes.c_uint16), ("pci_subsystem_device_id", ctypes.c_uint16),
        ("pci_class", ctypes.c_uint16), ("pci_revision", ctypes.c_uint8),
        ("vbios_oem_version", ctypes.c_uint8), ("vbios_version", ctypes.c_uint32),
        ("reg_ops_limit", ctypes.c_uint32), ("reserved1", ctypes.c_uint32),
        ("event_ioctl_nr_last", ctypes.c_int16), ("pad", ctypes.c_uint16),
        ("max_css_buffer_size", ctypes.c_uint32),
        ("ctxsw_ioctl_nr_last", ctypes.c_int16), ("prof_ioctl_nr_last", ctypes.c_int16),
        ("nvs_ioctl_nr_last", ctypes.c_int16), ("reserved2", ctypes.c_uint8 * 2),
        ("max_ctxsw_ring_buffer_size", ctypes.c_uint32), ("reserved3", ctypes.c_uint32),
        ("per_device_identifier", ctypes.c_uint64),
        ("num_ppc_per_gpc", ctypes.c_uint32), ("max_veid_count_per_tsg", ctypes.c_uint32),
        ("num_sub_partition_per_fbpa", ctypes.c_uint32), ("gpu_instance_id", ctypes.c_uint32),
        ("gr_instance_id", ctypes.c_uint32), ("max_gpfifo_entries", ctypes.c_uint32),
        ("max_dbg_tsg_timeslice", ctypes.c_uint32), ("reserved5", ctypes.c_uint32),
        ("device_instance_id", ctypes.c_uint64),
    ]

class nvgpu_gpu_get_characteristics(ctypes.Structure):
    _fields_ = [("gpu_characteristics_buf_size", ctypes.c_uint64), ("gpu_characteristics_buf_addr", ctypes.c_uint64)]

class nvmap_create_handle(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint32), ("handle", ctypes.c_uint32)]

class nvmap_alloc_handle(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("heap_mask", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("align", ctypes.c_uint32), ("numa_nid", ctypes.c_int32)]

class nvgpu_alloc_as_args(ctypes.Structure):
    _fields_ = [("big_page_size", ctypes.c_uint32), ("as_fd", ctypes.c_int32), ("flags", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32), ("va_range_start", ctypes.c_uint64), ("va_range_end", ctypes.c_uint64),
                ("va_range_split", ctypes.c_uint64), ("padding", ctypes.c_uint32 * 6)]

class nvgpu_as_bind_channel_args(ctypes.Structure):
    _fields_ = [("channel_fd", ctypes.c_uint32)]

class nvgpu_as_map_buffer_ex_args(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("compr_kind", ctypes.c_int16), ("incompr_kind", ctypes.c_int16),
                ("dmabuf_fd", ctypes.c_uint32), ("page_size", ctypes.c_uint32), ("buffer_offset", ctypes.c_uint64),
                ("mapping_size", ctypes.c_uint64), ("offset", ctypes.c_uint64)]

class nvgpu_gpu_open_tsg_args(ctypes.Structure):
    _fields_ = [("tsg_fd", ctypes.c_int32), ("flags", ctypes.c_uint32), ("token", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32), ("subctx_id", ctypes.c_uint32), ("_pad", ctypes.c_uint32)]

class nvgpu_tsg_bind_channel_ex_args(ctypes.Structure):
    _fields_ = [("channel_fd", ctypes.c_int32), ("subcontext_id", ctypes.c_uint32), ("reserved", ctypes.c_uint8 * 16)]

class nvgpu_tsg_create_subcontext_args(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("as_fd", ctypes.c_int32), ("veid", ctypes.c_uint32), ("reserved", ctypes.c_uint32)]

class nvgpu_gpu_open_channel_args(ctypes.Structure):
    _fields_ = [("channel_fd", ctypes.c_int32)]

class nvgpu_alloc_obj_ctx_args(ctypes.Structure):
    _fields_ = [("class_num", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("obj_id", ctypes.c_uint64)]

class nvgpu_channel_setup_bind_args(ctypes.Structure):
    _fields_ = [("num_gpfifo_entries", ctypes.c_uint32), ("num_inflight_jobs", ctypes.c_uint32),
                ("flags", ctypes.c_uint32), ("userd_dmabuf_fd", ctypes.c_int32), ("gpfifo_dmabuf_fd", ctypes.c_int32),
                ("work_submit_token", ctypes.c_uint32), ("userd_dmabuf_offset", ctypes.c_uint64),
                ("gpfifo_dmabuf_offset", ctypes.c_uint64), ("gpfifo_gpu_va", ctypes.c_uint64),
                ("userd_gpu_va", ctypes.c_uint64), ("usermode_mmio_gpu_va", ctypes.c_uint64),
                ("reserved", ctypes.c_uint32 * 9)]

class nvgpu_channel_wdt_args(ctypes.Structure):
    _fields_ = [("wdt_status", ctypes.c_uint32), ("timeout_ms", ctypes.c_uint32)]

# ioctl codes
NVGPU_GPU_IOCTL_GET_CHARACTERISTICS = _IOWR('G', 5, ctypes.sizeof(nvgpu_gpu_get_characteristics))
NVGPU_GPU_IOCTL_ALLOC_AS = _IOWR('G', 8, ctypes.sizeof(nvgpu_alloc_as_args))
NVGPU_GPU_IOCTL_OPEN_TSG = _IOWR('G', 9, ctypes.sizeof(nvgpu_gpu_open_tsg_args))
NVGPU_GPU_IOCTL_OPEN_CHANNEL = _IOWR('G', 11, ctypes.sizeof(nvgpu_gpu_open_channel_args))
NVMAP_IOC_CREATE = _IOWR('N', 0, ctypes.sizeof(nvmap_create_handle))
NVMAP_IOC_ALLOC = _IOW('N', 3, ctypes.sizeof(nvmap_alloc_handle))
NVMAP_IOC_GET_FD = _IOWR('N', 15, ctypes.sizeof(nvmap_create_handle))
NVGPU_AS_IOCTL_BIND_CHANNEL = _IOWR('A', 1, ctypes.sizeof(nvgpu_as_bind_channel_args))
NVGPU_AS_IOCTL_MAP_BUFFER_EX = _IOWR('A', 7, ctypes.sizeof(nvgpu_as_map_buffer_ex_args))
NVGPU_TSG_IOCTL_BIND_CHANNEL_EX = _IOWR('T', 11, ctypes.sizeof(nvgpu_tsg_bind_channel_ex_args))
NVGPU_TSG_IOCTL_CREATE_SUBCONTEXT = _IOWR('T', 18, ctypes.sizeof(nvgpu_tsg_create_subcontext_args))
NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX = _IOWR('H', 108, ctypes.sizeof(nvgpu_alloc_obj_ctx_args))
NVGPU_IOCTL_CHANNEL_SETUP_BIND = _IOWR('H', 128, ctypes.sizeof(nvgpu_channel_setup_bind_args))
NVGPU_IOCTL_CHANNEL_WDT = _IOW('H', 119, ctypes.sizeof(nvgpu_channel_wdt_args))

NVMAP_HEAP_IOVMM = 1 << 30
NVMAP_HANDLE_WRITE_COMBINE = 1
NVMAP_HANDLE_INNER_CACHEABLE = 2
SETUP_BIND_USERMODE = 1 << 3
SETUP_BIND_DETERMINISTIC = 1 << 1

# ============================================================================
# Helpers
# ============================================================================
def nvmap_create_buf(nvmap_fd, size, flags=NVMAP_HANDLE_WRITE_COMBINE):
    """Create + alloc + get_fd. Returns (handle, dmabuf_fd)."""
    c = nvmap_create_handle(); c.size = size
    nv_ioctl(nvmap_fd, NVMAP_IOC_CREATE, c)
    a = nvmap_alloc_handle(); a.handle = c.handle; a.heap_mask = NVMAP_HEAP_IOVMM
    a.flags = flags; a.align = 4096; a.numa_nid = 0
    nv_ioctl(nvmap_fd, NVMAP_IOC_ALLOC, a)
    g = nvmap_create_handle(); g.handle = c.handle
    nv_ioctl(nvmap_fd, NVMAP_IOC_GET_FD, g)
    return c.handle, g.size  # g.size = dmabuf fd

def gpu_map(as_fd, dmabuf_fd, size):
    """MAP_BUFFER_EX → returns GPU VA assigned by kernel."""
    m = nvgpu_as_map_buffer_ex_args()
    m.flags = 0; m.compr_kind = -1; m.incompr_kind = 0
    m.dmabuf_fd = dmabuf_fd; m.page_size = 4096
    m.buffer_offset = 0; m.mapping_size = 0; m.offset = 0
    nv_ioctl(as_fd, NVGPU_AS_IOCTL_MAP_BUFFER_EX, m)
    return m.offset

# ============================================================================
# Channel setup function — parameterized for A/B testing
# ============================================================================
def setup_channel(ctrl_fd, as_fd, nvmap_fd, tsg_fd, veid, compute_class,
                  gpfifo_size=8192, gpfifo_flags=NVMAP_HANDLE_WRITE_COMBINE,
                  map_gpfifo_to_gpu=False, entries=1024, label=""):
    """
    Full channel setup: OPEN_CHANNEL → AS_BIND → TSG_BIND → WDT → SETUP_BIND → ALLOC_OBJ_CTX.
    
    Arguments control what differs between variants:
      gpfifo_size: size of GPFIFO nvmap buffer (8KB vs 3MB)
      gpfifo_flags: WRITE_COMBINE vs INNER_CACHEABLE
      map_gpfifo_to_gpu: whether to MAP_BUFFER_EX before SETUP_BIND
      entries: number of GPFIFO entries
    """
    log(f"\n{'='*60}")
    log(f"VARIANT {label}: gpfifo_size={gpfifo_size}, flags={'IC' if gpfifo_flags==2 else 'WC'}, "
        f"gpu_map={'YES' if map_gpfifo_to_gpu else 'NO'}, entries={entries}")
    log(f"{'='*60}")

    # 1. Allocate GPFIFO buffer
    log("  1. Alloc GPFIFO buffer...")
    signal.alarm(5)
    gpfifo_handle, gpfifo_dmabuf = nvmap_create_buf(nvmap_fd, gpfifo_size, gpfifo_flags)
    log(f"     handle={gpfifo_handle}, dmabuf={gpfifo_dmabuf}, size={gpfifo_size}")
    signal.alarm(0)

    # 2. Optionally GPU-map it BEFORE channel setup
    gpfifo_gpu_va = 0
    if map_gpfifo_to_gpu:
        log("  2. MAP_BUFFER_EX (GPU map BEFORE SETUP_BIND)...")
        signal.alarm(5)
        gpfifo_gpu_va = gpu_map(as_fd, gpfifo_dmabuf, gpfifo_size)
        log(f"     gpu_va=0x{gpfifo_gpu_va:010x}")
        signal.alarm(0)
    else:
        log("  2. Skip GPU map (not mapping before SETUP_BIND)")

    # 3. Allocate separate USERD buffer (always WC, always 4KB)
    log("  3. Alloc USERD buffer...")
    signal.alarm(5)
    userd_handle, userd_dmabuf = nvmap_create_buf(nvmap_fd, 4096, NVMAP_HANDLE_WRITE_COMBINE)
    log(f"     handle={userd_handle}, dmabuf={userd_dmabuf}")
    signal.alarm(0)

    # 4. OPEN_CHANNEL
    log("  4. OPEN_CHANNEL...")
    signal.alarm(5)
    ch = nvgpu_gpu_open_channel_args(); ch.channel_fd = -1
    nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_OPEN_CHANNEL, ch)
    ch_fd = ch.channel_fd
    log(f"     ch_fd={ch_fd}")
    signal.alarm(0)

    # 5. AS_BIND
    log("  5. AS_BIND...")
    signal.alarm(5)
    ab = nvgpu_as_bind_channel_args(); ab.channel_fd = ch_fd
    nv_ioctl(as_fd, NVGPU_AS_IOCTL_BIND_CHANNEL, ab)
    log("     OK")
    signal.alarm(0)

    # 6. TSG_BIND
    log("  6. TSG_BIND...")
    signal.alarm(5)
    tb = nvgpu_tsg_bind_channel_ex_args(); tb.channel_fd = ch_fd; tb.subcontext_id = veid
    nv_ioctl(tsg_fd, NVGPU_TSG_IOCTL_BIND_CHANNEL_EX, tb)
    log("     OK")
    signal.alarm(0)

    # 7. WDT disable
    log("  7. WDT disable...")
    signal.alarm(5)
    wdt = nvgpu_channel_wdt_args(); wdt.wdt_status = 1
    nv_ioctl(ch_fd, NVGPU_IOCTL_CHANNEL_WDT, wdt)
    log("     OK")
    signal.alarm(0)

    # 8. SETUP_BIND
    log("  8. SETUP_BIND...")
    signal.alarm(10)
    sb = nvgpu_channel_setup_bind_args()
    sb.num_gpfifo_entries = entries
    sb.num_inflight_jobs = 0
    sb.gpfifo_dmabuf_fd = gpfifo_dmabuf
    sb.gpfifo_dmabuf_offset = 0
    sb.userd_dmabuf_fd = userd_dmabuf
    sb.userd_dmabuf_offset = 0
    sb.flags = SETUP_BIND_USERMODE | SETUP_BIND_DETERMINISTIC
    nv_ioctl(ch_fd, NVGPU_IOCTL_CHANNEL_SETUP_BIND, sb)
    log(f"     OK: token={sb.work_submit_token}")
    signal.alarm(0)

    # 9. ALLOC_OBJ_CTX
    log(f"  9. ALLOC_OBJ_CTX (0x{compute_class:04x})...")
    log(f"     >>> calling ioctl NOW on ch_fd={ch_fd}")
    signal.alarm(15)
    obj = nvgpu_alloc_obj_ctx_args(); obj.class_num = compute_class
    nv_ioctl(ch_fd, NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX, obj)
    log(f"     OK: obj_id=0x{obj.obj_id:x}")
    signal.alarm(0)

    log(f"  *** VARIANT {label} PASSED ***")
    return True

# ============================================================================
# Main
# ============================================================================
def main():
    variant = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
    log(f"=== Channel Variant Test — variant={variant} ===")
    log(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Common setup
    log("\n--- Common setup ---")
    nvmap_fd = os.open("/dev/nvmap", os.O_RDWR | os.O_SYNC)
    ctrl_fd = os.open("/dev/nvgpu/igpu0/ctrl", os.O_RDWR)
    log(f"  nvmap_fd={nvmap_fd}, ctrl_fd={ctrl_fd}")

    chars = nvgpu_gpu_characteristics()
    req = nvgpu_gpu_get_characteristics()
    req.gpu_characteristics_buf_size = ctypes.sizeof(chars)
    req.gpu_characteristics_buf_addr = ctypes.addressof(chars)
    nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_GET_CHARACTERISTICS, req)
    compute_class = chars.compute_class
    log(f"  compute_class=0x{compute_class:04x}, sm=0x{chars.sm_arch_sm_version:x}")

    PDE = 1 << 21
    as_args = nvgpu_alloc_as_args()
    as_args.big_page_size = 0; as_args.flags = 2
    as_args.va_range_start = PDE; as_args.va_range_end = (1<<40)-PDE
    nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_ALLOC_AS, as_args)
    as_fd = as_args.as_fd
    log(f"  as_fd={as_fd}")

    tsg = nvgpu_gpu_open_tsg_args()
    nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_OPEN_TSG, tsg)
    tsg_fd = tsg.tsg_fd
    log(f"  tsg_fd={tsg_fd}")

    subctx = nvgpu_tsg_create_subcontext_args()
    subctx.type = 1; subctx.as_fd = as_fd
    nv_ioctl(tsg_fd, NVGPU_TSG_IOCTL_CREATE_SUBCONTEXT, subctx)
    veid = subctx.veid
    log(f"  veid={veid}")

    # Run selected variant(s)
    variants_to_run = list("ABCDE") if variant == "ALL" else [variant]
    results = {}

    for v in variants_to_run:
        try:
            if v == "A":
                # Baseline: EXACT copy of test_nvgpu.py's working setup
                ok = setup_channel(ctrl_fd, as_fd, nvmap_fd, tsg_fd, veid, compute_class,
                    gpfifo_size=8192, gpfifo_flags=NVMAP_HANDLE_WRITE_COMBINE,
                    map_gpfifo_to_gpu=False, entries=1024, label="A (baseline)")
                results[v] = "PASS" if ok else "FAIL"

            elif v == "B":
                # Change: INNER_CACHEABLE instead of WRITE_COMBINE
                ok = setup_channel(ctrl_fd, as_fd, nvmap_fd, tsg_fd, veid, compute_class,
                    gpfifo_size=8192, gpfifo_flags=NVMAP_HANDLE_INNER_CACHEABLE,
                    map_gpfifo_to_gpu=False, entries=1024, label="B (cacheable)")
                results[v] = "PASS" if ok else "FAIL"

            elif v == "C":
                # Change: MAP_BUFFER_EX before SETUP_BIND
                ok = setup_channel(ctrl_fd, as_fd, nvmap_fd, tsg_fd, veid, compute_class,
                    gpfifo_size=8192, gpfifo_flags=NVMAP_HANDLE_WRITE_COMBINE,
                    map_gpfifo_to_gpu=True, entries=1024, label="C (gpu_map)")
                results[v] = "PASS" if ok else "FAIL"

            elif v == "D":
                # Change: Large buffer (3MB)
                ok = setup_channel(ctrl_fd, as_fd, nvmap_fd, tsg_fd, veid, compute_class,
                    gpfifo_size=0x300000, gpfifo_flags=NVMAP_HANDLE_WRITE_COMBINE,
                    map_gpfifo_to_gpu=False, entries=1024, label="D (large buf)")
                results[v] = "PASS" if ok else "FAIL"

            elif v == "E":
                # All changes combined (= tinygrad's approach)
                ok = setup_channel(ctrl_fd, as_fd, nvmap_fd, tsg_fd, veid, compute_class,
                    gpfifo_size=0x300000, gpfifo_flags=NVMAP_HANDLE_INNER_CACHEABLE,
                    map_gpfifo_to_gpu=True, entries=1024, label="E (tinygrad)")
                results[v] = "PASS" if ok else "FAIL"

        except Exception as e:
            log(f"  *** VARIANT {v} FAILED: {e} ***")
            results[v] = f"FAIL: {e}"

    # Summary
    log(f"\n{'='*60}")
    log("RESULTS SUMMARY")
    log(f"{'='*60}")
    for v, r in results.items():
        log(f"  Variant {v}: {r}")

    all_pass = all(r == "PASS" for r in results.values())
    log(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
