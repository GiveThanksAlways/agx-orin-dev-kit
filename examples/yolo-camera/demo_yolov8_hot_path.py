#!/usr/bin/env python3
"""YOLOv8 live camera demo — TinyGrad C Hot Path on Jetson AGX Orin.

Fastest tinygrad variant: after @TinyJit + BEAM kernel auto-tuning, the
GPU command queue is exported and replayed from C via raw MMIO doorbell
writes. Zero Python overhead in the inference hot loop.

Benchmark results (YOLOv8-n @ 320x320, Orin AGX 64GB MAXN):
  - With BEAM search:    9.5 ms / 106 FPS
  - Without BEAM search: 26 ms  /  38 FPS

Usage:
  cd ~/agx-orin-dev-kit/examples/yolo-camera
  nix develop
  make                                                 # build hot_path.so
  PARALLEL=0 JITBEAM=2 NV=1 python3 demo_yolov8_hot_path.py --stream

First run with JITBEAM=2 takes ~90 min (BEAM kernel search). Results are
permanently cached in ~/.cache/tinygrad/cache.db — subsequent runs warm
up in ~25 seconds.

SSH tunnel (view in browser on laptop):
  ssh -L 9999:localhost:8090 Orin-AGX-NixOS
  open http://localhost:9999/
"""
import argparse, sys, time, os, ctypes, threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np

# tinygrad imports
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import fetch, getenv
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad.engine.jit import TinyJit

# YOLOv8 building blocks
from examples.yolov8 import (
    YOLOv8,
    get_variant_multiples,
    get_weights_location,
    postprocess,
    scale_boxes,
)

# Hot path infrastructure (from control-loop)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOT_PATH_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "control-loop", "hot_path")


# ═══════════════════════════════════════════════════════════════════════════════
# Graph export: expression-based sym classification (robust for any model size)
# ═══════════════════════════════════════════════════════════════════════════════

from tinygrad.runtime.graph.hcq import HCQGraph

# Patch variable types (must match hot_path.h)
VAR_KICKOFF         = 0
VAR_TIMELINE_WAIT   = 1
VAR_TIMELINE_SIGNAL = 2


def _find_hcq_graph(jit_fn):
    """Find the first HCQGraph inside a TinyJit's captured jit cache."""
    captured = jit_fn.captured
    if captured is None:
        raise RuntimeError("TinyJit has no captured graph — run warmup first")
    for ei in captured._jit_cache:
        if isinstance(ei.prg, HCQGraph):
            return ei.prg
    raise RuntimeError("No HCQGraph found in jit cache")


def _find_all_hcq_graphs(jit_fn):
    """Find ALL HCQGraphs inside a TinyJit's captured jit cache."""
    captured = jit_fn.captured
    if captured is None:
        raise RuntimeError("TinyJit has no captured graph — run warmup first")
    graphs = [ei.prg for ei in captured._jit_cache if isinstance(ei.prg, HCQGraph)]
    if not graphs:
        raise RuntimeError("No HCQGraph found in jit cache")
    return graphs


def _classify_syms_by_expr(comp_queue, graph):
    """Classify syms by their UOp expression structure (not value matching).

    Each 64-bit value (timeline, kickoff) is split into two uint32 syms:
      - LOW:  expr & 0xFFFFFFFF  (the actual value to patch)
      - HIGH: expr >> 32         (always 0 for values < 2^32 — treat as const)

    Only LOW syms are patchable. HIGH syms are always 0 and never change.
    """
    kickoff_expr = graph.kickoff_var.arg[0]  # e.g. 'kickoff_var'
    tl_var_exprs = set()
    for _, var in graph.virt_timeline_vals.items():
        tl_var_exprs.add(var.arg[0])  # e.g. 'timeline_var_NV'

    classifications = []
    for sym in comp_queue.syms:
        s = str(sym)

        # High-bits syms (>> 32) are always 0 for our values → const
        if 'Ops.SHR' in s and 'arg=32' in s:
            classifications.append('const')
            continue

        if kickoff_expr in s:
            classifications.append('kickoff')
        elif any(tv in s for tv in tl_var_exprs):
            # timeline_var + 1 → SIGNAL; plain timeline_var → WAIT
            if 'Ops.ADD' in s and 'arg=1,' in s:
                classifications.append('tl_signal')
            else:
                classifications.append('tl_wait')
        elif 'timeline_sig_' in s:
            classifications.append('const')  # signal ADDRESS, not value
        else:
            classifications.append('const')

    n_ko = classifications.count('kickoff')
    n_tw = classifications.count('tl_wait')
    n_ts = classifications.count('tl_signal')
    print(f"  Sym classification: {len(comp_queue.syms)} syms — "
          f"{n_ko} kickoff, {n_tw} tl_wait, {n_ts} tl_signal, "
          f"{len(comp_queue.syms) - n_ko - n_tw - n_ts} const")

    if n_ko != 1 or n_tw != 1 or n_ts != 1:
        print(f"  WARNING: expected exactly 1 of each patchable type")

    return classifications


def export_hot_path_config(jit_fn, dev, x_buf, out_buf):
    """Export a hot_path_config_t-compatible dict from a warmed-up TinyJit."""
    graph = _find_hcq_graph(jit_fn)

    assert len(graph.devices) == 1, f"Expected 1 device, got {len(graph.devices)}"
    gdev = graph.devices[0]
    assert gdev is dev

    comp_queue = graph.comp_queues[dev]
    sym_classes = _classify_syms_by_expr(comp_queue, graph)

    # Build patch list from q_sints
    patches = []
    hw_base = comp_queue.hw_page.cpu_view().addr

    for off, sym_idx in comp_queue.q_sints:
        cat = sym_classes[sym_idx]
        if cat == 'const':
            continue
        addr = hw_base + off * 4
        if cat == 'kickoff':
            patches.append({'addr': addr, 'var_type': VAR_KICKOFF, 'mask': 0})
        elif cat == 'tl_wait':
            patches.append({'addr': addr, 'var_type': VAR_TIMELINE_WAIT, 'mask': 0})
        elif cat == 'tl_signal':
            patches.append({'addr': addr, 'var_type': VAR_TIMELINE_SIGNAL, 'mask': 0})

    # Build patch list from mv_sints
    import struct
    for mv, off, sym_idx, mask in comp_queue.mv_sints:
        cat = sym_classes[sym_idx]
        if cat == 'const':
            continue
        elem_size = struct.calcsize(mv.fmt)
        addr = mv.addr + off * elem_size
        mask_val = mask if mask is not None else 0
        if cat == 'kickoff':
            patches.append({'addr': addr, 'var_type': VAR_KICKOFF, 'mask': mask_val})
        elif cat == 'tl_wait':
            patches.append({'addr': addr, 'var_type': VAR_TIMELINE_WAIT, 'mask': mask_val})
        elif cat == 'tl_signal':
            patches.append({'addr': addr, 'var_type': VAR_TIMELINE_SIGNAL, 'mask': mask_val})

    # GPFifo info
    gpfifo = dev.compute_gpfifo

    # Queue signals to reset
    queue_sig_addrs = []
    for sig in graph.queue_signals_to_reset:
        queue_sig_addrs.append(sig.base_buf.cpu_view().addr)

    # Input/output buffer CPU addresses
    x_hcq = x_buf._buf
    o_hcq = out_buf._buf

    config = {
        'input_buf_addr':       x_hcq.cpu_view().addr,
        'output_buf_addr':      o_hcq.cpu_view().addr,
        'input_size':           x_buf.nbytes,
        'output_size':          out_buf.nbytes,
        'gpfifo_ring_addr':     gpfifo.ring.addr,
        'gpfifo_gpput_addr':    gpfifo.gpput.addr,
        'gpfifo_entries_count': gpfifo.entries_count,
        'gpfifo_token':         gpfifo.token,
        'gpfifo_put_value':     gpfifo.put_value,
        'cmdq_gpu_addr':        comp_queue.hw_page.va_addr,
        'cmdq_len_u32':         len(comp_queue._q),
        'gpu_mmio_addr':        dev.gpu_mmio.addr,
        'timeline_signal_addr': dev.timeline_signal.base_buf.cpu_view().addr,
        'timeline_value':       dev.timeline_value,
        'last_tl_value':        dev.timeline_value - 1,
        'kick_signal_addr':     graph.signals['KICK'].base_buf.cpu_view().addr,
        'kickoff_value':        graph.kickoff_value,
        'queue_signal_addrs':   queue_sig_addrs,
        'patches':              patches,
    }

    print(f"  Exported: {len(patches)} patches, "
          f"input={config['input_size']}B, output={config['output_size']}B, "
          f"cmdq={config['cmdq_len_u32']} words")
    return config


def export_hot_path_config_for_graph(graph, dev, x_buf, out_buf):
    """Export a hot_path_config_t-compatible dict from a specific HCQGraph."""
    assert len(graph.devices) == 1, f"Expected 1 device, got {len(graph.devices)}"
    assert graph.devices[0] is dev

    comp_queue = graph.comp_queues[dev]
    sym_classes = _classify_syms_by_expr(comp_queue, graph)

    patches = []
    hw_base = comp_queue.hw_page.cpu_view().addr

    for off, sym_idx in comp_queue.q_sints:
        cat = sym_classes[sym_idx]
        if cat == 'const':
            continue
        addr = hw_base + off * 4
        if cat == 'kickoff':
            patches.append({'addr': addr, 'var_type': VAR_KICKOFF, 'mask': 0})
        elif cat == 'tl_wait':
            patches.append({'addr': addr, 'var_type': VAR_TIMELINE_WAIT, 'mask': 0})
        elif cat == 'tl_signal':
            patches.append({'addr': addr, 'var_type': VAR_TIMELINE_SIGNAL, 'mask': 0})

    import struct
    for mv, off, sym_idx, mask in comp_queue.mv_sints:
        cat = sym_classes[sym_idx]
        if cat == 'const':
            continue
        elem_size = struct.calcsize(mv.fmt)
        addr = mv.addr + off * elem_size
        mask_val = mask if mask is not None else 0
        if cat == 'kickoff':
            patches.append({'addr': addr, 'var_type': VAR_KICKOFF, 'mask': mask_val})
        elif cat == 'tl_wait':
            patches.append({'addr': addr, 'var_type': VAR_TIMELINE_WAIT, 'mask': mask_val})
        elif cat == 'tl_signal':
            patches.append({'addr': addr, 'var_type': VAR_TIMELINE_SIGNAL, 'mask': mask_val})

    gpfifo = dev.compute_gpfifo
    queue_sig_addrs = [sig.base_buf.cpu_view().addr for sig in graph.queue_signals_to_reset]

    x_hcq = x_buf._buf
    o_hcq = out_buf._buf

    config = {
        'input_buf_addr':       x_hcq.cpu_view().addr,
        'output_buf_addr':      o_hcq.cpu_view().addr,
        'input_size':           x_buf.nbytes,
        'output_size':          out_buf.nbytes,
        'gpfifo_ring_addr':     gpfifo.ring.addr,
        'gpfifo_gpput_addr':    gpfifo.gpput.addr,
        'gpfifo_entries_count': gpfifo.entries_count,
        'gpfifo_token':         gpfifo.token,
        'gpfifo_put_value':     gpfifo.put_value,
        'cmdq_gpu_addr':        comp_queue.hw_page.va_addr,
        'cmdq_len_u32':         len(comp_queue._q),
        'gpu_mmio_addr':        dev.gpu_mmio.addr,
        'timeline_signal_addr': dev.timeline_signal.base_buf.cpu_view().addr,
        'timeline_value':       dev.timeline_value,
        'last_tl_value':        dev.timeline_value - 1,
        'kick_signal_addr':     graph.signals['KICK'].base_buf.cpu_view().addr,
        'kickoff_value':        graph.kickoff_value,
        'queue_signal_addrs':   queue_sig_addrs,
        'patches':              patches,
    }

    print(f"    {len(patches)} patches, cmdq={config['cmdq_len_u32']} words, "
          f"kick=0x{config['kick_signal_addr']:x}")
    return config


# ═══════════════════════════════════════════════════════════════════════════════
# CPU preprocessing (replaces tinygrad's GPU-based preprocess)
# ═══════════════════════════════════════════════════════════════════════════════

def cpu_preprocess(frame, imgsz):
    """Letterbox + normalize on CPU. Returns float32 NCHW array (1,3,H,W).

    Replicates tinygrad examples/yolov8.py compute_transform + preprocess:
      - Resize maintaining aspect ratio
      - Pad with (114,114,114) gray to square
      - BGR → RGB, HWC → CHW, divide by 255
    """
    h, w = frame.shape[:2]
    r = min(imgsz / h, imgsz / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))  # (nw, nh)
    if (w, h) != new_unpad:
        resized = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
    else:
        resized = frame

    dw = imgsz - new_unpad[0]
    dh = imgsz - new_unpad[1]
    top, bottom = int(round(dh / 2 - 0.1)), int(round(dh / 2 + 0.1))
    left, right = int(round(dw / 2 - 0.1)), int(round(dw / 2 + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    # BGR→RGB, HWC→CHW, normalize to [0,1]
    img = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(img[np.newaxis, ...])  # (1, 3, imgsz, imgsz)


# ═══════════════════════════════════════════════════════════════════════════════
# ctypes struct matching hot_path_config_t
# ═══════════════════════════════════════════════════════════════════════════════

MAX_PATCHES = 512
MAX_QUEUE_SIGNALS = 16

class PatchEntry(ctypes.Structure):
    _fields_ = [
        ("addr",     ctypes.c_uint64),
        ("var_type", ctypes.c_uint32),
        ("mask",     ctypes.c_uint32),
    ]

class HotPathConfig(ctypes.Structure):
    _fields_ = [
        ("input_buf_addr",       ctypes.c_uint64),
        ("output_buf_addr",      ctypes.c_uint64),
        ("input_size",           ctypes.c_uint32),
        ("output_size",          ctypes.c_uint32),

        ("gpfifo_ring_addr",     ctypes.c_uint64),
        ("gpfifo_gpput_addr",    ctypes.c_uint64),
        ("gpfifo_entries_count", ctypes.c_uint32),
        ("gpfifo_token",         ctypes.c_uint32),
        ("gpfifo_put_value",     ctypes.c_uint32),

        ("cmdq_gpu_addr",        ctypes.c_uint64),
        ("cmdq_len_u32",         ctypes.c_uint32),

        ("gpu_mmio_addr",        ctypes.c_uint64),

        ("timeline_signal_addr", ctypes.c_uint64),
        ("timeline_value",       ctypes.c_uint32),
        ("last_tl_value",        ctypes.c_uint32),

        ("kick_signal_addr",     ctypes.c_uint64),
        ("kickoff_value",        ctypes.c_uint32),

        ("num_queue_signals",    ctypes.c_uint32),
        ("queue_signal_addrs",   ctypes.c_uint64 * MAX_QUEUE_SIGNALS),

        ("num_patches",          ctypes.c_uint32),
        ("patches",              PatchEntry * MAX_PATCHES),

        ("gpfifo_entry",         ctypes.c_uint64),
    ]


def build_config_struct(cfg):
    """Convert export_graph dict → ctypes HotPathConfig."""
    c = HotPathConfig()
    c.input_buf_addr       = cfg['input_buf_addr']
    c.output_buf_addr      = cfg['output_buf_addr']
    c.input_size           = cfg['input_size']
    c.output_size          = cfg['output_size']
    c.gpfifo_ring_addr     = cfg['gpfifo_ring_addr']
    c.gpfifo_gpput_addr    = cfg['gpfifo_gpput_addr']
    c.gpfifo_entries_count = cfg['gpfifo_entries_count']
    c.gpfifo_token         = cfg['gpfifo_token']
    c.gpfifo_put_value     = cfg['gpfifo_put_value']
    c.cmdq_gpu_addr        = cfg['cmdq_gpu_addr']
    c.cmdq_len_u32         = cfg['cmdq_len_u32']
    c.gpu_mmio_addr        = cfg['gpu_mmio_addr']
    c.timeline_signal_addr = cfg['timeline_signal_addr']
    c.timeline_value       = cfg['timeline_value']
    c.last_tl_value        = cfg['last_tl_value']
    c.kick_signal_addr     = cfg['kick_signal_addr']
    c.kickoff_value        = cfg['kickoff_value']

    sigs = cfg['queue_signal_addrs']
    c.num_queue_signals = len(sigs)
    for i, addr in enumerate(sigs):
        c.queue_signal_addrs[i] = addr

    patches = cfg['patches']
    c.num_patches = len(patches)
    for i, p in enumerate(patches):
        c.patches[i].addr     = p['addr']
        c.patches[i].var_type = p['var_type']
        c.patches[i].mask     = p['mask']

    return c


# ═══════════════════════════════════════════════════════════════════════════════
# MJPEG HTTP streaming server
# ═══════════════════════════════════════════════════════════════════════════════

_stream_frame = None
_stream_lock = threading.Lock()

class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with _stream_lock:
                    jpg = _stream_frame
                if jpg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            pass
    def log_message(self, fmt, *a): pass

def start_stream_server(port):
    server = HTTPServer(("", port), MJPEGHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server

def update_stream_frame(frame):
    global _stream_frame
    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    with _stream_lock:
        _stream_frame = jpg.tobytes()


COCO_LABELS = None
def get_labels():
    global COCO_LABELS
    if COCO_LABELS is None:
        COCO_LABELS = fetch(
            "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
        ).read_text().split("\n")
    return COCO_LABELS


def annotate(frame, preds, dt, mode="C-HOT-PATH"):
    """Draw bounding boxes and FPS on a frame."""
    labels = get_labels()
    for pred in preds:
        conf = float(pred[4])
        if conf == 0:
            continue
        x1, y1, x2, y2 = map(int, pred[:4])
        cls_id = int(pred[5])
        label = f"{labels[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    fps = 1.0 / max(dt, 1e-6)
    cv2.putText(frame, f"{mode} | FPS: {fps:.1f}  inf: {dt*1000:.0f}ms",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame


# ═══════════════════════════════════════════════════════════════════════════════
# Args
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 C Hot Path demo (TinyGrad NV=1)")
    p.add_argument("--device", type=str, default="/dev/video0", help="V4L2 camera device path or index")
    p.add_argument("--variant", type=str, default="n", choices=["n","s","m","l","x"])
    p.add_argument("--width", type=int, default=2560)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--stereo", type=str, default="left", choices=["left","right","none"])
    p.add_argument("--headless", action="store_true")
    p.add_argument("--frames", type=int, default=0, help="0=unlimited")
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--video", type=str, default=None, help="Save MP4")
    p.add_argument("--stream", action="store_true", help="MJPEG HTTP stream")
    p.add_argument("--stream-port", type=int, default=8090)
    p.add_argument("--size", type=int, default=320, help="YOLO input (320=fastest)")
    p.add_argument("--warmup", type=int, default=5, help="Warmup frames for JIT+BEAM")
    p.add_argument("--bench", action="store_true", help="Run quick benchmark (no camera)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Core: JIT warmup → export → C dispatch
# ═══════════════════════════════════════════════════════════════════════════════

def setup_hot_path(model, imgsz, warmup_count):
    """Load model, warmup JIT, export graph(s), return C dispatch function.

    The JIT may split the model into multiple HCQGraphs (batch size 32 by
    default).  We export each graph separately and replay them sequentially
    in C — H2D before first graph, D2H after last graph.

    Returns: (c_dispatch, out_shape, dev, cfg_structs, lib)
      c_dispatch(input_np) → (output_np, time_ns)
    """
    dev = Device["NV"]

    # Static input: preprocessed image tensor (1, 3, H, W) float32
    static_x = Tensor.zeros(1, 3, imgsz, imgsz).contiguous().realize()
    dev.synchronize()

    # Probe output shape with a test forward pass
    print("Probing model output shape...")
    test_out = model(static_x)
    out_shape = test_out.shape
    out_numel = 1
    for s in out_shape:
        out_numel *= s
    print(f"  Model output: {out_shape} ({out_numel} floats, {out_numel*4} bytes)")

    # Static output buffer
    static_o = Tensor.zeros(*out_shape).contiguous().realize()
    dev.synchronize()

    # JIT-wrapped inference with static buffers
    @TinyJit
    def jit_run():
        static_o.assign(model(static_x)).realize()

    # Get CPU addresses for input buffer (for memcpy during warmup)
    x_buf = static_x._buffer()
    x_hcq = x_buf._buf
    in_addr = x_hcq.cpu_view().addr
    in_nbytes = x_buf.nbytes

    o_buf = static_o._buffer()
    o_hcq = o_buf._buf
    out_addr = o_hcq.cpu_view().addr
    out_nbytes = o_buf.nbytes

    # Warmup: run enough times for JIT to capture graph + BEAM to optimize
    jitbeam = getenv("JITBEAM", 0)
    if jitbeam:
        print(f"JITBEAM={jitbeam} — warmup will include kernel auto-tuning")
    print(f"Warming up ({warmup_count} frames)...")

    # Use random data for warmup
    rng = np.random.RandomState(42)
    for i in range(warmup_count):
        t0 = time.perf_counter()
        dummy = rng.rand(1, 3, imgsz, imgsz).astype(np.float32)
        ctypes.memmove(in_addr, dummy.ctypes.data, in_nbytes)
        jit_run()
        dev.synchronize()
        dt = time.perf_counter() - t0
        print(f"  warmup [{i+1}/{warmup_count}] {dt*1000:.0f}ms")

    print("JIT warmup complete — exporting HCQ graph(s) for C hot path...")

    # Find ALL HCQGraphs in the jit cache
    all_graphs = _find_all_hcq_graphs(jit_run)
    n_graphs = len(all_graphs)
    total_kernels = sum(len(g.jit_cache) for g in all_graphs)
    print(f"  Found {n_graphs} HCQGraph(s) ({total_kernels} total kernels)")

    # Export config for each graph
    cfg_dicts = []
    for gi, graph in enumerate(all_graphs):
        print(f"  Exporting graph {gi} ({len(graph.jit_cache)} kernels)...")
        cfg_dict = export_hot_path_config_for_graph(graph, dev, x_buf, o_buf)
        cfg_dicts.append(cfg_dict)

    # Load C hot path library
    hot_path_so = os.path.join(SCRIPT_DIR, "hot_path.so")
    if not os.path.exists(hot_path_so):
        # Try the control-loop location
        hot_path_so = os.path.join(HOT_PATH_DIR, "hot_path.so")
    if not os.path.exists(hot_path_so):
        print(f"ERROR: hot_path.so not found!")
        print(f"  Build it: cd {SCRIPT_DIR} && make")
        sys.exit(1)

    lib = ctypes.CDLL(hot_path_so)
    lib.hot_path_init.argtypes = [ctypes.c_void_p]
    lib.hot_path_init.restype = None
    lib.hot_path_run_iteration.argtypes = [
        ctypes.c_void_p,  # config
        ctypes.c_void_p,  # input data
        ctypes.c_void_p,  # output data
    ]
    lib.hot_path_run_iteration.restype = ctypes.c_uint64
    lib.hot_path_submit_graph.argtypes = [ctypes.c_void_p]
    lib.hot_path_submit_graph.restype = ctypes.c_uint64

    # Build and init C config structs for each graph
    cfg_structs = []
    for cfg_dict in cfg_dicts:
        cfg_struct = build_config_struct(cfg_dict)
        lib.hot_path_init(ctypes.byref(cfg_struct))
        cfg_structs.append(cfg_struct)

    # Allocate output numpy buffer
    out_np = np.zeros(out_numel, dtype=np.float32)

    # CRITICAL: keep references to static tensors alive so GC doesn't free
    # the GPU buffers that cfg_struct points to
    _pinned_refs = (static_x, static_o, x_buf, o_buf, jit_run, all_graphs)

    if n_graphs == 1:
        # Single graph — original fast path
        def c_dispatch(input_np):
            _ = _pinned_refs
            ns = lib.hot_path_run_iteration(
                ctypes.byref(cfg_structs[0]),
                input_np.ctypes.data,
                out_np.ctypes.data,
            )
            return out_np.reshape(out_shape), ns
    else:
        # Multiple graphs — H2D, submit each graph (chaining state), D2H
        def c_dispatch(input_np):
            _ = _pinned_refs
            # H2D: copy input to GPU buffer
            ctypes.memmove(in_addr, input_np.ctypes.data, in_nbytes)
            total_ns = 0
            # Submit all graphs sequentially, chaining shared GPU state
            for i, cfg in enumerate(cfg_structs):
                ns = lib.hot_path_submit_graph(ctypes.byref(cfg))
                total_ns += ns
                # Propagate gpfifo/timeline state to next graph
                if i + 1 < len(cfg_structs):
                    nxt = cfg_structs[i + 1]
                    nxt.gpfifo_put_value = cfg.gpfifo_put_value
                    nxt.timeline_value = cfg.timeline_value
                    nxt.last_tl_value = cfg.last_tl_value
            # Propagate final state back to first graph for next call
            last = cfg_structs[-1]
            cfg_structs[0].gpfifo_put_value = last.gpfifo_put_value
            cfg_structs[0].timeline_value = last.timeline_value
            cfg_structs[0].last_tl_value = last.last_tl_value
            # D2H: copy output from GPU buffer
            ctypes.memmove(out_np.ctypes.data, out_addr, out_nbytes)
            return out_np.reshape(out_shape), total_ns

    # Run a few iterations through C to verify it works
    print("Verifying C hot path dispatch...")
    dummy = rng.rand(1, 3, imgsz, imgsz).astype(np.float32)
    dummy_flat = np.ascontiguousarray(dummy)
    for i in range(3):
        _, ns = c_dispatch(dummy_flat)
        print(f"  C iteration {i+1}: {ns/1000:.0f} µs")

    print("C hot path ready!\n")
    return c_dispatch, out_shape, dev, cfg_structs, lib


# ═══════════════════════════════════════════════════════════════════════════════
# Quick benchmark (no camera)
# ═══════════════════════════════════════════════════════════════════════════════

def run_bench(args):
    """Benchmark C hot path dispatch latency (no camera)."""
    print("=" * 60)
    print("  C Hot Path Benchmark (no camera)")
    print("=" * 60)

    depth, width, ratio = get_variant_multiples(args.variant)
    model = YOLOv8(w=width, r=ratio, d=depth, num_classes=80)
    state_dict = safe_load(get_weights_location(args.variant))
    load_state_dict(model, state_dict)

    c_dispatch, out_shape, dev, cfg_structs, lib = setup_hot_path(
        model, args.size, args.warmup
    )

    n_iters = 100
    rng = np.random.RandomState(99)
    times_us = []

    # Pre-allocate a single reusable input buffer
    input_buf = np.ascontiguousarray(np.zeros((1, 3, args.size, args.size), dtype=np.float32))

    print(f"\nBenchmarking {n_iters} iterations...")
    for i in range(n_iters):
        np.copyto(input_buf, rng.rand(1, 3, args.size, args.size).astype(np.float32))
        _, ns = c_dispatch(input_buf)
        times_us.append(ns / 1000.0)
        if (i + 1) % 25 == 0:
            med = np.median(times_us[-25:])
            print(f"  [{i+1}/{n_iters}] last-25 median: {med:.0f} µs ({1e6/med:.1f} FPS)")

    a = np.array(times_us)
    print(f"\n{'─'*60}")
    print(f"  YOLOv8-{args.variant} @ {args.size}x{args.size} — C Hot Path")
    print(f"  median:  {np.median(a):.0f} µs  ({1e6/np.median(a):.1f} FPS)")
    print(f"  mean:    {np.mean(a):.0f} µs")
    print(f"  P99:     {np.percentile(a, 99):.0f} µs")
    print(f"  P99.9:   {np.percentile(a, 99.9):.0f} µs")
    print(f"  min/max: {np.min(a):.0f} / {np.max(a):.0f} µs")
    print(f"{'─'*60}")

    # Sync Python device state after C dispatch
    last_cfg = cfg_structs[-1]
    dev.compute_gpfifo.put_value = last_cfg.gpfifo_put_value
    dev.timeline_value = last_cfg.timeline_value


# ═══════════════════════════════════════════════════════════════════════════════
# Camera loop with C hot path dispatch
# ═══════════════════════════════════════════════════════════════════════════════

def run_camera(args):
    """Live camera loop with C hot path inference."""
    print("Loading YOLOv8-%s..." % args.variant)
    depth, width, ratio = get_variant_multiples(args.variant)
    model = YOLOv8(w=width, r=ratio, d=depth, num_classes=80)
    state_dict = safe_load(get_weights_location(args.variant))
    load_state_dict(model, state_dict)
    print("Model loaded")

    # Setup C hot path (JIT warmup + export + load .so)
    c_dispatch, out_shape, dev, cfg_structs, lib = setup_hot_path(
        model, args.size, args.warmup
    )

    # Open camera — accept device path ("/dev/video0") or integer index
    dev_arg = int(args.device) if args.device.isdigit() else args.device
    cap = cv2.VideoCapture(dev_arg, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: cannot open {args.device}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera: {args.device}  {actual_w}x{actual_h} @ {actual_fps:.0f} FPS")
    print(f"YOLO input: {args.size}x{args.size}  variant: {args.variant}")
    print(f"Mode: C HOT PATH (MMIO doorbell dispatch)")

    outdir = Path(args.outdir)
    if args.headless:
        outdir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    if args.video:
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)

    stream_server = None
    if args.stream:
        stream_server = start_stream_server(args.stream_port)
        print(f"MJPEG stream: http://0.0.0.0:{args.stream_port}/")

    frame_count = 0
    fps_smooth = 0.0

    # Pre-allocate contiguous input buffer
    input_buf = np.zeros((1, 3, args.size, args.size), dtype=np.float32)

    try:
        while True:
            _ = cap.grab()  # drop buffered frame
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break

            # Crop stereo pair to single eye
            if args.stereo != "none":
                h, w = frame.shape[:2]
                half = w // 2
                frame = frame[:, :half] if args.stereo == "left" else frame[:, half:]

            # CPU preprocess → float32 NCHW
            t0 = time.perf_counter()
            pre_np = cpu_preprocess(frame, args.size)
            np.copyto(input_buf, pre_np)

            # C hot path inference
            preds_raw, gpu_ns = c_dispatch(input_buf)

            # Scale boxes to original frame coordinates
            img1_shape = (args.size, args.size)
            preds = scale_boxes(img1_shape, preds_raw.copy(), frame.shape)
            dt = time.perf_counter() - t0

            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(dt, 1e-6)) if fps_smooth > 0 else 1.0 / max(dt, 1e-6)

            out = annotate(frame.copy(), preds, dt, mode="C-HOT-PATH")
            frame_count += 1

            if args.video:
                if video_writer is None:
                    h, w = out.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(args.video, fourcc, min(fps_smooth, 30), (w, h))
                video_writer.write(out)

            if args.stream:
                update_stream_frame(out)

            if args.headless or args.video:
                if args.headless:
                    path = outdir / f"frame_{frame_count:05d}.jpg"
                    cv2.imwrite(str(path), out)
                print(f"\r[{frame_count}] {fps_smooth:.1f} FPS  {dt*1000:.0f}ms  gpu={gpu_ns/1000:.0f}µs", end="", flush=True)
            elif not args.stream:
                cv2.imshow("YOLOv8 — C Hot Path (q to quit)", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print(f"\r[{frame_count}] {fps_smooth:.1f} FPS  {dt*1000:.0f}ms  gpu={gpu_ns/1000:.0f}µs", end="", flush=True)

            if args.frames > 0 and frame_count >= args.frames:
                break

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if not args.headless and not args.stream and not args.video:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        # Sync Python device state after C dispatch
        last_cfg = cfg_structs[-1]
        dev.compute_gpfifo.put_value = last_cfg.gpfifo_put_value
        dev.timeline_value = last_cfg.timeline_value

    print(f"\nDone — {frame_count} frames, avg {fps_smooth:.1f} FPS")
    if args.video:
        sz = os.path.getsize(args.video) / (1024 * 1024)
        print(f"Video saved: {args.video} ({sz:.1f} MB)")


def main():
    args = parse_args()
    if args.bench:
        run_bench(args)
    else:
        run_camera(args)


if __name__ == "__main__":
    main()
