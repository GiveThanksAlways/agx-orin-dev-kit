#!/usr/bin/env python3
"""Debug: compare C hot path output vs Python JIT output."""
import sys, os, ctypes, time
import numpy as np
sys.path.insert(0, os.path.realpath("../../external/tinygrad"))
os.environ.setdefault("NV", "1")

from tinygrad import Tensor, Device
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad.engine.jit import TinyJit
from examples.yolov8 import YOLOv8, get_variant_multiples, get_weights_location

# Load model
d, w, r = get_variant_multiples("n")
model = YOLOv8(w=w, r=r, d=d, num_classes=80)
state_dict = safe_load(get_weights_location("n"))
load_state_dict(model, state_dict)

dev = Device["NV"]
imgsz = 320

# Create a deterministic test input (looks like a real image)
rng = np.random.RandomState(12345)
test_input = rng.rand(1, 3, imgsz, imgsz).astype(np.float32)

# ─── Test 1: Direct model call (no JIT) ───
print("=" * 60)
print("Test 1: Direct model call (no JIT)")
print("=" * 60)
x = Tensor(test_input).realize()
out_direct = model(x).numpy()
print(f"  Output shape: {out_direct.shape}")
print(f"  Box range: {out_direct[:, :4].min():.4f} to {out_direct[:, :4].max():.4f}")
print(f"  Conf range: {out_direct[:, 4].min():.4f} to {out_direct[:, 4].max():.4f}")
n_det = (out_direct[:, 4] > 0.0).sum()
print(f"  Detections with conf > 0: {n_det}")
n_det25 = (out_direct[:, 4] > 0.25).sum()
print(f"  Detections with conf > 0.25: {n_det25}")
if n_det25 > 0:
    idx = out_direct[:, 4] > 0.25
    print(f"  Top confs: {out_direct[idx, 4][:5]}")

# ─── Test 2: JIT with static buffers (same pattern as hot path) ───
print("\n" + "=" * 60)
print("Test 2: JIT with static input/output buffers")
print("=" * 60)

static_x = Tensor.zeros(1, 3, imgsz, imgsz).contiguous().realize()
dev.synchronize()

# Probe output shape
test_out = model(static_x)
out_shape = test_out.shape
out_numel = 1
for s in out_shape:
    out_numel *= s

static_o = Tensor.zeros(*out_shape).contiguous().realize()
dev.synchronize()

@TinyJit
def jit_run():
    static_o.assign(model(static_x)).realize()

# Get buffer addresses
x_buf = static_x._buffer()
x_addr = x_buf._buf.cpu_view().addr
x_nbytes = x_buf.nbytes

o_buf = static_o._buffer()
o_addr = o_buf._buf.cpu_view().addr
o_nbytes = o_buf.nbytes

print(f"  Input buf:  addr=0x{x_addr:x}, size={x_nbytes}")
print(f"  Output buf: addr=0x{o_addr:x}, size={o_nbytes}")

# Warmup JIT (3 passes: interpret, capture, replay)
for i in range(3):
    dummy = rng.rand(1, 3, imgsz, imgsz).astype(np.float32)
    ctypes.memmove(x_addr, dummy.ctypes.data, x_nbytes)
    jit_run()
    dev.synchronize()

# Now run with our test input
ctypes.memmove(x_addr, test_input.ctypes.data, x_nbytes)
jit_run()
dev.synchronize()

# Read output via ctypes memcpy (same way C hot path does)
out_jit_c = np.zeros(out_numel, dtype=np.float32)
ctypes.memmove(out_jit_c.ctypes.data, o_addr, o_nbytes)
out_jit_c = out_jit_c.reshape(out_shape)

# Also read via tensor .numpy()
out_jit_np = static_o.numpy()

print(f"  Output shape: {out_jit_c.shape}")
print(f"  [ctypes] Box range: {out_jit_c[:, :4].min():.4f} to {out_jit_c[:, :4].max():.4f}")
print(f"  [ctypes] Conf range: {out_jit_c[:, 4].min():.4f} to {out_jit_c[:, 4].max():.4f}")
print(f"  [numpy]  Conf range: {out_jit_np[:, 4].min():.4f} to {out_jit_np[:, 4].max():.4f}")
n_det = (out_jit_c[:, 4] > 0.0).sum()
print(f"  Detections with conf > 0: {n_det}")
n_det25 = (out_jit_c[:, 4] > 0.25).sum()
print(f"  Detections with conf > 0.25: {n_det25}")

# Check if direct and JIT match
print(f"\n  Direct vs JIT max diff: {np.abs(out_direct - out_jit_c).max():.6f}")

# ─── Test 3: C Hot Path dispatch ───
print("\n" + "=" * 60)
print("Test 3: C Hot Path dispatch")
print("=" * 60)

# Import the export machinery
from demo_yolov8_hot_path import (
    export_hot_path_config, build_config_struct, HotPathConfig
)
from tinygrad.runtime.graph.hcq import HCQGraph

cfg_dict = export_hot_path_config(jit_run, dev, x_buf, o_buf)

# Load C library
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
hot_path_so = os.path.join(SCRIPT_DIR, "hot_path.so")
if not os.path.exists(hot_path_so):
    hot_path_so = os.path.join(os.path.dirname(SCRIPT_DIR), "control-loop", "hot_path", "hot_path.so")

lib = ctypes.CDLL(hot_path_so)
lib.hot_path_init.argtypes = [ctypes.c_void_p]
lib.hot_path_init.restype = None
lib.hot_path_run_iteration.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
lib.hot_path_run_iteration.restype = ctypes.c_uint64

cfg_struct = build_config_struct(cfg_dict)
lib.hot_path_init(ctypes.byref(cfg_struct))

# Run C dispatch with our test input
out_c = np.zeros(out_numel, dtype=np.float32)
input_flat = np.ascontiguousarray(test_input)
ns = lib.hot_path_run_iteration(
    ctypes.byref(cfg_struct),
    input_flat.ctypes.data,
    out_c.ctypes.data,
)
out_c = out_c.reshape(out_shape)

print(f"  C dispatch time: {ns/1000:.0f} µs")
print(f"  Output shape: {out_c.shape}")
print(f"  Box range: {out_c[:, :4].min():.4f} to {out_c[:, :4].max():.4f}")
print(f"  Conf range: {out_c[:, 4].min():.4f} to {out_c[:, 4].max():.4f}")
n_det = (out_c[:, 4] > 0.0).sum()
print(f"  Detections with conf > 0: {n_det}")
n_det25 = (out_c[:, 4] > 0.25).sum()
print(f"  Detections with conf > 0.25: {n_det25}")

# Also check the raw GPU buffer after C dispatch
out_gpu_raw = np.zeros(out_numel, dtype=np.float32)
ctypes.memmove(out_gpu_raw.ctypes.data, o_addr, o_nbytes)
out_gpu_raw = out_gpu_raw.reshape(out_shape)
print(f"\n  [raw GPU buf] Box range: {out_gpu_raw[:, :4].min():.4f} to {out_gpu_raw[:, :4].max():.4f}")
print(f"  [raw GPU buf] Conf range: {out_gpu_raw[:, 4].min():.4f} to {out_gpu_raw[:, 4].max():.4f}")

print(f"\n  JIT vs C hot path max diff: {np.abs(out_jit_c - out_c).max():.6f}")
print(f"  JIT vs C GPU buf max diff:  {np.abs(out_jit_c - out_gpu_raw).max():.6f}")

# Run C dispatch a second time to make sure it's stable
ns2 = lib.hot_path_run_iteration(
    ctypes.byref(cfg_struct),
    input_flat.ctypes.data,
    out_c.ctypes.data,
)
out_c2 = out_c.reshape(out_shape)
print(f"\n  2nd C dispatch: {ns2/1000:.0f} µs")
print(f"  2nd C conf range: {out_c2[:, 4].min():.4f} to {out_c2[:, 4].max():.4f}")
print(f"  1st vs 2nd max diff: {np.abs(out_c - out_c2).max():.6f}")

# Sync state back
dev.compute_gpfifo.put_value = cfg_struct.gpfifo_put_value
dev.timeline_value = cfg_struct.timeline_value

print("\nDONE")
