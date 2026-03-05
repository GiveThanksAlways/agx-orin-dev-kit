#!/usr/bin/env python3
"""Test C dispatch vs Python JIT on real camera frame."""
import sys, numpy as np, ctypes, os
sys.path.insert(0, ".")
import demo_yolov8_hot_path as demo
from tinygrad import Tensor, Device
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad.engine.jit import TinyJit
from tinygrad.runtime.graph.hcq import HCQGraph
from examples.yolov8 import YOLOv8, get_variant_multiples, get_weights_location
import cv2

d, w, r = get_variant_multiples("n")
model = YOLOv8(w=w, r=r, d=d, num_classes=80)
state_dict = safe_load(get_weights_location("n"))
load_state_dict(model, state_dict)

dev = Device["NV"]
imgsz = 320

# Camera frame
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)
for _ in range(3): cap.grab()
ret, frame = cap.read()
cap.release()
assert ret
frame = frame[:, :frame.shape[1]//2]
pre = demo.cpu_preprocess(frame, imgsz)
pre_flat = np.ascontiguousarray(pre)

# Setup static buffers + JIT
static_x = Tensor.zeros(1, 3, imgsz, imgsz).contiguous().realize()
dev.synchronize()
test_out = model(static_x)
out_shape = test_out.shape
numel = 1
for s in out_shape: numel *= s
static_o = Tensor.zeros(*out_shape).contiguous().realize()
dev.synchronize()

@TinyJit
def jit_run():
    static_o.assign(model(static_x)).realize()

x_buf = static_x._buffer()
x_hcq = x_buf._buf
in_addr = x_hcq.cpu_view().addr
in_nbytes = x_buf.nbytes
o_buf = static_o._buffer()
o_hcq = o_buf._buf
out_addr = o_hcq.cpu_view().addr
out_nbytes = o_buf.nbytes

# Warmup
rng = np.random.RandomState(42)
for i in range(5):
    dummy = rng.rand(1, 3, imgsz, imgsz).astype(np.float32)
    ctypes.memmove(in_addr, dummy.ctypes.data, in_nbytes)
    jit_run()
    dev.synchronize()

# STEP 1: Verify Python JIT works
ctypes.memmove(in_addr, pre_flat.ctypes.data, in_nbytes)
jit_run()
dev.synchronize()
python_out = np.zeros(numel, dtype=np.float32)
ctypes.memmove(python_out.ctypes.data, out_addr, out_nbytes)
python_out = python_out.reshape(out_shape)
print(f"Python JIT: max_conf={python_out[:,4].max():.4f}, dets={(python_out[:,4]>0.25).sum()}", flush=True)

# STEP 2: C dispatch
print(f"\ndev.timeline_value = {dev.timeline_value}", flush=True)

all_graphs = demo._find_all_hcq_graphs(jit_run)
print(f"Found {len(all_graphs)} graphs", flush=True)

cfg_dicts = []
for gi, g in enumerate(all_graphs):
    cfg = demo.export_hot_path_config_for_graph(g, dev, x_buf, o_buf)
    cfg_dicts.append(cfg)
    print(f"  graph {gi}: tl_val={cfg['timeline_value']}, last_tl={cfg['last_tl_value']}, kick_val={cfg['kickoff_value']}, put={cfg['gpfifo_put_value']}", flush=True)

hot_path_so = os.path.join(demo.HOT_PATH_DIR, "hot_path.so")
lib = ctypes.CDLL(hot_path_so)
lib.hot_path_init.argtypes = [ctypes.c_void_p]
lib.hot_path_init.restype = None
lib.hot_path_submit_graph.argtypes = [ctypes.c_void_p]
lib.hot_path_submit_graph.restype = ctypes.c_uint64

cfg_structs = []
for cfg in cfg_dicts:
    cs = demo.build_config_struct(cfg)
    lib.hot_path_init(ctypes.byref(cs))
    cfg_structs.append(cs)

# Zero output, copy input
zeros = np.zeros(numel, dtype=np.float32)
ctypes.memmove(out_addr, zeros.ctypes.data, out_nbytes)
ctypes.memmove(in_addr, pre_flat.ctypes.data, in_nbytes)

# C dispatch
for i, cs in enumerate(cfg_structs):
    ns = lib.hot_path_submit_graph(ctypes.byref(cs))
    print(f"  C graph {i}: {ns/1000:.0f}us, tl={cs.timeline_value}, last_tl={cs.last_tl_value}, kick={cs.kickoff_value}, put={cs.gpfifo_put_value}", flush=True)
    if i + 1 < len(cfg_structs):
        cfg_structs[i+1].gpfifo_put_value = cs.gpfifo_put_value
        cfg_structs[i+1].timeline_value = cs.timeline_value
        cfg_structs[i+1].last_tl_value = cs.last_tl_value

c_out = np.zeros(numel, dtype=np.float32)
ctypes.memmove(c_out.ctypes.data, out_addr, out_nbytes)
c_out = c_out.reshape(out_shape)
print(f"\nC dispatch: max_conf={c_out[:,4].max():.4f}, dets={(c_out[:,4]>0.25).sum()}", flush=True)
print(f"C first 10: {c_out.flatten()[:10]}", flush=True)
print(f"Diff from Python: {np.abs(python_out - c_out).sum():.6f}", flush=True)

# Sync back
last = cfg_structs[-1]
dev.compute_gpfifo.put_value = last.gpfifo_put_value
dev.timeline_value = last.timeline_value
print("DONE", flush=True)
