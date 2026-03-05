#!/usr/bin/env python3
"""Debug script: compare C hot path output vs direct model output."""
import sys, os, numpy as np, ctypes
sys.path.insert(0, os.path.realpath("../../external/tinygrad"))
sys.path.insert(0, ".")

from tinygrad import Tensor, Device
from tinygrad.nn.state import safe_load, load_state_dict
from examples.yolov8 import YOLOv8, get_variant_multiples, get_weights_location
from demo_yolov8_hot_path import setup_hot_path, cpu_preprocess
import cv2

d, w, r = get_variant_multiples("n")
model = YOLOv8(w=w, r=r, d=d, num_classes=80)
state_dict = safe_load(get_weights_location("n"))
load_state_dict(model, state_dict)
c_dispatch, out_shape, dev, cfg_struct, lib = setup_hot_path(model, 320, 5)

# Capture frame
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret, frame = cap.read()
cap.release()
frame = frame[:, :frame.shape[1]//2]
img = cpu_preprocess(frame, 320)
print(f"Input: shape={img.shape}, min={img.min():.4f}, max={img.max():.4f}, sum={img.sum():.2f}")

# Run C dispatch
out_c, ns = c_dispatch(img)

# Read back input buffer AFTER C dispatch
in_check = np.zeros_like(img)
ctypes.memmove(in_check.ctypes.data, cfg_struct.input_buf_addr, cfg_struct.input_size)
print(f"Input buffer after C dispatch: min={in_check.min():.4f}, max={in_check.max():.4f}")
print(f"Input matches original: {np.allclose(in_check, img)}, max diff: {np.abs(in_check - img).max():.8f}")

# Direct model (not JIT) — this creates new tensors, doesn't interfere with JIT
dev.synchronize()
out_direct = model(Tensor(img)).numpy()
print(f"Direct model: max conf={out_direct[:,4].max():.4f}, non-zero={(out_direct[:,4]>0).sum()}")
print(f"C hot path:   max conf={out_c[:,4].max():.4f}, non-zero={(out_c[:,4]>0).sum()}")

# Show direct model detections
dets = out_direct[out_direct[:,4] > 0]
print(f"Direct model detections ({len(dets)}):")
for i in range(min(5, len(dets))):
    print(f"  [{dets[i,0]:.2f}, {dets[i,1]:.2f}, {dets[i,2]:.2f}, {dets[i,3]:.2f}, {dets[i,4]:.4f}, {dets[i,5]:.0f}]")

# C output first 10 rows
print(f"C hot path first 10 rows:")
for i in range(10):
    r = out_c[i]
    print(f"  [{r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f}, {r[3]:.4f}, {r[4]:.6f}, {r[5]:.0f}]")

# Also check: how many C output rows have ANY non-zero value?
nonzero_rows = np.any(out_c != 0, axis=1).sum()
print(f"C output rows with any non-zero value: {nonzero_rows}/{out_c.shape[0]}")
print(f"C output overall: min={out_c.min():.6f}, max={out_c.max():.6f}")
