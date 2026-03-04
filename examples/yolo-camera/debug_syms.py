#!/usr/bin/env python3
"""Debug script to understand HCQGraph sym structure for YOLOv8."""
import os, sys, ctypes
os.environ.setdefault("NV", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "control-loop", "hot_path"))

from tinygrad import Tensor, dtypes, Device
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad.engine.jit import TinyJit
from tinygrad.runtime.graph.hcq import HCQGraph
from examples.yolov8 import YOLOv8, get_variant_multiples, get_weights_location
import numpy as np

depth, width, ratio = get_variant_multiples("n")
model = YOLOv8(w=width, r=ratio, d=depth, num_classes=80)
state_dict = safe_load(get_weights_location("n"))
load_state_dict(model, state_dict)

dev = Device["NV"]
imgsz = 320
static_x = Tensor.zeros(1, 3, imgsz, imgsz).contiguous().realize()
test_out = model(static_x)
out_shape = test_out.shape
print(f"Output shape: {out_shape}")
static_o = Tensor.zeros(*out_shape).contiguous().realize()
dev.synchronize()

@TinyJit
def jit_run():
    static_o.assign(model(static_x)).realize()

rng = np.random.RandomState(42)
x_buf = static_x._buffer()
in_addr = x_buf._buf.cpu_view().addr
in_nbytes = x_buf.nbytes

for i in range(5):
    dummy = rng.rand(1, 3, imgsz, imgsz).astype(np.float32)
    ctypes.memmove(in_addr, dummy.ctypes.data, in_nbytes)
    jit_run()
    dev.synchronize()
    print(f"  warmup [{i+1}/5] done")

# Find HCQGraph
captured = jit_run.captured
graph = None
for ei in captured._jit_cache:
    if isinstance(ei.prg, HCQGraph):
        graph = ei.prg
        break

if graph is None:
    print("ERROR: No HCQGraph found")
    sys.exit(1)

comp_queue = graph.comp_queues[dev]
prev = comp_queue._prev_resolved_syms
ko = graph.kickoff_value
tl = dev.timeline_value

print(f"\n=== SYM DEBUG ===")
print(f"kickoff_value={ko}, timeline_value={tl}")
print(f"tl_wait_ref={tl-2}, tl_signal_ref={tl-1}")
print(f"Num syms: {len(prev)}")
for i, val in enumerate(prev):
    if val is None:
        print(f"  sym[{i}] = None")
    else:
        match = ""
        if val == ko: match += " [KICKOFF]"
        if val == tl-2: match += " [TL_WAIT]"
        if val == tl-1: match += " [TL_SIGNAL]"
        print(f"  sym[{i}] = {val} (0x{val:x}){match}")

print(f"\nvirt_timeline_vals keys: {list(graph.virt_timeline_vals.keys())}")
print(f"kickoff_var: {graph.kickoff_var}")

# Print actual sym UOp expressions
print(f"\nSym UOp expressions:")
for i, sym in enumerate(comp_queue.syms):
    print(f"  sym[{i}]: {sym}")

# Also check q_sints and mv_sints
print(f"\nq_sints count: {len(comp_queue.q_sints)}")
print(f"mv_sints count: {len(comp_queue.mv_sints)}")

for off, sym_idx in comp_queue.q_sints[:20]:
    val = prev[sym_idx]
    print(f"  q_sint: off={off}, sym_idx={sym_idx}, resolved={val}")

# Check virt_timeline_vals variable expr
for d, var in graph.virt_timeline_vals.items():
    print(f"\nTimeline var for {d}: {var}")
    print(f"  var.expr = {var.expr}")
    print(f"  var.arg = {var.arg}")

# Check kickoff_var
print(f"\nkickoff_var.expr = {graph.kickoff_var.expr}")
print(f"kickoff_var.arg = {graph.kickoff_var.arg}")

# Try to find which syms contain timeline variable
tl_var_expr = None
for d, var in graph.virt_timeline_vals.items():
    tl_var_expr = var.expr
    break

print(f"\nLooking for timeline variable expr={tl_var_expr} in syms...")
for i, sym in enumerate(comp_queue.syms):
    s = str(sym)
    if tl_var_expr and tl_var_expr in s:
        print(f"  sym[{i}] CONTAINS timeline var: {s}")
    if "kickoff" in s.lower():
        print(f"  sym[{i}] CONTAINS kickoff: {s}")

print("\nDone.")
