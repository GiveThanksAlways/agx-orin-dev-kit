#!/usr/bin/env python3
"""
TinyGrad MLP — one linear flow, top to bottom
──────────────────────────────────────────────
Model: mlp_5k  (12→64→64→4, FP16, batch=1)
Pipeline: Tensor ops → UOp DAG → Scheduler → Codegen → PTX → cubin
          → NVProgram (QMD) → HCQGraph → GPFIFO doorbell → GPU

Run command (inside nix develop):
    NV=1 VIZ=1 JITBEAM=2 DEBUG=2 python3 demo_mlp_flow.py

SSH tunnel (open on your PC first so VIZ works):
    ssh -L 3000:localhost:3000 agent@192.168.8.162

Breakpoints to set in tinygrad source before running:
  BP 1  engine/realize.py     line 211  (ei.run — kernel about to fire)
  BP 2  runtime/ops_nv.py     line 122  (dev.gpu_mmio — hardware doorbell)
  Then VIZ opens at exit → UOp rewrite browser + profiler tab
"""

import os, sys, ctypes, time, statistics
assert os.environ.get("NV") == "1", "Run with NV=1  e.g.  NV=1 VIZ=1 JITBEAM=2 python3 demo_mlp_flow.py"
import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad import nn as tg_nn

# ── model: mlp_5k (12→64→64→4, FP16) ─────────────────────────────────────────
# Same input/output dims as real drone controllers: 12 state vars → 4 motor cmds
IN_DIM, OUT_DIM = 12, 4
DIMS = [IN_DIM, 64, 64, OUT_DIM]

rng = np.random.RandomState(42)
layers: list[tg_nn.Linear] = []
for fi, fo in zip(DIMS, DIMS[1:]):
    lin = tg_nn.Linear(fi, fo)
    lin.weight = Tensor((rng.randn(fo, fi) * (2.0 / fi) ** 0.5).astype(np.float16))
    lin.bias   = Tensor(np.zeros(fo, dtype=np.float16))
    layers.append(lin)

def forward(x: Tensor) -> Tensor:
    for i, layer in enumerate(layers):
        x = layer(x)
        if i < len(layers) - 1:
            x = x.relu()
    return x

total_params = sum(l.weight.shape[0] * l.weight.shape[1] + l.weight.shape[0] for l in layers)
print(f"\nmlp_5k  {DIMS}  FP16  ~{total_params:,} params  device={Device.DEFAULT}")
print(f"env: DEBUG={os.environ.get('DEBUG','0')} VIZ={os.environ.get('VIZ','0')} "
      f"JITBEAM={os.environ.get('JITBEAM','0')}\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Build the lazy graph — pure Python, GPU untouched
# ─────────────────────────────────────────────────────────────────────────────
x   = Tensor.zeros(1, IN_DIM,  dtype=dtypes.float16).contiguous().realize()
out = Tensor.zeros(1, OUT_DIM, dtype=dtypes.float16).contiguous().realize()

lazy_out = forward(x)
# ↑ At this point, lazy_out is a UOp DAG (expression tree) in Python memory.
#   No GPU work. Inspect lazy_out.uop in the debugger here.
print(f"[lazy]   out.uop.op = {lazy_out.uop.op}  — UOp tree built, GPU idle")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: JIT — captures the full kernel graph on first real run
#   cnt=0 → dry run (no capture)
#   cnt=1 → JIT capture: schedule → codegen → compile → HCQGraph assembled
#   cnt≥2 → steady-state: HCQGraph.__call__ patches ptrs + doorbell poke only
#
# Breakpoints fire during cnt=1:
#   BP1 (realize.py:211) — ei.run() for each of the 3 fused kernels
#   BP2 (ops_nv.py:122)  — dev.gpu_mmio write = hardware doorbell
# ─────────────────────────────────────────────────────────────────────────────
@TinyJit
def run():
    out.assign(forward(x)).realize()

WARMUP, BENCH = 5, 300
in_addr  = x._buffer()._buf.cpu_view().addr   # Tegra unified mem — zero copy
in_bytes = x.dtype.itemsize * IN_DIM
data = np.random.RandomState(99).randn(WARMUP + BENCH + 4, 1, IN_DIM).astype(np.float16)

for i in range(WARMUP):
    ctypes.memmove(in_addr, data[i].ctypes.data, in_bytes)
    t0 = time.perf_counter()
    run()
    Device["NV"].synchronize()
    label = {0: "dry-run (no capture)", 1: "JIT capture"}.get(i, "graph exec")
    print(f"[warmup {i}] {label}  {(time.perf_counter()-t0)*1e3:.1f} ms")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Steady-state benchmark — pure HCQGraph replay
# ─────────────────────────────────────────────────────────────────────────────
times_us: list[float] = []
for i in range(BENCH):
    ctypes.memmove(in_addr, data[WARMUP + i].ctypes.data, in_bytes)
    t0 = time.perf_counter_ns()
    run()
    Device["NV"].synchronize()
    times_us.append((time.perf_counter_ns() - t0) / 1e3)

med = statistics.median(times_us)
mn  = min(times_us)
p99 = sorted(times_us)[int(0.99 * BENCH)]
print(f"\n[bench]  median {med:.1f} µs  ({1e6/med:,.0f} Hz)   min {mn:.1f} µs   p99 {p99:.1f} µs")
print(f"[jit]    {len(run.captured._jit_cache) if run.captured else '?'} kernel(s) in HCQGraph\n")
print("Process exiting → VIZ/profiler server will start on port 3000")
