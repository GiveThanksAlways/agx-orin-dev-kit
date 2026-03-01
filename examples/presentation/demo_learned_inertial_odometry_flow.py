#!/usr/bin/env python3
"""
TinyGrad Pipeline Demo — Cioffi TCN (Learned Inertial Odometry)
───────────────────────────────────────────────────────────────
Paper:  Cioffi, Bauersfeld, Kaufmann, Scaramuzza.
        "Learned Inertial Odometry for Autonomous Drone Racing."
        IEEE RA-L 2023, arXiv:2210.15287
Model:  TCN ~250K params, input=(1,6,50), output=(1,3)
        7 TemporalBlocks (causal dilated Conv1d → GELU → residual → ReLU)
        Final: Linear(128,3) on last timestep → Δp displacement (x,y,z)
Input:  (1, 6, 50) — 3 gyro + 3 thrust channels, 50 timesteps (0.5s @ 100 Hz)
Output: (1, 3) — predicted displacement Δp

══════════════════════════════════════════════════════════════════════════════
 TinyGrad NV pipeline — how Python tensor code becomes GPU hardware commands
══════════════════════════════════════════════════════════════════════════════

   Python: model(x)                    ← you write this
     │
     ▼
   Tensor ops build a lazy UOp DAG     ← STEP 1 in this demo
     │  (pure Python — NO GPU work)       (forward pass returns a tree, not a value)
     │
     ▼
   .realize() / TinyJit triggers       ← STEP 2 (cnt=0: dry run, cnt=1: capture)
     │  execution of the graph
     │
     ├──► Scheduler                    ← create_schedule() in engine/schedule.py
     │      toposort the UOp DAG
     │      fuse compatible ops → list of ExecItems
     │
     ├──► Codegen (per ExecItem)       ← full_rewrite_to_sink() in codegen/__init__.py
     │      graph_rewrite passes:
     │        symbolic → split ranges → expand → devectorize → linearize
     │      BEAM search tries kernel layout combos, times them on GPU
     │        (or hand_coded_optimizations if BEAM=0)
     │
     ├──► Renderer → PTX source        ← NV renderer in runtime/ops_nv.py
     │      (visible with DEBUG≥4)
     │
     ├──► Compiler → cubin binary      ← nvrtc compiles PTX to native GPU code
     │
     ├──► NVProgram                    ← loads cubin to GPU VA, builds QMD struct
     │      (program_address, register count, shared mem, grid dims)
     │
     ▼  ┌──────────[ BP 1 ]───────────── engine/realize.py:212
   ei.run()                               ExecItem fires: one fused kernel
     │                                    Inspect ei.ast (kernel AST) + ei.bufs
     │
     ├──► HCQGraph                     ← runtime/graph/hcq.py
     │      Assembles static HWQueue command buffers:
     │        wait_signal → exec(QMD) → signal_done
     │      (built once at JIT capture; replayed on every subsequent call)
     │
     ├──► NVComputeQueue.exec()        ← writes QMD to command queue
     │      SEND_PCAS_A method → queue entry with program_address + args
     │
     ├──► _submit_to_gpfifo()          ← writes GPFIFO ring buffer entry
     │      gpput[0] = next slot
     │      memory_barrier()
     │
     ▼  ┌──────────[ BP 2 ]───────────── runtime/ops_nv.py:127
   dev.gpu_mmio[0x90 // 4] = token        MMIO doorbell poke — wakes the GPU
     │
     ▼
   GPU cmd processor                      fetches QMD from GPFIFO ring
     → loads cubin to SMs                 program_address resolved
     → dispatches grid of warps           warps execute on streaming multiprocessors
     → done                               signal written → host can read output

   ─── STEP 3: steady-state ──────────────────────────────────────────────
   After JIT capture (cnt≥2), TinyJit replays the HCQGraph directly:
     ctypes.memmove(addr, new_imu_data)   ← zero-copy via Tegra unified memory
     HCQGraph.__call__()                  ← patches input ptrs → doorbell poke
     (no Python per-kernel overhead — pure hardware replay)

══════════════════════════════════════════════════════════════════════════════

Run (inside nix develop, from examples/presentation/):
    NV=1 VIZ=1 JITBEAM=2 DEBUG=2 python3 demo_learned_inertial_odometry_flow.py

SSH tunnel (open on your PC first so VIZ serves to your browser):
    ssh -L 3000:localhost:3000 agent@192.168.8.162

Breakpoints to set in tinygrad source before running:
  BP 1  engine/realize.py:212   ei.run()  — fused kernel about to fire
  BP 2  runtime/ops_nv.py:127   dev.gpu_mmio[0x90//4] = token  — HW doorbell
  Then VIZ opens at exit → UOp rewrite browser + GPU profiler tab
"""

import os, sys, ctypes, time, statistics

assert os.environ.get("NV") == "1", \
    "Run with NV=1  e.g.  NV=1 VIZ=1 JITBEAM=2 DEBUG=2 python3 demo_learned_inertial_odometry_flow.py"

# ── import the Cioffi TCN builder from learned-inertial-odometry ──────────────
LIO_DIR = os.path.join(os.path.dirname(__file__), "..", "learned-inertial-odometry")
sys.path.insert(0, os.path.abspath(LIO_DIR))

import numpy as np
from cioffi_tcn import (
    INPUT_DIM, OUTPUT_DIM, SEQ_LEN,
    generate_weights, generate_input_pool, build_tinygrad_tcn, _count_params,
)
from tinygrad import Tensor, TinyJit, Device, dtypes

# ── build model ───────────────────────────────────────────────────────────────
weights = generate_weights()
model, param_count = build_tinygrad_tcn(weights, use_fp16=True)

print(f"\nCioffi TCN  input=({1},{INPUT_DIM},{SEQ_LEN})  output=({1},{OUTPUT_DIM})"
      f"  FP16  ~{param_count:,} params  device={Device.DEFAULT}")
print(f"env: DEBUG={os.environ.get('DEBUG','0')} VIZ={os.environ.get('VIZ','0')} "
      f"JITBEAM={os.environ.get('JITBEAM','0')}\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Build the lazy graph — pure Python, GPU untouched
#   model(x) returns a UOp DAG (expression tree). Nothing has been computed.
#   This is the first box in the pipeline diagram above.
# ─────────────────────────────────────────────────────────────────────────────
x   = Tensor.zeros(1, INPUT_DIM, SEQ_LEN, dtype=dtypes.float16).contiguous().realize()
out = Tensor.zeros(1, OUTPUT_DIM,         dtype=dtypes.float16).contiguous().realize()

lazy_out = model(x)
# ↑ lazy_out is a UOp DAG sitting in Python memory. GPU is idle.
#   In the debugger: inspect lazy_out.uop to see the full expression tree.
print(f"[lazy]   out.uop.op = {lazy_out.uop.op}  — UOp tree built, GPU idle")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: JIT — captures the full kernel graph on first real runs
#   cnt=0 → dry run (eager, no capture — exercises the full compile pipeline)
#   cnt=1 → JIT capture: schedule → codegen → BEAM → compile → HCQGraph built
#   cnt≥2 → steady-state: HCQGraph.__call__ patches ptrs → doorbell poke only
#
# Breakpoints fire during cnt=0 and cnt=1:
#   BP1 (realize.py:212) — ei.run() for each fused kernel
#   BP2 (ops_nv.py:127)  — dev.gpu_mmio write = hardware doorbell
# ─────────────────────────────────────────────────────────────────────────────
@TinyJit
def run():
    out.assign(model(x)).realize()

WARMUP, BENCH = 5, 300

# Direct-memory access for zero-copy input injection (Tegra unified memory)
in_addr  = x._buffer()._buf.cpu_view().addr
in_bytes = INPUT_DIM * SEQ_LEN * dtypes.float16.itemsize   # 6 * 50 * 2 = 600 bytes
pool = generate_input_pool(WARMUP + BENCH + 4)             # simulated IMU data

for i in range(WARMUP):
    ctypes.memmove(in_addr, pool[i].ctypes.data, in_bytes)
    t0 = time.perf_counter()
    run()
    Device["NV"].synchronize()
    label = {0: "dry-run (no capture)", 1: "JIT capture"}.get(i, "graph exec")
    print(f"[warmup {i}] {label}  {(time.perf_counter()-t0)*1e3:.1f} ms")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Steady-state benchmark — pure HCQGraph replay
#   No Python per-kernel overhead. Just: memmove → HCQGraph → doorbell → done.
# ─────────────────────────────────────────────────────────────────────────────
times_us: list[float] = []
for i in range(BENCH):
    ctypes.memmove(in_addr, pool[WARMUP + i].ctypes.data, in_bytes)
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
