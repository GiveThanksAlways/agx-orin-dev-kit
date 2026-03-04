"""BEAM search for YOLOv8 kernel optimization. Run with:
   PARALLEL=0 JITBEAM=2 NV=1 python3 beam_search.py
   
   Monitor progress: tail -f /tmp/beam_progress.log
"""
import sys, os, time
sys.path.insert(0, os.path.realpath("../../external/tinygrad"))

# Progress log
LOG = "/tmp/beam_progress.log"
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("Starting BEAM search for YOLOv8-n @ 320x320")
log(f"JITBEAM={os.environ.get('JITBEAM','unset')} BEAM={os.environ.get('BEAM','unset')} PARALLEL={os.environ.get('PARALLEL','unset')} NV={os.environ.get('NV','unset')}")

from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad.engine.jit import TinyJit
from examples.yolov8 import YOLOv8, get_variant_multiples, get_weights_location

log("Loading model...")
d, w, r = get_variant_multiples("n")
model = YOLOv8(w=w, r=r, d=d, num_classes=80)
state_dict = safe_load(get_weights_location("n"))
load_state_dict(model, state_dict)
log("Model loaded")

@TinyJit
def run(x): return model(x)

x = Tensor.randn(1, 3, 320, 320)

# Warmup pass 1 — graph capture, NO beam search yet
log("Warmup 1/3: graph capture...")
t0 = time.time()
y = run(x).realize()
Tensor.realize(y)
log(f"Warmup 1/3 done in {time.time()-t0:.1f}s")

# Warmup pass 2 — JIT kicks in, BEAM search runs on all kernels
log("Warmup 2/3: JIT + BEAM search (this is the slow one)...")
t0 = time.time()
y = run(x).realize()
Tensor.realize(y)
log(f"Warmup 2/3 done in {time.time()-t0:.1f}s")

# Warmup pass 3 — should be fast (cached)
log("Warmup 3/3: verification...")
t0 = time.time()
y = run(x).realize()
Tensor.realize(y)
log(f"Warmup 3/3 done in {time.time()-t0:.1f}s")

# Benchmark
import numpy as np
log("Benchmarking 100 iterations...")
times = []
for i in range(100):
    Tensor.realize(x)
    t0 = time.perf_counter_ns()
    y = run(x)
    Tensor.realize(y)
    elapsed = time.perf_counter_ns() - t0
    times.append(elapsed)
    if (i+1) % 25 == 0:
        a = np.array(times) / 1000
        log(f"  iter {i+1}/100: running median {np.median(a):.0f} us ({1e6/np.median(a):.1f} FPS)")

a = np.array(times) / 1000
log(f"\n=== RESULTS ===")
log(f"BEAM-optimized median: {np.median(a):.0f} us ({1e6/np.median(a):.1f} FPS)")
log(f"BEAM-optimized mean:   {np.mean(a):.0f} us ({1e6/np.mean(a):.1f} FPS)")
log(f"min/max: {np.min(a):.0f} / {np.max(a):.0f} us")
log(f"p5/p95:  {np.percentile(a,5):.0f} / {np.percentile(a,95):.0f} us")
log("DONE")
