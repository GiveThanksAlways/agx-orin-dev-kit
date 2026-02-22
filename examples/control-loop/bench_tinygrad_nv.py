#!/usr/bin/env python3
"""Control-loop latency benchmark: tinygrad NV=1 on Jetson AGX Orin.

Measures launch latency, inference time, total cycle time, and jitter
for two control-loop workloads (PID and sensor-fusion).

Usage (from tinygrad nix shell):
    cd examples/tinygrad && nix develop
    NV=1 python3 ../control-loop/bench_tinygrad_nv.py [--duration 60]

Or via run_all.sh which handles the nix shell automatically.
"""
import os, sys, time, argparse, csv
import numpy as np

os.environ.setdefault("NV", "1")

try:
    from tinygrad import Tensor, Device, TinyJit, dtypes
    from tinygrad import nn as tg_nn
except ImportError:
    print("ERROR: tinygrad not found. Run from tinygrad nix shell:")
    print("  cd examples/tinygrad && nix develop")
    sys.exit(1)

# ── Constants ────────────────────────────────────────────────────────────────
SEED        = 42
IN_DIM_PID  = 12   # position(3) + velocity(3) + orientation(3) + angular_vel(3)
IN_DIM_SF   = 24   # full raw sensor suite
OUT_DIM     = 4    # motor commands
HIDDEN      = 128

# ── Deterministic weight generation (shared with PyTorch benchmark) ──────────
def generate_weights(in_dim, out_dim, seed=SEED):
    rng = np.random.RandomState(seed)
    def _layer(fan_in, fan_out):
        s = 1.0 / np.sqrt(fan_in)
        w = (rng.randn(fan_out, fan_in) * s).astype(np.float16)
        b = (rng.randn(fan_out) * s).astype(np.float16)
        return w, b
    w1, b1 = _layer(in_dim, HIDDEN)
    w2, b2 = _layer(HIDDEN, HIDDEN)
    w3, b3 = _layer(HIDDEN, out_dim)
    return dict(w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)

# ── Model ────────────────────────────────────────────────────────────────────
class ControlMLP:
    """2-layer MLP: in → 128 → 128 → out, FP16."""
    def __init__(self, in_dim, out_dim, weights):
        self.l1 = tg_nn.Linear(in_dim, HIDDEN)
        self.l2 = tg_nn.Linear(HIDDEN, HIDDEN)
        self.l3 = tg_nn.Linear(HIDDEN, out_dim)
        # Load deterministic weights
        self.l1.weight = Tensor(weights['w1'])
        self.l1.bias   = Tensor(weights['b1'])
        self.l2.weight = Tensor(weights['w2'])
        self.l2.bias   = Tensor(weights['b2'])
        self.l3.weight = Tensor(weights['w3'])
        self.l3.bias   = Tensor(weights['b3'])

    def __call__(self, x):
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        return self.l3(x)

# ── CPU control components ───────────────────────────────────────────────────
class PIDController:
    def __init__(self, kp=1.0, kd=0.1):
        self.kp, self.kd = kp, kd
        self.prev_error = np.zeros(OUT_DIM, dtype=np.float32)

    def step(self, measurement, setpoint, dt):
        error = setpoint - measurement
        deriv = (error - self.prev_error) / max(dt, 1e-9)
        self.prev_error = error.copy()
        return self.kp * error + self.kd * deriv

class SteadyStateKalman:
    """Constant-gain Kalman filter (pre-converged). O(n²) per update."""
    def __init__(self, dim):
        self.x = np.zeros(dim, dtype=np.float32)
        self.K = np.eye(dim, dtype=np.float32) * 0.3  # steady-state gain

    def update(self, z):
        self.x += self.K @ (z - self.x)
        return self.x

# ── Helpers ──────────────────────────────────────────────────────────────────
def stats(data):
    a = np.asarray(data)
    return {
        'mean': np.mean(a), 'median': np.median(a), 'std': np.std(a),
        'min': np.min(a), 'max': np.max(a),
        'p99': np.percentile(a, 99), 'p999': np.percentile(a, 99.9),
        'max_dev': np.max(np.abs(a - np.mean(a))),
        'count': len(a),
    }

def print_stats(label, data):
    s = stats(data)
    freq = 1e6 / s['mean'] if s['mean'] > 0 else 0
    print(f"  {label}:")
    print(f"    mean={s['mean']:.1f} µs  median={s['median']:.1f} µs  std={s['std']:.1f} µs")
    print(f"    p99={s['p99']:.1f} µs  p99.9={s['p999']:.1f} µs  max={s['max']:.1f} µs")
    print(f"    achieved freq={freq:.0f} Hz  iterations={s['count']}")

def save_csv(path, columns, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(columns)
        for row in rows:
            w.writerow(row)

# ── Phase 1: Launch Latency ─────────────────────────────────────────────────
def bench_launch_latency(n_iters=10000):
    """Minimal GPU operation (1-element add) to measure pure launch overhead."""
    @TinyJit
    def noop(x):
        return (x + x).realize()

    dummy = np.zeros((1, 1), dtype=np.float16)
    for _ in range(5):
        _ = noop(Tensor(dummy)).numpy()

    times_us = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        _ = noop(Tensor(dummy)).numpy()
        t1 = time.perf_counter_ns()
        times_us.append((t1 - t0) / 1000.0)
    return times_us

# ── Phase 2: PID Control Loop ───────────────────────────────────────────────
def bench_pid_loop(duration_s):
    weights = generate_weights(IN_DIM_PID, OUT_DIM, seed=SEED)
    model = ControlMLP(IN_DIM_PID, OUT_DIM, weights)
    pid = PIDController()
    setpoint = np.zeros(OUT_DIM, dtype=np.float32)

    @TinyJit
    def run_model(x):
        return model(x).realize()

    # Pre-generate sensor data
    n_max = int(duration_s * 50000)
    rng = np.random.RandomState(123)
    sensor_pool = rng.randn(n_max, 1, IN_DIM_PID).astype(np.float16)

    # Warmup
    for i in range(5):
        _ = run_model(Tensor(sensor_pool[i])).numpy()

    inference_us, cycle_us = [], []
    idx = 0
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        t_cyc = time.perf_counter_ns()

        # Sensor read (pre-generated)
        sensor_data = sensor_pool[idx % n_max]

        # GPU inference
        t_inf = time.perf_counter_ns()
        result = run_model(Tensor(sensor_data)).numpy().flatten()
        t_inf_done = time.perf_counter_ns()

        # PID (CPU)
        dt = (t_inf_done - t_cyc) / 1e9
        _ = pid.step(result.astype(np.float32), setpoint, dt)

        t_cyc_done = time.perf_counter_ns()
        inference_us.append((t_inf_done - t_inf) / 1000.0)
        cycle_us.append((t_cyc_done - t_cyc) / 1000.0)
        idx += 1

    return inference_us, cycle_us

# ── Phase 3: Sensor Fusion Loop ─────────────────────────────────────────────
def bench_sensor_fusion_loop(duration_s):
    weights = generate_weights(IN_DIM_SF, OUT_DIM, seed=SEED + 1)
    model = ControlMLP(IN_DIM_SF, OUT_DIM, weights)
    pid = PIDController()
    kalman = SteadyStateKalman(IN_DIM_SF)
    setpoint = np.zeros(OUT_DIM, dtype=np.float32)

    @TinyJit
    def run_model(x):
        return model(x).realize()

    n_max = int(duration_s * 50000)
    rng = np.random.RandomState(456)
    raw_pool = rng.randn(n_max, IN_DIM_SF).astype(np.float32)

    # Warmup
    for i in range(5):
        dummy = raw_pool[i].reshape(1, -1).astype(np.float16)
        _ = run_model(Tensor(dummy)).numpy()

    inference_us, cycle_us = [], []
    idx = 0
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        t_cyc = time.perf_counter_ns()

        # Sensor read (raw noisy data)
        raw = raw_pool[idx % n_max]

        # Kalman filter (CPU)
        filtered = kalman.update(raw)

        # GPU inference
        t_inf = time.perf_counter_ns()
        inp = filtered.reshape(1, -1).astype(np.float16)
        result = run_model(Tensor(inp)).numpy().flatten()
        t_inf_done = time.perf_counter_ns()

        # PID (CPU)
        dt = (t_inf_done - t_cyc) / 1e9
        _ = pid.step(result.astype(np.float32), setpoint, dt)

        t_cyc_done = time.perf_counter_ns()
        inference_us.append((t_inf_done - t_inf) / 1000.0)
        cycle_us.append((t_cyc_done - t_cyc) / 1000.0)
        idx += 1

    return inference_us, cycle_us

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="tinygrad NV=1 control-loop benchmark")
    ap.add_argument('--duration', type=int, default=60, help='seconds per test (default: 60)')
    ap.add_argument('--output-dir', type=str, default='results', help='CSV output directory')
    args = ap.parse_args()

    if Device.DEFAULT != "NV":
        print(f"ERROR: Device is '{Device.DEFAULT}', expected 'NV'. Set NV=1.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== tinygrad NV=1 control-loop benchmark ===")
    print(f"Device: {Device.DEFAULT}")
    print(f"Duration per test: {args.duration}s")
    print(f"Output: {args.output_dir}/")
    print()

    # ── Phase 1: Launch Latency ──────────────────────────────────────────
    print("Phase 1: Launch latency (10 000 iterations, minimal 1-elem op)...")
    lat = bench_launch_latency(10000)
    save_csv(os.path.join(args.output_dir, 'tinygrad_nv_launch.csv'),
             ['latency_us'], [[v] for v in lat])
    print_stats("Launch latency", lat)
    print()

    # ── Phase 2: PID Loop ────────────────────────────────────────────────
    print(f"Phase 2: PID control loop ({args.duration}s)...")
    pid_inf, pid_cyc = bench_pid_loop(args.duration)
    save_csv(os.path.join(args.output_dir, 'tinygrad_nv_pid.csv'),
             ['inference_us', 'cycle_us'],
             list(zip(pid_inf, pid_cyc)))
    print_stats("Inference", pid_inf)
    print_stats("Total cycle", pid_cyc)
    print()

    # ── Phase 3: Sensor Fusion Loop ──────────────────────────────────────
    print(f"Phase 3: Sensor-fusion loop ({args.duration}s)...")
    sf_inf, sf_cyc = bench_sensor_fusion_loop(args.duration)
    save_csv(os.path.join(args.output_dir, 'tinygrad_nv_sensor_fusion.csv'),
             ['inference_us', 'cycle_us'],
             list(zip(sf_inf, sf_cyc)))
    print_stats("Inference", sf_inf)
    print_stats("Total cycle", sf_cyc)
    print()

    print("Done. CSVs written to", args.output_dir)

if __name__ == '__main__':
    main()
