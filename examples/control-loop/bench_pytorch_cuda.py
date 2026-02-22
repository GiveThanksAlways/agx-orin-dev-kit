#!/usr/bin/env python3
"""Control-loop latency benchmark: PyTorch CUDA on Jetson AGX Orin.

Tests both eager mode and CUDA-graph replay mode.

Usage (from control-loop nix shell):
    cd examples/control-loop && nix develop
    python3 bench_pytorch_cuda.py [--duration 60]
"""
import os, sys, time, argparse, csv
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("ERROR: PyTorch not found. Run from control-loop nix shell:")
    print("  cd examples/control-loop && nix develop")
    sys.exit(1)

# ── Constants (must match tinygrad benchmark exactly) ────────────────────────
SEED        = 42
IN_DIM_PID  = 12
IN_DIM_SF   = 24
OUT_DIM     = 4
HIDDEN      = 128

# ── Deterministic weight generation (identical to tinygrad benchmark) ────────
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
class TorchControlMLP(nn.Module):
    """Same architecture as tinygrad ControlMLP: in → 128 → 128 → out, FP16."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, HIDDEN)
        self.l2 = nn.Linear(HIDDEN, HIDDEN)
        self.l3 = nn.Linear(HIDDEN, out_dim)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.l3(x)

def load_weights(model, weights):
    with torch.no_grad():
        model.l1.weight.copy_(torch.from_numpy(weights['w1']))
        model.l1.bias.copy_(torch.from_numpy(weights['b1']))
        model.l2.weight.copy_(torch.from_numpy(weights['w2']))
        model.l2.bias.copy_(torch.from_numpy(weights['b2']))
        model.l3.weight.copy_(torch.from_numpy(weights['w3']))
        model.l3.bias.copy_(torch.from_numpy(weights['b3']))

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
    def __init__(self, dim):
        self.x = np.zeros(dim, dtype=np.float32)
        self.K = np.eye(dim, dtype=np.float32) * 0.3

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
def bench_launch_latency_eager(n_iters=10000):
    x = torch.zeros(1, 1, device='cuda', dtype=torch.float16)
    # warmup
    for _ in range(20):
        _ = (x + x)
        torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        _ = (x + x)
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)
    return times

def bench_launch_latency_graph(n_iters=10000):
    x = torch.zeros(1, 1, device='cuda', dtype=torch.float16)
    # warmup
    for _ in range(20):
        _ = (x + x)
        torch.cuda.synchronize()

    # capture graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = x + x
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        g.replay()
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)
    return times

# ── PID Loop Helper ─────────────────────────────────────────────────────────
def _pid_loop(model, static_input, duration_s, in_dim, use_graph, graph=None, static_output=None):
    pid = PIDController()
    setpoint = np.zeros(OUT_DIM, dtype=np.float32)

    n_max = int(duration_s * 50000)
    rng = np.random.RandomState(123)
    # Keep sensor data on CPU — transfer to GPU each iteration (realistic: sensors are on host)
    sensor_pool = rng.randn(n_max, 1, in_dim).astype(np.float16)

    inference_us, cycle_us = [], []
    idx = 0
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        t_cyc = time.perf_counter_ns()

        si = idx % n_max

        t_inf = time.perf_counter_ns()
        inp_gpu = torch.from_numpy(sensor_pool[si]).cuda()
        if use_graph:
            static_input.copy_(inp_gpu)
            graph.replay()
            torch.cuda.synchronize()
            result = static_output.cpu().numpy().flatten()
        else:
            out = model(inp_gpu)
            torch.cuda.synchronize()
            result = out.cpu().numpy().flatten()
        t_inf_done = time.perf_counter_ns()

        dt = (t_inf_done - t_cyc) / 1e9
        _ = pid.step(result.astype(np.float32), setpoint, dt)

        t_cyc_done = time.perf_counter_ns()
        inference_us.append((t_inf_done - t_inf) / 1000.0)
        cycle_us.append((t_cyc_done - t_cyc) / 1000.0)
        idx += 1

    return inference_us, cycle_us

# ── Sensor Fusion Loop Helper ────────────────────────────────────────────────
def _sf_loop(model, static_input, duration_s, in_dim, use_graph, graph=None, static_output=None):
    pid = PIDController()
    kalman = SteadyStateKalman(in_dim)
    setpoint = np.zeros(OUT_DIM, dtype=np.float32)

    n_max = int(duration_s * 50000)
    rng = np.random.RandomState(456)
    raw_pool = rng.randn(n_max, in_dim).astype(np.float32)

    inference_us, cycle_us = [], []
    idx = 0
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        t_cyc = time.perf_counter_ns()

        raw = raw_pool[idx % n_max]
        filtered = kalman.update(raw)

        t_inf = time.perf_counter_ns()
        inp_t = torch.from_numpy(filtered.reshape(1, -1).astype(np.float16)).cuda()
        if use_graph:
            static_input.copy_(inp_t)
            graph.replay()
            torch.cuda.synchronize()
            result = static_output.cpu().numpy().flatten()
        else:
            out = model(inp_t)
            torch.cuda.synchronize()
            result = out.cpu().numpy().flatten()
        t_inf_done = time.perf_counter_ns()

        dt = (t_inf_done - t_cyc) / 1e9
        _ = pid.step(result.astype(np.float32), setpoint, dt)

        t_cyc_done = time.perf_counter_ns()
        inference_us.append((t_inf_done - t_inf) / 1000.0)
        cycle_us.append((t_cyc_done - t_cyc) / 1000.0)
        idx += 1

    return inference_us, cycle_us

# ── Build model + optional CUDA graph ────────────────────────────────────────
def make_model_and_graph(in_dim, out_dim, weights, use_graph):
    model = TorchControlMLP(in_dim, out_dim).cuda().half().eval()
    load_weights(model, weights)

    static_input = torch.zeros(1, in_dim, device='cuda', dtype=torch.float16)
    graph, static_output = None, None

    # warmup
    with torch.inference_mode():
        for _ in range(20):
            _ = model(static_input)
        torch.cuda.synchronize()

    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            static_output = model(static_input)
        torch.cuda.synchronize()

    return model, static_input, graph, static_output

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="PyTorch CUDA control-loop benchmark")
    ap.add_argument('--duration', type=int, default=60)
    ap.add_argument('--output-dir', type=str, default='results')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available in this PyTorch build.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    dev_name = torch.cuda.get_device_name(0)
    print(f"=== PyTorch CUDA control-loop benchmark ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device:  {dev_name}")
    print(f"Duration per test: {args.duration}s")
    print(f"Output: {args.output_dir}/")
    print()

    with torch.inference_mode():
        # ── Launch latency ───────────────────────────────────────────────
        print("Phase 1a: Launch latency — eager (10 000 iters)...")
        lat_e = bench_launch_latency_eager(10000)
        save_csv(os.path.join(args.output_dir, 'pytorch_eager_launch.csv'),
                 ['latency_us'], [[v] for v in lat_e])
        print_stats("Eager launch", lat_e)

        print("Phase 1b: Launch latency — CUDA graph (10 000 iters)...")
        lat_g = bench_launch_latency_graph(10000)
        save_csv(os.path.join(args.output_dir, 'pytorch_graph_launch.csv'),
                 ['latency_us'], [[v] for v in lat_g])
        print_stats("Graph launch", lat_g)
        print()

        # ── PID loop — eager ─────────────────────────────────────────────
        for mode, use_graph in [("eager", False), ("graph", True)]:
            tag = f"pytorch_{mode}"
            print(f"Phase 2 ({mode}): PID control loop ({args.duration}s)...")
            w = generate_weights(IN_DIM_PID, OUT_DIM, seed=SEED)
            mdl, si, g, so = make_model_and_graph(IN_DIM_PID, OUT_DIM, w, use_graph)
            pid_inf, pid_cyc = _pid_loop(mdl, si, args.duration, IN_DIM_PID, use_graph, g, so)
            save_csv(os.path.join(args.output_dir, f'{tag}_pid.csv'),
                     ['inference_us', 'cycle_us'], list(zip(pid_inf, pid_cyc)))
            print_stats("Inference", pid_inf)
            print_stats("Total cycle", pid_cyc)
            del mdl, si, g, so
            torch.cuda.empty_cache()
            print()

        # ── Sensor-fusion loop ───────────────────────────────────────────
        for mode, use_graph in [("eager", False), ("graph", True)]:
            tag = f"pytorch_{mode}"
            print(f"Phase 3 ({mode}): Sensor-fusion loop ({args.duration}s)...")
            w = generate_weights(IN_DIM_SF, OUT_DIM, seed=SEED + 1)
            mdl, si, g, so = make_model_and_graph(IN_DIM_SF, OUT_DIM, w, use_graph)
            sf_inf, sf_cyc = _sf_loop(mdl, si, args.duration, IN_DIM_SF, use_graph, g, so)
            save_csv(os.path.join(args.output_dir, f'{tag}_sensor_fusion.csv'),
                     ['inference_us', 'cycle_us'], list(zip(sf_inf, sf_cyc)))
            print_stats("Inference", sf_inf)
            print_stats("Total cycle", sf_cyc)
            del mdl, si, g, so
            torch.cuda.empty_cache()
            print()

    print("Done. CSVs written to", args.output_dir)

if __name__ == '__main__':
    main()
