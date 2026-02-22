#!/usr/bin/env python3
"""Breakdown benchmark: WHERE does the time actually go?

Isolates each component of the control-loop cycle to show that NV=1's
hardware dispatch is fast — the overhead is in Python/data-transfer.

Tests (tinygrad NV=1):
  1. GPU-resident launch latency  — data on GPU, no H2D/D2H  (apples-to-apples with PyTorch)
  2. Component breakdown           — Tensor creation, dispatch, sync, readback, full round-trip
  3. MLP GPU-resident              — full MLP with data on GPU, no transfer
  4. Optimized control loop        — using assign() to avoid per-iteration allocation
  5. Naive control loop            — what we measured before (Tensor() + .numpy() each iter)

Tests (PyTorch CUDA):
  6. GPU-resident launch latency   — same as test 1 but PyTorch
  7. MLP GPU-resident              — same as test 3 but PyTorch
  8. Optimized control loop        — CUDA Graphs with static input copy

Usage:
  # tinygrad tests:
  cd examples/tinygrad && nix develop
  NV=1 python3 ../control-loop/bench_breakdown.py --backend tinygrad

  # PyTorch tests:
  cd examples/control-loop && nix develop
  python3 bench_breakdown.py --backend pytorch

  # Both (via run script):
  ./run_breakdown.sh
"""
import os, sys, time, argparse
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
SEED        = 42
IN_DIM_PID  = 12
OUT_DIM     = 4
HIDDEN      = 128
WARMUP      = 50
N_ITERS     = 10000

def generate_weights(in_dim=IN_DIM_PID, out_dim=OUT_DIM, seed=SEED):
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

def stats(data):
    a = np.asarray(data)
    return dict(mean=np.mean(a), median=np.median(a), std=np.std(a),
                min=np.min(a), max=np.max(a),
                p99=np.percentile(a, 99), p999=np.percentile(a, 99.9),
                count=len(a))

def print_row(label, data):
    s = stats(data)
    print(f"  {label:40s}  median={s['median']:8.1f} µs  mean={s['mean']:8.1f} µs  "
          f"std={s['std']:6.1f} µs  p99={s['p99']:8.1f} µs  min={s['min']:7.1f} µs")

# ═══════════════════════════════════════════════════════════════════════════════
# TINYGRAD BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════
def run_tinygrad():
    os.environ.setdefault("NV", "1")
    from tinygrad import Tensor, Device, TinyJit, dtypes
    from tinygrad import nn as tg_nn

    if Device.DEFAULT != "NV":
        print(f"ERROR: Device is '{Device.DEFAULT}', expected 'NV'. Set NV=1.")
        sys.exit(1)

    print(f"Backend: tinygrad NV=1")
    print(f"Device:  {Device.DEFAULT}")
    print(f"Iters:   {N_ITERS}")
    print()

    # ── Helper: sync ─────────────────────────────────────────────────────
    def sync():
        Device["NV"].synchronize()

    # ── Test 1: GPU-resident launch latency (1-elem add) ────────────────
    print("=" * 80)
    print("TEST 1: GPU-resident launch latency (1-elem add, data stays on GPU)")
    print("  This is the apples-to-apples comparison with PyTorch's launch test.")
    print("=" * 80)

    x_gpu = Tensor(np.zeros((1, 1), dtype=np.float16)).realize()
    sync()

    @TinyJit
    def noop_gpu(x):
        return (x + x).realize()

    for _ in range(WARMUP):
        _ = noop_gpu(x_gpu)
        sync()

    times = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter_ns()
        _ = noop_gpu(x_gpu)
        sync()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)
    print_row("GPU-resident dispatch+sync", times)
    result_gpu_launch = stats(times)
    print()

    # ── Test 2: Component breakdown ──────────────────────────────────────
    print("=" * 80)
    print("TEST 2: Component breakdown (isolating each piece)")
    print("=" * 80)

    dummy = np.zeros((1, 1), dtype=np.float16)

    # 2a: Tensor creation (H2D copy)
    sync()
    times_create = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter_ns()
        t = Tensor(dummy).realize()
        sync()
        t1 = time.perf_counter_ns()
        times_create.append((t1 - t0) / 1000.0)
    print_row("Tensor(numpy) + realize + sync", times_create)

    # 2b: .numpy() readback (D2H copy) on a pre-computed GPU tensor
    pre = Tensor(np.ones((1, 1), dtype=np.float16)).realize()
    sync()
    times_readback = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter_ns()
        _ = pre.numpy()
        t1 = time.perf_counter_ns()
        times_readback.append((t1 - t0) / 1000.0)
    print_row(".numpy() D2H readback", times_readback)

    # 2c: Just Device sync (when nothing is pending)
    sync()
    times_sync = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter_ns()
        sync()
        t1 = time.perf_counter_ns()
        times_sync.append((t1 - t0) / 1000.0)
    print_row("Device sync (nothing pending)", times_sync)

    # 2d: Full round-trip (what the original benchmark measured)
    @TinyJit
    def noop_rt(x):
        return (x + x).realize()

    for _ in range(WARMUP):
        _ = noop_rt(Tensor(dummy)).numpy()

    times_rt = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter_ns()
        _ = noop_rt(Tensor(dummy)).numpy()
        t1 = time.perf_counter_ns()
        times_rt.append((t1 - t0) / 1000.0)
    print_row("Full round-trip: Tensor()+JIT+.numpy()", times_rt)
    print()

    # ── Test 3: MLP GPU-resident ─────────────────────────────────────────
    print("=" * 80)
    print("TEST 3: MLP inference, GPU-resident (no H2D or D2H)")
    print("=" * 80)

    weights = generate_weights()

    class ControlMLP:
        def __init__(self, in_dim, out_dim, w):
            self.l1 = tg_nn.Linear(in_dim, HIDDEN)
            self.l2 = tg_nn.Linear(HIDDEN, HIDDEN)
            self.l3 = tg_nn.Linear(HIDDEN, out_dim)
            self.l1.weight, self.l1.bias = Tensor(w['w1']), Tensor(w['b1'])
            self.l2.weight, self.l2.bias = Tensor(w['w2']), Tensor(w['b2'])
            self.l3.weight, self.l3.bias = Tensor(w['w3']), Tensor(w['b3'])
        def __call__(self, x):
            return self.l3(self.l2(self.l1(x).relu()).relu())

    model = ControlMLP(IN_DIM_PID, OUT_DIM, weights)
    x_mlp = Tensor(np.random.randn(1, IN_DIM_PID).astype(np.float16)).realize()
    sync()

    @TinyJit
    def run_mlp_gpu(x):
        return model(x).realize()

    for _ in range(WARMUP):
        _ = run_mlp_gpu(x_mlp)
        sync()

    times_mlp = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter_ns()
        _ = run_mlp_gpu(x_mlp)
        sync()
        t1 = time.perf_counter_ns()
        times_mlp.append((t1 - t0) / 1000.0)
    print_row("MLP dispatch+sync (GPU-resident)", times_mlp)
    result_mlp_gpu = stats(times_mlp)

    # Also measure MLP full round-trip for comparison
    @TinyJit
    def run_mlp_rt(x):
        return model(x).realize()

    sensor = np.random.randn(1, IN_DIM_PID).astype(np.float16)
    for _ in range(WARMUP):
        _ = run_mlp_rt(Tensor(sensor)).numpy()

    times_mlp_rt = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter_ns()
        _ = run_mlp_rt(Tensor(sensor)).numpy()
        t1 = time.perf_counter_ns()
        times_mlp_rt.append((t1 - t0) / 1000.0)
    print_row("MLP full round-trip: Tensor()+JIT+.numpy()", times_mlp_rt)
    result_mlp_rt = stats(times_mlp_rt)
    print()

    # ── Test 4: Optimized control loop with assign() ─────────────────────
    print("=" * 80)
    print("TEST 4: Optimized control loop (assign + implicit input)")
    print("  Uses x.assign().realize() to update GPU buffer in-place,")
    print("  avoiding per-iteration Tensor/Buffer allocation.")
    print("=" * 80)

    model2 = ControlMLP(IN_DIM_PID, OUT_DIM, weights)
    static_x = Tensor.zeros(1, IN_DIM_PID).cast(dtypes.float16).realize()
    sync()

    @TinyJit
    def run_mlp_assign():
        return model2(static_x).realize()

    sensor_pool = np.random.randn(N_ITERS + WARMUP, 1, IN_DIM_PID).astype(np.float16)
    for i in range(WARMUP):
        static_x.assign(Tensor(sensor_pool[i].reshape(1, IN_DIM_PID))).realize()
        out = run_mlp_assign()
        _ = out.numpy()

    # Measure the assign+infer+readback path
    times_assign_full = []
    for i in range(N_ITERS):
        t0 = time.perf_counter_ns()
        static_x.assign(Tensor(sensor_pool[WARMUP + i].reshape(1, IN_DIM_PID))).realize()
        out = run_mlp_assign()
        result = out.numpy()
        t1 = time.perf_counter_ns()
        times_assign_full.append((t1 - t0) / 1000.0)
    print_row("assign()+MLP+.numpy() full cycle", times_assign_full)
    result_assign = stats(times_assign_full)
    print()

    # ── Test 5: Naive control loop (original methodology) ────────────────
    print("=" * 80)
    print("TEST 5: Naive control loop (Tensor() + MLP + .numpy() each iter)")
    print("  This is what the original benchmark measured.")
    print("=" * 80)

    model3 = ControlMLP(IN_DIM_PID, OUT_DIM, weights)

    @TinyJit
    def run_mlp_naive(x):
        return model3(x).realize()

    for i in range(WARMUP):
        _ = run_mlp_naive(Tensor(sensor_pool[i].reshape(1, IN_DIM_PID))).numpy()

    times_naive = []
    for i in range(N_ITERS):
        t0 = time.perf_counter_ns()
        out = run_mlp_naive(Tensor(sensor_pool[WARMUP + i].reshape(1, IN_DIM_PID))).numpy()
        t1 = time.perf_counter_ns()
        times_naive.append((t1 - t0) / 1000.0)
    print_row("Tensor()+MLP+.numpy() naive cycle", times_naive)
    result_naive = stats(times_naive)
    print()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 80)
    print("SUMMARY — Where does the time go?")
    print("=" * 80)
    print()
    s_create = stats(times_create)
    s_read = stats(times_readback)
    s_sync = stats(times_sync)
    s_gpu = result_gpu_launch
    print(f"  Component                            Median µs")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Tensor(numpy) creation + H2D copy    {s_create['median']:8.1f}")
    print(f"  GPU-resident 1-elem dispatch+sync    {s_gpu['median']:8.1f}")
    print(f"  .numpy() D2H readback                {s_read['median']:8.1f}")
    print(f"  Device sync (idle)                   {s_sync['median']:8.1f}")
    print(f"  ─────────────────────────────────────────────────")
    est = s_create['median'] + s_gpu['median'] + s_read['median']
    rt_med = stats(times_rt)['median']
    print(f"  Estimated sum                        {est:8.1f}")
    print(f"  Measured full round-trip              {rt_med:8.1f}")
    print()
    print(f"  MLP (GPU-resident, no transfer)      {result_mlp_gpu['median']:8.1f}")
    print(f"  MLP (full round-trip, naive)         {result_naive['median']:8.1f}")
    print(f"  MLP (assign + readback, optimized)   {result_assign['median']:8.1f}")
    print()
    if result_naive['median'] > 0:
        gpu_pct = (result_mlp_gpu['median'] / result_naive['median']) * 100
        overhead_pct = 100 - gpu_pct
        print(f"  GPU dispatch is {gpu_pct:.1f}% of the naive cycle time.")
        print(f"  Python/transfer overhead is {overhead_pct:.1f}% of the naive cycle time.")
    print()

    return {
        'gpu_launch': result_gpu_launch, 'mlp_gpu': result_mlp_gpu,
        'mlp_rt': result_mlp_rt, 'assign': result_assign, 'naive': result_naive,
        'tensor_create': s_create, 'readback': s_read, 'sync': s_sync,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PYTORCH BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════
def run_pytorch():
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    print(f"Backend: PyTorch {torch.__version__}")
    print(f"Device:  {torch.cuda.get_device_name(0)}")
    print(f"Iters:   {N_ITERS}")
    print()

    def sync():
        torch.cuda.synchronize()

    # ── Test 6: GPU-resident launch latency ──────────────────────────────
    print("=" * 80)
    print("TEST 6: GPU-resident launch latency (1-elem add)")
    print("=" * 80)

    x_gpu = torch.zeros(1, 1, device='cuda', dtype=torch.float16)

    # Eager
    for _ in range(WARMUP):
        _ = (x_gpu + x_gpu); sync()
    times_eager = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter_ns()
        _ = (x_gpu + x_gpu)
        sync()
        t1 = time.perf_counter_ns()
        times_eager.append((t1 - t0) / 1000.0)
    print_row("Eager dispatch+sync", times_eager)

    # CUDA Graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_g = x_gpu + x_gpu
    sync()
    for _ in range(WARMUP):
        g.replay(); sync()
    times_graph = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter_ns()
        g.replay()
        sync()
        t1 = time.perf_counter_ns()
        times_graph.append((t1 - t0) / 1000.0)
    print_row("CUDA Graph dispatch+sync", times_graph)
    print()

    # ── Test 7: Component breakdown ──────────────────────────────────────
    print("=" * 80)
    print("TEST 7: Component breakdown (PyTorch)")
    print("=" * 80)

    dummy = np.zeros((1, 1), dtype=np.float16)

    # CPU→GPU transfer
    times_h2d = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter_ns()
        t = torch.from_numpy(dummy).cuda()
        sync()
        t1 = time.perf_counter_ns()
        times_h2d.append((t1 - t0) / 1000.0)
    print_row("torch.from_numpy().cuda() H2D", times_h2d)

    # GPU→CPU readback
    pre = torch.ones(1, 1, device='cuda', dtype=torch.float16)
    times_d2h = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter_ns()
        _ = pre.cpu().numpy()
        t1 = time.perf_counter_ns()
        times_d2h.append((t1 - t0) / 1000.0)
    print_row(".cpu().numpy() D2H readback", times_d2h)

    # Full round-trip (1-elem)
    times_rt = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter_ns()
        r = torch.from_numpy(dummy).cuda()
        r = r + r
        sync()
        _ = r.cpu().numpy()
        t1 = time.perf_counter_ns()
        times_rt.append((t1 - t0) / 1000.0)
    print_row("Full round-trip: H2D+add+sync+D2H", times_rt)
    print()

    # ── Test 8: MLP GPU-resident ─────────────────────────────────────────
    print("=" * 80)
    print("TEST 8: MLP inference, GPU-resident (no H2D or D2H)")
    print("=" * 80)

    weights = generate_weights()

    class TorchMLP(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.l1 = nn.Linear(in_dim, HIDDEN)
            self.l2 = nn.Linear(HIDDEN, HIDDEN)
            self.l3 = nn.Linear(HIDDEN, out_dim)
        def forward(self, x):
            return self.l3(torch.relu(self.l2(torch.relu(self.l1(x)))))

    model = TorchMLP(IN_DIM_PID, OUT_DIM).cuda().half().eval()
    with torch.no_grad():
        model.l1.weight.copy_(torch.from_numpy(weights['w1']))
        model.l1.bias.copy_(torch.from_numpy(weights['b1']))
        model.l2.weight.copy_(torch.from_numpy(weights['w2']))
        model.l2.bias.copy_(torch.from_numpy(weights['b2']))
        model.l3.weight.copy_(torch.from_numpy(weights['w3']))
        model.l3.bias.copy_(torch.from_numpy(weights['b3']))

    x_mlp = torch.randn(1, IN_DIM_PID, device='cuda', dtype=torch.float16)

    with torch.inference_mode():
        # Eager
        for _ in range(WARMUP):
            _ = model(x_mlp); sync()
        times_eager_mlp = []
        for _ in range(N_ITERS):
            t0 = time.perf_counter_ns()
            _ = model(x_mlp)
            sync()
            t1 = time.perf_counter_ns()
            times_eager_mlp.append((t1 - t0) / 1000.0)
        print_row("Eager MLP dispatch+sync", times_eager_mlp)

        # CUDA Graph
        static_in = torch.zeros(1, IN_DIM_PID, device='cuda', dtype=torch.float16)
        gm = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gm):
            static_out = model(static_in)
        sync()
        for _ in range(WARMUP):
            gm.replay(); sync()
        times_graph_mlp = []
        for _ in range(N_ITERS):
            t0 = time.perf_counter_ns()
            gm.replay()
            sync()
            t1 = time.perf_counter_ns()
            times_graph_mlp.append((t1 - t0) / 1000.0)
        print_row("CUDA Graph MLP dispatch+sync", times_graph_mlp)

        # Full round-trip (naive, matching tinygrad's approach)
        sensor_pool = np.random.randn(N_ITERS + WARMUP, 1, IN_DIM_PID).astype(np.float16)
        for i in range(WARMUP):
            inp = torch.from_numpy(sensor_pool[i]).cuda()
            _ = model(inp).cpu().numpy()
        times_naive_mlp = []
        for i in range(N_ITERS):
            t0 = time.perf_counter_ns()
            inp = torch.from_numpy(sensor_pool[WARMUP + i]).cuda()
            out = model(inp)
            sync()
            result = out.cpu().numpy()
            t1 = time.perf_counter_ns()
            times_naive_mlp.append((t1 - t0) / 1000.0)
        print_row("Naive MLP: H2D+infer+sync+D2H", times_naive_mlp)

        # Optimized: CUDA Graph with static input copy
        static_in2 = torch.zeros(1, IN_DIM_PID, device='cuda', dtype=torch.float16)
        gm2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gm2):
            static_out2 = model(static_in2)
        sync()
        for i in range(WARMUP):
            static_in2.copy_(torch.from_numpy(sensor_pool[i]).cuda())
            gm2.replay(); sync()
            _ = static_out2.cpu().numpy()
        times_opt_mlp = []
        for i in range(N_ITERS):
            t0 = time.perf_counter_ns()
            static_in2.copy_(torch.from_numpy(sensor_pool[WARMUP + i]).cuda())
            gm2.replay()
            sync()
            result = static_out2.cpu().numpy()
            t1 = time.perf_counter_ns()
            times_opt_mlp.append((t1 - t0) / 1000.0)
        print_row("Optimized: copy+Graph+sync+D2H", times_opt_mlp)
    print()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 80)
    print("SUMMARY — PyTorch component breakdown")
    print("=" * 80)
    print()
    s_h2d = stats(times_h2d)
    s_d2h = stats(times_d2h)
    s_eager = stats(times_eager)
    s_graph = stats(times_graph)
    print(f"  Component                            Median µs")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  H2D transfer (1-elem)                {s_h2d['median']:8.1f}")
    print(f"  Eager dispatch+sync (1-elem)         {s_eager['median']:8.1f}")
    print(f"  CUDA Graph dispatch+sync (1-elem)    {s_graph['median']:8.1f}")
    print(f"  D2H readback (1-elem)                {s_d2h['median']:8.1f}")
    print()
    print(f"  Eager MLP (GPU-resident)             {stats(times_eager_mlp)['median']:8.1f}")
    print(f"  Graph MLP (GPU-resident)             {stats(times_graph_mlp)['median']:8.1f}")
    print(f"  Naive MLP (H2D+infer+D2H)           {stats(times_naive_mlp)['median']:8.1f}")
    print(f"  Optimized MLP (copy+Graph+D2H)       {stats(times_opt_mlp)['median']:8.1f}")
    print()

    return {
        'eager_launch': s_eager, 'graph_launch': s_graph,
        'h2d': s_h2d, 'd2h': s_d2h,
        'eager_mlp': stats(times_eager_mlp), 'graph_mlp': stats(times_graph_mlp),
        'naive_mlp': stats(times_naive_mlp), 'opt_mlp': stats(times_opt_mlp),
    }


def main():
    ap = argparse.ArgumentParser(description="Component breakdown benchmark")
    ap.add_argument('--backend', choices=['tinygrad', 'pytorch'], required=True)
    args = ap.parse_args()

    if args.backend == 'tinygrad':
        run_tinygrad()
    else:
        run_pytorch()

if __name__ == '__main__':
    main()
