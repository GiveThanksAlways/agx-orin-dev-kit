#!/usr/bin/env python3
"""
bench_hot_path.py — Benchmark: Python NV=1 vs C GPU hot path vs C NEON MLP.

Tests 7 MLP sizes to find the GPU/NEON crossover and sweet spots:
  Tiny:    12→64→64→4         (~5K params)    — PID-replacement, rate gyro filter
  Small:   12→128→128→4       (~18K params)   — learned hover controller, sensor fusion
  Medium:  12→256→256→256→4   (~135K params)  — full attitude policy, SLAM feature encoder
  Large:   12→512→512→4       (~268K params)  — visual-inertial nav, end-to-end landing
  XLarge:  12→512→512→512→4   (~530K params)  — GPU crossover zone
  XXLarge: 12→1024→1024→4     (~1.1M params)  — GPU sweet spot: path planner, obstacle avoidance
  Huge:    12→1024→1024→1024→4 (~2.1M params) — large policy, multi-agent coordination

For each size, runs:
  1. Python NV=1  — tinygrad @TinyJit with direct memmove (Approach C from bench_nv_wins.py)
  2. C GPU hot path — same GPU commands, replayed from C via raw MMIO doorbell
  3. C NEON MLP    — pure ARM NEON FP16 inference, no GPU

Usage (from repo root):
  cd examples/tinygrad && nix develop
  NV=1 JITBEAM=2 python3 ../../examples/control-loop/hot_path/bench_hot_path.py
"""
import os, sys, time, ctypes, struct
import numpy as np

# ── Ensure NV=1 is set ──
os.environ.setdefault("NV", "1")

from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad import nn as tg_nn

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOT_PATH_SO = os.path.join(SCRIPT_DIR, "hot_path.so")
NEON_MLP_SO = os.path.join(SCRIPT_DIR, "neon_mlp.so")

# Add hot_path dir to Python path for export_graph
sys.path.insert(0, SCRIPT_DIR)

# ── Config ──
SEED    = 42
IN_DIM  = 12       # pos(3) + vel(3) + rpy(3) + gyro(3)
OUT_DIM = 4        # thrust + roll_rate + pitch_rate + yaw_rate
WARMUP  = 30
N_ITERS = 10000

# MLP configs: (name, hidden_dims, description)
# Spans from tiny PID-replacement up through large policy networks to find
# the GPU sweet spot (where C GPU hot path beats NEON) and upper bound.
MLP_CONFIGS = [
    ("tiny",    [64, 64],              "12→64→64→4 (~5K params)"),
    ("small",   [128, 128],            "12→128→128→4 (~18K params)"),
    ("medium",  [256, 256, 256],       "12→256→256→256→4 (~135K params)"),
    ("large",   [512, 512],            "12→512→512→4 (~268K params)"),
    ("xlarge",  [512, 512, 512],       "12→512→512→512→4 (~530K params)"),
    ("xxlarge", [1024, 1024],          "12→1024→1024→4 (~1.1M params)"),
    ("huge",    [1024, 1024, 1024],    "12→1024→1024→1024→4 (~2.1M params)"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Model & data generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_weights(hidden_dims, seed=SEED):
    """Generate deterministic FP16 weights for an MLP with given hidden dims."""
    rng = np.random.RandomState(seed)
    layers = []
    dims = [IN_DIM] + hidden_dims + [OUT_DIM]
    for i in range(len(dims) - 1):
        fi, fo = dims[i], dims[i+1]
        s = 1.0 / np.sqrt(fi)
        w = (rng.randn(fo, fi) * s).astype(np.float16)
        b = (rng.randn(fo) * s).astype(np.float16)
        layers.append((w, b))
    return layers, dims


class ControlMLP:
    """Variable-depth MLP in tinygrad."""
    def __init__(self, layers_data):
        self.layers = []
        for w, b in layers_data:
            fo, fi = w.shape
            lin = tg_nn.Linear(fi, fo)
            lin.weight = Tensor(w)
            lin.bias = Tensor(b)
            self.layers.append(lin)

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.relu()
        return x


def generate_sensor_pool(n, seed=123):
    """Generate n FP16 state vectors [1, IN_DIM] as a contiguous array."""
    rng = np.random.RandomState(seed)
    # Realistic IMU-like data: accel ≈ [0,0,9.81], gyro ≈ 0, pos/vel/rpy ≈ 0
    pool = rng.randn(n, IN_DIM).astype(np.float16) * 0.1
    return pool


# ═══════════════════════════════════════════════════════════════════════════════
# Stats & printing
# ═══════════════════════════════════════════════════════════════════════════════

def stats(data):
    a = np.asarray(data, dtype=np.float64)
    return {
        'mean': np.mean(a), 'median': np.median(a), 'std': np.std(a),
        'min': np.min(a), 'max': np.max(a),
        'p99': np.percentile(a, 99), 'p999': np.percentile(a, 99.9),
    }


def print_row(label, s):
    freq = 1e6 / s['median'] if s['median'] > 0 else 0
    print(f"  {label:40s}  median={s['median']:8.1f} µs  "
          f"mean={s['mean']:8.1f} µs  std={s['std']:6.1f}  "
          f"p99={s['p99']:8.1f} µs  freq={freq:8.0f} Hz")


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark: Python NV=1 (direct memmove, from bench_nv_wins.py Approach C)
# ═══════════════════════════════════════════════════════════════════════════════

def bench_python_nv1(layers_data, sensor_pool):
    """Benchmark Python NV=1 with direct memmove via cpu_view."""
    dev = Device["NV"]

    model = ControlMLP(layers_data)
    static_x = Tensor.zeros(1, IN_DIM).cast(dtypes.float16).contiguous().realize()
    static_o = Tensor.zeros(1, OUT_DIM).cast(dtypes.float16).contiguous().realize()
    dev.synchronize()

    x_hcq = static_x._buffer()._buf
    o_hcq = static_o._buffer()._buf
    in_addr = x_hcq.cpu_view().addr
    out_addr = o_hcq.cpu_view().addr
    in_nbytes = IN_DIM * 2
    out_nbytes = OUT_DIM * 2

    @TinyJit
    def run():
        static_o.assign(model(static_x)).realize()

    # Warmup (captures JIT + BEAM optimizes)
    for i in range(WARMUP):
        ctypes.memmove(in_addr, sensor_pool[i].ctypes.data, in_nbytes)
        run()
        dev.synchronize()

    # Benchmark
    result_np = np.empty(OUT_DIM, dtype=np.float16)
    times_us = []
    for i in range(N_ITERS):
        t0 = time.perf_counter_ns()
        ctypes.memmove(in_addr, sensor_pool[WARMUP + i].ctypes.data, in_nbytes)
        run()
        dev.synchronize()
        ctypes.memmove(result_np.ctypes.data, out_addr, out_nbytes)
        t1 = time.perf_counter_ns()
        times_us.append((t1 - t0) / 1000.0)

    # Get reference output for correctness check
    ctypes.memmove(in_addr, sensor_pool[0].ctypes.data, in_nbytes)
    run()
    dev.synchronize()
    ref = np.empty(OUT_DIM, dtype=np.float16)
    ctypes.memmove(ref.ctypes.data, out_addr, out_nbytes)

    return stats(times_us), ref, run, static_x, static_o


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark: C GPU Hot Path
# ═══════════════════════════════════════════════════════════════════════════════

def _build_config_struct(cfg):
    """Convert export dict → ctypes struct matching hot_path_config_t."""

    class PatchEntry(ctypes.Structure):
        _fields_ = [
            ("addr",     ctypes.c_uint64),
            ("var_type", ctypes.c_uint32),
            ("mask",     ctypes.c_uint32),
        ]

    MAX_PATCHES = 512
    MAX_QUEUE_SIGNALS = 16

    class HotPathConfig(ctypes.Structure):
        _fields_ = [
            ("input_buf_addr",       ctypes.c_uint64),
            ("output_buf_addr",      ctypes.c_uint64),
            ("input_size",           ctypes.c_uint32),
            ("output_size",          ctypes.c_uint32),

            ("gpfifo_ring_addr",     ctypes.c_uint64),
            ("gpfifo_gpput_addr",    ctypes.c_uint64),
            ("gpfifo_entries_count", ctypes.c_uint32),
            ("gpfifo_token",         ctypes.c_uint32),
            ("gpfifo_put_value",     ctypes.c_uint32),

            ("cmdq_gpu_addr",        ctypes.c_uint64),
            ("cmdq_len_u32",         ctypes.c_uint32),

            ("gpu_mmio_addr",        ctypes.c_uint64),

            ("timeline_signal_addr", ctypes.c_uint64),
            ("timeline_value",       ctypes.c_uint32),
            ("last_tl_value",        ctypes.c_uint32),

            ("kick_signal_addr",     ctypes.c_uint64),
            ("kickoff_value",        ctypes.c_uint32),

            ("num_queue_signals",    ctypes.c_uint32),
            ("queue_signal_addrs",   ctypes.c_uint64 * MAX_QUEUE_SIGNALS),

            ("num_patches",          ctypes.c_uint32),
            ("patches",              PatchEntry * MAX_PATCHES),

            ("gpfifo_entry",         ctypes.c_uint64),
        ]

    c = HotPathConfig()
    c.input_buf_addr       = cfg['input_buf_addr']
    c.output_buf_addr      = cfg['output_buf_addr']
    c.input_size           = cfg['input_size']
    c.output_size          = cfg['output_size']

    c.gpfifo_ring_addr     = cfg['gpfifo_ring_addr']
    c.gpfifo_gpput_addr    = cfg['gpfifo_gpput_addr']
    c.gpfifo_entries_count = cfg['gpfifo_entries_count']
    c.gpfifo_token         = cfg['gpfifo_token']
    c.gpfifo_put_value     = cfg['gpfifo_put_value']

    c.cmdq_gpu_addr        = cfg['cmdq_gpu_addr']
    c.cmdq_len_u32         = cfg['cmdq_len_u32']

    c.gpu_mmio_addr        = cfg['gpu_mmio_addr']

    c.timeline_signal_addr = cfg['timeline_signal_addr']
    c.timeline_value       = cfg['timeline_value']
    c.last_tl_value        = cfg['last_tl_value']

    c.kick_signal_addr     = cfg['kick_signal_addr']
    c.kickoff_value        = cfg['kickoff_value']

    sigs = cfg['queue_signal_addrs']
    c.num_queue_signals = len(sigs)
    for i, addr in enumerate(sigs):
        c.queue_signal_addrs[i] = addr

    patches = cfg['patches']
    c.num_patches = len(patches)
    for i, p in enumerate(patches):
        c.patches[i].addr     = p['addr']
        c.patches[i].var_type = p['var_type']
        c.patches[i].mask     = p['mask']

    return c


def bench_c_gpu(cfg_dict, sensor_pool, ref_output, dev):
    """Benchmark C GPU hot path dispatch."""
    if not os.path.exists(HOT_PATH_SO):
        print(f"  SKIP: {HOT_PATH_SO} not found (run 'make' first)")
        return None

    lib = ctypes.CDLL(HOT_PATH_SO)
    lib.hot_path_init.argtypes = [ctypes.c_void_p]
    lib.hot_path_init.restype = None
    lib.hot_path_benchmark.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.hot_path_benchmark.restype = None

    cfg_struct = _build_config_struct(cfg_dict)
    lib.hot_path_init(ctypes.byref(cfg_struct))

    # Prepare sensor and action pools as numpy arrays for ctypes compatibility
    in_size = cfg_dict['input_size']
    out_size = cfg_dict['output_size']

    sensor_np = np.ascontiguousarray(sensor_pool[WARMUP:WARMUP+N_ITERS].reshape(-1))
    action_np = np.zeros(N_ITERS * out_size // 2, dtype=np.float16)
    times = (ctypes.c_uint64 * N_ITERS)()

    # Warmup (5 iterations)
    warmup_sensor = np.ascontiguousarray(sensor_pool[:5].reshape(-1))
    warmup_action = np.zeros(5 * out_size // 2, dtype=np.float16)
    warmup_times = (ctypes.c_uint64 * 5)()
    lib.hot_path_benchmark(
        ctypes.byref(cfg_struct),
        warmup_sensor.ctypes.data,
        warmup_action.ctypes.data,
        5,
        warmup_times,
    )

    # Benchmark
    lib.hot_path_benchmark(
        ctypes.byref(cfg_struct),
        sensor_np.ctypes.data,
        action_np.ctypes.data,
        N_ITERS,
        times,
    )

    times_us = [times[i] / 1000.0 for i in range(N_ITERS)]

    # Verify correctness: run one iteration with the reference input
    ref_sensor = np.ascontiguousarray(sensor_pool[0:1].reshape(-1))
    ref_action = np.zeros(out_size // 2, dtype=np.float16)
    ref_time = (ctypes.c_uint64 * 1)()
    lib.hot_path_benchmark(
        ctypes.byref(cfg_struct),
        ref_sensor.ctypes.data,
        ref_action.ctypes.data,
        1,
        ref_time,
    )
    if not np.allclose(ref_action, ref_output, atol=0.1, rtol=0.15):
        print(f"  WARNING: C GPU output mismatch! C={ref_action}, ref={ref_output}")
    else:
        print(f"  Correctness: PASS (C GPU matches Python NV=1)")

    # Sync Python device state with C code's final values to prevent desync
    dev.compute_gpfifo.put_value = cfg_struct.gpfifo_put_value
    dev.timeline_value = cfg_struct.timeline_value
    dev.timeline_signal.wait(cfg_struct.last_tl_value)

    return stats(times_us)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark: C NEON MLP
# ═══════════════════════════════════════════════════════════════════════════════

def bench_c_neon(layers_data, dims, sensor_pool, ref_output):
    """Benchmark ARM NEON FP16 MLP forward pass."""
    if not os.path.exists(NEON_MLP_SO):
        print(f"  SKIP: {NEON_MLP_SO} not found (run 'make' first)")
        return None

    lib = ctypes.CDLL(NEON_MLP_SO)

    # Define neon_mlp_t as opaque blob (we'll allocate it large enough)
    # With MAX_DIM=2048: scratch_a/b = 2*2048*2 = 8192 bytes, plus pointers/dims
    MLP_STRUCT_SIZE = 65536  # generous for MAX_DIM=2048

    lib.neon_mlp_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.neon_mlp_init.restype = ctypes.c_int

    lib.neon_mlp_load_layer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    lib.neon_mlp_load_layer.restype = None

    lib.neon_mlp_benchmark.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.neon_mlp_benchmark.restype = None

    lib.neon_mlp_free.argtypes = [ctypes.c_void_p]
    lib.neon_mlp_free.restype = None

    # Allocate and init
    mlp_buf = (ctypes.c_uint8 * MLP_STRUCT_SIZE)()
    n_layers = len(layers_data)
    c_dims = (ctypes.c_int * len(dims))(*dims)

    ret = lib.neon_mlp_init(mlp_buf, n_layers, c_dims)
    if ret != 0:
        print(f"  SKIP: neon_mlp_init failed (ret={ret})")
        return None

    # Load weights
    for l, (w, b) in enumerate(layers_data):
        w_contig = np.ascontiguousarray(w)
        b_contig = np.ascontiguousarray(b)
        lib.neon_mlp_load_layer(mlp_buf, l, w_contig.ctypes.data, b_contig.ctypes.data)

    # Prepare pools
    input_pool = np.ascontiguousarray(sensor_pool[WARMUP:WARMUP+N_ITERS].reshape(-1))
    output_pool = np.zeros(N_ITERS * OUT_DIM, dtype=np.float16)
    times = (ctypes.c_uint64 * N_ITERS)()

    # Warmup
    warmup_in = np.ascontiguousarray(sensor_pool[:5].reshape(-1))
    warmup_out = np.zeros(5 * OUT_DIM, dtype=np.float16)
    warmup_times = (ctypes.c_uint64 * 5)()
    lib.neon_mlp_benchmark(mlp_buf, warmup_in.ctypes.data, warmup_out.ctypes.data, 5, warmup_times)

    # Benchmark
    lib.neon_mlp_benchmark(mlp_buf, input_pool.ctypes.data, output_pool.ctypes.data, N_ITERS, times)

    times_us = [times[i] / 1000.0 for i in range(N_ITERS)]

    # Verify correctness
    ref_in = np.ascontiguousarray(sensor_pool[0:1].reshape(-1))
    ref_out = np.zeros(OUT_DIM, dtype=np.float16)
    ref_time = (ctypes.c_uint64 * 1)()
    lib.neon_mlp_benchmark(mlp_buf, ref_in.ctypes.data, ref_out.ctypes.data, 1, ref_time)

    # NEON and tinygrad may differ slightly due to different FP16 accumulation order
    if not np.allclose(ref_out, ref_output, atol=0.5, rtol=0.3):
        print(f"  WARNING: NEON output differs from GPU: NEON={ref_out}, GPU={ref_output}")
        print(f"  (FP16 accumulation order differences are expected)")
    else:
        print(f"  Correctness: PASS (NEON matches GPU within tolerance)")

    lib.neon_mlp_free(mlp_buf)
    return stats(times_us)


# ═══════════════════════════════════════════════════════════════════════════════
# Main benchmark driver
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    assert Device.DEFAULT == "NV", f"Device is '{Device.DEFAULT}', expected 'NV'. Set NV=1."
    dev = Device["NV"]

    print("=" * 95)
    print("Production Hot Path Benchmark: Python NV=1 vs C GPU vs C NEON")
    print(f"Device: {Device.DEFAULT}  |  JITBEAM={os.environ.get('JITBEAM', 'default')}  |  {N_ITERS} iterations")
    print("=" * 95)
    print()

    sensor_pool = generate_sensor_pool(N_ITERS + WARMUP + 100)
    all_results = {}

    for name, hidden_dims, desc in MLP_CONFIGS:
        print(f"{'─' * 95}")
        print(f"  MLP: {desc}")
        layers_data, dims = generate_weights(hidden_dims)
        n_params = sum(w.size + b.size for w, b in layers_data)
        print(f"  Parameters: {n_params:,}")
        print(f"{'─' * 95}")

        # 1. Python NV=1 benchmark
        print(f"\n  [1/3] Python NV=1 (direct memmove + @TinyJit + HCQGraph)...")
        py_stats, ref_output, jit_fn, static_x, static_o = bench_python_nv1(layers_data, sensor_pool)
        print_row("Python NV=1", py_stats)

        # 2. C GPU hot path
        print(f"\n  [2/3] C GPU hot path (raw MMIO doorbell dispatch)...")
        try:
            from export_graph import export_hot_path_config
            x_buf = static_x._buffer()
            o_buf = static_o._buffer()
            cfg_dict = export_hot_path_config(jit_fn, dev, x_buf, o_buf)
            c_gpu_stats = bench_c_gpu(cfg_dict, sensor_pool, ref_output, dev)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            c_gpu_stats = None

        if c_gpu_stats:
            print_row("C GPU hot path", c_gpu_stats)

        # 3. C NEON MLP
        print(f"\n  [3/3] C NEON MLP (ARM FP16 NEON intrinsics)...")
        neon_stats = bench_c_neon(layers_data, dims, sensor_pool, ref_output)
        if neon_stats:
            print_row("C NEON MLP", neon_stats)

        all_results[name] = {
            'desc': desc,
            'params': n_params,
            'python_nv1': py_stats,
            'c_gpu': c_gpu_stats,
            'c_neon': neon_stats,
        }

        # Clean up tinygrad state for next model
        del jit_fn, static_x, static_o
        dev.synchronize()
        print()

    # ── Summary table ──
    print("\n" + "=" * 95)
    print("SUMMARY: Median Latency (µs) and Frequency (Hz)")
    print("=" * 95)
    print(f"  {'Model':<35s} {'Python NV=1':>14s} {'C GPU':>14s} {'C NEON':>14s} "
          f"{'GPU speedup':>12s} {'GPU vs NEON':>12s}")
    print(f"  {'─'*35} {'─'*14} {'─'*14} {'─'*14} {'─'*12} {'─'*12}")

    for name, r in all_results.items():
        py_med = r['python_nv1']['median']
        py_freq = 1e6 / py_med if py_med > 0 else 0

        c_gpu_str = "N/A"
        speedup_str = "N/A"
        vs_neon_str = "N/A"
        if r['c_gpu']:
            c_med = r['c_gpu']['median']
            c_freq = 1e6 / c_med if c_med > 0 else 0
            c_gpu_str = f"{c_med:6.1f} / {c_freq:5.0f}"
            speedup_str = f"{py_med/c_med:.1f}x"

        neon_str = "N/A"
        if r['c_neon']:
            n_med = r['c_neon']['median']
            n_freq = 1e6 / n_med if n_med > 0 else 0
            neon_str = f"{n_med:6.1f} / {n_freq:5.0f}"
            if r['c_gpu']:
                ratio = n_med / r['c_gpu']['median']
                vs_neon_str = f"{'GPU' if ratio > 1 else 'NEON'} {max(ratio, 1/ratio):.1f}x"

        print(f"  {r['desc']:<35s} {py_med:6.1f} / {py_freq:5.0f}  "
              f"{c_gpu_str:>14s}  {neon_str:>14s}  {speedup_str:>12s}  {vs_neon_str:>12s}")

    print()
    print("Format: latency_µs / freq_Hz")
    print("GPU speedup = Python NV=1 median / C GPU median")
    print("GPU vs NEON = which is faster and by how much")
    print()


if __name__ == '__main__':
    main()
