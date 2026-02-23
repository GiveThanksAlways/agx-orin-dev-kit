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
NEON_MLP_F32_SO = os.path.join(SCRIPT_DIR, "neon_mlp_f32.so")

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
    """Generate deterministic FP16+FP32 weights for an MLP.
    Returns (layers_f16, layers_f32, dims). FP32 weights are the originals;
    FP16 weights are cast from FP32 (so FP16 error includes quantization)."""
    rng = np.random.RandomState(seed)
    layers_f16, layers_f32 = [], []
    dims = [IN_DIM] + hidden_dims + [OUT_DIM]
    for i in range(len(dims) - 1):
        fi, fo = dims[i], dims[i+1]
        s = 1.0 / np.sqrt(fi)
        w_f32 = (rng.randn(fo, fi) * s).astype(np.float32)
        b_f32 = (rng.randn(fo) * s).astype(np.float32)
        layers_f32.append((w_f32, b_f32))
        layers_f16.append((w_f32.astype(np.float16), b_f32.astype(np.float16)))
    return layers_f16, layers_f32, dims


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


def compute_fp64_reference(layers_f32, x_f16):
    """FP64 ground truth: high-precision forward pass using FP32 weights.
    The input is FP16 sensor data cast to FP64 for maximum precision."""
    h = x_f16.astype(np.float64).reshape(1, -1)
    for i, (w, b) in enumerate(layers_f32):
        h = h @ w.astype(np.float64).T + b.astype(np.float64)
        if i < len(layers_f32) - 1:
            h = np.maximum(h, 0)
    return h.flatten()


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
    print(f"  {label:40s}  med={s['median']:7.1f} µs  "
          f"p99={s['p99']:7.1f}  p99.9={s['p999']:7.1f}  "
          f"max={s['max']:7.1f} µs  {freq:7.0f} Hz")


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
        print(f"  Correctness: PASS (NEON FP16 matches GPU within tolerance)")

    lib.neon_mlp_free(mlp_buf)
    return stats(times_us), ref_out


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark: C NEON FP32 MLP
# ═══════════════════════════════════════════════════════════════════════════════

def bench_c_neon_f32(layers_f32, dims, sensor_pool_f32):
    """Benchmark ARM NEON FP32 MLP forward pass.
    Uses float32x4_t (4 elements per vector) — half the SIMD throughput of FP16
    but with full single-precision accuracy. This represents the common baseline
    where developers don't quantize to FP16."""
    if not os.path.exists(NEON_MLP_F32_SO):
        print(f"  SKIP: {NEON_MLP_F32_SO} not found (run 'make' first)")
        return None

    lib = ctypes.CDLL(NEON_MLP_F32_SO)

    MLP_STRUCT_SIZE = 65536

    lib.neon_mlp_f32_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.neon_mlp_f32_init.restype = ctypes.c_int

    lib.neon_mlp_f32_load_layer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    lib.neon_mlp_f32_load_layer.restype = None

    lib.neon_mlp_f32_benchmark.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.neon_mlp_f32_benchmark.restype = None

    lib.neon_mlp_f32_free.argtypes = [ctypes.c_void_p]
    lib.neon_mlp_f32_free.restype = None

    # Allocate and init
    mlp_buf = (ctypes.c_uint8 * MLP_STRUCT_SIZE)()
    n_layers = len(layers_f32)
    c_dims = (ctypes.c_int * len(dims))(*dims)

    ret = lib.neon_mlp_f32_init(mlp_buf, n_layers, c_dims)
    if ret != 0:
        print(f"  SKIP: neon_mlp_f32_init failed (ret={ret})")
        return None

    # Load FP32 weights
    for l, (w, b) in enumerate(layers_f32):
        w_contig = np.ascontiguousarray(w)
        b_contig = np.ascontiguousarray(b)
        lib.neon_mlp_f32_load_layer(mlp_buf, l, w_contig.ctypes.data, b_contig.ctypes.data)

    # Prepare pools (FP32)
    input_pool = np.ascontiguousarray(sensor_pool_f32[WARMUP:WARMUP+N_ITERS].reshape(-1))
    output_pool = np.zeros(N_ITERS * OUT_DIM, dtype=np.float32)
    times = (ctypes.c_uint64 * N_ITERS)()

    # Warmup
    warmup_in = np.ascontiguousarray(sensor_pool_f32[:5].reshape(-1))
    warmup_out = np.zeros(5 * OUT_DIM, dtype=np.float32)
    warmup_times = (ctypes.c_uint64 * 5)()
    lib.neon_mlp_f32_benchmark(mlp_buf, warmup_in.ctypes.data, warmup_out.ctypes.data, 5, warmup_times)

    # Benchmark
    lib.neon_mlp_f32_benchmark(mlp_buf, input_pool.ctypes.data, output_pool.ctypes.data, N_ITERS, times)

    times_us = [times[i] / 1000.0 for i in range(N_ITERS)]

    # Get reference output for accuracy
    ref_in = np.ascontiguousarray(sensor_pool_f32[0:1].reshape(-1))
    ref_out = np.zeros(OUT_DIM, dtype=np.float32)
    ref_time = (ctypes.c_uint64 * 1)()
    lib.neon_mlp_f32_benchmark(mlp_buf, ref_in.ctypes.data, ref_out.ctypes.data, 1, ref_time)

    print(f"  Correctness: computed (accuracy vs FP64 in summary)")

    lib.neon_mlp_f32_free(mlp_buf)
    return stats(times_us), ref_out


# ═══════════════════════════════════════════════════════════════════════════════
# Main benchmark driver
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    assert Device.DEFAULT == "NV", f"Device is '{Device.DEFAULT}', expected 'NV'. Set NV=1."
    dev = Device["NV"]

    print("=" * 105)
    print("Production Hot Path CRITICAL Benchmark: Python NV=1 vs C GPU vs NEON FP16 vs NEON FP32")
    print(f"Device: {Device.DEFAULT}  |  JITBEAM={os.environ.get('JITBEAM', 'default')}  |  {N_ITERS} iterations")
    print("=" * 105)
    print()

    sensor_pool = generate_sensor_pool(N_ITERS + WARMUP + 100)
    sensor_pool_f32 = sensor_pool.astype(np.float32)
    all_results = {}

    for name, hidden_dims, desc in MLP_CONFIGS:
        print(f"{'─' * 105}")
        print(f"  MLP: {desc}")
        layers_f16, layers_f32, dims = generate_weights(hidden_dims)
        n_params = sum(w.size + b.size for w, b in layers_f16)
        print(f"  Parameters: {n_params:,}")
        print(f"{'─' * 105}")

        # FP64 ground truth for accuracy analysis
        ref_f64 = compute_fp64_reference(layers_f32, sensor_pool[0])

        # 1. Python NV=1 benchmark
        print(f"\n  [1/4] Python NV=1 (direct memmove + @TinyJit + HCQGraph)...")
        py_stats, ref_output, jit_fn, static_x, static_o = bench_python_nv1(layers_f16, sensor_pool)
        print_row("Python NV=1", py_stats)

        # 2. C GPU hot path
        print(f"\n  [2/4] C GPU hot path (raw MMIO doorbell dispatch)...")
        c_gpu_stats = None
        try:
            from export_graph import export_hot_path_config
            x_buf = static_x._buffer()
            o_buf = static_o._buffer()
            cfg_dict = export_hot_path_config(jit_fn, dev, x_buf, o_buf)
            c_gpu_stats = bench_c_gpu(cfg_dict, sensor_pool, ref_output, dev)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

        if c_gpu_stats:
            print_row("C GPU hot path", c_gpu_stats)

        # 3. C NEON FP16
        print(f"\n  [3/4] C NEON FP16 (ARM float16x8_t, 8 elem/vector)...")
        neon_result = bench_c_neon(layers_f16, dims, sensor_pool, ref_output)
        neon_f16_stats = neon_result[0] if neon_result else None
        neon_ref_f16 = neon_result[1] if neon_result else None
        if neon_f16_stats:
            print_row("C NEON FP16", neon_f16_stats)

        # 4. C NEON FP32
        print(f"\n  [4/4] C NEON FP32 (ARM float32x4_t, 4 elem/vector)...")
        neon_f32_result = bench_c_neon_f32(layers_f32, dims, sensor_pool_f32)
        neon_f32_stats = neon_f32_result[0] if neon_f32_result else None
        neon_ref_f32 = neon_f32_result[1] if neon_f32_result else None
        if neon_f32_stats:
            print_row("C NEON FP32", neon_f32_stats)

        # Accuracy analysis vs FP64 reference
        accuracy = {}
        fp16_err = np.abs(ref_output.astype(np.float64) - ref_f64)
        accuracy['gpu_fp16_max'] = float(np.max(fp16_err))
        accuracy['gpu_fp16_mean'] = float(np.mean(fp16_err))
        if neon_ref_f16 is not None:
            nf16_err = np.abs(neon_ref_f16.astype(np.float64) - ref_f64)
            accuracy['neon_f16_max'] = float(np.max(nf16_err))
            accuracy['neon_f16_mean'] = float(np.mean(nf16_err))
        if neon_ref_f32 is not None:
            nf32_err = np.abs(neon_ref_f32.astype(np.float64) - ref_f64)
            accuracy['neon_f32_max'] = float(np.max(nf32_err))
            accuracy['neon_f32_mean'] = float(np.mean(nf32_err))

        def fmt_err(key):
            return f"{accuracy[key]:.6f}" if key in accuracy else "N/A"
        print(f"\n  Accuracy vs FP64: GPU_FP16 max={fmt_err('gpu_fp16_max')}, "
              f"NEON_F16 max={fmt_err('neon_f16_max')}, "
              f"NEON_F32 max={fmt_err('neon_f32_max')}")

        all_results[name] = {
            'desc': desc,
            'params': n_params,
            'python_nv1': py_stats,
            'c_gpu': c_gpu_stats,
            'c_neon_f16': neon_f16_stats,
            'c_neon_f32': neon_f32_stats,
            'accuracy': accuracy,
        }

        # Clean up tinygrad state for next model
        del jit_fn, static_x, static_o
        dev.synchronize()
        print()

    # ── LATENCY SUMMARY ──
    print("\n" + "=" * 115)
    print("LATENCY SUMMARY: Median (µs)")
    print("=" * 115)
    print(f"  {'Model':<22s} {'Params':>8s} {'Py NV=1':>9s} {'C GPU':>9s} "
          f"{'NEON F16':>9s} {'NEON F32':>9s} "
          f"{'GPU vs Py':>10s} {'GPU vs F16':>11s} {'GPU vs F32':>11s}")
    print(f"  {'─'*22} {'─'*8} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*10} {'─'*11} {'─'*11}")

    for name, r in all_results.items():
        py_med = r['python_nv1']['median']
        c_med = r['c_gpu']['median'] if r['c_gpu'] else None
        f16_med = r['c_neon_f16']['median'] if r['c_neon_f16'] else None
        f32_med = r['c_neon_f32']['median'] if r['c_neon_f32'] else None

        gpu_vs_py = f"{py_med/c_med:.1f}x" if c_med else "N/A"

        def ratio_str(a, b, a_name, b_name):
            if a is None or b is None: return "N/A"
            return f"{a_name} {b/a:.1f}x" if a < b else f"{b_name} {a/b:.1f}x"

        gpu_vs_f16 = ratio_str(c_med, f16_med, "GPU", "F16")
        gpu_vs_f32 = ratio_str(c_med, f32_med, "GPU", "F32")

        # Extract short param description
        p = r['params']
        p_str = f"{p/1e6:.1f}M" if p >= 1e6 else f"{p/1e3:.0f}K"

        print(f"  {r['desc'][:22]:<22s} {p_str:>8s} {py_med:9.1f} "
              f"{c_med if c_med else 0:9.1f} "
              f"{f16_med if f16_med else 0:9.1f} "
              f"{f32_med if f32_med else 0:9.1f} "
              f"{gpu_vs_py:>10s} {gpu_vs_f16:>11s} {gpu_vs_f32:>11s}")

    # ── FP16 vs FP32 NEON COMPARISON ──
    print(f"\n{'=' * 115}")
    print("FP16 vs FP32 NEON: How much does FP16 quantization help NEON?")
    print("=" * 115)
    print(f"  {'Model':<22s} {'NEON F16':>10s} {'NEON F32':>10s} {'F16 speedup':>12s} {'Note':>40s}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*12} {'─'*40}")
    for name, r in all_results.items():
        f16_med = r['c_neon_f16']['median'] if r['c_neon_f16'] else None
        f32_med = r['c_neon_f32']['median'] if r['c_neon_f32'] else None
        if f16_med and f32_med:
            speedup = f32_med / f16_med
            note = "FP16 NEON = 8 elem/vec, FP32 = 4 elem/vec" if name == list(all_results.keys())[0] else ""
            print(f"  {r['desc'][:22]:<22s} {f16_med:10.1f} {f32_med:10.1f} {speedup:10.1f}x  {note:>40s}")

    # ── JITTER / DETERMINISM ──
    print(f"\n{'=' * 115}")
    print("JITTER ANALYSIS: P99 / P99.9 / Max (µs) — critical for real-time control")
    print("=" * 115)
    print(f"  {'Model':<22s} {'C GPU  P99/P99.9/Max':>30s} {'NEON F16  P99/P99.9/Max':>30s} {'NEON F32  P99/P99.9/Max':>30s}")
    print(f"  {'─'*22} {'─'*30} {'─'*30} {'─'*30}")
    for name, r in all_results.items():
        def fmt_j(s):
            if s is None: return f"{'N/A':>30s}"
            return f"{s['p99']:8.1f} / {s['p999']:8.1f} / {s['max']:8.1f}"
        print(f"  {r['desc'][:22]:<22s} {fmt_j(r['c_gpu'])} {fmt_j(r['c_neon_f16'])} {fmt_j(r['c_neon_f32'])}")

    # ── ACCURACY ──
    print(f"\n{'=' * 115}")
    print("ACCURACY: Max Absolute Error vs FP64 Reference")
    print("  FP16 error = weight quantization (FP32→FP16) + FP16 accumulation rounding")
    print("  FP32 error = only FP32 accumulation rounding (negligible)")
    print("=" * 115)
    print(f"  {'Model':<22s} {'GPU FP16':>14s} {'NEON FP16':>14s} {'NEON FP32':>14s}")
    print(f"  {'─'*22} {'─'*14} {'─'*14} {'─'*14}")
    for name, r in all_results.items():
        a = r['accuracy']
        def fmt_a(key):
            return f"{a[key]:.6f}" if key in a else "N/A"
        print(f"  {r['desc'][:22]:<22s} {fmt_a('gpu_fp16_max'):>14s} "
              f"{fmt_a('neon_f16_max'):>14s} {fmt_a('neon_f32_max'):>14s}")

    # ── PyTorch COMPARISON ──
    print(f"\n{'=' * 115}")
    print("CONTEXT: C GPU Hot Path vs PyTorch CUDA Graphs (from BENCHMARK_REPORT.md, same Orin hardware)")
    print("=" * 115)
    print("  PyTorch CUDA Graphs (18K MLP, GPU-resident, no memcpy):  88 µs")
    print("  PyTorch eager       (18K MLP, GPU-resident, no memcpy): 402 µs")
    if 'small' in all_results and all_results['small']['c_gpu']:
        c_med = all_results['small']['c_gpu']['median']
        print(f"  C GPU Hot Path      (18K MLP, incl. unified-mem memcpy): {c_med:.1f} µs")
        print(f"  → C GPU Hot Path is {88/c_med:.1f}x faster than PyTorch CUDA Graphs")
        print(f"  → C GPU Hot Path is {402/c_med:.1f}x faster than PyTorch eager")
    print()
    print("  NOTE: PyTorch 'GPU-resident' = data already on GPU, zero H2D/D2H.")
    print("  Our C hot path includes unified-memory memcpy (<0.2 µs for 24B input).")
    print("  This is a near-level comparison. Both measure dispatch + compute + sync.")

    # ── CROSSOVER ANALYSIS ──
    print(f"\n{'=' * 115}")
    print("CROSSOVER: Where does C GPU Hot Path beat each NEON variant?")
    print("=" * 115)
    for label, key in [("NEON FP16", 'c_neon_f16'), ("NEON FP32", 'c_neon_f32')]:
        crossover_found = False
        for name, r in all_results.items():
            c_med = r['c_gpu']['median'] if r['c_gpu'] else None
            n_med = r[key]['median'] if r[key] else None
            if c_med and n_med and c_med < n_med and not crossover_found:
                print(f"  GPU beats {label} starting at ~{r['params']:,} params "
                      f"({r['desc'][:30]}): GPU {n_med/c_med:.1f}x faster")
                crossover_found = True
        if not crossover_found:
            print(f"  GPU never beats {label} in tested range")

    print()
    print("=" * 115)
    print("END OF BENCHMARK")
    print("=" * 115)


if __name__ == '__main__':
    main()
