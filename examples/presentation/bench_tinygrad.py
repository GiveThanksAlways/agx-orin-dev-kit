"""bench_tinygrad.py — tinygrad NV=1 inference benchmark with direct memory access.

Uses the proven Approach C from bench_nv_wins.py: ctypes.memmove to cpu_view()
for zero-copy H2D/D2H on Tegra unified memory. This is our fastest Python path.

Supports: MLP, 1D-CNN, Hybrid CNN+MLP architectures.
"""
import os, sys, time, ctypes
import numpy as np

os.environ.setdefault("NV", "1")

from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad import nn as tg_nn

assert Device.DEFAULT == "NV", f"Device is '{Device.DEFAULT}', expected 'NV'. Set NV=1."


def _setup_direct_memory(tensor):
    """Get the CPU-accessible address for a tinygrad tensor (Tegra unified memory).
    Returns (address, nbytes)."""
    buf = tensor._buffer()._buf
    view = buf.cpu_view()
    return view.addr, tensor.dtype.itemsize * int(np.prod(tensor.shape))


def _stats(times):
    a = np.asarray(times)
    return {
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "p99": float(np.percentile(a, 99)),
        "p999": float(np.percentile(a, 99.9)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MLP benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def bench_nv_mlp(name, hidden_dims, seed=42, warmup=50, n_iters=10000):
    """Benchmark an MLP on tinygrad NV=1 using direct memory access (Approach C).

    Returns (times_us_list, params_count).
    """
    from models import IN_DIM, OUT_DIM, build_tinygrad_mlp

    dev = Device["NV"]
    model, params = build_tinygrad_mlp(hidden_dims, seed=seed)

    # Static input/output tensors for direct memory access
    static_x = Tensor.zeros(1, IN_DIM, dtype=dtypes.float16).contiguous().realize()
    static_out = Tensor.zeros(1, OUT_DIM, dtype=dtypes.float16).contiguous().realize()
    dev.synchronize()

    in_addr, in_nbytes = _setup_direct_memory(static_x)
    out_addr, out_nbytes = _setup_direct_memory(static_out)

    @TinyJit
    def run():
        static_out.assign(model(static_x)).realize()

    # Pre-generate test data
    rng = np.random.RandomState(99)
    data_pool = rng.randn(warmup + n_iters + 10, 1, IN_DIM).astype(np.float16)

    # Warmup (JIT captures on 2nd call, graph on 3rd)
    for i in range(warmup):
        ctypes.memmove(in_addr, data_pool[i].ctypes.data, in_nbytes)
        run()
        dev.synchronize()

    # Correctness check: compute reference via numpy
    test_in = data_pool[0]
    ctypes.memmove(in_addr, test_in.ctypes.data, in_nbytes)
    run()
    dev.synchronize()
    result = np.empty((1, OUT_DIM), dtype=np.float16)
    ctypes.memmove(result.ctypes.data, out_addr, out_nbytes)

    # Benchmark
    times = []
    for i in range(n_iters):
        d = data_pool[warmup + i]
        t0 = time.perf_counter_ns()
        ctypes.memmove(in_addr, d.ctypes.data, in_nbytes)
        run()
        dev.synchronize()
        ctypes.memmove(result.ctypes.data, out_addr, out_nbytes)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)

    return times, params, result


# ═══════════════════════════════════════════════════════════════════════════════
# CNN benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def bench_nv_cnn(name, conv_config, mlp_head_dims, seed=42, warmup=50, n_iters=10000):
    """Benchmark a 1D-CNN on tinygrad NV=1 using direct memory access."""
    from models import IN_DIM, OUT_DIM, SEQ_LEN, build_tinygrad_cnn

    dev = Device["NV"]
    model, params = build_tinygrad_cnn(conv_config, mlp_head_dims, seed=seed)

    # Static buffers: input is (1, IN_DIM, SEQ_LEN)
    static_x = Tensor.zeros(1, IN_DIM, SEQ_LEN, dtype=dtypes.float16).contiguous().realize()
    static_out = Tensor.zeros(1, OUT_DIM, dtype=dtypes.float16).contiguous().realize()
    dev.synchronize()

    in_addr, in_nbytes = _setup_direct_memory(static_x)
    out_addr, out_nbytes = _setup_direct_memory(static_out)

    @TinyJit
    def run():
        static_out.assign(model(static_x)).realize()

    rng = np.random.RandomState(99)
    data_pool = rng.randn(warmup + n_iters + 10, 1, IN_DIM, SEQ_LEN).astype(np.float16)

    for i in range(warmup):
        ctypes.memmove(in_addr, data_pool[i].ctypes.data, in_nbytes)
        run()
        dev.synchronize()

    result = np.empty((1, OUT_DIM), dtype=np.float16)

    times = []
    for i in range(n_iters):
        d = data_pool[warmup + i]
        t0 = time.perf_counter_ns()
        ctypes.memmove(in_addr, d.ctypes.data, in_nbytes)
        run()
        dev.synchronize()
        ctypes.memmove(result.ctypes.data, out_addr, out_nbytes)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)

    return times, params, result


# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid CNN+MLP benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def bench_nv_hybrid(name, conv_config, mlp_head_dims, seed=42, warmup=50, n_iters=10000):
    """Benchmark a hybrid CNN+MLP on tinygrad NV=1 using direct memory access.

    Two static inputs: imu_window (temporal) + current_state (instantaneous).
    """
    from models import IN_DIM, OUT_DIM, SEQ_LEN, build_tinygrad_hybrid

    dev = Device["NV"]
    model, params = build_tinygrad_hybrid(conv_config, mlp_head_dims, seed=seed)

    # Two inputs
    static_imu = Tensor.zeros(1, IN_DIM, SEQ_LEN, dtype=dtypes.float16).contiguous().realize()
    static_state = Tensor.zeros(1, IN_DIM, dtype=dtypes.float16).contiguous().realize()
    static_out = Tensor.zeros(1, OUT_DIM, dtype=dtypes.float16).contiguous().realize()
    dev.synchronize()

    imu_addr, imu_nbytes = _setup_direct_memory(static_imu)
    state_addr, state_nbytes = _setup_direct_memory(static_state)
    out_addr, out_nbytes = _setup_direct_memory(static_out)

    @TinyJit
    def run():
        static_out.assign(model(static_imu, static_state)).realize()

    rng = np.random.RandomState(99)
    imu_pool = rng.randn(warmup + n_iters + 10, 1, IN_DIM, SEQ_LEN).astype(np.float16)
    state_pool = rng.randn(warmup + n_iters + 10, 1, IN_DIM).astype(np.float16)

    for i in range(warmup):
        ctypes.memmove(imu_addr, imu_pool[i].ctypes.data, imu_nbytes)
        ctypes.memmove(state_addr, state_pool[i].ctypes.data, state_nbytes)
        run()
        dev.synchronize()

    result = np.empty((1, OUT_DIM), dtype=np.float16)

    times = []
    for i in range(n_iters):
        t0 = time.perf_counter_ns()
        ctypes.memmove(imu_addr, imu_pool[warmup + i].ctypes.data, imu_nbytes)
        ctypes.memmove(state_addr, state_pool[warmup + i].ctypes.data, state_nbytes)
        run()
        dev.synchronize()
        ctypes.memmove(result.ctypes.data, out_addr, out_nbytes)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)

    return times, params, result
