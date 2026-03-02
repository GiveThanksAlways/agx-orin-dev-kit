"""bench_c_hot_path.py — C GPU hot path benchmark for tinygrad HCQGraph replay.

The C hot path eliminates all Python overhead from the NV=1 dispatch loop:
- No Python interpreter in the hot loop
- No ctypes marshalling per iteration (C loops directly)
- Same GPU commands as tinygrad NV=1, replayed via raw MMIO doorbell

The setup still uses Python to:
  1. Build the tinygrad model + @TinyJit warmup (captures HCQGraph)
  2. Export the graph config (buffer addrs, GPFifo, command queue, patches)
Then C takes over the dispatch loop.

This shows the TRUE GPU overhead floor — the minimum possible latency for
dispatching a tinygrad-compiled graph on Tegra.
"""
import os, sys, time, ctypes
import numpy as np

os.environ.setdefault("NV", "1")

from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad import nn as tg_nn

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOT_PATH_SO = os.path.join(SCRIPT_DIR, "hot_path.so")

# Add control-loop/hot_path to Python path for export_graph
CONTROL_LOOP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "control-loop", "hot_path")
sys.path.insert(0, CONTROL_LOOP_DIR)

MAX_PATCHES = 512
MAX_QUEUE_SIGNALS = 16


def _ensure_hot_path_so():
    """Build hot_path.so if not present."""
    if os.path.exists(HOT_PATH_SO):
        return True
    # Try building directly with CC from environment (set by nix dev shell)
    cc = os.environ.get("CC", "gcc")
    src = os.path.join(CONTROL_LOOP_DIR, "hot_path.c")
    if os.path.exists(src):
        import subprocess
        cmd = [cc, "-O3", "-Wall", "-Wextra", "-Wno-unused-parameter",
               "-fPIC", "-shared", f"-I{CONTROL_LOOP_DIR}", "-o", HOT_PATH_SO, src]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(HOT_PATH_SO):
            print(f"  Built hot_path.so")
            return True
        print(f"  Failed to build hot_path.so: {result.stderr}")
    return False


class _PatchEntry(ctypes.Structure):
    _fields_ = [
        ("addr",     ctypes.c_uint64),
        ("var_type", ctypes.c_uint32),
        ("mask",     ctypes.c_uint32),
    ]


class _HotPathConfig(ctypes.Structure):
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
        ("patches",              _PatchEntry * MAX_PATCHES),
        ("gpfifo_entry",         ctypes.c_uint64),
    ]


def _fill_config_struct(cfg_dict):
    """Convert export_graph dict → ctypes _HotPathConfig."""
    c = _HotPathConfig()
    c.input_buf_addr       = cfg_dict['input_buf_addr']
    c.output_buf_addr      = cfg_dict['output_buf_addr']
    c.input_size           = cfg_dict['input_size']
    c.output_size          = cfg_dict['output_size']
    c.gpfifo_ring_addr     = cfg_dict['gpfifo_ring_addr']
    c.gpfifo_gpput_addr    = cfg_dict['gpfifo_gpput_addr']
    c.gpfifo_entries_count = cfg_dict['gpfifo_entries_count']
    c.gpfifo_token         = cfg_dict['gpfifo_token']
    c.gpfifo_put_value     = cfg_dict['gpfifo_put_value']
    c.cmdq_gpu_addr        = cfg_dict['cmdq_gpu_addr']
    c.cmdq_len_u32         = cfg_dict['cmdq_len_u32']
    c.gpu_mmio_addr        = cfg_dict['gpu_mmio_addr']
    c.timeline_signal_addr = cfg_dict['timeline_signal_addr']
    c.timeline_value       = cfg_dict['timeline_value']
    c.last_tl_value        = cfg_dict['last_tl_value']
    c.kick_signal_addr     = cfg_dict['kick_signal_addr']
    c.kickoff_value        = cfg_dict['kickoff_value']

    sigs = cfg_dict['queue_signal_addrs']
    c.num_queue_signals = len(sigs)
    for i, addr in enumerate(sigs):
        c.queue_signal_addrs[i] = addr

    patches = cfg_dict['patches']
    c.num_patches = len(patches)
    for i, p in enumerate(patches):
        c.patches[i].addr     = p['addr']
        c.patches[i].var_type = p['var_type']
        c.patches[i].mask     = p['mask']

    return c


def _setup_direct_memory(tensor):
    """Get CPU-accessible address for a tinygrad tensor (Tegra unified memory)."""
    buf = tensor._buffer()._buf
    view = buf.cpu_view()
    return view.addr, tensor.dtype.itemsize * int(np.prod(tensor.shape))


def _bench_c_gpu_dispatch(cfg_dict, sensor_pool, n_iters, warmup, dev, use_fp16=True):
    """Run the C hot path benchmark. Returns list of times in µs."""
    np_dtype = np.float16 if use_fp16 else np.float32
    itemsize = 2 if use_fp16 else 4
    lib = ctypes.CDLL(HOT_PATH_SO)
    lib.hot_path_init.argtypes = [ctypes.c_void_p]
    lib.hot_path_init.restype = None
    lib.hot_path_benchmark.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.hot_path_benchmark.restype = None

    cfg_struct = _fill_config_struct(cfg_dict)
    lib.hot_path_init(ctypes.byref(cfg_struct))

    out_size = cfg_dict['output_size']

    # Warmup
    w_sensor = np.ascontiguousarray(sensor_pool[:warmup].reshape(-1))
    w_action = np.zeros(warmup * out_size // itemsize, dtype=np_dtype)
    w_times = (ctypes.c_uint64 * warmup)()
    lib.hot_path_benchmark(ctypes.byref(cfg_struct), w_sensor.ctypes.data,
                           w_action.ctypes.data, warmup, w_times)

    # Benchmark
    b_sensor = np.ascontiguousarray(sensor_pool[warmup:warmup+n_iters].reshape(-1))
    b_action = np.zeros(n_iters * out_size // itemsize, dtype=np_dtype)
    b_times = (ctypes.c_uint64 * n_iters)()
    lib.hot_path_benchmark(ctypes.byref(cfg_struct), b_sensor.ctypes.data,
                           b_action.ctypes.data, n_iters, b_times)

    times_us = [b_times[i] / 1000.0 for i in range(n_iters)]

    # Sync Python device state with C code's final values
    dev.compute_gpfifo.put_value = cfg_struct.gpfifo_put_value
    dev.timeline_value = cfg_struct.timeline_value
    dev.timeline_signal.wait(cfg_struct.last_tl_value)

    return times_us


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — MLP benchmark via C hot path
# ═══════════════════════════════════════════════════════════════════════════════

def bench_c_mlp(name, hidden_dims, seed=42, warmup=50, n_iters=10000, use_fp16=True, batch_size=1):
    """Benchmark an MLP via the C GPU hot path.

    Setup:
      1. Build tinygrad model + @TinyJit warmup → captures HCQGraph
      2. Export graph config (buffer addrs, GPFifo, patches)
      3. C hot_path.so replays the graph via raw MMIO

    Returns (times_us_list, params_count).
    """
    if not _ensure_hot_path_so():
        print(f"  SKIP: hot_path.so not available")
        return None, 0

    from models import IN_DIM, OUT_DIM, build_tinygrad_mlp
    from export_graph import export_hot_path_config

    np_dtype = np.float16 if use_fp16 else np.float32
    tg_dtype = dtypes.float16 if use_fp16 else dtypes.float32
    dev = Device["NV"]
    model, params = build_tinygrad_mlp(hidden_dims, seed=seed, use_fp16=use_fp16, batch_size=batch_size)

    # Static buffers
    static_x = Tensor.zeros(batch_size, IN_DIM, dtype=tg_dtype).contiguous().realize()
    static_out = Tensor.zeros(batch_size, OUT_DIM, dtype=tg_dtype).contiguous().realize()
    dev.synchronize()

    @TinyJit
    def run():
        static_out.assign(model(static_x)).realize()

    # Warmup tinygrad JIT (captures graph)
    rng = np.random.RandomState(99)
    data_pool = rng.randn(warmup + n_iters + 10, batch_size, IN_DIM).astype(np_dtype)
    in_addr, in_nbytes = _setup_direct_memory(static_x)

    for i in range(warmup):
        ctypes.memmove(in_addr, data_pool[i].ctypes.data, in_nbytes)
        run()
        dev.synchronize()

    # Export graph config for C hot path
    x_buf = static_x._buffer()
    o_buf = static_out._buffer()
    cfg_dict = export_hot_path_config(run, dev, x_buf, o_buf)

    # Run C hot path benchmark
    sensor_pool = data_pool[warmup:]  # fresh data for C benchmark
    times = _bench_c_gpu_dispatch(cfg_dict, sensor_pool, n_iters, min(warmup, 30), dev, use_fp16=use_fp16)

    # Clean up
    del run, static_x, static_out
    dev.synchronize()

    return times, params


def bench_c_cnn(name, conv_config, mlp_head_dims, seed=42, warmup=50, n_iters=10000, use_fp16=True, batch_size=1):
    """Benchmark a 1D-CNN via the C GPU hot path."""
    if not _ensure_hot_path_so():
        print(f"  SKIP: hot_path.so not available")
        return None, 0

    from models import IN_DIM, OUT_DIM, SEQ_LEN, build_tinygrad_cnn
    from export_graph import export_hot_path_config

    np_dtype = np.float16 if use_fp16 else np.float32
    tg_dtype = dtypes.float16 if use_fp16 else dtypes.float32
    dev = Device["NV"]
    model, params = build_tinygrad_cnn(conv_config, mlp_head_dims, seed=seed, use_fp16=use_fp16, batch_size=batch_size)

    static_x = Tensor.zeros(batch_size, IN_DIM, SEQ_LEN, dtype=tg_dtype).contiguous().realize()
    static_out = Tensor.zeros(batch_size, OUT_DIM, dtype=tg_dtype).contiguous().realize()
    dev.synchronize()

    @TinyJit
    def run():
        static_out.assign(model(static_x)).realize()

    rng = np.random.RandomState(99)
    data_pool = rng.randn(warmup + n_iters + 10, batch_size, IN_DIM, SEQ_LEN).astype(np_dtype)
    in_addr, in_nbytes = _setup_direct_memory(static_x)

    for i in range(warmup):
        ctypes.memmove(in_addr, data_pool[i].ctypes.data, in_nbytes)
        run()
        dev.synchronize()

    x_buf = static_x._buffer()
    o_buf = static_out._buffer()
    cfg_dict = export_hot_path_config(run, dev, x_buf, o_buf)

    sensor_pool = data_pool[warmup:]
    times = _bench_c_gpu_dispatch(cfg_dict, sensor_pool, n_iters, min(warmup, 30), dev, use_fp16=use_fp16)

    del run, static_x, static_out
    dev.synchronize()

    return times, params


def bench_c_hybrid(name, conv_config, mlp_head_dims, seed=42, warmup=50, n_iters=10000, use_fp16=True, batch_size=1):
    """Benchmark a Hybrid CNN+MLP via the C GPU hot path.

    For hybrid models with two inputs, we concatenate them into a single
    contiguous buffer that the C hot path memcpys in one shot.
    """
    if not _ensure_hot_path_so():
        print(f"  SKIP: hot_path.so not available")
        return None, 0

    from models import IN_DIM, OUT_DIM, SEQ_LEN, build_tinygrad_hybrid
    from export_graph import export_hot_path_config

    np_dtype = np.float16 if use_fp16 else np.float32
    tg_dtype = dtypes.float16 if use_fp16 else dtypes.float32
    dev = Device["NV"]
    model, params = build_tinygrad_hybrid(conv_config, mlp_head_dims, seed=seed, use_fp16=use_fp16, batch_size=batch_size)

    # Two inputs: IMU window + current state
    static_imu = Tensor.zeros(batch_size, IN_DIM, SEQ_LEN, dtype=tg_dtype).contiguous().realize()
    static_state = Tensor.zeros(batch_size, IN_DIM, dtype=tg_dtype).contiguous().realize()
    static_out = Tensor.zeros(batch_size, OUT_DIM, dtype=tg_dtype).contiguous().realize()
    dev.synchronize()

    @TinyJit
    def run():
        static_out.assign(model(static_imu, static_state)).realize()

    rng = np.random.RandomState(99)
    imu_pool = rng.randn(warmup + n_iters + 50, batch_size, IN_DIM, SEQ_LEN).astype(np_dtype)
    state_pool = rng.randn(warmup + n_iters + 50, batch_size, IN_DIM).astype(np_dtype)

    imu_addr, imu_nbytes = _setup_direct_memory(static_imu)
    state_addr, state_nbytes = _setup_direct_memory(static_state)

    for i in range(warmup):
        ctypes.memmove(imu_addr, imu_pool[i].ctypes.data, imu_nbytes)
        ctypes.memmove(state_addr, state_pool[i].ctypes.data, state_nbytes)
        run()
        dev.synchronize()

    # For hybrid, the C hot path only knows about one input buffer.
    # We need to check which buffer the graph's input_buf_addr points to.
    # The export will use x_buf (static_imu) as input. We'll handle the
    # second input by writing to its memory directly before C dispatches.
    #
    # Actually, the C hot path only does memcpy to input_buf_addr. For hybrid
    # models with two separate input tensors, the graph expects both to be
    # populated. The simplest approach: combine both into a sensor_pool that
    # matches the total input size, and ensure the C code copies the full thing.
    #
    # BUT: the two inputs are separate tensors at different GPU addresses.
    # The C hot path as-is only handles a single input buffer.
    #
    # Workaround: We'll benchmark the hybrid using Python-side memmoves for
    # the two inputs, but let C handle the GPU dispatch (the expensive part).
    # This is still a valid benchmark because the memmove is ~0.1 µs.

    # For now, use Python dispatch timing for the hybrid C hotpath since
    # the two-input case needs a different C API.
    # Let's measure what matters: the GPU dispatch overhead.
    # We'll time: memmove both inputs + C graph dispatch + memmove output.

    x_buf = static_imu._buffer()
    o_buf = static_out._buffer()
    cfg_dict = export_hot_path_config(run, dev, x_buf, o_buf)

    # Override input to cover both buffers — the C code does memcpy(input_buf_addr, ...)
    # But for hybrid, both input addresses are patched into the graph already.
    # The C hot path only needs to update the DATA in those buffers, not their addresses.
    # So we'll write imu data to imu_addr and state data to state_addr manually,
    # then call the C path which does: patch + submit + wait.
    # We need a custom benchmark loop for this.

    lib = ctypes.CDLL(HOT_PATH_SO)
    lib.hot_path_init.argtypes = [ctypes.c_void_p]
    lib.hot_path_init.restype = None
    lib.hot_path_run_iteration.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.hot_path_run_iteration.restype = ctypes.c_uint64

    cfg_struct = _fill_config_struct(cfg_dict)
    lib.hot_path_init(ctypes.byref(cfg_struct))

    out_nbytes = cfg_dict['output_size']
    action = np.zeros(OUT_DIM, dtype=np_dtype)

    # Warmup C path
    for i in range(min(warmup, 30)):
        # Write both inputs manually
        ctypes.memmove(imu_addr, imu_pool[warmup + i].ctypes.data, imu_nbytes)
        ctypes.memmove(state_addr, state_pool[warmup + i].ctypes.data, state_nbytes)
        # C dispatches (it will also memcpy cfg.input_buf_addr, but that's the imu buffer)
        dummy_in = imu_pool[warmup + i].reshape(-1)
        lib.hot_path_run_iteration(ctypes.byref(cfg_struct),
                                   dummy_in.ctypes.data,
                                   action.ctypes.data)

    # Benchmark
    times = []
    for i in range(n_iters):
        idx = warmup + 30 + i
        imu_data = imu_pool[idx]
        state_data = state_pool[idx]

        t0 = time.perf_counter_ns()
        # Write both inputs to their respective GPU buffers (unified memory)
        ctypes.memmove(imu_addr, imu_data.ctypes.data, imu_nbytes)
        ctypes.memmove(state_addr, state_data.ctypes.data, state_nbytes)
        # C does: patch + gpfifo submit + doorbell + spin-wait + memcpy output
        # We pass imu as the "sensor_data" (C will memcpy it to input_buf which IS imu)
        # The state buffer is already written above
        ns = lib.hot_path_run_iteration(ctypes.byref(cfg_struct),
                                        imu_data.reshape(-1).ctypes.data,
                                        action.ctypes.data)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)

    # Sync Python state
    dev.compute_gpfifo.put_value = cfg_struct.gpfifo_put_value
    dev.timeline_value = cfg_struct.timeline_value
    dev.timeline_signal.wait(cfg_struct.last_tl_value)

    del run, static_imu, static_state, static_out
    dev.synchronize()

    return times, params
