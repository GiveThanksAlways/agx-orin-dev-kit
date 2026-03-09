#!/usr/bin/env python3
"""bench_cioffi_tcn.py — Benchmark the Cioffi et al. Learned Inertial Odometry TCN.

Tests the exact architecture from the paper (arXiv:2210.15287) across backends:
  1. tinygrad NV=1    — raw Tegra ioctls, direct memmove, Python JIT
  2. tinygrad CUDA=1  — standard CUDA runtime via tinygrad
  3. tinygrad NV=1 + C Hot Path — same GPU kernels replayed from C
  4. TensorRT         — NVIDIA's optimized inference engine
  5. PyTorch eager    — standard PyTorch forward pass
  6. PyTorch CUDA Graphs — PyTorch with CUDA graph capture

Model: TCN, ~250K params, input=(1, 6, 50), output=(1, 3)
Platform: Jetson AGX Orin 64GB, JetPack 6, CUDA 12.6

Usage (from this directory, inside nix develop):
  NV=1 JITBEAM=2 python3 bench_cioffi_tcn.py                 # All backends
  NV=1 JITBEAM=2 python3 bench_cioffi_tcn.py --backend nv     # tinygrad NV=1 only
  NV=1 JITBEAM=2 python3 bench_cioffi_tcn.py --backend hotpath # C Hot Path only
  CUDA=1 python3 bench_cioffi_tcn.py --backend cuda            # tinygrad CUDA=1
  python3 bench_cioffi_tcn.py --backend trt                    # TensorRT only
  python3 bench_cioffi_tcn.py --backend pytorch                # PyTorch (needs CUDA torch)
  NV=1 JITBEAM=2 python3 bench_cioffi_tcn.py --iters 1000     # Quick run
  NV=1 JITBEAM=2 python3 bench_cioffi_tcn.py --precision fp32 # FP32 comparison
"""
import os, sys, argparse, time, json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cioffi_tcn import (
    INPUT_DIM, OUTPUT_DIM, NUM_CHANNELS, KERNEL_SIZE, SEQ_LEN, SEED,
    generate_weights, generate_input_pool, build_pytorch_tcn, export_onnx,
    build_tinygrad_tcn, _count_params,
)

SCRIPT_DIR = Path(__file__).parent
WEIGHTS_DIR = SCRIPT_DIR / "weights"
ONNX_DIR = SCRIPT_DIR / "onnx"


# ═══════════════════════════════════════════════════════════════════════════════
# Stats helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stats(times_us):
    a = np.asarray(times_us)
    return {
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "p99": float(np.percentile(a, 99)),
        "p999": float(np.percentile(a, 99.9)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "count": len(a),
    }


def print_result(label, times_us):
    s = compute_stats(times_us)
    hz = 1e6 / s["median"] if s["median"] > 0 else 0
    print(f"  {label:40s}  median={s['median']:8.1f} µs  "
          f"p99={s['p99']:8.1f}  p999={s['p999']:8.1f}  "
          f"max={s['max']:9.1f}  freq={hz:8.0f} Hz")
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# Backend: tinygrad NV=1
# ═══════════════════════════════════════════════════════════════════════════════

def bench_tinygrad_nv(weights, use_fp16, warmup, n_iters):
    """Benchmark tinygrad NV=1 with direct memmove (Approach C)."""
    import ctypes
    os.environ["NV"] = "1"
    from tinygrad import Tensor, dtypes, Device
    from tinygrad.engine.jit import TinyJit

    model, param_count = build_tinygrad_tcn(weights, use_fp16=use_fp16)

    tg_dtype = dtypes.float16 if use_fp16 else dtypes.float32
    np_dtype = np.float16 if use_fp16 else np.float32

    # Static buffers
    static_x = Tensor.zeros(1, INPUT_DIM, SEQ_LEN, dtype=tg_dtype).contiguous().realize()
    static_out = Tensor.zeros(1, OUTPUT_DIM, dtype=tg_dtype).contiguous().realize()

    in_addr = static_x._buffer()._buf.cpu_view().addr
    out_addr = static_out._buffer()._buf.cpu_view().addr
    in_nbytes = INPUT_DIM * SEQ_LEN * (2 if use_fp16 else 4)
    out_nbytes = OUTPUT_DIM * (2 if use_fp16 else 4)

    @TinyJit
    def run():
        static_out.assign(model(static_x)).realize()

    # Warmup
    pool = generate_input_pool(warmup + n_iters)
    for i in range(warmup):
        ctypes.memmove(in_addr, pool[i].ctypes.data, in_nbytes)
        run()
        Device["NV"].synchronize()

    # Benchmark
    result_buf = np.zeros(OUTPUT_DIM, dtype=np_dtype)
    times = []
    for i in range(n_iters):
        ctypes.memmove(in_addr, pool[warmup + i].ctypes.data, in_nbytes)
        t0 = time.perf_counter()
        run()
        Device["NV"].synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    ctypes.memmove(result_buf.ctypes.data, out_addr, out_nbytes)

    return times, param_count, result_buf


# ═══════════════════════════════════════════════════════════════════════════════
# Backend: tinygrad CUDA=1
# ═══════════════════════════════════════════════════════════════════════════════

def bench_tinygrad_cuda(weights, use_fp16, warmup, n_iters):
    """Benchmark tinygrad CUDA=1 (standard CUDA runtime backend)."""
    import ctypes
    # Note: caller must set CUDA=1 env var before importing tinygrad
    from tinygrad import Tensor, dtypes, Device
    from tinygrad.engine.jit import TinyJit

    model, param_count = build_tinygrad_tcn(weights, use_fp16=use_fp16)

    tg_dtype = dtypes.float16 if use_fp16 else dtypes.float32
    np_dtype = np.float16 if use_fp16 else np.float32

    static_x = Tensor.zeros(1, INPUT_DIM, SEQ_LEN, dtype=tg_dtype).contiguous().realize()
    static_out = Tensor.zeros(1, OUTPUT_DIM, dtype=tg_dtype).contiguous().realize()

    @TinyJit
    def run():
        static_out.assign(model(static_x)).realize()

    pool = generate_input_pool(warmup + n_iters)
    for i in range(warmup):
        static_x.assign(Tensor(pool[i:i+1], dtype=tg_dtype)).realize()
        run()
        Device["CUDA"].synchronize()

    times = []
    for i in range(n_iters):
        static_x.assign(Tensor(pool[warmup + i:warmup + i + 1], dtype=tg_dtype)).realize()
        t0 = time.perf_counter()
        run()
        Device["CUDA"].synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    result_buf = static_out.numpy().flatten().astype(np_dtype)
    return times, param_count, result_buf


# ═══════════════════════════════════════════════════════════════════════════════
# Backend: tinygrad NV=1 + C Hot Path
# ═══════════════════════════════════════════════════════════════════════════════

def bench_c_hot_path(weights, use_fp16, warmup, n_iters):
    """Benchmark C Hot Path: replay tinygrad HCQGraph from C."""
    import ctypes
    os.environ["NV"] = "1"
    from tinygrad import Tensor, dtypes, Device
    from tinygrad.engine.jit import TinyJit

    hp_dir = os.environ.get("HOT_PATH_DIR",
        str(SCRIPT_DIR.parent / "control-loop" / "hot_path"))

    so_path = os.path.join(hp_dir, "hot_path.so")
    if not os.path.exists(so_path):
        print(f"  ERROR: hot_path.so not found at {so_path}")
        print(f"  Build it first: cd {hp_dir} && make")
        return None, 0, None

    sys.path.insert(0, hp_dir)
    from export_graph import export_hot_path_config

    model, param_count = build_tinygrad_tcn(weights, use_fp16=use_fp16)
    tg_dtype = dtypes.float16 if use_fp16 else dtypes.float32
    np_dtype = np.float16 if use_fp16 else np.float32

    static_x = Tensor.zeros(1, INPUT_DIM, SEQ_LEN, dtype=tg_dtype).contiguous().realize()
    static_out = Tensor.zeros(1, OUTPUT_DIM, dtype=tg_dtype).contiguous().realize()

    in_addr = static_x._buffer()._buf.cpu_view().addr
    out_addr = static_out._buffer()._buf.cpu_view().addr
    in_nbytes = INPUT_DIM * SEQ_LEN * (2 if use_fp16 else 4)
    out_nbytes = OUTPUT_DIM * (2 if use_fp16 else 4)

    @TinyJit
    def run():
        static_out.assign(model(static_x)).realize()

    # Warmup to build HCQGraph
    pool = generate_input_pool(warmup + n_iters)
    for i in range(warmup):
        ctypes.memmove(in_addr, pool[i].ctypes.data, in_nbytes)
        run()
        Device["NV"].synchronize()

    # Export graph config for C hot path
    print("  Exporting HCQGraph config...")
    cfg = export_hot_path_config(run, Device["NV"],
                                  static_x._buffer(), static_out._buffer())

    # Load C shared library
    lib = ctypes.CDLL(so_path)

    # Build C config struct
    class PatchEntry(ctypes.Structure):
        _fields_ = [
            ("addr", ctypes.c_uint64),
            ("var_type", ctypes.c_uint32),
            ("mask", ctypes.c_uint32),
        ]

    MAX_PATCHES = 512
    MAX_QUEUE_SIGNALS = 16

    class HotPathConfig(ctypes.Structure):
        """Must match hot_path_config_t in hot_path.h exactly.
        ctypes follows C alignment rules automatically."""
        _fields_ = [
            ("input_buf_addr", ctypes.c_uint64),
            ("output_buf_addr", ctypes.c_uint64),
            ("input_size", ctypes.c_uint32),
            ("output_size", ctypes.c_uint32),
            ("gpfifo_ring_addr", ctypes.c_uint64),
            ("gpfifo_gpput_addr", ctypes.c_uint64),
            ("gpfifo_entries_count", ctypes.c_uint32),
            ("gpfifo_token", ctypes.c_uint32),
            ("gpfifo_put_value", ctypes.c_uint32),
            ("cmdq_gpu_addr", ctypes.c_uint64),
            ("cmdq_len_u32", ctypes.c_uint32),
            ("gpu_mmio_addr", ctypes.c_uint64),
            ("timeline_signal_addr", ctypes.c_uint64),
            ("timeline_value", ctypes.c_uint32),
            ("last_tl_value", ctypes.c_uint32),
            ("kick_signal_addr", ctypes.c_uint64),
            ("kickoff_value", ctypes.c_uint32),
            ("num_queue_signals", ctypes.c_uint32),
            ("queue_signal_addrs", ctypes.c_uint64 * MAX_QUEUE_SIGNALS),
            ("num_patches", ctypes.c_uint32),
            ("patches", PatchEntry * MAX_PATCHES),
            ("gpfifo_entry", ctypes.c_uint64),
        ]

    c_cfg = HotPathConfig()
    c_cfg.input_buf_addr = cfg["input_buf_addr"]
    c_cfg.output_buf_addr = cfg["output_buf_addr"]
    c_cfg.input_size = cfg["input_size"]
    c_cfg.output_size = cfg["output_size"]
    c_cfg.gpfifo_ring_addr = cfg["gpfifo_ring_addr"]
    c_cfg.gpfifo_gpput_addr = cfg["gpfifo_gpput_addr"]
    c_cfg.gpfifo_entries_count = cfg["gpfifo_entries_count"]
    c_cfg.gpfifo_token = cfg["gpfifo_token"]
    c_cfg.gpfifo_put_value = cfg["gpfifo_put_value"]
    c_cfg.cmdq_gpu_addr = cfg["cmdq_gpu_addr"]
    c_cfg.cmdq_len_u32 = cfg["cmdq_len_u32"]
    c_cfg.gpu_mmio_addr = cfg["gpu_mmio_addr"]
    c_cfg.timeline_signal_addr = cfg["timeline_signal_addr"]
    c_cfg.timeline_value = cfg["timeline_value"]
    c_cfg.last_tl_value = cfg["last_tl_value"]
    c_cfg.kick_signal_addr = cfg["kick_signal_addr"]
    c_cfg.kickoff_value = cfg["kickoff_value"]

    patches = cfg["patches"]
    c_cfg.num_patches = len(patches)
    for j, p in enumerate(patches):
        c_cfg.patches[j].addr = p["addr"]
        c_cfg.patches[j].var_type = p["var_type"]
        c_cfg.patches[j].mask = p["mask"]

    q_sigs = cfg["queue_signal_addrs"]
    c_cfg.num_queue_signals = len(q_sigs)
    for j, a in enumerate(q_sigs):
        c_cfg.queue_signal_addrs[j] = a

    lib.hot_path_init(ctypes.byref(c_cfg))

    # Prepare sensor pool as contiguous bytes
    sensor_pool = np.ascontiguousarray(pool[warmup:warmup + n_iters])
    action_pool = np.zeros((n_iters, INPUT_DIM, SEQ_LEN), dtype=np_dtype)
    # action pool: each output is OUTPUT_DIM elements
    action_pool_out = np.zeros(n_iters * out_nbytes, dtype=np.uint8)
    times_ns = (ctypes.c_uint64 * n_iters)()

    # C warmup (5 iterations to warm caches in C dispatch path)
    warmup_sensor = np.ascontiguousarray(pool[:5])
    warmup_action = np.zeros(5 * out_nbytes, dtype=np.uint8)
    warmup_times = (ctypes.c_uint64 * 5)()
    lib.hot_path_benchmark(
        ctypes.byref(c_cfg),
        warmup_sensor.ctypes.data_as(ctypes.c_void_p),
        warmup_action.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_uint32(5),
        warmup_times,
    )

    # Benchmark
    lib.hot_path_benchmark(
        ctypes.byref(c_cfg),
        sensor_pool.ctypes.data_as(ctypes.c_void_p),
        action_pool_out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_uint32(n_iters),
        times_ns,
    )

    times_us = [t / 1000.0 for t in times_ns]
    # Extract last action output
    result_buf = np.frombuffer(
        action_pool_out[(n_iters - 1) * out_nbytes : n_iters * out_nbytes],
        dtype=np_dtype)
    return times_us, param_count, result_buf


# ═══════════════════════════════════════════════════════════════════════════════
# Backend: TensorRT
# ═══════════════════════════════════════════════════════════════════════════════

def bench_tensorrt(weights, use_fp16, warmup, n_iters, use_tf32=False):
    """Benchmark TensorRT inference."""
    import ctypes

    ONNX_DIR.mkdir(exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)

    prec = "fp16" if use_fp16 else "fp32"
    onnx_path = str(ONNX_DIR / f"cioffi_tcn_{prec}.onnx")
    engine_path = str(ONNX_DIR / f"cioffi_tcn_{prec}.engine")

    if not os.path.exists(onnx_path):
        export_onnx(onnx_path, weights, use_fp16=False)

    # Try to import TensorRT
    try:
        import tensorrt as trt
    except ImportError:
        # Manual load from LD_LIBRARY_PATH
        trt_path = os.environ.get("TENSORRT_PATH", "")
        trt_lib = os.path.join(trt_path, "lib", "libnvinfer.so")
        if os.path.exists(trt_lib):
            ctypes.CDLL(trt_lib, mode=ctypes.RTLD_GLOBAL)
        try:
            import tensorrt as trt
        except ImportError:
            print("  ERROR: TensorRT Python bindings not available")
            return None, 0, None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Build engine if not cached
    if not os.path.exists(engine_path):
        print(f"  Building TRT engine ({prec})...")
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"  ONNX parse error: {parser.get_error(i)}")
                return None, 0, None

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        if use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if not use_tf32:
            config.clear_flag(trt.BuilderFlag.TF32)

        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            print("  ERROR: TRT engine build failed")
            return None, 0, None

        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        print(f"  Saved TRT engine: {engine_path}")
    else:
        print(f"  Using cached TRT engine: {engine_path}")

    # Load engine
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate CUDA buffers
    try:
        import pycuda.driver as cuda_drv
        import pycuda.autoinit
    except ImportError:
        # Use ctypes to call CUDA runtime directly
        print("  Using ctypes CUDA runtime for TRT buffer allocation")

    # Use numpy + cudart approach
    np_dtype = np.float16 if use_fp16 else np.float32
    input_shape = (1, INPUT_DIM, SEQ_LEN)
    output_shape = (1, OUTPUT_DIM)

    h_input = np.zeros(input_shape, dtype=np.float32)  # TRT inputs are FP32
    h_output = np.zeros(output_shape, dtype=np.float32)

    # Use TRT's built-in managed memory if available
    pool = generate_input_pool(warmup + n_iters)

    # Try using trt.Volume / simple approach
    input_nbytes = int(np.prod(input_shape) * 4)  # always FP32 for TRT IO
    output_nbytes = int(np.prod(output_shape) * 4)

    # Allocate GPU memory via ctypes cudaMalloc
    cudart = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
    d_input = ctypes.c_void_p()
    d_output = ctypes.c_void_p()
    cudart.cudaMalloc(ctypes.byref(d_input), input_nbytes)
    cudart.cudaMalloc(ctypes.byref(d_output), output_nbytes)

    # Create CUDA stream
    stream = ctypes.c_void_p()
    cudart.cudaStreamCreate(ctypes.byref(stream))

    # Set tensor addresses
    context.set_tensor_address("imu_data", d_input.value)
    context.set_tensor_address("displacement", d_output.value)

    # Warmup
    for i in range(warmup):
        h_input[:] = pool[i:i+1].astype(np.float32)
        cudart.cudaMemcpyAsync(
            d_input, h_input.ctypes.data,
            input_nbytes, 1, stream)  # 1 = cudaMemcpyHostToDevice
        context.execute_async_v3(stream.value)
        cudart.cudaMemcpyAsync(
            h_output.ctypes.data, d_output,
            output_nbytes, 2, stream)  # 2 = cudaMemcpyDeviceToHost
        cudart.cudaStreamSynchronize(stream)

    # Benchmark
    times = []
    for i in range(n_iters):
        h_input[:] = pool[warmup + i:warmup + i + 1].astype(np.float32)
        cudart.cudaMemcpyAsync(
            d_input, h_input.ctypes.data,
            input_nbytes, 1, stream)
        cudart.cudaStreamSynchronize(stream)

        t0 = time.perf_counter()
        context.execute_async_v3(stream.value)
        cudart.cudaStreamSynchronize(stream)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    cudart.cudaMemcpyAsync(
        h_output.ctypes.data, d_output,
        output_nbytes, 2, stream)
    cudart.cudaStreamSynchronize(stream)

    # Cleanup
    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    cudart.cudaStreamDestroy(stream)

    param_count = _count_params()
    result = h_output.flatten().astype(np_dtype)
    return times, param_count, result


# ═══════════════════════════════════════════════════════════════════════════════
# Backend: TensorRT (with H2D + D2H included in timing)
# ═══════════════════════════════════════════════════════════════════════════════

def bench_tensorrt_full(weights, use_fp16, warmup, n_iters, use_tf32=False):
    """Benchmark TensorRT with full H2D + compute + D2H in timing.

    This is the apples-to-apples comparison with tinygrad NV=1 which also
    includes data transfer in its timing.
    """
    import ctypes

    ONNX_DIR.mkdir(exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)

    prec = "fp16" if use_fp16 else "fp32"
    onnx_path = str(ONNX_DIR / f"cioffi_tcn_{prec}.onnx")
    engine_path = str(ONNX_DIR / f"cioffi_tcn_{prec}.engine")

    if not os.path.exists(onnx_path):
        export_onnx(onnx_path, weights, use_fp16=False)

    try:
        import tensorrt as trt
    except ImportError:
        trt_path = os.environ.get("TENSORRT_PATH", "")
        trt_lib = os.path.join(trt_path, "lib", "libnvinfer.so")
        if os.path.exists(trt_lib):
            ctypes.CDLL(trt_lib, mode=ctypes.RTLD_GLOBAL)
        try:
            import tensorrt as trt
        except ImportError:
            print("  ERROR: TensorRT Python bindings not available")
            return None, 0, None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    if not os.path.exists(engine_path):
        print(f"  Building TRT engine ({prec})...")
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"  ONNX parse error: {parser.get_error(i)}")
                return None, 0, None
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if not use_tf32:
            config.clear_flag(trt.BuilderFlag.TF32)
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            print("  ERROR: TRT engine build failed")
            return None, 0, None
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)

    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    np_dtype = np.float16 if use_fp16 else np.float32
    input_shape = (1, INPUT_DIM, SEQ_LEN)
    output_shape = (1, OUTPUT_DIM)
    h_input = np.zeros(input_shape, dtype=np.float32)
    h_output = np.zeros(output_shape, dtype=np.float32)

    input_nbytes = int(np.prod(input_shape) * 4)
    output_nbytes = int(np.prod(output_shape) * 4)

    cudart = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
    d_input = ctypes.c_void_p()
    d_output = ctypes.c_void_p()
    cudart.cudaMalloc(ctypes.byref(d_input), input_nbytes)
    cudart.cudaMalloc(ctypes.byref(d_output), output_nbytes)
    stream = ctypes.c_void_p()
    cudart.cudaStreamCreate(ctypes.byref(stream))

    context.set_tensor_address("imu_data", d_input.value)
    context.set_tensor_address("displacement", d_output.value)

    pool = generate_input_pool(warmup + n_iters)

    for i in range(warmup):
        h_input[:] = pool[i:i+1].astype(np.float32)
        cudart.cudaMemcpyAsync(d_input, h_input.ctypes.data, input_nbytes, 1, stream)
        context.execute_async_v3(stream.value)
        cudart.cudaMemcpyAsync(h_output.ctypes.data, d_output, output_nbytes, 2, stream)
        cudart.cudaStreamSynchronize(stream)

    # Benchmark: full round-trip (H2D + compute + D2H)
    times = []
    for i in range(n_iters):
        h_input[:] = pool[warmup + i:warmup + i + 1].astype(np.float32)
        t0 = time.perf_counter()
        cudart.cudaMemcpyAsync(d_input, h_input.ctypes.data, input_nbytes, 1, stream)
        context.execute_async_v3(stream.value)
        cudart.cudaMemcpyAsync(h_output.ctypes.data, d_output, output_nbytes, 2, stream)
        cudart.cudaStreamSynchronize(stream)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    cudart.cudaStreamDestroy(stream)

    param_count = _count_params()
    result = h_output.flatten().astype(np_dtype)
    return times, param_count, result


# ═══════════════════════════════════════════════════════════════════════════════
# Backend: TensorRT + Tegra Zero-Copy (cudaMallocManaged)
# ═══════════════════════════════════════════════════════════════════════════════

def bench_tensorrt_zerocopy(weights, use_fp16, warmup, n_iters, use_tf32=False):
    """Benchmark TensorRT with Tegra zero-copy (cudaMallocManaged).

    On Jetson (Tegra), CPU and GPU share the same DRAM. Using
    cudaMallocManaged instead of cudaMalloc eliminates the need for
    cudaMemcpyAsync H2D/D2H — we just memcpy directly into managed
    memory and the GPU reads it in-place.

    This tests the concept behind libinfer's infer_zerocopy() from Task 1.
    """
    import ctypes

    ONNX_DIR.mkdir(exist_ok=True)
    prec = "fp16" if use_fp16 else "fp32"
    onnx_path = str(ONNX_DIR / f"cioffi_tcn_{prec}.onnx")
    engine_path = str(ONNX_DIR / f"cioffi_tcn_{prec}.engine")

    if not os.path.exists(onnx_path):
        export_onnx(onnx_path, weights, use_fp16=False)

    try:
        import tensorrt as trt
    except ImportError:
        trt_path = os.environ.get("TENSORRT_PATH", "")
        trt_lib = os.path.join(trt_path, "lib", "libnvinfer.so")
        if os.path.exists(trt_lib):
            ctypes.CDLL(trt_lib, mode=ctypes.RTLD_GLOBAL)
        try:
            import tensorrt as trt
        except ImportError:
            print("  ERROR: TensorRT Python bindings not available")
            return None, 0, None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    if not os.path.exists(engine_path):
        print(f"  Building TRT engine ({prec})...")
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"  ONNX parse error: {parser.get_error(i)}")
                return None, 0, None
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if not use_tf32:
            config.clear_flag(trt.BuilderFlag.TF32)
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            print("  ERROR: TRT engine build failed")
            return None, 0, None
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)

    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    np_dtype = np.float16 if use_fp16 else np.float32
    input_shape = (1, INPUT_DIM, SEQ_LEN)
    output_shape = (1, OUTPUT_DIM)
    input_nbytes = int(np.prod(input_shape) * 4)
    output_nbytes = int(np.prod(output_shape) * 4)

    # Key difference: use cudaMallocManaged instead of cudaMalloc
    cudart = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
    d_input = ctypes.c_void_p()
    d_output = ctypes.c_void_p()
    # cudaMallocManaged flags: cudaMemAttachGlobal = 1
    cudart.cudaMallocManaged(ctypes.byref(d_input), input_nbytes, 1)
    cudart.cudaMallocManaged(ctypes.byref(d_output), output_nbytes, 1)

    stream = ctypes.c_void_p()
    cudart.cudaStreamCreate(ctypes.byref(stream))

    context.set_tensor_address("imu_data", d_input.value)
    context.set_tensor_address("displacement", d_output.value)

    pool = generate_input_pool(warmup + n_iters)

    # Create numpy views over managed memory (zero-copy!)
    inp_arr = (ctypes.c_float * (input_nbytes // 4)).from_address(d_input.value)
    out_arr = (ctypes.c_float * (output_nbytes // 4)).from_address(d_output.value)
    h_input_view = np.ctypeslib.as_array(inp_arr)
    h_output_view = np.ctypeslib.as_array(out_arr)

    # Warmup
    for i in range(warmup):
        h_input_view[:] = pool[i:i+1].astype(np.float32).ravel()
        context.execute_async_v3(stream.value)
        cudart.cudaStreamSynchronize(stream)

    # Benchmark: no cudaMemcpyAsync at all — direct CPU write to managed memory
    times = []
    for i in range(n_iters):
        h_input_view[:] = pool[warmup + i:warmup + i + 1].astype(np.float32).ravel()
        t0 = time.perf_counter()
        context.execute_async_v3(stream.value)
        cudart.cudaStreamSynchronize(stream)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    result = np.array(h_output_view[:OUTPUT_DIM], dtype=np.float32).astype(np_dtype)

    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    cudart.cudaStreamDestroy(stream)

    param_count = _count_params()
    return times, param_count, result


def bench_tensorrt_zerocopy_full(weights, use_fp16, warmup, n_iters, use_tf32=False):
    """TRT zero-copy, full round-trip timing (memcpy + compute + read-back).

    This is the apples-to-apples comparison with the stock TRT full path,
    showing the benefit of eliminating cudaMemcpyAsync on Tegra.
    """
    import ctypes

    ONNX_DIR.mkdir(exist_ok=True)
    prec = "fp16" if use_fp16 else "fp32"
    onnx_path = str(ONNX_DIR / f"cioffi_tcn_{prec}.onnx")
    engine_path = str(ONNX_DIR / f"cioffi_tcn_{prec}.engine")

    if not os.path.exists(onnx_path):
        export_onnx(onnx_path, weights, use_fp16=False)

    try:
        import tensorrt as trt
    except ImportError:
        trt_path = os.environ.get("TENSORRT_PATH", "")
        trt_lib = os.path.join(trt_path, "lib", "libnvinfer.so")
        if os.path.exists(trt_lib):
            ctypes.CDLL(trt_lib, mode=ctypes.RTLD_GLOBAL)
        try:
            import tensorrt as trt
        except ImportError:
            print("  ERROR: TensorRT Python bindings not available")
            return None, 0, None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    if not os.path.exists(engine_path):
        print(f"  Building TRT engine ({prec})...")
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                return None, 0, None
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if not use_tf32:
            config.clear_flag(trt.BuilderFlag.TF32)
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            return None, 0, None
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)

    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    np_dtype = np.float16 if use_fp16 else np.float32
    input_shape = (1, INPUT_DIM, SEQ_LEN)
    output_shape = (1, OUTPUT_DIM)
    input_nbytes = int(np.prod(input_shape) * 4)
    output_nbytes = int(np.prod(output_shape) * 4)

    cudart = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
    d_input = ctypes.c_void_p()
    d_output = ctypes.c_void_p()
    cudart.cudaMallocManaged(ctypes.byref(d_input), input_nbytes, 1)
    cudart.cudaMallocManaged(ctypes.byref(d_output), output_nbytes, 1)

    stream = ctypes.c_void_p()
    cudart.cudaStreamCreate(ctypes.byref(stream))

    context.set_tensor_address("imu_data", d_input.value)
    context.set_tensor_address("displacement", d_output.value)

    pool = generate_input_pool(warmup + n_iters)

    inp_arr = (ctypes.c_float * (input_nbytes // 4)).from_address(d_input.value)
    out_arr = (ctypes.c_float * (output_nbytes // 4)).from_address(d_output.value)
    h_input_view = np.ctypeslib.as_array(inp_arr)
    h_output_view = np.ctypeslib.as_array(out_arr)

    for i in range(warmup):
        h_input_view[:] = pool[i:i+1].astype(np.float32).ravel()
        context.execute_async_v3(stream.value)
        cudart.cudaStreamSynchronize(stream)

    # Full round-trip: write input + compute + read output
    times = []
    for i in range(n_iters):
        t0 = time.perf_counter()
        h_input_view[:] = pool[warmup + i:warmup + i + 1].astype(np.float32).ravel()
        context.execute_async_v3(stream.value)
        cudart.cudaStreamSynchronize(stream)
        _ = np.array(h_output_view[:OUTPUT_DIM], dtype=np.float32)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    result = np.array(h_output_view[:OUTPUT_DIM], dtype=np.float32).astype(np_dtype)

    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    cudart.cudaStreamDestroy(stream)

    param_count = _count_params()
    return times, param_count, result


# ═══════════════════════════════════════════════════════════════════════════════
# Backend: TensorRT + CUDA Graphs (capture enqueueV3, replay with cudaGraphLaunch)
# ═══════════════════════════════════════════════════════════════════════════════

def bench_tensorrt_cuda_graph(weights, use_fp16, warmup, n_iters, use_tf32=False):
    """Benchmark TensorRT with CUDA graph capture and replay.

    Captures enqueueV3 into a CUDA graph on the first call, then replays
    it with cudaGraphLaunch — eliminating per-call runtime dispatch overhead.

    This tests the concept behind the cuda-graph-replay Rust crate (Task 4).
    """
    import ctypes

    ONNX_DIR.mkdir(exist_ok=True)
    prec = "fp16" if use_fp16 else "fp32"
    onnx_path = str(ONNX_DIR / f"cioffi_tcn_{prec}.onnx")
    engine_path = str(ONNX_DIR / f"cioffi_tcn_{prec}.engine")

    if not os.path.exists(onnx_path):
        export_onnx(onnx_path, weights, use_fp16=False)

    try:
        import tensorrt as trt
    except ImportError:
        trt_path = os.environ.get("TENSORRT_PATH", "")
        trt_lib = os.path.join(trt_path, "lib", "libnvinfer.so")
        if os.path.exists(trt_lib):
            ctypes.CDLL(trt_lib, mode=ctypes.RTLD_GLOBAL)
        try:
            import tensorrt as trt
        except ImportError:
            print("  ERROR: TensorRT Python bindings not available")
            return None, 0, None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    if not os.path.exists(engine_path):
        print(f"  Building TRT engine ({prec})...")
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                return None, 0, None
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if not use_tf32:
            config.clear_flag(trt.BuilderFlag.TF32)
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            return None, 0, None
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)

    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    np_dtype = np.float16 if use_fp16 else np.float32
    input_shape = (1, INPUT_DIM, SEQ_LEN)
    output_shape = (1, OUTPUT_DIM)
    input_nbytes = int(np.prod(input_shape) * 4)
    output_nbytes = int(np.prod(output_shape) * 4)

    cudart = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
    d_input = ctypes.c_void_p()
    d_output = ctypes.c_void_p()
    cudart.cudaMalloc(ctypes.byref(d_input), input_nbytes)
    cudart.cudaMalloc(ctypes.byref(d_output), output_nbytes)

    stream = ctypes.c_void_p()
    cudart.cudaStreamCreate(ctypes.byref(stream))

    context.set_tensor_address("imu_data", d_input.value)
    context.set_tensor_address("displacement", d_output.value)

    pool = generate_input_pool(warmup + n_iters)
    h_input = np.zeros(input_shape, dtype=np.float32)
    h_output = np.zeros(output_shape, dtype=np.float32)

    # Standard warmup first (needed before capture)
    for i in range(3):
        h_input[:] = pool[i:i+1].astype(np.float32)
        cudart.cudaMemcpyAsync(d_input, h_input.ctypes.data, input_nbytes, 1, stream)
        context.execute_async_v3(stream.value)
        cudart.cudaMemcpyAsync(h_output.ctypes.data, d_output, output_nbytes, 2, stream)
        cudart.cudaStreamSynchronize(stream)

    # Capture CUDA graph
    # cudaStreamCaptureMode: cudaStreamCaptureModeGlobal = 0
    graph = ctypes.c_void_p()
    graph_exec = ctypes.c_void_p()

    cudart.cudaStreamBeginCapture(stream, 0)
    context.execute_async_v3(stream.value)
    cudart.cudaStreamEndCapture(stream, ctypes.byref(graph))
    cudart.cudaGraphInstantiate(ctypes.byref(graph_exec), graph, 0)

    print("  CUDA graph captured successfully")

    # Warmup with graph replay
    for i in range(warmup):
        h_input[:] = pool[i:i+1].astype(np.float32)
        cudart.cudaMemcpyAsync(d_input, h_input.ctypes.data, input_nbytes, 1, stream)
        cudart.cudaGraphLaunch(graph_exec, stream)
        cudart.cudaStreamSynchronize(stream)

    # Benchmark: H2D then graph launch (GPU-only timing)
    times = []
    for i in range(n_iters):
        h_input[:] = pool[warmup + i:warmup + i + 1].astype(np.float32)
        cudart.cudaMemcpyAsync(d_input, h_input.ctypes.data, input_nbytes, 1, stream)
        cudart.cudaStreamSynchronize(stream)
        t0 = time.perf_counter()
        cudart.cudaGraphLaunch(graph_exec, stream)
        cudart.cudaStreamSynchronize(stream)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    cudart.cudaMemcpyAsync(h_output.ctypes.data, d_output, output_nbytes, 2, stream)
    cudart.cudaStreamSynchronize(stream)

    # Cleanup
    cudart.cudaGraphExecDestroy(graph_exec)
    cudart.cudaGraphDestroy(graph)
    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    cudart.cudaStreamDestroy(stream)

    param_count = _count_params()
    result = h_output.flatten().astype(np_dtype)
    return times, param_count, result


def bench_tensorrt_zerocopy_cuda_graph(weights, use_fp16, warmup, n_iters, use_tf32=False):
    """TRT + Tegra zero-copy + CUDA graph: the full optimization stack.

    Combines both optimizations:
      - cudaMallocManaged (no H2D/D2H copies)
      - CUDA graph capture (no per-call enqueueV3 dispatch)

    This shows the theoretical best-case for TensorRT on Tegra.
    """
    import ctypes

    ONNX_DIR.mkdir(exist_ok=True)
    prec = "fp16" if use_fp16 else "fp32"
    onnx_path = str(ONNX_DIR / f"cioffi_tcn_{prec}.onnx")
    engine_path = str(ONNX_DIR / f"cioffi_tcn_{prec}.engine")

    if not os.path.exists(onnx_path):
        export_onnx(onnx_path, weights, use_fp16=False)

    try:
        import tensorrt as trt
    except ImportError:
        trt_path = os.environ.get("TENSORRT_PATH", "")
        trt_lib = os.path.join(trt_path, "lib", "libnvinfer.so")
        if os.path.exists(trt_lib):
            ctypes.CDLL(trt_lib, mode=ctypes.RTLD_GLOBAL)
        try:
            import tensorrt as trt
        except ImportError:
            print("  ERROR: TensorRT Python bindings not available")
            return None, 0, None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    if not os.path.exists(engine_path):
        print(f"  Building TRT engine ({prec})...")
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                return None, 0, None
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if not use_tf32:
            config.clear_flag(trt.BuilderFlag.TF32)
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            return None, 0, None
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)

    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    np_dtype = np.float16 if use_fp16 else np.float32
    input_shape = (1, INPUT_DIM, SEQ_LEN)
    output_shape = (1, OUTPUT_DIM)
    input_nbytes = int(np.prod(input_shape) * 4)
    output_nbytes = int(np.prod(output_shape) * 4)

    cudart = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
    d_input = ctypes.c_void_p()
    d_output = ctypes.c_void_p()
    cudart.cudaMallocManaged(ctypes.byref(d_input), input_nbytes, 1)
    cudart.cudaMallocManaged(ctypes.byref(d_output), output_nbytes, 1)

    stream = ctypes.c_void_p()
    cudart.cudaStreamCreate(ctypes.byref(stream))

    context.set_tensor_address("imu_data", d_input.value)
    context.set_tensor_address("displacement", d_output.value)

    pool = generate_input_pool(warmup + n_iters)

    inp_arr = (ctypes.c_float * (input_nbytes // 4)).from_address(d_input.value)
    out_arr = (ctypes.c_float * (output_nbytes // 4)).from_address(d_output.value)
    h_input_view = np.ctypeslib.as_array(inp_arr)
    h_output_view = np.ctypeslib.as_array(out_arr)

    # Standard warmup before capture
    for i in range(3):
        h_input_view[:] = pool[i:i+1].astype(np.float32).ravel()
        context.execute_async_v3(stream.value)
        cudart.cudaStreamSynchronize(stream)

    # Capture CUDA graph
    graph = ctypes.c_void_p()
    graph_exec = ctypes.c_void_p()

    cudart.cudaStreamBeginCapture(stream, 0)
    context.execute_async_v3(stream.value)
    cudart.cudaStreamEndCapture(stream, ctypes.byref(graph))
    cudart.cudaGraphInstantiate(ctypes.byref(graph_exec), graph, 0)

    print("  CUDA graph captured (zero-copy + graph replay)")

    # Warmup with graph replay
    for i in range(warmup):
        h_input_view[:] = pool[i:i+1].astype(np.float32).ravel()
        cudart.cudaGraphLaunch(graph_exec, stream)
        cudart.cudaStreamSynchronize(stream)

    # Benchmark: direct write to managed memory + graph launch
    times = []
    for i in range(n_iters):
        h_input_view[:] = pool[warmup + i:warmup + i + 1].astype(np.float32).ravel()
        t0 = time.perf_counter()
        cudart.cudaGraphLaunch(graph_exec, stream)
        cudart.cudaStreamSynchronize(stream)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    result = np.array(h_output_view[:OUTPUT_DIM], dtype=np.float32).astype(np_dtype)

    cudart.cudaGraphExecDestroy(graph_exec)
    cudart.cudaGraphDestroy(graph)
    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    cudart.cudaStreamDestroy(stream)

    param_count = _count_params()
    return times, param_count, result


# ═══════════════════════════════════════════════════════════════════════════════
# Backend: PyTorch (eager + CUDA Graphs)
# ═══════════════════════════════════════════════════════════════════════════════

def bench_pytorch(weights, use_fp16, warmup, n_iters):
    """Benchmark PyTorch eager and CUDA Graphs (if CUDA available)."""
    import torch

    model, param_count = build_pytorch_tcn(weights, use_fp16=use_fp16)

    has_cuda = torch.cuda.is_available()
    np_dtype = np.float16 if use_fp16 else np.float32
    torch_dtype = torch.float16 if use_fp16 else torch.float32

    pool = generate_input_pool(warmup + n_iters)
    results = {}

    if has_cuda:
        device = torch.device("cuda")
        model = model.to(device)

        # ── PyTorch Eager (CUDA) ──
        for i in range(warmup):
            x = torch.from_numpy(pool[i:i+1].astype(np.float32)).to(device)
            if use_fp16:
                x = x.half()
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()

        times_eager = []
        for i in range(n_iters):
            x = torch.from_numpy(pool[warmup + i:warmup + i + 1].astype(np.float32)).to(device)
            if use_fp16:
                x = x.half()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_eager.append((t1 - t0) * 1e6)

        results["eager"] = times_eager

        # ── PyTorch CUDA Graphs ──
        static_x = torch.zeros(1, INPUT_DIM, SEQ_LEN, dtype=torch_dtype, device=device)
        static_out = torch.zeros(1, OUTPUT_DIM, dtype=torch_dtype, device=device)

        # Warmup for graph capture
        with torch.no_grad():
            for _ in range(3):
                static_out.copy_(model(static_x))
        torch.cuda.synchronize()

        # Capture graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out.copy_(model(static_x))
        torch.cuda.synchronize()

        # Graph warmup
        for i in range(warmup):
            static_x.copy_(torch.from_numpy(pool[i:i+1].astype(np.float32)).to(device).to(torch_dtype))
            g.replay()
            torch.cuda.synchronize()

        times_graph = []
        for i in range(n_iters):
            static_x.copy_(
                torch.from_numpy(pool[warmup + i:warmup + i + 1].astype(np.float32)).to(device).to(torch_dtype))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_graph.append((t1 - t0) * 1e6)

        results["cuda_graphs"] = times_graph
        result_val = static_out.detach().cpu().numpy().flatten().astype(np_dtype)
    else:
        # CPU-only PyTorch
        model = model.float()  # CPU doesn't do FP16 well

        for i in range(warmup):
            x = torch.from_numpy(pool[i:i+1].astype(np.float32))
            with torch.no_grad():
                _ = model(x)

        times_cpu = []
        for i in range(n_iters):
            x = torch.from_numpy(pool[warmup + i:warmup + i + 1].astype(np.float32))
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(x)
            t1 = time.perf_counter()
            times_cpu.append((t1 - t0) * 1e6)

        results["cpu"] = times_cpu
        result_val = out.detach().numpy().flatten().astype(np_dtype)

    return results, param_count, result_val


# ═══════════════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table(all_results, param_count):
    """Print consolidated comparison table."""
    print("\n")
    print("=" * 100)
    print("SUMMARY — Cioffi TCN Benchmark on Jetson AGX Orin 64GB")
    print(f"  Model: TCN ~{param_count:,} params, input=(1, {INPUT_DIM}, {SEQ_LEN}), output=(1, {OUTPUT_DIM})")
    print(f"  Paper: Cioffi et al., 'Learned Inertial Odometry', RAL 2023")
    print("=" * 100)

    # Sort by median latency
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["median"])

    hdr = f"  {'Backend':35s} {'Median µs':>10s} {'P99 µs':>10s} {'Max µs':>10s} {'Hz':>10s} {'vs Best':>8s}"
    print(hdr)
    print(f"  {'─'*35} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")

    best_median = sorted_results[0][1]["median"]
    for name, s in sorted_results:
        hz = 1e6 / s["median"] if s["median"] > 0 else 0
        ratio = s["median"] / best_median if best_median > 0 else 0
        marker = "★" if s["median"] == best_median else ""
        print(f"  {name:35s} {s['median']:10.1f} {s['p99']:10.1f} "
              f"{s['max']:10.1f} {hz:10.0f} {ratio:7.2f}x {marker}")

    # IMU rate comparison
    print()
    imu_rates = [100, 200, 1000]
    for rate in imu_rates:
        period_us = 1e6 / rate
        print(f"  IMU @ {rate:4d} Hz ({period_us:,.0f} µs budget):")
        for name, s in sorted_results:
            headroom = period_us / s["median"] if s["median"] > 0 else 0
            status = "✓" if s["median"] < period_us else "✗"
            print(f"    {status} {name:33s} {s['median']:8.1f} µs = {headroom:6.1f}x headroom")

    print()
    print("=" * 100)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Benchmark Cioffi TCN")
    parser.add_argument("--backend", type=str, default="all",
                        choices=["all", "nv", "cuda", "hotpath", "trt",
                                "trt-zerocopy", "trt-graph", "trt-zc-graph",
                                "pytorch"],
                        help="Which backend(s) to benchmark")
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["fp16", "fp32"],
                        help="Precision (default: fp16)")
    parser.add_argument("--iters", type=int, default=5000,
                        help="Number of benchmark iterations (default: 5000)")
    parser.add_argument("--warmup", type=int, default=30,
                        help="Warmup iterations (default: 30)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    use_fp16 = args.precision == "fp16"
    prec_label = "FP16" if use_fp16 else "FP32"

    print("=" * 100)
    print(f"Cioffi et al. Learned Inertial Odometry TCN — Inference Benchmark")
    print(f"  Paper: arXiv:2210.15287 (IEEE RA-L 2023)")
    print(f"  Model: TCN, {_count_params():,} params, input=(1,{INPUT_DIM},{SEQ_LEN}), output=(1,{OUTPUT_DIM})")
    print(f"  Architecture: 7 TemporalBlocks, channels={NUM_CHANNELS}, kernel_size={KERNEL_SIZE}")
    print(f"  Precision: {prec_label}")
    print(f"  Iterations: {args.iters} (warmup: {args.warmup})")
    print(f"  Platform: Jetson AGX Orin 64GB, JetPack 6, CUDA 12.6")
    print("=" * 100)

    weights = generate_weights()
    all_results = {}
    param_count = _count_params()

    backends = (["nv", "cuda", "hotpath", "trt", "trt-zerocopy",
                 "trt-graph", "trt-zc-graph", "pytorch"]
                if args.backend == "all" else [args.backend])

    for backend in backends:
        print(f"\n{'─'*100}")

        if backend == "nv":
            print(f"  [tinygrad NV=1 {prec_label}]")
            try:
                times, pc, result = bench_tinygrad_nv(
                    weights, use_fp16, args.warmup, args.iters)
                s = print_result(f"tinygrad NV=1 ({pc:,} params)", times)
                all_results[f"tinygrad NV=1 {prec_label}"] = s
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()

        elif backend == "cuda":
            print(f"  [tinygrad CUDA=1 {prec_label}]")
            try:
                times, pc, result = bench_tinygrad_cuda(
                    weights, use_fp16, args.warmup, args.iters)
                s = print_result(f"tinygrad CUDA=1 ({pc:,} params)", times)
                all_results[f"tinygrad CUDA=1 {prec_label}"] = s
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()

        elif backend == "hotpath":
            print(f"  [C Hot Path {prec_label}]")
            try:
                times, pc, result = bench_c_hot_path(
                    weights, use_fp16, args.warmup, args.iters)
                if times is not None:
                    s = print_result(f"C Hot Path ({pc:,} params)", times)
                    all_results[f"C Hot Path {prec_label}"] = s
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()

        elif backend == "trt":
            # GPU-only timing
            print(f"  [TensorRT {prec_label} (GPU-only timing)]")
            try:
                times, pc, result = bench_tensorrt(
                    weights, use_fp16, args.warmup, args.iters)
                if times is not None:
                    s = print_result(f"TensorRT GPU-only ({pc:,} params)", times)
                    all_results[f"TensorRT GPU-only {prec_label}"] = s
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()

            # Full round-trip timing
            print(f"\n  [TensorRT {prec_label} (full H2D+compute+D2H)]")
            try:
                times_full, _, _ = bench_tensorrt_full(
                    weights, use_fp16, args.warmup, args.iters)
                if times_full is not None:
                    s = print_result(f"TensorRT full round-trip ({pc:,} params)", times_full)
                    all_results[f"TensorRT full {prec_label}"] = s
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()

        elif backend == "trt-zerocopy":
            # Zero-copy GPU-only timing
            print(f"  [TRT Zero-Copy {prec_label} (GPU-only timing)]")
            try:
                times, pc, result = bench_tensorrt_zerocopy(
                    weights, use_fp16, args.warmup, args.iters)
                if times is not None:
                    s = print_result(f"TRT Zero-Copy GPU-only ({pc:,} params)", times)
                    all_results[f"TRT Zero-Copy GPU-only {prec_label}"] = s
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()

            # Zero-copy full round-trip
            print(f"\n  [TRT Zero-Copy {prec_label} (full round-trip)]")
            try:
                times_full, _, _ = bench_tensorrt_zerocopy_full(
                    weights, use_fp16, args.warmup, args.iters)
                if times_full is not None:
                    s = print_result(f"TRT Zero-Copy full ({pc:,} params)", times_full)
                    all_results[f"TRT Zero-Copy full {prec_label}"] = s
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()

        elif backend == "trt-graph":
            print(f"  [TRT + CUDA Graph {prec_label}]")
            try:
                times, pc, result = bench_tensorrt_cuda_graph(
                    weights, use_fp16, args.warmup, args.iters)
                if times is not None:
                    s = print_result(f"TRT CUDA Graph ({pc:,} params)", times)
                    all_results[f"TRT CUDA Graph {prec_label}"] = s
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()

        elif backend == "trt-zc-graph":
            print(f"  [TRT Zero-Copy + CUDA Graph {prec_label}]")
            try:
                times, pc, result = bench_tensorrt_zerocopy_cuda_graph(
                    weights, use_fp16, args.warmup, args.iters)
                if times is not None:
                    s = print_result(f"TRT ZC+Graph ({pc:,} params)", times)
                    all_results[f"TRT ZC+Graph {prec_label}"] = s
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()

        elif backend == "pytorch":
            print(f"  [PyTorch {prec_label}]")
            try:
                pt_results, pc, result = bench_pytorch(
                    weights, use_fp16, args.warmup, args.iters)
                for variant, times in pt_results.items():
                    label = f"PyTorch {variant} ({pc:,} params)"
                    s = print_result(label, times)
                    all_results[f"PyTorch {variant} {prec_label}"] = s
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()

    # Summary
    if len(all_results) > 1:
        print_summary_table(all_results, param_count)

    # Save results
    if args.save:
        save_path = Path(args.save)
        save_data = {
            "model": {
                "name": "Cioffi TCN (Learned Inertial Odometry)",
                "paper": "arXiv:2210.15287",
                "params": param_count,
                "input_shape": [1, INPUT_DIM, SEQ_LEN],
                "output_shape": [1, OUTPUT_DIM],
                "channels": NUM_CHANNELS,
                "kernel_size": KERNEL_SIZE,
            },
            "platform": {
                "device": "Jetson AGX Orin 64GB",
                "jetpack": "6",
                "cuda": "12.6",
            },
            "config": {
                "precision": args.precision,
                "iters": args.iters,
                "warmup": args.warmup,
            },
            "results": all_results,
        }
        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
