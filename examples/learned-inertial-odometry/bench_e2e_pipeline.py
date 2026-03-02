#!/usr/bin/env python3
"""bench_e2e_pipeline.py — End-to-end benchmark of the Cioffi et al. pipeline.

Reproduces Fig. 2 from the paper: the complete sensor-to-state loop:

  ┌────────┐     ┌─────┐     ┌──────┐
  │IMU data│────▶│Buffer│────▶│ TCN  │────▶ Δp (3-DoF displacement)
  └───┬────┘     └─────┘     └──────┘        │
      │                                       │
      ▼                                       ▼
  ┌────────┐                           ┌──────────┐
  │IMU Prop│──────────────────────────▶│EKF Update│──▶ R, v, p, b_a, b_g
  └────────┘                           └──────────┘

Fair comparison methodology:
  - EKF is their EXACT ImuMSCKF code (numba JIT, FEJ, Mahalanobis gating)
  - ONLY the TCN inference backend changes between runs
  - Pipeline params match their Blackbird config: 0.5s window, 100 Hz, 20 Hz updates

Pipeline per update cycle (at 20 Hz, every 50 ms):
  1. IMU propagation: 5 samples @ 100 Hz (numba-JIT'd propagation + covariance)
  2. Buffer fill: collect 50 timesteps of (gyro, thrust) → (1, 6, 50) tensor
  3. TCN inference: forward pass → Δp (1, 3)   ← THIS is the only thing that changes
  4. EKF update: Kalman gain + state correction  (their code, unchanged)

Backends for step 3 (TCN inference):
  - tinygrad NV=1 (with TinyJit + direct memmove via Tegra unified memory)
  - tinygrad NV=1 + C Hot Path (zero-Python GPU dispatch)
  - TensorRT (CUDA runtime + H2D/D2H copies)

Usage:
  NV=1 JITBEAM=2 python3 bench_e2e_pipeline.py --backend all       # compare all
  NV=1 JITBEAM=2 python3 bench_e2e_pipeline.py --backend nv
  NV=1 JITBEAM=2 python3 bench_e2e_pipeline.py --backend hotpath
  python3 bench_e2e_pipeline.py --backend trt
  NV=1 JITBEAM=2 python3 bench_e2e_pipeline.py --data real          # Blackbird dataset
"""
import os, sys, argparse, time, json, ctypes
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cioffi_tcn import (
    INPUT_DIM, OUTPUT_DIM, NUM_CHANNELS, KERNEL_SIZE, SEQ_LEN,
    generate_weights, build_tinygrad_tcn, export_onnx, _count_params,
)
from cioffi_ekf import (
    create_filter, USING_REAL_EKF, generate_imu_stream, load_blackbird_data,
)

SCRIPT_DIR = Path(__file__).parent
ONNX_DIR = SCRIPT_DIR / "onnx"

# Pipeline parameters (from model_net_parameters_net_blackbird.json)
IMU_FREQ = 100       # Hz (Blackbird dataset rate)
UPDATE_FREQ = 20     # Hz (TCN inference / EKF update rate)
WINDOW_TIME = 0.5    # seconds (their actual config: 50 samples @ 100 Hz)
IMU_SAMPLES_PER_UPDATE = IMU_FREQ // UPDATE_FREQ  # 5
# clone_every = imu_freq / update_freq = 5 samples between state clones
# update_distance_num_clone = window_time * update_freq = 10 clones between updates
UPDATE_DISTANCE_NUM_CLONE = int(WINDOW_TIME * UPDATE_FREQ)  # 10

assert int(WINDOW_TIME * IMU_FREQ) == SEQ_LEN, \
    f"Window {WINDOW_TIME}s @ {IMU_FREQ}Hz = {int(WINDOW_TIME * IMU_FREQ)}, expected {SEQ_LEN}"


# ═══════════════════════════════════════════════════════════════════════════════
# IMU ring buffer (matches their NetInputBuffer concept)
# ═══════════════════════════════════════════════════════════════════════════════

class IMURingBuffer:
    """Ring buffer for IMU data, maintains a sliding window of SEQ_LEN samples."""

    def __init__(self, seq_len=SEQ_LEN, n_channels=INPUT_DIM):
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.buf = np.zeros((n_channels, seq_len), dtype=np.float32)
        self.count = 0

    def push(self, gyro, thrust):
        """Add one IMU sample. gyro: (3,), thrust: (3,)."""
        sample = np.concatenate([gyro.flatten(), thrust.flatten()])
        # Shift left and insert at right
        self.buf[:, :-1] = self.buf[:, 1:]
        self.buf[:, -1] = sample
        self.count += 1

    def is_full(self):
        return self.count >= self.seq_len

    def get_tensor_fp16(self):
        """Return (1, 6, SEQ_LEN) as FP16 for TCN input."""
        return self.buf.astype(np.float16).reshape(1, self.n_channels, self.seq_len)


# ═══════════════════════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stats(times_us):
    a = np.asarray(times_us)
    return {
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "p99": float(np.percentile(a, 99)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "count": len(a),
    }


def print_stats(label, times_us):
    s = compute_stats(times_us)
    hz = 1e6 / s["median"] if s["median"] > 0 else 0
    print(f"  {label:35s}  median={s['median']:8.1f} µs  "
          f"p99={s['p99']:8.1f}  max={s['max']:9.1f}  {hz:8.0f} Hz")
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# TCN inference backends (thin wrappers for the pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

class TinygradNVBackend:
    """tinygrad NV=1 with direct memmove."""

    def __init__(self, weights, use_fp16=True):
        os.environ["NV"] = "1"
        from tinygrad import Tensor, dtypes, Device
        from tinygrad.engine.jit import TinyJit

        self.Device = Device
        model, self.param_count = build_tinygrad_tcn(weights, use_fp16=use_fp16)
        tg_dtype = dtypes.float16 if use_fp16 else dtypes.float32
        self.np_dtype = np.float16 if use_fp16 else np.float32

        self.static_x = Tensor.zeros(1, INPUT_DIM, SEQ_LEN, dtype=tg_dtype).contiguous().realize()
        self.static_out = Tensor.zeros(1, OUTPUT_DIM, dtype=tg_dtype).contiguous().realize()

        self.in_addr = self.static_x._buffer()._buf.cpu_view().addr
        self.out_addr = self.static_out._buffer()._buf.cpu_view().addr
        self.in_nbytes = INPUT_DIM * SEQ_LEN * (2 if use_fp16 else 4)
        self.out_nbytes = OUTPUT_DIM * (2 if use_fp16 else 4)

        @TinyJit
        def _run():
            self.static_out.assign(model(self.static_x)).realize()
        self._run = _run

    def warmup(self, n=30):
        dummy = np.zeros((1, INPUT_DIM, SEQ_LEN), dtype=self.np_dtype)
        for _ in range(n):
            ctypes.memmove(self.in_addr, dummy.ctypes.data, self.in_nbytes)
            self._run()
            self.Device["NV"].synchronize()

    def infer(self, input_fp16):
        """Run TCN inference. input_fp16: (1, 6, 200) numpy array."""
        ctypes.memmove(self.in_addr, input_fp16.ctypes.data, self.in_nbytes)
        self._run()
        self.Device["NV"].synchronize()
        result = np.zeros(OUTPUT_DIM, dtype=self.np_dtype)
        ctypes.memmove(result.ctypes.data, self.out_addr, self.out_nbytes)
        return result

    def sync(self):
        self.Device["NV"].synchronize()


class TensorRTBackend:
    """TensorRT inference backend."""

    def __init__(self, weights, use_fp16=True):
        ONNX_DIR.mkdir(exist_ok=True)
        prec = "fp16" if use_fp16 else "fp32"
        self.onnx_path = str(ONNX_DIR / f"cioffi_tcn_{prec}.onnx")
        self.engine_path = str(ONNX_DIR / f"cioffi_tcn_{prec}.engine")
        self.np_dtype = np.float16 if use_fp16 else np.float32

        if not os.path.exists(self.onnx_path):
            export_onnx(self.onnx_path, weights, use_fp16=False)

        try:
            import tensorrt as trt
        except ImportError:
            trt_path = os.environ.get("TENSORRT_PATH", "")
            trt_lib = os.path.join(trt_path, "lib", "libnvinfer.so")
            if os.path.exists(trt_lib):
                ctypes.CDLL(trt_lib, mode=ctypes.RTLD_GLOBAL)
            import tensorrt as trt

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        if not os.path.exists(self.engine_path):
            print(f"  Building TRT engine ({prec})...")
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            with open(self.onnx_path, "rb") as f:
                assert parser.parse(f.read())
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            if use_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            config.clear_flag(trt.BuilderFlag.TF32)
            engine_bytes = builder.build_serialized_network(network, config)
            with open(self.engine_path, "wb") as f:
                f.write(engine_bytes)

        runtime = trt.Runtime(TRT_LOGGER)
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # CUDA buffers
        self.cudart = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
        input_nbytes = INPUT_DIM * SEQ_LEN * 4  # FP32 for TRT IO
        output_nbytes = OUTPUT_DIM * 4
        self.input_nbytes = input_nbytes
        self.output_nbytes = output_nbytes

        self.d_input = ctypes.c_void_p()
        self.d_output = ctypes.c_void_p()
        self.cudart.cudaMalloc(ctypes.byref(self.d_input), input_nbytes)
        self.cudart.cudaMalloc(ctypes.byref(self.d_output), output_nbytes)
        self.stream = ctypes.c_void_p()
        self.cudart.cudaStreamCreate(ctypes.byref(self.stream))

        self.context.set_tensor_address("imu_data", self.d_input.value)
        self.context.set_tensor_address("displacement", self.d_output.value)

        self.h_input = np.zeros((1, INPUT_DIM, SEQ_LEN), dtype=np.float32)
        self.h_output = np.zeros((1, OUTPUT_DIM), dtype=np.float32)
        self.param_count = _count_params()

    def warmup(self, n=10):
        for _ in range(n):
            self.cudart.cudaMemcpyAsync(
                self.d_input, self.h_input.ctypes.data,
                self.input_nbytes, 1, self.stream)
            self.context.execute_async_v3(self.stream.value)
            self.cudart.cudaStreamSynchronize(self.stream)

    def infer(self, input_fp16):
        """Run TRT inference. input_fp16: (1, 6, 200) numpy FP16 array."""
        self.h_input[:] = input_fp16.astype(np.float32)
        self.cudart.cudaMemcpyAsync(
            self.d_input, self.h_input.ctypes.data,
            self.input_nbytes, 1, self.stream)
        self.context.execute_async_v3(self.stream.value)
        self.cudart.cudaMemcpyAsync(
            self.h_output.ctypes.data, self.d_output,
            self.output_nbytes, 2, self.stream)
        self.cudart.cudaStreamSynchronize(self.stream)
        return self.h_output.flatten().astype(self.np_dtype)

    def sync(self):
        self.cudart.cudaStreamSynchronize(self.stream)

    def __del__(self):
        try:
            self.cudart.cudaFree(self.d_input)
            self.cudart.cudaFree(self.d_output)
            self.cudart.cudaStreamDestroy(self.stream)
        except Exception:
            pass


class CHotPathBackend:
    """C Hot Path backend — replays HCQGraph from C."""

    def __init__(self, weights, use_fp16=True):
        os.environ["NV"] = "1"
        from tinygrad import Tensor, dtypes, Device
        from tinygrad.engine.jit import TinyJit

        self.Device = Device
        self.np_dtype = np.float16 if use_fp16 else np.float32

        hp_dir = os.environ.get("HOT_PATH_DIR",
            str(SCRIPT_DIR.parent / "control-loop" / "hot_path"))
        self.so_path = os.path.join(hp_dir, "hot_path.so")
        if not os.path.exists(self.so_path):
            raise FileNotFoundError(f"hot_path.so not found at {self.so_path}")

        sys.path.insert(0, hp_dir)
        from export_graph import export_hot_path_config
        self._export = export_hot_path_config

        model, self.param_count = build_tinygrad_tcn(weights, use_fp16=use_fp16)
        tg_dtype = dtypes.float16 if use_fp16 else dtypes.float32

        self.static_x = Tensor.zeros(1, INPUT_DIM, SEQ_LEN, dtype=tg_dtype).contiguous().realize()
        self.static_out = Tensor.zeros(1, OUTPUT_DIM, dtype=tg_dtype).contiguous().realize()

        self.in_addr = self.static_x._buffer()._buf.cpu_view().addr
        self.out_addr = self.static_out._buffer()._buf.cpu_view().addr
        self.in_nbytes = INPUT_DIM * SEQ_LEN * (2 if use_fp16 else 4)
        self.out_nbytes = OUTPUT_DIM * (2 if use_fp16 else 4)

        @TinyJit
        def _run():
            self.static_out.assign(model(self.static_x)).realize()
        self._run = _run

        # Will be set after warmup
        self._lib = None
        self._c_cfg = None

    def warmup(self, n=30):
        dummy = np.zeros((1, INPUT_DIM, SEQ_LEN), dtype=self.np_dtype)
        for _ in range(n):
            ctypes.memmove(self.in_addr, dummy.ctypes.data, self.in_nbytes)
            self._run()
            self.Device["NV"].synchronize()

        # Export HCQGraph and prepare C hot path
        print("  Exporting HCQGraph for C Hot Path...")
        cfg = self._export(self._run, self.Device["NV"],
                           self.static_x._buffer(), self.static_out._buffer())

        self._lib = ctypes.CDLL(self.so_path)
        self._c_cfg = _build_c_config(cfg)
        self._lib.hot_path_init(ctypes.byref(self._c_cfg))

        # C warmup
        dummy_in = np.zeros(self.in_nbytes, dtype=np.uint8)
        dummy_out = np.zeros(self.out_nbytes, dtype=np.uint8)
        t = (ctypes.c_uint64 * 1)()
        for _ in range(5):
            self._lib.hot_path_benchmark(
                ctypes.byref(self._c_cfg),
                dummy_in.ctypes.data_as(ctypes.c_void_p),
                dummy_out.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_uint32(1), t)

    def infer(self, input_fp16):
        """Run C hot path inference."""
        sensor = np.ascontiguousarray(input_fp16.flatten().view(np.uint8))
        action = np.zeros(self.out_nbytes, dtype=np.uint8)
        t = (ctypes.c_uint64 * 1)()
        self._lib.hot_path_benchmark(
            ctypes.byref(self._c_cfg),
            sensor.ctypes.data_as(ctypes.c_void_p),
            action.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_uint32(1), t)
        return np.frombuffer(action, dtype=self.np_dtype)

    def sync(self):
        self.Device["NV"].synchronize()


class PyTorchCUDABackend:
    """PyTorch CUDA backend — their original inference code path."""

    def __init__(self, weights, use_fp16=True):
        import torch
        from cioffi_tcn import build_pytorch_tcn
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA not available (CPU-only torch?)")
        self.device = torch.device("cuda:0")
        self.np_dtype = np.float16 if use_fp16 else np.float32
        self.torch_dtype = torch.float16 if use_fp16 else torch.float32

        model, self.param_count = build_pytorch_tcn(weights, use_fp16=use_fp16)
        self.model = model.to(self.device).eval()

        # Pre-allocate CUDA tensors for CUDA Graphs
        self.static_input = torch.zeros(
            1, INPUT_DIM, SEQ_LEN, dtype=self.torch_dtype, device=self.device)
        self.static_output = torch.zeros(
            1, OUTPUT_DIM, dtype=self.torch_dtype, device=self.device)

        # CUDA Graph capture (after warmup)
        self._graph = None
        self._stream = torch.cuda.Stream()

    def warmup(self, n=30):
        import torch
        # Eager warmup
        for _ in range(n):
            with torch.no_grad():
                self.static_output.copy_(self.model(self.static_input))
        torch.cuda.synchronize()

        # Capture CUDA Graph for max throughput
        self._graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self._graph, stream=self._stream):
            self.static_output.copy_(self.model(self.static_input))
        torch.cuda.synchronize()

    def infer(self, input_fp16):
        """Run PyTorch CUDA inference."""
        import torch
        inp = torch.from_numpy(input_fp16.astype(
            np.float16 if self.torch_dtype == torch.float16 else np.float32
        )).to(self.device)
        self.static_input.copy_(inp)
        self._graph.replay()
        torch.cuda.synchronize()
        return self.static_output.detach().cpu().numpy().flatten().astype(self.np_dtype)

    def sync(self):
        import torch
        torch.cuda.synchronize()


def _build_c_config(cfg):
    """Build ctypes struct matching hot_path_config_t."""

    class PatchEntry(ctypes.Structure):
        _fields_ = [
            ("addr", ctypes.c_uint64),
            ("var_type", ctypes.c_uint32),
            ("mask", ctypes.c_uint32),
        ]

    MAX_PATCHES = 512
    MAX_QUEUE_SIGNALS = 16

    class HotPathConfig(ctypes.Structure):
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

    c = HotPathConfig()
    for field in ["input_buf_addr", "output_buf_addr", "input_size", "output_size",
                  "gpfifo_ring_addr", "gpfifo_gpput_addr", "gpfifo_entries_count",
                  "gpfifo_token", "gpfifo_put_value", "cmdq_gpu_addr", "cmdq_len_u32",
                  "gpu_mmio_addr", "timeline_signal_addr", "timeline_value",
                  "last_tl_value", "kick_signal_addr", "kickoff_value"]:
        setattr(c, field, cfg[field])

    sigs = cfg["queue_signal_addrs"]
    c.num_queue_signals = len(sigs)
    for i, a in enumerate(sigs):
        c.queue_signal_addrs[i] = a

    patches = cfg["patches"]
    c.num_patches = len(patches)
    for i, p in enumerate(patches):
        c.patches[i].addr = p["addr"]
        c.patches[i].var_type = p["var_type"]
        c.patches[i].mask = p["mask"]

    return c


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end pipeline benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(backend, ts, gyro, accel, thrust, n_updates, warmup_updates=10):
    """Run the full Cioffi pipeline and return per-update timing breakdown.

    Uses their REAL ImuMSCKF (with numba JIT, FEJ) when available.
    Only the TCN inference backend is swapped.

    Timing matches their pipeline:
      - IMU arrives at 100 Hz → propagate() at each sample
      - Every 5th sample: augment (clone) the state
      - Every 10th clone (= 50 samples = 0.5s): run TCN + EKF update

    Returns:
      times_total_us, times_imu_us, times_tcn_us, times_ekf_us, final_state_dict
    """
    ekf, is_real = create_filter()
    dt_us = int(1e6 / IMU_FREQ)  # 10000 µs at 100 Hz

    # Initialize filter with first accel measurement
    t_us = 0
    ba_init = np.zeros((3, 1))
    bg_init = np.zeros((3, 1))
    ekf.initialize(accel[0].reshape(3, 1), t_us, ba_init, bg_init)

    imu_buf = IMURingBuffer(seq_len=SEQ_LEN, n_channels=INPUT_DIM)

    # Fill the buffer initially (need SEQ_LEN samples before first inference)
    clone_counter = 0  # counts samples since last clone
    for i in range(SEQ_LEN):
        t_us += dt_us
        clone_counter += 1
        do_augment = (clone_counter % IMU_SAMPLES_PER_UPDATE == 0)
        t_aug = t_us if do_augment else None
        ekf.propagate(
            accel[i].reshape(3, 1),
            gyro[i].reshape(3, 1),
            t_us, t_augmentation_us=t_aug)
        imu_buf.push(gyro[i], thrust[i])

    imu_idx = SEQ_LEN

    def _do_one_update_cycle():
        """Run one update cycle: 5 IMU props + TCN + EKF update. Returns timing."""
        nonlocal imu_idx, t_us, clone_counter

        if imu_idx + IMU_SAMPLES_PER_UPDATE > len(ts):
            imu_idx = SEQ_LEN  # wrap around
            # Reset time to avoid timestamp issues
            t_us = imu_idx * dt_us

        t_total_start = time.perf_counter()

        # ── Step 1: IMU propagation (5 steps at 100 Hz) ──
        t_imu_start = time.perf_counter()
        for j in range(IMU_SAMPLES_PER_UPDATE):
            idx = imu_idx + j
            t_us += dt_us
            clone_counter += 1
            do_augment = (clone_counter % IMU_SAMPLES_PER_UPDATE == 0)
            t_aug = t_us if do_augment else None
            ekf.propagate(
                accel[idx].reshape(3, 1),
                gyro[idx].reshape(3, 1),
                t_us, t_augmentation_us=t_aug)
            imu_buf.push(gyro[idx], thrust[idx])
        t_imu_end = time.perf_counter()
        imu_idx += IMU_SAMPLES_PER_UPDATE

        # ── Step 2: TCN inference ──
        tcn_input = imu_buf.get_tensor_fp16()
        t_tcn_start = time.perf_counter()
        dp = backend.infer(tcn_input)
        t_tcn_end = time.perf_counter()

        # ── Step 3: EKF update (using their learnt_model_update + apply_update) ──
        t_ekf_start = time.perf_counter()
        N = ekf.state.N
        if N > UPDATE_DISTANCE_NUM_CLONE:
            t_oldest = ekf.state.si_timestamps_us[N - UPDATE_DISTANCE_NUM_CLONE - 1]
            t_newest = ekf.state.si_timestamps_us[-1]
            meas = dp.astype(np.float64).reshape(3, 1)
            meas_cov = np.eye(3)
            is_valid, innovation, H, R = ekf.learnt_model_update(
                meas, meas_cov, t_oldest, t_newest)
            if is_valid:
                ekf.apply_update(innovation.reshape(3, 1), H, R)
            # Marginalize old clones
            oldest_idx = ekf.state.si_timestamps_us.index(t_oldest)
            ekf.marginalize(oldest_idx)
        t_ekf_end = time.perf_counter()

        t_total_end = time.perf_counter()

        return (
            (t_total_end - t_total_start) * 1e6,
            (t_imu_end - t_imu_start) * 1e6,
            (t_tcn_end - t_tcn_start) * 1e6,
            (t_ekf_end - t_ekf_start) * 1e6,
        )

    # Warmup phase
    for _ in range(warmup_updates):
        _do_one_update_cycle()

    # Benchmark phase
    times_total, times_imu, times_tcn, times_ekf = [], [], [], []
    for _ in range(n_updates):
        t_total, t_imu, t_tcn, t_ekf = _do_one_update_cycle()
        times_total.append(t_total)
        times_imu.append(t_imu)
        times_tcn.append(t_tcn)
        times_ekf.append(t_ekf)

    final_state = {
        'R': ekf.state.s_R.copy(),
        'v': ekf.state.s_v.copy(),
        'p': ekf.state.s_p.copy(),
        'bg': ekf.state.s_bg.copy(),
        'ba': ekf.state.s_ba.copy(),
    }
    return times_total, times_imu, times_tcn, times_ekf, final_state


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="End-to-end Cioffi pipeline benchmark")
    parser.add_argument("--backend", type=str, default="nv",
                        choices=["nv", "hotpath", "trt", "pytorch", "all"],
                        help="TCN inference backend (default: nv)")
    parser.add_argument("--data", type=str, default="sim",
                        choices=["sim", "real"],
                        help="IMU data source: sim=simulated, real=Blackbird dataset")
    parser.add_argument("--iters", type=int, default=2000,
                        help="Number of update cycles to benchmark (default: 2000)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup update cycles (default: 20)")
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["fp16", "fp32"])
    parser.add_argument("--save", type=str, default=None,
                        help="Save results to JSON")
    args = parser.parse_args()

    use_fp16 = args.precision == "fp16"
    prec_label = "FP16" if use_fp16 else "FP32"
    param_count = _count_params()
    ekf_label = "ImuMSCKF (numba JIT, FEJ)" if USING_REAL_EKF else "Fallback (no numba)"

    print("=" * 100)
    print("Cioffi et al. — End-to-End Pipeline Benchmark (Fig. 2)")
    print(f"  Paper: 'Learned Inertial Odometry for Autonomous Drone Racing' (RAL 2023)")
    print(f"  Pipeline: IMU Prop ({IMU_SAMPLES_PER_UPDATE}×) → Buffer → TCN → EKF Update")
    print(f"  TCN: {param_count:,} params, input=(1,{INPUT_DIM},{SEQ_LEN}), output=(1,{OUTPUT_DIM})")
    print(f"  EKF: {ekf_label}")
    print(f"  IMU rate: {IMU_FREQ} Hz | Update rate: {UPDATE_FREQ} Hz | Window: {WINDOW_TIME}s")
    print(f"  Budget per update: {1e6/UPDATE_FREQ:,.0f} µs ({UPDATE_FREQ} Hz)")
    print(f"  Precision: {prec_label} | Iterations: {args.iters} | Data: {args.data}")
    print("=" * 100)

    # Load IMU data
    if args.data == "real":
        dataset_path = (SCRIPT_DIR.parent.parent /
            "external/learned_inertial_model_odometry/datasets/Blackbird/"
            "clover/yawForward/maxSpeed5p0/test/data.hdf5")
        if not dataset_path.exists():
            print(f"  ERROR: Dataset not found at {dataset_path}")
            print("  Falling back to simulated data.")
            args.data = "sim"
        else:
            print(f"  Loading Blackbird test dataset...")
            ts, gyro, accel, thrust = load_blackbird_data(str(dataset_path))
            print(f"  Loaded {len(ts)} IMU samples ({len(ts)/IMU_FREQ:.1f}s)")

    if args.data == "sim":
        n_seconds = max(30, (args.iters + args.warmup + SEQ_LEN) / UPDATE_FREQ + 5)
        print(f"  Generating {n_seconds:.0f}s of simulated IMU data...")
        ts, gyro, accel, thrust = generate_imu_stream(n_seconds, IMU_FREQ)
        print(f"  Generated {len(ts)} IMU samples")

    weights = generate_weights()

    # Determine backends to run
    if args.backend == "all":
        backend_names = ["nv", "hotpath", "trt", "pytorch"]
    else:
        backend_names = [args.backend]

    all_results = {}  # name → {times_total, times_imu, times_tcn, times_ekf, state}

    for bname in backend_names:
        print(f"\n{'─'*100}")
        print(f"  Backend: {bname}")
        print(f"{'─'*100}")

        try:
            backend = _create_backend(bname, weights, use_fp16)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  Warming up...")
        backend.warmup()

        print(f"  Running {args.iters} update cycles...")
        t_total, t_imu, t_tcn, t_ekf, state = \
            run_pipeline(backend, ts, gyro, accel, thrust, args.iters, args.warmup)

        all_results[bname] = {
            "times_total": t_total, "times_imu": t_imu,
            "times_tcn": t_tcn, "times_ekf": t_ekf, "state": state,
        }

        # Print per-backend breakdown
        s_total = print_stats("Total (end-to-end)", t_total)
        s_tcn   = print_stats(f"  TCN inference ({bname})", t_tcn)
        s_imu   = print_stats(f"  IMU propagation ({IMU_SAMPLES_PER_UPDATE}×)", t_imu)
        s_ekf   = print_stats("  EKF update", t_ekf)

        budget_us = 1e6 / UPDATE_FREQ
        headroom = budget_us / s_total["median"] if s_total["median"] > 0 else 0
        print(f"  Headroom: {headroom:.1f}x ({'✓ fits' if headroom >= 1 else '✗ TOO SLOW'})")

        # Cleanup GPU state between backends (important for --backend all)
        del backend

    # ── Comparison table (when multiple backends) ──
    if len(all_results) > 1:
        _print_comparison_table(all_results, args.iters, prec_label)

    # Save
    if args.save:
        save_data = {
            "pipeline": "Cioffi et al. Learned Inertial Odometry (Fig. 2)",
            "ekf": ekf_label,
            "precision": args.precision,
            "data_source": args.data,
            "imu_freq": IMU_FREQ,
            "update_freq": UPDATE_FREQ,
            "window_time": WINDOW_TIME,
            "imu_samples_per_update": IMU_SAMPLES_PER_UPDATE,
            "tcn_params": param_count,
            "iters": args.iters,
            "backends": {
                name: {
                    "total": compute_stats(r["times_total"]),
                    "imu_propagation": compute_stats(r["times_imu"]),
                    "tcn_inference": compute_stats(r["times_tcn"]),
                    "ekf_update": compute_stats(r["times_ekf"]),
                    "raw_total": [float(x) for x in r["times_total"]],
                    "raw_tcn": [float(x) for x in r["times_tcn"]],
                    "raw_imu": [float(x) for x in r["times_imu"]],
                    "raw_ekf": [float(x) for x in r["times_ekf"]],
                }
                for name, r in all_results.items()
            },
        }
        with open(args.save, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {args.save}")


def _create_backend(name, weights, use_fp16):
    """Factory for TCN inference backends."""
    if name == "nv":
        return TinygradNVBackend(weights, use_fp16)
    elif name == "hotpath":
        return CHotPathBackend(weights, use_fp16)
    elif name == "trt":
        return TensorRTBackend(weights, use_fp16)
    elif name == "pytorch":
        return PyTorchCUDABackend(weights, use_fp16)
    else:
        raise ValueError(f"Unknown backend: {name}")


def _print_comparison_table(all_results, n_iters, prec_label):
    """Print side-by-side comparison of all backends."""
    budget_us = 1e6 / UPDATE_FREQ

    print(f"\n{'═'*100}")
    print(f"  COMPARISON — End-to-End Pipeline ({prec_label}, {n_iters} iters)")
    print(f"  EKF: {'Their real ImuMSCKF (numba JIT)' if USING_REAL_EKF else 'Fallback (no numba)'}")
    print(f"  Only the TCN backend changes — everything else is identical.")
    print(f"{'═'*100}")

    # Build rows sorted by total median
    rows = []
    for name, r in all_results.items():
        s_total = compute_stats(r["times_total"])
        s_tcn = compute_stats(r["times_tcn"])
        s_imu = compute_stats(r["times_imu"])
        s_ekf = compute_stats(r["times_ekf"])
        headroom = budget_us / s_total["median"] if s_total["median"] > 0 else 0
        pct_tcn = s_tcn["median"] / s_total["median"] * 100 if s_total["median"] > 0 else 0
        rows.append({
            "name": name,
            "total_med": s_total["median"],
            "total_p99": s_total["p99"],
            "tcn_med": s_tcn["median"],
            "tcn_p99": s_tcn["p99"],
            "imu_med": s_imu["median"],
            "ekf_med": s_ekf["median"],
            "headroom": headroom,
            "pct_tcn": pct_tcn,
        })

    rows.sort(key=lambda r: r["total_med"])
    best_total = rows[0]["total_med"]
    best_tcn = min(r["tcn_med"] for r in rows)

    BACKEND_LABELS = {
        "nv":      "tinygrad NV=1   (zero-copy, no CUDA runtime)",
        "hotpath": "C Hot Path      (zero-Python GPU dispatch)",
        "trt":     "TensorRT        (CUDA runtime + H2D/D2H)",
        "pytorch": "PyTorch CUDA    (their original code path)",
    }

    header = (f"  {'Backend':<50s}  {'Total':>8s}  {'TCN':>8s}  "
              f"{'IMU':>7s}  {'EKF':>7s}  {'TCN%':>5s}  "
              f"{'Headroom':>8s}  {'vs Best':>8s}")
    print(header)
    print(f"  {'─'*50}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*5}  {'─'*8}  {'─'*8}")

    for r in rows:
        label = BACKEND_LABELS.get(r["name"], r["name"])
        slowdown = r["total_med"] / best_total if best_total > 0 else 0
        status = "✓" if r["headroom"] >= 1 else "✗"
        print(f"  {label:<50s}  {r['total_med']:7.0f}µ  {r['tcn_med']:7.0f}µ  "
              f"{r['imu_med']:6.0f}µ  {r['ekf_med']:6.0f}µ  {r['pct_tcn']:4.0f}%  "
              f"{status} {r['headroom']:5.1f}x  {slowdown:6.2f}x")

    print()
    print(f"  Budget: {budget_us:,.0f} µs per update ({UPDATE_FREQ} Hz)")
    print(f"  Best total: {best_total:,.0f} µs ({rows[0]['name']})")
    print(f"  Best TCN:   {best_tcn:,.0f} µs")

    # What-if at higher IMU rates (using best backend)
    best = rows[0]
    print(f"\n  What-if at higher IMU rates (using {best['name']}):")
    for imu_hz, prop_steps in [(200, 10), (400, 20), (1000, 50)]:
        est_imu = best["imu_med"] * (prop_steps / IMU_SAMPLES_PER_UPDATE)
        est_total = est_imu + best["tcn_med"] + best["ekf_med"]
        est_hz = 1e6 / est_total if est_total > 0 else 0
        status = "✓" if est_total < budget_us else "✗"
        print(f"    {status} IMU@{imu_hz}Hz ({prop_steps} prop/update): "
              f"~{est_total:.0f} µs → {est_hz:.0f} Hz potential update rate")

    print(f"\n{'═'*100}")


if __name__ == "__main__":
    main()
