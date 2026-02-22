#!/usr/bin/env python3
"""Optimized Control-Loop Benchmark: Making NV=1 Win on Jetson AGX Orin.

Tests three tinygrad NV=1 approaches + two PyTorch baselines for a realistic
quadrotor inner-loop control cycle.

tinygrad approaches (--backend tinygrad):
  A) naive       — Tensor() + .numpy() every iteration (baseline)
  B) buffer_api  — Buffer.copyin/copyout (skips Tensor creation, still uses SDMA)
  C) direct_mem  — ctypes.memmove via Tegra unified memory cpu_view (NO SDMA)

PyTorch approaches (--backend pytorch):
  D) eager       — torch.from_numpy().cuda() + .cpu().numpy()
  E) cuda_graph  — CUDA Graph with static_input.copy_() + .cpu().numpy()

Usage:
  # tinygrad:
  cd examples/tinygrad && nix develop
  NV=1 JITBEAM=2 python3 ../control-loop/bench_nv_wins.py --backend tinygrad

  # pytorch:
  cd examples/control-loop && nix develop
  python3 bench_nv_wins.py --backend pytorch
"""
import os, sys, time, argparse, ctypes
import numpy as np

SEED        = 42
IN_DIM      = 12   # state: pos(3) + vel(3) + rpy(3) + pqr(3)
OUT_DIM     = 4    # desired: thrust + roll_rate + pitch_rate + yaw_rate
HIDDEN      = 128
WARMUP      = 20
N_ITERS     = 20000

# ── Deterministic weights (shared between frameworks) ───────────────────────
def generate_weights(in_dim=IN_DIM, out_dim=OUT_DIM, seed=SEED):
    rng = np.random.RandomState(seed)
    def _layer(fi, fo):
        s = 1.0 / np.sqrt(fi)
        return (rng.randn(fo, fi) * s).astype(np.float16), (rng.randn(fo) * s).astype(np.float16)
    w1, b1 = _layer(in_dim, HIDDEN)
    w2, b2 = _layer(HIDDEN, HIDDEN)
    w3, b3 = _layer(HIDDEN, out_dim)
    return dict(w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)

# ── Realistic CPU control components ────────────────────────────────────────
class ComplementaryFilter:
    """Simple attitude estimator from IMU data."""
    def __init__(self, alpha=0.98, dt=0.001):
        self.alpha, self.dt = alpha, dt
        self.roll = self.pitch = self.yaw = 0.0

    def update(self, accel, gyro):
        # Accel-based angles
        ax, ay, az = accel
        a_roll = np.arctan2(ay, az)
        a_pitch = np.arctan2(-ax, np.sqrt(ay*ay + az*az))
        # Integrate gyro
        self.roll = self.alpha * (self.roll + gyro[0]*self.dt) + (1-self.alpha) * a_roll
        self.pitch = self.alpha * (self.pitch + gyro[1]*self.dt) + (1-self.alpha) * a_pitch
        self.yaw += gyro[2] * self.dt
        return self.roll, self.pitch, self.yaw

class PIDController:
    """Rate PID for angular velocity commands."""
    def __init__(self, kp=4.0, ki=0.02, kd=0.3, dt=0.001):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.integral = np.zeros(3, dtype=np.float32)
        self.prev_error = np.zeros(3, dtype=np.float32)

    def step(self, desired_rates, actual_rates):
        error = desired_rates - actual_rates
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -0.5, 0.5)
        deriv = (error - self.prev_error) / self.dt
        self.prev_error = error.copy()
        return self.kp * error + self.ki * self.integral + self.kd * deriv

MOTOR_MIX = np.array([  # thrust, roll, pitch, yaw → motor1-4
    [ 1,  1,  1,  1],
    [ 1, -1,  1, -1],
    [ 1, -1, -1,  1],
    [ 1,  1, -1, -1],
], dtype=np.float32) * 0.25

def motor_mixing(thrust, pid_out):
    """Convert thrust + PID angular corrections → 4 motor commands."""
    cmd = np.array([thrust, pid_out[0], pid_out[1], pid_out[2]], dtype=np.float32)
    motors = MOTOR_MIX @ cmd
    return np.clip(motors, 0.0, 1.0)

def simulate_imu(rng, n):
    """Generate realistic IMU data: accel ≈ [0,0,9.81] + noise, gyro ≈ 0 + noise."""
    accel = np.tile([0.0, 0.0, 9.81], (n, 1)) + rng.randn(n, 3) * 0.3
    gyro = rng.randn(n, 3) * 0.05  # rad/s
    return accel.astype(np.float32), gyro.astype(np.float32)

def build_state_vector(pos, vel, rpy, gyro):
    """12-element state vector for the policy network."""
    return np.concatenate([pos, vel, rpy, gyro]).astype(np.float16).reshape(1, IN_DIM)

# ── Stats & printing ────────────────────────────────────────────────────────
def stats(data):
    a = np.asarray(data)
    return dict(mean=np.mean(a), median=np.median(a), std=np.std(a),
                min=np.min(a), max=np.max(a),
                p99=np.percentile(a, 99), p999=np.percentile(a, 99.9), count=len(a))

def print_row(label, data):
    s = stats(data)
    freq = 1e6 / s['median'] if s['median'] > 0 else 0
    print(f"  {label:45s} median={s['median']:7.1f} µs  mean={s['mean']:7.1f} µs  "
          f"std={s['std']:5.1f} µs  p99={s['p99']:7.1f} µs  max={s['max']:8.1f} µs  "
          f"freq={freq:6.0f} Hz")

def verify_output(label, result, reference, atol=0.1):
    """Check that the output is numerically close to reference."""
    if not np.allclose(result, reference, atol=atol, rtol=0.1):
        print(f"  WARNING: {label} output mismatch! got={result}, expected={reference}")
        return False
    return True

# ═══════════════════════════════════════════════════════════════════════════════
# TINYGRAD BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════
def run_tinygrad():
    os.environ.setdefault("NV", "1")
    from tinygrad import Tensor, Device, TinyJit, dtypes
    from tinygrad import nn as tg_nn
    from tinygrad.helpers import from_mv

    assert Device.DEFAULT == "NV", f"Device is '{Device.DEFAULT}', expected 'NV'. Set NV=1."

    print(f"Backend: tinygrad NV=1 (JITBEAM={os.environ.get('JITBEAM', 'default')})")
    print(f"Device:  {Device.DEFAULT}")
    print(f"Iters:   {N_ITERS}")
    print()

    dev = Device["NV"]
    weights = generate_weights()

    class ControlMLP:
        def __init__(self, w):
            self.l1 = tg_nn.Linear(IN_DIM, HIDDEN)
            self.l2 = tg_nn.Linear(HIDDEN, HIDDEN)
            self.l3 = tg_nn.Linear(HIDDEN, OUT_DIM)
            self.l1.weight, self.l1.bias = Tensor(w['w1']), Tensor(w['b1'])
            self.l2.weight, self.l2.bias = Tensor(w['w2']), Tensor(w['b2'])
            self.l3.weight, self.l3.bias = Tensor(w['w3']), Tensor(w['b3'])
        def __call__(self, x):
            return self.l3(self.l2(self.l1(x).relu()).relu())

    # Pre-generate sensor data
    rng = np.random.RandomState(123)
    accel_pool, gyro_pool = simulate_imu(rng, N_ITERS + WARMUP + 100)

    # Pre-compute state vectors for all iterations
    cf = ComplementaryFilter()
    pos = np.zeros(3, dtype=np.float32)
    vel = np.zeros(3, dtype=np.float32)
    state_vectors = []
    for i in range(N_ITERS + WARMUP + 100):
        rpy = cf.update(accel_pool[i], gyro_pool[i])
        sv = build_state_vector(pos, vel, np.array(rpy, dtype=np.float32), gyro_pool[i])
        state_vectors.append(sv)
    state_pool = np.array(state_vectors).reshape(-1, 1, IN_DIM)

    in_nbytes = IN_DIM * 2  # FP16
    out_nbytes = OUT_DIM * 2

    # ── Compute reference output for correctness checking ────────────────
    ref_model = ControlMLP(weights)
    ref_out = ref_model(Tensor(state_pool[0])).numpy().flatten()
    del ref_model

    # ── Approach A: Naive (Tensor + .numpy()) ────────────────────────────
    print("=" * 90)
    print("A) NAIVE: Tensor(data) → @TinyJit MLP → .numpy() each iteration")
    print("=" * 90)

    model_a = ControlMLP(weights)

    @TinyJit
    def run_a(x):
        return model_a(x).realize()

    for i in range(WARMUP):
        _ = run_a(Tensor(state_pool[i])).numpy()

    verify_output("naive", run_a(Tensor(state_pool[0])).numpy().flatten(), ref_out)

    pid = PIDController()
    times_a = []
    for i in range(N_ITERS):
        t0 = time.perf_counter_ns()

        # GPU inference (full round-trip)
        policy_out = run_a(Tensor(state_pool[WARMUP + i])).numpy().flatten().astype(np.float32)

        # CPU control: PID + motor mixing
        thrust = float(np.clip(policy_out[0], 0, 1))
        desired_rates = policy_out[1:4]
        pid_correction = pid.step(desired_rates, gyro_pool[WARMUP + i])
        motors = motor_mixing(thrust, pid_correction)

        t1 = time.perf_counter_ns()
        times_a.append((t1 - t0) / 1000.0)

    print_row("Naive full control cycle", times_a)
    print()

    # ── Approach B: Buffer API (copyin/copyout, no Tensor creation) ──────
    print("=" * 90)
    print("B) BUFFER API: Buffer.copyin() → @TinyJit MLP → Buffer.copyout()")
    print("   Avoids Tensor object creation, still uses SDMA DMA engine")
    print("=" * 90)

    model_b = ControlMLP(weights)
    static_xb = Tensor.zeros(1, IN_DIM).cast(dtypes.float16).contiguous().realize()
    static_ob = Tensor.zeros(1, OUT_DIM).cast(dtypes.float16).contiguous().realize()
    dev.synchronize()

    xb_buf = static_xb._buffer()
    ob_buf = static_ob._buffer()

    @TinyJit
    def run_b():
        static_ob.assign(model_b(static_xb)).realize()

    for i in range(WARMUP):
        xb_buf.copyin(memoryview(state_pool[i].tobytes()))
        run_b()
        dev.synchronize()

    # Verify
    xb_buf.copyin(memoryview(state_pool[0].tobytes()))
    run_b()
    dev.synchronize()
    result_mv = memoryview(bytearray(out_nbytes))
    ob_buf.copyout(result_mv)
    verify_output("buffer_api", np.frombuffer(bytes(result_mv), dtype=np.float16).astype(np.float32), ref_out)

    pid = PIDController()
    result_buf = bytearray(out_nbytes)
    result_mv = memoryview(result_buf)
    times_b = []
    for i in range(N_ITERS):
        t0 = time.perf_counter_ns()

        # H2D via Buffer.copyin (SDMA, no Tensor creation)
        xb_buf.copyin(memoryview(state_pool[WARMUP + i].tobytes()))
        # GPU inference
        run_b()
        dev.synchronize()
        # D2H via Buffer.copyout (SDMA, no .numpy())
        ob_buf.copyout(result_mv)
        policy_out = np.frombuffer(bytes(result_mv), dtype=np.float16).astype(np.float32)

        # CPU control
        thrust = float(np.clip(policy_out[0], 0, 1))
        desired_rates = policy_out[1:4]
        pid_correction = pid.step(desired_rates, gyro_pool[WARMUP + i])
        motors = motor_mixing(thrust, pid_correction)

        t1 = time.perf_counter_ns()
        times_b.append((t1 - t0) / 1000.0)

    print_row("Buffer API full control cycle", times_b)
    print()

    # ── Approach C: Direct Memory (Tegra unified memory, NO SDMA) ────────
    print("=" * 90)
    print("C) DIRECT MEMORY: ctypes.memmove via Tegra cpu_view (NO SDMA engine)")
    print("   Leverages Tegra unified memory: GPU buffers are CPU-accessible via mmap")
    print("=" * 90)

    model_c = ControlMLP(weights)
    static_xc = Tensor.zeros(1, IN_DIM).cast(dtypes.float16).contiguous().realize()
    static_oc = Tensor.zeros(1, OUT_DIM).cast(dtypes.float16).contiguous().realize()
    dev.synchronize()

    xc_buf = static_xc._buffer()
    oc_buf = static_oc._buffer()
    xc_hcq = xc_buf._buf
    oc_hcq = oc_buf._buf

    # Verify cpu_view is available (Tegra unified memory)
    assert xc_hcq.view is not None, "Input buffer has no cpu_view — not running on Tegra?"
    assert oc_hcq.view is not None, "Output buffer has no cpu_view — not running on Tegra?"

    # Get CPU-accessible addresses from the MMIOInterface
    in_cpu_addr = xc_hcq.cpu_view().addr
    out_cpu_addr = oc_hcq.cpu_view().addr

    @TinyJit
    def run_c():
        static_oc.assign(model_c(static_xc)).realize()

    # Warmup: use direct memmove for input, but verify TinyJit captures correctly
    for i in range(WARMUP):
        ctypes.memmove(in_cpu_addr, state_pool[i].ctypes.data, in_nbytes)
        run_c()
        dev.synchronize()

    # Verify correctness
    ctypes.memmove(in_cpu_addr, state_pool[0].ctypes.data, in_nbytes)
    run_c()
    dev.synchronize()
    result_c = np.empty(OUT_DIM, dtype=np.float16)
    ctypes.memmove(result_c.ctypes.data, out_cpu_addr, out_nbytes)
    verify_output("direct_mem", result_c.astype(np.float32), ref_out)

    pid = PIDController()
    result_np = np.empty(OUT_DIM, dtype=np.float16)
    times_c = []
    for i in range(N_ITERS):
        t0 = time.perf_counter_ns()

        # H2D: direct memmove to GPU buffer (Tegra unified memory, no SDMA)
        ctypes.memmove(in_cpu_addr, state_pool[WARMUP + i].ctypes.data, in_nbytes)

        # GPU inference via JIT graph replay
        run_c()
        dev.synchronize()

        # D2H: direct memmove from GPU buffer (Tegra unified memory, no SDMA)
        ctypes.memmove(result_np.ctypes.data, out_cpu_addr, out_nbytes)
        policy_out = result_np.astype(np.float32)

        # CPU control: PID + motor mixing
        thrust = float(np.clip(policy_out[0], 0, 1))
        desired_rates = policy_out[1:4]
        pid_correction = pid.step(desired_rates, gyro_pool[WARMUP + i])
        motors = motor_mixing(thrust, pid_correction)

        t1 = time.perf_counter_ns()
        times_c.append((t1 - t0) / 1000.0)

    print_row("Direct memory full control cycle", times_c)

    # Also time just the GPU part (memmove + JIT + sync + memmove, no CPU control)
    times_c_gpu = []
    for i in range(N_ITERS):
        t0 = time.perf_counter_ns()
        ctypes.memmove(in_cpu_addr, state_pool[WARMUP + i].ctypes.data, in_nbytes)
        run_c()
        dev.synchronize()
        ctypes.memmove(result_np.ctypes.data, out_cpu_addr, out_nbytes)
        t1 = time.perf_counter_ns()
        times_c_gpu.append((t1 - t0) / 1000.0)

    print_row("Direct memory GPU round-trip only", times_c_gpu)
    print()

    # ── Summary ──────────────────────────────────────────────────────────
    sa, sb, sc, scg = stats(times_a), stats(times_b), stats(times_c), stats(times_c_gpu)
    print("=" * 90)
    print("TINYGRAD NV=1 SUMMARY")
    print("=" * 90)
    print(f"  {'Approach':45s} {'Median µs':>10s} {'Freq Hz':>10s} {'Jitter µs':>10s}")
    print(f"  {'─'*45} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'A) Naive (Tensor + .numpy())':<45s} {sa['median']:10.1f} {1e6/sa['median']:10.0f} {sa['std']:10.1f}")
    print(f"  {'B) Buffer API (copyin/copyout, SDMA)':<45s} {sb['median']:10.1f} {1e6/sb['median']:10.0f} {sb['std']:10.1f}")
    print(f"  {'C) Direct memory (memmove, no SDMA)':<45s} {sc['median']:10.1f} {1e6/sc['median']:10.0f} {sc['std']:10.1f}")
    print(f"  {'   C-gpu) GPU round-trip only':<45s} {scg['median']:10.1f} {1e6/scg['median']:10.0f} {scg['std']:10.1f}")
    print()

    return dict(naive=sa, buffer_api=sb, direct_mem=sc, direct_mem_gpu=scg)


# ═══════════════════════════════════════════════════════════════════════════════
# PYTORCH BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════
def run_pytorch():
    import torch
    import torch.nn as nn

    assert torch.cuda.is_available(), "CUDA not available"

    print(f"Backend: PyTorch {torch.__version__}")
    print(f"Device:  {torch.cuda.get_device_name(0)}")
    print(f"Iters:   {N_ITERS}")
    print()

    weights = generate_weights()

    class TorchMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(IN_DIM, HIDDEN)
            self.l2 = nn.Linear(HIDDEN, HIDDEN)
            self.l3 = nn.Linear(HIDDEN, OUT_DIM)
        def forward(self, x):
            return self.l3(torch.relu(self.l2(torch.relu(self.l1(x)))))

    def load_w(model, w):
        with torch.no_grad():
            model.l1.weight.copy_(torch.from_numpy(w['w1']))
            model.l1.bias.copy_(torch.from_numpy(w['b1']))
            model.l2.weight.copy_(torch.from_numpy(w['w2']))
            model.l2.bias.copy_(torch.from_numpy(w['b2']))
            model.l3.weight.copy_(torch.from_numpy(w['w3']))
            model.l3.bias.copy_(torch.from_numpy(w['b3']))

    def sync(): torch.cuda.synchronize()

    # Pre-generate sensor data (identical to tinygrad)
    rng = np.random.RandomState(123)
    accel_pool, gyro_pool = simulate_imu(rng, N_ITERS + WARMUP + 100)
    cf = ComplementaryFilter()
    pos = np.zeros(3, dtype=np.float32)
    vel = np.zeros(3, dtype=np.float32)
    state_vectors = []
    for i in range(N_ITERS + WARMUP + 100):
        rpy = cf.update(accel_pool[i], gyro_pool[i])
        sv = build_state_vector(pos, vel, np.array(rpy, dtype=np.float32), gyro_pool[i])
        state_vectors.append(sv)
    state_pool = np.array(state_vectors).reshape(-1, 1, IN_DIM)

    with torch.inference_mode():
        # ── Approach D: Eager ────────────────────────────────────────────
        print("=" * 90)
        print("D) PYTORCH EAGER: torch.from_numpy().cuda() → model() → .cpu().numpy()")
        print("=" * 90)

        model_d = TorchMLP().cuda().half().eval()
        load_w(model_d, weights)

        for i in range(WARMUP):
            _ = model_d(torch.from_numpy(state_pool[i]).cuda())
            sync()

        pid = PIDController()
        times_d = []
        for i in range(N_ITERS):
            t0 = time.perf_counter_ns()

            inp = torch.from_numpy(state_pool[WARMUP + i]).cuda()
            out = model_d(inp)
            sync()
            policy_out = out.cpu().numpy().flatten().astype(np.float32)

            thrust = float(np.clip(policy_out[0], 0, 1))
            desired_rates = policy_out[1:4]
            pid_correction = pid.step(desired_rates, gyro_pool[WARMUP + i])
            motors = motor_mixing(thrust, pid_correction)

            t1 = time.perf_counter_ns()
            times_d.append((t1 - t0) / 1000.0)

        print_row("Eager full control cycle", times_d)
        del model_d
        torch.cuda.empty_cache()
        print()

        # ── Approach E: CUDA Graph ───────────────────────────────────────
        print("=" * 90)
        print("E) PYTORCH CUDA GRAPH: static_input.copy_() → graph.replay() → .cpu().numpy()")
        print("=" * 90)

        model_e = TorchMLP().cuda().half().eval()
        load_w(model_e, weights)

        static_input = torch.zeros(1, IN_DIM, device='cuda', dtype=torch.float16)
        for _ in range(WARMUP):
            _ = model_e(static_input)
            sync()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_output = model_e(static_input)
        sync()

        # Additional warmup with graph
        for i in range(WARMUP):
            static_input.copy_(torch.from_numpy(state_pool[i]).cuda())
            g.replay()
            sync()

        pid = PIDController()
        times_e = []
        for i in range(N_ITERS):
            t0 = time.perf_counter_ns()

            static_input.copy_(torch.from_numpy(state_pool[WARMUP + i]).cuda())
            g.replay()
            sync()
            policy_out = static_output.cpu().numpy().flatten().astype(np.float32)

            thrust = float(np.clip(policy_out[0], 0, 1))
            desired_rates = policy_out[1:4]
            pid_correction = pid.step(desired_rates, gyro_pool[WARMUP + i])
            motors = motor_mixing(thrust, pid_correction)

            t1 = time.perf_counter_ns()
            times_e.append((t1 - t0) / 1000.0)

        print_row("CUDA Graph full control cycle", times_e)

        # GPU-only timing (no CPU control)
        times_e_gpu = []
        for i in range(N_ITERS):
            t0 = time.perf_counter_ns()
            static_input.copy_(torch.from_numpy(state_pool[WARMUP + i]).cuda())
            g.replay()
            sync()
            _ = static_output.cpu().numpy()
            t1 = time.perf_counter_ns()
            times_e_gpu.append((t1 - t0) / 1000.0)

        print_row("CUDA Graph GPU round-trip only", times_e_gpu)
        del model_e
        torch.cuda.empty_cache()
        print()

    sd, se, seg = stats(times_d), stats(times_e), stats(times_e_gpu)
    print("=" * 90)
    print("PYTORCH SUMMARY")
    print("=" * 90)
    print(f"  {'Approach':45s} {'Median µs':>10s} {'Freq Hz':>10s} {'Jitter µs':>10s}")
    print(f"  {'─'*45} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'D) Eager (from_numpy + cuda + cpu)':<45s} {sd['median']:10.1f} {1e6/sd['median']:10.0f} {sd['std']:10.1f}")
    print(f"  {'E) CUDA Graph (copy + replay + cpu)':<45s} {se['median']:10.1f} {1e6/se['median']:10.0f} {se['std']:10.1f}")
    print(f"  {'   E-gpu) GPU round-trip only':<45s} {seg['median']:10.1f} {1e6/seg['median']:10.0f} {seg['std']:10.1f}")
    print()

    return dict(eager=sd, cuda_graph=se, cuda_graph_gpu=seg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['tinygrad', 'pytorch'], required=True)
    args = ap.parse_args()

    if args.backend == 'tinygrad':
        run_tinygrad()
    else:
        run_pytorch()

if __name__ == '__main__':
    main()
