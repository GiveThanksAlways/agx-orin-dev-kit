# Production Hot Path Benchmark Results

**Platform:** Jetson AGX Orin 64GB, JetPack 6, NixOS, CUDA 12.6  
**Tinygrad:** NV=1 backend, JITBEAM=2  
**Compiler:** clang 21.1.7, `-O3 -march=armv8.2-a+fp16 -mtune=cortex-a78ae`  
**Iterations:** 10,000 per test

---

## Full Results (7 MLP sizes)

| # | MLP Config | Params | Python NV=1 | C GPU Hot Path | C NEON FP16 | GPU vs Python | GPU vs NEON | Winner |
|---|-----------|--------|-------------|----------------|-------------|---------------|-------------|--------|
| 1 | 12→64→64→4 | 5,252 | 105.9 µs / 9.4 kHz | 45.3 µs / 22.1 kHz | **1.1 µs / 919 kHz** | 2.3x | NEON 41.6x | **NEON** |
| 2 | 12→128→128→4 | 18,692 | 105.9 µs / 9.4 kHz | 46.2 µs / 21.7 kHz | **2.9 µs / 340 kHz** | 2.3x | NEON 15.7x | **NEON** |
| 3 | 12→256→256→256→4 | 135,940 | 124.6 µs / 8.0 kHz | 64.3 µs / 15.5 kHz | **17.1 µs / 58.5 kHz** | 1.9x | NEON 3.8x | **NEON** |
| 4 | 12→512→512→4 | 271,364 | 119.4 µs / 8.4 kHz | 58.6 µs / 17.1 kHz | **36.9 µs / 27.1 kHz** | 2.0x | NEON 1.6x | **NEON** |
| 5 | 12→512→512→512→4 | 534,020 | 143.2 µs / 7.0 kHz | **55.7 µs / 17.9 kHz** | 70.4 µs / 14.2 kHz | 2.6x | GPU 1.3x | **GPU** ⚡ |
| 6 | 12→1024→1024→4 | 1,067,012 | 150.2 µs / 6.7 kHz | **57.8 µs / 17.3 kHz** | 134.5 µs / 7.4 kHz | 2.6x | GPU 2.3x | **GPU** ⚡ |
| 7 | 12→1024→1024→1024→4 | 2,116,612 | 203.8 µs / 4.9 kHz | **78.3 µs / 12.8 kHz** | 228.5 µs / 4.4 kHz | 2.6x | GPU 2.9x | **GPU** ⚡ |

All correctness checks pass — C GPU output matches Python NV=1, NEON matches within FP16 tolerance.

---

## The Three Zones

### 🟢 NEON Zone (< ~400K params): Use pure ARM NEON

For models under ~400K parameters, NEON is dramatically faster because the GPU's
~45 µs launch overhead (GPFifo ring write → doorbell → GPU scheduler → execution →
signal) dwarfs the actual compute. NEON has zero launch overhead.

| Params | NEON Latency | GPU Latency | NEON Advantage |
|--------|-------------|-------------|----------------|
| 5K | 1.1 µs | 45.3 µs | 41.6x faster |
| 18K | 2.9 µs | 46.2 µs | 15.7x faster |
| 135K | 17.1 µs | 64.3 µs | 3.8x faster |
| 268K | 36.9 µs | 58.6 µs | 1.6x faster |

### ⚡ GPU Sweet Spot (~500K–2M+ params): Use C GPU Hot Path

Above ~500K parameters, GPU compute throughput overwhelms the launch overhead.
The C hot path keeps the overhead minimal (~10 µs patching + submit), and the
Orin's 2048 CUDA cores process the matrix operations faster than 12 NEON cores.

| Params | GPU Latency | NEON Latency | GPU Advantage |
|--------|-------------|--------------|---------------|
| 534K | 55.7 µs | 70.4 µs | 1.3x faster |
| 1.1M | 57.8 µs | 134.5 µs | 2.3x faster |
| 2.1M | 78.3 µs | 228.5 µs | 2.9x faster |

### 🔵 Crossover (~350K–500K params): Either works

At ~268K params NEON still leads (1.6x). At ~530K params GPU takes over (1.3x).
The exact crossover is approximately **~400K parameters** on Orin AGX 64GB.

---

## Real-World Model Size Guide

### ~5K params (12→64→64→4) — NEON: 1.1 µs / 919 kHz
**Use cases:** PID controller replacement, rate gyro filter, single-joint servo,
thermal compensation loop, simple sensor calibration.

**Examples:** Quadcopter rate controller (gyro → motor), heated bed PID
replacement (temp → PWM), basic vibration damper, single-axis gimbal stabilizer.

These are barely beyond a PID. The learned weights capture nonlinear dynamics
(asymmetric actuator response, sensor biases) that a PID can't. At sub-microsecond
NEON latency, you can run this at the sensor's native rate (often 8–32 kHz for MEMS IMUs).

---

### ~18K params (12→128→128→4) — NEON: 2.9 µs / 340 kHz
**Use cases:** Learned hover controller, complementary filter replacement,
sensor fusion (accel + gyro → orientation), basic EKF replacement,
force-torque estimation for grippers.

**Examples:** Drone attitude controller (full 3-axis), robot arm joint controller
with friction compensation, active vibration isolation, balancing robot,
legged locomotion single-leg stance controller.

Two 128-wide hidden layers can learn surprisingly complex dynamics — enough
for full 3-DOF attitude control with wind disturbance rejection. Running at
340 kHz means you have headroom for multiple cascaded controllers.

---

### ~135K params (12→256→256→256→4) — NEON: 17.1 µs / 58.5 kHz
**Use cases:** Full attitude policy network, SLAM feature encoder, visual-inertial
state estimator, terrain-adaptive locomotion, multi-joint coordination.

**Examples:** Quadcopter full flight controller (attitude + position inner loop),
hexapod gait generator, drone landing on moving platform, robotic arm reaching
controller with obstacle awareness, autonomous submarine depth controller.

Three 256-wide layers can represent complex nonlinear policies. At 58.5 kHz
this is more than enough for most robotic control loops (typical servo rate
is 1–10 kHz). NEON is still 3.8x faster than GPU here.

---

### ~268K params (12→512→512→4) — NEON: 36.9 µs / 27.1 kHz
**Use cases:** Visual-inertial navigation, end-to-end landing, multi-sensor
fusion (camera features + IMU + lidar), dexterous manipulation policy,
model-predictive-control neural approximator.

**Examples:** Drone autonomous landing with visual servoing, robot hand in-hand
manipulation, whole-body humanoid balance controller, autonomous vehicle
low-level steering. Also MPC approximation for fast replanning.

This is the **last size where NEON wins** (1.6x over GPU). Beyond this size,
you should switch to the C GPU hot path.

---

### ~530K params (12→512→512→512→4) — ⚡ GPU: 55.7 µs / 17.9 kHz
**Use cases:** Rich policy networks, multi-modal fusion, learned dynamics
models, sim-to-real transfer policies, contact-rich manipulation.

**Examples:** Dexterous robot hand manipulation (in-hand object rotation),
quadruped locomotion over rough terrain (sim2real), drone racing policy
with opponent awareness, autonomous forklift navigation + manipulation.

**This is where GPU takes the lead.** Three 512-wide layers have enough
compute to overcome the GPU's launch overhead. The C hot path delivers
17.9 kHz — more than enough for 1 kHz servo loops plus safety margins.

---

### ~1.1M params (12→1024→1024→4) — ⚡ GPU: 57.8 µs / 17.3 kHz
**Use cases:** Path planner neural networks, obstacle avoidance policies,
large learned dynamics models, visuomotor policies (after CNN feature
extraction), multi-agent coordination.

**Examples:** Self-driving vehicle local planner, warehouse robot path
planning, aerial manipulation (drone + arm), multi-drone formation control,
deformable object manipulation (cloth, rope), robot soccer player.

The GPU sweet spot: 2.3x faster than NEON. At 17.3 kHz you have massive
headroom for control loops that typically run at 100–1000 Hz. The extra
capacity can run multiple models or leave CPU free for perception.

---

### ~2.1M params (12→1024→1024→1024→4) — ⚡ GPU: 78.3 µs / 12.8 kHz
**Use cases:** Large policy networks, world models, multi-task policies,
learned cost-to-go functions for MPC, full visuomotor loops, language-conditioned
manipulation primitives.

**Examples:** Humanoid whole-body controller, autonomous vehicle behavioral
planner, agricultural robot crop manipulation, construction robot task planner,
surgical robot tissue manipulation, large multi-agent swarm coordination.

GPU advantage grows to 2.9x. Even at 2M+ parameters, the C hot path
maintains 12.8 kHz — well above real-time requirements for most systems.
Python NV=1 would give only 4.9 kHz here, so the C hot path provides
**2.6x more headroom**.

---

## Architecture

```
Python NV=1 (105–204 µs):
  TinyJit.__call__   →  HCQGraph.__call__  →  _apply_var_vals  →  _submit_to_gpfifo  →  GPU
  ^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^
  Python dispatch         Python graph mgmt     Python sym eval     Ring write + doorbell
  (~60 µs overhead)       (~10 µs)              (~15 µs)            (~5 µs)

C GPU Hot Path (45–78 µs):
  memcpy_in  →  apply_patches  →  submit_gpfifo  →  kick  →  GPU  →  memcpy_out
  (<1 µs)       (<1 µs)           (<1 µs)           (<1 µs)  (40–75 µs)  (<1 µs)

C NEON MLP (1–229 µs):
  memcpy_in  →  NEON FP16 forward pass  →  memcpy_out
  (<1 µs)       (1–229 µs)                  (<1 µs)
```

**Key insight:** The C GPU hot path reduces Python overhead from ~60 µs to <2 µs.
What remains is pure GPU execution time, which scales with model size. For small
models, that GPU execution time (~40 µs) is still much larger than NEON's total
time (~1 µs). But NEON scales linearly (O(n²) per layer pair), while GPU parallelism
grows with model size — creating the crossover at ~400K params.

---

## Files

| File | Purpose |
|------|---------|
| `hot_path.h` | C header: config struct, patch types, API |
| `hot_path.c` | C GPU dispatch: GPFifo submit, signal wait, patch apply |
| `neon_mlp.h` | NEON MLP header: struct, init/forward/benchmark API |
| `neon_mlp.c` | ARM NEON FP16 MLP: 4×8 FMLA unrolled forward pass |
| `export_graph.py` | Extract HCQGraph internals → C config struct |
| `bench_hot_path.py` | Benchmark driver: 7 sizes × 3 approaches |
| `Makefile` | Build both .so files with clang |

## Build & Run

```bash
cd examples/tinygrad && nix develop
cd ../../examples/control-loop/hot_path

# Build
CC=clang make -j2

# Run benchmark
NV=1 JITBEAM=2 python3 bench_hot_path.py
```
