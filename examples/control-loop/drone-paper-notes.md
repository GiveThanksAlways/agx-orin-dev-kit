# NV=1 TinyGrad × Real-Time Drone Control: Complete Knowledge Base v2

> **Purpose**: This document gives another AI (or engineer) everything needed to reason about where TinyGrad's NV=1 graph replay wins against TensorRT and other inference runtimes in the context of real-time drone control. It is the product of deep analysis of two papers and iterative brainstorming. Corrections from v1 are marked with ⚡.

---

## Table of Contents

1. [What NV=1 Actually Is (Corrected)](#what-nv1-actually-is)
2. [Paper Summaries](#paper-summaries)
3. [The CNN Bottleneck — Revisited](#the-cnn-bottleneck-revisited)
4. [Can We Replace VIO + CNN + Kalman + MLP?](#can-we-replace-the-whole-chain)
5. [Benchmarking Plan: NV=1 vs TensorRT](#benchmarking-plan)
6. [Where NV=1 Wins and Why](#where-nv1-wins)
7. [Hardware Context](#hardware-context)
8. [Honesty Checklist](#honesty-checklist)
9. [Key Numbers Reference](#key-numbers)

---

## What NV=1 Actually Is

⚡ **Critical correction from v1**: NV=1 is NOT "move everything to CPU." Here is what it actually does:

### The Mechanism

1. TinyGrad builds a computation graph (lazy evaluation).
2. On first execution, the graph compiles and runs on GPU (or whatever backend).
3. **NV=1 captures the entire execution trace** — every GPU kernel launch, memory allocation, buffer pointer.
4. It generates **~200 lines of C code** that replay exactly that sequence of GPU operations.
5. On subsequent calls, Python and TinyGrad are completely bypassed. The C code directly dispatches the same GPU kernels with the same shapes.

### What This Eliminates

- Python interpreter overhead (huge for small/medium models)
- TinyGrad framework dispatch (graph walking, scheduling, buffer management)
- JIT recompilation checks
- Memory allocation per-call (buffers are pre-allocated)
- Any dynamism — shapes must be static (hence "NV=1" — batch size 1, no variable dimensions)

### What This Does NOT Change

- The actual GPU kernel quality (same kernels as TinyGrad normally generates)
- The GPU compute time for large GEMM/convolution operations
- Memory bandwidth limitations
- GPU occupancy characteristics

### Key Implication

⚡ **NV=1 competes with TensorRT on the SAME hardware (GPU), not by moving work to CPU.** The question becomes: for which model types does eliminating framework overhead matter more than TensorRT's kernel-level optimizations (layer fusion, INT8 quantization, kernel auto-tuning)?

---

## Paper Summaries

### Paper 1: RTN-MPC (Salzmann et al., 2023)

**What**: Integrates neural network dynamics models into MPC optimization loops in real-time on embedded devices.

**Key trick**: Instead of differentiating through the full neural network at every SQP step, compute local linear approximations (Jacobians) in a batch on GPU, then pass them as parameters to the QP solver. This decouples model complexity from optimization time.

**Benchmark table (Table II) — critical reference**:

| Config | Params | Naive ARM Hz | RTN-MPC ARM Hz | RTN-MPC Jetson GPU Hz |
|--------|--------|-------------|----------------|----------------------|
| 2L×16 | 354 | 403 | 148 | 109 |
| 2L×128 | 17K | 20 | 135 | 107 |
| 5L×16 | 1.2K | 280 | 118 | 88 |
| 5L×128 | 67K | 6 | 85 | 63 |
| 12L×32 | 12K | 66 | 67 | 61 |
| 12L×512 | 182K | 2 | 67 | 61 |
| 20L×512 | 500K | <1 | 11 | 46 |
| 50L×512 | 13M | — | <1 | — |
| CNN ResNet-18 | 12M | — | <1 | 9 |

**⚠️ These Hz numbers are FULL MPC loop** (Jacobian batched computation + QP construction + QP solve + RK4 integration with 4 dynamics evaluations × N=10 nodes). The MLP forward pass is one component. For the "Naive" column, the MLP differentiation IS the bottleneck. For RTN-MPC, the QP solver often becomes the bottleneck instead.

**Hardware**: Jetson Xavier NX (ARM CPU + 384 CUDA cores + 48 Tensor Cores).

**Real-world result**: 82% tracking error reduction vs nominal MPC. GP-20 baseline crashes in naive integration. Their N-3-32 model on Jetson runs in real-time while naive crashes.

### Paper 2: Swift (Kaufmann et al., Nature 2023)

**What**: Autonomous drone racing system that beats world champion human pilots using onboard sensors only.

**Full pipeline**:
```
IMU (200 Hz) ──→ VIO (100 Hz, RealSense T265 onboard ASIC)
                      ↓
Camera (30 Hz) → Gate CNN (30 Hz, U-Net on TX2 GPU, 40ms) → IPPE pose
                      ↓                                        ↓
                  Kalman filter (100 Hz) ← ─── ─── ─── ─── ───┘
                      ↓
                  MLP policy (100 Hz, 2×128, TX2 CPU, 8ms) → thrust + body rates
                      ↓
                  Betaflight PID (1 kHz, STM32 @ 216 MHz) → motor PWM
```

**MLP details**:
- 2 hidden layers × 128 neurons, LeakyReLU (slope 0.2)
- Input: 31 dimensions (position 3 + velocity 3 + rotation matrix 9 = 15 for state, + 4 gate corners × 3 = 12 for gate pose, + 4 for previous action)
- Output: 4 dimensions (mass-normalized collective thrust + 3 body rates)
- ~21K parameters
- Trained with PPO (on-policy), 10⁸ environment steps, 50 min on i9+RTX3090
- Fine-tuned with real-world residual models (Gaussian process for perception noise, KNN for dynamics residual, from ~50s of flight data)

**Gate detector CNN details**:
- 6-level U-Net architecture
- Filter counts per level: (8, 16, 16, 16, 16, 16) with kernel sizes (3, 3, 3, 5, 7, 7)
- Final output layer: 12 filters (for 4 corners × 3 channels)
- LeakyReLU activation (α = 0.01)
- Input: 384×384 grayscale images (downsampled from T265 camera)
- Runs in TensorRT FP16 on Jetson TX2 GPU
- **40 ms per forward pass**
- This is the single largest latency component in the system

**VIO note**: The Intel RealSense T265 runs VIO on its own dedicated ASIC — it does NOT consume TX2 GPU/CPU resources for the core VIO computation. It outputs pose estimates at 100 Hz via USB. The TX2 just reads the result.

**Race results**: Won 15/25 head-to-head races against 3 world champions. Fastest single race time (17.465s vs Vanover's 17.956s). Median single lap 5.52s.

---

## The CNN Bottleneck — Revisited

⚡ **v1 said NV=1 couldn't help the CNN because it runs on CPU. This was wrong. NV=1 replays GPU kernels via C code.**

### Revised Analysis: NV=1 vs TensorRT for the Gate CNN

Both run on GPU. The comparison is now:

| Factor | TensorRT | NV=1 Graph Replay |
|--------|----------|-------------------|
| Dispatch overhead | Minimal (C++ runtime) | Minimal (~200 lines C) |
| Kernel quality | Hand-tuned cuDNN kernels, layer fusion, kernel auto-tuning | TinyGrad-generated kernels (may not be as optimized) |
| Quantization | INT8/FP16 with calibration | FP16 possible, INT8 less mature |
| Layer fusion | Aggressive (conv+bn+relu fused) | Depends on TinyGrad's graph optimization |
| Memory planning | Optimized tensor memory pools | Pre-allocated from trace |
| Startup cost | High (engine build can take minutes) | Low (one forward pass to trace) |
| Flexibility | Static shapes only (like NV=1) | Static shapes only |

### Where NV=1 Could Beat TensorRT

1. **Dispatch overhead for small/medium models**: For the gate CNN (~12 filters per level, small U-Net), the actual convolution compute may be small enough that TensorRT's dispatch overhead (engine execution, binding setup) is non-trivial. NV=1's raw C replay has essentially zero dispatch overhead.

2. **Models that TensorRT doesn't optimize well**: Custom architectures, unusual activation functions, non-standard layer patterns. TensorRT's fusion patterns are templated — if your model doesn't match, you get unoptimized fallback kernels.

3. **The "medium model" zone (10K–500K params)**: Where compute per kernel is small but there are many kernels. Framework overhead per kernel becomes a significant fraction of total time. This is NV=1's sweet spot.

### Where TensorRT Likely Wins

1. **Large convolutions**: cuDNN's hand-tuned kernels for standard conv2d shapes are extremely fast. TinyGrad's generated kernels may not match for large spatial convolutions.

2. **INT8 inference**: TensorRT has mature INT8 quantization with per-channel calibration. This 2× speedup over FP16 is hard to beat.

3. **Very large models (1M+ params)**: When compute dominates, kernel quality matters more than dispatch overhead.

### The Experiment We NEED To Run

⚡ **Before making any claims, we must benchmark**:

```
For each model type (MLP, CNN, LSTM):
  For each param count (1K, 10K, 50K, 100K, 500K, 1M, 5M, 12M):
    Measure:
      - TensorRT FP16 inference time (ms)
      - TensorRT INT8 inference time (ms)  
      - TinyGrad NV=1 graph replay time (ms)
      - TinyGrad NV=1 FP16 time (ms) if supported
      - Raw TinyGrad (no NV=1) time (ms) for comparison
    On:
      - Jetson TX2 (to match Swift's hardware)
      - Jetson Xavier NX (to match RTN-MPC's hardware)
      - Jetson AGX Orin (current gen)
      - Desktop GPU (for development/ceiling)
```

**Specific models to benchmark (matched to the papers)**:

| Model | Architecture | Params | Why |
|-------|-------------|--------|-----|
| Swift MLP | 2×128, LeakyReLU | ~21K | Direct Swift comparison |
| RTN-MPC small | 3×32 | ~2K | Baseline |
| RTN-MPC medium | 5×128 | ~67K | RTN-MPC sweet spot |
| RTN-MPC large | 12×512 | ~182K | NV=1 target zone |
| RTN-MPC XL | 20×512 | ~500K | NV=1 target zone |
| Swift Gate CNN | U-Net 6-level (8,16,16,16,16,16) | ~50K est. | The actual bottleneck |
| ResNet-18 | Standard | ~12M | RTN-MPC Table II entry |
| Small LSTM | 2-layer, 128 hidden | ~200K | Temporal modeling |
| Medium LSTM | 2-layer, 512 hidden | ~2M | RTN-MPC future work |
| Fusion MLP | 4×256 | ~200K | Hypothetical Kalman replacement |
| End-to-end MLP | 6×512 | ~1.3M | Hypothetical full replacement |

**What to measure per run**:
- Median latency over 10,000 inferences (after warmup)
- p99 latency (jitter matters for real-time control)
- GPU utilization %
- GPU memory usage
- Power draw (if measurable — matters for battery life)

**Expected outcome**: NV=1 likely wins for MLPs in the 50K–500K range where dispatch overhead is a significant fraction. For the U-Net CNN, it's a genuine open question. For large CNNs (ResNet-18), TensorRT likely wins on kernel quality.

---

## Can We Replace the Whole Chain?

### The Ambitious Claim: One 500K Model Replaces VIO + CNN + Kalman + MLP

Let's be rigorous about what each component actually does:

| Component | Function | State/Memory Required | Rate |
|-----------|----------|----------------------|------|
| VIO (T265) | Visual-inertial odometry — tracks camera pose over time | YES — maintains feature map, IMU pre-integration, sliding window of past poses | 100 Hz |
| Gate CNN | Detects gate corners in images | No (feedforward) | 30 Hz |
| Kalman filter | Fuses VIO drift correction with gate pose estimates | YES — maintains state estimate + covariance (6 states) | 100 Hz |
| MLP policy | Maps observed state → actions | No (feedforward, but takes previous action as input) | 100 Hz |

### The Hard Problem: VIO Requires Persistent State

VIO is fundamentally a **recursive state estimator**. It maintains:
- A map of visual features tracked across frames
- IMU pre-integration factors between keyframes
- A sliding window optimization (or filter) over past poses
- Bias estimates for IMU gyroscope and accelerometer

A pure feedforward MLP (or even a 500K param one) **cannot replace VIO** because it has no mechanism for persistent memory across time steps. You'd lose:
- Metric scale (how big is a meter?)
- Long-term position tracking
- Drift-free orientation estimation

### What CAN Be Replaced

**Realistic replacement targets**:

```
KEEP: VIO (T265 ASIC — it's free compute, runs on dedicated hardware)
REPLACE: Gate CNN + Kalman + MLP → Single learned model
```

This "Fusion+Control" model would:
- **Input**: VIO state estimate (15 dims) + raw gate corner detections OR small visual feature vector + previous action (4 dims) + IMU readings (6 dims)
- **Output**: thrust + body rates (4 dims)
- **Architecture**: 4-6 layer MLP, 256-512 neurons, ~200K-500K params
- **What it learns**: gate-aware state correction (replacing Kalman) + optimal control policy (replacing MLP) + possibly some visual feature extraction (replacing or augmenting CNN)

This is architecturally sound because:
- The Kalman filter is a linear correction — an MLP can learn a nonlinear version
- The MLP policy is already learned — just make it bigger with richer inputs
- RL training already handles the joint optimization of perception-aware control

### The Gate CNN Replacement Question

The gate CNN does something specific: segment gate corners in 384×384 images → 4 corner coordinates. Options:

**Option A: Keep the CNN, speed it up**
- Benchmark NV=1 vs TensorRT on the actual U-Net architecture
- If NV=1 wins (possible for this small U-Net), use it
- Low risk, direct improvement

**Option B: Replace CNN with a smaller regression network**
- Instead of segmentation (dense per-pixel output), train a direct regression head
- Input: downsampled image (e.g., 96×96) → Output: 4×2 corner coordinates
- Could be a small CNN (few conv layers) + MLP head, maybe 50K-200K params
- Much faster than the full U-Net
- Risk: less robust, might need more training data

**Option C: Remove the CNN entirely**
- Use VIO feature points that happen to be on the gate structure
- Or use a simpler color/geometry detector (gates have known shape and color)
- Classical CV + MLP fusion
- Lowest compute, highest risk to perception quality

### Honest Assessment

| Replacement | Feasible? | Params | NV=1 Advantage? |
|-------------|-----------|--------|----------------|
| Full VIO + CNN + Kalman + MLP → one model | ❌ No — VIO needs persistent state and dedicated hardware | — | — |
| CNN + Kalman + MLP → learned fusion+control | ✅ Yes — architecturally sound | 200K–500K | ✅ Strong — NV=1 sweet spot |
| CNN → smaller regression net | ✅ Likely — needs retraining | 50K–200K | ✅ Strong — NV=1 sweet spot |
| Kalman + MLP → richer learned policy | ✅ Yes — straightforward | 50K–200K | ✅ Strong |
| Full end-to-end (raw pixels → actions) | ⚠️ Research-grade only | 1M+ | ❓ Depends on benchmark |

### The Most Compelling Pitch

> "We can't replace the whole pipeline with one model — VIO needs persistent state and runs on free dedicated hardware anyway. But we CAN replace the three components that run on the TX2 (CNN + Kalman + MLP) with a single learned model in NV=1's sweet spot. This eliminates the 40ms CNN bottleneck, removes the hand-tuned Kalman filter, and runs the whole thing at 200+ Hz via NV=1 graph replay. The result: a simpler system, higher control rate, and potentially better robustness because the RL training jointly optimizes perception and control."

---

## Benchmarking Plan: NV=1 vs TensorRT

### Phase 1: Pure Inference Latency (No MPC, No Pipeline)

**Goal**: Find the crossover point where NV=1 graph replay beats TensorRT for each model type.

**MLP sweep**:
```python
# Sweep these configurations
for layers in [2, 3, 5, 8, 12, 20]:
    for width in [32, 64, 128, 256, 512]:
        params = estimate_params(layers, width)
        if 1_000 < params < 10_000_000:
            benchmark(model_type="MLP", layers=layers, width=width)
```

**CNN sweep**:
```python
# Key architectures
models = [
    "swift_unet_6level",           # ~50K params, the actual bottleneck
    "mobilenet_v2_0.25",           # ~200K, lightweight alternative  
    "custom_3conv_mlp_head",       # ~100K, regression approach
    "resnet18",                    # ~12M, RTN-MPC reference
]
input_sizes = [(384, 384), (192, 192), (96, 96)]  # test downsampling impact
```

**LSTM/GRU sweep**:
```python
for hidden_size in [64, 128, 256, 512]:
    for num_layers in [1, 2, 3]:
        for seq_len in [10, 25, 50, 100]:  # how much history
            benchmark(model_type="LSTM", hidden=hidden_size, 
                     layers=num_layers, seq_len=seq_len)
```

**For each configuration, measure**:
- TensorRT FP32 (baseline)
- TensorRT FP16
- TensorRT INT8 (where available)
- TinyGrad standard inference
- TinyGrad NV=1 graph replay
- ONNX Runtime (additional reference)
- PyTorch eager (to show framework overhead)

### Phase 2: Integration Benchmarks

Once we know the crossover points from Phase 1, test in realistic contexts:

**RTN-MPC integration**: Replace PyTorch Jacobian computation in RTN-MPC's data-driven dynamics phase with NV=1. Measure full MPC loop Hz improvement.

**Swift-like pipeline**: Build a simplified version of Swift's pipeline and swap components:
- Baseline: TensorRT CNN + numpy Kalman + PyTorch MLP
- Test 1: TensorRT CNN + numpy Kalman + NV=1 MLP
- Test 2: NV=1 CNN + numpy Kalman + NV=1 MLP
- Test 3: NV=1 fused (CNN+Kalman+MLP as single model)

### Phase 3: Novel Architecture Benchmarks

Test the new architectures that NV=1 makes possible:
- Multi-rate cascade: 500 Hz inner MLP + 100 Hz outer MLP
- Approximate MPC: distilled MPC → large MLP
- Fusion+control: single model replacing CNN+Kalman+MLP

### What the Benchmarks Should Produce

A **crossover chart** for each model type:

```
Latency (ms)
    │
    │  TensorRT ──────────────
    │  NV=1     ╲
    │            ╲──── NV=1 wins above this line
    │             ╲
    │              ───────────
    │
    └───────────────────────── Params
        1K   10K  100K  1M  10M
              ↑
           Crossover point
```

Expected crossover points (hypotheses to test):
- **MLP**: NV=1 likely competitive from ~10K params, possibly faster from ~50K due to dispatch overhead savings
- **CNN**: Unknown — depends on whether TinyGrad's conv kernels match cuDNN quality. Crossover could be at small CNNs (~50K) or may never happen for large ones
- **LSTM**: NV=1 likely wins for small-medium LSTMs because the sequential nature means many small kernel launches where dispatch overhead matters

---

## Where NV=1 Wins and Why

### The Fundamental Advantage

NV=1 graph replay is **~200 lines of C that directly dispatch GPU kernels**. No Python, no framework, no graph walking, no buffer management per call. For models where:

```
total_time = dispatch_overhead × num_kernels + compute_time
```

If `dispatch_overhead × num_kernels` is a significant fraction of `total_time`, NV=1 wins. This happens when:
- Models have many small operations (deep thin MLPs, LSTMs with many timesteps)
- Compute per operation is small (hidden dim ≤ 512)
- The batch size is 1 (real-time inference, no batching to amortize overhead)

### Specific Win Scenarios for Drone Control

| Scenario | Why NV=1 Wins | Expected Improvement |
|----------|--------------|---------------------|
| RTN-MPC dynamics MLP (200K-500K) | Many Jacobian evaluations (N=10 nodes × 4 RK4 evals = 40 forward passes per MPC step). Dispatch overhead multiplied 40×. | Could double the MPC control frequency |
| Multi-rate inner loop (5K-50K MLP at 500+ Hz) | At 500 Hz = 2ms budget. Dispatch overhead is significant fraction of 2ms. | Enables 500+ Hz with non-trivial models |
| Fusion+Control replacement model (200K) | Single model replaces CNN+Kalman+MLP pipeline. Eliminates pipeline stitching overhead. | 40ms → potentially 5-10ms (if CNN replacement works) |
| Fleet deployment (no TensorRT available) | TensorRT only works on NVIDIA GPUs. NV=1 can target other backends. | Enables non-NVIDIA deployment |
| Rapid iteration / deployment | TensorRT engine build takes minutes. NV=1 traces in one forward pass. | Development velocity |

### Scenarios Where NV=1 Probably Loses

| Scenario | Why TensorRT Wins |
|----------|------------------|
| Large CNN inference (ResNet-50+) | cuDNN kernels for large convolutions are extensively hand-tuned |
| INT8 quantized models | TensorRT's INT8 pipeline is mature with calibration tools |
| Very large models (10M+) | Compute completely dominates; dispatch overhead is negligible |
| Batch inference (batch > 1) | Both eliminate per-sample overhead, but TensorRT's kernel tuning for batched shapes is better |

---

## Hardware Context

### Jetson TX2 (Swift's hardware, 2017)
- CPU: 2× Denver 2 + 4× A57 @ 2 GHz
- GPU: 256 CUDA cores (Pascal) @ 1.3 GHz
- Memory: 8 GB LPDDR4
- TDP: 7.5-15W
- TensorRT support: Yes (older versions)

### Jetson Xavier NX (RTN-MPC's hardware, 2020)
- CPU: 6× Carmel (ARMv8.2) @ 1.9 GHz
- GPU: 384 CUDA cores + 48 Tensor Cores (Volta) @ 1.1 GHz
- Memory: 8 GB LPDDR4x
- TDP: 10-20W
- **~2-3× faster than TX2** on most workloads
- Tensor Cores give additional speedup for FP16/INT8

### Jetson AGX Orin (current gen, 2022)
- CPU: 12× A78AE @ 2.2 GHz
- GPU: 2048 CUDA cores + 64 Tensor Cores (Ampere) @ 1.3 GHz
- Memory: 32-64 GB LPDDR5
- TDP: 15-60W
- **~5-8× faster than Xavier NX**
- Would change ALL the numbers in both papers dramatically

### Comparison Rules
- ⚠️ NEVER compare TX2 Hz to Xavier NX Hz without stating the hardware difference
- ⚠️ NEVER compare "MPC loop Hz" to "pure inference Hz" — they measure different things
- ⚠️ NEVER compare NV=1 times to TensorRT times measured on different hardware
- ✅ ALWAYS state: hardware, precision (FP32/FP16/INT8), what's included in the timing, batch size

---

## Honesty Checklist

Before making any claim in a presentation, verify:

- [ ] **"NV=1 is Nx faster than TensorRT"** → On the same GPU? Same precision? Same model? Same batch size? Warmed up?
- [ ] **"This would improve Swift's lap time"** → The MLP is 8ms in a 40ms+ pipeline. Would faster MLP actually change race outcomes?
- [ ] **"We can replace the CNN"** → With what? Have you validated perception quality? What happens in low light? Motion blur?
- [ ] **"500K params can replace the whole pipeline"** → VIO needs persistent state. The T265 runs on its own ASIC. Are you actually replacing VIO or just the downstream processing?
- [ ] **"Higher control frequency = better performance"** → Only if the current frequency is the limiting factor. Swift at 100 Hz already has 40ms total latency vs 220ms for humans. Would 200 Hz help?
- [ ] **"This works on embedded"** → Which embedded? TX2? Xavier? Orin? They're very different.
- [ ] **"Graph replay eliminates overhead"** → True, but if compute dominates (large CNNs), overhead elimination doesn't matter much.

---

## Key Numbers

| Metric | Value | Source | Notes |
|--------|-------|--------|-------|
| Swift MLP params | ~21K | Nature paper | 2×128, LeakyReLU |
| Swift MLP inference | 8 ms | Nature paper | On TX2 CPU |
| Swift MLP rate | 100 Hz | Nature paper | Synchronous with Kalman output |
| Swift Gate CNN inference | 40 ms | Nature paper | TensorRT FP16 on TX2 GPU |
| Swift Gate CNN rate | 30 Hz | Nature paper | Camera frame rate limited |
| Swift Gate CNN architecture | 6-level U-Net | Nature paper | ~50K params estimated |
| Swift VIO | 100 Hz | Nature paper | Runs on T265 ASIC, not TX2 |
| Swift Kalman filter | < 1 ms | Estimated | 6-state linear filter |
| Swift total system latency | 40 ms | Nature paper | Perception-limited |
| Human pilot latency | 220 ms | Nature paper | Sensorimotor |
| Swift race win rate | 60% (15/25) | Nature paper | vs 3 world champions |
| RTN-MPC 17K MLP, ARM CPU | 135 Hz | RTN-MPC paper | Full MPC loop, Xavier NX |
| RTN-MPC 67K MLP, Jetson GPU | 63 Hz | RTN-MPC paper | Full MPC loop, Xavier NX |
| RTN-MPC 182K MLP, Jetson GPU | 61 Hz | RTN-MPC paper | Full MPC loop, Xavier NX |
| RTN-MPC 500K MLP, ARM CPU | 11 Hz | RTN-MPC paper | Full MPC loop, Xavier NX |
| RTN-MPC 500K MLP, Jetson GPU | 46 Hz | RTN-MPC paper | Full MPC loop, Xavier NX |
| RTN-MPC ResNet-18, Jetson GPU | 9 Hz | RTN-MPC paper | Full MPC loop, Xavier NX |
| RTN-MPC Naive dies at | ~10K params | RTN-MPC paper | ARM CPU, below 50 Hz |
| NV=1 graph replay code size | ~200 lines C | Our implementation | Static shapes, batch=1 |
| NV=1 sweet spot | 50K–1M params | Our benchmarks | Needs formal verification |
| Motor response bandwidth | ~500 Hz | General robotics | Physical actuator limit |
| Betaflight PID rate | 1 kHz | Swift paper | Runs on STM32 @ 216 MHz |
| Swift training time | 50 min | Nature paper | 10⁸ steps, i9 + RTX 3090 |
| Swift fine-tuning | 20M steps | Nature paper | ~10 min, with residual models |
| Swift real-world data needed | ~50 seconds of flight | Nature paper | 3 rollouts for residual ID |

---

## Open Questions for Future Investigation

1. **What are TinyGrad's generated GPU kernels actually like for convolutions?** Are they competitive with cuDNN? This is the key unknown for the CNN question.

2. **Can NV=1 do FP16?** If so, how does it compare to TensorRT FP16? This is critical for embedded GPU benchmarks.

3. **What's the actual dispatch overhead of TensorRT vs NV=1 for small models?** We hypothesize NV=1 is lower, but need measurements.

4. **Can we profile the Swift gate CNN to find the actual compute vs overhead split?** If overhead is 30% of the 40ms, NV=1 could shave 12ms — significant. If overhead is 5%, it's negligible.

5. **Has anyone trained a direct-regression gate detector (not segmentation)?** This would be a much smaller model and a natural NV=1 candidate.

6. **What is the minimum perception quality needed for Swift to still win races?** If a 50% less accurate gate detector only costs 0.1s per lap, the speed tradeoff might be worth it.


### Extra notes

MLP sweep: 2×128 (Swift's 21K), 5×128 (67K), 12×512 (182K), 20×512 (500K), 6×512 (1.3M) — all LeakyReLU, batch=1, input dim=31→4 to match Swift's actual shapes
CNN sweep: 3-conv+MLP head (50K), Swift's actual U-Net 6-level (8,16,16,16,16,16) on 384×384 grayscale, MobileNet-v2-0.25 (~200K), small ResNet-8 (~500K) — all single-image inference
LSTM sweep: 1-layer 128 hidden (50K), 2-layer 128 (200K), 2-layer 512 (2M) — sequence lengths 10, 25, 50 at batch=1, since these would model IMU history windows
Hybrid combos: small CNN (3-conv) → MLP head as single fused graph (~100-200K, the "gate regression" replacement), LSTM encoder → MLP policy head (~250K, temporal fusion+control), small CNN → LSTM → MLP (~500K, the full replacement for CNN+Kalman+MLP)
What to measure per config: median latency, p99 latency (jitter kills real-time), GPU util %, memory footprint — all NV=1 graph replay first, raw TinyGrad second for overhead comparison
What this leads to: the crossover chart — at what param count / architecture type does NV=1 overhead savings become meaningful? Then we take winners and benchmark head-to-head against TensorRT FP16 to see where we're competitive vs where cuDNN kernel quality dominates
Key targets to beat: 40ms (Swift CNN), 8ms (Swift MLP), 20ms (50Hz real-time floor), 10ms (100Hz), 2ms (500Hz inner loop) — if any config hits these thresholds, it unlocks a real application from the papers