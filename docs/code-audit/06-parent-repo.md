# Parent Repo Changes: NixOS, Frameworks, Projects

The `code-audit` branch merges all work from the development branches into
master. Beyond the 5 tinygrad source files, there are **~120 parent repo files**
across several categories.

## Table of Contents

1. [Repository Infrastructure](#repository-infrastructure)
2. [NixOS System Configuration](#nixos-system-configuration)
3. [LLM Frameworks](#llm-frameworks)
4. [Tinygrad Benchmarks](#tinygrad-benchmarks)
5. [Stereo Vision Camera](#stereo-vision-camera)
6. [ML Projects](#ml-projects)
7. [Kernel Source References](#kernel-source-references)
8. [Docker / Containerized Inference](#docker--containerized-inference)

---

## Repository Infrastructure

### AGENTS.md

Top-level guide for AI coding agents. Contains:
- Target hardware (Nvidia Jetson Orin AGX 64GB)
- Build commands (`nixos-rebuild switch --flake ...`)
- Folder layout table
- Quick reference for each example subfolder

### .gitmodules + .gitignore

```ini
[submodule "examples/tinygrad/tinygrad"]
    path = examples/tinygrad/tinygrad
    url = https://github.com/tinygrad/tinygrad.git
```

The tinygrad repo is a git submodule pinned to our `nv-agx-orin-dev-kit` branch.
Changes inside `examples/tinygrad/tinygrad/` track our fork's commits.

`.gitignore` additions: `__pycache__/`, `*.pyc`, `.beam_cache/`, `*.gguf`,
`*.safetensors`, `.beam/` — prevents accidentally committing multi-GB model
files or per-machine BEAM caches.

### .vscode/settings.json

VS Code workspace settings for the repo: Python path, file associations, etc.

---

## NixOS System Configuration

**Location**: `examples/nixos/`

A complete NixOS system configuration for the Jetson, with composable modules:

### flake.nix

Defines multiple NixOS configurations you can switch between:

```nix
nixosConfigurations = {
  nixos           = mkSystem {};              # base system
  nixos-perf      = mkSystem { perf = true; };  # + performance tuning
  nixos-llama-cpp = mkSystem { llama = true; };  # + llama.cpp server
  nixos-tabby     = mkSystem { tabby = true; };  # + TabbyAPI server
  nixos-telemetry = mkSystem { telemetry = true; };  # + Prometheus/Grafana
};
```

### modules/performance.nix

GPU clock pinning, CPU governor, memory overcommit, kernel parameters:

```nix
# Pin GPU to max frequency
systemd.services.gpu-max-freq = {
  script = ''
    echo $(cat /sys/class/devfreq/17000000.gpu/max_freq) \
      > /sys/class/devfreq/17000000.gpu/min_freq
  '';
};

# CPU performance governor
powerManagement.cpuFreqGovernor = "performance";
```

### modules/telemetry.nix

Full Prometheus + Grafana stack with:
- `prometheus-node-exporter`: CPU, memory, disk, network metrics
- Custom GPU exporter: reads from `/sys/class/devfreq/17000000.gpu/`
  and `/sys/devices/virtual/thermal/` for GPU freq, temp, utilization
- Pre-built Grafana dashboards (JSON provisioning)
- ~790 lines — the most complex module

### modules/llama-cpp-server.nix

systemd service for llama.cpp's HTTP server with CUDA:

```nix
ExecStart = "${llama-cpp}/bin/llama-server \
  -m ${config.llama.modelPath} \
  -ngl 99 --gpu-layers 99 \
  --host 0.0.0.0 --port 8080";
```

### modules/tabby-api.nix

Similar service for TabbyAPI (ExLlamaV2 backend) with GPU offloading.

---

## LLM Frameworks

### examples/LLM/

Minimal flake for tinygrad LLM inference. Uses the tinygrad submodule.

### examples/llama-cpp-orin/

Two variants:
- **Simple** (`llama-cpp-orin/flake.nix`): Builds llama.cpp from upstream with
  CUDA and OpenSSL. Dev shell only.
- **Overlay** (`llama-cpp-orin-nix-overlay/flake.nix`): Full overlay with
  wrapper scripts (`qwen3-coder`, `qwen3-server`, `llama-benchmark`). Exports
  `overlays.default` for reuse in other flakes.

### examples/vllm/

vLLM on Jetson — Docker-based because vLLM's Python dependency tree is too
complex for a pure Nix build:

```
flake.nix          → Nix dev shell with Docker tools
Dockerfile.jetson  → JetPack-based container with vLLM
run-vllm-docker.sh → Launch script with GPU passthrough
bench_vllm.py      → Throughput benchmarking script
```

### examples/MLC-LLM/

MLC LLM (Machine Learning Compilation) — Apache TVM-based inference:

```
flake.nix          → Dev shell with TVM, LLVM, CUDA
bench_mlc_llm.py   → Benchmark script comparing MLC vs tinygrad vs llama.cpp
```

---

## Tinygrad Benchmarks

**Location**: Inside the tinygrad submodule (`examples/tinygrad/tinygrad/bench_*.py`)

| Benchmark | What It Tests |
|---|---|
| `bench_cnn.py` | ResNet-18/50, MobileNetV2, EfficientNet, VGG-16 on NV vs CUDA |
| `bench_comprehensive.py` | matmul, conv2d, attention, LLM layers at multiple sizes |
| `bench_llama_3b.py` | LLaMA 3.2 3B inference (Q6_K, fp16) with BEAM |
| `bench_mixed_precision.py` | fp32/fp16/bf16 matmul and conv comparison |
| `bench_qwen3_beam.py` | Qwen3 0.6B with JITBEAM sweep (1, 2, 4, 8) |

All benchmarks:
- Support `NV=1` (native TegraIface) and `CUDA=1` (CUDA runtime) backends
- Include BEAM/JITBEAM optimization support
- Output markdown-formatted tables for documentation
- Report throughput in tok/s, TFLOPS, or GB/s depending on the operation

---

## Stereo Vision Camera

**Location**: `examples/binocular-camera/`

Hardware: **Waveshare Dual IMX219** — 8MP binocular stereo camera with ~60mm
baseline, connected via 2× CSI-2 (appearing as `/dev/video0` + `/dev/video1`).

### Scripts

| Script | Purpose |
|---|---|
| `capture_stereo.py` | V4L2/GStreamer synchronized stereo capture |
| `calibrate_stereo.py` | Checkerboard stereo calibration (intrinsics + extrinsics) |
| `depth_map.py` | Depth estimation: OpenCV SGBM vs tinygrad cost volume |
| `stereo_object_detect.py` | 3D object detection with TinyDetector CNN |
| `obstacle_avoidance.py` | Real-time clearance grid + heading suggestion |
| `hand_tracking_3d.py` | Hand keypoint detection + 3D gesture recognition |
| `bench_stereo_pipeline.py` | Pipeline throughput benchmark |

### Key Technique: Tinygrad Cost Volume

`depth_map.py` implements stereo depth estimation in tinygrad using a **cost
volume** approach:

1. For each possible disparity $d \in [0, \text{max\_disp})$, shift the right
   image by $d$ pixels
2. Compute the **Sum of Absolute Differences (SAD)** between left and shifted
   right images
3. Apply **box filter** (conv2d with a uniform kernel) to smooth the cost
4. Pick the disparity with minimum cost (**Winner-Take-All**)
5. Convert disparity to depth via the stereo Q matrix:
   $\text{depth} = \frac{f \cdot B}{d}$
   where $f$ is focal length and $B$ is baseline

This runs entirely on the GPU via tinygrad's NV/CUDA backend.

---

## ML Projects

**Location**: `examples/ml-projects/`

Five self-contained ML projects demonstrating tinygrad on Jetson:

### style-transfer/

**Neural Style Transfer** (Gatys et al., 2015): Transfer the artistic style of
one image onto the content of another using VGG feature matching.

Key concept: Minimize a loss that combines:
- **Content loss**: MSE between feature maps at layer 3
  (preserves image structure)
- **Style loss**: MSE between Gram matrices of features at multiple layers
  (captures texture patterns)

### audio-ml/

**Audio Classification**: Process audio on GPU using tinygrad for the entire
pipeline:
- STFT via DFT basis matrix as a tinygrad matmul (not FFT — matmul is
  well-optimized on GPU)
- Mel filterbank for perceptual frequency scaling
- CNN classifier on the mel spectrogram

### reinforcement-learning/

**CartPole**: Classic control problem solved two ways:
- **DQN**: Q-function network with experience replay buffer
- **REINFORCE**: Policy gradient with normalized returns

Includes a pure-Python CartPole environment (no gym dependency).

### generative/

**Generative Models**: Two architectures:
- **Convolutional Autoencoder**: Encoder (3 conv → FC) + Decoder (FC → 3
  transposed conv using nearest-neighbor upsampling)
- **Simplified DDPM**: Diffusion model with UNet (encoder/decoder with skip
  connections and time embedding)

### edge-deploy/

**Edge Deployment Optimization**: Practical tools for deploying models on Jetson:
- fp32 vs fp16 comparison (half-precision nearly 2× faster)
- Knowledge distillation (train small student from large teacher)
- Memory budget planner (estimates DRAM usage for common models)
- JIT fusion analysis (shows which ops tinygrad fuses into single kernels)

---

## Kernel Source References

**Location**: `examples/tinygrad/l4t-sources/`

Excerpts from the L4T (Linux for Tegra) kernel source code — the nvgpu and
nvmap drivers. These are **reference files only** (not compiled), included
so you can read the kernel-side implementation alongside TegraIface.

| Directory | What's Inside |
|---|---|
| `nvgpu/hal/fifo/` | Channel and usermode HAL for ga10b |
| `nvgpu/hal/init/` | GPU initialization (`hal_ga10b.c` — 2148 lines) |
| `nvgpu/os/linux/ioctl*.c` | Linux ioctl handlers (the kernel code our ioctls call) |
| `nvgpu/common/mm/` | Memory management: VM, page tables, allocators |
| `nvgpu/include/uapi/linux/` | UAPI headers (the struct definitions our ctypes mirror) |
| `nvidia-oot/include/linux/nvmap.h` | nvmap API header |
| `nvidia-oot/include/uapi/linux/nvmap.h` | nvmap ioctl definitions |

These are invaluable for understanding *why* TegraIface does things the way it
does — you can trace each ioctl from Python through the kernel handler to the
hardware register writes.

---

## Docker / Containerized Inference

**Location**: `examples/docker/`

For frameworks that are hard to nixify (complex Python dependency trees), we
provide Docker containers:

```
Dockerfile          → JetPack-based image with tinygrad + CUDA
docker-compose.yml  → Service definitions (model server, API endpoint)
api-server.py       → FastAPI wrapper around tinygrad LLM inference
download-model.sh   → Fetch GGUF models from HuggingFace
quickstart.sh       → One-command setup
```

The Docker approach trades reproducibility (Nix is fully deterministic) for
compatibility (pip install works for everything).

### Telemetry Viewer

**Location**: `examples/telemetry-viewer/`

A simple SSH tunnel script for viewing Grafana dashboards from a remote machine:

```bash
./connect-telemetry.sh <jetson-ip> <ssh-user>
# Forwards Grafana (3000), Prometheus (9090), node-exporter (9100)
# Then opens Chrome to localhost:3000
```

### Serial UART MCP

**Location**: `examples/serial-UART-mcp.py`

An MCP (Model Context Protocol) server that exposes Jetson's UART debug console
to AI coding agents. Useful for remote debugging when SSH is down — you can
read boot logs, interact with U-Boot, etc.
