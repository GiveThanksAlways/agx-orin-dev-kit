# Tinygrad + NixOS + tools others built = Nvidia Jetson Dev kit fun for us

GPU-accelerated inference on NVIDIA Jetson Orin using [tinygrad](https://github.com/tinygrad/tinygrad)'s NV backend — no CUDA runtime, just raw GPU command queues.

This repo contains a [tinygrad fork](external/tinygrad) with Tegra (Jetson) support, benchmarks, and a reproducible NixOS development environment.

## Quick Start

```bash
git clone --recurse-submodules git@github.com:GiveThanksAlways/agx-orin-dev-kit.git
cd agx-orin-dev-kit

# if you already cloned without --recurse-submodules:
git submodule update --init --recursive
```

## 1. Tinygrad with NV Backend

Run tinygrad directly on the GPU via NVIDIA's HW command queues (no `libcuda`):

```bash
cd examples/tinygrad
nix develop

NV=1 python3 -c 'from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())'
NV=1 python3 ../../external/tinygrad/examples/gpt2.py --count 10
```

## 2. Learned Inertial Odometry Benchmark

End-to-end sensor-fusion pipeline (IMU → TCN → EKF) from [Cioffi et al.](https://arxiv.org/pdf/2210.15287), benchmarked across tinygrad NV, PyTorch, and TensorRT:

```bash
cd examples/learned-inertial-odometry
nix develop

# tinygrad NV backend (beam-searches for fastest kernel schedule)
NV=1 JITBEAM=2 python3 bench_e2e_pipeline.py --backend nv

# compare all backends
NV=1 JITBEAM=2 python3 bench_e2e_pipeline.py --backend all
```
