# Fun ML Projects for Jetson Orin AGX 64GB

Machine learning experiments using **tinygrad** with the NV backend on the
Jetson Orin AGX. Every project runs on the iGPU (ga10b, SM 8.7) — no external
GPU needed.

## Setup

```bash
cd examples/ml-projects
nix develop
```

## Projects

### 1. Neural Style Transfer (`style-transfer/`)

Transfer artistic style (Van Gogh, Monet, abstract patterns) onto photographs.
Uses a VGG-like feature extractor + Gram matrix matching.

```bash
NV=1 python3 style-transfer/style_transfer.py --bench
NV=1 python3 style-transfer/style_transfer.py --content photo.jpg --style painting.jpg
```

**What you learn**: Feature extraction, Gram matrices (texture statistics),
optimization-based image generation, the content/style separation in CNNs.

### 2. Audio ML (`audio-ml/`)

GPU-accelerated mel spectrogram computation + audio classification CNN.
Computes STFT as a batched matmul against the DFT basis — pure tinygrad.

```bash
NV=1 python3 audio-ml/audio_classify.py --bench
NV=1 python3 audio-ml/audio_classify.py --file recording.wav
```

**What you learn**: Fourier transforms as matrix operations, mel scale
(human perception), audio → image representation for CNNs.

### 3. Reinforcement Learning (`reinforcement-learning/`)

CartPole environment + DQN/REINFORCE agents. The RL loop (environment step →
agent decision → reward → learn) runs entirely in tinygrad.

```bash
NV=1 python3 reinforcement-learning/rl_agents.py --agent dqn --episodes 500
NV=1 python3 reinforcement-learning/rl_agents.py --agent reinforce --episodes 500
NV=1 python3 reinforcement-learning/rl_agents.py --bench
```

**What you learn**: Q-learning vs policy gradients, exploration vs exploitation,
neural function approximation, why RL is hard (sample efficiency, stability).

### 4. Generative Models (`generative/`)

Convolutional autoencoder (learn compressed representations) and simplified
DDPM diffusion model (learn to reverse a noise process).

```bash
NV=1 python3 generative/generative_models.py --model autoencoder --bench
NV=1 python3 generative/generative_models.py --model diffusion --bench
NV=1 python3 generative/generative_models.py --model autoencoder --train --steps 100
```

**What you learn**: Latent spaces, encoder/decoder architectures, U-Nets,
diffusion process (forward noise + learned reverse), generative modeling.

### 5. Edge Deployment Toolkit (`edge-deploy/`)

Practical optimization techniques for deploying ML on edge devices:
quantization (fp32→fp16), knowledge distillation, memory budgeting, JIT fusion.

```bash
NV=1 python3 edge-deploy/edge_optimize.py --all
NV=1 python3 edge-deploy/edge_optimize.py --quantize
NV=1 python3 edge-deploy/edge_optimize.py --memory
NV=1 JITBEAM=2 python3 edge-deploy/edge_optimize.py --fusion
```

**What you learn**: Why quantization works (fp16 ≈ 2× bandwidth), distillation
(big→small model transfer), memory bandwidth as the bottleneck, kernel fusion.

## Design Philosophy

Every project follows the same principles:

1. **Pure tinygrad** — No PyTorch, TensorFlow, or TensorRT dependencies
2. **NV backend first** — Designed for Jetson's TegraIface/HCQ path
3. **Self-contained** — Synthetic data, no downloads (models use random weights)
4. **Benchmarkable** — Every script has `--bench` for NV vs CUDA comparison
5. **Educational** — Extensive docstrings explain the math and architecture
6. **BEAM-compatible** — Works with `JITBEAM=N` for kernel optimization

## Architecture Notes

The Orin AGX 64GB has unique characteristics for ML:

| Property | Value | Impact |
|---|---|---|
| GPU | ga10b (SM 8.7, 2048 cores) | Good for small-medium models |
| Memory | 64GB LPDDR5 unified | Huge model capacity |
| Bandwidth | ~102 GB/s effective | Memory-bound for LLM decode |
| TDP | 15-60W configurable | Edge power budget |
| FP16 | 2× tensor throughput | Quantize everything |

**Key insight**: The unified memory means zero-copy between CPU and GPU.
This is why the NV backend (which uses TegraIface for direct MMIO) is
faster than the CUDA backend (which goes through the CUDA runtime) for
bandwidth-bound workloads.
