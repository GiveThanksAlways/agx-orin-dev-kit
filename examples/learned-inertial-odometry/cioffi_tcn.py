"""cioffi_tcn.py — Cioffi et al. Learned Inertial Odometry TCN in multiple frameworks.

Faithfully reproduces the exact architecture from:
  Cioffi, Bauersfeld, Kaufmann, Scaramuzza.
  "Learned Inertial Odometry for Autonomous Drone Racing."
  IEEE RA-L, 2023. arXiv:2210.15287.
  Source: https://github.com/uzh-rpg/learned_inertial_model_odometry

Architecture (from src/learning/network/model_factory.py):
  Tcn(input_dim=6, output_dim=3,
      num_channels=[64, 64, 64, 64, 128, 128, 128],
      kernel_size=2, dropout=0.2, activation="GELU")

Each TemporalBlock:
  Conv1d(causal, dilated) → Chomp → GELU → Dropout
  Conv1d(causal, dilated) → Chomp → GELU → Dropout
  + residual connection (with 1x1 conv if channels change)
  → ReLU

7 blocks, dilations = [1, 2, 4, 8, 16, 32, 64]
Final: take last timestep → Linear(128, 3)
~250K parameters

Input: (batch, 6, seq_len) — 6 channels = 3 gyro + 3 thrust
Output: (batch, 3) — Δp (x, y, z displacement)

Their actual config (model_net_parameters_net_blackbird.json):
  window_time = 0.5s, sampling_freq = 100 Hz → 50 samples per window
  The TCN itself is sequence-length agnostic (Conv1d works at any length).
"""
import numpy as np

# Paper parameters (from model_net_parameters_net_blackbird.json)
INPUT_DIM = 6       # 3 gyro + 3 thrust
OUTPUT_DIM = 3      # Δp (x, y, z)
NUM_CHANNELS = [64, 64, 64, 64, 128, 128, 128]
KERNEL_SIZE = 2
DROPOUT = 0.0       # disabled at inference (eval mode)
SEQ_LEN = 50        # 50 timesteps (0.5s @ 100Hz — their Blackbird config)
SEED = 42

# ═══════════════════════════════════════════════════════════════════════════════
# Weight generation (deterministic, shared across frameworks)
# ═══════════════════════════════════════════════════════════════════════════════

def _count_params():
    """Count total parameters in the Cioffi TCN."""
    total = 0
    channels = [INPUT_DIM] + NUM_CHANNELS
    for i in range(len(NUM_CHANNELS)):
        c_in = channels[i]
        c_out = channels[i + 1]
        # conv1: weight_norm splits into weight_g (c_out,1,1) and weight_v (c_out,c_in,ks)
        # but total params = c_out * c_in * ks + c_out (bias) + c_out (g)
        total += c_out * c_in * KERNEL_SIZE + c_out + c_out  # conv1 (v + bias + g)
        total += c_out * c_out * KERNEL_SIZE + c_out + c_out  # conv2 (v + bias + g)
        if c_in != c_out:
            total += c_out * c_in + c_out  # downsample 1x1 conv
    # Final linear
    total += NUM_CHANNELS[-1] * OUTPUT_DIM + OUTPUT_DIM
    return total


def generate_weights(seed=SEED):
    """Generate deterministic weights for the TCN as numpy arrays.

    Returns a dict containing all weight tensors keyed by layer path.
    All weights are FP32; callers cast to their desired precision.
    """
    rng = np.random.RandomState(seed)
    weights = {}
    channels = [INPUT_DIM] + NUM_CHANNELS

    for i in range(len(NUM_CHANNELS)):
        c_in = channels[i]
        c_out = channels[i + 1]
        prefix = f"block_{i}"

        # The paper uses weight_norm, but for inference benchmarking we fold it:
        # weight_norm decomposes W = g * (v / ||v||), but at eval time the fused
        # weight W is what matters. We generate W directly.
        s1 = np.sqrt(2.0 / (c_in * KERNEL_SIZE))
        weights[f"{prefix}.conv1.weight"] = (rng.randn(c_out, c_in, KERNEL_SIZE) * s1).astype(np.float32)
        weights[f"{prefix}.conv1.bias"] = np.zeros(c_out, dtype=np.float32)

        s2 = np.sqrt(2.0 / (c_out * KERNEL_SIZE))
        weights[f"{prefix}.conv2.weight"] = (rng.randn(c_out, c_out, KERNEL_SIZE) * s2).astype(np.float32)
        weights[f"{prefix}.conv2.bias"] = np.zeros(c_out, dtype=np.float32)

        if c_in != c_out:
            s_ds = np.sqrt(2.0 / c_in)
            weights[f"{prefix}.downsample.weight"] = (rng.randn(c_out, c_in, 1) * s_ds).astype(np.float32)
            weights[f"{prefix}.downsample.bias"] = np.zeros(c_out, dtype=np.float32)

    # Final linear
    s_lin = np.sqrt(2.0 / NUM_CHANNELS[-1])
    weights["linear.weight"] = (rng.randn(OUTPUT_DIM, NUM_CHANNELS[-1]) * s_lin).astype(np.float32)
    weights["linear.bias"] = np.zeros(OUTPUT_DIM, dtype=np.float32)

    return weights


def generate_input_pool(n, seed=123):
    """Generate n input tensors (6, SEQ_LEN) as FP16.

    Simulates IMU data: gyro ~N(0, 0.5 rad/s), thrust ~N(0, 2 N).
    """
    rng = np.random.RandomState(seed)
    pool = rng.randn(n, INPUT_DIM, SEQ_LEN).astype(np.float32)
    pool[:, :3, :] *= 0.5   # gyro
    pool[:, 3:, :] *= 2.0   # thrust
    return pool.astype(np.float16)


# ═══════════════════════════════════════════════════════════════════════════════
# PyTorch implementation (for ONNX export and PyTorch benchmarking)
# ═══════════════════════════════════════════════════════════════════════════════

def build_pytorch_tcn(weights=None, use_fp16=True):
    """Build the Cioffi TCN in PyTorch. Returns (model, param_count).

    Uses standard nn.Conv1d (weight_norm folded) for clean ONNX export.
    """
    import torch
    import torch.nn as nn

    if weights is None:
        weights = generate_weights()

    class Chomp1d(nn.Module):
        def __init__(self, chomp_size):
            super().__init__()
            self.chomp_size = chomp_size
        def forward(self, x):
            return x[:, :, :-self.chomp_size].contiguous()

    class TemporalBlock(nn.Module):
        def __init__(self, n_in, n_out, ks, dilation, prefix):
            super().__init__()
            padding = (ks - 1) * dilation
            self.conv1 = nn.Conv1d(n_in, n_out, ks, dilation=dilation, padding=padding)
            self.chomp1 = Chomp1d(padding)
            self.act1 = nn.GELU()
            self.conv2 = nn.Conv1d(n_out, n_out, ks, dilation=dilation, padding=padding)
            self.chomp2 = Chomp1d(padding)
            self.act2 = nn.GELU()
            self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
            self.relu = nn.ReLU()

            # Load weights
            with torch.no_grad():
                self.conv1.weight.copy_(torch.from_numpy(weights[f"{prefix}.conv1.weight"]))
                self.conv1.bias.copy_(torch.from_numpy(weights[f"{prefix}.conv1.bias"]))
                self.conv2.weight.copy_(torch.from_numpy(weights[f"{prefix}.conv2.weight"]))
                self.conv2.bias.copy_(torch.from_numpy(weights[f"{prefix}.conv2.bias"]))
                if self.downsample is not None:
                    self.downsample.weight.copy_(torch.from_numpy(weights[f"{prefix}.downsample.weight"]))
                    self.downsample.bias.copy_(torch.from_numpy(weights[f"{prefix}.downsample.bias"]))

        def forward(self, x):
            out = self.act1(self.chomp1(self.conv1(x)))
            out = self.act2(self.chomp2(self.conv2(out)))
            res = self.downsample(x) if self.downsample is not None else x
            return self.relu(out + res)

    class CioffiTCN(nn.Module):
        def __init__(self):
            super().__init__()
            channels = [INPUT_DIM] + NUM_CHANNELS
            self.blocks = nn.ModuleList()
            for i in range(len(NUM_CHANNELS)):
                self.blocks.append(TemporalBlock(
                    channels[i], channels[i+1], KERNEL_SIZE,
                    dilation=2**i, prefix=f"block_{i}"))
            self.linear = nn.Linear(NUM_CHANNELS[-1], OUTPUT_DIM)
            with torch.no_grad():
                self.linear.weight.copy_(torch.from_numpy(weights["linear.weight"]))
                self.linear.bias.copy_(torch.from_numpy(weights["linear.bias"]))

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return self.linear(x[:, :, -1])

    model = CioffiTCN()
    model.eval()
    if use_fp16:
        model = model.half()

    param_count = sum(p.numel() for p in model.parameters())
    return model, param_count


def export_onnx(onnx_path, weights=None, use_fp16=False):
    """Export the Cioffi TCN to ONNX for TensorRT consumption.

    Always exports in FP32 — TensorRT handles precision conversion internally.
    """
    import torch

    if weights is None:
        weights = generate_weights()

    model, param_count = build_pytorch_tcn(weights, use_fp16=False)
    dummy = torch.randn(1, INPUT_DIM, SEQ_LEN, dtype=torch.float32)

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["imu_data"],
        output_names=["displacement"],
        dynamic_axes=None,  # fixed batch=1 for inference benchmarking
        opset_version=17,
        dynamo=False,  # use legacy tracer (no onnxscript dependency)
    )
    print(f"  Exported ONNX: {onnx_path} ({param_count:,} params)")
    return param_count


# ═══════════════════════════════════════════════════════════════════════════════
# tinygrad implementation
# ═══════════════════════════════════════════════════════════════════════════════

def build_tinygrad_tcn(weights=None, use_fp16=True):
    """Build the Cioffi TCN in tinygrad. Returns (model, param_count).

    Faithfully reproduces the paper's architecture:
      7 TemporalBlocks with causal dilated Conv1d → GELU → residual → ReLU
      Final: Linear(128, 3) on last timestep
    """
    from tinygrad import Tensor, dtypes

    if weights is None:
        weights = generate_weights()

    np_dtype = np.float16 if use_fp16 else np.float32

    class TgTemporalBlock:
        def __init__(self, n_in, n_out, ks, dilation, prefix):
            self.padding = (ks - 1) * dilation
            self.chomp = self.padding

            self.conv1_w = Tensor(weights[f"{prefix}.conv1.weight"].astype(np_dtype))
            self.conv1_b = Tensor(weights[f"{prefix}.conv1.bias"].astype(np_dtype))
            self.conv2_w = Tensor(weights[f"{prefix}.conv2.weight"].astype(np_dtype))
            self.conv2_b = Tensor(weights[f"{prefix}.conv2.bias"].astype(np_dtype))
            self.dilation = dilation

            self.has_downsample = n_in != n_out
            if self.has_downsample:
                self.ds_w = Tensor(weights[f"{prefix}.downsample.weight"].astype(np_dtype))
                self.ds_b = Tensor(weights[f"{prefix}.downsample.bias"].astype(np_dtype))

        def __call__(self, x):
            # Conv1d with causal padding + chomp
            p = self.padding
            # Causal pad on sequence dim (dim 2) for 4D tensor (N, C, seq, 1)
            # PyTorch-style flat padding: (W_left, W_right, H_left, H_right)
            out = x.pad((0, 0, p, 0))
            out = out.conv2d(self.conv1_w.unsqueeze(3), self.conv1_b, dilation=(self.dilation, 1))
            out = out.gelu()

            out = out.pad((0, 0, p, 0))
            out = out.conv2d(self.conv2_w.unsqueeze(3), self.conv2_b, dilation=(self.dilation, 1))
            out = out.gelu()

            if self.has_downsample:
                res = x.conv2d(self.ds_w.unsqueeze(3), self.ds_b)
            else:
                res = x
            return (out + res).relu()

    class TgCioffiTCN:
        def __init__(self):
            channels = [INPUT_DIM] + NUM_CHANNELS
            self.blocks = []
            for i in range(len(NUM_CHANNELS)):
                self.blocks.append(TgTemporalBlock(
                    channels[i], channels[i+1], KERNEL_SIZE,
                    dilation=2**i, prefix=f"block_{i}"))
            self.linear_w = Tensor(weights["linear.weight"].astype(np_dtype))
            self.linear_b = Tensor(weights["linear.bias"].astype(np_dtype))

        def __call__(self, x):
            # x shape: (1, 6, 200) → treat as (1, 6, 200, 1) for conv2d
            h = x.unsqueeze(3)
            for block in self.blocks:
                h = block(h)
            # h shape: (1, 128, 200, 1) → take last timestep
            h = h[:, :, -1, 0]  # (1, 128)
            return h.linear(self.linear_w.T, self.linear_b)

    model = TgCioffiTCN()
    param_count = sum(
        w.astype(np_dtype).size for w in weights.values()
    )
    return model, param_count
