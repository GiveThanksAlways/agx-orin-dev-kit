"""models.py — Architecture definitions for tinygrad NV=1 vs TensorRT benchmarks.

Defines MLPs, 1D-CNNs, and hybrid CNN+MLP architectures sized for real-world
drone/robot control loops. Each model can be instantiated in tinygrad or exported
to ONNX for TensorRT consumption.

All models use FP16 weights and operate on FP16 tensors (matching Orin's
Tensor Cores and our existing NV=1 benchmarks).

Architecture rationale (from drone/robotics literature):
  MLP:    Standard for learned controllers (Hwangbo 2017, Kaufmann 2020)
  1D-CNN: Temporal feature extraction from IMU time series (Loquercio 2021)
  Hybrid: CNN encoder + MLP policy head (common in end-to-end systems)
"""
import numpy as np

SEED = 42
IN_DIM = 12       # pos(3) + vel(3) + rpy(3) + gyro(3)
OUT_DIM = 4       # thrust + roll_rate + pitch_rate + yaw_rate
SEQ_LEN = 16      # temporal window for CNN models (16 IMU samples @ 1kHz = 16ms)


# ═══════════════════════════════════════════════════════════════════════════════
# Weight generation (deterministic, shared across frameworks)
# ═══════════════════════════════════════════════════════════════════════════════

def _kaiming_init(rng, fan_in, fan_out):
    """Kaiming He initialization (common in RL policy networks)."""
    s = np.sqrt(2.0 / fan_in)
    w = (rng.randn(fan_out, fan_in) * s).astype(np.float32)
    b = np.zeros(fan_out, dtype=np.float32)
    return w, b


def _conv1d_init(rng, in_ch, out_ch, kernel_size):
    """Kaiming init for 1D convolution weights: (out_ch, in_ch, kernel_size)."""
    fan_in = in_ch * kernel_size
    s = np.sqrt(2.0 / fan_in)
    w = (rng.randn(out_ch, in_ch, kernel_size) * s).astype(np.float32)
    b = np.zeros(out_ch, dtype=np.float32)
    return w, b


# ═══════════════════════════════════════════════════════════════════════════════
# Model configs — what we benchmark
# ═══════════════════════════════════════════════════════════════════════════════

# MLP configs: (name, hidden_dims, description, robotics_use_case)
MLP_CONFIGS = [
    ("mlp_5k",     [64, 64],           "12→64→64→4 (~5K)",
     "PID replacement, rate gyro filter"),
    ("mlp_18k",    [128, 128],         "12→128→128→4 (~18K)",
     "Learned hover controller, sensor fusion"),
    ("mlp_135k",   [256, 256, 256],    "12→256→256→256→4 (~135K)",
     "Full attitude policy, SLAM feature encoder"),
    ("mlp_270k",   [512, 512],         "12→512→512→4 (~270K)",
     "Visual-inertial nav, end-to-end landing"),
    ("mlp_530k",   [512, 512, 512],    "12→512→512→512→4 (~530K)",
     "GPU/NEON crossover zone"),
    ("mlp_1m",     [1024, 1024],       "12→1024→1024→4 (~1.1M)",
     "Path planner, obstacle avoidance"),
    ("mlp_2m",     [1024, 1024, 1024], "12→1024→1024→1024→4 (~2.1M)",
     "Large policy, multi-agent coordination"),
]

# 1D-CNN configs: (name, conv_layers, mlp_head, description, use_case)
# conv_layers: list of (out_channels, kernel_size, stride)
# These process IN_DIM channels over SEQ_LEN timesteps
CNN_CONFIGS = [
    ("cnn_small",
     [(32, 3, 1), (64, 3, 1)],
     [64],
     "Conv(32,k3)→Conv(64,k3)→FC(64)→4 (~30K)",
     "IMU denoising + attitude estimate from 16ms window"),
    ("cnn_medium",
     [(64, 3, 1), (128, 3, 1), (128, 3, 1)],
     [128],
     "Conv(64,k3)→Conv(128,k3)→Conv(128,k3)→FC(128)→4 (~150K)",
     "Temporal feature extraction for agile flight"),
    ("cnn_large",
     [(128, 3, 1), (256, 3, 1), (256, 3, 1)],
     [256, 128],
     "Conv(128,k3)→Conv(256,k3)→Conv(256,k3)→FC(256)→FC(128)→4 (~500K)",
     "End-to-end state estimation from IMU time series"),
]

# Hybrid CNN+MLP configs: (name, cnn_layers, mlp_layers, description, use_case)
# CNN processes temporal IMU window, MLP processes current state, outputs fused
HYBRID_CONFIGS = [
    ("hybrid_small",
     [(32, 3, 1), (64, 3, 1)],     # CNN encoder on IMU window
     [128, 64],                      # MLP head (CNN features + current state → action)
     "CNN(32,64)+MLP(128,64)→4 (~50K)",
     "Lightweight sensor fusion: IMU history + current state"),
    ("hybrid_medium",
     [(64, 3, 1), (128, 3, 1)],
     [256, 128],
     "CNN(64,128)+MLP(256,128)→4 (~200K)",
     "Agile flight: temporal patterns + reactive control"),
    ("hybrid_large",
     [(128, 3, 1), (256, 3, 1), (256, 3, 1)],
     [512, 256, 128],
     "CNN(128,256,256)+MLP(512,256,128)→4 (~700K)",
     "Full autonomy: history-aware perception + policy"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# tinygrad model builders
# ═══════════════════════════════════════════════════════════════════════════════

def build_tinygrad_mlp(hidden_dims, seed=SEED):
    """Build an MLP in tinygrad with deterministic weights. Returns (model, params_count)."""
    from tinygrad import Tensor, dtypes
    from tinygrad import nn as tg_nn

    rng = np.random.RandomState(seed)
    dims = [IN_DIM] + list(hidden_dims) + [OUT_DIM]

    layers = []
    total_params = 0
    for i in range(len(dims) - 1):
        fi, fo = dims[i], dims[i + 1]
        w, b = _kaiming_init(rng, fi, fo)
        lin = tg_nn.Linear(fi, fo)
        lin.weight = Tensor(w.astype(np.float16))
        lin.bias = Tensor(b.astype(np.float16))
        layers.append(lin)
        total_params += fo * fi + fo

    class MLP:
        def __init__(self):
            self.layers = layers
        def __call__(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:
                    x = x.relu()
            return x

    return MLP(), total_params


def build_tinygrad_cnn(conv_config, mlp_head_dims, seed=SEED):
    """Build a 1D-CNN in tinygrad. Input: (1, IN_DIM, SEQ_LEN). Returns (model, params)."""
    from tinygrad import Tensor, dtypes

    rng = np.random.RandomState(seed)
    conv_layers = []
    total_params = 0
    in_ch = IN_DIM

    # Build conv layers
    for out_ch, ks, stride in conv_config:
        w, b = _conv1d_init(rng, in_ch, out_ch, ks)
        conv_layers.append((
            Tensor(w.astype(np.float16)),
            Tensor(b.astype(np.float16)),
            stride
        ))
        total_params += out_ch * in_ch * ks + out_ch
        in_ch = out_ch

    # Compute flattened size after convolutions
    seq = SEQ_LEN
    for _, ks, stride in conv_config:
        seq = (seq - ks) // stride + 1  # valid padding
    flat_size = in_ch * seq

    # Build MLP head
    from tinygrad import nn as tg_nn
    mlp_layers = []
    dims = [flat_size] + list(mlp_head_dims) + [OUT_DIM]
    for i in range(len(dims) - 1):
        fi, fo = dims[i], dims[i + 1]
        w, b = _kaiming_init(rng, fi, fo)
        lin = tg_nn.Linear(fi, fo)
        lin.weight = Tensor(w.astype(np.float16))
        lin.bias = Tensor(b.astype(np.float16))
        mlp_layers.append(lin)
        total_params += fo * fi + fo

    class CNN1D:
        def __init__(self):
            self.convs = conv_layers
            self.mlp = mlp_layers
        def __call__(self, x):
            # x: (1, IN_DIM, SEQ_LEN)
            for w, b, stride in self.convs:
                # Manual conv1d: weight (out_ch, in_ch, ks), input (batch, in_ch, seq)
                x = x.conv2d(w.reshape(*w.shape, 1), b, stride=(stride, 1),
                             padding=0).reshape(x.shape[0], w.shape[0], -1)
            # Global average pool over time dimension
            x = x.mean(axis=2)  # (1, out_ch)
            for i, layer in enumerate(self.mlp):
                x = layer(x)
                if i < len(self.mlp) - 1:
                    x = x.relu()
            return x

    return CNN1D(), total_params


def build_tinygrad_hybrid(conv_config, mlp_head_dims, seed=SEED):
    """Build a hybrid CNN+MLP in tinygrad.

    Two inputs:
      - imu_window: (1, IN_DIM, SEQ_LEN) — temporal IMU history for CNN
      - current_state: (1, IN_DIM) — current state for direct MLP processing

    The CNN extracts temporal features, which are concatenated with current_state
    and fed through the MLP head.
    """
    from tinygrad import Tensor, dtypes
    from tinygrad import nn as tg_nn

    rng = np.random.RandomState(seed)
    conv_layers = []
    total_params = 0
    in_ch = IN_DIM

    for out_ch, ks, stride in conv_config:
        w, b = _conv1d_init(rng, in_ch, out_ch, ks)
        conv_layers.append((
            Tensor(w.astype(np.float16)),
            Tensor(b.astype(np.float16)),
            stride
        ))
        total_params += out_ch * in_ch * ks + out_ch
        in_ch = out_ch

    seq = SEQ_LEN
    for _, ks, stride in conv_config:
        seq = (seq - ks) // stride + 1
    cnn_out_dim = in_ch  # after global avg pool

    # MLP head: (cnn_features + current_state) → action
    mlp_in_dim = cnn_out_dim + IN_DIM
    mlp_layers = []
    dims = [mlp_in_dim] + list(mlp_head_dims) + [OUT_DIM]
    for i in range(len(dims) - 1):
        fi, fo = dims[i], dims[i + 1]
        w, b = _kaiming_init(rng, fi, fo)
        lin = tg_nn.Linear(fi, fo)
        lin.weight = Tensor(w.astype(np.float16))
        lin.bias = Tensor(b.astype(np.float16))
        mlp_layers.append(lin)
        total_params += fo * fi + fo

    class HybridModel:
        def __init__(self):
            self.convs = conv_layers
            self.mlp = mlp_layers
        def __call__(self, imu_window, current_state):
            # CNN on temporal window
            x = imu_window
            for w, b, stride in self.convs:
                x = x.conv2d(w.reshape(*w.shape, 1), b, stride=(stride, 1),
                             padding=0).reshape(x.shape[0], w.shape[0], -1)
            x = x.mean(axis=2)  # (1, cnn_out_dim)
            # Concatenate with current state
            x = x.cat(current_state, dim=1)  # (1, cnn_out_dim + IN_DIM)
            # MLP head
            for i, layer in enumerate(self.mlp):
                x = layer(x)
                if i < len(self.mlp) - 1:
                    x = x.relu()
            return x

    return HybridModel(), total_params


# ═══════════════════════════════════════════════════════════════════════════════
# ONNX export (for TensorRT)
# ═══════════════════════════════════════════════════════════════════════════════

def _numpy_mlp_forward(layers_f16, x):
    """Reference NumPy forward pass for an MLP (for ONNX weight verification)."""
    for i, (w, b) in enumerate(layers_f16):
        x = x @ w.T + b
        if i < len(layers_f16) - 1:
            x = np.maximum(x, 0)
    return x


def export_mlp_onnx(hidden_dims, path, seed=SEED):
    """Export an MLP to ONNX format for TensorRT. Returns (onnx_path, params_count, ref_output).

    We build the ONNX graph manually (no framework dependency) to keep
    the presentation self-contained — just numpy + struct.
    """
    rng = np.random.RandomState(seed)
    dims = [IN_DIM] + list(hidden_dims) + [OUT_DIM]

    # We'll build the ONNX protobuf manually using the onnx helper format.
    # This avoids needing `import onnx` which may not be in the flake.
    # Instead, we write a raw ONNX file using the binary protobuf format.
    #
    # Actually — let's generate a TensorRT engine directly from C API
    # in bench_tensorrt.py. The ONNX route adds a dependency we don't need.
    # TensorRT's INetworkDefinition API can build the same MLP/CNN directly.
    #
    # For now, return the weight data that bench_tensorrt.py will consume.
    layers_f16 = []
    total_params = 0
    for i in range(len(dims) - 1):
        fi, fo = dims[i], dims[i + 1]
        w, b = _kaiming_init(rng, fi, fo)
        layers_f16.append((w.astype(np.float16), b.astype(np.float16)))
        total_params += fo * fi + fo

    # Compute reference output
    test_input = np.random.RandomState(99).randn(1, IN_DIM).astype(np.float16)
    ref_output = _numpy_mlp_forward(layers_f16, test_input.astype(np.float32)).astype(np.float16)

    # Save weights as npz for TensorRT builder to consume
    weight_dict = {}
    for i, (w, b) in enumerate(layers_f16):
        weight_dict[f"w{i}"] = w
        weight_dict[f"b{i}"] = b
    np.savez(path, **weight_dict, test_input=test_input, ref_output=ref_output)

    return path, total_params, ref_output


def export_cnn_onnx(conv_config, mlp_head_dims, path, seed=SEED):
    """Export CNN weights as npz for TensorRT builder."""
    rng = np.random.RandomState(seed)
    weight_dict = {}
    total_params = 0
    in_ch = IN_DIM

    for i, (out_ch, ks, stride) in enumerate(conv_config):
        w, b = _conv1d_init(rng, in_ch, out_ch, ks)
        weight_dict[f"conv_w{i}"] = w.astype(np.float16)
        weight_dict[f"conv_b{i}"] = b.astype(np.float16)
        weight_dict[f"conv_stride{i}"] = np.array([stride])
        total_params += out_ch * in_ch * ks + out_ch
        in_ch = out_ch

    seq = SEQ_LEN
    for _, ks, stride in conv_config:
        seq = (seq - ks) // stride + 1
    flat_size = in_ch * seq

    dims = [flat_size] + list(mlp_head_dims) + [OUT_DIM]
    for i in range(len(dims) - 1):
        fi, fo = dims[i], dims[i + 1]
        w, b = _kaiming_init(rng, fi, fo)
        weight_dict[f"fc_w{i}"] = w.astype(np.float16)
        weight_dict[f"fc_b{i}"] = b.astype(np.float16)
        total_params += fo * fi + fo

    weight_dict["n_conv_layers"] = np.array([len(conv_config)])
    weight_dict["n_fc_layers"] = np.array([len(dims) - 1])

    test_input = np.random.RandomState(99).randn(1, IN_DIM, SEQ_LEN).astype(np.float16)
    np.savez(path, **weight_dict, test_input=test_input)
    return path, total_params


def export_hybrid_onnx(conv_config, mlp_head_dims, path, seed=SEED):
    """Export hybrid CNN+MLP weights as npz for TensorRT builder."""
    rng = np.random.RandomState(seed)
    weight_dict = {}
    total_params = 0
    in_ch = IN_DIM

    for i, (out_ch, ks, stride) in enumerate(conv_config):
        w, b = _conv1d_init(rng, in_ch, out_ch, ks)
        weight_dict[f"conv_w{i}"] = w.astype(np.float16)
        weight_dict[f"conv_b{i}"] = b.astype(np.float16)
        weight_dict[f"conv_stride{i}"] = np.array([stride])
        total_params += out_ch * in_ch * ks + out_ch
        in_ch = out_ch

    cnn_out_dim = in_ch
    mlp_in_dim = cnn_out_dim + IN_DIM
    dims = [mlp_in_dim] + list(mlp_head_dims) + [OUT_DIM]
    for i in range(len(dims) - 1):
        fi, fo = dims[i], dims[i + 1]
        w, b = _kaiming_init(rng, fi, fo)
        weight_dict[f"fc_w{i}"] = w.astype(np.float16)
        weight_dict[f"fc_b{i}"] = b.astype(np.float16)
        total_params += fo * fi + fo

    weight_dict["n_conv_layers"] = np.array([len(conv_config)])
    weight_dict["n_fc_layers"] = np.array([len(dims) - 1])
    weight_dict["cnn_out_dim"] = np.array([cnn_out_dim])

    test_imu = np.random.RandomState(99).randn(1, IN_DIM, SEQ_LEN).astype(np.float16)
    test_state = np.random.RandomState(98).randn(1, IN_DIM).astype(np.float16)
    np.savez(path, **weight_dict, test_imu=test_imu, test_state=test_state)
    return path, total_params
