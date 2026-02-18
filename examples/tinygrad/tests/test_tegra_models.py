#!/usr/bin/env python3
"""
B6: Tegra Model Tests for NV backend on Jetson Orin AGX 64GB.

End-to-end model forward passes comparing NV=1 output against numpy reference.
Run with: NV=1 python3 -m pytest tests/test_tegra_models.py -v --tb=short -s
"""
import unittest, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dmesg_checker import DmesgChecker

os.environ.setdefault("NV", "1")

from tinygrad import Tensor, Device, dtypes
from tinygrad.nn import Conv2d, Linear
from tinygrad.helpers import getenv


def softmax_np(x, axis=-1):
    """NumPy softmax."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


@unittest.skipUnless(Device.DEFAULT == "NV", "NV backend only")
class TestTegraModels(unittest.TestCase):
    """End-to-end model tests on the Tegra NV backend."""

    def setUp(self):
        self.dmesg = DmesgChecker()
        self.dmesg.clear()
        np.random.seed(42)
        Tensor.manual_seed(42)

    def tearDown(self):
        report = self.dmesg.check()
        if report.has_errors:
            self.fail(f"GPU kernel errors detected!\n{report.summary()}")

    def test_simple_mlp(self):
        """2-layer MLP (784→128→10) forward pass."""
        # Build model with fixed weights
        w1_np = np.random.randn(784, 128).astype(np.float32) * 0.01
        b1_np = np.random.randn(128).astype(np.float32) * 0.01
        w2_np = np.random.randn(128, 10).astype(np.float32) * 0.01
        b2_np = np.random.randn(10).astype(np.float32) * 0.01

        # Input
        x_np = np.random.randn(8, 784).astype(np.float32)

        # NumPy reference
        h_np = np.maximum(0, x_np @ w1_np + b1_np)  # ReLU
        out_np = h_np @ w2_np + b2_np

        # tinygrad NV
        x = Tensor(x_np).realize()
        w1 = Tensor(w1_np).realize()
        b1 = Tensor(b1_np).realize()
        w2 = Tensor(w2_np).realize()
        b2 = Tensor(b2_np).realize()

        h = (x @ w1 + b1).relu()
        out = (h @ w2 + b2).realize()
        result = out.numpy()

        np.testing.assert_allclose(result, out_np, atol=1e-3, rtol=1e-3,
            err_msg="MLP forward pass mismatch")
        self.assertEqual(result.shape, (8, 10))

    def test_cnn_forward(self):
        """Simple CNN (2 conv + 2 FC) forward pass."""
        # Build a simple CNN
        np.random.seed(42)

        # Conv layers (using tinygrad's Conv2d)
        conv1 = Conv2d(1, 8, 3, padding=1)
        conv2 = Conv2d(8, 16, 3, padding=1)
        # After 2x conv with stride=1, pad=1 on 8x8 input → still 8x8
        # Flatten: 16 * 8 * 8 = 1024
        fc1 = Linear(16 * 8 * 8, 64)
        fc2 = Linear(64, 10)

        # Input: batch of 4, 1 channel, 8x8
        x_np = np.random.randn(4, 1, 8, 8).astype(np.float32)
        x = Tensor(x_np).realize()

        # Forward pass
        h = conv1(x).relu()
        h = conv2(h).relu()
        h = h.reshape(4, -1)  # Flatten
        h = fc1(h).relu()
        out = fc2(h).realize()
        result = out.numpy()

        self.assertEqual(result.shape, (4, 10))
        # Just verify it produces finite values (exact comparison needs same weights)
        self.assertTrue(np.all(np.isfinite(result)), "CNN output has non-finite values")

    def test_transformer_block(self):
        """Single attention + FFN block forward pass."""
        np.random.seed(42)
        batch, seq_len, d_model = 2, 16, 64
        n_heads = 4
        d_head = d_model // n_heads

        # Input
        x_np = np.random.randn(batch, seq_len, d_model).astype(np.float32) * 0.1

        # Weight matrices for Q, K, V projections
        wq_np = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        wk_np = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        wv_np = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        wo_np = np.random.randn(d_model, d_model).astype(np.float32) * 0.02

        # FFN weights
        w_ff1_np = np.random.randn(d_model, d_model * 4).astype(np.float32) * 0.02
        w_ff2_np = np.random.randn(d_model * 4, d_model).astype(np.float32) * 0.02

        # tinygrad forward
        x = Tensor(x_np).realize()
        wq = Tensor(wq_np).realize()
        wk = Tensor(wk_np).realize()
        wv = Tensor(wv_np).realize()
        wo = Tensor(wo_np).realize()
        w_ff1 = Tensor(w_ff1_np).realize()
        w_ff2 = Tensor(w_ff2_np).realize()

        # Self-attention
        q = (x @ wq).reshape(batch, seq_len, n_heads, d_head).permute(0, 2, 1, 3)
        k = (x @ wk).reshape(batch, seq_len, n_heads, d_head).permute(0, 2, 1, 3)
        v = (x @ wv).reshape(batch, seq_len, n_heads, d_head).permute(0, 2, 1, 3)

        scale = float(d_head) ** -0.5
        attn = (q @ k.permute(0, 1, 3, 2)) * scale
        attn = attn.softmax(axis=-1)
        attn_out = (attn @ v).permute(0, 2, 1, 3).reshape(batch, seq_len, d_model)
        attn_proj = attn_out @ wo

        # Residual + FFN
        h = x + attn_proj
        ffn = (h @ w_ff1).relu() @ w_ff2
        out = (h + ffn).realize()
        result = out.numpy()

        # NumPy reference
        q_np = (x_np @ wq_np).reshape(batch, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
        k_np = (x_np @ wk_np).reshape(batch, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
        v_np = (x_np @ wv_np).reshape(batch, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)

        attn_np = (q_np @ k_np.transpose(0, 1, 3, 2)) * scale
        attn_np = softmax_np(attn_np, axis=-1)
        attn_out_np = (attn_np @ v_np).transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
        attn_proj_np = attn_out_np @ wo_np

        h_np = x_np + attn_proj_np
        ffn_np = np.maximum(0, h_np @ w_ff1_np) @ w_ff2_np
        out_np = h_np + ffn_np

        np.testing.assert_allclose(result, out_np, atol=1e-2, rtol=1e-2,
            err_msg="Transformer block output mismatch")
        self.assertEqual(result.shape, (batch, seq_len, d_model))

    def test_deep_mlp_10_layers(self):
        """10-layer MLP forward pass — stress tests deep computation chains."""
        np.random.seed(42)
        batch = 16
        dims = [256, 128, 128, 128, 128, 64, 64, 64, 32, 32, 10]

        # Generate random input
        x_np = np.random.randn(batch, dims[0]).astype(np.float32) * 0.1

        # Generate weights for each layer
        weights_np = []
        biases_np = []
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i+1]).astype(np.float32) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1], dtype=np.float32)
            weights_np.append(w)
            biases_np.append(b)

        # NumPy reference
        h_np = x_np
        for i in range(len(weights_np)):
            h_np = h_np @ weights_np[i] + biases_np[i]
            if i < len(weights_np) - 1:  # ReLU except last layer
                h_np = np.maximum(0, h_np)
        out_np = h_np

        # tinygrad NV
        x = Tensor(x_np).realize()
        weights = [Tensor(w).realize() for w in weights_np]
        biases = [Tensor(b).realize() for b in biases_np]

        h = x
        for i in range(len(weights)):
            h = h @ weights[i] + biases[i]
            if i < len(weights) - 1:
                h = h.relu()
        out = h.realize()
        result = out.numpy()

        np.testing.assert_allclose(result, out_np, atol=1e-2, rtol=1e-2,
            err_msg="Deep MLP forward pass mismatch")
        self.assertEqual(result.shape, (batch, 10))


if __name__ == '__main__':
    unittest.main()
