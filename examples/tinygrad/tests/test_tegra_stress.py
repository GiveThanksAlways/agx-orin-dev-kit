#!/usr/bin/env python3
"""
B5: Tegra Stress Tests for NV backend on Jetson Orin AGX 64GB.

Longer-running tests that exercise sustained workloads, backpressure, and resource churn.
Run with: NV=1 python3 -m pytest tests/test_tegra_stress.py -v --tb=short -s
"""
import unittest, os, sys, time, gc
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dmesg_checker import DmesgChecker

os.environ.setdefault("NV", "1")

from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv


def count_fds():
    return len(os.listdir(f"/proc/{os.getpid()}/fd"))


@unittest.skipUnless(Device.DEFAULT == "NV", "NV backend only")
class TestTegraStress(unittest.TestCase):
    """Stress tests for the Tegra NV backend."""

    def setUp(self):
        self.dmesg = DmesgChecker()
        self.dmesg.clear()

    def tearDown(self):
        report = self.dmesg.check()
        if report.has_errors:
            self.fail(f"GPU kernel errors detected!\n{report.summary()}")

    def test_rapid_kernel_launches_10k(self):
        """Submit 10,000 trivial kernels back-to-back."""
        acc = Tensor([0.0]).realize()
        one = Tensor([1.0]).realize()

        for i in range(10000):
            acc = (acc + one).realize()

        result = acc.numpy()[0]
        np.testing.assert_allclose(result, 10000.0, atol=1.0,
            err_msg=f"Expected 10000.0, got {result}")

    def test_signal_chain_pipeline(self):
        """Build chained compute pipeline, 1000 iterations."""
        x = Tensor([1.0]).realize()
        for i in range(1000):
            x = (x + Tensor([0.001])).realize()

        result = x.numpy()[0]
        expected = 1.0 + 0.001 * 1000
        np.testing.assert_allclose(result, expected, atol=0.1,
            err_msg=f"Signal chain drift: expected ~{expected}, got {result}")

    def test_sustained_matmul_60s(self):
        """1024x1024 matmul in loop for 60 seconds. Check for hangs and numerical drift."""
        np.random.seed(42)
        a_np = np.random.randn(1024, 1024).astype(np.float32) * 0.01
        b_np = np.random.randn(1024, 1024).astype(np.float32) * 0.01
        expected = a_np @ b_np

        a = Tensor(a_np).realize()
        b = Tensor(b_np).realize()

        # Warmup
        c = (a @ b).realize()
        _ = c.numpy()

        start = time.time()
        count = 0
        last_result = None
        while time.time() - start < 60:
            c = (a @ b).realize()
            if count % 100 == 0:
                last_result = c.numpy()
            count += 1

        elapsed = time.time() - start
        print(f"\n  Sustained matmul: {count} iterations in {elapsed:.1f}s "
              f"({count/elapsed:.1f} iter/s)")

        # Verify no numerical drift
        if last_result is not None:
            np.testing.assert_allclose(last_result, expected, atol=1e-2, rtol=1e-2,
                err_msg="Numerical drift detected in sustained matmul")

    def test_mixed_compute_copy(self):
        """Interleave compute and data transfer operations, 1000 iterations."""
        for i in range(1000):
            # Compute
            a = Tensor([float(i), float(i+1)]).realize()
            b = (a * Tensor([2.0, 3.0])).realize()

            # Transfer (readback)
            result = b.numpy()

            if i % 200 == 0:
                expected = np.array([i * 2.0, (i+1) * 3.0], dtype=np.float32)
                np.testing.assert_allclose(result, expected, atol=1e-3,
                    err_msg=f"Mismatch at iteration {i}")

    def test_backpressure(self):
        """Submit commands faster than GPU can execute. Verify GPFIFO handles backpressure."""
        # Build up a large batch of pending ops without realizing
        tensors = []
        base = Tensor.ones(1024).realize()
        x = base
        for i in range(500):
            x = x + Tensor.ones(1024)
            if i % 50 == 49:
                x = x.realize()
                tensors.append(x)

        # Now realize the final result
        x = x.realize()
        result = x.numpy()

        # Should be 1 + 500 = 501
        np.testing.assert_allclose(result[:5], [501.0] * 5, atol=1.0,
            err_msg="Backpressure test: incorrect result")

    def test_shared_memory_kernel(self):
        """Launch kernels that use shared memory (matmul triggers this)."""
        np.random.seed(42)
        # Matmul uses shared memory for tiling
        a = Tensor(np.random.randn(256, 256).astype(np.float32)).realize()
        b = Tensor(np.random.randn(256, 256).astype(np.float32)).realize()

        for _ in range(10):
            c = (a @ b).realize()
            _ = c.numpy()

        # Verify correctness
        expected = a.numpy() @ b.numpy()
        result = c.numpy()
        np.testing.assert_allclose(result, expected, atol=1e-2, rtol=1e-2)

    def test_concurrent_tensor_ops(self):
        """Multiple tensor operations scheduled rapidly (like a training step)."""
        np.random.seed(42)

        for step in range(100):
            # Simulate a mini training step
            x = Tensor(np.random.randn(64, 32).astype(np.float32)).realize()
            w = Tensor(np.random.randn(32, 16).astype(np.float32)).realize()
            b = Tensor(np.random.randn(16).astype(np.float32)).realize()

            # Forward pass: matmul + bias + relu
            out = (x @ w + b).relu().realize()
            result = out.numpy()

            self.assertEqual(result.shape, (64, 16))
            self.assertTrue(np.all(result >= 0), "ReLU output should be non-negative")

    def test_memory_churn(self):
        """Rapidly create and destroy tensors of varying sizes for 30s."""
        np.random.seed(42)
        sizes = [64, 256, 1024, 4096, 16384, 65536, 262144]

        fd_before = count_fds()
        start = time.time()
        count = 0

        while time.time() - start < 30:
            size = sizes[count % len(sizes)]
            t = Tensor.zeros(size).contiguous().realize()
            _ = t.numpy()
            del t
            count += 1
            if count % 500 == 0:
                gc.collect()

        gc.collect()
        time.sleep(0.2)

        fd_after = count_fds()
        fd_leak = fd_after - fd_before
        elapsed = time.time() - start

        print(f"\n  Memory churn: {count} alloc/free cycles in {elapsed:.1f}s "
              f"({count/elapsed:.0f}/s), fd leak: {fd_leak}")

        self.assertLess(fd_leak, 100,
            f"FD leak during memory churn: {fd_before} → {fd_after} (+{fd_leak})")


if __name__ == '__main__':
    unittest.main()
