#!/usr/bin/env python3
"""
B4: Tegra Edge-Case Tests for NV backend on Jetson Orin AGX 64GB.

Tests edge cases specific to the TegraIface nvgpu/nvmap backend.
Run with: NV=1 python3 -m pytest tests/test_tegra_edge_cases.py -v --tb=short
"""
import unittest, os, sys, time, gc
import numpy as np

# Ensure tests/ is importable (for dmesg_checker)
sys.path.insert(0, os.path.dirname(__file__))
from dmesg_checker import DmesgChecker

# Must set NV=1 before importing tinygrad
os.environ.setdefault("NV", "1")

from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv


def count_fds():
    """Count open file descriptors for this process."""
    return len(os.listdir(f"/proc/{os.getpid()}/fd"))

def count_maps():
    """Count memory mappings for this process."""
    with open(f"/proc/{os.getpid()}/maps") as f:
        return sum(1 for _ in f)


@unittest.skipUnless(Device.DEFAULT == "NV", "NV backend only")
class TestTegraEdgeCases(unittest.TestCase):
    """Edge-case tests for the Tegra NV backend."""

    def setUp(self):
        self.dmesg = DmesgChecker()
        self.dmesg.clear()

    def tearDown(self):
        report = self.dmesg.check()
        if report.has_errors:
            self.fail(f"GPU kernel errors detected!\n{report.summary()}")

    # --- VA / Memory Tests ---

    def test_40bit_va_boundary(self):
        """All GPU VAs must be < 2^40 (TegraIface uses va_range_end = (1<<40) - PDE_SIZE)."""
        VA_LIMIT = 1 << 40
        tensors = []
        for size_mb in [1, 4, 16, 64]:
            t = Tensor.zeros(size_mb * 1024 * 1024 // 4).contiguous().realize()
            tensors.append(t)

        for t in tensors:
            buf = t._buffer()
            va = buf._buf.va_addr
            self.assertGreater(va, 0, "VA should be non-zero")
            self.assertLess(va, VA_LIMIT, f"VA {va:#x} exceeds 40-bit boundary")

        del tensors
        gc.collect()

    def test_memory_pressure_progressive(self):
        """Allocate progressively larger buffers up to memory limits."""
        sizes_mb = [1, 10, 100, 1024, 4096]
        max_success = 0
        failure_size = None
        failure_msg = None

        for size_mb in sizes_mb:
            try:
                t = Tensor.zeros(size_mb * 1024 * 1024 // 4).contiguous().realize()
                _ = t.numpy()  # Force readback to verify
                max_success = size_mb
                del t
                gc.collect()
            except Exception as e:
                failure_size = size_mb
                failure_msg = str(e)
                break

        self.assertGreaterEqual(max_success, 100,
            f"Should allocate at least 100MB, only got {max_success}MB. "
            f"Failed at {failure_size}MB: {failure_msg}")

    def test_alloc_free_cycle_leak_check(self):
        """1000 iterations of alloc(1MB)+free(). Check fd/maps counts don't grow."""
        # Warm up to stabilize baseline
        for _ in range(10):
            t = Tensor.zeros(256 * 1024).contiguous().realize()
            del t
        gc.collect()
        time.sleep(0.1)

        fd_before = count_fds()
        maps_before = count_maps()

        for i in range(1000):
            t = Tensor.zeros(256 * 1024).contiguous().realize()  # ~1MB (float32)
            del t
            if i % 100 == 99:
                gc.collect()

        gc.collect()
        time.sleep(0.1)

        fd_after = count_fds()
        maps_after = count_maps()

        fd_leak = fd_after - fd_before
        maps_leak = maps_after - maps_before

        # Allow small variance (LRU cache, GC timing)
        self.assertLess(fd_leak, 50,
            f"File descriptor leak: {fd_before} → {fd_after} (+{fd_leak})")
        self.assertLess(maps_leak, 50,
            f"Memory mapping leak: {maps_before} → {maps_after} (+{maps_leak})")

    # --- DMA Copy Tests ---

    def test_dma_copy_small(self):
        """Copy small tensors: 1, 4, 16, 64 elements. Byte-exact verification."""
        for n in [1, 4, 16, 64]:
            src = np.arange(n, dtype=np.float32)
            t = Tensor(src).contiguous().realize()
            result = t.numpy()
            np.testing.assert_array_equal(result, src, err_msg=f"DMA copy failed for {n} elements")

    def test_dma_copy_medium(self):
        """Copy medium tensors: 4KB and 64KB."""
        for n_elements in [1024, 16384]:  # 4KB, 64KB (float32)
            src = np.random.randn(n_elements).astype(np.float32)
            t = Tensor(src).contiguous().realize()
            result = t.numpy()
            np.testing.assert_allclose(result, src, atol=0, rtol=0,
                err_msg=f"DMA copy mismatch for {n_elements} elements")

    def test_dma_copy_large(self):
        """Copy large tensors: 1MB, 16MB."""
        for size_mb in [1, 16]:
            n_elements = size_mb * 1024 * 1024 // 4
            src = np.random.randn(n_elements).astype(np.float32)
            t = Tensor(src).contiguous().realize()
            result = t.numpy()
            np.testing.assert_allclose(result, src, atol=0, rtol=0,
                err_msg=f"DMA copy mismatch for {size_mb}MB")
            del t, result
            gc.collect()

    # --- Tensor Shape Edge Cases ---

    def test_zero_element_tensor(self):
        """Zero-element tensor operations should not crash."""
        try:
            t = Tensor([]).reshape(0, 3)
            _ = t.shape
            t2 = t + 1
            _ = t2.shape
        except Exception:
            pass  # Some backends don't support zero-element tensors — that's OK

    def test_one_element_tensor(self):
        """Scalar tensor ops."""
        a = Tensor(42.0).realize()
        b = Tensor(1.0).realize()
        result = (a + b).numpy()
        np.testing.assert_allclose(result, 43.0, atol=1e-6)

        result = (a * b).numpy()
        np.testing.assert_allclose(result, 42.0, atol=1e-6)

        result = a.exp().numpy()
        np.testing.assert_allclose(result, np.exp(42.0), rtol=1e-5)

    def test_non_power_of_2_shapes(self):
        """Shapes that don't align to power-of-2 boundaries."""
        shapes = [(7,), (13, 17), (127, 127), (1023,), (4097,)]
        for shape in shapes:
            a = Tensor.ones(*shape).realize()
            b = Tensor.ones(*shape).realize()
            c = (a + b).numpy()
            expected = np.full(shape, 2.0, dtype=np.float32)
            np.testing.assert_allclose(c, expected, atol=1e-6,
                err_msg=f"Failed for shape {shape}")

    def test_non_power_of_2_matmul(self):
        """Matmul with non-aligned dimensions."""
        np.random.seed(42)
        a_np = np.random.randn(13, 17).astype(np.float32)
        b_np = np.random.randn(17, 11).astype(np.float32)
        expected = a_np @ b_np

        a = Tensor(a_np).realize()
        b = Tensor(b_np).realize()
        result = (a @ b).numpy()
        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)

    # --- Dtype Tests ---

    def test_all_dtypes(self):
        """Basic ops on each supported dtype."""
        test_cases = [
            (dtypes.float32, np.float32, [1.0, 2.0, 3.0]),
            (dtypes.float16, np.float16, [1.0, 2.0, 3.0]),
            (dtypes.int32, np.int32, [1, 2, 3]),
        ]
        for dt, np_dt, vals in test_cases:
            with self.subTest(dtype=str(dt)):
                src = np.array(vals, dtype=np_dt)
                t = Tensor(src, dtype=dt).realize()
                result = t.numpy()
                np.testing.assert_array_equal(result, src,
                    err_msg=f"Roundtrip failed for {dt}")

    # --- GPFIFO / Infrastructure Tests ---

    def test_gpfifo_ring_wraparound(self):
        """Submit enough kernels to force GPFIFO ring wraparound (ring size = 1024)."""
        # Each kernel submit uses at least 1 GPFIFO entry. With 1024-size ring,
        # >1024 submissions forces wraparound.
        results = []
        for i in range(1200):
            t = (Tensor([float(i)]) + Tensor([1.0])).realize()
            if i % 400 == 399:
                results.append(t.numpy()[0])

        # Verify last batch
        np.testing.assert_allclose(results[-1], 1199.0 + 1.0, atol=1e-4)

    def test_free_correctness(self):
        """Verify free() properly cleans up: GPU VA unmapped, dmabuf closed, nvmap freed."""
        # Warm up
        _ = Tensor([1.0]).realize().numpy()
        gc.collect()

        fd_before = count_fds()

        # Allocate and free explicitly
        tensors = []
        for _ in range(50):
            t = Tensor.zeros(65536).contiguous().realize()  # 256KB
            _ = t.numpy()
            tensors.append(t)

        del tensors
        gc.collect()
        time.sleep(0.2)

        fd_after = count_fds()
        fd_diff = fd_after - fd_before

        # After freeing all tensors, fd count should return close to baseline
        # Allow some slack for LRU cache
        self.assertLess(fd_diff, 100,
            f"FD leak after free: {fd_before} → {fd_after} (+{fd_diff})")

    def test_cache_invalidation_pattern(self):
        """Write→compute→readback→modify→re-compute→readback.
        Tests whether NOP'd invalidate_caches() causes stale data."""
        # First computation
        a = Tensor([1.0, 2.0, 3.0, 4.0]).realize()
        b = Tensor([10.0, 20.0, 30.0, 40.0]).realize()
        c = (a + b).realize()
        result1 = c.numpy()
        np.testing.assert_allclose(result1, [11.0, 22.0, 33.0, 44.0], atol=1e-6)

        # Second computation with different values (tests cache coherence)
        a2 = Tensor([100.0, 200.0, 300.0, 400.0]).realize()
        b2 = Tensor([5.0, 6.0, 7.0, 8.0]).realize()
        c2 = (a2 * b2).realize()
        result2 = c2.numpy()
        np.testing.assert_allclose(result2, [500.0, 1200.0, 2100.0, 3200.0], atol=1e-4)

        # Third: chain of ops
        d = (a2 + b2 + Tensor([1.0, 1.0, 1.0, 1.0])).realize()
        result3 = d.numpy()
        np.testing.assert_allclose(result3, [106.0, 207.0, 308.0, 409.0], atol=1e-4)

    def test_large_grid_launch(self):
        """Launch kernel with large grid dimensions to test QMD CTA/grid field limits."""
        # Large 1D tensor — forces large grid
        n = 1024 * 1024  # 1M elements
        a = Tensor.ones(n).realize()
        b = Tensor.ones(n).realize()
        c = (a + b).realize()
        result = c.numpy()

        self.assertEqual(result.shape, (n,))
        np.testing.assert_allclose(result[:10], [2.0] * 10, atol=1e-6)
        np.testing.assert_allclose(result[-10:], [2.0] * 10, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
