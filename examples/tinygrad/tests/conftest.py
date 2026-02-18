#!/usr/bin/env python3
"""
conftest.py — Shared pytest infrastructure for NV/Tegra backend testing.

Provides:
  - Backend detection fixtures
  - Output comparison helpers (NV=1 vs numpy reference)
  - Timing utilities for micro-benchmarks
  - Memory leak detection (fd/maps counting)
  - Automatic dmesg checking (opt-in via marker)

Usage:
  # In any test file:
  def test_something(backend):
      print(f"Running on {backend}")

  @pytest.mark.dmesg
  def test_gpu_op():
      '''This test will fail if GPU kernel errors appear in dmesg.'''
      result = (Tensor([1.0]) + Tensor([2.0])).numpy()
      assert result[0] == 3.0

  def test_matmul_perf(timed):
      a = Tensor.randn(1024, 1024).realize()
      b = Tensor.randn(1024, 1024).realize()
      stats = timed(lambda: (a @ b).realize())
      assert stats["median_ms"] < 50

  def test_no_leak(count_fds, count_maps):
      fd_before = count_fds()
      # ... do stuff ...
      assert count_fds() - fd_before < 10
"""

import os
import sys
import time
import gc
import pytest

# Ensure tests/ directory is importable
sys.path.insert(0, os.path.dirname(__file__))

from dmesg_checker import DmesgChecker


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def get_backend() -> str:
    """Return the active tinygrad backend name: 'NV', 'CUDA', 'CPU', etc."""
    from tinygrad import Device
    return Device.DEFAULT


@pytest.fixture
def backend():
    """Pytest fixture returning the active backend string."""
    return get_backend()


@pytest.fixture
def require_nv():
    """Skip test unless NV backend is active."""
    if get_backend() != "NV":
        pytest.skip("NV backend required")


@pytest.fixture
def require_gpu():
    """Skip test unless a GPU backend (NV or CUDA) is active."""
    if get_backend() not in ("NV", "CUDA"):
        pytest.skip("GPU backend required (NV or CUDA)")


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

def _timed(fn, warmup=10, iters=90):
    """
    Benchmark a callable. Returns dict with timing stats in milliseconds.

    Args:
        fn: Callable to benchmark (should include synchronize if needed)
        warmup: Number of warmup iterations (not measured)
        iters: Number of timed iterations

    Returns:
        dict with keys: median_ms, mean_ms, min_ms, p99_ms, iters
    """
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    times.sort()
    return {
        "median_ms": times[len(times) // 2],
        "mean_ms": sum(times) / len(times),
        "min_ms": times[0],
        "p99_ms": times[int(len(times) * 0.99)],
        "iters": iters,
    }


@pytest.fixture
def timed():
    """Pytest fixture providing the timed() benchmarking function."""
    return _timed


# ---------------------------------------------------------------------------
# Memory leak detection
# ---------------------------------------------------------------------------

def _count_fds() -> int:
    """Count open file descriptors for this process."""
    return len(os.listdir(f"/proc/{os.getpid()}/fd"))


def _count_maps() -> int:
    """Count memory mappings for this process."""
    with open(f"/proc/{os.getpid()}/maps") as f:
        return sum(1 for _ in f)


@pytest.fixture
def count_fds():
    """Pytest fixture providing count_fds() function."""
    return _count_fds


@pytest.fixture
def count_maps():
    """Pytest fixture providing count_maps() function."""
    return _count_maps


@pytest.fixture
def leak_check():
    """
    Context-manager fixture for fd/maps leak detection.

    Usage:
        def test_no_leak(leak_check):
            with leak_check(max_fd_leak=10, max_maps_leak=10):
                # ... allocate and free tensors ...
    """
    import contextlib

    @contextlib.contextmanager
    def _leak_check(max_fd_leak=50, max_maps_leak=50):
        # Stabilize baseline
        gc.collect()
        time.sleep(0.1)
        fd_before = _count_fds()
        maps_before = _count_maps()

        yield

        gc.collect()
        time.sleep(0.1)
        fd_after = _count_fds()
        maps_after = _count_maps()

        fd_leak = fd_after - fd_before
        maps_leak = maps_after - maps_before

        assert fd_leak < max_fd_leak, \
            f"File descriptor leak: {fd_before} → {fd_after} (+{fd_leak})"
        assert maps_leak < max_maps_leak, \
            f"Memory mapping leak: {maps_before} → {maps_after} (+{maps_leak})"

    return _leak_check


# ---------------------------------------------------------------------------
# Dmesg checking (opt-in via @pytest.mark.dmesg)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _dmesg_check(request):
    """
    Automatic dmesg checking for tests marked with @pytest.mark.dmesg.

    Usage:
        @pytest.mark.dmesg
        def test_gpu_operation():
            ...
    """
    marker = request.node.get_closest_marker("dmesg")
    if marker is None:
        yield
        return

    fail_on_warning = marker.kwargs.get("fail_on_warning", False)
    checker = DmesgChecker(fail_on_warning=fail_on_warning)
    checker.clear()
    yield
    report = checker.check()
    if report.has_errors:
        pytest.fail(f"GPU kernel errors detected!\n{report.summary()}")
    if fail_on_warning and report.has_warnings:
        pytest.fail(f"GPU kernel warnings detected!\n{report.summary()}")


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "dmesg: check kernel logs (dmesg) for GPU errors after test")
    config.addinivalue_line("markers", "slow: mark test as slow-running")


# ---------------------------------------------------------------------------
# Backend comparison helper (for use in test code, not as fixture)
# ---------------------------------------------------------------------------

def compare_with_numpy(tinygrad_fn, numpy_fn, atol=1e-4, rtol=1e-4):
    """
    Run a tinygrad computation and compare against numpy reference.

    Args:
        tinygrad_fn: Callable returning a tinygrad Tensor (will be .numpy()'d)
        numpy_fn: Callable returning a numpy array (reference)
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        tuple of (tinygrad_result, numpy_result) as numpy arrays
    """
    import numpy as np

    tg_result = tinygrad_fn()
    if hasattr(tg_result, 'numpy'):
        tg_result = tg_result.numpy()

    np_result = numpy_fn()

    np.testing.assert_allclose(tg_result, np_result, atol=atol, rtol=rtol,
        err_msg="tinygrad output does not match numpy reference")

    return tg_result, np_result
