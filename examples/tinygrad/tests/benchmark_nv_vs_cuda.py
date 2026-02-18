#!/usr/bin/env python3
"""
Phase C: Performance Benchmarks — NV=1 vs CUDA=1 on Jetson Orin AGX 64GB.

Run with:
  NV=1 python3 tests/benchmark_nv_vs_cuda.py 2>&1 | tee tests/results_bench_nv.log
  CUDA=1 python3 tests/benchmark_nv_vs_cuda.py 2>&1 | tee tests/results_bench_cuda.log

Output: JSON results file + human-readable summary.
"""
import os, sys, time, json, argparse, gc
import numpy as np

os.environ.setdefault("NV", "1") if "CUDA" not in os.environ else None

from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv


def timed(fn, warmup=10, iters=90):
    """Benchmark a function. Returns dict with median/mean/p99/min times in ms."""
    # Warmup
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
        "median_ms": times[len(times)//2],
        "mean_ms": sum(times)/len(times),
        "min_ms": times[0],
        "p99_ms": times[int(len(times)*0.99)],
        "iters": iters,
    }


def bench_matmul(results):
    """C1: Matmul compute throughput."""
    print("\n=== C1: Matmul Throughput ===")
    configs = [
        (256, dtypes.float32), (512, dtypes.float32), (1024, dtypes.float32),
        (2048, dtypes.float32), (4096, dtypes.float32),
        (1024, dtypes.float16), (2048, dtypes.float16), (4096, dtypes.float16),
    ]

    matmul_results = []
    for n, dt in configs:
        np_dt = np.float16 if dt == dtypes.float16 else np.float32
        a = Tensor(np.random.randn(n, n).astype(np_dt)).realize()
        b = Tensor(np.random.randn(n, n).astype(np_dt)).realize()

        def run():
            c = (a @ b).realize()
            Device[Device.DEFAULT].synchronize()

        t = timed(run, warmup=5, iters=50 if n <= 1024 else 20)
        flops = 2 * n * n * n
        gflops = flops / (t["median_ms"] / 1000) / 1e9

        entry = {"size": n, "dtype": str(dt), "gflops": round(gflops, 1), **t}
        matmul_results.append(entry)
        print(f"  {n}x{n} {dt}: {gflops:.1f} GFLOPS (median {t['median_ms']:.2f} ms)")

        del a, b
        gc.collect()

    results["matmul"] = matmul_results


def bench_bandwidth(results):
    """C2: Memory bandwidth."""
    print("\n=== C2: Memory Bandwidth ===")
    bw_results = []

    for size_mb in [1, 16, 256]:
        n_elements = size_mb * 1024 * 1024 // 4
        src_np = np.random.randn(n_elements).astype(np.float32)
        size_bytes = n_elements * 4

        # Host → Device (copyin)
        def copyin():
            t = Tensor(src_np).contiguous().realize()
            Device[Device.DEFAULT].synchronize()
            return t

        t_in = timed(copyin, warmup=5, iters=30)
        gb_s_in = (size_bytes / 1e9) / (t_in["median_ms"] / 1000)

        # Device → Host (copyout)
        t_dev = Tensor(src_np).contiguous().realize()
        def copyout():
            _ = t_dev.numpy()

        t_out = timed(copyout, warmup=5, iters=30)
        gb_s_out = (size_bytes / 1e9) / (t_out["median_ms"] / 1000)

        # Device → Device (DMA copy)
        a = Tensor(src_np).contiguous().realize()
        def d2d():
            b = (a + 0).contiguous().realize()  # force copy via identity
            Device[Device.DEFAULT].synchronize()

        t_d2d = timed(d2d, warmup=5, iters=30)
        gb_s_d2d = (size_bytes / 1e9) / (t_d2d["median_ms"] / 1000)

        # Allocation latency
        def alloc_only():
            t = Tensor.zeros(n_elements).contiguous().realize()
            Device[Device.DEFAULT].synchronize()
            del t

        t_alloc = timed(alloc_only, warmup=5, iters=30)

        entry = {
            "size_mb": size_mb,
            "copyin_gb_s": round(gb_s_in, 2), "copyin_ms": round(t_in["median_ms"], 3),
            "copyout_gb_s": round(gb_s_out, 2), "copyout_ms": round(t_out["median_ms"], 3),
            "d2d_gb_s": round(gb_s_d2d, 2), "d2d_ms": round(t_d2d["median_ms"], 3),
            "alloc_ms": round(t_alloc["median_ms"], 3),
        }
        bw_results.append(entry)
        print(f"  {size_mb}MB: copyin={gb_s_in:.2f} GB/s, copyout={gb_s_out:.2f} GB/s, "
              f"d2d={gb_s_d2d:.2f} GB/s, alloc={t_alloc['median_ms']:.3f} ms")

        del a, t_dev
        gc.collect()

    results["bandwidth"] = bw_results


def bench_kernel_launch(results):
    """C3: Kernel launch overhead."""
    print("\n=== C3: Kernel Launch Overhead ===")

    # Trivial kernel: 1-element add
    a = Tensor([1.0]).realize()
    b = Tensor([1.0]).realize()

    def trivial_kernel():
        c = (a + b).realize()
        Device[Device.DEFAULT].synchronize()

    t = timed(trivial_kernel, warmup=50, iters=1000)

    entry = {
        "trivial_median_us": round(t["median_ms"] * 1000, 1),
        "trivial_p99_us": round(t["p99_ms"] * 1000, 1),
        "trivial_min_us": round(t["min_ms"] * 1000, 1),
    }
    results["kernel_launch"] = entry
    print(f"  Trivial kernel: median={entry['trivial_median_us']:.1f} µs, "
          f"p99={entry['trivial_p99_us']:.1f} µs")

    del a, b
    gc.collect()


def bench_elementwise(results):
    """C4: Element-wise ops (bandwidth-limited)."""
    print("\n=== C4: Element-wise Ops ===")
    ew_results = []

    ops = {
        "add": lambda a: a + a,
        "mul": lambda a: a * a,
        "exp": lambda a: a.exp(),
        "relu": lambda a: a.relu(),
    }

    for n_elements in [1_000_000, 10_000_000]:
        a = Tensor(np.random.randn(n_elements).astype(np.float32) * 0.1).realize()
        size_bytes = n_elements * 4

        for op_name, op_fn in ops.items():
            def run(fn=op_fn, x=a):
                r = fn(x).realize()
                Device[Device.DEFAULT].synchronize()

            t = timed(run, warmup=10, iters=50)
            # Effective bandwidth: read input + write output
            bw_bytes = size_bytes * 2  # read + write
            gb_s = (bw_bytes / 1e9) / (t["median_ms"] / 1000)

            entry = {"op": op_name, "elements": n_elements, "gb_s": round(gb_s, 2),
                     "median_ms": round(t["median_ms"], 3)}
            ew_results.append(entry)
            print(f"  {op_name} {n_elements//1_000_000}M: {gb_s:.2f} GB/s "
                  f"(median {t['median_ms']:.3f} ms)")

        del a
        gc.collect()

    results["elementwise"] = ew_results


def bench_model_inference(results):
    """C5: Model inference (simple MLP)."""
    print("\n=== C5: Model Inference ===")

    np.random.seed(42)

    # Simple MLP: 784→256→128→10
    w1 = Tensor(np.random.randn(784, 256).astype(np.float32) * 0.01).realize()
    b1 = Tensor(np.random.randn(256).astype(np.float32) * 0.01).realize()
    w2 = Tensor(np.random.randn(256, 128).astype(np.float32) * 0.01).realize()
    b2 = Tensor(np.random.randn(128).astype(np.float32) * 0.01).realize()
    w3 = Tensor(np.random.randn(128, 10).astype(np.float32) * 0.01).realize()
    b3 = Tensor(np.random.randn(10).astype(np.float32) * 0.01).realize()

    x = Tensor(np.random.randn(64, 784).astype(np.float32)).realize()

    def forward():
        h = (x @ w1 + b1).relu()
        h = (h @ w2 + b2).relu()
        out = (h @ w3 + b3).realize()
        Device[Device.DEFAULT].synchronize()

    t = timed(forward, warmup=10, iters=100)

    entry = {
        "model": "MLP-784-256-128-10",
        "batch_size": 64,
        "median_ms": round(t["median_ms"], 3),
        "p99_ms": round(t["p99_ms"], 3),
        "throughput_samples_per_sec": round(64 / (t["median_ms"] / 1000), 0),
    }
    results["model_inference"] = entry
    print(f"  MLP: median={t['median_ms']:.3f} ms, "
          f"throughput={entry['throughput_samples_per_sec']:.0f} samples/s")


def main():
    parser = argparse.ArgumentParser(description="NV vs CUDA benchmark suite")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file")
    parser.add_argument("--skip", nargs="*", default=[], help="Skip benchmarks (matmul, bandwidth, kernel_launch, elementwise, model)")
    args = parser.parse_args()

    backend = Device.DEFAULT
    print(f"Backend: {backend}")
    print(f"Device: Jetson Orin AGX 64GB")

    results = {
        "backend": backend,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": "Jetson Orin AGX 64GB",
    }

    if "matmul" not in args.skip:
        bench_matmul(results)
    if "bandwidth" not in args.skip:
        bench_bandwidth(results)
    if "kernel_launch" not in args.skip:
        bench_kernel_launch(results)
    if "elementwise" not in args.skip:
        bench_elementwise(results)
    if "model" not in args.skip:
        bench_model_inference(results)

    # Output
    output_file = args.output or f"tests/results_bench_{backend.lower()}.json"
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print("\n=== Summary ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
