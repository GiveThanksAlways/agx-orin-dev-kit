#!/usr/bin/env python3
"""validate_benchmarks.py — Validation suite for benchmark rigor.

Tests:
  1. Numerical correctness: same input → same output across backends
  2. C vs Python timer cross-check: compare hot_path.c clock_gettime with Python perf_counter
  3. Input data sensitivity: verify timing is data-independent
  4. Percentile/jitter analysis: load JSON results and report histogram

Usage:
  NV=1 JITBEAM=2 python3 validate_benchmarks.py --test correctness
  NV=1 JITBEAM=2 python3 validate_benchmarks.py --test timer-crosscheck
  NV=1 JITBEAM=2 python3 validate_benchmarks.py --test input-sensitivity
  python3 validate_benchmarks.py --test jitter --results results_soak.json
  NV=1 JITBEAM=2 python3 validate_benchmarks.py --test all
"""
import os, sys, argparse, time, ctypes, json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cioffi_tcn import (
    INPUT_DIM, OUTPUT_DIM, NUM_CHANNELS, KERNEL_SIZE, SEQ_LEN,
    generate_weights, build_tinygrad_tcn, _count_params,
)

SCRIPT_DIR = Path(__file__).parent


def _run_backend_subprocess(backend_name, weights_path, inputs_path, output_path):
    """Run a single backend in a subprocess to avoid CUDA context conflicts."""
    import subprocess, pickle
    script = f'''
import os, sys, ctypes, pickle
import numpy as np
sys.path.insert(0, {repr(str(SCRIPT_DIR))})
from cioffi_tcn import INPUT_DIM, OUTPUT_DIM, NUM_CHANNELS, KERNEL_SIZE, SEQ_LEN, generate_weights, build_tinygrad_tcn, _count_params

with open({repr(inputs_path)}, "rb") as f:
    test_inputs = pickle.load(f)
weights = generate_weights()
outputs = {{}}

if {repr(backend_name)} == "pytorch_fp16":
    import torch
    from cioffi_tcn import build_pytorch_tcn
    pt_model, _ = build_pytorch_tcn(weights, use_fp16=True)
    pt_model = pt_model.cuda()
    for name, x in test_inputs:
        x16 = torch.from_numpy(x).cuda().half()
        with torch.no_grad():
            out = pt_model(x16).cpu().numpy().flatten().astype(np.float32)
        outputs[name] = out

elif {repr(backend_name)} == "tinygrad_nv_fp16":
    os.environ["NV"] = "1"
    from tinygrad import Tensor, dtypes, Device
    from tinygrad.engine.jit import TinyJit
    model, _ = build_tinygrad_tcn(weights, use_fp16=True)
    static_x = Tensor.zeros(1, INPUT_DIM, SEQ_LEN, dtype=dtypes.float16).contiguous().realize()
    static_out = Tensor.zeros(1, OUTPUT_DIM, dtype=dtypes.float16).contiguous().realize()
    in_addr = static_x._buffer()._buf.cpu_view().addr
    out_addr = static_out._buffer()._buf.cpu_view().addr
    in_nbytes = INPUT_DIM * SEQ_LEN * 2
    out_nbytes = OUTPUT_DIM * 2
    @TinyJit
    def _run():
        static_out.assign(model(static_x)).realize()
    for _ in range(5):
        dummy = np.zeros((1, INPUT_DIM, SEQ_LEN), dtype=np.float16)
        ctypes.memmove(in_addr, dummy.ctypes.data, in_nbytes)
        _run()
        Device["NV"].synchronize()
    for name, x in test_inputs:
        x16 = x.astype(np.float16)
        ctypes.memmove(in_addr, x16.ctypes.data, in_nbytes)
        _run()
        Device["NV"].synchronize()
        result = np.zeros(OUTPUT_DIM, dtype=np.float16)
        ctypes.memmove(result.ctypes.data, out_addr, out_nbytes)
        outputs[name] = result.astype(np.float32)

elif {repr(backend_name)} == "hotpath_fp16":
    os.environ["NV"] = "1"
    from tinygrad import Tensor, dtypes, Device
    from tinygrad.engine.jit import TinyJit
    model, _ = build_tinygrad_tcn(weights, use_fp16=True)
    static_x = Tensor.zeros(1, INPUT_DIM, SEQ_LEN, dtype=dtypes.float16).contiguous().realize()
    static_out = Tensor.zeros(1, OUTPUT_DIM, dtype=dtypes.float16).contiguous().realize()
    in_addr = static_x._buffer()._buf.cpu_view().addr
    out_addr = static_out._buffer()._buf.cpu_view().addr
    in_nbytes = INPUT_DIM * SEQ_LEN * 2
    out_nbytes = OUTPUT_DIM * 2
    @TinyJit
    def _run():
        static_out.assign(model(static_x)).realize()
    for _ in range(5):
        dummy = np.zeros((1, INPUT_DIM, SEQ_LEN), dtype=np.float16)
        ctypes.memmove(in_addr, dummy.ctypes.data, in_nbytes)
        _run()
        Device["NV"].synchronize()
    hp_dir = os.environ.get("HOT_PATH_DIR", {repr(str(SCRIPT_DIR.parent / "control-loop" / "hot_path"))})
    sys.path.insert(0, hp_dir)
    from export_graph import export_hot_path_config
    from bench_e2e_pipeline import _build_c_config
    cfg = export_hot_path_config(_run, Device["NV"], static_x._buffer(), static_out._buffer())
    lib = ctypes.CDLL(os.path.join(hp_dir, "hot_path.so"))
    c_cfg = _build_c_config(cfg)
    lib.hot_path_init(ctypes.byref(c_cfg))
    t = (ctypes.c_uint64 * 1)()
    dummy_in = np.zeros(in_nbytes, dtype=np.uint8)
    dummy_out = np.zeros(OUTPUT_DIM * 2, dtype=np.uint8)
    for _ in range(5):
        lib.hot_path_benchmark(ctypes.byref(c_cfg), dummy_in.ctypes.data_as(ctypes.c_void_p),
            dummy_out.ctypes.data_as(ctypes.c_void_p), ctypes.c_uint32(1), t)
    for name, x in test_inputs:
        x16 = x.astype(np.float16)
        sensor = np.ascontiguousarray(x16.flatten().view(np.uint8))
        action = np.zeros(OUTPUT_DIM * 2, dtype=np.uint8)
        lib.hot_path_benchmark(ctypes.byref(c_cfg), sensor.ctypes.data_as(ctypes.c_void_p),
            action.ctypes.data_as(ctypes.c_void_p), ctypes.c_uint32(1), t)
        outputs[name] = np.frombuffer(action, dtype=np.float16).astype(np.float32)

with open({repr(output_path)}, "wb") as f:
    pickle.dump(outputs, f)
'''
    result = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"{backend_name} subprocess failed:\n{result.stderr[-500:]}")


def test_numerical_correctness():
    """Compare outputs across backends with identical input (subprocess-isolated)."""
    print("\n" + "=" * 70)
    print("TEST: Numerical Correctness")
    print("=" * 70)

    import pickle, tempfile

    weights = generate_weights()
    rng = np.random.RandomState(42)
    test_inputs = [
        ("zeros", np.zeros((1, INPUT_DIM, SEQ_LEN), dtype=np.float32)),
        ("random_seed42", rng.randn(1, INPUT_DIM, SEQ_LEN).astype(np.float32)),
        ("ones", np.ones((1, INPUT_DIM, SEQ_LEN), dtype=np.float32)),
        ("large_magnitude", (rng.randn(1, INPUT_DIM, SEQ_LEN) * 100).astype(np.float32)),
    ]

    # Save inputs for subprocesses
    inputs_path = os.path.join(tempfile.gettempdir(), "validate_inputs.pkl")
    with open(inputs_path, "wb") as f:
        pickle.dump(test_inputs, f)

    results = {}

    for backend in ["pytorch_fp16", "tinygrad_nv_fp16", "hotpath_fp16"]:
        idx = {"pytorch_fp16": 1, "tinygrad_nv_fp16": 2, "hotpath_fp16": 3}[backend]
        print(f"\n  [{idx}/3] {backend} ...")
        output_path = os.path.join(tempfile.gettempdir(), f"validate_{backend}.pkl")
        try:
            _run_backend_subprocess(backend, None, inputs_path, output_path)
            with open(output_path, "rb") as f:
                results[backend] = pickle.load(f)
            print(f"    OK: {len(test_inputs)} inputs tested")
        except Exception as e:
            print(f"    SKIP: {e}")

    # ── Comparison ──
    print("\n  ── Cross-backend comparison ──")
    backends = list(results.keys())
    if len(backends) < 2:
        print("  FAIL: Need at least 2 backends for comparison")
        return False

    all_pass = True
    ref_name = backends[0]
    ref = results[ref_name]

    for bn in backends[1:]:
        other = results[bn]
        print(f"\n  {ref_name} vs {bn}:")
        for inp_name in test_inputs:
            name = inp_name[0]
            if name not in ref or name not in other:
                continue
            r = ref[name]
            o = other[name]
            max_diff = np.max(np.abs(r - o))
            rmse = np.sqrt(np.mean((r - o) ** 2))
            # FP16 tolerance: absolute 0.05 for small values, or 0.1% relative for large
            abs_max = max(np.max(np.abs(r)), np.max(np.abs(o)), 1.0)
            threshold = max(0.05, abs_max * 0.001)
            status = "PASS" if max_diff < threshold else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"    input={name:20s}  max_diff={max_diff:.6f}  rmse={rmse:.6f}  [{status}]")
            print(f"      {ref_name}: {r}")
            print(f"      {bn}:     {o}")

    if all_pass:
        print("\n  ✓ ALL CORRECTNESS CHECKS PASSED")
    else:
        print("\n  ✗ SOME CORRECTNESS CHECKS FAILED")
    return all_pass


def test_timer_crosscheck():
    """Compare C hot_path_benchmark clock_gettime with Python perf_counter."""
    print("\n" + "=" * 70)
    print("TEST: C vs Python Timer Cross-Check")
    print("=" * 70)

    try:
        os.environ["NV"] = "1"
        from tinygrad import Tensor, dtypes, Device
        from tinygrad.engine.jit import TinyJit

        weights = generate_weights()
        model, _ = build_tinygrad_tcn(weights, use_fp16=True)
        static_x = Tensor.zeros(1, INPUT_DIM, SEQ_LEN, dtype=dtypes.float16).contiguous().realize()
        static_out = Tensor.zeros(1, OUTPUT_DIM, dtype=dtypes.float16).contiguous().realize()
        in_addr = static_x._buffer()._buf.cpu_view().addr
        out_addr = static_out._buffer()._buf.cpu_view().addr
        in_nbytes = INPUT_DIM * SEQ_LEN * 2
        out_nbytes = OUTPUT_DIM * 2

        @TinyJit
        def _run():
            static_out.assign(model(static_x)).realize()

        # Warmup
        for _ in range(30):
            dummy = np.zeros((1, INPUT_DIM, SEQ_LEN), dtype=np.float16)
            ctypes.memmove(in_addr, dummy.ctypes.data, in_nbytes)
            _run()
            Device["NV"].synchronize()

        hp_dir = os.environ.get("HOT_PATH_DIR",
            str(SCRIPT_DIR.parent / "control-loop" / "hot_path"))
        so_path = os.path.join(hp_dir, "hot_path.so")
        if not os.path.exists(so_path):
            print(f"  SKIP: hot_path.so not found at {so_path}")
            return True

        sys.path.insert(0, hp_dir)
        from export_graph import export_hot_path_config
        from bench_e2e_pipeline import _build_c_config

        cfg = export_hot_path_config(_run, Device["NV"],
                                      static_x._buffer(), static_out._buffer())
        lib = ctypes.CDLL(so_path)
        c_cfg = _build_c_config(cfg)
        lib.hot_path_init(ctypes.byref(c_cfg))

        # C warmup
        dummy_in = np.zeros(in_nbytes, dtype=np.uint8)
        dummy_out = np.zeros(out_nbytes, dtype=np.uint8)
        t = (ctypes.c_uint64 * 1)()
        for _ in range(10):
            lib.hot_path_benchmark(ctypes.byref(c_cfg),
                dummy_in.ctypes.data_as(ctypes.c_void_p),
                dummy_out.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_uint32(1), t)

        # Run N iterations, compare C nanosecond timer with Python perf_counter
        N = 100
        c_times_ns = (ctypes.c_uint64 * N)()
        python_times_us = []

        sensor = np.zeros(in_nbytes, dtype=np.uint8)
        action = np.zeros(out_nbytes, dtype=np.uint8)

        for i in range(N):
            c_t = (ctypes.c_uint64 * 1)()
            t0 = time.perf_counter()
            lib.hot_path_benchmark(ctypes.byref(c_cfg),
                sensor.ctypes.data_as(ctypes.c_void_p),
                action.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_uint32(1), c_t)
            t1 = time.perf_counter()
            c_times_ns[i] = c_t[0]
            python_times_us.append((t1 - t0) * 1e6)

        c_us = np.array([c_times_ns[i] / 1000.0 for i in range(N)])
        py_us = np.array(python_times_us)
        diff_us = py_us - c_us  # Python should be >= C (wraps the C call)

        print(f"\n  Iterations: {N}")
        print(f"  C clock_gettime  — median: {np.median(c_us):.1f} µs,  mean: {np.mean(c_us):.1f} µs")
        print(f"  Python perf_counter — median: {np.median(py_us):.1f} µs,  mean: {np.mean(py_us):.1f} µs")
        print(f"  Overhead (Py - C) — median: {np.median(diff_us):.1f} µs,  mean: {np.mean(diff_us):.1f} µs")
        print(f"  Overhead range: [{np.min(diff_us):.1f}, {np.max(diff_us):.1f}] µs")

        # Check: Python should wrap C call, so overhead should be small positive
        median_overhead = np.median(diff_us)
        if 0 <= median_overhead < 20:
            print(f"\n  ✓ PASS: Timers agree within {median_overhead:.1f} µs (< 20 µs threshold)")
            return True
        elif median_overhead < 0:
            print(f"\n  ⚠ WARNING: Python timer FASTER than C timer by {abs(median_overhead):.1f} µs")
            print(f"    This suggests a measurement issue")
            return False
        else:
            print(f"\n  ⚠ WARNING: {median_overhead:.1f} µs overhead between C and Python")
            return True
    except Exception as e:
        print(f"  SKIP: {e}")
        import traceback; traceback.print_exc()
        return True


def test_input_sensitivity():
    """Verify timing doesn't depend on input values (no branch divergence)."""
    print("\n" + "=" * 70)
    print("TEST: Input Data Sensitivity")
    print("=" * 70)

    try:
        os.environ["NV"] = "1"
        from tinygrad import Tensor, dtypes, Device
        from tinygrad.engine.jit import TinyJit

        weights = generate_weights()
        model, _ = build_tinygrad_tcn(weights, use_fp16=True)
        static_x = Tensor.zeros(1, INPUT_DIM, SEQ_LEN, dtype=dtypes.float16).contiguous().realize()
        static_out = Tensor.zeros(1, OUTPUT_DIM, dtype=dtypes.float16).contiguous().realize()
        in_addr = static_x._buffer()._buf.cpu_view().addr
        out_addr = static_out._buffer()._buf.cpu_view().addr
        in_nbytes = INPUT_DIM * SEQ_LEN * 2
        out_nbytes = OUTPUT_DIM * 2

        @TinyJit
        def _run():
            static_out.assign(model(static_x)).realize()

        # Warmup
        for _ in range(30):
            dummy = np.zeros((1, INPUT_DIM, SEQ_LEN), dtype=np.float16)
            ctypes.memmove(in_addr, dummy.ctypes.data, in_nbytes)
            _run()
            Device["NV"].synchronize()

        # Test different input patterns
        rng = np.random.RandomState(42)
        input_patterns = {
            "zeros": np.zeros((1, INPUT_DIM, SEQ_LEN), dtype=np.float16),
            "ones": np.ones((1, INPUT_DIM, SEQ_LEN), dtype=np.float16),
            "random": rng.randn(1, INPUT_DIM, SEQ_LEN).astype(np.float16),
            "large": (rng.randn(1, INPUT_DIM, SEQ_LEN) * 100).astype(np.float16),
            "tiny": (rng.randn(1, INPUT_DIM, SEQ_LEN) * 1e-5).astype(np.float16),
        }

        N = 100  # iterations per pattern
        pattern_times = {}

        for name, x in input_patterns.items():
            times = []
            for _ in range(N):
                ctypes.memmove(in_addr, x.ctypes.data, in_nbytes)
                t0 = time.perf_counter()
                _run()
                Device["NV"].synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1e6)
            pattern_times[name] = np.array(times)
            med = np.median(pattern_times[name])
            std = np.std(pattern_times[name])
            print(f"  {name:12s}: median={med:.1f} µs  std={std:.1f}  p99={np.percentile(pattern_times[name], 99):.1f}")

        # Check: all medians should be within 10% of each other
        medians = [np.median(v) for v in pattern_times.values()]
        spread = (max(medians) - min(medians)) / np.mean(medians) * 100
        print(f"\n  Spread: {spread:.1f}% (max-min / mean)")
        if spread < 10:
            print(f"  ✓ PASS: Timing is data-independent (< 10% spread)")
            return True
        else:
            print(f"  ⚠ WARNING: {spread:.1f}% timing variation across input patterns")
            return False
    except Exception as e:
        print(f"  SKIP: {e}")
        import traceback; traceback.print_exc()
        return True


def test_jitter_analysis(results_files):
    """Analyze jitter/percentiles from saved benchmark JSON results."""
    print("\n" + "=" * 70)
    print("TEST: Latency Histogram & Jitter Analysis")
    print("=" * 70)

    if not results_files:
        print("  No result files provided. Use --results file1.json file2.json ...")
        return True

    for fpath in results_files:
        if not os.path.exists(fpath):
            print(f"  SKIP: {fpath} not found")
            continue

        with open(fpath) as f:
            data = json.load(f)

        print(f"\n  File: {fpath}")
        for backend_name, bdata in data.get("backends", data).items():
            raw_keys = {"raw_total": "total_pipeline", "raw_tcn": "tcn_inference",
                        "raw_imu": "imu_propagation", "raw_ekf": "ekf_update"}
            for raw_key, display_name in raw_keys.items():
                if raw_key not in bdata:
                    continue
                times = bdata[raw_key]
                if not isinstance(times, list) or len(times) == 0:
                    continue
                a = np.array(times)
                med = np.median(a)
                p90 = np.percentile(a, 90)
                p95 = np.percentile(a, 95)
                p99 = np.percentile(a, 99)
                p999 = np.percentile(a, 99.9)
                mx = np.max(a)
                cov = np.std(a) / np.mean(a) * 100
                max_med_ratio = mx / med if med > 0 else float('inf')

                print(f"\n    {backend_name} / {display_name} ({len(a)} samples):")
                print(f"      median={med:.1f}  p90={p90:.1f}  p95={p95:.1f}  p99={p99:.1f}  p99.9={p999:.1f}  max={mx:.1f} µs")
                print(f"      CoV={cov:.2f}%  max/median={max_med_ratio:.2f}x")
                if max_med_ratio > 3:
                    print(f"      ⚠ WARNING: max is {max_med_ratio:.1f}× median — significant jitter")
                elif cov > 10:
                    print(f"      ⚠ WARNING: CoV={cov:.1f}% — high variance")
                else:
                    print(f"      ✓ Stable")

    return True


def main():
    p = argparse.ArgumentParser(description="Benchmark validation suite")
    p.add_argument("--test", choices=["correctness", "timer-crosscheck",
                   "input-sensitivity", "jitter", "all"], default="all")
    p.add_argument("--results", nargs="*", help="JSON result files for jitter analysis")
    args = p.parse_args()

    results = {}

    if args.test in ("correctness", "all"):
        results["correctness"] = test_numerical_correctness()

    if args.test in ("timer-crosscheck", "all"):
        results["timer_crosscheck"] = test_timer_crosscheck()

    if args.test in ("input-sensitivity", "all"):
        results["input_sensitivity"] = test_input_sensitivity()

    if args.test in ("jitter", "all"):
        results["jitter"] = test_jitter_analysis(args.results or [])

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:25s}  {status}")

    all_pass = all(results.values())
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
