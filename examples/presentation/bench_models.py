#!/usr/bin/env python3
"""bench_models.py — Unified benchmark: tinygrad NV=1 vs TensorRT on Jetson AGX Orin.

Tests MLP, 1D-CNN, and Hybrid CNN+MLP architectures across both frameworks.
Focuses on the real-world question: where does NV=1's dispatch advantage outweigh
TensorRT's kernel optimization, and vice versa?

Usage:
  cd examples/presentation && nix develop
  NV=1 JITBEAM=2 python3 bench_models.py                  # All architectures
  NV=1 JITBEAM=2 python3 bench_models.py --arch mlp        # MLP only
  NV=1 JITBEAM=2 python3 bench_models.py --arch cnn        # CNN only
  NV=1 JITBEAM=2 python3 bench_models.py --arch hybrid     # Hybrid only
  NV=1 JITBEAM=2 python3 bench_models.py --skip-tensorrt   # tinygrad only
  NV=1 JITBEAM=2 python3 bench_models.py --iters 5000      # Fewer iterations (faster)
"""
import os, sys, argparse, time, json
import numpy as np
from pathlib import Path

# Ensure this directory is on PYTHONPATH for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    MLP_CONFIGS, CNN_CONFIGS, HYBRID_CONFIGS,
    IN_DIM, OUT_DIM, SEQ_LEN,
    export_mlp_onnx, export_cnn_onnx, export_hybrid_onnx,
)

WEIGHTS_DIR = Path(__file__).parent / "weights"


# ═══════════════════════════════════════════════════════════════════════════════
# Stats & formatting
# ═══════════════════════════════════════════════════════════════════════════════

def stats(times):
    a = np.asarray(times)
    return {
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "p99": float(np.percentile(a, 99)),
        "p999": float(np.percentile(a, 99.9)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "count": len(a),
    }


def print_stats(label, times):
    s = stats(times)
    freq = 1e6 / s["median"] if s["median"] > 0 else 0
    print(f"  {label:45s} median={s['median']:8.1f} µs  "
          f"std={s['std']:6.1f}  p99={s['p99']:8.1f}  "
          f"max={s['max']:9.1f}  freq={freq:7.0f} Hz")


def print_comparison(nv_times, trt_times, label):
    """Print head-to-head comparison between NV=1 and TensorRT."""
    nv = stats(nv_times)
    trt = stats(trt_times)
    ratio = trt["median"] / nv["median"] if nv["median"] > 0 else 0

    if ratio > 1:
        winner = f"NV=1 wins {ratio:.2f}x"
    elif ratio > 0:
        winner = f"TensorRT wins {1/ratio:.2f}x"
    else:
        winner = "N/A"

    print(f"\n  {'Metric':20s} {'NV=1':>12s} {'TensorRT':>12s} {'Ratio':>12s}")
    print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*12}")
    print(f"  {'Median (µs)':20s} {nv['median']:12.1f} {trt['median']:12.1f} {winner:>12s}")
    print(f"  {'Frequency (Hz)':20s} {1e6/nv['median']:12.0f} {1e6/trt['median']:12.0f}")
    print(f"  {'Std (µs)':20s} {nv['std']:12.1f} {trt['std']:12.1f}")
    print(f"  {'P99 (µs)':20s} {nv['p99']:12.1f} {trt['p99']:12.1f}")
    print(f"  {'P99.9 (µs)':20s} {nv['p999']:12.1f} {trt['p999']:12.1f}")
    print(f"  {'Max (µs)':20s} {nv['max']:12.1f} {trt['max']:12.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main benchmark runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_mlp_benchmarks(args):
    """Run MLP benchmarks across all sizes."""
    from bench_tinygrad import bench_nv_mlp
    results = []

    print("\n" + "=" * 90)
    print("MLP ARCHITECTURES — tinygrad NV=1 vs TensorRT")
    print("  Input: (1, 12) — pos/vel/rpy/gyro state vector")
    print("  Output: (1, 4) — thrust + angular rates")
    print("  Real-world: learned controllers, sensor fusion, reactive policies")
    print("=" * 90)

    for name, hidden_dims, desc, use_case in MLP_CONFIGS:
        print(f"\n{'─' * 90}")
        print(f"  {name}: {desc}")
        print(f"  Use case: {use_case}")
        print(f"{'─' * 90}")

        # Export weights for TensorRT
        weight_path = WEIGHTS_DIR / f"{name}.npz"
        export_mlp_onnx(hidden_dims, str(weight_path))

        # tinygrad NV=1
        print(f"\n  [tinygrad NV=1]")
        nv_times, params, nv_result = bench_nv_mlp(
            name, hidden_dims, warmup=args.warmup, n_iters=args.iters)
        print_stats(f"NV=1 direct memmove ({params:,} params)", nv_times)

        result_entry = {
            "name": name, "arch": "mlp", "params": params,
            "hidden_dims": hidden_dims, "desc": desc, "use_case": use_case,
            "nv": stats(nv_times),
        }

        # TensorRT
        if not args.skip_tensorrt:
            print(f"\n  [TensorRT FP16]")
            try:
                from bench_tensorrt import bench_trt_mlp
                trt_times, _ = bench_trt_mlp(
                    name, hidden_dims, str(weight_path),
                    warmup=args.warmup, n_iters=args.iters)
                print_stats(f"TensorRT FP16 ({params:,} params)", trt_times)
                print_comparison(nv_times, trt_times, name)
                result_entry["trt"] = stats(trt_times)
            except Exception as e:
                print(f"  TensorRT FAILED: {e}")
                result_entry["trt_error"] = str(e)

        results.append(result_entry)

    return results


def run_cnn_benchmarks(args):
    """Run 1D-CNN benchmarks."""
    from bench_tinygrad import bench_nv_cnn
    results = []

    print("\n" + "=" * 90)
    print("1D-CNN ARCHITECTURES — tinygrad NV=1 vs TensorRT")
    print(f"  Input: (1, {IN_DIM}, {SEQ_LEN}) — {SEQ_LEN}-sample IMU time window")
    print("  Output: (1, 4) — thrust + angular rates")
    print("  Real-world: temporal feature extraction from sensor history")
    print("=" * 90)

    for name, conv_layers, mlp_head, desc, use_case in CNN_CONFIGS:
        print(f"\n{'─' * 90}")
        print(f"  {name}: {desc}")
        print(f"  Use case: {use_case}")
        print(f"{'─' * 90}")

        weight_path = WEIGHTS_DIR / f"{name}.npz"
        export_cnn_onnx(conv_layers, mlp_head, str(weight_path))

        print(f"\n  [tinygrad NV=1]")
        nv_times, params, nv_result = bench_nv_cnn(
            name, conv_layers, mlp_head, warmup=args.warmup, n_iters=args.iters)
        print_stats(f"NV=1 direct memmove ({params:,} params)", nv_times)

        result_entry = {
            "name": name, "arch": "cnn", "params": params,
            "conv_layers": conv_layers, "mlp_head": mlp_head,
            "desc": desc, "use_case": use_case,
            "nv": stats(nv_times),
        }

        if not args.skip_tensorrt:
            print(f"\n  [TensorRT FP16]")
            try:
                from bench_tensorrt import bench_trt_cnn
                trt_times, _ = bench_trt_cnn(
                    name, conv_layers, mlp_head, str(weight_path),
                    warmup=args.warmup, n_iters=args.iters)
                print_stats(f"TensorRT FP16 ({params:,} params)", trt_times)
                print_comparison(nv_times, trt_times, name)
                result_entry["trt"] = stats(trt_times)
            except Exception as e:
                print(f"  TensorRT FAILED: {e}")
                result_entry["trt_error"] = str(e)

        results.append(result_entry)

    return results


def run_hybrid_benchmarks(args):
    """Run Hybrid CNN+MLP benchmarks."""
    from bench_tinygrad import bench_nv_hybrid
    results = []

    print("\n" + "=" * 90)
    print("HYBRID CNN+MLP ARCHITECTURES — tinygrad NV=1 vs TensorRT")
    print(f"  Inputs: IMU window (1, {IN_DIM}, {SEQ_LEN}) + Current state (1, {IN_DIM})")
    print("  Output: (1, 4) — thrust + angular rates")
    print("  Real-world: history-aware perception fused with reactive control")
    print("=" * 90)

    for name, conv_layers, mlp_head, desc, use_case in HYBRID_CONFIGS:
        print(f"\n{'─' * 90}")
        print(f"  {name}: {desc}")
        print(f"  Use case: {use_case}")
        print(f"{'─' * 90}")

        weight_path = WEIGHTS_DIR / f"{name}.npz"
        export_hybrid_onnx(conv_layers, mlp_head, str(weight_path))

        print(f"\n  [tinygrad NV=1]")
        nv_times, params, nv_result = bench_nv_hybrid(
            name, conv_layers, mlp_head, warmup=args.warmup, n_iters=args.iters)
        print_stats(f"NV=1 direct memmove ({params:,} params)", nv_times)

        result_entry = {
            "name": name, "arch": "hybrid", "params": params,
            "conv_layers": conv_layers, "mlp_head": mlp_head,
            "desc": desc, "use_case": use_case,
            "nv": stats(nv_times),
        }

        if not args.skip_tensorrt:
            print(f"\n  [TensorRT FP16]")
            try:
                from bench_tensorrt import bench_trt_hybrid
                trt_times, _ = bench_trt_hybrid(
                    name, conv_layers, mlp_head, str(weight_path),
                    warmup=args.warmup, n_iters=args.iters)
                print_stats(f"TensorRT FP16 ({params:,} params)", trt_times)
                print_comparison(nv_times, trt_times, name)
                result_entry["trt"] = stats(trt_times)
            except Exception as e:
                print(f"  TensorRT FAILED: {e}")
                result_entry["trt_error"] = str(e)

        results.append(result_entry)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Summary & analysis
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(all_results):
    """Print a consolidated summary table and analysis."""
    print("\n")
    print("=" * 100)
    print("SUMMARY — tinygrad NV=1 vs TensorRT on Jetson AGX Orin 64GB")
    print("=" * 100)

    # Summary table
    header = (f"{'Model':18s} {'Arch':7s} {'Params':>10s} "
              f"{'NV=1 (µs)':>10s} {'TRT (µs)':>10s} {'Ratio':>8s} "
              f"{'NV Hz':>8s} {'TRT Hz':>8s} {'Winner':>10s}")
    print(f"\n{header}")
    print("─" * len(header))

    wins = {"nv": 0, "trt": 0, "tie": 0}
    crossover_params = None

    for r in all_results:
        nv_med = r["nv"]["median"]
        nv_hz = 1e6 / nv_med if nv_med > 0 else 0

        if "trt" in r:
            trt_med = r["trt"]["median"]
            trt_hz = 1e6 / trt_med if trt_med > 0 else 0
            ratio = trt_med / nv_med if nv_med > 0 else 0

            if ratio > 1.05:
                winner = "NV=1"
                wins["nv"] += 1
            elif ratio < 0.95:
                winner = "TensorRT"
                wins["trt"] += 1
                if crossover_params is None:
                    crossover_params = r["params"]
            else:
                winner = "~tied"
                wins["tie"] += 1

            print(f"{r['name']:18s} {r['arch']:7s} {r['params']:10,d} "
                  f"{nv_med:10.1f} {trt_med:10.1f} {ratio:8.2f}x "
                  f"{nv_hz:8.0f} {trt_hz:8.0f} {winner:>10s}")
        else:
            err = r.get("trt_error", "skipped")
            print(f"{r['name']:18s} {r['arch']:7s} {r['params']:10,d} "
                  f"{nv_med:10.1f} {'---':>10s} {'---':>8s} "
                  f"{nv_hz:8.0f} {'---':>8s} {'---':>10s}")

    # Analysis
    print(f"\n{'─' * 80}")
    print(f"Score: NV=1 wins {wins['nv']}, TensorRT wins {wins['trt']}, tied {wins['tie']}")

    if crossover_params:
        print(f"\nCrossover estimate: TensorRT starts winning around ~{crossover_params:,} params")
        print("  Below this: NV=1's dispatch advantage dominates (overhead-bound)")
        print("  Above this: TensorRT's kernel optimization dominates (compute-bound)")

    # Real-world context
    print(f"\n{'─' * 80}")
    print("REAL-WORLD CONTEXT")
    print("─" * 80)
    print("""
  Frequency targets for robotics/drones:
    500 Hz  — basic stabilization (most hobby drones)
    1 kHz   — industrial robot arms, precision agriculture
    2 kHz   — racing drones, fast servo control
    4 kHz   — high-performance servos, vibration control

  Sensor bottlenecks (what limits you before GPU speed):
    SPI IMU (ICM-42688):     ~10 µs per read @ 10 MHz → 100 kHz max
    I2C barometer (BMP390):  ~500 µs per read @ 400 kHz → 2 kHz max
    UART to ESC:             ~50 µs @ 2 Mbaud per motor → ~5 kHz max
    CAN bus (automotive):    ~100 µs per frame → 10 kHz max
    Ethernet camera:         ~5-33 ms per frame → 30-200 Hz

  The question is: does your control loop NEED 4+ kHz?
    - If sensors cap you at 1 kHz → both NV=1 and TensorRT are fast enough
    - If you're doing inner-loop rate control + learned policy → yes, 4 kHz matters
    - If you have a fast sensor (IMU @ 4 kHz) + learned filter/policy → NV=1 wins

  What NV=1 uniquely enables:
    1. Zero CUDA runtime → auditable, reproducible, no background thread stalls
    2. Tegra unified memory bypass → <1 µs H2D/D2H for small tensors
    3. Deterministic latency → zero 100ms stalls (critical for safety)
    4. Minimal dependencies → works on bare Tegra kernel drivers, no JetPack SDK""")

    return wins


def save_results(all_results, path):
    """Save raw results as JSON for further analysis."""
    with open(path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": "Jetson AGX Orin 64GB",
            "jitbeam": os.environ.get("JITBEAM", "default"),
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\nRaw results saved to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="tinygrad NV=1 vs TensorRT benchmark")
    parser.add_argument("--arch", choices=["mlp", "cnn", "hybrid", "all"], default="all",
                        help="Architecture type to benchmark (default: all)")
    parser.add_argument("--iters", type=int, default=10000,
                        help="Iterations per benchmark (default: 10000)")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Warmup iterations (default: 50)")
    parser.add_argument("--skip-tensorrt", action="store_true",
                        help="Skip TensorRT benchmarks (tinygrad only)")
    parser.add_argument("--output", type=str, default="results.json",
                        help="Output JSON file for raw results")
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(exist_ok=True)

    print("=" * 90)
    print("tinygrad NV=1 vs TensorRT — Presentation Benchmarks")
    print("=" * 90)
    print(f"  Platform:    Jetson AGX Orin 64GB")
    print(f"  JITBEAM:     {os.environ.get('JITBEAM', 'default')}")
    print(f"  Iterations:  {args.iters}")
    print(f"  Warmup:      {args.warmup}")
    print(f"  Backends:    tinygrad NV=1" + ("" if args.skip_tensorrt else " + TensorRT FP16"))
    print(f"  Archs:       {args.arch}")

    all_results = []

    if args.arch in ("mlp", "all"):
        all_results.extend(run_mlp_benchmarks(args))

    if args.arch in ("cnn", "all"):
        all_results.extend(run_cnn_benchmarks(args))

    if args.arch in ("hybrid", "all"):
        all_results.extend(run_hybrid_benchmarks(args))

    # Summary
    print_summary(all_results)

    # Save
    output_path = Path(__file__).parent / args.output
    save_results(all_results, str(output_path))


if __name__ == "__main__":
    main()
