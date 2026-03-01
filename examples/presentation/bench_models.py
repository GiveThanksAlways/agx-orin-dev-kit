#!/usr/bin/env python3
"""bench_models.py — Unified benchmark: tinygrad NV=1 vs C Hot Path vs TensorRT.

Tests MLP, 1D-CNN, and Hybrid CNN+MLP architectures across three backends:
  1. tinygrad NV=1 (Python) — full JIT dispatch via Tegra ioctls
  2. C GPU Hot Path — replays tinygrad HCQGraphs from C, zero Python overhead
  3. TensorRT — NVIDIA's optimized inference engine

Usage:
  cd examples/presentation && nix develop
  NV=1 JITBEAM=2 python3 bench_models.py                            # All architectures (FP16, batch=1)
  NV=1 JITBEAM=2 python3 bench_models.py --precision fp32            # FP32 head-to-head
  NV=1 JITBEAM=2 python3 bench_models.py --precision tf32            # TF32 (tensor core FP32)
  NV=1 JITBEAM=2 python3 bench_models.py --batch 1,8,16             # Multiple batch sizes
  NV=1 JITBEAM=2 python3 bench_models.py --arch mlp                 # MLP only
  NV=1 JITBEAM=2 python3 bench_models.py --arch cnn                 # CNN only
  NV=1 JITBEAM=2 python3 bench_models.py --arch hybrid              # Hybrid only
  NV=1 JITBEAM=2 python3 bench_models.py --skip-tensorrt            # tinygrad only
  NV=1 JITBEAM=2 python3 bench_models.py --skip-hotpath             # skip C hot path
  NV=1 JITBEAM=2 python3 bench_models.py --iters 5000               # Fewer iterations (faster)
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


def print_comparison(nv_times, label, hp_times=None, trt_times=None):
    """Print head-to-head comparison between all available backends."""
    nv = stats(nv_times)
    cols = [("NV=1", nv)]
    if hp_times is not None:
        cols.append(("C Hot Path", stats(hp_times)))
    if trt_times is not None:
        cols.append(("TensorRT", stats(trt_times)))

    hdr = f"\n  {'Metric':20s}"
    sep = f"  {'─'*20}"
    for name, _ in cols:
        hdr += f" {name:>12s}"
        sep += f" {'─'*12}"
    print(hdr)
    print(sep)

    for metric, label_str in [("median", "Median (µs)"), ("std", "Std (µs)"),
                               ("p99", "P99 (µs)"), ("p999", "P99.9 (µs)"),
                               ("max", "Max (µs)")]:
        line = f"  {label_str:20s}"
        for _, s in cols:
            line += f" {s[metric]:12.1f}"
        print(line)

    line = f"  {'Frequency (Hz)':20s}"
    for _, s in cols:
        hz = 1e6 / s["median"] if s["median"] > 0 else 0
        line += f" {hz:12.0f}"
    print(line)

    medians = {name: s["median"] for name, s in cols}
    best_name = min(medians, key=medians.get)
    for name, med in medians.items():
        if name != best_name and med > 0:
            ratio = med / medians[best_name]
            print(f"  → {best_name} vs {name}: {best_name} wins {ratio:.2f}x")


# ═══════════════════════════════════════════════════════════════════════════════
# Main benchmark runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_mlp_benchmarks(args):
    """Run MLP benchmarks across all sizes."""
    from bench_tinygrad import bench_nv_mlp
    results = []
    use_fp16 = args.use_fp16
    use_tf32 = args.use_tf32
    prec_label = "TF32" if use_tf32 else ("FP16" if use_fp16 else "FP32")

    for batch_size in args.batch_sizes:
        bs_label = f"batch={batch_size}"
        print("\n" + "=" * 90)
        print(f"MLP ARCHITECTURES ({prec_label}, {bs_label}) — tinygrad NV=1 vs C Hot Path vs TensorRT")
        print(f"  Input: ({batch_size}, 12) — pos/vel/rpy/gyro state vector")
        print(f"  Output: ({batch_size}, 4) — thrust + angular rates")
        print("  Real-world: learned controllers, sensor fusion, reactive policies")
        print("=" * 90)

        for name, hidden_dims, desc, use_case in MLP_CONFIGS:
            print(f"\n{'─' * 90}")
            print(f"  {name}: {desc} [{bs_label}]")
            print(f"  Use case: {use_case}")
            print(f"{'─' * 90}")

            # Export weights for TensorRT
            prec_suffix = "tf32" if use_tf32 else ("fp16" if use_fp16 else "fp32")
            weight_path = WEIGHTS_DIR / f"{name}_{prec_suffix}.npz"
            export_mlp_onnx(hidden_dims, str(weight_path), use_fp16=use_fp16, batch_size=batch_size)

            # tinygrad NV=1
            print(f"\n  [tinygrad NV=1 {prec_label}]")
            nv_times, params, nv_result = bench_nv_mlp(
                name, hidden_dims, warmup=args.warmup, n_iters=args.iters,
                use_fp16=use_fp16, batch_size=batch_size)
            print_stats(f"NV=1 {prec_label} direct memmove ({params:,} params)", nv_times)

            result_entry = {
                "name": name, "arch": "mlp", "params": params,
                "hidden_dims": hidden_dims, "desc": desc, "use_case": use_case,
                "batch_size": batch_size,
                "nv": stats(nv_times),
            }

            # C GPU Hot Path
            hp_times = None
            if not args.skip_hotpath:
                print(f"\n  [C GPU Hot Path {prec_label}]")
                try:
                    from bench_c_hot_path import bench_c_mlp
                    hp_times, _ = bench_c_mlp(
                        name, hidden_dims, warmup=args.warmup, n_iters=args.iters,
                        use_fp16=use_fp16, batch_size=batch_size)
                    if hp_times is not None:
                        print_stats(f"C hot path ({params:,} params)", hp_times)
                        result_entry["hotpath"] = stats(hp_times)
                except Exception as e:
                    print(f"  C Hot Path FAILED: {e}")
                    result_entry["hotpath_error"] = str(e)

            # TensorRT
            trt_times = None
            if not args.skip_tensorrt:
                print(f"\n  [TensorRT {prec_label}]")
                try:
                    from bench_tensorrt import bench_trt_mlp
                    trt_times, _ = bench_trt_mlp(
                        name, hidden_dims, str(weight_path),
                        warmup=args.warmup, n_iters=args.iters,
                        use_fp16=use_fp16, use_tf32=use_tf32, batch_size=batch_size)
                    print_stats(f"TensorRT {prec_label} ({params:,} params)", trt_times)
                    result_entry["trt"] = stats(trt_times)
                except Exception as e:
                    print(f"  TensorRT FAILED: {e}")
                    result_entry["trt_error"] = str(e)

            # Comparison
            print_comparison(nv_times, name, hp_times=hp_times, trt_times=trt_times)
            results.append(result_entry)

    return results


def run_cnn_benchmarks(args):
    """Run 1D-CNN benchmarks."""
    from bench_tinygrad import bench_nv_cnn
    results = []
    use_fp16 = args.use_fp16
    use_tf32 = args.use_tf32
    prec_label = "TF32" if use_tf32 else ("FP16" if use_fp16 else "FP32")

    for batch_size in args.batch_sizes:
        bs_label = f"batch={batch_size}"
        print("\n" + "=" * 90)
        print(f"1D-CNN ARCHITECTURES ({prec_label}, {bs_label}) — tinygrad NV=1 vs C Hot Path vs TensorRT")
        print(f"  Input: ({batch_size}, {IN_DIM}, {SEQ_LEN}) — {SEQ_LEN}-sample IMU time window")
        print("  Output: (1, 4) — thrust + angular rates")
        print("  Real-world: temporal feature extraction from sensor history")
        print("=" * 90)

        for name, conv_layers, mlp_head, desc, use_case in CNN_CONFIGS:
            print(f"\n{'─' * 90}")
            print(f"  {name}: {desc} [{bs_label}]")
            print(f"  Use case: {use_case}")
            print(f"{'─' * 90}")

            prec_suffix = "tf32" if use_tf32 else ("fp16" if use_fp16 else "fp32")
            weight_path = WEIGHTS_DIR / f"{name}_{prec_suffix}.npz"
            export_cnn_onnx(conv_layers, mlp_head, str(weight_path), use_fp16=use_fp16, batch_size=batch_size)

            print(f"\n  [tinygrad NV=1 {prec_label}]")
            nv_times, params, nv_result = bench_nv_cnn(
                name, conv_layers, mlp_head, warmup=args.warmup, n_iters=args.iters,
                use_fp16=use_fp16, batch_size=batch_size)
            print_stats(f"NV=1 {prec_label} direct memmove ({params:,} params)", nv_times)

            result_entry = {
                "name": name, "arch": "cnn", "params": params,
                "conv_layers": conv_layers, "mlp_head": mlp_head,
                "desc": desc, "use_case": use_case,
                "batch_size": batch_size,
                "nv": stats(nv_times),
            }

            hp_times = None
            if not args.skip_hotpath:
                print(f"\n  [C GPU Hot Path {prec_label}]")
                try:
                    from bench_c_hot_path import bench_c_cnn
                    hp_times, _ = bench_c_cnn(
                        name, conv_layers, mlp_head, warmup=args.warmup, n_iters=args.iters,
                        use_fp16=use_fp16, batch_size=batch_size)
                    if hp_times is not None:
                        print_stats(f"C hot path ({params:,} params)", hp_times)
                        result_entry["hotpath"] = stats(hp_times)
                except Exception as e:
                    print(f"  C Hot Path FAILED: {e}")
                    result_entry["hotpath_error"] = str(e)

            trt_times = None
            if not args.skip_tensorrt:
                print(f"\n  [TensorRT {prec_label}]")
                try:
                    from bench_tensorrt import bench_trt_cnn
                    trt_times, _ = bench_trt_cnn(
                        name, conv_layers, mlp_head, str(weight_path),
                        warmup=args.warmup, n_iters=args.iters,
                        use_fp16=use_fp16, use_tf32=use_tf32, batch_size=batch_size)
                    print_stats(f"TensorRT {prec_label} ({params:,} params)", trt_times)
                    result_entry["trt"] = stats(trt_times)
                except Exception as e:
                    print(f"  TensorRT FAILED: {e}")
                    result_entry["trt_error"] = str(e)

            print_comparison(nv_times, name, hp_times=hp_times, trt_times=trt_times)
            results.append(result_entry)

    return results


def run_hybrid_benchmarks(args):
    """Run Hybrid CNN+MLP benchmarks."""
    from bench_tinygrad import bench_nv_hybrid
    results = []
    use_fp16 = args.use_fp16
    use_tf32 = args.use_tf32
    prec_label = "TF32" if use_tf32 else ("FP16" if use_fp16 else "FP32")

    for batch_size in args.batch_sizes:
        bs_label = f"batch={batch_size}"
        print("\n" + "=" * 90)
        print(f"HYBRID CNN+MLP ARCHITECTURES ({prec_label}, {bs_label}) — tinygrad NV=1 vs C Hot Path vs TensorRT")
        print(f"  Inputs: IMU window ({batch_size}, {IN_DIM}, {SEQ_LEN}) + Current state ({batch_size}, {IN_DIM})")
        print("  Output: (1, 4) — thrust + angular rates")
        print("  Real-world: history-aware perception fused with reactive control")
        print("=" * 90)

        for name, conv_layers, mlp_head, desc, use_case in HYBRID_CONFIGS:
            print(f"\n{'─' * 90}")
            print(f"  {name}: {desc} [{bs_label}]")
            print(f"  Use case: {use_case}")
            print(f"{'─' * 90}")

            prec_suffix = "tf32" if use_tf32 else ("fp16" if use_fp16 else "fp32")
            weight_path = WEIGHTS_DIR / f"{name}_{prec_suffix}.npz"
            export_hybrid_onnx(conv_layers, mlp_head, str(weight_path), use_fp16=use_fp16, batch_size=batch_size)

            print(f"\n  [tinygrad NV=1 {prec_label}]")
            nv_times, params, nv_result = bench_nv_hybrid(
                name, conv_layers, mlp_head, warmup=args.warmup, n_iters=args.iters,
                use_fp16=use_fp16, batch_size=batch_size)
            print_stats(f"NV=1 {prec_label} direct memmove ({params:,} params)", nv_times)

            result_entry = {
                "name": name, "arch": "hybrid", "params": params,
                "conv_layers": conv_layers, "mlp_head": mlp_head,
                "desc": desc, "use_case": use_case,
                "batch_size": batch_size,
                "nv": stats(nv_times),
            }

            hp_times = None
            if not args.skip_hotpath:
                print(f"\n  [C GPU Hot Path {prec_label}]")
                try:
                    from bench_c_hot_path import bench_c_hybrid
                    hp_times, _ = bench_c_hybrid(
                        name, conv_layers, mlp_head, warmup=args.warmup, n_iters=args.iters,
                        use_fp16=use_fp16, batch_size=batch_size)
                    if hp_times is not None:
                        print_stats(f"C hot path ({params:,} params)", hp_times)
                        result_entry["hotpath"] = stats(hp_times)
                except Exception as e:
                    print(f"  C Hot Path FAILED: {e}")
                    result_entry["hotpath_error"] = str(e)

            trt_times = None
            if not args.skip_tensorrt:
                print(f"\n  [TensorRT {prec_label}]")
                try:
                    from bench_tensorrt import bench_trt_hybrid
                    trt_times, _ = bench_trt_hybrid(
                        name, conv_layers, mlp_head, str(weight_path),
                        warmup=args.warmup, n_iters=args.iters,
                        use_fp16=use_fp16, use_tf32=use_tf32, batch_size=batch_size)
                    print_stats(f"TensorRT {prec_label} ({params:,} params)", trt_times)
                    result_entry["trt"] = stats(trt_times)
                except Exception as e:
                    print(f"  TensorRT FAILED: {e}")
                    result_entry["trt_error"] = str(e)

            print_comparison(nv_times, name, hp_times=hp_times, trt_times=trt_times)
            results.append(result_entry)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Summary & analysis
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(all_results):
    """Print a consolidated summary table and analysis."""
    print("\n")
    print("=" * 120)
    print("SUMMARY — tinygrad NV=1 vs C Hot Path vs TensorRT on Jetson AGX Orin 64GB")
    print("=" * 120)

    has_hp = any("hotpath" in r for r in all_results)
    has_trt = any("trt" in r for r in all_results)

    # Build header dynamically
    header = f"{'Model':18s} {'Arch':7s} {'Params':>10s} {'NV=1 (µs)':>10s}"
    if has_hp:
        header += f" {'C HP (µs)':>10s}"
    if has_trt:
        header += f" {'TRT (µs)':>10s}"
    header += f" {'NV Hz':>8s}"
    if has_hp:
        header += f" {'HP Hz':>8s}"
    if has_trt:
        header += f" {'TRT Hz':>8s}"
    header += f" {'Winner':>12s}"

    print(f"\n{header}")
    print("─" * len(header))

    wins = {"nv": 0, "hotpath": 0, "trt": 0, "tie": 0}

    for r in all_results:
        nv_med = r["nv"]["median"]
        nv_hz = 1e6 / nv_med if nv_med > 0 else 0

        medians = {"NV=1": nv_med}
        if "hotpath" in r:
            medians["C Hot Path"] = r["hotpath"]["median"]
        if "trt" in r:
            medians["TensorRT"] = r["trt"]["median"]

        best = min(medians, key=medians.get)
        if best == "NV=1":
            wins["nv"] += 1
        elif best == "C Hot Path":
            wins["hotpath"] += 1
        elif best == "TensorRT":
            wins["trt"] += 1

        line = f"{r['name']:18s} {r['arch']:7s} {r['params']:10,d} {nv_med:10.1f}"
        if has_hp:
            hp_med = r.get("hotpath", {}).get("median")
            line += f" {hp_med:10.1f}" if hp_med else f" {'---':>10s}"
        if has_trt:
            trt_med = r.get("trt", {}).get("median")
            line += f" {trt_med:10.1f}" if trt_med else f" {'---':>10s}"

        line += f" {nv_hz:8.0f}"
        if has_hp:
            hp_med = r.get("hotpath", {}).get("median")
            line += f" {1e6/hp_med:8.0f}" if hp_med else f" {'---':>8s}"
        if has_trt:
            trt_med = r.get("trt", {}).get("median")
            line += f" {1e6/trt_med:8.0f}" if trt_med else f" {'---':>8s}"

        line += f" {best:>12s}"
        print(line)

    # Analysis
    print(f"\n{'─' * 80}")
    score_parts = []
    if wins["nv"]:
        score_parts.append(f"NV=1 wins {wins['nv']}")
    if wins["hotpath"]:
        score_parts.append(f"C Hot Path wins {wins['hotpath']}")
    if wins["trt"]:
        score_parts.append(f"TensorRT wins {wins['trt']}")
    if wins["tie"]:
        score_parts.append(f"tied {wins['tie']}")
    print(f"Score: {', '.join(score_parts)}")

    if has_hp:
        print(f"\n  C Hot Path = same tinygrad GPU kernels, zero Python overhead")
        print(f"  Shows the true GPU-compute floor for tinygrad-compiled graphs")
        print(f"  Gap between NV=1 and C Hot Path = Python dispatch overhead (~60µs)")

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


def save_results(all_results, path, precision="fp16", batch_sizes=None):
    """Save raw results as JSON for further analysis."""
    with open(path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": "Jetson AGX Orin 64GB",
            "precision": precision,
            "batch_sizes": batch_sizes or [1],
            "jitbeam": os.environ.get("JITBEAM", "default"),
            "allow_tf32": os.environ.get("ALLOW_TF32", "0"),
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\nRaw results saved to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="tinygrad NV=1 vs C Hot Path vs TensorRT benchmark")
    parser.add_argument("--arch", choices=["mlp", "cnn", "hybrid", "all"], default="all",
                        help="Architecture type to benchmark (default: all)")
    parser.add_argument("--precision", choices=["fp16", "fp32", "tf32"], default="fp16",
                        help="Numeric precision for all backends (default: fp16)")
    parser.add_argument("--batch", type=str, default="1",
                        help="Comma-separated batch sizes, e.g. '1,8,16' (default: 1)")
    parser.add_argument("--iters", type=int, default=10000,
                        help="Iterations per benchmark (default: 10000)")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Warmup iterations (default: 50)")
    parser.add_argument("--skip-tensorrt", action="store_true",
                        help="Skip TensorRT benchmarks")
    parser.add_argument("--skip-hotpath", action="store_true",
                        help="Skip C GPU hot path benchmarks")
    parser.add_argument("--output", type=str, default="results.json",
                        help="Output JSON file for raw results")
    args = parser.parse_args()
    args.use_fp16 = (args.precision == "fp16")
    args.use_tf32 = (args.precision == "tf32")
    args.batch_sizes = [int(b.strip()) for b in args.batch.split(",")]
    prec_label = "TF32" if args.use_tf32 else ("FP16" if args.use_fp16 else "FP32")

    # Set ALLOW_TF32 for tinygrad when using TF32 precision
    if args.use_tf32:
        os.environ["ALLOW_TF32"] = "1"

    WEIGHTS_DIR.mkdir(exist_ok=True)

    print("=" * 90)
    print(f"tinygrad NV=1 vs C Hot Path vs TensorRT — {prec_label} Presentation Benchmarks")
    print("=" * 90)
    print(f"  Platform:    Jetson AGX Orin 64GB")
    print(f"  Precision:   {prec_label}")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  JITBEAM:     {os.environ.get('JITBEAM', 'default')}")
    print(f"  ALLOW_TF32:  {os.environ.get('ALLOW_TF32', '0')}")
    print(f"  Iterations:  {args.iters}")
    print(f"  Warmup:      {args.warmup}")
    backends = f"tinygrad NV=1 {prec_label}"
    if not args.skip_hotpath:
        backends += f" + C Hot Path {prec_label}"
    if not args.skip_tensorrt:
        backends += f" + TensorRT {prec_label}"
    print(f"  Backends:    {backends}")
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
    save_results(all_results, str(output_path), precision=args.precision, batch_sizes=args.batch_sizes)


if __name__ == "__main__":
    main()
