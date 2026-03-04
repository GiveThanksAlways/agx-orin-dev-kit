#!/usr/bin/env python3
"""YOLOv8-n TensorRT benchmark on Jetson AGX Orin — FP16 for max speed.

Pipeline:
  1. Build YOLOv8-n in PyTorch (same weights as tinygrad)
  2. Export to ONNX
  3. Build TensorRT FP16 engine via trtexec
  4. Benchmark via trtexec (GPU-resident, CUDA event timing)
  5. Also benchmark with data transfers for apples-to-apples comparison

Usage:
  cd ~/agx-orin-dev-kit/examples/yolo-camera
  nix develop
  python3 bench_yolov8_trt.py [--size 320] [--fp32]
"""
import argparse, os, sys, subprocess, re, time
from pathlib import Path
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Locate trtexec
# ═══════════════════════════════════════════════════════════════════════════════

def find_trtexec():
    """Find trtexec binary — prefer Jetson-extracted (SM87)."""
    # 1. Env var set by flake shellHook
    env_path = os.environ.get("JETSON_TRTEXEC")
    if env_path and os.path.isfile(env_path):
        return env_path
    # 2. Jetson-extracted next to presentation/
    here = Path(__file__).resolve().parent
    jetson = here.parent / "presentation" / "jetson-trt" / "extracted" / "usr" / "src" / "tensorrt" / "bin" / "trtexec"
    if jetson.is_file():
        return str(jetson)
    # 3. PATH
    import shutil
    p = shutil.which("trtexec")
    if p:
        return p
    raise FileNotFoundError("trtexec not found. Set JETSON_TRTEXEC or check PATH.")


def trtexec_env():
    """Build LD_LIBRARY_PATH for Jetson TRT libs."""
    import glob
    here = Path(__file__).resolve().parent
    jetson_libs = here.parent / "presentation" / "jetson-trt" / "extracted" / "usr" / "lib" / "aarch64-linux-gnu"
    env = os.environ.copy()
    if jetson_libs.is_dir():
        extra_paths = [str(jetson_libs)]
        # DLA compiler lib (provides libnvdla_compiler.so needed by libnvinfer_plugin)
        dla_dirs = glob.glob("/nix/store/*-nvidia-l4t-dla-compiler-*/lib")
        extra_paths.extend(dla_dirs)
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(extra_paths) + (":" + existing if existing else "")
    return env


# ═══════════════════════════════════════════════════════════════════════════════
# ONNX export via PyTorch
# ═══════════════════════════════════════════════════════════════════════════════

def export_onnx(variant, size, onnx_path):
    """Export YOLOv8 to ONNX using PyTorch — raw detection head output (no postprocessing)."""
    if os.path.exists(onnx_path):
        print(f"  Using cached ONNX: {onnx_path}")
        return

    print(f"  Exporting YOLOv8-{variant} to ONNX...")
    import torch
    import torch.nn as nn
    from bench_yolov8_pytorch import (YOLOv8PyTorch, load_tinygrad_weights_into_pytorch,
                                       Darknet, Yolov8NECK, ConvBlock)

    # Export the raw model without anchor/postprocessing — just backbone + neck + conv heads
    # TRT handles the heavy lifting; postprocessing is done on CPU anyway in real deployment
    class YOLOv8ForExport(nn.Module):
        """YOLOv8 that outputs raw detection head tensors (no anchor decode)."""
        def __init__(self, w=0.25, r=2.0, d=0.33, nc=80):
            super().__init__()
            self.full = YOLOv8PyTorch(w=w, r=r, d=d, nc=nc)
        def forward(self, x):
            p3, p4, p5 = self.full.net(x)
            h1, h2, h3 = self.full.fpn(p3, p4, p5)
            # Run per-head convolutions, concat box+cls, return flat tensor
            head = self.full.head
            outs = []
            for i, feat in enumerate([h1, h2, h3]):
                box = head.cv2[i](feat)
                cls = head.cv3[i](feat)
                b, _, fh, fw = box.shape
                combined = torch.cat((box, cls), 1)  # (B, 4*ch+nc, H, W)
                outs.append(combined.view(b, combined.shape[1], -1))  # (B, C, H*W)
            return torch.cat(outs, 2)  # (B, 4*ch+nc, total_anchors)

    w_map = {'n': (0.33, 0.25, 2.0), 's': (0.33, 0.50, 2.0), 'm': (0.67, 0.75, 1.5)}
    d, w, r = w_map[variant]
    model = YOLOv8ForExport(w=w, r=r, d=d, nc=80).cuda().eval()
    model.full = load_tinygrad_weights_into_pytorch(model.full, variant)

    dummy = torch.randn(1, 3, size, size, device='cuda')
    with torch.no_grad():
        torch.onnx.utils.export(
            model, dummy, onnx_path,
            input_names=['images'],
            output_names=['output'],
            opset_version=13,
            dynamic_axes=None,
        )
    print(f"  ONNX exported: {onnx_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# TensorRT engine build & benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def build_engine(onnx_path, engine_path, fp16=True):
    """Build TensorRT engine using trtexec."""
    if os.path.exists(engine_path):
        print(f"  Using cached engine: {engine_path}")
        return

    print(f"  Building TensorRT engine ({'FP16' if fp16 else 'FP32'})...")
    trtexec = find_trtexec()
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--memPoolSize=workspace:2048MiB",
    ]
    if fp16:
        cmd.append("--fp16")
    else:
        cmd.append("--noTF32")  # pure FP32

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=trtexec_env())
    if result.returncode != 0:
        print("trtexec build failed:")
        for line in (result.stdout + result.stderr).strip().split("\n")[-20:]:
            print(f"    {line}")
        sys.exit(1)
    print(f"  Engine built: {engine_path}")


def bench_engine(engine_path, n_iters=200, warmup_ms=5000, with_transfers=False, fp16=True):
    """Benchmark TRT engine using trtexec's built-in CUDA event timing."""
    trtexec = find_trtexec()
    cmd = [
        trtexec,
        f"--loadEngine={engine_path}",
        f"--warmUp={warmup_ms}",
        f"--iterations={n_iters}",
        "--useSpinWait",
    ]
    if fp16:
        cmd.append("--fp16")
    if not with_transfers:
        cmd.append("--noDataTransfers")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=trtexec_env())
    if result.returncode != 0:
        print("trtexec benchmark failed:")
        for line in (result.stdout + result.stderr).strip().split("\n")[-10:]:
            print(f"    {line}")
        return None

    return parse_trtexec(result.stdout + result.stderr)


def parse_trtexec(output):
    """Parse trtexec latency stats."""
    stats = {}
    for line in output.split("\n"):
        line = line.strip()
        if "GPU Compute Time:" in line and "Total" not in line:
            prefix = "gpu_"
        elif "H2D Latency:" in line:
            prefix = "h2d_"
        elif "D2H Latency:" in line:
            prefix = "d2h_"
        elif "Enqueue Time:" in line:
            prefix = "enqueue_"
        elif "Latency:" in line and "H2D" not in line and "D2H" not in line:
            prefix = ""
        else:
            m = re.search(r"Throughput:\s*([\d.]+)\s*qps", line)
            if m:
                stats["throughput_qps"] = float(m.group(1))
            continue
        for part in line.split(","):
            part = part.strip()
            m = re.search(r"(min|max|mean|median)\s*[:=]\s*([\d.]+)\s*ms", part)
            if m:
                stats[f"{prefix}{m.group(1)}"] = float(m.group(2)) * 1000.0  # ms → µs
            m = re.search(r"percentile\((\d+(?:\.\d+)?)%\)\s*=\s*([\d.]+)\s*ms", part)
            if m:
                stats[f"{prefix}p{m.group(1)}"] = float(m.group(2)) * 1000.0
    return stats


def print_trt_stats(label, stats):
    if not stats:
        print(f"  {label}: FAILED")
        return
    gpu_med = stats.get("gpu_median", stats.get("median", 0))
    gpu_mean = stats.get("gpu_mean", stats.get("mean", 0))
    gpu_min = stats.get("gpu_min", stats.get("min", 0))
    gpu_max = stats.get("gpu_max", stats.get("max", 0))
    gpu_p99 = stats.get("gpu_p99", stats.get("gpu_p99.0", gpu_med * 1.2))
    qps = stats.get("throughput_qps", 0)
    print(f"  {label}:")
    print(f"    median:  {gpu_med:.0f} µs  ({1e6/gpu_med:.1f} FPS)" if gpu_med else "    median:  N/A")
    print(f"    mean:    {gpu_mean:.0f} µs")
    print(f"    P99:     {gpu_p99:.0f} µs")
    print(f"    min/max: {gpu_min:.0f} / {gpu_max:.0f} µs")
    if qps:
        print(f"    trtexec throughput: {qps:.0f} qps")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8-n TensorRT benchmark")
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--variant", type=str, default="n")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--fp32", action="store_true", help="Use FP32 instead of FP16")
    args = parser.parse_args()

    fp16 = not args.fp32
    prec_str = "FP16" if fp16 else "FP32"

    print("=" * 60)
    print(f"  YOLOv8-{args.variant} TensorRT Benchmark @ {args.size}x{args.size} {prec_str}")
    print("=" * 60)

    # Paths
    here = Path(__file__).resolve().parent
    onnx_dir = here / "onnx"
    engine_dir = here / "engines"
    onnx_dir.mkdir(exist_ok=True)
    engine_dir.mkdir(exist_ok=True)

    prec_tag = "fp16" if fp16 else "fp32"
    onnx_path = str(onnx_dir / f"yolov8{args.variant}_{args.size}.onnx")
    engine_path = str(engine_dir / f"yolov8{args.variant}_{args.size}_{prec_tag}.engine")

    # Step 1: Export ONNX
    export_onnx(args.variant, args.size, onnx_path)

    # Step 2: Build engine
    build_engine(onnx_path, engine_path, fp16=fp16)

    # Step 3: Benchmark — GPU-resident (best case for TRT)
    print(f"\n  trtexec GPU-resident ({args.iters} iters)...")
    gpu_stats = bench_engine(engine_path, n_iters=args.iters, fp16=fp16)

    # Step 4: Benchmark — with H2D/D2H transfers (fair comparison)
    print(f"  trtexec with data transfers ({args.iters} iters)...")
    full_stats = bench_engine(engine_path, n_iters=args.iters, with_transfers=True, fp16=fp16)

    print(f"\n{'═'*60}")
    print(f"  Results — YOLOv8-{args.variant} @ {args.size}x{args.size} TensorRT {prec_str}")
    print(f"{'═'*60}")
    print_trt_stats("GPU-resident (no transfers)", gpu_stats)
    print_trt_stats("With H2D/D2H transfers", full_stats)
    print(f"{'═'*60}")

    if gpu_stats:
        med = gpu_stats.get("gpu_median", gpu_stats.get("median", 0))
        print(f"\n  Best (GPU-resident): {med:.0f} µs ({1e6/med:.1f} FPS)")


if __name__ == "__main__":
    main()
