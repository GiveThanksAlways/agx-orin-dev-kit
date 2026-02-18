#!/usr/bin/env python3
"""
Stereo Pipeline Throughput Benchmark — tinygrad on Jetson Orin AGX.

Measures end-to-end pipeline throughput for stereo vision tasks,
comparing NV=1 vs CUDA=1 backends. Benchmarks:

  1. Stereo depth estimation (cost volume matching)
  2. CNN inference (detector + keypoint models)
  3. Full pipeline (capture → preprocess → depth → inference → postprocess)

Usage:
    NV=1 python3 bench_stereo_pipeline.py
    CUDA=1 python3 bench_stereo_pipeline.py
    NV=1 JITBEAM=2 python3 bench_stereo_pipeline.py  # With BEAM search
"""
import os, sys, time
import numpy as np

def get_backend_name():
    if os.environ.get("NV") == "1": return "NV"
    if os.environ.get("CUDA") == "1": return "CUDA"
    return "CPU"

def bench_depth_estimation(sizes=None, max_disps=None, n_iter=10):
    """Benchmark stereo depth computation at various resolutions."""
    from depth_map import stereo_cost_volume_tinygrad

    if sizes is None:
        sizes = [(320, 240), (640, 480), (1280, 720)]
    if max_disps is None:
        max_disps = [32, 64, 128]

    print("=== Stereo Depth Estimation ===")
    print(f"{'Resolution':>12} {'MaxDisp':>8} {'Time(ms)':>10} {'FPS':>8} {'Mpix/s':>10}")
    print("-" * 56)

    results = []
    for W, H in sizes:
        for md in max_disps:
            if md > W // 2:
                continue  # Skip unreasonable disparity values

            left = np.random.randint(0, 255, (H, W), dtype=np.uint8)
            right = np.roll(left, 20, axis=1)

            # Warmup
            _ = stereo_cost_volume_tinygrad(left, right, md)

            times = []
            for _ in range(n_iter):
                t0 = time.time()
                _ = stereo_cost_volume_tinygrad(left, right, md)
                times.append(time.time() - t0)

            avg_ms = np.mean(times) * 1000
            fps = 1000.0 / avg_ms
            mpix = (W * H) / (avg_ms / 1000) / 1e6
            print(f"{W}x{H:>4} {md:>8} {avg_ms:>10.1f} {fps:>8.1f} {mpix:>10.1f}")
            results.append({"size": (W, H), "max_disp": md, "ms": avg_ms, "fps": fps})

    return results

def bench_cnn_inference(n_iter=20):
    """Benchmark CNN model forward passes."""
    from tinygrad import Tensor
    from tinygrad.nn import Conv2d, Linear

    print("\n=== CNN Inference (Forward Pass) ===")
    print(f"{'Model':>20} {'Params':>10} {'Time(ms)':>10} {'FPS':>8}")
    print("-" * 54)

    results = []

    # TinyDetector (from stereo_object_detect.py)
    class TinyDetector:
        def __init__(self):
            self.conv1 = Conv2d(3, 16, 3, stride=2, padding=1)
            self.conv2 = Conv2d(16, 32, 3, stride=2, padding=1)
            self.conv3 = Conv2d(32, 64, 3, stride=2, padding=1)
            self.conv4 = Conv2d(64, 128, 3, stride=2, padding=1)
            self.head = Conv2d(128, 10, 1)  # 5 classes + 4 bbox + 1 obj

        def __call__(self, x):
            x = self.conv1(x).relu()
            x = self.conv2(x).relu()
            x = self.conv3(x).relu()
            x = self.conv4(x).relu()
            return self.head(x)

    # HandKeypointNet
    class HandNet:
        def __init__(self):
            self.conv1 = Conv2d(3, 32, 3, stride=2, padding=1)
            self.conv2 = Conv2d(32, 64, 3, stride=2, padding=1)
            self.conv3 = Conv2d(64, 128, 3, stride=2, padding=1)
            self.conv4 = Conv2d(128, 256, 3, stride=2, padding=1)
            self.fc1 = Linear(256 * 8 * 8, 512)
            self.fc2 = Linear(512, 42)

        def __call__(self, x):
            x = self.conv1(x).relu()
            x = self.conv2(x).relu()
            x = self.conv3(x).relu()
            x = self.conv4(x).relu()
            x = x.reshape(x.shape[0], -1)
            x = self.fc1(x).relu()
            return self.fc2(x)

    models = [
        ("TinyDetector", TinyDetector(), Tensor.randn(1, 3, 224, 224)),
        ("HandKeypointNet", HandNet(), Tensor.randn(1, 3, 128, 128)),
    ]

    for name, model, x in models:
        # Count params
        import inspect
        params = 0
        for attr_name in dir(model):
            attr = getattr(model, attr_name)
            if hasattr(attr, 'weight'):
                params += np.prod(attr.weight.shape)
                if hasattr(attr, 'bias') and attr.bias is not None:
                    params += np.prod(attr.bias.shape)

        # Warmup
        for _ in range(3):
            model(x).numpy()

        times = []
        for _ in range(n_iter):
            t0 = time.time()
            model(x).numpy()
            times.append(time.time() - t0)

        avg_ms = np.mean(times) * 1000
        fps = 1000.0 / avg_ms
        print(f"{name:>20} {params:>10,.0f} {avg_ms:>10.2f} {fps:>8.1f}")
        results.append({"model": name, "params": params, "ms": avg_ms, "fps": fps})

    # Batch scaling for detector
    print(f"\n{'Batch Size':>12} {'Tot Time(ms)':>12} {'Per Image(ms)':>14} {'Images/s':>10}")
    print("-" * 54)
    det = TinyDetector()
    for bs in [1, 2, 4, 8]:
        x = Tensor.randn(bs, 3, 224, 224)
        # Warmup
        det(x).numpy()
        times = []
        for _ in range(n_iter):
            t0 = time.time()
            det(x).numpy()
            times.append(time.time() - t0)
        avg_ms = np.mean(times) * 1000
        per_img = avg_ms / bs
        ips = bs * 1000.0 / avg_ms
        print(f"{bs:>12} {avg_ms:>12.2f} {per_img:>14.2f} {ips:>10.1f}")

    return results

def bench_full_pipeline():
    """Benchmark the full stereo + detection pipeline."""
    from depth_map import stereo_cost_volume_tinygrad
    from tinygrad import Tensor
    from tinygrad.nn import Conv2d

    print("\n=== Full Pipeline: Capture → Depth → Detection → 3D ===")
    print("(Using synthetic data to isolate compute from camera I/O)")

    W, H = 640, 480
    max_disp = 64

    # Simulate camera frames
    left_gray = np.random.randint(0, 255, (H, W), dtype=np.uint8)
    right_gray = np.roll(left_gray, 20, axis=1)
    left_color = np.stack([left_gray]*3, axis=-1)

    # Build detector
    class QuickDet:
        def __init__(self):
            self.c1 = Conv2d(3, 16, 3, stride=2, padding=1)
            self.c2 = Conv2d(16, 32, 3, stride=2, padding=1)
            self.c3 = Conv2d(32, 64, 3, stride=2, padding=1)
            self.out = Conv2d(64, 6, 1)
        def __call__(self, x):
            return self.out(self.c3(self.c2(self.c1(x).relu()).relu()).relu())

    det = QuickDet()

    # Warmup
    import cv2
    input_t = Tensor(cv2.resize(left_color, (224, 224))[:,:,::-1].astype(np.float32).transpose(2,0,1) / 255.0).reshape(1,3,224,224)
    det(input_t).numpy()
    stereo_cost_volume_tinygrad(left_gray, right_gray, max_disp)

    n_iter = 10
    stage_times = {"depth": [], "preprocess": [], "detection": [], "postprocess": [], "total": []}

    for _ in range(n_iter):
        t_total = time.time()

        # Stage 1: Depth estimation
        t0 = time.time()
        disp = stereo_cost_volume_tinygrad(left_gray, right_gray, max_disp)
        stage_times["depth"].append(time.time() - t0)

        # Stage 2: Preprocess for detection
        t0 = time.time()
        resized = cv2.resize(left_color, (224, 224))
        rgb = resized[:,:,::-1].astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))
        input_t = Tensor(chw).reshape(1, 3, 224, 224)
        stage_times["preprocess"].append(time.time() - t0)

        # Stage 3: CNN detection
        t0 = time.time()
        output = det(input_t).numpy()
        stage_times["detection"].append(time.time() - t0)

        # Stage 4: Postprocess (decode + 3D projection)
        t0 = time.time()
        # Simulate decoding bounding boxes and depth lookup
        _ = np.max(output)
        _ = np.median(disp[disp > 0]) if np.any(disp > 0) else 0
        stage_times["postprocess"].append(time.time() - t0)

        stage_times["total"].append(time.time() - t_total)

    print(f"\nPipeline stage breakdown (640×480, max_disp={max_disp}):")
    print(f"{'Stage':>15} {'Mean(ms)':>10} {'Std(ms)':>10} {'% Total':>10}")
    print("-" * 48)

    total_mean = np.mean(stage_times["total"]) * 1000
    for stage in ["depth", "preprocess", "detection", "postprocess"]:
        mean_ms = np.mean(stage_times[stage]) * 1000
        std_ms = np.std(stage_times[stage]) * 1000
        pct = mean_ms / total_mean * 100
        print(f"{stage:>15} {mean_ms:>10.2f} {std_ms:>10.2f} {pct:>9.1f}%")

    print(f"{'TOTAL':>15} {total_mean:>10.2f} {np.std(stage_times['total'])*1000:>10.2f} {'100.0':>9}%")
    print(f"\nEnd-to-end FPS: {1000.0/total_mean:.1f}")

def main():
    backend = get_backend_name()
    beam = os.environ.get("JITBEAM", "0")

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Stereo Pipeline Benchmark — Jetson Orin AGX    ║")
    print(f"║  Backend: {backend:>4}  JITBEAM: {beam:>2}                      ║")
    print(f"╚══════════════════════════════════════════════════╝\n")

    bench_depth_estimation()
    bench_cnn_inference()
    bench_full_pipeline()

    print("\n=== Benchmark Complete ===")

if __name__ == "__main__":
    main()
