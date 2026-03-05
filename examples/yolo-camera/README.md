# YOLOv8 Camera Demos — Jetson AGX Orin

Real-time YOLOv8-n object detection on the **NVIDIA Jetson AGX Orin 64GB**
with live USB camera streaming via MJPEG over SSH.

Three inference backends are benchmarked and demo'd: TinyGrad (NV=1 +
C Hot Path + BEAM kernel auto-tuning), PyTorch (CUDA Graphs), and
TensorRT (FP16).

## Benchmark Results

YOLOv8-n @ 320x320, Orin AGX 64GB, MAXN power mode, all FP32 unless noted.

| Backend                        | Median    | FPS     | Notes                         |
| ------------------------------ | --------- | ------- | ----------------------------- |
| TensorRT 10.3 FP16             | 2,288 us  | **437** | trtexec, GPU-resident         |
| PyTorch 2.9.1 CUDA Graphs      | 6,313 us  | **158** | cuDNN, static graph replay    |
| **TinyGrad BEAM + C Hot Path** | 9,458 us  | **106** | auto-tuned PTX, MMIO dispatch |
| PyTorch 2.9.1 torch.compile    | 12,593 us | 79      | inductor backend              |
| TinyGrad C Hot Path (no BEAM)  | 26,277 us | 38      | default kernel schedule       |
| PyTorch 2.9.1 Eager            | 23,150 us | 43      | no graph optimizations        |

BEAM kernel auto-tuning improved TinyGrad from **38 FPS to 106 FPS** (2.8x),
closing the gap to PyTorch CUDA Graphs from 4.2x down to 1.5x.

## Quick Start

```bash
cd ~/agx-orin-dev-kit/examples/yolo-camera
nix develop
make            # build C hot path shared library (first time only)
```

### Run Demos

```bash
# TinyGrad C Hot Path + BEAM (recommended — 106 FPS inference)
PARALLEL=0 JITBEAM=2 NV=1 python3 demo_yolov8_hot_path.py --stream

# PyTorch CUDA Graphs (158 FPS inference)
python3 demo_yolov8_pytorch.py --stream

# TensorRT FP16 (437 FPS inference, live demo falls back to PyTorch)
python3 bench_yolov8_trt.py --size 320   # build engine
python3 demo_yolov8_trt.py --stream
```

### Run Benchmarks

```bash
# TinyGrad
PARALLEL=0 JITBEAM=2 NV=1 python3 demo_yolov8_hot_path.py --bench --size 320

# PyTorch (Eager + torch.compile + CUDA Graphs)
python3 bench_yolov8_pytorch.py --size 320

# TensorRT FP16 (via trtexec)
python3 bench_yolov8_trt.py --size 320
```

### View Stream (SSH)

```bash
# On laptop:
ssh -L 9999:localhost:8090 Orin-AGX-NixOS
# Then open: http://localhost:9999/
```

## BEAM Kernel Auto-Tuning

First run with `JITBEAM=2` performs an exhaustive search over kernel
schedules for all 186 GPU kernels (~90 min one-time cost). Results are
permanently cached in `~/.cache/tinygrad/cache.db`. Subsequent runs use
the cache and warm up in ~25 seconds.

`PARALLEL=0` is required because multiprocessing spawn hangs inside
`nix develop`. This forces single-threaded beam search.

## Files

| File                      | Description                                         |
| ------------------------- | --------------------------------------------------- |
| `demo_yolov8_hot_path.py` | TinyGrad C Hot Path camera demo (fastest tinygrad)  |
| `demo_yolov8_pytorch.py`  | PyTorch CUDA Graphs camera demo                     |
| `demo_yolov8_trt.py`      | TensorRT camera demo (falls back to PyTorch)        |
| `demo_yolov8_camera.py`   | TinyGrad @TinyJit camera demo (Python dispatch)     |
| `bench_yolov8_pytorch.py` | PyTorch benchmark (3 modes)                         |
| `bench_yolov8_trt.py`     | TensorRT FP16 benchmark (ONNX export + trtexec)     |
| `beam_search.py`          | Standalone BEAM search runner with progress logging |
| `flake.nix`               | Nix devshell (tinygrad, PyTorch, OpenCV, TRT)       |
| `Makefile`                | Builds `hot_path.so` from C sources                 |

## Hardware

- **Board:** NVIDIA Jetson AGX Orin 64GB Developer Kit
- **GPU:** Ampere SM87, 2048 CUDA cores, 64 tensor cores
- **OS:** NixOS 25.11 with JetPack 6 / L4T r36.4.4 / CUDA 12.6
- **Camera:** MMlove USB stereo global-shutter (2560x720 MJPG @ 60 FPS)
