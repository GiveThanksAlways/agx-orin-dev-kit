#!/usr/bin/env python3
"""YOLOv8-n PyTorch benchmark on Jetson AGX Orin — CUDA Graphs for max speed.

Three modes tested (fairness hierarchy):
  1. Eager mode          — baseline, no optimizations
  2. torch.compile       — inductor-optimized graph
  3. CUDA Graphs         — zero-overhead GPU replay (closest to C hot path)

Usage:
  cd ~/agx-orin-dev-kit/examples/yolo-camera
  nix develop
  python3 bench_yolov8_pytorch.py [--size 320] [--iters 200] [--warmup 20]
"""
import argparse, time, sys, os
import numpy as np
import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════════════════
# YOLOv8-n architecture in pure PyTorch (matching tinygrad's implementation)
# ═══════════════════════════════════════════════════════════════════════════════

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class ConvBlock(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3,3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBlock(c1, c_, k[0], 1)
        self.cv2 = ConvBlock(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvBlock(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvBlock((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3,3), e=1.0) for _ in range(n))
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvBlock(c1, c_, 1, 1)
        self.cv2 = ConvBlock(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='nearest')
    def forward(self, x):
        return self.up(x)

class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False)
        self.conv.weight.data = torch.arange(c1, dtype=torch.float32).view(1, c1, 1, 1)
        self.c1 = c1
    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class Darknet(nn.Module):
    def __init__(self, w, r, d):
        super().__init__()
        self.b1 = nn.Sequential(
            ConvBlock(3, int(64*w), 3, 2, 1),
            ConvBlock(int(64*w), int(128*w), 3, 2, 1))
        self.b2 = nn.Sequential(
            C2f(int(128*w), int(128*w), round(3*d), True),
            ConvBlock(int(128*w), int(256*w), 3, 2, 1),
            C2f(int(256*w), int(256*w), round(6*d), True))
        self.b3 = nn.Sequential(
            ConvBlock(int(256*w), int(512*w), 3, 2, 1),
            C2f(int(512*w), int(512*w), round(6*d), True))
        self.b4 = nn.Sequential(
            ConvBlock(int(512*w), int(512*w*r), 3, 2, 1),
            C2f(int(512*w*r), int(512*w*r), round(3*d), True))
        self.b5 = SPPF(int(512*w*r), int(512*w*r), 5)
    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x4 = self.b4(x3)
        x5 = self.b5(x4)
        return x2, x3, x5

class Yolov8NECK(nn.Module):
    def __init__(self, w, r, d):
        super().__init__()
        self.up = Upsample(2)
        self.n1 = C2f(int(512*w*(1+r)), int(512*w), round(3*d), False)
        self.n2 = C2f(int(768*w), int(256*w), round(3*d), False)
        self.n3 = ConvBlock(int(256*w), int(256*w), 3, 2, 1)
        self.n4 = C2f(int(768*w), int(512*w), round(3*d), False)
        self.n5 = ConvBlock(int(512*w), int(512*w), 3, 2, 1)
        self.n6 = C2f(int(512*w*(1+r)), int(512*w*r), round(3*d), False)
    def forward(self, p3, p4, p5):
        x = self.n1(torch.cat((self.up(p5), p4), 1))
        h1 = self.n2(torch.cat((self.up(x), p3), 1))
        h2 = self.n4(torch.cat((self.n3(h1), x), 1))
        h3 = self.n6(torch.cat((self.n5(h2), p5), 1))
        return h1, h2, h3

class DetectionHead(nn.Module):
    def __init__(self, nc=80, ch=16, filters=()):
        super().__init__()
        self.nc = nc
        self.ch = ch
        self.nl = len(filters)
        self.no = nc + ch * 4
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.dfl = DFL(ch)
        c1 = max(filters[0], nc)
        c2 = max(filters[0] // 4, ch * 4)
        self.cv2 = nn.ModuleList(
            nn.Sequential(ConvBlock(x, c2, 3), ConvBlock(c2, c2, 3), nn.Conv2d(c2, 4*ch, 1)) for x in filters)
        self.cv3 = nn.ModuleList(
            nn.Sequential(ConvBlock(x, c1, 3), ConvBlock(c1, c1, 3), nn.Conv2d(c1, nc, 1)) for x in filters)

    def forward(self, feats):
        shape = feats[0].shape
        x = [torch.cat((self.cv2[i](feats[i]), self.cv3[i](feats[i])), 1) for i in range(self.nl)]
        # Make anchors
        anchors, strides = [], []
        for i, s in enumerate([8, 16, 32]):
            _, _, h, w = x[i].shape
            sx = torch.arange(w, device=x[i].device, dtype=x[i].dtype) + 0.5
            sy = torch.arange(h, device=x[i].device, dtype=x[i].dtype) + 0.5
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchors.append(torch.stack((sx, sy), -1).reshape(-1, 2))
            strides.append(torch.full((h*w,), s, device=x[i].device, dtype=x[i].dtype))
        anchors = torch.cat(anchors, 0).T.unsqueeze(0)
        strides = torch.cat(strides, 0).unsqueeze(0).unsqueeze(0)

        y = [i.view(shape[0], self.no, -1) for i in x]
        x_cat = torch.cat(y, 2)
        box, cls = x_cat[:, :self.ch*4], x_cat[:, self.ch*4:]
        dbox = self._dist2bbox(self.dfl(box), anchors) * strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def _dist2bbox(self, distance, anchor_points):
        lt, rb = distance.chunk(2, 1)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), 1)

class YOLOv8PyTorch(nn.Module):
    def __init__(self, w=0.25, r=2.0, d=0.33, nc=80):
        super().__init__()
        self.net = Darknet(w, r, d)
        self.fpn = Yolov8NECK(w, r, d)
        self.head = DetectionHead(nc, filters=(int(256*w), int(512*w), int(512*w*r)))
    def forward(self, x):
        p3, p4, p5 = self.net(x)
        h1, h2, h3 = self.fpn(p3, p4, p5)
        return self.head([h1, h2, h3])


def load_tinygrad_weights_into_pytorch(pt_model, variant='n'):
    """Load tinygrad safetensors weights into the PyTorch model."""
    sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '../../external/tinygrad')))
    from tinygrad.nn.state import safe_load
    from examples.yolov8 import get_weights_location
    weights = safe_load(get_weights_location(variant))

    pt_sd = pt_model.state_dict()

    # Map tinygrad key conventions to PyTorch:
    #   bottleneck -> m  (C2f's ModuleList is called 'bottleneck' in tinygrad, 'm' in our PyTorch)
    #   net.b5.0.  -> net.b5.  (SPPF is wrapped in Sequential in tinygrad, direct attr in ours)
    loaded = 0
    for key, tensor_data in weights.items():
        pt_key = key.replace("bottleneck", "m").replace("net.b5.0.", "net.b5.")
        np_arr = tensor_data.numpy()
        if pt_key in pt_sd:
            t = torch.from_numpy(np_arr)
            if pt_sd[pt_key].shape == t.shape:
                pt_sd[pt_key] = t
                loaded += 1
            elif pt_sd[pt_key].shape == () and t.numel() == 1:
                pt_sd[pt_key] = t.squeeze()
                loaded += 1
    pt_model.load_state_dict(pt_sd, strict=False)
    print(f"  Loaded {loaded}/{len(pt_sd)} parameters from tinygrad safetensors")
    return pt_model


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark modes
# ═══════════════════════════════════════════════════════════════════════════════

def bench_eager(model, x_static, n_warmup, n_iters):
    """Eager mode — no compilation."""
    print("\n[1/3] Eager mode...")
    for _ in range(n_warmup):
        _ = model(x_static)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        _ = model(x_static)
        torch.cuda.synchronize()
        times.append(time.perf_counter_ns() - t0)
    return times


def bench_compiled(model, x_static, n_warmup, n_iters):
    """torch.compile with inductor backend."""
    print("\n[2/3] torch.compile (inductor)...")
    try:
        compiled = torch.compile(model, mode="max-autotune")
    except Exception as e:
        print(f"  torch.compile failed: {e}")
        return None

    for _ in range(n_warmup):
        _ = compiled(x_static)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        _ = compiled(x_static)
        torch.cuda.synchronize()
        times.append(time.perf_counter_ns() - t0)
    return times


def bench_cuda_graphs(model, x_static, n_warmup, n_iters):
    """CUDA Graphs — zero-overhead GPU replay (best case for PyTorch)."""
    print("\n[3/3] CUDA Graphs...")
    # Warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmup):
            _ = model(x_static)
    torch.cuda.current_stream().wait_stream(s)

    # Capture graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_static = model(x_static)

    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        g.replay()
        torch.cuda.synchronize()
        times.append(time.perf_counter_ns() - t0)
    return times


def print_stats(name, times_ns):
    if times_ns is None:
        print(f"  {name}: SKIPPED")
        return
    a = np.array(times_ns) / 1000.0  # ns → µs
    print(f"  {name}:")
    print(f"    median:  {np.median(a):.0f} µs  ({1e6/np.median(a):.1f} FPS)")
    print(f"    mean:    {np.mean(a):.0f} µs")
    print(f"    P99:     {np.percentile(a, 99):.0f} µs")
    print(f"    min/max: {np.min(a):.0f} / {np.max(a):.0f} µs")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8-n PyTorch benchmark")
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--variant", type=str, default="n")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  YOLOv8-{args.variant} PyTorch Benchmark @ {args.size}x{args.size}")
    print(f"  PyTorch {torch.__version__}, CUDA: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # Load model
    w_map = {'n': (0.33, 0.25, 2.0), 's': (0.33, 0.50, 2.0), 'm': (0.67, 0.75, 1.5)}
    d, w, r = w_map[args.variant]
    model = YOLOv8PyTorch(w=w, r=r, d=d, nc=80).cuda().eval()
    model = load_tinygrad_weights_into_pytorch(model, args.variant)
    model.eval()

    # Static input (GPU-resident, matching C hot path benchmark)
    x_static = torch.randn(1, 3, args.size, args.size, device='cuda', dtype=torch.float32)

    with torch.no_grad():
        eager_times = bench_eager(model, x_static, args.warmup, args.iters)
        compile_times = bench_compiled(model, x_static, args.warmup, args.iters)
        graph_times = bench_cuda_graphs(model, x_static, args.warmup, args.iters)

    print(f"\n{'═'*60}")
    print(f"  Results — YOLOv8-{args.variant} @ {args.size}x{args.size}")
    print(f"{'═'*60}")
    print_stats("Eager", eager_times)
    print_stats("torch.compile", compile_times)
    print_stats("CUDA Graphs", graph_times)
    print(f"{'═'*60}")

    # Save best result for comparison script
    best = graph_times or compile_times or eager_times
    median_us = np.median(np.array(best) / 1000.0)
    print(f"\n  Best: {median_us:.0f} µs ({1e6/median_us:.1f} FPS)")


if __name__ == "__main__":
    main()
