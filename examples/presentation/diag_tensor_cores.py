#!/usr/bin/env python3
"""diag_tensor_cores.py — Verify whether tinygrad Tensor Cores (WMMA) engage.

Tests matmul at various batch sizes and precisions, inspecting the compiled
AST for Ops.WMMA to confirm tensor core usage.

Usage (from nix dev shell):
  NV=1 python3 diag_tensor_cores.py
  NV=1 ALLOW_TF32=1 python3 diag_tensor_cores.py   # also test TF32 TC
  NV=1 DEBUG=3 python3 diag_tensor_cores.py          # verbose TC matching
"""
import os, sys
os.environ.setdefault("NV", "1")

from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import Ops
from tinygrad.engine.realize import get_program

def check_tc(M, K, N, dtype_in, dtype_out, label=""):
    """Check if a (M,K)@(K,N) matmul triggers WMMA tensor cores."""
    a = Tensor.rand(M, K, dtype=dtype_in)
    b = Tensor.rand(K, N, dtype=dtype_in)
    r = a.matmul(b, dtype=dtype_out)
    sched = r.schedule()
    # The matmul kernel is typically the last scheduled item
    ast = sched[-1].ast
    try:
        program = get_program(ast, Device[Device.DEFAULT].renderer)
        wmma_count = sum(1 for u in program.uops if u.op is Ops.WMMA)
        tc_opts = [x for x in program.applied_opts if hasattr(x, 'op') and str(x.op) == 'OptOps.TC']
        return wmma_count, len(tc_opts), program.src
    except Exception as e:
        return 0, 0, f"ERROR: {e}"


def check_linear(batch, in_features, out_features, dtype_in, label=""):
    """Check if nn.Linear at given batch size triggers TC."""
    from tinygrad import nn
    lin = nn.Linear(in_features, out_features)
    dtype_map = {dtypes.float16: dtypes.float16, dtypes.float32: dtypes.float32}
    x = Tensor.rand(batch, in_features, dtype=dtype_in)
    lin.weight = Tensor.rand(out_features, in_features, dtype=dtype_in)
    lin.bias = Tensor.rand(out_features, dtype=dtype_in)
    r = lin(x)
    sched = r.schedule()
    # Find the kernel with the largest reduce (that's the matmul)
    for item in reversed(sched):
        ast = item.ast
        try:
            program = get_program(ast, Device[Device.DEFAULT].renderer)
            wmma_count = sum(1 for u in program.uops if u.op is Ops.WMMA)
            if wmma_count > 0 or len(sched) <= 2:
                return wmma_count, program.src
        except Exception:
            continue
    return 0, "no matmul kernel found"


def main():
    ren = Device[Device.DEFAULT].renderer
    print(f"Device: {Device.DEFAULT}")
    print(f"Renderer: {ren.__class__.__name__}")
    if hasattr(ren, 'arch'):
        print(f"Arch: {ren.arch}")
    tc_list = getattr(ren, 'tensor_cores', [])
    print(f"Tensor Cores available: {len(tc_list)}")
    for tc in tc_list:
        print(f"  {tc}")
    print(f"ALLOW_TF32: {os.environ.get('ALLOW_TF32', '0')}")
    print()

    # ── Part 1: Raw matmul at various batch sizes ──
    print("=" * 80)
    print("PART 1: Raw matmul — does WMMA trigger at different M (batch) sizes?")
    print("=" * 80)

    configs = [
        # (M, K, N, dtype_in, dtype_out, label)
        (1,   12,  64,   dtypes.float16, dtypes.float16, "batch=1  FP16  12x64  (mlp_5k layer1)"),
        (1,   64,  64,   dtypes.float16, dtypes.float16, "batch=1  FP16  64x64  (mlp_5k layer2)"),
        (1,  1024, 1024, dtypes.float16, dtypes.float16, "batch=1  FP16  1024x1024 (mlp_1m)"),
        (8,   12,  64,   dtypes.float16, dtypes.float16, "batch=8  FP16  12x64"),
        (8,   64,  64,   dtypes.float16, dtypes.float16, "batch=8  FP16  64x64"),
        (8,  1024, 1024, dtypes.float16, dtypes.float16, "batch=8  FP16  1024x1024"),
        (16,  12,  64,   dtypes.float16, dtypes.float16, "batch=16 FP16  12x64"),
        (16,  64,  64,   dtypes.float16, dtypes.float16, "batch=16 FP16  64x64"),
        (16, 1024, 1024, dtypes.float16, dtypes.float16, "batch=16 FP16  1024x1024"),
        # FP32
        (1,  1024, 1024, dtypes.float32, dtypes.float32, "batch=1  FP32  1024x1024"),
        (8,  1024, 1024, dtypes.float32, dtypes.float32, "batch=8  FP32  1024x1024"),
        (16, 1024, 1024, dtypes.float32, dtypes.float32, "batch=16 FP32  1024x1024"),
    ]

    print(f"\n{'Label':50s} {'WMMA':>5s} {'TC opts':>7s}")
    print("─" * 65)
    for M, K, N, di, do, label in configs:
        wmma, tc, src = check_tc(M, K, N, di, do, label)
        status = "✓ TC" if wmma > 0 else "✗ no TC"
        print(f"  {label:48s} {wmma:5d} {tc:7d}    {status}")

    # ── Part 2: nn.Linear at different batch sizes ──
    print()
    print("=" * 80)
    print("PART 2: nn.Linear — realistic layer sizes from benchmark models")
    print("=" * 80)

    linear_configs = [
        # (batch, in_features, out_features, dtype, label)
        (1,  12,   64,   dtypes.float16, "batch=1  Linear(12,64) FP16"),
        (1,  1024, 1024, dtypes.float16, "batch=1  Linear(1024,1024) FP16"),
        (1,  2048, 2048, dtypes.float16, "batch=1  Linear(2048,2048) FP16"),
        (8,  12,   64,   dtypes.float16, "batch=8  Linear(12,64) FP16"),
        (8,  1024, 1024, dtypes.float16, "batch=8  Linear(1024,1024) FP16"),
        (8,  2048, 2048, dtypes.float16, "batch=8  Linear(2048,2048) FP16"),
        (16, 12,   64,   dtypes.float16, "batch=16 Linear(12,64) FP16"),
        (16, 1024, 1024, dtypes.float16, "batch=16 Linear(1024,1024) FP16"),
        (16, 2048, 2048, dtypes.float16, "batch=16 Linear(2048,2048) FP16"),
        # FP32 / TF32
        (1,  1024, 1024, dtypes.float32, "batch=1  Linear(1024,1024) FP32"),
        (8,  1024, 1024, dtypes.float32, "batch=8  Linear(1024,1024) FP32"),
        (16, 1024, 1024, dtypes.float32, "batch=16 Linear(1024,1024) FP32"),
    ]

    print(f"\n{'Label':50s} {'WMMA':>5s}")
    print("─" * 58)
    for batch, in_f, out_f, dtype, label in linear_configs:
        wmma, src = check_linear(batch, in_f, out_f, dtype, label)
        status = "✓ TC" if wmma > 0 else "✗ no TC"
        print(f"  {label:48s} {wmma:5d}    {status}")

    # ── Part 3: Summary ──
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
  Tensor Cores require minimum tile sizes (e.g., 8x16x16 for SM80+).
  At batch=1, the M dimension is 1, which is too small for TC tiles.
  TC_OPT=2 can pad M→16, but that wastes 16x compute — BEAM rejects it.

  Expected results:
    batch=1:  no WMMA (M=1 too small)
    batch=8:  WMMA likely for large K,N (M=8 matches TC M=16 with padding)
    batch=16: WMMA likely for most sizes (M=16 = exact TC tile)

  FP32 without ALLOW_TF32=1: no WMMA (TF32 tensor cores gated off)
  FP32 with ALLOW_TF32=1: WMMA possible (TF32 mode, 19-bit mantissa)
""")


if __name__ == "__main__":
    main()
