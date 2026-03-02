#!/usr/bin/env python3
"""Generate summary report from completed benchmark JSON files."""
import json, os

files = [
    ("results_fp16_b1.json", "FP16 batch=1"),
    ("results_fp32_b1.json", "FP32 batch=1 (pure, --noTF32)"),
    ("results_tf32_b1.json", "TF32 batch=1 (default FP32 = TF32 enabled)"),
]

for fname, label in files:
    if not os.path.exists(fname):
        print(f"\n===== {label}: FILE NOT FOUND =====")
        continue
    with open(fname) as f:
        data = json.load(f)

    beam = data.get("jitbeam", "?")
    tf32 = data.get("allow_tf32", "0")
    print(f"\n{'='*100}")
    print(f"  {label}  (JITBEAM={beam}, ALLOW_TF32={tf32})")
    print(f"{'='*100}")
    print(f"  {'Model':18s} {'Arch':7s} {'Params':>10s}  {'NV=1 µs':>9s}  {'HP µs':>9s}  {'TRT µs':>9s}  {'Winner':>12s}  {'Speedup':>8s}")
    print(f"  {'-'*18} {'-'*7} {'-'*10}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*12}  {'-'*8}")

    hp_wins = 0; trt_wins = 0; nv_wins = 0
    for r in data["results"]:
        nv = r["nv"]["median"]
        hp = r.get("hotpath", {}).get("median")
        trt = r.get("trt", {}).get("median")

        medians = {"NV=1": nv}
        if hp is not None: medians["C Hot Path"] = hp
        if trt is not None: medians["TensorRT"] = trt
        best_name = min(medians, key=medians.get)
        best_val = medians[best_name]

        if best_name == "C Hot Path": hp_wins += 1
        elif best_name == "TensorRT": trt_wins += 1
        else: nv_wins += 1

        # Speedup of winner vs second best
        sorted_m = sorted(medians.values())
        speedup = sorted_m[1] / sorted_m[0] if len(sorted_m) > 1 and sorted_m[0] > 0 else 1.0

        hp_s = f"{hp:9.1f}" if hp is not None else "      ---"
        trt_s = f"{trt:9.1f}" if trt is not None else "      ---"
        print(f"  {r['name']:18s} {r['arch']:7s} {r['params']:10,d}  {nv:9.1f}  {hp_s}  {trt_s}  {best_name:>12s}  {speedup:7.2f}x")

    total = len(data["results"])
    print(f"\n  SCORE: C Hot Path {hp_wins}/{total}, TensorRT {trt_wins}/{total}, NV=1 {nv_wins}/{total}")

# Cross-precision comparison
print(f"\n\n{'='*100}")
print(f"  CROSS-PRECISION COMPARISON (batch=1)")
print(f"{'='*100}")
print(f"  {'Model':18s} {'Arch':7s}  {'FP16 HP':>9s}  {'FP16 TRT':>9s}  {'FP32 HP':>9s}  {'FP32 TRT':>9s}  {'TF32 HP':>9s}  {'TF32 TRT':>9s}")
print(f"  {'-'*18} {'-'*7}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")

all_data = {}
for fname, key in [("results_fp16_b1.json","fp16"), ("results_fp32_b1.json","fp32"), ("results_tf32_b1.json","tf32")]:
    if os.path.exists(fname):
        with open(fname) as f:
            all_data[key] = {r["name"]: r for r in json.load(f)["results"]}

if len(all_data) == 3:
    for name in all_data["fp16"]:
        r16 = all_data["fp16"].get(name, {})
        r32 = all_data["fp32"].get(name, {})
        rtf = all_data["tf32"].get(name, {})
        arch = r16.get("arch", "?")

        def get_med(r, key):
            return r.get(key, {}).get("median")

        fp16_hp = get_med(r16, "hotpath")
        fp16_trt = get_med(r16, "trt")
        fp32_hp = get_med(r32, "hotpath")
        fp32_trt = get_med(r32, "trt")
        tf32_hp = get_med(rtf, "hotpath")
        tf32_trt = get_med(rtf, "trt")

        def fmt(v):
            return f"{v:9.1f}" if v is not None else "      ---"

        print(f"  {name:18s} {arch:7s}  {fmt(fp16_hp)}  {fmt(fp16_trt)}  {fmt(fp32_hp)}  {fmt(fp32_trt)}  {fmt(tf32_hp)}  {fmt(tf32_trt)}")

# Key findings
print(f"\n\n{'='*100}")
print(f"  KEY FINDINGS")
print(f"{'='*100}")
print("""
  1. FP16 batch=1: C Hot Path wins 7/17, TensorRT wins 10/17
     - TRT dominates large MLPs (2M+, 4M+, 8M) and large CNNs via cuBLAS tensor cores
     - HP wins small-to-medium models where launch overhead matters more than compute

  2. FP32 batch=1 (pure --noTF32): C Hot Path wins 10/17, TensorRT wins 7/17
     - Without TF32 tensor cores, TRT loses its cuBLAS advantage on large models
     - HP's launch-overhead advantage becomes the dominant factor
     - TRT CNN perf drops dramatically (cnn_xxlarge: 793µs vs 320µs FP16)

  3. TF32 batch=1: C Hot Path wins 11/17, TensorRT wins 6/17
     - TF32 barely helps tinygrad at batch=1 (M=1 can't use 8x16x16 WMMA tiles)
     - TRT benefits from TF32 on CNNs (cnn_xxlarge: 543µs TF32 vs 793µs pure FP32)
     - HP dominance increases because its speedups compound while TRT only partially recovers

  4. STILL TO RUN (batch=8 and batch=16):
     - FP16 batch=8  — critical test where tinygrad tensor cores SHOULD engage (M=8 >= TC tile)
     - TF32 batch=8  — tensor core vs tensor core comparison
     - FP16 batch=16 — deeper TC utilization
     - TF32 batch=16 — max TC engagement

  WHY BATCH>=8 MATTERS:
     At batch=1, matmuls are M=1 x K x N (gemv). Tensor cores need M>=8 for 8x16x16 tiles.
     At batch=8, matmuls become M=8 x K x N — first time tinygrad's WMMA ops can fire.
     This is the pivotal test to see if tinygrad can close/reverse the gap vs TRT on large models.
""")
