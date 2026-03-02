#!/usr/bin/env python3
"""Generate comprehensive report from all benchmark results."""
import json, os, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Map of all result files to load
RESULT_SETS = {
    # batch=1
    "fp16_b1": ("results_fp16_b1.json", "FP16 batch=1"),
    "fp32_b1": ("results_fp32_b1.json", "FP32 batch=1 (pure --noTF32)"),
    "tf32_b1": ("results_tf32_b1.json", "TF32 batch=1"),
    # batch=8 (split by arch)
    "fp16_b8_mlp": ("results_fp16_b8_mlp.json", "FP16 batch=8 MLP"),
    "fp16_b8_cnn": ("results_fp16_b8_cnn.json", "FP16 batch=8 CNN"),
    "fp16_b8_hybrid": ("results_fp16_b8_hybrid.json", "FP16 batch=8 Hybrid"),
    "tf32_b8_mlp": ("results_tf32_b8_mlp.json", "TF32 batch=8 MLP"),
    "tf32_b8_cnn": ("results_tf32_b8_cnn.json", "TF32 batch=8 CNN"),
    "tf32_b8_hybrid": ("results_tf32_b8_hybrid.json", "TF32 batch=8 Hybrid"),
    # batch=16
    "fp16_b16_mlp": ("results_fp16_b16_mlp.json", "FP16 batch=16 MLP"),
}

all_data = {}
for key, (fname, label) in RESULT_SETS.items():
    if os.path.exists(fname):
        with open(fname) as f:
            d = json.load(f)
        all_data[key] = d
        #print(f"  Loaded {fname}: {len(d['results'])} models", file=sys.stderr)

def get_results(key):
    if key in all_data:
        return all_data[key]["results"]
    return []

def score(results):
    hp_w, trt_w, nv_w = 0, 0, 0
    for r in results:
        nv = r["nv"]["median"]
        hp = r.get("hotpath", {}).get("median")
        trt = r.get("trt", {}).get("median")
        medians = {"NV": nv}
        if hp: medians["HP"] = hp
        if trt: medians["TRT"] = trt
        best = min(medians, key=medians.get)
        if best == "HP": hp_w += 1
        elif best == "TRT": trt_w += 1
        else: nv_w += 1
    return hp_w, trt_w, nv_w

def print_table(results, label):
    if not results:
        print(f"\n  {label}: NO DATA")
        return
    hp_w, trt_w, nv_w = score(results)
    total = len(results)
    print(f"\n  {label}")
    print(f"  {'Model':18s} {'Arch':7s} {'Params':>10s}  {'NV µs':>8s}  {'HP µs':>8s}  {'TRT µs':>8s}  {'Winner':>12s}  {'x':>6s}")
    print(f"  {'-'*18} {'-'*7} {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*6}")
    for r in results:
        nv = r["nv"]["median"]
        hp = r.get("hotpath", {}).get("median")
        trt = r.get("trt", {}).get("median")
        medians = {"NV=1": nv}
        if hp: medians["HP"] = hp
        if trt: medians["TRT"] = trt
        best = min(medians, key=medians.get)
        best_v = medians[best]
        sorted_v = sorted(medians.values())
        ratio = sorted_v[1]/sorted_v[0] if len(sorted_v) > 1 and sorted_v[0] > 0 else 1.0
        hp_s = f"{hp:8.1f}" if hp else "     ---"
        trt_s = f"{trt:8.1f}" if trt else "     ---"
        print(f"  {r['name']:18s} {r['arch']:7s} {r['params']:10,d}  {nv:8.1f}  {hp_s}  {trt_s}  {best:>12s}  {ratio:5.2f}x")
    print(f"  SCORE: HP {hp_w}/{total} | TRT {trt_w}/{total} | NV {nv_w}/{total}")
    return hp_w, trt_w, nv_w

# Combine batch=8 split files
fp16_b8_all = get_results("fp16_b8_mlp") + get_results("fp16_b8_cnn") + get_results("fp16_b8_hybrid")
tf32_b8_all = get_results("tf32_b8_mlp") + get_results("tf32_b8_cnn") + get_results("tf32_b8_hybrid")

print("="*100)
print("COMPLETE BENCHMARK REPORT")
print("tinygrad NV=1 vs C Hot Path vs TensorRT — Jetson AGX Orin 64GB")
print("="*100)

# batch=1
print("\n" + "="*80)
print("BATCH=1 RESULTS (single inference — the drone/robot control loop case)")
print("="*80)
print_table(get_results("fp16_b1"), "FP16 batch=1 (JITBEAM=8)")
print_table(get_results("fp32_b1"), "FP32 batch=1 pure --noTF32 (JITBEAM=8)")
print_table(get_results("tf32_b1"), "TF32 batch=1 (JITBEAM=8)")

# batch=8
print("\n" + "="*80)
print("BATCH=8 RESULTS (batched inference — multi-agent, ensemble, or throughput)")
print("="*80)
print_table(fp16_b8_all, "FP16 batch=8 (MLP/CNN: JITBEAM=8, Hybrid: JITBEAM=2)")
print_table(tf32_b8_all, "TF32 batch=8 (CNN/Hybrid: JITBEAM=2, MLP: JITBEAM=8)")

# batch=16 MLPs
print("\n" + "="*80)
print("BATCH=16 RESULTS (higher throughput, deeper TC utilization)")
print("="*80)
print_table(get_results("fp16_b16_mlp"), "FP16 batch=16 MLP only (JITBEAM=2)")

# Grand totals
print("\n" + "="*80)
print("GRAND SCOREBOARD")
print("="*80)

all_runs = [
    ("FP16 b=1", get_results("fp16_b1")),
    ("FP32 b=1", get_results("fp32_b1")),
    ("TF32 b=1", get_results("tf32_b1")),
    ("FP16 b=8", fp16_b8_all),
    ("TF32 b=8", tf32_b8_all),
    ("FP16 b=16 MLP", get_results("fp16_b16_mlp")),
]

grand_hp, grand_trt, grand_nv, grand_total = 0, 0, 0, 0
print(f"\n  {'Run':20s}  {'HP wins':>8s}  {'TRT wins':>9s}  {'NV wins':>8s}  {'Total':>6s}")
print(f"  {'-'*20}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*6}")
for label, results in all_runs:
    if not results: continue
    hp_w, trt_w, nv_w = score(results)
    total = len(results)
    grand_hp += hp_w; grand_trt += trt_w; grand_nv += nv_w; grand_total += total
    print(f"  {label:20s}  {hp_w:8d}  {trt_w:9d}  {nv_w:8d}  {total:6d}")
print(f"  {'-'*20}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*6}")
print(f"  {'GRAND TOTAL':20s}  {grand_hp:8d}  {grand_trt:9d}  {grand_nv:8d}  {grand_total:6d}")
print(f"\n  C Hot Path win rate: {grand_hp}/{grand_total} = {100*grand_hp/grand_total:.0f}%")
print(f"  TensorRT win rate:   {grand_trt}/{grand_total} = {100*grand_trt/grand_total:.0f}%")

# Key model comparisons across batch sizes
print("\n" + "="*80)
print("BATCH SIZE SCALING — Selected Models")
print("="*80)

# Build lookup by (name, batch_size, precision)
lookup = {}
for key, results in [("fp16_b1", get_results("fp16_b1")),
                      ("fp16_b8", fp16_b8_all),
                      ("fp16_b16", get_results("fp16_b16_mlp"))]:
    for r in results:
        lookup[(r["name"], key)] = r

key_models = ["mlp_18k", "mlp_530k", "mlp_2m", "mlp_8m", "cnn_small", "cnn_xxlarge", "hybrid_large"]
print(f"\n  {'Model':18s}  {'b1 HP':>8s}  {'b1 TRT':>8s}  {'b8 HP':>8s}  {'b8 TRT':>8s}  {'b16 HP':>8s}  {'b16 TRT':>8s}")
print(f"  {'-'*18}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
for name in key_models:
    def g(key):
        r = lookup.get((name, key))
        if not r: return "---", "---"
        hp = r.get("hotpath", {}).get("median")
        trt = r.get("trt", {}).get("median")
        return (f"{hp:.0f}" if hp else "---"), (f"{trt:.0f}" if trt else "---")
    b1_hp, b1_trt = g("fp16_b1")
    b8_hp, b8_trt = g("fp16_b8")
    b16_hp, b16_trt = g("fp16_b16")
    print(f"  {name:18s}  {b1_hp:>8s}  {b1_trt:>8s}  {b8_hp:>8s}  {b8_trt:>8s}  {b16_hp:>8s}  {b16_trt:>8s}")
