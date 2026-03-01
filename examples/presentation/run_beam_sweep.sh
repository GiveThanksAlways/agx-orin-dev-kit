#!/usr/bin/env bash
# run_beam_sweep.sh — Run the 3-way benchmark at multiple BEAM search levels.
#
# BEAM search in tinygrad optimizes GPU kernel schedules by exploring
# different tiling/vectorization/fusion strategies. Higher BEAM = more
# optimization time but potentially faster kernels.
#
# Usage (from nix dev shell):
#   cd examples/presentation && nix develop
#   bash run_beam_sweep.sh                                    # Full sweep FP16 batch=1 (default)
#   bash run_beam_sweep.sh --precision fp32                   # Full sweep FP32
#   bash run_beam_sweep.sh --precision tf32                   # Full sweep TF32 (tensor cores)
#   bash run_beam_sweep.sh --batch 1,8,16                    # Multiple batch sizes
#   bash run_beam_sweep.sh --arch mlp --precision fp32
#   bash run_beam_sweep.sh --skip-tensorrt
#
# TensorRT results are the same across BEAM levels (it has its own optimizer),
# so we only run TRT once (on the first BEAM level) and reuse cached engines.

set -euo pipefail
cd "$(dirname "$0")"

# Pass through any extra args (--arch, --iters, --skip-tensorrt, --precision, --batch, etc.)
EXTRA_ARGS="${*}"

# Detect precision from args (default fp16)
PRECISION="fp16"
prev_arg=""
for arg in "$@"; do
    if [[ "$prev_arg" == "--precision" ]]; then
        PRECISION="$arg"
    fi
    prev_arg="$arg"
done

# Detect batch from args (default 1)
BATCH="1"
prev_arg=""
for arg in "$@"; do
    if [[ "$prev_arg" == "--batch" ]]; then
        BATCH="$arg"
    fi
    prev_arg="$arg"
done

ITERS=10000
BEAM_LEVELS=(2 4 8)
BEAM_LABELS=("light" "medium" "heavy")

echo "═══════════════════════════════════════════════════════════════════════"
echo "  BEAM Search Sweep — tinygrad NV=1 vs C Hot Path vs TensorRT"
echo "  Precision: ${PRECISION^^}"
echo "  Batch:     ${BATCH}"
echo "  Levels: BEAM=${BEAM_LEVELS[0]} (light), BEAM=${BEAM_LEVELS[1]} (medium), BEAM=${BEAM_LEVELS[2]} (heavy)"
echo "  Iterations: ${ITERS}"
echo "  Extra args: ${EXTRA_ARGS:-none}"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Build batch tag for filenames (replace commas with dashes)
BATCH_TAG="${BATCH//,/-}"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  BEAM Search Sweep — tinygrad NV=1 vs C Hot Path vs TensorRT"
echo "  Precision: ${PRECISION^^}"
echo "  Levels: BEAM=${BEAM_LEVELS[0]} (light), BEAM=${BEAM_LEVELS[1]} (medium), BEAM=${BEAM_LEVELS[2]} (heavy)"
echo "  Iterations: ${ITERS}"
echo "  Extra args: ${EXTRA_ARGS:-none}"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

for i in "${!BEAM_LEVELS[@]}"; do
    beam="${BEAM_LEVELS[$i]}"
    label="${BEAM_LABELS[$i]}"
    outfile="results_beam${beam}_${PRECISION}_b${BATCH_TAG}.json"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  BEAM=${beam} (${label}) ${PRECISION^^} batch=${BATCH} — output: ${outfile}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    NV=1 JITBEAM="${beam}" python3 bench_models.py \
        --iters "${ITERS}" \
        --output "${outfile}" \
        ${EXTRA_ARGS} \
        2>&1 | tee "log_beam${beam}_${PRECISION}_b${BATCH_TAG}.txt"

    echo ""
    echo "  ✓ BEAM=${beam} ${PRECISION^^} batch=${BATCH} complete → ${outfile}"
    echo ""
done

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  BEAM Sweep Complete (${PRECISION^^}, batch=${BATCH})"
echo "  Results: results_beam{2,4,8}_${PRECISION}_b${BATCH_TAG}.json"
echo "  Logs:    log_beam{2,4,8}_${PRECISION}_b${BATCH_TAG}.txt"
echo "═══════════════════════════════════════════════════════════════════════"

# Quick comparison table
echo ""
echo "Quick BEAM comparison (median µs from each run):"
SWEEP_PRECISION="${PRECISION}" SWEEP_BATCH_TAG="${BATCH_TAG}" python3 -c "
import json, sys, os
precision = os.environ.get('SWEEP_PRECISION', 'fp16')
batch_tag = os.environ.get('SWEEP_BATCH_TAG', '1')
beam_levels = [2, 4, 8]
all_data = {}
for b in beam_levels:
    fname = f'results_beam{b}_{precision}_b{batch_tag}.json'
    try:
        with open(fname) as f:
            data = json.load(f)
        for r in data['results']:
            key = f\"{r['name']}_b{r.get('batch_size', 1)}\"
            if key not in all_data:
                all_data[key] = {'arch': r['arch'], 'params': r['params'], 'batch': r.get('batch_size', 1)}
            all_data[key][f'nv_b{b}'] = r['nv']['median']
            if 'hotpath' in r:
                all_data[key][f'hp_b{b}'] = r['hotpath']['median']
            if 'trt' in r:
                all_data[key]['trt'] = r['trt']['median']
    except FileNotFoundError:
        print(f'  WARNING: {fname} not found')

if not all_data:
    sys.exit(0)

# Header
hdr = f\"{'Model':18s} {'Arch':7s} {'Params':>8s} {'Batch':>5s}\"
for b in beam_levels:
    hdr += f' NV B={b:d}'
    hdr += f'  HP B={b:d}'
hdr += '      TRT'
print(hdr)
print('─' * len(hdr))

for name, d in all_data.items():
    line = f\"{name:18s} {d['arch']:7s} {d['params']:8,d} {d['batch']:5d}\"
    for b in beam_levels:
        nv = d.get(f'nv_b{b}')
        hp = d.get(f'hp_b{b}')
        line += f' {nv:7.1f}' if nv else '     ---'
        line += f' {hp:7.1f}' if hp else '     ---'
    trt = d.get('trt')
    line += f' {trt:8.1f}' if trt else '      ---'
    print(line)
"
