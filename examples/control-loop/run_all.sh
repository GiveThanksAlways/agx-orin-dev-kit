#!/usr/bin/env bash
set -euo pipefail
# ── Control-Loop Benchmark Runner ────────────────────────────────────────────
#
# Runs tinygrad NV=1 and PyTorch CUDA benchmarks back-to-back, then analysis.
#
# Usage:
#   cd examples/control-loop
#   bash run_all.sh [--duration 60]
#
# Prerequisites:
#   - Jetson AGX Orin with JetPack 6 / jetpack-nixos
#   - Both flakes built: ../tinygrad/flake.nix  and  ./flake.nix
#   - GPU clocks locked (the NV backend does this automatically via sysfs)
#
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS="$SCRIPT_DIR/results"
DURATION="${1:-60}"

mkdir -p "$RESULTS"

echo "================================================================"
echo " Control-Loop Benchmark — Jetson AGX Orin"
echo " Duration per test: ${DURATION}s"
echo " Results: $RESULTS/"
echo "================================================================"
echo ""

# ── Phase 1: tinygrad NV=1 ──────────────────────────────────────────────────
echo ">>> Running tinygrad NV=1 benchmark..."
echo "    (entering tinygrad nix shell)"
nix develop "$REPO_ROOT/examples/tinygrad#" \
    --command bash -c "
        cd '$SCRIPT_DIR'
        NV=1 python3 bench_tinygrad_nv.py --duration $DURATION --output-dir '$RESULTS'
    "
echo ""
echo ">>> tinygrad NV=1 benchmark complete."
echo ""

# ── Phase 2: PyTorch CUDA ───────────────────────────────────────────────────
echo ">>> Running PyTorch CUDA benchmark..."
echo "    (entering control-loop nix shell)"
nix develop "$SCRIPT_DIR#" \
    --command bash -c "
        cd '$SCRIPT_DIR'
        python3 bench_pytorch_cuda.py --duration $DURATION --output-dir '$RESULTS'
    "
echo ""
echo ">>> PyTorch CUDA benchmark complete."
echo ""

# ── Phase 3: Analysis ───────────────────────────────────────────────────────
echo ">>> Running analysis..."
nix develop "$SCRIPT_DIR#" \
    --command bash -c "
        cd '$SCRIPT_DIR'
        python3 analyze.py --input-dir '$RESULTS' --output-dir '$RESULTS' --duration $DURATION
    "
echo ""
echo "================================================================"
echo " DONE — Report: $RESULTS/report.md"
echo " Plots:  $RESULTS/*.png"
echo "================================================================"
