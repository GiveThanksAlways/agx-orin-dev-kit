#!/usr/bin/env bash
# Run breakdown benchmark for both backends.
# Usage: cd examples/control-loop && ./run_breakdown.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Breakdown Benchmark: Where Does the Time Actually Go?         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo

# ── tinygrad NV=1 ────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Running tinygrad NV=1 breakdown..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nix develop ../tinygrad --command bash -c \
  'cd "$(dirname "$0")" && NV=1 python3 bench_breakdown.py --backend tinygrad' \
  -- "$PWD" 2>&1 | tee results/breakdown_tinygrad.txt

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Running PyTorch CUDA breakdown..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nix develop . --command python3 bench_breakdown.py --backend pytorch 2>&1 | tee results/breakdown_pytorch.txt

echo
echo "Done. Results in results/breakdown_*.txt"
