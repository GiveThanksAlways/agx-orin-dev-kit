#!/usr/bin/env bash
# connect-telemetry.sh -- run on your PC to view Jetson dashboards
#
# Usage:
#   ./connect-telemetry.sh <jetson-ip> [ssh-user]
#
# Then open http://localhost:3301 in Chrome.
# Ctrl-C to disconnect.

set -euo pipefail

JETSON_IP="${1:?Usage: $0 <jetson-ip> [ssh-user]}"
SSH_USER="${2:-root}"
GRAFANA_PORT=3301
PROMETHEUS_PORT=9090

echo ""
echo "=== Jetson Telemetry Tunnel ==="
echo ""
echo "  Forwarding:"
echo "    Grafana:    http://localhost:${GRAFANA_PORT}"
echo "    Prometheus: http://localhost:${PROMETHEUS_PORT}"
echo ""
echo "  Open Chrome -> http://localhost:${GRAFANA_PORT}"
echo "  Dashboard:  Dashboards -> Jetson -> Jetson Mission Control"
echo ""
echo "  Ctrl-C to disconnect."
echo ""

ssh -N \
  -L "${GRAFANA_PORT}:localhost:${GRAFANA_PORT}" \
  -L "${PROMETHEUS_PORT}:localhost:${PROMETHEUS_PORT}" \
  -L "9100:localhost:9100" \
  -L "9101:localhost:9101" \
  -L "8889:localhost:8889" \
  "${SSH_USER}@${JETSON_IP}"
