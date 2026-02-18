# Telemetry Viewer (run from your PC)

View Jetson GPU/system dashboards in Chrome on your local machine.

## On the Jetson (one-time)

Build with the telemetry configuration from `examples/nixos`:

```bash
sudo nixos-rebuild switch --flake ./examples/nixos#nixos-telemetry
```

Or add the module to your own config:

```nix
# import the telemetry module
imports = [ ./modules/telemetry.nix ];

services.jetson-telemetry = {
  enable = true;
  # enableOpenTelemetry = true;  # optional: adds OTLP collector
};
```

## On your PC

```bash
cd examples/telemetry-viewer
./connect-telemetry.sh <jetson-ip>
# or with a specific user:
./connect-telemetry.sh <jetson-ip> myuser
```

Open Chrome: `http://localhost:3301`

Navigate: Dashboards -> Jetson -> Jetson Mission Control

## Dashboard panels

- GPU load gauge + utilization over time
- GPU frequency
- Per-core CPU usage
- RAM/SWAP gauges and timeseries
- All thermal sensors with color thresholds
- Power draw per rail (VDD_IN, CPU_GPU_CV, SOC)
- EMC (memory controller) load gauge
- VIC / APE engine stats

## Disable telemetry

```nix
services.jetson-telemetry.enable = false;
```

Rebuild. All telemetry services stop. Zero overhead.
