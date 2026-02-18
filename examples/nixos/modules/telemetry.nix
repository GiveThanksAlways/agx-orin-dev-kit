# telemetry.nix -- Grafana + Prometheus + tegrastats for Jetson Orin
#
# Self-contained NixOS module. Import and enable:
#
#   imports = [ ./modules/telemetry.nix ];
#   services.jetson-telemetry.enable = true;
#
# Grafana UI: http://<host>:3301
# Prometheus: http://<host>:9090

{ config, lib, pkgs, ... }:

let
  inherit (lib)
    mkEnableOption
    mkIf
    mkOption
    types;

  cfg = config.services.jetson-telemetry;

  # Tegrastats parser -- writes Prometheus text-format metrics
  tegrastatsExporter = pkgs.writeShellScriptBin "tegrastats-exporter" ''
    #!/usr/bin/env bash
    # No set -e: ((expr)) returning 0 is falsy in bash and would kill the script
    echo "Starting tegrastats exporter"

    TEGRASTATS_BIN="${pkgs.nvidia-jetpack.l4t-tools}/bin/tegrastats"
    METRICS_FILE="/var/lib/jetson-telemetry/metrics/tegrastats-metrics.prom.txt"
    INTERVAL_MS=500

    mkdir -p "$(dirname "$METRICS_FILE")"

    "$TEGRASTATS_BIN" --interval "$INTERVAL_MS" | while IFS= read -r line; do
      timestamp=$(date +%s)

      {
        echo "# HELP jetson_ram_used_mb RAM used in megabytes"
        echo "# TYPE jetson_ram_used_mb gauge"
        echo "# HELP jetson_ram_total_mb RAM total in megabytes"
        echo "# TYPE jetson_ram_total_mb gauge"
        echo "# HELP jetson_swap_used_mb SWAP used in megabytes"
        echo "# TYPE jetson_swap_used_mb gauge"
        echo "# HELP jetson_swap_total_mb SWAP total in megabytes"
        echo "# TYPE jetson_swap_total_mb gauge"
        echo "# HELP jetson_cpu_usage_percent CPU usage percentage per core"
        echo "# TYPE jetson_cpu_usage_percent gauge"
        echo "# HELP jetson_cpu_freq_mhz CPU frequency in MHz"
        echo "# TYPE jetson_cpu_freq_mhz gauge"
        echo "# HELP jetson_gpu_usage_percent GPU usage percentage"
        echo "# TYPE jetson_gpu_usage_percent gauge"
        echo "# HELP jetson_gpu_freq_mhz GPU frequency in MHz"
        echo "# TYPE jetson_gpu_freq_mhz gauge"
        echo "# HELP jetson_emc_freq_percent EMC frequency percentage"
        echo "# TYPE jetson_emc_freq_percent gauge"
        echo "# HELP jetson_emc_freq_mhz EMC frequency in MHz"
        echo "# TYPE jetson_emc_freq_mhz gauge"
        echo "# HELP jetson_temperature_celsius Temperature in Celsius"
        echo "# TYPE jetson_temperature_celsius gauge"
        echo "# HELP jetson_power_mw Power consumption in milliwatts"
        echo "# TYPE jetson_power_mw gauge"

        # RAM
        if [[ $line =~ RAM\ ([0-9]+)/([0-9]+)MB ]]; then
          echo "jetson_ram_used_mb ''${BASH_REMATCH[1]}"
          echo "jetson_ram_total_mb ''${BASH_REMATCH[2]}"
        fi

        # SWAP
        if [[ $line =~ SWAP\ ([0-9]+)/([0-9]+)MB ]]; then
          echo "jetson_swap_used_mb ''${BASH_REMATCH[1]}"
          echo "jetson_swap_total_mb ''${BASH_REMATCH[2]}"
        fi

        # CPU per-core (handles "off" cores gracefully)
        if [[ $line =~ CPU\ \[([^\]]+)\] ]]; then
          cpu_data="''${BASH_REMATCH[1]}"
          core_num=0
          IFS=',' read -ra CORES <<< "$cpu_data"
          for core in "''${CORES[@]}"; do
            if [[ $core =~ ([0-9]+)%@([0-9]+) ]]; then
              echo "jetson_cpu_usage_percent{core=\"$core_num\"} ''${BASH_REMATCH[1]}"
              echo "jetson_cpu_freq_mhz{core=\"$core_num\"} ''${BASH_REMATCH[2]}"
            fi
            core_num=$((core_num + 1))
          done
        fi

        # GPU — handles both "GR3D_FREQ 0%@1300" and "GR3D_FREQ 0%" formats
        if [[ $line =~ GR3D_FREQ\ ([0-9]+)%@([0-9]+) ]]; then
          echo "jetson_gpu_usage_percent ''${BASH_REMATCH[1]}"
          echo "jetson_gpu_freq_mhz ''${BASH_REMATCH[2]}"
        elif [[ $line =~ GR3D_FREQ\ ([0-9]+)% ]]; then
          echo "jetson_gpu_usage_percent ''${BASH_REMATCH[1]}"
        fi

        # EMC
        if [[ $line =~ EMC_FREQ\ ([0-9]+)%@([0-9]+) ]]; then
          echo "jetson_emc_freq_percent ''${BASH_REMATCH[1]}"
          echo "jetson_emc_freq_mhz ''${BASH_REMATCH[2]}"
        elif [[ $line =~ EMC_FREQ\ ([0-9]+)% ]]; then
          echo "jetson_emc_freq_percent ''${BASH_REMATCH[1]}"
        fi

        # Temperatures — match both upper and lowercase sensor names (cpu, soc0, tj, etc.)
        temp_line="$line"
        while [[ $temp_line =~ ([A-Za-z0-9_]+)@([0-9.-]+)C ]]; do
          sensor="''${BASH_REMATCH[1]}"
          temp="''${BASH_REMATCH[2]}"
          echo "jetson_temperature_celsius{sensor=\"$sensor\"} $temp"
          temp_line="''${temp_line/''${sensor}@''${temp}C/}"
        done

        # Power rails — match any VDD_*/VIN_* rail pattern: "NAME curr/avg" in mW
        power_line="$line"
        while [[ $power_line =~ (V[A-Z0-9_]+)\ ([0-9]+)mW/([0-9]+)mW ]]; do
          rail="''${BASH_REMATCH[1]}"
          curr="''${BASH_REMATCH[2]}"
          echo "jetson_power_mw{rail=\"$rail\"} $curr"
          power_line="''${power_line/''${rail} ''${curr}mW/}"
        done

      } > "$METRICS_FILE.tmp"

      mv -f "$METRICS_FILE.tmp" "$METRICS_FILE"
    done
  '';

  # Grafana dashboard JSON
  gpuDashboard = pkgs.writeText "jetson-gpu-dashboard.json" (builtins.toJSON {
    title = "Jetson Mission Control";
    tags = ["jetson" "nvidia" "gpu" "telemetry"];
    timezone = "browser";
    refresh = "2s";
    time = { from = "now-5m"; to = "now"; };
    style = "dark";
    panels = [
      # ── Row 0: GPU ──
      {
        id = 1; title = "GPU Load"; type = "gauge";
        gridPos = { h = 8; w = 6; x = 0; y = 0; };
        targets = [{ expr = "jetson_gpu_usage_percent"; legendFormat = "GPU %"; }];
        fieldConfig.defaults = {
          unit = "percent"; min = 0; max = 100;
          thresholds.mode = "absolute";
          thresholds.steps = [
            { value = null; color = "green"; }
            { value = 60; color = "yellow"; }
            { value = 85; color = "red"; }
          ];
        };
      }
      {
        id = 2; title = "GPU Utilization"; type = "timeseries";
        gridPos = { h = 8; w = 18; x = 6; y = 0; };
        targets = [{ expr = "jetson_gpu_usage_percent"; legendFormat = "GPU Usage %"; }];
        fieldConfig.defaults = {
          unit = "percent"; min = 0; max = 100;
          custom.fillOpacity = 20; custom.lineWidth = 2; custom.gradientMode = "scheme";
        };
      }
      # ── Row 1: CPU ──
      {
        id = 3; title = "CPU Usage by Core"; type = "timeseries";
        gridPos = { h = 8; w = 14; x = 0; y = 8; };
        targets = [{ expr = "jetson_cpu_usage_percent"; legendFormat = "Core {{core}}"; }];
        fieldConfig.defaults = {
          unit = "percent"; min = 0; max = 100;
          custom.fillOpacity = 15; custom.lineWidth = 1;
        };
      }
      {
        id = 4; title = "CPU Frequency"; type = "timeseries";
        gridPos = { h = 8; w = 10; x = 14; y = 8; };
        targets = [{ expr = "jetson_cpu_freq_mhz"; legendFormat = "Core {{core}}"; }];
        fieldConfig.defaults = {
          unit = "MHz";
          custom.fillOpacity = 10; custom.lineWidth = 1;
        };
      }
      # ── Row 2: Memory + EMC ──
      {
        id = 5; title = "RAM Used"; type = "gauge";
        gridPos = { h = 8; w = 4; x = 0; y = 16; };
        targets = [{ expr = "jetson_ram_used_mb / jetson_ram_total_mb * 100"; legendFormat = "RAM %"; }];
        fieldConfig.defaults = {
          unit = "percent"; min = 0; max = 100;
          thresholds.mode = "absolute";
          thresholds.steps = [
            { value = null; color = "green"; }
            { value = 70; color = "yellow"; }
            { value = 90; color = "red"; }
          ];
        };
      }
      {
        id = 6; title = "Memory (MB)"; type = "timeseries";
        gridPos = { h = 8; w = 8; x = 4; y = 16; };
        targets = [
          { expr = "jetson_ram_used_mb"; legendFormat = "RAM"; }
          { expr = "jetson_swap_used_mb"; legendFormat = "SWAP"; }
        ];
        fieldConfig.defaults = {
          unit = "mbytes";
          custom.fillOpacity = 10; custom.lineWidth = 2;
        };
      }
      {
        id = 7; title = "EMC Load"; type = "gauge";
        gridPos = { h = 8; w = 4; x = 12; y = 16; };
        targets = [{ expr = "jetson_emc_freq_percent"; legendFormat = "EMC %"; }];
        fieldConfig.defaults = {
          unit = "percent"; min = 0; max = 100;
          thresholds.mode = "absolute";
          thresholds.steps = [
            { value = null; color = "green"; }
            { value = 60; color = "yellow"; }
            { value = 85; color = "red"; }
          ];
        };
      }
      {
        id = 8; title = "Memory Controller"; type = "timeseries";
        gridPos = { h = 8; w = 8; x = 16; y = 16; };
        targets = [{ expr = "jetson_emc_freq_mhz"; legendFormat = "EMC MHz"; }];
        fieldConfig.defaults = {
          unit = "MHz";
          custom.fillOpacity = 10; custom.lineWidth = 2;
        };
      }
      # ── Row 3: Thermals + Power ──
      {
        id = 9; title = "Temperatures"; type = "timeseries";
        gridPos = { h = 8; w = 12; x = 0; y = 24; };
        targets = [{ expr = "jetson_temperature_celsius"; legendFormat = "{{sensor}}"; }];
        fieldConfig.defaults = {
          unit = "celsius"; custom.fillOpacity = 5;
          thresholds.mode = "absolute";
          thresholds.steps = [
            { value = null; color = "green"; }
            { value = 70; color = "yellow"; }
            { value = 85; color = "red"; }
          ];
        };
      }
      {
        id = 10; title = "Power Consumption"; type = "timeseries";
        gridPos = { h = 8; w = 8; x = 12; y = 24; };
        targets = [{ expr = "jetson_power_mw"; legendFormat = "{{rail}}"; }];
        fieldConfig.defaults = {
          unit = "mwatt"; custom.fillOpacity = 15; custom.gradientMode = "scheme";
        };
      }
      {
        id = 11; title = "Total Power"; type = "stat";
        gridPos = { h = 8; w = 4; x = 20; y = 24; };
        targets = [{ expr = "sum(jetson_power_mw)"; legendFormat = "Total"; }];
        fieldConfig.defaults = {
          unit = "mwatt";
          thresholds.mode = "absolute";
          thresholds.steps = [
            { value = null; color = "green"; }
            { value = 15000; color = "yellow"; }
            { value = 30000; color = "red"; }
          ];
        };
      }
    ];
  });

  # Node Exporter dashboard JSON
  nodeExporterDashboard = pkgs.writeText "jetson-node-exporter-dashboard.json" (builtins.toJSON {
    title = "Jetson System Metrics";
    tags = ["jetson" "system" "node-exporter"];
    timezone = "browser";
    refresh = "5s";
    time = { from = "now-15m"; to = "now"; };
    style = "dark";
    panels = [
      # ── Row 0: Overview ──
      {
        id = 1; title = "CPU Usage"; type = "gauge";
        gridPos = { h = 6; w = 6; x = 0; y = 0; };
        targets = [{
          expr = "100 - (avg(rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)";
          legendFormat = "CPU %";
        }];
        fieldConfig.defaults = {
          unit = "percent"; min = 0; max = 100;
          thresholds.mode = "absolute";
          thresholds.steps = [
            { value = null; color = "green"; }
            { value = 60; color = "yellow"; }
            { value = 85; color = "red"; }
          ];
        };
      }
      {
        id = 2; title = "Load Average"; type = "timeseries";
        gridPos = { h = 6; w = 6; x = 6; y = 0; };
        targets = [
          { expr = "node_load1"; legendFormat = "1m"; }
          { expr = "node_load5"; legendFormat = "5m"; }
          { expr = "node_load15"; legendFormat = "15m"; }
        ];
        fieldConfig.defaults = {
          custom.fillOpacity = 10; custom.lineWidth = 2;
        };
      }
      {
        id = 3; title = "Memory Used"; type = "gauge";
        gridPos = { h = 6; w = 6; x = 12; y = 0; };
        targets = [{
          expr = "(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100";
          legendFormat = "Memory %";
        }];
        fieldConfig.defaults = {
          unit = "percent"; min = 0; max = 100;
          thresholds.mode = "absolute";
          thresholds.steps = [
            { value = null; color = "green"; }
            { value = 70; color = "yellow"; }
            { value = 90; color = "red"; }
          ];
        };
      }
      {
        id = 4; title = "Uptime"; type = "stat";
        gridPos = { h = 6; w = 6; x = 18; y = 0; };
        targets = [{
          expr = "time() - node_boot_time_seconds";
          legendFormat = "Uptime";
        }];
        fieldConfig.defaults = {
          unit = "s";
          thresholds.mode = "absolute";
          thresholds.steps = [{ value = null; color = "blue"; }];
        };
      }
      # ── Row 1: CPU Detailed ──
      {
        id = 5; title = "CPU Usage by Mode"; type = "timeseries";
        gridPos = { h = 8; w = 24; x = 0; y = 6; };
        targets = [
          { expr = "avg(rate(node_cpu_seconds_total{mode=\"user\"}[5m])) * 100"; legendFormat = "user"; }
          { expr = "avg(rate(node_cpu_seconds_total{mode=\"system\"}[5m])) * 100"; legendFormat = "system"; }
          { expr = "avg(rate(node_cpu_seconds_total{mode=\"iowait\"}[5m])) * 100"; legendFormat = "iowait"; }
          { expr = "avg(rate(node_cpu_seconds_total{mode=\"irq\"}[5m])) * 100"; legendFormat = "irq"; }
          { expr = "avg(rate(node_cpu_seconds_total{mode=\"softirq\"}[5m])) * 100"; legendFormat = "softirq"; }
        ];
        fieldConfig.defaults = {
          unit = "percent"; min = 0;
          custom.fillOpacity = 30; custom.lineWidth = 1; custom.stacking.mode = "normal";
        };
      }
      # ── Row 2: Memory + Filesystem ──
      {
        id = 6; title = "Memory Usage"; type = "timeseries";
        gridPos = { h = 8; w = 12; x = 0; y = 14; };
        targets = [
          { expr = "node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes"; legendFormat = "Used"; }
          { expr = "node_memory_Cached_bytes"; legendFormat = "Cached"; }
          { expr = "node_memory_Buffers_bytes"; legendFormat = "Buffers"; }
          { expr = "node_memory_MemAvailable_bytes"; legendFormat = "Available"; }
        ];
        fieldConfig.defaults = {
          unit = "bytes";
          custom.fillOpacity = 20; custom.lineWidth = 1;
        };
      }
      {
        id = 7; title = "Filesystem Usage"; type = "bargauge";
        gridPos = { h = 8; w = 12; x = 12; y = 14; };
        targets = [{
          expr = "100 - (node_filesystem_avail_bytes{fstype!~\"tmpfs|overlay|squashfs\"} / node_filesystem_size_bytes{fstype!~\"tmpfs|overlay|squashfs\"} * 100)";
          legendFormat = "{{mountpoint}}";
        }];
        fieldConfig.defaults = {
          unit = "percent"; min = 0; max = 100;
          thresholds.mode = "absolute";
          thresholds.steps = [
            { value = null; color = "green"; }
            { value = 70; color = "yellow"; }
            { value = 90; color = "red"; }
          ];
        };
        options.orientation = "horizontal";
      }
      # ── Row 3: Network ──
      {
        id = 8; title = "Network Receive"; type = "timeseries";
        gridPos = { h = 8; w = 12; x = 0; y = 22; };
        targets = [{
          expr = "rate(node_network_receive_bytes_total{device!=\"lo\"}[5m])";
          legendFormat = "{{device}} RX";
        }];
        fieldConfig.defaults = {
          unit = "Bps";
          custom.fillOpacity = 15; custom.lineWidth = 2; custom.gradientMode = "scheme";
        };
      }
      {
        id = 9; title = "Network Transmit"; type = "timeseries";
        gridPos = { h = 8; w = 12; x = 12; y = 22; };
        targets = [{
          expr = "rate(node_network_transmit_bytes_total{device!=\"lo\"}[5m])";
          legendFormat = "{{device}} TX";
        }];
        fieldConfig.defaults = {
          unit = "Bps";
          custom.fillOpacity = 15; custom.lineWidth = 2; custom.gradientMode = "scheme";
        };
      }
      # ── Row 4: Disk I/O ──
      {
        id = 10; title = "Disk Read"; type = "timeseries";
        gridPos = { h = 8; w = 12; x = 0; y = 30; };
        targets = [{
          expr = "rate(node_disk_read_bytes_total[5m])";
          legendFormat = "{{device}}";
        }];
        fieldConfig.defaults = {
          unit = "Bps";
          custom.fillOpacity = 15; custom.lineWidth = 2;
        };
      }
      {
        id = 11; title = "Disk Write"; type = "timeseries";
        gridPos = { h = 8; w = 12; x = 12; y = 30; };
        targets = [{
          expr = "rate(node_disk_written_bytes_total[5m])";
          legendFormat = "{{device}}";
        }];
        fieldConfig.defaults = {
          unit = "Bps";
          custom.fillOpacity = 15; custom.lineWidth = 2;
        };
      }
      # ── Row 5: Thermal ──
      {
        id = 12; title = "Thermal Zones"; type = "timeseries";
        gridPos = { h = 8; w = 24; x = 0; y = 38; };
        targets = [{
          expr = "node_thermal_zone_temp";
          legendFormat = "{{type}} zone{{zone}}";
        }];
        fieldConfig.defaults = {
          unit = "celsius"; custom.fillOpacity = 5; custom.lineWidth = 2;
          thresholds.mode = "absolute";
          thresholds.steps = [
            { value = null; color = "green"; }
            { value = 70; color = "yellow"; }
            { value = 85; color = "red"; }
          ];
        };
      }
    ];
  });

  # OpenTelemetry Collector dashboard JSON
  otelDashboard = pkgs.writeText "jetson-otel-dashboard.json" (builtins.toJSON {
    title = "OpenTelemetry Collector";
    tags = ["jetson" "opentelemetry" "otel"];
    timezone = "browser";
    refresh = "10s";
    time = { from = "now-15m"; to = "now"; };
    style = "dark";
    panels = [
      # ── Row 0: Collector Health ──
      {
        id = 1; title = "Collector Uptime"; type = "stat";
        gridPos = { h = 6; w = 8; x = 0; y = 0; };
        targets = [{
          expr = "otelcol_process_uptime";
          legendFormat = "Uptime";
        }];
        fieldConfig.defaults = {
          unit = "s";
          thresholds.mode = "absolute";
          thresholds.steps = [{ value = null; color = "green"; }];
        };
      }
      {
        id = 2; title = "Collector CPU"; type = "timeseries";
        gridPos = { h = 6; w = 8; x = 8; y = 0; };
        targets = [{
          expr = "rate(otelcol_process_cpu_seconds[5m])";
          legendFormat = "CPU";
        }];
        fieldConfig.defaults = {
          unit = "percentunit";
          custom.fillOpacity = 20; custom.lineWidth = 2;
        };
      }
      {
        id = 3; title = "Collector Memory (RSS)"; type = "timeseries";
        gridPos = { h = 6; w = 8; x = 16; y = 0; };
        targets = [{
          expr = "otelcol_process_memory_rss";
          legendFormat = "RSS";
        }];
        fieldConfig.defaults = {
          unit = "bytes";
          custom.fillOpacity = 20; custom.lineWidth = 2;
        };
      }
      # ── Row 1: Metrics Pipeline ──
      {
        id = 4; title = "Metrics Received"; type = "timeseries";
        gridPos = { h = 8; w = 12; x = 0; y = 6; };
        targets = [
          {
            expr = "rate(otelcol_receiver_accepted_metric_points[5m])";
            legendFormat = "{{receiver}} accepted";
          }
          {
            expr = "rate(otelcol_receiver_refused_metric_points[5m])";
            legendFormat = "{{receiver}} refused";
          }
        ];
        fieldConfig.defaults = {
          unit = "cps";
          custom.fillOpacity = 15; custom.lineWidth = 2;
        };
      }
      {
        id = 5; title = "Metrics Exported"; type = "timeseries";
        gridPos = { h = 8; w = 12; x = 12; y = 6; };
        targets = [
          {
            expr = "rate(otelcol_exporter_sent_metric_points[5m])";
            legendFormat = "{{exporter}} sent";
          }
          {
            expr = "rate(otelcol_exporter_send_failed_metric_points[5m])";
            legendFormat = "{{exporter}} failed";
          }
        ];
        fieldConfig.defaults = {
          unit = "cps";
          custom.fillOpacity = 15; custom.lineWidth = 2;
        };
      }
      # ── Row 2: Application Metrics ──
      {
        id = 6; title = "Application Metrics (via OTLP)"; type = "timeseries";
        gridPos = { h = 8; w = 24; x = 0; y = 14; };
        targets = [{
          expr = "{job=\"otel-collector\", __name__!~\"otelcol_.*|up|scrape_.*\"}";
          legendFormat = "{{__name__}}";
        }];
        fieldConfig.defaults = {
          custom.fillOpacity = 10; custom.lineWidth = 2;
        };
      }
    ];
  });

in
{
  options.services.jetson-telemetry = {
    enable = mkEnableOption "Jetson GPU/system telemetry (Grafana + Prometheus)";

    port = mkOption {
      type = types.port;
      default = 3301;
      description = "Grafana web UI port.";
    };

    dataDir = mkOption {
      type = types.path;
      default = "/var/lib/jetson-telemetry";
      description = "Data directory for metrics storage.";
    };

    enableTegrastats = mkOption {
      type = types.bool;
      default = true;
      description = "Enable tegrastats GPU exporter (500ms sampling).";
    };

    enableNodeExporter = mkOption {
      type = types.bool;
      default = true;
      description = "Enable Prometheus node exporter for system metrics.";
    };

    retentionTime = mkOption {
      type = types.str;
      default = "30d";
      description = "Prometheus data retention period.";
    };

    enableOpenTelemetry = mkOption {
      type = types.bool;
      default = false;
      description = "Enable OpenTelemetry collector (OTLP -> Prometheus).";
    };
  };

  config = mkIf cfg.enable {
    environment.systemPackages = with pkgs;
      [ pkgs.nvidia-jetpack.l4t-tools pkgs.python3 ]
      ++ lib.optionals (pkgs.nvidia-jetpack.l4tAtLeast "36")
        [ pkgs.nvidia-jetpack.nvidia-smi ];

    systemd.tmpfiles.rules = [
      "d ${cfg.dataDir} 0755 root root -"
      "d ${cfg.dataDir}/metrics 0755 root root -"
      "d ${cfg.dataDir}/prometheus 0755 root root -"
      "d ${cfg.dataDir}/grafana 0755 root root -"
      "d ${cfg.dataDir}/grafana/dashboards 0755 root root -"
      "L+ ${cfg.dataDir}/grafana/dashboards/jetson-gpu.json - - - - ${gpuDashboard}"
      "L+ ${cfg.dataDir}/grafana/dashboards/jetson-system.json - - - - ${nodeExporterDashboard}"
      "L+ ${cfg.dataDir}/grafana/dashboards/jetson-otel.json - - - - ${otelDashboard}"
    ];

    # Tegrastats exporter
    systemd.services.tegrastats-exporter = mkIf cfg.enableTegrastats {
      description = "Tegrastats GPU metrics exporter";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];
      serviceConfig = {
        Type = "simple";
        ExecStart = "${tegrastatsExporter}/bin/tegrastats-exporter";
        Restart = "on-failure";
        RestartSec = "10s";
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ cfg.dataDir ];
      };
    };

    # HTTP server exposing .prom files for Prometheus scraping
    systemd.services.tegrastats-http-server = mkIf cfg.enableTegrastats {
      description = "HTTP server for tegrastats Prometheus metrics";
      wantedBy = [ "multi-user.target" ];
      after = [ "tegrastats-exporter.service" ];
      requires = [ "tegrastats-exporter.service" ];
      serviceConfig = {
        Type = "simple";
        ExecStart = "${pkgs.python3}/bin/python3 -m http.server 9101 --directory ${cfg.dataDir}/metrics --bind 0.0.0.0";
        WorkingDirectory = "${cfg.dataDir}/metrics";
        Restart = "on-failure";
        RestartSec = "5s";
        ReadOnlyPaths = [ "${cfg.dataDir}/metrics" ];
        ProtectSystem = "strict";
        ProtectHome = true;
      };
    };

    # Node exporter
    services.prometheus.exporters.node = mkIf cfg.enableNodeExporter {
      enable = true;
      port = 9100;
      enabledCollectors = [
        "cpu" "loadavg" "meminfo" "diskstats"
        "filesystem" "netdev" "thermal_zone"
      ];
    };

    # Prometheus
    services.prometheus = {
      enable = true;
      port = 9090;
      retentionTime = cfg.retentionTime;
      scrapeConfigs = [
        {
          job_name = "jetson-tegrastats";
          scrape_interval = "5s";
          metrics_path = "/tegrastats-metrics.prom.txt";
          static_configs = [{
            targets = [ "localhost:9101" ];
            labels = { device = "jetson"; source = "tegrastats"; };
          }];
        }
        {
          job_name = "jetson-system";
          scrape_interval = "15s";
          static_configs = [{
            targets = [ "localhost:9100" ];
            labels = { device = "jetson"; source = "node-exporter"; };
          }];
        }
      ] ++ lib.optionals cfg.enableOpenTelemetry [
        {
          job_name = "otel-collector";
          scrape_interval = "5s";
          static_configs = [{
            targets = [ "localhost:8889" ];
            labels = { device = "jetson"; source = "opentelemetry"; };
          }];
        }
      ];
    };

    # OpenTelemetry collector (opt-in)
    systemd.services.otel-collector = mkIf cfg.enableOpenTelemetry (let
      otelConfig = pkgs.writeText "otel-config.yaml" ''
        receivers:
          otlp:
            protocols:
              grpc:
                endpoint: "0.0.0.0:4317"
              http:
                endpoint: "0.0.0.0:4318"
        exporters:
          prometheus:
            endpoint: "0.0.0.0:8889"
            namespace: "otel"
        telemetry:
          metrics:
            address: "0.0.0.0:8889"
            prometheus:
              enable: true
        service:
          pipelines:
            metrics:
              receivers: [otlp]
              exporters: [prometheus]
      '';
    in {
      description = "OpenTelemetry Collector";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];
      serviceConfig = {
        Type = "simple";
        ExecStart = "${pkgs.opentelemetry-collector-contrib}/bin/otelcol-contrib --config=${otelConfig}";
        Restart = "on-failure";
        RestartSec = "5s";
      };
    });

    # Grafana
    services.grafana = {
      enable = true;
      settings = {
        server = {
          http_addr = "0.0.0.0";
          http_port = cfg.port;
          domain = "localhost";
        };
        "auth.anonymous" = {
          enabled = true;
          org_role = "Viewer";
        };
        analytics.reporting_enabled = false;
      };
      provision = {
        enable = true;
        datasources.settings.datasources = [{
          name = "Prometheus";
          type = "prometheus";
          access = "proxy";
          url = "http://localhost:9090";
          isDefault = true;
          editable = false;
        }];
        dashboards.settings = {
          apiVersion = 1;
          providers = [{
            name = "Jetson Telemetry";
            folder = "Jetson";
            type = "file";
            disableDeletion = false;
            updateIntervalSeconds = 30;
            allowUiUpdates = true;
            options.path = "${cfg.dataDir}/grafana/dashboards";
          }];
        };
      };
    };

    # Firewall
    networking.firewall.allowedTCPPorts = [
      cfg.port 9090 9100 9101
    ] ++ lib.optionals cfg.enableOpenTelemetry [ 4317 4318 8889 ];

    # Activation message
    system.activationScripts.jetson-telemetry-info = lib.mkIf cfg.enable ''
      echo ""
      echo "=== JETSON TELEMETRY ACTIVE ==="
      echo "  Grafana:    http://localhost:${toString cfg.port}"
      echo "  Prometheus: http://localhost:9090"
      echo "  SSH tunnel: ssh -L ${toString cfg.port}:localhost:${toString cfg.port} -L 9090:localhost:9090 user@<jetson-ip>"
      echo ""
    '';
  };
}
