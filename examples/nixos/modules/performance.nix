# performance.nix — Orin AGX performance tuning module
#
# Composable module for maximizing inference / compute performance
# on Jetson Orin AGX.  Import it and flip the switch:
#
#   imports = [ ./modules/performance.nix ];
#   services.orin-perf.enable = true;

{ config, lib, pkgs, ... }:

let
  cfg = config.services.orin-perf;
in
{
  options.services.orin-perf = {
    enable = lib.mkEnableOption "Orin AGX performance tuning";

    powerMode = lib.mkOption {
      type = lib.types.enum [ "MAXN" "30W" "50W" ];
      default = "MAXN";
      description = "NVPModel power mode. MAXN = max performance.";
    };

    lockClocks = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Lock GPU/CPU clocks to max (jetson_clocks).";
    };

    hugepages = lib.mkOption {
      type = lib.types.int;
      default = 4096;
      description = "Number of 2MB hugepages to allocate.";
    };

    zramSwap = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Enable zram swap for memory pressure during model loading.";
    };
  };

  config = lib.mkIf cfg.enable {
    # Set NVPModel power mode at boot
    systemd.services.orin-power-mode = {
      description = "Set Orin AGX power mode";
      wantedBy = [ "multi-user.target" ];
      after = [ "multi-user.target" ];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStart =
          let
            modeId = {
              "MAXN" = "0";
              "30W" = "1";
              "50W" = "2";
            }.${cfg.powerMode};
          in
          pkgs.writeShellScript "set-power-mode" ''
            export PATH="/usr/sbin:/usr/bin:$PATH"
            nvpmodel -m ${modeId} || true
            ${lib.optionalString cfg.lockClocks ''
              sleep 2
              jetson_clocks || true
            ''}
          '';
      };
    };

    # Kernel params for max performance
    boot.kernel.sysctl = {
      "vm.swappiness" = 10;
      "vm.overcommit_memory" = 1;
      "vm.max_map_count" = 1048576;
    };

    boot.kernelParams = [
      "hugepagesz=2M"
      "hugepages=${toString cfg.hugepages}"
      "transparent_hugepage=always"
    ];

    # ZRAM swap
    zramSwap = lib.mkIf cfg.zramSwap {
      enable = true;
      memoryPercent = 50;
      algorithm = "zstd";
    };

    # tmpfs scratch space
    fileSystems."/tmp/llm-scratch" = {
      device = "tmpfs";
      fsType = "tmpfs";
      options = [ "size=4G" "mode=1777" ];
    };
  };
}
