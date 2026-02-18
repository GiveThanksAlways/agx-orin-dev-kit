# tabby-api.nix — NixOS module for TabbyAPI on Jetson Orin
#
# ExLlamaV2-based OpenAI-compatible server, great for IDE / OpenCode
# code completion workloads.
#
#   imports = [ ./modules/tabby-api.nix ];
#   services.tabby-api = {
#     enable    = true;
#     modelDir  = "/models";
#     modelName = "Qwen3-Coder-Next-Q4_K_M.gguf";
#   };

{ config, lib, pkgs, ... }:

let
  cfg = config.services.tabby-api;
in
{
  options.services.tabby-api = {
    enable = lib.mkEnableOption "TabbyAPI for code completion serving on Jetson Orin";

    modelDir = lib.mkOption {
      type = lib.types.str;
      description = "Directory containing GGUF/EXL2 models.";
      example = "/models";
    };

    modelName = lib.mkOption {
      type = lib.types.str;
      description = "Model directory name or GGUF filename within modelDir.";
      example = "Qwen3-Coder-Next-Q4_K_M.gguf";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "0.0.0.0";
      description = "Host to bind to.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 5000;
      description = "Port for the OpenAI-compatible API.";
    };

    maxSeqLen = lib.mkOption {
      type = lib.types.int;
      default = 4096;
      description = "Maximum sequence length.";
    };

    cacheMode = lib.mkOption {
      type = lib.types.enum [ "FP16" "Q8" "Q6" "Q4" ];
      default = "Q4";
      description = "KV cache quantization. Q4 saves most VRAM on Orin.";
    };

    extraConfig = lib.mkOption {
      type = lib.types.attrs;
      default = { };
      description = "Extra config options merged into config.yml.";
    };

    src = lib.mkOption {
      type = lib.types.str;
      default = "https://github.com/theroyallab/tabbyAPI";
      description = "TabbyAPI source repository URL.";
    };
  };

  config = lib.mkIf cfg.enable {
    environment.etc."tabby-api/config.json".text = builtins.toJSON ({
      model = {
        model_dir = cfg.modelDir;
        model_name = cfg.modelName;
        max_seq_len = cfg.maxSeqLen;
        cache_mode = cfg.cacheMode;
        gpu_split_auto = true;
      };
      network = {
        host = cfg.host;
        port = cfg.port;
      };
      developer = {
        unsafe_launch = true;
      };
    } // cfg.extraConfig);

    systemd.services.tabby-api = {
      description = "TabbyAPI - ExLlamaV2 OpenAI-compatible Server";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];

      environment = {
        CUDA_VISIBLE_DEVICES = "0";
      };

      serviceConfig = {
        Type = "simple";
        Restart = "on-failure";
        RestartSec = 10;

        StateDirectory = "tabby-api";
        WorkingDirectory = "/var/lib/tabby-api";

        ExecStartPre = pkgs.writeShellScript "tabby-api-setup" ''
          if [ ! -d /var/lib/tabby-api/repo ]; then
            ${pkgs.git}/bin/git clone --depth 1 ${cfg.src} /var/lib/tabby-api/repo
          fi
          cd /var/lib/tabby-api/repo
          ${pkgs.git}/bin/git pull --ff-only || true
        '';

        ExecStart = ''
          ${pkgs.python3}/bin/python /var/lib/tabby-api/repo/main.py \
            --config /etc/tabby-api/config.json
        '';
      };
    };

    networking.firewall.allowedTCPPorts = [ cfg.port ];
  };
}
