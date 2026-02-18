# llama-cpp-server.nix — NixOS module for llama.cpp server on Jetson Orin
#
# Lowest overhead GGUF-native serving with an OpenAI-compatible API.
#
#   imports = [ ./modules/llama-cpp-server.nix ];
#   services.llama-cpp-server = {
#     enable = true;
#     model  = "/models/my-model.gguf";
#   };

{ config, lib, pkgs, ... }:

let
  cfg = config.services.llama-cpp-server;
in
{
  options.services.llama-cpp-server = {
    enable = lib.mkEnableOption "llama.cpp server for GGUF models on Jetson Orin";

    model = lib.mkOption {
      type = lib.types.str;
      description = "Path to GGUF model file.";
      example = "/models/Qwen3-Coder-Next-Q4_K_M.gguf";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "0.0.0.0";
      description = "Host to bind to.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8080;
      description = "Port for the OpenAI-compatible API.";
    };

    nGpuLayers = lib.mkOption {
      type = lib.types.int;
      default = 99;
      description = "Number of layers to offload to GPU. 99 = all layers.";
    };

    contextSize = lib.mkOption {
      type = lib.types.int;
      default = 4096;
      description = "Context window size in tokens.";
    };

    threads = lib.mkOption {
      type = lib.types.int;
      default = 8;
      description = "Number of CPU threads (Orin AGX has 12 cores).";
    };

    batchSize = lib.mkOption {
      type = lib.types.int;
      default = 512;
      description = "Batch size for prompt processing.";
    };

    useMmap = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Use mmap for faster model loading.";
    };

    flashAttn = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Enable flash attention (supported on Orin SM 8.7).";
    };

    extraArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = "Extra CLI arguments for llama-server.";
    };

    package = lib.mkOption {
      type = lib.types.package;
      default =
        if pkgs ? llama-cpp then pkgs.llama-cpp
        else if pkgs ? llamaCpp then pkgs.llamaCpp
        else throw "llama-cpp package not found. Add it to your overlay or install via nixpkgs.";
      description = "llama.cpp package to use (must include llama-server).";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.llama-cpp-server = {
      description = "llama.cpp OpenAI-compatible Server";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];

      environment = {
        CUDA_VISIBLE_DEVICES = "0";
      };

      serviceConfig = {
        Type = "simple";
        Restart = "on-failure";
        RestartSec = 5;

        DynamicUser = true;
        StateDirectory = "llama-cpp";

        ExecStart =
          let
            args = [
              "${cfg.package}/bin/llama-server"
              "--model"
              cfg.model
              "--host"
              cfg.host
              "--port"
              (toString cfg.port)
              "--n-gpu-layers"
              (toString cfg.nGpuLayers)
              "--ctx-size"
              (toString cfg.contextSize)
              "--threads"
              (toString cfg.threads)
              "--batch-size"
              (toString cfg.batchSize)
            ]
            ++ lib.optionals cfg.useMmap [ "--use-mmap" ]
            ++ lib.optionals cfg.flashAttn [ "--flash-attn" ]
            ++ cfg.extraArgs;
          in
          lib.concatStringsSep " " args;
      };
    };

    networking.firewall.allowedTCPPorts = [ cfg.port ];
  };
}
