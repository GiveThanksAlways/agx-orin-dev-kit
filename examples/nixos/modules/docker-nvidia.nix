# Docker + NVIDIA Container Toolkit for Jetson Orin AGX
#
# Enables Docker with GPU passthrough via jetpack-nixos's
# nvidia-container-toolkit module.  This is the minimal config
# required to run GPU-accelerated containers (vLLM, MLC LLM, etc.)
#
# Usage in your NixOS config:
#   imports = [ ./modules/docker-nvidia.nix ];
#   services.docker-nvidia.enable = true;
#
# Then:
#   docker run --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all <image> nvidia-smi

{ config, lib, pkgs, ... }:

let
  cfg = config.services.docker-nvidia;
in
{
  options.services.docker-nvidia = {
    enable = lib.mkEnableOption "Docker with NVIDIA Container Toolkit for Jetson";

    users = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ "agent" "spencer" ];
      description = "Users to add to the docker group (avoids needing sudo for docker commands).";
    };

    storageDriver = lib.mkOption {
      type = lib.types.str;
      default = "overlay2";
      description = "Docker storage driver.";
    };
  };

  config = lib.mkIf cfg.enable {
    # Enable Docker daemon
    virtualisation.docker = {
      enable = true;
      enableNvidia = true;            # triggers jetpack-nixos nvidia-container-toolkit
      storageDriver = cfg.storageDriver;

      # Auto-prune old images weekly to save disk
      autoPrune = {
        enable = true;
        dates = "weekly";
      };
    };

    # Add specified users to docker group
    users.users = lib.listToAttrs (map (user: {
      name = user;
      value = { extraGroups = [ "docker" ]; };
    }) cfg.users);

    # Open no extra ports — containers bind to host ports as needed

    # Helpful packages available system-wide when Docker is on
    environment.systemPackages = with pkgs; [
      docker-compose
    ];
  };
}
