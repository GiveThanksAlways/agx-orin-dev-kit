# Pill 09: Building Your Own Jetson System

## tl;dr

This is the practical wrap-up. You now know every layer — from Nix expressions to overlays to modules to flash scripts. This pill puts it all together: how to go from zero to a running NixOS system on your Jetson, how to customize it, how to add services, and how to maintain it. This is your reference for actually doing the work.

## Prerequisites

1. **Jetson dev kit** (Orin AGX, Orin NX, Orin Nano, Xavier, or Thor)
2. **x86_64 Linux machine** for initial flash (USB-A to USB-C cable)
3. **Nix installed** with flakes enabled on the x86_64 machine
4. [Optional] **NVMe SSD** for rootfs (recommended over eMMC)

## Step 1: Create Your Flake

```nix
# flake.nix
{
  description = "My Jetson NixOS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    jetpack.url = "github:anduril/jetpack-nixos/master";
    jetpack.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, jetpack, ... }: {
    nixosConfigurations.my-jetson = nixpkgs.lib.nixosSystem {
      modules = [
        jetpack.nixosModules.default
        ./configuration.nix
      ];
    };
  };
}
```

### What `follows` Does

```nix
jetpack.inputs.nixpkgs.follows = "nixpkgs";
```

This forces jetpack-nixos to use **your** nixpkgs instead of its own pinned version. One nixpkgs, one eval, consistent packages.

## Step 2: Write Your Configuration

```nix
# configuration.nix
{ config, lib, pkgs, ... }: {
  # ──── Hardware ────
  hardware.nvidia-jetpack.enable = true;
  hardware.nvidia-jetpack.som = "orin-agx";       # Your SoM
  hardware.nvidia-jetpack.carrierBoard = "devkit"; # Your carrier board
  hardware.nvidia-jetpack.configureCuda = true;
  hardware.graphics.enable = true;

  # ──── Boot ────
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;
  boot.kernelParams = [ "console=ttyTCU0,115200" ];

  # ──── OTA firmware updates ────
  hardware.nvidia-jetpack.firmware.autoUpdate = true;

  # ──── Network ────
  networking.hostName = "jetson";
  networking.networkmanager.enable = true;
  # OR for static IP:
  # networking.interfaces.eth0.ipv4.addresses = [{
  #   address = "192.168.1.100"; prefixLength = 24;
  # }];

  # ──── Users ────
  users.users.myuser = {
    isNormalUser = true;
    extraGroups = [ "wheel" "video" "render" ];
    initialPassword = "changeme";
    openssh.authorizedKeys.keys = [
      "ssh-ed25519 AAAA..."
    ];
  };

  # ──── Services ────
  services.openssh.enable = true;
  nix.settings.experimental-features = [ "nix-command" "flakes" ];

  # ──── Packages ────
  environment.systemPackages = with pkgs; [
    vim git htop
  ];

  # ──── Firewall ────
  networking.firewall.allowedTCPPorts = [ 22 ];

  system.stateVersion = "25.11";
}
```

### Choosing Your SoM

| `som` value | Device | SoC |
|-------------|--------|-----|
| `"orin-agx"` | Orin AGX 32GB/64GB | T234 |
| `"orin-agx-industrial"` | Orin AGX Industrial | T234 |
| `"orin-nx"` | Orin NX 8GB/16GB | T234 |
| `"orin-nano"` | Orin Nano 4GB/8GB | T234 |
| `"xavier-agx"` | Xavier AGX | T194 |
| `"xavier-agx-industrial"` | Xavier AGX Industrial | T194 |
| `"xavier-nx"` | Xavier NX (SD) | T194 |
| `"xavier-nx-emmc"` | Xavier NX (eMMC) | T194 |
| `"thor-agx"` | Thor AGX | T264 |

### Choosing Your JetPack Version

```nix
# Default is JP6. To use JP5 or JP7:
hardware.nvidia-jetpack.majorVersion = "7";  # JP7: L4T R38, CUDA 13.0
hardware.nvidia-jetpack.majorVersion = "6";  # JP6: L4T R36, CUDA 12.6 (default)
hardware.nvidia-jetpack.majorVersion = "5";  # JP5: L4T R35, CUDA 11.4
```

## Step 3: Hardware Configuration

Generate or write your `hardware-configuration.nix`:

```nix
# hardware-configuration.nix
{ config, lib, pkgs, modulesPath, ... }: {
  imports = [
    (modulesPath + "/installer/scan/not-detected.nix")
  ];

  # Root filesystem
  fileSystems."/" = {
    device = "/dev/disk/by-label/nixos";
    fsType = "ext4";
  };

  # EFI System Partition
  fileSystems."/boot" = {
    device = "/dev/disk/by-label/ESP";
    fsType = "vfat";
  };

  # Swap (optional, Orin AGX 64GB may not need it)
  # swapDevices = [{ device = "/dev/disk/by-label/swap"; }];

  nixpkgs.hostPlatform = lib.mkDefault "aarch64-linux";
}
```

## Step 4: Initial Flash

Put the Jetson in **recovery mode**:
1. Power off the device
2. Hold the **REC** (recovery) button
3. Press **RESET** (or plug in power)
4. Release REC after 2 seconds
5. Connect USB-C from Jetson to x86_64 host

From the x86_64 host:

```bash
# Build and run the flash script
nix build .#nixosConfigurations.my-jetson.config.system.build.flashScript
sudo ./result/bin/initrd-flash-jetson

# Or more directly:
nix run .#nixosConfigurations.my-jetson.config.system.build.flashScript
```

This:
1. Boots a temporary Linux on the Jetson via USB (RCM)
2. Programs QSPI firmware (UEFI, OP-TEE, device trees)
3. Partitions and writes the rootfs
4. Reboots into NixOS

**First flash takes 10-30 minutes** depending on your setup.

## Step 5: Normal Updates

After the initial flash, updates are just:

```bash
# SSH into the Jetson, or from the device directly:
sudo nixos-rebuild switch --flake /path/to/your/flake#my-jetson
```

If `firmware.autoUpdate = true`, firmware updates happen automatically via UEFI capsules.

## Multiple Configurations

The power of flakes — define multiple system variants:

```nix
outputs = { self, nixpkgs, jetpack, ... }:
  let
    baseModules = [
      jetpack.nixosModules.default
      ./configuration.nix
    ];
  in {
    nixosConfigurations = {
      # Base system
      nixos = nixpkgs.lib.nixosSystem {
        modules = baseModules;
      };

      # Performance-tuned
      nixos-perf = nixpkgs.lib.nixosSystem {
        modules = baseModules ++ [
          ./modules/performance.nix
          { services.orin-perf.enable = true; }
        ];
      };

      # With llama.cpp server
      nixos-llama = nixpkgs.lib.nixosSystem {
        modules = baseModules ++ [
          ./modules/performance.nix
          ./modules/llama-cpp-server.nix
          {
            services.orin-perf.enable = true;
            services.llama-cpp-server = {
              enable = true;
              model = "/models/my-model.gguf";
              nGpuLayers = 99;
              contextSize = 4096;
            };
          }
        ];
      };

      # With telemetry
      nixos-telemetry = nixpkgs.lib.nixosSystem {
        modules = baseModules ++ [
          ./modules/telemetry.nix
          {
            services.jetson-telemetry = {
              enable = true;
              enableTegrastats = true;
              enableNodeExporter = true;
            };
          }
        ];
      };
    };
  };
```

Switch between them:

```bash
sudo nixos-rebuild switch --flake .#nixos-perf
sudo nixos-rebuild switch --flake .#nixos-llama
sudo nixos-rebuild switch --flake .#nixos-telemetry
# Roll back:
sudo nixos-rebuild switch --rollback
```

## Writing Custom Modules

### Pattern: Service Module

```nix
# modules/my-inference.nix
{ config, lib, pkgs, ... }:
let
  cfg = config.services.my-inference;
in {
  options.services.my-inference = {
    enable = lib.mkEnableOption "my inference service";
    model = lib.mkOption {
      type = lib.types.path;
      description = "Path to model file";
    };
    port = lib.mkOption {
      type = lib.types.port;
      default = 8080;
    };
    gpuLayers = lib.mkOption {
      type = lib.types.int;
      default = 99;
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.my-inference = {
      description = "My Inference Service";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];
      serviceConfig = {
        ExecStart = ''
          ${pkgs.nvidia-jetpack.cudaPackages.llama-cpp}/bin/llama-server \
            --model ${cfg.model} \
            --port ${toString cfg.port} \
            --n-gpu-layers ${toString cfg.gpuLayers}
        '';
        Restart = "on-failure";
        # Security hardening
        DynamicUser = true;
        SupplementaryGroups = [ "video" "render" ];
      };
    };

    networking.firewall.allowedTCPPorts = [ cfg.port ];
  };
}
```

### Pattern: Performance Module

```nix
# modules/performance.nix
{ config, lib, pkgs, ... }: {
  options.services.orin-perf = {
    enable = lib.mkEnableOption "Orin performance tuning";
  };

  config = lib.mkIf config.services.orin-perf.enable {
    hardware.nvidia-jetpack.maxClock = true;
    services.nvpmodel.profileNumber = 0;  # MAXN

    boot.kernelParams = [
      "hugepagesz=2M" "hugepages=512"
      "transparent_hugepage=always"
    ];

    zramSwap = {
      enable = true;
      memoryPercent = 50;
    };
  };
}
```

## Using CUDA Packages

```nix
# In your config or a dev shell
{ pkgs, ... }: {
  environment.systemPackages = with pkgs; [
    # Access the Jetson CUDA toolkit
    nvidia-jetpack.cudaPackages.cuda_nvcc
    nvidia-jetpack.cudaPackages.cudnn
    nvidia-jetpack.cudaPackages.tensorrt

    # Or use nixpkgs' CUDA ecosystem (configured by jetpack)
    cudaPackages.cuda_nvcc
  ];
}
```

### Dev Shell for CUDA Development

```nix
# In your flake.nix outputs
devShells.aarch64-linux.default = let
  pkgs = import nixpkgs {
    system = "aarch64-linux";
    overlays = [ jetpack.overlays.default ];
    config = {
      allowUnfree = true;
      cudaSupport = true;
      cudaCapabilities = [ "8.7" ];
    };
  };
in pkgs.mkShell {
  packages = with pkgs.nvidia-jetpack.cudaPackages; [
    cuda_nvcc
    cuda_cudart
    libcublas
    cudnn
  ];

  shellHook = ''
    export CUDA_PATH=${pkgs.nvidia-jetpack.cudaPackages.cuda_nvcc}
    echo "CUDA dev shell ready"
  '';
};
```

## Docker with GPU Support

```nix
# modules/docker-nvidia.nix
{ config, lib, pkgs, ... }: {
  virtualisation.docker.enable = true;
  hardware.nvidia-container-toolkit.enable = true;

  # Add your user to the docker group
  users.users.myuser.extraGroups = [ "docker" ];
}
```

```bash
# Test GPU in Docker
docker run --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all ubuntu nvidia-smi
```

## Troubleshooting

### Build fails with "CUDA capability not supported"

```nix
# Ensure capability matches your SoM
hardware.nvidia-jetpack.som = "orin-agx";  # Sets capability 8.7 automatically
```

### "GPU not found" after boot

```nix
# Ensure graphics are enabled
hardware.graphics.enable = true;
# Check kernel modules are loaded:
# lsmod | grep nvgpu    (JP5/6)
# lsmod | grep nvidia   (JP7)
```

### Flash script won't run

```bash
# Flash scripts are x86_64 only!
# Must run from an x86_64 Linux machine
nix build .#nixosConfigurations.my-jetson.config.system.build.flashScript \
  --system x86_64-linux
```

### Reverting a bad firmware update

UEFI capsule updates use A/B partitions. If an update fails, the previous firmware slot is used automatically. For manual recovery, use recovery mode + flash script.

### Checking firmware version

```bash
# Current firmware version (from DMI)
cat /sys/devices/virtual/dmi/id/bios_version

# Check if update is pending
ota-check-firmware -b
```

## The Complete Picture

```
YOU WRITE:                          JETPACK-NIXOS PROVIDES:
─────────────                       ──────────────────────
flake.nix                           nixosModules.default
  └─ inputs: nixpkgs, jetpack         ├─ kernel (5.10/5.15/6.8)
  └─ modules: [                        ├─ GPU drivers (l4t-3d-core)
       jetpack.nixosModules.default    ├─ firmware (l4t-firmware)
       ./configuration.nix             ├─ kernel modules (nvgpu/nvidia)
     ]                                 ├─ CUDA packages
                                       ├─ udev rules
configuration.nix                      ├─ power management
  └─ som = "orin-agx"                 ├─ fan control
  └─ carrierBoard = "devkit"           ├─ device trees
  └─ configureCuda = true              ├─ UEFI firmware (EDK2)
  └─ your services...                  ├─ OP-TEE
  └─ your packages...                  ├─ flash scripts
                                       └─ capsule OTA updates
         │
         ▼
   nixos-rebuild switch
         │
         ▼
   Complete, reproducible, immutable system
   Firmware + OS + drivers + services
   All from one flake.nix
```

## Commands Cheat Sheet

```bash
# ──── Building ────
sudo nixos-rebuild switch --flake .#my-jetson        # Build & activate
sudo nixos-rebuild test --flake .#my-jetson           # Activate without persisting
sudo nixos-rebuild boot --flake .#my-jetson           # Set as next boot, don't activate
sudo nixos-rebuild switch --rollback                   # Roll back to previous generation

# ──── Flashing (from x86_64 host) ────
nix build .#nixosConfigurations.my-jetson.config.system.build.flashScript
sudo ./result/bin/initrd-flash-jetson

# ──── Inspection ────
nix eval .#nixosConfigurations.my-jetson.config.hardware.nvidia-jetpack.som
nix repl                                              # Then :lf . to load flake
nixos-option hardware.nvidia-jetpack                  # List jetpack options

# ──── CUDA ────
nvidia-smi                                            # GPU status (JP7)
cat /sys/devices/gpu.0/load                           # GPU load (JP5/6)
nvcc --version                                        # CUDA compiler version
cat /etc/nv_tegra_release                             # L4T version

# ──── Power & Thermal ────
nvpmodel -q                                           # Current power profile
jetson_clocks --show                                  # Clock speeds
tegrastats                                            # Live system/GPU stats

# ──── Firmware ────
cat /sys/devices/virtual/dmi/id/bios_version          # Firmware version
ota-check-firmware -b                                 # Check firmware match
```

## Summary

| Step | What | Command |
|------|------|---------|
| 1 | Create flake | Write `flake.nix` with jetpack input |
| 2 | Configure | Write `configuration.nix` with SoM + services |
| 3 | Flash | `nix run .#...flashScript` from x86_64 |
| 4 | Update | `nixos-rebuild switch --flake .#config` |
| 5 | Firmware OTA | Automatic with `firmware.autoUpdate = true` |
| 6 | Rollback | `nixos-rebuild switch --rollback` |

**The whole point**: Your Jetson dev kit — drivers, kernel, firmware, CUDA, services, everything — defined in a handful of `.nix` files. Reproducible. Version-controlled. Rollback-able. One `nixos-rebuild switch` away from any configuration you can imagine.

---

**Previous**: [← Pill 08: Flashing, Firmware & OTA Updates](08-flashing-firmware-ota.md)
**Back to**: [README — Table of Contents](README.md)
