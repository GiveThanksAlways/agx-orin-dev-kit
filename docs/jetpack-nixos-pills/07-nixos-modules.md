# Pill 07: NixOS Modules — Declaring Your System

## tl;dr

A NixOS **module** is a function that declares options and sets configuration values. The module system merges dozens of modules into one coherent configuration. jetpack-nixos provides a module (`nixosModules.default`) that declares Jetson-specific options (`hardware.nvidia-jetpack.*`) and uses them to configure the kernel, drivers, firmware, and services. Understanding the module system is understanding how NixOS becomes a working system.

## What Is a NixOS Module?

A module is a function that returns `{ options, config, imports }`:

```nix
# The simplest possible module
{ config, lib, pkgs, ... }: {
  options = {
    # Declare new options
    services.myService.enable = lib.mkEnableOption "my service";
  };

  config = lib.mkIf config.services.myService.enable {
    # Set values when enabled
    systemd.services.myService = {
      description = "My Service";
      wantedBy = [ "multi-user.target" ];
      serviceConfig.ExecStart = "${pkgs.myApp}/bin/myapp";
    };
  };
}
```

**Two halves**:
1. **`options`**: "What can be configured?" (the schema)
2. **`config`**: "What values should be set?" (the implementation)

## The Module System: Merge Everything

NixOS takes dozens of modules and **merges** them:

```
Module 1 (jetpack): options.hardware.nvidia-jetpack.* + config.boot.kernelPackages = ...
Module 2 (your config): config.hardware.nvidia-jetpack.som = "orin-agx"
Module 3 (graphics): config.hardware.graphics.* = ...
Module 4 (ssh): config.services.openssh.enable = true
  ...all merged into one final config...
```

The module system resolves dependencies:
- Module A sets `option.X = true`
- Module B reads `config.X` and conditionally sets `option.Y`
- Module C reads `config.Y` and configures a service

This all resolves lazily — no ordering needed.

## Option Types

### Common Types

```nix
{ lib, ... }: {
  options = {
    # Boolean (with enable helper)
    hardware.nvidia-jetpack.enable = lib.mkEnableOption "NVIDIA Jetson device support";

    # String enum
    hardware.nvidia-jetpack.som = lib.mkOption {
      type = lib.types.enum [
        "generic" "orin-agx" "orin-nx" "orin-nano"
        "xavier-agx" "xavier-nx" "thor-agx"
      ];
      default = "generic";
      description = "Jetson SoM to target";
    };

    # Boolean
    hardware.nvidia-jetpack.maxClock = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Always run at max clock speed";
    };

    # Integer (optional)
    services.nvpmodel.profileNumber = lib.mkOption {
      type = lib.types.nullOr lib.types.int;
      default = null;
      description = "Power model profile ID";
    };

    # Path
    services.nvfancontrol.configFile = lib.mkOption {
      type = lib.types.path;
      description = "Fan control config file";
    };

    # String
    hardware.nvidia-jetpack.majorVersion = lib.mkOption {
      type = lib.types.enum [ "5" "6" "7" ];
      default = "6";
    };

    # List of strings
    boot.kernelParams = lib.mkOption {
      type = lib.types.listOf lib.types.str;
    };
  };
}
```

### Type Composition

```nix
types.nullOr types.int        # null or int
types.listOf types.str         # list of strings
types.attrsOf types.package    # attrset of packages
types.either types.str types.int  # string or int
types.submodule { ... }        # Nested module with its own options
```

## Priority Functions: `mkDefault`, `mkForce`, `mkBefore`

When multiple modules set the same option, NixOS uses **priority** to resolve conflicts:

```nix
# Priority levels (lower number = higher priority):
# 50   mkForce
# 100  default (normal assignment)
# 1000 mkDefault
# 1500 mkOptionDefault

# Multiple modules set the same option:
# Module A:
boot.kernelParams = [ "console=ttyTCU0,115200" ];  # priority 100

# Module B:
boot.kernelParams = lib.mkDefault [ "quiet" ];  # priority 1000 (lower priority)

# Result: [ "console=ttyTCU0,115200" ] wins (higher priority = 100)
```

### `mkDefault` — "I'm a sensible default, override me"

```nix
# From modules/devices.nix
services.nvpmodel.enable = mkIf (nvpModelConf ? "${cfg.som}") (mkDefault true);
# User can override: services.nvpmodel.enable = false;
```

### `mkForce` — "I know what I'm doing, override everything"

```nix
# From modules/graphics.nix
hardware.nvidia.enabled = lib.mkForce false;
# Prevents upstream NVIDIA desktop modules from conflicting with Jetson
```

### `mkBefore` — "Put my values first in the list"

```nix
# From modules/default.nix
nixpkgs.overlays = mkBefore [
  overlay
  (import ../overlay-with-config.nix config)
];
# Ensures jetpack overlays are applied BEFORE user overlays
```

### `mkMerge` — "Combine these conditionally"

```nix
config = lib.mkMerge [
  # Always applied
  {
    boot.kernelModules = [ "nvgpu" ];
  }

  # Only for JP6+
  (lib.mkIf (jetpackAtLeast "6") {
    hardware.deviceTree.dtbSource = config.boot.kernelPackages.devicetree;
  })
];
```

### `mkIf` — "Only set this if condition is true"

```nix
config = mkIf cfg.enable {
  # Everything in here is only set when hardware.nvidia-jetpack.enable = true
  boot.kernelPackages = pkgs.nvidia-jetpack.kernelPackages;
  hardware.firmware = [ pkgs.nvidia-jetpack.l4t-firmware ];
};
```

## The jetpack-nixos Module Tree

```nix
# modules/default.nix is the root
{
  imports = [
    ./capsule-updates.nix          # OTA firmware updates
    ./cuda.nix                     # CUDA configuration
    ./devices.nix                  # Per-SoM hardware config
    ./flash-script.nix             # Flash options + firmware settings
    ./graphics.nix                 # GPU driver wiring
    ./nvargus-daemon.nix           # Camera daemon
    ./nvfancontrol.nix             # Fan control service
    ./nvidia-container-toolkit.nix # Docker GPU passthrough
    ./nvpmodel.nix                 # Power profiles
    ./optee.nix                    # Trusted execution
  ];
}
```

Each sub-module is focused on one concern. Let's look at the important ones.

## `modules/default.nix` — The Root Module

This is the powerhouse. When you set `hardware.nvidia-jetpack.enable = true`, it:

### Declares Options

```nix
options.hardware.nvidia-jetpack = {
  enable = mkEnableOption "NVIDIA Jetson device support";
  som = mkOption { type = types.enum [ "generic" "orin-agx" ... ]; };
  carrierBoard = mkOption { type = types.enum [ "generic" "devkit" ... ]; };
  majorVersion = mkOption { type = types.enum [ "5" "6" "7" ]; };
  maxClock = mkOption { type = types.bool; default = false; };
  kernel.realtime = mkOption { type = types.bool; default = false; };
};
```

### Sets the Kernel

```nix
boot.kernelPackages =
  (if cfg.kernel.realtime
    then pkgs.nvidia-jetpack.rtkernelPackages
    else pkgs.nvidia-jetpack.kernelPackages
  ).extend pkgs.nvidia-jetpack.kernelPackagesOverlay;
```

### Sets Kernel Parameters

```nix
boot.kernelParams = [
  "nvidia.rm_firmware_active=all"
] ++ lib.optionals cfg.console.enable [
  "console=tty0"
  "console=ttyTCU0,115200"  # Tegra Combined UART
];
```

### Loads Kernel Modules

```nix
boot.kernelModules = if (jetpackAtLeast "7")
  then [ "nvidia-uvm" ]
  else [ "nvgpu" ];

boot.extraModulePackages = lib.optional (jetpackAtLeast "6")
  config.boot.kernelPackages.nvidia-oot-modules;
```

### Sets Hardware Firmware

```nix
hardware.firmware = with pkgs.nvidia-jetpack; [
  l4t-firmware
] ++ lib.optionals (lib.versionOlder cfg.majorVersion "7") [
  cudaPackages.vpi-firmware
] ++ lib.optionals (l4tOlder "38") [
  l4t-xusb-firmware
];
```

### Applies Overlays

```nix
nixpkgs.overlays = mkBefore [
  overlay                                  # The main jetpack overlay
  (final: prev: {                          # Version selection overlay
    nvidia-jetpack = final."nvidia-jetpack${cfg.majorVersion}";
    cudaPackages = final."cudaPackages_${...}";
  })
  (import ../overlay-with-config.nix config)  # Config-dependent overlay
];
```

### Configures udev Rules

```nix
services.udev.packages = [
  (pkgs.runCommand "jetson-udev-rules" { } ''
    install -D -t $out/etc/udev/rules.d \
      ${pkgs.nvidia-jetpack.l4t-init}/etc/udev/rules.d/99-tegra-devices.rules
    # Patch paths to use Nix store binaries
    sed -i -e 's#/bin/mknod#${lib.getExe' pkgs.coreutils "mknod"}#' ...
  '')
];
```

### Sets Up Services

```nix
systemd.services.jetson_clocks = mkIf cfg.maxClock {
  description = "Set maximum clock speed";
  serviceConfig.ExecStart = "${pkgs.nvidia-jetpack.l4t-tools}/bin/jetson_clocks";
  after = [ "nvpmodel.service" ];
  wantedBy = [ "multi-user.target" ];
};
```

## `modules/devices.nix` — Per-SoM Configuration

Maps each SoM to its specific configs:

```nix
# Power model configs
nvpModelConf = {
  orin-agx           = ".../nvpmodel_p3701_0000.conf";
  orin-agx-industrial = ".../nvpmodel_p3701_0008.conf";
  orin-nx            = ".../nvpmodel_p3767_0000.conf";
  orin-nano          = ".../nvpmodel_p3767_0003.conf";
  thor-agx           = ".../nvpmodel_p3834_0008.conf";
  xavier-agx         = ".../nvpmodel_t194.conf";
};

# Fan control configs
nvfancontrolConf = {
  orin-agx           = ".../nvfancontrol_p3701_0000.conf";
  orin-nx            = ".../nvfancontrol_p3767_0000.conf";
  # ...
};

# Flash partition templates
hardware.nvidia-jetpack.flashScriptOverrides = mkMerge [
  (mkIf (cfg.som == "orin-agx") {
    targetBoard = mkDefault "jetson-agx-orin-devkit";
    partitionTemplate = mkDefault ".../flash_t234_qspi.xml";
  })
  (mkIf (cfg.som == "orin-nx" || cfg.som == "orin-nano") {
    targetBoard = mkDefault "jetson-orin-nano-devkit";
    partitionTemplate = mkDefault ".../flash_t234_qspi.xml";
  })
  # ... one block per SoM
];
```

## `modules/graphics.nix` — GPU Driver Wiring

```nix
# Sets the GPU driver package
hardware.graphics.package = pkgs.nvidia-jetpack.l4t-3d-core;

# Extra packages needed for GPU
hardware.graphics.extraPackages = [
  jetson-graphics-extra-packages  # Symlink join of:
    # l4t-camera, l4t-core, l4t-cuda, l4t-gbm,
    # l4t-multimedia, l4t-nvsci, l4t-pva, l4t-wayland
];

# Force NVIDIA X11 driver
services.xserver.drivers = lib.mkForce (lib.singleton {
  name = "nvidia";
  modules = [ pkgs.nvidia-jetpack.l4t-3d-core ];
});

# Critically: disable upstream NVIDIA modules (they're for desktop GPUs)
hardware.nvidia.enabled = lib.mkForce false;
```

## `modules/cuda.nix` — CUDA Integration

```nix
# When enabled, configures nixpkgs for CUDA support
nixpkgs.config = mkIf cfg.configureCuda {
  cudaSupport = true;
  cudaCapabilities =
    lib.optionals isXavier [ "7.2" ] ++
    lib.optionals isOrin [ "8.7" ] ++
    lib.optionals isThor [ "11.0" ];
};

# Assertions to prevent invalid CUDA + JetPack combos
assertions = [
  {
    assertion = cfg.majorVersion == "6" -> (cudaAtLeast "12.4" && cudaOlder "13.0");
    message = "JetPack 6 supports CUDA 12.4-12.9";
  }
];
```

## `modules/nvpmodel.nix` — Power Profiles

```nix
# NVPModel controls CPU/GPU frequency, online cores, etc.
options.services.nvpmodel = {
  enable = mkEnableOption "NVPModel";
  configFile = mkOption { type = types.path; };
  profileNumber = mkOption { type = types.nullOr types.int; default = null; };
};

config = mkIf cfg.enable {
  systemd.services.nvpmodel = {
    ExecStart = "${pkgs.nvidia-jetpack.l4t-nvpmodel}/bin/nvpmodel -f ${cfg.configFile}";
    wantedBy = [ "multi-user.target" ];
  };
};
```

### Power Profiles

| Profile | Mode | CPU Cores | GPU Freq | Power |
|---------|------|-----------|----------|-------|
| 0 | MAXN | All | Max | Full |
| 1 | 50W | 8 | Medium | 50W |
| 2 | 30W | 6 | Low | 30W |
| 3 | 15W | 4 | Low | 15W |

## `modules/nvfancontrol.nix` — Fan Control

```nix
systemd.services.nvfancontrol = mkIf cfg.enable {
  ExecStart = "${pkgs.nvidia-jetpack.l4t-nvfancontrol}/bin/nvfancontrol -f ${cfg.configFile}";
  wantedBy = [ "multi-user.target" ];
};
```

## `modules/nvidia-container-toolkit.nix` — Docker GPU

Enables GPU passthrough for Docker containers:

```nix
hardware.nvidia-container-toolkit = {
  enable = true;
  # Jetson uses CSV-based discovery (not standard GPU discovery)
  mount-nvidia-executables = false;
  mount-nvidia-docker-1-directories = false;
  device-name-strategy = "by-index";
};
```

After enabling:
```bash
docker run --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all ubuntu nvidia-smi
```

## Writing Your Own Module

You can extend jetpack-nixos with your own modules:

```nix
# modules/performance.nix
{ config, lib, pkgs, ... }: {
  options.services.orin-perf = {
    enable = lib.mkEnableOption "Orin performance tuning";
  };

  config = lib.mkIf config.services.orin-perf.enable {
    # Lock clocks to max
    hardware.nvidia-jetpack.maxClock = true;

    # Use MAXN power profile
    services.nvpmodel.profileNumber = 0;

    # Hugepages for CUDA
    boot.kernelParams = [
      "hugepagesz=2M" "hugepages=512"
      "transparent_hugepage=always"
    ];

    # zram swap for the 64GB model
    zramSwap = {
      enable = true;
      memoryPercent = 50;
    };
  };
}
```

Use it:
```nix
nixosConfigurations.nixos-perf = nixpkgs.lib.nixosSystem {
  modules = baseModules ++ [
    ./modules/performance.nix
    ({ ... }: { services.orin-perf.enable = true; })
  ];
};
```

## The Module Evaluation Flow

```
1. All modules are imported and merged

2. options tree is built:
   hardware.nvidia-jetpack.enable: bool
   hardware.nvidia-jetpack.som: enum
   boot.kernelPackages: package
   boot.kernelParams: listOf str
   ... thousands of options

3. config values are resolved:
   hardware.nvidia-jetpack.enable = true (from your config)
   hardware.nvidia-jetpack.som = "orin-agx" (from your config)
   
4. mkIf guards evaluate:
   cfg.enable is true → all jetpack config blocks activate
   
5. Dependent options cascade:
   som = "orin-agx" → socType = "t234"
                     → boot.kernelPackages = orin kernel
                     → nvpmodel = p3701_0000.conf
                     → partitionTemplate = flash_t234_qspi.xml

6. Final config is complete:
   A single, merged attribute set describing the entire system
```

## The Configuration You Write

```nix
# configuration.nix
{ config, lib, pkgs, ... }: {
  # Jetson hardware
  hardware.nvidia-jetpack.enable = true;
  hardware.nvidia-jetpack.som = "orin-agx";
  hardware.nvidia-jetpack.carrierBoard = "devkit";
  hardware.nvidia-jetpack.configureCuda = true;
  hardware.graphics.enable = true;

  # Boot
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;

  # This is enough. The jetpack module handles:
  # - Kernel selection (5.15.148 for Orin)
  # - GPU driver (l4t-3d-core)
  # - Firmware (l4t-firmware)
  # - Kernel modules (nvgpu)
  # - Kernel parameters (nvidia.rm_firmware_active=all)
  # - udev rules (99-tegra-devices.rules)
  # - Power management (nvpmodel, nvfancontrol)
  # - CUDA configuration (sm_87, cudaSupport=true)
  # - Device tree compilation
  # - Flash script generation
  # - Capsule update support
}
```

## Summary

| Concept | What | Example |
|---------|------|---------|
| Module | `options` + `config` | `modules/default.nix` |
| `mkEnableOption` | Boolean on/off | `hardware.nvidia-jetpack.enable` |
| `mkOption` | Typed config option | `som`, `carrierBoard`, `majorVersion` |
| `mkIf` | Conditional config | Only apply when `enable = true` |
| `mkMerge` | Combine configs | Per-SoM device settings |
| `mkDefault` | Low-priority value | Sensible defaults, easily overridden |
| `mkForce` | High-priority value | Disable upstream NVIDIA modules |
| `mkBefore` | List ordering | Jetpack overlays applied first |
| Module merge | All modules → one config | The whole system from 10+ modules |

**The mental model**: Each NixOS module is a **layer of declarations**. The module system merges them all into one flat configuration with conflict resolution via priorities. jetpack-nixos provides a module that declares Jetson-specific options and, when enabled, configures every aspect of the system — from the kernel to GPU drivers to power management — all driven by a single `som = "orin-agx"` setting.

---

**Previous**: [← Pill 06: CUDA, cuDNN, TensorRT & GPU Packages](06-cuda-gpu-packages.md)
**Next**: [Pill 08: Flashing, Firmware & OTA Updates →](08-flashing-firmware-ota.md)
