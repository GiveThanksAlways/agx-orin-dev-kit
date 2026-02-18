# Pill 04: jetpack-nixos — The 30,000ft View

## tl;dr

jetpack-nixos is a **Nix overlay + NixOS module** that packages NVIDIA's entire JetPack SDK for NixOS. It takes NVIDIA's proprietary `.deb` packages, kernel source, BSP blobs, and firmware — and wraps them in Nix's reproducible build system. The result: your Jetson's kernel, drivers, CUDA stack, firmware, and flash scripts are all declared in Nix expressions, pinned to exact versions, and buildable from scratch.

## What Is JetPack?

NVIDIA JetPack SDK is the software stack for Jetson embedded platforms:

```
┌─────────────────────────────────────────────────┐
│                  JetPack SDK                     │
├─────────────────────────────────────────────────┤
│  AI Frameworks   │  TensorRT, cuDNN, CUDA       │
├──────────────────┼──────────────────────────────┤
│  Multimedia      │  GStreamer, V4L2, Argus       │
├──────────────────┼──────────────────────────────┤
│  Graphics        │  EGL, Vulkan, OpenGL ES       │
├──────────────────┼──────────────────────────────┤
│  L4T (Linux for  │  Kernel, drivers, firmware,   │
│  Tegra)          │  bootloader, flash tools      │
├──────────────────┼──────────────────────────────┤
│  Hardware        │  Xavier, Orin, Thor SoCs       │
└─────────────────────────────────────────────────┘
```

NVIDIA packages this for Ubuntu. jetpack-nixos packages it for NixOS.

## The Problem jetpack-nixos Solves

NVIDIA distributes JetPack as:
- **`.deb` files** from `repo.download.nvidia.com/jetson/` (driver libraries, CUDA, etc.)
- **Git repos** (kernel source, device trees, firmware build scripts)
- **BSP tarball** (Board Support Package — bootloader, flash tools, partition layouts)
- **x86_64-only flash tools** (NVIDIA's flashing requires an x86 host)

None of this is designed for Nix. jetpack-nixos:
1. **Pins** every source with content hashes in JSON manifests
2. **Extracts** `.deb` files and patches ELF binaries to use Nix store paths
3. **Builds** the kernel from NVIDIA's git source with Jetson-specific configs
4. **Wraps** flash tools to work on NixOS
5. **Exposes** everything through a standard Nix overlay and NixOS module

## Architecture Overview

```
flake.nix                                     ← Entry point
│
├── inputs: nixpkgs, cuda-legacy
│
├── overlays.default                          ← Package layer
│   ├── cuda-legacy.overlays.default          ← CUDA 11.4 packages
│   ├── import ./overlay.nix                  ← Main overlay
│   │   ├── nvidia-jetpack5 (mk-overlay.nix)  ← JP5 scope
│   │   ├── nvidia-jetpack6 (mk-overlay.nix)  ← JP6 scope
│   │   ├── nvidia-jetpack7 (mk-overlay.nix)  ← JP7 scope
│   │   ├── nvidia-jetpack (auto-selected)     ← Default scope
│   │   ├── cudaPackages_* overrides           ← CUDA integration
│   │   └── _cuda extensions                   ← Deb-based CUDA packages
│   └── guard marker                           ← Idempotency
│
├── nixosModules.default                      ← System layer
│   ├── modules/default.nix                   ← Root module
│   │   ├── modules/cuda.nix                  ← CUDA config
│   │   ├── modules/devices.nix               ← Per-SoM config
│   │   ├── modules/graphics.nix              ← GPU drivers
│   │   ├── modules/flash-script.nix          ← Flash options
│   │   ├── modules/capsule-updates.nix       ← OTA firmware
│   │   ├── modules/optee.nix                 ← Trusted execution
│   │   ├── modules/nvpmodel.nix              ← Power profiles
│   │   ├── modules/nvfancontrol.nix          ← Fan control
│   │   └── modules/nvidia-container-toolkit.nix ← Docker GPU
│   └── import ./overlay-with-config.nix      ← Config-dependent
│
├── nixosConfigurations                       ← Pre-built configs
│   ├── installer_minimal                     ← ISO images
│   └── orin-agx-devkit, orin-nx-devkit, ...  ← All supported devices
│
├── packages                                  ← Build outputs
│   ├── x86_64-linux: flash-*, iso_minimal    ← Flash scripts, ISOs
│   └── aarch64-linux: iso_minimal            ← Native ISOs
│
└── legacyPackages                            ← Package access
    ├── nvidia-jetpack (default)
    ├── nvidia-jetpack5
    ├── nvidia-jetpack6
    └── nvidia-jetpack7
```

## The Two Halves: Overlay + Module

### The Overlay: "What Packages Exist"

`overlay.nix` + `mk-overlay.nix` define what packages are available:

```nix
# overlay.nix adds these to nixpkgs:
pkgs.nvidia-jetpack5.kernel          # 5.10 kernel for JP5
pkgs.nvidia-jetpack5.l4t-3d-core     # GPU driver for JP5
pkgs.nvidia-jetpack5.flash-tools     # Flash tools for JP5
pkgs.nvidia-jetpack5.cudaPackages    # CUDA 11.4

pkgs.nvidia-jetpack6.kernel          # 5.15 kernel for JP6
pkgs.nvidia-jetpack6.l4t-3d-core     # GPU driver for JP6
pkgs.nvidia-jetpack6.flash-tools     # Flash tools for JP6
pkgs.nvidia-jetpack6.cudaPackages    # CUDA 12.6

pkgs.nvidia-jetpack7.kernel          # 6.8 kernel for JP7
# ... etc

pkgs.nvidia-jetpack                  # Points to JP5, JP6, or JP7
pkgs.cudaPackages                    # Points to appropriate CUDA version
```

### The Module: "How the System Is Configured"

`modules/default.nix` declares NixOS options and wires packages into the system:

```nix
# What the module does when you set:
hardware.nvidia-jetpack.enable = true;
hardware.nvidia-jetpack.som = "orin-agx";

# It sets:
boot.kernelPackages = pkgs.nvidia-jetpack.kernelPackages;  # Jetson kernel
boot.kernelModules = [ "nvgpu" ];                           # GPU kernel module
boot.kernelParams = [ "nvidia.rm_firmware_active=all" ];    # Driver params
hardware.firmware = [ pkgs.nvidia-jetpack.l4t-firmware ];   # Firmware blobs
hardware.graphics.package = pkgs.nvidia-jetpack.l4t-3d-core; # GPU driver
services.nvpmodel.configFile = "...orin-agx config...";     # Power profile
services.nvfancontrol.configFile = "...orin-agx config..."; # Fan control
# ... and much more
```

## `mk-overlay.nix`: The Package Factory

This is the heart of jetpack-nixos. It's a **factory function** that takes version parameters and produces a complete package scope:

```nix
# Called three times with different version tuples:
{ jetpackMajorMinorPatchVersion   # "6.2.1"
, l4tMajorMinorPatchVersion       # "36.4.4"
, cudaMajorMinorPatchVersion      # "12.6.10"
, cudaDriverMajorMinorVersion     # "540.4.0"
, bspHash                         # sha256 of the BSP tarball
, bspPatches ? []
, bspPostPatch ? []
}:
final: _:                         # overlay args (final nixpkgs, prev unused)
let
  sourceInfo = import ./sourceinfo { ... };  # Fetch pinned sources
in
makeScope final.newScope (self: {
  # Version info
  inherit jetpackMajorMinorPatchVersion l4tMajorMinorPatchVersion;

  # Source packages
  inherit (sourceInfo) debs gitRepos;
  bspSrc = ...;  # BSP unpacked + patched

  # L4T packages (from debs)
  l4t-3d-core = ...;
  l4t-cuda = ...;
  l4t-firmware = ...;
  # ... 25+ L4T packages

  # Kernel
  kernel = self.callPackage ./pkgs/kernels/r${l4tMajorVersion} { };
  kernelPackages = final.linuxPackagesFor self.kernel;

  # Flash tools
  flash-tools = self.callPackage ./pkgs/flash-tools { };

  # UEFI firmware
  uefi-firmware = ...;

  # OP-TEE
  buildTOS = ...;

  # Benchmarks, samples, tests
  samples = ...;
  tests = ...;
})
```

### The Factory Pattern Visualized

```
mk-overlay.nix("5.1.5", "35.6.2", "11.4.298", ...)
  → nvidia-jetpack5 scope
    ├── kernel (5.10.216)
    ├── l4t-3d-core (from r35.6 debs)
    ├── cudaPackages (CUDA 11.4)
    └── flash-tools (JP5)

mk-overlay.nix("6.2.1", "36.4.4", "12.6.10", ...)
  → nvidia-jetpack6 scope
    ├── kernel (5.15.148)
    ├── l4t-3d-core (from r36.4 debs)
    ├── cudaPackages (CUDA 12.6)
    └── flash-tools (JP6)

mk-overlay.nix("7.0", "38.2.1", "13.0.2", ...)
  → nvidia-jetpack7 scope
    ├── kernel (6.8.12)
    ├── l4t-3d-core (from r38.2 debs)
    ├── cudaPackages (CUDA 13.0)
    └── flash-tools (JP7)
```

Same function, different inputs, different outputs. That's functional programming.

## `overlay-with-config.nix`: The Config Bridge

The overlay provides generic packages. But some packages need to know your specific hardware:
- What SoM are you on? (Orin AGX vs Orin NX vs Xavier)
- What carrier board? (devkit vs custom)
- What firmware options? (boot logo, secure boot keys, debug mode)

`overlay-with-config.nix` bridges the gap:

```nix
# Receives the NixOS config
config:
final: prev: {
  nvidia-jetpack = prev.nvidia-jetpack.overrideScope (finalJetpack: prevJetpack: {
    # Device detection
    socType = if lib.hasPrefix "orin-" cfg.som then "t234" else ...;
    chipId = if lib.hasPrefix "orin-" cfg.som then "0x23" else ...;

    # Firmware customization
    uefi-firmware = prevJetpack.uefi-firmware.override {
      bootLogo = cfg.firmware.uefi.logo;
      debugMode = cfg.firmware.uefi.debugMode;
    };

    # Flash script generation (needs full config)
    mkFlashScript = flash-tools: args: import ./device-pkgs/flash-script.nix { ... };
    bup = ...;  # Board Update Package
    signedFirmware = ...;  # Signed firmware image
  });
}
```

### Why Two Overlays?

```
overlay.nix              → Generic packages (don't need NixOS config)
overlay-with-config.nix  → Config-specific packages (need to know your SoM)
```

This separation exists because:
1. `overlay.nix` can be used **without** NixOS (e.g., in a `nix develop` shell)
2. `overlay-with-config.nix` only makes sense in a NixOS system context
3. Keeps config-dependent logic cleanly separated

## The Data Flow

When you run `sudo nixos-rebuild switch --flake .#nixos`:

```
1. Nix reads flake.nix
   └── Resolves inputs from flake.lock (nixpkgs, jetpack)

2. Evaluates nixosConfigurations.nixos
   └── Calls nixpkgs.lib.nixosSystem with your modules

3. Merges modules:
   ├── jetpack.nixosModules.default
   │   ├── Applies overlays to nixpkgs (overlay.nix + overlay-with-config.nix)
   │   └── Imports all sub-modules (cuda, devices, graphics, flash, ...)
   └── ./configuration.nix
       └── Your settings (som, carrierBoard, packages, services, ...)

4. Evaluates merged config:
   ├── boot.kernelPackages → nvidia-jetpack.kernelPackages
   │   └── Builds kernel 5.15.148 from NVIDIA git source
   ├── hardware.firmware → [ l4t-firmware, vpi-firmware, ... ]
   │   └── Fetches + extracts firmware debs
   ├── hardware.graphics.package → l4t-3d-core
   │   └── Fetches + extracts GPU driver deb
   ├── systemd.services → nvpmodel, nvfancontrol, ...
   └── ... hundreds more attributes

5. Builds the system closure:
   └── /nix/store/hash-nixos-system-25.11/
       ├── kernel
       ├── initrd
       ├── firmware/
       ├── etc/
       └── sw/  (all installed packages)

6. Activates the new system:
   └── Switches bootloader, restarts services, updates symlinks
```

## Supported Devices

jetpack-nixos supports every Jetson SoM that NVIDIA currently ships:

| SoM | SoC | JetPack | CUDA Arch |
|-----|-----|---------|-----------|
| Orin AGX | T234 | 5, 6 | sm_87 |
| Orin AGX Industrial | T234 | 5, 6 | sm_87 |
| Orin NX | T234 | 5, 6 | sm_87 |
| Orin NX (Super) | T234 | 5, 6 | sm_87 |
| Orin Nano | T234 | 5, 6 | sm_87 |
| Orin Nano (Super) | T234 | 5, 6 | sm_87 |
| Thor AGX | T264 | 7 | sm_110 |
| Xavier AGX | T194 | 5 | sm_72 |
| Xavier AGX Industrial | T194 | 5 | sm_72 |
| Xavier NX | T194 | 5 | sm_72 |
| Xavier NX (eMMC) | T194 | 5 | sm_72 |

## The `nvidia-jetpack` Scope Contents

```nix
pkgs.nvidia-jetpack = {
  # Version info
  jetpackMajorMinorPatchVersion;  # "6.2.1"
  l4tMajorMinorPatchVersion;      # "36.4.4"
  cudaMajorMinorVersion;           # "12.6"

  # Sources (pinned)
  debs;       # All .deb files, by repo
  gitRepos;   # All git repos
  bspSrc;     # Board Support Package, unpacked + patched

  # L4T packages (runtime)
  l4t-3d-core;       # GPU driver (EGL, Vulkan, GLES)
  l4t-camera;        # Argus camera framework
  l4t-core;          # Core runtime libs
  l4t-cuda;          # CUDA driver integration
  l4t-firmware;      # Device firmware
  l4t-gbm;           # GBM buffer management
  l4t-gstreamer;     # Hardware codec plugins
  l4t-init;          # Init scripts, udev rules
  l4t-multimedia;    # V4L2 multimedia
  l4t-nvfancontrol;  # Fan control daemon
  l4t-nvml;          # nvidia-smi
  l4t-nvpmodel;      # Power profiles
  l4t-nvsci;         # IPC framework
  l4t-tools;         # jetson_clocks, tegrastats
  l4t-wayland;       # Wayland support
  # ... and more, version-dependent

  # Kernel
  kernel;          # Linux kernel for this JetPack version
  kernelPackages;  # Kernel + out-of-tree modules
  rtkernel;        # PREEMPT_RT variant
  rtkernelPackages;

  # Build infrastructure
  buildFromDebs;    # Deb → Nix package helper
  callPackage;      # Scope-aware callPackage
  callPackages;     # Multi-output callPackage
  cudaPackages;     # CUDA package set for this JetPack

  # Firmware & flash
  flash-tools;     # NVIDIA flash.sh wrapper
  uefi-firmware;   # UEFI bootloader
  bspSrc;          # Board Support Package source

  # OP-TEE
  buildTOS;        # Trusted OS builder
  opteeClient;     # OP-TEE client library

  # Containers
  containerDeps;   # Dependencies for GPU containers
  l4tCsv;          # Library CSV for CDI

  # Utilities
  otaUtils;        # OTA update tools
  tegra-eeprom-tool;  # Board ID reader
  patchgpt;        # GPT partition editor

  # Tests & samples
  samples;         # CUDA/multimedia sample builds
  tests;           # Integration tests
};
```

## Key Design Decisions

### 1. Deb-to-Nix Pipeline (Not Compiling from Source)

NVIDIA's GPU driver, CUDA, and multimedia libraries are **proprietary binaries**. jetpack-nixos can't compile them from source. Instead:

```
NVIDIA .deb files → dpkg extract → autoPatchelf → Nix store path
```

`autoPatchelf` rewrites ELF binaries to reference dependencies in the Nix store instead of `/usr/lib`.

### 2. Kernel: Built from Source

Unlike the proprietary libraries, the kernel IS open source. jetpack-nixos builds it from NVIDIA's git repos with full NixOS kernel infrastructure:

```nix
kernel = buildLinux {
  src = gitRepos."kernel-oot";
  version = "5.15.148";
  structuredExtraConfig = { TEGRA_BPMP = lib.kernel.yes; ... };
};
```

### 3. Flash Tools: x86_64-Only

NVIDIA's flash tools only run on x86_64. jetpack-nixos handles this:

```nix
# Flash scripts are built for x86_64 even when targeting aarch64
flasherPkgs = import pkgs.path { system = "x86_64-linux"; };
```

### 4. Three JetPack Generations Coexist

All three (JP5, JP6, JP7) are defined simultaneously. Nix's laziness means only the one you use gets evaluated:

```nix
nvidia-jetpack5 = ...;  # Only evaluated if referenced
nvidia-jetpack6 = ...;  # Only evaluated if referenced
nvidia-jetpack7 = ...;  # Only evaluated if referenced
nvidia-jetpack = final.nvidia-jetpack6;  # This is the one that evaluates
```

## Summary

| Component | Role | Key File |
|-----------|------|----------|
| Flake | Entry point, exposes all outputs | `flake.nix` |
| Main overlay | Creates JP5/6/7 scopes, overrides CUDA | `overlay.nix` |
| Package factory | Version params → package scope | `mk-overlay.nix` |
| Config overlay | Injects NixOS config into packages | `overlay-with-config.nix` |
| NixOS module | Declares options, wires system | `modules/default.nix` |
| Source info | Pins all NVIDIA sources | `sourceinfo/` |
| L4T packages | Runtime libraries from debs | `pkgs/l4t/` |
| Kernel | Built from NVIDIA git | `pkgs/kernels/r{35,36,38}/` |
| Flash tools | x86_64 flash script wrappers | `pkgs/flash-tools/` |
| Device packages | Config-dependent flash/firmware | `device-pkgs/` |

**The mental model**: jetpack-nixos is a **translation layer** between NVIDIA's Ubuntu-centric JetPack SDK and NixOS's reproducible build system. The overlay provides the packages. The module wires them into a working system. The factory pattern handles version multiplexing. The result: your entire Jetson system — kernel, drivers, firmware, CUDA — is a pure function of your `flake.nix` + `flake.lock`.

---

**Previous**: [← Pill 03: Overlays & Overrides](03-overlays-and-overrides.md)
**Next**: [Pill 05: BSP, Kernel & L4T Packages →](05-bsp-kernel-l4t.md)
