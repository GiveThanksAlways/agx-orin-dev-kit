# Jetpack NixOS Pills

*A tl;dr first-principles guide to understanding Nix, NixOS, flakes, overlays, and the jetpack-nixos stack — built for Jetson AGX Orin 64GB*

## Welcome

This collection of "pills" (inspired by [Nix Pills](https://nixos.org/guides/nix-pills/) and our own [TinyGrad Pills](../../examples/tinygrad/tinygrad/TinyGrad-pills/README.md)) teaches you everything you need to know to understand, use, and extend jetpack-nixos. Each pill is focused, practical, and builds on previous knowledge.

**Target audience**: Intermediate NixOS/Nix users who want first-principles mastery of how jetpack-nixos packages NVIDIA's JetPack SDK, from the Nix language up through flashing firmware.

**Hardware**: Jetson AGX Orin 64GB running NixOS via jetpack-nixos.

## Reading Order

### Foundation Pills (Start Here)
0. **[Pill 00: Nix — The Language Under Everything](00-nix-language.md)** — Expressions, functions, attribute sets, laziness. The bedrock.
1. **[Pill 01: Derivations — How Nix Builds Things](01-derivations.md)** — The single concept that makes Nix work.
2. **[Pill 02: Flakes — Pinned, Composable, Reproducible](02-flakes.md)** — Inputs, outputs, lock files. Why flakes exist.
3. **[Pill 03: Overlays & Overrides — Layering the Package Set](03-overlays-and-overrides.md)** — The mechanism that lets jetpack-nixos inject Jetson packages into nixpkgs.

### jetpack-nixos Architecture Pills
4. **[Pill 04: jetpack-nixos — The 30,000ft View](04-jetpack-nixos-architecture.md)** — How flake.nix, overlay.nix, mk-overlay.nix, and modules/ fit together.
5. **[Pill 05: BSP, Kernel & L4T Packages](05-bsp-kernel-l4t.md)** — Board Support Package, kernel builds, the deb-to-nix pipeline.
6. **[Pill 06: CUDA, cuDNN, TensorRT & GPU Packages](06-cuda-gpu-packages.md)** — How CUDA package sets work, version selection, JetPack generations.

### NixOS System Pills
7. **[Pill 07: NixOS Modules — Declaring Your System](07-nixos-modules.md)** — Options, config, mkIf, mkMerge, and how jetpack-nixos wires hardware.
8. **[Pill 08: Flashing, Firmware & OTA Updates](08-flashing-firmware-ota.md)** — Flash scripts, UEFI, OP-TEE, capsule updates. The full boot chain.
9. **[Pill 09: Building Your Own Jetson System](09-building-your-system.md)** — From zero to running NixOS on Orin. Practical walkthrough.

## Quick Reference

### The jetpack-nixos Stack at a Glance

```text
┌─────────────────────────────────────────────────────┐
│                   YOUR CONFIG                        │
│  hardware.nvidia-jetpack.enable = true;             │
│  hardware.nvidia-jetpack.som = "orin-agx";          │
│  hardware.nvidia-jetpack.carrierBoard = "devkit";   │
└─────────────────────┬───────────────────────────────┘
                      │ NixOS module system
┌─────────────────────▼───────────────────────────────┐
│               modules/default.nix                    │
│  Wires: kernel, firmware, drivers, CUDA, services   │
│  Imports: cuda.nix, devices.nix, graphics.nix, ...  │
└─────────────────────┬───────────────────────────────┘
                      │ Applies overlays to nixpkgs
┌─────────────────────▼───────────────────────────────┐
│                overlay.nix                           │
│  Creates: nvidia-jetpack5/6/7 package scopes        │
│  Overrides: cudaPackages_11_4, _12_6, etc.          │
└─────────────────────┬───────────────────────────────┘
                      │ Per-version factory
┌─────────────────────▼───────────────────────────────┐
│               mk-overlay.nix                         │
│  Input: version tuple (JP version, L4T, CUDA)       │
│  Output: makeScope with all jetpack packages        │
│  Contains: kernel, L4T packages, flash-tools, etc.  │
└─────────────────────┬───────────────────────────────┘
                      │ Upstream sources
┌─────────────────────▼───────────────────────────────┐
│             sourceinfo/default.nix                   │
│  Reads: r36.4-debs.json, r36.4.4-gitrepos.json     │
│  Fetches: .deb files + git repos from NVIDIA        │
└─────────────────────────────────────────────────────┘
```

### Key Files at a Glance

| File | What It Does |
|------|-------------|
| `flake.nix` | Entry point. Exposes overlay, NixOS module, packages, flash scripts |
| `overlay.nix` | Creates `nvidia-jetpack{5,6,7}`, overrides `cudaPackages` |
| `mk-overlay.nix` | Factory: version params → package scope (`makeScope`) |
| `overlay-with-config.nix` | Injects NixOS config into packages (flash scripts, firmware) |
| `modules/default.nix` | NixOS module root. Wires kernel, drivers, firmware, services |
| `modules/cuda.nix` | Sets `cudaSupport = true`, `cudaCapabilities` per SoM |
| `modules/devices.nix` | Per-SoM config: nvpmodel, fan control, partition templates |
| `modules/graphics.nix` | EGL, Vulkan, X11/Wayland driver packages |
| `modules/flash-script.nix` | Flash options, firmware variants, secure boot |
| `sourceinfo/default.nix` | Fetches NVIDIA's .deb and git sources by version |
| `pkgs/l4t/` | All L4T runtime packages (GPU driver, camera, multimedia, etc.) |
| `pkgs/kernels/r{35,36,38}/` | Kernel builds for JetPack 5/6/7 |

### JetPack Generations

| JetPack | L4T | Kernel | CUDA | SoMs |
|---------|-----|--------|------|------|
| 5.1.5 | R35.6.2 | 5.10 | 11.4 | Xavier, Orin |
| 6.2.1 | R36.4.4 | 5.15 | 12.6 | Orin |
| 7.0 | R38.2.1 | 6.8 | 13.0 | Thor |

### Essential Commands

```bash
# Build & switch NixOS configuration
sudo nixos-rebuild switch --flake .#nixos --show-trace

# Flash firmware to Jetson (from x86_64 host)
nix build .#flash-orin-agx-devkit
sudo ./result/bin/flash-jetson

# Enter dev shell with CUDA
nix develop

# Check what JetPack version is active
cat /etc/nv_tegra_release

# Power model control
sudo nvpmodel -q    # Query current mode
sudo nvpmodel -m 0  # Set MAXN
```

## Philosophy

These pills follow the same principles as our TinyGrad Pills:

1. **First principles** — Start from the language, build up to the system
2. **tl;dr format** — Dense with information, zero fluff
3. **Show the code** — Every concept illustrated with real jetpack-nixos code
4. **Build intuition** — Once you see that Nix is "just functions and attribute sets", the whole stack clicks

## License

Same as jetpack-nixos (MIT)
