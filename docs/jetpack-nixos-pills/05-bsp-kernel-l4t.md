# Pill 05: BSP, Kernel & L4T Packages

## tl;dr

The **BSP** (Board Support Package) is NVIDIA's tarball containing flash tools, bootloader, and partition layouts. The **kernel** is built from NVIDIA's git repos with Jetson-specific configs. **L4T packages** are NVIDIA's proprietary libraries (GPU driver, multimedia, camera) extracted from `.deb` files and patched for the Nix store. These three form the foundation that everything else builds on.

## What Is L4T?

**Linux for Tegra** (L4T) is NVIDIA's Linux distribution for Jetson. It's what JetPack is built on:

```
JetPack SDK
├── L4T (Linux for Tegra)           ← The OS layer
│   ├── Kernel (NVIDIA-patched)
│   ├── Bootloader (UEFI, CBoot)
│   ├── Drivers (GPU, multimedia, camera)
│   ├── Firmware blobs
│   └── Flash tools
├── CUDA Toolkit
├── cuDNN
├── TensorRT
└── Samples & tools
```

L4T is versioned independently from JetPack. The mapping:

| JetPack | L4T | Kernel | SoC Family |
|---------|-----|--------|------------|
| 5.1.5 | R35.6.2 | 5.10.216 | Xavier + Orin (T194/T234) |
| 6.2.1 | R36.4.4 | 5.15.148 | Orin (T234) |
| 7.0 | R38.2.1 | 6.8.12 | Thor (T264) |

## The Source Pinning System

### How NVIDIA Sources Are Tracked

Every NVIDIA binary and git repo is tracked in JSON manifest files under `sourceinfo/`:

```
sourceinfo/
├── default.nix              # Manifest loader
├── r35.6-debs.json          # JP5 .deb hashes
├── r35.6.2-gitrepos.json    # JP5 git repo hashes
├── r36.4-debs.json          # JP6 .deb hashes
├── r36.4.4-gitrepos.json    # JP6 git repo hashes
├── r38.2-debs.json          # JP7 .deb hashes
├── r38.2.1-gitrepos.json    # JP7 git repo hashes
├── debs-update.py           # Script to refresh deb hashes
└── gitrepos-update.py       # Script to refresh git hashes
```

### Deb Manifest Format

```json
{
  "common": {
    "nvidia-l4t-3d-core": {
      "source": "nvidia-l4t-3d-core",
      "version": "36.4.4-20250107105206",
      "src": {
        "url": "https://repo.download.nvidia.com/jetson/common/pool/main/n/nvidia-l4t-3d-core/nvidia-l4t-3d-core_36.4.4-20250107105206_arm64.deb",
        "hash": "sha256:XXXXX..."
      }
    }
  },
  "t234": { ... },
  "t194": { ... }
}
```

Three repos per version: `common` (shared), `t234` (Orin-specific), `t194` (Xavier-specific).

### Git Repos Manifest Format

```json
{
  "kernel-oot": {
    "url": "https://nv-tegra.nvidia.com/r/linux-nv-oot",
    "rev": "jetson_36.4.4",
    "hash": "sha256:YYYY..."
  },
  "kernel-devicetree": {
    "url": "https://nv-tegra.nvidia.com/r/device-tree/...",
    "rev": "jetson_36.4.4",
    "hash": "sha256:ZZZZ..."
  }
}
```

### `sourceinfo/default.nix`: The Loader

```nix
{ l4tMajorMinorPatchVersion, lib, fetchurl, fetchgit }:
let
  # Select the right manifest based on L4T version
  debsJson = builtins.fromJSON (builtins.readFile
    ./r${lib.versions.majorMinor l4tMajorMinorPatchVersion}-debs.json);
  gitReposJson = builtins.fromJSON (builtins.readFile
    ./r${l4tMajorMinorPatchVersion}-gitrepos.json);
in {
  # Convert JSON to fetchurl calls
  debs = lib.mapAttrs (repo: pkgs:
    lib.mapAttrs (name: pkg: pkg // {
      src = fetchurl { inherit (pkg.src) url hash; };
    }) pkgs
  ) debsJson;

  # Convert JSON to fetchgit calls
  gitRepos = lib.mapAttrs (name: repo:
    fetchgit { inherit (repo) url rev hash; }
  ) gitReposJson;
}
```

**Key insight**: Every single upstream source is pinned by content hash. `nix build` will fail if ANY binary changes upstream. Reproducibility is enforced at the data layer.

## The BSP (Board Support Package)

### What's in the BSP?

The BSP tarball (`Jetson_Linux_R36.4.4_aarch64.tbz2`) unpacks to `Linux_for_Tegra/`:

```
Linux_for_Tegra/
├── bootloader/               # Bootloader binaries, partition configs
│   ├── flash.sh              # The primary flash script
│   ├── generic/cfg/          # Partition templates per board
│   │   ├── flash_t234_qspi.xml
│   │   └── flash_t234_qspi_sdmmc.xml
│   ├── tegrabct               # Boot configuration tools
│   └── *.bin, *.dtb           # Firmware blobs
├── flash.sh                   # Symlink to bootloader/flash.sh
├── kernel/                    # Kernel Image, DTBs (pre-built)
├── nv_tegra/                  # Driver packages for Ubuntu
├── rootfs/                    # Default Ubuntu rootfs (unused by NixOS)
└── tools/                     # Utility scripts
```

### How jetpack-nixos Packages the BSP

```nix
# From mk-overlay.nix
bspSrc = final.applyPatches {
  src = final.runCommand "l4t-unpacked" {
    src = final.fetchurl {
      url = "https://developer.download.nvidia.com/embedded/L4T/r${major}_Release_v${minor}.${patch}/release/Jetson_Linux_R${l4tVersion}_aarch64.tbz2";
      hash = bspHash;  # Pinned in overlay.nix per JP version
    };
    nativeBuildInputs = [ final.buildPackages.bzip2_1_1 ];
  } ''
    bzip2 -d -c $src | tar xf -
    mv Linux_for_Tegra $out
  '';
  patches = bspPatches;    # Version-specific patches (e.g., pkgs/r38-bsp.patch)
  postPatch = bspPostPatch; # Copy overlay files (MB1 BCT updates)
};
```

The BSP is:
1. **Fetched** with a pinned hash
2. **Unpacked** with bzip2
3. **Patched** (fixes for NixOS compatibility)
4. **Used by**: flash-tools, UEFI firmware builds, partition templates

## The Kernel

### Kernel Build Overview

jetpack-nixos builds NVIDIA's kernel from source, not from the BSP pre-built image:

```nix
# Simplified from pkgs/kernels/r36/default.nix
{ buildLinux, ... }:
buildLinux {
  version = "5.15.148";
  src = gitRepos."kernel-oot";   # NVIDIA's kernel source (git)
  
  kernelPatches = [
    # NixOS compatibility patches
  ];
  
  structuredExtraConfig = with lib.kernel; {
    # Tegra-specific kernel configs
    ARCH_TEGRA = yes;
    TEGRA_BPMP = yes;
    TEGRA_HOST1X = yes;
    PCIE_TEGRA194 = yes;
    
    # GPU support
    DRM_TEGRA = module;
    
    # Storage
    MMC_SDHCI_TEGRA = yes;
    NVME = yes;
    
    # ... hundreds more
  };
}
```

### Kernel Versions by Release

| Directory | L4T Release | Kernel | Features |
|-----------|-------------|--------|----------|
| `pkgs/kernels/r35/` | R35.x (JP5) | 5.10.216 | Xavier + Orin, optional RT, display driver OOT |
| `pkgs/kernels/r36/` | R36.x (JP6) | 5.15.148 | Orin only, OOT modules, separate devicetree |
| `pkgs/kernels/r38/` | R38.x (JP7) | 6.8.12 | Thor, Open RM driver |

### Out-of-Tree (OOT) Kernel Modules

Starting with JP6, NVIDIA ships some drivers as out-of-tree modules:

```nix
# From mk-overlay.nix
kernelPackagesOverlay = final: _:
  if self.l4tAtLeast "36" then {
    # These build as kernel modules against the running kernel
    devicetree = final.callPackage ./pkgs/kernels/r${l4tMajorVersion}/devicetree.nix { ... };
    nvidia-oot-modules = final.callPackage ./pkgs/kernels/r${l4tMajorVersion}/oot-modules.nix { ... };
  } else {
    nvidia-display-driver = final.callPackage ./pkgs/kernels/r${l4tMajorVersion}/display-driver.nix { ... };
  };
```

These are loaded via `boot.extraModulePackages` in the NixOS module:

```nix
boot.extraModulePackages = lib.optional (jetpackAtLeast "6")
  config.boot.kernelPackages.nvidia-oot-modules;
```

### Real-Time Kernel

Each kernel version has an RT (PREEMPT_RT) variant:

```nix
rtkernel = self.callPackage ./pkgs/kernels/r${l4tMajorVersion} {
  kernelPatches = [];
  realtime = true;  # Applies PREEMPT_RT patches
};
rtkernelPackages = final.linuxPackagesFor self.rtkernel;
```

Enable with:
```nix
hardware.nvidia-jetpack.kernel.realtime = true;
```

## L4T Packages: The Deb-to-Nix Pipeline

### The Process

NVIDIA distributes Jetson-specific libraries as `.deb` files. jetpack-nixos converts them:

```
Step 1: fetchurl (with pinned hash)
  → /nix/store/hash-nvidia-l4t-3d-core_36.4.4_arm64.deb

Step 2: dpkg -x (extract)
  → /usr/lib/aarch64-linux-gnu/libEGL_nvidia.so.0
  → /usr/lib/aarch64-linux-gnu/libGLESv2_nvidia.so.2
  → /etc/...
  → /usr/share/...

Step 3: Reorganize for Nix
  → $out/lib/libEGL_nvidia.so.0
  → $out/lib/libGLESv2_nvidia.so.2

Step 4: autoPatchelf
  → Rewrites ELF RPATH from /usr/lib/... to /nix/store/...
  → Fixes all library dependencies to point to Nix store paths
```

### `buildFromDebs`: The Helper Function

```nix
# pkgs/buildFromDebs.nix
{ stdenv, dpkg, autoPatchelfHook, ... }:
{ name, version, srcs, ... }:
stdenv.mkDerivation {
  inherit name version;
  
  srcs = map (pkg: pkg.src) srcs;  # The .deb files
  
  nativeBuildInputs = [ dpkg autoPatchelfHook ];
  
  unpackPhase = ''
    for src in $srcs; do
      dpkg -x $src .
    done
  '';
  
  installPhase = ''
    # Move extracted files to $out
    # Remove Ubuntu-specific paths
    # autoPatchelf runs in fixupPhase automatically
  '';
}
```

### The L4T Package Catalog

All packages are in `pkgs/l4t/`, auto-discovered by `packagesFromDirectoryRecursive`:

```
pkgs/l4t/
├── l4t-3d-core.nix          # GPU driver: EGL, Vulkan, OpenGL ES, DRM
├── l4t-camera.nix            # Camera: Argus, libcamera, nvarguscamerasrc
├── l4t-core.nix              # Core runtime: libnvrm, libnvos, libtegra_soc
├── l4t-cuda.nix              # CUDA driver: libcuda.so, cuda_compat
├── l4t-cupva.nix             # CuPVA: NVIDIA's programmable vision accelerator
├── l4t-dla-compiler.nix      # DLA: Deep Learning Accelerator compiler (JP6)
├── l4t-firmware.nix          # Firmware blobs for boot, USB, etc.
├── l4t-gbm.nix               # GBM: Generic Buffer Management for Wayland
├── l4t-gstreamer.nix         # GStreamer NVIDIA plugins: nvvidconv, nvjpegdec, etc.
├── l4t-init.nix              # udev rules, init scripts
├── l4t-multimedia.nix        # V4L2: hardware codec, video/audio processing
├── l4t-nvfancontrol.nix      # Fan control daemon + config files
├── l4t-nvml.nix              # NVIDIA Management Library (nvidia-smi)
├── l4t-nvpmodel.nix          # Power model profiles per SoM
├── l4t-nvsci.nix             # NvSci: IPC framework for autonomous machines
├── l4t-opencv.nix            # OpenCV with CUDA acceleration
├── l4t-pva.nix               # PVA: Programmable Vision Accelerator runtime
├── l4t-tools.nix             # Utilities: jetson_clocks, tegrastats, etc.
├── l4t-wayland.nix           # Wayland protocol extensions
└── l4t-xusb-firmware.nix     # USB firmware (pre-JP7)
```

### Key L4T Packages in Detail

#### `l4t-3d-core` — The GPU Driver

The most critical L4T package. Provides:
- `libEGL_nvidia.so.0` — EGL implementation
- `libGLESv2_nvidia.so.2` — OpenGL ES
- `libvulkan_nvidia.so` — Vulkan driver
- `nvidia_drv.so` — X11 driver
- `libnvidia-glcore.so` — OpenGL core

Used as `hardware.graphics.package` in the NixOS module.

#### `l4t-cuda` — CUDA Driver Integration

Provides `libcuda.so` (the CUDA driver API) and `cuda_compat` (forward compatibility for running newer CUDA on older JetPack):

```nix
# cuda_compat allows CUDA 12.x apps to run on JetPack 5's CUDA 11.4 driver
# by shimming the driver API
```

#### `l4t-firmware` — Firmware Blobs

Kernel firmware files needed at boot:
- Tegra BPMP firmware
- Audio/display/power firmware
- Placed in `/lib/firmware/` at boot

#### `l4t-tools` — System Utilities

```bash
jetson_clocks     # Lock all clocks to maximum
tegrastats        # Real-time SoC monitoring (GPU%, CPU%, thermal, power)
nvpmodel          # Power/performance profile switching
```

## Version-Gating: `l4tAtLeast` and `l4tOlder`

L4T packages are conditionally included based on version:

```nix
# From mk-overlay.nix — these helpers are created per JP version
l4tAtLeast = versionAtLeast l4tMajorMinorPatchVersion;  # e.g., l4tAtLeast "36"
l4tOlder = versionOlder l4tMajorMinorPatchVersion;       # e.g., l4tOlder "38"
```

Used to conditionally include packages:

```nix
# From pkgs/l4t/default.nix
{
  # Available in all versions
  inherit l4t-3d-core l4t-camera l4t-core l4t-cuda l4t-firmware;

  # JP6 only
  l4t-dla-compiler = lib.optionalAttrs (l4tAtLeast "36" && l4tOlder "38") { ... };

  # JP6+
  l4t-nvml = lib.optionalAttrs (l4tAtLeast "36") { ... };
  nvidia-smi = self.l4t-nvml;  # Alias

  # JP7+
  driverDebs = lib.optionalAttrs (l4tAtLeast "38") { ... };
  l4t-firmware-openrm = lib.optionalAttrs (l4tAtLeast "38") { ... };
}
```

And in the NixOS module:

```nix
hardware.firmware = with pkgs.nvidia-jetpack; [
  l4t-firmware
] ++ lib.optionals (lib.versionOlder cfg.majorVersion "7") [
  cudaPackages.vpi-firmware  # Not needed on JP7
] ++ lib.optionals (l4tOlder "38") [
  l4t-xusb-firmware          # USB firmware: pre-JP7 only
];
```

## `dlopenOverride`: Fixing Library Loading

Some NVIDIA libraries use `dlopen("libfoo.so")` with hardcoded paths. On NixOS, libraries aren't in `/usr/lib`, so this fails. jetpack-nixos uses `LD_PRELOAD` to intercept `dlopen` calls:

```nix
# pkgs/dlopen-override/
# A small C library that intercepts dlopen() and redirects
# library paths to the Nix store
dlopenOverride = final.callPackage ./pkgs/dlopen-override { };
```

This ensures NVIDIA's proprietary libraries can find each other in the Nix store.

## Updating Source Manifests

When NVIDIA releases a new JetPack:

```bash
# Update deb manifests
cd sourceinfo
python3 debs-update.py r36.4      # Scrapes NVIDIA's repo, updates hashes
python3 gitrepos-update.py r36.4.4 # Updates git repo revisions + hashes
```

This produces updated JSON files with fresh hashes. Commit them, and `nix build` uses the new sources.

## Summary

| Component | Source | Packaging Method |
|-----------|--------|-----------------|
| BSP | NVIDIA tarball (pinned hash) | `fetchurl` → `tar xf` → `applyPatches` |
| Kernel | NVIDIA git repos (pinned rev) | `fetchgit` → `buildLinux` |
| L4T packages | NVIDIA .deb files (pinned hashes) | `fetchurl` → `dpkg -x` → `autoPatchelf` |
| CUDA toolkit | See Pill 06 | Mixed (deb + redist) |
| Firmware | Inside `.deb` files | Extract → `/lib/firmware/` |

**The mental model**: NVIDIA provides three types of sources — a BSP tarball, git repos, and `.deb` files. jetpack-nixos pins ALL of them with content hashes in JSON manifests, fetches them reproducibly, extracts/builds them into Nix store paths, and patches ELF binaries to work outside of Ubuntu's `/usr/lib`. The result: NVIDIA's proprietary stack, running on NixOS, with full reproducibility guarantees.

---

**Previous**: [← Pill 04: jetpack-nixos — The 30,000ft View](04-jetpack-nixos-architecture.md)
**Next**: [Pill 06: CUDA, cuDNN, TensorRT & GPU Packages →](06-cuda-gpu-packages.md)
