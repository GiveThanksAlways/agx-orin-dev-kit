# Pill 06: CUDA, cuDNN, TensorRT & GPU Packages

## tl;dr

CUDA on Jetson is different from CUDA on desktop. The driver is baked into L4T, the toolkit comes from NVIDIA's Jetson `.deb` repos (not the standard CUDA redistributables), and `cuda_compat` lets you run newer CUDA apps on older driver versions. jetpack-nixos integrates all of this by overriding nixpkgs' `cudaPackages` infrastructure to use Jetson-specific sources, while keeping the standard API surface so `pkgs.cudaPackages.cuda_cudart` "just works."

## CUDA on Jetson vs Desktop

### Desktop GPU (GeForce/Datacenter)

```
User App → libcuda.so (driver API) → nvidia.ko → PCIe GPU
                                        ↑
                                  Desktop kernel module
```

You install the driver separately. CUDA toolkit is independent. Everything is modular.

### Jetson (Tegra SoC)

```
User App → libcuda.so (l4t-cuda) → nvgpu.ko → Integrated GPU
                                      ↑
                                L4T kernel module (part of the kernel)
```

The GPU is **integrated into the SoC**. The driver is part of L4T. There's no separate "driver installer." The CUDA driver (`libcuda.so`) ships as an L4T package, and it must match the kernel module version exactly.

### Key Differences

| Aspect | Desktop | Jetson |
|--------|---------|--------|
| GPU | Discrete (PCIe) | Integrated (SoC) |
| Driver | Separate installer | Part of L4T |
| Kernel module | `nvidia.ko` | `nvgpu.ko` (JP5/6), `nvidia.ko` (JP7) |
| CUDA toolkit | nvidia.com/cuda | Jetson-specific `.deb` repos |
| Version coupling | Loose (driver/toolkit independent) | Tight (must match L4T version) |

## CUDA Version Mapping

Each JetPack version ships a specific CUDA version:

| JetPack | L4T | Native CUDA | cuda_compat Range |
|---------|-----|-------------|-------------------|
| 5.1.5 | R35.6.2 | **11.4** | Up to 12.2 |
| 6.2.1 | R36.4.4 | **12.6** | 12.4 – 12.9 |
| 7.0 | R38.2.1 | **13.0** | — |

### What Is `cuda_compat`?

NVIDIA provides a forward-compatibility layer. If your JetPack ships CUDA 11.4 but you want to run a CUDA 12.2 application:

```
App (compiled for CUDA 12.2)
    ↓
cuda_compat shim (translates 12.2 API → 11.4 driver)
    ↓
libcuda.so (11.4 driver)
    ↓
nvgpu.ko (L4T kernel)
```

`cuda_compat` is part of `l4t-cuda`:
```nix
# l4t-cuda includes:
# - libcuda.so (the CUDA driver)
# - cuda_compat/ (forward compatibility libraries)
```

## How `cudaPackages` Works in nixpkgs

Standard nixpkgs has a CUDA package set infrastructure:

```nix
pkgs.cudaPackages = {
  cuda_cudart;    # CUDA runtime
  cuda_nvcc;      # NVIDIA CUDA compiler
  cuda_cccl;      # CUDA C++ Core Libraries
  cublas;         # cuBLAS
  cudnn;          # cuDNN
  tensorrt;       # TensorRT
  nccl;           # Multi-GPU communication
  # ... 30+ packages
};

# Version-specific sets
pkgs.cudaPackages_11_4;
pkgs.cudaPackages_12_6;
pkgs.cudaPackages_13_0;
```

Each set is constructed from **manifests** — JSON data describing available packages, URLs, and hashes. nixpkgs downloads pre-built **CUDA redistributables** from NVIDIA's standard CDN.

## How jetpack-nixos Overrides `cudaPackages`

For Jetson (aarch64-linux), the standard redistributables don't work. jetpack-nixos replaces them with packages built from Jetson-specific `.deb` files.

### Step 1: Override the CUDA Package Sets

```nix
# From overlay.nix
cudaPackages_11_4 = prev.cudaPackages_11_4.override (prevArgs:
  if system == "aarch64-linux" then {
    # Use Jetson-specific CUDA 11.4 release
    manifests.cuda.release_label = "11.4.298";
  } else {
    # On x86_64, use cuDNN 8.6 to match JetPack 5's version
    manifests = prevArgs.manifests // {
      cudnn = final._cuda.manifests.cudnn."8.6.0";
    };
  });

# Set the default CUDA version for the system
cudaPackages = final.cudaPackages_11;
```

### Step 2: CUDA Extensions (Deb-Based Packages)

```nix
# From cuda-packages-11-4-extension.nix
# This replaces standard CUDA redist packages with Jetson deb-built versions

# The core mechanism:
debWrapBuildRedist = name: attrs:
  let
    # Find the existing package from nixpkgs' manifest
    original = prevCudaPackages.${name};
    
    # Find the matching .deb files
    debs = debsForSourcePackage name;
  in
  # Replace the source with extracted .deb contents
  original.overrideAttrs (old: {
    src = buildFromDebs { inherit name; srcs = debs; };
  });
```

Packages handled this way:
```nix
{
  # Standard CUDA packages, rebuilt from Jetson debs
  cuda_cudart         # CUDA runtime
  cuda_nvcc           # CUDA compiler
  cuda_nvrtc          # Runtime compilation
  cuda_nvml_dev       # NVIDIA Management Library
  cuda_profiler_api   # Profiling API
  cuda_cccl           # C++ Core Libraries
  cublas              # BLAS library
  cufft               # FFT library
  curand              # Random number generation
  cusolver            # Linear algebra solver
  cusparse            # Sparse matrix library
  libnpp              # Performance primitives
  # ... and more
}
```

### Step 3: Special Packages

Some packages need custom handling:

#### cuDNN

```nix
# cuDNN headers are in a separate deb
cudnn = prevBuildRedist.overrideAttrs {
  postPatch = ''
    # Symlink headers from the dev package
    ln -s ${cudnnDevHeaders}/include/* include/
  '';
};
```

#### TensorRT

TensorRT is particularly complex — it has both aarch64 (Jetson) and x86_64 (host tools) variants:

```nix
tensorrt = buildFromDebs {
  name = "tensorrt";
  srcs = [
    debs."libnvinfer8"
    debs."libnvinfer-dev"
    debs."libnvinfer-plugin8"
    debs."libnvonnxparsers8"
    debs."libnvparsers8"
  ];
  # Complex post-processing to organize headers and libraries
};

# Nsight tools — host-only (x86_64)
nsight_systems_host = ...;  # Profiling tool
nsight_compute_host = ...;  # Kernel analysis tool
```

#### DLA (Deep Learning Accelerator)

Jetson Orin has a dedicated DLA engine. `libcudla` exposes it:

```nix
libcudla = buildFromDebs {
  name = "libcudla";
  srcs = debsForSourcePackage "nvidia-l4t-dla-compiler";
};
# Only available on aarch64-linux (Jetson)
```

### Step 4: Per-JetPack TensorRT Versions

Different JetPack versions need specific TensorRT versions:

```nix
# From overlay.nix — per-CUDA-set TensorRT pinning
cudaPackages_12_6 = prev.cudaPackages_12_6.override (prevArgs: {
  manifests = prevArgs.manifests // {
    tensorrt = final._cuda.manifests.tensorrt."10.7.0";
  };
});
cudaPackages_12_8 = prev.cudaPackages_12_8.override (prevArgs: {
  manifests = prevArgs.manifests // {
    tensorrt = final._cuda.manifests.tensorrt."10.7.0";
  };
});
cudaPackages_13_0 = prev.cudaPackages_13_0.override (prevArgs: {
  manifests = prevArgs.manifests // {
    tensorrt = final._cuda.manifests.tensorrt."10.14.1";
  };
});
```

## The CUDA Capability System

Each Jetson SoC has a specific CUDA compute capability:

| SoC | Compute Capability | Marketing Name |
|-----|-------------------|----------------|
| T194 (Xavier) | sm_72 | Volta |
| T234 (Orin) | sm_87 | Ampere |
| T264 (Thor) | sm_110 | Blackwell |

jetpack-nixos sets this automatically based on your `som`:

```nix
# From modules/cuda.nix
nixpkgs.config = mkIf cfg.configureCuda {
  cudaSupport = true;
  cudaCapabilities =
    lib.optionals (isXavier || isGeneric) [ "7.2" ] ++
    lib.optionals (isOrin || isGeneric) [ "8.7" ] ++
    lib.optionals isThor [ "11.0" ];
};
```

When you build a CUDA application, `nvcc` compiles kernels only for your SoC's architecture. No wasted time compiling for GPUs you don't have.

### `pkgsForCudaArch`

nixpkgs provides architecture-specific package sets:

```nix
# From the jetpack-nixos flake.nix
legacyPackages = {
  nvidia-jetpack6 = pkgs.pkgsForCudaArch.sm_87.nvidia-jetpack6;  # Orin
  nvidia-jetpack7 = pkgs.pkgsForCudaArch.sm_110.nvidia-jetpack7; # Thor
};
```

This ensures CUDA packages are compiled with the right `-arch=sm_XX` flags.

## Using CUDA in Your Packages

### In a NixOS Configuration

```nix
# configuration.nix
{
  hardware.nvidia-jetpack.configureCuda = true;  # Enables cudaSupport globally
  
  environment.systemPackages = with pkgs; [
    # These automatically build with CUDA since cudaSupport = true
    opencv4
    ffmpeg
    llama-cpp   # Gets CUDA backend
  ];
}
```

### In a Dev Shell

```nix
# flake.nix
devShells.aarch64-linux.default = pkgs.mkShell {
  packages = with pkgs.nvidia-jetpack.cudaPackages; [
    cuda_cudart
    cuda_nvcc
    cublas
    cudnn
    tensorrt
  ];
  
  shellHook = ''
    export CUDA_PATH=${pkgs.nvidia-jetpack.cudaPackages.cuda_cudart}
    export CUDNN_PATH=${pkgs.nvidia-jetpack.cudaPackages.cudnn}
  '';
};
```

### Building a CUDA Application

```nix
# my-cuda-app.nix
{ stdenv, cudaPackages, cmake }:
stdenv.mkDerivation {
  pname = "my-cuda-app";
  version = "1.0";
  
  src = ./.;
  
  nativeBuildInputs = [ cmake cudaPackages.cuda_nvcc ];
  buildInputs = [
    cudaPackages.cuda_cudart
    cudaPackages.cublas
    cudaPackages.cudnn
  ];
  
  cmakeFlags = [
    "-DCUDA_TOOLKIT_ROOT_DIR=${cudaPackages.cuda_cudart}"
    "-DCMAKE_CUDA_ARCHITECTURES=87"  # Orin
  ];
}
```

## The `_cuda` Extension System

nixpkgs provides an extension mechanism for CUDA packages:

```nix
# From overlay.nix
_cuda = prev._cuda.extend (_: prevCuda: {
  extensions = prevCuda.extensions ++ [
    # General Jetson CUDA extensions
    (import ./pkgs/cuda-extensions { inherit (final) lib; })

    # CUDA 11.4 specific (deb-based replacements)
    (import ./cuda-packages-11-4-extension.nix { inherit (final) lib; inherit system; })
  ];
});
```

Extensions are applied automatically to all `cudaPackages_*` sets. They can:
- Replace packages (deb-built versions instead of redist)
- Add new packages (DLA, Jetson-specific tools)
- Modify existing packages (patch for Tegra compatibility)

## The Relationship Between nvidia-jetpack and cudaPackages

```
pkgs.nvidia-jetpack.cudaPackages
  ↑
  │ Points to the version-matched CUDA set
  │
pkgs.cudaPackages (= pkgs.cudaPackages_11 for JP5)
  ↑
  │ Overridden by overlay.nix
  │
pkgs.cudaPackages_11_4
  ↑
  │ Modified by cuda-packages-11-4-extension.nix
  │ (replaces redist packages with deb-extracted Jetson versions)
  │
nixpkgs standard cudaPackages_11_4
  ↑
  │ Standard nixpkgs manifest-based CUDA packages
  │
NVIDIA CUDA redistributable manifests
```

## Version Selection Logic

The default `nvidia-jetpack` and `cudaPackages` are selected based on the CUDA version:

```nix
# From overlay.nix
nvidia-jetpack =
  if final.cudaPackages.cudaOlder "12.3" then final.nvidia-jetpack5    # CUDA 11.4
  else if final.cudaPackages.cudaOlder "13.0" then final.nvidia-jetpack6  # CUDA 12.6
  else final.nvidia-jetpack7;                                             # CUDA 13.0

# The NixOS module then pins both:
nvidia-jetpack = final."nvidia-jetpack${cfg.majorVersion}";
cudaPackages = final."cudaPackages_${majorVersion}";
```

This ensures CUDA packages and JetPack version always stay in sync.

## Practical Commands

```bash
# Check CUDA version
nvcc --version
cat /usr/local/cuda/version.txt  # Traditional path
nvidia-smi  # Available on JP6+ (l4t-nvml)

# Check compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# List CUDA packages available
nix eval '.#legacyPackages.aarch64-linux.nvidia-jetpack.cudaPackages' \
  --apply 'x: builtins.attrNames x'

# Build a specific CUDA package
nix build '.#legacyPackages.aarch64-linux.nvidia-jetpack.cudaPackages.cublas'

# Check which CUDA your system uses
cat /etc/nv_tegra_release
```

## Summary

| Concept | What | Key Details |
|---------|------|-------------|
| CUDA on Jetson | Integrated GPU, tight L4T coupling | Driver = `l4t-cuda`, not separate |
| `cuda_compat` | Forward compatibility shim | Newer CUDA apps on older drivers |
| `cudaPackages` | nixpkgs CUDA package set | Overridden with Jetson `.deb` packages |
| `cudaCapabilities` | Compute capability (sm_XX) | Auto-set from SoM (7.2, 8.7, 11.0) |
| Deb-to-Nix CUDA | Replace redist with debs | `cuda-packages-11-4-extension.nix` |
| TensorRT | Inference optimizer | Version-matched per JetPack |
| cuDNN | Deep learning primitives | From Jetson deb repos |
| `_cuda.extensions` | Package set customization | How jetpack-nixos hooks into nixpkgs |

**The mental model**: nixpkgs has a generic CUDA infrastructure built around redistributable packages from nvidia.com. Jetson uses different packages — `.deb` files from NVIDIA's Jetson-specific repos. jetpack-nixos intercepts the `cudaPackages` infrastructure, replaces the package sources with Jetson debs, pins the versions to match L4T, and auto-configures compute capabilities. The result: `pkgs.cudaPackages.cublas` transparently returns the Jetson-specific cuBLAS, compiled for your Orin's sm_87 GPU.

---

**Previous**: [← Pill 05: BSP, Kernel & L4T Packages](05-bsp-kernel-l4t.md)
**Next**: [Pill 07: NixOS Modules — Declaring Your System →](07-nixos-modules.md)
