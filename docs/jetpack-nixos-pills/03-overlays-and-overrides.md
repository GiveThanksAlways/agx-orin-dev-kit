# Pill 03: Overlays & Overrides — Layering the Package Set

## tl;dr

An **overlay** is a function `final: prev: { ... }` that modifies the nixpkgs package set. It's how jetpack-nixos injects `nvidia-jetpack`, overrides `cudaPackages`, and adds Jetson-specific kernel builds into nixpkgs — without forking nixpkgs. Overlays are the layering mechanism that makes the whole stack work.

## The Core Concept

nixpkgs is a giant attribute set of ~80,000+ packages. An overlay is a function that takes that set and returns modifications:

```nix
# An overlay is a function with two arguments:
final: prev: {
  # prev = the package set BEFORE this overlay
  # final = the package set AFTER ALL overlays (the fixed point)
  
  my-package = prev.my-package.override { someFlag = true; };
  new-package = final.callPackage ./new-package.nix { };
}
```

### `final` vs `prev` (The Fixed Point)

This is the most confusing part of Nix. Let's make it clear:

```nix
overlay = final: prev: {
  # prev.python3  → Python BEFORE our overlay modified anything
  # final.python3 → Python AFTER all overlays (including ours) are applied
  
  # Use prev when you want the UNMODIFIED version
  my-python = prev.python3;
  
  # Use final when you want the FULLY-RESOLVED version (after all overlays)
  my-app = final.callPackage ./app.nix { };
  # This would use final.python3, which might be modified by another overlay
};
```

**Rule of thumb**:
- **`prev`**: Use when overriding something (you need the original)
- **`final`**: Use when building something new (you want the latest version of dependencies)

### Why It's Called "Fixed Point"

Overlays use a mathematical concept called a **fixed point**. `final` is the result of applying ALL overlays simultaneously, including itself. Nix resolves this lazily:

```
prev (original nixpkgs)
  ↓ overlay 1 applied
  ↓ overlay 2 applied
  ↓ overlay 3 applied
final (all overlays applied)
```

But `final` in overlay 1 CAN reference things added by overlay 3. Nix's laziness makes this work without explicitly ordering the resolution.

## How jetpack-nixos Applies Overlays

### Layer 1: The Main Overlay (`overlay.nix`)

```nix
# overlay.nix
final: prev: {
  # Creates three JetPack package scopes — one per generation
  nvidia-jetpack5 = import ./mk-overlay.nix {
    jetpackMajorMinorPatchVersion = "5.1.5";
    l4tMajorMinorPatchVersion = "35.6.2";
    cudaMajorMinorPatchVersion = "11.4.298";
    # ...
  } final prev;

  nvidia-jetpack6 = import ./mk-overlay.nix {
    jetpackMajorMinorPatchVersion = "6.2.1";
    l4tMajorMinorPatchVersion = "36.4.4";
    cudaMajorMinorPatchVersion = "12.6.10";
    # ...
  } final prev;

  nvidia-jetpack7 = import ./mk-overlay.nix {
    jetpackMajorMinorPatchVersion = "7.0";
    l4tMajorMinorPatchVersion = "38.2.1";
    cudaMajorMinorPatchVersion = "13.0.2";
    # ...
  } final prev;

  # The "default" nvidia-jetpack is selected by CUDA version
  nvidia-jetpack =
    if final.cudaPackages.cudaOlder "12.3" then final.nvidia-jetpack5
    else if final.cudaPackages.cudaOlder "13.0" then final.nvidia-jetpack6
    else final.nvidia-jetpack7;

  # Override CUDA package sets for Jetson
  cudaPackages_11_4 = prev.cudaPackages_11_4.override (...);
  cudaPackages_12_6 = prev.cudaPackages_12_6.override (...);
  cudaPackages = final.cudaPackages_11;
}
```

### Layer 2: Composition in the Flake

```nix
# From flake.nix
overlays.default = final: prev:
  if prev."jetpack-nixos-overlay-applied-${self.narHash}" or false then
    { }   # Guard: don't apply twice
  else
    nixpkgs.lib.composeManyExtensions [
      cuda-legacy.overlays.default    # Layer 1: CUDA legacy packages
      (import ./overlay.nix)          # Layer 2: jetpack-nixos packages
      (_: _: { "jetpack-nixos-overlay-applied-${self.narHash}" = true; })  # Layer 3: mark applied
    ] final prev;
```

`composeManyExtensions` takes a list of overlays and combines them into one:

```nix
# Equivalent to manually chaining:
# overlay3(overlay2(overlay1(nixpkgs)))
# But with proper final/prev threading
```

### Layer 3: Config-Dependent Overlay (`overlay-with-config.nix`)

Some packages need NixOS config values (like which SoM you're using). This overlay injects those:

```nix
# overlay-with-config.nix — applied by the NixOS module
config:
final: prev: {
  nvidia-jetpack = prev.nvidia-jetpack.overrideScope (finalJetpack: prevJetpack: {
    # Now packages can see the NixOS config
    socType =
      if lib.hasPrefix "orin-" cfg.som then "t234"
      else if lib.hasPrefix "xavier-" cfg.som then "t194"
      else throw "Unknown SoC type";

    # Override UEFI firmware with user's boot logo, debug settings
    uefi-firmware = prevJetpack.uefi-firmware.override {
      bootLogo = cfg.firmware.uefi.logo;
      debugMode = cfg.firmware.uefi.debugMode;
    };

    # Override flash-tools with user's patches
    flash-tools = prevJetpack.flash-tools.overrideAttrs ({ patches ? [], ... }: {
      patches = patches ++ cfg.flashScriptOverrides.patches;
    });
  });
}
```

### The Full Overlay Stack

```
nixpkgs (original ~80K packages)
  │
  ├── cuda-legacy overlay          Adds CUDA 11.4 packages
  ├── overlay.nix                  Adds nvidia-jetpack{5,6,7}, overrides cudaPackages
  ├── applied marker               Prevents double-application
  ├── version selection overlay    Sets nvidia-jetpack = jetpack{5,6,7} based on config
  └── overlay-with-config.nix      Injects NixOS config (SoM, firmware options)
  │
  ▼
final nixpkgs (original + all JetPack packages + config-specific overrides)
```

## `overrideScope`: Modifying Package Scopes

`nvidia-jetpack` is a **scope** (created by `makeScope`). You can modify it with `overrideScope`:

```nix
# Override a package inside the nvidia-jetpack scope
nvidia-jetpack = prev.nvidia-jetpack.overrideScope (finalJetpack: prevJetpack: {
  # Add new package
  socType = "t234";

  # Modify existing package
  flash-tools = prevJetpack.flash-tools.overrideAttrs (old: {
    patches = old.patches ++ [ ./my-patch.patch ];
  });
});
```

This is how `overlay-with-config.nix` works — it takes the base `nvidia-jetpack` scope and overlays device-specific overrides onto it.

### Scope vs Top-Level Overlay

| Mechanism | Scope | What It Modifies |
|-----------|-------|-----------------|
| Top-level overlay | `final: prev: { ... }` | The global nixpkgs package set |
| `overrideScope` | `finalScope: prevScope: { ... }` | A package scope (like `nvidia-jetpack`) |

They have the same `final/prev` semantics but operate at different levels.

## The Three Override Mechanisms

### 1. `.override` — Change Function Arguments

Modifies the arguments passed to `callPackage`:

```nix
# Original: callPackage ./cuda.nix { }
# cuda.nix takes { stdenv, fetchurl, gcc ? pkgs.gcc12 }

# Override gcc version:
cuda = prev.cuda.override { gcc = final.gcc13; };

# Override CUDA package set manifests:
cudaPackages_12_6 = prev.cudaPackages_12_6.override (prevArgs: {
  manifests = prevArgs.manifests // {
    tensorrt = final._cuda.manifests.tensorrt."10.7.0";
  };
});
```

### 2. `.overrideAttrs` — Change Derivation Attributes

Modifies the build recipe (phases, sources, patches):

```nix
# Add patches to flash-tools
flash-tools = prevJetpack.flash-tools.overrideAttrs ({ patches ? [], postPatch ? "", ... }: {
  patches = patches ++ cfg.flashScriptOverrides.patches;
  postPatch = postPatch + cfg.flashScriptOverrides.postPatch;
});
```

### 3. Overlay — Replace or Add Top-Level Packages

```nix
# Replace an entire package
final: prev: {
  nvidia-jetpack = prev.nvidia-jetpack.overrideScope (...);
  cudaPackages = final.cudaPackages_12_6;  # Switch default CUDA version
};
```

### When to Use What

| Want to... | Use |
|-----------|-----|
| Change a build dependency | `.override` |
| Add a patch or modify build flags | `.overrideAttrs` |
| Replace a package entirely | Overlay |
| Modify packages inside a scope | `.overrideScope` |
| Add a new package to nixpkgs | Overlay |

## `lib.composeManyExtensions`: Combining Overlays

```nix
lib.composeManyExtensions [
  overlay1   # Applied first
  overlay2   # Applied second
  overlay3   # Applied third
]
# Returns: a single overlay that applies all three in order
```

**Order matters for `prev`** (earlier overlays' changes are visible in later overlays' `prev`). But `final` is the same everywhere (the unified result).

## Practical Example: Adding Your Own Package

```nix
# In your flake.nix
{
  outputs = { self, nixpkgs, jetpack, ... }: {
    nixosConfigurations.my-jetson = nixpkgs.lib.nixosSystem {
      modules = [
        jetpack.nixosModules.default
        ./configuration.nix
        
        # Add a custom overlay
        ({ ... }: {
          nixpkgs.overlays = [
            (final: prev: {
              # Your custom CUDA app, using jetpack's CUDA
              my-cuda-app = final.callPackage ./my-cuda-app.nix {
                inherit (final.nvidia-jetpack.cudaPackages) cuda_cudart cuda_nvcc;
              };
            })
          ];
        })
      ];
    };
  };
}
```

## How `cudaPackages` Gets Wired

This is a key example of overlays in action. jetpack-nixos needs CUDA packages built from NVIDIA's Jetson .deb files instead of the standard x86 redistributables:

```nix
# Step 1: overlay.nix overrides cudaPackages_11_4
cudaPackages_11_4 = prev.cudaPackages_11_4.override (prevArgs:
  if system == "aarch64-linux" then {
    manifests.cuda.release_label = "11.4.298";
  } else { ... });

# Step 2: overlay.nix sets the default CUDA to 11.4
cudaPackages_11 = final.cudaPackages_11_4;
cudaPackages = final.cudaPackages_11;

# Step 3: cuda-packages-11-4-extension.nix replaces individual packages
# with deb-extracted versions for aarch64-linux
_cuda = prev._cuda.extend (_: prevCuda: {
  extensions = prevCuda.extensions ++ [
    (import ./pkgs/cuda-extensions { ... })
    (import ./cuda-packages-11-4-extension.nix { ... })
  ];
});
```

The result: when you write `pkgs.cudaPackages.cuda_cudart` on your Jetson, you get NVIDIA's Jetson-specific CUDA runtime extracted from a `.deb` file, not the generic x86 binary.

## The Guard Pattern

```nix
overlays.default = final: prev:
  if prev."jetpack-nixos-overlay-applied-${self.narHash}" or false
  then { }
  else { ... };
```

Overlays can accidentally be applied twice (e.g., both by your config and by a dependency). The guard pattern uses a marker attribute to detect this and short-circuit:

1. First application: marker doesn't exist → apply overlay, set marker
2. Second application: marker exists → return `{}` (no-op)

The `self.narHash` in the marker name ensures different versions of jetpack-nixos don't interfere.

## Debugging Overlays

```bash
# See what packages an overlay adds/modifies
nix eval '.#legacyPackages.aarch64-linux.nvidia-jetpack' --apply 'x: builtins.attrNames x'

# Check if your overlay is applied
nix eval '.#legacyPackages.aarch64-linux.nvidia-jetpack.l4tMajorMinorPatchVersion'

# Trace evaluation issues
nix build .#package --show-trace

# Check overlay order
nix eval '.#nixosConfigurations.nixos.config.nixpkgs.overlays' --apply 'x: builtins.length x'
```

## Summary

| Concept | What | jetpack-nixos Usage |
|---------|------|-------------------|
| Overlay | `final: prev: { ... }` | Injects nvidia-jetpack into nixpkgs |
| `final` | Package set after all overlays | Building new packages |
| `prev` | Package set before this overlay | Overriding existing packages |
| `composeManyExtensions` | Combine multiple overlays | cuda-legacy + overlay.nix + marker |
| `.override` | Change function args | Switch CUDA manifests/versions |
| `.overrideAttrs` | Change build recipe | Add flash-tools patches |
| `overrideScope` | Modify package scope | Device-specific overrides |
| Guard pattern | Prevent double application | `"applied-${narHash}"` marker |

**The mental model**: nixpkgs is a **base layer**. Each overlay is a **transparent sheet** laid on top that modifies what's visible. jetpack-nixos applies several sheets: CUDA legacy packages, Jetson packages, version selection, device configuration. The final result is a package set where `pkgs.nvidia-jetpack`, `pkgs.cudaPackages`, and `pkgs.boot.kernelPackages` all point to Jetson-specific implementations.

---

**Previous**: [← Pill 02: Flakes — Pinned, Composable, Reproducible](02-flakes.md)
**Next**: [Pill 04: jetpack-nixos — The 30,000ft View →](04-jetpack-nixos-architecture.md)
