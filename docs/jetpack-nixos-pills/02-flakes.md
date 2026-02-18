# Pill 02: Flakes — Pinned, Composable, Reproducible

## tl;dr

A **flake** is a Nix project with a standardized interface. It has exactly two files that matter: `flake.nix` (the code) and `flake.lock` (the pinned dependency versions). Flakes solve the "works on my machine" problem by ensuring that every input — nixpkgs, jetpack-nixos, everything — is locked to a specific git revision. No channels. No mutable state. Just pinned, reproducible builds.

## The Problem Flakes Solve

**Before flakes (channels)**:
```bash
# Which nixpkgs are you using? Who knows!
nix-channel --list
# nixpkgs https://nixos.org/channels/nixos-24.05
# But when did you last update? What commit?
# ¯\_(ツ)_/¯
```

Two developers, same `configuration.nix`, different `nix-channel --update` times → different system builds. Not reproducible.

**With flakes**:
```bash
cat flake.lock | grep -A3 nixpkgs
# "locked": {
#   "rev": "abc123def456...",  ← EXACT commit
#   "narHash": "sha256-XXXX..."  ← Content hash
# }
```

Everyone building this flake gets **the exact same nixpkgs commit**. Always.

## Anatomy of `flake.nix`

Every flake has the same structure:

```nix
{
  description = "What this flake does";

  inputs = {
    # Dependencies — other flakes (or non-flake sources)
  };

  outputs = { self, ... }:
    # What this flake produces — packages, modules, overlays, etc.
    {
      # ...
    };
}
```

That's it. Three fields. `description` (optional), `inputs` (what you depend on), `outputs` (what you produce).

### Inputs: Your Dependencies

```nix
inputs = {
  # A flake from GitHub
  nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

  # Another flake — jetpack-nixos
  jetpack.url = "github:anduril/jetpack-nixos/master";

  # Make jetpack use OUR nixpkgs, not its own
  jetpack.inputs.nixpkgs.follows = "nixpkgs";
};
```

#### Input URL Schemes

```nix
# GitHub
"github:owner/repo"              # Default branch
"github:owner/repo/branch"       # Specific branch
"github:owner/repo/abc123"       # Specific commit

# Path (local flake)
"path:/home/user/my-flake"       # Absolute
"path:./subfolder"               # Relative

# Git
"git+https://example.com/repo"   # HTTPS
"git+ssh://git@github.com/repo"  # SSH

# Tarball
"https://example.com/archive.tar.gz"
```

#### `follows`: Deduplication

Without `follows`:
```
Your flake
├── nixpkgs (rev A)        ← Your pinned nixpkgs
└── jetpack-nixos
    └── nixpkgs (rev B)    ← jetpack's own pinned nixpkgs (DIFFERENT!)
```

Two different nixpkgs → two different stdenv → things may not compose correctly.

With `follows`:
```nix
jetpack.inputs.nixpkgs.follows = "nixpkgs";
```

```
Your flake
├── nixpkgs (rev A)
└── jetpack-nixos
    └── nixpkgs → (rev A)  ← SAME as yours
```

One nixpkgs. Consistent. This is how the real example flake does it:

```nix
# From examples/nixos/flake.nix
inputs = {
  nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
  jetpack.url = "github:anduril/jetpack-nixos/master";
  jetpack.inputs.nixpkgs.follows = "nixpkgs";
};
```

### Outputs: What You Produce

The `outputs` function receives all resolved inputs and returns an attrset of "things this flake provides":

```nix
outputs = { self, nixpkgs, jetpack, ... }: {
  # NixOS system configurations
  nixosConfigurations.my-jetson = nixpkgs.lib.nixosSystem { ... };

  # Packages
  packages.aarch64-linux.my-tool = ...;

  # NixOS modules (reusable config fragments)
  nixosModules.default = ...;

  # Overlays (package set modifications)
  overlays.default = ...;

  # Development shells
  devShells.aarch64-linux.default = ...;

  # Custom checks (CI)
  checks.aarch64-linux.formatting = ...;
};
```

#### Standard Output Types

| Output | Type | Used For |
|--------|------|----------|
| `nixosConfigurations.<name>` | NixOS system | `nixos-rebuild switch --flake .#name` |
| `packages.<system>.<name>` | Derivation | `nix build .#name` |
| `overlays.<name>` | Overlay function | `nixpkgs.overlays = [ flake.overlays.default ]` |
| `nixosModules.<name>` | NixOS module | `imports = [ flake.nixosModules.default ]` |
| `devShells.<system>.<name>` | Dev environment | `nix develop .#name` |
| `legacyPackages.<system>` | Full package set | For complex nested packages |
| `checks.<system>.<name>` | Test derivation | `nix flake check` |
| `formatter.<system>` | Formatter package | `nix fmt` |

## The jetpack-nixos Flake Dissected

Let's walk through the actual `flake.nix`:

### Inputs

```nix
inputs = {
  nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";

  cuda-legacy = {
    url = "github:nixos-cuda/cuda-legacy";
    inputs.nixpkgs.follows = "nixpkgs";
  };
};
```

Two inputs: nixpkgs and cuda-legacy (provides CUDA 11.4 packages no longer in nixpkgs).

### The Overlay Output

```nix
overlays.default = final: prev:
  if prev."jetpack-nixos-overlay-applied-${self.narHash}" or false then
    { }  # Already applied — return nothing
  else
    nixpkgs.lib.composeManyExtensions [
      cuda-legacy.overlays.default    # Apply cuda-legacy first
      (import ./overlay.nix)          # Then our overlay
      (_: _: { "jetpack-nixos-overlay-applied-${self.narHash}" = true; })  # Mark as applied
    ] final prev;
```

**Key insight**: The overlay is **idempotent**. It records that it's been applied using a marker attribute. If applied twice, the second application is a no-op. This prevents double-application bugs.

### The NixOS Module Output

```nix
nixosModules.default = import ./modules/default.nix self.overlays.default;
```

The module is a function that takes the overlay and returns a NixOS module. The module applies the overlay to the system's nixpkgs and wires up kernel, firmware, drivers, and services.

### NixOS Configurations (for testing/CI)

```nix
nixosConfigurations = {
  installer_minimal = nixpkgs.lib.nixosSystem {
    modules = [ aarch64_config installer_minimal_config ];
  };
} // supportedNixOSConfigurations;
```

The flake defines configurations for every supported Jetson device (Orin AGX, Orin NX, Orin Nano, Xavier variants, Thor). These are primarily for CI testing and ISO generation.

### Packages Output

```nix
packages = {
  x86_64-linux = {
    # Flash scripts (x86_64 only — NVIDIA's tools are x86)
    flash-orin-agx-devkit = ...;
    flash-orin-nx-devkit = ...;

    # ISOs for installation
    iso_minimal = ...;
  };
  aarch64-linux = {
    iso_minimal = ...;
  };
};
```

### legacyPackages Output

```nix
legacyPackages = forAllSystems ({ system, ... }:
  let
    pkgs = (import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaCapabilities = if system == "aarch64-linux" then [ "7.2" "8.7" ] else [];
        cudaSupport = true;
      };
      overlays = [ self.overlays.default ];
    });
  in
  pkgs.nvidia-jetpack // {
    inherit (pkgs) nvidia-jetpack5;
    inherit (pkgs.pkgsForCudaArch.sm_87) nvidia-jetpack6;
    inherit (pkgs.pkgsForCudaArch.sm_110) nvidia-jetpack7;
  }
);
```

This is how you access individual JetPack packages without a full NixOS system build.

## Your Configuration Flake

Here's how a **user** consumes jetpack-nixos. This is the real `examples/nixos/flake.nix`:

```nix
{
  description = "NixOS configuration for Jetson AGX Orin";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    jetpack.url = "github:anduril/jetpack-nixos/master";
    jetpack.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, jetpack, ... }:
    let
      baseModules = [
        jetpack.nixosModules.default    # ← Brings in the jetpack overlay + modules
        ./configuration.nix              # ← Your system config
      ];
    in {
      nixosConfigurations.nixos = nixpkgs.lib.nixosSystem {
        modules = baseModules;
      };
    };
}
```

**That's it.** Three lines of `inputs`, one module list, one `nixosSystem` call. The jetpack module handles everything else.

### Multiple Configurations

```nix
# You can define multiple configurations and switch between them
nixosConfigurations.nixos-perf = nixpkgs.lib.nixosSystem {
  modules = baseModules ++ [
    ./modules/performance.nix
    ({ ... }: { services.orin-perf.enable = true; })
  ];
};

# Build: sudo nixos-rebuild switch --flake .#nixos-perf
```

## `flake.lock`: The Pin File

`flake.lock` is auto-generated JSON that pins every input to an exact revision:

```json
{
  "nodes": {
    "nixpkgs": {
      "locked": {
        "lastModified": 1707826012,
        "narHash": "sha256-XXXXXXXXXX...",
        "owner": "NixOS",
        "repo": "nixpkgs",
        "rev": "abc123def456789...",
        "type": "github"
      }
    },
    "jetpack": {
      "locked": {
        "lastModified": 1707825000,
        "narHash": "sha256-YYYYYYYYYY...",
        "rev": "def789abc123456...",
        "type": "github"
      }
    }
  }
}
```

### Updating the Lock File

```bash
# Update ALL inputs to latest
nix flake update

# Update only one input
nix flake update nixpkgs

# Update to a specific commit
nix flake update jetpack --override-input jetpack github:anduril/jetpack-nixos/abc123

# Check what would change
nix flake update --dry-run
```

### The Reproducibility Guarantee

```
flake.nix  +  flake.lock  →  EXACT same build
```

Commit both files. Anyone cloning your repo and running `nixos-rebuild switch --flake .#nixos` gets **byte-for-byte identical** system builds (given the same build platform).

## `nixpkgs.lib.nixosSystem`: Building a System

This is the function that turns modules into a full NixOS system:

```nix
nixpkgs.lib.nixosSystem {
  modules = [
    # Module 1: jetpack-nixos (injects overlay + hardware modules)
    jetpack.nixosModules.default

    # Module 2: your config
    ./configuration.nix

    # Module 3: inline module
    ({ ... }: {
      hardware.nvidia-jetpack.enable = true;
      hardware.nvidia-jetpack.som = "orin-agx";
    })
  ];
}
```

`nixosSystem` does:
1. Merges all modules (options + config)
2. Evaluates the merged configuration
3. Produces the system closure (all packages, kernel, firmware, etc.)
4. Returns it as `.config.system.build.toplevel`

## Dev Shells: Development Environments

Flakes can define development environments:

```nix
# From examples/llama-cpp-orin-nix-overlay/flake.nix
devShells.aarch64-linux.default = pkgs.mkShell {
  packages = [
    pkgs.llama-cpp
    pkgs.cudatoolkit
  ];
  shellHook = ''
    echo "llama.cpp dev shell ready"
    export CUDA_PATH=${pkgs.cudatoolkit}
  '';
};
```

Enter with:
```bash
nix develop          # Uses default devShell
nix develop .#other  # Uses named devShell
```

## `self`: Self-Reference

The `self` parameter in `outputs` refers to the flake itself:

```nix
outputs = { self, nixpkgs, ... }: {
  # Reference our own overlay
  nixosModules.default = import ./modules/default.nix self.overlays.default;

  # Reference our own NixOS configs
  packages.x86_64-linux.iso = self.nixosConfigurations.installer.config.system.build.isoImage;

  # Use our own narHash for dedup markers
  overlays.default = final: prev:
    if prev."applied-${self.narHash}" or false then { } else { ... };
};
```

## `forAllSystems`: Multi-Platform

A common pattern for supporting multiple architectures:

```nix
let
  allSystems = [ "x86_64-linux" "aarch64-linux" ];
  forAllSystems = f: nixpkgs.lib.genAttrs allSystems (system: f {
    pkgs = nixpkgs.legacyPackages.${system};
    inherit system;
  });
in {
  packages = forAllSystems ({ pkgs, ... }: {
    my-tool = pkgs.callPackage ./my-tool.nix { };
  });
  # Result: packages.x86_64-linux.my-tool AND packages.aarch64-linux.my-tool
}
```

## Flake Commands Cheat Sheet

```bash
# Build a package
nix build .#package-name

# Build a NixOS system
nix build .#nixosConfigurations.nixos.config.system.build.toplevel

# Switch NixOS system
sudo nixos-rebuild switch --flake .#nixos

# Enter dev shell
nix develop .#shell-name

# Show flake outputs
nix flake show

# Show flake metadata (inputs, revisions)
nix flake metadata

# Check flake for errors
nix flake check

# Update lock file
nix flake update

# Evaluate an expression from the flake
nix eval .#nixosConfigurations.nixos.config.hardware.nvidia-jetpack.som
```

## Common Patterns

### Override an Input Temporarily

```bash
# Use local jetpack-nixos instead of GitHub
sudo nixos-rebuild switch --flake .#nixos \
  --override-input jetpack path:/home/agent/jetpack-nixos
```

### Pin to a Specific Commit

```bash
nix flake update jetpack --override-input jetpack github:anduril/jetpack-nixos/abc123
```

### Use `--no-update-lock-file`

```bash
# Don't modify flake.lock (useful in CI or dev shells)
nix develop --no-update-lock-file
```

## Summary

| Concept | What | Why |
|---------|------|-----|
| `flake.nix` | Code: inputs + outputs | Standardized project interface |
| `flake.lock` | Pinned input revisions | Reproducibility across machines |
| `inputs` | Dependencies (other flakes) | nixpkgs, jetpack-nixos, cuda-legacy |
| `follows` | Input deduplication | Ensure one nixpkgs for all |
| `outputs` | What the flake provides | Packages, modules, overlays, configs |
| `nixosSystem` | Modules → bootable system | The function that builds NixOS |
| `overlays.default` | Package set modification | How jetpack injects its packages |
| `nixosModules.default` | Reusable config fragment | How jetpack wires hardware support |
| `self` | Self-reference | Access own outputs from within |
| `nix flake update` | Update lock file | Get newer versions of inputs |

**The mental model**: A flake is a **pure function** from pinned inputs to outputs. `flake.lock` pins the inputs. `flake.nix` defines the function. The outputs are packages, systems, modules — whatever you need. Compose flakes by listing them as inputs. That's it.

---

**Previous**: [← Pill 01: Derivations — How Nix Builds Things](01-derivations.md)
**Next**: [Pill 03: Overlays & Overrides — Layering the Package Set →](03-overlays-and-overrides.md)
