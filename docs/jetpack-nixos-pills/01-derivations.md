# Pill 01: Derivations — How Nix Builds Things

## tl;dr

A **derivation** is the single concept that makes Nix work. It's a build instruction: "take these inputs, run this builder script, produce this output at a deterministic store path." The entire Nix store — every package, every kernel, every firmware blob — is a tree of derivations. Understand derivations, understand Nix.

## What Is a Derivation?

A derivation is a **pure function from inputs to outputs**:

```
Inputs (source code, dependencies, build script)
        │
        ▼
   ┌─────────┐
   │  BUILD   │  (runs in a sandbox: no network, no $HOME, no /usr)
   └────┬────┘
        │
        ▼
Output (/nix/store/<hash>-<name>/)
```

The **hash** in the output path is computed from ALL inputs. Change any input → different hash → different output path. This is what makes Nix reproducible.

## The Raw Derivation

At the lowest level, a derivation is just an attrset passed to `builtins.derivation`:

```nix
builtins.derivation {
  name = "hello";
  system = "aarch64-linux";
  builder = "/bin/sh";
  args = [ "-c" "echo hello > $out" ];
}
```

This produces: `/nix/store/abc123...-hello`

**Nobody writes this.** Instead, we use `stdenv.mkDerivation`.

## `stdenv.mkDerivation`: The Real Interface

`stdenv` (standard environment) wraps the raw derivation with a proper build environment:

```nix
stdenv.mkDerivation {
  pname = "my-package";
  version = "1.0";

  src = fetchurl {
    url = "https://example.com/my-package-1.0.tar.gz";
    sha256 = "sha256-AAAA...";
  };

  nativeBuildInputs = [ cmake pkg-config ];  # Build-time tools
  buildInputs = [ openssl zlib ];             # Libraries to link against

  # Build phases (all optional, have sensible defaults)
  configurePhase = ''
    cmake -B build -DCMAKE_INSTALL_PREFIX=$out
  '';

  buildPhase = ''
    cmake --build build -j$NIX_BUILD_CORES
  '';

  installPhase = ''
    cmake --install build
  '';
}
```

### The Build Phases

`mkDerivation` runs phases in order. Each has a default behavior you can override:

```
1. unpackPhase      Extract source archive
2. patchPhase       Apply patches
3. configurePhase   Run ./configure or cmake
4. buildPhase       Run make
5. checkPhase       Run tests (optional)
6. installPhase     Copy results to $out
7. fixupPhase       Patch ELF binaries, fix references
```

### `$out` — The Sacred Variable

Every derivation gets an `$out` environment variable pointing to its output path in the Nix store. Everything the build produces must go under `$out`:

```nix
installPhase = ''
  mkdir -p $out/bin
  cp my-binary $out/bin/
  mkdir -p $out/lib
  cp libfoo.so $out/lib/
'';
```

Result: `/nix/store/hash-my-package/bin/my-binary` and `/nix/store/hash-my-package/lib/libfoo.so`

## The Nix Store: The Immutable Package Database

Every derivation output lives in `/nix/store/`:

```
/nix/store/
├── abc123...-linux-5.15.148/          # Kernel
├── def456...-nvidia-jetpack-l4t-3d-core/  # GPU driver
├── ghi789...-cudaPackages-12.6/       # CUDA toolkit
├── jkl012...-nixos-system-nixos-25.11/    # Your entire system
└── ... thousands more
```

**Key properties**:
1. **Immutable** — Once built, never modified
2. **Content-addressed** — Hash computed from inputs
3. **Garbage-collected** — Unused packages are swept away
4. **Shared** — Two configs needing the same package share one store path

## How jetpack-nixos Uses Derivations

### L4T Packages: Debs → Nix Store

NVIDIA ships JetPack as `.deb` files. jetpack-nixos converts them to Nix derivations:

```nix
# Simplified from pkgs/buildFromDebs.nix
buildFromDebs {
  name = "nvidia-l4t-3d-core";
  version = l4tMajorMinorPatchVersion;
  srcs = debsForSourcePackage "nvidia-l4t-3d-core";

  # Extract .deb, fix paths, patch ELF binaries
  postPatch = ''
    # autoPatchelf will fix library paths to point to Nix store
  '';
}
```

The deb-to-nix pipeline:
```
NVIDIA's .deb from repo.download.nvidia.com
        │
        ▼
sourceinfo/r36.4-debs.json     (pinned hash)
        │
        ▼
fetchurl { url = ...; hash = ...; }     (reproducible fetch)
        │
        ▼
dpkg -x *.deb $out             (extract)
        │
        ▼
autoPatchelf                    (fix library paths to /nix/store/...)
        │
        ▼
/nix/store/hash-l4t-3d-core/   (proper Nix package)
```

### Kernel: Git Repo → Nix Store

```nix
# Simplified from pkgs/kernels/r36/default.nix
kernel = buildLinux {
  version = "5.15.148";
  src = gitRepos."kernel-oot";     # Fetched from NVIDIA's git
  kernelPatches = [ ... ];
  structuredExtraConfig = {
    TEGRA_BPMP = lib.kernel.yes;
    # ... hundreds of Jetson-specific kernel configs
  };
};
```

### BSP: The Board Support Package

```nix
# From mk-overlay.nix
bspSrc = final.applyPatches {
  src = final.runCommand "l4t-unpacked" {
    src = final.fetchurl {
      url = "https://developer.download.nvidia.com/.../Jetson_Linux_R${l4tVersion}_aarch64.tbz2";
      hash = bspHash;  # Pinned hash ensures reproducibility
    };
  } ''
    bzip2 -d -c $src | tar xf -
    mv Linux_for_Tegra $out
  '';
  patches = bspPatches;
};
```

## `nativeBuildInputs` vs `buildInputs`

This matters for cross-compilation (x86_64 host → aarch64 Jetson):

```nix
{
  nativeBuildInputs = [ cmake ];  # Runs on BUILD machine (x86_64)
  buildInputs = [ cudatoolkit ];  # Links against TARGET machine (aarch64)
}
```

| Attribute | Runs on | Used for |
|-----------|---------|----------|
| `nativeBuildInputs` | Build machine | Compilers, code generators, build tools |
| `buildInputs` | Target machine | Libraries to link against |

jetpack-nixos uses this for flash scripts (run on x86_64, target aarch64):

```nix
# flasherPkgs is x86_64, config targets aarch64
flasherPkgs = import pkgs.path {
  system = "x86_64-linux";    # Build platform
  inherit (pkgs) config;
};
```

## Fixed-Output Derivations: Fetching from the Internet

Normal derivations run in a sandbox with no network. **Fixed-output derivations** are special: they CAN access the network, but must produce a known hash:

```nix
fetchurl {
  url = "https://developer.download.nvidia.com/.../Jetson_Linux_R36.4.4_aarch64.tbz2";
  hash = "sha256-ps4RwiEAqwl25BmVkYJBfIPWL0JyUBvIcU8uB24BDzs=";
  # If the download doesn't match this hash → build fails
}
```

This is how jetpack-nixos pins ALL upstream NVIDIA sources — `sourceinfo/r36.4-debs.json` contains hashes for every `.deb`:

```json
{
  "nvidia-l4t-3d-core": {
    "src": {
      "url": "https://repo.download.nvidia.com/jetson/...",
      "hash": "sha256:XXXX..."
    }
  }
}
```

Change the hash → Nix refuses to use the old cached version → forces re-download → reproducibility guaranteed.

## Store Paths: The Input-Addressed Model

The output hash is derived from **all inputs**:

```
hash = sha256(
  builder script
  + all source inputs
  + all dependency store paths
  + all environment variables
  + system architecture
)
```

This means:
- Same inputs → same hash → same output → **cache hit**
- Different GCC version → different hash → different output
- Different kernel config flag → different hash → different kernel

## `runCommand`: Quick One-Off Derivations

For simple derivations that just run a shell command:

```nix
# From mk-overlay.nix — unpacking BSP
final.runCommand "l4t-unpacked" {
  src = bspTarball;
  nativeBuildInputs = [ final.buildPackages.bzip2_1_1 ];
} ''
  bzip2 -d -c $src | tar xf -
  mv Linux_for_Tegra $out
''
```

```nix
# From mk-overlay.nix — generating L4T JSON
final.runCommand "l4t.json" {
  nativeBuildInputs = [ final.buildPackages.python3 ];
} ''
  python3 ${./pkgs/containers/gen_l4t_json.py} ${self.l4tCsv} > $out
''
```

## `overrideAttrs`: Modifying Derivations

Since derivations are immutable, you can't change them. But you can create a **new derivation** based on an existing one:

```nix
# Original
flash-tools = self.callPackage ./pkgs/flash-tools { };

# Modified (from overlay-with-config.nix)
flash-tools = prevJetpack.flash-tools.overrideAttrs ({ patches ? [], postPatch ? "", ... }: {
  patches = patches ++ cfg.flashScriptOverrides.patches;
  postPatch = postPatch + cfg.flashScriptOverrides.postPatch;
});
```

`overrideAttrs` takes a function `oldAttrs -> newAttrs`. The `//` merge happens implicitly.

## `.override`: Modifying Function Arguments

Different from `overrideAttrs`. This changes the **arguments passed to `callPackage`**:

```nix
# Original: callPackage passes default cudnn
cudaPackages_12_6 = prev.cudaPackages_12_6;

# Override: change the TensorRT manifest version
cudaPackages_12_6 = prev.cudaPackages_12_6.override (prevArgs: {
  manifests = prevArgs.manifests // {
    tensorrt = final._cuda.manifests.tensorrt."10.7.0";
  };
});
```

| Method | What It Changes | Analogy |
|--------|----------------|---------|
| `.override` | Arguments to the package function | Changing constructor args |
| `.overrideAttrs` | Attributes of the derivation | Changing build recipe |

## `makeScope` and `newScope`: Package Sets

A **scope** is a self-contained package set where packages can see each other:

```nix
# From mk-overlay.nix
makeScope final.newScope (self: {
  # self.callPackage auto-injects from THIS scope (not the global pkgs)
  flash-tools = self.callPackage ./pkgs/flash-tools { };
  kernel = self.callPackage ./pkgs/kernels/r36 { };

  # Packages here can reference each other
  kernelPackages = final.linuxPackagesFor self.kernel;
})
```

`nvidia-jetpack` is a scope. When you write `pkgs.nvidia-jetpack.flash-tools`, you're accessing `flash-tools` from that scope.

This is why jetpack-nixos packages can depend on each other without polluting the global package set.

## The Derivation DAG

Derivations form a **Directed Acyclic Graph**:

```
                    nixos-system
                   /     |      \
              kernel  firmware  nvidia-jetpack
             /    \      |       /    |     \
        linux-src  dt  l4t-fw  l4t-3d cuda  flash-tools
                          |      |      |
                       deb-src  deb   deb-src
                          |      |      |
                      fetchurl fetchurl fetchurl
```

Every leaf is a `fetchurl` (fixed-output derivation). Every internal node is a build step. The entire graph is deterministic.

## Garbage Collection

```bash
# List all store paths (roots keep things alive)
nix-store --gc --print-roots

# Delete unreferenced paths
nix-collect-garbage

# Delete generations older than 7 days
nix-collect-garbage --delete-older-than 7d

# Check what would be deleted
nix-store --gc --print-dead
```

A store path is a **root** if:
1. It's the current system generation
2. It's referenced by a GC root in `/nix/var/nix/gcroots/`
3. It's in the current user's profile

Everything else can be garbage-collected.

## Debugging Derivations

```bash
# Show the derivation file (.drv)
nix derivation show nixpkgs#hello

# Build with verbose output
nix build .#package --show-trace -L

# Build and keep the build directory on failure
nix build .#package --keep-failed

# Enter a shell with the build environment
nix develop .#package

# Find what depends on a store path
nix-store --query --referrers /nix/store/hash-package

# Find what a store path depends on
nix-store --query --references /nix/store/hash-package

# Show the full dependency tree
nix-store --query --tree /nix/store/hash-package
```

## Summary

| Concept | What | Why |
|---------|------|-----|
| Derivation | Build instruction → store path | Everything in Nix is a derivation |
| `stdenv.mkDerivation` | The standard way to build packages | Handles phases, patching, ELF fixing |
| `$out` | Output path in `/nix/store/` | Where build results go |
| Store path hash | Computed from all inputs | Guarantees reproducibility |
| `fetchurl` | Fixed-output network fetch | Pins upstream sources with hashes |
| `callPackage` | Auto-inject dependencies | The pattern that makes Nix ergonomic |
| `overrideAttrs` | Modify build recipe | Customize without forking |
| `.override` | Modify function arguments | Change dependency versions |
| `makeScope` | Self-contained package set | How `nvidia-jetpack` is organized |
| `runCommand` | Quick shell-script derivation | BSP unpacking, JSON generation |

**The mental model**: Your entire Jetson NixOS system is one giant derivation tree. At the leaves: `fetchurl` calls to NVIDIA's servers. In the middle: build steps that extract, patch, compile, and link. At the root: your bootable system. Every node is immutable, reproducible, and cacheable.

---

**Previous**: [← Pill 00: Nix — The Language Under Everything](00-nix-language.md)
**Next**: [Pill 02: Flakes — Pinned, Composable, Reproducible →](02-flakes.md)
