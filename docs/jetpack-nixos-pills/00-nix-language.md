# Pill 00: Nix — The Language Under Everything

## tl;dr

Nix is a **purely functional, lazy, dynamically-typed language** with exactly one job: build software reproducibly. Everything in NixOS — packages, kernel, drivers, services, your entire Jetson system — is an expression in this language. Master the language, master the system.

## The Core Mental Model

Nix has **five fundamental types** that matter:

```
1. Strings      "hello"
2. Numbers      42, 3.14
3. Booleans     true, false
4. Lists        [ 1 2 3 ]
5. Attribute Sets  { key = value; }
```

And **one fundamental operation**: the function.

```nix
# A function. That's it. input: body.
x: x + 1
```

Everything else — packages, NixOS configurations, overlays, flakes — is built from these primitives. There are no classes, no objects, no mutation, no side effects (except `builtins.fetchurl` and friends).

## Attribute Sets: The Universal Container

An **attribute set** (attrset) is Nix's dictionary/map/object. It's the single most important data structure.

```nix
# Simple attribute set
{
  name = "jetpack-nixos";
  version = "6.2.1";
  cuda = true;
}
```

### Nested Access

```nix
let
  cfg = {
    hardware.nvidia-jetpack = {
      enable = true;
      som = "orin-agx";
    };
  };
in
  cfg.hardware.nvidia-jetpack.som  # "orin-agx"
```

### The `//` Merge Operator

Merges two attrsets. Right side wins on conflict:

```nix
{ a = 1; b = 2; } // { b = 3; c = 4; }
# Result: { a = 1; b = 3; c = 4; }
```

This is how jetpack-nixos builds device configs:

```nix
# From flake.nix — merging base jetpack config with per-device settings
{
  hardware.nvidia-jetpack = { enable = true; } // cfg;
  # cfg might be { som = "orin-agx"; carrierBoard = "devkit"; }
  # Result: { enable = true; som = "orin-agx"; carrierBoard = "devkit"; }
}
```

### `rec` — Self-Referencing Sets

Normal attrsets can't reference their own members. `rec` enables that:

```nix
rec {
  major = "36";
  minor = "4";
  full = "${major}.${minor}";  # "36.4"
}
```

**Warning**: `rec` can cause infinite recursion. Prefer `let` bindings when possible.

## Functions: The Only Abstraction

### Basic Syntax

```nix
# Single argument
double = x: x * 2;
double 5  # 10

# Multiple arguments (curried)
add = x: y: x + y;
add 3 4  # 7

# Partial application (automatic)
add3 = add 3;
add3 4  # 7
```

### Pattern Matching (Destructuring)

The dominant function style in Nix. Takes an attrset, destructures it:

```nix
# Basic pattern
greet = { name, age }: "Hello ${name}, age ${toString age}";
greet { name = "Orin"; age = 3; }  # "Hello Orin, age 3"

# With defaults
greet = { name, age ? 0 }: "Hello ${name}, age ${toString age}";
greet { name = "Orin"; }  # "Hello Orin, age 0"

# With ellipsis (ignore extra attrs)
greet = { name, ... }: "Hello ${name}";
greet { name = "Orin"; som = "orin-agx"; carrierBoard = "devkit"; }  # "Hello Orin"
```

**This is everywhere in jetpack-nixos.** Look at `mk-overlay.nix`:

```nix
# The entire mk-overlay.nix is a function that takes version parameters
{ jetpackMajorMinorPatchVersion
, l4tMajorMinorPatchVersion
, cudaMajorMinorPatchVersion
, cudaDriverMajorMinorVersion
, bspHash
, bspPatches ? []
, bspPostPatch ? []
}:
final: _:   # <-- This is a two-argument overlay function
let
  # ... body uses these parameters to build all packages
in
  # ... returns attribute set of packages
```

### `callPackage` Pattern

The **most important pattern in nixpkgs**. Auto-injects function arguments from the package set:

```nix
# A package file: pkgs/my-package.nix
{ stdenv, fetchurl, cmake }:  # <-- These are auto-injected
stdenv.mkDerivation {
  name = "my-package";
  src = fetchurl { ... };
  nativeBuildInputs = [ cmake ];
}

# callPackage reads the function's parameters and fills them from pkgs
pkgs.callPackage ./pkgs/my-package.nix { }
# Equivalent to:
# import ./pkgs/my-package.nix { stdenv = pkgs.stdenv; fetchurl = pkgs.fetchurl; cmake = pkgs.cmake; }

# Override specific args:
pkgs.callPackage ./pkgs/my-package.nix { cmake = pkgs.cmake_3_20; }
```

jetpack-nixos uses this heavily:

```nix
# From mk-overlay.nix
flash-tools = self.callPackage ./pkgs/flash-tools { };
board-automation = self.callPackage ./pkgs/board-automation { };
kernel = self.callPackage ./pkgs/kernels/r${l4tMajorVersion} { kernelPatches = []; };
```

## `let` Bindings: Local Scope

```nix
let
  x = 10;
  y = 20;
  sum = x + y;
in
  sum * 2  # 60
```

**Key rule**: `let` bindings are **recursive** — they can reference each other:

```nix
let
  l4tVersion = "36.4.4";
  majorVersion = builtins.head (builtins.split "\\." l4tVersion);
  # majorVersion = "36" — references l4tVersion
in
  majorVersion
```

## `with`: Scope Injection

Brings all attributes of a set into scope:

```nix
let
  pkgs = { gcc = "gcc-13"; cmake = "cmake-3.28"; };
in
  with pkgs; [ gcc cmake ]  # [ "gcc-13" "cmake-3.28" ]
```

Common in NixOS configs:

```nix
environment.systemPackages = with pkgs; [
  vim
  git
  file
];
# Same as: [ pkgs.vim pkgs.git pkgs.file ]
```

**Warning**: `with` can shadow variables. Avoid `with` in library code. Fine in configs.

## `inherit`: Shorthand for Same-Named Attrs

```nix
# These are equivalent:
{ x = x; y = y; }
{ inherit x y; }

# Inherit from another set:
let config = { som = "orin-agx"; cuda = true; };
in { inherit (config) som cuda; }
# Result: { som = "orin-agx"; cuda = true; }
```

Real usage from `mk-overlay.nix`:

```nix
makeScope final.newScope (self: {
  inherit (sourceInfo) debs gitRepos;
  inherit jetpackMajorMinorPatchVersion l4tMajorMinorPatchVersion cudaMajorMinorVersion;
  # ...
})
```

## `import`: File-Level Composition

`import` evaluates a `.nix` file and returns its value:

```nix
# file: add.nix
x: y: x + y

# usage:
let add = import ./add.nix;
in add 3 4  # 7
```

Most `.nix` files are **functions** that get imported and called:

```nix
# overlay.nix exports a function: final: prev: { ... }
# flake.nix imports and uses it:
overlays.default = final: prev:
  lib.composeManyExtensions [
    cuda-legacy.overlays.default
    (import ./overlay.nix)  # <-- imports the overlay function
  ] final prev;
```

## Laziness: Why Nix Doesn't Explode

Nix is **lazy** — expressions are only evaluated when their results are needed.

```nix
let
  expensive = builtins.throw "BOOM";  # Never evaluated
  cheap = 42;
in
  cheap  # 42 — expensive is never touched
```

**Why this matters for jetpack-nixos**: The flake defines packages for ALL Jetson SoMs (Xavier, Orin, Thor), ALL JetPack versions (5, 6, 7), and ALL configurations. But only the ones you actually build get evaluated. You can define:

```nix
nvidia-jetpack5 = import ./mk-overlay.nix { ... jp5 params ... } final prev;
nvidia-jetpack6 = import ./mk-overlay.nix { ... jp6 params ... } final prev;
nvidia-jetpack7 = import ./mk-overlay.nix { ... jp7 params ... } final prev;
```

And if you only use JetPack 6, only `nvidia-jetpack6` evaluates. The other two are never computed.

## String Interpolation

```nix
let
  version = "36.4.4";
  major = "36";
in
  "L4T R${major} version ${version}"
# "L4T R36 version 36.4.4"
```

### Multi-Line Strings

```nix
''
  mkdir -p $out
  cp -r ${src}/* $out/
  echo "Built version ${version}"
''
```

Two single quotes (`''`). Whitespace is auto-trimmed to the leftmost non-blank column.

### Paths vs Strings

```nix
./my-file.nix          # Path (relative to current file)
/absolute/path.nix     # Path (absolute)
"./my-file.nix"        # String (NOT a path!)

# Paths are automatically copied to the Nix store when used in derivations
src = ./my-source;  # Copies to /nix/store/hash-my-source
```

## `if`/`then`/`else`: Expressions, Not Statements

Everything in Nix is an expression. `if` returns a value:

```nix
let
  som = "orin-agx";
  socType =
    if lib.hasPrefix "orin-" som then "t234"
    else if lib.hasPrefix "xavier-" som then "t194"
    else if lib.hasPrefix "thor-" som then "t264"
    else throw "Unknown SoC type";
in
  socType  # "t234"
```

Real code from `overlay-with-config.nix`:

```nix
socType =
  if cfg.som == null then null
  else if lib.hasPrefix "thor-" cfg.som then "t264"
  else if lib.hasPrefix "orin-" cfg.som then "t234"
  else if lib.hasPrefix "xavier-" cfg.som then "t194"
  else throw "Unknown SoC type";
```

## Lists

```nix
# Lists are space-separated, NOT comma-separated
[ 1 2 3 ]
[ "a" "b" "c" ]
[ { name = "orin"; } { name = "xavier"; } ]

# List operations
builtins.head [ 1 2 3 ]     # 1
builtins.tail [ 1 2 3 ]     # [ 2 3 ]
builtins.length [ 1 2 3 ]   # 3
builtins.elem 2 [ 1 2 3 ]   # true
[ 1 2 ] ++ [ 3 4 ]          # [ 1 2 3 4 ]

# map, filter
map (x: x * 2) [ 1 2 3 ]               # [ 2 4 6 ]
builtins.filter (x: x > 2) [ 1 2 3 4 ] # [ 3 4 ]
```

## `builtins` vs `lib`

Two sources of utility functions:

```nix
# builtins — built into the Nix evaluator (always available)
builtins.attrNames { a = 1; b = 2; }  # [ "a" "b" ]
builtins.map (x: x + 1) [ 1 2 3 ]     # [ 2 3 4 ]
builtins.toString 42                    # "42"

# lib — from nixpkgs (must be imported)
lib.hasPrefix "orin-" "orin-agx"        # true
lib.optionals true [ "a" "b" ]          # [ "a" "b" ]
lib.optionalString false "hello"        # ""
lib.versions.major "36.4.4"             # "36"
lib.versions.minor "36.4.4"             # "4"
lib.composeManyExtensions [ ... ]       # Compose overlay functions
```

## Putting It All Together

Here's a simplified version of what jetpack-nixos does, using only the concepts above:

```nix
# A "mini jetpack-nixos" in ~20 lines
let
  # 1. Version parameters (just data)
  versions = {
    jetpack = "6.2.1";
    l4t = "36.4.4";
    cuda = "12.6";
  };

  # 2. Function to build packages for a version (curried)
  mkPackages = versions: pkgs: {
    kernel = pkgs.callPackage ./kernel.nix { inherit (versions) l4t; };
    cuda = pkgs.callPackage ./cuda.nix { inherit (versions) cuda; };
    flash-tools = pkgs.callPackage ./flash-tools.nix { inherit (versions) l4t; };
  };

  # 3. Overlay: inject packages into nixpkgs (function of two args)
  overlay = final: prev: {
    nvidia-jetpack = mkPackages versions final;
  };

  # 4. NixOS module: declare hardware options, wire packages
  nixosModule = { config, pkgs, ... }: {
    options.hardware.jetson.enable = lib.mkEnableOption "Jetson";
    config = lib.mkIf config.hardware.jetson.enable {
      boot.kernelPackages = pkgs.nvidia-jetpack.kernel;
      hardware.firmware = [ pkgs.nvidia-jetpack.firmware ];
    };
  };
in
  { inherit overlay nixosModule; }
```

**That's all Nix is.** Functions that take attrsets and return attrsets, composed into layers. The actual jetpack-nixos just has more layers and more details — but the structure is identical.

## Common Gotchas

### 1. Semicolons End Bindings, Not Statements

```nix
{ a = 1; b = 2; }  # ← Semicolons after each binding
let x = 1; in x    # ← Semicolons after let bindings
```

### 2. Function Application is Whitespace

```nix
f x      # Call f with x
f x y    # Call f with x, then call result with y
f (x y)  # Call x with y, then call f with result
```

### 3. No `return` Keyword

The body of a function IS the return value:

```nix
add = x: y: x + y;   # Returns x + y
# NOT: add = x: y: return (x + y);
```

### 4. Strings Need `toString`

```nix
"version " + 42           # ERROR
"version " + toString 42  # "version 42"
"version ${toString 42}"  # "version 42"
```

## Summary

| Concept | What It Is | Where You'll See It |
|---------|-----------|-------------------|
| Attribute set | `{ key = val; }` | Everywhere. Configs, packages, options |
| Function | `x: body` | Package definitions, overlays, modules |
| `//` merge | Right-biased merge | Config composition, device variants |
| `callPackage` | Auto-inject function args | `pkgs.callPackage ./pkg.nix {}` |
| `let`/`in` | Local bindings | Version variables, intermediate values |
| `import` | Load a `.nix` file | `import ./overlay.nix` |
| Laziness | Evaluate only what's needed | Multiple JetPack versions co-existing |
| `inherit` | Shorthand for same-name attrs | Passing through version strings |
| `with` | Scope injection | `with pkgs; [ vim git ]` |

---

**Next**: [Pill 01: Derivations — How Nix Builds Things →](01-derivations.md)
