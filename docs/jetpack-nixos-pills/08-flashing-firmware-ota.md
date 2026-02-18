# Pill 08: Flashing, Firmware & OTA Updates

## tl;dr

A Jetson can't just boot from an SD card image. Its firmware lives in QSPI flash memory and must be programmed via NVIDIA's flash tools — which only run on **x86_64 Linux**. jetpack-nixos generates flash scripts as Nix derivations, builds UEFI firmware from source (EDK2), bundles OP-TEE, and supports **capsule OTA updates** so firmware upgrades happen automatically on `nixos-rebuild switch`. This pill explains the entire chain from "I have a NixOS config" to "my Jetson boots."

## Why Flashing Is Different

Desktop PCs: BIOS/UEFI is pre-installed. You boot a USB stick and install.

Jetson: Firmware is stored in **QSPI NOR flash** (a small chip soldered to the board). It must be programmed before the device can boot at all.

```
┌────────────────────────────────────────────┐
│         QSPI NOR Flash (32-64MB)           │
│  ┌──────────┬──────────┬────────────────┐  │
│  │  MB1/MB2 │ OP-TEE   │ UEFI (EDK2)   │  │
│  │ (NVIDIA  │ (secure  │ (bootloader)   │  │
│  │  boot    │  world)  │                │  │
│  │  chain)  │          │                │  │
│  └──────────┴──────────┴────────────────┘  │
└────────────────────────────────────────────┘
        ↓ Boots
┌────────────────────────────────────────────┐
│  eMMC / NVMe / SD card                     │
│  ┌──────────┬──────────────────────────┐   │
│  │   ESP    │   NixOS rootfs           │   │
│  │ systemd- │   /nix/store/...         │   │
│  │ boot     │                          │   │
│  └──────────┴──────────────────────────┘   │
└────────────────────────────────────────────┘
```

## The Boot Chain

```
Power on
  → MB1 (Microboot 1, in QSPI)       — NVIDIA binary, hardware init
  → MB2 (Microboot 2, in QSPI)       — NVIDIA binary, loads firmware
  → OP-TEE (in QSPI)                 — Secure world OS (ARM TrustZone)
  → UEFI / EDK2 (in QSPI)           — Standard UEFI bootloader
  → systemd-boot (on ESP partition)  — Linux bootloader
  → NixOS kernel + initrd            — Your system
```

MB1 and MB2 are NVIDIA-proprietary blobs. You can't replace them. Everything from OP-TEE onwards is buildable from source in jetpack-nixos.

## Flash Methods

### Method 1: Initrd Flash (Recommended)

The modern method. Uses RCM (Recovery Mode) to boot a temporary Linux initrd on the Jetson that flashes itself:

```
x86_64 host                          Jetson (in recovery mode)
     │                                      │
     │ ── USB RCM boot ──────────────►      │
     │    (uploads kernel + initrd)         │
     │                                      │ Boots temporary Linux
     │                                      │ initrd runs flash tools
     │                                      │ Writes QSPI partitions
     │                                      │ Writes rootfs
     │                                      │ Reboots
     │                                      │
     │                                      │ → Normal NixOS boot
```

```bash
# Put Jetson in recovery mode (hold REC button, press RESET)
# Then from x86_64 host:
nix run .#initrdFlashScript
```

### Method 2: Legacy Flash

Older method. The x86_64 host directly programs the Jetson over USB:

```bash
nix run .#legacyFlashScript
```

### Method 3: RCM Boot (No Flash)

Boots the Jetson from the host without writing to permanent storage. Useful for testing:

```bash
nix run .#rcmBoot
```

## How Flash Scripts Are Generated

### The Factory

Flash scripts are built in `device-pkgs/default.nix`, which is called with **two** package sets:

```nix
# Called from mk-overlay.nix
devicePkgs = pkgs.callPackage ../device-pkgs {
  inherit config;         # The NixOS config (aarch64)
  inherit nvidia-jetpack; # From x86_64 package set!
};
```

The flash tools are NVIDIA's prebuilt x86_64 binaries, so they need an x86_64 package set, but the firmware, kernel, and rootfs target aarch64.

### The Flash Script Template

`device-pkgs/flash-script.nix` generates a shell script:

```bash
# Simplified view of what the flash script does:

# 1. Copy flash-tools to a writable workdir
cp -r ${flash-tools}/. "$WORKDIR"
chmod -R u+w "$WORKDIR"
cd "$WORKDIR"

# 2. Add NVIDIA tools to PATH
export PATH=${flashDeps}/bin:$PATH

# 3. Tell flash.sh we're not building rootfs here
export NO_ROOTFS=1
export NO_RECOVERY_IMG=1
export NO_ESP_IMG=1

# 4. Copy in our built firmware
cp ${uefi-firmware}/uefi_jetson.bin bootloader/uefi_jetson.bin
cp ${tosImage}/tos.img bootloader/tos-optee_t234.img

# 5. Copy in device trees
cp -r ${dtbsDir}/. kernel/dtb/

# 6. Copy partition template
cp ${partitionTemplate} flash.xml

# 7. Flash!
./flash.sh -c flash.xml jetson-agx-orin-devkit mmcblk0p1
```

### Single-Variant Optimization

When only one firmware variant is defined (single board), jetpack-nixos pre-executes flash.sh with `--no-flash` at Nix build time, producing a `flashcmd.txt` that can be run anywhere:

```nix
# In device-pkgs/default.nix
useFlashCmd = builtins.length cfg.firmware.variants == 1;
```

This is faster and more reproducible — the signing and configuration happen in the Nix sandbox, and the final script just runs `flashcmd.txt`.

## Firmware Variants

Each Jetson SoM can have multiple hardware revisions (FAB numbers, board SKUs):

```nix
# From modules/flash-script.nix, for Orin AGX:
orin-agx = [
  { boardid = "3701"; boardsku = "0000"; fab = "300"; chipsku = "00:00:00:D0"; }  # base
  { boardid = "3701"; boardsku = "0004"; fab = "300"; chipsku = "00:00:00:D2"; }  # 32GB
  { boardid = "3701"; boardsku = "0005"; fab = "300"; chipsku = "00:00:00:D0"; }  # 64GB
];
```

With a single variant, the flash script is pre-compiled. With multiple, the flash script auto-detects the board at flash time.

## UEFI Firmware (EDK2)

jetpack-nixos builds NVIDIA's EDK2 fork from source:

```nix
# Simplified from the UEFI firmware build
uefi-firmware = buildUefiFirmware {
  # NVIDIA's forks of EDK2
  src = fetchFromGitHub { owner = "NVIDIA"; repo = "edk2-nvidia"; ... };
  edk2Src = fetchFromGitHub { owner = "NVIDIA"; repo = "edk2"; ... };
  
  # Configurable!
  logo = cfg.firmware.uefi.logo;  # NixOS logo by default
  debugMode = cfg.firmware.uefi.debugMode;
  patches = cfg.firmware.uefi.edk2NvidiaPatches;
};
```

### UEFI Options You Can Set

```nix
hardware.nvidia-jetpack.firmware.uefi = {
  # Custom boot logo (converted to BMP automatically)
  logo = ./my-logo.svg;
  
  # Debug output on serial console
  debugMode = true;
  errorLevelInfo = true;
  
  # Patch the EDK2 source
  edk2NvidiaPatches = [ ./my-uefi-fix.patch ];
  
  # UEFI Secure Boot
  secureBoot = {
    enrollDefaultKeys = true;
    defaultPkEslFile = ./keys/PK.esl;
    defaultKekEslFile = ./keys/KEK.esl;
    defaultDbEslFile = ./keys/db.esl;
    signer = {
      cert = ./keys/signer.crt;
      key = "/run/keys/signer.key";  # NOT in Nix store!
    };
  };
};
```

### Boot Order

```nix
# Control UEFI boot priority
hardware.nvidia-jetpack.firmware.initialBootOrder = [
  "nvme"      # Try NVMe first
  "emmc"      # Then eMMC
  "sd"        # Then SD card
  "usb"       # Then USB
  "pxev4"     # Then PXE
];
```

This gets compiled into a device tree overlay (`DefaultBootOrder.dtbo`) that's embedded in the flash image.

## OP-TEE (Trusted Execution)

OP-TEE provides ARM TrustZone — a secure execution environment isolated from Linux:

```nix
# In configuration.nix
hardware.nvidia-jetpack.firmware.optee = {
  # tee-supplicant daemon (enabled by default)
  supplicant.enable = true;
  
  # PKCS#11 support (hardware crypto tokens)
  pkcs11Support = true;
  
  # Load custom Trusted Applications
  supplicant.trustedApplications = [ myTa ];
  
  # Log levels (0=none, 4=debug)
  coreLogLevel = 2;
  taLogLevel = 2;
};
```

OP-TEE runs in the "secure world" alongside Linux (the "normal world"):

```
┌─── ARM CPU ────────────────────────────────┐
│                                            │
│  Normal World          │  Secure World     │
│  ┌──────────────┐      │  ┌─────────────┐  │
│  │    Linux     │◄────►│  │   OP-TEE    │  │
│  │              │ SMC  │  │   OS        │  │
│  │  tee-suppl.  │ calls│  │  ┌───────┐  │  │
│  │  (userspace) │      │  │  │  TAs   │  │  │
│  └──────────────┘      │  │  └───────┘  │  │
│                        │  └─────────────┘  │
└────────────────────────┴───────────────────┘
```

## Secure Boot

Full chain of trust from power-on to Linux kernel:

```nix
hardware.nvidia-jetpack.firmware.secureBoot = {
  # PKC key — validates firmware partitions
  pkcFile = "/run/keys/pkc.pem";  # NEVER put in Nix store!
  
  # SBK key — encrypts firmware partitions
  sbkFile = "/run/keys/sbk.key";  # NEVER put in Nix store!
  
  # For Nix sandbox access to keys
  requiredSystemFeatures = [ "jetson-keys" ];
  
  # Pre-signing setup (e.g., HSM initialization)
  preSignCommands = pkgs: ''
    export PKCS11_MODULE=${pkgs.opensc}/lib/opensc-pkcs11.so
  '';
};
```

## Capsule OTA Updates

The crown jewel: automatic firmware updates via `nixos-rebuild switch`.

### How It Works

1. NixOS builds a **UEFI capsule** — a signed firmware update blob
2. During boot loader installation, the capsule is placed on the ESP
3. On next reboot, UEFI detects the capsule and applies it
4. Firmware updates atomically (A/B partition scheme)

```
nixos-rebuild switch
  → builds new NixOS generation
  → builds new firmware (UEFI, OP-TEE, DTBs)
  → packages as UEFI capsule
  → ota-check-firmware: "is current firmware different?"
  → if yes: ota-apply-capsule-update → places capsule on ESP
  → next reboot: UEFI applies capsule
```

### Enabling Capsule Updates

```nix
hardware.nvidia-jetpack.firmware.autoUpdate = true;
```

That's it. The module handles everything else.

### What Happens Under the Hood

From `modules/capsule-updates.nix`:

```nix
# Hooks into bootloader installation
boot.loader.systemd-boot.extraInstallCommands = lib.getExe updateFirmware;

# The update script:
# 1. Skip if running in a VM (image builds, etc.)
if systemd-detect-virt --quiet; then exit 0; fi

# 2. Skip if explicitly disabled
if [[ -v JETPACK_NIXOS_SKIP_CAPSULE_UPDATE ]]; then exit 0; fi

# 3. Check if firmware version matches
if ! ota-check-firmware -b; then
  # 4. Setup EFI variables for the update
  ota-setup-efivars jetson-agx-orin-devkit
  # 5. Apply the capsule
  ota-apply-capsule-update ${uefiCapsuleUpdate}
else
  # 6. Firmware matches, abort any pending update
  ota-abort-capsule-update
fi
```

### Safety: The `nixos-rebuild test` Guard

If you use `nixos-rebuild test` (which doesn't create a permanent generation), the capsule update is **skipped**:

```nix
# Compare persisted generation to active system
latest_generation=$(readlink -f /nix/var/nix/profiles/system)
current_system=$(readlink -f /run/current-system)
if [[ $latest_generation != "$current_system" ]]; then
  echo "Skipping capsule update, current active system not persisted"
  exit 0
fi
```

### Capsule Authentication

For production deployments, you can sign capsule updates:

```nix
hardware.nvidia-jetpack.firmware.uefi.capsuleAuthentication = {
  enable = true;
  trustedPublicCertPemFile = ./keys/capsule-trust.der;
  otherPublicCertPemFile = ./keys/capsule-signer.pem;
  signerPrivateCertPemFile = "/run/keys/capsule-private.pem";
};
```

## Partition Template

The flash process uses an XML template that describes all firmware partitions:

```xml
<!-- Simplified partition layout for T234 (Orin) -->
<partition_layout>
  <device type="qspi" instance="0">
    <partition name="mb1"     type="mb1_bootloader" />
    <partition name="mb2"     type="mb2_bootloader" />
    <partition name="tos"     type="tos" />           <!-- OP-TEE -->
    <partition name="eks"     type="eks" />           <!-- Encrypted keystore -->
    <partition name="uefi"    type="uefi" />          <!-- EDK2 -->
    <partition name="dtb"     type="kernel_dtb" />    <!-- Device tree -->
    <!-- A/B redundancy for OTA -->
    <partition name="mb1_b"   type="mb1_bootloader" />
    <partition name="mb2_b"   type="mb2_bootloader" />
    <partition name="tos_b"   type="tos" />
    <partition name="uefi_b"  type="uefi" />
  </device>
</partition_layout>
```

The A/B scheme means updates are atomic — if power fails during an update, the device boots from the previous slot.

## Fuse Programming

**Warning: Fuse burning is PERMANENT and IRREVERSIBLE.**

Fuses enable hardware security features (secure boot enforcement):

```bash
# Generated fuse script
nix run .#fuseScript -- -X odmfuse.xml
```

```nix
# Fuse arguments
hardware.nvidia-jetpack.flashScriptOverrides.fuseArgs = [ 
  # Additional fuse arguments at runtime
];
```

## Flash Script Outputs

Everything is exposed as `system.build` attributes:

```nix
# In your NixOS config, after building:
config.system.build.flashScript         # Recommended (initrd flash)
config.system.build.initrdFlashScript   # Same as flashScript
config.system.build.legacyFlashScript   # Old method (direct USB)
config.system.build.rcmBoot             # RCM boot (no flash)
config.system.build.fuseScript          # Fuse programming (DANGER)
config.system.build.signedFirmware      # Signed firmware images
config.system.build.uefiCapsuleUpdate   # Capsule for OTA
```

### Building Flash Scripts

```bash
# Build the flash script (runs on x86_64!)
nix build .#nixosConfigurations.my-jetson.config.system.build.flashScript

# Or use the flake app shortcut
nix run .#flash-orin-agx
```

## The Cross-Architecture Challenge

Flash tools are x86_64 binaries, but the target system is aarch64:

```
x86_64 host:
  ├── flash-tools (NVIDIA's prebuilt x86 binaries)
  ├── tegraflash.py (Python scripts that orchestrate flashing)
  ├── signing tools
  └── USB communication tools

aarch64 target:
  ├── kernel Image
  ├── initrd
  ├── device trees
  ├── UEFI firmware (built from source)
  └── OP-TEE (built from source)
```

jetpack-nixos handles this by building firmware as aarch64 derivations but wrapping them in x86_64 flash scripts.

## Summary

| Concept | What |
|---------|------|
| QSPI | Small flash chip holding firmware (MB1, MB2, OP-TEE, UEFI) |
| Initrd flash | Boot temporary Linux via USB, flash from within |
| Legacy flash | Direct USB programming from x86_64 host |
| RCM boot | Boot without flashing (testing) |
| Firmware variants | Board IDs, SKUs, FABs for hardware revision detection |
| UEFI / EDK2 | Bootloader, built from NVIDIA's fork, fully configurable |
| OP-TEE | ARM TrustZone secure OS for Trusted Applications |
| Capsule OTA | Firmware updates via `nixos-rebuild switch` |
| A/B partitions | Atomic firmware updates with rollback |
| Secure boot | PKC + SBK keys for firmware authentication/encryption |
| Capsule auth | Sign OTA capsules for production security |
| Flash script | Nix derivation wrapping NVIDIA's x86_64 flash tools |

**The mental model**: Flashing is the chicken-and-egg problem of embedded systems. jetpack-nixos solves it by turning the entire flash process into a reproducible Nix derivation — your firmware, your kernel, your device trees, all built from your NixOS config, packaged into a script that programs the hardware. After the initial flash, capsule OTA updates mean you never need to plug in a USB cable again.

---

**Previous**: [← Pill 07: NixOS Modules — Declaring Your System](07-nixos-modules.md)
**Next**: [Pill 09: Building Your Own Jetson System →](09-building-your-system.md)
