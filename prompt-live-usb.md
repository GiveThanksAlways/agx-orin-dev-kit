# Prompt: Flash Stock JetPack 6.2 Ubuntu to USB for Bare-Metal Tinygrad Testing

> **Give this entire document to an AI agent running on your x86_64 Ubuntu 22.04 host (or WSL2).**
> It contains everything needed to prepare a bootable USB drive for the Jetson AGX Orin.

---

## Background

I have an **NVIDIA Jetson AGX Orin 64GB Dev Kit** running NixOS (jetpack-nixos) on its internal NVMe.
I've already validated tinygrad's Tegra backend inside an Incus system container on that box,
but the container shares the host kernel — I need a **true bare-metal Ubuntu test**.

**Goal:** Flash stock JetPack 6.2 (L4T r36.4.4) Ubuntu to a USB drive, boot the Orin from it,
run tinygrad's full test suite, then unplug the USB to return to NixOS.
The NVMe must NOT be touched.

---

## What You (the AI agent) Need to Do

You are running on an **x86_64 Ubuntu 22.04** machine (may be WSL2).
You have access to a terminal. Follow these phases in order.

---

## Phase 1 — Prerequisites

Install required packages on the x86_64 host:

```bash
sudo apt-get update
sudo apt-get install -y \
  wget tar qemu-user-static binfmt-support \
  python3 python3-yaml sshpass abootimg \
  nfs-kernel-server libxml2-utils cpio
```

Verify you have **at least 50 GB free** in your working directory:

```bash
df -h .
```

---

## Phase 2 — Download JetPack 6.2 BSP + Root Filesystem

```bash
mkdir -p ~/jetpack-usb && cd ~/jetpack-usb

# L4T r36.4.4 Driver Package (BSP)
wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.4/release/Jetson_Linux_R36.4.4_aarch64.tbz2

# Sample Root Filesystem (Ubuntu 22.04 based)
wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.4/release/Tegra_Linux_Sample-Root-Filesystem_R36.4.4_aarch64.tbz2
```

> **Note:** These are ~1.5 GB each. If the exact URLs have changed, find the current
> "L4T r36.4.4" BSP and Sample Root Filesystem downloads at
> <https://developer.nvidia.com/embedded/jetson-linux-archive>.

---

## Phase 3 — Extract BSP and Apply Root Filesystem

```bash
cd ~/jetpack-usb

# Extract BSP
tar xf Jetson_Linux_R36.4.4_aarch64.tbz2

# Extract rootfs into the BSP's rootfs directory
cd Linux_for_Tegra/rootfs
sudo tar xpf ../../Tegra_Linux_Sample-Root-Filesystem_R36.4.4_aarch64.tbz2

# Go back to BSP root and apply NVIDIA binaries into the rootfs
cd ..
sudo ./apply_binaries.sh
```

This populates the rootfs with NVIDIA's L4T libraries, firmware, and drivers.

---

## Phase 4 — Inject Tinygrad Test Script into the Rootfs

Create a first-boot test script that will run automatically (or on demand) after booting:

```bash
sudo tee rootfs/home/ubuntu/run-tinygrad-tests.sh << 'TESTSCRIPT'
#!/usr/bin/env bash
set -euo pipefail

LOG="/home/ubuntu/tinygrad-test-results.txt"
echo "=== Tinygrad Bare-Metal Test Run ===" | tee "$LOG"
echo "Date: $(date)" | tee -a "$LOG"
echo "Kernel: $(uname -r)" | tee -a "$LOG"
echo "Hostname: $(hostname)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ── Install dependencies ──
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y \
  python3.12 python3.12-venv python3.12-dev \
  clang build-essential git \
  cuda-nvrtc-12-6 cuda-nvrtc-dev-12-6 cuda-cudart-dev-12-6

# ── Set up environment ──
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/cuda-12.6/targets/aarch64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NVRTC_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib/libnvrtc.so.12
sudo ldconfig

# ── Clone tinygrad ──
cd /home/ubuntu
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
git checkout refactor-2   # ← our test branch

# ── Create venv and install ──
python3.12 -m venv /home/ubuntu/venv
source /home/ubuntu/venv/bin/activate
pip install --upgrade pip
pip install -e '.[testing]'

echo "" | tee -a "$LOG"

# ── NV=1 Tests ──
echo "========== NV=1 test_ops ==========" | tee -a "$LOG"
NV=1 python3 -m pytest test/test_ops.py -v --tb=short 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "========== NV=1 supporting tests ==========" | tee -a "$LOG"
NV=1 python3 -m pytest \
  test/test_schedule.py test/test_tensor.py \
  test/test_custom_kernel.py test/test_pickle.py \
  -v --tb=short 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "========== NV=1 test_jit ==========" | tee -a "$LOG"
NV=1 python3 -m pytest test/test_jit.py -v --tb=short 2>&1 | tee -a "$LOG" || true

echo "" | tee -a "$LOG"
echo "========== NV=1 test_randomness ==========" | tee -a "$LOG"
NV=1 python3 -m pytest test/test_randomness.py -v --tb=short 2>&1 | tee -a "$LOG" || true

echo "" | tee -a "$LOG"
echo "========== NV=1 test_graph ==========" | tee -a "$LOG"
NV=1 python3 -m pytest test/backend/test_graph.py -v --tb=short 2>&1 | tee -a "$LOG" || true

echo "" | tee -a "$LOG"

# ── CUDA=1 Tests ──
echo "========== CUDA=1 test_ops ==========" | tee -a "$LOG"
CUDA=1 python3 -m pytest test/test_ops.py -v --tb=short 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "========== CUDA=1 supporting tests ==========" | tee -a "$LOG"
CUDA=1 python3 -m pytest \
  test/test_schedule.py test/test_tensor.py \
  test/test_custom_kernel.py test/test_pickle.py \
  -v --tb=short 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "========== CUDA=1 test_jit ==========" | tee -a "$LOG"
CUDA=1 python3 -m pytest test/test_jit.py -v --tb=short 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "========== CUDA=1 test_randomness ==========" | tee -a "$LOG"
CUDA=1 python3 -m pytest test/test_randomness.py -v --tb=short 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "========== CUDA=1 test_graph ==========" | tee -a "$LOG"
CUDA=1 python3 -m pytest test/backend/test_graph.py -v --tb=short 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "========== DONE ==========" | tee -a "$LOG"
echo "Results saved to $LOG"
TESTSCRIPT

sudo chmod +x rootfs/home/ubuntu/run-tinygrad-tests.sh
```

Also ensure the default user account works:

```bash
# Set a password for the ubuntu user (JetPack default user)
sudo chroot rootfs /bin/bash -c "echo 'ubuntu:ubuntu' | chpasswd"
# Ensure ubuntu user owns the test script
sudo chroot rootfs /bin/bash -c "chown ubuntu:ubuntu /home/ubuntu/run-tinygrad-tests.sh"
```

---

## Phase 5 — Flash the USB Drive

### 5a. Put the Jetson in Recovery Mode

On the **Jetson AGX Orin Dev Kit** (physically):

1. Power off the Jetson
2. Hold the **REC** (Force Recovery) button (middle button on the carrier board)
3. While holding REC, press and release the **RESET** button (leftmost button)
4. Release REC after ~2 seconds
5. Connect a **USB-C cable** from the Jetson's flashing port to your x86_64 host

Verify recovery mode on the x86_64 host:

```bash
lsusb | grep -i nvidia
# Should show: "NVIDIA Corp. APX"
```

> **WSL2 Note:** You need to pass the USB device from Windows to WSL2 first:
> ```powershell
> # In an elevated PowerShell on Windows:
> winget install usbipd
> usbipd list            # Find the NVIDIA APX device
> usbipd bind --busid <BUS_ID>
> usbipd attach --wsl --busid <BUS_ID>
> ```
> Then `lsusb` inside WSL2 should show the NVIDIA device.

### 5b. Flash to USB

**IMPORTANT:** This flashes the rootfs to an **external USB drive**, not the internal NVMe.
The Jetson must have a USB drive plugged into one of its USB-A ports.

```bash
cd ~/jetpack-usb/Linux_for_Tegra

# Flash to external USB device (sda1)
# The --external-device flag targets the USB drive
sudo ./tools/kernel_flash/l4t_initrd_flash.sh \
  --external-device sda1 \
  -c tools/kernel_flash/flash_l4t_t234_nvme.xml \
  --showlogs \
  jetson-agx-orin-devkit external
```

> This takes 10–30 minutes. It will:
> 1. Upload a small initrd to the Jetson via USB
> 2. The Jetson boots into initrd mode
> 3. The initrd partitions and writes the rootfs to the USB drive (sda)
> 4. QSPI firmware may be updated (same L4T version — harmless, won't break NixOS)

Wait for `Flashing completed successfully` or similar message.

---

## Phase 6 — Boot from USB and Run Tests

### On the Jetson (physically):

1. Disconnect the USB-C flashing cable
2. Make sure the USB drive is plugged into a USB-A port on the Jetson
3. Power on the Jetson
4. **Immediately press ESC** when you see the NVIDIA splash screen to enter UEFI boot menu
5. Select the **USB drive** from the boot device list
6. Ubuntu will boot from USB (first boot may take a couple minutes for setup)

### Log in and run tests:

```bash
# Default credentials: ubuntu / ubuntu
# (you may be prompted to change password on first login)

# Run the test script
chmod +x ~/run-tinygrad-tests.sh
~/run-tinygrad-tests.sh

# Results will be saved to ~/tinygrad-test-results.txt
```

### When done:

1. Power off the Jetson
2. Remove the USB drive
3. Power on — Jetson boots back to NixOS on NVMe automatically

---

## Expected Results (Container Baseline to Compare Against)

| Test File | NV=1 Container | CUDA=1 Container | Bare-Metal (fill in) |
|---|---|---|---|
| test_ops | 409 passed, 7 skipped | 409 passed, 7 skipped | |
| test_schedule + test_tensor + test_custom_kernel + test_pickle | 243 passed, 15 skipped | 243 passed, 15 skipped | |
| test_jit | CRASHED (NV=1) | 44 passed, 9 skipped | |
| test_randomness | CRASHED (2) (NV=1) | 30 passed, 2 skipped | |
| test_graph | CRASHED (3) (NV=1) | 7 passed, 3 skipped | |

**Key question we're answering:** Do the NV=1 crashes in test_jit, test_randomness, and test_graph
reproduce on bare-metal Ubuntu (stock kernel), or are they container artifacts?

---

## Troubleshooting

- **`lsusb` doesn't show NVIDIA APX:** Re-do recovery mode sequence. Make sure you're using the correct USB-C port (the one closest to the power jack on AGX Orin Dev Kit).
- **WSL2 can't see USB:** Install `usbipd-win`, bind and attach the device. See WSL2 note above.
- **Flash fails with "no USB device":** The 32GB+ USB drive must be plugged into the Jetson's USB-A port (not the host). The USB-C cable is only for recovery/flashing communication.
- **UEFI doesn't show USB option:** Try a different USB-A port on the Jetson. Ensure the flash completed successfully.
- **CUDA tests fail on boot:** Verify `/usr/local/cuda-12.6` exists. Run `ldconfig` and check `ldconfig -p | grep nvrtc`.
- **`refactor-2` branch not found:** The branch may have been merged. Try `main` instead, or check `git branch -r` for available branches.