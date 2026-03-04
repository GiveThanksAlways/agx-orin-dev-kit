# IMU-3D — MPU-9250 Real-Time 3D Visualization

Live Three.js visualization of a GY-9250 (MPU-9250) IMU connected to a
Jetson AGX Orin over I2C.  Move the sensor and watch it mirrored in your
browser — accelerometer, gyroscope, magnetometer, orientation, all streaming
at 50 Hz.

![demo](https://img.shields.io/badge/Three.js-3D-06b6d4?style=flat-square)
![imu](https://img.shields.io/badge/MPU--9250-I2C-22c55e?style=flat-square)
![nix](https://img.shields.io/badge/Nix-flake-7eb0d5?style=flat-square)

## Wiring

Connect the GY-9250 module to the Jetson AGX Orin **J40 40-pin header**:

```
 GY-9250 Module           Jetson AGX Orin (J40 Header)
┌──────────────┐          ┌───────────────────────────┐
│ VCC  ────────┼──────────┤ Pin 2   (5.0 V)           │
│ GND  ────────┼──────────┤ Pin 6   (GND)             │
│ SDA  ────────┼──────────┤ Pin 3   (I2C5_DAT)        │
│ SCL  ────────┼──────────┤ Pin 5   (I2C5_CLK)        │
│ AD0  ────────┤ (NC)     │  → pulled low on-board    │
│ NCS  ────────┤ (NC)     │  → pulled high on-board   │
│ INT  ────────┤ (NC)     │                           │
└──────────────┘          └───────────────────────────┘
```

**No external pull-up resistors** needed — the GY-9250 module has on-board
pull-ups on SDA and SCL.

Power: connect VCC to **Pin 2 (5 V)** — the module's on-board regulator steps
it down to 3.3 V for the MPU-9250.

Default I2C address: **0x68** (AD0 pulled low on-board).

## Prerequisites

Apply the NixOS sensors configuration so the I2C bus and `inv_mpu6050` driver are ready:

```bash
cd ../nixos
sudo nixos-rebuild switch --flake .#nixos-sensors
```

Verify the sensor is visible:

```bash
i2cdetect -y 7          # should show 0x68
i2cget -y 7 0x68 0x75   # WHO_AM_I → 0x71
```

## Quick Start

```bash
cd examples/IMU-3D
nix develop

# Live IMU (sensor must be wired + NixOS sensors config applied)
python server.py

# …or test the UI without hardware
python server.py --mock
```

Then open **http://192.168.8.162:9090** in a browser on your laptop.

> VS Code's SSH remote extension will also auto-detect the port and offer to
> forward it — you can use `localhost:9090` in that case.
>
> **Manual SSH tunnel** (from PowerShell / Terminal on your laptop):
> ```
> ssh -L 9090:localhost:9090 Orin-AGX-NixOS
> ```
> Then open `http://localhost:9090`.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mock` | off | Simulated IMU data (no hardware needed) |
| `--bus N` | `7` | I2C bus number |
| `--addr 0xNN` | `0x68` | MPU-9250 I2C address |
| `--port N` | `9090` | HTTP / WebSocket port |
| `--rate N` | `50` | Sample rate in Hz |

## What You See

- **3D PCB model** that rotates in real time matching the physical IMU
- **Accelerometer** — X / Y / Z in g (gravity)
- **Gyroscope** — X / Y / Z in °/s (angular rate)
- **Magnetometer** — X / Y / Z in µT (magnetic field)
- **Orientation** — Roll / Pitch / Yaw in degrees (from Madgwick AHRS)
- **Quaternion** — raw w / x / y / z from the sensor fusion filter
- **Temperature** — on-chip thermometer
- **Sample rate** — live Hz counter

Buttons:
- **Calibrate** — resets the AHRS filter (makes current orientation = zero)
- **Reset Camera** — snaps the 3D camera back to default view

## Architecture

```
┌──────────────────┐     I2C      ┌─────────────┐   WebSocket   ┌──────────┐
│  GY-9250 Module  │────────────▸ │  server.py   │─────────────▸ │  Browser │
│  (MPU-9250)      │   /dev/i2c-7 │  (aiohttp)   │  JSON @ 50Hz │ Three.js │
└──────────────────┘              └─────────────┘               └──────────┘
```

- **server.py** reads raw accel/gyro/mag over smbus2, runs a Madgwick 6-DOF
  AHRS filter to estimate orientation, and broadcasts JSON packets over
  WebSocket to all connected browsers.
- **index.html** renders a Three.js scene with the PCB model, applies the
  quaternion for 3D rotation, and displays live telemetry gauges.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `WHO_AM_I returned 0xff` | Check wiring — SDA/SCL may be swapped or disconnected |
| `Permission denied: /dev/i2c-7` | Your user needs the `i2c` group — check `services.orin-sensors.cameraUsers` |
| `smbus2 not available` | Run inside `nix develop`, or the server falls back to `--mock` mode |
| Page loads but board doesn't move | Confirm WebSocket connects (green dot in top bar) |
| Yaw drifts slowly | Expected with 6-DOF filter (no mag fusion) — calibrate periodically |
