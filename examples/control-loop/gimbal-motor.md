# Gimbal Motor Rig — Neural vs PID Control with tinygrad NV=1

**Goal**: Build a direct-drive brushless motor rig with a pendulum arm, control it with a neural network policy running at 4.8 kHz on tinygrad NV=1, and demonstrate measurable improvement over classical PID. Use hardware that can later go onto a drone gimbal or robot arm joint.

---

## Table of Contents

1. [Software Architecture](#1-software-architecture)
2. [Why Neural Network Beats PID Here — Specific Cases](#2-why-neural-network-beats-pid-here--specific-cases)
3. [Inner Loop vs Outer Loop — What Each Layer Does](#3-inner-loop-vs-outer-loop--what-each-layer-does)
4. [Hardware — What to Buy](#4-hardware--what-to-buy)
5. [Wiring](#5-wiring)
6. [STM32 Firmware — Inner Loop (PID / FOC)](#6-stm32-firmware--inner-loop-pid--foc)
7. [Orin Software — Outer Loop (Neural Policy)](#7-orin-software--outer-loop-neural-policy)
8. [Build Plan](#8-build-plan)
9. [Test & Benchmark Plan](#9-test--benchmark-plan)
10. [Training the Neural Policy](#10-training-the-neural-policy)
11. [Path to Drone Gimbal / Robot Arm](#11-path-to-drone-gimbal--robot-arm)

---

## 1. Software Architecture

### 1a. Two-Layer Control Architecture

This mirrors how every real drone and robot works: a fast inner loop on an MCU, a smart outer loop on the main compute.

```
┌──────────────────────────────────────────────────────────────────────┐
│  ORIN AGX — OUTER LOOP (tinygrad NV=1)                               │
│  Rate: 1-4.8 kHz                                                     │
│                                                                      │
│  SPI read AS5048A  (shaft angle, 14-bit)         → 2 bytes, ~5µs    │
│  SPI read ICM-42688 (gyro xyz + accel xyz)       → 12 bytes, ~10µs  │
│                                                                      │
│  Build state vector:                                                  │
│    [angle, angular_vel, gyro_xyz, accel_xyz, prev_torque,            │
│     target_angle, target_vel] = 13 floats → 26 bytes FP16           │
│                                                                      │
│  ctypes.memmove(gpu_in, state, 26)                  ← < 1µs         │
│  run_policy()  # @TinyJit HCQGraph                  ← ~100µs        │
│  dev.synchronize()                                   ← ~100µs        │
│  ctypes.memmove(torque, gpu_out, 2)                 ← < 1µs         │
│                                                                      │
│  UART write torque command to STM32 (2 bytes)       ← ~5µs          │
│                                                                      │
│  Total: ~220 µs = ~4.5 kHz                                          │
└──────────────────┬───────────────────────────────────────────────────┘
                   │ UART (ttyTHS1 ↔ STM32 USART, 2 Mbaud)
                   │ Protocol: [0xAA, torque_hi, torque_lo, checksum]
┌──────────────────▼───────────────────────────────────────────────────┐
│  STM32H743 — INNER LOOP (FOC / current control)                     │
│  Rate: 10-40 kHz                                                     │
│                                                                      │
│  Receive torque command from Orin                                    │
│  Read motor hall sensors / encoder (built into motor driver)         │
│  FOC algorithm: torque → d/q currents → PWM duty cycles             │
│  Output PWM to motor driver (DRV8302)                                │
│                                                                      │
│  If no Orin command for > 5 ms: safety hold (zero torque)            │
└──────────────────────────────────────────────────────────────────────┘
```

### 1b. What the Neural Policy Learns (Outer Loop)

The neural network replaces a PID + gain-scheduling + feedforward controller. Its inputs and outputs:

```python
# Input state vector (13 floats, FP16)
state = [
    angle,              # AS5048A: current shaft angle (rad), from absolute encoder
    angular_velocity,   # Computed: d(angle)/dt (rad/s), or from gyro
    gyro_x, gyro_y, gyro_z,   # ICM-42688: angular rates
    accel_x, accel_y, accel_z, # ICM-42688: accelerations (gravity + disturbance)
    prev_torque,        # Last commanded torque (for smooth transitions)
    target_angle,       # Desired angle from trajectory planner
    target_velocity,    # Desired angular velocity
    target_accel,       # Desired angular acceleration (feedforward)
]

# Output (1 float, FP16)
torque = model(state)   # Range [-1, 1], scaled to motor's max torque
```

### 1c. Orin Control Loop Code

```python
import os, ctypes, time, struct
os.environ["NV"] = "1"
import numpy as np
from tinygrad import Tensor, Device, TinyJit, dtypes

dev = Device["NV"]

# Dimensions
STATE_DIM = 13
ACTION_DIM = 1

# Neural policy: 13→128→128→1 (~18K params)
class MotorPolicy:
    def __init__(self):
        from tinygrad import nn
        self.l1 = nn.Linear(STATE_DIM, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, ACTION_DIM)
    def __call__(self, x):
        return self.l3(self.l2(self.l1(x).relu()).relu()).tanh()

policy = MotorPolicy()
# TODO: load trained weights from file

# Static GPU buffers (zero-copy via Tegra unified memory)
static_state = Tensor.zeros(1, STATE_DIM, dtype=dtypes.float16).contiguous().realize()
static_action = Tensor.zeros(1, ACTION_DIM, dtype=dtypes.float16).contiguous().realize()
dev.synchronize()

state_addr = static_state._buffer()._buf.cpu_view().addr
action_addr = static_action._buffer()._buf.cpu_view().addr

@TinyJit
def run_policy():
    static_action.assign(policy(static_state)).realize()

# Warmup
for _ in range(20):
    run_policy(); dev.synchronize()

# ─── Sensor setup ─────────────────────────
import spidev

spi_enc = spidev.SpiDev()  # AS5048A on SPI0, CS1
spi_enc.open(0, 1)
spi_enc.max_speed_hz = 10_000_000
spi_enc.mode = 0b01  # CPOL=0, CPHA=1 for AS5048A

spi_imu = spidev.SpiDev()  # ICM-42688 on SPI0, CS0
spi_imu.open(0, 0)
spi_imu.max_speed_hz = 10_000_000
spi_imu.mode = 0b11

import serial
uart = serial.Serial('/dev/ttyTHS1', 2_000_000, timeout=0.001)

# ─── Main loop ─────────────────────────────
state_np = np.zeros((1, STATE_DIM), dtype=np.float16)
result_np = np.empty(ACTION_DIM, dtype=np.float16)
state_nbytes = STATE_DIM * 2
action_nbytes = ACTION_DIM * 2

target_angle = 0.0  # setpoint (updated by trajectory generator)
prev_angle = 0.0
prev_torque = 0.0
dt = 1.0 / 4800  # estimated loop period

while True:
    t0 = time.perf_counter_ns()

    # 1. Read encoder (AS5048A: read angle register 0x3FFF)
    raw = spi_enc.xfer2([0xFF, 0xFF])
    angle_raw = ((raw[0] & 0x3F) << 8) | raw[1]  # 14-bit angle
    angle_rad = angle_raw * (2 * 3.14159265 / 16384.0)

    # 2. Read IMU (ICM-42688: burst read accel + gyro)
    imu_data = spi_imu.xfer2([0x1F | 0x80] + [0]*12)  # read 12 bytes starting at ACCEL_XOUT_H
    accel = np.frombuffer(bytes(imu_data[1:7]), dtype='>i2').astype(np.float32) * (9.81 / 16384.0)
    gyro = np.frombuffer(bytes(imu_data[7:13]), dtype='>i2').astype(np.float32) * (3.14159 / 180.0 / 131.0)

    # 3. Compute angular velocity
    angular_vel = (angle_rad - prev_angle) / dt
    prev_angle = angle_rad

    # 4. Pack state vector
    state_np[0, 0] = angle_rad
    state_np[0, 1] = angular_vel
    state_np[0, 2:5] = gyro
    state_np[0, 5:8] = accel
    state_np[0, 8] = prev_torque
    state_np[0, 9] = target_angle
    state_np[0, 10] = 0.0   # target velocity
    state_np[0, 11] = 0.0   # target acceleration
    state_np[0, 12] = 0.0   # reserved

    # 5. GPU inference via Tegra unified memory
    ctypes.memmove(state_addr, state_np.ctypes.data, state_nbytes)
    run_policy()
    dev.synchronize()
    ctypes.memmove(result_np.ctypes.data, action_addr, action_nbytes)

    torque_cmd = float(result_np[0])
    prev_torque = torque_cmd

    # 6. Send torque to STM32 via UART
    torque_int16 = int(np.clip(torque_cmd * 32767, -32767, 32767))
    packet = struct.pack('>BhB', 0xAA, torque_int16, (0xAA + (torque_int16 >> 8) + (torque_int16 & 0xFF)) & 0xFF)
    uart.write(packet)

    t1 = time.perf_counter_ns()
    # Timing feedback for benchmarking
```

---

## 2. Why Neural Network Beats PID Here — Specific Cases

This section is about the specific motor rig, not abstract advantages. Each case is testable on the hardware.

### 2a. Motor Cogging Torque

Brushless gimbal motors have **cogging torque** — periodic resistance from magnetic detents (6N×poles interactions per revolution). A GM4108H has 22 poles × 24 slots, producing 132 cogging periods per revolution.

**PID behavior**: Constant gains. At low speed, cogging causes the motor to "stutter" between detents. PID reacts *after* the position error appears. At very low speed (< 10 rpm), PID-controlled gimbal motors exhibit visible jitter.

**Neural network behavior**: Trained on data that includes cogging, the NN learns a feedforward torque profile that *preemptively* cancels cogging. Before the motor reaches a detent, the NN applies extra torque. Think of it as the network learning `cogging_compensation(angle)` as an implicit function.

**Test procedure**: Command slow constant-velocity rotation (1 rpm). Measure angle error std deviation.
- PID: expect ~0.5-2° error oscillation at cogging frequency
- NN: expect < 0.2° — the NN compensates cogging proactively

### 2b. Gravity Compensation with Pendulum

With a weighted rod attached to the motor shaft, gravity torque is `τ = m·g·L·sin(θ)`. This is a nonlinear function of angle.

**PID behavior**: Uses constant I-gain to accumulate steady-state error. Works at one angle, but when the target angle changes, the integrator must wind up/down. This causes overshoot and slow settling.

**Neural network behavior**: Learns `gravity_compensation(angle, angular_vel)` directly. No integrator windup. The NN outputs the exact gravity torque plus the dynamic correction simultaneously.

**Test procedure**: Command step changes in target angle (0° → 45° → 90° → 45° → 0°).
- PID: measure settling time and overshoot at each step
- NN: expect 30-50% faster settling, near-zero overshoot

### 2c. Payload Change (Add/Remove Weight)

Attach different weights to the rod tip (e.g., 50g, 100g, 200g).

**PID behavior**: Tuned for one payload. When you add weight, the system becomes underdamped (oscillates) or overdamped (sluggish) depending on the original tuning. Typically requires manual retuning of P/I/D gains for each payload.

**Neural network behavior**: If trained with randomized payloads (domain randomization), the NN adapts to different inertias without retuning. The network implicitly estimates the payload from the observed dynamics (how fast the arm responds to torque commands).

**Test procedure**: Tune PID for 100g payload. Then test with 50g and 200g without changing gains. Run the NN (trained with 50-200g randomization) with the same payloads.
- PID: expect degraded performance (overshoot/sluggishness) at non-nominal payloads
- NN: expect consistent performance across all payloads

### 2d. Disturbance Rejection (Flick the Rod)

Physically disturb the rod by tapping it with a finger.

**PID behavior**: Reacts after the disturbance causes a position error. The response time is limited by the derivative gain (which amplifies sensor noise) and the loop rate.

**Neural network behavior**: With gyro data in the state vector, the NN can detect the onset of a disturbance within 1 cycle (0.2 ms at 4.8 kHz) from the angular acceleration, before the position error grows large. This is like having a perfect D-term without noise amplification.

**Test procedure**: Hold target at 0°. Flick the rod. Measure peak deflection and recovery time.
- PID at 500 Hz: recovery in ~100-200 ms
- NN at 4.8 kHz: recovery in ~20-50 ms (hypothesis: 2-5x faster due to both better policy AND higher rate)

### 2e. Trajectory Tracking (Sine Wave)

Command a sinusoidal target: `target = A·sin(2π·f·t)`.

**PID behavior**: The tracking error increases with frequency due to phase lag. At high frequencies, PID can't keep up, especially with the nonlinear gravity torque.

**Neural network behavior**: With target velocity and acceleration in the input, the NN can implement feedforward compensation. It predicts the needed torque to follow the trajectory, then corrects for errors.

**Test procedure**: Sweep sine frequency from 0.5 Hz to 20 Hz with 30° amplitude.
- PID: measure tracking error (RMSE) vs frequency
- NN: expect lower RMSE, especially above 5 Hz where PID phase lag dominates

---

## 3. Inner Loop vs Outer Loop — What Each Layer Does

### 3a. Our Architecture Mapped to Industry Standards

```
┌─────────────────────────────────────────────────────────────────┐
│               Industry Standard         Our Demo                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PLANNER (10-30 Hz):                                            │
│    Industry: trajectory optimization, path planning             │
│    Our demo: simple trajectory generator (sine wave, steps)     │
│    Runs on: Orin CPU thread                                     │
│                                                                 │
│  OUTER LOOP / REACTIVE POLICY (50-5000 Hz):                     │
│    Industry: PID (PX4) or RL policy (Unitree, ANYmal)           │
│    Our demo: neural policy on tinygrad NV=1, 4.8 kHz            │
│    Runs on: Orin GPU via HCQGraph                               │
│    Input: angle, gyro, accel, target                            │
│    Output: torque command                                        │
│    THIS IS WHERE NV=1 WINS.                                     │
│                                                                 │
│  INNER LOOP / ACTUATOR CONTROL (1-40 kHz):                      │
│    Industry: FOC current loop, PD torque control                │
│    Our demo: PID on STM32H743, 10-40 kHz                        │
│    Runs on: STM32 MCU (hard real-time)                          │
│    Input: torque command from Orin                               │
│    Output: PWM to motor driver                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3b. Why the STM32 Inner Loop is Necessary

The STM32 handles things that must be hard real-time:

1. **Motor commutation (FOC)**: BLDC motors need precise 3-phase current control at 10-40 kHz. A missed cycle causes audible noise, vibration, or motor damage. Linux can't guarantee this.

2. **Safety watchdog**: If the Orin freezes (kernel panic, GPU hang, Python exception), the STM32 detects missing UART commands and safely shuts down the motor within 5 ms.

3. **Current limiting**: Protects the motor and driver from overcurrent. Must react within one PWM cycle (~25-100 µs).

The Orin tells the STM32 *what torque to apply*. The STM32 figures out *how to apply it* (which phase, what current, what PWM duty cycle). This separation is universal in robotics.

### 3c. Comparison to PX4/ArduPilot Architecture

| Layer | PX4 on Pixhawk | Our Demo |
|-------|----------------|----------|
| Inner loop HW | STM32F7/H7 on Pixhawk | STM32H743 on Nucleo |
| Inner loop software | PX4 rate controller (PID, 8 kHz) | SimpleFOC or custom PID (10-40 kHz) |
| Outer loop HW | Same STM32 (attitude controller) | Orin AGX (GPU neural policy) |
| Outer loop software | PX4 attitude PID (250-400 Hz) | tinygrad NV=1 neural policy (4.8 kHz) |
| Planner | PX4 mission controller (50 Hz) | Trajectory generator (10-50 Hz) |
| Communication | Internal (same MCU) | UART (2 Mbaud, ~5 µs per packet) |

**The key upgrade**: PX4's outer loop is a PID running at 250-400 Hz on a microcontroller. Our outer loop is a neural network running at 4.8 kHz on a GPU. Same architecture, 12-20x faster outer loop, with learned nonlinear dynamics instead of fixed PID gains.

---

## 4. Hardware — What to Buy

### 4a. Motor Assembly

| # | Component | Specific Part | Est. Cost | Why |
|---|-----------|--------------|----------:|-----|
| 1 | **Gimbal Motor** | iFlight GM4108H-120T (hollow shaft, 90W) | $25 | Real gimbal motor. Direct-drive (no gearbox) = control is harder = NN advantage. Same motor used on DJI Ronin-class gimbals. Hollow shaft for routing encoder cable. |
| 2 | **Motor Driver Board** | SimpleFOC Mini v1.1 | $20 | Open source FOC driver. Supports torque mode (current control). 2S-5S LiPo input. SPI interface for encoder passthrough. Pairs with DRV8302 or uses onboard DRV8313. Arduino-compatible firmware. |
| 3 | **Alternative Motor Driver** | DRV8302 BLDC driver board (if more power needed) | $30 | 45V, 30A. Full FOC support. Used in many open-source robot projects. |

### 4b. Sensors

| # | Component | Specific Part | Est. Cost | Why |
|---|-----------|--------------|----------:|-----|
| 4 | **Magnetic Encoder** | AS5048A breakout (AMS eval kit or Adafruit-compatible) | $12 | 14-bit absolute position. SPI at 10 MHz. Mounts on motor shaft with diametrically magnetized magnet (included or ~$2). This IS the encoder used on commercial robot joints. |
| 5 | **Diametric Magnet** | 6mm × 2.5mm N52 (for AS5048A) | $2 | Glues to motor shaft end. AS5048A reads field direction. |
| 6 | **IMU** | ICM-42688-P breakout (SparkFun SEN-22914) | $15 | 6-axis gyro+accel. SPI. Measures angular rate and vibration of the motor assembly. Same sensor as Pixhawk 6X. |

### 4c. MCU (Inner Loop)

| # | Component | Specific Part | Est. Cost | Why |
|---|-----------|--------------|----------:|-----|
| 7 | **STM32 Dev Board** | NUCLEO-H743ZI2 | $25 | 480 MHz Cortex-M7. Built-in debugger. 3 SPI, 4 I2C, 8 UART. Can run SimpleFOC library for FOC motor control. PX4-compatible MCU. |

### 4d. Mechanical

| # | Component | Specific Part | Est. Cost | Why |
|---|-----------|--------------|----------:|-----|
| 8 | **Motor Mount** | 3D printed bracket (or aluminum L-bracket) | $5 | Mounts GM4108H to extrusion. STL files available from iFlight community. |
| 9 | **Frame** | 2020 aluminum extrusion, 200mm (2×) + corner brackets | $15 | Rigid base. Standard maker part. |
| 10 | **Pendulum Arm** | 6mm carbon fiber tube, 150mm long | $5 | Light, stiff. Attaches to motor shaft. |
| 11 | **Weights** | M6 nuts + bolts (act as adjustable payload) | $3 | Thread onto end of carbon tube. Add/remove for payload tests. |

### 4e. Power & Interconnect

| # | Component | Specific Part | Est. Cost | Why |
|---|-----------|--------------|----------:|-----|
| 12 | **Motor Power** | 12V 5A power supply (or 3S LiPo for portability) | $15 | Powers motor driver. 12V matches 2S-3S LiPo range. |
| 13 | **Jumper wires** | M-F and F-F assortment | $8 | Connections between Orin header, STM32, sensors |
| 14 | **Breadboard** | Full-size 830 points | $6 | For prototyping sensor connections |
| 15 | **USB-C cable** | For STM32 Nucleo | $5 | Power + flashing |

### 4f. Cost Total

| Category | Cost |
|----------|-----:|
| Motor + driver | $45-55 |
| Sensors (encoder + IMU + magnet) | $29 |
| MCU | $25 |
| Mechanical (frame + arm + weights) | $28 |
| Power + interconnect | $34 |
| **Total** | **$161-171** |

### 4g. Combined Purchase List (Sensor Fusion Demo + Gimbal Motor)

Since you're building both demos, many parts are shared:

| Part | SF Demo | Gimbal Demo | Qty to Buy |
|------|:---:|:---:|---:|
| ICM-42688-P breakout | ✅ | ✅ | 1 (shared or buy 2 for convenience) |
| AS5048A breakout + magnet | — | ✅ | 1 |
| BMP390 breakout | ✅ | — | 1 |
| PMW3901 optical flow | ✅ | — | 1 |
| VL53L1X ToF | ✅ | — | 1 |
| NUCLEO-H743ZI2 | ✅ (shared) | ✅ (shared) | 1 |
| GM4108H-120T motor | — | ✅ | 1 |
| SimpleFOC Mini | — | ✅ | 1 |
| 12V 5A supply | — | ✅ | 1 |
| 2020 extrusion + brackets | ✅ (shared) | ✅ (shared) | 1 set |
| Carbon tube + weights | — | ✅ | 1 set |
| Breadboard + wires | ✅ (shared) | ✅ (shared) | 1 set |

**Combined total: ~$230-250** for both demos with shared parts.

---

## 5. Wiring

### 5a. Orin 40-Pin Header Connections

| Signal | Orin Pin | Goes To |
|--------|----------|---------|
| SPI0_SCLK | 23 | AS5048A CLK + ICM-42688 CLK |
| SPI0_MOSI | 19 | AS5048A MOSI + ICM-42688 MOSI |
| SPI0_MISO | 21 | AS5048A MISO + ICM-42688 MISO |
| SPI0_CS0 | 24 | ICM-42688 CS |
| SPI0_CS1 | 26 | AS5048A CS |
| UART1_TX (ttyTHS1) | 8 | STM32 USART RX |
| UART1_RX (ttyTHS1) | 10 | STM32 USART TX |
| 3.3V | 1/17 | Sensor power + STM32 logic reference |
| GND | 6/9/14/20/25 | Common ground |

### 5b. STM32 Connections

| Signal | STM32 Nucleo Pin | Goes To |
|--------|-----------------|---------|
| USART3_TX | PD8 (CN10 pin 14) | Orin UART1_RX (pin 10) |
| USART3_RX | PD9 (CN10 pin 12) | Orin UART1_TX (pin 8) |
| SPI1_MOSI | PA7 | SimpleFOC driver (if SPI interface) |
| SPI1_MISO | PA6 | SimpleFOC driver |
| SPI1_SCLK | PA5 | SimpleFOC driver |
| TIM1_CH1 | PE9 | Motor phase A (PWM) |
| TIM1_CH2 | PE11 | Motor phase B (PWM) |
| TIM1_CH3 | PE13 | Motor phase C (PWM) |
| ADC1_IN0 | PA0 | Motor driver current sense A |
| ADC1_IN1 | PA1 | Motor driver current sense B |
| GND | GND | Common ground with Orin + driver |

### 5c. Motor Driver Connections

| Signal | SimpleFOC / DRV8302 | Goes To |
|--------|---------------------|---------|
| PWM_A | IN_A | STM32 TIM1_CH1 |
| PWM_B | IN_B | STM32 TIM1_CH2 |
| PWM_C | IN_C | STM32 TIM1_CH3 |
| ISENSE_A | SO_A | STM32 ADC1_IN0 |
| ISENSE_B | SO_B | STM32 ADC1_IN1 |
| MOTOR_A | Phase A | Motor wire A |
| MOTOR_B | Phase B | Motor wire B |
| MOTOR_C | Phase C | Motor wire C |
| VIN | + | 12V power supply |
| GND | - | Common ground |

### 5d. Wiring Diagram (Simplified)

```
    ┌─────────────┐
    │  12V Supply  │
    └──────┬──────┘
           │
    ┌──────▼──────┐       ┌───────────────┐
    │  SimpleFOC  │──PWM──│   STM32H743   │
    │  Motor      │       │   Nucleo      │
    │  Driver     │       │               │
    └──────┬──────┘       │  USART3 ◄────────── Orin UART1 (pins 8,10)
           │              │               │
    ┌──────▼──────┐       └───────────────┘
    │  GM4108H    │
    │  Motor      │
    │    │        │
    │  ┌─▼──┐    │
    │  │mag │    │       ┌───────────────┐
    │  └─┬──┘    │       │  Orin AGX     │
    │    │       │       │  40-pin       │
    ┌────▼────┐  │       │               │
    │ AS5048A │──SPI0────│  SPI0 CS1     │
    │ encoder │  │       │  (pins 19-26) │
    └─────────┘  │       │               │
                 │       │               │
    ┌─────────┐  │       │               │
    │ICM-42688│──SPI0────│  SPI0 CS0     │
    │  IMU    │  │       │  (pin 24)     │
    └─────────┘  │       │               │
                         └───────────────┘
```

---

## 6. STM32 Firmware — Inner Loop (PID / FOC)

### 6a. Firmware Architecture

```c
// main.c — STM32H743 FOC motor controller
// Built with STM32CubeIDE or PlatformIO + Arduino framework (SimpleFOC library)

#include <SimpleFOC.h>

// Motor + driver
BLDCMotor motor = BLDCMotor(11);  // 11 pole pairs for GM4108H
BLDCDriver3PWM driver = BLDCDriver3PWM(PE9, PE11, PE13);

// UART from Orin
volatile float target_torque = 0.0f;
volatile uint32_t last_cmd_time = 0;

void setup() {
    // Motor driver setup
    driver.voltage_power_supply = 12;
    driver.init();
    motor.linkDriver(&driver);

    // FOC configuration
    motor.torque_controller = TorqueControlType::foc_current;
    motor.controller = MotionControlType::torque;
    motor.init();
    motor.initFOC();

    // UART to Orin (2 Mbaud)
    Serial3.begin(2000000);
}

void loop() {
    // Inner loop: FOC commutation (~10-40 kHz, handled by SimpleFOC)
    motor.loopFOC();

    // Parse Orin commands (non-blocking)
    if (Serial3.available() >= 4) {
        uint8_t buf[4];
        Serial3.readBytes(buf, 4);
        if (buf[0] == 0xAA) {
            int16_t raw = (buf[1] << 8) | buf[2];
            uint8_t chk = (buf[0] + buf[1] + buf[2]) & 0xFF;
            if (chk == buf[3]) {  // checksum valid
                target_torque = raw / 32767.0f;  // normalize to [-1, 1]
                last_cmd_time = millis();
            }
        }
    }

    // Safety: zero torque if no command for 5 ms
    if (millis() - last_cmd_time > 5) {
        target_torque = 0.0f;
    }

    motor.move(target_torque * motor.voltage_limit);
}
```

### 6b. Two Modes for Testing

| Mode | Description | When to Use |
|------|-------------|-------------|
| **Passthrough** | STM32 directly applies torque from Orin. Inner loop = FOC only. | For benchmarking neural policy at full speed. |
| **PID + NN hybrid** | STM32 runs PD on encoder, + adds NN feedforward from Orin. | For comparing PID-only vs PID+NN vs NN-only. |

---

## 7. Orin Software — Outer Loop (Neural Policy)

### 7a. Software Stack

```
NixOS (jetpack-nixos)
├── nix develop → tinygrad flake (examples/tinygrad/flake.nix)
│   ├── tinygrad with NV=1 (Tegra direct ioctls)
│   ├── Python 3.13
│   └── spidev, smbus2, pyserial (pip install in shell)
├── Device tree overlays enabled: SPI0, UART1
└── /dev/spidev0.0, /dev/spidev0.1, /dev/ttyTHS1
```

### 7b. Benchmark Scripts

```bash
# Run neural policy benchmark (motor not connected, dummy STM32 echo)
cd examples/tinygrad && nix develop
NV=1 JITBEAM=2 python3 ../control-loop/gimbal_bench.py --mode nn-only

# Run PID baseline (on Orin CPU, same sensors, same UART)
python3 ../control-loop/gimbal_bench.py --mode pid-only

# Run comparison (alternate NN and PID, same trajectory)
python3 ../control-loop/gimbal_bench.py --mode compare

# Run PyTorch CUDA Graphs baseline
cd examples/control-loop && nix develop
python3 gimbal_bench.py --mode pytorch-nn
```

### 7c. Scripts to Create

| Script | Purpose |
|--------|---------|
| `gimbal_bench.py` | Main benchmark script with NN / PID / PyTorch modes |
| `gimbal_control.py` | Production control loop (runs until Ctrl-C) |
| `gimbal_trajectory.py` | Trajectory generators (sine, step, random) |
| `sensor_drivers.py` | SPI/I2C sensor read functions |
| `uart_protocol.py` | UART communication with STM32 |
| `train_policy.py` | MuJoCo/Isaac sim → train policy → export weights |

---

## 8. Build Plan

### Phase 1: STM32 Bring-Up

| Step | Task | Verification |
|------|------|-------------|
| 1.1 | Flash STM32 with SimpleFOC example | LED blinks, serial output works |
| 1.2 | Connect motor + driver to STM32 | Motor spins in open-loop mode |
| 1.3 | Add AS5048A encoder to STM32 SPI | Read angle, motor tracks angle setpoint |
| 1.4 | Implement FOC torque mode | Motor holds position against hand push |
| 1.5 | Add UART command parser (from Orin) | Send commands from PC, motor responds |

### Phase 2: Sensor Integration on Orin

| Step | Task | Verification |
|------|------|-------------|
| 2.1 | Enable SPI0 on Orin 40-pin (device tree overlay) | `/dev/spidev0.0` exists |
| 2.2 | Wire AS5048A to Orin SPI0 CS1 | Read angle from Python, matches motor position |
| 2.3 | Wire ICM-42688 to Orin SPI0 CS0 | Read gyro/accel, values make sense when moving motor |
| 2.4 | Wire UART to STM32 | Echo test: Orin sends byte, STM32 echoes back |
| 2.5 | Benchmark SPI read speed | Confirm < 20 µs for both sensors |

### Phase 3: PID Baseline on Orin

| Step | Task | Expected Result |
|------|------|----------------|
| 3.1 | Implement PID controller on Orin (CPU, Python) | Motor tracks setpoint at ~2-5 kHz |
| 3.2 | Tune PID gains for 100g payload | Stable, reasonable tracking |
| 3.3 | Run benchmark suite (step response, sine tracking, disturbance) | Record PID baseline numbers |

### Phase 4: Neural Policy on Orin

| Step | Task | Expected Result |
|------|------|----------------|
| 4.1 | Create motor simulation in MuJoCo (same dimensions/inertia) | Sim matches real motor dynamics |
| 4.2 | Train RL policy in sim (PPO, ~100K params) | Policy stabilizes pendulum in sim |
| 4.3 | Export policy weights to numpy | Load into tinygrad model |
| 4.4 | Run neural policy on real motor (tinygrad NV=1, 4.8 kHz) | Motor tracks setpoint |
| 4.5 | Run comparison benchmarks (NN vs PID, same tests) | Generate comparison plots |

### Phase 5: PyTorch Comparison

| Step | Task | Expected Result |
|------|------|----------------|
| 5.1 | Implement same policy in PyTorch, CUDA Graphs | Motor tracks setpoint at ~2.5 kHz |
| 5.2 | Run same benchmark suite | Compare NN-tinygrad vs NN-pytorch vs PID |

### Phase 6: Advanced Tests

| Step | Task | Expected Result |
|------|------|----------------|
| 6.1 | Payload change test (50g, 100g, 200g) | NN handles all, PID degrades |
| 6.2 | Disturbance rejection (finger flick) | NN recovers faster |
| 6.3 | High-frequency trajectory tracking (up to 20 Hz sine) | NN has lower RMSE above 5 Hz |
| 6.4 | Model size sweep (64→128→256→512 hidden) | Plot performance vs model size |

---

## 9. Test & Benchmark Plan

### 9a. Metrics

| Metric | What | Sensor |
|--------|------|--------|
| Tracking RMSE | RMS angular error during trajectory following | AS5048A encoder |
| Settling time | Time from step command to within 2% of target | AS5048A encoder |
| Overshoot | Peak error after step command | AS5048A encoder |
| Disturbance recovery time | Time to recover after flick | AS5048A + ICM-42688 |
| Control loop rate | Achieved frequency (Hz) | perf_counter_ns timing |
| Cycle jitter | Std deviation of cycle time | perf_counter_ns timing |
| End-to-end latency | IMU event → motor torque change | Oscilloscope (optional) |

### 9b. Test Matrix

| Test | Controller | Rate | Payload | Trajectory | Expected Winner |
|------|-----------|-----:|--------:|-----------|:-:|
| 1 | PID (CPU) | 4.8 kHz | 100g | Step 0°→45° | Baseline |
| 2 | NN (tinygrad NV=1) | 4.8 kHz | 100g | Step 0°→45° | **NN** (less overshoot) |
| 3 | NN (PyTorch CG) | 2.6 kHz | 100g | Step 0°→45° | NN (slower rate) |
| 4 | PID (CPU) | 4.8 kHz | 100g | Sine 5 Hz | Baseline |
| 5 | NN (tinygrad NV=1) | 4.8 kHz | 100g | Sine 5 Hz | **NN** (feedforward) |
| 6 | PID (CPU) | 4.8 kHz | 200g | Step 0°→45° | PID (tuned for 100g) will oscillate |
| 7 | NN (tinygrad NV=1) | 4.8 kHz | 200g | Step 0°→45° | **NN** (generalizes) |
| 8 | PID (CPU) | 4.8 kHz | 100g | Disturbance (flick) | Baseline |
| 9 | NN (tinygrad NV=1) | 4.8 kHz | 100g | Disturbance (flick) | **NN** (preemptive) |
| 10 | NN (tinygrad NV=1) | 4.8 kHz | 100g | Slow rotation 1 rpm | **NN** (anti-cogging) |

### 9c. Model Size Sweep

Run test #5 (sine 5 Hz tracking) with different model sizes:

| Model | Params | Expected Rate | Expected RMSE |
|-------|-------:|:---:|---|
| 13→64→64→1 | 5.3K | ~5 kHz | Slightly worse (underfitting) |
| 13→128→128→1 | 18.7K | ~4.8 kHz | Good |
| 13→256→256→1 | 70K | ~4 kHz | Slightly better |
| 13→512→512→1 | 272K | ~3 kHz | Best tracking, lower rate |
| 13→1024→1024→1 | 1M | ~1.5 kHz | Diminishing returns in tracking |

The interesting question: **does higher rate with smaller model beat lower rate with bigger model?** This determines the optimal operating point.

---

## 10. Training the Neural Policy

### 10a. Simulation Setup

```python
# MuJoCo model: motor + pendulum
# Match real hardware:
#   - Motor: GM4108H-120T specs (torque constant, cogging model)
#   - Arm: carbon tube 150mm, 8g
#   - Weight: 50-200g at tip (domain randomized)
#   - Encoder: 14-bit resolution, 1 LSB noise
#   - Gyro: typical ICM-42688 noise profile

# Reward function:
#   r = -|angle_error|² - 0.1×|angular_vel_error|² - 0.01×|torque|²
#   + bonus for low tracking error over sine trajectory
```

### 10b. Training Stack

**Options (all run on Orin AGX)**:

1. **Isaac Gym / Isaac Lab** — NVIDIA's GPU-parallelized sim. 4096+ environments simultaneously. Best for fast training but heavy CUDA dependency.

2. **MuJoCo + Stable-Baselines3** — CPU sim, PyTorch RL. Simpler setup, slower training. Good for a first pass.

3. **Tinygrad + custom sim** — Write the pendulum dynamics in tinygrad. Train the policy entirely in tinygrad. Purest approach but requires implementing PPO from scratch.

Recommended: **MuJoCo + Stable-Baselines3 for training**, then export weights to tinygrad for deployment. This separates the training framework (PyTorch, mature RL libraries) from the deployment framework (tinygrad NV=1, fast inference).

### 10c. Domain Randomization (for sim-to-real transfer)

Randomize these parameters during training so the policy generalizes to real hardware:

| Parameter | Range | Why |
|-----------|-------|-----|
| Payload mass | 50-250g | Different payloads on real arm |
| Arm length | 130-170mm | Manufacturing tolerance |
| Motor torque constant | ±20% | Motor-to-motor variation |
| Encoder noise | ±2 LSB | Real sensor noise |
| Gyro noise/bias | Typical ICM-42688 spec | Real sensor |
| Cogging torque amplitude | 0-5% of rated torque | Varies with motor |
| Friction (static + viscous) | Typical BLDC friction range | Bearing variation |
| Control delay | 0-500 µs | Simulates variable loop timing |

### 10d. Export Pipeline

```bash
# 1. Train in MuJoCo
python3 train_policy.py --env pendulum_motor --algo PPO --steps 10M

# 2. Export to numpy weights
python3 export_weights.py --checkpoint best_model.zip --output policy_weights.npz

# 3. Load in tinygrad
python3 gimbal_control.py --weights policy_weights.npz
```

---

## 11. Path to Drone Gimbal / Robot Arm

### 11a. From Demo to 2-Axis Gimbal

The GM4108H motor IS a gimbal motor. To go from this demo to a camera gimbal:

| Change | What | Effort |
|--------|------|--------|
| Add second motor | Mount second GM4108H orthogonal to first | Mechanical + 1 more encoder + expand model |
| Expand policy | 13→26 inputs (both joints), 1→2 outputs | Retrain, same tinygrad/STM32 code structure |
| Add camera | Mount small camera (e.g., Arducam for Orin CSI) | Separate from control loop (CSI is DMA) |
| Mount on drone frame | Attach gimbal to vibration-dampened plate | Standard gimbal mount |

The control code changes minimally — expand `STATE_DIM` and `ACTION_DIM`, retrain the policy.

### 11b. From Demo to Robot Arm

| Joint Count | Params (est.) | Expected Rate | NV=1 Advantage |
|:-:|:-:|:-:|:-:|
| 1 joint (this demo) | 18K | 4.8 kHz | 2.6x |
| 3 joints (wrist) | 26K | 4.5 kHz | 2.5x |
| 6 joints (full arm) | 38K | 4.2 kHz | 2.4x |
| 12 joints (dual arm) | 55K | 3.8 kHz | 2.2x |
| 23 joints (humanoid upper) | 85K | 3.3 kHz | 2.0x |

For all practical robot arm joint counts (up to 12), we maintain > 2x advantage over PyTorch CUDA Graphs and > 3 kHz control rate. The model grows linearly with joint count (more inputs, same hidden width), so compute barely changes.

### 11c. What You'll Learn Building This

| Skill | Applies to |
|-------|-----------|
| SPI sensor integration on Orin 40-pin | Any Jetson-based robot |
| Real-time UART communication Orin ↔ STM32 | Any companion-computer ↔ MCU architecture |
| FOC motor control on STM32 | Any BLDC actuator (drone ESC, robot joint, gimbal) |
| Neural policy training in sim → deploy on real hardware | Core skill for learned robot control |
| tinygrad NV=1 direct memory inference | Unique Tegra advantage — publishable research |
| Two-layer control architecture (MCU + GPU) | Industry standard (PX4 + companion computer) |
