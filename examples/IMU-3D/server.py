#!/usr/bin/env python3
"""IMU-3D: Real-time MPU-9250 → Three.js 3D visualization server.

Reads accelerometer, gyroscope, and magnetometer data from an MPU-9250
over I2C, runs a Madgwick AHRS filter to estimate orientation, and
streams everything to a Three.js web frontend over WebSocket.

Usage:
    nix develop
    python server.py                    # live IMU on I2C bus 7
    python server.py --mock             # simulated IMU data
    python server.py --bus 1 --port 9000

Then open http://<orin-ip>:8080 in your browser.
"""

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

from aiohttp import web

try:
    import smbus2
except ImportError:
    smbus2 = None

STATIC = Path(__file__).parent / "static"

# ── MPU-9250 Register Map ─────────────────────────────────────────────

WHO_AM_I      = 0x75
PWR_MGMT_1    = 0x6B
SMPLRT_DIV    = 0x19
CONFIG_REG    = 0x1A
GYRO_CONFIG   = 0x1B
ACCEL_CONFIG  = 0x1C
INT_PIN_CFG   = 0x37
ACCEL_XOUT_H  = 0x3B

# AK8963 magnetometer (accessed via I2C bypass)
AK_ADDR       = 0x0C
AK_WIA        = 0x00
AK_ST1        = 0x02
AK_HXL        = 0x03
AK_CNTL1      = 0x0A
AK_ASAX       = 0x10


# ── Madgwick AHRS Filter (6-DOF) ──────────────────────────────────────

class MadgwickAHRS:
    """Madgwick's gradient-descent IMU filter (accelerometer + gyroscope).

    Outputs a unit quaternion [w, x, y, z] representing the orientation
    of the sensor relative to the Earth frame (Z-up).
    """

    def __init__(self, sample_freq: float = 50.0, beta: float = 0.1):
        self.beta = beta
        self.inv_freq = 1.0 / sample_freq
        self.q = [1.0, 0.0, 0.0, 0.0]

    def reset(self):
        self.q = [1.0, 0.0, 0.0, 0.0]

    def update(self, gx: float, gy: float, gz: float,
               ax: float, ay: float, az: float) -> list:
        """Feed gyro (rad/s) and accel (any consistent unit)."""
        q0, q1, q2, q3 = self.q

        # Normalise accelerometer
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm < 1e-10:
            # Can't correct without gravity reference — integrate gyro only
            qDot0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
            qDot1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
            qDot2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
            qDot3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)
            q0 += qDot0 * self.inv_freq
            q1 += qDot1 * self.inv_freq
            q2 += qDot2 * self.inv_freq
            q3 += qDot3 * self.inv_freq
            rn = 1.0 / math.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
            self.q = [q0*rn, q1*rn, q2*rn, q3*rn]
            return self.q

        rn = 1.0 / norm
        ax *= rn; ay *= rn; az *= rn

        # Auxiliary variables
        _2q0 = 2.0*q0; _2q1 = 2.0*q1; _2q2 = 2.0*q2; _2q3 = 2.0*q3
        _4q0 = 4.0*q0; _4q1 = 4.0*q1; _4q2 = 4.0*q2
        _8q1 = 8.0*q1; _8q2 = 8.0*q2
        q0q0 = q0*q0; q1q1 = q1*q1; q2q2 = q2*q2; q3q3 = q3*q3

        # Gradient descent corrective step
        s0 = _4q0*q2q2 + _2q2*ax + _4q0*q1q1 - _2q1*ay
        s1 = (_4q1*q3q3 - _2q3*ax + 4.0*q0q0*q1 - _2q0*ay
              - _4q1 + _8q1*q1q1 + _8q1*q2q2 + _4q1*az)
        s2 = (4.0*q0q0*q2 + _2q0*ax + _4q2*q3q3 - _2q3*ay
              - _4q2 + _8q2*q1q1 + _8q2*q2q2 + _4q2*az)
        s3 = 4.0*q1q1*q3 - _2q1*ax + 4.0*q2q2*q3 - _2q2*ay

        rn = 1.0 / math.sqrt(s0*s0 + s1*s1 + s2*s2 + s3*s3 + 1e-12)
        s0 *= rn; s1 *= rn; s2 *= rn; s3 *= rn

        # Rate of change = gyro term − β × gradient
        qDot0 = 0.5*(-q1*gx - q2*gy - q3*gz) - self.beta*s0
        qDot1 = 0.5*(q0*gx + q2*gz - q3*gy) - self.beta*s1
        qDot2 = 0.5*(q0*gy - q1*gz + q3*gx) - self.beta*s2
        qDot3 = 0.5*(q0*gz + q1*gy - q2*gx) - self.beta*s3

        # Integrate
        q0 += qDot0 * self.inv_freq
        q1 += qDot1 * self.inv_freq
        q2 += qDot2 * self.inv_freq
        q3 += qDot3 * self.inv_freq

        # Normalise quaternion
        rn = 1.0 / math.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
        self.q = [q0*rn, q1*rn, q2*rn, q3*rn]
        return self.q


# ── MPU-9250 Hardware Driver ───────────────────────────────────────────

class MPU9250:
    """Read accel/gyro/mag/temp from MPU-9250 over Linux I2C (smbus2)."""

    ACCEL_SCALE = 16384.0   # ±2 g
    GYRO_SCALE  = 131.0     # ±250 °/s
    MAG_SCALE   = 0.15      # µT per LSB (16-bit mode)

    def __init__(self, bus: int = 7, addr: int = 0x68):
        if smbus2 is None:
            raise RuntimeError("smbus2 not installed — use 'nix develop' or '--mock'")
        self.bus = smbus2.SMBus(bus)
        self.addr = addr
        self.mag_cal = [1.0, 1.0, 1.0]
        self._init_mpu()
        self._init_mag()

    # ── Initialisation ────────────────────────────────────

    def _init_mpu(self):
        # Wake up (clear SLEEP bit)
        self.bus.write_byte_data(self.addr, PWR_MGMT_1, 0x00)
        time.sleep(0.1)

        # Verify WHO_AM_I
        who = self.bus.read_byte_data(self.addr, WHO_AM_I)
        if who not in (0x71, 0x73):
            raise RuntimeError(
                f"WHO_AM_I returned 0x{who:02x} — expected 0x71 (MPU-9250). "
                "Check wiring or I2C bus number."
            )
        print(f"✓ MPU-9250 detected (WHO_AM_I=0x{who:02x})")

        # Sample rate: 1 kHz / (1 + 19) = 50 Hz
        self.bus.write_byte_data(self.addr, SMPLRT_DIV, 19)
        # DLPF bandwidth 42 Hz
        self.bus.write_byte_data(self.addr, CONFIG_REG, 0x03)
        # Gyro ±250 °/s
        self.bus.write_byte_data(self.addr, GYRO_CONFIG, 0x00)
        # Accel ±2 g
        self.bus.write_byte_data(self.addr, ACCEL_CONFIG, 0x00)
        # Enable I2C bypass to talk directly to AK8963
        self.bus.write_byte_data(self.addr, INT_PIN_CFG, 0x02)
        time.sleep(0.01)

    def _init_mag(self):
        try:
            # Fuse ROM access mode → read sensitivity adjustment
            self.bus.write_byte_data(AK_ADDR, AK_CNTL1, 0x0F)
            time.sleep(0.01)
            asa = self.bus.read_i2c_block_data(AK_ADDR, AK_ASAX, 3)
            self.mag_cal = [(a - 128) * 0.5 / 128.0 + 1.0 for a in asa]

            # Power-down → continuous measurement mode 2 (100 Hz, 16-bit)
            self.bus.write_byte_data(AK_ADDR, AK_CNTL1, 0x00)
            time.sleep(0.01)
            self.bus.write_byte_data(AK_ADDR, AK_CNTL1, 0x16)
            time.sleep(0.01)

            wia = self.bus.read_byte_data(AK_ADDR, AK_WIA)
            print(f"✓ AK8963 magnetometer detected (WIA=0x{wia:02x})")
        except Exception as e:
            print(f"⚠ Magnetometer init failed ({e}) — mag data will be zero")
            self.mag_cal = [0.0, 0.0, 0.0]

    # ── Reading ───────────────────────────────────────────

    @staticmethod
    def _signed16(msb, lsb):
        v = (msb << 8) | lsb
        return v - 65536 if v > 32767 else v

    def read_all(self):
        """Return (accel_g, gyro_dps, mag_ut, temp_c)."""
        # Burst read: accel(6) + temp(2) + gyro(6) = 14 bytes from 0x3B
        d = self.bus.read_i2c_block_data(self.addr, ACCEL_XOUT_H, 14)

        ax = self._signed16(d[0],  d[1])  / self.ACCEL_SCALE
        ay = self._signed16(d[2],  d[3])  / self.ACCEL_SCALE
        az = self._signed16(d[4],  d[5])  / self.ACCEL_SCALE
        temp = self._signed16(d[6], d[7]) / 333.87 + 21.0
        gx = self._signed16(d[8],  d[9])  / self.GYRO_SCALE
        gy = self._signed16(d[10], d[11]) / self.GYRO_SCALE
        gz = self._signed16(d[12], d[13]) / self.GYRO_SCALE

        # Magnetometer (AK8963 — little-endian!)
        mx = my = mz = 0.0
        try:
            st1 = self.bus.read_byte_data(AK_ADDR, AK_ST1)
            if st1 & 0x01:
                md = self.bus.read_i2c_block_data(AK_ADDR, AK_HXL, 7)
                mx = self._signed16(md[1], md[0]) * self.mag_cal[0] * self.MAG_SCALE
                my = self._signed16(md[3], md[2]) * self.mag_cal[1] * self.MAG_SCALE
                mz = self._signed16(md[5], md[4]) * self.mag_cal[2] * self.MAG_SCALE
                # Reading ST2 (md[6]) signals ready for next measurement
        except Exception:
            pass

        return [ax, ay, az], [gx, gy, gz], [mx, my, mz], temp

    def close(self):
        self.bus.close()


# ── Mock IMU (simulated data for testing) ──────────────────────────────

class MockIMU:
    """Generates physically-consistent fake IMU data for UI testing."""

    def read_all(self):
        t = time.time()

        # Smooth sinusoidal orientation
        roll  = math.sin(t * 0.4) * 0.5
        pitch = math.sin(t * 0.3) * 0.4
        yaw   = t * 0.2

        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)

        # Gravity rotated into body frame (g)
        accel = [
            -sp * 9.81 / 9.81,
            sr * cp * 9.81 / 9.81,
            cr * cp,
        ]
        # Approximate angular rates (°/s)
        gyro = [
            math.cos(t * 0.4) * 0.4 * 57.3 * 0.5,
            math.cos(t * 0.3) * 0.3 * 57.3 * 0.4,
            0.2 * 57.3,
        ]
        # Earth field rotated to body frame (µT)
        cy, sy = math.cos(yaw), math.sin(yaw)
        mag_n, mag_d = 25.0, 40.0
        mag = [
            mag_n * cp * cy + mag_d * sp,
            mag_n * (sr * sp * cy - cr * sy) - mag_d * sr * cp,
            mag_n * (cr * sp * cy + sr * sy) - mag_d * cr * cp,
        ]

        temp = 34.0 + math.sin(t * 0.05) * 2.0
        return accel, gyro, mag, temp

    def close(self):
        pass


# ── Web Server ─────────────────────────────────────────────────────────

async def index_handler(request):
    return web.FileResponse(STATIC / "index.html")


async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    request.app["clients"].add(ws)
    print(f"↳ client connected ({len(request.app['clients'])} total)")
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("cmd") == "calibrate":
                    request.app["ahrs"].reset()
                    print("↻ AHRS calibrated (quaternion reset)")
    except Exception:
        pass
    finally:
        request.app["clients"].discard(ws)
        print(f"↳ client disconnected ({len(request.app['clients'])} remaining)")
    return ws


async def imu_loop(app):
    """Background task: read IMU → update AHRS → broadcast over WebSocket."""
    imu  = app["imu"]
    ahrs = app["ahrs"]
    rate = app["rate"]
    loop = asyncio.get_event_loop()
    interval = 1.0 / rate

    while True:
        try:
            accel, gyro, mag, temp = await loop.run_in_executor(None, imu.read_all)

            # Madgwick expects gyro in rad/s
            gx_rad = math.radians(gyro[0])
            gy_rad = math.radians(gyro[1])
            gz_rad = math.radians(gyro[2])
            q = ahrs.update(gx_rad, gy_rad, gz_rad, accel[0], accel[1], accel[2])

            packet = json.dumps({
                "t":     round(time.time(), 4),
                "accel": [round(v, 4) for v in accel],
                "gyro":  [round(v, 2) for v in gyro],
                "mag":   [round(v, 2) for v in mag],
                "temp":  round(temp, 1),
                "q":     [round(v, 6) for v in q],
            })

            dead = []
            for ws in list(app["clients"]):
                try:
                    await ws.send_str(packet)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                app["clients"].discard(ws)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"⚠ IMU read error: {e}")

        await asyncio.sleep(interval)


async def on_startup(app):
    app["imu_task"] = asyncio.create_task(imu_loop(app))


async def on_cleanup(app):
    app["imu_task"].cancel()
    try:
        await app["imu_task"]
    except asyncio.CancelledError:
        pass
    app["imu"].close()


# ── Entry Point ────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="IMU-3D visualisation server")
    p.add_argument("--mock",  action="store_true",       help="use simulated IMU data")
    p.add_argument("--bus",   type=int,   default=7,     help="I2C bus number (default: 7)")
    p.add_argument("--addr",  type=str,   default="0x68",help="MPU-9250 I2C address")
    p.add_argument("--port",  type=int,   default=9090,  help="HTTP/WebSocket port")
    p.add_argument("--rate",  type=int,   default=50,    help="sample rate in Hz")
    args = p.parse_args()

    if args.mock or smbus2 is None:
        if not args.mock:
            print("⚠ smbus2 not available — falling back to mock mode")
        imu = MockIMU()
        print("● Mock IMU active (simulated data)")
    else:
        addr = int(args.addr, 16)
        imu = MPU9250(bus=args.bus, addr=addr)
        print(f"● MPU-9250 on I2C bus {args.bus} @ {args.addr}")

    ahrs = MadgwickAHRS(sample_freq=float(args.rate), beta=0.1)

    app = web.Application()
    app["imu"]     = imu
    app["ahrs"]    = ahrs
    app["clients"] = set()
    app["rate"]    = args.rate

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/",   index_handler)

    host = "0.0.0.0"
    port = args.port
    print(f"")
    print(f"  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  IMU-3D Server                                          │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  │  Local    → http://localhost:{port:<24}│")
    print(f"  │  Network  → http://192.168.8.162:{port:<20}│")
    print(f"  │  Rate     → {args.rate} Hz{' ' * 38}│")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  │  From your laptop (PowerShell / Terminal):              │")
    print(f"  │                                                         │")
    print(f"  │    ssh -L {port}:localhost:{port} Orin-AGX-NixOS{' ' * (15 - len(str(port)))}│")
    print(f"  │                                                         │")
    print(f"  │  Then open → http://localhost:{port:<24}│")
    print(f"  └─────────────────────────────────────────────────────────┘")
    print(f"")

    web.run_app(app, host=host, port=port, print=None)


if __name__ == "__main__":
    main()
