# sensors.nix — IMU + USB camera support for Jetson AGX Orin
#
# Enables:
#   - MPU-9250 IMU over I2C-7 (inv_mpu6050 driver, addr 0x68)
#   - USB UVC stereo camera (uvcvideo driver)
#   - /dev/iio:device* for IMU data (accel, gyro, magnetometer)
#   - /dev/video* for camera streams
#
# Usage:
#   imports = [ ./modules/sensors.nix ];
#   services.orin-sensors = {
#     enable = true;
#     imuI2cBus = 7;
#     imuI2cAddr = "0x68";
#     cameraUsers = [ "agent" "spencer" ];
#   };

{ config, lib, pkgs, ... }:

let
  cfg = config.services.orin-sensors;
in
{
  options.services.orin-sensors = {
    enable = lib.mkEnableOption "Orin AGX IMU + camera sensor support";

    imuI2cBus = lib.mkOption {
      type = lib.types.int;
      default = 7;
      description = "I2C bus number the MPU-9250 is wired to.";
    };

    imuI2cAddr = lib.mkOption {
      type = lib.types.str;
      default = "0x68";
      description = "I2C address of the MPU-9250 (AD0=GND → 0x68, AD0=VCC → 0x69).";
    };

    cameraUsers = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ "agent" ];
      description = "Users to add to the 'video' and 'i2c' groups for sensor access.";
    };

    enableImu = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Enable MPU-9250 IMU (inv_mpu6050 over I2C).";
    };

    enableCamera = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Enable USB UVC camera support.";
    };
  };

  config = lib.mkIf cfg.enable {

    # --- Kernel modules ---
    boot.kernelModules =
      (lib.optionals cfg.enableImu [
        "i2c-dev"         # userspace I2C access (/dev/i2c-*)
        "inv-mpu6050-i2c" # InvenSense MPU-6050/9250 I2C driver
        "industrialio"    # IIO subsystem core
      ])
      ++ (lib.optionals cfg.enableCamera [
        "uvcvideo"        # USB Video Class driver
        "videobuf2-vmalloc"
      ]);

    # --- udev rules: permissions for video + IIO + I2C devices ---
    services.udev.extraRules = ''
      # UVC cameras — grant video group access
      SUBSYSTEM=="video4linux", GROUP="video", MODE="0660"

      # IIO devices (IMU accel/gyro/mag) — grant video group access
      SUBSYSTEM=="iio", GROUP="video", MODE="0660"
      KERNEL=="iio:device*", GROUP="video", MODE="0660"

      # I2C bus — grant i2c group access for diagnostics (i2cdetect etc.)
      SUBSYSTEM=="i2c-dev", GROUP="i2c", MODE="0660"
    '';

    # --- Create the i2c group ---
    users.groups.i2c = {};

    # --- Add camera/i2c users to the required groups ---
    users.users = builtins.listToAttrs (map (u: {
      name = u;
      value = {
        extraGroups = [ "video" "i2c" ];
      };
    }) cfg.cameraUsers);

    # --- Bind MPU-9250 to inv_mpu6050 at boot ---
    systemd.services.mpu9250-bind = lib.mkIf cfg.enableImu {
      description = "Bind MPU-9250 IMU to inv_mpu6050 driver on I2C bus ${toString cfg.imuI2cBus}";
      wantedBy = [ "multi-user.target" ];
      after = [ "systemd-modules-load.service" ];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStart = pkgs.writeShellScript "bind-mpu9250" ''
          set -euo pipefail
          BUS=${toString cfg.imuI2cBus}
          ADDR=${cfg.imuI2cAddr}

          SYSFS="/sys/bus/i2c/devices/i2c-''${BUS}"

          # If the device is already bound, skip
          if [ -d "/sys/bus/i2c/devices/''${BUS}-00''${ADDR#0x}" ]; then
            echo "MPU-9250 already bound at ''${BUS}-00''${ADDR#0x}"
            exit 0
          fi

          # Instantiate the device on the I2C bus
          echo "inv_mpu6050 ''${ADDR}" > "''${SYSFS}/new_device" || {
            echo "Warning: could not bind MPU-9250 — check wiring"
            exit 0
          }
          echo "MPU-9250 bound on I2C bus ''${BUS} at ''${ADDR}"
        '';
      };
    };

    # --- Convenience packages ---
    environment.systemPackages = with pkgs; [
      i2c-tools       # i2cdetect, i2cget, i2cdump
      v4l-utils       # v4l2-ctl, media-ctl for camera introspection
    ];
  };
}
