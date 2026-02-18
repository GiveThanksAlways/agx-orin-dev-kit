{
  description = "Binocular Camera (Dual IMX219) stereo vision + ML on Jetson Orin AGX";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    jetpack-nixos.url = "github:anduril/jetpack-nixos";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, jetpack-nixos, flake-utils }:
    flake-utils.lib.eachSystem [ "aarch64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [ jetpack-nixos.overlays.default ];
        };
        jetpack = pkgs.nvidia-jetpack6;
        cuda = jetpack.cudaPackages;

        pythonPackages = pkgs.python3Packages;
      in
      {
        devShells.default = pkgs.mkShell {
          name = "binocular-camera-shell";

          buildInputs = [
            pkgs.python3
            pythonPackages.pip
            pythonPackages.virtualenv
            pythonPackages.numpy

            # Camera / image processing
            pkgs.v4l-utils       # v4l2-ctl for camera control
            pkgs.ffmpeg          # Video capture / encoding
            pkgs.gst_all_1.gstreamer
            pkgs.gst_all_1.gst-plugins-base
            pkgs.gst_all_1.gst-plugins-good

            # Build tools
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.gcc

            # CUDA (for GPU-accelerated image processing)
            (pkgs.lib.getLib cuda.cuda_cudart)
            (pkgs.lib.getLib cuda.libcublas)
            (pkgs.lib.getLib cuda.cuda_nvrtc)
          ];

          CUDA_HOME = "${pkgs.lib.getDev cuda.cuda_cudart}";

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            (pkgs.lib.getLib cuda.cuda_cudart)
            (pkgs.lib.getLib cuda.libcublas)
            (pkgs.lib.getLib cuda.cuda_nvrtc)
            jetpack.l4t-cuda
            jetpack.l4t-core
            pkgs.stdenv.cc.cc
          ];

          shellHook = ''
            echo ""
            echo "=== Binocular Camera (Dual IMX219) Dev Shell ==="
            echo "    Jetson Orin AGX / JetPack 6 / CUDA 12.6"
            echo ""

            # Create venv for pip packages (OpenCV, tinygrad, etc.)
            VENV_DIR="$PWD/.venv"
            if [ ! -d "$VENV_DIR" ]; then
              echo "Creating Python venv..."
              python3 -m venv "$VENV_DIR" --system-site-packages
              source "$VENV_DIR/bin/activate"
              pip install --upgrade pip
              pip install opencv-python-headless tinygrad Pillow
            else
              source "$VENV_DIR/bin/activate"
            fi

            echo ""
            echo "Camera Setup:"
            echo "  v4l2-ctl --list-devices        # List cameras"
            echo "  v4l2-ctl -d /dev/video0 --all  # Left camera info"
            echo "  v4l2-ctl -d /dev/video1 --all  # Right camera info"
            echo ""
            echo "Projects:"
            echo "  python3 scripts/capture_stereo.py           # Capture stereo pairs"
            echo "  python3 scripts/calibrate_stereo.py         # Stereo calibration"
            echo "  python3 scripts/depth_map.py                # Compute depth maps"
            echo "  python3 scripts/stereo_object_detect.py     # 3D object detection"
            echo "  python3 scripts/obstacle_avoidance.py       # Obstacle avoidance demo"
            echo "  python3 scripts/hand_tracking_3d.py         # 3D hand tracking"
            echo ""
            echo "Benchmarks:"
            echo "  python3 scripts/bench_stereo_pipeline.py    # Pipeline throughput"
            echo ""
          '';
        };
      }
    );
}
