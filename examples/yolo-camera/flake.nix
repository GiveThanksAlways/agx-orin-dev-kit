{
  description = "YOLOv8 live camera demo & benchmark — TinyGrad vs PyTorch vs TensorRT on Jetson AGX Orin";

  inputs = {
    # Reuse control-loop's nixpkgs + jetpack-nixos so CUDA torch is cached.
    control-loop.url = "path:../control-loop";
    nixpkgs.follows = "control-loop/nixpkgs";
    jetpack-nixos.follows = "control-loop/jetpack-nixos";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, jetpack-nixos, flake-utils, control-loop }:
    flake-utils.lib.eachSystem [ "aarch64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
            cudaCapabilities = [ "8.7" ];
            cudaForwardCompat = false;
          };
          overlays = [
            jetpack-nixos.overlays.default
            (final: _: { inherit (final.nvidia-jetpack6) cudaPackages; })
          ];
        };
        jetpack = pkgs.nvidia-jetpack6;
        cuda = jetpack.cudaPackages;

        cuda-root = pkgs.runCommand "tinygrad-cuda-root" {} ''
          mkdir -p $out
          ln -s ${pkgs.lib.getDev cuda.cuda_cudart}/include $out/include
          ln -s ${jetpack.l4t-cuda}/lib/libcuda.so $out/libcuda.so
          ln -s ${jetpack.l4t-cuda}/lib/libcuda.so.1 $out/libcuda.so.1
          ln -s ${jetpack.l4t-cuda}/lib/libcuda.so.1.1 $out/libcuda.so.1.1
        '';

        # OpenCV without CUDA — hits the binary cache (no 30min source build).
        # Only TinyGrad uses the GPU (NV=1); OpenCV just does camera capture + drawing.
        opencv-cpu = pkgs.python312Packages.opencv4.override { enableCuda = false; };

        pythonEnv = pkgs.python312.withPackages (ps: [
          ps.torch        # CUDA-enabled torch (same as control-loop, already built)
          ps.numpy
          ps.pillow
          opencv-cpu
          ps.requests
          ps.tqdm
          ps.onnx         # for ONNX export (PyTorch → TensorRT)
          ps.protobuf
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          name = "yolo-camera";

          buildInputs = [
            pythonEnv
            pkgs.git
            pkgs.clang
            pkgs.v4l-utils   # v4l2-ctl for camera diagnostics
            pkgs.libv4l       # libv4l2.so — required by OpenCV V4L2 backend
            cuda.tensorrt     # TensorRT (trtexec + libs)
          ];

          CC = "${pkgs.clang}/bin/clang";
          CXX = "${pkgs.clang}/bin/clang++";

          CUDA_PATH = "${cuda-root}";
          NVRTC_PATH = "${pkgs.lib.getLib cuda.cuda_nvrtc}/lib/libnvrtc.so";
          NVJITLINK_PATH = "${pkgs.lib.getLib cuda.libnvjitlink}/lib/libnvJitLink.so";
          LIBC_PATH = "${pkgs.glibc}/lib/libc.so.6";

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            (pkgs.lib.getLib cuda.cuda_cudart)
            (pkgs.lib.getLib cuda.libcublas)
            (pkgs.lib.getLib cuda.libcusparse)
            (pkgs.lib.getLib cuda.libcusolver)
            (pkgs.lib.getLib cuda.libcufft)
            (pkgs.lib.getLib cuda.libcurand)
            (pkgs.lib.getLib cuda.cuda_nvrtc)
            (pkgs.lib.getLib cuda.libnvjitlink)
            (pkgs.lib.getLib cuda.cudnn)
            cuda.tensorrt
            jetpack.l4t-cuda
            jetpack.l4t-core
            pkgs.stdenv.cc.cc
            pkgs.libv4l
          ];

          shellHook = ''
            export CC="${pkgs.clang}/bin/clang"
            export CXX="${pkgs.clang}/bin/clang++"

            # Find tinygrad source
            if [ -d "$PWD/../../external/tinygrad/tinygrad" ]; then
              TINYGRAD_PATH="$(realpath "$PWD/../../external/tinygrad")"
            elif [ -d "$PWD/external/tinygrad/tinygrad" ]; then
              TINYGRAD_PATH="$PWD/external/tinygrad"
            else
              echo "ERROR: cannot find tinygrad at external/tinygrad/"
              echo "Run: git submodule update --init --recursive"
              return 1
            fi
            export PYTHONPATH="$TINYGRAD_PATH:$PWD:$PYTHONPATH"

            # Jetson-extracted TRT (SM87) — prefer over nix SBSA TRT
            JETSON_TRT="$PWD/../presentation/jetson-trt/extracted/usr"
            if [ -d "$JETSON_TRT" ]; then
              export JETSON_TRTEXEC="$JETSON_TRT/src/tensorrt/bin/trtexec"
              export LD_LIBRARY_PATH="$JETSON_TRT/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH"
            fi

            echo ""
            echo "=== YOLOv8 Benchmark & Demo (Orin AGX / CUDA 12.6) ==="
            echo "Using tinygrad from: $TINYGRAD_PATH"
            echo ""
            echo "── Benchmarks ──────────────────────────────────────"
            echo "  python3 bench_yolov8_pytorch.py   # PyTorch CUDA Graphs"
            echo "  python3 bench_yolov8_trt.py        # TensorRT FP16"
            echo "  NV=1 python3 demo_yolov8_hot_path.py --bench  # TinyGrad C Hot Path"
            echo ""
            echo "── Camera Demos ────────────────────────────────────"
            echo "  NV=1 python3 demo_yolov8_hot_path.py --stream   # TinyGrad C Hot Path"
            echo "  NV=1 python3 demo_yolov8_camera.py --stream     # TinyGrad @TinyJit"
            echo "  python3 demo_yolov8_pytorch.py --stream          # PyTorch CUDA Graphs"
            echo "  python3 demo_yolov8_trt.py --stream              # TensorRT FP16"
            echo ""
          '';
        };
      }
    );
}
