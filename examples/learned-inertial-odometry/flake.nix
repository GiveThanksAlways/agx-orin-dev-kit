{
  description = "Benchmark: Cioffi et al. Learned Inertial Odometry TCN on Jetson Orin AGX";

  inputs = {
    # Follow control-loop's nixpkgs + jetpack-nixos so that torch (built from
    # source with CUDA) is already in the nix store — zero rebuild time.
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

        # CUDA root for tinygrad (needs include/ + libcuda.so)
        cuda-root = pkgs.runCommand "tinygrad-cuda-root" {} ''
          mkdir -p $out
          ln -s ${pkgs.lib.getDev cuda.cuda_cudart}/include $out/include
          ln -s ${jetpack.l4t-cuda}/lib/libcuda.so $out/libcuda.so
          ln -s ${jetpack.l4t-cuda}/lib/libcuda.so.1 $out/libcuda.so.1
          ln -s ${jetpack.l4t-cuda}/lib/libcuda.so.1.1 $out/libcuda.so.1.1
        '';

        # TensorRT root for the benchmark
        trt-root = pkgs.runCommand "trt-root" {} ''
          mkdir -p $out/lib $out/include
          for f in ${cuda.tensorrt}/lib/*.so*; do
            ln -s "$f" $out/lib/
          done
          for f in ${cuda.tensorrt}/include/*; do
            ln -s "$f" $out/include/
          done
        '';

        # python312 to match control-loop (same torch derivation hash = cached)
        pythonEnv = pkgs.python312.withPackages (ps: [
          ps.torch        # CUDA-enabled torch (same as control-loop, already built)
          ps.numpy
          ps.tqdm
          ps.tabulate
          ps.onnx
          ps.protobuf
          ps.h5py
          ps.numba        # needed for Cioffi's real scekf.py (their numba-JIT'd EKF)
          ps.scipy        # needed for Cioffi's net_input_utils.py (interpolation)
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          name = "cioffi-tcn-bench";

          buildInputs = [
            pythonEnv
            pkgs.git
            pkgs.clang
            cuda.tensorrt
          ];

          CC = "${pkgs.clang}/bin/clang";
          CXX = "${pkgs.clang}/bin/clang++";

          CUDA_PATH = "${cuda-root}";
          NVRTC_PATH = "${pkgs.lib.getLib cuda.cuda_nvrtc}/lib/libnvrtc.so";
          NVJITLINK_PATH = "${pkgs.lib.getLib cuda.libnvjitlink}/lib/libnvJitLink.so";
          LIBC_PATH = "${pkgs.glibc}/lib/libc.so.6";
          TENSORRT_PATH = "${trt-root}";

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
          ];

          shellHook = ''
            export CC="${pkgs.clang}/bin/clang"
            export CXX="${pkgs.clang}/bin/clang++"

            # Find tinygrad
            if [ -d "$PWD/../../external/tinygrad/tinygrad" ]; then
              TINYGRAD_PATH="$(realpath "$PWD/../../external/tinygrad")"
            elif [ -d "$PWD/external/tinygrad/tinygrad" ]; then
              TINYGRAD_PATH="$PWD/external/tinygrad"
            else
              echo "WARNING: cannot find tinygrad at external/tinygrad/"
              echo "Run: git submodule update --init --recursive"
            fi
            if [ -n "''${TINYGRAD_PATH:-}" ]; then
              export PYTHONPATH="$TINYGRAD_PATH:$PYTHONPATH"
            fi

            # Find learned_inertial_model_odometry (their real code)
            if [ -d "$PWD/../../external/learned_inertial_model_odometry/src" ]; then
              CIOFFI_REPO="$(realpath "$PWD/../../external/learned_inertial_model_odometry")"
              export CIOFFI_REPO
              export PYTHONPATH="$CIOFFI_REPO/src:$PYTHONPATH"
            fi

            # Hot path shared objects
            HP_DIR="$(realpath "$PWD/../control-loop/hot_path" 2>/dev/null || true)"
            if [ -n "$HP_DIR" ] && [ -f "$HP_DIR/hot_path.so" ]; then
              export HOT_PATH_DIR="$HP_DIR"
            fi

            echo ""
            echo "=== Cioffi TCN Benchmark Shell (Orin AGX / CUDA 12.6) ==="
            echo "  Paper: Learned Inertial Odometry (Cioffi et al., RAL 2023)"
            echo "  Model: TCN ~250K params, input=(1,6,50), output=(1,3)"
            echo "  EKF:   Their real ImuMSCKF (numba JIT, FEJ)"
            echo "  PyTorch: CUDA-enabled (from control-loop flake)"
            echo ""
            echo "  tinygrad:  ''${TINYGRAD_PATH:-NOT FOUND}"
            echo "  Cioffi:    ''${CIOFFI_REPO:-NOT FOUND}"
            echo "  Hot Path:  ''${HOT_PATH_DIR:-NOT FOUND}"
            echo "  TensorRT:  ${cuda.tensorrt}"
            echo ""
            echo "Usage:"
            echo "  # TCN-only benchmark (all backends)"
            echo "  NV=1 JITBEAM=2 python3 bench_cioffi_tcn.py --backend all"
            echo ""
            echo "  # End-to-end pipeline (all backends, side-by-side)"
            echo "  NV=1 JITBEAM=2 python3 bench_e2e_pipeline.py --backend all"
            echo ""
            echo "  # Individual backends"
            echo "  NV=1 JITBEAM=2 python3 bench_e2e_pipeline.py --backend nv"
            echo "  NV=1 JITBEAM=2 python3 bench_e2e_pipeline.py --backend hotpath"
            echo "  python3 bench_e2e_pipeline.py --backend trt"
            echo "  python3 bench_e2e_pipeline.py --backend pytorch"
            echo ""
          '';
        };
      }
    );
}
