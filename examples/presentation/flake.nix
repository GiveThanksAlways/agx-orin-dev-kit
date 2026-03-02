{
  description = "Presentation benchmarks: tinygrad NV=1 vs TensorRT on Jetson Orin AGX";

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
          overlays = [
            jetpack-nixos.overlays.default
            (final: _: { inherit (final.nvidia-jetpack6) cudaPackages; })
          ];
        };
        jetpack = pkgs.nvidia-jetpack6;
        cuda = jetpack.cudaPackages;

        # Same CUDA root as the tinygrad flake — tinygrad needs include/ + libcuda.so
        cuda-root = pkgs.runCommand "tinygrad-cuda-root" {} ''
          mkdir -p $out
          ln -s ${pkgs.lib.getDev cuda.cuda_cudart}/include $out/include
          ln -s ${jetpack.l4t-cuda}/lib/libcuda.so $out/libcuda.so
          ln -s ${jetpack.l4t-cuda}/lib/libcuda.so.1 $out/libcuda.so.1
          ln -s ${jetpack.l4t-cuda}/lib/libcuda.so.1.1 $out/libcuda.so.1.1
        '';

        # TensorRT include + lib root for the C benchmark harness
        trt-root = pkgs.runCommand "trt-root" {} ''
          mkdir -p $out/lib $out/include
          for f in ${cuda.tensorrt}/lib/*.so*; do
            ln -s "$f" $out/lib/
          done
          for f in ${cuda.tensorrt}/include/*; do
            ln -s "$f" $out/include/
          done
        '';

        pythonEnv = pkgs.python3.withPackages (ps: [
          ps.numpy
          ps.tqdm
          ps.tabulate
          ps.onnx          # ONNX model export for TensorRT
          ps.protobuf      # required by onnx
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          name = "presentation-bench";

          buildInputs = [
            pythonEnv
            pkgs.git
            pkgs.clang
            cuda.tensorrt       # TensorRT libs + headers
          ];

          CC = "${pkgs.clang}/bin/clang";
          CXX = "${pkgs.clang}/bin/clang++";

          # tinygrad env vars (same as examples/tinygrad/flake.nix)
          CUDA_PATH = "${cuda-root}";
          NVRTC_PATH = "${pkgs.lib.getLib cuda.cuda_nvrtc}/lib/libnvrtc.so";
          NVJITLINK_PATH = "${pkgs.lib.getLib cuda.libnvjitlink}/lib/libnvJitLink.so";
          LIBC_PATH = "${pkgs.glibc}/lib/libc.so.6";

          # TensorRT paths for the benchmark scripts
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
            cuda.tensorrt                        # libnvinfer.so etc.
            jetpack.l4t-cuda
            jetpack.l4t-core
            pkgs.stdenv.cc.cc
          ];

          shellHook = ''
            export CC="${pkgs.clang}/bin/clang"
            export CXX="${pkgs.clang}/bin/clang++"

            # Find tinygrad from parent repo
            if [ -d "$PWD/external/tinygrad/tinygrad" ]; then
              TINYGRAD_PATH="$PWD/external/tinygrad"
            elif [ -d "$(realpath "$PWD/../../external/tinygrad" 2>/dev/null)/tinygrad" ]; then
              TINYGRAD_PATH="$(realpath "$PWD/../../external/tinygrad")"
            else
              echo "WARNING: cannot find tinygrad submodule at external/tinygrad/"
              echo "tinygrad benchmarks will not work."
              echo "From repo root: git submodule update --init --recursive"
            fi

            if [ -n "''${TINYGRAD_PATH:-}" ]; then
              export PYTHONPATH="$TINYGRAD_PATH:$PYTHONPATH"
            fi

            echo ""
            echo "=== Presentation Benchmark Shell (Orin AGX / CUDA 12.6) ==="
            echo "  tinygrad NV=1 vs TensorRT — MLP, CNN, Hybrid architectures"
            echo ""
            echo "  TensorRT:  ${cuda.tensorrt}"
            echo "  tinygrad:  ''${TINYGRAD_PATH:-NOT FOUND}"
            echo ""
            echo "Usage:"
            echo "  NV=1 JITBEAM=2 python3 bench_models.py          # Run all benchmarks"
            echo "  NV=1 JITBEAM=2 python3 bench_models.py --arch mlp   # MLP only"
            echo "  NV=1 JITBEAM=2 python3 bench_models.py --arch cnn   # CNN only"
            echo "  NV=1 JITBEAM=2 python3 bench_models.py --arch hybrid # Hybrid only"
            echo ""
          '';
        };
      }
    );
}
