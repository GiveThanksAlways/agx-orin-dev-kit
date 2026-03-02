{
  description = "Tinygrad dev shell for Jetson Orin AGX (CUDA + NV backends)";

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
            # Must match control-loop's config exactly so torch's derivation hash
            # is identical — Nix reuses the cached build instead of rebuilding.
            cudaSupport = true;
            cudaCapabilities = [ "8.7" ];
            cudaForwardCompat = false;
          };
          overlays = [
            jetpack-nixos.overlays.default
            # jetpack-nixos defaults cudaPackages to CUDA 11.4 for broad compatibility.
            # Override it to JetPack 6 / CUDA 12.6 so that nixpkgs packages like torch
            # get built against the correct CUDA version for Orin AGX.
            # See jetpack-nixos README: "Re-using jetpack-nixos's CUDA package set".
            (final: _: { inherit (final.nvidia-jetpack6) cudaPackages; })
          ];
        };
        jetpack = pkgs.nvidia-jetpack6;
        cuda = jetpack.cudaPackages;

        # Combined CUDA root directory that tinygrad's CUDA_PATH can point to.
        # compiler_cuda.py does: -I{CUDA_PATH}/include  → needs include/ subdir
        # DLL.findlib searches CUDA_PATH directory for libcuda.so*  → needs libcuda.so at top level
        # This derivation creates a flat layout with both.
        cuda-root = pkgs.runCommand "tinygrad-cuda-root" {} ''
          mkdir -p $out
          ln -s ${pkgs.lib.getDev cuda.cuda_cudart}/include $out/include
          ln -s ${jetpack.l4t-cuda}/lib/libcuda.so $out/libcuda.so
          ln -s ${jetpack.l4t-cuda}/lib/libcuda.so.1 $out/libcuda.so.1
          ln -s ${jetpack.l4t-cuda}/lib/libcuda.so.1.1 $out/libcuda.so.1.1
        '';

        # python312 to match control-loop (same torch derivation hash = cached)
        pythonEnv = pkgs.python312.withPackages (ps: [
          ps.torch        # CUDA-enabled torch (same as control-loop, already built)
          ps.numpy
          ps.tqdm
          ps.requests
          ps.pillow
          ps.tiktoken  # for GPT-2 tokenization
          ps.pytest  # for running tinygrad's test suite
          ps.hypothesis  # for property-based tests (test_jit.py, test_tensor.py)
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          name = "tinygrad-orin";

          buildInputs = [
            pythonEnv
            pkgs.git
            pkgs.clang  # tinygrad needs clang for CPU JIT compilation
          ];

          # Force clang as CC so tinygrad CPU backend works (gcc doesn't support --target)
          CC = "${pkgs.clang}/bin/clang";
          CXX = "${pkgs.clang}/bin/clang++";

          # tinygrad uses its own library finder (DLL.findlib in runtime/support/c.py)
          # that searches hardcoded FHS paths (/usr/lib, /lib64, etc.) which don't exist on NixOS.
          # It checks <NAME>_PATH env vars first, so we point those directly at the .so files.
          # See: grep -rn 'c.DLL' tinygrad/runtime/ for the full list of libraries.
          #
          # CUDA backend libs (required):
          #
          # IMPORTANT: tinygrad's NVRTCCompiler reads CUDA_PATH and appends /include to it
          # (tinygrad/runtime/support/compiler_cuda.py line ~47).
          # tinygrad's DLL.findlib also searches CUDA_PATH directory for libcuda.so*.
          # We use a merged derivation that has both include/ and lib/libcuda.so.
          CUDA_PATH = "${cuda-root}";
          NVRTC_PATH = "${pkgs.lib.getLib cuda.cuda_nvrtc}/lib/libnvrtc.so";
          NVJITLINK_PATH = "${pkgs.lib.getLib cuda.libnvjitlink}/lib/libnvJitLink.so";
          # System libs (tinygrad loads libc via ctypes for io_uring/mmap/etc.):
          LIBC_PATH = "${pkgs.glibc}/lib/libc.so.6";

          # LD_LIBRARY_PATH is still needed so the dynamic linker can resolve
          # transitive dependencies (e.g. libcudart, libcublas, libstdc++).
          # Also includes glibc so tinygrad's CPU backend can find libgcc_s.so.1.
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            (pkgs.lib.getLib cuda.cuda_cudart)   # libcudart.so
            (pkgs.lib.getLib cuda.libcublas)      # libcublas.so
            (pkgs.lib.getLib cuda.libcusparse)    # libcusparse.so
            (pkgs.lib.getLib cuda.libcusolver)    # libcusolver.so
            (pkgs.lib.getLib cuda.libcufft)       # libcufft.so
            (pkgs.lib.getLib cuda.libcurand)      # libcurand.so
            (pkgs.lib.getLib cuda.cuda_nvrtc)     # libnvrtc.so
            (pkgs.lib.getLib cuda.libnvjitlink)   # libnvJitLink.so
            (pkgs.lib.getLib cuda.cudnn)          # libcudnn.so
            jetpack.l4t-cuda                      # libcuda.so (Jetson CUDA driver)
            jetpack.l4t-core                      # libnvcucompat.so
            pkgs.stdenv.cc.cc                     # libstdc++.so
          ];

          shellHook = ''
            # Force clang as the C/C++ compiler — tinygrad's CPU backend uses
            # --target=aarch64-none-unknown-elf which is a clang-only flag.
            # mkShell's stdenv sets CC=gcc, so we override here.
            export CC="${pkgs.clang}/bin/clang"
            export CXX="${pkgs.clang}/bin/clang++"

            # Local-source workflow: tinygrad is imported from the repo's submodule at
            # external/tinygrad/.  The submodule used to live at examples/tinygrad/tinygrad/
            # and was moved; both locations are checked so the shell works regardless of
            # where you run `nix develop` from.
            #
            # Expected usage:   cd examples/tinygrad && nix develop
            # Also works:       nix develop examples/tinygrad# (from repo root)

            if [ -d "$PWD/external/tinygrad/tinygrad" ]; then
              # Running nix develop from the repo root
              TINYGRAD_PATH="$PWD/external/tinygrad"
            elif [ -d "$(realpath "$PWD/../../external/tinygrad" 2>/dev/null)/tinygrad" ]; then
              # Running nix develop from examples/tinygrad/ (expected)
              TINYGRAD_PATH="$(realpath "$PWD/../../external/tinygrad")"
            elif [ -d "$PWD/tinygrad/tinygrad" ]; then
              # Legacy: submodule still at examples/tinygrad/tinygrad/
              TINYGRAD_PATH="$PWD/tinygrad"
            else
              echo ""
              echo "ERROR: cannot find tinygrad submodule."
              echo "From the repo root, run:"
              echo "  git submodule update --init --recursive"
              echo ""
              return 1
            fi

            export PYTHONPATH="$TINYGRAD_PATH:$PYTHONPATH"

            echo ""
            echo "=== tinygrad dev shell (Orin AGX / CUDA 12.6, local source mode) ==="
            echo "Using tinygrad from: $TINYGRAD_PATH"
            echo ""
            echo "Quick test (CPU):"
            echo "  python3 -c 'from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())'"
            echo ""
            echo "Quick test (NV backend — Jetson Orin):"
            echo "  NV=1 python3 -c 'from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())'"
            echo ""
            echo "GPT-2 (auto-downloads weights):"
            echo "  NV=1 python3 \$TINYGRAD_PATH/examples/gpt2.py --count 20"
            echo ""
            echo "Radix KV-cache test (GPT-2 + multi-agent prefix reuse):"
            echo "  NV=1 python3 test_radix_gpt2.py"
            echo ""
            echo "PYTHONPATH includes \$TINYGRAD_PATH for tinygrad + extra modules."
            echo ""
          '';
        };

        devShells.detective = pkgs.mkShell {
          packages = with pkgs; [
            binutils  # objdump, etc.
            gcc
            git
            wget
            unzip
            gnumake
            ctags  # For source navigation
            strace  # For ioctl tracing
            gdb
          ];
        };
      }
    );
}
