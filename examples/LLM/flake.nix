{
  description = "TensorRT and MLC-LLM dev shell for Jetson Orin AGX";

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
      in
      {
        devShells.default = pkgs.mkShell {
          name = "tensorrt-mlc-llm-shell";
          buildInputs = [
            pkgs.python3
            pkgs.python3Packages.pip
            pkgs.python3Packages.virtualenv
            cuda.tensorrt
            pkgs.cmake
          ];
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath ([
            cuda.tensorrt
            cuda.cuda_cudart
            cuda.libcublas
            cuda.libcusparse
            cuda.libcusolver
            cuda.libcufft
            cuda.libcurand
            cuda.cuda_nvrtc
            cuda.cudnn
            pkgs.stdenv.cc.cc
          ]);
          shellHook = ''
            echo
            echo "TensorRT and MLC-LLM dev shell for Jetson Orin AGX (JetPack 6 / CUDA 12.6)"
            echo
            TRT_LIB_PATH="${cuda.tensorrt}/lib"
            if [ -d "$TRT_LIB_PATH" ]; then
              echo "TensorRT libraries found in $TRT_LIB_PATH"
            else
              echo "TensorRT library directory not found: $TRT_LIB_PATH"
              echo "If you see import errors, check JetPack overlay version and LD_LIBRARY_PATH."
            fi
            echo
            VENV_DIR="$PWD/.venv"
            if [ ! -d "$VENV_DIR" ]; then
              echo "Creating Python venv in .venv/ ..."
              python3 -m venv "$VENV_DIR"
            fi
            source "$VENV_DIR/bin/activate"
            echo
            echo "To install MLC-LLM:"
            echo "  pip install mlc-llm"
            echo
            echo "To test TensorRT Python bindings:"
            echo "  python -c 'import tensorrt; print(tensorrt.__version__)'"
            echo "If this fails, ensure LD_LIBRARY_PATH includes $TRT_LIB_PATH and that the overlay is correct."
            echo
            echo "For C++/CLI, link against libraries in $TRT_LIB_PATH."
            echo
          '';
        };
      }
    );
}