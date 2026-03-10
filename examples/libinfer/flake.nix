{
  description = "libinfer dev shell — Rust + TensorRT + CUDA on Jetson AGX Orin";

  inputs = {
    # Follow control-loop's inputs so TensorRT etc. are already cached.
    # We don't use PyTorch — just the same nixpkgs + jetpack-nixos revisions.
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
          config.allowUnfree = true;
          overlays = [
            jetpack-nixos.overlays.default
            (final: _: { inherit (final.nvidia-jetpack6) cudaPackages; })
          ];
        };
        jetpack = pkgs.nvidia-jetpack6;
        cuda = jetpack.cudaPackages;
      in
      {
        devShells.default = pkgs.mkShell {
          name = "libinfer-dev";

          nativeBuildInputs = [
            pkgs.rustc
            pkgs.cargo
            pkgs.rustfmt
            pkgs.clang
            pkgs.pkg-config
            pkgs.binutils        # ar, needed by cc crate
          ];

          buildInputs = [
            pkgs.spdlog
            pkgs.fmt
            pkgs.openssl
            cuda.tensorrt
            cuda.cuda_cudart
            cuda.cuda_nvcc      # crt/host_defines.h needed by cuda_runtime_api.h
          ];

          CC = "${pkgs.clang}/bin/clang";
          CXX = "${pkgs.clang}/bin/clang++";

          # build.rs reads these three env vars
          TENSORRT_LIBRARIES = "${pkgs.lib.getLib cuda.tensorrt}/lib";
          CUDA_INCLUDE_DIRS = "${pkgs.lib.getDev cuda.cuda_cudart}/include";
          CUDA_LIBRARIES = "${pkgs.lib.getLib cuda.cuda_cudart}/lib";

          # clang++ needs TensorRT + CUDA headers on the include path
          CPLUS_INCLUDE_PATH = builtins.concatStringsSep ":" [
            "${cuda.tensorrt}/include"
            "${pkgs.lib.getDev cuda.cuda_cudart}/include"
            "${pkgs.lib.getDev cuda.cuda_nvcc}/include"
            "${pkgs.lib.getDev pkgs.fmt}/include"
            "${pkgs.lib.getDev pkgs.spdlog}/include"
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            (pkgs.lib.getLib cuda.cuda_cudart)
            (pkgs.lib.getLib cuda.tensorrt)
            jetpack.l4t-cuda
            pkgs.stdenv.cc.cc
          ];

          LIBCLANG_PATH = "${pkgs.libclang.lib}/lib/";

          shellHook = ''
            export CC="${pkgs.clang}/bin/clang"
            export CXX="${pkgs.clang}/bin/clang++"
          '';
        };
      });
}
