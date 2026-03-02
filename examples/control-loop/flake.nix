{
  description = "PyTorch CUDA on Jetson AGX Orin (built from source via nixpkgs)";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";
    jetpack-nixos.url = "github:anduril/jetpack-nixos";
    jetpack-nixos.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, jetpack-nixos }:
    let
      system = "aarch64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          # Tells nixpkgs to build torch (and all CUDA-aware packages) with CUDA.
          cudaSupport = true;
          # Target Orin AGX (SM 8.7). Keeps compile time sane — skips other arches.
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

      # Python 3.12 — fully supported in nixpkgs 25.11.
      # torch builds from source with CUDA when cudaSupport = true above.
      pythonEnv = pkgs.python312.withPackages (ps: [
        ps.torch
        ps.numpy
        ps.matplotlib
      ]);
    in
    {
      packages.${system}.default = pythonEnv;

      devShells.${system}.default = pkgs.mkShell {
        packages = [ pythonEnv ];
        shellHook = ''
          echo "PyTorch $(python3 -c 'import torch; print(torch.__version__)') ready."
          python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
        '';
      };
    };
}
