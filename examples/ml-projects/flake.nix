{
  description = "Fun ML projects for Jetson Orin AGX — tinygrad + NV backend";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
  };

  outputs = { self, nixpkgs }: let
    system = "aarch64-linux";
    pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
  in {
    devShells.${system}.default = pkgs.mkShell {
      name = "ml-projects";

      packages = with pkgs; [
        # Python
        python311
        python311Packages.pip
        python311Packages.numpy
        python311Packages.pillow

        # Audio processing
        portaudio
        ffmpeg
        sox
        libsndfile

        # CUDA
        cudaPackages.cuda_cudart
        cudaPackages.cuda_nvcc

        # Build tools
        cmake
        gcc
        pkg-config
      ];

      shellHook = ''
        # Create venv if needed
        if [ ! -d .venv ]; then
          echo "Creating Python venv..."
          python3 -m venv .venv --system-site-packages
          source .venv/bin/activate
          pip install --quiet opencv-python-headless tinygrad soundfile librosa
        else
          source .venv/bin/activate
        fi

        export CUDA_PATH=${pkgs.cudaPackages.cuda_cudart}

        echo ""
        echo "=== ML Projects Dev Shell ==="
        echo "Available projects:"
        echo "  style-transfer/      Neural style transfer on images"
        echo "  audio-ml/            Audio classification + mel spectrogram"
        echo "  reinforcement-learning/ Q-learning + policy gradient agents"
        echo "  generative/          Autoencoder + diffusion experiments"
        echo "  edge-deploy/         Model optimization for edge inference"
        echo ""
        echo "Run with: NV=1 python3 <project>/<script>.py"
      '';
    };
  };
}
