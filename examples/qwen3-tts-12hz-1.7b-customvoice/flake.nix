{
  description = "Qwen3-TTS-12Hz-1.7B-CustomVoice dev shell for Jetson Orin AGX";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "aarch64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          name = "qwen3-tts-customvoice-orin";

          buildInputs = [
            pkgs.python3
            pkgs.python3Packages.pip
            pkgs.python3Packages.virtualenv
            pkgs.ffmpeg
            pkgs.libsndfile
          ];

          shellHook = ''
            echo
            echo "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice dev shell (Jetson Orin AGX)"
            echo
            VENV_DIR="$PWD/.venv"
            if [ ! -d "$VENV_DIR" ]; then
              echo "Creating Python venv in .venv/ ..."
              python3 -m venv "$VENV_DIR"
            fi
            source "$VENV_DIR/bin/activate"
            echo
            echo "Install runtime deps:"
            echo "  pip install -U transformers accelerate soundfile"
            echo
            echo "Then run your Qwen3-TTS inference script against:"
            echo "  Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            echo
          '';
        };
      }
    );
}
