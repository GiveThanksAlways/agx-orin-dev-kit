{
  description = "MLC LLM inference for Jetson Orin AGX — Docker-first with native fallback";

  # ┌──────────────────────────────────────────────────────────────────────┐
  # │ PREREQUISITE: Docker + NVIDIA Container Toolkit must be enabled     │
  # │ in NixOS.  See ../nixos/ for the docker-bench configuration:        │
  # │                                                                     │
  # │   sudo nixos-rebuild switch --flake ../nixos#nixos-docker-bench     │
  # │                                                                     │
  # │ Then:  nix develop        # enters shell with docker + benchmark    │
  # │        ./run-mlc-docker.sh                                          │
  # │        python3 bench_mlc_llm.py                                     │
  # └──────────────────────────────────────────────────────────────────────┘

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
        # ── Primary dev shell: Docker-based MLC LLM ──────────────────
        # Uses dustynv's Jetson-optimized MLC container.
        # Provides python3 for the benchmark client script.
        devShells.default = pkgs.mkShell {
          name = "mlc-llm-orin";

          buildInputs = [
            pkgs.python3
            pkgs.python3Packages.requests
            pkgs.curl
            pkgs.jq
            pkgs.wget
          ];

          shellHook = ''
            echo ""
            echo "=== MLC LLM dev shell (Orin AGX — Docker) ==="
            echo ""

            # Pre-flight: verify docker daemon
            if ! command -v docker &>/dev/null; then
              echo "WARNING: 'docker' not in PATH."
              echo "Enable Docker in NixOS config first:"
              echo "  sudo nixos-rebuild switch --flake ../nixos#nixos-docker-bench"
              echo ""
            elif ! docker info &>/dev/null 2>&1; then
              echo "WARNING: Docker daemon not running or permission denied."
              echo "  sudo systemctl start docker"
              echo ""
            else
              echo "Docker: $(docker --version)"
              echo ""
            fi

            echo "Usage:"
            echo "  ./run-mlc-docker.sh                         # LLaMA 3.2 1B q4f16 (default)"
            echo "  ./run-mlc-docker.sh <model-spec>            # custom MLC model"
            echo "  python3 bench_mlc_llm.py                    # after container is up"
            echo ""
            echo "Docker images tried (in order):"
            echo "  1. mlc-jetson:latest         (local build from Dockerfile.jetson)"
            echo "  2. dustynv/mlc:r36.4.0       (dustynv's JetPack 6 pre-built)"
            echo ""
          '';
        };
      }
    );
}
