{
  description = "vLLM inference engine for Jetson Orin AGX — Docker-first with native fallback";

  # ┌──────────────────────────────────────────────────────────────────────┐
  # │ PREREQUISITE: Docker + NVIDIA Container Toolkit must be enabled     │
  # │ in NixOS.  See ../nixos/ for the docker-bench configuration:        │
  # │                                                                     │
  # │   sudo nixos-rebuild switch --flake ../nixos#nixos-docker-bench     │
  # │                                                                     │
  # │ Then:  nix develop        # enters shell with docker + benchmark    │
  # │        ./run-vllm-docker.sh                                         │
  # │        python3 bench_vllm.py                                        │
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
        # ── Primary dev shell: Docker-based vLLM ──────────────────────
        # Uses dustynv's Jetson-optimized container or builds from
        # Dockerfile.jetson.  Provides docker CLI, curl, jq, python3
        # for benchmarking.
        devShells.default = pkgs.mkShell {
          name = "vllm-orin";

          buildInputs = [
            pkgs.python3
            pkgs.python3Packages.requests
            pkgs.curl
            pkgs.jq
            pkgs.wget
          ];

          shellHook = ''
            echo ""
            echo "=== vLLM dev shell (Orin AGX — Docker) ==="
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
              echo "  # or: add yourself to 'docker' group and re-login"
              echo ""
            else
              echo "Docker: $(docker --version)"
              echo ""
            fi

            echo "Usage:"
            echo "  ./run-vllm-docker.sh                                # LLaMA 3.2 1B (default)"
            echo "  ./run-vllm-docker.sh meta-llama/Llama-3.2-3B-Instruct 8000"
            echo "  python3 bench_vllm.py                               # after server is up"
            echo "  python3 bench_vllm.py --server http://localhost:8000 --model <name>"
            echo ""
            echo "Docker images tried (in order):"
            echo "  1. vllm-jetson:latest    (local build from Dockerfile.jetson)"
            echo "  2. dustynv/vllm:r36.4.0  (dustynv's JetPack 6 pre-built)"
            echo ""
          '';
        };
      }
    );
}
