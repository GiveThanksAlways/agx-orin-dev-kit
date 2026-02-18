{
  description = "Advanced llama.cpp dev environment for Jetson Orin AGX";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "aarch64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
            cudaCapabilities = [ "8.7" ]; # Orin AGX sm_87
          };
          overlays = [ self.overlays.default ];
        };
      in
      {
        # Dev shell with all tools
        devShells.default = pkgs.mkShell {
          name = "llama-cpp-advanced";

          buildInputs = with pkgs; [
            llama-cpp-orin
            qwen3-coder
            qwen3-server
            llama-benchmark
            # Utilities
            curl wget htop
          ];

          shellHook = ''
            echo ""
            echo "llama.cpp overlay shell for Orin AGX"
            echo ""
            echo "Commands:"
            echo "  qwen3-coder           Chat with Qwen3-Coder-Next (Q5_K_XL)"
            echo "  qwen3-coder Q4_K_M    Use smaller quant (~35GB)"
            echo "  qwen3-server          API server on :8080"
            echo "  llama-benchmark       Run performance benchmark"
            echo ""
            echo "Raw llama.cpp:"
            echo "  llama-cli -hf unsloth/Qwen3-Coder-Next-GGUF:Q5_K_XL -ngl 999"
            echo "  llama-server -hf ... --host 0.0.0.0 --port 8080"
            echo ""
            echo "Model sizes: Q4_K_M (~35GB) | Q5_K_M (~45GB) | Q5_K_XL (~57GB)"
            echo ""

            mkdir -p ~/.cache/huggingface
          '';
        };

        # Packages
        packages = {
          default = pkgs.llama-cpp-orin;
          llama-cpp = pkgs.llama-cpp-orin;
          qwen3-coder = pkgs.qwen3-coder;
          qwen3-server = pkgs.qwen3-server;
          benchmark = pkgs.llama-benchmark;
        };

        # Apps - run directly with nix run
        apps = {
          default = {
            type = "app";
            program = "${pkgs.qwen3-coder}/bin/qwen3-coder";
          };
          server = {
            type = "app";
            program = "${pkgs.qwen3-server}/bin/qwen3-server";
          };
          benchmark = {
            type = "app";
            program = "${pkgs.llama-benchmark}/bin/llama-benchmark";
          };
          llama-cli = {
            type = "app";
            program = "${pkgs.llama-cpp-orin}/bin/llama-cli";
          };
        };
      }
    ) // {
      # Overlay - can be used by other flakes
      overlays.default = final: prev: {
        # llama.cpp with CUDA for Orin
        llama-cpp-orin = prev.llama-cpp.override {
          cudaSupport = true;
        };

        # Interactive chat wrapper
        qwen3-coder = final.writeShellScriptBin "qwen3-coder" ''
          set -euo pipefail
          QUANT="''${1:-Q5_K_XL}"
          MODEL="unsloth/Qwen3-Coder-Next-GGUF:$QUANT"

          echo "Starting Qwen3-Coder-Next ($QUANT)..."
          echo "Model will download to ~/.cache/huggingface on first run"
          echo ""

          # -cnv enables conversation mode (multi-turn chat)
          # Remove -cnv if using older llama.cpp versions
          exec ${final.llama-cpp-orin}/bin/llama-cli \
            -hf "$MODEL" \
            --gpu-layers 999 \
            --ctx-size 32768 \
            --temp 0.6 \
            ''${LLAMA_EXTRA_FLAGS:--cnv} \
            "''${@:2}"
        '';

        # API server wrapper
        qwen3-server = final.writeShellScriptBin "qwen3-server" ''
          set -euo pipefail
          QUANT="''${1:-Q5_K_XL}"
          PORT="''${2:-8080}"
          MODEL="unsloth/Qwen3-Coder-Next-GGUF:$QUANT"

          echo "Starting Qwen3-Coder-Next API server..."
          echo "Model: $MODEL"
          echo "Endpoint: http://0.0.0.0:$PORT"
          echo ""
          echo "Test with:"
          echo "  curl http://localhost:$PORT/v1/chat/completions \\"
          echo "    -H 'Content-Type: application/json' \\"
          echo "    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
          echo ""

          exec ${final.llama-cpp-orin}/bin/llama-server \
            -hf "$MODEL" \
            --gpu-layers 999 \
            --ctx-size 32768 \
            --host 0.0.0.0 \
            --port "$PORT" \
            "''${@:3}"
        '';

        # Benchmark wrapper
        llama-benchmark = final.writeShellScriptBin "llama-benchmark" ''
          set -euo pipefail
          QUANT="''${1:-Q5_K_XL}"
          MODEL="unsloth/Qwen3-Coder-Next-GGUF:$QUANT"

          echo "Benchmarking $MODEL on Orin AGX..."
          echo ""

          exec ${final.llama-cpp-orin}/bin/llama-bench \
            -hf "$MODEL" \
            --gpu-layers 999 \
            -p 512 \
            -n 128 \
            "''${@:2}"
        '';
      };
    };
}
