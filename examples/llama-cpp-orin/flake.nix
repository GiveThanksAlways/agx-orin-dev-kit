{
  description = "llama.cpp for Jetson Orin AGX";

  inputs = {
    llama-cpp.url = "github:ggml-org/llama.cpp";
  };

  outputs = { self, llama-cpp, ... }:
    let
      system = "aarch64-linux";
      # Use llama-cpp's nixpkgs to avoid derivation hash mismatch
      pkgs = llama-cpp.inputs.nixpkgs.legacyPackages.${system};
      llamaCpp = (llama-cpp.packages.${system}.jetson-orin).overrideAttrs (old: {
        buildInputs = old.buildInputs ++ [ pkgs.openssl ];
        cmakeFlags = old.cmakeFlags ++ [ "-DLLAMA_OPENSSL=ON" ];
      });
    in
    {
      packages.${system}.default = llamaCpp;

      devShells.${system}.default = pkgs.mkShell {
        name = "llama-cpp-orin";
        packages = [ llamaCpp ];
        shellHook = ''
          echo ""
          echo "llama.cpp for Jetson Orin AGX"
          echo "Version: $(llama-cli --version 2>&1 | head -1)"
          echo ""
          echo "Model: $HOME/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf"
          echo "  Qwen3-Coder-Next MXFP4 MoE (80B.A3B) | 44.7 GiB | 262k context"
          echo "  Tested: ~193 t/s prefill | ~15.8 t/s generation | 8 GiB free after load"
          echo "  NOTE: use 131072 ctx (not 262144) to avoid OOM in long multi-turn sessions"
          echo ""
          echo "Recommended server command (128k ctx, single user, UMA trick):"
          echo ""
          echo "  llama-server \\"
          echo "    --model \$HOME/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf \\"
          echo "    --host 0.0.0.0 --port 5000 \\"
          echo "    --ctx-size 131072 --threads 12 \\"
          echo "    --parallel 1 \\"
          echo "    --n-gpu-layers 999 --fit off \\"
          echo "    --cache-type-k q8_0 --cache-type-v q8_0 \\"
          echo "    --flash-attn on --mmap --no-warmup \\"
          echo "    --cache-ram 0 \\"
          echo "    --alias Qwen3-Coder-Next"
          echo ""
          echo "Key flags:"
          echo "  --n-gpu-layers 999 --fit off   routes all layers through CUDA on UMA (the magic)"
          echo "  --parallel 1                   single user, all KV cache to one slot"
          echo "  --cache-type-k/v q8_0          compress KV cache (262k ctx = ~3 GiB vs ~6 GiB fp16)"
          echo "  --cache-ram 0                  disable prompt cache snapshots (OOM risk at 262k ctx)"
          echo "  (no --ubatch-size)             default ubatch=512 is fastest for this MoE model"
          echo ""
          echo "Test with llama-cli (benchmark prefill/generation speed):"
          echo ""
          echo "  llama-cli \\"
          echo "    --model \$HOME/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf \\"
          echo "    --ctx-size 131072 --threads 12 \\"
          echo "    --n-gpu-layers 999 --fit off \\"
          echo "    --cache-type-k q8_0 --cache-type-v q8_0 \\"
          echo "    --flash-attn on --mmap --no-warmup --perf \\"
          echo "    -f /path/to/some/file"
          echo ""
          echo "  Expected: ~193 t/s prefill | ~15.8 t/s generation"
          echo ""
          echo "Check from the Orin:"
          echo "  curl http://127.0.0.1:5000/v1/models"
          echo "  curl http://127.0.0.1:5000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"Qwen3-Coder-Next\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":false,\"max_tokens\":64}'"
          echo ""
          echo "Check from your PC:"
          echo "  curl http://192.168.8.162:5000/v1/models"
          echo ""
          echo "Watch GPU while serving:"
          echo "  sudo tegrastats | grep --color=always -E 'GR3D_FREQ [1-9][0-9]*%|RAM|CPU|$'"
          echo ""
        '';
      };
    };
}
