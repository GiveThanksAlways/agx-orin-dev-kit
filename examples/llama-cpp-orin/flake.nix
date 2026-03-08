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
          echo "Local Qwen3 models I found:"
          echo "  $HOME/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf"
          echo ""
          echo "Single-user Qwen3-Coder-Next presets for 64 GB Orin:"
          echo "  small  (8k ctx):  llama-server --model $HOME/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf --host 0.0.0.0 --port 5000 --ctx-size 8192 --threads 8 --batch-size 64 --ubatch-size 64 --n-parallel 1 --n-gpu-layers auto --flash-attn on --mmap --no-warmup"
          echo "  medium (16k ctx): llama-server --model $HOME/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf --host 0.0.0.0 --port 5000 --ctx-size 16384 --threads 8 --batch-size 64 --ubatch-size 64 --n-parallel 1 --n-gpu-layers auto --flash-attn on --mmap --no-warmup"
          echo "  large  (32k ctx): llama-server --model $HOME/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf --host 0.0.0.0 --port 5000 --ctx-size 32768 --threads 8 --batch-size 64 --ubatch-size 64 --n-parallel 1 --n-gpu-layers auto --flash-attn on --mmap --no-warmup"
          echo ""
          echo "Recommended starting point for OpenCode: medium (16k ctx)."
          echo ""
          echo "Check from the Orin:"
          echo "  curl http://127.0.0.1:5000/v1/models"
          echo "  curl http://127.0.0.1:5000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":false,\"max_tokens\":64}'"
          echo ""
          echo "Check from your PC:"
          echo "  curl http://192.168.8.162:5000/v1/models"
          echo ""
          echo "Watch GPU while serving:"
          echo "  sudo tegrastats"
          echo "  sudo tegrastats | grep --color=always -E 'GR3D_FREQ [1-9][0-9]*%|RAM|CPU|$'"
          echo ""
        '';
      };
    };
}
