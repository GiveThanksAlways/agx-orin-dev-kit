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
          echo "Chat:    llama-cli -hf unsloth/Qwen3-Coder-Next-GGUF:Q5_K_XL --gpu-layers 999"
          echo "Server:  llama-server -hf ... --gpu-layers 999 --host 0.0.0.0 --port 5000"
          echo "Bench:   llama-bench -hf ..."
          echo ""
        '';
      };
    };
}
