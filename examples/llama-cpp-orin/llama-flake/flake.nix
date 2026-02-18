{
  description = "Pinned llama.cpp with override";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    llama-cpp.url = "github:ggml-org/llama.cpp/9a5f57795c01c6e67a53eeedeae67ed63aaf7f8e";
  };

  outputs = { self, llama-cpp, nixpkgs }: {
    packages.aarch64-linux.default =
      (llama-cpp.packages.aarch64-linux.jetson-orin).overrideAttrs (old: {
        buildInputs = old.buildInputs ++ [ nixpkgs.legacyPackages.aarch64-linux.openssl ];
        cmakeFlags = old.cmakeFlags ++ [ "-DLLAMA_OPENSSL=ON" ];
      });
  };
}