# llama.cpp overlay for Jetson Orin AGX

Full-featured Nix overlay with wrapper scripts for llama.cpp on Orin AGX.
Provides `qwen3-coder`, `qwen3-server`, and `llama-benchmark` commands
out of the box. The overlay is reusable from other flakes.

## Quick start

```bash
nix develop       # enter dev shell with all tools
qwen3-coder       # interactive chat (Q5_K_XL, ~57 GB)
```

## Commands

| Command | Description |
|---|---|
| `qwen3-coder` | Interactive chat with Qwen3-Coder-Next (Q5_K_XL) |
| `qwen3-coder Q4_K_M` | Use a smaller quant (~35 GB) |
| `qwen3-server` | Start OpenAI-compatible API on :8080 |
| `qwen3-server Q4_K_M 5000` | Server with custom quant and port |
| `llama-benchmark` | Run llama-bench on Orin |

## API server

```bash
qwen3-server

# Test:
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

## Using the overlay from another flake

```nix
{
  inputs.llama-orin.url = "path:./examples/llama-cpp-orin-nix-overlay";

  outputs = { nixpkgs, llama-orin, ... }: {
    packages.aarch64-linux = let
      pkgs = import nixpkgs {
        system = "aarch64-linux";
        overlays = [ llama-orin.overlays.default ];
      };
    in { inherit (pkgs) llama-cpp-orin qwen3-coder qwen3-server; };
  };
}
```

## Model quants

| Quant | Size | VRAM | Quality |
|---|---|---|---|
| Q4_K_M | ~35 GB | ~40 GB | Good |
| Q5_K_M | ~45 GB | ~50 GB | Better |
| Q5_K_XL | ~57 GB | ~64 GB | Best (default) |
