# vLLM on Jetson Orin AGX

Dev shell and NixOS module for [vLLM](https://github.com/vllm-project/vllm)
on Jetson Orin AGX with CUDA support.

## Dev shell

```bash
nix develop
vllm serve <model>
```

## NixOS module

This flake exports `nixosModules.default` providing `services.vllm-serving`.
Import it from your system flake:

```nix
# flake.nix inputs:
vllm.url = "path:../vLLM";

# modules:
modules = [
  vllm.nixosModules.default
  ({ ... }: {
    services.vllm-serving = {
      enable = true;
      model = "/models/Qwen3-Coder-Next-Q4_K_M.gguf";
      tokenizer = "Qwen/Qwen3-Coder-Next";
      quantization = "gguf";
      gpuMemoryUtilization = 0.90;
      maxModelLen = 4096;
    };
  })
];
```

## Options

| Option | Default | Description |
|---|---|---|
| `model` | -- | Path to GGUF model or HuggingFace ID |
| `tokenizer` | `""` | HuggingFace tokenizer (needed for GGUF) |
| `port` | `8000` | API port |
| `gpuMemoryUtilization` | `0.90` | GPU memory fraction |
| `maxModelLen` | `4096` | Max sequence length |
| `quantization` | `null` | Quantization method (gguf, awq, gptq) |
| `extraArgs` | `[]` | Extra CLI flags |
| `environment` | `{}` | Extra env vars |

API endpoint: `http://<host>:<port>/v1`
