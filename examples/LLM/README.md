on Jetson Orin AGX with CUDA support.
vllm.url = "path:../vLLM";

# MLC-LLM & TensorRT on Jetson Orin AGX

**Quickstart:**

```bash
nix develop
# Activate Python venv
source .venv/bin/activate
# Install MLC-LLM
pip install mlc-llm
# Test TensorRT Python bindings
python -c 'import tensorrt; print(tensorrt.__version__)'
```

## TL;DR
- Dev shell provides CUDA 12.6, TensorRT, Python, cmake
- Use MLC-LLM for fast LLM inference: https://github.com/mlc-ai/mlc-llm
- Use TensorRT for optimized GPU inference: https://developer.nvidia.com/tensorrt
- Works out-of-the-box on JetPack 6 (Orin AGX)

## Benchmarking
- Run MLC-LLM: `mlc_llm serve --model <model>`
- Use TensorRT for custom models/scripts

## Troubleshooting
- If TensorRT libraries not found, check JetPack overlay version
- For Python issues, ensure `.venv` is activated

## More info
- See upstream docs:
  - [MLC-LLM](https://github.com/mlc-ai/mlc-llm)
  - [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/)
