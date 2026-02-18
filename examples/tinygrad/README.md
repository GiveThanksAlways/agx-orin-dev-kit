# Tinygrad on Jetson Orin AGX 64GB

CUDA and NV backend dev shell for tinygrad LLM examples.

## Setup

```bash
cd examples/tinygrad
nix develop

# tinygrad is auto-cloned + installed on first shell entry
```

## Run (CUDA backend)

```bash
# Smoke test
CUDA=1 python3 -c 'from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())'

# GPT-2 (small model, quick sanity check)
CUDA=1 python3 tinygrad/examples/gpt2.py --count 20

# LLaMA (download weights first -- see tinygrad repo docs)
CUDA=1 python3 tinygrad/examples/llama.py --prompt "Hello world"
```

## Run (NV backend -- experimental)

The NV backend talks directly to the NVIDIA driver.
It may not fully work on Orin yet but worth testing.

```bash
NV=1 python3 -c 'from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())'
NV=1 python3 tinygrad/examples/gpt2.py --count 10
```

## Notes

- 64GB unified memory means most 7B-class models fit without quantization
- For max perf, set power mode first: `sudo nvpmodel -m 0 && sudo jetson_clocks`
- If CUDA test fails, check `LD_LIBRARY_PATH` includes CUDA libs (the dev shell sets this)
- tinygrad repo: https://github.com/tinygrad/tinygrad
