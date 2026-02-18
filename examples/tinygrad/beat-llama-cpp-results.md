# Beat llama.cpp — Results

## Mission: Beat llama.cpp on LLaMA 3.2 1B Q6_K decode throughput

**Device**: Jetson Orin AGX 64GB, JetPack 6, L4T r36.4.4, SM 8.7, CUDA 12.6, LPDDR5  
**Model**: LLaMA 3.2 1B Instruct Q6_K (GGUF)  
**Metric**: Steady-state decode tokens/second (batch=1, autoregressive)

## Results

| Config | tok/s | ms/tok | vs llama.cpp |
|--------|-------|--------|-------------|
| llama.cpp (llama-bench tg128) | 25.62 | 39.0 | baseline |
| llama.cpp (FA=1) | 27.82 | 36.0 | 108.6% |
| tinygrad NV=1 (before fix) | 3.85 | 260.0 | 15% |
| tinygrad NV=1 + matvec fix (MV_TPR=8) | 18.34 | 54.5 | 72% |
| tinygrad NV=1 + matvec fix (MV_TPR=32) | 29.90 | 33.4 | 117% |
| tinygrad CUDA=1 + matvec fix (MV_TPR=32) | 29.97 | 33.4 | 117% |
| tinygrad NV=1 + matvec + JITBEAM=2 | 34.21 | 29.2 | 134% |
| **tinygrad NV=1 + matvec + JITBEAM=4** | **36.71** | **27.2** | **143%** |
| tinygrad CUDA=1 + matvec + JITBEAM=4 | 31.83 | 31.4 | 124% |
| tinygrad NV=1 + JITBEAM=4 (no MV override) | 31.92 | 31.3 | 125% |

### Key Insight: NV=1 vs CUDA=1 with BEAM

BEAM search benefits NV=1 much more than CUDA=1 (+23% vs +6%). With JITBEAM=4, NV=1 is **15% faster** than CUDA=1 (36.71 vs 31.83 tok/s). This is because NV's lower kernel dispatch overhead means the GPU pipeline stays fuller when kernel configs are optimized — better kernels amplify the NV advantage.

## Root Cause

The matvec heuristic in `tinygrad/codegen/opt/heuristic.py` was **never triggering** for LLM matmul kernels. It expected `REDUCE(MUL(INDEX, INDEX))` but:

1. fp16 matmul with fp32 accumulation has `REDUCE(CAST(MUL(INDEX, INDEX)))` — CAST wrapping
2. Fused RMSNorm+matmul produces `REDUCE(MUL(MUL(x, norm_factor), weight))` — nested MULs

Without matvec, all matmul kernels fell through to the generic `GROUPTOP(16)` heuristic, which produced:
- Only **16 threads per block** (half a warp!)
- **Non-coalesced memory access** (threads 262144 elements apart)
- **No vectorization** (scalar half loads)
- **Serial shared-memory reduction**

The result: matmul kernels achieved only **1.5 GB/s** of the ~100+ GB/s available memory bandwidth.

## Fix (commit `2439279b1`)

Two changes to `hand_coded_optimizations()`:

1. **CAST unwrap**: `if mulop.op is Ops.CAST: mulop = mulop.src[0]`
2. **Recursive INDEX finder**: Instead of `MUL(INDEX, INDEX)`, recursively find INDEX nodes through MUL/CAST chains up to depth 3. This handles fused operations like RMSNorm→matmul.

After fix with `MV_THREADS_PER_ROW=32`:
- Matmul kernels now use **128 threads** (32 GROUP + 4 LOCAL)
- Coalesced memory access via GROUP reduction
- 4-way UPCAST for ILP
- Bandwidth: **29-47 GB/s** (up from 1.5 GB/s)

## MV Parameter Tuning

| MV_TPR | BS | RPT | tok/s | notes |
|--------|-----|-----|-------|-------|
| 8 | 4 | 4 | 18.34 | defaults |
| 32 | 2 | 4 | 29.03 | |
| **32** | **4** | **4** | **29.43** | **best** |
| 32 | 4 | 2 | 28.20 | |
| 32 | 4 | 8 | 22.10 | too much upcast |
| 32 | 8 | 4 | 24.24 | |
| 64 | 4 | 4 | 25.93 | |

**Optimal for Orin**: `MV_THREADS_PER_ROW=32 MV_BLOCKSIZE=4 MV_ROWS_PER_THREAD=4`

## How to run

```bash
cd /home/agent/jetpack-nixos/examples/tinygrad

# NV backend (direct kernel interface, no CUDA driver)
nix develop -c bash -c 'NV=1 MV_THREADS_PER_ROW=32 python3 -c "
from tinygrad.apps.llm import Transformer, models
from tinygrad import Tensor
from tinygrad.helpers import fetch
import time
gguf = fetch(models[\"llama3.2:1b\"], \"Llama-3.2-1B-Instruct-Q6_K.gguf\", subdir=\"llama3-1b-instruct\")
model, kv = Transformer.from_gguf(Tensor(gguf))
tokens = [128000, 9906]
times = []
for i, tok in enumerate(model.generate(tokens)):
    times.append(time.time())
    if i >= 15: break
dts = [times[i]-times[i-1] for i in range(5, len(times))]
avg = sum(dts)/len(dts)
print(f\"Steady-state: {avg*1000:.1f}ms = {1/avg:.2f} tok/s\")
"'

# CUDA backend
nix develop -c bash -c 'CUDA=1 MV_THREADS_PER_ROW=32 python3 ...'

# llama.cpp baseline
cd ../llama-cpp-orin && nix develop -c llama-bench -m ~/.cache/tinygrad/llama3-1b-instruct/Llama-3.2-1B-Instruct-Q6_K.gguf -p 128 -n 128
```

## Kernel-Level Profiling (with matvec fix, MV_TPR=32)

Per-token decode breakdown (230 kernels, 16 transformer layers):

| Kernel | Count | Total (µs) | % | BW (GB/s) | Role |
|--------|------:|----------:|----:|----------:|------|
| `r_512_32_4_4_64` | 30 | 20,560 | 37.4% | 49 | gate/up proj (2048→8192) |
| `r_128_32_4_4_256` | 15 | 10,924 | 19.9% | 46 | down proj (8192→2048) |
| `r_8016_32_4_4_64` | 1 | 10,110 | 18.4% | 52 | lm_head (2048→128256, reads ~500MB!) |
| `r_128_32_4_4_64` | 30 | 5,931 | 10.8% | 43 | Q/O proj (2048→2048) |
| `r_2_8_4_32_4_4_64` | 14 | 1,608 | 2.9% | 57 | QKV+RoPE fused |
| `r_16_128` | 45 | 1,375 | 2.5% | ~0 | RMSNorm reductions |
| attention kernels | 60 | 1,669 | 3.0% | ~2-15 | softmax, attn score/value |
| other | 35 | 1,742 | 3.2% | varies | embeddings, KV cache, final |

**Note:** DEBUG=2 inflates total from 33.4ms → 54.9ms due to per-kernel sync overhead. Actual bandwidth utilization is ~48% of the ~102 GB/s effective LPDDR5 bandwidth.

## Remaining optimization opportunities

- **On-the-fly Q6_K dequant** (weights are currently expanded to fp16 in memory, 2.44× bandwidth waste). Fusing dequant into matmul would read 0.97 GB instead of 2.5 GB per token — theoretical 2.5× speedup on matvec kernels.
- **lm_head is 500MB at fp16** — 18.4% of compute. Keeping it quantized would save ~300MB reads per token.
- **MV_THREADS_PER_ROW default**: Should be 32 for SM 8.7+ (warp-width GROUP reduction). Currently defaults to 8.
- **Tensor core utilization** for batch>1 / prefill workloads
- **Wider BEAM search** (JITBEAM=8 or higher) may find additional wins at the cost of longer first-run compilation
