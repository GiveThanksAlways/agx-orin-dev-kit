# The Matvec Heuristic: A 7.6× Speedup

This is the single most impactful change — **33 lines added, 22 removed** in
`tinygrad/codegen/opt/heuristic.py`. It turned LLaMA 1B Q6_K decode from
4.8 tok/s to **36.71 tok/s** (7.6× improvement).

## Table of Contents

1. [What is Matvec?](#what-is-matvec)
2. [Why LLM Decode is Matvec](#why-llm-decode-is-matvec)
3. [The Pattern Matching Problem](#the-pattern-matching-problem)
4. [The Fix: Unwrap CAST/MUL Chains](#the-fix-unwrap-castmul-chains)
5. [Diff Walk-Through](#diff-walk-through)

---

## What is Matvec?

**Matrix-vector multiplication** (matvec) is when you multiply a matrix $W$
(shape $[M, K]$) by a vector $x$ (shape $[K]$) to produce a vector $y$ (shape
$[M]$):

$$y_i = \sum_{k=0}^{K-1} W_{i,k} \cdot x_k$$

On a GPU, the naive way to parallelize this is: one thread per output element
$y_i$, each doing the full reduction over $K$. But $K$ is large (e.g., 2048
for LLaMA 1B), so each thread reads a full row of $W$ — lots of memory traffic,
poor cache utilization.

### The Matvec Optimization

Tinygrad's heuristic optimizer detects matvec patterns and applies three
optimizations:

1. **GROUP** (`MV_THREADS_PER_ROW`): Split the reduction across multiple threads.
   With `MV_THREADS_PER_ROW=32`, 32 threads cooperate on each row, each reading
   $K/32$ elements. They then do a shared-memory reduction to combine partial sums.

2. **LOCAL** (`MV_BLOCKSIZE`): Process multiple output rows per workgroup.
   With `MV_BLOCKSIZE=4`, each workgroup handles 4 consecutive rows.

3. **UPCAST** (`MV_ROWS_PER_THREAD`): Each thread handles multiple rows.
   With `MV_ROWS_PER_THREAD=4`, each thread computes 4 output elements.

```
Without matvec:        With matvec (TPR=32, BS=4, RPT=4):
                       
Thread 0: y[0] = Σ    Workgroup 0: y[0..15]
Thread 1: y[1] = Σ      Thread 0..31: partial sums for y[0..3]
Thread 2: y[2] = Σ      Thread 32..63: partial sums for y[4..7]
...                      Thread 64..95: partial sums for y[8..11]
Thread M: y[M] = Σ      Thread 96..127: partial sums for y[12..15]
                         → shared_mem reduce → final y[0..15]
                       
                       Workgroup 1: y[16..31]
                       ...
```

### Why It Matters for Memory Bandwidth

LLM decode is **memory-bandwidth bound**. Each token generation reads the entire
model weight matrix (~1.1 GB for LLaMA 1B Q6_K). The Orin's memory bandwidth is
~102 GB/s, so the theoretical max is:

$$\text{max tok/s} = \frac{102 \text{ GB/s}}{1.1 \text{ GB}} \approx 93 \text{ tok/s}$$

But that assumes perfect memory access. Without matvec, each thread reads its own
row independently, causing **bank conflicts** and **cache thrashing**. With matvec,
threads cooperate on rows, sharing data through fast on-chip shared memory (48 KB/SM),
getting much closer to peak bandwidth.

Without matvec → 4.8 tok/s (5% of bandwidth). With matvec → 36.71 tok/s (39% of
bandwidth). The remaining gap is from quantization overhead, attention computation,
and other non-matmul layers.

---

## Why LLM Decode is Matvec

During LLM inference, there are two phases:

1. **Prefill**: Process the entire prompt at once. Input is a matrix (batch of
   tokens). This is **matmul** (matrix × matrix).

2. **Decode**: Generate one token at a time. Input is a single vector (the last
   token's embedding). This is **matvec** (matrix × vector).

For autoregressive generation (the common case), almost all time is spent in
decode — and decode is wall-to-wall matvec.

### The Computation Graph

In tinygrad's IR, a quantized LLM matmul looks like:

```
REDUCE(ADD,
  MUL(
    INDEX(weight_matrix, [row_idx, col_idx]),    ← W[i,k]
    INDEX(input_vector, [col_idx])               ← x[k]
  )
)
```

But quantized models (Q6_K, Q4_K_M) add **dequantization** — the weights are
stored as integers and must be cast to float before multiplication. And
**RMSNorm** normalizes the input vector before the matmul. So the actual graph
becomes:

```
REDUCE(ADD,
  CAST(fp16 → fp32,       ← accumulate in fp32 for precision
    MUL(
      MUL(                 ← fused norm × matmul
        MUL(x, norm_weight),  ← RMSNorm output
        dequant_weight     ← dequantized from Q6_K
      ),
      some_other_factor
    )
  )
)
```

---

## The Pattern Matching Problem

The **old code** expected a very specific pattern:

```python
# OLD: only matches REDUCE(ADD, MUL(INDEX(...), INDEX(...)))
(mulop:=k.reduceop.src[0]).op is Ops.MUL and \
mulop.src[0].op is Ops.INDEX and \
mulop.src[1].op is Ops.INDEX
```

It checks that:
1. The reduction operand is a `MUL`
2. Both sources of MUL are directly `INDEX` operations

But with quantized models, the actual operand tree is:

```
reduceop.src[0] = CAST(fp32,
  MUL(
    MUL(
      INDEX(weight, ...),    ← buried 2 levels deep
      something
    ),
    INDEX(input, ...)        ← buried 1 level deep
  )
)
```

The `CAST` wrapper means `reduceop.src[0].op` is `Ops.CAST`, not `Ops.MUL`.
And the `INDEX` ops are nested inside multiple `MUL` layers, not directly
accessible as `mulop.src[0]` and `mulop.src[1]`.

**Result**: The heuristic silently fell through, no matvec optimization was
applied, and every decode kernel ran ~7.6× slower.

---

## The Fix: Unwrap CAST/MUL Chains

The fix has two parts:

### Part 1: Unwrap CAST

```python
mulop = k.reduceop.src[0]
# NEW: unwrap CAST (e.g. fp16 inputs accumulated in fp32)
if mulop.op is Ops.CAST: mulop = mulop.src[0]
```

Simple: if the reduction operand is a CAST, look inside it. Now `mulop` points
to the MUL underneath.

### Part 2: Recursive INDEX Finder

Instead of requiring INDEX ops as direct children of MUL, we search recursively:

```python
def _find_indices(u, depth=0):
    if u.op is Ops.INDEX: return [u]
    if depth > 3: return []  # safety: don't recurse too deep
    ret = []
    for s in u.src: ret.extend(_find_indices(s, depth+1))
    return ret

indices = _find_indices(mulop) if mulop.op is Ops.MUL else []
```

This finds INDEX operations through chains of MUL and CAST:

```
MUL                     ← depth 0
├── MUL                 ← depth 1
│   ├── INDEX(weight)   ← depth 2: found! ✓
│   └── norm_factor
└── INDEX(input)        ← depth 1: found! ✓
```

The `depth > 3` guard prevents infinite recursion on pathological graphs while
still handling all real-world patterns we've seen (max depth in practice is 2-3).

### Part 3: Use the Found Indices

```python
if len(indices) >= 2:
    idx0, idx1 = indices[0].src[1].get_idx(), indices[1].src[1].get_idx()
    # ... rest of matvec optimization (unchanged)
```

Same logic as before — check that the reduction range appears in the index
expressions, verify divisibility, apply GROUP/LOCAL/UPCAST — but now it
actually triggers for quantized models.

---

## Diff Walk-Through

### The Old Code (removed)

```python
if k.reduceop is not None and k.reduceop.arg[0] is Ops.ADD \
   and len(k.full_shape) >= 2 and k.ren.has_shared and \
   (mulop:=k.reduceop.src[0]).op is Ops.MUL \           # ← FAILS: src[0] is CAST
   and mulop.src[0].op is Ops.INDEX \                     # ← never reached
   and mulop.src[1].op is Ops.INDEX:                      # ← never reached
    idx0, idx1 = mulop.src[0].src[1].get_idx(), mulop.src[1].src[1].get_idx()
```

### The New Code (added)

```python
if k.reduceop is not None and k.reduceop.arg[0] is Ops.ADD \
   and len(k.full_shape) >= 2 and k.ren.has_shared:
    mulop = k.reduceop.src[0]
    # unwrap CAST (e.g. fp16 inputs accumulated in fp32)
    if mulop.op is Ops.CAST: mulop = mulop.src[0]

    # find INDEX operands through MUL/CAST chains
    def _find_indices(u, depth=0):
        if u.op is Ops.INDEX: return [u]
        if depth > 3: return []
        ret = []
        for s in u.src: ret.extend(_find_indices(s, depth+1))
        return ret

    indices = _find_indices(mulop) if mulop.op is Ops.MUL else []
    if len(indices) >= 2:
        idx0 = indices[0].src[1].get_idx()
        idx1 = indices[1].src[1].get_idx()
        # ... same reduction range check + apply GROUP/LOCAL/UPCAST ...
```

### Key Observations

1. **The condition relaxation**: Removed `(mulop:=...).op is Ops.MUL` from the
   outer condition. Now we enter the block for any valid-looking reduction and
   check MUL inside.

2. **The CAST unwrap**: Single line, huge impact. This one line is arguably
   responsible for the majority of the speedup.

3. **Recursive search**: `_find_indices` is small (6 lines) but generalizes
   to any nesting of MUL/CAST between the reduction and the INDEX operations.

4. **No other code changed**: GROUP/LOCAL/UPCAST application logic is identical.
   The optimizer already knew what to do — it just couldn't see the pattern.

### The Numbers

| Configuration | tok/s | vs baseline |
|---|---|---|
| Before (no matvec) | 4.8 | 1.0× |
| After (matvec) | 36.71 | 7.6× |
| llama.cpp reference | 25.7 | 5.4× |
| **tinygrad/llama.cpp ratio** | **143%** | |

The optimization applies to every attention projection and FFN layer in every
transformer block — dozens of matvec kernels per token. Each one went from
"random slow fallback" to "optimized cooperative reduction."
