#!/usr/bin/env python3
"""
Radix KV-cache benchmark — Llama-3.2-1B-Instruct on Jetson Orin AGX (NV=1).

Demonstrates and measures the SGLang-style radix prefix-tree KV-cache sharing
strategy for multi-agent LLM inference running on the TinyGrad NV backend.

Theory
------
In multi-agent serving, N agents share a long system prompt (L tokens) then
each append a short private question (P tokens).  Without a KV cache:
  • Cost per agent: O((L+P)²) attention + O(L+P) weight loads (per layer)
  • Total:          N × that

With the radix tree:
  • Agent 1:        full O((L+P)²) prefill → snapshot stored at depth L
  • Agents 2..N:    restore snapshot, then O(P²) + O(P×L) prefill for P tokens
  • Savings:        (N-1) × O(L²) attention + (N-1) × L × weight-load steps

On Jetson Orin AGX, each token step is dominated by loading ~2 GB of model
weights from LPDDR5X (128 GB/s peak).  Skipping 130 shared-prefix steps per
agent saves ~260 GB of memory reads per agent — a directly measurable speedup.

Benchmark outputs
-----------------
  • Prefill time per agent   — baseline vs radix (ms)
  • Total prefill speedup    — N-agent combined
  • Snapshot payload size    — bytes stored in the radix tree for the shared KV
  • KV memory reduction      — N × naive vs 1 × shared snapshot
  • Correctness              — first generated token identical (greedy, temp=0)

Usage (from repo root, inside tinygrad nix dev shell)
------------------------------------------------------
    cd examples/tinygrad && nix develop
    # Then inside the shell:
    NV=1 python3 test_radix_llama3.py
    NV=1 python3 test_radix_llama3.py --agents 5 --max_ctx 512
    NV=1 python3 test_radix_llama3.py --out results_radix.json
"""
from __future__ import annotations

import sys, time, heapq, threading, argparse, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── PYTHONPATH: resolve tinygrad submodule from any CWD ──────────────────────
_here = Path(__file__).resolve()
_repo = _here.parents[2]                           # agx-orin-dev-kit/
_tg   = _repo / "external" / "tinygrad"            # submodule
_tgex = _tg   / "examples"                         # llama3.py lives here
for _p in [str(_tg), str(_tgex)]:
  if _p not in sys.path:
    sys.path.insert(0, _p)

# ── tinygrad imports ──────────────────────────────────────────────────────────
from tinygrad import Tensor, dtypes, Device, GlobalCounters  # noqa: E402
from tinygrad.helpers import fetch, colored                   # noqa: E402

# ── Llama-3 loader (from external/tinygrad/examples/llama3.py) ───────────────
# build_transformer handles GGUF loading, weight remapping and model init.
# Tokenizer wraps tiktoken with Llama-3 special tokens.
from llama3 import Tokenizer, build_transformer              # type: ignore


# =============================================================================
# Inline SGLang-style Radix KV Cache
# =============================================================================
# Self-contained: no dependency on examples/Qwen3-Coder-Next/radix_kv_cache.py
# This version stores LlamaKVSnap objects keyed on prefix token sequences.

@dataclass
class LlamaKVSnap:
  """Frozen KV-cache state at a prefix boundary.

  ``layers``: per-layer dict mapping layer index → Tensor of shape
  ``(2, B, prefix_len, n_kv_heads, head_dim)`` in fp16.
  The factor-2 first dim = [K, V].
  """
  prefix_len: int
  layers: Dict[int, Tensor]


class _N:
  """Radix tree node (internal)."""
  __slots__ = ("ch", "par", "tids", "snap", "ref", "ts")

  def __init__(self) -> None:
    self.ch:   Dict[int, "_N"]        = {}
    self.par:  Optional["_N"]         = None
    self.tids: List[int]              = []        # edge label
    self.snap: Optional[LlamaKVSnap] = None
    self.ref:  int                    = 0          # pin count
    self.ts:   float                  = time.monotonic()

  def __lt__(self, other: "_N") -> bool:           # for heapq LRU ordering
    return self.ts < other.ts


def _cpl(a: List[int], b: List[int]) -> int:
  """Length of the longest common prefix of lists a and b."""
  n = 0
  for x, y in zip(a, b):
    if x != y:
      break
    n += 1
  return n


class RadixKVCache:
  """SGLang-style radix prefix tree for multi-agent KV-cache sharing.

  Public API
  ----------
  match(tokens)           → (matched_len, LlamaKVSnap | None)
  insert(tokens, snap)    → None
  cached_tokens           → int (total edge-label tokens in tree)
  snap_bytes(snap)        → int (bytes used by one snapshot)
  """

  def __init__(self, budget: int = 16_384) -> None:
    self.budget = budget
    self._lock  = threading.Lock()
    self.root   = _N()
    self.root.ref = 1   # root is always pinned
    self._tot:  int = 0

  # ── Public ──────────────────────────────────────────────────────────────────

  def match(self, toks: List[int]) -> Tuple[int, Optional[LlamaKVSnap]]:
    """Walk the tree; return (longest_matched_len, snapshot_at_that_depth)."""
    with self._lock:
      node  = self.root
      pos, best_l, best_s = 0, 0, None
      now   = time.monotonic()
      while pos < len(toks):
        f = toks[pos]
        if f not in node.ch:
          break
        c = node.ch[f]
        n = _cpl(c.tids, toks[pos:])
        if n == 0:
          break
        c.ts = now
        if n < len(c.tids):
          break             # query ends inside an edge (no snapshot here)
        pos += n
        if c.snap is not None:
          best_l, best_s = pos, c.snap
        node = c
      return best_l, best_s

  def insert(self, toks: List[int], snap: LlamaKVSnap) -> None:
    """Insert tokens[:snap.prefix_len] with its snapshot."""
    with self._lock:
      self._ins(self.root, toks[: snap.prefix_len], snap)
      self._evict()

  @property
  def cached_tokens(self) -> int:
    return self._tot

  @staticmethod
  def snap_bytes(snap: LlamaKVSnap) -> int:
    """Bytes consumed by one snapshot (sum of all layer KV slices)."""
    return sum(v.nbytes() for v in snap.layers.values())

  # ── Internal ────────────────────────────────────────────────────────────────

  def _ins(self, node: _N, toks: List[int], snap: LlamaKVSnap) -> None:
    pos = 0
    while pos < len(toks):
      f   = toks[pos]
      rem = toks[pos:]

      if f not in node.ch:
        # New leaf for the entire remaining suffix
        lf        = _N()
        lf.par    = node
        lf.tids   = list(rem)
        lf.snap   = snap
        lf.ts     = time.monotonic()
        node.ch[f] = lf
        self._tot += len(rem)
        return

      c = node.ch[f]
      n = _cpl(c.tids, rem)

      if n == len(c.tids):
        # Full edge match — descend
        c.ts = time.monotonic()
        if c.snap is None:
          c.snap = snap
        pos += n
        node = c
        continue

      # Mid-edge split at position n
      #   Before:  node → [A B C D]           (child c)
      #   After:   node → [A B] → [C D]       (split_node → c)
      #                         → [X Y ...]   (new_leaf for diverging suffix)
      sn       = _N()
      sn.par   = node
      sn.tids  = c.tids[:n]
      sn.ts    = time.monotonic()

      c.tids   = c.tids[n:]   # trim child edge
      c.par    = sn
      sn.ch[c.tids[0]] = c
      node.ch[f] = sn

      div = rem[n:]
      if div:
        lf        = _N()
        lf.par    = sn
        lf.tids   = list(div)
        lf.snap   = snap
        lf.ts     = time.monotonic()
        sn.ch[div[0]] = lf
        self._tot += len(div)
      else:
        sn.snap = snap
      return

  def _evict(self) -> None:
    """LRU evict unpinned leaf nodes until total tokens ≤ budget."""
    if self._tot <= self.budget:
      return
    leaves: List[_N] = []
    self._collect_leaves(self.root, leaves)
    heapq.heapify(leaves)
    need = self._tot - self.budget
    freed = 0
    while freed < need and leaves:
      n = heapq.heappop(leaves)
      if n.ref > 0:
        continue
      freed     += len(n.tids)
      self._tot -= len(n.tids)
      if n.par and n.tids:
        n.par.ch.pop(n.tids[0], None)
      n.snap = None       # release Tensor refs
      # parent may now be an evictable leaf
      p = n.par
      if p and p is not self.root and not p.ch and p.ref == 0:
        heapq.heappush(leaves, p)

  def _collect_leaves(self, node: _N, out: List[_N]) -> None:
    if not node.ch and node is not self.root and node.ref == 0:
      out.append(node)
      return
    for c in node.ch.values():
      self._collect_leaves(c, out)


# =============================================================================
# KV-cache snapshot / restore  (llama Attention layout)
# =============================================================================
# tinygrad llama Attention stores:
#   self.cache_kv  shape: (2, B, max_context, n_kv_heads, head_dim)
#   index 0 = K,  index 1 = V
#   Filled in-place: cache_kv[:, :, start_pos:start_pos+seqlen, :, :].assign(...)

def take_snapshot(model, prefix_len: int) -> LlamaKVSnap:
  """Clone positions [0, prefix_len) from every attention layer's cache_kv.

  The clone is a .contiguous().realize() copy so the snapshot owns its buffer
  and is not aliased to the live KV cache.
  """
  layers: Dict[int, Tensor] = {}
  for i, layer in enumerate(model.layers):
    attn = layer.attention
    if hasattr(attn, "cache_kv"):
      # cache_kv: (2, B, max_ctx, n_kv_heads, head_dim)
      sl = attn.cache_kv[:, :, :prefix_len, :, :].contiguous()
      sl.realize()
      layers[i] = sl
  return LlamaKVSnap(prefix_len=prefix_len, layers=layers)


def restore_snapshot(model, snap: LlamaKVSnap) -> None:
  """Write snapshot back into each layer's cache_kv buffer in-place.

  Uses .assign().realize() which modifies the existing buffer object without
  reallocating.  This is critical for TinyJit correctness: the JIT compiles
  references to the buffer address, so the buffer must remain the same object.
  Stale data beyond snap.prefix_len is irrelevant because llama's attention
  reads cache_kv[..., :start_pos+seqlen, ...] — always bounded by start_pos.
  """
  pl = snap.prefix_len
  for i, layer in enumerate(model.layers):
    attn = layer.attention
    if i not in snap.layers:
      continue
    src = snap.layers[i]           # (2, B, pl, n_kv_heads, head_dim)
    if not hasattr(attn, "cache_kv"):
      # Lazily create cache_kv matching the snapshot's shape
      _, B, _, Hk, D = src.shape
      attn.cache_kv = Tensor.zeros(
        2, B, attn.max_context, Hk, D, dtype=src.dtype, device=src.device,
      ).contiguous().realize()
    attn.cache_kv[:, :, :pl, :, :].assign(src).realize()


# =============================================================================
# Chat-template helpers (Llama-3 instruct format)
# =============================================================================

def _role_toks(tok: Tokenizer, role: str) -> List[int]:
  BH  = tok.special_tokens["<|start_header_id|>"]
  EH  = tok.special_tokens["<|end_header_id|>"]
  return [BH] + tok.encode(role) + [EH] + tok.encode("\n\n")


def _msg_toks(tok: Tokenizer, role: str, content: str) -> List[int]:
  EOT = tok.special_tokens["<|eot_id|>"]
  return _role_toks(tok, role) + tok.encode(content.strip()) + [EOT]


# =============================================================================
# Benchmark corpus
# =============================================================================

SYSTEM_PROMPT = (
  "You are an expert AI assistant running on a Jetson Orin AGX 64GB via TinyGrad "
  "with the NV backend.  You have deep knowledge of CUDA, GPU architecture, memory "
  "hierarchies, and embedded AI deployment at the edge.  You prioritize concise, "
  "technically accurate responses.  When writing code, prefer Python with TinyGrad. "
  "Always consider Jetson's unified-memory architecture in your optimizations."
)

AGENT_QUERIES = [
  "What is the peak memory bandwidth of the Jetson Orin AGX?",
  "Explain how GQA reduces KV-cache memory compared to standard MHA.",
  "How does the NV=1 backend differ from CUDA=1 in TinyGrad?",
  "What is the Roofline model and how does it apply to LLM inference on Jetson?",
  "Describe how radix-tree prefix caching works in SGLang.",
]


# =============================================================================
# Token-by-token prefill / decode helpers
# =============================================================================

def prefill_range(model, toks: List[int], start: int, end: int, device) -> None:
  """Run model forward for toks[start], toks[start+1], … toks[end-1]."""
  for i in range(start, end):
    model(Tensor([[toks[i]]], device=device), i, 0.0, 0, 0.8, 0.0, 0.0).realize()


def decode_one(model, toks: List[int], pos: int, device) -> int:
  """Greedy-decode one token at position pos (temperature=0 → argmax)."""
  return int(
    model(Tensor([[toks[pos]]], device=device), pos, 0.0, 0, 0.8, 0.0, 0.0).item()
  )


def warmup_jit(model, device) -> None:
  """Run a few forward passes to trigger TinyJit compilation before timing."""
  print("  Warming up JIT… ", end="", flush=True)
  GlobalCounters.reset()
  # start_pos=0 → non-JIT path (creates cache_kv lazily)
  model(Tensor([[1]], device=device), 0, 0.0, 0, 0.8, 0.0, 0.0).realize()
  # start_pos ≠ 0 (shape 1×1) → TinyJit compilation
  for i in range(1, 6):
    model(Tensor([[i % 100 + 1]], device=device), i, 0.0, 0, 0.8, 0.0, 0.0).realize()
  GlobalCounters.reset()
  print("done")


# =============================================================================
# Main benchmark
# =============================================================================

def run_benchmark(
  model,
  tokenizer: Tokenizer,
  device: str,
  n_agents: int  = 5,
  seed:     int  = 42,
) -> dict:
  Tensor.manual_seed(seed)

  # ── Build shared-prefix token list (system turn only) ─────────────────────
  shared_toks: List[int] = (
    [tokenizer.bos_id]
    + _msg_toks(tokenizer, "system", SYSTEM_PROMPT)
  )
  shared_len = len(shared_toks)
  queries    = AGENT_QUERIES[:n_agents]

  print(f"\n{'═' * 70}")
  print(f"  Radix KV-Cache Benchmark — Llama-3.2-1B on {device}")
  print(f"  Shared-prefix length : {shared_len} tokens")
  print(f"  Agents               : {n_agents}")
  print(f"{'═' * 70}")

  warmup_jit(model, device)

  # ── Build per-agent full token lists once ─────────────────────────────────
  agent_toks: List[List[int]] = []
  for q in queries:
    toks = (
      list(shared_toks)
      + _msg_toks(tokenizer, "user", q)
      + _role_toks(tokenizer, "assistant")
    )
    agent_toks.append(toks)

  private_lens = [len(t) - shared_len for t in agent_toks]
  total_lens   = [len(t) for t in agent_toks]

  # ──────────────────────────────────────────────────────────────────────────
  # BASELINE: full prefill from position 0 for every agent
  # ──────────────────────────────────────────────────────────────────────────
  print(f"\n[ BASELINE — full prefill from token 0 for every agent ]\n")
  b_times:  List[float] = []
  b_first:  List[int]   = []

  for qi, (toks, plen, tlen) in enumerate(zip(agent_toks, private_lens, total_lens)):
    t0 = time.perf_counter()
    prefill_range(model, toks, 0, len(toks) - 1, device)
    elapsed = time.perf_counter() - t0
    ft = decode_one(model, toks, len(toks) - 1, device)
    b_times.append(elapsed)
    b_first.append(ft)
    print(
      f"  A{qi + 1}  {tlen:3d} tok  (+{plen:2d} private)  "
      f"prefill = {elapsed * 1e3:6.0f} ms  first_tok = {ft}"
    )

  b_total = sum(b_times)
  print(f"\n  TOTAL baseline prefill: {b_total * 1e3:.0f} ms\n")

  # ──────────────────────────────────────────────────────────────────────────
  # RADIX CACHE: snapshot after first miss, restore for subsequent agents
  # ──────────────────────────────────────────────────────────────────────────
  print(f"[ RADIX KV CACHE — snapshot on miss, restore on hit ]\n")
  cache = RadixKVCache(budget=32_768)
  r_times:  List[float] = []
  r_first:  List[int]   = []
  r_labels: List[str]   = []
  snap_ref:  Optional[LlamaKVSnap] = None

  for qi, (toks, plen, tlen) in enumerate(zip(agent_toks, private_lens, total_lens)):
    hit_len, snap = cache.match(toks)

    t0 = time.perf_counter()

    if snap is not None and hit_len >= shared_len:
      # ── Cache HIT ──────────────────────────────────────────────────────────
      # Restore positions 0..shared_len-1 from snapshot, then prefill private
      # suffix starting at shared_len.  The call to restore_snapshot uses
      # .assign().realize() which modifies the buffer in-place — the TinyJit
      # compiled kernel (referenced by buffer address) remains valid.
      restore_snapshot(model, snap)
      prefill_range(model, toks, shared_len, len(toks) - 1, device)
      elapsed = time.perf_counter() - t0
      label = "HIT "
    else:
      # ── Cache MISS ─────────────────────────────────────────────────────────
      # Full prefill from position 0 (overwrites existing cache_kv positions),
      # then capture a snapshot at the shared prefix boundary.
      prefill_range(model, toks, 0, len(toks) - 1, device)
      elapsed = time.perf_counter() - t0
      snap_ref = take_snapshot(model, shared_len)
      cache.insert(list(shared_toks), snap_ref)
      label = "MISS"

    ft = decode_one(model, toks, len(toks) - 1, device)
    match_sym = colored("✓", "green") if ft == b_first[qi] else colored("✗", "red")

    r_times.append(elapsed)
    r_first.append(ft)
    r_labels.append(label)
    print(
      f"  A{qi + 1}  [{label}]  {tlen:3d} tok  (+{plen:2d} private)  "
      f"prefill = {elapsed * 1e3:6.0f} ms  first_tok = {ft}  {match_sym}"
    )

  r_total    = sum(r_times)
  all_match  = (b_first == r_first)
  speedup    = b_total / r_total if r_total > 0 else float("inf")
  saved_tok  = (n_agents - 1) * shared_len

  print(f"\n  TOTAL radix-cache prefill: {r_total * 1e3:.0f} ms\n")

  # ── Snapshot memory ────────────────────────────────────────────────────────
  if snap_ref is not None:
    snap_b = RadixKVCache.snap_bytes(snap_ref)
  else:
    # Estimate: n_layers × 2 × B × prefix_len × n_kv_heads × head_dim × 2B
    n_layers   = len(model.layers)
    first_attn = model.layers[0].attention
    snap_b = (
      n_layers
      * 2 * 1 * shared_len
      * first_attn.n_kv_heads * first_attn.head_dim
      * 2                                            # fp16
    )

  snap_mb        = snap_b / 1e6
  baseline_kv_mb = n_agents * snap_mb
  reduction      = baseline_kv_mb / snap_mb if snap_mb > 0 else 0

  # ── Summary table ─────────────────────────────────────────────────────────
  SPD = colored(f"{speedup:.2f}×", "green" if speedup >= 2.0 else "yellow")
  COR = colored("PASS", "green") if all_match else colored("FAIL", "red")

  print(f"{'─' * 70}")
  print(f"  {'Metric':<42} {'Baseline':>10}   {'Radix':>9}")
  print(f"{'─' * 70}")
  print(f"  {'Total prefill time':<42} {b_total * 1e3:>10.0f}ms  {r_total * 1e3:>8.0f}ms")
  for i in range(n_agents):
    lbl = f"    Agent {i + 1}  ({total_lens[i]:3d} tok, +{private_lens[i]:2d} private)"
    print(f"  {lbl:<42} {b_times[i] * 1e3:>10.0f}ms  {r_times[i] * 1e3:>8.0f}ms")
  print(f"{'─' * 70}")
  print(f"  {'Speedup (total prefill time)':<42} {'':>10}   {SPD}")
  print(f"  {'Shared-prefix compute steps avoided':<42} {'':>10}   {saved_tok:>7d} tok")
  print(f"  {'Snapshot in radix tree (shared KV)':<42} {'':>10}   {snap_mb:>7.1f} MB")
  print(f"  {'KV state: N × naive vs 1 × shared':<42} {baseline_kv_mb:>8.1f} MB  {snap_mb:>7.1f} MB")
  print(f"  {'KV memory reduction factor':<42} {'':>10}   {reduction:.1f}×")
  print(f"  {'Correctness (all first tokens match)':<42} {'':>10}   {COR}")
  print(f"{'═' * 70}\n")

  return {
    "device":              device,
    "n_agents":            n_agents,
    "shared_prefix_tokens": shared_len,
    "baseline_total_ms":   round(b_total * 1e3, 1),
    "radix_total_ms":      round(r_total * 1e3, 1),
    "speedup":             round(speedup, 3),
    "saved_shared_tokens": saved_tok,
    "snapshot_mb":         round(snap_mb, 2),
    "kv_mem_reduction_x":  round(reduction, 1),
    "correctness_pass":    all_match,
    "per_agent": [
      {
        "query":       queries[i],
        "total_tok":   total_lens[i],
        "private_tok": private_lens[i],
        "baseline_ms": round(b_times[i] * 1e3, 1),
        "radix_ms":    round(r_times[i] * 1e3, 1),
        "label":       r_labels[i].strip(),
        "tok_match":   b_first[i] == r_first[i],
      }
      for i in range(n_agents)
    ],
  }


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Radix KV-cache benchmark — Llama-3.2-1B on Jetson Orin (NV=1)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--agents",  type=int,  default=5,
                      help="Number of simulated agents to benchmark")
  parser.add_argument("--max_ctx", type=int,  default=512,
                      help="KV cache context size in tokens (keep ≤ 512 to fit in memory)")
  parser.add_argument("--model",   type=Path, default=None,
                      help="Path to GGUF (auto-downloaded to ~/.cache/tinygrad if absent)")
  parser.add_argument("--out",     type=Path, default=None,
                      help="Write JSON results to this file")
  args = parser.parse_args()

  device = Device.DEFAULT
  print(f"Backend : {device}")

  # ── Auto-download Llama-3.2-1B-Instruct Q6_K (~1.1 GB, one-time) ──────────
  tok_path  = fetch(
    "https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model",
    "tokenizer.model", subdir="llama3-1b-instruct",
  )
  gguf_path = fetch(
    "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/"
    "Llama-3.2-1B-Instruct-Q6_K.gguf",
    "Llama-3.2-1B-Instruct-Q6_K.gguf", subdir="llama3-1b-instruct",
  )
  model_path = args.model or gguf_path

  tokenizer = Tokenizer(str(tok_path))

  print(f"Loading  : Llama-3.2-1B-Instruct Q6_K from {model_path.name}")
  t0    = time.perf_counter()
  model = build_transformer(
    model_path,
    model_size  = "1B",
    device      = device,
    max_context = args.max_ctx,
  )
  print(f"Loaded in {time.perf_counter() - t0:.1f}s")

  results = run_benchmark(
    model, tokenizer, device,
    n_agents = min(args.agents, len(AGENT_QUERIES)),
  )

  if args.out:
    args.out.write_text(json.dumps(results, indent=2))
    print(f"Results written to {args.out}")
  else:
    print(json.dumps(results, indent=2))
