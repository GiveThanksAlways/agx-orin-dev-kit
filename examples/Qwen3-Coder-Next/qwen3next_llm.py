#!/usr/bin/env python3
"""
Qwen3-Coder-Next (qwen3next) — TinyGrad implementation
Architecture: Hybrid Gated Delta Net + Full GQA Attention, 512-expert MoE + shared expert
Pattern: [DeltaNet×3, Attention×1] × 12 = 48 layers
Params:  ~80B total, ~3B active per token

Loads directly from GGUF (MXFP4_MOE quantized). Expert weights use sparse
access: only the top-K selected experts (~40 MB) are dequantized per layer per
token, instead of all 512 experts (~2.1 GB). This keeps peak nvmap-pinned
memory under ~8 GB, fitting comfortably in Jetson Orin's 64 GB unified memory.

Supports CUDA=1 (PTX) and NV=1 (native Nvidia) backends on Jetson Orin AGX.

Usage:
  # Interactive chat (inside tinygrad nix dev shell)
  NV=1 python3 qwen3next_llm.py

  # Benchmark
  CUDA=1 python3 qwen3next_llm.py --benchmark 20

  # OpenAI-compatible server
  NV=1 python3 qwen3next_llm.py --serve

Reference: docs/qwen3next-architecture.md for full forward pass pseudocode.
"""
from __future__ import annotations
import sys, argparse, typing, re, unicodedata, json, uuid, time, functools, math, pathlib
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, dtypes
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, stderr_log, colored

# ─── Tokenizer (qwen2 preset from llm.py) ──────────────────────────────────────
class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="qwen2"):
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","olmo"): raise ValueError(f"Invalid tokenizer preset '{preset}'")
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}
    def ucat_range(pre: str): return "".join(re.escape(chr(cp)) for cp in range(0x323b0) if unicodedata.category(chr(cp)).startswith(pre))
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    self._split_to_word = re.compile("(?i:'s|'t|'re|'ve|'m|'ll|'d)|" + \
      f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+")
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")
    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.preset = preset
  @staticmethod
  def from_gguf_kv(kv:dict):
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    return SimpleTokenizer(dict(normal_tokens), dict(special_tokens), kv.get("tokenizer.ggml.pre", "qwen2"))
  def _encode_word(self, word:bytes) -> list[int]:
    if (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    while True:
      i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j]+parts[j+1], sys.maxsize), j) for j in range(len(parts)-1)])[1]
      if i == -1: break
      parts[i:i+2] = [parts[i] + parts[i+1]]
    try: return [self._normal_tokens[p] for p in parts]
    except KeyError: raise RuntimeError("token not found")
  def _encode_sentence(self, chunk:str) -> list[int]:
    return [tok for word in self._split_to_word.findall(chunk) for tok in self._encode_word(word.encode())]
  def encode(self, text:str) -> list[int]:
    tokens: list[int] = []
    pos = 0
    for match in self._split_to_sentence.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    return tokens + self._encode_sentence(text[pos:])
  def decode(self, ids:list[int]) -> str: return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def role(self, role:str): return self.encode("<|im_start|>" + role + "\n")
  def end_turn(self, eos_id:int): return [eos_id] + self.encode("\n")

# ─── RoPE (partial — only first rope_dim dims of head) ─────────────────────────
@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).contiguous()

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  """Apply RoPE to first rope_dim dims of x: (B,H,T,D). Passes remaining dims through."""
  half = freqs_cis.shape[-1] // 2   # rope_dim/2 pairs to rotate
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)  # each (1,1,T,half)
  x1, x2 = x[..., :half], x[..., half:2*half]
  x_rest = x[..., 2*half:] if x.shape[-1] > 2*half else None
  rotated = (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)
  return rotated.cat(x_rest, dim=-1) if x_rest is not None else rotated

# ─── Expert Weights (on-demand GGUF dequant) ──────────────────────────────────
class ExpertWeights:
  """MoE expert weights with on-demand GGUF dequantization.

  The packed weight tensor (num_experts, out, in) is ~285 MB MXFP4 packed,
  expanding to ~2.1 GB float32 after dequant (512 experts × 512 × 2048).
  With 48 layers × 3 matrices, that's ~307 GB — impossible on 64 GB Jetson.

  Solution: store GGUF byte coordinates and create dequant chains ON DEMAND
  during inference — only for the top-K selected experts per token.

  Loading: set_gguf_source() stores coordinates (instant, zero memory).
  Inference: _gguf_sparse_forward() reads ~544 KB per expert from GGUF,
  dequants to fp16, and runs the matmul. Only 10 experts × ~2 MB = 20 MB
  live at a time instead of 2.1 GB.

  An LRU cache (default 32 experts per matrix) avoids repeated dequant
  for frequently-used experts. MoE routing follows a power law, so after
  warmup the cache hit rate is typically 50-80%."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
    self._experts: list[Tensor]|None = None  # only used for pre-split fallback
    self._gguf_info: tuple|None = None       # set by set_gguf_source()
    self._cache: dict[int, Tensor] = {}      # expert_id → realized Tensor
    self._cache_order: list[int] = []         # LRU eviction order
    self._cache_max: int = 32                 # max cached experts per matrix

  def set_gguf_source(self, disk_tensor:Tensor, data_start:int, tensor_off:int,
                      ggml_type:int, shape:tuple, use_half:int):
    """Store GGUF coordinates for on-demand per-expert dequant.
    This is instant — no data is read or processed until forward pass."""
    from tinygrad.nn.state import ggml_data_to_tensor as _dequant
    n_experts, out_dim, in_dim = shape
    elements = out_dim * in_dim
    raw_bytes = _ggml_raw_bytes(elements, ggml_type)
    self._gguf_info = (disk_tensor, data_start, tensor_off, ggml_type,
                       n_experts, out_dim, in_dim, elements, raw_bytes, use_half)

  def _get_expert(self, e:int) -> Tensor:
    """Get a single expert weight matrix, from cache or fresh GGUF dequant."""
    if e in self._cache:
      # Move to end of LRU order
      self._cache_order.remove(e)
      self._cache_order.append(e)
      return self._cache[e]

    # On-demand dequant from GGUF
    disk_tensor, data_start, tensor_off, ggml_type, \
      n_experts, out_dim, in_dim, elements, raw_bytes, use_half = self._gguf_info
    from tinygrad.nn.state import ggml_data_to_tensor
    start = data_start + tensor_off + e * raw_bytes
    raw = disk_tensor[start : start + raw_bytes].to(None)  # ~544 KB DISK → NV
    t = ggml_data_to_tensor(raw, elements, ggml_type).reshape(out_dim, in_dim)
    if use_half: t = t.cast(dtypes.float16)
    t = t.contiguous()
    t.realize()  # realize now so we cache the concrete NV buffer (~2 MB)

    # LRU eviction
    if len(self._cache) >= self._cache_max:
      evict_id = self._cache_order.pop(0)
      del self._cache[evict_id]
    self._cache[e] = t
    self._cache_order.append(e)
    return t

  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    if self._gguf_info is not None:
      return self._gguf_sparse_forward(sel, x)
    if self._experts is not None:
      return self._sparse_forward(sel, x)
    # Dense fallback (only used before GGUF source is set)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).squeeze(-2)

  def _gguf_sparse_forward(self, sel:Tensor, x:Tensor) -> Tensor:
    """On-demand dequant: read selected experts from GGUF → matmul.
    sel: (B, T, K) expert indices, x: (B, T, 1, D) input.
    For single-token gen: B=1, T=1, K=10 → 10× ~544 KB reads, ~20 MB fp16."""
    try:
      sel_np = sel.numpy()
    except Exception:
      sel_np = sel.to("CPU").numpy()
    unique_ids = sorted(set(int(e) for e in sel_np.flatten()))

    # Get experts from cache or dequant on demand
    expert_stack = Tensor.stack(*[self._get_expert(e) for e in unique_ids])

    if len(unique_ids) == sel_np.shape[-1] and all(unique_ids[i] == int(sel_np.flat[i]) for i in range(len(unique_ids))):
      selected = expert_stack.unsqueeze(0).unsqueeze(0)
    else:
      remap = {orig: new for new, orig in enumerate(unique_ids)}
      remapped_flat = [remap[int(e)] for e in sel_np.flatten()]
      remapped = Tensor(remapped_flat, dtype=dtypes.int32).reshape(sel_np.shape).to(x.device)
      selected = expert_stack[remapped]

    return (x.unsqueeze(-2) @ selected.transpose(-1, -2)).squeeze(-2)

  def _sparse_forward(self, sel:Tensor, x:Tensor) -> Tensor:
    """Fallback for pre-split experts (legacy path)."""
    try:
      sel_np = sel.numpy()
    except Exception:
      sel_np = sel.to("CPU").numpy()
    unique_ids = sorted(set(int(e) for e in sel_np.flatten()))
    expert_stack = Tensor.stack(*[self._experts[e] for e in unique_ids])
    if len(unique_ids) == sel_np.shape[-1] and all(unique_ids[i] == int(sel_np.flat[i]) for i in range(len(unique_ids))):
      selected = expert_stack.unsqueeze(0).unsqueeze(0)
    else:
      remap = {orig: new for new, orig in enumerate(unique_ids)}
      remapped_flat = [remap[int(e)] for e in sel_np.flatten()]
      remapped = Tensor(remapped_flat, dtype=dtypes.int32).reshape(sel_np.shape).to(x.device)
      selected = expert_stack[remapped]
    return (x.unsqueeze(-2) @ selected.transpose(-1, -2)).squeeze(-2)

# ─── Delta Net Recurrent Block ─────────────────────────────────────────────────
class DeltaNetBlock:
  """Gated Delta Net — linear attention with learnable recurrent state matrix.
  State: [B, S_v, S_k, H_v] = [B, 128, 128, 32] matrix bank per layer.
  Conv state: [B, conv_k-1, conv_channels] = [B, 3, 8192]."""

  def __init__(self, dim:int, d_inner:int, n_k_heads:int, n_v_heads:int,
               head_k_dim:int, head_v_dim:int, conv_kernel:int, norm_eps:float,
               hidden_dim:int, num_experts:int, num_experts_per_tok:int):
    self.dim, self.d_inner = dim, d_inner           # 2048, 4096
    self.n_k_heads, self.n_v_heads = n_k_heads, n_v_heads  # 16, 32
    self.head_k_dim, self.head_v_dim = head_k_dim, head_v_dim  # 128, 128
    self.conv_kernel = conv_kernel                   # 4
    self.conv_channels = d_inner + 2 * n_k_heads * head_k_dim  # 8192
    self.heads_per_group = n_v_heads // n_k_heads    # 2
    self.ba_dim = 2 * self.heads_per_group * n_k_heads  # 64

    # Pre/Post norms
    self.attn_norm = nn.RMSNorm(dim, norm_eps)
    self.post_attention_norm = nn.RMSNorm(dim, norm_eps)

    # Delta Net projections
    self.attn_qkv = nn.Linear(dim, self.conv_channels, bias=False)  # →8192
    self.attn_gate = nn.Linear(dim, d_inner, bias=False)             # →4096 (Z gate)
    self.ssm_ba = nn.Linear(dim, self.ba_dim, bias=False)            # →64 (beta+alpha)

    # SSM parameters
    self.ssm_a = Tensor.zeros(n_v_heads)                             # [32]
    self.ssm_dt_bias = Tensor.zeros(n_v_heads)                       # [32]
    self.ssm_conv1d_weight = Tensor.zeros(conv_kernel, self.conv_channels)  # [4, 8192]
    self.ssm_norm = nn.RMSNorm(head_v_dim, norm_eps)                 # [128]
    self.ssm_out = nn.Linear(d_inner, dim, bias=False)               # 4096→2048

    # MoE FFN
    self.num_experts_per_tok = num_experts_per_tok
    self.ffn_gate_inp = nn.Linear(dim, num_experts, bias=False)
    self.ffn_gate_exps = ExpertWeights(num_experts, dim, hidden_dim)
    self.ffn_up_exps = ExpertWeights(num_experts, dim, hidden_dim)
    self.ffn_down_exps = ExpertWeights(num_experts, hidden_dim, dim)
    # Shared expert
    self.ffn_gate_shexp = nn.Linear(dim, hidden_dim, bias=False)
    self.ffn_up_shexp = nn.Linear(dim, hidden_dim, bias=False)
    self.ffn_down_shexp = nn.Linear(hidden_dim, dim, bias=False)
    self.ffn_gate_inp_shexp = Tensor.zeros(dim)  # scalar gate weight

  def init_state(self, B:int=1) -> tuple[Tensor, Tensor]:
    delta_state = Tensor.zeros(B, self.head_v_dim, self.head_k_dim, self.n_v_heads, dtype=dtypes.float32).contiguous()
    conv_state = Tensor.zeros(B, self.conv_kernel - 1, self.conv_channels, dtype=dtypes.float32).contiguous()
    return delta_state, conv_state

  def _delta_net(self, x:Tensor, delta_state:Tensor, conv_state:Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Single-token Gated Delta Net. x: (B,1,D)."""
    B = x.shape[0]
    S_k, S_v, H_k, H_v = self.head_k_dim, self.head_v_dim, self.n_k_heads, self.n_v_heads
    hpg = self.heads_per_group  # 2

    # Projections
    qkv = self.attn_qkv(x).reshape(B, self.conv_channels)       # (B, 8192)
    z = self.attn_gate(x)                                         # (B, 1, 4096)
    ba = self.ssm_ba(x).reshape(B, 2 * hpg, H_k)                # (B, 4, 16)

    # Beta-alpha split: first hpg rows are beta, last hpg rows are alpha
    beta_raw = ba[:, :hpg].reshape(B, H_v, 1)                    # (B, 32, 1)
    alpha = ba[:, hpg:].reshape(B, H_v)                           # (B, 32)

    # Decay gate: ssm_a is negative (log-space), alpha+bias gives dt, softplus makes positive
    gate = self.ssm_a * (alpha + self.ssm_dt_bias).softplus()     # (B, 32), negative

    # Causal 1D convolution: prepend conv_state, take last conv_kernel frames
    conv_in = conv_state.cat(qkv.unsqueeze(1), dim=1)            # (B, 4, 8192)
    new_conv = conv_in[:, 1:, :].contiguous()                     # (B, 3, 8192)
    conv_out = (conv_in * self.ssm_conv1d_weight).sum(axis=1).silu()  # (B, 8192)

    # Split into V, Q, K — V gets first d_inner dims, Q/K interleaved in rest
    v = conv_out[:, :self.d_inner].reshape(B, S_v, H_v)          # (B, 128, 32)
    qk = conv_out[:, self.d_inner:].reshape(B, S_k, H_k, 2)
    q_small = qk[:, :, :, 0]                                      # (B, 128, 16)
    k_small = qk[:, :, :, 1]                                      # (B, 128, 16)

    # Expand Q,K from n_k_heads→n_v_heads via repeat_interleave
    # [h0,h1,...,h15] → [h0,h0,h1,h1,...,h15,h15] for heads_per_group=2
    q = q_small.unsqueeze(-1).expand(B, S_k, H_k, hpg).reshape(B, S_k, H_v)
    k = k_small.unsqueeze(-1).expand(B, S_k, H_k, hpg).reshape(B, S_k, H_v)

    # L2 normalize Q, K (over head_dim axis=1), scale Q
    eps = 1e-6
    q = q / (q.square().sum(1, keepdim=True).sqrt() + eps) * (1.0 / math.sqrt(S_v))
    k = k / (k.square().sum(1, keepdim=True).sqrt() + eps)
    beta = beta_raw.sigmoid()                                     # (B, 32, 1)

    # --- Recurrent update on state [B, S_v, S_k, H_v] ---
    # 1. Decay
    g = gate.reshape(B, 1, 1, H_v).exp()                         # decay ∈ (0,1)
    state = delta_state * g

    # 2. Read at key k: matrix-vector product along S_k dim
    k_e = k.reshape(B, 1, S_k, H_v)                              # (B, 1, 128, 32)
    kv_mem = (state * k_e).sum(2, keepdim=True)                   # (B, S_v, 1, H_v)

    # 3. Delta rule: beta * (v - read)
    v_t = v.reshape(B, S_v, 1, H_v)                              # (B, 128, 1, 32)
    delta = beta.reshape(B, 1, 1, H_v) * (v_t - kv_mem)          # (B, 128, 1, 32)

    # 4. Write: rank-1 outer product update
    state = state + k_e * delta                                    # (B, S_v, S_k, H_v)

    # 5. Query: read at key q
    q_e = q.reshape(B, 1, S_k, H_v)                              # (B, 1, 128, 32)
    output = (state * q_e).sum(2)                                  # (B, S_v, H_v)

    # Gated per-head norm + output projection
    output = self.ssm_norm(output.transpose(1, 2))                # (B, H_v, S_v) → normed [128]
    output = output.reshape(B, 1, self.d_inner) * z.silu()        # gate with silu(Z)
    return self.ssm_out(output), state.contiguous(), new_conv

  def _moe_ffn(self, h:Tensor) -> Tensor:
    """MoE FFN with shared expert and scalar sigmoid gate."""
    h_norm = self.post_attention_norm(h)
    x = h_norm.unsqueeze(2)                                       # (B, T, 1, D)

    # Routed experts (top-k selection)
    probs, sel = self.ffn_gate_inp(h_norm).softmax(-1).topk(self.num_experts_per_tok)
    x_down = self.ffn_down_exps(sel, self.ffn_gate_exps(sel, x).silu() * self.ffn_up_exps(sel, x))
    moe_out = (x_down * probs.unsqueeze(-1)).sum(axis=2)

    # Shared expert (SwiGLU + scalar sigmoid gate)
    shexp = self.ffn_down_shexp(self.ffn_gate_shexp(h_norm).silu() * self.ffn_up_shexp(h_norm))
    sg = (h_norm * self.ffn_gate_inp_shexp).sum(-1, keepdim=True).sigmoid()
    return h + moe_out + shexp * sg

  def __call__(self, x:Tensor, ds:Tensor, cs:Tensor) -> tuple[Tensor, Tensor, Tensor]:
    h = self.attn_norm(x)
    attn_out, new_ds, new_cs = self._delta_net(h, ds, cs)
    return self._moe_ffn(x + attn_out).contiguous(), new_ds, new_cs

# ─── Full Attention Block ──────────────────────────────────────────────────────
class FullAttentionBlock:
  """Standard GQA attention with sigmoid gating + MoE FFN. Uses KV cache."""
  def __init__(self, dim:int, n_heads:int, n_kv_heads:int, head_dim:int,
               rope_theta:float, rope_dim:int, norm_eps:float, max_context:int,
               hidden_dim:int, num_experts:int, num_experts_per_tok:int):
    self.dim, self.n_heads, self.n_kv_heads = dim, n_heads, n_kv_heads
    self.head_dim, self.rope_theta, self.rope_dim = head_dim, rope_theta, rope_dim
    self.max_context = max_context

    # Norms
    self.attn_norm = nn.RMSNorm(dim, norm_eps)
    self.post_attention_norm = nn.RMSNorm(dim, norm_eps)

    # Attention: Q projection includes gate (n_heads * head_dim * 2)
    self.attn_q = nn.Linear(dim, n_heads * head_dim * 2, bias=False)  # →8192
    self.attn_k = nn.Linear(dim, n_kv_heads * head_dim, bias=False)   # →512
    self.attn_v = nn.Linear(dim, n_kv_heads * head_dim, bias=False)   # →512
    self.attn_output = nn.Linear(n_heads * head_dim, dim, bias=False)  # 4096→2048
    self.attn_q_norm = nn.RMSNorm(head_dim, norm_eps)
    self.attn_k_norm = nn.RMSNorm(head_dim, norm_eps)

    # MoE FFN (same structure as DeltaNetBlock)
    self.num_experts_per_tok = num_experts_per_tok
    self.ffn_gate_inp = nn.Linear(dim, num_experts, bias=False)
    self.ffn_gate_exps = ExpertWeights(num_experts, dim, hidden_dim)
    self.ffn_up_exps = ExpertWeights(num_experts, dim, hidden_dim)
    self.ffn_down_exps = ExpertWeights(num_experts, hidden_dim, dim)
    self.ffn_gate_shexp = nn.Linear(dim, hidden_dim, bias=False)
    self.ffn_up_shexp = nn.Linear(dim, hidden_dim, bias=False)
    self.ffn_down_shexp = nn.Linear(hidden_dim, dim, bias=False)
    self.ffn_gate_inp_shexp = Tensor.zeros(dim)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    h = self.attn_norm(x)

    # Joint Q+gate → split per head
    qg = self.attn_q(h).reshape(B, T, self.n_heads, self.head_dim * 2)
    q, gate = qg[..., :self.head_dim], qg[..., self.head_dim:]
    k = self.attn_k(h).reshape(B, T, self.n_kv_heads, self.head_dim)
    v = self.attn_v(h).reshape(B, T, self.n_kv_heads, self.head_dim)

    # Per-head QK norm
    q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    q, k, v, gate = (t.transpose(1, 2) for t in (q, k, v, gate))  # →(B,H,T,D)

    # Partial RoPE (first rope_dim dims only)
    fc = precompute_freqs_cis(self.rope_dim, self.max_context, self.rope_theta)[start_pos:start_pos+T]
    q, k = apply_rope(q, fc), apply_rope(k, fc)

    # KV cache
    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.zeros(2, B, self.n_kv_heads, self.max_context, self.head_dim,
                                   dtype=k.dtype, device=k.device).contiguous().realize()
    self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v)).realize()
    k_full = self.cache_kv[0, :, :, :start_pos+T, :]
    v_full = self.cache_kv[1, :, :, :start_pos+T, :]

    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(int(start_pos)+1) if T > 1 else None
    attn = q.scaled_dot_product_attention(k_full, v_full, attn_mask=mask, enable_gqa=True)

    # Sigmoid gating
    attn = (attn * gate.sigmoid()).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(attn)

  def _moe_ffn(self, h:Tensor) -> Tensor:
    h_norm = self.post_attention_norm(h)
    x = h_norm.unsqueeze(2)
    probs, sel = self.ffn_gate_inp(h_norm).softmax(-1).topk(self.num_experts_per_tok)
    x_down = self.ffn_down_exps(sel, self.ffn_gate_exps(sel, x).silu() * self.ffn_up_exps(sel, x))
    moe_out = (x_down * probs.unsqueeze(-1)).sum(axis=2)
    shexp = self.ffn_down_shexp(self.ffn_gate_shexp(h_norm).silu() * self.ffn_up_shexp(h_norm))
    sg = (h_norm * self.ffn_gate_inp_shexp).sum(-1, keepdim=True).sigmoid()
    return h + moe_out + shexp * sg

  def __call__(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    x = x + self._attention(x, start_pos)
    return self._moe_ffn(x).contiguous()

# ─── Qwen3Next Transformer ────────────────────────────────────────────────────
class Qwen3NextTransformer:
  """Hybrid Delta Net + Full Attention with MoE.
  Layers 0..N-1 with full_attn_interval=4 → [delta, delta, delta, attn] repeating."""

  def __init__(self, *, num_blocks:int, dim:int, hidden_dim:int, n_heads:int, n_kv_heads:int,
               norm_eps:float, vocab_size:int, head_dim:int, rope_theta:float, rope_dim:int,
               max_context:int, num_experts:int, num_experts_per_tok:int,
               d_inner:int, n_k_heads:int, n_v_heads:int, head_k_dim:int, head_v_dim:int,
               conv_kernel:int, full_attn_interval:int):
    self.num_blocks = num_blocks
    self.max_context = max_context
    self.full_attn_interval = full_attn_interval

    self.token_embd = nn.Embedding(vocab_size, dim)
    self.output_norm = nn.RMSNorm(dim, norm_eps)
    self.output = nn.Linear(dim, vocab_size, bias=False)

    # Build hybrid layer lists
    self.delta_blocks: list[DeltaNetBlock] = []
    self.attn_blocks: list[FullAttentionBlock] = []
    self.layer_type: list[str] = []       # "delta" | "attn" per layer index
    self.delta_idx: list[int] = []        # layer → delta_blocks index (-1 if attn)
    self.attn_idx: list[int] = []         # layer → attn_blocks index (-1 if delta)

    for i in range(num_blocks):
      is_rec = (i % full_attn_interval) != (full_attn_interval - 1)
      if is_rec:
        self.layer_type.append("delta")
        self.delta_idx.append(len(self.delta_blocks))
        self.attn_idx.append(-1)
        self.delta_blocks.append(DeltaNetBlock(dim, d_inner, n_k_heads, n_v_heads,
          head_k_dim, head_v_dim, conv_kernel, norm_eps, hidden_dim, num_experts, num_experts_per_tok))
      else:
        self.layer_type.append("attn")
        self.delta_idx.append(-1)
        self.attn_idx.append(len(self.attn_blocks))
        self.attn_blocks.append(FullAttentionBlock(dim, n_heads, n_kv_heads, head_dim,
          rope_theta, rope_dim, norm_eps, max_context, hidden_dim, num_experts, num_experts_per_tok))

    self._delta_states: list[tuple[Tensor, Tensor]] | None = None
    if DEBUG >= 1:
      print(f"  {len(self.delta_blocks)} delta + {len(self.attn_blocks)} attn = {num_blocks} layers")

  def _init_states(self, B:int=1):
    self._delta_states = [blk.init_state(B) for blk in self.delta_blocks]

  def forward(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    if self._delta_states is None: self._init_states(tokens.shape[0])
    x = self.token_embd(tokens)
    for i in range(self.num_blocks):
      if self.layer_type[i] == "delta":
        di = self.delta_idx[i]
        x, ds, cs = self.delta_blocks[di](x, self._delta_states[di][0], self._delta_states[di][1])
        self._delta_states[di] = (ds, cs)
      else:
        x = self.attn_blocks[self.attn_idx[i]](x, start_pos)
    return self.output(self.output_norm(x))[:, -1, :].softmax(-1, dtype="float").argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp=0) -> Tensor:
    return self.forward(tokens, start_pos)

  @staticmethod
  def from_gguf(gguf_path:str|pathlib.Path, max_context:int|None=None) -> tuple[Qwen3NextTransformer, dict]:
    """Load from GGUF with per-tensor device migration for large models.

    Why per-tensor: tinygrad's built-in gguf_load does gguf.to(None) to move the
    entire file to a compute device, which works for ~16 GB GGUFs on desktop GPUs.
    But Qwen3-Coder-Next's 40.7 GB GGUF exceeds Tegra's nvmap single-allocation limit
    (mmap of 43 GB dmabuf → EINVAL). Our _gguf_load_chunked solves this by moving
    each tensor's raw bytes individually (~200-600 MB each, well within limits).

    Loading phases:
    1. Parse GGUF header on DISK (zero memory — TensorIO reads small chunks)
    2. Per-tensor: tight DISK slice → .to(Device.DEFAULT) → ggml_data_to_tensor
    3. Cast to fp16, load into model (all lazy at this point)
    4. Batch-realize non-expert params (~2 GB on NV)
    5. Expert weights stay lazy — split into per-expert slices for sparse dequant"""

    stderr_log("Loading GGUF metadata...\n")

    # Per-tensor GGUF loader: avoids the single 43 GB allocation that breaks Tegra.
    # Expert weights are deferred — their GGUF coordinates are returned for per-expert loading.
    gguf_p = pathlib.Path(gguf_path) if isinstance(gguf_path, str) else gguf_path
    kv, state_dict, disk_tensor, data_start, expert_tensor_info = _gguf_load_chunked(gguf_p)

    arch = kv['general.architecture']
    if arch != 'qwen3next': raise ValueError(f"Expected qwen3next, got '{arch}'")

    mc = kv.get(f'{arch}.context_length', 131072)
    max_context = min(max_context, mc) if max_context is not None else min(mc, 2048)  # cap for memory safety
    n_heads = kv[f'{arch}.attention.head_count']
    n_kv_heads = kv[f'{arch}.attention.head_count_kv']
    head_dim = kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads)
    d_inner = kv.get(f'{arch}.ssm.inner_size', 4096)
    n_k_heads = kv.get(f'{arch}.ssm.group_count', 16)
    n_v_heads = kv.get(f'{arch}.ssm.time_step_rank', 32)
    head_k_dim = kv.get(f'{arch}.ssm.state_size', 128)
    head_v_dim = d_inner // n_v_heads
    conv_kernel = kv.get(f'{arch}.ssm.conv_kernel', 4)
    full_attn_interval = kv.get(f'{arch}.full_attention_interval', 4)
    rope_dim = kv.get(f'{arch}.rope.dimension_count', 64)
    num_experts = kv.get(f'{arch}.expert_count', 0)
    num_experts_per_tok = kv.get(f'{arch}.expert_used_count', 0)
    hidden_dim = kv.get(f'{arch}.expert_feed_forward_length', kv[f'{arch}.feed_forward_length'])

    if 'output.weight' not in state_dict:
      state_dict['output.weight'] = state_dict['token_embd.weight']

    stderr_log(f"  arch={arch}, layers={kv[f'{arch}.block_count']}, dim={kv[f'{arch}.embedding_length']}, "
               f"experts={num_experts}×top{num_experts_per_tok}, max_ctx={max_context}\n")

    model = Qwen3NextTransformer(
      num_blocks=kv[f'{arch}.block_count'], dim=kv[f'{arch}.embedding_length'],
      hidden_dim=hidden_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
      norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim, rope_theta=kv[f'{arch}.rope.freq_base'], rope_dim=rope_dim,
      max_context=max_context, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok,
      d_inner=d_inner, n_k_heads=n_k_heads, n_v_heads=n_v_heads,
      head_k_dim=head_k_dim, head_v_dim=head_v_dim,
      conv_kernel=conv_kernel, full_attn_interval=full_attn_interval)

    # Remap GGUF flat names (blk.N.xxx) → model's delta_blocks/attn_blocks paths
    _remap_state_dict(model, state_dict)

    # GGUF type-30 (BF16) shared-expert scalar gate vectors: skip and use model defaults.
    # These are tiny (dim=2048) scalars that default to zeros, which makes the shared expert
    # gate sigmoid(0)=0.5 — a reasonable default until we debug BF16 dequant on NV.
    skipped_shexp_gate = 0
    for k in list(state_dict.keys()):
      if k.endswith("ffn_gate_inp_shexp"):
        del state_dict[k]
        skipped_shexp_gate += 1
    if skipped_shexp_gate:
      stderr_log(f"  skipping {skipped_shexp_gate} BF16 ffn_gate_inp_shexp tensors (using model defaults)\n")

    # Cast all weights to float16 (canonical tinygrad pattern — saves memory, no precision
    # loss since MXFP4 source is only 4-bit). Tensors are already on the compute device
    # from .to(None) above, so no manual .to() needed.
    if getenv("HALF", 1):
      state_dict = {k: v.cast(dtypes.float16) for k, v in state_dict.items()}

    # Load into model (lazy — nothing realized yet)
    stderr_log(f"  Loading state dict ({len(state_dict)} tensors)...\n")
    nn.state.load_state_dict(model, state_dict, strict=False, verbose=False, consume=True, realize=False)

    # Make all params contiguous (canonical pattern — prevents repacking on every forward pass)
    all_named = nn.state.get_state_dict(model)
    for s in all_named.values(): s.replace(s.contiguous())

    # Identify expert params — these stay lazy for sparse dequant during inference.
    # 512 experts × 48 layers × 3 matrices = ~307 GB if fully realized → impossible.
    # Instead, only top-K selected experts (~40 MB) are realized per forward pass.
    expert_param_ids = set()
    for blk_list in (model.delta_blocks, model.attn_blocks):
      for blk in blk_list:
        for attr_name in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps'):
          expert_param_ids.add(id(getattr(blk, attr_name).weight))

    non_expert = [v for v in all_named.values() if id(v) not in expert_param_ids]
    expert = [v for v in all_named.values() if id(v) in expert_param_ids]

    # Realize non-expert params (~4.7 GB fp16 on Jetson's unified memory).
    # Batch-realize lets tinygrad's memory planner reuse intermediate buffers.
    stderr_log(f"  Realizing {len(non_expert)} non-expert params...\n")
    Tensor.realize(*non_expert)
    stderr_log(f"  {len(expert)} expert weight placeholders left (will be loaded per-expert)\n")

    # ── Wire up on-demand GGUF dequant for expert weights ──
    # Instead of pre-creating 73,728 dequant chains (512 experts × 144 tensors),
    # we store GGUF byte coordinates in each ExpertWeights instance.
    # During inference, only the top-K selected experts are dequanted on-demand
    # from the GGUF file: 10 × ~544 KB reads per matrix per token.
    # With LRU caching, frequently-used experts are served from memory.
    #
    # This makes loading INSTANT (was 20+ minutes) and keeps memory usage tight:
    #   Per matrix: 10 selected × 2 MB fp16 + 32 cached × 2 MB = ~84 MB
    #   Total:      144 matrices × 84 MB ≈ 12 GB cache headroom (fits in 62 GB)
    stderr_log(f"  Wiring {len(expert_tensor_info)} expert tensors for on-demand GGUF dequant...\n")
    fai = model.full_attn_interval
    use_half = getenv("HALF", 1)
    for name, (off, typ, shape) in expert_tensor_info.items():
      parts = name.split(".")
      blk_i = int(parts[1])
      attr_name = parts[2]  # ffn_gate_exps, ffn_up_exps, or ffn_down_exps
      is_rec = (blk_i % fai) != (fai - 1)
      blk = model.delta_blocks[model.delta_idx[blk_i]] if is_rec else model.attn_blocks[model.attn_idx[blk_i]]
      ew = getattr(blk, attr_name)
      ew.set_gguf_source(disk_tensor, data_start, off, ggml_type=typ, shape=shape, use_half=use_half)
    stderr_log(f"  Expert weights ready (on-demand dequant, LRU cache=32 per matrix)\n")

    return model, kv

  def generate(self, tokens:list[int], start_pos=0):
    t = Tensor([tokens[start_pos:]], dtype="int32")
    while len(tokens) < self.max_context:
      t = self(t, start_pos)
      next_id = int(t.item())
      tokens.append(next_id)
      start_pos = len(tokens) - 1
      yield next_id


# ─── Per-Tensor GGUF Loader (avoids Tegra nvmap single-allocation limit) ─────
def _ggml_raw_bytes(n: int, ggml_type: int) -> int:
  """Compute exact raw byte count for n elements of a GGML quantized type.
  These match the t[:byte_count] slicing done inside ggml_data_to_tensor."""
  if ggml_type == 30: return 2 * n  # BF16: 2 bytes per element
  native_sizes = {0: 4, 1: 2, 16: 1, 17: 2, 18: 4}  # f32, f16, i8, i16, i32
  if ggml_type in native_sizes: return native_sizes[ggml_type] * n
  quant_params = {2: (32, 18), 3: (32, 20), 8: (32, 34), 12: (256, 144), 14: (256, 210), 39: (32, 17)}
  if ggml_type in quant_params:
    nel, nb = quant_params[ggml_type]
    return (n // nel) * nb
  raise ValueError(f"Unknown GGML type {ggml_type}")

def _gguf_load_chunked(gguf_path: pathlib.Path) -> tuple[dict, dict[str, Tensor], Tensor, int, dict]:
  """Parse GGUF from disk and load tensors individually to Device.DEFAULT.

  Returns: (kv_data, state_dict, disk_tensor, data_start, expert_tensor_info)

  Why per-tensor: tinygrad's built-in gguf_load does gguf.to(None) to move the
  entire file to a compute device — works for ~16 GB GGUFs but Qwen3-Coder-Next's
  40.7 GB GGUF exceeds Tegra's nvmap single-allocation limit (43 GB dmabuf → EINVAL).

  Expert weight tensors (ffn_gate_exps, ffn_up_exps, ffn_down_exps) are NOT loaded
  into state_dict. Their GGUF coordinates are returned in expert_tensor_info so the
  caller can create per-expert dequant chains from individual ~544 KB GGUF slices.

  This is CRITICAL for Jetson Orin 64GB memory:
    Old approach: dequant full (512, out, in) = 2.1 GB float32 per expert matrix
    New approach:  dequant top-10 experts = 10 × 4 MB = 40 MB per expert matrix
    Savings: 50× per matrix, 150× across gate/up/down per layer"""
  from tinygrad.nn.state import ggml_data_to_tensor, TensorIO
  from tinygrad.helpers import round_up, prod
  import struct, io

  # DISK-backed tensor: mmap'd by the kernel, zero physical memory cost
  disk_tensor = Tensor(gguf_path)

  # Reuse tinygrad's TensorIO for header parsing (reads small chunks from DISK)
  reader = io.BufferedReader(TensorIO(disk_tensor), 1_000_000)
  kv_data: dict = {}
  state_dict: dict[str, Tensor] = {}

  def read_unpack(fmt: str, nb: int): return struct.unpack(fmt, reader.read(nb))[0]
  def read_str(): return str(reader.read(read_uint64()), "utf-8")
  def read_arr():
    reader_fn, n = readers[read_int32()], read_uint64()  # noqa: F841 (shadows outer reader intentionally)
    return [reader_fn() for _ in range(n)]

  readers: dict = {8: read_str, 9: read_arr, **{t: functools.partial(read_unpack, "<"+f, nb) for t,f,nb in
    [(0,"c",1),(1,"b",1),(2,"H",2),(3,"h",2),(4,"I",4),(5,"i",4),(6,"f",4),(7,"?",1),(10,"Q",8),(11,"q",8),(12,"d",8)]}}
  read_uint32, read_int32, read_uint64, read_int64 = readers[4], readers[5], readers[10], readers[11]

  magic, version = reader.read(4), read_int32()
  n_tensors, n_kv = read_int64(), read_int64()
  if magic != b"GGUF" or version not in [2, 3]: raise ValueError("Invalid GGUF format!")

  for _ in range(n_kv):
    k, typ = read_str(), read_int32()
    kv_data[k] = readers[typ]()

  t_infos = [(read_str(), tuple(read_uint64() for _ in range(read_uint32())), read_int32(), read_uint64()) for _ in range(n_tensors)]
  alignment = kv_data.get("general.alignment", 32)
  data_start = round_up(reader.tell(), alignment)

  stderr_log(f"  GGUF header: {n_tensors} tensors, data_start=0x{data_start:x}\n")

  # Separate expert weight tensors from non-expert tensors.
  # Expert weights are NOT loaded here — they'll be split per-expert later.
  expert_tensor_info: dict[str, tuple[int, int, tuple]] = {}  # name → (off, typ, python_shape)
  _expert_suffixes = ('ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight')

  for name, dims, typ, off in t_infos:
    if any(name.endswith(s) for s in _expert_suffixes):
      # Record GGUF coordinates for per-expert loading later
      expert_tensor_info[name] = (off, typ, tuple(reversed(dims)))
      continue

    # Non-expert: DISK slice → .to(Device.DEFAULT) → dequant (all lazy)
    n = prod(dims)
    raw_bytes = _ggml_raw_bytes(n, typ)
    raw_slice = disk_tensor[data_start + off : data_start + off + raw_bytes]
    raw_on_device = raw_slice.to(None)
    state_dict[name] = ggml_data_to_tensor(raw_on_device, n, typ).reshape(*reversed(dims))

  stderr_log(f"  {len(state_dict)} non-expert tensors loaded, {len(expert_tensor_info)} expert tensors deferred\n")
  return kv_data, state_dict, disk_tensor, data_start, expert_tensor_info


def _remap_state_dict(model: Qwen3NextTransformer, sd: dict[str, Tensor]):
  """Remap GGUF flat names (blk.N.xxx) → model's delta_blocks/attn_blocks paths."""
  new: dict[str, Tensor] = {}
  fai = model.full_attn_interval

  for name, tensor in list(sd.items()):
    if not name.startswith("blk."):
      new[name] = tensor
      continue

    parts = name.split(".")
    blk_i = int(parts[1])
    rest = ".".join(parts[2:])
    is_rec = (blk_i % fai) != (fai - 1)
    pfx = f"delta_blocks.{model.delta_idx[blk_i]}" if is_rec else f"attn_blocks.{model.attn_idx[blk_i]}"

    # Map GGUF names that don't match model attribute names directly
    mapping = {
      "ssm_a": "ssm_a",
      "ssm_dt.bias": "ssm_dt_bias",
      "ssm_conv1d.weight": "ssm_conv1d_weight",
      "ffn_gate_inp_shexp.weight": "ffn_gate_inp_shexp",
    }
    if rest == "ssm_conv1d.weight":
      tensor = tensor.transpose(-1, -2)
    mapped = mapping.get(rest)
    new[f"{pfx}.{mapped}" if mapped else f"{pfx}.{rest}"] = tensor

  sd.clear()
  sd.update(new)

# ─── HTTP Server (OpenAI-compatible) ──────────────────────────────────────────
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler

CHAT_HTML = b'''<!DOCTYPE html><html><head><title>qwen3next chat</title><style>
  * { margin: 0 }
  body { background: #212121; color: #e3e3e3; font-family: system-ui;
         height: 100vh; display: flex; flex-direction: column }
  #chat { flex: 1; overflow-y: auto; padding: 20px }
  .msg { padding: 10px 16px; margin: 8px 0; white-space: pre-wrap; border-radius: 18px }
  .user { background: #2f2f2f; margin-left: auto; width: fit-content; max-width: 70% }
  #input { max-width: 768px; width: 100%; margin: 20px auto; padding: 14px 20px;
           background: #2f2f2f; color: inherit; font: inherit;
           border: none; outline: none; resize: none; border-radius: 24px; field-sizing: content }
</style></head><body><div id="chat"></div>
<textarea id="input" rows="1" placeholder="Ask anything"></textarea>
<script>
  input.onkeydown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }
  const msgs = [];
  async function send() {
    if (!input.value.trim()) return;
    msgs.push({role: 'user', content: input.value.trim()});
    chat.innerHTML += '<div class="msg user">' + input.value.trim().replace(/</g, '&lt;') + '</div>';
    input.value = '';
    const d = document.createElement('div'); d.className = 'msg'; chat.appendChild(d);
    const r = await fetch('/v1/chat/completions', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: 'qwen3next', messages: msgs, stream: true})});
    for (const rd = r.body.getReader(), dec = new TextDecoder();;) {
      const {done, value} = await rd.read();
      if (done) break;
      for (const ln of dec.decode(value).split('\\n'))
        if (ln.startsWith('data: ') && !ln.includes('[DONE]'))
          try { d.textContent += JSON.parse(ln.slice(6)).choices[0]?.delta?.content || '' } catch {}
      chat.scrollTop = chat.scrollHeight;
    }
    msgs.push({role: 'assistant', content: d.textContent});
  }
</script></body></html>'''

class Handler(HTTPRequestHandler):
  def log_request(self, code='-', size='-'): pass
  def do_GET(self): self.send_data(CHAT_HTML, content_type="text/html")
  def run_model(self, ids:list[int], model_name:str, include_usage=False):
    stderr_log(f"{self.path}  {colored('--', 'BLACK')}  in:{len(ids):5d}  {colored('--', 'BLACK')}  ")
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    yield {"choices": [{"index":0, "delta":{"role":"assistant","content":""}, "finish_reason":None}], **tmpl}
    out: list[int] = []
    st = time.perf_counter()
    for next_id in model.generate(ids):
      if len(out) == 0: stderr_log(f"prefill:{len(ids)/((pt:=time.perf_counter())-st):4.0f} tok/s  {colored('--', 'BLACK')}  ")
      if next_id == eos_id: break
      out.append(next_id)
      yield {"choices": [{"index":0, "delta":{"content":tok.decode([next_id])}, "finish_reason":None}], **tmpl}
    yield {"choices": [{"index":0, "delta":{},"finish_reason":"stop"}], **tmpl}
    if include_usage:
      yield {"choices": [], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}, **tmpl}
    stderr_log(f"out:{len(out):5d}  {colored('--', 'BLACK')}  gen: {len(out)/(time.perf_counter()-pt):4.0f} tok/s\n")

  def do_POST(self):
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if self.path == "/v1/chat/completions":
      ids: list[int] = []
      for msg in body["messages"]:
        ids += tok.role(msg["role"])
        content = msg["content"]
        if isinstance(content, str): ids += tok.encode(content)
        elif isinstance(content, list):
          for c in content:
            if c["type"] == "text": ids += tok.encode(c["text"])
        ids += tok.end_turn(eos_id)
      ids += tok.role("assistant")
      chunks = self.run_model(ids, body["model"], not body.get("stream") or body.get("stream_options",{}).get("include_usage", False))
      if body.get("stream"): self.stream_json(chunks)
      else:
        out = []
        for c in chunks: out.append(c["choices"][0]["delta"].get("content", "") if c["choices"] else "")
        self.send_data(json.dumps({**c, "object":"chat.completion",
          "choices":[{"index":0, "message":{"role":"assistant","content":"".join(out)}, "finish_reason":"stop"}]}).encode())

# ─── Main ──────────────────────────────────────────────────────────────────────
DEFAULT_GGUF = pathlib.Path.home() / ".cache/tinygrad/downloads/qwen3-coder-next/Qwen3-Coder-Next-MXFP4_MOE.gguf"
GGUF_URL = "https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/resolve/main/Qwen3-Coder-Next-MXFP4_MOE.gguf"

def _preflight_memory_check(gguf_path: pathlib.Path):
  """Check system memory and tune kernel params for large model loading on Jetson."""
  import os
  gguf_size_gb = gguf_path.stat().st_size / (1024**3)

  # Read available memory
  try:
    with open("/proc/meminfo") as f:
      meminfo = {k.strip(): int(v.split()[0]) for line in f if ':' in line for k, v in [line.split(':', 1)]}
    avail_gb = meminfo.get('MemAvailable', 0) / (1024**2)
    total_gb = meminfo.get('MemTotal', 0) / (1024**2)
  except Exception:
    avail_gb, total_gb = 0, 0

  stderr_log(f"  System RAM: {total_gb:.1f} GB total, {avail_gb:.1f} GB available\n")
  stderr_log(f"  GGUF file:  {gguf_size_gb:.1f} GB (mmap'd, pages evictable)\n")

  # Estimate memory needs: ~3 GB non-expert params + ~2 GB states + ~3 GB OS headroom
  non_expert_est_gb = 3.0
  os_headroom_gb = 4.0
  min_needed_gb = non_expert_est_gb + os_headroom_gb

  if avail_gb < min_needed_gb:
    stderr_log(f"  WARNING: Only {avail_gb:.1f} GB available, need ~{min_needed_gb:.0f} GB minimum\n")
    stderr_log(f"  Dropping page caches to free memory...\n")
    try:
      os.system("sync")
      with open("/proc/sys/vm/drop_caches", "w") as f: f.write("3")
    except (PermissionError, OSError):
      stderr_log(f"  (could not drop caches — run as root or set vm.drop_caches sysctl)\n")

  # Tune kernel VM params for large mmap workloads
  try:
    with open("/proc/sys/vm/min_free_kbytes") as f: mfk = int(f.read().strip())
    # Ensure at least 256MB min_free to prevent OOM under memory pressure
    if mfk < 262144:
      try:
        with open("/proc/sys/vm/min_free_kbytes", "w") as f: f.write("262144")
        stderr_log(f"  Raised min_free_kbytes: {mfk} → 262144 (256 MB)\n")
      except (PermissionError, OSError): pass
  except Exception: pass

def _decode_with_stats(model: Qwen3NextTransformer, tok: SimpleTokenizer, eos_id: int,
                       ids: list[int], start_pos: int, max_new_tokens: int,
                       stream_output: bool=True) -> tuple[list[int], float, float]:
  t0 = time.perf_counter()
  first_token_ts: float|None = None
  out: list[int] = []

  for next_id in model.generate(ids, start_pos):
    now = time.perf_counter()
    if first_token_ts is None: first_token_ts = now
    if next_id == eos_id or len(out) >= max_new_tokens: break
    out.append(next_id)
    if stream_output:
      sys.stdout.write(tok.decode([next_id]))
      sys.stdout.flush()

  t1 = time.perf_counter()
  prefill_tokens = max(1, len(ids) - start_pos)
  prefill_s = max(1e-9, (first_token_ts or t1) - t0)
  gen_s = max(1e-9, t1 - (first_token_ts or t1))
  prefill_tps = prefill_tokens / prefill_s
  gen_tps = len(out) / gen_s if len(out) > 0 else 0.0
  return out, prefill_tps, gen_tps

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Qwen3-Coder-Next — TinyGrad")
  parser.add_argument("--gguf", type=str, default=str(DEFAULT_GGUF), help="Path to GGUF file")
  parser.add_argument("--max_context", type=int, default=2048, help="Max context length (keep low for memory)")
  parser.add_argument("--serve", nargs='?', type=int, const=11434, metavar="PORT", help="Run OpenAI API server")
  parser.add_argument("--benchmark", nargs='?', type=int, const=20, metavar="COUNT", help="Benchmark tok/s")
  parser.add_argument("--prompt", type=str, default=None, help="Run one-shot prompt and exit")
  parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens for --prompt/chat turns")
  args = parser.parse_args()

  gguf_path = pathlib.Path(args.gguf)
  if not gguf_path.exists():
    print(f"GGUF not found at {gguf_path}\nDownload:\n  curl -L -o '{DEFAULT_GGUF}' '{GGUF_URL}'")
    sys.exit(1)

  print(f"Loading Qwen3-Coder-Next from {gguf_path.name} ...")
  _preflight_memory_check(gguf_path)
  t0 = time.perf_counter()
  model, kv = Qwen3NextTransformer.from_gguf(gguf_path, args.max_context)
  print(f"Loaded in {time.perf_counter()-t0:.1f}s")

  if args.benchmark:
    param_bytes = sum(x.nbytes() for x in nn.state.get_parameters(model))
    gen = model.generate([0], 0)
    for i in range(args.benchmark):
      GlobalCounters.reset()
      with Timing(on_exit=lambda x: f", {1e9/x:6.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s, param {param_bytes/x:7.2f} GB/s"):
        next(gen)
    sys.exit(0)

  tok = SimpleTokenizer.from_gguf_kv(kv)
  bos_id: int|None = kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None
  eos_id: int = kv['tokenizer.ggml.eos_token_id']

  if args.serve is not None:
    print(f"Serving on http://localhost:{args.serve}")
    TCPServerWithReuse(('', args.serve), Handler).serve_forever()

  if args.prompt is not None:
    ids: list[int] = [bos_id] if bos_id is not None else []
    ids += tok.role("user") + tok.encode(args.prompt) + tok.end_turn(eos_id) + tok.role("assistant")
    out, prefill_tps, gen_tps = _decode_with_stats(model, tok, eos_id, ids, max(len(ids)-1, 0), args.max_new_tokens, stream_output=True)
    print(f"\n[prompt] prefill: {prefill_tps:.1f} tok/s, gen: {gen_tps:.1f} tok/s, out: {len(out)} tok")
    sys.exit(0)

  ids: list[int] = [bos_id] if bos_id is not None else []
  while 1:
    start_pos = max(len(ids) - 1, 0)
    try:
      ids += tok.role("user") + tok.encode(input('>>> ')) + tok.end_turn(eos_id) + tok.role("assistant")
    except EOFError:
      break
    out, prefill_tps, gen_tps = _decode_with_stats(model, tok, eos_id, ids, start_pos, args.max_new_tokens, stream_output=True)
    print(f"\n[turn] prefill: {prefill_tps:.1f} tok/s, gen: {gen_tps:.1f} tok/s, out: {len(out)} tok\n")
