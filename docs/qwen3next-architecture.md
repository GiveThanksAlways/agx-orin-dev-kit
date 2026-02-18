# Qwen3-Coder-Next (qwen3next) Architecture — Complete Forward Pass

> Reverse-engineered from `ggml-org/llama.cpp` `src/models/qwen3next.cpp`, `src/models/models.h`, `src/llama-arch.cpp`, and `gguf-py/gguf/constants.py` + `tensor_mapping.py`.

## Critical Correction: This is NOT Mamba2

Despite the class inheriting from `llm_graph_context_mamba`, **qwen3next does NOT use Mamba2 SSM**.
It uses **Gated Delta Net** — a linear attention mechanism with a learnable recurrent state matrix.
The "SSM" tensor names in GGUF are a naming convention carry-over; the actual math is completely different.

---

## 1. Model Hyperparameters

| Parameter | Symbol | Value | Notes |
|-----------|--------|-------|-------|
| Hidden dim | `n_embd` / `D` | 2048 | Model dimension |
| SSM inner dim | `d_inner` | 4096 | Delta net expanded dim |
| SSM state dim | `head_k_dim` / `S_k` | 128 | = `ssm_d_state` |
| SSM value head dim | `head_v_dim` / `S_v` | 128 | = `d_inner / num_v_heads` = 4096/32 |
| Conv kernel | `conv_kernel_size` | 4 | 1D causal convolution |
| Num K heads | `num_k_heads` / `H_k` | 16 | = `ssm_n_group` |
| Num V heads | `num_v_heads` / `H_v` | 32 | = `ssm_dt_rank` |
| Attn heads | `n_head` | 16 | Full attention heads |
| Attn KV heads | `n_head_kv` | 2 | GQA for full attention |
| Head dim (attn) | `n_embd_head` | 256 | = `key_length` = `value_length` |
| RoPE dim | `n_rot` | 64 | For full attention layers |
| Full attn interval | `full_attention_interval` | 4 | Every 4th layer is full attention |
| Num layers | `n_layer` | N | Total transformer layers |
| Num experts | `n_expert` | varies | MoE routed experts |
| Num experts used | `n_expert_used` | varies | Top-K routing |
| Conv channels | - | 8192 | = `d_inner + 2*H_k*S_k` = 4096+2×16×128 |

Derived: `H_v / H_k = 32/16 = 2` (each K head maps to 2 V heads)

---

## 2. Layer Type Dispatch

```
for il in range(n_layer):
    x = RMSNorm(x, attn_norm.weight)
    
    if is_recurrent(il):              # layers 0,1,2, 4,5,6, 8,9,10, ...
        x = delta_net_layer(x)        # Gated Delta Net (linear attention)
    else:                              # layers 3, 7, 11, ... (every 4th)
        x = full_attention_layer(x)   # Standard GQA attention with gating
    
    x = x + residual                  # attention residual
    ffn_residual = x
    x = RMSNorm(x, post_attention_norm.weight)
    x = moe_ffn(x)                   # MoE + shared expert
    x = x + ffn_residual             # FFN residual
```

The `is_recurrent(il)` check uses `full_attention_interval=4`:
- Layer indices 0,1,2 → recurrent (delta net)
- Layer index 3 → full attention
- Layer indices 4,5,6 → recurrent
- Layer index 7 → full attention
- ...pattern repeats

---

## 3. GGUF Tensor → Code Mapping

### Global Tensors

| GGUF Name | Code Name | Shape | Purpose |
|-----------|-----------|-------|---------|
| `token_embd.weight` | `tok_embd` | `[D, vocab]` | Token embedding |
| `output_norm.weight` | `output_norm` | `[D]` | Final RMS norm |
| `output.weight` | `output` | `[D, vocab]` | LM head |

### Per-Layer: Delta Net (Recurrent) Layers

| GGUF Name | Code Name | Shape | Purpose |
|-----------|-----------|-------|---------|
| `blk.{i}.attn_norm.weight` | `attn_norm` | `[2048]` | Pre-attention RMS norm |
| `blk.{i}.attn_qkv.weight` | `wqkv` | `[2048, 8192]` | Joint QKV projection |
| `blk.{i}.attn_gate.weight` | `wqkv_gate` | `[2048, 4096]` | Z gate projection |
| `blk.{i}.ssm_ba.weight` | `ssm_beta_alpha` | `[2048, 64]` | Joint beta+alpha projection |
| `blk.{i}.ssm_a` | `ssm_a` | `[32]` | Decay rate: `-exp(A_log)` |
| `blk.{i}.ssm_dt.bias` | `ssm_dt` | `[32]` | Alpha bias (added before softplus) |
| `blk.{i}.ssm_conv1d.weight` | `ssm_conv1d` | `[4, 8192]` | 1D causal conv kernel |
| `blk.{i}.ssm_norm.weight` | `ssm_norm` | `[128]` | Per-head RMS norm (head_v_dim) |
| `blk.{i}.ssm_out.weight` | `ssm_out` | `[4096, 2048]` | Output projection |
| `blk.{i}.post_attention_norm.weight` | `attn_post_norm` | `[2048]` | Post-attention RMS norm |

### Per-Layer: Full Attention Layers

| GGUF Name | Code Name | Shape | Purpose |
|-----------|-----------|-------|---------|
| `blk.{i}.attn_norm.weight` | `attn_norm` | `[2048]` | Pre-attention RMS norm |
| `blk.{i}.attn_q.weight` | `wq` | `[2048, 8192]` | Joint Q+gate (16 heads × 512) |
| `blk.{i}.attn_k.weight` | `wk` | `[2048, 512]` | K projection (2 KV heads × 256) |
| `blk.{i}.attn_v.weight` | `wv` | `[2048, 512]` | V projection (2 KV heads × 256) |
| `blk.{i}.attn_q_norm.weight` | `attn_q_norm` | `[256]` | Q RMS norm (per head) |
| `blk.{i}.attn_k_norm.weight` | `attn_k_norm` | `[256]` | K RMS norm (per head) |
| `blk.{i}.attn_output.weight` | `wo` | `[4096, 2048]` | Output projection |
| `blk.{i}.post_attention_norm.weight` | `attn_post_norm` | `[2048]` | Post-attention RMS norm |

### Per-Layer: MoE FFN (All Layers)

| GGUF Name | Code Name | Shape | Purpose |
|-----------|-----------|-------|---------|
| `blk.{i}.ffn_gate_inp.weight` | `ffn_gate_inp` | `[2048, n_expert]` | Router logits |
| `blk.{i}.ffn_gate_exps.weight` | `ffn_gate_exps` | packed | Expert gate (SwiGLU) |
| `blk.{i}.ffn_up_exps.weight` | `ffn_up_exps` | packed | Expert up projection |
| `blk.{i}.ffn_down_exps.weight` | `ffn_down_exps` | packed | Expert down projection |
| `blk.{i}.ffn_gate_shexp.weight` | `ffn_gate_shexp` | `[2048, ff_shexp]` | Shared expert gate |
| `blk.{i}.ffn_up_shexp.weight` | `ffn_up_shexp` | `[2048, ff_shexp]` | Shared expert up |
| `blk.{i}.ffn_down_shexp.weight` | `ffn_down_shexp` | `[ff_shexp, 2048]` | Shared expert down |
| `blk.{i}.ffn_gate_inp_shexp.weight` | `ffn_gate_inp_shexp` | `[2048]` | Shared expert scalar gate |

---

## 4. Delta Net Layer — Full Forward Pass (Token Generation)

This is the core recurrent layer. For single-token generation (`n_seq_tokens == 1`):

### 4.1 Input Projections

```python
# x: [D=2048]  (single token)

# QKV projection
qkv_mixed = x @ wqkv.T                    # [2048] @ [2048, 8192].T → [8192]

# Z gate projection  
z = x @ wqkv_gate.T                       # [2048] @ [2048, 4096].T → [4096]

# Beta-Alpha projection
mixed_ba = x @ ssm_beta_alpha.T           # [2048] @ [2048, 64].T → [64]
```

### 4.2 Beta-Alpha Split

```python
# Reshape mixed_ba: [64] → [2*H_v/H_k, H_k] = [4, 16]
mixed_ba = mixed_ba.reshape(4, 16)         # [ba_new_dim=4, num_k_heads=16]

# Split into beta (first 2 per head) and alpha (last 2 per head)
b = mixed_ba[:2, :]                        # [2, 16] = [H_v/H_k, H_k]
a = mixed_ba[2:, :]                        # [2, 16]

# Reshape to merge head dims
beta  = b.reshape(32, 1)                   # [H_v=32, 1]
alpha = a.reshape(32)                      # [H_v=32]
```

### 4.3 Compute Decay Gate

```python
# ssm_dt is bias [32], ssm_a contains -exp(A_log) [32]
alpha_biased   = alpha + ssm_dt_bias       # [32] + [32] → [32]
alpha_softplus = softplus(alpha_biased)     # softplus(x) = log(1 + exp(x))
gate = alpha_softplus * ssm_a              # [32] * [32] → [32]
# gate is NEGATIVE (ssm_a = -exp(A_log) < 0), representing log-decay per step
```

### 4.4 Causal Convolution (with State)

```python
# conv_channels = d_inner + 2*H_k*S_k = 4096 + 2*16*128 = 8192
# conv_states: [conv_kernel-1, conv_channels] = [3, 8192] from cache

# Prepend cached states to current input
conv_input = cat([conv_states, qkv_mixed.unsqueeze(0)], dim=0)  # [4, 8192]

# Update conv state cache (store last 3 columns)
new_conv_states = conv_input[1:]           # [3, 8192]

# 1D convolution (no padding, valid mode) 
conv_out = ssm_conv(conv_input, ssm_conv1d_weight)  # [4, 8192] conv [4, 8192] → [8192]

# SiLU activation
conv_out = silu(conv_out)                  # [8192]
```

### 4.5 Split QKV from Conv Output

```python
# Split conv output (total 8192) into Q, K, V
# Q: num_k_heads * head_k_dim = 16 * 128 = 2048
# K: num_k_heads * head_k_dim = 16 * 128 = 2048  
# V: num_v_heads * head_v_dim = 32 * 128 = 4096
# Total: 2048 + 2048 + 4096 = 8192 ✓

# The split order in GGUF/llama.cpp for qkv_mixed is V first, then Q, then K
# (d_inner for V, then 2 * n_group * d_state for QK)
# Actually: V is first d_inner=4096, then [Q,K] interleaved in the remaining 4096

v_conv = conv_out[:4096].reshape(128, 32)          # [S_v=128, H_v=32]
qk_conv = conv_out[4096:]                           # [4096]
q_conv = qk_conv.reshape(128, 16, 2)[:,:,0]        # [S_k=128, H_k=16] 
k_conv = qk_conv.reshape(128, 16, 2)[:,:,1]        # [S_k=128, H_k=16]

# If H_k != H_v, repeat Q and K to match V heads
# num_v_heads(32) / num_k_heads(16) = 2, so repeat each K-head 2x
q_conv = q_conv.repeat_interleave(2, dim=1)         # [128, 32]
k_conv = k_conv.repeat_interleave(2, dim=1)         # [128, 32]
```

### 4.6 Delta Net Autoregressive Update

This is the heart of the recurrent mechanism:

```python
# State shape: [S_v=128, S_v*H_v = 128*32 = 4096, 1, n_seqs]
# Reshaped to: [S_v=128, S_v=128, H_v=32, n_seqs]
# For single seq: state is [128, 128, 32]  (a matrix per V-head)

S_k = 128  # q/k head dim (= S_v in this model)
S_v = 128  # v head dim
H_v = 32   # num v heads

# 1. L2-normalize Q and K
q = l2_norm(q_conv)                        # [S_k=128, H_v=32]
k = l2_norm(k_conv)                        # [S_k=128, H_v=32]

# 2. Scale Q
scale = 1.0 / sqrt(S_v)                   # 1/sqrt(128)
q = q * scale                              # [128, 32]

# 3. Sigmoid on beta (update gate)
beta = sigmoid(beta)                       # [H_v=32, 1]

# 4. Exponentiate gate (decay)
g = exp(gate)                              # [H_v=32] — values < 1 since gate is negative
g = g.reshape(1, 1, H_v)                  # broadcast shape

# 5. Apply decay to state
state = state * g                          # [128, 128, 32] * [1, 1, 32]

# 6. Compute kv_mem = state @ k  (sum over S_v dim)
k_t = k.reshape(1, S_v, H_v)              # [1, 128, 32]
kv_mem = (state * k_t).sum(dim=1)         # [128, 1, 32]  — contract over S_v

# 7. Delta rule: v_new = beta * (v - kv_mem)
v_t = v_conv.reshape(S_v, 1, H_v)         # [128, 1, 32]
delta = beta * (v_t - kv_mem)              # [128, 1, 32]

# 8. State update: state += outer(k, delta)
k_outer = k.reshape(1, S_v, H_v).expand(S_v, S_v, H_v)  # broadcast
state = state + k_outer * delta            # [128, 128, 32]

# 9. Query the state: output = state @ q  (sum over S_v dim)
q_t = q.reshape(1, S_v, H_v)              # [1, 128, 32]
output = (state * q_t).sum(dim=1)         # [128, 1, 32]
```

**Interpretation**: The state is a bank of 32 matrices, each 128×128. At each step:
1. Decay all states by `exp(gate)` (multiplicative forgetting)
2. Read old value at key `k`: `kv_mem = S @ k`
3. Compute correction: `delta = beta * (v_new - kv_mem)`
4. Write: `S += k ⊗ delta` (rank-1 update)
5. Read at query `q`: `output = S @ q`

### 4.7 Gated Normalization + Output Projection

```python
# output: [S_v=128, 1, H_v=32] → flatten to [4096]
output = output.reshape(4096)

# z gate from step 4.1: [4096]
# Gated norm: RMSNorm(output) * SiLU(z)
output_normed = rms_norm(output.reshape(32, 128), ssm_norm_weight)  # per-head RMSNorm, [128] weights
output_normed = output_normed.reshape(4096)
output_gated = output_normed * silu(z)                              # [4096]

# Output projection
result = output_gated @ ssm_out.T          # [4096] @ [4096, 2048].T → [2048]
```

---

## 5. Full Attention Layer — Forward Pass

Every `full_attention_interval`-th layer (layer 3, 7, 11, ...):

```python
# x: [D=2048, n_tokens]

# Joint Q+Gate projection
Qcur_full = x @ wq.T                      # [2048] @ [2048, 8192].T → [8192]
Qcur_full = Qcur_full.reshape(512, 16)    # [n_embd_head*2, n_head] = [512, 16]

# Split into Q and gate
Qcur = Qcur_full[:256, :]                 # [256, 16] — query
gate  = Qcur_full[256:, :]                # [256, 16] — attention gate

# K, V projections
Kcur = x @ wk.T                           # [2048] @ [2048, 512].T → [512] → [256, 2]
Vcur = x @ wv.T                           # [2048] @ [2048, 512].T → [512] → [256, 2]

# Q, K normalization (per-head RMSNorm)
Qcur = rms_norm(Qcur, attn_q_norm)        # [256] weights applied per head
Kcur = rms_norm(Kcur, attn_k_norm)        # [256] weights applied per head

# RoPE (on first rope_dim=64 of each head)
Qcur = apply_rope(Qcur, positions, rope_dim=64, ...)
Kcur = apply_rope(Kcur, positions, rope_dim=64, ...)

# Standard GQA attention (16 Q heads, 2 KV heads → group size 8)
# KV cache is used here (standard transformer KV cache, not recurrent state)
attn_out = grouped_query_attention(Qcur, Kcur, Vcur, scale=1/sqrt(256))
# attn_out: [256, 16] per token → [4096]

# Sigmoid gating
gate_sigmoid = sigmoid(gate)               # [256, 16] → [4096]
attn_out = attn_out * gate_sigmoid         # element-wise

# Output projection  
result = attn_out @ wo.T                   # [4096] @ [4096, 2048].T → [2048]
```

---

## 6. MoE FFN — Forward Pass

Applied to EVERY layer (both recurrent and attention layers):

```python
# x: [D=2048] (after post_attention_norm)

# === Routed Experts ===
# Router
logits = x @ ffn_gate_inp.T               # [2048] @ [2048, n_expert].T → [n_expert]
probs = softmax(logits)                     # top-K routing

# For each selected expert i (top-K):
#   gate_i = x @ ffn_gate_exps[i].T       # [2048] → [ff_dim]
#   up_i   = x @ ffn_up_exps[i].T         # [2048] → [ff_dim]
#   expert_out_i = (silu(gate_i) * up_i) @ ffn_down_exps[i].T  # [ff_dim] → [2048]
# moe_out = sum(prob_i * expert_out_i for selected experts)
moe_out = moe_ffn(x, routed_experts, probs)  # [2048]

# === Shared Expert ===
if ffn_up_shexp is not None:
    # Standard SwiGLU FFN
    gate_shexp = x @ ffn_gate_shexp.T     # [2048] → [ff_shexp]
    up_shexp   = x @ ffn_up_shexp.T       # [2048] → [ff_shexp]
    ffn_shexp  = (silu(gate_shexp) * up_shexp) @ ffn_down_shexp.T  # → [2048]
    
    # Shared expert gate (scalar sigmoid gate)
    shared_gate = x @ ffn_gate_inp_shexp.T  # [2048] @ [2048, 1]? → scalar
    shared_gate = sigmoid(shared_gate)       # scalar in [0,1]
    
    # Apply gate
    ffn_shexp = ffn_shexp * shared_gate      # [2048] * scalar
    
    # Combine
    result = moe_out + ffn_shexp             # [2048]
```

**So `ffn_gate_inp_shexp.weight` [2048] is**: A linear projection that produces a **scalar per token**, sigmoided to get a gate value in [0,1], which controls how much the shared expert output contributes. This is identical to the Qwen2MoE / DeepSeek shared expert gating mechanism.

---

## 7. Recurrent State Maintenance

### State Shapes (per recurrent layer)

| State | Shape | Size (bytes, fp32) |
|-------|-------|--------------------|
| Delta net state (S) | `[S_v, S_v*H_v, 1, n_seqs]` = `[128, 4096, 1, 1]` | 128×4096×4 = 2MB |
| Conv1d state | `[conv_kernel-1, conv_channels, n_seqs]` = `[3, 8192, 1]` | 3×8192×4 = 96KB |

### How States Flow

```
Token 1 (prefill/prompt):
  - Delta net uses CHUNKING (chunk_size=64)
  - Conv states initialized, state matrix built up over chunks
  - Final state saved to cache

Token 2+ (generation):
  - Conv state: last 3 input columns carried forward
  - Delta net state: [128, 128, 32] matrix carried forward
  - Each step: decay → read → delta update → write → query
```

### Full Attention Layers
Full attention layers use a standard **KV cache** (not recurrent state). Key and Value tensors
are appended to the cache at each step, and attention is computed over the full sequence.

---

## 8. Complete Single-Token Pseudocode (TinyGrad-Ready)

```python
import numpy as np
from tinygrad import Tensor

class Qwen3NextLayer:
    """One transformer layer — either delta_net or full_attention."""
    
    def __init__(self, weights, layer_idx, is_recurrent, hparams):
        self.w = weights
        self.il = layer_idx
        self.is_recurrent = is_recurrent
        self.hp = hparams
    
    def __call__(self, x, pos, delta_state, conv_state, kv_cache):
        """
        x:           [D=2048]
        delta_state: [128, 128, 32] or None
        conv_state:  [3, 8192] or None
        kv_cache:    (K_cache, V_cache) or None
        Returns:     (output, new_delta_state, new_conv_state, new_kv_cache)
        """
        D = self.hp['n_embd']          # 2048
        residual = x
        
        # Pre-attention norm
        x = rms_norm(x, self.w['attn_norm'])
        
        if self.is_recurrent:
            out, delta_state, conv_state = self.delta_net_forward(x, delta_state, conv_state)
        else:
            out, kv_cache = self.full_attn_forward(x, pos, kv_cache)
        
        # Attention residual
        x = out + residual
        ffn_residual = x
        
        # Post-attention norm
        x = rms_norm(x, self.w['post_attention_norm'])
        
        # MoE FFN
        x = self.moe_ffn(x)
        
        # FFN residual
        x = x + ffn_residual
        
        return x, delta_state, conv_state, kv_cache
    
    def delta_net_forward(self, x, state, conv_state):
        """Gated Delta Net — single token autoregressive."""
        D = 2048
        d_inner = 4096
        S_k = 128                      # head_k_dim = ssm_d_state
        S_v = 128                      # head_v_dim = d_inner / H_v
        H_k = 16                       # num_k_heads = ssm_n_group
        H_v = 32                       # num_v_heads = ssm_dt_rank
        conv_k = 4
        conv_ch = 8192                 # d_inner + 2*H_k*S_k
        
        # === Projections ===
        qkv_mixed = x @ self.w['attn_qkv'].T           # [8192]
        z = x @ self.w['attn_gate'].T                   # [4096]
        mixed_ba = x @ self.w['ssm_ba'].T               # [64]
        
        # === Beta-Alpha split ===
        mixed_ba = mixed_ba.reshape(4, 16)              # [2*H_v/H_k, H_k]
        beta_raw = mixed_ba[:2].reshape(H_v, 1)         # [32, 1]
        alpha = mixed_ba[2:].reshape(H_v)               # [32]
        
        # === Decay gate ===
        gate = self.w['ssm_a'] * (alpha + self.w['ssm_dt']).softplus()  # [32], negative
        
        # === Conv1d ===
        conv_input = Tensor.cat(conv_state, qkv_mixed.unsqueeze(0), dim=0)  # [4, 8192]
        new_conv_state = conv_input[1:]                  # [3, 8192]
        # Depthwise conv1d: each channel independently
        conv_out = (conv_input * self.w['ssm_conv1d']).sum(dim=0)  # [8192]
        conv_out = conv_out.silu()
        
        # === Split QKV ===
        v = conv_out[:d_inner].reshape(S_v, H_v)        # [128, 32]
        qk = conv_out[d_inner:]                          # [4096]
        q = qk.reshape(S_k, H_k, 2)[:,:,0]              # [128, 16]
        k = qk.reshape(S_k, H_k, 2)[:,:,1]              # [128, 16]
        # Repeat Q,K from H_k=16 to H_v=32
        q = q.repeat(1, 2)                               # [128, 32]
        k = k.repeat(1, 2)                               # [128, 32]
        
        # === Delta Net Autoregressive ===
        eps = 1e-6
        q = q / (q.square().sum(0, keepdim=True).sqrt() + eps)  # L2 norm per head
        k = k / (k.square().sum(0, keepdim=True).sqrt() + eps)
        q = q * (1.0 / S_v**0.5)                        # scale
        beta = beta_raw.sigmoid()                         # [32, 1]
        
        # state: [128, 128, 32]
        # Decay
        g = gate.exp().reshape(1, 1, H_v)               # [1, 1, 32], values in (0,1)
        state = state * g
        
        # Read: kv_mem = sum_j state[i,j,h] * k[j,h]
        kv_mem = (state * k.reshape(1, S_v, H_v)).sum(dim=1, keepdim=True)  # [128, 1, 32]
        
        # Delta update
        v_t = v.reshape(S_v, 1, H_v)
        delta = beta.reshape(1, 1, H_v) * (v_t - kv_mem)  # [128, 1, 32]
        
        # Write: state += k_outer * delta
        state = state + k.reshape(1, S_v, H_v) * delta  # broadcast: [128, 128, 32]
        
        # Query: output = sum_j state[i,j,h] * q[j,h]
        output = (state * q.reshape(1, S_v, H_v)).sum(dim=1)  # [128, 32]
        
        # === Gated Norm + Output ===
        output = output.reshape(H_v, S_v)                # [32, 128]
        output = rms_norm_per_head(output, self.w['ssm_norm'])  # [128] weights per head
        output = output.reshape(d_inner)                  # [4096]
        output = output * z.silu()                        # gated
        result = output @ self.w['ssm_out'].T             # [4096] → [2048]
        
        return result, state, new_conv_state
    
    def full_attn_forward(self, x, pos, kv_cache):
        """Full GQA attention with sigmoid gating."""
        D = 2048
        n_head = 16
        n_kv = 2
        d_head = 256
        
        # Joint Q+Gate
        qg = (x @ self.w['attn_q'].T).reshape(512, n_head)  # [512, 16]
        q = qg[:256]                                      # [256, 16]
        gate = qg[256:]                                    # [256, 16]
        
        # K, V
        k = (x @ self.w['attn_k'].T).reshape(d_head, n_kv)  # [256, 2]
        v = (x @ self.w['attn_v'].T).reshape(d_head, n_kv)  # [256, 2]
        
        # Norms
        q = rms_norm_per_head(q.reshape(n_head, d_head), self.w['attn_q_norm']).reshape(d_head, n_head)
        k = rms_norm_per_head(k.reshape(n_kv, d_head), self.w['attn_k_norm']).reshape(d_head, n_kv)
        
        # RoPE (first 64 dims)
        q = apply_rope(q, pos, rope_dim=64)
        k = apply_rope(k, pos, rope_dim=64)
        
        # Update KV cache
        K_cache, V_cache = kv_cache
        K_cache = Tensor.cat(K_cache, k.unsqueeze(-1), dim=-1)
        V_cache = Tensor.cat(V_cache, v.unsqueeze(-1), dim=-1)
        
        # GQA: 16 Q heads, 2 KV heads → group size 8
        # Standard scaled dot-product attention
        attn_out = gqa_attention(q, K_cache, V_cache, scale=1.0/16.0)  # 1/sqrt(256)
        # attn_out: [256, 16] → [4096]
        
        # Sigmoid gating
        attn_out = attn_out * gate.sigmoid()
        
        # Output projection
        result = attn_out.reshape(4096) @ self.w['attn_output'].T  # → [2048]
        
        return result, (K_cache, V_cache)
    
    def moe_ffn(self, x):
        """MoE with shared expert."""
        # Routed experts
        logits = x @ self.w['ffn_gate_inp'].T             # [n_expert]
        probs = logits.softmax()
        top_k_indices = probs.topk(self.hp['n_expert_used'])
        
        moe_out = Tensor.zeros(2048)
        for idx in top_k_indices:
            gate_e = (x @ self.w['ffn_gate_exps'][idx].T).silu()
            up_e   = x @ self.w['ffn_up_exps'][idx].T
            expert = (gate_e * up_e) @ self.w['ffn_down_exps'][idx].T
            moe_out = moe_out + probs[idx] * expert
        
        # Shared expert
        if 'ffn_up_shexp' in self.w:
            gate_s = (x @ self.w['ffn_gate_shexp'].T).silu()
            up_s   = x @ self.w['ffn_up_shexp'].T
            shexp_out = (gate_s * up_s) @ self.w['ffn_down_shexp'].T  # [2048]
            
            # Scalar sigmoid gate
            sg = (x @ self.w['ffn_gate_inp_shexp'].T).sigmoid()  # scalar
            shexp_out = shexp_out * sg
            
            moe_out = moe_out + shexp_out
        
        return moe_out


def rms_norm(x, weight, eps=1e-6):
    """RMS normalization."""
    rms = (x.square().mean() + eps).sqrt()
    return (x / rms) * weight

def rms_norm_per_head(x, weight, eps=1e-6):
    """RMS norm applied independently per head. x: [n_heads, head_dim], weight: [head_dim]."""
    rms = (x.square().mean(dim=-1, keepdim=True) + eps).sqrt()
    return (x / rms) * weight
```

---

## 9. Tensor `ssm_ba.weight` Explained

`ssm_ba.weight` (GGUF name: `blk.{i}.ssm_ba`) is `LLM_TENSOR_SSM_BETA_ALPHA`.

**HuggingFace origin**: `model.layers.{bid}.linear_attn.in_proj_ba`

It is a **combined beta+alpha projection** unique to qwen3next (qwen3.5 has separate `ssm_beta` and `ssm_alpha` tensors instead).

- **Shape**: `[2048, 64]` where 64 = `2 × (H_v/H_k) × H_k` = `2 × 2 × 16`
- **Beta** (first half): Update gate — controls how much to overwrite old state with new value. Passed through `sigmoid()`.
- **Alpha** (second half): Decay parameter — added to `ssm_dt` bias, passed through `softplus()`, then multiplied by `ssm_a` (which is `-exp(A_log)`) to produce the per-step multiplicative decay factor for the state matrix.

---

## 10. Delta Net Chunking (Prefill)

For prompt processing (`n_seq_tokens > 1`), the delta net uses a chunked algorithm with `CHUNK_SIZE=64`:

1. L2-normalize Q, K; scale Q by `1/sqrt(S_v)`; sigmoid beta
2. Reshape into chunks of 64 tokens
3. For each chunk:
   - Compute cumulative decay mask within chunk
   - Compute intra-chunk attention via triangular solve
   - Apply cross-chunk state updates
4. The state carries over between chunks, producing the final state for generation

This is mathematically equivalent to the autoregressive version but much faster for long sequences due to parallelism.

---

## Summary for TinyGrad Implementation

1. **Two layer types**: Delta Net (recurrent, ~75% of layers) and Full Attention (every 4th layer)
2. **Delta Net state**: One `[128, 128, 32]` matrix per recurrent layer = ~2MB fp32 per layer
3. **Conv state**: `[3, 8192]` per recurrent layer = ~96KB fp32
4. **KV cache**: Standard per full-attention layer
5. **Key ops**: L2 norm, sigmoid, softplus, exp, SiLU, RMSNorm, 1D conv, matrix-vector products
6. **MoE**: Standard top-K routing + shared expert with scalar sigmoid gate on every layer
