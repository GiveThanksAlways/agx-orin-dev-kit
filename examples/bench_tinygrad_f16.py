#!/usr/bin/env python3
"""Tinygrad F16 benchmark with proper warmup/measurement separation.

Usage (from examples/tinygrad dev shell):
  NV=1 python3 ../bench_tinygrad_f16.py
  NV=1 JITBEAM=2 python3 ../bench_tinygrad_f16.py
  NV=1 JITBEAM=4 python3 ../bench_tinygrad_f16.py
"""
import time, statistics, sys, os
from tinygrad import Tensor, Device
from tinygrad.helpers import getenv
from tinygrad.apps.llm import Transformer, SimpleTokenizer

DEFAULT_GGUF = "/home/agent/.cache/tinygrad/downloads/llama3.2-1b-f16/Llama-3.2-1B-Instruct-f16.gguf"
GGUF = os.environ.get("GGUF", DEFAULT_GGUF)
WARMUP_TOKENS = 10   # tokens to generate for JIT warmup (not measured)
GEN_TOKENS = 128     # tokens to measure

jitbeam = getenv("JITBEAM", 0)
beam = getenv("BEAM", 0)
label = f"NV=1"
if jitbeam: label += f" JITBEAM={jitbeam}"
if beam: label += f" BEAM={beam}"

print(f"Device: {Device.DEFAULT}")
print(f"Config: {label}")
print(f"GGUF:   {GGUF}")

# Load model
print("Loading model...")
t0 = time.time()
model, kv = Transformer.from_gguf(Tensor.from_url(GGUF))
tok = SimpleTokenizer.from_gguf_kv(kv)
bos_id = kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None
eos_id = kv['tokenizer.ggml.eos_token_id']
print(f"Model loaded in {time.time()-t0:.1f}s")

# Build prompt tokens (chat template)
prompt = ("Explain the key principles of the transformer architecture in deep learning, "
          "including self-attention mechanisms, positional encoding, and how they enable "
          "parallel processing of sequential data.")
ids = ([bos_id] if bos_id else []) + tok.role("user") + tok.encode(prompt) + tok.end_turn(eos_id) + tok.role("assistant")
prompt_len = len(ids)
print(f"Prompt tokens: {prompt_len}")

# Phase 1: Prefill + warmup (JIT compilation + BEAM search happens here)
print(f"Phase 1: Prefill + {WARMUP_TOKENS} warmup tokens (JIT compile)...")
t_prefill_start = time.time()
gen = model.generate(ids, 0)
warmup_times = []
generated = []
for i in range(WARMUP_TOKENS):
    t_s = time.time()
    next_id = next(gen)
    dt = time.time() - t_s
    warmup_times.append(dt)
    generated.append(next_id)

# First token is prefill + JIT, rest of warmup is JIT settling
prefill_time = warmup_times[0]
prefill_tps = prompt_len / prefill_time
print(f"  Prefill: {prefill_tps:.1f} tok/s ({prefill_time*1000:.0f}ms for {prompt_len} tokens)")
if len(warmup_times) > 1:
    warmup_decode = warmup_times[1:]
    warmup_avg = sum(warmup_decode) / len(warmup_decode) * 1000
    print(f"  Warmup decode avg: {warmup_avg:.1f}ms/tok ({len(warmup_decode)} tokens)")

# Phase 2: Steady-state decode measurement
print(f"Phase 2: Measuring {GEN_TOKENS} decode tokens...")
per_tok_times = []
for i in range(GEN_TOKENS):
    t_s = time.time()
    next_id = next(gen)
    dt = time.time() - t_s
    per_tok_times.append(dt)
    generated.append(next_id)
    if next_id == eos_id:
        break

decode_time = sum(per_tok_times)
decode_tps = len(per_tok_times) / decode_time
latencies_ms = [t * 1000 for t in per_tok_times]
p50 = statistics.median(latencies_ms)
sorted_lat = sorted(latencies_ms)
n = len(sorted_lat)
p10 = sorted_lat[n // 10]
p90 = sorted_lat[9 * n // 10]

print()
print(f"=== tinygrad {label} F16 Results ===")
print(f"Prefill:     {prefill_tps:.1f} tok/s ({prefill_time*1000:.0f}ms for {prompt_len} tokens)")
print(f"Decode:      {decode_tps:.1f} tok/s ({len(per_tok_times)} tokens, steady-state)")
print(f"P50 latency: {p50:.2f} ms/tok")
print(f"P10 latency: {p10:.2f} ms/tok")
print(f"P90 latency: {p90:.2f} ms/tok")
print(f"Jitter:      {p90 - p10:.2f} ms")
print(f"Output:      {tok.decode(generated)[:200]}")
