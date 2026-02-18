#!/usr/bin/env python3
"""
Tinygrad LLM benchmark — measures prefill, decode, and per-token latency.
Designed to match the cross-framework benchmark metrics exactly.

Usage:
    NV=1 MV_THREADS_PER_ROW=32 python3 bench_tinygrad_direct.py --model <gguf_path> --num-tokens 128
    NV=1 MV_THREADS_PER_ROW=32 JITBEAM=4 python3 bench_tinygrad_direct.py --model <gguf_path>
"""
import argparse, time, statistics, json, sys, os

def bench(model_path, num_tokens=128, num_runs=3, warmup_tokens=5):
    from tinygrad import Tensor, Device
    from tinygrad.apps.llm import Transformer
    from tinygrad.helpers import fetch

    backend = "NV" if os.environ.get("NV") else ("CUDA" if os.environ.get("CUDA") else "CPU")
    jitbeam = os.environ.get("JITBEAM", "0")
    mv_tpr = os.environ.get("MV_THREADS_PER_ROW", "8")

    print(f"\n{'='*70}")
    print(f"Tinygrad Direct Benchmark")
    print(f"{'='*70}")
    print(f"  Model:      {os.path.basename(model_path)}")
    print(f"  Backend:    {backend}")
    print(f"  JITBEAM:    {jitbeam}")
    print(f"  MV_TPR:     {mv_tpr}")
    print(f"  Gen tokens: {num_tokens}")
    print(f"  Runs:       {num_runs}")
    print(f"  Device:     {Device.DEFAULT}")
    print(f"{'='*70}\n")

    # ── Load model ──
    print("Loading model...", end=" ", flush=True)
    t_load = time.perf_counter()
    model, kv = Transformer.from_gguf(Tensor(open(model_path, "rb").read()))
    t_load = time.perf_counter() - t_load
    print(f"done ({t_load:.1f}s)")

    # Use ~100 token prompt to match cross-framework bench
    prompt_text = (
        "Explain the architecture of a transformer neural network in detail, "
        "covering the encoder and decoder blocks, multi-head self-attention mechanism, "
        "positional encoding schemes, feed-forward layers, layer normalization, "
        "residual connections, and how these components work together to process "
        "sequential data for tasks like machine translation, text generation, "
        "and language understanding. Include discussion of the key innovations "
        "that made transformers superior to previous recurrent approaches."
    )

    # Tokenize prompt
    from tinygrad.apps.llm import Tokenizer
    tok_path = None
    # Try to get tokenizer from the GGUF metadata
    try:
        tokenizer = model.tokenizer
        prompt_tokens = tokenizer.encode(prompt_text, allow_special=True)
    except Exception:
        # Fallback: use a simple token list (128000 = BOS for LLaMA 3)
        prompt_tokens = [128000] + [9906] * 50  # approximate

    prompt_len = len(prompt_tokens)
    print(f"Prompt tokens: {prompt_len}\n")

    # ── Warmup ──
    print(f"Warming up ({warmup_tokens} tokens)...", end=" ", flush=True)
    for tok in model.generate(prompt_tokens[:5]):
        warmup_tokens -= 1
        if warmup_tokens <= 0:
            break
    # Reset KV cache
    for kv_entry in kv:
        kv_entry.zero_()
    print("done\n")

    all_prefill_tps = []
    all_decode_tps = []
    all_token_latencies = []

    for run in range(num_runs):
        print(f"  Run {run+1}/{num_runs}...", end=" ", flush=True)

        # Reset KV cache
        for kv_entry in kv:
            kv_entry.zero_()

        token_times = []
        tt = prompt_tokens.copy()

        # Time the generation (prefill happens on first token)
        t_start = time.perf_counter()
        gen_count = 0
        first_token_time = None

        for tok in model.generate(tt):
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            token_times.append(now)
            gen_count += 1
            if gen_count >= num_tokens:
                break

        if not token_times or first_token_time is None:
            print("FAILED")
            continue

        # Prefill = time from start to first generated token
        ttft = first_token_time - t_start
        prefill_tps = prompt_len / ttft if ttft > 0 else 0

        # Decode = steady-state tokens (skip first 2 for warmup)
        if len(token_times) > 3:
            decode_deltas = [token_times[i] - token_times[i-1] for i in range(3, len(token_times))]
            decode_tps = len(decode_deltas) / sum(decode_deltas) if decode_deltas else 0
        else:
            decode_time = token_times[-1] - first_token_time
            decode_tps = (gen_count - 1) / decode_time if decode_time > 0 else 0
            decode_deltas = []

        all_prefill_tps.append(prefill_tps)
        all_decode_tps.append(decode_tps)
        if decode_deltas:
            all_token_latencies.append(decode_deltas)

        print(f"prefill={prefill_tps:.1f} tok/s  decode={decode_tps:.2f} tok/s  TTFT={ttft*1000:.0f}ms")

    if not all_decode_tps:
        print("ERROR: All runs failed")
        sys.exit(1)

    # ── Aggregate ──
    avg_prefill = statistics.mean(all_prefill_tps)
    avg_decode = statistics.mean(all_decode_tps)

    all_deltas_ms = []
    for deltas in all_token_latencies:
        all_deltas_ms.extend([d * 1000 for d in deltas])
    p50_ms = statistics.median(all_deltas_ms) if all_deltas_ms else (1000 / avg_decode if avg_decode > 0 else 0)

    print(f"\n{'─'*70}")
    print(f"Results (averaged over {len(all_decode_tps)} runs):")
    print(f"  Prefill throughput:      {avg_prefill:>8.1f} tok/s")
    print(f"  Decode throughput:       {avg_decode:>8.2f} tok/s")
    print(f"  Decode latency P50:      {p50_ms:>8.2f} ms/tok")
    if all_deltas_ms:
        p10 = sorted(all_deltas_ms)[max(0, len(all_deltas_ms)//10)]
        p90 = sorted(all_deltas_ms)[min(len(all_deltas_ms)-1, len(all_deltas_ms)*9//10)]
        jitter = p90 - p10
        print(f"  Decode latency P10:      {p10:>8.2f} ms/tok")
        print(f"  Decode latency P90:      {p90:>8.2f} ms/tok")
        print(f"  Latency jitter (P90-P10):{jitter:>8.2f} ms")
    print(f"{'─'*70}")

    result = {
        "framework": f"tinygrad {backend}",
        "model": os.path.basename(model_path),
        "jitbeam": int(jitbeam),
        "mv_tpr": int(mv_tpr),
        "num_tokens": num_tokens,
        "num_runs": len(all_decode_tps),
        "prefill_tok_s": round(avg_prefill, 1),
        "decode_tok_s": round(avg_decode, 2),
        "p50_ms": round(p50_ms, 2),
    }
    if all_deltas_ms:
        result["p10_ms"] = round(sorted(all_deltas_ms)[max(0, len(all_deltas_ms)//10)], 2)
        result["p90_ms"] = round(sorted(all_deltas_ms)[min(len(all_deltas_ms)-1, len(all_deltas_ms)*9//10)], 2)
        result["jitter_ms"] = round(result["p90_ms"] - result["p10_ms"], 2)

    print(f"\nJSON: {json.dumps(result)}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tinygrad direct LLM benchmark")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--num-runs", type=int, default=3)
    args = parser.parse_args()
    bench(args.model, args.num_tokens, args.num_runs)
