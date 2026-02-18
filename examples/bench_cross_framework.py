#!/usr/bin/env python3
"""
Cross-framework LLM benchmark — measures 3 metrics via OpenAI-compatible API.

Metrics:
  1. Prefill throughput (prompt tok/s) — how fast the model processes input tokens
  2. Decode throughput  (generation tok/s) — steady-state autoregressive speed
  3. Per-token decode latency P50 (median ms/tok) — consistency/jitter

Works with any OpenAI-compatible server: vLLM, MLC LLM, tinygrad, llama.cpp, etc.

Usage:
    python3 bench_cross_framework.py --server http://localhost:8000 --model "model-name" --num-tokens 128
"""
import argparse, json, time, sys, statistics

def benchmark(server_url, model, num_tokens=128, prompt_text=None, num_runs=3):
    try:
        import requests
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        import requests

    # Use a longer prompt to measure prefill properly (~100 tokens)
    if prompt_text is None:
        prompt_text = (
            "Explain the architecture of a transformer neural network in detail, "
            "covering the encoder and decoder blocks, multi-head self-attention mechanism, "
            "positional encoding schemes, feed-forward layers, layer normalization, "
            "residual connections, and how these components work together to process "
            "sequential data for tasks like machine translation, text generation, "
            "and language understanding. Include discussion of the key innovations "
            "that made transformers superior to previous recurrent approaches."
        )

    print(f"\n{'='*70}")
    print(f"Cross-Framework Benchmark (OpenAI API)")
    print(f"{'='*70}")
    print(f"  Server:     {server_url}")
    print(f"  Model:      {model}")
    print(f"  Gen tokens: {num_tokens}")
    print(f"  Runs:       {num_runs}")
    print(f"  Prompt:     '{prompt_text[:60]}...'")
    print(f"{'='*70}\n")

    # Check server readiness (try /health, then /v1/models)
    try:
        resp = requests.get(f"{server_url}/health", timeout=5)
        if resp.status_code != 200:
            resp = requests.get(f"{server_url}/v1/models", timeout=5)
            if resp.status_code != 200:
                print(f"ERROR: Server not ready"); sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to {server_url}"); sys.exit(1)
    print("Server ready ✓\n")

    # ── Warmup (2 short requests) ──
    print("Warming up...", end="", flush=True)
    for _ in range(2):
        try:
            requests.post(f"{server_url}/v1/completions",
                json={"model": model, "prompt": "Hi", "max_tokens": 5, "temperature": 0},
                timeout=120)
        except Exception:
            pass
    print(" done\n")

    all_prefill_tps = []
    all_decode_tps = []
    all_token_latencies = []  # list of lists

    for run in range(num_runs):
        print(f"  Run {run+1}/{num_runs}...", end=" ", flush=True)

        # ── Streaming request to measure per-token times ──
        token_times = []
        t_start = time.perf_counter()

        resp = requests.post(
            f"{server_url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt_text,
                "max_tokens": num_tokens,
                "temperature": 0,
                "stream": True,
            },
            stream=True,
            timeout=300,
        )

        first_token_time = None
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: ") and line != "data: [DONE]":
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                token_times.append(now)

        t_end = time.perf_counter()

        if not token_times or first_token_time is None:
            print("FAILED (no tokens received)")
            continue

        n_tokens = len(token_times)

        # ── Non-streaming request to get token counts ──
        resp2 = requests.post(
            f"{server_url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt_text,
                "max_tokens": num_tokens,
                "temperature": 0,
            },
            timeout=300,
        )
        usage = resp2.json().get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", len(prompt_text.split()))
        completion_tokens = usage.get("completion_tokens", n_tokens)

        # ── Compute metrics ──
        # Prefill = time from request start to first token
        ttft = first_token_time - t_start
        prefill_tps = prompt_tokens / ttft if ttft > 0 else 0

        # Decode = tokens after first, measured from streaming deltas
        # Skip first 2 tokens (warmup) for steady-state measurement
        if len(token_times) > 3:
            decode_deltas = [token_times[i] - token_times[i-1] for i in range(3, len(token_times))]
            decode_tps = len(decode_deltas) / sum(decode_deltas) if decode_deltas else 0
        else:
            total_decode_time = t_end - first_token_time
            decode_tps = (n_tokens - 1) / total_decode_time if total_decode_time > 0 and n_tokens > 1 else 0
            decode_deltas = []

        all_prefill_tps.append(prefill_tps)
        all_decode_tps.append(decode_tps)
        if decode_deltas:
            all_token_latencies.append(decode_deltas)

        print(f"prefill={prefill_tps:.1f} tok/s  decode={decode_tps:.2f} tok/s  ({n_tokens} tokens)")

    if not all_decode_tps:
        print("\nERROR: All runs failed")
        sys.exit(1)

    # ── Aggregate results ──
    avg_prefill = statistics.mean(all_prefill_tps)
    avg_decode = statistics.mean(all_decode_tps)

    # P50 latency from all runs combined
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
        print(f"  Decode latency P10:      {p10:>8.2f} ms/tok")
        print(f"  Decode latency P90:      {p90:>8.2f} ms/tok")
        jitter = p90 - p10
        print(f"  Latency jitter (P90-P10):{jitter:>8.2f} ms")
    print(f"{'─'*70}")

    result = {
        "model": model,
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
    parser = argparse.ArgumentParser(description="Cross-framework LLM benchmark")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--prompt", default=None, help="Custom prompt (default: ~100 token transformer explanation)")
    args = parser.parse_args()
    benchmark(args.server, args.model, args.num_tokens, args.prompt, args.num_runs)
