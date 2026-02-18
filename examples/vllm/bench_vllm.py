#!/usr/bin/env python3
"""
vLLM Benchmark for Jetson Orin AGX.
Tests throughput via OpenAI-compatible API (works with Docker or native vLLM).

Usage:
    # Against local vLLM server (Docker or native)
    python3 bench_vllm.py
    python3 bench_vllm.py --server http://localhost:8000 --model meta-llama/Llama-3.2-1B-Instruct
    python3 bench_vllm.py --model meta-llama/Llama-3.2-3B-Instruct --num-tokens 50
"""
import argparse, json, time, sys

def benchmark_vllm(server_url, model, num_tokens=25, prompt="Hello, how are you today?"):
    try:
        import requests
    except ImportError:
        print("Installing requests...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        import requests

    print(f"\n{'='*70}")
    print(f"vLLM Benchmark (OpenAI API)")
    print(f"Server: {server_url}")
    print(f"Model: {model}")
    print(f"Prompt: '{prompt}'")
    print(f"Max tokens: {num_tokens}")
    print(f"{'='*70}\n")

    # Check server health (try /health, fall back to /v1/models)
    try:
        resp = requests.get(f"{server_url}/health", timeout=5)
        if resp.status_code == 200:
            print("Server is healthy ✓\n")
        else:
            # Try /v1/models as fallback (MLC LLM doesn't have /health)
            resp2 = requests.get(f"{server_url}/v1/models", timeout=5)
            if resp2.status_code == 200:
                print("Server is ready ✓\n")
            else:
                print(f"ERROR: Server not healthy (status {resp.status_code})")
                sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to {server_url}")
        print("Start the server first: ./run-vllm-docker.sh or ./run-mlc-docker.sh")
        sys.exit(1)

    # Warmup
    print("Warming up...", flush=True)
    requests.post(
        f"{server_url}/v1/completions",
        json={"model": model, "prompt": "Hi", "max_tokens": 3, "temperature": 0},
        timeout=60
    )

    # Benchmark: non-streaming for total time
    print("Benchmarking (non-streaming)...", flush=True)
    t0 = time.time()
    resp = requests.post(
        f"{server_url}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": num_tokens, "temperature": 0},
        timeout=120
    )
    t_total = time.time() - t0
    
    data = resp.json()
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", num_tokens)
    prompt_tokens = usage.get("prompt_tokens", len(prompt.split()))
    
    # Non-streaming throughput
    total_tok_s = completion_tokens / t_total if t_total > 0 else 0

    # Benchmark: streaming for per-token latency
    print("Benchmarking (streaming)...", flush=True)
    times = []
    t_start = time.time()
    resp = requests.post(
        f"{server_url}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": num_tokens, "temperature": 0, "stream": True},
        stream=True,
        timeout=120
    )
    
    for line in resp.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: ") and line != "data: [DONE]":
                times.append(time.time())
    t_end = time.time()

    # Calculate streaming metrics
    if len(times) > 3:
        # First token = TTFT (time to first token)
        ttft = times[0] - t_start
        # Decode times (skip first 2 tokens)
        decode_times = [times[j] - times[j-1] for j in range(3, len(times))]
        if decode_times:
            stream_tok_s = len(decode_times) / sum(decode_times)
            stream_ms = sum(decode_times) / len(decode_times) * 1000
        else:
            stream_tok_s = total_tok_s
            stream_ms = 1000 / total_tok_s if total_tok_s > 0 else 0
    else:
        ttft = t_total
        stream_tok_s = total_tok_s
        stream_ms = 1000 / total_tok_s if total_tok_s > 0 else 0

    print(f"\nResults:")
    print(f"  Prompt tokens: {prompt_tokens}")
    print(f"  Completion tokens: {completion_tokens}")
    print(f"  TTFT (time to first token): {ttft*1000:.1f} ms")
    print(f"  Non-streaming throughput: {total_tok_s:.2f} tok/s")
    print(f"  Streaming decode throughput: {stream_tok_s:.2f} tok/s")
    print(f"  Streaming decode latency: {stream_ms:.1f} ms/token")

    result = {
        "engine": "vllm",
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": ttft * 1000,
        "total_tok_s": total_tok_s,
        "stream_tok_s": stream_tok_s,
        "stream_ms_token": stream_ms
    }

    print(f"\nJSON: {json.dumps(result)}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM benchmark")
    parser.add_argument("--server", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct", help="Model name")
    parser.add_argument("--num-tokens", type=int, default=25, help="Number of tokens to generate")
    parser.add_argument("--prompt", default="Hello, how are you today?", help="Prompt text")
    args = parser.parse_args()
    
    benchmark_vllm(args.server, args.model, args.num_tokens, args.prompt)
