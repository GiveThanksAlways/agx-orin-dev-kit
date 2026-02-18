#!/usr/bin/env python3
"""
MLC LLM Benchmark for Jetson Orin AGX.
Tests decode throughput — works against both a running MLC server (Docker)
and a local native install.

Usage (against Docker server):
    ./run-mlc-docker.sh serve
    python3 bench_mlc_llm.py --server http://localhost:8001

Usage (native, if installed):
    python3 bench_mlc_llm.py --native HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC

Usage (legacy):
    python3 bench_mlc_llm.py HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC 25
"""
import argparse, json, time, sys


def benchmark_server(server_url, model, num_tokens=25, prompt="Hello, how are you today?"):
    """Benchmark against running MLC LLM OpenAI-compatible server."""
    try:
        import requests
    except ImportError:
        print("Installing requests...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        import requests

    print(f"\n{'='*70}")
    print(f"MLC LLM Benchmark (Server Mode)")
    print(f"Server: {server_url}")
    print(f"Model: {model}")
    print(f"Prompt: '{prompt}'")
    print(f"Max tokens: {num_tokens}")
    print(f"{'='*70}\n")

    # Check server
    try:
        resp = requests.get(f"{server_url}/v1/models", timeout=5)
        if resp.status_code != 200:
            print(f"ERROR: Server returned status {resp.status_code}")
            sys.exit(1)
        models = resp.json()
        print(f"Server models: {json.dumps(models.get('data', []), indent=2)}")
        print()
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to {server_url}")
        print("Start MLC LLM server first: ./run-mlc-docker.sh serve")
        sys.exit(1)

    # Warmup
    print("Warming up...", flush=True)
    requests.post(
        f"{server_url}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 3},
        timeout=60
    )

    # Non-streaming benchmark
    print("Benchmarking (non-streaming)...", flush=True)
    t0 = time.time()
    resp = requests.post(
        f"{server_url}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": num_tokens, "temperature": 0},
        timeout=120
    )
    t_total = time.time() - t0

    data = resp.json()
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", num_tokens)
    total_tok_s = completion_tokens / t_total if t_total > 0 else 0

    # Streaming benchmark
    print("Benchmarking (streaming)...", flush=True)
    times = []
    t_start = time.time()
    resp = requests.post(
        f"{server_url}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": num_tokens, "temperature": 0, "stream": True},
        stream=True, timeout=120
    )
    for line in resp.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: ") and line != "data: [DONE]":
                times.append(time.time())

    if len(times) > 3:
        ttft = times[0] - t_start
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
    print(f"  Completion tokens: {completion_tokens}")
    print(f"  TTFT: {ttft*1000:.1f} ms")
    print(f"  Non-streaming throughput: {total_tok_s:.2f} tok/s")
    print(f"  Streaming decode throughput: {stream_tok_s:.2f} tok/s")
    print(f"  Streaming decode latency: {stream_ms:.1f} ms/token")

    result = {
        "engine": "mlc-llm",
        "mode": "server",
        "model": model,
        "completion_tokens": completion_tokens,
        "ttft_ms": ttft * 1000,
        "total_tok_s": total_tok_s,
        "stream_tok_s": stream_tok_s,
        "stream_ms_token": stream_ms
    }
    print(f"\nJSON: {json.dumps(result)}")
    return result


def benchmark_native(model_path, num_tokens=25, prompt="Hello, how are you?"):
    """Benchmark using local MLC LLM install (the MLCEngine API)."""
    try:
        from mlc_llm import MLCEngine
    except ImportError:
        print("ERROR: mlc_llm not installed.")
        print("Use --server mode against a running Docker container instead,")
        print("or install: pip install mlc-ai-nightly -f https://mlc.ai/wheels")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"MLC LLM Benchmark (Native Mode)")
    print(f"Model: {model_path}")
    print(f"Prompt: '{prompt}'")
    print(f"Max tokens: {num_tokens}")
    print(f"{'='*70}\n")

    print("Loading model...", flush=True)
    t0 = time.time()
    engine = MLCEngine(model_path)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s\n")

    # Warmup
    print("Warming up...", flush=True)
    for resp in engine.chat.completions.create(
        messages=[{"role": "user", "content": "Hi"}],
        model=model_path, max_tokens=5, stream=True
    ):
        pass

    # Benchmark
    print("Benchmarking decode...", flush=True)
    times = []
    tokens_generated = 0

    t_start = time.time()
    for resp in engine.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_path, max_tokens=num_tokens, stream=True
    ):
        times.append(time.time())
        if resp.choices[0].delta.content:
            tokens_generated += 1
    t_end = time.time()

    total_time = t_end - t_start
    if tokens_generated > 3:
        decode_times = [times[j] - times[j-1] for j in range(3, len(times))]
        if decode_times:
            avg_ms = sum(decode_times) / len(decode_times) * 1000
            tok_s = len(decode_times) / sum(decode_times)
        else:
            avg_ms = total_time / tokens_generated * 1000
            tok_s = tokens_generated / total_time
    else:
        avg_ms = total_time / max(tokens_generated, 1) * 1000
        tok_s = max(tokens_generated, 1) / total_time

    print(f"\nResults:")
    print(f"  Tokens generated: {tokens_generated}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Decode throughput: {tok_s:.2f} tok/s")
    print(f"  Decode latency: {avg_ms:.1f} ms/token")
    print(f"  Model load time: {load_time:.1f}s")

    result = {
        "engine": "mlc-llm",
        "mode": "native",
        "model": model_path,
        "tokens": tokens_generated,
        "tok_s": tok_s,
        "ms_token": avg_ms,
        "load_time_s": load_time
    }
    print(f"\nJSON: {json.dumps(result)}")

    engine.terminate()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLC LLM benchmark")
    parser.add_argument("--server", default=None, help="MLC LLM server URL (e.g. http://localhost:8001)")
    parser.add_argument("--native", default=None, help="Model path for native MLC LLM engine benchmark")
    parser.add_argument("--model", default="HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC", help="Model name/path")
    parser.add_argument("--num-tokens", type=int, default=25, help="Number of tokens to generate")
    parser.add_argument("--prompt", default="Hello, how are you today?", help="Prompt text")

    # Legacy positional args support
    args, remaining = parser.parse_known_args()
    if remaining and not args.server and not args.native:
        args.native = remaining[0]
        if len(remaining) > 1:
            args.num_tokens = int(remaining[1])

    if args.server:
        benchmark_server(args.server, args.model, args.num_tokens, args.prompt)
    elif args.native:
        benchmark_native(args.native, args.num_tokens, args.prompt)
    else:
        # Default: try server mode on localhost:8001
        print("No --server or --native specified. Trying server at http://localhost:8001...")
        benchmark_server("http://localhost:8001", args.model, args.num_tokens, args.prompt)
