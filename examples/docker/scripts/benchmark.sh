#!/bin/bash
# Benchmark TensorRT-LLM inference performance
# Usage: ./benchmark.sh engine-name [num-iterations]

set -e

ENGINE_NAME=${1:-qwen3}
ITERATIONS=${2:-10}
ENGINE_DIR="/workspace/engines/$ENGINE_NAME"

if [ ! -d "$ENGINE_DIR" ]; then
    echo "❌ Engine not found: $ENGINE_DIR"
    echo "   Build it first with: ./convert-and-run.sh $ENGINE_NAME"
    exit 1
fi

echo "⚡ TensorRT-LLM Performance Benchmark"
echo "====================================="
echo "Engine: $ENGINE_NAME"
echo "Iterations: $ITERATIONS"
echo

PROMPT="Write a detailed explanation of machine learning in 200 words."

echo "Running warmup..."
python /workspace/run-model.py \
    --engine "$ENGINE_DIR" \
    --prompt "$PROMPT" \
    --max-tokens 200 > /dev/null 2>&1

echo "Running benchmark..."
echo

total_time=0
for i in $(seq 1 $ITERATIONS); do
    echo -n "Run $i/$ITERATIONS... "
    start=$(date +%s.%N)
    
    python /workspace/run-model.py \
        --engine "$ENGINE_DIR" \
        --prompt "$PROMPT" \
        --max-tokens 200 > /dev/null 2>&1
    
    end=$(date +%s.%N)
    runtime=$(echo "$end - $start" | bc)
    total_time=$(echo "$total_time + $runtime" | bc)
    
    echo "${runtime}s"
done

echo
avg_time=$(echo "scale=2; $total_time / $ITERATIONS" | bc)
tokens_per_sec=$(echo "scale=2; 200 / $avg_time" | bc)

echo "📊 Results:"
echo "   Average time: ${avg_time}s"
echo "   Tokens/sec: $tokens_per_sec"
echo "   Total time: ${total_time}s"
echo

if (( $(echo "$tokens_per_sec > 50" | bc -l) )); then
    echo "🚀 Excellent performance!"
elif (( $(echo "$tokens_per_sec > 20" | bc -l) )); then
    echo "✅ Good performance"
else
    echo "⚠️  Consider INT8/INT4 quantization for faster inference"
fi
