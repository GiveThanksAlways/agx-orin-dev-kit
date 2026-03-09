/**
 * bench_trt_variants.cpp — Benchmark TensorRT inference variants on Jetson Orin
 *
 * Tests 4 TRT configurations on the Cioffi TCN (250K params):
 *   1. Stock:           cudaMemcpyAsync H2D → enqueueV3 → cudaMemcpyAsync D2H
 *   2. Zero-Copy:       cudaMallocManaged → memcpy → enqueueV3 (no cudaMemcpyAsync)
 *   3. CUDA Graph:      cudaMalloc + cudaGraphLaunch (captured enqueueV3)
 *   4. ZC + Graph:      cudaMallocManaged + cudaGraphLaunch (both optimizations)
 *
 * Compile (inside nix develop):
 *   clang++ -O3 -std=c++17 bench_trt_variants.cpp \
 *     -I$TENSORRT_PATH/include -I$CUDA_PATH/include \
 *     -L$(dirname $(readlink -f $(which trtexec)))/../lib \
 *     -lnvinfer -lcudart -o bench_trt_variants
 *
 * Usage:
 *   ./bench_trt_variants <engine_path> [iterations=5000] [warmup=30]
 */

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <vector>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ─── Minimal TRT Logger ───────────────────────────────────────────────────────
class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING)
            fprintf(stderr, "[TRT] %s\n", msg);
    }
};

// ─── Stats ────────────────────────────────────────────────────────────────────
struct Stats {
    double median, mean, p99, p999, min, max, std_dev;
    int count;
};

Stats compute_stats(std::vector<double> &times) {
    std::sort(times.begin(), times.end());
    int n = times.size();
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / n;
    double sq_sum = 0;
    for (auto t : times) sq_sum += (t - mean) * (t - mean);

    return {
        .median = times[n / 2],
        .mean = mean,
        .p99 = times[(int)(n * 0.99)],
        .p999 = times[(int)(n * 0.999)],
        .min = times[0],
        .max = times[n - 1],
        .std_dev = std::sqrt(sq_sum / n),
        .count = n,
    };
}

void print_stats(const char *label, Stats &s) {
    double hz = s.median > 0 ? 1e6 / s.median : 0;
    printf("  %-40s median=%8.1f µs  p99=%8.1f  p999=%8.1f  max=%9.1f  freq=%8.0f Hz\n",
           label, s.median, s.p99, s.p999, s.max, hz);
}

// ─── Engine loader ────────────────────────────────────────────────────────────
nvinfer1::ICudaEngine *load_engine(const char *path, nvinfer1::IRuntime *runtime) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) {
        fprintf(stderr, "Cannot open engine: %s\n", path);
        exit(1);
    }
    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(size);
    f.read(buf.data(), size);
    return runtime->deserializeCudaEngine(buf.data(), size);
}

// ─── Get tensor sizes ─────────────────────────────────────────────────────────
size_t tensor_bytes(nvinfer1::ICudaEngine *engine, const char *name) {
    auto dims = engine->getTensorShape(name);
    size_t n = 1;
    for (int i = 0; i < dims.nbDims; i++) n *= dims.d[i];
    auto dtype = engine->getTensorDataType(name);
    size_t elem = (dtype == nvinfer1::DataType::kFLOAT) ? 4 :
                  (dtype == nvinfer1::DataType::kHALF)  ? 2 :
                  (dtype == nvinfer1::DataType::kINT8)   ? 1 : 4;
    return n * elem;
}

// ─── JSON output helper ───────────────────────────────────────────────────────
void print_json_stats(const char *label, Stats &s) {
    printf("    \"%s\": {\"median\": %.1f, \"mean\": %.1f, \"p99\": %.1f, "
           "\"p999\": %.1f, \"min\": %.1f, \"max\": %.1f, \"std\": %.1f, \"count\": %d}",
           label, s.median, s.mean, s.p99, s.p999, s.min, s.max, s.std_dev, s.count);
}

// ═════════════════════════════════════════════════════════════════════════════
// Variant 1: Stock TRT (cudaMalloc + cudaMemcpyAsync)
// ═════════════════════════════════════════════════════════════════════════════
Stats bench_stock(nvinfer1::ICudaEngine *engine, int warmup, int n_iters,
                  bool full_roundtrip) {
    auto *ctx = engine->createExecutionContext();

    const char *in_name = "imu_data";
    const char *out_name = "displacement";
    size_t in_bytes = tensor_bytes(engine, in_name);
    size_t out_bytes = tensor_bytes(engine, out_name);

    void *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, in_bytes));
    CHECK_CUDA(cudaMalloc(&d_out, out_bytes));

    std::vector<float> h_in(in_bytes / sizeof(float), 0.1f);
    std::vector<float> h_out(out_bytes / sizeof(float));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    ctx->setTensorAddress(in_name, d_in);
    ctx->setTensorAddress(out_name, d_out);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        CHECK_CUDA(cudaMemcpyAsync(d_in, h_in.data(), in_bytes,
                                    cudaMemcpyHostToDevice, stream));
        ctx->enqueueV3(stream);
        CHECK_CUDA(cudaMemcpyAsync(h_out.data(), d_out, out_bytes,
                                    cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // Benchmark
    std::vector<double> times(n_iters);
    for (int i = 0; i < n_iters; i++) {
        if (full_roundtrip) {
            auto t0 = std::chrono::high_resolution_clock::now();
            CHECK_CUDA(cudaMemcpyAsync(d_in, h_in.data(), in_bytes,
                                        cudaMemcpyHostToDevice, stream));
            ctx->enqueueV3(stream);
            CHECK_CUDA(cudaMemcpyAsync(h_out.data(), d_out, out_bytes,
                                        cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
            auto t1 = std::chrono::high_resolution_clock::now();
            times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        } else {
            CHECK_CUDA(cudaMemcpyAsync(d_in, h_in.data(), in_bytes,
                                        cudaMemcpyHostToDevice, stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
            auto t0 = std::chrono::high_resolution_clock::now();
            ctx->enqueueV3(stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));
            auto t1 = std::chrono::high_resolution_clock::now();
            times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
    }

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaStreamDestroy(stream));
    delete ctx;

    return compute_stats(times);
}

// ═════════════════════════════════════════════════════════════════════════════
// Variant 2: Tegra Zero-Copy (cudaMallocManaged — no cudaMemcpyAsync)
// ═════════════════════════════════════════════════════════════════════════════
Stats bench_zerocopy(nvinfer1::ICudaEngine *engine, int warmup, int n_iters,
                     bool full_roundtrip) {
    auto *ctx = engine->createExecutionContext();

    const char *in_name = "imu_data";
    const char *out_name = "displacement";
    size_t in_bytes = tensor_bytes(engine, in_name);
    size_t out_bytes = tensor_bytes(engine, out_name);

    // Key: use cudaMallocManaged instead of cudaMalloc
    void *d_in, *d_out;
    CHECK_CUDA(cudaMallocManaged(&d_in, in_bytes));
    CHECK_CUDA(cudaMallocManaged(&d_out, out_bytes));

    std::vector<float> h_in(in_bytes / sizeof(float), 0.1f);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    ctx->setTensorAddress(in_name, d_in);
    ctx->setTensorAddress(out_name, d_out);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        std::memcpy(d_in, h_in.data(), in_bytes);
        ctx->enqueueV3(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // Benchmark
    std::vector<double> times(n_iters);
    for (int i = 0; i < n_iters; i++) {
        if (full_roundtrip) {
            auto t0 = std::chrono::high_resolution_clock::now();
            std::memcpy(d_in, h_in.data(), in_bytes);
            ctx->enqueueV3(stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));
            // read-back: just access managed memory directly
            volatile float out0 = ((float *)d_out)[0];
            (void)out0;
            auto t1 = std::chrono::high_resolution_clock::now();
            times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        } else {
            std::memcpy(d_in, h_in.data(), in_bytes);
            auto t0 = std::chrono::high_resolution_clock::now();
            ctx->enqueueV3(stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));
            auto t1 = std::chrono::high_resolution_clock::now();
            times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
    }

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaStreamDestroy(stream));
    delete ctx;

    return compute_stats(times);
}

// ═════════════════════════════════════════════════════════════════════════════
// Variant 3: CUDA Graph (capture enqueueV3 → replay with cudaGraphLaunch)
// ═════════════════════════════════════════════════════════════════════════════
Stats bench_cuda_graph(nvinfer1::ICudaEngine *engine, int warmup, int n_iters) {
    auto *ctx = engine->createExecutionContext();

    const char *in_name = "imu_data";
    const char *out_name = "displacement";
    size_t in_bytes = tensor_bytes(engine, in_name);
    size_t out_bytes = tensor_bytes(engine, out_name);

    void *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, in_bytes));
    CHECK_CUDA(cudaMalloc(&d_out, out_bytes));

    std::vector<float> h_in(in_bytes / sizeof(float), 0.1f);
    std::vector<float> h_out(out_bytes / sizeof(float));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    ctx->setTensorAddress(in_name, d_in);
    ctx->setTensorAddress(out_name, d_out);

    // Warmup (regular) before capture
    for (int i = 0; i < 3; i++) {
        CHECK_CUDA(cudaMemcpyAsync(d_in, h_in.data(), in_bytes,
                                    cudaMemcpyHostToDevice, stream));
        ctx->enqueueV3(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // Capture CUDA graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    ctx->enqueueV3(stream);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    printf("  CUDA graph captured\n");

    // Warmup with graph replay
    for (int i = 0; i < warmup; i++) {
        CHECK_CUDA(cudaMemcpyAsync(d_in, h_in.data(), in_bytes,
                                    cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // Benchmark: GPU-only timing (graph launch)
    std::vector<double> times(n_iters);
    for (int i = 0; i < n_iters; i++) {
        CHECK_CUDA(cudaMemcpyAsync(d_in, h_in.data(), in_bytes,
                                    cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        auto t0 = std::chrono::high_resolution_clock::now();
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaStreamDestroy(stream));
    delete ctx;

    return compute_stats(times);
}

// ═════════════════════════════════════════════════════════════════════════════
// Variant 4: Zero-Copy + CUDA Graph (both optimizations)
// ═════════════════════════════════════════════════════════════════════════════
Stats bench_zerocopy_cuda_graph(nvinfer1::ICudaEngine *engine, int warmup, int n_iters) {
    auto *ctx = engine->createExecutionContext();

    const char *in_name = "imu_data";
    const char *out_name = "displacement";
    size_t in_bytes = tensor_bytes(engine, in_name);
    size_t out_bytes = tensor_bytes(engine, out_name);

    void *d_in, *d_out;
    CHECK_CUDA(cudaMallocManaged(&d_in, in_bytes));
    CHECK_CUDA(cudaMallocManaged(&d_out, out_bytes));

    std::vector<float> h_in(in_bytes / sizeof(float), 0.1f);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    ctx->setTensorAddress(in_name, d_in);
    ctx->setTensorAddress(out_name, d_out);

    // Warmup before capture
    for (int i = 0; i < 3; i++) {
        std::memcpy(d_in, h_in.data(), in_bytes);
        ctx->enqueueV3(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // Capture
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    ctx->enqueueV3(stream);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    printf("  CUDA graph captured (zero-copy + graph)\n");

    // Warmup
    for (int i = 0; i < warmup; i++) {
        std::memcpy(d_in, h_in.data(), in_bytes);
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // Benchmark
    std::vector<double> times(n_iters);
    for (int i = 0; i < n_iters; i++) {
        std::memcpy(d_in, h_in.data(), in_bytes);
        auto t0 = std::chrono::high_resolution_clock::now();
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaStreamDestroy(stream));
    delete ctx;

    return compute_stats(times);
}

// ═════════════════════════════════════════════════════════════════════════════
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <engine_path> [iterations=5000] [warmup=30]\n", argv[0]);
        return 1;
    }

    const char *engine_path = argv[1];
    int n_iters = argc > 2 ? atoi(argv[2]) : 5000;
    int warmup = argc > 3 ? atoi(argv[3]) : 30;

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("TensorRT Variant Benchmark — Cioffi TCN on Jetson AGX Orin\n");
    printf("  Engine: %s\n", engine_path);
    printf("  Iterations: %d (warmup: %d)\n", n_iters, warmup);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    TrtLogger logger;
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine *engine = load_engine(engine_path, runtime);
    if (!engine) {
        fprintf(stderr, "Failed to load engine\n");
        return 1;
    }

    // Print I/O info
    const char *in_name = "imu_data";
    const char *out_name = "displacement";
    printf("  Input:  %s (%zu bytes)\n", in_name, tensor_bytes(engine, in_name));
    printf("  Output: %s (%zu bytes)\n\n", out_name, tensor_bytes(engine, out_name));

    // ── Variant 1: Stock (GPU-only timing) ──
    printf("─── [1] TRT Stock (cudaMalloc + cudaMemcpyAsync) ───\n");
    {
        printf("  GPU-only timing (enqueueV3 + sync):\n");
        auto s = bench_stock(engine, warmup, n_iters, false);
        print_stats("TRT Stock GPU-only", s);

        printf("  Full round-trip (H2D + compute + D2H):\n");
        auto s_full = bench_stock(engine, warmup, n_iters, true);
        print_stats("TRT Stock full", s_full);
    }
    printf("\n");

    // ── Variant 2: Zero-Copy (cudaMallocManaged) ──
    printf("─── [2] TRT Zero-Copy (cudaMallocManaged) ───\n");
    {
        printf("  GPU-only timing (enqueueV3 + sync):\n");
        auto s = bench_zerocopy(engine, warmup, n_iters, false);
        print_stats("TRT Zero-Copy GPU-only", s);

        printf("  Full round-trip (memcpy + compute + read):\n");
        auto s_full = bench_zerocopy(engine, warmup, n_iters, true);
        print_stats("TRT Zero-Copy full", s_full);
    }
    printf("\n");

    // ── Variant 3: CUDA Graph ──
    printf("─── [3] TRT + CUDA Graph (cudaGraphLaunch) ───\n");
    {
        auto s = bench_cuda_graph(engine, warmup, n_iters);
        print_stats("TRT CUDA Graph", s);
    }
    printf("\n");

    // ── Variant 4: Zero-Copy + CUDA Graph ──
    printf("─── [4] TRT Zero-Copy + CUDA Graph ───\n");
    {
        auto s = bench_zerocopy_cuda_graph(engine, warmup, n_iters);
        print_stats("TRT ZC + Graph", s);
    }
    printf("\n");

    // ── Summary ──
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("SUMMARY\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    {
        auto stock_gpu = bench_stock(engine, 10, 2000, false);
        auto stock_full = bench_stock(engine, 10, 2000, true);
        auto zc_gpu = bench_zerocopy(engine, 10, 2000, false);
        auto zc_full = bench_zerocopy(engine, 10, 2000, true);
        auto graph = bench_cuda_graph(engine, 10, 2000);
        auto zc_graph = bench_zerocopy_cuda_graph(engine, 10, 2000);

        printf("  %-40s %10s %10s %10s %10s\n", "Variant", "Median µs", "P99 µs", "Max µs", "Hz");
        printf("  %-40s %10s %10s %10s %10s\n", "────────────────────────────────────────",
               "──────────", "──────────", "──────────", "──────────");

        auto row = [](const char *label, Stats &s) {
            double hz = s.median > 0 ? 1e6 / s.median : 0;
            printf("  %-40s %10.1f %10.1f %10.1f %10.0f\n",
                   label, s.median, s.p99, s.max, hz);
        };
        row("TRT Stock (GPU-only)", stock_gpu);
        row("TRT Stock (full round-trip)", stock_full);
        row("TRT Zero-Copy (GPU-only)", zc_gpu);
        row("TRT Zero-Copy (full round-trip)", zc_full);
        row("TRT CUDA Graph", graph);
        row("TRT Zero-Copy + CUDA Graph", zc_graph);

        printf("\n");
        printf("  Speedup from zero-copy (full):  %.2fx\n",
               stock_full.median / zc_full.median);
        printf("  Speedup from CUDA graph:        %.2fx\n",
               stock_gpu.median / graph.median);
        printf("  Speedup from ZC + graph:        %.2fx (vs stock full)\n",
               stock_full.median / zc_graph.median);

        // JSON output for scripting
        printf("\n{\"results\": {\n");
        print_json_stats("stock_gpu", stock_gpu); printf(",\n");
        print_json_stats("stock_full", stock_full); printf(",\n");
        print_json_stats("zerocopy_gpu", zc_gpu); printf(",\n");
        print_json_stats("zerocopy_full", zc_full); printf(",\n");
        print_json_stats("cuda_graph", graph); printf(",\n");
        print_json_stats("zerocopy_cuda_graph", zc_graph); printf("\n");
        printf("}}\n");
    }

    delete engine;
    delete runtime;
    return 0;
}
