/**
 * bench_multi_model.cpp — Multi-model layered benchmark for Jetson Orin
 *
 * Simulates Saronic's real perception pipeline:
 *   Detection (YOLOv8n) → Segmentation (YOLOv8n-seg) → Pose (YOLOv8n-pose)
 *
 * Compares 3 execution strategies:
 *   1. Sequential:  3x enqueueV3 + sync between each
 *   2. Pipelined:   3x enqueueV3 on same stream, one sync at end
 *   3. CUDA Graph:  Capture all 3 enqueueV3 → single cudaGraphLaunch
 *
 * Compile:
 *   clang++ -O3 -std=c++17 bench_multi_model.cpp \
 *     -I$TENSORRT_PATH/include -I$CUDA_PATH/include \
 *     -L$(dirname $(readlink -f $(which trtexec)))/../lib \
 *     -lnvinfer -lcudart -o bench_multi_model
 *
 * Usage:
 *   ./bench_multi_model <det.engine> <seg.engine> <pose.engine> [iters=5000] [warmup=50]
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

class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING)
            fprintf(stderr, "[TRT] %s\n", msg);
    }
};

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
    printf("  %-44s median=%8.1f µs  p99=%8.1f  p999=%8.1f  max=%9.1f  freq=%6.0f Hz\n",
           label, s.median, s.p99, s.p999, s.max, hz);
}

void print_json_stats(const char *label, Stats &s) {
    printf("    \"%s\": {\"median\": %.1f, \"mean\": %.1f, \"p99\": %.1f, "
           "\"p999\": %.1f, \"min\": %.1f, \"max\": %.1f, \"std\": %.1f, \"count\": %d}",
           label, s.median, s.mean, s.p99, s.p999, s.min, s.max, s.std_dev, s.count);
}

nvinfer1::ICudaEngine *load_engine(const char *path, nvinfer1::IRuntime *runtime) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) { fprintf(stderr, "Cannot open: %s\n", path); exit(1); }
    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(size);
    f.read(buf.data(), size);
    return runtime->deserializeCudaEngine(buf.data(), size);
}

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

// Per-model state
struct ModelState {
    const char *label;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *ctx;
    std::vector<void *> bufs;   // GPU/managed buffers
    std::vector<size_t> sizes;
};

ModelState setup_model(const char *label, const char *path,
                       nvinfer1::IRuntime *runtime, cudaStream_t stream, bool managed) {
    ModelState m;
    m.label = label;
    m.engine = load_engine(path, runtime);
    if (!m.engine) { fprintf(stderr, "Failed to load: %s\n", path); exit(1); }
    m.ctx = m.engine->createExecutionContext();

    for (int i = 0; i < m.engine->getNbIOTensors(); i++) {
        const char *name = m.engine->getIOTensorName(i);
        size_t bytes = tensor_bytes(m.engine, name);
        void *ptr;
        if (managed) {
            CHECK_CUDA(cudaMallocManaged(&ptr, bytes));
        } else {
            CHECK_CUDA(cudaMalloc(&ptr, bytes));
        }
        CHECK_CUDA(cudaMemset(ptr, 0, bytes));
        m.ctx->setTensorAddress(name, ptr);
        m.bufs.push_back(ptr);
        m.sizes.push_back(bytes);
    }

    // Print I/O
    printf("  %-12s ", label);
    size_t total = 0;
    for (int i = 0; i < m.engine->getNbIOTensors(); i++) {
        const char *name = m.engine->getIOTensorName(i);
        auto mode = m.engine->getTensorIOMode(name);
        size_t bytes = tensor_bytes(m.engine, name);
        total += bytes;
    }
    printf("%zu tensors, %.1f KB total I/O\n", m.bufs.size(), total / 1024.0);
    return m;
}

void free_model(ModelState &m) {
    for (auto p : m.bufs) cudaFree(p);
    delete m.ctx;
    delete m.engine;
}

// ═══════════════════════════════════════════════════════════════════════
// Strategy 1: Sequential — sync after each model (worst case)
// ═══════════════════════════════════════════════════════════════════════
Stats bench_sequential(ModelState *models, int n_models, cudaStream_t stream,
                       int warmup, int n_iters) {
    for (int w = 0; w < warmup; w++) {
        for (int m = 0; m < n_models; m++) {
            models[m].ctx->enqueueV3(stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));
        }
    }
    std::vector<double> times(n_iters);
    for (int i = 0; i < n_iters; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int m = 0; m < n_models; m++) {
            models[m].ctx->enqueueV3(stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    return compute_stats(times);
}

// ═══════════════════════════════════════════════════════════════════════
// Strategy 2: Pipelined — all enqueues on same stream, one sync
// ═══════════════════════════════════════════════════════════════════════
Stats bench_pipelined(ModelState *models, int n_models, cudaStream_t stream,
                      int warmup, int n_iters) {
    for (int w = 0; w < warmup; w++) {
        for (int m = 0; m < n_models; m++)
            models[m].ctx->enqueueV3(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    std::vector<double> times(n_iters);
    for (int i = 0; i < n_iters; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int m = 0; m < n_models; m++)
            models[m].ctx->enqueueV3(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    return compute_stats(times);
}

// ═══════════════════════════════════════════════════════════════════════
// Strategy 3: Single CUDA Graph capturing all models
// ═══════════════════════════════════════════════════════════════════════
Stats bench_graph(ModelState *models, int n_models, cudaStream_t stream,
                  int warmup, int n_iters) {
    // Warmup before capture
    for (int w = 0; w < 5; w++) {
        for (int m = 0; m < n_models; m++)
            models[m].ctx->enqueueV3(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // Capture
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int m = 0; m < n_models; m++)
        models[m].ctx->enqueueV3(stream);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    size_t numNodes = 0;
    CHECK_CUDA(cudaGraphGetNodes(graph, nullptr, &numNodes));
    printf("  CUDA Graph captured: %zu nodes across %d models\n", numNodes, n_models);

    // Warmup with graph
    for (int w = 0; w < warmup; w++) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    std::vector<double> times(n_iters);
    for (int i = 0; i < n_iters; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    return compute_stats(times);
}

// ═══════════════════════════════════════════════════════════════════════
int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <det.engine> <seg.engine> <pose.engine> [iters=5000] [warmup=50]\n"
            "\n"
            "Benchmarks a 3-model perception pipeline:\n"
            "  Detection → Segmentation → Pose Estimation\n"
            "\n"
            "Tests Sequential (sync between each), Pipelined (one sync),\n"
            "and CUDA Graph (single graph launch) execution strategies.\n",
            argv[0]);
        return 1;
    }

    const char *det_path = argv[1];
    const char *seg_path = argv[2];
    const char *pose_path = argv[3];
    int n_iters = argc > 4 ? atoi(argv[4]) : 5000;
    int warmup = argc > 5 ? atoi(argv[5]) : 50;

    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("Multi-Model Layered Pipeline Benchmark — Jetson AGX Orin\n");
    printf("  Detection  → %s\n", det_path);
    printf("  Segmentation → %s\n", seg_path);
    printf("  Pose         → %s\n", pose_path);
    printf("  Iterations: %d (warmup: %d)\n", n_iters, warmup);
    printf("═══════════════════════════════════════════════════════════════════════\n\n");

    TrtLogger logger;
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    printf("Loading engines:\n");
    ModelState det = setup_model("Detection", det_path, runtime, stream, false);
    ModelState seg = setup_model("Segmentation", seg_path, runtime, stream, false);
    ModelState pose = setup_model("Pose", pose_path, runtime, stream, false);
    ModelState models[] = {det, seg, pose};
    printf("\n");

    // ── Strategy 1: Sequential ──
    printf("─── [1] Sequential (sync between each model) ───\n");
    printf("  Simulates naive pipeline: enqueue→sync→enqueue→sync→enqueue→sync\n");
    auto s_seq = bench_sequential(models, 3, stream, warmup, n_iters);
    print_stats("Sequential (3x enqueue+sync)", s_seq);
    printf("\n");

    // ── Strategy 2: Pipelined ──
    printf("─── [2] Pipelined (all enqueues, one sync) ───\n");
    printf("  Better pipeline: enqueue→enqueue→enqueue→sync\n");
    auto s_pipe = bench_pipelined(models, 3, stream, warmup, n_iters);
    print_stats("Pipelined (3x enqueue, 1 sync)", s_pipe);
    printf("\n");

    // ── Strategy 3: CUDA Graph ──
    printf("─── [3] CUDA Graph (single graph captures all 3 models) ───\n");
    printf("  Optimal: one cudaGraphLaunch replays det+seg+pose\n");
    auto s_graph = bench_graph(models, 3, stream, warmup, n_iters);
    print_stats("CUDA Graph (3 models, 1 launch)", s_graph);
    printf("\n");

    // ── Summary ──
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("SUMMARY — 3-model perception pipeline\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");
    {
        // Fresh runs for clean numbers
        auto seq = bench_sequential(models, 3, stream, 10, 2000);
        auto pipe = bench_pipelined(models, 3, stream, 10, 2000);
        auto graph = bench_graph(models, 3, stream, 10, 2000);

        printf("  %-44s %10s %10s %10s %10s\n", "Strategy", "Median µs", "P99 µs", "Max µs", "Hz");
        printf("  %-44s %10s %10s %10s %10s\n",
               "────────────────────────────────────────────",
               "──────────", "──────────", "──────────", "──────────");

        auto row = [](const char *label, Stats &s) {
            double hz = s.median > 0 ? 1e6 / s.median : 0;
            printf("  %-44s %10.1f %10.1f %10.1f %10.0f\n",
                   label, s.median, s.p99, s.max, hz);
        };
        row("Sequential (3x sync)", seq);
        row("Pipelined (1 sync)", pipe);
        row("CUDA Graph (1 launch)", graph);

        printf("\n");
        printf("  Speedup: Pipelined vs Sequential: %.2fx\n", seq.median / pipe.median);
        printf("  Speedup: Graph vs Sequential:     %.2fx\n", seq.median / graph.median);
        printf("  Speedup: Graph vs Pipelined:      %.2fx\n", pipe.median / graph.median);
        printf("\n");
        printf("  Jitter reduction (p99/median):\n");
        printf("    Sequential: %.4f  Pipelined: %.4f  Graph: %.4f\n",
               seq.p99 / seq.median, pipe.p99 / pipe.median, graph.p99 / graph.median);

        // JSON
        printf("\n{\"pipeline\": {\n");
        print_json_stats("sequential", seq); printf(",\n");
        print_json_stats("pipelined", pipe); printf(",\n");
        print_json_stats("graph", graph); printf("\n");
        printf("}}\n");
    }

    free_model(det);
    free_model(seg);
    free_model(pose);
    CHECK_CUDA(cudaStreamDestroy(stream));
    delete runtime;
    return 0;
}
