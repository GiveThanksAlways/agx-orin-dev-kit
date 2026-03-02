/*
 * neon_mlp_f32.c — ARM NEON FP32 MLP forward pass for Cortex-A78AE.
 *
 * Float32 baseline using float32x4_t FMLA instructions.
 * Same structure as neon_mlp.c but with FP32 precision:
 *   - float32x4_t = 4 elements per 128-bit register (vs 8 for FP16)
 *   - 2x less SIMD throughput than FP16, but higher precision
 *   - Weights padded to multiples of 4 (vs 8 for FP16)
 *
 * Build: gcc -O3 -march=armv8.2-a+fp16 -mtune=cortex-a78ae -shared -fPIC \
 *            -o neon_mlp_f32.so neon_mlp_f32.c
 */
#include "neon_mlp_f32.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline int pad4(int x) { return (x + 3) & ~3; }

static inline uint64_t timespec_to_ns(struct timespec *ts) {
    return (uint64_t)ts->tv_sec * 1000000000ULL + (uint64_t)ts->tv_nsec;
}

/* Horizontal sum of float32x4_t → scalar float */
static inline float hsum_f32x4(float32x4_t v) {
    float32x2_t lo = vget_low_f32(v);
    float32x2_t hi = vget_high_f32(v);
    float32x2_t sum2 = vadd_f32(lo, hi);
    float32x2_t sum1 = vpadd_f32(sum2, sum2);
    return vget_lane_f32(sum1, 0);
}

int neon_mlp_f32_init(neon_mlp_f32_t *mlp, int n_layers, const int *dims) {
    if (n_layers > NEON_MLP_F32_MAX_LAYERS || n_layers < 1) return -1;
    for (int i = 0; i <= n_layers; i++) {
        if (dims[i] > NEON_MLP_F32_MAX_DIM || dims[i] < 1) return -1;
    }

    mlp->n_layers = n_layers;
    memcpy(mlp->dims, dims, (n_layers + 1) * sizeof(int));

    for (int l = 0; l < n_layers; l++) {
        int in_padded = pad4(dims[l]);
        int out_dim = dims[l + 1];

        mlp->weights[l] = (float *)aligned_alloc(64, out_dim * in_padded * sizeof(float));
        mlp->biases[l]  = (float *)aligned_alloc(64, pad4(out_dim) * sizeof(float));
        if (!mlp->weights[l] || !mlp->biases[l]) return -1;

        memset(mlp->weights[l], 0, out_dim * in_padded * sizeof(float));
        memset(mlp->biases[l],  0, pad4(out_dim) * sizeof(float));
    }

    memset(mlp->scratch_a, 0, sizeof(mlp->scratch_a));
    memset(mlp->scratch_b, 0, sizeof(mlp->scratch_b));
    return 0;
}

void neon_mlp_f32_load_layer(neon_mlp_f32_t *mlp, int layer,
                              const float *weight_data,
                              const float *bias_data) {
    int in_dim = mlp->dims[layer];
    int in_padded = pad4(in_dim);
    int out_dim = mlp->dims[layer + 1];

    for (int i = 0; i < out_dim; i++) {
        memcpy(mlp->weights[layer] + i * in_padded,
               weight_data + i * in_dim,
               in_dim * sizeof(float));
    }

    memcpy(mlp->biases[layer], bias_data, out_dim * sizeof(float));
}

uint64_t neon_mlp_f32_forward(neon_mlp_f32_t *mlp,
                               const float *input,
                               float *output) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    const float *in = input;
    float *out;
    int use_a = 1;

    for (int l = 0; l < mlp->n_layers; l++) {
        int in_dim   = mlp->dims[l];
        int in_padded = pad4(in_dim);
        int out_dim  = mlp->dims[l + 1];
        const float *W = mlp->weights[l];
        const float *b = mlp->biases[l];

        if (l == mlp->n_layers - 1) {
            out = output;
        } else {
            out = use_a ? mlp->scratch_a : mlp->scratch_b;
            use_a = !use_a;
        }

        for (int i = 0; i < out_dim; i++) {
            const float *w_row = W + i * in_padded;
            float32x4_t acc0 = vdupq_n_f32(0);
            float32x4_t acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0);
            float32x4_t acc3 = vdupq_n_f32(0);

            int j = 0;
            /* Unrolled 4×4 = 16 elements per iteration */
            for (; j + 16 <= in_padded; j += 16) {
                acc0 = vfmaq_f32(acc0, vld1q_f32(w_row + j),      vld1q_f32(in + j));
                acc1 = vfmaq_f32(acc1, vld1q_f32(w_row + j + 4),  vld1q_f32(in + j + 4));
                acc2 = vfmaq_f32(acc2, vld1q_f32(w_row + j + 8),  vld1q_f32(in + j + 8));
                acc3 = vfmaq_f32(acc3, vld1q_f32(w_row + j + 12), vld1q_f32(in + j + 12));
            }
            /* Remaining 4-element blocks */
            for (; j + 4 <= in_padded; j += 4) {
                acc0 = vfmaq_f32(acc0, vld1q_f32(w_row + j), vld1q_f32(in + j));
            }

            /* Reduce accumulators */
            acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float sum = hsum_f32x4(acc0) + b[i];

            /* ReLU for all layers except the last */
            if (l < mlp->n_layers - 1 && sum < 0.0f) sum = 0.0f;

            out[i] = sum;
        }

        if (l < mlp->n_layers - 1) {
            int padded_out = pad4(out_dim);
            for (int k = out_dim; k < padded_out; k++) out[k] = 0.0f;
        }

        in = out;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    return timespec_to_ns(&t1) - timespec_to_ns(&t0);
}

void neon_mlp_f32_benchmark(neon_mlp_f32_t *mlp,
                             const float *input_pool,
                             float *output_pool,
                             uint32_t n_iters,
                             uint64_t *times_ns) {
    int in_dim  = mlp->dims[0];
    int out_dim = mlp->dims[mlp->n_layers];

    for (uint32_t i = 0; i < n_iters; i++) {
        times_ns[i] = neon_mlp_f32_forward(
            mlp,
            input_pool + (size_t)i * in_dim,
            output_pool + (size_t)i * out_dim
        );
    }
}

void neon_mlp_f32_free(neon_mlp_f32_t *mlp) {
    for (int l = 0; l < mlp->n_layers; l++) {
        free(mlp->weights[l]);
        free(mlp->biases[l]);
        mlp->weights[l] = NULL;
        mlp->biases[l] = NULL;
    }
}
