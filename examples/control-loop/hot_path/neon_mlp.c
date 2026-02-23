/*
 * neon_mlp.c — ARM NEON FP16 MLP forward pass for Cortex-A78AE.
 *
 * Matrix-vector multiply using float16x8_t FMLA instructions.
 * Weights are padded to multiples of 8 for clean vectorization.
 *
 * Build: gcc -O3 -march=armv8.2-a+fp16 -mtune=cortex-a78ae -shared -fPIC \
 *            -o neon_mlp.so neon_mlp.c
 */
#include "neon_mlp.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline int pad8(int x) { return (x + 7) & ~7; }

static inline uint64_t timespec_to_ns(struct timespec *ts) {
    return (uint64_t)ts->tv_sec * 1000000000ULL + (uint64_t)ts->tv_nsec;
}

/* Horizontal sum of float16x8_t → scalar __fp16 */
static inline __fp16 hsum_f16x8(float16x8_t v) {
    float16x4_t lo = vget_low_f16(v);
    float16x4_t hi = vget_high_f16(v);
    float16x4_t sum4 = vadd_f16(lo, hi);
    float16x4_t sum2 = vpadd_f16(sum4, sum4);
    float16x4_t sum1 = vpadd_f16(sum2, sum2);
    return vget_lane_f16(sum1, 0);
}

int neon_mlp_init(neon_mlp_t *mlp, int n_layers, const int *dims) {
    if (n_layers > NEON_MLP_MAX_LAYERS || n_layers < 1) return -1;
    for (int i = 0; i <= n_layers; i++) {
        if (dims[i] > NEON_MLP_MAX_DIM || dims[i] < 1) return -1;
    }

    mlp->n_layers = n_layers;
    memcpy(mlp->dims, dims, (n_layers + 1) * sizeof(int));

    for (int l = 0; l < n_layers; l++) {
        int in_padded = pad8(dims[l]);
        int out_dim = dims[l + 1];

        /* Allocate aligned weight matrix: [out_dim][in_padded] */
        mlp->weights[l] = (__fp16 *)aligned_alloc(64, out_dim * in_padded * sizeof(__fp16));
        mlp->biases[l]  = (__fp16 *)aligned_alloc(64, pad8(out_dim) * sizeof(__fp16));
        if (!mlp->weights[l] || !mlp->biases[l]) return -1;

        /* Zero-fill (padding must be 0 for correct dot products) */
        memset(mlp->weights[l], 0, out_dim * in_padded * sizeof(__fp16));
        memset(mlp->biases[l],  0, pad8(out_dim) * sizeof(__fp16));
    }

    memset(mlp->scratch_a, 0, sizeof(mlp->scratch_a));
    memset(mlp->scratch_b, 0, sizeof(mlp->scratch_b));
    return 0;
}

void neon_mlp_load_layer(neon_mlp_t *mlp, int layer,
                         const __fp16 *weight_data,
                         const __fp16 *bias_data) {
    int in_dim = mlp->dims[layer];
    int in_padded = pad8(in_dim);
    int out_dim = mlp->dims[layer + 1];

    /* Copy weights row by row, padding each row to in_padded */
    for (int i = 0; i < out_dim; i++) {
        memcpy(mlp->weights[layer] + i * in_padded,
               weight_data + i * in_dim,
               in_dim * sizeof(__fp16));
        /* Padding is already zeroed from init */
    }

    /* Copy biases */
    memcpy(mlp->biases[layer], bias_data, out_dim * sizeof(__fp16));
}

uint64_t neon_mlp_forward(neon_mlp_t *mlp,
                          const __fp16 *input,
                          __fp16 *output) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    const __fp16 *in = input;
    __fp16 *out;
    int use_a = 1; /* alternate scratch buffers */

    for (int l = 0; l < mlp->n_layers; l++) {
        int in_dim   = mlp->dims[l];
        int in_padded = pad8(in_dim);
        int out_dim  = mlp->dims[l + 1];
        const __fp16 *W = mlp->weights[l];
        const __fp16 *b = mlp->biases[l];

        /* Last layer → write directly to output */
        if (l == mlp->n_layers - 1) {
            out = output;
        } else {
            out = use_a ? mlp->scratch_a : mlp->scratch_b;
            use_a = !use_a;
        }

        /*
         * For small in_dim (≤ 16), unrolled scalar is faster than
         * the NEON path due to reduced instruction overhead.
         * But NEON is what we're benchmarking, so always use it.
         */
        for (int i = 0; i < out_dim; i++) {
            const __fp16 *w_row = W + i * in_padded;
            float16x8_t acc0 = vdupq_n_f16(0);
            float16x8_t acc1 = vdupq_n_f16(0);
            float16x8_t acc2 = vdupq_n_f16(0);
            float16x8_t acc3 = vdupq_n_f16(0);

            int j = 0;
            /* Unrolled 4×8 = 32 elements per iteration */
            for (; j + 32 <= in_padded; j += 32) {
                acc0 = vfmaq_f16(acc0, vld1q_f16(w_row + j),      vld1q_f16(in + j));
                acc1 = vfmaq_f16(acc1, vld1q_f16(w_row + j + 8),  vld1q_f16(in + j + 8));
                acc2 = vfmaq_f16(acc2, vld1q_f16(w_row + j + 16), vld1q_f16(in + j + 16));
                acc3 = vfmaq_f16(acc3, vld1q_f16(w_row + j + 24), vld1q_f16(in + j + 24));
            }
            /* Remaining 8-element blocks */
            for (; j + 8 <= in_padded; j += 8) {
                acc0 = vfmaq_f16(acc0, vld1q_f16(w_row + j), vld1q_f16(in + j));
            }

            /* Reduce accumulators */
            acc0 = vaddq_f16(vaddq_f16(acc0, acc1), vaddq_f16(acc2, acc3));
            __fp16 sum = hsum_f16x8(acc0) + b[i];

            /* ReLU for all layers except the last */
            if (l < mlp->n_layers - 1 && sum < (__fp16)0) sum = (__fp16)0;

            out[i] = sum;
        }

        /* Input for next layer is this layer's output.
         * If out_dim isn't a multiple of 8, pad the remaining
         * scratch entries with 0 (needed for next layer's NEON loads). */
        if (l < mlp->n_layers - 1) {
            int padded_out = pad8(out_dim);
            for (int k = out_dim; k < padded_out; k++) out[k] = (__fp16)0;
        }

        in = out;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    return timespec_to_ns(&t1) - timespec_to_ns(&t0);
}

void neon_mlp_benchmark(neon_mlp_t *mlp,
                        const __fp16 *input_pool,
                        __fp16 *output_pool,
                        uint32_t n_iters,
                        uint64_t *times_ns) {
    int in_dim  = mlp->dims[0];
    int out_dim = mlp->dims[mlp->n_layers];

    for (uint32_t i = 0; i < n_iters; i++) {
        times_ns[i] = neon_mlp_forward(
            mlp,
            input_pool + (size_t)i * in_dim,
            output_pool + (size_t)i * out_dim
        );
    }
}

void neon_mlp_free(neon_mlp_t *mlp) {
    for (int l = 0; l < mlp->n_layers; l++) {
        free(mlp->weights[l]);
        free(mlp->biases[l]);
        mlp->weights[l] = NULL;
        mlp->biases[l] = NULL;
    }
}
