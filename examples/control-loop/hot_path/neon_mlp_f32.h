/*
 * neon_mlp_f32.h — ARM NEON FP32 MLP forward pass for Cortex-A78AE (Orin).
 *
 * FP32 baseline for comparing against FP16 NEON and GPU dispatch.
 * Uses ARM NEON float32x4_t intrinsics (4 elements per vector vs 8 for FP16).
 * This represents the common real-world case where developers don't quantize to FP16.
 */
#ifndef NEON_MLP_F32_H
#define NEON_MLP_F32_H

#include <stdint.h>
#include <stddef.h>
#include <arm_neon.h>

#define NEON_MLP_F32_MAX_LAYERS 8
#define NEON_MLP_F32_MAX_DIM    2048

typedef struct {
    int n_layers;
    int dims[NEON_MLP_F32_MAX_LAYERS + 1];

    /* Weights stored row-major: W[layer][out_dim][in_dim_padded] */
    /* in_dim_padded = round_up(in_dim, 4) for NEON float32x4_t alignment */
    float *weights[NEON_MLP_F32_MAX_LAYERS];
    float *biases[NEON_MLP_F32_MAX_LAYERS];

    /* Scratch buffers for intermediate activations */
    float scratch_a[NEON_MLP_F32_MAX_DIM];
    float scratch_b[NEON_MLP_F32_MAX_DIM];
} neon_mlp_f32_t;

int neon_mlp_f32_init(neon_mlp_f32_t *mlp, int n_layers, const int *dims);

void neon_mlp_f32_load_layer(neon_mlp_f32_t *mlp, int layer,
                              const float *weight_data,
                              const float *bias_data);

uint64_t neon_mlp_f32_forward(neon_mlp_f32_t *mlp,
                               const float *input,
                               float *output);

void neon_mlp_f32_benchmark(neon_mlp_f32_t *mlp,
                             const float *input_pool,
                             float *output_pool,
                             uint32_t n_iters,
                             uint64_t *times_ns);

void neon_mlp_f32_free(neon_mlp_f32_t *mlp);

#endif /* NEON_MLP_F32_H */
