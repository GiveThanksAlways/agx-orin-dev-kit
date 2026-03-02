/*
 * neon_mlp.h — ARM NEON FP16 MLP forward pass for Cortex-A78AE (Orin).
 *
 * Pure CPU baseline for comparing against GPU dispatch.
 * Uses ARM NEON float16x8_t intrinsics for vectorized inference.
 */
#ifndef NEON_MLP_H
#define NEON_MLP_H

#include <stdint.h>
#include <stddef.h>
#include <arm_neon.h>

/* Up to 8 layers (7 hidden + 1 output) */
#define NEON_MLP_MAX_LAYERS 8
#define NEON_MLP_MAX_DIM    2048

typedef struct {
    int n_layers;                            /* number of weight matrices */
    int dims[NEON_MLP_MAX_LAYERS + 1];       /* [in, h1, h2, ..., out] */

    /* Weights stored row-major: W[layer][out_dim][in_dim_padded] */
    /* in_dim_padded = round_up(in_dim, 8) for NEON alignment     */
    __fp16 *weights[NEON_MLP_MAX_LAYERS];  /* allocated per layer */
    __fp16 *biases[NEON_MLP_MAX_LAYERS];   /* allocated per layer */

    /* Scratch buffers for intermediate activations */
    __fp16 scratch_a[NEON_MLP_MAX_DIM];
    __fp16 scratch_b[NEON_MLP_MAX_DIM];
} neon_mlp_t;

/*
 * Initialize MLP: allocate and zero internal weight storage.
 * dims is an array of [in, h1, h2, ..., out] with n_layers+1 entries.
 * Returns 0 on success, -1 on failure.
 */
int neon_mlp_init(neon_mlp_t *mlp, int n_layers, const int *dims);

/*
 * Load weights for a single layer from row-major FP16 data.
 * Weight shape: [out_dim, in_dim] stored row-major.
 * Internally pads in_dim to multiple of 8 and stores.
 */
void neon_mlp_load_layer(neon_mlp_t *mlp, int layer,
                         const __fp16 *weight_data,
                         const __fp16 *bias_data);

/*
 * Forward pass: input[in_dim] → output[out_dim]
 * All hidden layers use ReLU; output layer has no activation.
 * Returns cycle time in nanoseconds.
 */
uint64_t neon_mlp_forward(neon_mlp_t *mlp,
                          const __fp16 *input,
                          __fp16 *output);

/*
 * Benchmark N iterations.
 * input_pool: N * in_dim _Float16 values.
 * output_pool: N * out_dim _Float16 values (written).
 * times_ns: N uint64_t values (per-iteration timing).
 */
void neon_mlp_benchmark(neon_mlp_t *mlp,
                        const __fp16 *input_pool,
                        __fp16 *output_pool,
                        uint32_t n_iters,
                        uint64_t *times_ns);

/* Free allocated weight storage. */
void neon_mlp_free(neon_mlp_t *mlp);

#endif /* NEON_MLP_H */
