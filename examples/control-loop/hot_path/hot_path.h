/*
 * hot_path.h — C production hot path for tinygrad NV=1 HCQGraph dispatch on Tegra.
 *
 * Replays a pre-built GPU command queue via raw GPFifo MMIO doorbell writes.
 * Zero ioctls in the hot path — submission is one volatile MMIO write.
 *
 * Usage: Python builds the JIT graph, exports addresses to hot_path_config_t,
 * then C takes over the dispatch loop via ctypes .so loading.
 */
#ifndef HOT_PATH_H
#define HOT_PATH_H

#include <stdint.h>
#include <stddef.h>

#define MAX_PATCHES        512
#define MAX_QUEUE_SIGNALS  16

/* Patch variable types — what value a patch location receives each iteration */
#define VAR_KICKOFF         0   /* value = kickoff_value                 */
#define VAR_TIMELINE_WAIT   1   /* value = timeline_value - 1 (wait val) */
#define VAR_TIMELINE_SIGNAL 2   /* value = timeline_value (signal val)   */

typedef struct {
    uint64_t addr;          /* CPU address to write (volatile uint32_t*) */
    uint32_t var_type;      /* VAR_KICKOFF / VAR_TIMELINE_WAIT / VAR_TIMELINE_SIGNAL */
    uint32_t mask;          /* 0 = full 32-bit write; else bitmask for partial update */
} patch_entry_t;

typedef struct {
    /* ── Input/output GPU buffers (CPU-mapped via Tegra unified memory) ── */
    uint64_t input_buf_addr;
    uint64_t output_buf_addr;
    uint32_t input_size;
    uint32_t output_size;

    /* ── GPFifo (ring buffer for GPU command submission) ── */
    uint64_t gpfifo_ring_addr;      /* CPU addr of ring[] (volatile uint64_t*) */
    uint64_t gpfifo_gpput_addr;     /* CPU addr of gpput  (volatile uint32_t*) */
    uint32_t gpfifo_entries_count;
    uint32_t gpfifo_token;          /* work submit token for doorbell */
    uint32_t gpfifo_put_value;      /* current put index (increments each submit) */

    /* ── Command queue (pre-built by tinygrad, bound to GPU hw_page) ── */
    uint64_t cmdq_gpu_addr;         /* GPU virtual address of command queue */
    uint32_t cmdq_len_u32;          /* length of command queue in uint32 words */

    /* ── MMIO doorbell (usermode GPU register page) ── */
    uint64_t gpu_mmio_addr;         /* CPU addr of MMIO page (doorbell at +0x90) */

    /* ── Timeline signal (GPU writes here on completion) ── */
    uint64_t timeline_signal_addr;  /* CPU addr of 8-byte signal value */
    uint32_t timeline_value;        /* next timeline value to use */
    uint32_t last_tl_value;         /* last submitted timeline value to wait on */

    /* ── Kick signal (CPU writes here to unblock GPU) ── */
    uint64_t kick_signal_addr;      /* CPU addr of 8-byte kick signal */
    uint32_t kickoff_value;         /* current kickoff counter */

    /* ── Queue signals to reset before each kick ── */
    uint32_t num_queue_signals;
    uint64_t queue_signal_addrs[MAX_QUEUE_SIGNALS];

    /* ── Patch list: locations in cmdq/kernargs to update per iteration ── */
    uint32_t num_patches;
    patch_entry_t patches[MAX_PATCHES];

    /* ── Pre-computed GPFifo entry (address + length, constant) ── */
    uint64_t gpfifo_entry;
} hot_path_config_t;

/*
 * Run one GPU inference iteration (synchronous: submit + wait).
 *
 * 1. memcpy sensor_data → input buffer (unified memory, ~0.1 µs)
 * 2. Patch command queue with new kickoff/timeline values (~0.1 µs)
 * 3. Submit to GPFifo: ring write + doorbell poke (~0.5 µs)
 * 4. Reset queue signals + set KICK signal (~0.1 µs)
 * 5. Spin-wait for GPU completion (~5-25 µs depending on model)
 * 6. memcpy output buffer → action_output (~0.1 µs)
 *
 * Returns: cycle time in nanoseconds.
 */
uint64_t hot_path_run_iteration(
    hot_path_config_t *cfg,
    const void *sensor_data,
    void *action_output
);

/*
 * Run N iterations, recording per-iteration timing in times_ns[].
 * sensor_pool is N * input_size contiguous bytes.
 * action_pool is N * output_size contiguous bytes (output).
 */
void hot_path_benchmark(
    hot_path_config_t *cfg,
    const void *sensor_pool,
    void *action_pool,
    uint32_t n_iters,
    uint64_t *times_ns
);

/*
 * Finalize config after Python populates it.
 * Precomputes the GPFifo entry from cmdq_gpu_addr + cmdq_len.
 */
void hot_path_init(hot_path_config_t *cfg);

/*
 * Submit one HCQ graph (patch + submit + wait + kick).
 * No memcpy — caller handles H2D/D2H.
 * Returns: latency in nanoseconds.
 */
uint64_t hot_path_submit_graph(hot_path_config_t *cfg);

#endif /* HOT_PATH_H */
