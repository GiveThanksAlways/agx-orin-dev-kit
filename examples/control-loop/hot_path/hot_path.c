/*
 * hot_path.c — C production hot path for tinygrad NV=1 HCQGraph dispatch.
 *
 * Replays a pre-built GPU command queue by writing to the GPFifo ring buffer
 * and poking the doorbell MMIO register. Zero ioctls, zero CUDA, zero syscalls.
 *
 * Build: gcc -O3 -march=armv8.2-a+fp16 -mtune=cortex-a78ae -shared -fPIC \
 *            -o hot_path.so hot_path.c
 */
#include "hot_path.h"
#include <string.h>
#include <time.h>
#include <stdatomic.h>

/* ── ARM memory barrier (ensures CPU writes visible to GPU via ACE-Lite) ── */
#define dmb_sy() __asm__ __volatile__("dmb sy" ::: "memory")

/* ── Helpers ── */
static inline uint64_t timespec_to_ns(struct timespec *ts) {
    return (uint64_t)ts->tv_sec * 1000000000ULL + (uint64_t)ts->tv_nsec;
}

/* ── Spin-wait until signal value >= target ── */
static inline void wait_signal(uint64_t signal_addr, uint64_t target) {
    volatile uint64_t *sig = (volatile uint64_t *)signal_addr;
    while (__atomic_load_n(sig, __ATOMIC_ACQUIRE) < target) {
        __asm__ __volatile__("yield" ::: "memory");
    }
}

/* ── Apply variable patches to the command queue / kernel args ── */
static inline void apply_patches(hot_path_config_t *cfg,
                                 uint32_t kickoff,
                                 uint32_t tl_wait,
                                 uint32_t tl_signal) {
    uint32_t vals[3];
    vals[VAR_KICKOFF]         = kickoff;
    vals[VAR_TIMELINE_WAIT]   = tl_wait;
    vals[VAR_TIMELINE_SIGNAL] = tl_signal;

    for (uint32_t i = 0; i < cfg->num_patches; i++) {
        volatile uint32_t *addr = (volatile uint32_t *)cfg->patches[i].addr;
        uint32_t new_val = vals[cfg->patches[i].var_type];
        uint32_t mask = cfg->patches[i].mask;
        if (mask) {
            *addr = (*addr & ~mask) | (new_val & mask);
        } else {
            *addr = new_val;
        }
    }
}

/* ── Submit to GPU via GPFifo: ring write + doorbell poke ── */
static inline void submit_gpfifo(hot_path_config_t *cfg) {
    volatile uint64_t *ring  = (volatile uint64_t *)cfg->gpfifo_ring_addr;
    volatile uint32_t *gpput = (volatile uint32_t *)cfg->gpfifo_gpput_addr;
    volatile uint32_t *mmio  = (volatile uint32_t *)cfg->gpu_mmio_addr;

    uint32_t put = cfg->gpfifo_put_value % cfg->gpfifo_entries_count;
    ring[put] = cfg->gpfifo_entry;
    gpput[0] = (put + 1) % cfg->gpfifo_entries_count;

    /* Memory barrier: ensure ring + gpput writes are visible before doorbell */
    dmb_sy();

    /* Poke doorbell — the single MMIO write that kicks the GPU */
    mmio[0x90 / 4] = cfg->gpfifo_token;

    cfg->gpfifo_put_value++;
}

/* ── Initialize: precompute the constant GPFifo entry ── */
void hot_path_init(hot_path_config_t *cfg) {
    /*
     * GPFifo entry format (from tinygrad ops_nv.py _submit_to_gpfifo):
     *   (cmdq_addr/4 << 2) | (len << 42) | (1 << 41)
     *
     * cmdq_addr is the GPU virtual address of the bound command queue.
     * len is the number of uint32 words in the command queue.
     */
    cfg->gpfifo_entry = ((cfg->cmdq_gpu_addr / 4) << 2)
                      | ((uint64_t)cfg->cmdq_len_u32 << 42)
                      | (1ULL << 41);
}

/* ── Run one synchronous GPU iteration ── */
uint64_t hot_path_run_iteration(
    hot_path_config_t *cfg,
    const void *sensor_data,
    void *action_output)
{
    struct timespec t0, t1;

    /* 1. Wait for previous GPU completion (first call: already satisfied) */
    wait_signal(cfg->timeline_signal_addr, cfg->last_tl_value);

    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* 2. H2D: copy sensor data into GPU input buffer (Tegra unified memory) */
    memcpy((void *)(uintptr_t)cfg->input_buf_addr, sensor_data, cfg->input_size);

    /* 3. Increment kickoff */
    cfg->kickoff_value++;

    /* 4. Compute patch values */
    uint32_t tl_wait   = cfg->timeline_value - 1;
    uint32_t tl_signal = cfg->timeline_value;

    /* 5. Patch command queue with new variable values */
    apply_patches(cfg, cfg->kickoff_value, tl_wait, tl_signal);

    /* 6. Submit to GPU via GPFifo */
    submit_gpfifo(cfg);

    /* 7. Update timeline tracking */
    cfg->last_tl_value = cfg->timeline_value;
    cfg->timeline_value++;

    /* 8. Reset queue signals (must happen before KICK) */
    for (uint32_t i = 0; i < cfg->num_queue_signals; i++) {
        volatile uint64_t *sig = (volatile uint64_t *)cfg->queue_signal_addrs[i];
        *sig = 0;
    }

    /* 9. KICK — unblocks the GPU which is waiting on this signal */
    {
        volatile uint64_t *kick = (volatile uint64_t *)cfg->kick_signal_addr;
        *kick = cfg->kickoff_value;
    }

    /* 10. Wait for GPU completion (synchronous for benchmarking) */
    wait_signal(cfg->timeline_signal_addr, cfg->last_tl_value);

    /* 11. D2H: copy action from GPU output buffer */
    memcpy(action_output, (const void *)(uintptr_t)cfg->output_buf_addr, cfg->output_size);

    clock_gettime(CLOCK_MONOTONIC, &t1);

    return timespec_to_ns(&t1) - timespec_to_ns(&t0);
}

/* ── Submit one HCQ graph: patch + submit + wait + kick (no memcpy) ── */
uint64_t hot_path_submit_graph(hot_path_config_t *cfg)
{
    struct timespec t0, t1;

    /* 1. Wait for previous GPU completion (first call: already satisfied) */
    wait_signal(cfg->timeline_signal_addr, cfg->last_tl_value);

    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* 2. Increment kickoff */
    cfg->kickoff_value++;

    /* 3. Compute patch values */
    uint32_t tl_wait   = cfg->timeline_value - 1;
    uint32_t tl_signal = cfg->timeline_value;

    /* 4. Patch command queue with new variable values */
    apply_patches(cfg, cfg->kickoff_value, tl_wait, tl_signal);

    /* 5. Submit to GPU via GPFifo */
    submit_gpfifo(cfg);

    /* 6. Update timeline tracking */
    cfg->last_tl_value = cfg->timeline_value;
    cfg->timeline_value++;

    /* 7. Reset queue signals (must happen before KICK) */
    for (uint32_t i = 0; i < cfg->num_queue_signals; i++) {
        volatile uint64_t *sig = (volatile uint64_t *)cfg->queue_signal_addrs[i];
        *sig = 0;
    }

    /* 8. KICK — unblocks the GPU which is waiting on this signal */
    {
        volatile uint64_t *kick = (volatile uint64_t *)cfg->kick_signal_addr;
        *kick = cfg->kickoff_value;
    }

    /* 9. Wait for GPU completion */
    wait_signal(cfg->timeline_signal_addr, cfg->last_tl_value);

    clock_gettime(CLOCK_MONOTONIC, &t1);

    return timespec_to_ns(&t1) - timespec_to_ns(&t0);
}

/* ── Benchmark N iterations ── */
void hot_path_benchmark(
    hot_path_config_t *cfg,
    const void *sensor_pool,
    void *action_pool,
    uint32_t n_iters,
    uint64_t *times_ns)
{
    const uint8_t *sp = (const uint8_t *)sensor_pool;
    uint8_t *ap = (uint8_t *)action_pool;

    for (uint32_t i = 0; i < n_iters; i++) {
        times_ns[i] = hot_path_run_iteration(
            cfg,
            sp + (size_t)i * cfg->input_size,
            ap + (size_t)i * cfg->output_size
        );
    }
}
