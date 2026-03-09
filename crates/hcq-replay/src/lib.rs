//! # hcq-replay
//!
//! Rust hot path for replaying tinygrad NV=1 HCQGraph command queues on
//! NVIDIA Tegra (Jetson AGX Orin) via raw MMIO doorbell writes.
//!
//! This crate is the Rust equivalent of `hot_path.c` — it replays a
//! pre-built GPU command queue with zero ioctls, zero CUDA runtime calls,
//! and zero Python overhead. Typical per-iteration latency: ~46 µs.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use hcq_replay::{HcqGraph, HotPathConfig};
//!
//! // Config populated from export_graph.py via JSON
//! let config: HotPathConfig = /* ... */;
//!
//! // SAFETY: all addresses in config must be valid mmap'd pointers
//! let mut graph = unsafe { HcqGraph::from_config(config) };
//!
//! // Hot loop
//! let sensor_data: &[u8] = &[/* IMU bytes */];
//! let mut output = vec![0u8; graph.output_size()];
//! graph.run_iteration(sensor_data, &mut output);
//! ```

mod config;
mod gpfifo;
mod patch;
mod signal;

pub use config::{HotPathConfig, PatchEntry, VarType};
pub use gpfifo::GpFifo;

use std::time::Instant;

/// A ready-to-run HCQ graph that can replay GPU commands via raw MMIO.
///
/// Construction is `unsafe` because it accepts raw mmap'd pointers from
/// the Python export pipeline. Once constructed, `run_iteration()` is safe
/// (the mmap'd regions are guaranteed valid for the graph's lifetime by
/// the caller's contract).
pub struct HcqGraph {
    // I/O buffer addresses (Tegra unified memory, CPU-accessible)
    input_buf: *mut u8,
    output_buf: *const u8,
    input_size: usize,
    output_size: usize,

    // GPU submission
    gpfifo: GpFifo,

    // Signals
    timeline_signal: *const u64,
    kick_signal: *mut u64,
    queue_signals: Vec<*mut u64>,

    // Patch list
    patches: Vec<PatchEntry>,

    // Counters
    timeline_value: u32,
    last_tl_value: u32,
    kickoff_value: u32,
}

// HcqGraph is not thread-safe (mutable GPU state) but can be moved between threads.
unsafe impl Send for HcqGraph {}

impl HcqGraph {
    /// Create an HcqGraph from a config exported by `export_graph.py`.
    ///
    /// # Safety
    ///
    /// All address fields in `config` must point to valid mmap'd memory regions
    /// that remain valid for the lifetime of this HcqGraph. These are typically
    /// obtained from tinygrad's NV backend via `cpu_view().addr`.
    pub unsafe fn from_config(config: HotPathConfig) -> Self {
        let gpfifo = GpFifo::new(
            config.gpfifo_ring_addr as *mut u64,
            config.gpfifo_gpput_addr as *mut u32,
            config.gpu_mmio_addr as *mut u32,
            config.gpfifo_entries_count,
            config.gpfifo_token,
            config.gpfifo_put_value,
            config.cmdq_gpu_addr,
            config.cmdq_len_u32,
        );

        let queue_signals = config
            .queue_signal_addrs
            .iter()
            .map(|&addr| addr as *mut u64)
            .collect();

        HcqGraph {
            input_buf: config.input_buf_addr as *mut u8,
            output_buf: config.output_buf_addr as *const u8,
            input_size: config.input_size as usize,
            output_size: config.output_size as usize,
            gpfifo,
            timeline_signal: config.timeline_signal_addr as *const u64,
            kick_signal: config.kick_signal_addr as *mut u64,
            queue_signals,
            patches: config.patches,
            timeline_value: config.timeline_value,
            last_tl_value: config.last_tl_value,
            kickoff_value: config.kickoff_value,
        }
    }

    /// Run one synchronous GPU inference iteration.
    ///
    /// 1. Wait for previous GPU completion
    /// 2. Copy `sensor_data` → GPU input buffer (unified memory memcpy)
    /// 3. Patch command queue with new kickoff/timeline values
    /// 4. Submit to GPFifo (ring write + MMIO doorbell)
    /// 5. Reset queue signals + set KICK signal
    /// 6. Wait for GPU completion
    /// 7. Copy GPU output buffer → `output`
    ///
    /// Returns elapsed time in nanoseconds.
    pub fn run_iteration(&mut self, sensor_data: &[u8], output: &mut [u8]) -> u64 {
        assert!(sensor_data.len() <= self.input_size);
        assert!(output.len() <= self.output_size);

        // 1. Wait for previous GPU completion
        signal::wait(self.timeline_signal, self.last_tl_value as u64);

        let t0 = Instant::now();

        // 2. H2D: copy into unified memory input buffer
        unsafe {
            core::ptr::copy_nonoverlapping(
                sensor_data.as_ptr(),
                self.input_buf,
                sensor_data.len(),
            );
        }

        // 3. Increment kickoff
        self.kickoff_value += 1;

        // 4. Patch command queue
        let tl_wait = self.timeline_value - 1;
        let tl_signal = self.timeline_value;
        patch::apply(&self.patches, self.kickoff_value, tl_wait, tl_signal);

        // 5. Submit to GPU via GPFifo
        self.gpfifo.submit();

        // 6. Update timeline
        self.last_tl_value = self.timeline_value;
        self.timeline_value += 1;

        // 7. Reset queue signals
        for &sig_ptr in &self.queue_signals {
            unsafe {
                core::ptr::write_volatile(sig_ptr, 0);
            }
        }

        // 8. KICK — unblocks the GPU
        unsafe {
            core::ptr::write_volatile(self.kick_signal, self.kickoff_value as u64);
        }

        // 9. Wait for GPU completion
        signal::wait(self.timeline_signal, self.last_tl_value as u64);

        // 10. D2H: copy from unified memory output buffer
        unsafe {
            core::ptr::copy_nonoverlapping(self.output_buf, output.as_mut_ptr(), output.len());
        }

        t0.elapsed().as_nanos() as u64
    }

    /// Submit the graph without H2D/D2H copies.
    /// Caller handles I/O buffer access directly.
    pub fn submit_graph(&mut self) -> u64 {
        signal::wait(self.timeline_signal, self.last_tl_value as u64);

        let t0 = Instant::now();

        self.kickoff_value += 1;
        let tl_wait = self.timeline_value - 1;
        let tl_signal = self.timeline_value;

        patch::apply(&self.patches, self.kickoff_value, tl_wait, tl_signal);
        self.gpfifo.submit();

        self.last_tl_value = self.timeline_value;
        self.timeline_value += 1;

        for &sig_ptr in &self.queue_signals {
            unsafe { core::ptr::write_volatile(sig_ptr, 0) };
        }
        unsafe { core::ptr::write_volatile(self.kick_signal, self.kickoff_value as u64) };

        signal::wait(self.timeline_signal, self.last_tl_value as u64);
        t0.elapsed().as_nanos() as u64
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Get a mutable slice over the GPU input buffer (Tegra unified memory).
    ///
    /// # Safety
    /// Caller must ensure no GPU operation is in flight that reads this buffer.
    pub unsafe fn input_buffer_mut(&mut self) -> &mut [u8] {
        core::slice::from_raw_parts_mut(self.input_buf, self.input_size)
    }

    /// Get a slice over the GPU output buffer (Tegra unified memory).
    ///
    /// # Safety
    /// Caller must ensure no GPU operation is in flight that writes this buffer.
    pub unsafe fn output_buffer(&self) -> &[u8] {
        core::slice::from_raw_parts(self.output_buf, self.output_size)
    }
}
