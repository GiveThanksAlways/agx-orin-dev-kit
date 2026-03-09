/// Patch variable types — what value a patch location receives each iteration.
/// Must match hot_path.h: VAR_KICKOFF=0, VAR_TIMELINE_WAIT=1, VAR_TIMELINE_SIGNAL=2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum VarType {
    Kickoff = 0,
    TimelineWait = 1,
    TimelineSignal = 2,
}

/// A single patch entry: one location in the command queue or kernel args
/// that must be updated with a fresh kickoff/timeline value each iteration.
#[derive(Debug, Clone)]
pub struct PatchEntry {
    /// CPU address to write (volatile u32 pointer into mmap'd command queue).
    pub addr: u64,
    /// Which variable this patch writes.
    pub var_type: VarType,
    /// Bitmask for partial updates (0 = full 32-bit write).
    pub mask: u32,
}

/// Full configuration matching hot_path_config_t from hot_path.h.
///
/// All `_addr` fields are raw CPU virtual addresses obtained from tinygrad's
/// `cpu_view().addr` / mmap. These point into Tegra unified memory or
/// GPU MMIO register pages.
#[derive(Debug, Clone)]
pub struct HotPathConfig {
    // I/O buffers (Tegra unified memory, CPU-mapped)
    pub input_buf_addr: u64,
    pub output_buf_addr: u64,
    pub input_size: u32,
    pub output_size: u32,

    // GPFifo (ring buffer for GPU command submission)
    pub gpfifo_ring_addr: u64,
    pub gpfifo_gpput_addr: u64,
    pub gpfifo_entries_count: u32,
    pub gpfifo_token: u32,
    pub gpfifo_put_value: u32,

    // Command queue (pre-built by tinygrad, bound to GPU hw_page)
    pub cmdq_gpu_addr: u64,
    pub cmdq_len_u32: u32,

    // MMIO doorbell (usermode GPU register page)
    pub gpu_mmio_addr: u64,

    // Timeline signal (GPU writes here on completion)
    pub timeline_signal_addr: u64,
    pub timeline_value: u32,
    pub last_tl_value: u32,

    // Kick signal (CPU writes here to unblock GPU)
    pub kick_signal_addr: u64,
    pub kickoff_value: u32,

    // Queue signals to reset before each kick
    pub queue_signal_addrs: Vec<u64>,

    // Patch list
    pub patches: Vec<PatchEntry>,
}
