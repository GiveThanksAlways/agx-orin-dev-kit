use core::sync::atomic::{AtomicU64, Ordering};

/// Spin-wait until the signal value at `signal_addr` reaches `target`.
///
/// Uses atomic acquire load semantics to ensure we see GPU writes.
/// On ARM, emits `yield` hint while spinning to save power.
#[inline(always)]
pub fn wait(signal_addr: *const u64, target: u64) {
    // Reinterpret the raw pointer as an AtomicU64 reference.
    // SAFETY: the pointer came from tinygrad's mmap'd signal allocation,
    // which is always 8-byte aligned and valid for the graph's lifetime.
    let atomic = unsafe { &*(signal_addr as *const AtomicU64) };

    while atomic.load(Ordering::Acquire) < target {
        // ARM WFE/YIELD hint: tells the CPU to enter a low-power state
        // while spinning, reducing power consumption and bus traffic.
        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!("yield", options(nomem, nostack, preserves_flags));
        }
        // x86: PAUSE hint
        #[cfg(target_arch = "x86_64")]
        core::hint::spin_loop();
    }
}
