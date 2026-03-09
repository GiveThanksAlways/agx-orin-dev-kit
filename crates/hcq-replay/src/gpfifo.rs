/// GPFifo: GPU command submission via ring buffer + MMIO doorbell.
///
/// This is the Tegra submission path — writing a 64-bit entry to the ring buffer,
/// updating the put pointer, then poking the doorbell MMIO register at offset 0x90.
pub struct GpFifo {
    ring: *mut u64,
    gpput: *mut u32,
    doorbell: *mut u32,
    entries_count: u32,
    token: u32,
    put_value: u32,
    entry: u64,
}

impl GpFifo {
    /// Create a new GpFifo from raw mmap'd addresses.
    ///
    /// # Safety
    /// All pointers must be valid mmap'd addresses for the GPU channel.
    pub unsafe fn new(
        ring: *mut u64,
        gpput: *mut u32,
        doorbell: *mut u32,
        entries_count: u32,
        token: u32,
        put_value: u32,
        cmdq_gpu_addr: u64,
        cmdq_len_u32: u32,
    ) -> Self {
        // GPFifo entry format (from tinygrad ops_nv.py _submit_to_gpfifo):
        //   (cmdq_addr/4 << 2) | (len << 42) | (1 << 41)
        let entry = ((cmdq_gpu_addr / 4) << 2)
            | ((cmdq_len_u32 as u64) << 42)
            | (1u64 << 41);

        GpFifo {
            ring,
            gpput,
            doorbell,
            entries_count,
            token,
            put_value,
            entry,
        }
    }

    /// Submit the pre-built command queue to the GPU.
    ///
    /// 1. Write entry to ring buffer at current put index
    /// 2. Update gpput pointer
    /// 3. ARM dmb sy memory barrier
    /// 4. Poke MMIO doorbell at offset 0x90
    #[inline(always)]
    pub fn submit(&mut self) {
        let idx = (self.put_value % self.entries_count) as usize;

        unsafe {
            // Write ring entry
            core::ptr::write_volatile(self.ring.add(idx), self.entry);

            // Update gpput
            core::ptr::write_volatile(
                self.gpput,
                (self.put_value + 1) % self.entries_count,
            );

            // ARM memory barrier: ensure ring + gpput writes are visible before doorbell.
            // On aarch64 this compiles to `dmb sy`.
            #[cfg(target_arch = "aarch64")]
            core::arch::asm!("dmb sy", options(nostack, preserves_flags));

            #[cfg(not(target_arch = "aarch64"))]
            core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);

            // Poke doorbell — the single MMIO write that kicks the GPU.
            // Offset 0x90 = 0x90/4 = 36 u32 words from base.
            core::ptr::write_volatile(self.doorbell.add(0x90 / 4), self.token);
        }

        self.put_value += 1;
    }
}
