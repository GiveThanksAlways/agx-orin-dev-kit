use crate::config::{PatchEntry, VarType};

/// Apply variable patches to the command queue / kernel args.
///
/// Each patch entry is a u32 location in the pre-built command queue that
/// must be updated with the current kickoff or timeline value.
///
/// Supports bitmask partial writes for fields packed into shared u32 words
/// (common in QMD structs where timeline values share a word with other fields).
#[inline(always)]
pub fn apply(patches: &[PatchEntry], kickoff: u32, tl_wait: u32, tl_signal: u32) {
    for patch in patches {
        let new_val = match patch.var_type {
            VarType::Kickoff => kickoff,
            VarType::TimelineWait => tl_wait,
            VarType::TimelineSignal => tl_signal,
        };

        unsafe {
            let addr = patch.addr as *mut u32;
            if patch.mask != 0 {
                // Partial write: read-modify-write with bitmask
                let old = core::ptr::read_volatile(addr);
                core::ptr::write_volatile(addr, (old & !patch.mask) | (new_val & patch.mask));
            } else {
                // Full 32-bit write
                core::ptr::write_volatile(addr, new_val);
            }
        }
    }
}
