use std::ffi::c_void;
use std::ptr;

// Raw CUDA runtime bindings — just the functions we need.
// These are linked via -lcudart from the system CUDA installation.
extern "C" {
    fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
}

/// Owned CUDA stream with RAII cleanup.
pub struct CudaStream {
    pub(crate) raw: *mut c_void,
}

impl CudaStream {
    pub fn new() -> Result<Self, i32> {
        let mut raw: *mut c_void = ptr::null_mut();
        let err = unsafe { cudaStreamCreate(&mut raw) };
        if err != 0 {
            return Err(err);
        }
        Ok(CudaStream { raw })
    }

    pub fn synchronize(&self) -> Result<(), i32> {
        let err = unsafe { cudaStreamSynchronize(self.raw) };
        if err != 0 { Err(err) } else { Ok(()) }
    }

    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { cudaStreamDestroy(self.raw) };
        }
    }
}

// CudaStream is Send (can be moved between threads) but not Sync.
unsafe impl Send for CudaStream {}
