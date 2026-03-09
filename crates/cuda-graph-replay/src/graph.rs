use crate::stream::CudaStream;
use std::ffi::c_void;
use std::ptr;

// CUDA graph capture mode
const CUDA_STREAM_CAPTURE_MODE_GLOBAL: i32 = 0;

// Raw CUDA graph API bindings
extern "C" {
    fn cudaStreamBeginCapture(stream: *mut c_void, mode: i32) -> i32;
    fn cudaStreamEndCapture(stream: *mut c_void, graph: *mut *mut c_void) -> i32;
    fn cudaGraphInstantiate(
        exec: *mut *mut c_void,
        graph: *mut c_void,
        log: *mut u8,
        log_size: usize,
    ) -> i32;
    fn cudaGraphLaunch(exec: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaGraphExecDestroy(exec: *mut c_void) -> i32;
    fn cudaGraphDestroy(graph: *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: i32,
        stream: *mut c_void,
    ) -> i32;
}

const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

fn check_cuda(err: i32, msg: &str) -> Result<(), String> {
    if err != 0 {
        Err(format!("{}: CUDA error code {}", msg, err))
    } else {
        Ok(())
    }
}

/// State of the CUDA graph runner.
enum State {
    /// Not yet captured — next call will do capture.
    NeedCapture,
    /// Graph captured and instantiated — subsequent calls use replay.
    Ready {
        graph: *mut c_void,
        exec: *mut c_void,
    },
}

/// Captures and replays CUDA graphs for lower-latency inference.
///
/// The workflow:
/// 1. First call to `run()`: captures stream operations into a CUDA graph
/// 2. Subsequent calls to `replay()`: replays the captured graph
///
/// This eliminates per-call enqueueV3 overhead (~30-60 µs saved on Tegra).
pub struct CudaGraphRunner {
    stream: CudaStream,
    state: State,
    // Pre-allocated GPU buffers for I/O
    input_buf: *mut c_void,
    output_buf: *mut c_void,
    input_size: usize,
    output_size: usize,
}

extern "C" {
    fn cudaMalloc(devptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(devptr: *mut c_void) -> i32;
}

impl CudaGraphRunner {
    /// Create a new CUDA graph runner with pre-allocated GPU buffers.
    ///
    /// `input_size` and `output_size` are in bytes.
    pub fn new(input_size: usize, output_size: usize) -> Result<Self, String> {
        let stream = CudaStream::new().map_err(|e| format!("cudaStreamCreate: {}", e))?;

        let mut input_buf: *mut c_void = ptr::null_mut();
        let mut output_buf: *mut c_void = ptr::null_mut();

        check_cuda(
            unsafe { cudaMalloc(&mut input_buf, input_size) },
            "cudaMalloc input",
        )?;
        check_cuda(
            unsafe { cudaMalloc(&mut output_buf, output_size) },
            "cudaMalloc output",
        )?;

        Ok(CudaGraphRunner {
            stream,
            state: State::NeedCapture,
            input_buf,
            output_buf,
            input_size,
            output_size,
        })
    }

    /// Begin CUDA graph capture on our stream.
    pub fn begin_capture(&self) -> Result<(), String> {
        check_cuda(
            unsafe {
                cudaStreamBeginCapture(self.stream.as_raw(), CUDA_STREAM_CAPTURE_MODE_GLOBAL)
            },
            "cudaStreamBeginCapture",
        )
    }

    /// End capture, instantiate the graph, and store the executable.
    pub fn end_capture(&mut self) -> Result<(), String> {
        let mut graph: *mut c_void = ptr::null_mut();
        check_cuda(
            unsafe { cudaStreamEndCapture(self.stream.as_raw(), &mut graph) },
            "cudaStreamEndCapture",
        )?;

        let mut exec: *mut c_void = ptr::null_mut();
        check_cuda(
            unsafe { cudaGraphInstantiate(&mut exec, graph, ptr::null_mut(), 0) },
            "cudaGraphInstantiate",
        )?;

        self.state = State::Ready { graph, exec };
        Ok(())
    }

    /// Replay the captured CUDA graph.
    ///
    /// Input data must already be in the GPU buffer (via `copy_input()`).
    /// After return, output is available via `copy_output()`.
    pub fn replay(&self) -> Result<(), String> {
        match &self.state {
            State::NeedCapture => Err("Graph not captured yet — call begin_capture/end_capture first".into()),
            State::Ready { exec, .. } => {
                check_cuda(
                    unsafe { cudaGraphLaunch(*exec, self.stream.as_raw()) },
                    "cudaGraphLaunch",
                )?;
                check_cuda(
                    unsafe { cudaStreamSynchronize(self.stream.as_raw()) },
                    "cudaStreamSynchronize",
                )
            }
        }
    }

    /// Copy input data from host to GPU buffer.
    pub fn copy_input(&self, data: &[u8]) -> Result<(), String> {
        assert!(data.len() <= self.input_size);
        check_cuda(
            unsafe {
                cudaMemcpyAsync(
                    self.input_buf,
                    data.as_ptr() as *const c_void,
                    data.len(),
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                    self.stream.as_raw(),
                )
            },
            "cudaMemcpyAsync H2D",
        )
    }

    /// Copy output data from GPU buffer to host.
    pub fn copy_output(&self, output: &mut [u8]) -> Result<(), String> {
        assert!(output.len() <= self.output_size);
        check_cuda(
            unsafe {
                cudaMemcpyAsync(
                    output.as_mut_ptr() as *mut c_void,
                    self.output_buf,
                    output.len(),
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                    self.stream.as_raw(),
                )
            },
            "cudaMemcpyAsync D2H",
        )?;
        check_cuda(
            unsafe { cudaStreamSynchronize(self.stream.as_raw()) },
            "cudaStreamSynchronize after D2H",
        )
    }

    /// Get the raw CUDA stream handle (for passing to TRT enqueueV3 during capture).
    pub fn stream_raw(&self) -> *mut c_void {
        self.stream.as_raw()
    }

    /// Get the raw input GPU buffer pointer.
    pub fn input_buffer_ptr(&self) -> *mut c_void {
        self.input_buf
    }

    /// Get the raw output GPU buffer pointer.
    pub fn output_buffer_ptr(&self) -> *mut c_void {
        self.output_buf
    }

    pub fn is_captured(&self) -> bool {
        matches!(self.state, State::Ready { .. })
    }
}

impl Drop for CudaGraphRunner {
    fn drop(&mut self) {
        if let State::Ready { graph, exec } = self.state {
            unsafe {
                cudaGraphExecDestroy(exec);
                cudaGraphDestroy(graph);
            }
        }
        unsafe {
            if !self.input_buf.is_null() {
                cudaFree(self.input_buf);
            }
            if !self.output_buf.is_null() {
                cudaFree(self.output_buf);
            }
        }
    }
}

unsafe impl Send for CudaGraphRunner {}
