//! # cuda-graph-replay
//!
//! Capture and replay CUDA graphs from Rust for lower-latency inference.
//!
//! Instead of calling `enqueueV3` per iteration (which goes through the full
//! CUDA runtime → ioctl → driver path), this crate captures the execution into
//! a CUDA graph on the first call, then replays it with `cudaGraphLaunch` on
//! subsequent calls — skipping all runtime dispatch overhead.
//!
//! ## How CUDA Graphs Work
//!
//! ```text
//! First call (capture):
//!   cudaStreamBeginCapture → enqueueV3 → cudaStreamEndCapture
//!   → cudaGraphInstantiate → cudaGraphExec
//!
//! Subsequent calls (replay):
//!   cudaGraphLaunch(exec, stream) → cudaStreamSynchronize
//!   (no enqueueV3, no runtime dispatch, no ioctl per kernel)
//! ```
//!
//! ## Integration with libinfer
//!
//! ```rust,no_run
//! use cuda_graph_replay::CudaGraphRunner;
//!
//! // Wrap a libinfer engine's infer() call in a CUDA graph
//! let mut runner = CudaGraphRunner::new();
//! // First call captures, subsequent calls replay
//! ```

mod graph;
mod stream;

pub use graph::CudaGraphRunner;
pub use stream::CudaStream;
