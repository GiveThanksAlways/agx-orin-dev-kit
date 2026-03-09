# Saronic Collaboration PRD

> Branch: `saronic-dev`
> Target: upstream PRs to [saronic-technologies/libinfer](https://github.com/saronic-technologies/libinfer)

---

## Task 1: Tegra Zero-Copy for libinfer

**Goal**: Eliminate unnecessary `cudaMemcpyAsync` H2D/D2H copies on Jetson (Tegra unified memory).

**Files changed** (in `external/libinfer/`):
- [x] `src/engine.h` — Add `mUseManagedMemory` flag, zero-copy buffer views, `infer_zerocopy()` method
- [x] `src/engine.cpp` — `cudaMallocManaged` path in `load()`, skip copies in `infer_zerocopy()`
- [x] `src/lib.rs` — Expose `get_input_buffer_ptr()`, `get_output_buffer_ptr()`, `infer_zerocopy()` via cxx bridge
- [x] `examples/benchmark_zerocopy.rs` — Benchmark comparing stock `infer()` vs `infer_zerocopy()`

**Acceptance**:
- [ ] `cargo build` succeeds on AGX Orin (aarch64)
- [ ] `benchmark_zerocopy` shows measurable latency drop vs stock `infer()`
- [x] Stock `infer()` path unchanged (no regression for discrete GPUs)

---

## Task 2: Rust HCQ-Replay Crate (mmio-replay)

**Goal**: Port the C hot path (`examples/control-loop/hot_path/hot_path.c`) to a standalone Rust crate.

**Files created** (new crate: `crates/hcq-replay/`):
- [x] `Cargo.toml`
- [x] `src/lib.rs` — Public API: `HcqGraph::from_config()`, `.run_iteration()`
- [x] `src/config.rs` — `HotPathConfig` struct (mirrors `hot_path_config_t`)
- [x] `src/gpfifo.rs` — GPFifo ring submission + MMIO doorbell
- [x] `src/signal.rs` — AtomicU64 spin-wait
- [x] `src/patch.rs` — Command queue patcher

**Acceptance**:
- [ ] Crate compiles on aarch64-linux (no CUDA dependency at build time)
- [x] API is `unsafe` at construction (raw mmap addresses) but safe at `run_iteration()`
- [x] Same `hot_path_config_t` layout loadable from Python via `export_graph.py`

---

## Task 3: Benchmark libinfer on Cioffi TCN

**Goal**: Get actual TensorRT-via-Rust latency numbers for our 250K-param TCN.

**Files created**:
- [ ] `examples/benchmark_libinfer_tcn.rs` — Benchmark script in libinfer examples

**Steps** (run manually on Orin):
- [ ] Export TCN to ONNX via `cioffi_tcn.py --export-onnx`
- [ ] Build TRT engine via `trtexec --onnx=... --saveEngine=... --fp16`
- [ ] Run `cargo run --release --example benchmark_zerocopy -- --path cioffi_tcn.engine`
- [ ] Record numbers in results table

---

## Task 4: Rust Hot Path for PyTorch CUDA Graphs

**Goal**: Provide a Rust crate that captures and replays PyTorch CUDA graphs with minimal overhead, giving Saronic a lower-latency path than their current Python PyTorch inference.

**Files created** (new crate: `crates/cuda-graph-replay/`):
- [x] `Cargo.toml`
- [x] `src/lib.rs` — Public API: `CudaGraphRunner::new()`, `.replay()`
- [x] `src/graph.rs` — CUDA graph capture/replay via cuda_runtime bindings
- [x] `src/stream.rs` — CUDA stream management
- [ ] `examples/bench_cuda_graph.rs` — Benchmark

**Acceptance**:
- [ ] Captures a CUDA graph from a TRT engine's inference call
- [ ] `replay()` uses `cudaGraphLaunch` (no re-enqueue overhead)
- [ ] Benchmarks show latency reduction vs stock `infer()`
