# libinfer — Analysis & Integration Notes

> **Origin**: [saronic-technologies/libinfer](https://github.com/saronic-technologies/libinfer) (forked to GiveThanksAlways)
> **What**: Rust crate providing a safe, idiomatic API to TensorRT inference via `cxx` FFI bridge
> **Size**: ~2800 lines total (465 C++, 227 Rust, rest is examples)

---

## TL;DR — How libinfer Works (First Principles)

libinfer is a thin Rust wrapper around TensorRT's C++ API. It solves one problem well:
**load a pre-built `.engine` file and run inference from Rust with minimal ceremony**.

### The 4 Structural Pieces

```text
┌─────────────────────────────────────────────────────────┐
│  1. build.rs  — Build-time glue                         │
│     • Finds CUDA + TensorRT + spdlog + fmt via env vars │
│     • Invokes cxx-build to generate C++↔Rust bridge     │
│     • Compiles engine.cpp with clang++ -O3 -std=c++17   │
│     • Links: cudart, nvinfer, nvinfer_plugin, nvonnxparser│
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│  2. src/lib.rs  — Rust API surface                      │
│     • #[cxx::bridge] defines shared types:              │
│       Options, InputTensor, OutputTensor, TensorInfo    │
│     • Declares extern "C++" functions:                  │
│       load_engine(), infer(), get_input/output_dims()   │
│     • Engine::new() → Result<UniquePtr<Engine>>         │
│     • engine.pin_mut().infer(&inputs) → Vec<OutputTensor>│
│     • unsafe impl Send — movable across threads         │
│       (but NOT Sync — not thread-safe)                  │
└─────────────────┬───────────────────────────────────────┘
                  │ cxx bridge
┌─────────────────▼───────────────────────────────────────┐
│  3. src/engine.h + engine.cpp  — C++ TensorRT core      │
│                                                         │
│  Engine lifecycle:                                      │
│    ┌──────────────┐                                     │
│    │ Constructor   │ Set device, init spdlog logger      │
│    └──────┬───────┘                                     │
│           ▼                                             │
│    ┌──────────────┐                                     │
│    │ load()       │ Read .engine file from disk          │
│    │              │ createInferRuntime → IRuntime        │
│    │              │ deserializeCudaEngine → ICudaEngine  │
│    │              │ createExecutionContext → IExecContext │
│    │              │ cudaStreamCreate (1 inference stream)│
│    │              │ Enumerate I/O tensors:               │
│    │              │   • Cache metadata (name/dtype/dims) │
│    │              │   • cudaMallocAsync per tensor       │
│    │              │   • Extract batch dim (min/opt/max)  │
│    └──────┬───────┘                                     │
│           ▼                                             │
│    ┌──────────────┐                                     │
│    │ infer()      │ Per-call hot path:                   │
│    │              │   1. Map input tensors by name       │
│    │              │   2. Validate batch size constraints │
│    │              │   3. setInputShape (dynamic batch)   │
│    │              │   4. cudaMemcpyAsync H→D (inputs)    │
│    │              │   5. setTensorAddress for all I/O    │
│    │              │   6. enqueueV3 (launch inference)    │
│    │              │   7. cudaMemcpyAsync D→H (outputs)   │
│    │              │   8. cudaStreamSynchronize           │
│    │              │   9. Return Vec<OutputTensor>        │
│    └──────┬───────┘                                     │
│           ▼                                             │
│    ┌──────────────┐                                     │
│    │ ~Engine()    │ cudaFree all buffers                 │
│    │              │ cudaStreamDestroy                    │
│    └──────────────┘ (never throws — logs errors)        │
│                                                         │
│  No cudaStreamSynchronize between H→D and enqueueV3:   │
│  • Same CUDA stream = ordered execution                 │
│  • Pageable memory cudaMemcpyAsync blocks CPU anyway    │
│  • TRT manages auxiliary streams internally             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  4. flake.nix  — Nix dev shell                          │
│     • jetpack-nixos for aarch64 CUDA + TensorRT         │
│     • clang++ compiler, spdlog, fmt, cxx-rs             │
│     • Sets TENSORRT_LIBRARIES, CUDA_LIBRARIES, etc.     │
└─────────────────────────────────────────────────────────┘
```

### Data Flow (one inference call)

```
Rust Vec<u8> (host heap)
    │   cudaMemcpyAsync H→D
    ▼   (pageable → blocks CPU, staged to GPU)
GPU buffer (cudaMalloc'd at load time)
    │   context->enqueueV3(stream)
    ▼   (TensorRT kernel execution on same stream)
GPU output buffer
    │   cudaMemcpyAsync D→H
    ▼   cudaStreamSynchronize
Rust Vec<u8> (allocated via new_output_buffer)
    │   reinterpret bytes as f32/u8/i64 in Rust
    ▼
OutputTensor { name, data: Vec<u8>, dtype }
```

### Examples Breakdown

| Example | Purpose |
|---------|---------|
| `basic.rs` | Load engine, create zero inputs, run N iterations |
| `benchmark.rs` | Warmup 1024 iters, then measure latency/throughput at batch 1/2/4/8/16 |
| `dynamic.rs` | Exercise dynamic batch size range (min/opt/max) |
| `functional_test.rs` | Load known input.bin, compare output against features.txt |
| `onnx_tensorrt_compare.rs` | Load both ONNX (via `ort` crate) and TRT engine, compare outputs for correctness + benchmark both |

---

## How This Relates to Our Existing Stack

### Our Stack Today

```
                    ┌──────────────────────────┐
                    │  Cioffi TCN (250K params) │
                    │  Input: (1,6,50) IMU      │
                    │  Output: (1,3) Δposition  │
                    └─────────┬────────────────┘
                              │
              ┌───────────────┼───────────────────┐
              ▼               ▼                   ▼
    ┌─────────────┐  ┌───────────────┐  ┌──────────────┐
    │  PyTorch    │  │  tinygrad NV  │  │  TensorRT    │
    │  (baseline) │  │  @TinyJit +   │  │  (trtexec)   │
    │             │  │  HCQGraph     │  │              │
    └─────────────┘  └───────┬───────┘  └──────────────┘
                             │
                    ┌────────▼────────┐
                    │  C Hot Path     │
                    │  raw MMIO replay│
                    │  ~46 µs/iter    │
                    └─────────────────┘
```

### Where libinfer Fits

libinfer gives us a **Rust-native TensorRT path** — the missing piece:

```
              ┌───────────────┼───────────────────┐
              ▼               ▼                   ▼
    ┌─────────────┐  ┌───────────────┐  ┌──────────────────┐
    │  tinygrad NV│  │  C Hot Path   │  │  libinfer (Rust) │
    │  Python     │  │  MMIO replay  │  │  TensorRT via cxx│
    │  ~107 µs    │  │  ~46 µs       │  │  ~??? µs         │
    └─────────────┘  └───────────────┘  └──────────────────┘
```

---

## TinyGrad Integration Possibilities

### 1. Replace TensorRT Backend with TinyGrad NV — "libinfer-tinygrad"

**Concept**: Keep libinfer's Rust API surface but swap the C++ TensorRT internals for tinygrad's NV pipeline.

**How it could work**:
- tinygrad compiles models to PTX/cubin (it already does this with NV=1)
- Instead of TensorRT's engine file, you'd load tinygrad's JIT-captured HCQGraph
- The Rust side still does `engine.infer(&inputs) → outputs`
- Under the hood: direct MMIO doorbell poke instead of CUDA runtime calls

**Challenges**:
- tinygrad's NV backend is deeply Python — the HCQGraph, command queue builder, signal management, and MMIO doorbell are all in Python/ctypes
- Would need to either:
  - (a) Serialize the HCQGraph + command queue to a static binary format that Rust/C can replay
  - (b) Use PyO3 to call tinygrad from Rust (defeats the purpose)
  - (c) Rewrite the HCQGraph replay loop in Rust (this is essentially what the C hot path already does!)

**Verdict**: Option (c) is essentially "port the C hot path to Rust" — which is already a good idea on its own (see below).

### 2. Rust Hot Path — Port C MMIO Replay to Rust

**Concept**: Rewrite `hot_path.c` in Rust, gaining:
- Memory safety for mmap'd GPU buffers (no stale pointer UB)
- Zero-cost abstractions for the patch list / signal management
- `Send`/`!Sync` type safety matching the actual GPU execution model
- Integration with libinfer's crate ecosystem

**What to port (from hot_path.c, ~200 lines of actual logic)**:
```rust
// Core data structure
struct HotPathConfig {
    io_buf_in: *mut u8,           // Tegra unified memory input
    io_buf_out: *const u8,        // Tegra unified memory output
    gpfifo_ring: *mut u64,        // GPFifo ring buffer
    gpput: *mut u32,              // GPFifo put pointer
    mmio_doorbell: *mut u32,      // GPU MMIO page (offset 0x90)
    cmdq_gpu_va: u64,             // Command queue GPU virtual address
    timeline_signal: AtomicU64,   // Timeline semaphore
    kick_signal: AtomicU64,       // Kickoff signal
    patches: Vec<PatchEntry>,     // Command queue patch list
    gpfifo_entry: u64,            // Pre-computed ring entry
}

// The hot loop (~10 lines of actual work)
fn run_iteration(&self, input: &[u8], output: &mut [u8]) {
    self.wait_signal(&self.timeline_signal, expected);  // spin-poll
    self.io_buf_in.copy_from_slice(input);               // unified mem
    self.apply_patches(kickoff, timeline);                // patch CQ
    self.submit_gpfifo();                                 // dmb sy + MMIO
    self.wait_signal(&self.timeline_signal, expected+1);  // spin-poll
    output.copy_from_slice(self.io_buf_out);
}
```

**Why Rust is better than C here**:
- The C code uses raw `volatile uint64_t*` and manual `__sync_synchronize()` — Rust's `AtomicU64` with `Ordering::Acquire/Release` is both safer and equally fast
- The mmap'd GPU addresses can be wrapped in owned types with `Drop` impl
- Patch entries use bitmask partial writes — Rust's type system can prevent off-by-one in the mask/shift logic
- Integrates naturally with libinfer's Cargo/Nix build

### 3. Hybrid: TinyGrad Compile → Rust Replay

**Concept**: Use tinygrad as the **compiler** and Rust as the **runtime**.

```
 Python (offline / one-time)          Rust (runtime / hot path)
┌────────────────────────┐          ┌─────────────────────────┐
│ tinygrad NV=1          │          │ Load serialized graph   │
│ model(dummy_input)     │  export  │ mmap GPU buffers        │
│ @TinyJit captures      │ ──────► │ Replay HCQGraph via     │
│ HCQGraph               │  .bin   │ MMIO doorbell           │
│ export_graph.py        │          │ ~46 µs per iteration    │
└────────────────────────┘          └─────────────────────────┘
```

**This is the most exciting path because**:
- tinygrad's BEAM search finds optimal kernel schedules (often beating TensorRT for small models)
- The command queue is a static binary blob — trivial to serialize
- No TensorRT dependency at runtime (just raw MMIO + Tegra unified memory)
- No Python at runtime
- The existing `export_graph.py` already extracts everything needed (gpfifo, patches, buffer addresses, signal locations)

**What's needed**:
1. A serialization format for HCQGraph state (JSON/msgpack/flatbuffers)
2. Rust deserializer + mmap setup
3. Rust GPFifo submission + MMIO doorbell (trivial — ~30 lines of unsafe)
4. Signal spin-wait with atomic acquire semantics

### 4. Side-by-Side Benchmark: libinfer (TensorRT) vs tinygrad NV

**Concept**: Use the same Cioffi TCN model, benchmark both paths on AGX Orin.

```bash
# Export TCN to ONNX → TensorRT engine
python cioffi_tcn.py --export-onnx
trtexec --onnx=cioffi_tcn.onnx --saveEngine=cioffi_tcn.engine

# Benchmark via libinfer
cargo run --example benchmark -- --path cioffi_tcn.engine -i 32768

# Benchmark via tinygrad NV (existing)
NV=1 python bench_cioffi_tcn.py

# Benchmark via C hot path (existing)
NV=1 python bench_hot_path.py
```

This gives us a direct comparison: TensorRT-via-Rust vs tinygrad-via-Python vs tinygrad-via-C-MMIO.

---

## libinfer's Approach — What's Good, What's Missing

### Strengths
- **Clean Rust/C++ boundary** via `cxx`: no manual `unsafe extern "C"` bindings, shared type definitions, automatic Pin safety
- **Correct CUDA stream model**: single stream, no redundant syncs between H2D and enqueue, sync only before D2H read
- **Dynamic batch support**: min/opt/max profile shapes, runtime batch size validation
- **Multi-tensor I/O**: handles models with multiple named inputs/outputs (common in transformers)
- **Nix-ready**: ships a `flake.nix` targeting jetpack-nixos (exactly our setup)
- **Destructor safety**: `~Engine()` logs but never throws (correct C++ in a mixed Rust/C++ environment)

### Limitations & Future Improvements

| Area | Current State | Improvement |
|------|--------------|-------------|
| **Transfer overhead** | H2D/D2H via `cudaMemcpyAsync` pageable memory every call | Accept device pointers (zero-copy on Tegra unified memory) |
| **No async** | `infer()` blocks until `cudaStreamSynchronize` | Expose `infer_async()` + `poll()` for pipelined execution |
| **No stream control** | Creates its own internal stream | Accept user CUDA stream for integration with larger pipelines |
| **Engine building** | Must use external `trtexec` or Python API | Add Rust-native ONNX→TRT engine builder |
| **FP16/INT8 awareness** | Handles data types but no runtime quantization | Expose TRT builder flags for precision modes |
| **No graph replay** | Each `infer()` calls `enqueueV3` through the CUDA runtime | CUDA Graphs / HCQ-style static replay for lower latency |
| **Profiling** | None | Expose CUDA events for per-kernel timing |
| **Thread safety** | `Send` but not `Sync` (correct) | Could add `Mutex<Engine>` wrapper for shared access |
| **Pinned memory** | Uses pageable host memory | `cudaMallocHost` for true async H2D (esp. on discrete GPUs) |
| **Tegra optimization** | No special Tegra path | Use unified memory `cudaMallocManaged` + zero-copy on Jetson |

### The Big Gap: Dispatch Latency

libinfer goes through the full CUDA runtime per inference call:
```
Rust → cxx bridge → C++ infer() → cudaMemcpyAsync → enqueueV3 → cudaStreamSync
```

Our C hot path bypasses everything:
```
C → memcpy(unified_mem) → MMIO doorbell → spin-wait
```

The CUDA runtime overhead (ioctl to driver, context switch, etc.) adds ~30-60 µs — which is the entire latency budget for a 20 kHz control loop. libinfer is great for ~1 ms+ inference calls (YOLO, transformers, etc.) but not for the sub-100 µs regime we target.

**This is exactly why the "TinyGrad compile → Rust replay" approach (option 3) is the sweet spot**: you get tinygrad's kernel optimization + Rust's safety + raw MMIO dispatch latency.

---

## Concrete Next Steps

1. **Benchmark libinfer on our TCN** — Export Cioffi TCN → TRT engine, run `benchmark.rs` to get baseline TensorRT-via-Rust numbers
2. **Port C hot path to Rust** — Replace `hot_path.c` with a Rust crate (keep the same `export_graph.py` pipeline)
3. **Serialize HCQGraph** — Add a binary export format to `export_graph.py` that the Rust crate can load
4. **Unify the API** — Make both TensorRT (via libinfer) and tinygrad-MMIO (via Rust hot path) implement a common `Infer` trait:
   ```rust
   trait Infer {
       fn infer(&mut self, input: &[u8]) -> &[u8];
       fn input_dims(&self) -> &[usize];
       fn output_dims(&self) -> &[usize];
   }
   ```
5. **Tegra zero-copy for libinfer** — PR upstream to add `cudaMallocManaged` path for Jetson, eliminating H2D/D2H entirely
