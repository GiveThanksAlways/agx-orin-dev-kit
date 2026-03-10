# libinfer Optimization Handoff — For the Next AI

This document summarizes everything done so far on Saronic's [libinfer](https://github.com/saronic-technologies/libinfer) TensorRT→Rust inference library, what remains, and what would bring real engineering value from a first-principles perspective.

**Platform**: Jetson AGX Orin 64GB · NixOS · CUDA 12.6 · TensorRT 10.7 · SM 8.7  
**Codebase**: `external/libinfer/` (git submodule, branch `feat/cuda-graph-replay`)  
**Parent repo**: `agx-orin-dev-kit/` (branch `saronic-dev`)

---

## 1. Git History — Where We Left Off

### libinfer (`external/libinfer/`)

```
d182d65 feat: Direct I/O buffers + 5-mode benchmark (1.18× pipeline, 34→40 Hz)
9a5dce5 feat: CUDA Graph capture/replay + zero-copy fast path for TensorRT inference
be20561 (origin/main) publish version 0.0.5 (#30)
23b9087 fix: 11x inference regression from FFI resize loop and spurious stream sync (#29)
```

**Commit `9a5dce5`** — Added CUDA graph capture/replay + IO_COHERENCE managed memory:
- `capture_cuda_graph(batch_size)` / `run_cuda_graph()` in engine.h/cpp
- `Engine::new_managed()` / `Engine::new_cuda_graph_managed()` in lib.rs
- `get_input_buffer_ptr()` / `get_output_buffer_ptr()` / `get_{input,output}_buffer_size()` / `get_num_{inputs,outputs}()` — raw pointer + size queries
- Original 3-mode benchmark in saronic_demo_pipeline.rs

**Commit `d182d65`** — Added safe Direct I/O API + 5-mode isolated benchmark:
- `write_input_buffer(index, &[u8])` / `read_output_buffer(index, &mut [u8])` — safe Rust slice API, no `unsafe` needed in user code
- Adapts automatically: `memcpy` for managed mem, `cudaMemcpyAsync` for device mem
- 5-mode benchmark isolating each optimization layer independently
- `BENCHMARK_REPORT.md` — full results document

### Parent repo (`agx-orin-dev-kit/`)

```
c8640a9 chore: update libinfer submodule (Direct I/O + 5-mode benchmark)
454df96 feat: libinfer CUDA Graph zero-copy fast path + nix dev shell
```

### Benchmark Results (5000 iters, Jetson AGX Orin 64GB)

| Mode | Pipeline µs | Hz | Technique |
|---|---|---|---|
| 1. Stock `infer()` | 29,410 | 34 | Baseline |
| 2. CUDA Graph | 26,548 | 38 | `infer_cuda_graph()` |
| 3. IO_COHERENCE | 28,856 | 35 | `infer_zerocopy()` via managed mem |
| 4. IO_COHERENCE + Graph | 25,868 | 39 | Combined |
| 5. **Direct I/O** | **24,855** | **40** | `write_input_buffer` + `run_cuda_graph` + `read_output_buffer` |

Additive breakdown (3-model YOLOv8n FP16 pipeline):
- CUDA graph dispatch: **−2,862 µs** (63% of savings)
- IO_COHERENCE memory: **−680 µs** (15%)  
- Direct I/O buffers: **−1,012 µs** (22%)
- **Total: −4,555 µs/frame (1.18×)**

---

## 2. Files You Need to Know

| File | Lines | What |
|---|---|---|
| `external/libinfer/src/engine.h` | ~130 | C++ Engine class declaration. All public API lives here. |
| `external/libinfer/src/engine.cpp` | ~900 | C++ implementation. TRT runtime, CUDA memory, graph capture. |
| `external/libinfer/src/lib.rs` | ~300 | Rust FFI via cxx bridge. Wraps C++ Engine for safe Rust. |
| `external/libinfer/build.rs` | ~35 | Cargo build script. Links CUDA/TRT/spdlog. |
| `external/libinfer/Cargo.toml` | ~50 | Dependencies (cxx 1.0.116, clap, tracing, ort). |
| `external/libinfer/examples/saronic_demo_pipeline.rs` | ~510 | 5-mode benchmark. Good template for production usage. |
| `external/libinfer/BENCHMARK_REPORT.md` | ~260 | Full results + elevator pitch for Saronic. |
| `examples/libinfer/flake.nix` | ~80 | Nix dev shell (CUDA + TRT + spdlog). |

### How to build

```bash
cd examples/libinfer && nix develop --command bash -c \
  "cd ../../external/libinfer && cargo build --release --example saronic_demo_pipeline"
```

### How to run benchmark

```bash
ONNX=/path/to/engines
nix develop --command bash -c "cd ../../external/libinfer && \
  RUST_LOG=info cargo run --release --example saronic_demo_pipeline -- \
    --det  $ONNX/yolov8n_640_fp16.engine \
    --seg  $ONNX/yolov8n_seg_640_fp16.engine \
    --pose $ONNX/yolov8n_pose_640_fp16.engine"
```

Engine files are at: `/home/agent/agx-orin-dev-kit/examples/learned-inertial-odometry/onnx/`

---

## 3. Upstream Issues We Can Address

Saronic's libinfer has 3 open issues on GitHub:

| Issue | Title | Status | Our Work |
|---|---|---|---|
| [#14](https://github.com/saronic-technologies/libinfer/issues/14) | Support CUDA graphs | **We implemented this** — `capture_cuda_graph()` + `run_cuda_graph()` |
| [#16](https://github.com/saronic-technologies/libinfer/issues/16) | Implement passing and returning device pointers | **Partially done** — `get_input_buffer_ptr()` / `get_output_buffer_ptr()` exist, but safe `write_input_buffer` / `read_output_buffer` is the better approach |
| [#15](https://github.com/saronic-technologies/libinfer/issues/15) | Cargo Test and CI | **Not started** — Good contribution opportunity |

Open PRs:
- [#31](https://github.com/saronic-technologies/libinfer/pull/31) — Heterogeneously dynamic inputs (active, 1 approval)
- [#28](https://github.com/saronic-technologies/libinfer/pull/28) — Better dynamic axes (stale)

---

## 4. High-Value Improvements — What To Do Next

### Tier 1: Safety & Correctness (Saronic cares deeply about Rust safety)

#### A. Fix `getDataTypeSize()` missing default case
**File**: `engine.cpp:40-54`  
**Severity**: Critical  
**What**: The `switch` statement has no `default` case. If a new `TensorDataType` is added, it returns uninitialized stack data (UB).  
**Fix**: Add `default: throw std::runtime_error("Unsupported tensor data type");`

#### B. Add bounds validation to `write_input_buffer()` / `read_output_buffer()`
**File**: `engine.cpp:851-890`  
**Severity**: High  
**What**: No check that `data.size()` matches the expected buffer size. A user can write past the end of a GPU buffer → memory corruption.  
**Fix**: Compare `data.size()` against the actual buffer size (stored during `load()`), throw if mismatched.

#### C. Remove/gate the raw pointer APIs
**File**: `engine.h:68-72`, `lib.rs` unsafe blocks  
**What**: `get_input_buffer_ptr()` / `get_output_buffer_ptr()` expose `*mut u8` to GPU memory with no bounds checking. Now that `write_input_buffer()` / `read_output_buffer()` exist, consider deprecating or removing the raw pointer path.  
**Pitch to Saronic**: "Safe Rust API achieving identical performance to raw pointers, with zero `unsafe` in user code."

#### D. Document thread safety model properly
**File**: `lib.rs`  
**What**: `unsafe impl Send for ffi::Engine {}` exists but Engine is NOT thread-safe (single IExecutionContext, no mutex). This is correct — Engine can be *moved* between threads, just not *shared*. But the docs don't explain this. Add clear documentation that Engine is `Send` but not `Sync`, and concurrent inference requires separate Engine instances.

#### E. Add `std::call_once` for logger initialization
**File**: `engine.cpp:108-131`  
**What**: Logger initialization has a TOCTOU race — two threads constructing Engine simultaneously can both enter the `!spdlog::get("libinfer")` block. Fix with `std::call_once`.

### Tier 2: Performance (More µs to reclaim)

#### F. Pre-compute tensor index mappings during `load()`
**File**: `engine.cpp` — impacts `infer()`, `infer_cuda_graph()`, `infer_zerocopy()`, `write_input_buffer()`, `read_output_buffer()`  
**Severity**: High — currently O(n²)  
**What**: Every call to `write_input_buffer(index)` iterates through ALL metadata to find the nth input. Every output read in `infer()` iterates all prior tensors to count outputs. This is O(n²) in tensor count.  
**Fix**: During `load()`, build `std::vector<size_t> mInputTensorIndices` and `mOutputTensorIndices` that map logical input/output index → metadata index. Then all lookups are O(1).  
**Estimated savings**: ~50-100 µs per frame for a 3-model pipeline (eliminates ~18 linear scans per frame).

#### G. Cache `setInputShape()` calls
**File**: `engine.cpp:330-335`  
**What**: `mContext->setInputShape()` is called on EVERY inference even when shape hasn't changed (same batch size, same model). TensorRT validates the shape each time.  
**Fix**: Store `mLastBatchSize` and only call `setInputShape()` when batch size changes. For fixed-batch models (which Saronic uses), this eliminates one TRT API call per inference.

#### H. Eliminate HashMap from `infer()` hot path
**File**: `engine.cpp:318-320`  
**What**: `infer()` constructs an `unordered_map<string, InputTensor*>` on every call to map tensor names to inputs. For a model with 1 input, this is a heap allocation + hash computation + string copy — pure waste.  
**Fix**: Pre-validate tensor ordering once during the first call, then use direct index mapping. Or require inputs be passed in the same order as the engine's metadata.  
**Estimated savings**: ~200-300 µs per frame (the HashMap+string alloc is measurably ~100µs per model).

#### I. Investigate multi-stream pipelining
**What**: Currently all 3 models run sequentially on the same CUDA stream. Detection doesn't depend on segmentation — they could overlap.  
**Approach**: Create separate IExecutionContext + CUDA stream per model. Launch all 3 inference calls, then synchronize once. On Orin (2048 CUDA cores), there's enough SM occupancy for overlap.  
**Risk**: TRT memory fragmentation. Need to measure whether overlapping small models actually helps vs. saturating the GPU with one model at a time.  
**Potential**: If det+seg+pose overlap even 30%, that's another ~8 Hz.

#### J. Pinned host memory for device-memory path
**What**: For Mode 2 (CUDA graph on device memory), H2D copies use pageable memory (`Vec<u8>` on heap). `cudaMemcpyAsync` with pageable memory secretly synchronizes. Allocating inputs with `cudaMallocHost` (pinned) enables truly async H2D overlap with compute.  
**Fix**: Add `Engine::new_pinned()` constructor that allocates host-side staging buffers with `cudaMallocHost`. Return slices to pinned memory.  
**Benefit**: Async H2D while previous graph is still executing.

### Tier 3: Ecosystem & Production Value

#### K. Contribute CUDA graph support upstream (closes [#14](https://github.com/saronic-technologies/libinfer/issues/14))
**What**: Clean up our `capture_cuda_graph()` / `run_cuda_graph()` API to match libinfer's code style and submit as a PR against `saronic-technologies/libinfer` main.  
**Includes**: The safe `write_input_buffer` / `read_output_buffer` API (closes most of [#16](https://github.com/saronic-technologies/libinfer/issues/16)).

#### L. Add CI and integration tests (addresses [#15](https://github.com/saronic-technologies/libinfer/issues/15))
**What**: Saronic has no CI for libinfer. Create a GitHub Actions workflow that:
- Builds on x86_64 (CUDA stub) and aarch64 (cross-compile)
- Runs unit tests with mock engine files
- Validates the cxx bridge compiles
- Tests the nix flake builds
**Value**: Shows engineering maturity, prevents regressions like the 11× regression in #29.

#### M. Integrate libdebayer into the pipeline
**What**: Saronic also has [libdebayer](https://github.com/saronic-technologies/libdebayer) — CUDA-accelerated Bayer demosaicing for camera sensors. In a real perception pipeline, raw Bayer frames → debayer → YOLOv8.  
**Optimization**: If libdebayer outputs to device memory and libinfer accepts device pointers (via `write_input_buffer` on managed mem or the raw pointer API), you can skip the GPU→CPU→GPU round-trip between debayer and inference.  
**Architecture**: `camera → GPU debayer → [device memory] → GPU inference → CPU postprocess`  
**Potential**: Eliminates one full frame copy (~3-5 ms for 1080p Bayer).

#### N. Multi-model engine sharing (TRT engine cache)
**What**: Loading 15 engines (5 modes × 3 models) uses 254 MiB GPU. In production, you'd want one engine per model with runtime-selectable optimization level.  
**Design**: Engine pool that lazily captures CUDA graphs on first use, shares TRT runtime across instances.

---

## 5. TinyGrad vs. TensorRT — Is It Worth Trying?

**Short answer**: For YOLOv8-class models, TensorRT will win. But it's worth benchmarking to quantify the gap.

**Why TRT wins on YOLO**:
- TRT's layer fusion (conv+bn+relu → single kernel) is heavily optimized for the YOLO architecture
- INT8 calibration with TRT is production-ready; tinygrad doesn't support INT8
- TRT's `enqueueV3` path is extremely mature for fixed-topology networks

**Where tinygrad could compete**:
- Dynamic models (transformers, attention with variable sequence length)
- Models where TRT's fusion heuristics don't fire
- Development speed — tinygrad can JIT compile and run a model in seconds vs. TRT's multi-minute engine build

**What to try**: Build the same YOLOv8n-pose model in tinygrad (it supports ONNX import), run on the same Orin with `NV=1`, and compare latency. Our tinygrad TegraIface work (`external/tinygrad/`) already has the NV backend running. If tinygrad is within 2× of TRT on these models, that's interesting. If it's 5×+ slower, TRT is the clear choice for production.

**Benchmark script to write**:
```python
# Compare TinyGrad NV backend vs TensorRT via libinfer
# Same model, same input, same hardware
import tinygrad
from tinygrad import Tensor
# ... load ONNX, run both, compare
```

---

## 6. Saronic's Other Repos — Cross-Cutting Opportunities

| Repo | What | Opportunity |
|---|---|---|
| [libdebayer](https://github.com/saronic-technologies/libdebayer) | CUDA Bayer→RGB (Menon 2007) | Integrate device→device with libinfer (skip CPU roundtrip) |
| [libsbf-rs](https://github.com/saronic-technologies/libsbf-rs) | Saronic Binary Format (telemetry serialization) | Could carry inference metadata (latency, confidence) in telemetry stream |
| [libpid-rs](https://github.com/saronic-technologies/libpid-rs) | PID controller crate | Inference output → PID → actuator. Could profile end-to-end control loop latency |
| [zencan](https://github.com/saronic-technologies/zencan) (fork) | CANOpen for Rust | CAN bus is the actuator interface — inference latency directly impacts control Hz |
| [foxglove-protos](https://github.com/saronic-technologies/foxglove-protos) | Foxglove viz protobufs | Could publish inference results + timing to Foxglove for real-time monitoring |
| [hubris](https://github.com/saronic-technologies/hubris) (fork) | Oxide's embedded kernel | Tells us Saronic uses microcontrollers too — libinfer optimization frees CPU for MCU comms |
| [state_predictor_challenge](https://github.com/saronic-technologies/state_predictor_challenge) | State estimation challenge | They care about sensor fusion — our learned-inertial-odometry work is directly relevant |

**The big picture**: Saronic builds autonomous surface vessels. The perception pipeline is:
```
Camera (Bayer) → libdebayer → libinfer (YOLO det/seg/pose) → state estimation → libpid → zencan → actuators
```
Every microsecond we save in libinfer is a microsecond available for state estimation and control. At 40 Hz instead of 34 Hz, the control loop gets 17% more frequent updates, which reduces tracking error in rough seas.

---

## 7. Summary of Recommended Priority

| Priority | Task | Estimated Impact | Difficulty |
|---|---|---|---|
| **P0** | Fix `getDataTypeSize()` UB (A) | Safety | Trivial |
| **P0** | Add bounds checking to write/read buffers (B) | Safety | Easy |
| **P1** | Pre-compute tensor index maps (F) | −100 µs | Medium |
| **P1** | Eliminate HashMap in `infer()` (H) | −300 µs | Medium |
| **P1** | Cache `setInputShape()` (G) | −50 µs | Easy |
| **P1** | Upstream PR for CUDA graphs (#14) (K) | Community value | Medium |
| **P2** | Deprecate raw pointer APIs (C) | Safety posture | Easy |
| **P2** | Add CI (L) | Engineering maturity | Medium |
| **P2** | Multi-stream pipelining (I) | +8 Hz potential | Hard |
| **P2** | libdebayer integration (M) | −3-5 ms | Hard |
| **P3** | Pinned memory for device path (J) | Async overlap | Medium |
| **P3** | TinyGrad vs TRT benchmark (§5) | Decision data | Easy |
| **P3** | Logger race fix (E) | Correctness | Trivial |

---

## 8. Key Technical Context for the Next AI

- **IO_COHERENCE**: The Orin's iGPU shares LPDDR5 with the CPU. `cudaMallocManaged` with IO_COHERENCE hints lets GPU and CPU access the same physical pages via the SMMU. This eliminates DMA copies between "host" and "device" — but only helps when I/O size is large enough that DMA overhead > page fault overhead (we saw detection model at 7.6 KB was *slower* with managed mem, while segmentation at 11.8 KB was 1.29× faster).

- **CUDA graphs**: Record the entire TRT execution (all kernel launches) into a single graph object, then replay with one API call. Eliminates the per-layer launch overhead (~400-950 µs for YOLOv8n).

- **The `infer()` overhead problem**: Saronic's original `infer()` creates a `HashMap<String, *mut>`, allocates `Vec<u8>` for each output, copies output data into the new Vec, and validates metadata — every single call. We measured this at ~340 µs per model per call (1020 µs for 3 models). Our `write_input_buffer`/`read_output_buffer` API eliminates all of this.

- **Nix build**: The project uses `examples/libinfer/flake.nix` as the dev shell. It depends on `examples/control-loop/flake.nix` for the nixpkgs pin (so PyTorch etc. are cached). Build with `nix develop --command bash -c "cd ../../external/libinfer && cargo build --release"`.

- **cxx bridge**: libinfer uses [cxx](https://cxx.rs/) for C++↔Rust FFI (not bindgen). The bridge is defined in `lib.rs` — C++ functions are declared in `extern "C++"` blocks and automatically generated. This means any new C++ API needs both a `.h` declaration and a `lib.rs` bridge entry.

- **License**: MPL-2.0. Our changes are derivative works and should be contributed back or kept under the same license.
