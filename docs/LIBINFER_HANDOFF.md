# libinfer Optimization Handoff — For the Next AI

This document summarizes everything done so far on Saronic's [libinfer](https://github.com/saronic-technologies/libinfer) TensorRT→Rust inference library, what remains, and what would bring real engineering value from a first-principles perspective.

**Platform**: Jetson AGX Orin 64GB · NixOS · CUDA 12.6 · TensorRT 10.7 · SM 8.7  
**Codebase**: `external/libinfer/` (git submodule, branch `feat/cuda-graph-replay`)  
**Parent repo**: `agx-orin-dev-kit/` (branch `saronic-dev`)

---

## 1. Git History — Where We Left Off

### libinfer (`external/libinfer/`)

```
a9a4d3e bench: baseline comparison benchmark + report
fc52327 docs: update benchmark report with dynamic input changes and latest numbers
728440c feat: heterogeneous dynamic inputs + safety/perf improvements
d182d65 feat: Direct I/O buffers + 5-mode benchmark (1.18× pipeline, 34→40 Hz)
9a5dce5 feat: CUDA Graph capture/replay + zero-copy fast path for TensorRT inference
be20561 (origin/main) publish version 0.0.5
23b9087 fix: 11x inference regression from FFI resize loop and spurious stream sync
```

**Commit `728440c`** — Incorporated upstream dynamic inputs + safety/perf rewrite:
- Complete rewrite of engine.h, engine.cpp, lib.rs merging upstream
  per-input heterogeneous dynamic shape support with our CUDA graph, managed
  memory, and Direct I/O optimizations
- Per-input `InputShapeProfile` from TRT optimization profiles
- Output buffers sized via TRT shape propagation (not manual mOutputLengths)
- Pre-computed `mInputIndices`/`mOutputIndices` for O(1) hot-path lookups
- Shape caching via `mLastInputShapes` to skip redundant setInputShape() calls
- Safety: getDataTypeSize() default case, bounds checking, std::call_once, Send docs
- Removed raw pointer APIs in favor of safe write/read buffer API
- Backward-compatible `get_batch_dims()` via `get_input_shape_profiles()`

**Commit `d182d65`** — Added safe Direct I/O API + 5-mode isolated benchmark:
- `write_input_buffer(index, &[u8])` / `read_output_buffer(index, &mut [u8])` — safe Rust slice API, no `unsafe` needed in user code
- Adapts automatically: `memcpy` for managed mem, `cudaMemcpyAsync` for device mem
- 5-mode benchmark isolating each optimization layer independently
- `BENCHMARK_REPORT.md` — full results document

**Commit `9a5dce5`** — Added CUDA graph capture/replay + IO_COHERENCE managed memory:
- `capture_cuda_graph(batch_size)` / `run_cuda_graph()` in engine.h/cpp
- `Engine::new_managed()` / `Engine::new_cuda_graph_managed()` in lib.rs
- Original 3-mode benchmark in saronic_demo_pipeline.rs

### Benchmark Results (4,096 iters + 15s sustained, Jetson AGX Orin 64GB)

#### Single-Model (YOLOv8n FP16)

| Engine | Stock µs | CUDA Graph µs | Speedup |
|---|---|---|---|
| YOLOv8n 640 FP16 | 4,102 | 3,886 | **1.06×** |
| YOLOv8n 320 FP16 | 1,866 | 1,635 | **1.14×** |

#### 3-Model Pipeline (det → seg → pose, 15s sustained)

| Mode | Hz | Median µs | P99 µs |
|---|---:|---:|---:|
| 1. Stock | 34.0 | 29,350 | 30,317 |
| 2. CUDA Graph | 37.5 | 26,545 | 27,305 |
| 3. IO_COHERENCE | 34.7 | 28,860 | 29,581 |
| 4. IO_COHERENCE + Graph | 38.5 | 25,917 | 26,481 |
| 5. **Direct I/O** (managed+graph) | **39.9** | **25,160** | **25,464** |

Production impact: **+6 Hz (34 → 40)**, 4,275 µs/frame freed, P99 tail 16% tighter.

Additive breakdown: CUDA Graph −2,748µs (64%), IO_COHERENCE −652µs (15%), Direct I/O −874µs (20%).

---

## 2. Files You Need to Know

| File | Lines | What |
|---|---|---|
| `external/libinfer/src/engine.h` | ~187 | C++ Engine class declaration. TensorMetadata, index vectors, shape cache, CUDA graph state. |
| `external/libinfer/src/engine.cpp` | ~968 | C++ implementation. TRT runtime, per-input dynamic shapes, CUDA graph, managed memory, Direct I/O. |
| `external/libinfer/src/lib.rs` | ~295 | Rust FFI via cxx bridge. InputShapeProfile, ShapeProfile, BatchDims, Send safety docs. |
| `external/libinfer/build.rs` | ~35 | Cargo build script. Links CUDA/TRT/spdlog. |
| `external/libinfer/Cargo.toml` | ~50 | Dependencies (cxx 1.0.116, clap, tracing, ort). |
| `external/libinfer/examples/saronic_demo_pipeline.rs` | ~510 | 5-mode benchmark. Good template for production usage. |
| `external/libinfer/BENCHMARK_REPORT.md` | ~280 | Full results + technical writeup. |
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

## 3. Upstream Awareness (NOT upstreaming — stealth fork only)

> **IMPORTANT**: We are working on our private fork only (`feat/cuda-graph-replay` branch).
> Do NOT open PRs or issues against `saronic-technologies/libinfer`.
> This section is context-only so you know what Saronic's team is working on upstream.

Saronic's public libinfer has 3 open issues:

| Issue | Title | What It Means For Us |
|---|---|---|
| upstream CUDA graph issue | Support CUDA graphs | We already implemented this on our fork. If they ship their own version, we may need to rebase. Watch for conflicts. |
| upstream device-pointer issue | Implement passing and returning device pointers | We solved this with `write_input_buffer` / `read_output_buffer`. Their approach may differ. |
| upstream CI issue | Cargo Test and CI | No CI exists upstream. We should add tests to our fork to protect our changes. |

**Approved PRs we MUST incorporate** (see Step Zero in §4):
- upstream dynamic-inputs branch — **Heterogeneously dynamic inputs** (branch: `mpw/heterogeneously_dynamic_inputs`, approved)
- upstream dynamic-axes branch — **Better dynamic axes** (branch: `better-dynamic-axes`, approved)

Both are approved and will land on `main` soon. We need to pull these into our fork FIRST so our optimizations build on top of their latest code, not alongside it. This avoids painful merge conflicts later and means our CUDA graph / Direct I/O work benefits from their dynamic input improvements.

notes from the upstream CUDA graph issue:

- "CUDA graphs would be a great optimization to have, but we need to also support dynamic batching, so we will have to cache them."

notes from the upstream device-pointer issue:

- "In the spirit of this library, we want more code is as reasonable to exist on the Rust side versus the C++ side. To that end, it would greatly enhance flexibility if we could pass device pointers to libinfer in order to execute operations. This will allow chained CUDA operations to be feasible without additional host to device memcpys."

notes from the upstream CI issue:

- "Testing is a little tricky right now because of the build process, and because it requires a GPU, but it would be great to get these running."

---

## 4. High-Value Improvements — What To Do Next

### ~~Step Zero: Incorporate Saronic's Approved PRs~~ ✅ DONE

upstream dynamic-inputs's heterogeneous dynamic input approach has been incorporated into our
rewrite (commit `d43e31f`). The engine core now supports per-input shape
profiles, TRT shape propagation for output buffer sizing, and independent
dynamic dimension resolution per input. upstream dynamic-axes's changes are superseded by
upstream dynamic-inputs's approach.

#### Branches to fetch

| PR | Branch | What It Adds |
|---|---|---|
| upstream dynamic-inputs branch | `mpw/heterogeneously_dynamic_inputs` | Heterogeneously dynamic inputs — each input tensor can have independently dynamic dimensions (not just batch). This changes how `setInputShape()` works and how tensor metadata is stored. |
| upstream dynamic-axes branch | `better-dynamic-axes` | Better dynamic axes — improves how optimization profiles handle dynamic dimensions. Changes to engine loading and shape validation. |

#### Remote setup (ALREADY DONE)

```
origin    git@github.com:GiveThanksAlways/libinfer.git  (fetch + push) ← OUR fork
upstream  git@github.com:saronic-technologies/libinfer.git (fetch only) ← Saronic's repo
```

> **STEALTH SAFETY**: The upstream push URL is set to `no_push_to_upstream` so
> `git push upstream` will always fail. We ONLY fetch from upstream. All pushes
> go to `origin` (our fork). If you ever need to re-add the remote:
>
> ```bash
> git remote add upstream git@github.com:saronic-technologies/libinfer.git
> git remote set-url --push upstream no_push_to_upstream
> ```

#### How to do it (all local, stealth)

Branches `upstream/better-dynamic-axes` and `upstream/mpw/heterogeneously_dynamic_inputs`
have already been fetched. To refresh them later: `git fetch upstream`.

```bash
cd external/libinfer

# If branches aren't fetched yet:
git fetch upstream mpw/heterogeneously_dynamic_inputs
git fetch upstream better-dynamic-axes

# Option A: Rebase our work on top of both (preferred — clean history)
git checkout feat/cuda-graph-replay
git rebase upstream/better-dynamic-axes     # older PR first
git rebase upstream/mpw/heterogeneously_dynamic_inputs  # newer PR on top
# Then resolve any conflicts in engine.cpp / lib.rs

# Option B: Merge if rebase gets messy
git merge upstream/better-dynamic-axes --no-edit
git merge upstream/mpw/heterogeneously_dynamic_inputs --no-edit

# Push rebased branch to OUR fork only
git push origin feat/cuda-graph-replay --force-with-lease
```

#### Why this matters

1. **Our `write_input_buffer()` / `read_output_buffer()` must handle dynamic shapes.** upstream dynamic-inputs changes how input dimensions are validated — if we don't incorporate it, our Direct I/O path may break on dynamically-shaped models.
2. **CUDA graph caching for dynamic batches.** The upstream CUDA graph issue explicitly says "we need to also support dynamic batching, so we will have to cache them." upstream dynamic-inputs's heterogeneous dynamic inputs is the prerequisite for this. Our `capture_cuda_graph()` currently only handles a single fixed batch size — we'll need to extend it to cache multiple graphs keyed by input shape.
3. **The upstream device-pointer issue alignment.** They want device pointers to live on the Rust side. upstream dynamic-axes's better dynamic axes changes the Rust FFI surface. If we build our device pointer APIs without this, the types won't match upstream's direction.
4. **Conflict avoidance.** Both PRs touch `engine.cpp`, `engine.h`, and `lib.rs` — the exact files we modified. Incorporating now (while changes are small) is far easier than rebasing after 10 more commits.

#### After incorporating, verify:

```bash
# Rebuild and make sure our benchmark still works
cd /home/agent/agx-orin-dev-kit/examples/libinfer
nix develop --command bash -c "\
  cd ../../external/libinfer && \
  cargo build --release --example saronic_demo_pipeline && \
  echo 'BUILD OK'"
```

If the benchmark still passes with identical numbers (±5%), we're good. If shapes changed, update `DirectBufs` and `make_inputs()` accordingly.

---

### Tier 1: Safety & Correctness ✅ ALL DONE

All Tier 1 items have been completed in commit `d43e31f`:

#### ~~A. Fix `getDataTypeSize()` missing default case~~ ✅
Added `default: throw std::runtime_error("Unsupported tensor data type in getDataTypeSize");`

#### ~~B. Add bounds validation to `write_input_buffer()` / `read_output_buffer()`~~ ✅
Both methods now validate `data.size()` against `mBufferSizes[metaIdx]` and throw on overflow.

#### ~~C. Remove/gate the raw pointer APIs~~ ✅
Removed `get_input_buffer_ptr()` / `get_output_buffer_ptr()`. Only safe `write_input_buffer` / `read_output_buffer` remain.

#### ~~D. Document thread safety model properly~~ ✅
`unsafe impl Send for ffi::Engine {}` now has full safety documentation explaining Engine is Send but not Sync.

#### ~~E. Add `std::call_once` for logger initialization~~ ✅
Logger init uses `static std::once_flag sLoggerInitFlag` with `std::call_once` in the Engine constructor.

### Tier 2: Performance (More µs to reclaim)

#### ~~F. Pre-compute tensor index mappings during `load()`~~ ✅
Pre-computed `mInputIndices` / `mOutputIndices` vectors built during `load()`, enabling O(1) lookup on every hot path.

#### ~~G. Cache `setInputShape()` calls~~ ✅
`mLastInputShapes` vector stores previous shapes per tensor. `setInputShape()` is only called when shape actually changes (via `dimsEqual()` comparison).

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

#### K. Add CI and integration tests to our fork
**What**: Neither upstream nor our fork has CI. Add a local test harness that:
- Validates the cxx bridge compiles
- Runs unit tests with mock engine files
- Regression-tests the Direct I/O path against known-good outputs
- Tests the nix flake builds
**Value**: Protects our fork from regressions (upstream had an 11× regression in a previous release — we don't want that).

#### L. Track upstream changes for rebase-readiness
**What**: Periodically check `saronic-technologies/libinfer` main for new commits (especially the heterogeneous dynamic inputs feature). If they land CUDA graph support themselves, our fork will need careful conflict resolution.
**Action**: `git fetch origin && git log origin/main --oneline` to see what's new.

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
| **P0** | **Incorporate approved upstream dynamic input branches (Step Zero)** | **Foundation — do first** | **Medium** |
| **P0** | Fix `getDataTypeSize()` UB (A) | Safety | Trivial |
| **P0** | Add bounds checking to write/read buffers (B) | Safety | Easy |
| **P1** | Pre-compute tensor index maps (F) | −100 µs | Medium |
| **P1** | Eliminate HashMap in `infer()` (H) | −300 µs | Medium |
| **P1** | Cache `setInputShape()` (G) | −50 µs | Easy |
| **P1** | Add CI / regression tests to fork (K) | Protects our work | Medium |
| **P2** | Deprecate raw pointer APIs (C) | Safety posture | Easy |
| **P2** | Track upstream for rebase-readiness (L) | Avoids drift | Low |
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

- **License**: MPL-2.0. Our fork changes are derivative works under the same license. We are NOT upstreaming at this time — all work stays on our private fork (`feat/cuda-graph-replay` branch in `external/libinfer/`, `saronic-dev` branch in parent repo).

## extra context

```text

extra context for <!-- upstream dynamic-axes branch -->:
------------------------------------------
Description
Added support for as many dynamic axes as you want wherever you want. It doesn't just have to be the first dimension of a tensor.

Testing
All the examples run
Also integrated into a certain saronic downstream program and it works

Notes
Finished adding support for dynamic axes in libinfer wherever whenever: Some comparison

libinfer 0.0.4 DETR benchmark.rs

2025-08-31T21:32:50.467814Z  INFO benchmark: inference calls    : 4096
2025-08-31T21:32:50.467820Z  INFO benchmark: total latency      : 54.53879
2025-08-31T21:32:50.467823Z  INFO benchmark: avg. frame latency : 0.0066575673
2025-08-31T21:32:50.467825Z  INFO benchmark: avg. frame fps     : 150.20502
2025-08-31T21:32:50.467826Z  INFO benchmark: avg. batch latency : 0.013315135
2025-08-31T21:32:50.467828Z  INFO benchmark: avg. batch fps     : 75.10251
libinfer 0.0.5 (dynamic axes of death) DETR benchmark.rs

2025-08-31T21:29:09.053947Z  INFO benchmark: inference calls    : 4096
2025-08-31T21:29:09.053951Z  INFO benchmark: total latency      : 30.404646
2025-08-31T21:29:09.053954Z  INFO benchmark: avg. batch latency : 0.0074230093
2025-08-31T21:29:09.053955Z  INFO benchmark: avg. batch fps     : 134.71625
libinfer 0.0.4 yolov8

2025-08-31T21:37:40.372793Z  INFO benchmark: inference calls    : 4096
2025-08-31T21:37:40.372798Z  INFO benchmark: total latency      : 50.759754
2025-08-31T21:37:40.372800Z  INFO benchmark: avg. frame latency : 0.012392518
2025-08-31T21:37:40.372802Z  INFO benchmark: avg. frame fps     : 80.69385
2025-08-31T21:37:40.372803Z  INFO benchmark: avg. batch latency : 0.012392518
2025-08-31T21:37:40.372805Z  INFO benchmark: avg. batch fps     : 80.69385
libinfer 0.0.5 yolov8

2025-08-31T21:39:45.790839Z  INFO benchmark: inference calls    : 4096
2025-08-31T21:39:45.790845Z  INFO benchmark: total latency      : 59.263123
2025-08-31T21:39:45.790847Z  INFO benchmark: avg. batch latency : 0.014468536
2025-08-31T21:39:45.790849Z  INFO benchmark: avg. batch fps     : 69.11549
I found a major optimization bug where we were prematurely synchronizing the cuda stream. I introduced this in 0.0.4. By removing this we have a pretty massive performance improvement on larger models. Strangely I am getting better performance on the new tracker trained DETR model than yolov8. The DETR model is quite a bit larger and has two transformers so I am suprised. Not complaining though, this is nearly a 2x performance improvement

We are still IO bound on f32 output tensors. Will save that for 0.0.6
```

```text
extra context for: <!-- upstream dynamic-inputs branch -->
-------------------------------------
Adds per-input heterogeneous dynamic shape support; each input tensor now has its own min/opt/max shape profile from TensorRT, replacing the single global batch size. Input buffers allocated per their own max shape; output buffers sized via TensorRT shape propagation after setting all inputs to max. infer() resolves each input's dynamic dimension independently using precomputed metadata (one integer division per dynamic input, zero heap allocations).

Also removed get_output_len() and mOutputLengths, output sizes now queried dynamically from mContext->getTensorShape() after input shapes are set

Questions:

remove get_batch_dims()?
Benchmarks:
FP16 engine (3 tensors, 1 dynamic dim on [1])
dynD	Avg	p50	p99	Throughput
1	5.57ms	5.57ms	5.65ms	179.6 infer/s
16	11.47ms	11.48ms	11.76ms	87.2 infer/s
64	38.96ms	39.07ms	39.67ms	25.7 infer/s
FP32 engine (3 tensors, 1 dynamic dim on [1])
dynD	Avg	p50	p99	Throughput
1	20.00ms	20.00ms	20.18ms	50.0 infer/s
16	37.37ms	37.23ms	39.86ms	26.8 infer/s
64	127.2ms	—	—	~7.9 infer/s
YOLOv8n (static, single input)
Metric	Value
Avg	1.750ms
p50	1.763ms
p99	1.781ms
Min	1.642ms
Max	1.787ms
Throughput	571.5 infer/s
```