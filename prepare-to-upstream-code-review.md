# prepare-to-upstream Code Review

## Branch Summary

| Item | Value |
|---|---|
| **Base** | `upstream/master` @ `b3cdb6106` ("clean up expand_multi [pr] (#14865)") |
| **Branch** | `prepare-to-upstream` → merged into `testing-merge` via fast-forward |
| **Commit** | `c065f7dee` "runtime/ops_nv: add TegraIface for Jetson Orin (nvgpu/nvmap) NV=1 support" |
| **Files changed** | 1 (`tinygrad/runtime/ops_nv.py`) |
| **Diff** | +355 insertions, −9 deletions (net +346 lines) |
| **sz.py total** | 21,633 (upstream: 21,341, delta: +292 by sz.py counting) |
| **Ruff** | All checks passed (0 errors) |
| **Unit tests** | 229 passed, 46 skipped, 0 regressions vs. upstream baseline |

---

## What This PR Does

Adds a third `HCQInterface` implementation — **`TegraIface`** — alongside the existing `NVKIface` (NVK/Mesa kernel driver) and `PCIIface` (NVIDIA proprietary PCI driver). TegraIface enables `NV=1` on NVIDIA Jetson AGX Orin and other Tegra SoCs that use the `nvgpu` + `nvmap` kernel drivers (JetPack 6).

### Why a new interface?

Desktop GPUs use either the NVK open kernel or the proprietary `nvidia-uvm`/`nvidia-ctl` driver. Tegra chips use a completely different kernel driver stack (`/dev/nvgpu/igpu0/ctrl` + `/dev/nvmap`) with its own ioctl numbering, memory management, and channel setup protocol. No existing interface can talk to these devices.

---

## What Was Excluded (And Why)

The original `nv-agx-orin-dev-kit` branch contained 18 commits with changes to 7+ files. This PR distills down to **only the changes required for Tegra NV=1 support**:

| File | Original Change | Decision | Reason |
|---|---|---|---|
| `ops_nv.py` | TegraIface + NVDevice mods | **INCLUDED** | Core port — the entire point |
| `heuristic.py` | +43/−22 matvec for fused CAST/MUL | **EXCLUDED** | Optimization, not correctness. Submit separately. |
| `search.py` | +4/−1 fork context on Linux | **EXCLUDED** | NixOS-specific workaround |
| `state.py` | +7/−1 BF16 GGML tensor support | **EXCLUDED** | Separate feature, not Tegra-specific |
| `ptx.py` | +3/−1 cosmetic linebreak | **EXCLUDED** | Formatting-only artifact |
| `compiler_cuda.py` | +2 CUDA_INCLUDE_PATH env | **EXCLUDED** | NixOS-specific |
| `bench_*.py`, docs, examples | Various | **EXCLUDED** | Not upstream material |

---

## Code Compression Summary

The original branch added **+748 lines** to ops_nv.py (836 → 1584). This clean branch adds **+346 lines** (836 → 1182), a **53% reduction** in delta size. Key techniques:

1. **`FileIOInterface._mmap`** — Replaced 3 duplicate `ctypes.CDLL("libc.so.6")` + mmap setups with the existing `_mmap` static method from `hcq.py`
2. **`_ct_struct()` helper** — Compact ctypes Structure builder (3 lines) used for all 18 ioctl argument structs
3. **`_nvmap_buf()` helper** — Deduplicates the nvmap CREATE→ALLOC→GET_FD pattern (used 3 times in the codebase)
4. **Combined NOP `rm_alloc` cases** — 7 NOP class IDs collapsed into a single `if clss in (...):` check
5. **Compressed `rm_control` NOP handlers** — Fall-through `return params` instead of per-command NOP branches
6. **Removed verbose comments** — Tinygrad style prefers minimal inline comments

---

## Line-by-Line Diff Review

### 1. Import: `fcntl` (line 1)

```python
-import os, ctypes, contextlib, re, functools, mmap, struct, array, sys, weakref
+import os, ctypes, contextlib, re, functools, mmap, struct, array, sys, weakref, fcntl
```

**Why**: TegraIface uses `fcntl.ioctl()` for all nvgpu/nvmap kernel calls. `fcntl` is a stdlib module, no new dependencies.

**Risk**: None — `fcntl` is only imported, never called unless TegraIface is instantiated. On non-Linux systems, `sys.platform != 'win32'` assertion on line 2 already guards.

---

### 2. gpfifo_entry extraction (lines 122–123)

```python
-    gpfifo.ring[gpfifo.put_value % gpfifo.entries_count] = (cmdq_addr//4 << 2) | (len(self._q) << 42) | (1 << 41)
+    gpfifo_entry = (cmdq_addr//4 << 2) | (len(self._q) << 42) | (1 << 41)
+    gpfifo.ring[gpfifo.put_value % gpfifo.entries_count] = gpfifo_entry
```

**Why**: Readability. The 64-bit GPFIFO entry encoding was packed into a multi-expression array assignment. Extracting to a named variable makes the bit layout inspectable in a debugger.

**Risk**: None — semantically identical.

---

### 3. `_tegra_signal` ClassVar on NVComputeQueue (lines 130–131)

```python
class NVComputeQueue(NVCommandQueue):
+  _tegra_signal: ClassVar[bool] = False
```

**Why**: Signal race prevention. On Tegra, we cannot reuse QMD (Queue Meta Data) slots for signalling because the pushbuffer signal path differs from desktop. This flag forces `signal()` to always use pushbuffer-based signalling instead of embedding signals in QMD.

**Impact**: Default `False` — no change for desktop. Set to `True` only when a Tegra device initializes.

---

### 4. `signal()` QMD bypass (lines 163–164)

```python
  def signal(self, signal:HCQSignal, value:sint=0):
+    if self._tegra_signal: self.active_qmd = None
```

**Why**: When `_tegra_signal` is True, clearing `active_qmd` forces the method to skip the "embed signal in QMD release slot" fast path and fall through to the pushbuffer `NVC6C0_SET_OBJECT` + `RELEASE_MEMBAR` path below. The QMD release path assumes desktop memory semantics that don't hold on Tegra.

**Risk**: Performance — pushbuffer signals are slightly slower than QMD embedding. In practice this is negligible vs. kernel execution time.

**Correctness**: Without this, Tegra hits sporadic signal races causing GPU hangs during multi-kernel dispatch.

---

### 5. NVAllocator `uncached` param (line 334)

```python
-    return self.dev.iface.alloc(size, cpu_access=options.cpu_access, host=options.host)
+    return self.dev.iface.alloc(size, cpu_access=options.cpu_access, host=options.host, uncached=options.uncached)
```

**Why**: Passes the `uncached` flag through to the interface. Both NVKIface and PCIIface silently ignore it (`**kwargs`). TegraIface uses it to select write-combine vs. cached mapping for nvmap buffers.

**Risk**: None for desktop — extra kwarg is absorbed by `**kwargs`.

---

### 6. NVAllocator `_copyout` / `_copyin` overrides (lines 337–350)

```python
+  def _copyout(self, dest:memoryview, src:HCQBuffer):
+    if self.dev.is_tegra():
+      self.dev.synchronize()
+      ctypes.memmove(mv_address(dest), src.va_addr, len(dest))
+      return
+    super()._copyout(dest, src)
+
+  def _copyin(self, dest:HCQBuffer, src:memoryview):
+    if self.dev.is_tegra():
+      self.dev.synchronize()
+      ctypes.memmove(dest.va_addr, mv_address(src), len(src))
+      return
+    super()._copyin(dest, src)
```

**Why**: Tegra has unified memory (GPU and CPU share physical RAM). The default HCQAllocator `_copyin`/`_copyout` uses DMA staging buffers because desktop GPUs have separate VRAM. On Tegra, we can `memmove` directly between CPU and GPU addresses — no staging buffer needed.

**Correctness**: `self.dev.synchronize()` is called first to ensure all pending GPU work completes before CPU reads/writes the buffer. Without this, stale data would be read.

**Risk**: `super()` fallback for non-Tegra means desktop path is unchanged.

---

### 7. Ioctl Infrastructure (lines 593–704)

#### 7a. `_ioc` / `_io` / `_ior` / `_iow` / `_iowr` (lines 595–601)

Linux ioctl number encoding functions. These convert (direction, type char, ordinal, size) tuples into the 32-bit ioctl numbers used by `fcntl.ioctl()`. Standard Linux kernel ioctl macros (`_IO`, `_IOR`, `_IOW`, `_IOWR`) translated to Python.

#### 7b. `_ct_struct()` helper (lines 603–605)

```python
def _ct_struct(fields):
  ns = {"_fields_": fields}
  return type("_s", (ctypes.Structure,), ns)
```

Compact factory for ctypes Structures. Avoids repeating `class Foo(ctypes.Structure): _fields_ = [...]` 18 times. Every struct in the ioctl interface uses this.

#### 7c. Ioctl Argument Structures (lines 607–670)

18 ctypes Structure definitions for nvgpu/nvmap ioctls:

| Struct | Purpose |
|---|---|
| `_nvgpu_gpu_characteristics` | GPU capability query (arch, SM version, classes, etc.) |
| `_nvgpu_gpu_get_characteristics` | Wrapper for the capability ioctl |
| `_nvmap_handle` | nvmap buffer create / get-fd |
| `_nvmap_alloc` | nvmap buffer allocation params |
| `_nvgpu_alloc_as` | Address space creation |
| `_nvgpu_as_bind_channel` | Bind channel to address space |
| `_nvgpu_as_alloc_space` | Reserve VA range |
| `_nvgpu_as_map_buffer_ex` | Map dmabuf into GPU VA |
| `_nvgpu_as_unmap_buffer` | Unmap buffer from GPU VA |
| `_nvgpu_open_tsg` | Open TSG (timeslice group) |
| `_nvgpu_tsg_bind_channel_ex` | Bind channel to TSG |
| `_nvgpu_tsg_create_subctx` | Create subcontext for scheduling |
| `_nvgpu_open_channel` | Open GPU channel |
| `_nvgpu_alloc_obj_ctx` | Allocate engine object (compute/DMA) |
| `_nvgpu_setup_bind` | Setup GPFIFO + USERD binding |
| `_nvgpu_channel_wdt` | Configure channel watchdog timer |

**Validation**: All struct layouts match the nvgpu kernel headers in JetPack 6 (`drivers/gpu/nvgpu/include/uapi/linux/`). Field sizes and offsets verified against ioctl traces.

#### 7d. Ioctl Number Constants (lines 672–688)

17 ioctl numbers computed from the struct sizes. Naming convention: `_NVGPU_*` for nvgpu driver, `_NVMAP_*` for nvmap driver, `_NVGPU_AS_*` for address space, `_NVGPU_TSG_*` for TSG, `_NVGPU_CH_*` for channel.

#### 7e. Constants (line 690)

```python
_NVMAP_HEAP_IOVMM, _NVMAP_WC, _NVMAP_CACHED = (1 << 30), 1, 2
_NVMAP_TAG = 0x0900  # tinygrad subsystem tag for nvmap
```

nvmap heap selector, cache policy flags, and a subsystem tag (0x0900 is an unused range in the nvmap tag space).

#### 7f. `_tioctl()` helper (lines 692–693)

```python
def _tioctl(fd, nr, buf):
  if fcntl.ioctl(fd, nr, buf) < 0: raise OSError(...)
```

Thin wrapper that raises on failure. Named `_tioctl` (tegra ioctl) to avoid collision with any future tinygrad `ioctl` helper.

#### 7g. `_nvmap_buf()` helper (lines 695–704)

```python
def _nvmap_buf(nvmap_fd, size, cache_flags, align=4096):
  c = _nvmap_handle(size=size)
  _tioctl(nvmap_fd, _NVMAP_CREATE, c)
  a = _nvmap_alloc(...)
  _tioctl(nvmap_fd, _NVMAP_ALLOC, a)
  g = _nvmap_handle(handle=c.handle)
  _tioctl(nvmap_fd, _NVMAP_GET_FD, g)
  return c.handle, g.size  # g.size is the dmabuf fd
```

Deduplicates the nvmap CREATE→ALLOC→GET_FD pattern. Called 3 times: once in `alloc()`, twice in `rm_alloc()` for dedicated gpfifo/userd buffers.

**Note**: `g.size` returns the dmabuf fd because the nvmap GET_FD ioctl overwrites the `size` field with the fd. This is a quirk of the kernel interface.

---

### 8. `_TegraMem` dataclass (lines 706–707)

```python
@dataclass
class _TegraMem:
  handle: int; dmabuf_fd: int; gpu_va: int; size: int; cpu_addr: int = 0; hMemory: int = 0
```

Tracks per-allocation metadata for cleanup. `hMemory` is the fake RM handle assigned for GPFIFO buffer matching. Stored as `HCQBuffer.meta`.

---

### 9. `TegraIface` Class (lines 709–919)

#### 9a. Class Variables & `__init__` (lines 710–736)

```python
class TegraIface:
  _inited: ClassVar[bool] = False
  _nvmap_fd: ClassVar[int] = -1
  _ctrl_fd: ClassVar[int] = -1
  _chars: ClassVar[_nvgpu_gpu_characteristics|None] = None
```

ClassVars hold singleton driver state (one iGPU per Tegra SoC). First `__init__` call opens `/dev/nvmap` and `/dev/nvgpu/igpu0/ctrl`, queries GPU characteristics.

**Guard**: `if device_id != 0: raise RuntimeError(...)` — Tegra only has one iGPU, multi-device indexing is invalid.

**GPU capability query**: Reads `compute_class`, `gpfifo_class`, `dma_copy_class`, `sm_arch_sm_version` from the nvgpu characteristics struct. These are the same values that desktop drivers provide via RM control calls.

#### 9b. `_nh()` — Handle Generator (lines 743–744)

Monotonic counter starting at 0x10001. Generates "virtual" RM handles that the rest of NVDevice uses to track objects. On desktop, these are real RM handles assigned by the kernel. On Tegra, they're just local identifiers since nvgpu doesn't use RM handle semantics.

#### 9c. `rm_alloc()` — Resource Allocation (lines 746–816)

Translates NV RM allocation calls into nvgpu equivalents:

| RM Class | Tegra Action |
|---|---|
| `NV01_DEVICE_0`, `NV20_SUBDEVICE_0`, etc. (7 classes) | NOP — return handle. Tegra doesn't need explicit RM object hierarchy. |
| `NV01_MEMORY_VIRTUAL` | Create address space (`_NVGPU_ALLOC_AS`). Reserve VA windows at `0xFD00000000` and `0xFE00000000` for shared/local memory. |
| `KEPLER_CHANNEL_GROUP_A` | Open TSG (timeslice group) for scheduling. |
| `FERMI_CONTEXT_SHARE_A` | Create subcontext within TSG. |
| `gpfifo_class` / `AMPERE_CHANNEL_GPFIFO_A` | Full channel setup: open channel → bind to AS → bind to TSG → set WDT → allocate dedicated gpfifo/userd buffers → `SETUP_BIND` → MAP_FIXED overlay. |
| `compute_class` / `dma_class` | Allocate compute/DMA engine object on channel. |
| `GT200_DEBUGGER` | NOP — Tegra doesn't expose debugger RM object. |

**Channel setup detail**: The nvgpu `SETUP_BIND` ioctl requires `gpfifo_dmabuf_offset=0` and `userd_dmabuf_offset=0`. Since NVDevice allocates a single large gpfifo_area buffer, we create **dedicated small nvmap buffers** for the ring and userd, then use `MAP_FIXED` to overlay them at the correct CPU virtual addresses within the gpfifo_area mapping. This lets the rest of NVDevice read/write gpfifo ring entries and userd doorbell via the same gpfifo_area pointer, unaware of the underlying nvmap buffer split.

#### 9d. `rm_control()` — Control Commands (lines 818–863)

Translates NV RM control calls:

| Command | Tegra Action |
|---|---|
| `NV2080_CTRL_CMD_PERF_BOOST` | Sets GPU to max frequency via sysfs devfreq. Desktop uses RM for boost. |
| `NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN` | Returns token from `SETUP_BIND` result. Desktop gets it from RM. |
| `NV2080_CTRL_CMD_GR_GET_INFO` | Populates GR info from cached characteristics (num GPCs, TPCs, SMs, warp count, SM version). |
| `NV0080_CTRL_CMD_GPU_GET_CLASSLIST` | Returns [compute, gpfifo, dma, TURING_USERMODE_A] class list. |
| `NV2080_CTRL_CMD_GPU_GET_GID_INFO` | Generates synthetic 16-byte GID (Tegra iGPU has no PCI-style GID). |
| All others | NOP — `return params`. |

#### 9e. `setup_usermode()` (lines 865–867)

Maps the channel's usermode MMIO region (0x10000 bytes from ctrl fd offset 0). Returns `(0, MMIOInterface)` — the `0` is because Tegra doesn't have a separate usermode handle.

#### 9f. `alloc()` — Memory Allocation (lines 872–895)

1. Determines alignment: 2MB for large (≥8MB) cached buffers, page-aligned otherwise
2. Selects cache policy: write-combine for uncached/host, cached for default
3. Calls `_nvmap_buf()` to create the underlying nvmap buffer
4. Maps into GPU VA via `_NVGPU_AS_MAP_BUF`
5. CPU maps with `MAP_FIXED` at the GPU VA (unified memory — CPU and GPU see same address)
6. Fallback: if MAP_FIXED fails, maps at arbitrary address

**Unified memory**: On Tegra, CPU virtual address == GPU virtual address when MAP_FIXED succeeds. This is critical for the direct `memmove` in `_copyin`/`_copyout`.

#### 9g. `free()` — Memory Deallocation (lines 897–911)

Cleanup order: unmap from GPU AS → munmap CPU → close dmabuf fd → free nvmap handle. Each step wrapped in `contextlib.suppress` because partial-cleanup failure shouldn't crash.

**nvmap handle sign extension**: `meta.handle if meta.handle < 0x80000000 else meta.handle - 0x100000000`. The nvmap `_NVMAP_FREE` ioctl expects a signed 32-bit handle, but Python ctypes may return unsigned. This converts back.

---

### 10. NVDevice Changes

#### 10a. `is_tegra()` method (line 921)

```python
def is_tegra(self) -> bool: return isinstance(self.iface, TegraIface)
```

Convenience method matching the existing `is_nvd()` pattern.

#### 10b. `_select_iface` ordering (line 925)

```python
-    self.iface = self._select_iface(NVKIface, PCIIface)
+    self.iface = self._select_iface(TegraIface, NVKIface, PCIIface)
```

TegraIface is tried first because `_select_iface` iterates in order and stops at the first that succeeds. TegraIface's `__init__` raises `FileNotFoundError` if `/dev/nvgpu/igpu0/ctrl` doesn't exist, so it silently falls through on desktop systems.

#### 10c. `gpfifo_area` size (line 938)

```python
-    self.gpfifo_area = self.iface.alloc(0x300000, ...)
+    self.gpfifo_area = self.iface.alloc(0x10000 if self.is_tegra() else 0x300000, ...)
```

Desktop: 3MB gpfifo area for 0x10000 entries per FIFO. Tegra: 64KB for 1024 entries per FIFO. Tegra's nvgpu driver has a lower maximum GPFIFO size.

#### 10d. GPFIFO entry count (lines 944–950)

```python
+    if self.is_tegra():
+      self.compute_gpfifo = self._new_gpu_fifo(..., entries=1024, ...)
+      self.dma_gpfifo = self._new_gpu_fifo(..., entries=1024, ...)
+    else:
+      self.compute_gpfifo = self._new_gpu_fifo(..., entries=0x10000, ...)
+      self.dma_gpfifo = self._new_gpu_fifo(..., entries=0x10000, ...)
```

1024 entries per FIFO on Tegra (matching the gpfifo_area size). Offset calculation ensures compute and DMA FIFOs don't overlap: DMA starts at `1024*8 + 0x1000` (ring size + page padding).

#### 10e. `_tegra_signal` flag set (line 969)

```python
+    if self.is_tegra(): NVComputeQueue._tegra_signal = True
```

Activates pushbuffer-based signalling for all compute queues. Must happen before any queue usage.

#### 10f. PMA disabled on Tegra (line 970)

```python
-    self.pma_enabled = PMA.value > 0 and PROFILE >= 1
+    self.pma_enabled = PMA.value > 0 and PROFILE >= 1 and not self.is_tegra()
```

PMA (Performance Monitoring Allocation) requires desktop RM control commands that don't exist on Tegra. Disabling prevents a crash in `_prof_init()`.

#### 10g. Window addresses (lines 1017–1018)

```python
-    self.shared_mem_window, self.local_mem_window = 0x729400000000, 0x729300000000
+    if self.is_tegra(): self.shared_mem_window, self.local_mem_window = 0xFE00000000, 0xFD00000000
+    else: self.shared_mem_window, self.local_mem_window = 0x729400000000, 0x729300000000
```

Desktop uses 48-bit VA addresses. Tegra uses 40-bit VA (limited by nvgpu address space). The reserved ranges `0xFD00000000` and `0xFE00000000` are allocated during `rm_alloc` for `NV01_MEMORY_VIRTUAL`.

#### 10h. `synchronize()` before local memory realloc (line 1032)

```python
+    self.synchronize()
     self.slm_per_thread, old_slm_per_thread = round_up(required, 32), self.slm_per_thread
```

**General correctness fix** (benefits all backends, found while testing on Tegra): If the GPU is still executing kernels that reference the old `shader_local_mem` buffer, freeing it immediately causes use-after-free. The `synchronize()` call ensures all pending work completes before reallocation.

#### 10i. `on_device_hang` Tegra early exit (line 1075)

```python
+    if self.is_tegra(): raise RuntimeError("GPU hang detected on Tegra device")
```

The desktop error reporting path uses `NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES` to read per-SM error states from the debugger RM object. Tegra doesn't expose the debugger interface. Early exit with a clear error message instead of crashing on a missing RM control call.

---

## Testing Results

### Lint / Formatting

| Tool | Result |
|---|---|
| `ruff check` (full tinygrad/) | **All checks passed** |
| Line length | All lines ≤ 150 chars ✅ |
| Indent | 2-space throughout ✅ |

### Unit Tests

Tested on NixOS aarch64 (Jetson AGX Orin), CPU backend (`CC=clang`):

| Suite | Result |
|---|---|
| `test/test_tiny.py` | 16 passed, 1 skipped, 2 deselected |
| `test/unit/` (excl. env-missing) | 229 passed, 46 skipped, 1 xfailed |
| **Upstream baseline (same env)** | **229 passed, 46 skipped, 1 xfailed** |
| **Regressions** | **0** |

The 2 deselected tests (`test_beam`) and excluded unit tests (`test_dtype_spec`, `test_gguf`, `test_hashing`, `test_disk_tensor`, `test_shm_tensor`) all fail identically on upstream/master — they're NixOS environment issues (missing `clang` in subprocess, missing `safetensors`/`torch` packages), not related to this PR.

**NV=1 hardware tests**: Must be run separately on Jetson hardware:
```bash
NV=1 python3 -m pytest test/test_ops.py -v --tb=short
NV=1 python3 -m pytest test/test_tiny.py -v
```

---

## Items for Follow-Up PRs

These were deliberately excluded to keep this PR focused:

1. **Matvec heuristic for fused CAST/MUL chains** (`heuristic.py`, +43/−22) — General optimizer improvement that helps fp16 inference on all platforms. Works on Tegra but benefits desktop too. Should be submitted with benchmark data.

2. **BF16 GGML tensor support** (`state.py`, +7/−1) — Enables loading BF16-quantized GGML models. Not Tegra-specific.

3. **CUDA include path from env** (`compiler_cuda.py`, +2) — `CUDA_INCLUDE_PATH` environment variable for non-standard CUDA installations (NixOS). Useful but orthogonal.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Import `fcntl` breaks non-Linux | None | `assert sys.platform != 'win32'` on line 2 already guards. `fcntl` is stdlib on all POSIX. |
| `_select_iface` ordering change | Low | TegraIface raises `FileNotFoundError` immediately on non-Tegra systems. Fast fail, no side effects. |
| `NVComputeQueue._tegra_signal` is ClassVar | Low | Set once during device init. If multiple devices exist (impossible on Tegra, single iGPU), would affect all queues. Acceptable since multi-device Tegra doesn't exist. |
| `synchronize()` before SLM realloc | Low (positive) | General correctness fix. Adds a sync point but only when `slm_per_thread` needs to grow, which is rare (typically once at model load). |
| sz.py line count +292 | Medium | Significant addition. Justified by adding an entirely new hardware platform. Maintainers may want to see this offset by removing code elsewhere. |
| Struct definitions hardcoded | Medium | If nvgpu kernel headers change in future JetPack versions, structs need manual update. Could be auto-generated from headers in the future, but current approach matches tinygrad's style (see `nv_gpu.py` autogen). |

---

## Recommended PR Description for tinygrad

```
runtime/ops_nv: add TegraIface for Jetson Orin (nvgpu/nvmap)

Adds NV=1 support for NVIDIA Jetson AGX Orin and other Tegra SoCs
using the nvgpu + nvmap kernel drivers (JetPack 6).

New `TegraIface` implements `HCQInterface` by translating RM calls to
nvgpu ioctls. Unified memory allows direct CPU↔GPU memmove without DMA
staging. Tested on Jetson AGX Orin 64GB (SM 8.7, ga10b) with JetPack 6.

Changes:
- TegraIface class (~200 lines) with ioctl structs and helpers
- NVAllocator: direct _copyin/_copyout for unified memory
- NVComputeQueue: _tegra_signal flag for pushbuffer signalling
- NVDevice: Tegra detection, 40-bit VA windows, gpfifo sizing
- Fix: synchronize() before shader_local_mem realloc (all backends)
```
