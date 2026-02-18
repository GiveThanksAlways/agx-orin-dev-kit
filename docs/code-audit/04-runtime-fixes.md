# Runtime Fixes: QMD Race, Copyout, and Memory Safety

These changes are scattered across `ops_nv.py` but share a theme: making the NV
backend correct on Tegra's hardware timing characteristics. Each one prevented a
class of crash or correctness bug.

## Table of Contents

1. [The QMD Race Condition](#the-qmd-race-condition)
2. [Direct Memmove (Copyout/Copyin)](#direct-memmove-copyoutcopyin)
3. [Shader Local Memory Synchronization](#shader-local-memory-synchronization)
4. [PMA Disabled on Tegra](#pma-disabled-on-tegra)
5. [Tegra Fault Reporting](#tegra-fault-reporting)

---

## The QMD Race Condition

### What's a QMD?

A **QMD (Queue Meta Data)** is a control structure that describes a GPU kernel
launch. It contains:
- Kernel entry point (program address)
- Grid dimensions (how many thread blocks)
- Block dimensions (how many threads per block)
- Shared memory size
- **Signal release**: what value to write where when the kernel finishes

In tinygrad's NV backend, the signal release mechanism is embedded in the QMD:

```
QMD:
  program_address: 0x1234000
  grid_dim_x: 128
  block_dim_x: 256
  release0_address: 0xABCD000  ← signal location
  release0_payload:  42        ← value to write on completion
```

### The Race

On desktop GPUs, kernel launch latency is measured in **microseconds** (PCIe
round-trip). By the time the CPU submits the next kernel, the GPU has almost
certainly read the previous QMD.

On Tegra, the doorbell is an **MMIO write** (nanoseconds), not a PCIe transaction.
The CPU can submit the next iteration *before* the GPU reads the current QMD:

```
Time →
CPU:  [Submit kernel A] [Overwrite QMD with kernel B] [Submit kernel B] ...
GPU:  ...................[Read QMD]..................................↑ ← reads B's data, not A's!
                         ^^^^^^^^
                         GPU reads QMD *after* CPU overwrote it
```

If kernel A's QMD had `release0_payload = 42` but kernel B overwrote it with
`release0_payload = 43`, the GPU writes 43 when kernel A finishes — the CPU
sees the wrong signal value and assumes kernel B finished (but it hasn't started).

### The Fix: _tegra_signal

```python
class NVComputeQueue(NVCommandQueue):
    _tegra_signal: ClassVar[bool] = False

    def signal(self, signal: HCQSignal, value: sint = 0):
        # On Tegra, force pushbuffer-based signal release
        if self._tegra_signal: self.active_qmd = None
        # ... rest of signal method
```

Setting `self.active_qmd = None` forces tinygrad to emit the signal as a
**pushbuffer command** instead of embedding it in the QMD:

- **QMD-based** (default): Signal value stored in shared QMD memory → race
- **Pushbuffer-based** (Tegra): Signal command appended to the pushbuffer →
  each submit gets a unique cmdq_page offset (bump-allocated), so values are
  immutable

```
Pushbuffer for kernel A (offset 0x0000):
  COMPUTE_DISPATCH(program=0x1234, grid=128, ...)
  MEM_OP_B(address=0xABCD, data=42)  ← signal baked into pushbuffer

Pushbuffer for kernel B (offset 0x2000):
  COMPUTE_DISPATCH(program=0x5678, grid=64, ...)
  MEM_OP_B(address=0xABCD, data=43)  ← different offset, can't overwrite A
```

The flag is set during device initialization:

```python
# In NVDevice.__init__:
if self.is_tegra(): NVComputeQueue._tegra_signal = True
```

And is a `ClassVar` (class-level, not instance-level) because all queues share
the same hardware timing characteristics.

---

## Direct Memmove (Copyout/Copyin)

### The Problem

Tinygrad's default `HCQAllocator._copyout()` uses a **DMA staging path**:

1. Allocate a 2MB write-combine staging buffer
2. Submit a DMA copy from GPU buffer to staging buffer (chunk by chunk)
3. Wait for DMA completion
4. CPU reads from staging buffer into destination

On desktop, this makes sense — GPU memory (VRAM) is on the other side of PCIe,
and DMA is faster than CPU reads across the bus.

On Tegra, GPU buffers are in **unified system RAM**, mapped with
`INNER_CACHEABLE`. Reading them directly from the CPU is cache-speed fast. The
DMA staging path is strictly worse:

```
Default (DMA staging):          Direct (Tegra):
GPU buf → DMA → staging → CPU  GPU buf → CPU
(2× data movement)             (1× data movement)
(per-chunk submission overhead) (one memmove call)
(uncached staging reads)        (cached reads)
```

### The Fix

```python
class NVAllocator(HCQAllocator['NVDevice']):
    def _copyout(self, dest: memoryview, src: HCQBuffer):
        if self.dev.is_tegra():
            self.dev.synchronize()  # ensure GPU writes are complete
            ctypes.memmove(mv_address(dest), src.va_addr, len(dest))
            return
        super()._copyout(dest, src)

    def _copyin(self, dest: HCQBuffer, src: memoryview):
        if self.dev.is_tegra():
            self.dev.synchronize()  # ensure no in-flight GPU reads
            ctypes.memmove(dest.va_addr, mv_address(src), len(src))
            return
        super()._copyin(dest, src)
```

**Why `synchronize()` first?** On Tegra, the SMMU provides IO-coherency — after
the GPU writes, the CPU can read the latest data without cache flushes. But we
still need to wait for the GPU to finish its current work:

- **copyout**: GPU might still be writing to `src`. Wait until all kernels
  complete, then read.
- **copyin**: GPU might be reading from `dest` for an in-flight kernel. Wait
  until done, then overwrite.

`synchronize()` waits for all queued GPU work to complete by polling the
timeline signal.

**Why `va_addr` works as a CPU pointer**: Because of the `MAP_FIXED` trick in
`alloc()` — we mmap the dmabuf at the same virtual address as the GPU VA. So
`src.va_addr` is simultaneously a valid GPU address and a valid CPU pointer.

### The `uncached` Flag

The alloc method also gained an `uncached` parameter:

```python
def _alloc(self, size, options):
    uncached = options.uncached
    return self.dev.iface.alloc(size, ..., uncached=uncached)
```

This passes through to the cache flag selection in `TegraIface.alloc()`:

```python
cache_flags = _NVMAP_HANDLE_WRITE_COMBINE if (uncached or host) \
              else _NVMAP_HANDLE_INNER_CACHEABLE
```

Write-combine buffers are used for GPU control structures (GPFIFO, USERD, QMD)
where the CPU writes but the GPU reads. Inner-cacheable buffers are used for
data (model weights, activations) where both CPU and GPU read/write.

---

## Shader Local Memory Synchronization

### The Problem

When a GPU kernel needs more registers than the hardware provides, the compiler
"spills" variables to **local memory** (per-thread scratch space in DRAM). Tinygrad
allocates this as a regular buffer (`shader_local_mem`).

If a new kernel needs more local memory than currently allocated, tinygrad
reallocates a larger buffer. But there may be **in-flight kernels** still using
the old buffer:

```
Time →
CPU:  [launch kernel A (needs 32B/thread)]
      [kernel B needs 64B/thread → realloc shader_local_mem → free old buffer]
GPU:  .........[kernel A running, using old shader_local_mem]...→ CRASH!
```

### The Fix

```python
def _ensure_has_local_memory(self, required):
    if self.slm_per_thread >= required: return

    # Must synchronize before freeing old buffer — GPU may still be using it
    self.synchronize()

    self.slm_per_thread = round_up(required, 32)
    # ... realloc ...
```

One line: `self.synchronize()`. Wait for all GPU work to complete before freeing
the old buffer. Simple, correct, and the performance impact is negligible (this
reallocation happens at most a handful of times during model warmup).

---

## PMA Disabled on Tegra

```python
self.pma_enabled = PMA.value > 0 and PROFILE >= 1 and not self.is_tegra()
```

**PMA (Performance Monitoring Aggregator)** is NVIDIA's hardware performance
counter system for profiling. It works through RM control calls to the debugger
object (`GT200_DEBUGGER`). On Tegra, the nvgpu driver doesn't expose the
`GT200_DEBUGGER` RM class (it's desktop-only), so PMA would crash. We simply
disable it.

(Tegra has its own profiling via Nsight Systems and nvprof, which use different
kernel interfaces.)

---

## Tegra Fault Reporting

When the GPU hangs (infinite loop, bad memory access), tinygrad tries to read
detailed fault info from the debugger object. On Tegra, this isn't available:

```python
if self.is_tegra():
    raise RuntimeError(
        "GPU hang detected on Tegra device "
        "(no detailed fault info available via nvgpu)"
    )
```

This replaces a crash (trying to call RM debug controls on a stubbed debugger)
with a clean error message. GPU hangs on Tegra are debugged using:
- `dmesg` (kernel logs from nvgpu.ko)
- `/sys/kernel/debug/gpu.0/` (debugfs entries)
- `nvgpu_error.h` error codes in dmesg
