# Agent Prompts for Sequential Phase Execution

These prompts are designed for three agents run sequentially (2 → 3 → 4), each building on the previous phase's output.

**Context files to include for ALL agents:** `nv-attempt.md`, `Learning-Phase1.md`, `phase1.md`, `test_nvgpu.py`

---

## Prompt for Agent 2 — Phase 2: Memory Management

**Additional context file:** `phase2.md`

```
You are continuing a multi-phase project to build a tinygrad NV backend ("TegraIface") for the NVIDIA Jetson Orin AGX 64GB. Phase 1 is COMPLETE — all nvgpu/nvmap ioctls have been reverse-engineered and 7/7 tests pass in test_nvgpu.py. Your job is Phase 2: Memory Management.

## Your Goal

Prove that CPU<->GPU shared memory works end-to-end on the Jetson Orin using nvmap + nvgpu ioctls. By the end of this phase, you must have:

1. A working `mmap()` of nvmap buffers into CPU address space (read + write from Python)
2. GPU VA mapping of the same buffers via MAP_BUFFER_EX (already proven in Phase 1 — extend it)
3. A reusable `TegraAllocator` helper class in test_nvgpu.py that handles: create → alloc → get_fd → mmap → gpu_map → cleanup
4. Verification that CPU writes are visible to GPU (and vice versa) — the Orin has IO_COHERENCE so this should work without explicit cache flushes, but VERIFY it
5. Tests with multiple buffer sizes: 4KB, 64KB, 1MB, 16MB, 64MB

## Critical Technical Details

- **Device paths:** `/dev/nvmap` (fd for mmap AND ioctls), `/dev/nvgpu/igpu0/ctrl`
- **Heap:** Use IOVMM (1<<30). SYSMEM (1<<31) does NOT work. This was proven in Phase 1.
- **Alloc flags:** NVMAP_HANDLE_WRITE_COMBINE=1 works. Try NVMAP_HANDLE_CACHEABLE=5 too and document the difference.
- **mmap details:** mmap the dmabuf_fd (from NVMAP_IOC_GET_FD), NOT the nvmap device fd directly. The dmabuf fd is what you mmap. Use offset=0, PROT_READ|PROT_WRITE, MAP_SHARED.
- **GPU mapping:** Use MAP_BUFFER_EX on the AS fd (already working in test_nvgpu.py test 6). compr_kind=-1, incompr_kind=0 (pitch linear).
- **Unified memory:** Orin has VRAM=0, meaning CPU and GPU share the same physical DRAM. The IO_COHERENCE flag IS set. This means CPU writes should be immediately visible to GPU without any flush — but test this empirically.
- **Coherence test strategy:** Since you don't have command submission yet (that's Phase 3), the simplest coherence test is: (a) mmap a buffer, (b) write a known pattern from CPU, (c) map it to GPU VA, (d) mmap it AGAIN to a different CPU pointer and read back — verifying the data survives round-tripping through the nvmap/dmabuf path. Alternatively, you can use the NVMAP_IOC_WRITE/NVMAP_IOC_READ ioctls (nr=5 and nr=4 in nvmap.h) if they exist.

## Dev Environment

- You are on NixOS. Enter the dev shell with: `cd /home/agent/jetpack-nixos/examples/tinygrad && nix develop --command bash`
- Inside the shell, run tests with: `python3 test_nvgpu.py`
- The existing test_nvgpu.py has all the ioctl wrappers and structs from Phase 1. EXTEND this file — don't create a new one.
- You can also use the `detective` shell for strace: `nix develop .#detective --command bash` (but this shell has NO Python)
- The nvmap header is at: `l4t-sources/nvgpu/include/uapi/linux/nvmap.h` — read it for mmap and ioctl details
- The nvgpu headers are at: `l4t-sources/nvgpu/include/uapi/linux/nvgpu*.h`

## Iteration Loop

1. Read the existing test_nvgpu.py to understand the current state
2. Read the nvmap.h header to find mmap-related ioctls and semantics
3. Add a `test_mmap_buffer()` function that creates a buffer, mmaps it, writes a pattern, reads it back
4. Add a `test_gpu_coherence()` function that verifies CPU<->GPU memory visibility
5. Build the `TegraAllocator` class
6. Run tests, debug failures, iterate
7. When all tests pass, update phase2.md with your findings, discoveries, and any gotchas

## What Success Looks Like

- test_nvgpu.py has new tests that all pass (keep the existing 7 tests too!)
- CPU can write bytes to a buffer and read them back via mmap
- The same buffer is simultaneously mapped into GPU VA space
- TegraAllocator class exists and is clean/reusable
- phase2.md is updated with complete findings
- You should end with something like "9/9 tests pass" (7 existing + at least 2 new)

## Progress Tracking

Keep the existing `Results: X/Y tests passed` summary at the end of main(). Add your new tests incrementally so every run prints a running score (e.g. "8/9 tests passed" → "9/9 tests passed"). This makes progress visible at a glance.

## Key Pitfall Warnings (learned from Phase 1)

- Struct sizes MUST match the kernel exactly. Use ctypes.sizeof() to verify.
- Don't use _pack_=1 unless the kernel struct also uses __packed__. Most nvgpu structs use natural alignment.
- If an ioctl returns EINVAL, read the kernel source (l4t-sources/) to find the exact validation check.
- The nvmap handle vs dmabuf fd distinction matters: handle is for nvmap ioctls, dmabuf fd is for mmap and for passing to nvgpu (MAP_BUFFER_EX).
- Always close fds and clean up handles to avoid resource leaks that cause later tests to fail.

## Final Deliverable: Learning Document

When you are done with Phase 2, write a `Learning-Phase2.md` that teaches the reader how things work — the linear history/iteration of how you discovered things, what broke, what surprised you, and the key concepts. Follow the same spirit as `Learning-Phase1.md`: chronological, honest about mistakes, with diagrams and code snippets that explain the "why" not just the "what."

Good Luck and God Speed
```

---

## Prompt for Agent 3 — Phase 3: Command Submission

**Additional context files:** `phase3.md`, and also include `phase2.md` (Agent 3 depends on Phase 2's allocator)

```
You are continuing a multi-phase project to build a tinygrad NV backend ("TegraIface") for the NVIDIA Jetson Orin AGX 64GB. Phase 1 (ioctl reverse-engineering) is COMPLETE with 7/7 tests passing. Phase 2 (memory management) should already be complete — its TegraAllocator class and mmap tests are in test_nvgpu.py. Your job is Phase 3: Command Submission.

## Your Goal

Push actual GPU commands through the GPFIFO and execute a compute shader on the Jetson Orin's ga10b GPU (Ampere, SM 8.7, compute class 0xc7c0). By the end of this phase, you must have:

1. mmap of the userd buffer (4KB) — this is the usermode doorbell for GPFIFO submission
2. mmap of the GPFIFO buffer (8KB = 1024 entries x 8 bytes)
3. A working GPFIFO push mechanism: write GPFIFO entries → update GP_PUT → ring doorbell
4. A trivial compute shader compiled via nvrtc (PTX → SASS), loaded into GPU memory, dispatched via QMD, and its output verified
5. Syncpoint-based completion waiting

## Critical Technical Details

### Usermode Submit (how CUDA does it — no SUBMIT_GPFIFO ioctl!)
CUDA on Jetson uses "usermode submit" — it does NOT call the SUBMIT_GPFIFO ioctl. Instead:
1. Write GPFIFO entries (8 bytes each) into the GPFIFO buffer (already allocated in SETUP_BIND)
2. Update GP_PUT register in the userd region (offset from `AmpereAControlGPFifo.GPPut`)
3. Write the work_submit_token to the doorbell register (at userd + `AmpereAControlGPFifo.GPPut` offset or via host1x)

The work_submit_token from Phase 1 is 511. The syncpoint ID is 17 with GPU VA 0xffffe10000.

### GPFIFO Entry Format (8 bytes)
```c
// Each GPFIFO entry is a 64-bit value:
// bits[2:0]   = ENTRY_OPCODE (1 = normal GPU VA, 0 = conditional/nop)
// bits[4:3]   = reserved
// bits[40:5]  = GPU_VA >> 2  (GPU virtual address of push buffer, 4-byte aligned)
// bits[41]    = PRIV (privileged)
// bits[42]    = LEVEL (0 = main, 1 = subroutine)
// bits[52:43] = LENGTH (number of 4-byte words in push buffer)
// bits[63:53] = reserved/sync
```
Study tinygrad's `NVComputeQueue` and `NVCopyQueue` in ops_nv.py — they format these entries.

### Push Buffer / Method Format
The push buffer contains GPU "methods" — 4-byte header + data pairs:
- Header: `(count << 28) | (subchannel << 13) | (method_offset >> 2)`
- For compute methods on subchannel 0, this addresses registers in the compute class (0xc7c0)

### QMD (Queue Meta Data)
The QMD is a ~256-byte struct that describes a compute dispatch. It contains:
- Shader program address (GPU VA of the compiled SASS binary)
- Grid dimensions (blocks_x, blocks_y, blocks_z)  
- Block dimensions (threads_x, threads_y, threads_z)
- Shared memory size, local memory, constant buffer bindings
- The QMD format for Ampere (GA10B = Ampere mobile) should be IDENTICAL to desktop Ampere

**Study tinygrad's QMD construction:** `tinygrad/tinygrad/runtime/ops_nv.py` has QMD building in the `NVComputeQueue` class. Also look at `tinygrad/tinygrad/runtime/autogen/nv_gpu.py` for the QMD field definitions (search for `qmd` or `ComputeQmd`).

### Shader Compilation
Use `libcuda.so` + `libnvrtc.so` for PTX compilation:
```python
import ctypes
nvrtc = ctypes.CDLL("libnvrtc.so")
# nvrtcCreateProgram, nvrtcCompileProgram, nvrtcGetCUBIN/nvrtcGetCUBINSize
```
Target SM 8.7 (`--gpu-architecture=sm_87`). Start with the simplest possible kernel:
```c
extern "C" __global__ void test_kernel(float *out) {
    out[threadIdx.x] = threadIdx.x * 1.0f;
}
```

### Memory Layout
You'll need these buffers, all allocated via nvmap (using Phase 2's allocator or by extending test_nvgpu.py's existing alloc code):
- **Push buffer:** ~4KB, holds GPU methods + QMD launch command
- **Shader binary:** size from nvrtc, holds compiled SASS
- **Output buffer:** e.g. 256 floats = 1KB, where the kernel writes results
- **QMD buffer:** ~256 bytes (but align to 256 bytes)
All must be mmap'd to CPU (for writing methods/reading results) AND mapped to GPU VA (for GPU execution).

### Syncpoint Wait
After ringing the doorbell, poll the syncpoint to detect completion:
- Read the hwsyncpt value from `/sys/devices/platform/host1x/[syncpt_id]/min` 
- Or mmap the syncpoint GPU VA (0xffffe10000) and poll in a loop
- Or use the `host1x-fence` device

### The Full Submission Sequence
1. Allocate push buffer, shader, output, QMD buffers (nvmap + mmap + MAP_BUFFER_EX)
2. Compile PTX shader → SASS binary → copy to shader buffer
3. Format QMD with shader address, grid/block dims, constant buffer binding to output
4. Write push buffer: method to launch QMD (SEND_PCAS + SEND_SIGNALING_PCAS on compute class)
5. Write GPFIFO entry pointing to push buffer
6. Update GP_PUT in userd
7. Write work_submit_token to doorbell
8. Wait for syncpoint increment
9. Read output buffer from CPU, verify correctness

## Dev Environment

- NixOS dev shell: `cd /home/agent/jetpack-nixos/examples/tinygrad && nix develop --command bash`
- Run tests: `python3 test_nvgpu.py`
- EXTEND test_nvgpu.py — don't create a separate file
- tinygrad source is at: `tinygrad/tinygrad/runtime/ops_nv.py` — study NVComputeQueue, NVCopyQueue, and the QMD setup
- tinygrad autogen GPU structs: `tinygrad/tinygrad/runtime/autogen/nv_gpu.py` — has QMD field offsets
- Kernel headers: `l4t-sources/nvgpu/include/uapi/linux/nvgpu*.h`
- Kernel source: `l4t-sources/nvgpu/common/fifo/channel.c` — has SETUP_BIND and submit logic
- nvrtc is available in the nix shell (comes with CUDA)

## Iteration Loop

1. Read tinygrad's NVComputeQueue and QMD code to understand the push buffer + QMD format
2. Read the autogen nv_gpu.py for QMD struct field definitions
3. Start simple: just try to mmap the userd buffer and write to it
4. Then try a minimal GPFIFO push (even a NOP) and verify the GPU processes it (syncpoint increments)
5. Then compile a shader, build a QMD, and do a real compute dispatch
6. Verify output, iterate on failures
7. Update phase3.md with findings

## What Success Looks Like

- A compute shader runs on the GPU and produces correct output, verified from CPU
- The entire flow (alloc → compile → QMD → push → doorbell → wait → verify) works in Python
- Syncpoint-based completion detection works
- phase3.md is updated with the exact push buffer format, QMD fields used, doorbell mechanism, and any discoveries

## Progress Tracking

Keep the existing `Results: X/Y tests passed` summary at the end of main(). Add your tests incrementally — for example: test 8 = mmap userd, test 9 = GPFIFO NOP push, test 10 = shader compile, test 11 = full compute dispatch + verify. Every run should print a running score like "9/11 tests passed" so progress is visible at a glance.

## Key Pitfall Warnings

- The userd and GPFIFO buffers were ALREADY allocated in Phase 1's `test_full_channel_setup()`. You need their dmabuf_fds to mmap them. Either save them from Phase 1's setup or re-extract them.
- GP_PUT offset in userd: look at tinygrad's `AmpereAControlGPFifo` struct for the exact offset. It's likely at offset 0x68 or 0x90 — check the autogen.
- The doorbell token (work_submit_token=511) must be written to the correct location. On Jetson this may be via host1x or an MMIO register. Study the strace output or kernel source.
- Method offsets are in units of 4 bytes (not byte offsets). So method 0x2000 means register offset 0x8000.
- QMD must be 256-byte aligned in GPU memory.
- Shader must be loaded at a GPU VA. The QMD points to this VA.
- If the GPU hangs or the channel gets stuck, you may need to tear down and recreate the channel. The watchdog is disabled (from Phase 1), so hangs won't auto-recover.
- If you get stuck on the doorbell mechanism, strace a simple CUDA program to see exactly what memory location it writes and what value: `strace -e trace=write,writev,ioctl -f python3 -c "import cuda; ..."`

## Final Deliverable: Learning Document

When you are done with Phase 3, write a `Learning-Phase3.md` that teaches the reader how things work — the linear history/iteration of how you discovered things, what broke, what surprised you, and the key concepts. Follow the same spirit as `Learning-Phase1.md`: chronological, honest about mistakes, with diagrams and code snippets that explain the "why" not just the "what."

Good Luck and God Speed

---

## Prompt for Agent 4 — Phase 4: TegraIface Integration

**Additional context files:** `phase4.md`, tinygrad's `ops_nv.py` (critical!), and `tinygrad/tinygrad/runtime/support/hcq.py`

```
You are continuing a multi-phase project to build a tinygrad NV backend ("TegraIface") for the NVIDIA Jetson Orin AGX 64GB. Phase 1 (ioctl reverse-engineering), Phase 2 (memory management), and Phase 3 (command submission) are COMPLETE. Their results are in test_nvgpu.py, phase2.md, and phase3.md. Your job is Phase 4: TegraIface Integration — building the actual tinygrad backend class.

## Your Goal

Build a `TegraIface` class that plugs into tinygrad's NV runtime, enabling `NV=1` on Jetson Orin AGX. By the end of this phase:

1. `TegraIface` class in `tinygrad/tinygrad/runtime/ops_nv.py` that implements the same interface as `NVKIface` and `PCIIface`
2. `NVDevice` modified to try `TegraIface` when on Jetson (detected via `/dev/nvgpu/igpu0/ctrl`)
3. Basic operations working: `NV=1 python3 -c "from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())"`
4. Full test: `NV=1 python3 -c "from tinygrad import Tensor; print((Tensor([1,2,3]) + Tensor([4,5,6])).numpy())"` → `[5 7 9]`

## Architecture — What TegraIface Replaces

tinygrad's NV backend has a clear separation:
- **Interface layer** (NVKIface/PCIIface): Handles device-specific driver communication (memory alloc, channel creation, command submission)
- **GPU programming layer** (NVComputeQueue/NVCopyQueue): Formats push buffers, QMDs, methods — this is architecture-specific (Ampere/Hopper/Blackwell) but NOT driver-specific

TegraIface ONLY replaces the interface layer. The GPU programming layer stays the same because ga10b IS Ampere.

## The Interface Contract

Study `NVKIface` and `PCIIface` carefully. Here's what TegraIface must implement:

### Required Methods

```python
class TegraIface:
    def __init__(self, dev, device_id):
        """Initialize: open /dev/nvmap + /dev/nvgpu/igpu0/ctrl, detect GPU.
        Set self.compute_class, self.gpfifo_class, self.dma_class, self.viddec_class.
        These come from GET_CHARACTERISTICS (already decoded in test_nvgpu.py)."""
        
    def rm_alloc(self, parent, clss, params=None, root=None) -> int:
        """On NVKIface this allocates RM objects. On TegraIface, we need to translate
        RM class numbers to nvgpu equivalents. Key mappings:
        - NV01_DEVICE_0 → just store metadata (no nvgpu equivalent, Tegra has no device objects)
        - NV20_SUBDEVICE_0 → same (no equivalent)
        - NV01_MEMORY_VIRTUAL → create the AS fd (ALLOC_AS)
        - FERMI_VASPACE_A → we already have AS from ALLOC_AS
        - KEPLER_CHANNEL_GROUP_A → OPEN_TSG
        - FERMI_CONTEXT_SHARE_A → CREATE_SUBCONTEXT
        - AMPERE_CHANNEL_GPFIFO_A → OPEN_CHANNEL + AS_BIND + TSG_BIND + SETUP_BIND
        - AMPERE_COMPUTE_B (0xc7c0) → ALLOC_OBJ_CTX on channel
        - AMPERE_DMA_COPY_B → ALLOC_OBJ_CTX on channel
        - GT200_DEBUGGER → skip (not needed on Tegra)
        Return a handle integer that NVDevice can use for rm_control calls."""
        
    def rm_control(self, obj, cmd, params=None):
        """Translate RM control calls to nvgpu equivalents.
        Key ones:
        - NV2080_CTRL_CMD_GPU_GET_GID_INFO → return a synthetic UUID
        - NV2080_CTRL_CMD_PERF_BOOST → use devfreq sysfs to set GPU freq
        - NV2080_CTRL_CMD_GR_GET_INFO → translate to GET_CHARACTERISTICS fields
        - NV0080_CTRL_CMD_GPU_GET_CLASSLIST → return our known class list
        - NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN → return saved token from SETUP_BIND
        - NVA06C_CTRL_CMD_GPFIFO_SCHEDULE → NOP (nvgpu channels auto-schedule)
        Return the params object."""
        
    def setup_usermode(self):
        """Return (usermode_handle, mmio_interface_for_doorbell).
        On Tegra, the userd buffer IS the usermode region. mmap it and return an MMIOInterface."""
        
    def setup_vm(self, vaspace):
        """On NVKIface this registers with UVM. On Tegra, we already have the AS.
        This can be a NOP since we set up the AS in __init__."""
        
    def setup_gpfifo_vm(self, gpfifo):
        """On NVKIface this registers a channel with UVM. On Tegra, NOP (channel is already bound to AS)."""
        
    def alloc(self, size, host=False, uncached=False, cpu_access=False, contiguous=False, **kwargs) -> HCQBuffer:
        """Allocate GPU memory via nvmap. Steps:
        1. nvmap CREATE → ALLOC (IOVMM heap) → GET_FD
        2. MAP_BUFFER_EX on AS fd to get GPU VA
        3. If cpu_access or host: mmap the dmabuf_fd
        4. Return HCQBuffer with va_addr=GPU_VA, view=MMIOInterface if mmap'd"""
        
    def free(self, mem: HCQBuffer):
        """Unmap from GPU VA (UNMAP_BUFFER on AS fd), munmap CPU, close dmabuf fd."""
        
    def map(self, mem: HCQBuffer):
        """Cross-device mapping (for multi-GPU). Tegra is single-GPU, so this is simple."""
```

### Key Differences from NVKIface

| Aspect | NVKIface | TegraIface |
|--------|----------|------------|
| Device open | /dev/nvidiactl + /dev/nvidia-uvm | /dev/nvmap + /dev/nvgpu/igpu0/ctrl |
| RM objects | Full RM hierarchy (root→device→subdevice) | No RM hierarchy; use nvgpu ioctls directly |
| Memory | UVM (nvidia-uvm) | nvmap (CREATE→ALLOC→GET_FD→mmap) |
| VA management | UVM manages VA | nvgpu AS manages VA (ALLOC_AS + MAP_BUFFER_EX) |
| Channel | RM GPFIFO alloc | nvgpu TSG + OPEN_CHANNEL + SETUP_BIND |
| Submission | UVM doorbell (same as desktop) | userd doorbell (write to mmap'd userd buffer) |
| Compute | Same Ampere QMD/methods | Same Ampere QMD/methods (ga10b = Ampere) |

### NVDevice Modifications

In `NVDevice.__init__()`, change the iface selection:

```python
# Current:
self.iface = self._select_iface(NVKIface, PCIIface)

# Change to:
self.iface = self._select_iface(TegraIface, NVKIface, PCIIface)
```

TegraIface should be tried FIRST. Its `__init__` should raise an exception if `/dev/nvgpu/igpu0/ctrl` doesn't exist, causing `_select_iface` to fall through to NVKIface/PCIIface.

### The rm_alloc Translation Challenge

The hardest part is translating NVDevice's RM calls (which assume NVKIface) to nvgpu calls. NVDevice does things like:
```python
self.nvdevice = self.iface.rm_alloc(self.iface.root, nv_gpu.NV01_DEVICE_0, device_params)
self.subdevice = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV20_SUBDEVICE_0, ...)
self.virtmem = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV01_MEMORY_VIRTUAL, ...)
```

For TegraIface, these RM classes don't exist. You need to:
1. Return fake handles for NV01_DEVICE_0, NV20_SUBDEVICE_0 (they're just hierarchy containers)
2. For NV01_MEMORY_VIRTUAL, actually call ALLOC_AS and return the AS fd as the handle
3. For KEPLER_CHANNEL_GROUP_A, call OPEN_TSG
4. For AMPERE_CHANNEL_GPFIFO_A, do the full channel setup pipeline from Phase 1
5. For compute/DMA classes, call ALLOC_OBJ_CTX

Use a handle counter + dictionary to map fake handles → nvgpu state (fds, etc).

## Dev Environment

- NixOS: `cd /home/agent/jetpack-nixos/examples/tinygrad && nix develop --command bash`
- tinygrad source: `tinygrad/tinygrad/runtime/ops_nv.py` — THIS IS THE FILE YOU EDIT
- Also read: `tinygrad/tinygrad/runtime/support/hcq.py` — has HCQCompiled, _select_iface, HCQBuffer
- Also read: `tinygrad/tinygrad/runtime/autogen/nv_gpu.py` — has RM class numbers, QMD structs
- Phase 1 test: `test_nvgpu.py` — has ALL the nvgpu/nvmap ctypes structs and ioctl wrappers you need. Copy them into ops_nv.py or import from a shared module.
- Headers: `l4t-sources/nvgpu/include/uapi/linux/nvgpu*.h`, `l4t-sources/nvgpu/include/uapi/linux/nvmap.h`

## Iteration Loop

1. **FIRST: Deep-read NVKIface and NVDevice.__init__()** — understand every rm_alloc, rm_control, and setup call. Map each one to its nvgpu equivalent.
2. **Stub out TegraIface** with all methods raising NotImplementedError
3. **Implement detection** — `__init__` tries opening `/dev/nvgpu/igpu0/ctrl`, raises if not found
4. **Implement memory** — `alloc()` and `free()` using nvmap
5. **Implement channel setup** — translate the RM channel creation to nvgpu TSG+channel pipeline
6. **Implement rm_alloc/rm_control translation** — the big mapping layer
7. **Test incrementally**: first just `from tinygrad import Tensor`, then `Tensor([1])`, then operations
8. **Update phase4.md** with findings

## What Success Looks Like

- `NV=1 python3 -c "from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())"` → `[1 2 3]`
- `NV=1 python3 -c "from tinygrad import Tensor; print((Tensor.randn(4,4) @ Tensor.randn(4,4)).numpy())"` → correct matrix multiply
- No crashes, no hangs, clean error handling
- TegraIface is a clean, maintainable class in ops_nv.py
- phase4.md updated with architecture decisions and test results

## Progress Tracking

Build incrementally and test at each step. Good milestones to print/confirm:
1. `TegraIface.__init__` succeeds (device detected)
2. `rm_alloc(NV01_DEVICE_0)` returns without error
3. `alloc()` returns a valid HCQBuffer
4. Channel setup completes
5. `Tensor([1,2,3])` creates without crash
6. `.numpy()` returns correct values
7. Tensor add works
8. Matmul works
Log each milestone so progress is visible.

## Key Pitfall Warnings

- **Don't modify the GPU programming layer** (NVComputeQueue, NVCopyQueue, QMD format). ga10b uses IDENTICAL Ampere class methods as desktop. Only the DRIVER layer differs.
- **NVDevice makes many rm_alloc/rm_control calls** — you MUST handle all of them, even if some become NOPs on Tegra. If rm_alloc returns a handle that NVDevice later passes to rm_control, your handle mapping must be consistent.
- **HCQBuffer contract:** alloc() must return HCQBuffer with correct va_addr (GPU VA), size, view (MMIOInterface for CPU access), and meta (for free/map to use).
- **VA allocator:** NVKIface uses BumpAllocator for VA addresses. TegraIface needs one too, but the range is different: 0x200000 to 0xFFFFE00000 (from ALLOC_AS in Phase 1). Or let MAP_BUFFER_EX auto-assign with offset=0.
- **The gpfifo_area allocation** in NVDevice is 0x300000 bytes with `force_devmem=True` and WC mapping. TegraIface.alloc() must handle the `force_devmem` kwarg (just allocate normally since Tegra has no separate VRAM).
- **sm_version:** NVKIface gets this via RM control calls (NV2080_CTRL_CMD_GR_GET_INFO). TegraIface must return it from GET_CHARACTERISTICS: `sm_arch_sm_version=0x807` → this is already in the right format for tinygrad.
- **The `_query_gpu_info` method** in NVDevice asks for `num_gpcs`, `num_tpc_per_gpc`, `num_sm_per_tpc`, `max_warps_per_sm`, `sm_version` via rm_control. TegraIface must return these from GET_CHARACTERISTICS or other nvgpu ioctls.

## Final Deliverable: Learning Document

When you are done with Phase 4, write a `Learning-Phase4.md` that teaches the reader how things work — the linear history/iteration of how you discovered things, what broke, what surprised you, and the key concepts. Follow the same spirit as `Learning-Phase1.md`: chronological, honest about mistakes, with diagrams and code snippets that explain the "why" not just the "what."

Good Luck and God Speed


<!-- Starting section for robust testing and performance -->

---

## Prompt for Agent 5 — Phase 5: Robust Testing & Performance Benchmarking

**Additional context files:** `robust-testing-and-performance.md` (master tracking doc — read ALL of it), `tests/dmesg_checker.py`, tinygrad's `ops_nv.py` (the backend under test), `tinygrad/tinygrad/runtime/support/hcq.py` (HCQ framework), and optionally `nv-attempt.md` (full build history)

```
You are continuing a multi-phase project to build and harden a tinygrad NV backend ("TegraIface") for the NVIDIA Jetson Orin AGX 64GB. Phases 1-4 are COMPLETE — the TegraIface class is fully functional and 20/20 HCQ tests pass. Your job is Phase 5: Robust Testing & Performance Benchmarking.

## Your Goal

Complete all remaining phases in `robust-testing-and-performance.md` — specifically B2 through B6 (correctness tests) and Phase C (performance benchmarks). By the end, you should have:

1. **B2** — `test_ops.py` results for NV=1 vs CUDA=1 (hundreds of tensor op tests)
2. **B3** — `test_jit.py` results for NV=1 vs CUDA=1
3. **B4** — `tests/test_tegra_edge_cases.py` written and passing (15 edge-case tests)
4. **B5** — `tests/test_tegra_stress.py` written and passing (8 stress tests)
5. **B6** — `tests/test_tegra_models.py` written and passing (4 model tests)
6. **Phase C** — Performance benchmark results (matmul, bandwidth, launch overhead, element-wise, model inference)
7. **`robust-testing-and-performance.md`** updated with all results
8. Any bugs found → fixed in `ops_nv.py` and documented

## What Is Already Done (DO NOT REDO)

- ✅ **B1 (test_hcq.py):** 20/20 pass — results already in the doc
- ✅ **dmesg_checker.py:** Built and working — use it during tests
- ✅ **QMD reuse race:** FIXED (pushbuffer-based signal on Tegra)
- ✅ **nvmap tag warnings:** FIXED (`_NVMAP_TAG_TINYGRAD = 0x0900`)
- ✅ All changes committed on branch `nv-agx-orin-dev-kit`

## Critical Technical Details

### Device
- **Jetson Orin AGX 64GB:** JetPack 6, L4T r36.4.4, Kernel 5.15.148, CUDA 12.6
- **GPU:** ga10b iGPU, Ampere arch, SM 8.7, compute class 0xc7c0
- **Memory:** Unified (VRAM=0) — CPU and GPU share the same DRAM. IO-coherent via AXI fabric.
- **tinygrad:** v0.12.0, commit cc9bf8cc on branch `nv-agx-orin-dev-kit`

### Key Code Locations
- **TegraIface:** `tinygrad/tinygrad/runtime/ops_nv.py` ~L575-1100
- **NVDevice:** `tinygrad/tinygrad/runtime/ops_nv.py` ~L1100+
- **HCQ framework:** `tinygrad/tinygrad/runtime/support/hcq.py`
- **dmesg checker:** `tests/dmesg_checker.py`
- **Test files to create:** `tests/test_tegra_edge_cases.py`, `tests/test_tegra_stress.py`, `tests/test_tegra_models.py`, `tests/benchmark_nv_vs_cuda.py`

### Known Issues You Will Encounter
1. **`test_map_cpu_buffer_to_device` fails** — `TegraAllocator.map()` is a no-op. CPU buffers can't be DMA-copied to GPU. If tests need this, implement `map()` or work around it.
2. **Error cascade in test suite** — When one test fails and sets `device.error_state`, ALL subsequent tests fail in `setUp()`. Workaround: run tests individually with `python3 -m unittest TestClass.test_name` or use `--failfast`.
3. **`invalidate_caches()` is NOP'd** — The `NV2080_CTRL_CMD_FB_FLUSH_GPU_CACHE` RM control is a no-op in TegraIface. This might cause stale data issues under certain access patterns. Test this in B4.
4. **`TegraIface.free()` has a contradictory None check** — The inner `if mem.view is None` is always False when inside `if mem.view is not None`. May be correct by accident. Audit in B4.
5. **`tu104_gr_init_commit_rtv_cb`** — harmless WARNING in dmesg on every GR context init. Known harmless, suppressed by dmesg_checker.

### Test Infrastructure Patterns
- **No pip, no pytest** — Use `python3 -m unittest` (it's in the stdlib). tinygrad's test suite also supports this.
- **Backend selection:** `NV=1` for our Tegra backend, `CUDA=1` for the reference CUDA backend
- **Tolerance:** Use `atol=1e-4` for float32 comparisons between NV=1 and CUDA=1
- **dmesg checking:** Wrap tests with `DmesgChecker` context manager to catch GPU errors:
  ```python
  from dmesg_checker import DmesgChecker
  with DmesgChecker() as dc:
      run_test()
  assert dc.report.is_clean, dc.report.summary()
  ```

## Dev Environment

- **NixOS dev shell:** `cd /home/agent/jetpack-nixos/examples/tinygrad && nix develop --command bash`
- **Run tinygrad tests:** `cd tinygrad && NV=1 python3 -m unittest discover -s test -p "test_ops.py" -v`
- **Run individual test:** `NV=1 python3 -m unittest test.test_ops.TestOps.test_add`
- **Run custom tests:** `NV=1 python3 -m unittest tests.test_tegra_edge_cases -v`
- **Check kernel logs:** `dmesg | tail -30` or `python3 tests/dmesg_checker.py`
- **tinygrad source:** `tinygrad/` subdir (it's a git submodule)
- **Our test files:** `tests/` subdir (at same level as `tinygrad/`)

## Execution Order

### Step 1: B2 — Run test_ops.py
```bash
cd /home/agent/jetpack-nixos/examples/tinygrad/tinygrad

# NV=1 — our backend
NV=1 python3 -m pytest test/test_ops.py -v --tb=short 2>&1 | tee ../tests/results_ops_nv.log

# CUDA=1 — reference
CUDA=1 python3 -m pytest test/test_ops.py -v --tb=short 2>&1 | tee ../tests/results_ops_cuda.log
```
Count passes/failures/errors for both. Record in the doc. If NV=1 has failures that CUDA=1 doesn't, investigate and fix in ops_nv.py.

### Step 2: B3 — Run test_jit.py
```bash
NV=1 python3 -m pytest test/test_jit.py -v --tb=short 2>&1 | tee ../tests/results_jit_nv.log
CUDA=1 python3 -m pytest test/test_jit.py -v --tb=short 2>&1 | tee ../tests/results_jit_nv.log
```

### Step 3: B4 — Write and run test_tegra_edge_cases.py
Create `tests/test_tegra_edge_cases.py` with the 15 tests listed in `robust-testing-and-performance.md`. Each test should:
- Use `unittest.TestCase`
- Clear and check dmesg around GPU operations
- Compare against expected values or CUDA=1 output
- Clean up allocations (no resource leaks)

### Step 4: B5 — Write and run test_tegra_stress.py
Create `tests/test_tegra_stress.py` with the 8 stress tests. These are longer-running (some take 30-60s). Use timeouts.

### Step 5: B6 — Write and run test_tegra_models.py
Create `tests/test_tegra_models.py` with the 4 model tests. Use random weights (seeded for reproducibility). Compare NV=1 vs CUDA=1 outputs.

### Step 6: Phase C — Performance Benchmarks
Only after all B phases are green. Create `tests/benchmark_nv_vs_cuda.py` that:
- Runs each benchmark (matmul, bandwidth, latency, element-wise, model inference)
- Outputs JSON results
- Calculates NV/CUDA % ratios

### Step 7: Update the doc
Fill in ALL the results tables in `robust-testing-and-performance.md`. Update the Results Log section.

## Bug Fixing

When you find NV=1 failures that don't occur on CUDA=1:
1. Isolate the failing test to a minimal reproducer
2. Check dmesg for GPU errors (sked exception, MMU fault, etc.)
3. Look at the TegraIface code path the test exercises
4. Fix in `ops_nv.py`
5. Re-run the test AND all B1 tests (to check for regressions)
6. Document the fix in the "Known Bugs" table and Results Log

## Important Patterns From Prior Work

- **QMD reuse race was the hardest bug** — it manifested as occasional val=198 instead of 200 in `test_exec_2_kernels_100_times`. Root cause: fast MMIO doorbell on Tegra. Fix: `NVComputeQueue._tegra_signal = True` forces pushbuffer-based signal release. NEW bugs may have similar timing-dependent symptoms.
- **Making cmdq_page WC causes sked exceptions** — PBDMA requires cacheable pushbuffer on Tegra. Never change cmdq_page cacheability.
- **INNER_CACHEABLE memory IS coherent** — Tegra's AXI fabric provides IO coherence. Don't add unnecessary cache flushes.
- **Error cascade** — One failed test poisons `device.error_state` and all subsequent tests fail. Run tests individually to avoid this.

## What Success Looks Like

- All B2-B6 results recorded in the doc
- NV=1 pass rate matches or is very close to CUDA=1 pass rate (accounting for known limitations like single GPU)
- Phase C benchmark tables completely filled in with NV/CUDA % ratios
- Any bugs found are either fixed (with regression testing) or documented as known issues
- `robust-testing-and-performance.md` is a complete, publishable report of NV backend quality on Jetson Orin
- All test files committed to the repo

## Commit Strategy

Commit after each major milestone:
1. After B2+B3 results (update doc)
2. After writing B4+B5+B6 test files
3. After all B tests pass (update doc with results)
4. After Phase C benchmarks (update doc with results)

Use descriptive commit messages like: "Add B2/B3 test results: X/Y ops pass, Z/W jit pass"

Good Luck and God Speed
```

