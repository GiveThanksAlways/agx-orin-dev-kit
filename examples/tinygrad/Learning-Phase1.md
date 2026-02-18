# Learning Phase 1: How We Reverse-Engineered the Jetson Orin GPU Interface

**This document is a teaching walkthrough** of how Phase 1 was completed. It follows the actual chronological order of the work, explains the methodology, the mistakes, the debugging, and the key concepts you need to understand to work with the nvgpu/nvmap kernel interface on NVIDIA Jetson.

---

## Table of Contents

1. [Background: Why We're Doing This](#1-background-why-were-doing-this)
2. [The Starting Point: What We Knew](#2-the-starting-point-what-we-knew)
3. [Step 1: Get the Kernel Source](#3-step-1-get-the-kernel-source)
4. [Step 2: Strace CUDA to See What It Does](#4-step-2-strace-cuda-to-see-what-it-does)
5. [Step 3: Decode Every Ioctl](#5-step-3-decode-every-ioctl)
6. [Step 4: Build Python ctypes Structs](#6-step-4-build-python-ctypes-structs)
7. [Step 5: Test Basic Ioctls](#7-step-5-test-basic-ioctls)
8. [Step 6: The Hard Stuff — ALLOC_AS and PDE Alignment](#8-step-6-the-hard-stuff--alloc_as-and-pde-alignment)
9. [Step 7: The Channel Pipeline](#9-step-7-the-channel-pipeline)
10. [Step 8: SETUP_BIND — The Final Boss](#10-step-8-setup_bind--the-final-boss)
11. [Key Concepts You Need to Know](#11-key-concepts-you-need-to-know)
12. [Tools and Methodology](#12-tools-and-methodology)
13. [Common Pitfalls](#13-common-pitfalls)

---

## 1. Background: Why We're Doing This

The Jetson Orin AGX 64GB has a powerful GPU (Ampere architecture, SM 8.7), but it uses a completely different driver stack than desktop NVIDIA GPUs. Desktop GPUs use `nvidia.ko` with the RM (Resource Manager) API. Jetson uses `nvgpu.ko` + `nvmap.ko` with a totally different set of ioctls.

tinygrad's NV backend (`ops_nv.py`) has two interfaces:
- **NVKIface** — talks to `/dev/nvidiactl` via RM ioctls (needs nvidia-uvm, which doesn't exist on Jetson)
- **PCIIface** — directly maps PCI BARs (Jetson GPU is on platform bus, not PCI)

Both fail on Jetson. We need a third interface: **TegraIface** — that talks to nvgpu/nvmap.

Before we can build TegraIface, we need to know exactly what ioctls to call, in what order, with what struct layouts. That's what Phase 1 is about.

---

## 2. The Starting Point: What We Knew

From the initial investigation (documented in `nv-attempt.md`), we knew:
- The GPU is accessed via `/dev/nvgpu/igpu0/ctrl` and `/dev/nvmap`
- Channel/TSG/AS are sub-devices returned as file descriptors from ioctls on the ctrl device
- CUDA works (meaning libcuda.so knows how to talk to nvgpu.ko)
- The kernel source is available in the L4T BSP

What we didn't know:
- The exact struct layouts for each ioctl
- The correct initialization sequence
- What parameters the kernel validates and rejects
- Whether the QMD (compute dispatch) format matches desktop Ampere

---

## 3. Step 1: Get the Kernel Source

### What we did

Downloaded the L4T r36.4.4 BSP sources from NVIDIA:
```bash
wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.4/sources/public_sources.tbz2
tar xf public_sources.tbz2
# Inside: Linux_for_Tegra/source/kernel_oot_modules_src.tbz2
tar xf kernel_oot_modules_src.tbz2
```

### What we found

The key header files that define the ioctl interface:

| File | What it defines |
|------|-----------------|
| `nvgpu/include/uapi/linux/nvgpu.h` | Channel ioctls (Magic 'H'), TSG ioctls (Magic 'T'), struct definitions |
| `nvgpu/include/uapi/linux/nvgpu-ctrl.h` | Ctrl-GPU ioctls (Magic 'G'), GPU characteristics, REGISTER_BUFFER |
| `nvgpu/include/uapi/linux/nvgpu-as.h` | Address space ioctls (Magic 'A'), MAP_BUFFER_EX |
| `nvidia-oot/include/uapi/linux/nvmap.h` | nvmap ioctls (Magic 'N'), memory allocation |

### Key concept: Linux ioctl encoding

Every Linux ioctl has a 32-bit code that encodes:
```
bits 31-30: direction (00=none, 01=write, 10=read, 11=read+write)
bits 29-16: size of the struct (14 bits, max 16383 bytes)
bits 15-8:  magic number (identifies the device type)
bits 7-0:   command number (identifies the specific ioctl)
```

For example, `GET_CHARACTERISTICS` is `_IOWR('G', 5, 16)`:
- Direction: READ|WRITE (0x3)
- Magic: 'G' (0x47)
- Command: 5
- Size: 16 bytes (the wrapper struct)

This encoding is why strace output shows things like `_IOC(_IOC_READ|_IOC_WRITE, 0x47, 0x5, 0x10)`.

---

## 4. Step 2: Strace CUDA to See What It Does

### Why strace?

The kernel headers tell you what ioctls EXIST, but not:
- Which ones CUDA actually uses
- In what ORDER they must be called
- What VALUES to pass in the structs

Stracing a working CUDA program gives you the complete, correct initialization sequence.

### How we ran it

We created a nix dev shell (`detective`) with strace and gcc, then:
```bash
strace -f -e trace=ioctl,openat,mmap -o /tmp/cuda_trace.txt \
  python3 -c "
import ctypes
cuda = ctypes.CDLL('/path/to/libcuda.so.1')
cuda.cuInit(0)
dev = ctypes.c_int()
cuda.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.POINTER(ctypes.c_void_p)()
cuda.cuCtxCreate(ctypes.byref(ctx), 0, dev.value)
"
```

### The output

2783 lines of ioctl calls. 1793 total ioctls. 39 unique ioctl codes.

### The eureka moment: ZERO SUBMIT_GPFIFO ioctls

When we grepped for `SUBMIT_GPFIFO` (ioctl H:107, which is how the kernel-mode submit path works), there were **zero matches**. CUDA on Jetson uses **usermode submit** — it writes GPFIFO entries directly to mapped memory and rings a hardware doorbell. No ioctl needed for each GPU submission!

This is the same approach tinygrad uses on desktop via `NVKIface`. It's the fast path.

---

## 5. Step 3: Decode Every Ioctl

### Methodology

For each strace line like:
```
ioctl(4, _IOC(_IOC_READ|_IOC_WRITE, 0x47, 0x5, 0x10), 0xffffffff9890) = 0
```

We decoded:
1. **fd=4** — which device? (we tracked fd assignments from `openat` calls)
2. **Magic=0x47='G'** — ctrl-gpu device
3. **Nr=0x5** — command 5 → look up in `nvgpu-ctrl.h` → `GET_CHARACTERISTICS`
4. **Size=0x10=16** — struct size (confirms which overload)
5. **Return=0** — success

We built a complete table of all 39 ioctls, their frequencies, and their sequence.

### Reading the strace sequence

The sequence revealed the initialization order:
1. Open `/dev/nvmap` and `/dev/nvgpu/igpu0/ctrl`
2. Discovery phase: GET_CHARACTERISTICS, TPC_MASKS, engine info, clocks
3. Memory setup: ALLOC_AS, allocate buffers via nvmap
4. Channel setup: OPEN_TSG → CREATE_SUBCONTEXT → OPEN_CHANNEL → bind → SETUP_BIND
5. Compute setup: GET_USER_SYNCPOINT → ALLOC_OBJ_CTX(compute class)

This sequence became the roadmap for our Python test.

---

## 6. Step 4: Build Python ctypes Structs

### The challenge

The kernel header says:
```c
struct nvgpu_gpu_characteristics {
    __u32 arch;
    __u32 impl;
    __u32 rev;
    __u32 num_gpc;
    __s32 numa_domain_id;
    __u64 L2_cache_size;
    // ... 70+ more fields, 328 bytes total
};
```

We need to translate this to Python ctypes EXACTLY — same field order, same sizes, same alignment. One wrong field and the entire struct is shifted, causing garbage data.

### Python ctypes translation

```python
class nvgpu_gpu_characteristics(ctypes.Structure):
    _fields_ = [
        ("arch",            c_uint32),    # offset 0
        ("impl",            c_uint32),    # offset 4
        ("rev",             c_uint32),    # offset 8
        ("num_gpc",         c_uint32),    # offset 12
        ("numa_domain_id",  c_int32),     # offset 16 (__s32!)
        ("_pad0",           c_uint32),    # offset 20 — IMPLICIT PADDING
        ("L2_cache_size",   c_uint64),    # offset 24 (u64 needs 8-byte alignment)
        # ...
    ]
```

### The alignment trap

**This is the #1 source of bugs.** On aarch64, the compiler inserts padding to align fields:
- `u64` fields need 8-byte alignment
- After a `u32` at offset 16, the next `u64` goes at offset 24, NOT 20
- You MUST add explicit padding fields in your ctypes struct

If you get this wrong, `ctypes.sizeof()` will be wrong, and the ioctl will return garbage or EINVAL.

### Verification technique

Always verify your struct size matches the kernel:
```python
assert ctypes.sizeof(nvgpu_gpu_characteristics) == 328
assert ctypes.sizeof(nvgpu_alloc_as_args) == 64
assert ctypes.sizeof(nvgpu_channel_setup_bind_args) == 104
```

If the size doesn't match, you have an alignment error.

---

## 7. Step 5: Test Basic Ioctls

### GET_CHARACTERISTICS — the first win

This was the first ioctl we got working. It's a two-level indirection:
1. Create a wrapper struct with `buf_size` (328) and `buf_addr` (pointer to characteristics buffer)
2. Call the ioctl with the wrapper struct
3. The kernel fills in the characteristics buffer

Result: arch=0x0170, SM 8.7, compute_class=0xc7c0. The GPU is talking to us!

### NVMAP CREATE + ALLOC — memory allocation

```
NVMAP_CREATE(size=4096)  → handle=0x80000D46
NVMAP_ALLOC(handle, heap=IOVMM, align=4096, kind=0)  → success
NVMAP_GET_FD(handle)  → dmabuf_fd=5
```

**Trap:** The `nvmap_alloc_handle` struct is 17 bytes of actual data + 3 bytes padding = 20 bytes. We had to use `_pack_ = 1` and add explicit `_pad` bytes to match the kernel's 20-byte struct size.

**Trap:** The `GET_AVAILABLE_HEAPS` ioctl reports only VPR and FSI heaps. But IOVMM (1<<30) actually works! And SYSMEM (1<<31) does NOT work (ENOMEM). You have to just try things.

---

## 8. Step 6: The Hard Stuff — ALLOC_AS and PDE Alignment

### The problem

`ALLOC_AS` creates a GPU address space. It kept returning EINVAL. We tried different parameters, different flags — nothing worked.

### The detective work

We extracted the kernel source (`common/mm/as.c`) and read the validation code:

```c
// From gk20a_vm_alloc_share():
if (va_range_start == 0 || va_range_end == 0) {
    return -EINVAL;  // ← THIS!
}
if (!IS_ALIGNED(va_range_start, pde_size) || !IS_ALIGNED(va_range_end, pde_size)) {
    return -EINVAL;  // ← AND THIS!
}
```

So `va_range_start` and `va_range_end` MUST be non-zero AND aligned to PDE size.

### Finding the PDE size

The GPU characteristics report `pde_coverage_bit_count = 47`. But that's the TOP-LEVEL PDE, not the alignment requirement for ALLOC_AS. The actual PDE alignment for ga10b is 2^21 = 2MB.

How we found this: **brute force.** We tried every power of 2 as alignment:
```python
for bits in range(12, 48):
    pde = 1 << bits
    args.va_range_start = pde
    args.va_range_end = (1 << 40) - pde
    try:
        ioctl(ctrl_fd, ALLOC_AS, args)
        print(f"pde_bits={bits} WORKS!")  # bits=21 works!
        break
    except:
        pass
```

**Lesson:** When the kernel source validation tells you something must be "aligned" but doesn't tell you the alignment value, brute force is a valid strategy.

### Working parameters

```python
args.big_page_size = 0        # ga10b has no big pages
args.flags = 2                # UNIFIED_VA (required for compute)
args.va_range_start = 0x200000     # 2MB (PDE aligned)
args.va_range_end = 0xFFFFE00000   # ~1TB - 2MB
args.va_range_split = 0       # must be 0 for UNIFIED_VA
```

---

## 9. Step 7: The Channel Pipeline

### The correct order (discovered from strace)

This was one of the most frustrating parts. The channel setup has to happen in a specific order, and getting it wrong gives you EINVAL with no helpful error message.

**Correct order (each step depends on previous):**

```
1. OPEN_TSG (on ctrl fd)
   → returns TSG fd

2. CREATE_SUBCONTEXT (on TSG fd)
   type=ASYNC, as_fd=<AS fd>
   → returns VEID (virtual engine ID)

3. OPEN_CHANNEL (on ctrl fd)
   runlist_id=-1 (auto)
   → returns channel fd

4. AS BIND_CHANNEL (on AS fd)     ← MUST BE BEFORE TSG BIND!
   channel_fd=<channel fd>

5. TSG BIND_CHANNEL_EX (on TSG fd)
   channel_fd=<channel fd>, subcontext_id=<VEID>

6. WDT disable (on channel fd)
   wdt_status=1 (disable)

7. Allocate GPFIFO + userd buffers (via nvmap)

8. SETUP_BIND (on channel fd)
   flags = USERMODE_SUPPORT | DETERMINISTIC

9. GET_USER_SYNCPOINT (on channel fd)

10. ALLOC_OBJ_CTX (on channel fd)
    class_num=0xc7c0 (compute)
```

### Mistakes we made and fixed

**Wrong bind order:** We initially bound the channel to TSG before binding to AS. This caused EINVAL because the kernel checks that a channel has an AS before it can be bound to a TSG.

**Wrong OPEN_CHANNEL struct:** The kernel header defines `nvgpu_gpu_open_channel_args` as a union — it's just a single `s32` (4 bytes). We initially used a 16-byte struct with multiple fields, which caused ENOTTY (wrong ioctl size).

**Wrong CREATE_SUBCONTEXT struct:** We initially used two `u64` fields (16 bytes total but wrong layout). The correct struct is: `type(u32), as_fd(s32), veid(u32, out), reserved(u32)` — still 16 bytes but with completely different field types.

---

## 10. Step 8: SETUP_BIND — The Final Boss

### The problem

Everything up to step 7 worked, but SETUP_BIND kept returning EINVAL. We had the right struct layout (verified by size check: 104 bytes). We had the right buffer sizes. We had the right dmabuf fds.

### The investigation

We extracted the kernel source (`common/fifo/channel.c`) and found `channel_setup_bind_prechecks()`:

```c
if ((args->flags & USERMODE_SUPPORT) != 0U &&
    (args->flags & SUPPORT_DETERMINISTIC) == 0U) {
    nvgpu_err(g, "need deterministic for usermode submit");
    err = -EINVAL;
}
```

**The kernel requires DETERMINISTIC flag when using USERMODE_SUPPORT.** Our flags were `(1<<3)` (USERMODE only). They needed to be `(1<<3) | (1<<1)` (USERMODE + DETERMINISTIC).

### Why this makes sense

Deterministic mode means:
- The channel won't use async job tracking
- The watchdog must be disabled (which we already did)
- Power management is handled via refcount rather than idle detection
- This is required for usermode submit because usermode submit bypasses the kernel's job tracking

### The fix

One line change:
```python
# Before (FAILS):
setup.flags = NVGPU_CHANNEL_SETUP_BIND_FLAGS_USERMODE_SUPPORT  # (1<<3)

# After (WORKS):
setup.flags = (NVGPU_CHANNEL_SETUP_BIND_FLAGS_USERMODE_SUPPORT |
               NVGPU_CHANNEL_SETUP_BIND_FLAGS_DETERMINISTIC)    # (1<<3)|(1<<1)
```

### The result

```
SETUP_BIND succeeded!
  Work submit token: 511
  Syncpoint ID: 17, max=30000, GPU VA=0xffffe10000
  Compute class 0xc7c0 allocated!
```

**7/7 tests pass. We have proven direct GPU access works without CUDA.**

---

## 11. Key Concepts You Need to Know

### GPU Architecture Hierarchy

```
GPU (ga10b)
  └── GPC (Graphics Processing Cluster) — 1 on Orin AGX
        └── TPC (Texture Processing Cluster) — 4 per GPC
              └── SM (Streaming Multiprocessor) — 1 per TPC = 4 total
```

### The nvgpu Object Model

```
                    /dev/nvmap (memory)
                        │
                  nvmap handle
                  (CREATE → ALLOC → GET_FD → dmabuf)
                        │
/dev/nvgpu/igpu0/ctrl ──┤
    │                   │
    ├── ALLOC_AS ─────► AS fd
    │                   │  (GPU virtual address space)
    │                   │  MAP_BUFFER_EX maps dmabufs into GPU VA
    │                   │
    ├── OPEN_TSG ─────► TSG fd
    │                   │  (Time Slice Group — scheduling unit)
    │                   │  CREATE_SUBCONTEXT → VEID
    │                   │  BIND_CHANNEL_EX → binds channel
    │                   │
    └── OPEN_CHANNEL ─► Channel fd
                        │  (actual command submission endpoint)
                        │  SETUP_BIND → GPFIFO + userd + doorbell token
                        │  ALLOC_OBJ_CTX → compute class object
                        │  GET_USER_SYNCPOINT → completion tracking
```

### Memory Model on Jetson Orin

Jetson has **unified memory** — CPU and GPU share the same physical DRAM (64GB). This means:
- No explicit CPU↔GPU copies needed
- nvmap allocates from a shared pool (IOVMM heap)
- CPU mmaps the handle for CPU access
- MAP_BUFFER_EX maps the same memory into GPU virtual address space
- Coherence is maintained by IO_COHERENCE hardware

### Usermode Submit

Desktop NVIDIA (via RM) and older nvgpu (kernel-mode submit) use an ioctl for every GPU submission:
```
CPU: write commands to push buffer
CPU: ioctl(SUBMIT_GPFIFO, {gpu_va, length})  ← expensive syscall!
Kernel: validates, copies GPFIFO entry, kicks GPU
```

Usermode submit bypasses this:
```
CPU: write commands to push buffer
CPU: write GPFIFO entry directly to mapped memory  ← just a memory write!
CPU: write to userd doorbell  ← another memory write
GPU: picks up new work from GPFIFO ring buffer
```

This is much faster — zero syscall overhead for submissions. CUDA on Jetson uses this, and our TegraIface will too.

### GPFIFO Ring Buffer

The GPFIFO is a circular buffer of 8-byte entries. Each entry points to a "push buffer" in GPU memory:
```
GPFIFO entry (8 bytes):
  bits 63-40: length (in dwords)
  bits 39-0:  GPU virtual address of push buffer segment
```

The push buffer contains the actual GPU commands (method calls on the compute class).

### Syncpoints

Syncpoints are hardware counters used for CPU↔GPU synchronization:
- GPU increments a syncpoint when work completes
- CPU reads the syncpoint value to check completion
- The syncpoint is mapped at a GPU VA so the GPU can write to it directly

Our channel got syncpoint ID=17 at GPU VA 0xffffe10000.

### Compute Class 0xc7c0

The compute class is the GPU's compute engine interface. It accepts "methods" (register writes) that configure and launch compute kernels. Key methods include:
- QMD (Queue Meta Data) setup — grid dimensions, shared memory, shader address
- Class methods for binding textures, constants, etc.
- Launch method — triggers actual kernel execution

This is the same class as desktop Ampere, so tinygrad's existing QMD formatting code should work.

---

## 12. Tools and Methodology

### Nix Dev Shells

We used two nix dev shells (defined in `flake.nix`):
- **default** — has Python 3.13, tinygrad, CUDA libraries. Use for running tests.
- **detective** — has strace, gcc, gdb, file, hexdump. Use for reverse engineering.

```bash
# Run test:
nix develop --command python3 test_nvgpu.py

# Run strace:
nix develop .#detective --command strace -f -e trace=ioctl ...
```

### The Iteration Loop

For every new ioctl, we followed this pattern:

1. **Read the kernel header** — what struct does it expect? What fields?
2. **Write the Python ctypes struct** — match field order, types, alignment
3. **Verify struct size** — `ctypes.sizeof()` must match kernel header
4. **Call the ioctl** — fill in the struct, call `fcntl.ioctl()`
5. **If EINVAL** — read the kernel source (`.c` file) to find the validation code
6. **Fix and retry** — adjust parameters, struct layout, or call order

### Reading Kernel Source for Validation

When an ioctl returns EINVAL, the answer is almost always in the kernel `.c` file:

```bash
# Find the file:
find l4t-sources/nvgpu -name "*.c" | xargs grep -l "setup_bind"

# Read the validation function:
grep -n "EINVAL" common/fifo/channel.c
# → shows every place the kernel rejects your parameters
```

### Strace Decoding Workflow

```bash
# 1. Capture the trace
strace -f -e trace=ioctl -o /tmp/cuda_trace.txt python3 -c "..."

# 2. Count unique ioctls
grep "ioctl" /tmp/cuda_trace.txt | grep -oP "0x47, 0x[0-9a-f]+" | sort | uniq -c | sort -rn

# 3. Find specific sequences
grep "0x48, 0x80" /tmp/cuda_trace.txt  # Find all SETUP_BIND calls

# 4. Read context around a call
sed -n '2088,2106p' /tmp/cuda_trace.txt  # See what happens before/after
```

---

## 13. Common Pitfalls

### 1. Struct alignment on aarch64

Always check that your ctypes struct size matches the kernel header size. If not, you have alignment padding wrong. aarch64 aligns `u64` to 8 bytes, `u32` to 4 bytes.

### 2. Signed vs unsigned integers

Some fields are `__s32` (signed) not `__u32` (unsigned). Using the wrong type flips the interpretation of -1, which is often used as a default/invalid value.

### 3. The nvmap GET_FD trick

`NVMAP_IOC_GET_FD` reuses the `nvmap_create_handle` struct. The `handle` field is input, and the fd is returned in the `size` field(!). This is a union in the kernel but looks confusing.

### 4. Device path discovery

Old docs say `/dev/nvhost-ctrl-gpu`. The actual path on JetPack 6 is `/dev/nvgpu/igpu0/ctrl`. Channel/TSG/AS are NOT opened as separate device files — they're returned as fds from ioctls on the ctrl device.

### 5. IOVMM heap vs SYSMEM

Even though `GET_AVAILABLE_HEAPS` doesn't list IOVMM, it works. SYSMEM (listed in some docs) does NOT work (returns ENOMEM). Always try IOVMM first.

### 6. VA range requirements

ALLOC_AS absolutely requires non-zero, PDE-aligned VA ranges. Don't pass zeros. PDE alignment is 2^21 = 2MB for ga10b.

### 7. Print everything

When debugging ioctls, print every field of the output struct after the call. Values that look wrong (e.g., syncpoint_id = 4292935680) usually mean your struct field order is wrong.

---

## Timeline of the Actual Work

| Step | What Happened | Key Discovery |
|------|--------------|---------------|
| 1 | Downloaded L4T BSP, extracted headers | Found UAPI headers with all struct definitions |
| 2 | Built strace helper, straced CUDA + GPT-2 | ZERO SUBMIT_GPFIFO — usermode submit! |
| 3 | Decoded all 39 ioctls from strace | Complete initialization sequence mapped |
| 4 | Wrote initial test_nvgpu.py | First working ioctl: GET_CHARACTERISTICS |
| 5 | Fixed characteristics struct alignment | numa_domain_id is s32, needs padding before u64 |
| 6 | Got NVMAP CREATE+ALLOC working | IOVMM heap works, SYSMEM doesn't |
| 7 | Fixed nvmap_alloc_handle size (17→20) | Need _pack_=1 plus explicit trailing padding |
| 8 | ALLOC_AS kept failing EINVAL | Extracted kernel source, found VA range validation |
| 9 | Brute-forced PDE alignment | 2^21 = 2MB works for ga10b |
| 10 | MAP_BUFFER_EX worked first try | GPU VA = 0xffffa00000 |
| 11 | CREATE_SUBCONTEXT EINVAL | Wrong struct — was 2×u64, correct is 4×u32 |
| 12 | OPEN_CHANNEL ENOTTY | Wrong size — was 16 bytes, correct is 4 bytes (union) |
| 13 | BIND_CHANNEL_EX EINVAL | Wrong bind order — AS bind must come BEFORE TSG bind |
| 14 | SETUP_BIND EINVAL | Missing DETERMINISTIC flag — kernel source says required |
| 15 | Fixed syncpoint struct | Field order was wrong: gpu_va first, not last |
| 16 | **ALL 7/7 TESTS PASS** | Compute class 0xc7c0 allocated! |

---

## What's Next

With Phase 1 complete, we've proven that Python can talk directly to the GPU via nvgpu/nvmap ioctls and set up a full compute channel. The next phases are:

- **Phase 2:** Actually read/write GPU memory (mmap nvmap handles, verify CPU↔GPU coherence)
- **Phase 3:** Push actual compute commands (GPFIFO entries, QMD, shader dispatch)
- **Phase 4:** Wrap it all in a `TegraIface` class that plugs into tinygrad

The hardest reverse-engineering work is done. Phases 2-4 are engineering with a known API.
