# Compact Pipeline: First Principles

How tinygrad turns `Tensor([1,2,3]) + Tensor([4,5,6])` into GPU warps executing on
Orin's 2048 CUDA cores, stage by stage. Each section traces the data through one box
of the compact pipeline and links to the exact code that implements it.

Reference pipeline:

```text
  Tensor ops  →  UOp DAG-AST  →  Scheduler  →  Linearizer + BEAM  →  Renderer
  →  Compiler  →  HCQProgram  →  HCQGraph  →  GPFIFO + doorbell  →  GPU warps
```

---

## 1. Tensor Ops — Building the Lazy Graph

**What happens:** You write Python math. Nothing executes. Each operation just
appends a node to a directed acyclic graph (DAG).

```python
# tinygrad/tensor.py:118  (Tensor.__init__)
class Tensor:
  __slots__ = "uop", "requires_grad", "grad"

  def __init__(self, data, device=None, dtype=None, ...):
    # Every tensor is a single UOp at its core
    if isinstance(data, UOp):
      self.uop = data          # already a graph node
    elif isinstance(data, (int, float, bool)):
      self.uop = UOp.const(dtype, data)  # leaf node
```

When you do `a + b`, Python calls `__add__` which calls `_apply_uop`. This
creates a new `UOp(Ops.ADD, ...)` whose `.src` tuple points to `a.uop` and
`b.uop`. That's it — no computation, just graph construction.

**Why lazy?** Because tinygrad needs to see the *whole computation* before it can
decide how to fuse, optimize, and schedule it. Eager execution (like PyTorch default)
would force each op into a separate kernel, killing memory bandwidth.

---

## 2. UOp DAG-AST — The Universal Intermediate Representation

**What it is:** Every operation in tinygrad — math, memory access, control flow,
buffer creation — is a `UOp`. The entire computation is one big DAG of UOps.

```python
# tinygrad/uop/ops.py:122
@dataclass(eq=False, slots=True)
class UOp(OpMixin, metaclass=UOpMetaClass):
  op: Ops                          # what operation (ADD, MUL, LOAD, STORE, ...)
  dtype: DType = dtypes.void       # result type (float32, int8, ...)
  src: tuple[UOp, ...] = tuple()   # inputs — edges in the DAG
  arg: Any = None                  # op-specific data (constant value, axis, ...)
  tag: Any = None                  # metadata for scheduler tracking
```

The `Ops` enum (`tinygrad/uop/__init__.py:12`) defines ~80 operation types organized
into categories:

```python
# tinygrad/uop/__init__.py:12-79
class Ops(FastEnum):
  # --- defines ---
  DEFINE_VAR = auto(); SPECIAL = auto(); DEFINE_LOCAL = auto(); DEFINE_REG = auto()

  # --- structural ---
  SINK = auto(); AFTER = auto(); PROGRAM = auto(); LINEAR = auto()

  # --- memory ---
  INDEX = auto(); LOAD = auto(); STORE = auto()

  # --- math (unary) ---
  CAST = auto(); EXP2 = auto(); LOG2 = auto(); SIN = auto(); SQRT = auto(); NEG = auto()

  # --- math (binary) ---
  ADD = auto(); MUL = auto(); SHL = auto(); MAX = auto(); CMPLT = auto()

  # --- math (ternary) ---
  WHERE = auto(); MULACC = auto()    # fused multiply-accumulate

  # --- control flow ---
  BARRIER = auto(); RANGE = auto(); IF = auto(); END = auto()

  # --- constants ---
  CONST = auto()
```

### Why UOps instead of an AST per operation type?

Three reasons:

1. **One representation for all passes.** The same DAG flows through scheduling,
   optimization, linearization, and rendering. No conversion between IRs.

2. **Pattern matching rewrites.** Tinygrad's optimizer is built on `PatternMatcher`
   (`tinygrad/uop/ops.py`) — you write a pattern like "match `ADD(x, 0)`" and a
   replacement "return `x`". This works because every node is the same `UOp` type.

3. **Deduplication.** `UOpMetaClass` automatically caches identical UOps via
   `weakref`. If two operations produce the same (op, dtype, src, arg), only one
   node exists. This makes the graph smaller and equality checks free.

### DAG structure

```text
  CONST(1)  CONST(4)
      \       /
       ADD           ← one UOp node, src=(left, right)
        |
      STORE
        |
       SINK          ← root: collects all outputs
```

Navigation: `uop.src` gives parents (inputs), `uop.toposort()` gives all reachable
nodes in dependency order, `uop.backward_slice` gives the full subgraph.

### GroupOp — Semantic Classification

```python
# tinygrad/uop/__init__.py:125+
class GroupOp:
  Unary = {EXP2, LOG2, SIN, SQRT, RECIPROCAL, NEG, TRUNC}
  Binary = {ADD, MUL, SHL, SHR, IDIV, MAX, MOD, CMPLT, ...}
  ALU = Unary | Binary | {WHERE, MULACC}
  Elementwise = ALU | {CAST, BITCAST}
  Movement = {RESHAPE, EXPAND, PERMUTE, PAD, SHRINK, FLIP}
```

`GroupOp` matters for fusion: everything in `Elementwise` can fuse into one kernel.
Everything in `Movement` creates a kernel boundary.

---

## 3. Scheduler — Fusing Ops into GPU Kernels

**The problem:** You have a graph of 50 UOps. If each becomes its own kernel, you
pay 50x kernel launch overhead and 50x memory round-trips (write intermediate
result to DRAM, read it back). Fusion merges adjacent ops into one kernel so
intermediates stay in registers.

**Entry point:** `complete_create_schedule_with_vars()` in
`tinygrad/engine/schedule.py:145`.

The real fusion logic lives in `get_rangeify_map()`:

```python
# tinygrad/schedule/rangeify.py:566
def get_rangeify_map(sink: UOp) -> dict[UOp, UOp]:
    # 1. Tag each UOp with a numeric ID for tracking
    tsink = graph_rewrite(sink, add_tags, ...)

    # 2. Convert movement ops (reshape, permute) into RANGE operations
    #    This determines the iteration space for each op
    tsink, rctx = run_rangeify(tsink, ...)

    # 3. Algebraic simplification + constant folding
    tsink = graph_rewrite(tsink, symbolic + pm_reduce_simplify + ...)

    # 4. Insert STORE ops where kernel outputs go to buffers
    tsink = graph_rewrite(tsink, pm_add_buffers + pm_add_range_tags, ...)

    # 5. Split into separate kernels — THIS IS WHERE FUSION DECISIONS FINALIZE
    tsink = graph_rewrite(tsink, split_kernels, ...)
    #                            ^^^^^^^^^^^^^^
    # tinygrad/schedule/rangeify.py:532
    # split_kernels = PatternMatcher([
    #   (UPat((Ops.STORE, Ops.END), name="x"), split_store),
    # ])
```

### Fusion rule (simplified)

- **Same iteration space = fused.** If two elementwise ops (ADD, MUL, etc.)
  operate on the same shape and don't have a reduce between them, they share
  the same RANGE dimensions and end up in one kernel.

- **Reduce = kernel boundary.** A reduction (sum, max) changes the iteration
  space (some dimensions collapse), so it forces a new kernel.

- **Example:** `(x + 1.0) * 2.0 - 0.5` → three ALU ops, same shape → **1 kernel**.
  `(x * 2.0).sum()` → elementwise + reduce → **2 kernels**.

### The output: ExecItem

```python
# tinygrad/engine/realize.py:207
@dataclass
class ExecItem:
  ast: UOp            # the kernel's UOp AST (rooted at SINK)
  bufs: list[Buffer]  # GPU buffers (inputs + outputs)
  metadata: tuple     # tracing info
  fixedvars: dict     # constant variable bindings
  prg: Runner|None    # compiled program (filled later)
```

`create_schedule()` (`tinygrad/engine/schedule.py:18`) takes the fused graph and
topologically sorts the kernels respecting data dependencies (via `AFTER` UOps).

---

## 4. Linearizer + BEAM Search — Picking the Fastest Kernel Shape

After fusion, each kernel is still a DAG of UOps with loop dimensions (RANGE nodes).
Two questions remain: (a) what order to iterate, and (b) how to map iterations to
GPU threads/blocks.

### Optimization (BEAM or heuristic)

**Entry:** `apply_opts()` at `tinygrad/codegen/opt/postrange.py:337`.

The optimizer wraps the kernel AST in a `Scheduler` object that tracks which
optimizations have been applied:

```python
# tinygrad/codegen/opt/postrange.py:17
class Scheduler:
    ast: UOp              # kernel AST
    ren: Renderer         # target renderer
    applied_opts: list    # optimizations applied so far
```

If `BEAM >= 1` (env var), tinygrad runs **BEAM search** — a greedy best-first
search over the optimization space:

```python
# tinygrad/codegen/opt/search.py:120
def beam_search(s: Scheduler, rawbufs: list[Buffer], amt: int, ...) -> Scheduler:
    beam = [(s, float("inf"))]     # start: unoptimized kernel, infinite time
    seen_libs = set()

    while not exiting:
        # 1. Generate ALL valid next optimizations for each candidate
        candidates = flatten([get_kernel_actions(si).values() for si, _ in beam])

        # 2. Compile each candidate in parallel (multiprocessing pool)
        # 3. Time each on real GPU hardware (3 runs, take minimum)
        for i, proc in pool.imap_unordered(_try_compile, enumerate(candidates)):
            p, lib, compile_et = proc
            tms = _time_program(p, lib, var_vals, rawbufs, ...)
            timed.append((candidates[i], min(tms)))

        # 4. Keep top `amt` fastest, discard rest
        opts = sorted(timed, key=lambda x: x[1])
        beam = opts[:amt]

        # 5. Stop when improvement < 0.01µs
        exiting = (beam[0][1] - opts[0][1]) < min_progress
```

### What the optimizations do

The action space (`tinygrad/codegen/opt/search.py:13-25`):

| Action | What it does | GPU effect |
|--------|-------------|------------|
| `UPCAST` | More work per thread (vectorize) | Fewer threads, more registers |
| `UNROLL` | Unroll inner reduce loop N times | Larger kernel, fewer loop iterations |
| `LOCAL` | Map a global dim to shared memory tiles | Threads cooperate via shared mem |
| `GROUP` | Group reduce across threads | Partial sums in shared memory |
| `THREAD` | Set threads per block | Occupancy tuning |
| `TC` | Use tensor cores (WMMA) | Matrix multiply acceleration |
| `SWAP` | Reorder loop nesting | Cache line utilization |

### Linearizer — DAG to Linear Instruction List

After optimization, the DAG must become a linear sequence of instructions for the
renderer. This is `linearize()`:

```python
# tinygrad/codegen/late/linearizer.py:7
def linearize(sink: UOp) -> list[UOp]:
    lst = list(sink.toposort())

    # Assign priority to each UOp type
    for u in reversed(lst):
        match u.op:
            case Ops.PARAM:      priority = -20   # params first (load pointers)
            case Ops.LOAD:       priority = -1    # loads early (hide latency)
            case Ops.STORE:      priority = 1     # stores late (after compute)
            case Ops.RANGE:      priority = 5     # loop starts late
            case _:              priority = 0

    # Toposort with priority: heap-based, respects data deps but prefers
    # the "ideal" priority order. High-run-count ops scheduled later.
    heap = [(-nkey[sink], sink)]
    # ... heap-based toposort ...
    return output_list
```

The key insight: the linearizer doesn't *change* the computation — it just decides
the *order* of the instructions. Placing LOADs early helps hide memory latency
(the GPU can fetch data while computing something else). Placing STOREs late
means results accumulate in registers before writing out.

---

## 5. Renderer — UOps to PTX Assembly

The linearized `list[UOp]` is now a flat sequence of micro-ops. The renderer
walks this list and emits one PTX instruction per UOp.

**Entry:** `do_render()` at `tinygrad/codegen/__init__.py:174`:

```python
# tinygrad/codegen/__init__.py:174
def do_render(ctx: Renderer, prg: UOp, lin: UOp) -> UOp:
    src = ctx.render(list(lin.src))    # lin.src = linearized UOp list
    return prg.replace(src=prg.src + (UOp(Ops.SOURCE, arg=src),))
```

For NVIDIA, `ctx` is `PTXRenderer` (`tinygrad/renderer/ptx.py:148`).

### How each UOp becomes PTX

The renderer maintains a register map `r: dict[UOp, str]` — each UOp gets a
virtual register name like `%alu_f32_0`.

**Math ops** go through `asm_for_op` (`tinygrad/renderer/ptx.py:18`):

```python
# tinygrad/renderer/ptx.py:18-33
asm_for_op = {
    Ops.ADD:  lambda d,a,b,dt,name: f"add.{name} {d}, {a}, {b};",
    Ops.MUL:  lambda d,a,b,dt,name: f"mul{'.lo' if dtypes.is_int(dt) else ''}.{name} {d}, {a}, {b};",
    Ops.MAX:  lambda d,a,b,dt,name: f"max.{name} {d}, {a}, {b};",
    Ops.EXP2: lambda d,a,dt,name:   f"ex2.approx.{name} {d}, {a};",
    Ops.SQRT: lambda d,a,dt,name:   f"sqrt.approx.{name} {d}, {a};",
    Ops.MULACC: lambda d,a,b,c,dt,name: f"fma.rn.{name} {d}, {a}, {b}, {c};",
    # ...
}
```

So `UOp(Ops.ADD, dtypes.float32, src=(x, y))` becomes:
```ptx
add.f32  %alu_f32_0, %val_f32_0, %val_f32_1;
```

**Memory ops** use pattern matching (`tinygrad/renderer/ptx.py:66-106`):

```python
# tinygrad/renderer/ptx.py (string_rewrite pattern matcher)
# LOAD → ld.global.f32 %val, [%ptr + 0];
# STORE → st.global.f32 [%ptr + 0], %val;
# Gated LOAD with predicate:
#   @%p0  ld.global.f32 %f0, [%rd0+0];     ← load if predicate true
#   @!%p0 mov.b32 %f0, %f1;                ← use fallback if false
```

**Thread indexing** — `SPECIAL` UOps map to PTX thread/block IDs:

```python
# tinygrad/renderer/ptx.py:128
# SPECIAL("g0") → mov.u32 %g0, %ctaid.x;   (block index)
# SPECIAL("l0") → mov.u32 %l0, %tid.x;     (thread index within block)
```

### Final assembly

`render_kernel()` (`tinygrad/renderer/ptx.py:170`) wraps everything:

```ptx
.version 7.5
.target sm_87
.address_size 64
.visible .entry r_2_3 (
    .param .u64 data0,
    .param .u64 data1,
    .param .u64 data2
)
.maxntid 1
{
    .reg .f32 %alu_f32<2>;
    .reg .s64 %val_s64<4>;

    ld.param.u64    %rd0, [data0+0];    // load buffer pointers
    ld.param.u64    %rd1, [data1+0];
    ld.param.u64    %rd2, [data2+0];
    ld.global.f32   %f0, [%rd1+0];      // load input a
    ld.global.f32   %f1, [%rd2+0];      // load input b
    add.f32         %f2, %f0, %f1;      // a + b
    st.global.f32   [%rd0+0], %f2;      // store result
    ret;
}
```

---

## 6. Compiler — PTX to CUBIN ELF

The PTX source is human-readable assembly. The GPU doesn't execute PTX — it
executes CUBIN (Compute Unified Binary), an ELF file containing SASS machine
code for the specific GPU architecture.

**For NV=1 (tinygrad's direct GPU path):** `NVPTXCompiler`
(`tinygrad/runtime/support/compiler_cuda.py:80`):

```python
# tinygrad/runtime/support/compiler_cuda.py:80
class NVPTXCompiler(PTXCompiler):
    def compile(self, src: str) -> bytes:
        # 1. Replace placeholder strings with actual arch
        ptxsrc = super().compile(src)  # "TARGET" → "sm_87", "VERSION" → "7.5"

        # 2. Use NVIDIA's JIT linker (libnvjitlink.so)
        jitlink.nvJitLinkCreate(handle, 1, [f'-arch={self.arch}'])
        jitlink.nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX, ptxsrc, len(ptxsrc))
        jitlink.nvJitLinkComplete(handle)          # ← actual compilation happens here

        # 3. Extract CUBIN binary
        data = jitlink.nvJitLinkGetLinkedCubin(handle)
        return data   # bytes: ELF binary with SASS machine code
```

**For CUDA backend:** `NVRTCCompiler` (`compiler_cuda.py:50`) uses `libnvrtc.so`
instead — same idea, different API:

```python
# tinygrad/runtime/support/compiler_cuda.py:56
nvrtc.nvrtcCreateProgram(prog, src.encode(), ...)
nvrtc.nvrtcCompileProgram(prog, ['--gpu-architecture=sm_87'])
data = nvrtc.nvrtcGetCUBIN(prog)    # CUBIN bytes out
```

Both libraries are part of the CUDA toolkit. They're the same compiler CUDA uses —
tinygrad just calls it directly with PTX instead of going through the CUDA runtime.

**Compilation is cached** — `compile_cached()` in the `Compiler` base class hashes
the source and stores the binary on disk. Repeated runs skip compilation entirely.

**Where it runs in the pipeline:**

```python
# tinygrad/codegen/__init__.py:178
def do_compile(ctx: Renderer, prg: UOp, source: UOp) -> UOp:
    lib = ctx.compiler.compile_cached(source.arg)   # arg = PTX string
    return prg.replace(src=prg.src + (UOp(Ops.BINARY, arg=lib),))
```

The whole linearize → render → compile flow is orchestrated by `pm_to_program`
(`tinygrad/codegen/__init__.py:158`):

```python
# tinygrad/codegen/__init__.py:158
pm_to_program = PatternMatcher([
    # Step 1: PROGRAM + SINK → linearize
    (UPat(Ops.PROGRAM, src=(UPat(Ops.SINK), UPat(Ops.DEVICE))), do_linearize),
    # Step 2: PROGRAM + LINEAR → render
    (UPat(Ops.PROGRAM, src=(_, _, UPat(Ops.LINEAR))),            do_render),
    # Step 3: PROGRAM + SOURCE → compile
    (UPat(Ops.PROGRAM, src=(_, _, _, UPat(Ops.SOURCE))),         do_compile),
])
```

Each pattern matches a different stage of the PROGRAM UOp's growth (it accumulates
LINEAR, SOURCE, BINARY children as each pass runs).

---

## 7. HCQProgram — The Cached GPU Program

After compilation, we have a CUBIN binary (bytes). `NVProgram`
(`tinygrad/runtime/ops_nv.py:169`) loads this binary and extracts everything the
GPU needs to dispatch a kernel:

```python
class NVProgram(HCQProgram):
    def __init__(self, dev, name, lib, ...):
        # Parse CUBIN ELF to extract:
        image, sections, relocs = elf_loader(self.lib, force_section_align=128)
        # - .text section      → shader machine code (SASS)
        # - .nv.shared.*       → shared memory size
        # - .nv.info           → register count, local memory size
        # - Relocations        → fix up absolute addresses for this GPU
```

It also pre-builds a **QMD** (Queue Meta Data) template — a 256-byte descriptor
(`tinygrad/runtime/ops_nv.py:55`) that tells the GPU everything about a compute
dispatch:

```python
# tinygrad/runtime/ops_nv.py:55
class QMD:
    """256-byte (V03/Ampere) or 384-byte (V05/Blackwell) descriptor."""
    # Fields set at init time:
    #   program_address      → GPU VA of the shader binary
    #   register_count       → how many registers the shader uses
    #   shared_mem_size      → shared memory per block
    #   qmd_major_version=3  → V03 for Ampere (ga10b / Orin)
```

The QMD is "template" because some fields change per-dispatch (grid dimensions,
constant buffer address). Those get patched at exec time.

---

## 8. HCQGraph — Building WAIT → EXEC → SIGNAL Sequences

When tinygrad's JIT (`tinygrad/engine/jit.py`) records a sequence of kernel
launches, it batches them into an **HCQGraph** (`tinygrad/runtime/graph/hcq.py:11`).

The graph pre-builds the entire GPU command stream in memory:

```python
# tinygrad/runtime/graph/hcq.py:11
class HCQGraph(MultiGraphRunner):
    def __init__(self, jit_cache: list[ExecItem], ...):
        # For each kernel in the batch:
        for j, ji in enumerate(jit_cache):
            # 1. WAIT: stall until dependencies are done
            for sig, val in deps:
                enqueue_queue.wait(sig, val)

            # 2. EXEC: launch the kernel
            enqueue_queue.exec(ji.prg._prg, self.ji_args[j],
                               tuple(ji.prg.p.global_size),
                               tuple(ji.prg.p.local_size))

            # 3. SIGNAL: write completion value for dependents
            enqueue_queue.signal(signal, signal_val)
```

Each `wait/exec/signal` call doesn't talk to the GPU — it writes **GPU method
commands** (push buffer entries) into a pre-allocated buffer.

### NVComputeQueue.exec — Writing QMD + Methods

```python
# tinygrad/runtime/ops_nv.py:141
class NVComputeQueue(NVCommandQueue):
    def exec(self, prg, args_state, global_size, local_size):
        # Copy QMD template, patch grid/block dims + constant buf address
        qmd_buf.cpu_view()[:] = prg.qmd.mv
        qmd.write(cta_raster_width=global_size[0], ...)
        qmd.set_constant_buf_addr(0, args_state.buf.va_addr)

        # Write GPU method: "load this QMD and dispatch"
        if self.active_qmd is None:
            self.nvm(1, NVC6C0_SEND_PCAS_A, qmd_buf.va_addr >> 8)
        else:
            # Chain: link previous QMD to this one (avoids extra method)
            self.active_qmd.write(dependent_qmd0_pointer=qmd_buf.va_addr >> 8, ...)
```

The `nvm()` method (`tinygrad/runtime/ops_nv.py:98`) writes GPU push buffer entries:

```python
# tinygrad/runtime/ops_nv.py:98
def nvm(self, subchannel, mthd, *args, typ=2):
    # Format: [header][data words]
    # header = (type << 28) | (count << 16) | (subchannel << 13) | (method >> 2)
    self.q((typ << 28) | (len(args) << 16) | (subchannel << 13) | (mthd >> 2), *args)
```

### Wait and Signal as Semaphore Operations

```python
# tinygrad/runtime/ops_nv.py:103  (wait)
def wait(self, signal, value):
    self.nvm(0, NVC56F_SEM_ADDR_LO, *signal_addr, *value,
             flags(operation="acq_circ_geq", payload_size="64bit"))
    # GPU stalls until: mem[signal_addr] >= value

# tinygrad/runtime/ops_nv.py:164  (signal via QMD release)
# GPU writes `value` to `signal_addr` when kernel completes
# This is embedded IN the QMD, not a separate method — zero overhead
```

---

## 9. Push Buffer → GPFIFO → Doorbell — One MMIO Write

The entire command queue (WAIT+EXEC+SIGNAL for every kernel in the graph) is
in GPU-visible memory. Now we need to tell the GPU to execute it.

```python
# tinygrad/runtime/ops_nv.py:123
def _submit_to_gpfifo(self, dev, gpfifo):
    # 1. Write GPFIFO entry: "command queue is at this address, this long"
    gpfifo.ring[gpfifo.put_value % gpfifo.entries_count] = \
        (cmdq_addr // 4 << 2) | (len(self._q) << 42) | (1 << 41)
    #    ^^^^ GPU VA           ^^^^ length in words    ^^^^ privilege bit

    # 2. Update GPPut: "I added one entry to the ring"
    gpfifo.gpput[0] = (gpfifo.put_value + 1) % gpfifo.entries_count

    # 3. Memory barrier: ensure writes are visible before doorbell
    System.memory_barrier()     # ARM: dmb sy

    # 4. THE MMIO WRITE — this is the only thing that wakes the GPU
    dev.gpu_mmio[0x90 // 4] = gpfifo.token
    #            ^^^^
    #  Offset 0x90 = NV_USERMODE_NOTIFY_CHANNEL_PENDING
    #  Same on desktop and Tegra — the hardware convergence point

    gpfifo.put_value += 1
```

**Everything before the doorbell is just writing to shared memory.** The GPU is
idle. The single volatile write to `gpu_mmio[0x90 // 4]` is what wakes the
GPU's host interface and makes it process the GPFIFO ring.

### HCQGraph.__call__ — Execution

```python
# tinygrad/runtime/graph/hcq.py:193
def __call__(self, input_buffers, var_vals, wait=False):
    self.kickoff_value += 1

    # Resolve all symbolic variables (timeline signals, buffer addresses)
    hcq_var_vals = { self.kickoff_var.expr: self.kickoff_value, ... }

    # Submit each device's queue — each submit = one GPFIFO + one doorbell
    for dev in self.devices:
        self.comp_queues[dev].submit(dev, hcq_var_vals)

    # Kick: write kickoff signal so GPU WAITs unblock
    self.signals['KICK'].value = self.kickoff_value
```

On Orin, the submit path is **zero syscalls** — the GPFIFO ring, GPPut register,
and doorbell are all mmapped into userspace. No `ioctl(SUBMIT_GPFIFO)` needed.

---

## 10. GPU Hardware Execution

After the doorbell write, the GPU takes over:

```text
 GPU Host Interface
   |  reads GPFIFO ring → fetches push buffer from GPU memory
   v
 PBDMA (Push Buffer DMA)
   |  decodes method headers (the nvm() calls we wrote)
   |  processes WAIT: stalls until semaphore value met
   |  processes EXEC: reads QMD, sets up dispatch
   v
 GPC (Graphics Processing Cluster)
   |  loads CUBIN binary from program_address into I-cache
   |  allocates registers (from QMD register_count)
   |  allocates shared memory (from QMD shared_mem_size)
   v
 SM Dispatch (ga10b: 16 SMs × 128 CUDA cores = 2048 total)
   |  grid_dim blocks distributed across SMs
   |  each block split into 32-thread warps
   |  warps execute in lock-step (SIMT)
   v
 Memory
   |  loads/stores → LPDDR5 64GB (unified, IO coherent)
   |  L2 cache (4MB) shared across all SMs
   |  no PCIe, no VRAM copy — direct DRAM access
   v
 Completion
   |  processes SIGNAL: writes timeline value to semaphore address
   |  IO coherent → CPU sees it immediately (no cache flush needed)
   v
 CPU polls semaphore → reads output buffer → done
```

**IO coherence** (`GET_CHARACTERISTICS` flag bit 20) is why there's no explicit
cache management on Orin. When the GPU writes the completion semaphore, the CPU
sees the updated value immediately through hardware cache coherence. Desktop GPUs
with separate VRAM across PCIe don't have this.

---

## Summary: The Whole Pipeline in Code Locations

| Stage | What | Key File | Key Line(s) |
|-------|------|----------|-------------|
| **Tensor ops** | Build lazy UOp graph | `tinygrad/tensor.py` | 118 (\_\_init\_\_) |
| **UOp DAG** | Universal IR node | `tinygrad/uop/ops.py` | 122 (UOp class) |
| **Ops enum** | Operation types | `tinygrad/uop/__init__.py` | 12-79 |
| **Scheduler** | Fuse ops into kernels | `tinygrad/schedule/rangeify.py` | 566 (get_rangeify_map) |
| **Kernel split** | Decide kernel boundaries | `tinygrad/schedule/rangeify.py` | 532 (split_kernels) |
| **Schedule sort** | Topological order | `tinygrad/engine/schedule.py` | 18 (create_schedule) |
| **Optimizer** | BEAM or heuristic | `tinygrad/codegen/opt/search.py` | 120 (beam_search) |
| **Actions** | Optimization space | `tinygrad/codegen/opt/search.py` | 13-25 |
| **Linearizer** | DAG → ordered list | `tinygrad/codegen/late/linearizer.py` | 7 (linearize) |
| **PTX renderer** | UOps → PTX asm | `tinygrad/renderer/ptx.py` | 148 (PTXRenderer) |
| **PTX ops** | Op → instruction map | `tinygrad/renderer/ptx.py` | 18 (asm_for_op) |
| **Compiler** | PTX → CUBIN ELF | `tinygrad/runtime/support/compiler_cuda.py` | 80 (NVPTXCompiler) |
| **Pipeline glue** | linearize→render→compile | `tinygrad/codegen/__init__.py` | 158 (pm_to_program) |
| **NVProgram** | Load CUBIN + build QMD | `tinygrad/runtime/ops_nv.py` | 169+ |
| **QMD** | 256-byte dispatch descriptor | `tinygrad/runtime/ops_nv.py` | 55 (QMD class) |
| **HCQGraph** | Batch WAIT→EXEC→SIGNAL | `tinygrad/runtime/graph/hcq.py` | 11 |
| **Push buffer** | GPU method commands | `tinygrad/runtime/ops_nv.py` | 98 (nvm) |
| **GPFIFO submit** | Ring buffer + doorbell | `tinygrad/runtime/ops_nv.py` | 123 (\_submit_to_gpfifo) |
| **Doorbell** | The one MMIO write | `tinygrad/runtime/ops_nv.py` | 130 (gpu_mmio[0x90//4]) |

All paths relative to `/home/agent/agx-orin-dev-kit/external/tinygrad/`.
