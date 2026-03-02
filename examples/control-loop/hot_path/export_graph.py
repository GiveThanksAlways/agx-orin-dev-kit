#!/usr/bin/env python3
"""
export_graph.py — Extract tinygrad HCQGraph internals for the C hot path runtime.

After TinyJit warmup, this module inspects the captured HCQGraph to extract:
  - GPU buffer addresses (input/output, CPU-mapped via Tegra unified memory)
  - GPFifo ring, gpput, doorbell addresses
  - Command queue GPU address and length
  - Timeline/kick signal addresses
  - Patch map: which locations in the command queue change per iteration

The exported config populates a hot_path_config_t struct (defined in hot_path.h)
for the C hot path to replay the exact same GPU commands without Python overhead.

Usage:
    from export_graph import export_hot_path_config
    cfg = export_hot_path_config(jit_fn, dev, x_buf, out_buf)
"""
import struct
from tinygrad.runtime.graph.hcq import HCQGraph

# Patch variable types (must match hot_path.h)
VAR_KICKOFF         = 0
VAR_TIMELINE_WAIT   = 1
VAR_TIMELINE_SIGNAL = 2


def _find_hcq_graph(jit_fn):
    """Find the HCQGraph inside a TinyJit's captured jit cache."""
    captured = jit_fn.captured
    if captured is None:
        raise RuntimeError("TinyJit has no captured graph — run warmup first")

    for ei in captured._jit_cache:
        if isinstance(ei.prg, HCQGraph):
            return ei.prg

    raise RuntimeError("No HCQGraph found in jit cache — JIT may not have graphed")


def _classify_syms(comp_queue, graph, dev):
    """
    Classify each sym in the compute queue using resolved values from the last run.

    After warmup, comp_queue._prev_resolved_syms holds the values from the last
    _apply_var_vals call. We match these against known runtime values:

      kickoff sym    → resolved to graph.kickoff_value
      tl_wait sym    → resolved to dev.timeline_value - 2  (tl_var during last call)
      tl_signal sym  → resolved to dev.timeline_value - 1  (tl_var+1 during last call)

    Returns: list of category strings, one per sym:
      'const', 'kickoff', 'tl_wait', 'tl_signal'
    """
    prev = comp_queue._prev_resolved_syms
    if prev is None or len(prev) == 0:
        raise RuntimeError("No resolved syms — was the graph warmed up?")

    ko = graph.kickoff_value
    # After last __call__: _apply_var_vals used tl_var = (dev.timeline_value - 1) - 1
    # because during that call, dev.timeline_value was one less (before next_timeline()).
    # So: tl_wait resolved = dev.timeline_value - 2, tl_signal resolved = dev.timeline_value - 1
    tl_wait_ref   = dev.timeline_value - 2
    tl_signal_ref = dev.timeline_value - 1

    classifications = []
    for i, val in enumerate(prev):
        if val is None:
            classifications.append('const')
        elif val == ko and ko != tl_wait_ref and ko != tl_signal_ref:
            classifications.append('kickoff')
        elif val == tl_wait_ref and val != ko:
            classifications.append('tl_wait')
        elif val == tl_signal_ref and val != ko:
            classifications.append('tl_signal')
        else:
            classifications.append('const')

    n_ko = classifications.count('kickoff')
    n_tw = classifications.count('tl_wait')
    n_ts = classifications.count('tl_signal')
    print(f"  Sym classification: {len(prev)} syms — "
          f"{n_ko} kickoff, {n_tw} tl_wait, {n_ts} tl_signal, "
          f"{len(prev) - n_ko - n_tw - n_ts} const")

    if n_ko == 0:
        print(f"  WARNING: no kickoff syms found (ko={ko}, tl_wait_ref={tl_wait_ref}, tl_signal_ref={tl_signal_ref})")
    if n_tw == 0 or n_ts == 0:
        print(f"  WARNING: no timeline syms found (tl_wait_ref={tl_wait_ref}, tl_signal_ref={tl_signal_ref})")

    return classifications


def export_hot_path_config(jit_fn, dev, x_buf, out_buf):
    """
    Export a hot_path_config_t-compatible dict from a warmed-up TinyJit.

    Args:
        jit_fn:  The @TinyJit-decorated function (must be warmed up)
        dev:     The NV device (Device["NV"])
        x_buf:   The input buffer (static_x._buffer())
        out_buf: The output buffer (static_o._buffer())

    Returns:
        dict with all fields matching hot_path_config_t
    """
    graph = _find_hcq_graph(jit_fn)

    # There should be exactly one device for our single-GPU graph
    assert len(graph.devices) == 1, f"Expected 1 device, got {len(graph.devices)}"
    gdev = graph.devices[0]
    assert gdev is dev, "Graph device doesn't match provided device"

    comp_queue = graph.comp_queues[dev]

    # Classify syms using last-run resolved values (no sym_infer needed)
    sym_classes = _classify_syms(comp_queue, graph, dev)

    # Build patch list from q_sints (command queue uint32 word patches)
    patches = []
    hw_base = comp_queue.hw_page.cpu_view().addr

    for off, sym_idx in comp_queue.q_sints:
        cat = sym_classes[sym_idx]
        if cat == 'const':
            continue

        addr = hw_base + off * 4  # _q is uint32 array

        if cat == 'kickoff':
            patches.append({'addr': addr, 'var_type': VAR_KICKOFF, 'mask': 0})
        elif cat == 'tl_wait':
            patches.append({'addr': addr, 'var_type': VAR_TIMELINE_WAIT, 'mask': 0})
        elif cat == 'tl_signal':
            patches.append({'addr': addr, 'var_type': VAR_TIMELINE_SIGNAL, 'mask': 0})

    # Build patch list from mv_sints (kernel args / QMD patches)
    for mv, off, sym_idx, mask in comp_queue.mv_sints:
        cat = sym_classes[sym_idx]
        if cat == 'const':
            continue

        elem_size = struct.calcsize(mv.fmt)
        addr = mv.addr + off * elem_size
        mask_val = mask if mask is not None else 0

        if cat == 'kickoff':
            patches.append({'addr': addr, 'var_type': VAR_KICKOFF, 'mask': mask_val})
        elif cat == 'tl_wait':
            patches.append({'addr': addr, 'var_type': VAR_TIMELINE_WAIT, 'mask': mask_val})
        elif cat == 'tl_signal':
            patches.append({'addr': addr, 'var_type': VAR_TIMELINE_SIGNAL, 'mask': mask_val})

    # GPFifo info
    gpfifo = dev.compute_gpfifo

    # Queue signals to reset
    queue_sig_addrs = []
    for sig in graph.queue_signals_to_reset:
        queue_sig_addrs.append(sig.base_buf.cpu_view().addr)

    # Input/output buffer CPU addresses
    x_hcq = x_buf._buf
    o_hcq = out_buf._buf
    assert x_hcq.view is not None, "Input buffer has no CPU view — not on Tegra?"
    assert o_hcq.view is not None, "Output buffer has no CPU view — not on Tegra?"

    config = {
        'input_buf_addr':       x_hcq.cpu_view().addr,
        'output_buf_addr':      o_hcq.cpu_view().addr,
        'input_size':           x_buf.nbytes,
        'output_size':          out_buf.nbytes,

        'gpfifo_ring_addr':     gpfifo.ring.addr,
        'gpfifo_gpput_addr':    gpfifo.gpput.addr,
        'gpfifo_entries_count': gpfifo.entries_count,
        'gpfifo_token':         gpfifo.token,
        'gpfifo_put_value':     gpfifo.put_value,

        'cmdq_gpu_addr':        comp_queue.hw_page.va_addr,
        'cmdq_len_u32':         len(comp_queue._q),

        'gpu_mmio_addr':        dev.gpu_mmio.addr,

        'timeline_signal_addr': dev.timeline_signal.base_buf.cpu_view().addr,
        'timeline_value':       dev.timeline_value,
        'last_tl_value':        dev.timeline_value - 1,

        'kick_signal_addr':     graph.signals['KICK'].base_buf.cpu_view().addr,
        'kickoff_value':        graph.kickoff_value,

        'queue_signal_addrs':   queue_sig_addrs,
        'patches':              patches,
    }

    print(f"  Exported config: {len(patches)} patches, "
          f"input={config['input_size']}B, output={config['output_size']}B, "
          f"cmdq={config['cmdq_len_u32']} words, "
          f"ko={config['kickoff_value']}, tl={config['timeline_value']}, "
          f"gpfifo_put={config['gpfifo_put_value']}")
    return config
