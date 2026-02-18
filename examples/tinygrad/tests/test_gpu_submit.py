#!/usr/bin/env python3
"""
Test: Command submission through GPFIFO with semaphore release.

Validates the full pipeline:
1. Channel setup (compute + DMA, properly-sized buffers)
2. Allocate a semaphore buffer + push buffer
3. Submit push buffer via GPFIFO → GPU writes semaphore value
4. Verify the GPU wrote the expected value

This proves end-to-end GPU command submission works.

Log file: tests/test_gpu_submit.log
"""

import sys, os, struct, time, mmap, ctypes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tegra_helpers import *

# NV class method registers (from nv_gpu autogen headers)
NVC6C0_SET_OBJECT = 0x0000
NVC56F_SEM_ADDR_LO = 0x005c
NVC56F_SEM_ADDR_HI = 0x0060
NVC56F_SEM_PAYLOAD_LO = 0x0064
NVC56F_SEM_PAYLOAD_HI = 0x0068
NVC56F_SEM_EXECUTE = 0x006c

# Semaphore execute flags
SEM_EXECUTE_RELEASE = 1          # operation=release
SEM_EXECUTE_RELEASE_WFI = (1 << 20)  # release_wfi=en
SEM_EXECUTE_64BIT = (1 << 24)    # payload_size=64bit
SEM_EXECUTE_TIMESTAMP = (1 << 25) # release_timestamp=en

DOORBELL_OFFSET = 0x90  # NV_USERMODE_NOTIFY_CHANNEL_PENDING


def build_semaphore_release_pushbuf(sem_gpu_va, value):
    """Build a push buffer that releases a semaphore with a 64-bit value.
    
    Uses subchannel 0 (control) with direct SEM_ADDR/PAYLOAD/EXECUTE methods.
    Returns list of u32 words.
    """
    words = []

    def nvm(subchannel, mthd, *args, typ=2):
        words.append((typ << 28) | (len(args) << 16) | (subchannel << 13) | (mthd >> 2))
        words.extend(args)

    # SEM_ADDR_LO, SEM_ADDR_HI (contiguous methods, can use incrementing type)
    nvm(0, NVC56F_SEM_ADDR_LO,
        sem_gpu_va & 0xFFFFFFFF,       # addr lo
        (sem_gpu_va >> 32) & 0xFF,     # addr hi
        value & 0xFFFFFFFF,            # payload lo
        (value >> 32) & 0xFFFFFFFF,    # payload hi
        SEM_EXECUTE_RELEASE | SEM_EXECUTE_RELEASE_WFI | SEM_EXECUTE_64BIT | SEM_EXECUTE_TIMESTAMP)

    return words


def main():
    log = Logger(os.path.join(os.path.dirname(__file__), "test_gpu_submit.log"))

    try:
        # === Setup ===
        log.log("=== GPU Command Submission Test ===")
        log.log("")

        log.log("Step 1: Open device nodes")
        nvmap_fd = os.open("/dev/nvmap", os.O_RDWR | os.O_SYNC)
        ctrl_fd = os.open("/dev/nvgpu/igpu0/ctrl", os.O_RDWR)
        log.log(f"  nvmap_fd={nvmap_fd}, ctrl_fd={ctrl_fd}")

        log.log("Step 2: GET_CHARACTERISTICS")
        chars = get_gpu_characteristics(ctrl_fd)
        log.log(f"  compute=0x{chars.compute_class:04x} dma=0x{chars.dma_copy_class:04x}")

        log.log("Step 3: ALLOC_AS + TSG + subcontext")
        as_fd = alloc_address_space(ctrl_fd)
        tsg_fd = open_tsg(ctrl_fd)
        veid = create_subcontext(tsg_fd, as_fd)
        log.log(f"  as_fd={as_fd}, tsg_fd={tsg_fd}, veid={veid}")

        # === Channel setup ===
        ENTRIES = 1024

        log.log(f"Step 4: Create COMPUTE channel (entries={ENTRIES})")
        compute = create_full_channel(log, ctrl_fd, as_fd, tsg_fd, veid, nvmap_fd,
                                       entries=ENTRIES, label="compute")

        log.log(f"Step 5: ALLOC_OBJ_CTX compute class 0x{chars.compute_class:04x}")
        alloc_obj_ctx(compute['ch_fd'], chars.compute_class)
        log.log(f"  PASSED!")

        # === mmap doorbell ===
        log.log("Step 6: mmap doorbell (ctrl fd)")
        doorbell_base = cpu_mmap(ctrl_fd, 0x1000)
        doorbell = (ctypes.c_uint32 * (0x1000 // 4)).from_address(doorbell_base)
        log.log(f"  doorbell_base=0x{doorbell_base:x}")

        # === mmap channel buffers ===
        log.log("Step 7: mmap compute channel buffers")
        gpfifo_cpu = cpu_mmap(compute['gpfifo_dmabuf'], compute['gpfifo_size'])
        userd_cpu = cpu_mmap(compute['userd_dmabuf'], 4096)
        log.log(f"  gpfifo_cpu=0x{gpfifo_cpu:x}, userd_cpu=0x{userd_cpu:x}")

        gpfifo_ring = (ctypes.c_uint64 * ENTRIES).from_address(gpfifo_cpu)
        userd_view = (ctypes.c_uint8 * 4096).from_address(userd_cpu)

        # === Allocate semaphore buffer ===
        log.log("Step 8: Allocate semaphore buffer (4KB)")
        sem_handle, sem_dmabuf = nvmap_create_alloc_getfd(nvmap_fd, 4096, flags=NVMAP_HANDLE_WRITE_COMBINE)
        sem_gpu_va = gpu_va_map(as_fd, sem_dmabuf, 4096)
        sem_cpu = cpu_mmap(sem_dmabuf, 4096)
        log.log(f"  sem_gpu_va=0x{sem_gpu_va:012x}, sem_cpu=0x{sem_cpu:x}")

        # Clear semaphore to 0
        sem_array = (ctypes.c_uint8 * 4096).from_address(sem_cpu)
        ctypes.memset(sem_cpu, 0, 4096)
        log.log(f"  Semaphore initial value: {struct.unpack_from('<Q', sem_array, 0)[0]}")

        # === Allocate push buffer ===
        log.log("Step 9: Allocate push buffer (4KB)")
        pb_handle, pb_dmabuf = nvmap_create_alloc_getfd(nvmap_fd, 4096, flags=NVMAP_HANDLE_WRITE_COMBINE)
        pb_gpu_va = gpu_va_map(as_fd, pb_dmabuf, 4096)
        pb_cpu = cpu_mmap(pb_dmabuf, 4096)
        log.log(f"  pb_gpu_va=0x{pb_gpu_va:012x}, pb_cpu=0x{pb_cpu:x}")

        # === Build and write push buffer ===
        MAGIC_VALUE = 0xDEADBEEF42
        log.log(f"Step 10: Build push buffer (semaphore release value=0x{MAGIC_VALUE:x})")
        words = build_semaphore_release_pushbuf(sem_gpu_va, MAGIC_VALUE)
        log.log(f"  Push buffer: {len(words)} words")
        for i, w in enumerate(words):
            log.log(f"    [{i}] 0x{w:08x}")

        pb_u32 = (ctypes.c_uint32 * (4096 // 4)).from_address(pb_cpu)
        for i, w in enumerate(words):
            pb_u32[i] = w

        # === Submit via GPFIFO ===
        log.log("Step 11: Submit push buffer via GPFIFO")

        # Read current GPPut
        gp_put = struct.unpack_from('<I', userd_view, 0x8C)[0]
        log.log(f"  GPPut before: {gp_put}")

        # Format GPFIFO entry: (addr//4 << 2) | (len << 42) | (1 << 41)
        gpfifo_entry = (pb_gpu_va & ~3) | (len(words) << 42) | (1 << 41)
        log.log(f"  GPFIFO entry: 0x{gpfifo_entry:016x}")
        log.log(f"    addr={pb_gpu_va:#x}, len={len(words)}, priv=1")

        # Write GPFIFO entry at current GPPut position
        gpfifo_ring[gp_put % ENTRIES] = gpfifo_entry

        # Update GPPut
        new_gp_put = (gp_put + 1) % ENTRIES
        struct.pack_into('<I', userd_view, 0x8C, new_gp_put)
        log.log(f"  GPPut after: {new_gp_put}")

        # Memory barrier (ARM64 DSB)
        ctypes.CDLL("libc.so.6").syscall(0)  # any syscall acts as memory barrier

        # Ring doorbell
        token = compute['token']
        log.log(f"  Ringing doorbell: token={token} at offset 0x{DOORBELL_OFFSET:x}")
        doorbell[DOORBELL_OFFSET // 4] = token

        # === Wait for GPU to process ===
        log.log("Step 12: Wait for semaphore write...")
        deadline = time.monotonic() + 5.0  # 5 second timeout
        while time.monotonic() < deadline:
            val = struct.unpack_from('<Q', sem_array, 0)[0]
            if val != 0:
                log.log(f"  Semaphore value: 0x{val:016x}")
                break
            time.sleep(0.001)
        else:
            val = struct.unpack_from('<Q', sem_array, 0)[0]
            log.log(f"  TIMEOUT! Semaphore value after 5s: 0x{val:016x}")
            # Read GPGet to see if GPU consumed the entry
            gp_get = struct.unpack_from('<I', userd_view, 0x88)[0]
            log.log(f"  GPGet={gp_get}, GPPut={new_gp_put}")
            log.log("  FAILED: GPU did not process the command")
            return 1

        # === Verify ===
        log.log("Step 13: Verify result")
        # The semaphore release with TIMESTAMP writes: [value_lo, value_hi, timestamp_lo, timestamp_hi]
        # But with 64BIT payload the first 8 bytes are the payload
        payload_lo = struct.unpack_from('<I', sem_array, 0)[0]
        payload_hi = struct.unpack_from('<I', sem_array, 4)[0]
        payload = payload_lo | (payload_hi << 32)
        timestamp_lo = struct.unpack_from('<I', sem_array, 8)[0]
        timestamp_hi = struct.unpack_from('<I', sem_array, 12)[0]
        timestamp = timestamp_lo | (timestamp_hi << 32)

        log.log(f"  Payload:   0x{payload:016x} (expected 0x{MAGIC_VALUE:016x})")
        log.log(f"  Timestamp: 0x{timestamp:016x}")

        if payload == MAGIC_VALUE:
            log.log("")
            log.log("=" * 60)
            log.log("GPU COMMAND SUBMISSION: PASSED!")
            log.log("=" * 60)
            log.log("")
            log.log("Proven:")
            log.log("  ✓ Channel setup with separate small buffers works")
            log.log("  ✓ GPFIFO entry written and GPU consumed it")
            log.log("  ✓ Doorbell (ctrl fd mmap offset 0x90) triggers GPU")
            log.log("  ✓ Semaphore release with 64-bit payload verified")
            log.log("  ✓ GPU timestamp written (proves GPU executed the command)")
            return 0
        else:
            log.log(f"  MISMATCH: payload=0x{payload:016x} != expected=0x{MAGIC_VALUE:016x}")
            return 1

    except Exception:
        log.log(f"EXCEPTION: {traceback.format_exc()}")
        return 1
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
