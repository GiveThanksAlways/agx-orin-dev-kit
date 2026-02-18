#!/usr/bin/env bash
# This script sets up the same environment as `nix develop .#default`
# (CUDA env vars + PYTHONPATH) so we can strace it from the detective shell.
# Usage: strace -f -e trace=ioctl,openat -o /tmp/cuda_trace.txt bash strace-cuda.sh

export CUDA_PATH="/nix/store/1n10gbic244vpavmarf1z0d90fqh2akm-nvidia-l4t-cuda-36.4.4-20250616085344/lib/libcuda.so.1"
export NVRTC_PATH="/nix/store/sqnpv4p8a274pfn62l55w9dyvqynn6bd-cuda12.6-cuda_nvrtc-12.6.85-lib/lib/libnvrtc.so"
export NVJITLINK_PATH="/nix/store/2s5x3d3kqh3j2z3a0l11ar254077hifh-cuda12.6-libnvjitlink-12.6.85-lib/lib/libnvJitLink.so"
export LIBC_PATH="/nix/store/86wgxj5p9yry03cg7czian66bvz1r1bj-glibc-2.40-218/lib/libc.so.6"
export LD_LIBRARY_PATH="/nix/store/fy2vics08wx3zz9875krb6s22zwsdm3b-cuda12.6-cuda_cudart-12.6.77/lib:/nix/store/hhsr52l65s6wkd9nb3cbfc9wmkg2w4w6-cuda12.6-libcublas-12.6.4.1-lib/lib:/nix/store/6l1bgjjgxd47j75qk9li02z26lkv2ckx-cuda12.6-libcusparse-12.5.4.2-lib/lib:/nix/store/b58rbz6gj4agzjx09xbh4my8pk524y83-cuda12.6-libcusolver-11.7.1.2-lib/lib:/nix/store/pknxqnf7l4i925qryg77vjd512za3m55-cuda12.6-libcufft-11.3.0.4-lib/lib:/nix/store/cbrn9jlsrbcpzqnv18q6fn2g8n58b9v6-cuda12.6-libcurand-10.3.7.77-lib/lib:/nix/store/sqnpv4p8a274pfn62l55w9dyvqynn6bd-cuda12.6-cuda_nvrtc-12.6.85-lib/lib:/nix/store/2s5x3d3kqh3j2z3a0l11ar254077hifh-cuda12.6-libnvjitlink-12.6.85-lib/lib:/nix/store/xpx5v1gxkis1mffqvkk7qb3qsyd50bas-cuda12.6-cudnn-9.13.0.50-lib/lib:/nix/store/1n10gbic244vpavmarf1z0d90fqh2akm-nvidia-l4t-cuda-36.4.4-20250616085344/lib:/nix/store/ql1gsviskdaqar61hggv6gxp4d07v79d-nvidia-l4t-core-36.4.4-20250616085344/lib:/nix/store/b072105qs6av7xadbl69sn8xrqm09bgx-gcc-14.3.0-lib/lib"
export PYTHONPATH="/home/agent/jetpack-nixos/examples/tinygrad/tinygrad:$PYTHONPATH"

PYTHON="/nix/store/d86p3fy3827yqc5kzwbqiqvkf1mm73p6-python3-3.13.11-env/bin/python3"

# Minimal CUDA program: allocate a tensor on GPU and read it back.
# This exercises: device init, memory alloc, compute, transfer.
exec $PYTHON -c "
import os
os.environ['CUDA'] = '1'
from tinygrad import Tensor
t = Tensor([1.0, 2.0, 3.0])
print('Result:', t.numpy())
"
