"""
Monkey-patch for numba 0.63 + numpy 2.3 compatibility.

numba 0.63 cannot lower `static_setitem(Array, UniTuple(slice, 2), Array)`
with numpy 2.3's array representation. This module provides drop-in
replacement JIT functions for scekf.propagate_rvt_and_jac and
scekf.propagate_covariance that use explicit element-by-element assignment
instead of 2D slice assignment.

Usage:
    import numba_compat  # patches scekf at import time
"""

import numpy as np
from numba import jit

# Import the JIT'd helpers we need (these work fine — no 2D slice assignment)
from filter.python.src.utils.math_utils import mat_exp, Jr_exp


@jit(nopython=True, cache=True)
def _set_block(dst, r0, c0, src):
    """Write src[i,j] into dst[r0+i, c0+j] element-by-element."""
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            dst[r0 + i, c0 + j] = src[i, j]


@jit(nopython=True, cache=True)
def _add_block(dst, r0, c0, src):
    """dst[r0:r0+n, c0:c0+m] += src"""
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            dst[r0 + i, c0 + j] += src[i, j]


@jit(nopython=True, parallel=False, cache=True)
def propagate_rvt_and_jac(R_k, v_k, p_k, b_gk, b_ak, gyr, acc, g, dt):
    def hat(v):
        v = v.flatten()
        R = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        return R

    dtheta = (gyr - b_gk) * dt
    dRd = mat_exp(dtheta)
    Rd = R_k @ dRd
    dv_w = R_k @ (acc - b_ak) * dt
    dp_w = 0.5 * dv_w * dt
    gdt = g * dt
    gdt22 = 0.5 * gdt * dt
    vd = v_k + dv_w + gdt
    pd = p_k + v_k * dt + dp_w + gdt22

    A = np.eye(15)
    _set_block(A, 3, 0, -hat(dv_w))
    _set_block(A, 6, 0, -hat(dp_w))
    _set_block(A, 6, 3, np.eye(3) * dt)
    _set_block(A, 0, 9, -Rd @ Jr_exp(dtheta) * dt)
    _set_block(A, 3, 12, -R_k * dt)
    _set_block(A, 6, 12, -0.5 * R_k * dt * dt)

    return Rd, vd, pd, A


@jit(nopython=True, parallel=False, cache=True)
def propagate_covariance(A_aug, B_aug, dt, Sigma, W, Q):
    dim_new_state = A_aug.shape[0] - A_aug.shape[1]  # either 0 or 9
    n15d = 15 + dim_new_state
    n = A_aug.shape[0]
    top = n - n15d  # number of rows/cols in top-left block

    A = np.ascontiguousarray(A_aug[top:, n - 15:])
    AT = np.ascontiguousarray(A.T)

    ret = np.zeros((n, n))

    # top-left: Sigma[:-15, :-15]
    for i in range(top):
        for j in range(top):
            ret[i, j] = Sigma[i, j]

    # top-right: Sigma[:-15, -15:] @ AT
    tmp_tr = np.ascontiguousarray(Sigma[:top, n - 15:]) @ AT
    for i in range(top):
        for j in range(n15d):
            ret[i, top + j] = tmp_tr[i, j]

    # bottom-left: A @ Sigma[-15:, :-15]
    tmp_bl = A @ np.ascontiguousarray(Sigma[n - 15:, :top])
    for i in range(n15d):
        for j in range(top):
            ret[top + i, j] = tmp_bl[i, j]

    # bottom-right: A @ Sigma[-15:, -15:] @ AT
    tmp_br = A @ np.ascontiguousarray(Sigma[n - 15:, n - 15:]) @ AT
    for i in range(n15d):
        for j in range(n15d):
            ret[top + i, top + j] = tmp_br[i, j]

    # add noise from input: B_aug @ W @ B_aug.T
    noise = B_aug @ W @ B_aug.T
    _add_block(ret, top, top, noise)

    # add noise from bias model
    _add_block(ret, n - 6, n - 6, dt * Q)

    return ret


def patch():
    """Replace scekf module-level JIT functions with compatible versions."""
    from filter.python.src import scekf
    scekf.propagate_rvt_and_jac = propagate_rvt_and_jac
    scekf.propagate_covariance = propagate_covariance


# Auto-patch on import
patch()
