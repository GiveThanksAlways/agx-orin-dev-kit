"""cioffi_ekf.py — Wrapper for the Cioffi et al. ImuMSCKF filter.

Uses their EXACT code from:
  external/learned_inertial_model_odometry/src/filter/python/src/scekf.py

We import their real ImuMSCKF which includes:
  - numba JIT on propagation functions (their optimization)
  - FEJ (First-Estimate Jacobians) for MSCKF consistency
  - Mahalanobis gating (disabled by default, mahalanobis_factor=-1.0)
  - Full augmented-state covariance propagation
  - Proper state cloning and marginalization

The ONLY thing we swap is the TCN inference backend (step [5c] in their
pipeline). Everything else runs their code byte-for-byte. This makes the
benchmark a fair head-to-head: their EKF + PyTorch vs. their EKF + tinygrad.

If numba/scipy is unavailable (e.g., testing without full deps), falls back
to a simplified EKF clearly marked as such.
"""
import os, sys
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Import their real code (requires numba, scipy on PYTHONPATH)
# ═══════════════════════════════════════════════════════════════════════════════

USING_REAL_EKF = False

_cioffi_repo = os.environ.get("CIOFFI_REPO", "")
if _cioffi_repo:
    _cioffi_src = os.path.join(_cioffi_repo, "src")
    if os.path.isdir(_cioffi_src) and _cioffi_src not in sys.path:
        sys.path.insert(0, _cioffi_src)

try:
    # Monkey-patch numba 0.63 + numpy 2.3 static_setitem bug BEFORE importing scekf
    import numba_compat  # noqa: F401  (patches scekf JIT functions)
    from filter.python.src.scekf import ImuMSCKF
    from filter.python.src.utils.dotdict import dotdict
    from filter.python.src.utils.math_utils import mat_exp as _mat_exp
    USING_REAL_EKF = True
except ImportError as e:
    ImuMSCKF = None
    dotdict = None
    _import_error = str(e)


# ═══════════════════════════════════════════════════════════════════════════════
# Filter tuning (their default CLI values from main_filter.py)
# ═══════════════════════════════════════════════════════════════════════════════

def make_default_filter_tuning():
    """Create filter_tuning dotdict matching their CLI defaults."""
    if not USING_REAL_EKF:
        return None
    return dotdict({
        "g_norm": 9.8082,
        "sigma_na": 1e-1,
        "sigma_ng": 1e-3,
        "sigma_nba": 1e-2,
        "sigma_nbg": 1e-4,
        "init_attitude_sigma": 10.0 / 180.0 * np.pi,
        "init_yaw_sigma": 10.0 / 180.0 * np.pi,
        "init_vel_sigma": 1.0,
        "init_pos_sigma": 1.0,
        "init_bg_sigma": 1e-4,
        "init_ba_sigma": 1e-4,
        "use_const_cov": True,
        "const_cov_val_x": 0.01,
        "const_cov_val_y": 0.01,
        "const_cov_val_z": 0.01,
        "meascov_scale": 1.0,
        "mahalanobis_factor": -1.0,      # negative = disabled (their default)
        "mahalanobis_fail_scale": 0.0,
    })


def create_filter(tuning=None):
    """Create the EKF filter instance.

    Returns (filter, is_real) where is_real indicates whether we're using
    their actual ImuMSCKF (True) or the simplified fallback (False).
    """
    if USING_REAL_EKF:
        if tuning is None:
            tuning = make_default_filter_tuning()
        return ImuMSCKF(tuning), True
    else:
        print(f"  WARNING: Using simplified EKF fallback ({_import_error})")
        print(f"  For faithful results, install numba + scipy and set CIOFFI_REPO")
        return _FallbackMSCKF(), False


# ═══════════════════════════════════════════════════════════════════════════════
# Fallback: simplified MSCKF when numba/scipy unavailable
# (clearly marked — only for testing without full deps)
# ═══════════════════════════════════════════════════════════════════════════════

def _hat(v):
    v = v.flatten()
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def _mat_exp_simple(omega):
    omega = omega.flatten()
    angle = np.linalg.norm(omega)
    if angle < 1e-10:
        return np.eye(3) + _hat(omega)
    axis = omega / angle
    s, c = np.sin(angle), np.cos(angle)
    return c * np.eye(3) + (1 - c) * np.outer(axis, axis) + s * _hat(axis)

def _Jr_exp(omega):
    omega = omega.flatten()
    angle = np.linalg.norm(omega)
    if angle < 1e-10:
        return np.eye(3)
    axis = omega / angle
    s, c = np.sin(angle), np.cos(angle)
    return (s/angle)*np.eye(3) + (1-s/angle)*np.outer(axis, axis) - ((1-c)/angle)*_hat(axis)

def _rot_from_gravity(acc):
    acc = acc.reshape(3, 1)
    a_n = acc / np.linalg.norm(acc)
    b_n = np.array([[0], [0], [1.0]])
    omega = np.cross(a_n.T, b_n.T).T
    c = 1.0 / (1.0 + float(a_n.T @ b_n))
    return np.eye(3) + _hat(omega) + c * _hat(omega) @ _hat(omega)


class _FallbackMSCKF:
    """Simplified MSCKF fallback — same API as ImuMSCKF but without numba/FEJ.

    *** This is NOT their real code. Use only when numba is unavailable. ***
    The API mirrors ImuMSCKF so bench_e2e_pipeline.py works either way.
    """

    def __init__(self):
        self.g = np.array([[0], [0], [-9.8082]])
        sigma_na, sigma_ng = 1e-1, 1e-3
        sigma_nba, sigma_nbg = 1e-2, 1e-4
        self.W = np.diag([sigma_ng**2]*3 + [sigma_na**2]*3)
        self.Q = np.diag([sigma_nbg**2]*3 + [sigma_nba**2]*3)

        self.initialized = False
        self.state = type('State', (), {
            's_R': None, 's_v': None, 's_p': None,
            's_ba': None, 's_bg': None, 's_timestamp_us': -1,
            'N': 0, 'si_Rs': [], 'si_ps': [], 'si_vs': [],
            'si_timestamps_us': [],
        })()
        self.Sigma = np.eye(15) * 1e-4

    def initialize(self, acc, t_us, init_ba=None, init_bg=None):
        self.state.s_R = _rot_from_gravity(acc)
        self.state.s_v = np.zeros((3, 1))
        self.state.s_p = np.zeros((3, 1))
        self.state.s_bg = init_bg if init_bg is not None else np.zeros((3, 1))
        self.state.s_ba = init_ba if init_ba is not None else np.zeros((3, 1))
        self.state.s_timestamp_us = t_us
        self.state.N = 0
        self.state.si_Rs = []
        self.state.si_ps = []
        self.state.si_vs = []
        self.state.si_timestamps_us = []
        self.Sigma = np.diag([
            0.01, 0.01, 0.001, 1.0, 1.0, 1.0,
            1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8,
            0.04, 0.04, 0.04,
        ])
        self.initialized = True

    def propagate(self, acc, gyr, t_us, t_augmentation_us=None):
        R = self.state.s_R
        v, p = self.state.s_v, self.state.s_p
        bg, ba = self.state.s_bg, self.state.s_ba
        dt_us = t_us - self.state.s_timestamp_us
        dt = dt_us * 1e-6 if dt_us > 0 else 0.01

        acc = acc.reshape(3, 1)
        gyr = gyr.reshape(3, 1)

        dtheta = (gyr - bg).flatten() * dt
        dRd = _mat_exp_simple(dtheta)
        R_new = R @ dRd
        dv_w = R @ (acc - ba) * dt
        v_new = v + dv_w + self.g * dt
        p_new = p + v * dt + 0.5 * dv_w * dt + 0.5 * self.g * dt * dt

        A = np.eye(15)
        A[3:6, 0:3] = -_hat(dv_w)
        A[6:9, 0:3] = -_hat(0.5 * dv_w * dt)
        A[6:9, 3:6] = np.eye(3) * dt
        A[0:3, 9:12] = -R_new @ _Jr_exp(dtheta) * dt
        A[3:6, 12:15] = -R * dt
        A[6:9, 12:15] = -0.5 * R * dt * dt

        B = np.zeros((15, 6))
        B[0:3, 0:3] = -A[0:3, 9:12]
        B[3:6, 3:6] = -A[3:6, 12:15]
        B[6:9, 3:6] = -A[6:9, 12:15]

        Sigma_new = A @ self.Sigma @ A.T + B @ self.W @ B.T * dt + self.Q * dt
        self.Sigma = 0.5 * (Sigma_new + Sigma_new.T)

        self.state.s_R = R_new
        self.state.s_v = v_new
        self.state.s_p = p_new
        self.state.s_timestamp_us = t_us

        if t_augmentation_us is not None and t_augmentation_us == t_us:
            self.state.si_Rs.append(R_new.copy())
            self.state.si_ps.append(p_new.copy())
            self.state.si_vs.append(v_new.copy())
            self.state.si_timestamps_us.append(t_us)
            self.state.N += 1

    def learnt_model_update(self, meas, meas_cov, t_begin_us, t_end_us):
        ts = self.state.si_timestamps_us
        if t_begin_us not in ts or t_end_us not in ts:
            return False, None, None, None
        i_begin = ts.index(t_begin_us)
        i_end = ts.index(t_end_us)
        pred = self.state.si_ps[i_end] - self.state.si_ps[i_begin]
        innovation = meas.reshape(3, 1) - pred
        H = np.zeros((3, 15))
        H[:, 6:9] = np.eye(3)
        return True, innovation, H, meas_cov

    def apply_update(self, innovation, H, R):
        S = H @ self.Sigma @ H.T + R
        K = self.Sigma @ H.T @ np.linalg.inv(S)
        dX = K @ innovation
        self.state.s_R = _mat_exp_simple(dX[:3].flatten()) @ self.state.s_R
        self.state.s_v += dX[3:6]
        self.state.s_p += dX[6:9]
        self.state.s_bg += dX[9:12]
        self.state.s_ba += dX[12:15]
        self.Sigma = self.Sigma - K @ H @ self.Sigma
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)
        return True

    def marginalize(self, cut_idx):
        if cut_idx > 0 and cut_idx <= self.state.N:
            self.state.si_Rs = self.state.si_Rs[cut_idx:]
            self.state.si_ps = self.state.si_ps[cut_idx:]
            self.state.si_vs = self.state.si_vs[cut_idx:]
            self.state.si_timestamps_us = self.state.si_timestamps_us[cut_idx:]
            self.state.N -= cut_idx

    def get_past_state(self, t_us):
        ts = self.state.si_timestamps_us
        if t_us in ts:
            i = ts.index(t_us)
            return self.state.si_Rs[i], self.state.si_vs[i], self.state.si_ps[i]
        return self.state.s_R, self.state.s_v, self.state.s_p

    def get_state_dict(self):
        return {
            'R': self.state.s_R.copy(),
            'v': self.state.s_v.copy(),
            'p': self.state.s_p.copy(),
            'bg': self.state.s_bg.copy(),
            'ba': self.state.s_ba.copy(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Simulated IMU data generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_imu_stream(n_seconds, imu_freq=100, seed=42):
    """Generate simulated IMU data for the full pipeline benchmark.

    Returns:
      ts: (N,) timestamps in seconds
      gyro: (N, 3) gyroscope [rad/s]
      accel: (N, 3) accelerometer [m/s^2]
      thrust: (N, 3) mass-normalized thrust in body frame

    Simulates gentle hovering flight with small perturbations.
    """
    rng = np.random.RandomState(seed)
    N = int(n_seconds * imu_freq)
    dt = 1.0 / imu_freq
    ts = np.arange(N) * dt

    gyro = rng.randn(N, 3).astype(np.float64) * 0.05
    accel = np.zeros((N, 3), dtype=np.float64)
    accel[:, 2] = 9.81
    accel += rng.randn(N, 3) * 0.1

    thrust = np.zeros((N, 3), dtype=np.float64)
    thrust[:, 2] = 9.81 + rng.randn(N) * 0.5
    thrust[:, :2] = rng.randn(N, 2) * 0.1

    return ts, gyro, accel, thrust


def load_blackbird_data(dataset_path):
    """Load real Blackbird dataset from HDF5 for benchmarking."""
    import h5py
    with h5py.File(dataset_path, 'r') as f:
        ts = f['ts'][:]
        gyro = f['gyro_calib'][:]
        accel = f['accel_calib'][:]
        thrust = f['thrust'][:]
    return ts, gyro, accel, thrust
