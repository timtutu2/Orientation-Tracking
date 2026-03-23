import numpy as np
import pickle
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from load_data import read_data
import matplotlib

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT   = os.path.join(_SCRIPT_DIR, '..', '..', 'data')
_TRAIN_DIR   = os.path.join(_DATA_ROOT, 'trainset')
_TEST_DIR    = os.path.join(_DATA_ROOT, 'testset')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from transforms3d.euler import mat2euler, quat2euler
from transforms3d.quaternions import quat2mat


# ─────────────────────────────────────────────────────────────────────────────
# IMU Sensor Constants
# ─────────────────────────────────────────────────────────────────────────────

# ADXL335 accelerometer + LPR530AL / LY530ALH gyroscope (4× signal output)
# at Vref = 3.3 V
VREF = 3300.0             # Reference voltage in mV
ACCEL_SENSITIVITY = 330.0  # mV / g
GYRO_SENSITIVITY  = 3.33   # mV / (deg/s)  — using 4× amplified output

# scale_factor = Vref / 1023 / sensitivity
# value = (raw_adc - bias) × scale_factor
ACCEL_SCALE = VREF / 1023.0 / ACCEL_SENSITIVITY            # g per ADC count
GYRO_SCALE  = VREF / 1023.0 / GYRO_SENSITIVITY * (np.pi / 180.0)  # rad/s per ADC count


# ─────────────────────────────────────────────────────────────────────────────
# Data I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(dataset_id, data_dir):
    """Load IMU, Vicon (if available), and camera (if available) data."""
    imu_file   = os.path.join(data_dir, 'imu',   f'imuRaw{dataset_id}.p')
    vicon_file = os.path.join(data_dir, 'vicon', f'viconRot{dataset_id}.p')
    cam_file   = os.path.join(data_dir, 'cam',   f'cam{dataset_id}.p')

    imu_raw   = read_data(imu_file)
    vicon_raw = read_data(vicon_file) if os.path.exists(vicon_file) else None
    cam_raw   = read_data(cam_file)   if os.path.exists(cam_file)   else None

    return imu_raw, vicon_raw, cam_raw


def parse_imu(imu_raw):
    if isinstance(imu_raw, dict):
        key = 'vals' if 'vals' in imu_raw else list(imu_raw.keys())[0]
        data = np.array(imu_raw[key])
    else:
        data = np.array(imu_raw)

    ts        = data[0].flatten()
    accel_raw = data[1:4]   # (3, N)
    gyro_raw  = data[4:7]   # (3, N)
    return ts, accel_raw, gyro_raw


def parse_vicon(vicon_raw):
    if isinstance(vicon_raw, dict):
        rots = np.array(vicon_raw['rots'])
        ts   = np.array(vicon_raw['ts']).flatten()
    elif isinstance(vicon_raw, (list, tuple)) and len(vicon_raw) == 2:
        rots = np.array(vicon_raw[0])
        ts   = np.array(vicon_raw[1]).flatten()
    else:
        rots = np.array(vicon_raw)
        ts   = np.arange(rots.shape[-1], dtype=float)

    # Ensure shape (N, 3, 3); the raw data is usually (3, 3, N)
    if rots.ndim == 3 and rots.shape[0] == 3 and rots.shape[1] == 3:
        rots = np.transpose(rots, (2, 0, 1))

    return rots, ts


# ─────────────────────────────────────────────────────────────────────────────
# IMU Calibration
# ─────────────────────────────────────────────────────────────────────────────

def estimate_bias(accel_raw, gyro_raw, vicon_rots, n_static=200):
    n_static = min(n_static, accel_raw.shape[1])

    # Gyro: mean during static period → offset to zero
    bias_gyro = np.mean(gyro_raw[:, :n_static], axis=1)

    # Accel: expected = gravity in body frame at initial orientation
    # g_world = [0, 0, 1] g (specific force pointing "up" in world frame)
    # g_body  = R0^T @ g_world
    R0 = vicon_rots[0]
    g_body = R0.T @ np.array([0.0, 0.0, 1.0])   # expected accel in g units
    mean_accel = np.mean(accel_raw[:, :n_static], axis=1)
    bias_accel = mean_accel - g_body / ACCEL_SCALE

    return bias_accel, bias_gyro


def estimate_bias_no_vicon(accel_raw, gyro_raw, n_static=200):
    n_static = min(n_static, accel_raw.shape[1])
    bias_gyro  = np.mean(gyro_raw[:, :n_static],  axis=1)
    mean_accel = np.mean(accel_raw[:, :n_static], axis=1)
    # Expect gravity along +z in body frame when level
    g_body = np.array([0.0, 0.0, 1.0])
    bias_accel = mean_accel - g_body / ACCEL_SCALE
    return bias_accel, bias_gyro


def calibrate_imu(accel_raw, gyro_raw, bias_accel, bias_gyro):
    accel = (accel_raw - bias_accel[:, None]) * ACCEL_SCALE
    omega = (gyro_raw  - bias_gyro[:, None])  * GYRO_SCALE
    return accel, omega


# ─────────────────────────────────────────────────────────────────────────────
# Quaternion Utilities  (w, x, y, z  convention,  PyTorch tensors)
# ─────────────────────────────────────────────────────────────────────────────

def qmult(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def qinv(q):
    s = torch.tensor([1., -1., -1., -1.], dtype=q.dtype, device=q.device)
    return q * s


def qexp(v):
    theta      = torch.linalg.norm(v, dim=-1, keepdim=True)
    safe_theta = theta.clamp(min=1e-7)
    # sinc(theta) = sin(theta)/theta → 1 as theta→0; safe_theta avoids 0/0
    sinc = torch.sin(safe_theta) / safe_theta
    return torch.cat([torch.cos(theta), sinc * v], dim=-1)


def qlog(q):
    w = q[..., :1]
    v = q[..., 1:]
    v_norm     = torch.linalg.norm(v, dim=-1, keepdim=True)
    # Clamp v_norm so that theta/v_norm is finite and its gradient is finite
    # even when v ≈ 0.  When clamped, the coefficient ≈ atan2(eps, w)/eps ≈ 1/w ≈ 1.
    safe_v_norm = v_norm.clamp(min=1e-7)
    theta       = torch.atan2(safe_v_norm, w)
    coeff       = theta / safe_v_norm          # ≈ 1 near identity; exact elsewhere
    return torch.cat([torch.zeros_like(w), coeff * v], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Motion Model and Observation Model
# ─────────────────────────────────────────────────────────────────────────────

def motion_model(qt, tau_omega):
    return qmult(qt, qexp(tau_omega / 2.0))


def observation_model(qt):
    T = qt.shape[0]
    g = torch.tensor([0., 0., 0., 1.], dtype=qt.dtype, device=qt.device)
    g = g.unsqueeze(0).expand(T, -1)
    return qmult(qmult(qinv(qt), g), qt)


# ─────────────────────────────────────────────────────────────────────────────
# Cost Function  (Eq. 3)
# ─────────────────────────────────────────────────────────────────────────────

def cost_function(q1T_flat, tau_omega, accel_meas, T):
    q1T = q1T_flat.reshape(T, 4)

    # Build full trajectory  q_0 (fixed) | q_{1:T}
    q0    = torch.tensor([[1., 0., 0., 0.]], dtype=q1T_flat.dtype, device=q1T_flat.device)
    q_all = torch.cat([q0, q1T], dim=0)   # (T+1, 4)

    qt  = q_all[:-1]   # (T, 4): q_0, …, q_{T-1}
    qt1 = q_all[1:]    # (T, 4): q_1, …, q_T  (= q1T)

    # ── Motion model error ──────────────────────────────────────────────────
    f_qt    = motion_model(qt, tau_omega)          # predicted q_{t+1}
    rel_rot = qmult(qinv(qt1), f_qt)              # q_{t+1}^{-1} ⊗ f(q_t,…)
    log_rel = qlog(rel_rot)                        # (T, 4)
    motion_cost = 0.5 * torch.sum(
        torch.linalg.norm(2.0 * log_rel, dim=-1) ** 2
    )

    # ── Observation model error ─────────────────────────────────────────────
    h_qt = observation_model(qt1)                  # (T, 4)
    zeros = torch.zeros(T, 1, dtype=accel_meas.dtype, device=accel_meas.device)
    accel_quat = torch.cat([zeros, accel_meas], dim=-1)   # [0, a_t]
    obs_cost = 0.5 * torch.sum(
        torch.linalg.norm(accel_quat - h_qt, dim=-1) ** 2
    )

    return motion_cost + obs_cost


# ─────────────────────────────────────────────────────────────────────────────
# Simple Gyroscope Integration (open-loop, for calibration verification)
# ─────────────────────────────────────────────────────────────────────────────

def integrate_gyro(omega, ts, q0=None):
    N = omega.shape[1]
    if q0 is None:
        q0 = np.array([1., 0., 0., 0.])

    quats = np.zeros((N, 4))
    quats[0] = q0
    tau = np.diff(ts)   # (N-1,)

    for t in range(N - 1):
        qt        = torch.tensor(quats[t],            dtype=torch.float64).unsqueeze(0)
        tau_omega = torch.tensor(tau[t] * omega[:, t], dtype=torch.float64).unsqueeze(0)
        qt1       = motion_model(qt, tau_omega).squeeze(0)
        qt1       = qt1 / torch.linalg.norm(qt1)
        quats[t + 1] = qt1.detach().numpy()

    return quats   # (N, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Projected Gradient Descent  (main optimiser)
# ─────────────────────────────────────────────────────────────────────────────

def pgd_orientation_tracking(omega, accel, ts, n_iter=300, lr=0.01, verbose=True):
    N   = omega.shape[1]
    T   = N - 1                             # states to optimise: q_1 … q_T
    tau = np.diff(ts)                       # (T,) time steps

    # τ_t · ω_t  for t = 0, …, T-1
    tau_omega_np = tau[:, None] * omega[:, :T].T   # (T, 3)
    # a_t         for t = 1, …, T
    accel_np = accel[:, 1:N].T                     # (T, 3)

    tau_omega_t = torch.tensor(tau_omega_np, dtype=torch.float64)
    accel_t     = torch.tensor(accel_np,     dtype=torch.float64)

    # Initialise q_{1:T} from gyro integration
    q_init = integrate_gyro(omega, ts)   # (N, 4)
    q_opt  = q_init[1:N].copy()          # (T, 4)

    if verbose:
        print(f"  PGD: T={T}, n_iter={n_iter}, lr={lr:.5f}")

    prev_cost = float('inf')

    for it in range(n_iter):
        # Fresh leaf tensor → ensures clean computation graph each iteration
        q_var = torch.tensor(q_opt, dtype=torch.float64, requires_grad=True)

        cost = cost_function(q_var.reshape(-1), tau_omega_t, accel_t, T)
        cost.backward()

        grad      = q_var.grad.detach().numpy()   # (T, 4)
        cost_val  = cost.item()

        # Gradient step (fixed learning rate — adaptive decay causes LR to
        # collapse to ~0 on non-convex landscapes, freezing the optimiser)
        q_opt = q_opt - lr * grad

        # Project: normalise each quaternion
        norms = np.linalg.norm(q_opt, axis=1, keepdims=True)
        q_opt = q_opt / np.maximum(norms, 1e-12)

        # Convergence check
        if abs(cost_val - prev_cost) < 1e-6:
            if verbose:
                print(f"    Converged at iter {it+1}: cost={cost_val:.6f}")
            break
        prev_cost = cost_val

        if verbose and (it + 1) % 50 == 0:
            print(f"    iter {it+1:4d}/{n_iter}: cost={cost_val:.4f}")

    # Prepend fixed q_0 = [1, 0, 0, 0]
    q0    = np.array([[1., 0., 0., 0.]])
    quats = np.vstack([q0, q_opt])   # (N, 4)
    return quats


# ─────────────────────────────────────────────────────────────────────────────
# Euler Angle Conversions
# ─────────────────────────────────────────────────────────────────────────────

def quats_to_euler(quats):
    angles = np.zeros((len(quats), 3))
    for i, q in enumerate(quats):
        n = np.linalg.norm(q)
        if n > 1e-9:
            angles[i] = quat2euler(q / n, axes='sxyz')
    return angles   # roll, pitch, yaw


def rotmats_to_euler(rots):
    angles = np.zeros((len(rots), 3))
    for i, R in enumerate(rots):
        angles[i] = mat2euler(R, axes='sxyz')
    return angles   # roll, pitch, yaw


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_euler_comparison(est_ts, euler_est, vicon_ts, euler_vicon,
                          title, save_path=None):
    labels = ['Roll', 'Pitch', 'Yaw']
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    fig.suptitle(title, fontsize=14)

    t0 = est_ts[0]
    for i, (ax, label) in enumerate(zip(axes, labels)):
        if vicon_ts is not None and euler_vicon is not None:
            ax.plot(vicon_ts - t0, np.degrees(euler_vicon[:, i]),
                    'b-', linewidth=1.5, label='VICON (ground truth)')
        ax.plot(est_ts - t0, np.degrees(euler_est[:, i]),
                'r-', linewidth=1.0, alpha=0.85, label='Estimated')
        ax.set_ylabel(f'{label} (°)', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)', fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline for One Dataset
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(dataset_id, data_dir, n_iter=300, lr=0.01, n_static=200,
                output_dir='results', show_plots=False):
    print(f"\n{'='*60}")
    print(f"  Dataset {dataset_id}")
    print(f"{'='*60}")

    # ── Load ──────────────────────────────────────────────────────────────
    imu_raw, vicon_raw, cam_raw = load_dataset(dataset_id, data_dir)
    imu_ts, accel_raw, gyro_raw = parse_imu(imu_raw)

    has_vicon = vicon_raw is not None
    vicon_rots = vicon_ts = euler_vicon = None

    if has_vicon:
        vicon_rots, vicon_ts = parse_vicon(vicon_raw)
        euler_vicon = rotmats_to_euler(vicon_rots)
        print(f"  IMU samples  : {len(imu_ts)}")
        print(f"  Vicon samples: {len(vicon_ts)}")
    else:
        print(f"  IMU samples  : {len(imu_ts)}")
        print(f"  Vicon        : not available (test set)")

    # ── Calibrate ─────────────────────────────────────────────────────────
    if has_vicon:
        bias_accel, bias_gyro = estimate_bias(
            accel_raw, gyro_raw, vicon_rots, n_static)
    else:
        bias_accel, bias_gyro = estimate_bias_no_vicon(
            accel_raw, gyro_raw, n_static)

    print(f"  Accel bias (ADC): {np.round(bias_accel, 2)}")
    print(f"  Gyro  bias (ADC): {np.round(bias_gyro,  2)}")

    accel, omega = calibrate_imu(accel_raw, gyro_raw, bias_accel, bias_gyro)

    os.makedirs(output_dir, exist_ok=True)

    # ── Gyroscope integration ──────────────────────────────────────────────
    print("\n  [1/2] Gyroscope integration (open-loop)...")
    q_gyro    = integrate_gyro(omega, imu_ts)
    euler_gyro = quats_to_euler(q_gyro)

    fig_gyro = plot_euler_comparison(
        imu_ts, euler_gyro,
        vicon_ts, euler_vicon,
        title=f"Dataset {dataset_id}: Gyroscope Integration (Open-loop)",
        save_path=os.path.join(output_dir, f"dataset{dataset_id}_gyro.png"),
    )
    if show_plots:
        plt.show()
    plt.close(fig_gyro)

    # ── Projected Gradient Descent ─────────────────────────────────────────
    print(f"\n  [2/2] Projected Gradient Descent (n_iter={n_iter}, lr={lr})...")
    q_pgd    = pgd_orientation_tracking(omega, accel, imu_ts,
                                        n_iter=n_iter, lr=lr)
    euler_pgd = quats_to_euler(q_pgd)

    fig_pgd = plot_euler_comparison(
        imu_ts, euler_pgd,
        vicon_ts, euler_vicon,
        title=f"Dataset {dataset_id}: PGD Orientation Tracking",
        save_path=os.path.join(output_dir, f"dataset{dataset_id}_pgd.png"),
    )
    if show_plots:
        plt.show()
    plt.close(fig_pgd)

    # Save quaternion trajectory for panorama construction
    np.save(os.path.join(output_dir, f"dataset{dataset_id}_quats.npy"), q_pgd)
    np.save(os.path.join(output_dir, f"dataset{dataset_id}_ts.npy"),    imu_ts)
    print(f"\n  Quaternion trajectory saved to: {output_dir}/")
    print(f"  Done.")

    return q_pgd, imu_ts


# ─────────────────────────────────────────────────────────────────────────────
# Hard-coded configuration
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(_SCRIPT_DIR, 'results')

DATASET    = 1      # dataset ID to run (train: 1–9 / test: 10–11)
RUN_ALL    = True # set True to run all datasets (1–9 train + 10–11 test)

N_ITER     = 300
LR         = 0.01
N_STATIC   = 500
SHOW_PLOTS = False


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if RUN_ALL:
        for ds in range(1, 10):
            run_dataset(ds, _TRAIN_DIR, N_ITER, LR, N_STATIC, OUTPUT_DIR, SHOW_PLOTS)
        for ds in [10, 11]:
            run_dataset(ds, _TEST_DIR, N_ITER, LR, N_STATIC, OUTPUT_DIR, SHOW_PLOTS)
    else:
        data_dir = _TRAIN_DIR if DATASET <= 9 else _TEST_DIR
        run_dataset(DATASET, data_dir, N_ITER, LR, N_STATIC, OUTPUT_DIR, SHOW_PLOTS)


if __name__ == '__main__':
    main()
