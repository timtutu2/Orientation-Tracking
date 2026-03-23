import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from load_data import read_data

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT  = os.path.join(_SCRIPT_DIR, '..', '..', 'data')
_TRAIN_DIR  = os.path.join(_DATA_ROOT, 'trainset')
_TEST_DIR   = os.path.join(_DATA_ROOT, 'testset')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transforms3d.quaternions import quat2mat, mat2quat


# ─────────────────────────────────────────────────────────────────────────────
# IMU Sensor Constants  (copy from orientation_tracking.py to stay standalone)
# ─────────────────────────────────────────────────────────────────────────────

VREF               = 3300.0   # mV
ACCEL_SENSITIVITY  = 330.0    # mV / g
GYRO_SENSITIVITY   = 3.33     # mV / (deg/s)  [4× output]

ACCEL_SCALE = VREF / 1023.0 / ACCEL_SENSITIVITY               # g / ADC count
GYRO_SCALE  = VREF / 1023.0 / GYRO_SENSITIVITY * (np.pi / 180.0)  # rad/s / ADC count


# ─────────────────────────────────────────────────────────────────────────────
# Data I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_imu(imu_raw):
    data = np.array(imu_raw['vals'] if isinstance(imu_raw, dict) else imu_raw)
    ts        = data[0].flatten()
    accel_raw = data[1:4]
    gyro_raw  = data[4:7]
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

    # Normalise to (N, 3, 3); raw shape is usually (3, 3, N)
    if rots.ndim == 3 and rots.shape[0] == 3 and rots.shape[1] == 3:
        rots = np.transpose(rots, (2, 0, 1))
    return rots, ts


def parse_cam(cam_raw):
    if isinstance(cam_raw, dict):
        images = np.array(cam_raw['cam'])
        ts     = np.array(cam_raw['ts']).flatten()
    elif isinstance(cam_raw, (list, tuple)) and len(cam_raw) == 2:
        images = np.array(cam_raw[0])
        ts     = np.array(cam_raw[1]).flatten()
    else:
        images = np.array(cam_raw)
        ts     = np.arange(images.shape[-1], dtype=float)

    # Bring to (K, H, W, 3) regardless of raw layout
    if images.ndim == 4:
        if images.shape[2] == 3 and images.shape[3] != 3:
            images = np.transpose(images, (3, 0, 1, 2))   # (H,W,3,K) → (K,H,W,3)
        elif images.shape[1] == 3 and images.shape[0] != 3:
            images = np.transpose(images, (0, 2, 3, 1))   # (K,3,H,W) → (K,H,W,3)
        elif images.shape[0] == 3:
            images = np.transpose(images, (3, 1, 2, 0))   # (3,H,W,K) → (K,H,W,3)

    return images.astype(np.uint8), ts


# ─────────────────────────────────────────────────────────────────────────────
# IMU Calibration (needed when re-running orientation tracking)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_bias(accel_raw, gyro_raw, vicon_rots, n_static=200):
    n_static   = min(n_static, accel_raw.shape[1])
    bias_gyro  = np.mean(gyro_raw[:, :n_static], axis=1)
    R0         = vicon_rots[0]
    g_body     = R0.T @ np.array([0.0, 0.0, 1.0])
    mean_accel = np.mean(accel_raw[:, :n_static], axis=1)
    bias_accel = mean_accel - g_body / ACCEL_SCALE
    return bias_accel, bias_gyro


def estimate_bias_no_vicon(accel_raw, gyro_raw, n_static=200):
    n_static   = min(n_static, accel_raw.shape[1])
    bias_gyro  = np.mean(gyro_raw[:, :n_static], axis=1)
    mean_accel = np.mean(accel_raw[:, :n_static], axis=1)
    bias_accel = mean_accel - np.array([0.0, 0.0, 1.0]) / ACCEL_SCALE
    return bias_accel, bias_gyro


def calibrate_imu(accel_raw, gyro_raw, bias_accel, bias_gyro):
    accel = (accel_raw - bias_accel[:, None]) * ACCEL_SCALE
    omega = (gyro_raw  - bias_gyro[:, None])  * GYRO_SCALE
    return accel, omega


# ─────────────────────────────────────────────────────────────────────────────
# Orientation Lookup
# ─────────────────────────────────────────────────────────────────────────────

def find_closest_past(query_ts, ref_ts):
    idx = np.searchsorted(ref_ts, query_ts, side='right') - 1
    idx = np.clip(idx, 0, len(ref_ts) - 1)
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Panorama Construction
# ─────────────────────────────────────────────────────────────────────────────

def build_panorama(images, cam_ts, quats, orient_ts,
                   pano_H=360, pano_W=720,
                   focal_length=280.0):
    K, H, W, _ = images.shape
    cx, cy = W / 2.0, H / 2.0

    panorama = np.zeros((pano_H, pano_W, 3), dtype=np.uint8)

    # ── Pre-compute normalised pixel rays in camera/body frame (vectorised) ──
    # Pixel grid: u = column index (0…W-1), v = row index (0…H-1)
    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float64),
                                 np.arange(H, dtype=np.float64))  # (H, W)

    # Body-frame ray components (before normalisation):
    #   x-axis = optical axis      → always focal_length
    #   y-axis = right in image    → u - cx
    #   z-axis = up in image       → cy - v
    dx = np.full((H, W), focal_length)
    dy = u_grid - cx
    dz = cy - v_grid

    D      = np.stack([dx, dy, dz], axis=-1)         # (H, W, 3)
    D_flat = D.reshape(-1, 3)                         # (H*W, 3)
    norms  = np.linalg.norm(D_flat, axis=1, keepdims=True)
    D_norm = D_flat / norms                           # (H*W, 3) unit rays

    print(f"  Building panorama: {K} frames → {pano_H}×{pano_W} output")

    for k in range(K):
        # ── 1. Look up orientation ───────────────────────────────────────
        idx = int(find_closest_past(cam_ts[k], orient_ts))
        q   = quats[idx]               # (4,) = (w, x, y, z)
        R   = quat2mat(q)              # (3, 3) body-to-world

        # ── 2. Rotate rays to world frame ────────────────────────────────
        D_world = (R @ D_norm.T).T     # (H*W, 3)

        # ── 3. Spherical coordinates ─────────────────────────────────────
        phi   = np.arctan2(D_world[:, 1], D_world[:, 0])          # azimuth [-π, π]
        theta = np.arcsin(np.clip(D_world[:, 2], -1.0, 1.0))      # elevation [-π/2, π/2]

        # ── 4. Equirectangular mapping ────────────────────────────────────
        # phi=−π → px=0; phi=+π → px=pano_W
        # theta=+π/2 (zenith) → py=0; theta=−π/2 (nadir) → py=pano_H
        px = ((phi + np.pi) / (2.0 * np.pi) * pano_W).astype(int) % pano_W
        py = ((np.pi / 2.0 - theta) / np.pi * pano_H).astype(int)
        py = np.clip(py, 0, pano_H - 1)

        # ── 5. Write pixel colours ────────────────────────────────────────
        img_flat = images[k].reshape(-1, 3)
        panorama[py, px] = img_flat

        if (k + 1) % 200 == 0 or k == K - 1:
            print(f"    {k+1:4d}/{K} frames processed")

    return panorama


def build_panorama_from_vicon(images, cam_ts, vicon_rots, vicon_ts,
                               pano_H=360, pano_W=720, focal_length=280.0):
    # Convert rotation matrices to quaternions so we can reuse build_panorama
    quats = np.array([mat2quat(R) for R in vicon_rots])  # (N, 4) w,x,y,z
    return build_panorama(images, cam_ts, quats, vicon_ts,
                          pano_H=pano_H, pano_W=pano_W, focal_length=focal_length)


# ─────────────────────────────────────────────────────────────────────────────
# Orientation Tracking  (thin wrapper — imports the sibling module if available)
# ─────────────────────────────────────────────────────────────────────────────

def run_orientation_tracking(dataset_id, data_dir, n_iter=300, lr=0.01, n_static=200):
    sibling = os.path.join(os.path.dirname(__file__), '..', 'Orientation Tracking')
    if os.path.isdir(sibling):
        sys.path.insert(0, os.path.abspath(sibling))

    try:
        from orientation_tracking import (
            load_dataset, parse_imu as _parse_imu, parse_vicon as _parse_vicon,
            calibrate_imu as _calibrate_imu,
            estimate_bias as _estimate_bias,
            estimate_bias_no_vicon as _estimate_bias_no_vicon,
            pgd_orientation_tracking,
        )
        print("  Using orientation_tracking.py (PGD)")
        imu_raw, vicon_raw, _ = load_dataset(dataset_id, data_dir)
        imu_ts, accel_raw, gyro_raw = _parse_imu(imu_raw)
        if vicon_raw is not None:
            vicon_rots, _ = _parse_vicon(vicon_raw)
            bias_a, bias_g = _estimate_bias(accel_raw, gyro_raw, vicon_rots, n_static)
        else:
            bias_a, bias_g = _estimate_bias_no_vicon(accel_raw, gyro_raw, n_static)
        accel, omega = _calibrate_imu(accel_raw, gyro_raw, bias_a, bias_g)
        quats = pgd_orientation_tracking(omega, accel, imu_ts, n_iter=n_iter, lr=lr)
        return quats, imu_ts

    except ImportError:
        print("  WARNING: orientation_tracking not available; using gyro integration only.")
        return _gyro_only_tracking(dataset_id, data_dir, n_static)


def _gyro_only_tracking(dataset_id, data_dir, n_static=200):
    imu_file  = os.path.join(data_dir, 'imu',   f'imuRaw{dataset_id}.p')
    vicon_file = os.path.join(data_dir, 'vicon', f'viconRot{dataset_id}.p')

    imu_raw = read_data(imu_file)
    imu_ts, accel_raw, gyro_raw = parse_imu(imu_raw)

    vicon_rots = None
    if os.path.exists(vicon_file):
        vicon_raw  = read_data(vicon_file)
        vicon_rots, _ = parse_vicon(vicon_raw)
        bias_a, bias_g = estimate_bias(accel_raw, gyro_raw, vicon_rots, n_static)
    else:
        bias_a, bias_g = estimate_bias_no_vicon(accel_raw, gyro_raw, n_static)

    _, omega = calibrate_imu(accel_raw, gyro_raw, bias_a, bias_g)

    N     = omega.shape[1]
    quats = np.zeros((N, 4))
    quats[0] = [1., 0., 0., 0.]
    tau   = np.diff(imu_ts)

    for t in range(N - 1):
        q   = quats[t]
        w   = omega[:, t] * tau[t]
        ang = np.linalg.norm(w) / 2.0
        if ang < 1e-10:
            dq = np.array([1., 0., 0., 0.])
        else:
            ax = w / (2.0 * ang)
            dq = np.array([np.cos(ang), *(np.sin(ang) * ax)])

        # Hamilton product q ⊗ dq
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = dq
        qt1 = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
        quats[t + 1] = qt1 / np.linalg.norm(qt1)

    return quats, imu_ts


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline for One Dataset
# ─────────────────────────────────────────────────────────────────────────────

def run_panorama(dataset_id, data_dir,
                 quats_path=None, ts_path=None,
                 use_vicon=False,
                 n_iter=300, lr=0.01, n_static=200,
                 pano_H=360, pano_W=720,
                 focal_length=280.0,
                 output_dir='results'):
    print(f"\n{'='*62}")
    print(f"  Panorama  –  Dataset {dataset_id}")
    print(f"{'='*62}")

    os.makedirs(output_dir, exist_ok=True)

    # ── Check camera data availability ────────────────────────────────────
    cam_file = os.path.join(data_dir, 'cam', f'cam{dataset_id}.p')
    if not os.path.exists(cam_file):
        print(f"  No camera data found ({cam_file}).  Skipping.")
        return None

    print(f"  Loading camera data from: {cam_file}")
    cam_raw = read_data(cam_file)
    images, cam_ts = parse_cam(cam_raw)
    K, H, W, _ = images.shape
    print(f"  Camera: {K} frames, each {H}×{W} px")

    # ── Load / compute orientations ───────────────────────────────────────
    if quats_path and ts_path and os.path.exists(quats_path) and os.path.exists(ts_path):
        # ── Option 1: pre-saved quaternions ──────────────────────────────
        quats    = np.load(quats_path)
        orient_ts = np.load(ts_path)
        print(f"  Orientations: loaded from {quats_path}")

        panorama = build_panorama(images, cam_ts, quats, orient_ts,
                                   pano_H=pano_H, pano_W=pano_W,
                                   focal_length=focal_length)

    elif use_vicon:
        # ── Option 2: Vicon ground-truth ──────────────────────────────────
        vicon_file = os.path.join(data_dir, 'vicon', f'viconRot{dataset_id}.p')
        if not os.path.exists(vicon_file):
            print(f"  Vicon file not found ({vicon_file}).  Falling back to tracking.")
            quats, orient_ts = run_orientation_tracking(
                dataset_id, data_dir, n_iter, lr, n_static)
            panorama = build_panorama(images, cam_ts, quats, orient_ts,
                                       pano_H=pano_H, pano_W=pano_W,
                                       focal_length=focal_length)
        else:
            vicon_raw = read_data(vicon_file)
            vicon_rots, vicon_ts = parse_vicon(vicon_raw)
            print(f"  Orientations: Vicon ground-truth ({len(vicon_ts)} samples)")
            panorama = build_panorama_from_vicon(
                images, cam_ts, vicon_rots, vicon_ts,
                pano_H=pano_H, pano_W=pano_W, focal_length=focal_length)
            # Also save vicon quats for reference
            quats     = np.array([mat2quat(R) for R in vicon_rots])
            orient_ts = vicon_ts

    else:
        # ── Option 3: run PGD orientation tracking on-the-fly ────────────
        print("  Orientations: running orientation tracking (PGD)…")
        quats, orient_ts = run_orientation_tracking(
            dataset_id, data_dir, n_iter, lr, n_static)
        np.save(os.path.join(output_dir, f'dataset{dataset_id}_quats.npy'), quats)
        np.save(os.path.join(output_dir, f'dataset{dataset_id}_ts.npy'),    orient_ts)
        print(f"  Quaternions saved to {output_dir}/")

        panorama = build_panorama(images, cam_ts, quats, orient_ts,
                                   pano_H=pano_H, pano_W=pano_W,
                                   focal_length=focal_length)

    # ── Save panorama ─────────────────────────────────────────────────────
    save_path = os.path.join(output_dir, f'dataset{dataset_id}_panorama.png')
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.imshow(panorama)
    ax.axis('off')
    ax.set_title(f'Panorama – Dataset {dataset_id}   ({pano_H}×{pano_W})',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Panorama saved: {save_path}")

    return panorama


# ─────────────────────────────────────────────────────────────────────────────
# Hard-coded configuration
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR   = os.path.join(_SCRIPT_DIR, 'results')
OT_RESULTS   = os.path.join(_SCRIPT_DIR, '..', 'Orientation Tracking', 'results')

DATASET      = 1          # dataset ID to run (train: 1,2,8,9 / test: 10,11)
RUN_ALL      = True      # set True to run all datasets with camera data
USE_VICON    = False      # set True to use Vicon ground-truth orientations

PANO_H       = 960
PANO_W       = 1920
FOCAL_LENGTH = 280.0
N_ITER       = 300
LR           = 0.01
N_STATIC     = 500


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def _quats_for(ds):
    q = os.path.join(OT_RESULTS, f'dataset{ds}_quats.npy')
    t = os.path.join(OT_RESULTS, f'dataset{ds}_ts.npy')
    return (q, t) if (os.path.exists(q) and os.path.exists(t)) else (None, None)


def main():
    if RUN_ALL:
        for ds in [1, 2, 8, 9]:
            q_path, t_path = _quats_for(ds)
            run_panorama(ds, _TRAIN_DIR, q_path, t_path, USE_VICON,
                         N_ITER, LR, N_STATIC, PANO_H, PANO_W, FOCAL_LENGTH, OUTPUT_DIR)
        for ds in [10, 11]:
            q_path, t_path = _quats_for(ds)
            run_panorama(ds, _TEST_DIR, q_path, t_path, False,
                         N_ITER, LR, N_STATIC, PANO_H, PANO_W, FOCAL_LENGTH, OUTPUT_DIR)
    else:
        data_dir = _TRAIN_DIR if DATASET <= 9 else _TEST_DIR
        q_path, t_path = _quats_for(DATASET)
        run_panorama(DATASET, data_dir, q_path, t_path, USE_VICON,
                     N_ITER, LR, N_STATIC, PANO_H, PANO_W, FOCAL_LENGTH, OUTPUT_DIR)


if __name__ == '__main__':
    main()
