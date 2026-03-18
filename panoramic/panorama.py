"""
ECE 276A PR1 – Part 2: Panoramic Image Construction
=====================================================

Constructs an equirectangular panoramic image by stitching RGB camera
images (320×240) over time, using the orientation estimates q_{1:T}
from Part 1 (Orientation Tracking).

Algorithm
---------
For each camera frame k:
  1. Find the closest-past orientation estimate (largest IMU timestamp ≤ cam timestamp).
  2. Convert the quaternion to a rotation matrix R (body-to-world).
  3. For each pixel (u, v) in the camera image:
       - Build a unit direction ray in the body/camera frame using a pinhole model.
       - Rotate the ray to world frame:  d_world = R @ d_body
       - Convert (x, y, z) to spherical coordinates (azimuth φ, elevation θ).
       - Map to equirectangular panorama pixel (px, py).
  4. Write the pixel colour (simple overwrite, no blending).

Camera model
------------
  • Optical axis aligned with the IMU / body x-axis.
  • y-axis points right in the image; z-axis points up.
  • Pinhole: focal length f ≈ 280 px (tunable), principal point (cx, cy).
  • Ray direction in body frame: d = [f, u-cx, cy-v] (then normalised).

Panorama layout
---------------
  • Equirectangular projection
  • Azimuth   φ ∈ [-π, π]      → horizontal axis (0 to pano_W)
  • Elevation θ ∈ [-π/2, π/2] → vertical axis   (0 to pano_H, top = +π/2)

Usage
-----
  # Use pre-saved orientations from Orientation Tracking results:
  python panorama.py --dataset 1 \
      --quats "../Orientation Tracking/results/dataset1_quats.npy" \
      --ts    "../Orientation Tracking/results/dataset1_ts.npy"

  # Re-run orientation tracking on-the-fly (requires PyTorch):
  python panorama.py --dataset 1 --data_dir ../../trainset

  # Run all datasets that have camera data (train: 1,2,8,9; test: 10,11):
  python panorama.py --all --data_dir_train ../../trainset --data_dir_test ../../testset

  # Use Vicon ground-truth orientations instead of estimates:
  python panorama.py --dataset 1 --use_vicon
"""

import os
import sys
import pickle
import argparse

import numpy as np
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

def read_data(fname):
    """Load a Python-2/3 compatible pickle file."""
    with open(fname, 'rb') as f:
        if sys.version_info[0] < 3:
            return pickle.load(f)
        return pickle.load(f, encoding='latin1')


def parse_imu(imu_raw):
    """
    Extract timestamps, raw accel, raw gyro from the 7×N IMU array.

    Returns
    -------
    ts        : (N,)   unix timestamps in seconds
    accel_raw : (3, N) raw ADC accelerometer values
    gyro_raw  : (3, N) raw ADC gyroscope values
    """
    data = np.array(imu_raw['vals'] if isinstance(imu_raw, dict) else imu_raw)
    ts        = data[0].flatten()
    accel_raw = data[1:4]
    gyro_raw  = data[4:7]
    return ts, accel_raw, gyro_raw


def parse_vicon(vicon_raw):
    """
    Extract (N, 3, 3) body-to-world rotation matrices and timestamps from Vicon data.

    Returns
    -------
    rots : (N, 3, 3)
    ts   : (N,)
    """
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
    """
    Extract images and timestamps from camera pickle data.

    Raw shape: H × W × 3 × K  (240 × 320 × 3 × K)

    Returns
    -------
    images : (K, H, W, 3) uint8 RGB images
    ts     : (K,)         unix timestamps
    """
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
    """Estimate accel and gyro biases using the initial static period + Vicon R0."""
    n_static   = min(n_static, accel_raw.shape[1])
    bias_gyro  = np.mean(gyro_raw[:, :n_static], axis=1)
    R0         = vicon_rots[0]
    g_body     = R0.T @ np.array([0.0, 0.0, 1.0])
    mean_accel = np.mean(accel_raw[:, :n_static], axis=1)
    bias_accel = mean_accel - g_body / ACCEL_SCALE
    return bias_accel, bias_gyro


def estimate_bias_no_vicon(accel_raw, gyro_raw, n_static=200):
    """Bias estimation without Vicon: assumes device starts level, z-axis up."""
    n_static   = min(n_static, accel_raw.shape[1])
    bias_gyro  = np.mean(gyro_raw[:, :n_static], axis=1)
    mean_accel = np.mean(accel_raw[:, :n_static], axis=1)
    bias_accel = mean_accel - np.array([0.0, 0.0, 1.0]) / ACCEL_SCALE
    return bias_accel, bias_gyro


def calibrate_imu(accel_raw, gyro_raw, bias_accel, bias_gyro):
    """Convert raw ADC to physical units: value = (raw - bias) × scale."""
    accel = (accel_raw - bias_accel[:, None]) * ACCEL_SCALE
    omega = (gyro_raw  - bias_gyro[:, None])  * GYRO_SCALE
    return accel, omega


# ─────────────────────────────────────────────────────────────────────────────
# Orientation Lookup
# ─────────────────────────────────────────────────────────────────────────────

def find_closest_past(query_ts, ref_ts):
    """
    For each query timestamp find the index of the closest-past reference
    timestamp (largest ref_ts[i] ≤ query_ts).  Clamps to valid range.

    Parameters
    ----------
    query_ts : scalar or array-like
    ref_ts   : sorted 1-D array

    Returns
    -------
    idx : int or (M,) int array
    """
    idx = np.searchsorted(ref_ts, query_ts, side='right') - 1
    idx = np.clip(idx, 0, len(ref_ts) - 1)
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Panorama Construction
# ─────────────────────────────────────────────────────────────────────────────

def build_panorama(images, cam_ts, quats, orient_ts,
                   pano_H=360, pano_W=720,
                   focal_length=280.0):
    """
    Stitch camera frames into an equirectangular panoramic image.

    For each camera frame the function:
      1. Looks up the closest-past orientation quaternion.
      2. Converts the quaternion to a body-to-world rotation matrix R.
      3. Builds a direction ray for each pixel using the pinhole camera model
         (optical axis = body x-axis, right = body y-axis, up = body z-axis).
      4. Rotates rays to world frame and maps to equirectangular coordinates.
      5. Writes pixel colours (overwrite; no blending needed per spec).

    Parameters
    ----------
    images       : (K, H, W, 3) uint8  camera frames
    cam_ts       : (K,)         float  camera timestamps (seconds)
    quats        : (N, 4)       float  orientation quaternions (w, x, y, z)
    orient_ts    : (N,)         float  timestamps matching quats
    pano_H       : int                 output panorama height  (default 360)
    pano_W       : int                 output panorama width   (default 720)
    focal_length : float               pinhole focal length in pixels (default 280)

    Returns
    -------
    panorama : (pano_H, pano_W, 3) uint8  equirectangular panorama
    """
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
    """
    Same as build_panorama but accepts Vicon rotation matrices directly
    (ground-truth orientations) rather than estimated quaternions.

    Parameters
    ----------
    images      : (K, H, W, 3) uint8
    cam_ts      : (K,) float
    vicon_rots  : (N, 3, 3) float  body-to-world rotation matrices
    vicon_ts    : (N,) float
    """
    # Convert rotation matrices to quaternions so we can reuse build_panorama
    quats = np.array([mat2quat(R) for R in vicon_rots])  # (N, 4) w,x,y,z
    return build_panorama(images, cam_ts, quats, vicon_ts,
                          pano_H=pano_H, pano_W=pano_W, focal_length=focal_length)


# ─────────────────────────────────────────────────────────────────────────────
# Orientation Tracking  (thin wrapper — imports the sibling module if available)
# ─────────────────────────────────────────────────────────────────────────────

def run_orientation_tracking(dataset_id, data_dir, n_iter=300, lr=0.01, n_static=200):
    """
    Run PGD orientation tracking for one dataset.

    Tries to import orientation_tracking.py from the sibling folder
    '../Orientation Tracking/'.  Falls back to basic gyroscope integration
    if that import fails (e.g. PyTorch not installed).

    Returns
    -------
    quats    : (N, 4) float  quaternion trajectory (w,x,y,z)
    imu_ts   : (N,)   float  IMU timestamps
    """
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
    """
    Fallback: open-loop gyroscope integration (no PyTorch required).
    Less accurate than PGD but still produces a usable panorama.
    """
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
    """
    Build and save a panoramic image for one dataset.

    Orientation source priority (highest to lowest):
      1. Pre-saved .npy files  (--quats / --ts flags)
      2. Vicon ground-truth    (--use_vicon flag)
      3. On-the-fly PGD        (falls back to gyro integration if PyTorch absent)

    Parameters
    ----------
    dataset_id   : int    dataset number (1–9 train, 10–11 test)
    data_dir     : str    root folder containing imu/, cam/, vicon/ sub-folders
    quats_path   : str    path to pre-saved quaternions .npy  (optional)
    ts_path      : str    path to pre-saved IMU timestamps .npy  (optional)
    use_vicon    : bool   use Vicon ground-truth orientations
    n_iter       : int    PGD iterations (if running tracker)
    lr           : float  PGD learning rate
    n_static     : int    static samples for bias estimation
    pano_H/W     : int    panorama dimensions in pixels
    focal_length : float  camera focal length in pixels
    output_dir   : str    directory to save the output PNG

    Returns
    -------
    panorama : (pano_H, pano_W, 3) uint8  or None if no camera data
    """
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
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='ECE 276A PR1 – Panoramic Image Construction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset / data location
    parser.add_argument('--dataset',    type=int,   default=1,
                        help='Dataset ID  (default: 1)')
    parser.add_argument('--data_dir',   type=str,   default='../../trainset',
                        help='Root data folder containing imu/, cam/, vicon/ '
                             '(default: ../../trainset)')
    parser.add_argument('--all',        action='store_true',
                        help='Run all datasets that have camera data '
                             '(train 1,2,8,9 + test 10,11).  '
                             'Use --data_dir_train and --data_dir_test.')
    parser.add_argument('--data_dir_train', type=str, default='../../trainset',
                        help='Training data root (used with --all)')
    parser.add_argument('--data_dir_test',  type=str, default='../../testset',
                        help='Test data root (used with --all)')

    # Orientation source
    orient = parser.add_mutually_exclusive_group()
    orient.add_argument('--quats',      type=str,   default=None,
                        help='Path to pre-saved quaternions .npy')
    orient.add_argument('--use_vicon',  action='store_true',
                        help='Use Vicon ground-truth orientations')
    parser.add_argument('--ts',         type=str,   default=None,
                        help='Path to pre-saved IMU timestamps .npy  '
                             '(required with --quats)')

    # Tracking hyper-parameters (ignored when --quats or --use_vicon)
    parser.add_argument('--n_iter',     type=int,   default=300,
                        help='PGD iterations  (default: 300)')
    parser.add_argument('--lr',         type=float, default=0.01,
                        help='PGD learning rate  (default: 0.01)')
    parser.add_argument('--n_static',   type=int,   default=200,
                        help='Static samples for bias estimation  (default: 200)')

    # Panorama settings
    parser.add_argument('--pano_H',       type=int,   default=360,
                        help='Panorama height in pixels  (default: 360)')
    parser.add_argument('--pano_W',       type=int,   default=720,
                        help='Panorama width in pixels  (default: 720)')
    parser.add_argument('--focal_length', type=float, default=280.0,
                        help='Camera focal length in pixels  (default: 280)')

    # Output
    parser.add_argument('--output_dir',   type=str,   default='results',
                        help='Output directory  (default: results/)')

    args = parser.parse_args()

    if args.all:
        train_datasets = [1, 2, 8, 9]
        test_datasets  = [10, 11]
        print("Running all datasets with camera data…")
        for ds in train_datasets:
            # Auto-discover pre-saved quats in sibling results folder
            sibling_results = os.path.join(
                os.path.dirname(__file__),
                '..', 'Orientation Tracking', 'results')
            q_path = os.path.join(sibling_results, f'dataset{ds}_quats.npy')
            t_path = os.path.join(sibling_results, f'dataset{ds}_ts.npy')
            if not (os.path.exists(q_path) and os.path.exists(t_path)):
                q_path = t_path = None
            run_panorama(
                dataset_id   = ds,
                data_dir     = args.data_dir_train,
                quats_path   = q_path,
                ts_path      = t_path,
                use_vicon    = args.use_vicon,
                n_iter       = args.n_iter,
                lr           = args.lr,
                n_static     = args.n_static,
                pano_H       = args.pano_H,
                pano_W       = args.pano_W,
                focal_length = args.focal_length,
                output_dir   = args.output_dir,
            )
        for ds in test_datasets:
            sibling_results = os.path.join(
                os.path.dirname(__file__),
                '..', 'Orientation Tracking', 'results')
            q_path = os.path.join(sibling_results, f'dataset{ds}_quats.npy')
            t_path = os.path.join(sibling_results, f'dataset{ds}_ts.npy')
            if not (os.path.exists(q_path) and os.path.exists(t_path)):
                q_path = t_path = None
            run_panorama(
                dataset_id   = ds,
                data_dir     = args.data_dir_test,
                quats_path   = q_path,
                ts_path      = t_path,
                use_vicon    = False,           # no Vicon in test set
                n_iter       = args.n_iter,
                lr           = args.lr,
                n_static     = args.n_static,
                pano_H       = args.pano_H,
                pano_W       = args.pano_W,
                focal_length = args.focal_length,
                output_dir   = args.output_dir,
            )
    else:
        run_panorama(
            dataset_id   = args.dataset,
            data_dir     = args.data_dir,
            quats_path   = args.quats,
            ts_path      = args.ts,
            use_vicon    = args.use_vicon,
            n_iter       = args.n_iter,
            lr           = args.lr,
            n_static     = args.n_static,
            pano_H       = args.pano_H,
            pano_W       = args.pano_W,
            focal_length = args.focal_length,
            output_dir   = args.output_dir,
        )


if __name__ == '__main__':
    main()
