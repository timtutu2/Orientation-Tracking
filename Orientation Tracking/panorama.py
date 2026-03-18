"""
ECE 276A PR1: Panorama Construction

Stitches camera images into a 360° equirectangular panorama by:
  1. Loading orientation estimates q_{1:T} (from orientation_tracking.py)
  2. For each camera frame, finding the closest-past orientation estimate
  3. Projecting every image pixel to the world-frame unit sphere
  4. Mapping the sphere direction to panorama pixel coordinates

Camera model:
  - Optical axis aligned with the body / IMU x-axis
  - Pinhole model: focal length f ≈ 280 px (tunable), principal point (cx, cy)
  - Body frame: x forward (optical), y right in image, z up in image

Panorama:
  - Equirectangular: azimuth φ ∈ [−π, π], elevation θ ∈ [−π/2, π/2]
  - Default size: 720 × 360 pixels

Usage:
  # Run orientation tracking first, then:
  python panorama.py --dataset 1 --data_dir ../../trainset

  # Or provide pre-saved orientations:
  python panorama.py --dataset 1 --quats results/dataset1_quats.npy \
                                  --ts    results/dataset1_ts.npy
"""

import numpy as np
import pickle
import sys
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transforms3d.quaternions import quat2mat

# Local import – orientation_tracking must be on the path
sys.path.insert(0, os.path.dirname(__file__))
from orientation_tracking import (
    read_data, load_dataset, parse_imu,
    calibrate_imu, estimate_bias, estimate_bias_no_vicon,
    parse_vicon,
    pgd_orientation_tracking, integrate_gyro,
)


# ─────────────────────────────────────────────────────────────────────────────
# Camera data loading
# ─────────────────────────────────────────────────────────────────────────────

def parse_cam(cam_raw):
    """
    Extract images and timestamps from camera pickle data.

    Raw format (from IMU reference):  H × W × 3 × K  (240 × 320 × 3 × K)

    Returns
    -------
    images : (K, H, W, 3)  uint8 RGB images
    ts     : (K,)          unix timestamps
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

    # Raw shape is (H, W, 3, K) — transpose to (K, H, W, 3)
    if images.ndim == 4:
        if images.shape[2] == 3 and images.shape[3] != 3:
            # (H, W, 3, K)
            images = np.transpose(images, (3, 0, 1, 2))
        elif images.shape[1] == 3 and images.shape[0] != 3:
            # (K, 3, H, W)
            images = np.transpose(images, (0, 2, 3, 1))
        elif images.shape[0] == 3:
            # (3, H, W, K) — unusual
            images = np.transpose(images, (3, 1, 2, 0))
        # else already (K, H, W, 3)

    return images.astype(np.uint8), ts


# ─────────────────────────────────────────────────────────────────────────────
# Orientation lookup
# ─────────────────────────────────────────────────────────────────────────────

def find_closest_past(query_ts, ref_ts):
    """
    For each query timestamp, return the index of the largest ref timestamp
    that is ≤ query (closest-in-the-past policy).

    Timestamps outside [ref_ts[0], ref_ts[-1]] are clamped.
    """
    idx = np.searchsorted(ref_ts, query_ts, side='right') - 1
    idx = np.clip(idx, 0, len(ref_ts) - 1)
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Panorama construction
# ─────────────────────────────────────────────────────────────────────────────

def build_panorama(images, cam_ts, quats, imu_ts,
                   pano_H=360, pano_W=720,
                   focal_length=280.0):
    """
    Stitch camera images into an equirectangular panorama.

    Parameters
    ----------
    images       : (K, H, W, 3)  camera images
    cam_ts       : (K,)          camera timestamps
    quats        : (N, 4)        orientation quaternions (w,x,y,z)
    imu_ts       : (N,)          IMU/orientation timestamps
    pano_H/pano_W: int           panorama height / width in pixels
    focal_length : float         camera focal length in pixels

    Returns
    -------
    panorama : (pano_H, pano_W, 3) uint8 equirectangular image
    """
    K, H, W, _ = images.shape
    cx, cy = W / 2.0, H / 2.0

    panorama = np.zeros((pano_H, pano_W, 3), dtype=np.uint8)

    # Pre-compute pixel-grid directions in camera/body frame (vectorised)
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))   # (H, W)

    # Camera frame: x = optical axis, y = right in image, z = up in image
    d_x = np.full((H, W), focal_length)
    d_y = (u_grid - cx)          # positive → right
    d_z = (cy - v_grid)          # positive → up

    D = np.stack([d_x, d_y, d_z], axis=-1)             # (H, W, 3)
    D_flat = D.reshape(-1, 3)                           # (H*W, 3)
    D_norm = D_flat / np.linalg.norm(D_flat, axis=1, keepdims=True)  # (H*W, 3)

    print(f"  Building panorama from {K} frames  ({pano_H}×{pano_W})...")

    for k in range(K):
        # Find closest-past orientation estimate
        idx = find_closest_past(cam_ts[k], imu_ts)[()]
        q   = quats[idx]                        # (4,) unit quaternion (w,x,y,z)
        R   = quat2mat(q)                       # (3, 3) body-to-world rotation

        # Rotate body-frame directions to world frame
        D_world = (R @ D_norm.T).T              # (H*W, 3)

        # Spherical coordinates
        phi   = np.arctan2(D_world[:, 1], D_world[:, 0])       # azimuth  [-π, π]
        theta = np.arcsin(np.clip(D_world[:, 2], -1.0, 1.0))   # elevation [-π/2, π/2]

        # Map to panorama pixel indices
        px = ((phi + np.pi) / (2.0 * np.pi) * pano_W).astype(int) % pano_W
        py = ((np.pi / 2.0 - theta) / np.pi * pano_H).astype(int)
        py = np.clip(py, 0, pano_H - 1)

        # Write pixel colours (simple overwrite – no blending)
        img_flat = images[k].reshape(-1, 3)
        panorama[py, px] = img_flat

        if (k + 1) % 100 == 0 or k == K - 1:
            print(f"    Processed {k+1}/{K} frames")

    return panorama


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_panorama(dataset_id, data_dir, quats_path=None, ts_path=None,
                 n_iter=300, lr=0.01, n_static=200,
                 pano_H=360, pano_W=720, focal_length=280.0,
                 output_dir='results'):
    """
    Build a panorama for one dataset.

    If quats_path / ts_path are provided, loads pre-saved orientations.
    Otherwise re-runs the PGD orientation tracker.
    """
    print(f"\n{'='*60}")
    print(f"  Panorama – Dataset {dataset_id}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # ── Load orientation estimates ────────────────────────────────────────
    if quats_path and ts_path and os.path.exists(quats_path) and os.path.exists(ts_path):
        quats  = np.load(quats_path)
        imu_ts = np.load(ts_path)
        print(f"  Loaded orientations from: {quats_path}")
    else:
        print("  Running orientation tracking first...")
        imu_raw, vicon_raw, _ = load_dataset(dataset_id, data_dir)
        imu_ts, accel_raw, gyro_raw = parse_imu(imu_raw)

        if vicon_raw is not None:
            vicon_rots, _ = parse_vicon(vicon_raw)
            bias_accel, bias_gyro = estimate_bias(accel_raw, gyro_raw, vicon_rots, n_static)
        else:
            bias_accel, bias_gyro = estimate_bias_no_vicon(accel_raw, gyro_raw, n_static)

        from orientation_tracking import calibrate_imu
        accel, omega = calibrate_imu(accel_raw, gyro_raw, bias_accel, bias_gyro)
        quats = pgd_orientation_tracking(omega, accel, imu_ts, n_iter=n_iter, lr=lr)
        np.save(os.path.join(output_dir, f"dataset{dataset_id}_quats.npy"), quats)
        np.save(os.path.join(output_dir, f"dataset{dataset_id}_ts.npy"),    imu_ts)

    # ── Load camera data ──────────────────────────────────────────────────
    cam_file = os.path.join(data_dir, 'cam', f'cam{dataset_id}.p')
    if not os.path.exists(cam_file):
        print(f"  No camera data found for dataset {dataset_id}.  Skipping panorama.")
        return None

    print(f"  Loading camera data...")
    cam_raw = read_data(cam_file)
    images, cam_ts = parse_cam(cam_raw)
    print(f"  Camera frames: {len(images)}, image size: {images.shape[1]}×{images.shape[2]}")

    # ── Build panorama ────────────────────────────────────────────────────
    panorama = build_panorama(images, cam_ts, quats, imu_ts,
                              pano_H=pano_H, pano_W=pano_W,
                              focal_length=focal_length)

    # ── Save ──────────────────────────────────────────────────────────────
    save_path = os.path.join(output_dir, f"dataset{dataset_id}_panorama.png")
    plt.figure(figsize=(18, 9))
    plt.imshow(panorama)
    plt.axis('off')
    plt.title(f"Panorama – Dataset {dataset_id}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Panorama saved: {save_path}")

    return panorama


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ECE 276A PR1 – Panorama Construction')
    parser.add_argument('--dataset',      type=int,   default=1,
                        help='Dataset ID (1-9 train, 10-11 test)')
    parser.add_argument('--data_dir',     type=str,   default='../../trainset',
                        help='Path to data directory')
    parser.add_argument('--quats',        type=str,   default=None,
                        help='Path to pre-saved quats .npy (optional)')
    parser.add_argument('--ts',           type=str,   default=None,
                        help='Path to pre-saved timestamps .npy (optional)')
    parser.add_argument('--n_iter',       type=int,   default=300,
                        help='PGD iterations (if re-running tracker)')
    parser.add_argument('--lr',           type=float, default=0.01,
                        help='PGD learning rate')
    parser.add_argument('--n_static',     type=int,   default=200,
                        help='Static samples for bias estimation')
    parser.add_argument('--pano_H',       type=int,   default=360,
                        help='Panorama height in pixels')
    parser.add_argument('--pano_W',       type=int,   default=720,
                        help='Panorama width in pixels')
    parser.add_argument('--focal_length', type=float, default=280.0,
                        help='Camera focal length in pixels')
    parser.add_argument('--output_dir',   type=str,   default='results',
                        help='Output directory')
    args = parser.parse_args()

    run_panorama(
        dataset_id    = args.dataset,
        data_dir      = args.data_dir,
        quats_path    = args.quats,
        ts_path       = args.ts,
        n_iter        = args.n_iter,
        lr            = args.lr,
        n_static      = args.n_static,
        pano_H        = args.pano_H,
        pano_W        = args.pano_W,
        focal_length  = args.focal_length,
        output_dir    = args.output_dir,
    )


if __name__ == '__main__':
    main()
