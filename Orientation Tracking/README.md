# ECE 276A PR1 — Orientation Tracking & Panorama

## Overview

Two main scripts implement the full pipeline for 3-D orientation tracking from
IMU data and panoramic image stitching.

| File | Description |
|------|-------------|
| `orientation_tracking.py` | IMU calibration, gyro integration, PGD optimizer |
| `panorama.py` | Panorama construction from orientation estimates |

---

## orientation_tracking.py

### Algorithm

1. **IMU Calibration** — bias estimation from the initial static period using the
   first Vicon rotation to determine the expected gravity direction in body frame.
2. **Gyroscope Integration** (open-loop verification) — integrates `q_{t+1} = q_t ⊗ exp([0, τω/2])`.
3. **Projected Gradient Descent** — minimises the combined cost (Eq. 3):

```
c(q_{1:T}) = ½ Σ ‖2 log(q_{t+1}⁻¹ ⊗ f(qt, τtωt))‖²
           + ½ Σ ‖[0, at] − h(qt)‖²
```

   After each gradient step, each quaternion is normalised (`q / ‖q‖`) to
   enforce the unit-norm constraint.

Gradients are computed with `torch.autograd` (`.backward()`).

### Usage

```bash
# Single training dataset
python orientation_tracking.py --dataset 1 --data_dir ../../trainset

# All training datasets
python orientation_tracking.py --all --data_dir ../../trainset

# Test dataset (no Vicon)
python orientation_tracking.py --dataset 10 --data_dir ../../testset
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `1` | Dataset ID (1–9 train, 10–11 test) |
| `--data_dir` | `../../trainset` | Path to data directory |
| `--n_iter` | `300` | PGD iterations |
| `--lr` | `0.01` | Initial step size |
| `--n_static` | `200` | Samples used for bias estimation |
| `--output_dir` | `results` | Where to save plots and `.npy` files |
| `--show` | off | Show interactive matplotlib windows |
| `--all` | off | Run all training datasets 1–9 |

### Outputs (per dataset, in `output_dir/`)

- `dataset{N}_gyro.png` — Roll/pitch/yaw from open-loop gyro integration vs Vicon
- `dataset{N}_pgd.png` — Roll/pitch/yaw from PGD vs Vicon
- `dataset{N}_quats.npy` — Optimised quaternion trajectory `(N, 4)`
- `dataset{N}_ts.npy` — IMU timestamps `(N,)`

---

## panorama.py

Stitches camera images into a 360° equirectangular panorama.

### Algorithm

For each camera frame (timestamp `t_cam`):
1. Look up the closest-past orientation estimate by IMU timestamp.
2. Compute the body-to-world rotation matrix `R = quat2mat(q)`.
3. For every image pixel `(u, v)`, compute the camera-frame ray
   `d = normalize([f, u−cx, cy−v])` (x = optical axis, y right, z up).
4. Rotate to world frame: `d_world = R @ d`.
5. Map to spherical coordinates `(φ, θ)` and project to panorama pixel.

### Usage

```bash
# Use pre-saved orientations (fast)
python panorama.py --dataset 1 --data_dir ../../trainset \
    --quats results/dataset1_quats.npy --ts results/dataset1_ts.npy

# Re-run orientation tracking internally (slower)
python panorama.py --dataset 1 --data_dir ../../trainset
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--focal_length` | `280.0` | Camera focal length in pixels |
| `--pano_H` / `--pano_W` | `360` / `720` | Panorama dimensions |

---

## Sensor Constants

| Quantity | Value | Source |
|----------|-------|--------|
| ADC reference voltage | 3300 mV | IMU reference |
| Accel sensitivity | 330 mV/g | ADXL335 datasheet at 3.3 V |
| Gyro sensitivity | 3.33 mV/(°/s) | LPR530AL / LY530ALH 4× output |
| Accel scale | 0.00977 g/count | Vref / 1023 / 330 |
| Gyro scale | 0.000570 rad/s/count | Vref / 1023 / 3.33 × π/180 |

## Dependencies

```
numpy
torch
transforms3d
matplotlib
scipy  (for panorama.py only — interp1d)
```
