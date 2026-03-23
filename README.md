# ECE 276A PR1 – Orientation Tracking & Panoramic Image Construction

**UCSD ECE 276A: Sensing & Estimation in Robotics**

This project implements IMU-based 3D orientation tracking using Projected Gradient Descent (PGD) over a quaternion trajectory, followed by panoramic image construction via equirectangular projection of camera frames.

---

## Overview

The pipeline has two main stages:

1. **Orientation Tracking** — Estimate the 6-DoF orientation of an IMU over time by minimising a joint motion + observation cost over the quaternion trajectory `q_{1:T}`.
2. **Panoramic Image Construction** — Use the estimated orientations to stitch RGB camera frames into a full 360° equirectangular panorama.

---

## Project Structure

```
PR1/
├── code/
│   ├── load_data.py                        # Pickle data loader utility
│   ├── rotplot.py                          # 3D rotation matrix visualiser
│   ├── Orientation Tracking/
│   │   ├── orientation_tracking.py         # IMU calibration + PGD tracker
│   │   └── results/                        # Output plots and .npy trajectories
│   │       ├── dataset{N}_gyro.png         # Open-loop gyro integration result
│   │       ├── dataset{N}_pgd.png          # PGD optimised orientation result
│   │       ├── dataset{N}_quats.npy        # Saved quaternion trajectory (N, 4)
│   │       └── dataset{N}_ts.npy           # Saved IMU timestamps (N,)
│   └── panoramic/
│       ├── panorama.py                     # Equirectangular panorama builder
│       └── results/                        # Output panorama images
│           └── dataset{N}_panorama.png
├── data/
│   ├── trainset/
│   │   ├── imu/        imuRaw{1-9}.p       # Raw IMU ADC data (7 × N)
│   │   ├── cam/        cam{1,2,8,9}.p      # RGB camera frames (240×320×3×K)
│   │   └── vicon/      viconRot{1-9}.p     # Vicon ground-truth rotation matrices
│   └── testset/
│       ├── imu/        imuRaw{10,11}.p
│       └── cam/        cam{10,11}.p
└── docs/                                   # Sensor datasheets and reference material
```

> **Data**: The train/test datasets are not included in this repository.  
> Download from: https://ucsdcloud-my.sharepoint.com/:f:/g/personal/natanasov_ucsd_edu/IgBek3pBUhg4Rr-TP7oMZXikARvskUMNOCnORDml2BNz878?e=e3WLBo

---

## Dependencies

```bash
pip install numpy torch matplotlib transforms3d
```

| Package | Purpose |
|---|---|
| `numpy` | Array math and data handling |
| `torch` | Automatic differentiation for PGD |
| `matplotlib` | Euler angle plots and panorama output |
| `transforms3d` | Quaternion / rotation matrix conversions |

---

## Hardware / Sensor Details

The IMU consists of:
- **ADXL335** accelerometer — sensitivity 330 mV/g at Vref = 3.3 V
- **LPR530AL** (pitch/roll) + **LY530ALH** (yaw) gyroscopes — 4× amplified output, sensitivity 3.33 mV/(deg/s)

Raw 10-bit ADC values are converted to physical units via:

```
accel [g]      = (raw_adc − bias) × Vref / 1023 / 330
angular_vel [rad/s] = (raw_adc − bias) × Vref / 1023 / 3.33 × π/180
```

---

## Part 1 – Orientation Tracking

### Method

The quaternion convention used is **(w, x, y, z)** (scalar-first).

**Motion model** (quaternion kinematics):
```
f(q_t, τ_t ω_t) = q_t ⊗ exp([0, τ_t ω_t / 2])
```

**Observation model** (gravity in body frame):
```
h(q_t) = q_t^{-1} ⊗ [0, 0, 0, 1] ⊗ q_t
```

**Cost function** (Eq. 3 from spec):
```
c(q_{1:T}) = ½ Σ ‖2 log(q_{t+1}^{-1} ⊗ f(q_t, τ_t ω_t))‖²
           + ½ Σ ‖[0, a_t] − h(q_t)‖²
```

Minimised using **Projected Gradient Descent**: after each gradient step, each quaternion is re-normalised (projected onto the unit hypersphere H*). Gradients are computed via PyTorch autograd. The trajectory is initialised from open-loop gyroscope integration.

**Bias estimation** uses the first `n_static` IMU samples (assumed stationary), with the Vicon initial rotation to resolve the gravity direction for the accelerometer.

### Running

Edit the configuration block at the bottom of `orientation_tracking.py` and run:

```bash
cd "code/Orientation Tracking"
python orientation_tracking.py
```

Key configuration variables:

| Variable | Default | Description |
|---|---|---|
| `RUN_ALL` | `True` | Run all 11 datasets (train 1–9, test 10–11) |
| `DATASET` | `1` | Single dataset to run when `RUN_ALL=False` |
| `N_ITER` | `300` | PGD iterations |
| `LR` | `0.01` | Gradient descent step size |
| `N_STATIC` | `500` | Static samples for bias estimation |

Output is saved to `Orientation Tracking/results/`:
- `dataset{N}_gyro.png` — open-loop gyro integration vs. Vicon (roll/pitch/yaw)
- `dataset{N}_pgd.png` — PGD optimised estimates vs. Vicon
- `dataset{N}_quats.npy` — quaternion trajectory for use by panorama builder
- `dataset{N}_ts.npy` — corresponding IMU timestamps

---

## Part 2 – Panoramic Image Construction

### Method

For each camera frame (320×240 RGB):
1. Find the closest-past orientation estimate (largest IMU timestamp ≤ camera timestamp).
2. Convert the quaternion to a body-to-world rotation matrix **R**.
3. For each pixel `(u, v)`, build a unit ray in the body frame using a pinhole model:
   - Optical axis = body x-axis; right = y-axis; up = z-axis
   - `d_body = [f, u − cx, cy − v]` (normalised), with focal length f ≈ 280 px
4. Rotate rays to world frame: `d_world = R @ d_body`
5. Map to equirectangular coordinates:
   - Azimuth φ = atan2(y, x) → horizontal axis
   - Elevation θ = arcsin(z) → vertical axis
6. Write pixel colour (last-write wins, no blending).

Output is a 960×1920 (or configurable) equirectangular panorama PNG.

### Running

**Recommended:** Run orientation tracking first to generate `.npy` files, then build panoramas:

```bash
cd code/panoramic
python panorama.py
```

Key configuration variables:

| Variable | Default | Description |
|---|---|---|
| `RUN_ALL` | `True` | Run all datasets that have camera data |
| `DATASET` | `1` | Single dataset when `RUN_ALL=False` |
| `PANO_H / PANO_W` | `960 / 1920` | Output panorama resolution |
| `FOCAL_LENGTH` | `280.0` | Pinhole focal length in pixels |
| `USE_VICON` | `False` | Use Vicon ground-truth instead of PGD estimates |

Datasets with camera data: **1, 2, 8, 9** (train) and **10, 11** (test).

Orientation sources (in priority order):
1. Pre-saved `.npy` files from `Orientation Tracking/results/`
2. Vicon ground-truth (if `USE_VICON=True`)
3. On-the-fly PGD tracking (falls back to gyro-only if PyTorch unavailable)

---

## Utilities

### `load_data.py`

Loads `.p` (pickle) data files, handling both Python 2 and Python 3 encodings.

```python
from load_data import read_data
imu_data = read_data("path/to/imuRaw1.p")
```

### `rotplot.py`

Visualises a 3×3 rotation matrix as an oriented 3D box.

```python
from rotplot import rotplot
import numpy as np
rotplot(np.eye(3))   # identity orientation
```

---

## Results

After running both scripts, results are stored in:

```
code/Orientation Tracking/results/   ← roll/pitch/yaw comparison plots + .npy files
code/panoramic/results/              ← equirectangular panorama PNGs
```

Training datasets (1–9) include Vicon ground-truth overlaid in the orientation plots (blue = Vicon, red = estimated). Test datasets (10–11) show estimated trajectories only.
