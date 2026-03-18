# ECE 276A PR1 – Part 2: Panoramic Image Construction

Constructs an equirectangular (360°) panoramic image by stitching the
RGB camera images over time using the orientation estimates `q_{1:T}`.

## Algorithm

For each camera frame:
1. **Orientation lookup** – find the closest-past IMU orientation estimate
   (largest timestamp ≤ camera timestamp) per the project spec.
2. **Rotation matrix** – convert the quaternion `q = (w,x,y,z)` to a
   body-to-world rotation matrix `R`.
3. **Ray projection** – for every pixel `(u,v)` build a unit direction
   ray in the body/camera frame using a pinhole model:
   ```
   d_body = normalise([f,  u - cx,  cy - v])
   ```
   where the optical axis is the body x-axis, y points right, z points up.
4. **World-frame direction** – `d_world = R @ d_body`
5. **Spherical coords** – azimuth `φ = atan2(dy, dx)`,
   elevation `θ = arcsin(dz)`.
6. **Equirectangular mapping**:
   ```
   px = (φ + π) / (2π) × W
   py = (π/2 − θ) / π  × H
   ```
7. **Write** pixel colour (overwrite; no blending required by spec).

## Environment

Activate the conda environment before running:
```bash
conda activate opencv-env
```

## Usage

```bash
cd code/panoramic

# Option A – use pre-saved orientation estimates from Part 1:
python panorama.py --dataset 1 \
    --quats "../Orientation Tracking/results/dataset1_quats.npy" \
    --ts    "../Orientation Tracking/results/dataset1_ts.npy"

# Option B – use Vicon ground-truth orientations (training sets only):
python panorama.py --dataset 1 --use_vicon

# Option C – run orientation tracking on-the-fly (needs PyTorch):
python panorama.py --dataset 1 --data_dir ../../trainset

# Run ALL datasets that have camera data (1,2,8,9 train + 10,11 test):
python panorama.py --all \
    --data_dir_train ../../trainset \
    --data_dir_test  ../../testset

# Custom panorama resolution / focal length:
python panorama.py --dataset 1 --pano_H 720 --pano_W 1440 --focal_length 280
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | 1 | Dataset ID (1–9 train, 10–11 test) |
| `--data_dir` | `../../trainset` | Data root folder |
| `--all` | — | Process all camera datasets |
| `--data_dir_train` | `../../trainset` | Train root (with `--all`) |
| `--data_dir_test` | `../../testset` | Test root (with `--all`) |
| `--quats` | — | Path to pre-saved quaternions `.npy` |
| `--ts` | — | Path to pre-saved timestamps `.npy` |
| `--use_vicon` | — | Use Vicon ground-truth orientations |
| `--n_iter` | 300 | PGD iterations |
| `--lr` | 0.01 | PGD learning rate |
| `--n_static` | 200 | Static samples for bias estimation |
| `--pano_H` | 360 | Output height (pixels) |
| `--pano_W` | 720 | Output width (pixels) |
| `--focal_length` | 280.0 | Camera focal length (pixels) |
| `--output_dir` | `results/` | Where to save output PNGs |

## Output

Panorama images are saved to `results/dataset{N}_panorama.png`.
