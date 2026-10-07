"""Empirical check: is LIDAR_SCALE=/40 or /50 more appropriate for the alpha dataset?

Samples lidar point clouds from train arrow files, computes the distance (r)
distribution over FOV-valid points AND over the actual non-empty range-view depth
cells (what the network actually sees), then reports /40 vs /50 mappings.
"""
import glob
import io
import os

import numpy as np
import pyarrow as pa

ROOT = "/home/carol/chaeyeon-kim/processed"
H, W = 32, 1024
FOV = (-30.0, 10.0)
N_FILES = 4          # how many train files to sample
FRAMES_PER_FILE = 80 # evenly-spaced frames per file


def range_view_depth(points, H=H, W=W, fov_degrees=FOV):
    """Mirror data_muvo_2D.point_cloud_to_range_view; return (valid_r, nonempty_depth)."""
    fov_down = np.deg2rad(fov_degrees[0])
    fov_up = np.deg2rad(fov_degrees[1])
    fov = fov_up - fov_down

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    yaw = np.arctan2(y, x)
    pitch = np.arcsin(z / (r + 1e-8))

    yaw_img = (yaw / np.pi + 1.0) / 2.0
    pitch_img = (pitch - fov_down) / fov
    valid = np.isfinite(r) & (r > 0) & (pitch_img >= 0.0) & (pitch_img <= 1.0)

    r_v = r[valid]
    yaw_img = yaw_img[valid]
    pitch_img = pitch_img[valid]

    col = np.clip((yaw_img * (W - 1)).astype(int), 0, W - 1)
    row = np.clip(((1.0 - pitch_img) * (H - 1)).astype(int), 0, H - 1)
    order = np.argsort(r_v)[::-1]
    depth = np.full((H, W), -1.0, dtype=np.float32)
    depth[row[order], col[order]] = r_v[order]   # nearest wins
    nonempty = depth[depth >= 0.0]
    return r_v, nonempty


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "train_run_*.arrow")))[:N_FILES]
    all_valid_r = []
    all_rv_depth = []
    for fp in files:
        mmap = pa.memory_map(fp, "r")
        table = pa.ipc.open_stream(mmap).read_all()
        n = table.num_rows
        idxs = np.linspace(0, n - 1, FRAMES_PER_FILE).astype(int)
        for k in idxs:
            pts = np.array(table.slice(int(k), 1).column("lidar")[0].as_py())
            if pts.ndim != 2 or pts.shape[0] == 0:
                continue
            vr, nd = range_view_depth(pts)
            all_valid_r.append(vr)
            all_rv_depth.append(nd)
        print(f"  {os.path.basename(fp)}: {n} rows, sampled {len(idxs)}")

    valid_r = np.concatenate(all_valid_r)
    rv_depth = np.concatenate(all_rv_depth)

    def report(name, arr):
        qs = [50, 90, 95, 99, 99.9, 100]
        pv = np.percentile(arr, qs)
        print(f"\n=== {name} (n={arr.size:,}) ===")
        print(f"  mean={arr.mean():.2f}m  " + "  ".join(f"p{q}={v:.2f}m" for q, v in zip(qs, pv)))
        for scale in (40.0, 50.0):
            sv = pv / scale
            mx = arr.max() / scale
            frac_gt1 = (arr / scale > 1.0).mean() * 100
            frac_gt2 = (arr / scale > 2.0).mean() * 100
            print(f"  /{scale:.0f}:  max={mx:.3f}  p99={sv[3]:.3f}  p99.9={sv[4]:.3f}"
                  f"   | >1.0: {frac_gt1:5.2f}%   >2.0: {frac_gt2:5.2f}%")

    report("raw distance r over FOV-valid points", valid_r)
    report("range-view non-empty depth cells (what the net sees)", rv_depth)


if __name__ == "__main__":
    main()
