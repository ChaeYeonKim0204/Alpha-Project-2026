"""data_muvo_2D.py — `data_muvo.py` 베이스 + PixelAugmentation 포팅.

변경점:
- `from config_muvo_2D import cfg`로 import swap.
- 신규 `PixelAugmentation` (`muvo_2d/muvo/models/preprocess.py:295-333` 포팅).
  Blur / Sharpen / ColorJitter, single `cfg.DATA.AUGMENTATION.ENABLED` flag.
- `_make_img_transform` (encoder input)만 augmentation 적용,
  `_make_img_transform_raw` (recon target)는 augmentation 안 함.

Dataset이 반환하는 dict key/shape는 byte-identical (image, image_raw, lidar, action,
speed, run_id, start_row).
"""
import os
import io
import bisect
import glob
import math
import pickle

import numpy as np
import pyarrow as pa
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from config_muvo_2D import cfg


# ─────────────────────────────────────────────
# Image / LiDAR transforms
# ─────────────────────────────────────────────

class CenterCropToSize:
    """Center-crop PIL image to (target_h, target_w). Both axes."""

    def __init__(self, size):
        self.target_h, self.target_w = tuple(size)

    def __call__(self, img):
        width, height = img.size
        if width < self.target_w or height < self.target_h:
            raise ValueError(
                f"image {height}x{width} is smaller than crop {self.target_h}x{self.target_w}."
            )
        top  = (height - self.target_h) // 2
        left = (width  - self.target_w) // 2
        return TF.crop(img, top=top, left=left, height=self.target_h, width=self.target_w)


class PixelAugmentation:
    """upstream `muvo_2d/muvo/models/preprocess.py:295-333` 포팅.

    Blur / Sharpen / ColorJitter — 각각 독립 확률로 적용. blur와 sharpen은
    `blur_prob + sharpen_prob <= 1`로 mutex. ColorJitter는 그 위에 독립 적용.

    PIL Image (encoder input)에 직접 적용 — Tensor 변환 전 단계에서 실행.
    """

    def __init__(self, aug_cfg):
        self.enabled = bool(getattr(aug_cfg, "ENABLED", True))
        self.blur_prob = float(getattr(aug_cfg, "BLUR_PROB", 0.3))
        self.sharpen_prob = float(getattr(aug_cfg, "SHARPEN_PROB", 0.3))
        assert self.blur_prob + self.sharpen_prob <= 1.0, (
            f"blur_prob + sharpen_prob must be ≤ 1; got {self.blur_prob}+{self.sharpen_prob}"
        )
        self.blur_window = int(getattr(aug_cfg, "BLUR_WINDOW", 5))
        self.blur_std = tuple(getattr(aug_cfg, "BLUR_STD", (0.1, 1.7)))
        self.sharpen_factor = tuple(getattr(aug_cfg, "SHARPEN_FACTOR", (1.0, 5.0)))

        color_prob = float(getattr(aug_cfg, "COLOR_PROB", 0.3))
        self.color_jitter = transforms.RandomApply(
            torch.nn.ModuleList([
                transforms.ColorJitter(
                    brightness=float(getattr(aug_cfg, "BRIGHTNESS", 0.3)),
                    contrast=float(getattr(aug_cfg, "CONTRAST", 0.3)),
                    saturation=float(getattr(aug_cfg, "SATURATION", 0.3)),
                    hue=float(getattr(aug_cfg, "HUE", 0.1)),
                ),
            ]),
            p=color_prob,
        )

    def __call__(self, img):
        if not self.enabled:
            return img
        rand_value = float(torch.rand(1).item())
        if rand_value < self.blur_prob:
            std = float(torch.empty(1).uniform_(self.blur_std[0], self.blur_std[1]).item())
            img = TF.gaussian_blur(img, kernel_size=self.blur_window, sigma=std)
        elif rand_value < self.blur_prob + self.sharpen_prob:
            factor = float(torch.empty(1).uniform_(self.sharpen_factor[0], self.sharpen_factor[1]).item())
            img = TF.adjust_sharpness(img, sharpness_factor=factor)
        img = self.color_jitter(img)
        return img


def _make_img_transform(size=None, augment=True):
    """Encoder input transform. augment=True일 때만 PixelAugmentation 적용."""
    size = size or cfg.DATA.IMAGE_INPUT_SIZE
    ops = [CenterCropToSize(size)]
    if augment and getattr(cfg.DATA, "AUGMENTATION", None) is not None:
        ops.append(PixelAugmentation(cfg.DATA.AUGMENTATION))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(ops)


def _make_img_transform_raw(size=None):
    """Reconstruction target. 같은 spatial crop이지만 augmentation도 normalisation도
    적용하지 않아 [0, 1] 범위 그대로 decoder target으로 쓸 수 있게."""
    size = size or cfg.DATA.RGB_RECON_SIZE
    return transforms.Compose([
        CenterCropToSize(size),
        transforms.ToTensor(),
    ])


# ─────────────────────────────────────────────
# LiDAR point cloud → range view
# ─────────────────────────────────────────────

def point_cloud_to_range_view(points, H=None, W=None, fov_degrees=None):
    if H is None or W is None:
        H, W = cfg.DATA.LIDAR_RANGE_VIEW_SIZE
    fov_down_deg, fov_up_deg = fov_degrees or cfg.DATA.LIDAR_FOV_DEGREES
    fov_down = np.deg2rad(fov_down_deg)
    fov_up   = np.deg2rad(fov_up_deg)
    fov      = fov_up - fov_down

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r     = np.sqrt(x**2 + y**2 + z**2)
    yaw   = np.arctan2(y, x)
    pitch = np.arcsin(z / (r + 1e-8))

    yaw_img   = (yaw / np.pi + 1.0) / 2.0
    pitch_img = (pitch - fov_down) / fov
    valid     = np.isfinite(r) & (r > 0) & (pitch_img >= 0.0) & (pitch_img <= 1.0)

    x, y, z, r = x[valid], y[valid], z[valid], r[valid]
    yaw_img   = yaw_img[valid]
    pitch_img = pitch_img[valid]

    col = np.clip((yaw_img * (W - 1)).astype(int), 0, W - 1)
    row = np.clip(((1.0 - pitch_img) * (H - 1)).astype(int), 0, H - 1)

    order = np.argsort(r)[::-1]
    rv = np.zeros((4, H, W), dtype=np.float32)
    rv[0, row[order], col[order]] = x[order]
    rv[1, row[order], col[order]] = y[order]
    rv[2, row[order], col[order]] = z[order]
    rv[3, row[order], col[order]] = r[order]
    return rv


# ─────────────────────────────────────────────
# Arrow manifest helpers
# ─────────────────────────────────────────────

def build_arrow_manifest(root=None, out_path=None):
    root     = root     or cfg.DATA.ROOT
    out_path = out_path or cfg.DATA.MANIFEST_PATH
    files    = sorted(glob.glob(os.path.join(root, "*.arrow")))
    manifest = {
        "root":       root,
        "train":      [p for p in files if os.path.basename(p).startswith("train_")],
        "validation": [p for p in files if os.path.basename(p).startswith("validation_")],
        "test":       [p for p in files if os.path.basename(p).startswith("test_")],
    }
    with open(out_path, "wb") as f:
        pickle.dump(manifest, f)
    return manifest


def load_arrow_manifest(path=None, root=None, rebuild=False):
    path = path or cfg.DATA.MANIFEST_PATH
    if rebuild or not os.path.exists(path):
        return build_arrow_manifest(root=root, out_path=path)
    with open(path, "rb") as f:
        return pickle.load(f)


def _as_arrow_path(name_or_path):
    if os.path.isabs(name_or_path):
        return name_or_path
    return os.path.join(cfg.DATA.ROOT, name_or_path)


def _select_from_manifest(manifest, split, preferred=None, use_all=False):
    files = manifest[split]
    if use_all:
        return files
    if preferred is not None:
        preferred_path = _as_arrow_path(preferred)
        if preferred_path in files or os.path.exists(preferred_path):
            return [preferred_path]
    return files[:1]


# ─────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────

class MUVODataset(Dataset):
    def __init__(self, arrow_file, seq_len=4, stride=None, sample_every_n=None,
                 image_input_size=None, rgb_recon_size=None, lidar_size=None,
                 frame_step=None, require_contiguous_frames=True, augment=True):
        self.seq_len        = int(seq_len)
        self.sample_every_n = int(sample_every_n or cfg.DATA.SAMPLE_EVERY_N)
        self.stride         = self.seq_len if stride is None else int(stride)
        self.arrow_file     = _as_arrow_path(arrow_file)
        self.lidar_size     = tuple(lidar_size or cfg.DATA.LIDAR_RANGE_VIEW_SIZE)
        self.img_transform     = _make_img_transform(image_input_size or cfg.DATA.IMAGE_INPUT_SIZE,
                                                     augment=augment)
        self.img_transform_raw = _make_img_transform_raw(rgb_recon_size or cfg.DATA.RGB_RECON_SIZE)
        self.frame_step               = int(frame_step or cfg.DATA.FRAME_STEP)
        self.require_contiguous_frames = require_contiguous_frames

        self._mmap  = pa.memory_map(self.arrow_file, "r")
        reader      = pa.ipc.open_stream(self._mmap)
        self.table  = reader.read_all()

        self.n = self.table.num_rows

        self.index   = []
        run_ids      = self.table.column("run_id").to_pylist()
        frames       = self.table.column("frame").to_pylist() if "frame" in self.table.column_names else None
        max_offset   = (self.seq_len - 1) * self.sample_every_n

        for i in range(0, self.n - max_offset, self.stride):
            offsets = [j * self.sample_every_n for j in range(self.seq_len)]
            if not all(run_ids[i + off] == run_ids[i] for off in offsets):
                continue
            if self.require_contiguous_frames and frames is not None:
                expected = self.frame_step * self.sample_every_n
                if not all(
                    frames[i + offsets[j]] - frames[i + offsets[j - 1]] == expected
                    for j in range(1, self.seq_len)
                ):
                    continue
            self.index.append(i)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i = self.index[idx]
        images, images_raw, lidars, actions, speeds = [], [], [], [], []

        for j in range(self.seq_len):
            k   = i + j * self.sample_every_n
            row = self.table.slice(k, 1)

            img_bytes = row.column("image_front")[0].as_py()["bytes"]
            img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            images.append(self.img_transform(img))
            images_raw.append(self.img_transform_raw(img))

            lidar_np = np.array(row.column("lidar")[0].as_py())
            lidar_rv = point_cloud_to_range_view(lidar_np, *self.lidar_size) / float(cfg.DATA.LIDAR_SCALE)
            lidars.append(torch.tensor(lidar_rv))

            actions.append([row.column("throttle")[0].as_py(), row.column("steer")[0].as_py()])
            speeds.append(row.column("speed_kmh")[0].as_py())

        return {
            "image":     torch.stack(images),
            "image_raw": torch.stack(images_raw),
            "lidar":     torch.stack(lidars),
            "action":    torch.tensor(actions),
            "speed":     torch.tensor(speeds),
            "run_id":    self.table.column("run_id")[i].as_py(),
            "start_row": i,
        }


class StreamWindowDataset(MUVODataset):
    """Interleave contiguous window streams so TBPTT hidden states stay aligned per batch position."""

    def __init__(self, arrow_file, seq_len=4, num_streams=1, stride=None, **kwargs):
        super().__init__(arrow_file, seq_len=seq_len, stride=stride, **kwargs)
        self.num_streams = max(1, int(num_streams))
        self.index       = self._make_stream_order(self.index, self.num_streams)

    @staticmethod
    def _make_stream_order(index, num_streams):
        if num_streams <= 1 or len(index) == 0:
            return index

        chunk_size = math.ceil(len(index) / num_streams)
        chunks = [
            index[i * chunk_size: min((i + 1) * chunk_size, len(index))]
            for i in range(num_streams)
        ]
        chunks  = [c for c in chunks if c]
        min_len = min(len(c) for c in chunks)

        ordered = []
        for t in range(min_len):
            for chunk in chunks:
                ordered.append(chunk[t])
        return ordered


class MultiArrowStreamDataset(Dataset):
    def __init__(self, arrow_files, seq_len=4, num_streams=1, stride=None, **kwargs):
        self.datasets = [
            StreamWindowDataset(path, seq_len=seq_len, num_streams=num_streams, stride=stride, **kwargs)
            for path in arrow_files
        ]
        self.datasets = [ds for ds in self.datasets if len(ds) > 0]
        if not self.datasets:
            raise ValueError("No non-empty Arrow datasets were found.")
        lengths                  = [len(ds) for ds in self.datasets]
        self.cumulative_lengths  = np.cumsum(lengths).tolist()

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        ds_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        prev   = 0 if ds_idx == 0 else self.cumulative_lengths[ds_idx - 1]
        return self.datasets[ds_idx][idx - prev]
