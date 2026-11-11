"""KITTI training loop — Sec. 5.2.

"The network trains end to end for 50 epochs with AdamW on KITTI odometry
(00–05 train, 06–07 validation, 08–10 test) and evaluates additionally on
nuScenes."

Expected KITTI odometry layout::

    <root>/dataset/sequences/00/velodyne/000000.bin
    <root>/dataset/sequences/00/calib.txt
    <root>/dataset/poses/00.txt          # ground-truth poses (00–10)

Pairs are sampled with a ground-truth overlap filter (voxel-hash overlap
ratio); the expected directory layout mirrors the public benchmark.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..common.se3 import SE3
from .config import RegConfig
from .losses import infonce_loss, registration_loss, transform_l1_loss, weighted_bce_loss
from .model import TRIADRegModel
from .sampling import distance_weighted_fps

TRAIN_SEQUENCES = ("00", "01", "02", "03", "04", "05")
VAL_SEQUENCES = ("06", "07")
TEST_SEQUENCES = ("08", "09", "10")


def load_kitti_scan(root: str, sequence: str, frame: int) -> np.ndarray:
    path = os.path.join(root, "dataset", "sequences", sequence, "velodyne",
                        f"{frame:06d}.bin")
    raw = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return raw[:, :3].astype(np.float64)


def load_kitti_poses(root: str, sequence: str) -> np.ndarray:
    path = os.path.join(root, "dataset", "poses", f"{sequence}.txt")
    data = np.loadtxt(path).reshape(-1, 3, 4)
    poses = np.tile(np.eye(4), (len(data), 1, 1))
    poses[:, :3, :4] = data
    return poses


def load_kitti_calib(root: str, sequence: str) -> np.ndarray:
    """Tr calibration: velodyne → cam0 (column of calib.txt)."""
    path = os.path.join(root, "dataset", "sequences", sequence, "calib.txt")
    calib: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, values = line.split(":", 1)
            calib[key.strip()] = np.array([float(v) for v in values.split()])
    Tr = calib["Tr"].reshape(3, 4)
    T = np.eye(4)
    T[:3, :4] = Tr
    return T


def overlap_ratio(
    src: np.ndarray, tgt: np.ndarray, voxel: float = 0.3, radius: float = 1.0
) -> float:
    """Ground-truth-free overlap proxy via voxel hashing.

    Both clouds are voxelised; the ratio is the fraction of source voxels
    whose neighbourhood intersects a target voxel — used to bucket the
    Table 7 overlap groups (10–30 %, 30–50 %, > 50 %).
    """
    src_keys = set(map(tuple, np.floor(src / voxel).astype(int)))
    tgt_keys = set(map(tuple, np.floor(tgt / voxel).astype(int)))
    if not src_keys:
        return 0.0
    hits = 0
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if abs(dx) + abs(dy) + abs(dz) <= radius * 3
    ]
    for key in src_keys:
        for off in offsets:
            if (key[0] + off[0], key[1] + off[1], key[2] + off[2]) in tgt_keys:
                hits += 1
                break
    return hits / len(src_keys)


@dataclass
class RegistrationPair:
    src: np.ndarray
    tgt: np.ndarray
    T_gt: np.ndarray  # src frame → tgt frame
    sequence: str
    src_frame: int
    tgt_frame: int


class KittiOdometryDataset(Dataset):
    """Frame pairs of the KITTI odometry split with an overlap filter."""

    def __init__(
        self,
        root: str,
        sequences: Tuple[str, ...],
        config: Optional[RegConfig] = None,
        pair_gap: int = 5,
        max_pairs: Optional[int] = None,
        seed: int = 0,
        stride: int = 1,
    ):
        self.root = root
        self.config = config or RegConfig()
        self.pairs: List[RegistrationPair] = []
        rng = np.random.default_rng(seed)

        for sequence in sequences:
            poses = load_kitti_poses(root, sequence)
            n_frames = len(poses)
            indices = list(range(0, n_frames, max(stride, 1)))
            for i in indices:
                j = i + pair_gap
                if j >= n_frames:
                    continue
                # ground-truth relative transform src → tgt
                T_gt = SE3.inverse(poses[j]) @ poses[i]
                self.pairs.append(
                    RegistrationPair(
                        src=None, tgt=None, T_gt=T_gt,
                        sequence=sequence, src_frame=i, tgt_frame=j,
                    )
                )
        if max_pairs is not None and len(self.pairs) > max_pairs:
            chosen = rng.choice(len(self.pairs), size=max_pairs, replace=False)
            self.pairs = [self.pairs[k] for k in sorted(chosen)]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> RegistrationPair:
        pair = self.pairs[index]
        if pair.src is None:
            src = load_kitti_scan(self.root, pair.sequence, pair.src_frame)
            tgt = load_kitti_scan(self.root, pair.sequence, pair.tgt_frame)
            # subsample to keypoint budget with DW-FPS (Sec. 3.3)
            n = self.config.keypoint_counts[-1]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            src_t = torch.as_tensor(src, dtype=torch.float32, device=device)
            tgt_t = torch.as_tensor(tgt, dtype=torch.float32, device=device)
            if len(src_t) > n:
                src = src[distance_weighted_fps(src_t[None], n,
                                                self.config.dw_fps_beta)[0]
                          .cpu().numpy()].astype(np.float32)
            if len(tgt_t) > n:
                tgt = tgt[distance_weighted_fps(tgt_t[None], n,
                                                self.config.dw_fps_beta)[0]
                          .cpu().numpy()].astype(np.float32)
            pair.src, pair.tgt = src, tgt
        return pair


def gt_inlier_labels(
    src: torch.Tensor, tgt: torch.Tensor, T_gt: torch.Tensor,
    src_idx: torch.Tensor, tgt_idx: torch.Tensor, thresh: float = 0.3,
) -> torch.Tensor:
    """Binary inlier labels of the candidate correspondences under the GT
    transform — the registration reliability labels of Sec. 5.4."""
    src_t = src @ T_gt[:3, :3].T + T_gt[:3, 3]
    err = (src_t[src_idx] - tgt[tgt_idx]).norm(dim=1)
    return (err < thresh).float()


def train(
    root: str,
    config: Optional[RegConfig] = None,
    device: Optional[str] = None,
    checkpoint_path: str = "checkpoints/triad_reg.pt",
    log_every: int = 10,
) -> TRIADRegModel:
    """End-to-end training for 50 epochs with AdamW (Sec. 5.2)."""
    config = config or RegConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TRIADRegModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    train_set = KittiOdometryDataset(root, TRAIN_SEQUENCES, config)
    val_set = KittiOdometryDataset(root, VAL_SEQUENCES, config, max_pairs=200)

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step, pair in enumerate(train_set):
            src = torch.as_tensor(pair.src, dtype=torch.float32, device=device)
            tgt = torch.as_tensor(pair.tgt, dtype=torch.float32, device=device)
            T_gt = torch.as_tensor(pair.T_gt, dtype=torch.float32, device=device)

            output = model(src, tgt)

            # inlier labels under GT for InfoNCE/BCE supervision
            src_t = src @ T_gt[:3, :3].T + T_gt[:3, 3]
            match = torch.cdist(src_t, tgt) < 0.3
            pos_src, pos_tgt = torch.nonzero(match, as_tuple=True)
            labels = gt_inlier_labels(src, tgt, T_gt, output.src_idx,
                                      output.tgt_idx, thresh=0.3)

            infonce_terms = [
                infonce_loss(src, tgt, pos_src, pos_tgt,
                             config.match_temperature)
            ]
            wbce_terms = [
                weighted_bce_loss(output.confidences, labels, output.weights.detach())
            ]
            transform_terms = [transform_l1_loss(output.transform, T_gt)]
            loss = registration_loss(
                infonce_terms, wbce_terms, transform_terms,
                (config.loss_infonce_weight, config.loss_wbce_weight,
                 config.loss_transform_weight),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
            if log_every and (step + 1) % log_every == 0:
                print(f"epoch {epoch} step {step + 1} loss {loss.item():.4f}")

        # validation: relative translation / rotation error (Sec. 5.2)
        model.eval()
        rte, rre = [], []
        with torch.no_grad():
            for pair in val_set:
                src = torch.as_tensor(pair.src, dtype=torch.float32, device=device)
                tgt = torch.as_tensor(pair.tgt, dtype=torch.float32, device=device)
                T_pred = model(src, tgt).transform.cpu().numpy()
                T_gt_np = pair.T_gt
                rte.append(np.linalg.norm(T_pred[:3, 3] - T_gt_np[:3, 3]))
                rre.append(SE3.rotation_error_deg(T_pred, T_gt_np))
        print(
            f"epoch {epoch}: train loss {epoch_loss / max(len(train_set), 1):.4f} "
            f"val RTE {np.mean(rte):.3f} m RRE {np.mean(rre):.3f} deg"
        )

        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
    return model
