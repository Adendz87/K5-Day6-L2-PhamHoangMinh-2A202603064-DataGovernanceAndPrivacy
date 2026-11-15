"""Hình học box. Mọi box là mảng (n, 4) = x, y, w, h với (x, y) là góc trái-trên (đã kiểm tra trên fixture)."""
from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU từng cặp giữa a (n,4) và b (m,4) -> (n,m)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1 = a[:, 0:1], a[:, 1:2]
    ax2, ay2 = ax1 + a[:, 2:3], ay1 + a[:, 3:4]
    bx1, by1 = b[:, 0][None, :], b[:, 1][None, :]
    bx2, by2 = bx1 + b[:, 2][None, :], by1 + b[:, 3][None, :]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = a[:, 2:3] * a[:, 3:4] + (b[:, 2] * b[:, 3])[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def center_dist_norm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Khoảng cách tâm giữa từng cặp, chuẩn hoá theo kích thước đối tượng (căn bậc hai diện tích lớn hơn)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ac = np.stack([a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2], axis=1)
    bc = np.stack([b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)
    d = np.linalg.norm(ac[:, None, :] - bc[None, :, :], axis=2)
    sa = np.sqrt(np.clip(a[:, 2] * a[:, 3], 1e-6, None))[:, None]
    sb = np.sqrt(np.clip(b[:, 2] * b[:, 3], 1e-6, None))[None, :]
    return d / np.maximum(sa, sb)
