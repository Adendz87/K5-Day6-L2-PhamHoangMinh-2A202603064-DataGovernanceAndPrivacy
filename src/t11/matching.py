"""Ghép annotation <-> prediction của detector, trong MỘT ảnh.

Stage 1  Hungarian, KHÔNG phân biệt class, chỉ ghép khi IoU >= match_iou. Ghép không phân biệt class
         là cố ý: nếu bắt buộc cùng class thì class swap sẽ trông như "không có prediction" và lẫn với
         detector bỏ sót. Cặp cùng class được cộng nhẹ 0.1 để thắng khi có hai ứng viên.
Stage 2  Với phần còn lại, ghép theo khoảng cách tâm (chuẩn hoá theo kích thước). Đây là cách nhận ra box
         bị DỊCH nhiều (IoU gần 0) — nếu bỏ stage này, một box bị dịch sinh ra hai tín hiệu giả
         ("annotation không khớp" + "prediction mồ côi => missing"), tức là đếm trùng một lỗi.
Còn lại  annotation không ghép được = stage 0; prediction không ghép được = "mồ côi" (ứng viên missing).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import center_dist_norm, iou_matrix

_BIG = 1e6


@dataclass
class ImageMatch:
    pred_of_ann: np.ndarray      # (nA,) chỉ số prediction, -1 nếu không ghép được
    stage: np.ndarray            # (nA,) 0 = không ghép, 1 = ghép theo IoU, 2 = ghép theo khoảng cách tâm
    iou: np.ndarray              # (nA,) IoU với prediction đã ghép (0 nếu không có)
    best_iou_any: np.ndarray     # (nA,) IoU lớn nhất với BẤT KỲ prediction nào (không loại trừ)
    orphan_preds: list[int]      # prediction không ghép được với annotation nào


def match_image(a_boxes, a_cls, p_boxes, p_cls, match_iou: float = 0.30, shift_gate: float = 2.0) -> ImageMatch:
    a_boxes = np.asarray(a_boxes, dtype=float).reshape(-1, 4)
    p_boxes = np.asarray(p_boxes, dtype=float).reshape(-1, 4)
    nA, nP = len(a_boxes), len(p_boxes)
    pred_of = np.full(nA, -1, dtype=int)
    stage = np.zeros(nA, dtype=int)
    iou = iou_matrix(a_boxes, p_boxes)
    best_any = iou.max(axis=1) if nP else np.zeros(nA)
    if nA == 0 or nP == 0:
        return ImageMatch(pred_of, stage, np.zeros(nA), best_any, list(range(nP)))

    same = np.array([[str(a) == str(p) for p in p_cls] for a in a_cls], dtype=float)

    # ---- stage 1
    cost = np.where(iou >= match_iou, -(iou + 0.1 * same), _BIG)
    r, c = linear_sum_assignment(cost)
    for i, j in zip(r, c):
        if cost[i, j] < _BIG / 2:
            pred_of[i], stage[i] = j, 1

    # ---- stage 2
    free_a = [i for i in range(nA) if pred_of[i] < 0]
    free_p = [j for j in range(nP) if j not in set(pred_of[pred_of >= 0].tolist())]
    if free_a and free_p:
        d = center_dist_norm(a_boxes[free_a], p_boxes[free_p])
        s2 = same[np.ix_(free_a, free_p)]
        cost2 = np.where(d <= shift_gate, d + 0.5 * (1.0 - s2), _BIG)
        r, c = linear_sum_assignment(cost2)
        for ri, ci in zip(r, c):
            if cost2[ri, ci] < _BIG / 2:
                pred_of[free_a[ri]], stage[free_a[ri]] = free_p[ci], 2

    matched_iou = np.array([iou[i, pred_of[i]] if pred_of[i] >= 0 else 0.0 for i in range(nA)])
    used = set(pred_of[pred_of >= 0].tolist())
    orphans = [j for j in range(nP) if j not in used]
    return ImageMatch(pred_of, stage, matched_iou, best_any, orphans)
