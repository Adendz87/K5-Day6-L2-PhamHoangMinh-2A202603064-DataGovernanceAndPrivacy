"""Matching Adapter interfacing with protected src/t11/matching.py and geometry.py."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from service.adapters.yolo_adapter import YOLOPrediction
from service.core.annotation_normalizer import NormalizedAnnotation
from src.t11.geometry import center_dist_norm, iou_matrix
from src.t11.matching import ImageMatch, match_image


@dataclass
class MatchResult:
    image_id: str
    match: ImageMatch
    annotations: List[NormalizedAnnotation]
    predictions: List[YOLOPrediction]
    iou_matrix: np.ndarray


class MatchingAdapter:
    """Safely bridges incoming data to the protected matching engine."""

    def __init__(self, match_iou: float = 0.30, shift_gate: float = 2.0) -> None:
        self.match_iou = match_iou
        self.shift_gate = shift_gate

    def match_image_data(
        self,
        image_id: str,
        annotations: List[NormalizedAnnotation],
        predictions: List[YOLOPrediction],
    ) -> MatchResult:
        if len(annotations) > 0:
            a_boxes = np.array([[a.x, a.y, a.w, a.h] for a in annotations], dtype=float)
            a_cls = [a.class_name for a in annotations]
        else:
            a_boxes = np.zeros((0, 4), dtype=float)
            a_cls = []

        if len(predictions) > 0:
            p_boxes = np.array([[p.x, p.y, p.w, p.h] for p in predictions], dtype=float)
            p_cls = [p.class_name for p in predictions]
        else:
            p_boxes = np.zeros((0, 4), dtype=float)
            p_cls = []

        match = match_image(
            a_boxes,
            a_cls,
            p_boxes,
            p_cls,
            match_iou=self.match_iou,
            shift_gate=self.shift_gate,
        )

        ious = iou_matrix(a_boxes, p_boxes)

        return MatchResult(
            image_id=image_id,
            match=match,
            annotations=annotations,
            predictions=predictions,
            iou_matrix=ious,
        )
