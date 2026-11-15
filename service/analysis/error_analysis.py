"""Error type analysis and candidate detection layer."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from service.adapters.sam3_adapter import BaseSAM3Adapter
from service.adapters.yolo_adapter import YOLOPrediction
from service.analysis.matching_adapter import MatchResult
from service.core.annotation_normalizer import NormalizedAnnotation
from src.t11.geometry import iou_matrix


@dataclass
class ScoredAnnotationEvidence:
    ann_id: str
    image_id: str
    risk_score: float
    risk_level: str
    possible_error_type: str
    reason_code: str
    suggested_action: str
    review_required: bool
    evidence_dict: Dict[str, Any]


@dataclass
class MissingCandidateEvidence:
    image_id: str
    item_id: str  # "img:<image_id>"
    risk_score: float
    risk_level: str
    model_bbox: Tuple[float, float, float, float]
    predicted_class: str
    prediction_confidence: float
    suggested_action: str
    evidence_dict: Dict[str, Any]


class ErrorAnalyzer:
    """Analyzes matches and produces structured error evidence."""

    def __init__(
        self,
        w_dup: float = 0.95,
        w_swap: float = 0.90,
        w_shift: float = 0.85,
        w_missing: float = 0.85,
        w_unmatched: float = 0.35,
        dup_iou: float = 0.50,
        orphan_overlap: float = 0.30,
        missing_conf_floor: float = 0.40,
        sam3_adapter: Optional[BaseSAM3Adapter] = None,
    ) -> None:
        self.w_dup = w_dup
        self.w_swap = w_swap
        self.w_shift = w_shift
        self.w_missing = w_missing
        self.w_unmatched = w_unmatched
        self.dup_iou = dup_iou
        self.orphan_overlap = orphan_overlap
        self.missing_conf_floor = missing_conf_floor
        self.sam3_adapter = sam3_adapter

    def _score_to_risk_level(self, score: float) -> str:
        if score < 0.20:
            return "RISK_LOW"
        elif score < 0.50:
            return "RISK_MEDIUM"
        elif score < 0.80:
            return "RISK_HIGH"
        return "RISK_CRITICAL"

    def analyze_image(
        self,
        match_result: MatchResult,
        image_path: Optional[Any] = None,
    ) -> Tuple[List[ScoredAnnotationEvidence], List[MissingCandidateEvidence]]:
        """Perform error classification and evidence generation on matched image data."""
        ann_evidences: List[ScoredAnnotationEvidence] = []
        missing_evidences: List[MissingCandidateEvidence] = []

        m = match_result.match
        anns = match_result.annotations
        preds = match_result.predictions
        img_id = match_result.image_id

        # 1. Duplicates check between annotations in this image
        dup_map: Dict[int, Tuple[int, float]] = {}
        if len(anns) >= 2:
            a_boxes = np.array([[a.x, a.y, a.w, a.h] for a in anns])
            pair_ious = iou_matrix(a_boxes, a_boxes)
            for i in range(len(anns)):
                for j in range(i + 1, len(anns)):
                    if anns[i].class_name == anns[j].class_name and pair_ious[i, j] >= self.dup_iou:
                        dup_map[i] = (j, float(pair_ious[i, j]))
                        dup_map[j] = (i, float(pair_ious[i, j]))

        # 2. Score existing annotations
        for i, a in enumerate(anns):
            pred_idx = int(m.pred_of_ann[i]) if len(m.pred_of_ann) > i else -1
            stage = int(m.stage[i]) if len(m.stage) > i else 0
            iou_val = float(m.iou[i]) if len(m.iou) > i else 0.0

            scores: Dict[str, float] = {}
            ev_dict: Dict[str, Any] = {
                "human_bbox": [a.x, a.y, a.w, a.h],
                "human_class": a.class_name,
                "iou": iou_val,
                "match_stage": stage,
            }

            # Check duplicate
            if i in dup_map:
                partner_idx, d_iou = dup_map[i]
                partner = anns[partner_idx]
                scores["duplicate"] = self.w_dup * (0.6 + 0.4 * min(1.0, (d_iou - self.dup_iou) / 0.4))
                ev_dict["partner_ann_id"] = partner.ann_id

            if pred_idx >= 0 and pred_idx < len(preds):
                p = preds[pred_idx]
                ev_dict["model_bbox"] = [p.x, p.y, p.w, p.h]
                ev_dict["predicted_class"] = p.class_name
                ev_dict["prediction_confidence"] = p.conf

                # Class swap
                if p.class_name != a.class_name:
                    swap_s = self.w_swap * p.conf * (0.6 + 0.4 * min(1.0, iou_val / 0.7))
                    scores["class_swap"] = swap_s
                    ev_dict["class_disagreement"] = True

                # Box shift
                if stage == 2 or iou_val < 0.65:
                    shift_s = self.w_shift * (1.0 - iou_val) * (0.6 + 0.4 * p.conf)
                    scores["box_shift"] = shift_s

                    # Conditional SAM 3 evidence
                    if self.sam3_adapter and image_path:
                        sam_res = self.sam3_adapter.evaluate_box_evidence(
                            image_path,
                            (a.x, a.y, a.w, a.h),
                            (p.x, p.y, p.w, p.h),
                        )
                        ev_dict["sam3_mask_ref"] = sam_res.mask_ref
                        ev_dict["geometric_diff"] = sam_res.geometric_diff
            else:
                # Unmatched annotation
                scores["unmatched"] = self.w_unmatched * 0.7

            # Determine top error
            if scores:
                top_err = max(scores, key=lambda k: scores[k])
                top_score = scores[top_err]
            else:
                top_err = "none"
                top_score = 0.05

            risk_level = self._score_to_risk_level(top_score)
            review_req = top_score >= 0.20

            # Suggested action
            action = ""
            if top_err == "class_swap" and "predicted_class" in ev_dict:
                action = f"edit class:{ev_dict['predicted_class']}"
            elif top_err == "box_shift" and "model_bbox" in ev_dict:
                b = ev_dict["model_bbox"]
                action = f"edit box:{b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f}"
            elif top_err == "duplicate":
                action = f"reject duplicate of {ev_dict.get('partner_ann_id', '')}"
            elif top_err == "unmatched":
                action = "verify: manual inspection"

            ann_evidences.append(
                ScoredAnnotationEvidence(
                    ann_id=a.ann_id,
                    image_id=img_id,
                    risk_score=round(float(top_score), 4),
                    risk_level=risk_level,
                    possible_error_type=top_err,
                    reason_code=top_err.upper(),
                    suggested_action=action,
                    review_required=review_req,
                    evidence_dict=ev_dict,
                )
            )

        # 3. Detect missing annotation candidates (orphan predictions with high confidence)
        for orphan_idx in m.orphan_preds:
            if orphan_idx < len(preds):
                p = preds[orphan_idx]
                if p.conf >= self.missing_conf_floor:
                    miss_score = self.w_missing * p.conf
                    missing_evidences.append(
                        MissingCandidateEvidence(
                            image_id=img_id,
                            item_id=f"img:{img_id}",
                            risk_score=round(float(miss_score), 4),
                            risk_level=self._score_to_risk_level(miss_score),
                            model_bbox=(p.x, p.y, p.w, p.h),
                            predicted_class=p.class_name,
                            prediction_confidence=p.conf,
                            suggested_action=f"add {p.class_name} box:{p.x:.1f},{p.y:.1f},{p.w:.1f},{p.h:.1f}",
                            evidence_dict={
                                "model_bbox": [p.x, p.y, p.w, p.h],
                                "predicted_class": p.class_name,
                                "prediction_confidence": p.conf,
                                "reason_code": "MISSING",
                            },
                        )
                    )

        return ann_evidences, missing_evidences
