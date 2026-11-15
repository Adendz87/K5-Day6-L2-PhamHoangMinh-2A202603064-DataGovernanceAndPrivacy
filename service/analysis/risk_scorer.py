"""Multi-level Risk Scorer (Annotation → Image → Job level)."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List

from service.analysis.error_analysis import (
    MissingCandidateEvidence,
    ScoredAnnotationEvidence,
)


@dataclass
class ImageRiskSummary:
    image_id: str
    image_score: float
    risk_level: str
    annotation_count: int
    review_required: bool
    possible_error_types: List[str]
    processing_status: str = "IMAGE_DONE"


@dataclass
class JobRiskSummary:
    job_id: str
    run_id: str
    job_score: float
    risk_level: str
    total_images: int
    processed_images: int
    failed_images: int
    total_annotations: int
    review_required_count: int
    missing_candidates: int
    error_type_counts: Dict[str, int]
    scoring_version: str = "real-world-v1"


class RiskScorer:
    """Computes multi-level hierarchical risk scores."""

    def __init__(self, scoring_version: str = "real-world-v1") -> None:
        self.scoring_version = scoring_version

    @staticmethod
    def score_to_risk_level(score: float) -> str:
        if score < 0.20:
            return "RISK_LOW"
        elif score < 0.50:
            return "RISK_MEDIUM"
        elif score < 0.80:
            return "RISK_HIGH"
        return "RISK_CRITICAL"

    def compute_image_risk(
        self,
        image_id: str,
        ann_evidences: List[ScoredAnnotationEvidence],
        missing_evidences: List[MissingCandidateEvidence],
    ) -> ImageRiskSummary:
        """Compute aggregated image risk score:

        image_risk = max(ann_scores)*0.6 + p90(ann_scores)*0.3 + missing*0.1
        """
        ann_scores = [e.risk_score for e in ann_evidences]
        miss_scores = [m.risk_score for m in missing_evidences]

        max_ann = max(ann_scores) if ann_scores else 0.0
        p90_ann = float(np.percentile(ann_scores, 90)) if len(ann_scores) >= 2 else max_ann
        miss_component = max(miss_scores) if miss_scores else 0.0

        if not ann_scores and not miss_scores:
            image_score = 0.0
        elif not ann_scores:
            image_score = miss_component
        else:
            image_score = max_ann * 0.6 + p90_ann * 0.3 + miss_component * 0.1

        image_score = round(float(min(1.0, max(0.0, image_score))), 4)
        risk_level = self.score_to_risk_level(image_score)

        error_types = list(
            {e.possible_error_type for e in ann_evidences if e.possible_error_type != "none"}
        )
        if missing_evidences:
            error_types.append("missing")

        review_req = any(e.review_required for e in ann_evidences) or len(missing_evidences) > 0

        return ImageRiskSummary(
            image_id=image_id,
            image_score=image_score,
            risk_level=risk_level,
            annotation_count=len(ann_evidences),
            review_required=review_req,
            possible_error_types=error_types,
            processing_status="IMAGE_DONE",
        )

    def compute_job_risk(
        self,
        job_id: str,
        run_id: str,
        image_summaries: List[ImageRiskSummary],
        ann_evidences: List[ScoredAnnotationEvidence],
        missing_evidences: List[MissingCandidateEvidence],
    ) -> JobRiskSummary:
        """Compute Job-level risk score: 80th percentile of image scores."""
        img_scores = [img.image_score for img in image_summaries]
        if img_scores:
            job_score = float(np.percentile(img_scores, 80))
        else:
            job_score = 0.0

        job_score = round(float(job_score), 4)
        risk_level = self.score_to_risk_level(job_score)

        review_count = sum(1 for a in ann_evidences if a.review_required)
        error_counts: Dict[str, int] = {}
        for a in ann_evidences:
            if a.possible_error_type != "none":
                error_counts[a.possible_error_type] = error_counts.get(a.possible_error_type, 0) + 1
        if missing_evidences:
            error_counts["missing"] = len(missing_evidences)

        return JobRiskSummary(
            job_id=job_id,
            run_id=run_id,
            job_score=job_score,
            risk_level=risk_level,
            total_images=len(image_summaries),
            processed_images=len(image_summaries),
            failed_images=0,
            total_annotations=len(ann_evidences),
            review_required_count=review_count,
            missing_candidates=len(missing_evidences),
            error_type_counts=error_counts,
            scoring_version=self.scoring_version,
        )
