"""Result Aggregator formatting final rankings and job outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from service.analysis.error_analysis import (
    MissingCandidateEvidence,
    ScoredAnnotationEvidence,
)
from service.analysis.risk_scorer import ImageRiskSummary, JobRiskSummary


class ResultAggregator:
    """Aggregates all evaluation data into a ranked review queue and summary."""

    @staticmethod
    def rank_items(
        ann_evidences: List[ScoredAnnotationEvidence],
        missing_evidences: List[MissingCandidateEvidence],
    ) -> List[Tuple[str, float, str, str]]:
        """Rank all items (annotations + image missing candidates) by suspicion score descending.

        Returns list of (item_id, score, error_type, suggested_action).
        """
        items: List[Tuple[str, float, str, str]] = []

        for a in ann_evidences:
            items.append((a.ann_id, a.risk_score, a.possible_error_type, a.suggested_action))

        for m in missing_evidences:
            items.append((m.item_id, m.risk_score, "missing", m.suggested_action))

        # Stable sort by score descending
        items.sort(key=lambda x: x[1], reverse=True)
        return items

    @staticmethod
    def build_job_payload(
        job_summary: JobRiskSummary,
        ranked_items: List[Tuple[str, float, str, str]],
    ) -> Dict[str, Any]:
        """Build dictionary suitable for Protobuf JobResult conversion."""
        top_ids = [item[0] for item in ranked_items]
        item_scores = {item[0]: item[1] for item in ranked_items}
        item_reasons = {item[0]: item[2] for item in ranked_items}

        return {
            "job_id": job_summary.job_id,
            "run_id": job_summary.run_id,
            "total_annotations_evaluated": job_summary.total_annotations,
            "total_images_evaluated": job_summary.total_images,
            "total_errors_flagged": job_summary.review_required_count + job_summary.missing_candidates,
            "overall_risk": job_summary.risk_level,
            "top_item_ids": top_ids,
            "item_scores": item_scores,
            "item_reason_codes": item_reasons,
        }
