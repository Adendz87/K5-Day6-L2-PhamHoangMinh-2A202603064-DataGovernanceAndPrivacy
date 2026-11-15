"""Result Manager translating internal results into Protobuf EvaluateResponse messages."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from service.analysis.error_analysis import (
    MissingCandidateEvidence,
    ScoredAnnotationEvidence,
)
from service.analysis.risk_scorer import ImageRiskSummary, JobRiskSummary
from service.generated import pb2


class ResultManager:
    """Constructs Protobuf response messages from analyzed results."""

    @staticmethod
    def _to_proto_risk(risk_str: str) -> pb2.RiskLevel:
        return getattr(pb2.RiskLevel, risk_str, pb2.RiskLevel.RISK_UNKNOWN)

    @staticmethod
    def _to_proto_bbox(b: Any) -> pb2.BoundingBox:
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            return pb2.BoundingBox(x=float(b[0]), y=float(b[1]), w=float(b[2]), h=float(b[3]))
        elif isinstance(b, dict):
            return pb2.BoundingBox(
                x=float(b.get("x", 0.0)),
                y=float(b.get("y", 0.0)),
                w=float(b.get("w", 0.0)),
                h=float(b.get("h", 0.0)),
            )
        return pb2.BoundingBox()

    def make_annotation_response(
        self,
        job_id: str,
        run_id: str,
        seq: int,
        evidence: ScoredAnnotationEvidence,
    ) -> pb2.EvaluateResponse:
        ed = evidence.evidence_dict
        ev_proto = pb2.Evidence(
            human_bbox=self._to_proto_bbox(ed.get("human_bbox")),
            model_bbox=self._to_proto_bbox(ed.get("model_bbox")),
            iou=float(ed.get("iou", 0.0)),
            class_disagreement=bool(ed.get("class_disagreement", False)),
            human_class=str(ed.get("human_class", "")),
            predicted_class=str(ed.get("predicted_class", "")),
            prediction_confidence=float(ed.get("prediction_confidence", 0.0)),
            match_stage=str(ed.get("match_stage", "")),
            sam3_mask_ref=str(ed.get("sam3_mask_ref", "")),
            reason_code=evidence.reason_code,
            geometric_diff=float(ed.get("geometric_diff", 0.0)),
        )

        res = pb2.AnnotationResult(
            job_id=job_id,
            run_id=run_id,
            result_sequence=seq,
            ann_id=evidence.ann_id,
            image_id=evidence.image_id,
            suspicion_score=evidence.risk_score,
            risk_level=self._to_proto_risk(evidence.risk_level),
            evidence=ev_proto,
            reason_code=evidence.reason_code,
        )
        return pb2.EvaluateResponse(annotation_result=res)

    def make_image_response(
        self,
        job_id: str,
        run_id: str,
        seq: int,
        summary: ImageRiskSummary,
        evidence_list: Optional[List[pb2.Evidence]] = None,
    ) -> pb2.EvaluateResponse:
        res = pb2.ImageResult(
            job_id=job_id,
            run_id=run_id,
            result_sequence=seq,
            image_id=summary.image_id,
            risk_level=self._to_proto_risk(summary.risk_level),
            suspicion_score=summary.image_score,
            error_ann_ids=[],
            evidence=evidence_list or [],
            final_status=pb2.ImageStatus.IMAGE_DONE,
        )
        return pb2.EvaluateResponse(image_result=res)

    def make_missing_candidate_response(
        self,
        job_id: str,
        run_id: str,
        seq: int,
        missing: MissingCandidateEvidence,
    ) -> pb2.EvaluateResponse:
        res = pb2.MissingCandidate(
            job_id=job_id,
            run_id=run_id,
            result_sequence=seq,
            image_id=missing.image_id,
            item_id=missing.item_id,
            model_bbox=self._to_proto_bbox(missing.model_bbox),
            prediction_confidence=missing.prediction_confidence,
            predicted_class=missing.predicted_class,
            suspicion_score=missing.risk_score,
        )
        return pb2.EvaluateResponse(missing_candidate=res)

    def make_job_result_response(
        self,
        job_id: str,
        run_id: str,
        seq: int,
        job_payload: Dict[str, Any],
    ) -> pb2.EvaluateResponse:
        res = pb2.JobResult(
            job_id=job_id,
            run_id=run_id,
            result_sequence=seq,
            total_annotations_evaluated=job_payload.get("total_annotations_evaluated", 0),
            total_images_evaluated=job_payload.get("total_images_evaluated", 0),
            total_errors_flagged=job_payload.get("total_errors_flagged", 0),
            overall_risk=self._to_proto_risk(job_payload.get("overall_risk", "RISK_LOW")),
            top_item_ids=job_payload.get("top_item_ids", []),
            item_scores=job_payload.get("item_scores", {}),
            item_reason_codes=job_payload.get("item_reason_codes", {}),
        )
        return pb2.EvaluateResponse(job_result=res)

    def make_job_completed_response(
        self,
        job_id: str,
        run_id: str,
        seq: int,
        duration_ms: int = 0,
        output_path: str = "",
    ) -> pb2.EvaluateResponse:
        res = pb2.JobCompleted(
            job_id=job_id,
            run_id=run_id,
            result_sequence=seq,
            final_status=pb2.JobStatus.COMPLETED,
            duration_ms=duration_ms,
            output_path=output_path,
        )
        return pb2.EvaluateResponse(job_completed=res)
