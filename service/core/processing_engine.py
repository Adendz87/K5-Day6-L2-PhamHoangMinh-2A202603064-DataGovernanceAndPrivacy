"""Core Processing Engine for T11 Service.

Orchestrates input reception, YOLO & SAM 3 inference, matching, error analysis,
risk scoring, and streaming result generation.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from service.adapters.sam3_adapter import create_sam3_adapter
from service.adapters.yolo_adapter import YOLOPrediction, create_yolo_adapter
from service.analysis.error_analysis import (
    ErrorAnalyzer,
    MissingCandidateEvidence,
    ScoredAnnotationEvidence,
)
from service.analysis.matching_adapter import MatchingAdapter
from service.analysis.risk_scorer import ImageRiskSummary, RiskScorer
from service.core.annotation_normalizer import NormalizedAnnotation
from service.core.batch_manager import BatchManager
from service.core.database import JobDatabase
from service.core.input_manager import InputManager
from service.core.job_lifecycle import ImageState, JobLifecycleManager, JobState
from service.core.resource_manager import ResourceManager
from service.core.result_storage import ResultStorage
from service.core.temporary_storage import TemporaryStorage
from service.generated import pb2
from service.result.aggregator import ResultAggregator
from service.result.result_manager import ResultManager
from service.server.config import ServiceConfig

logger = logging.getLogger("t11.processing_engine")


class ProcessingEngine:
    """Complete, self-contained processing pipeline for evaluation jobs."""

    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self.storage = TemporaryStorage(
            config.storage.temp_path,
            max_size_gb=config.transfer.max_job_size_gb,
        )
        self.db = JobDatabase(config.storage.database_path)
        self.lifecycle = JobLifecycleManager(self.db)
        self.input_manager = InputManager(
            self.storage,
            self.db,
            self.lifecycle,
            ack_window_chunks=config.transfer.ack_window_chunks,
        )
        self.result_storage = ResultStorage(config.storage.result_path, self.db)
        self.resource_manager = ResourceManager(
            max_concurrent_jobs=config.processing.max_concurrent_jobs,
        )
        self.batch_manager = BatchManager(
            batch_size=config.processing.batch_size,
            queue_max_size=config.processing.queue_max_size,
        )

        # Adapters
        self.yolo_adapter = create_yolo_adapter(config.models.yolo)
        self.sam3_adapter = (
            create_sam3_adapter(config.models.sam3) if config.models.sam3.enabled else None
        )
        self.matching_adapter = MatchingAdapter(
            match_iou=config.scoring.match_iou,
            shift_gate=config.scoring.shift_gate,
        )
        self.error_analyzer = ErrorAnalyzer(
            w_dup=config.scoring.w_dup,
            w_swap=config.scoring.w_swap,
            w_shift=config.scoring.w_shift,
            w_missing=config.scoring.w_missing,
            w_unmatched=config.scoring.w_unmatched,
            dup_iou=config.scoring.dup_iou,
            orphan_overlap=config.scoring.orphan_overlap,
            missing_conf_floor=config.scoring.missing_conf_floor,
            sam3_adapter=self.sam3_adapter,
        )
        self.risk_scorer = RiskScorer(scoring_version=config.scoring.version)
        self.result_manager = ResultManager()

    def process_job(
        self,
        job_id: str,
        run_id: str,
    ) -> Generator[pb2.EvaluateResponse, None, None]:
        """Execute the full evaluation pipeline for a job that has received EndOfInput.

        Yields EvaluateResponse messages: progress, per-annotation, per-image, missing candidates,
        job result, and job completed.
        """
        import json as _json

        start_time = time.perf_counter()
        self.lifecycle.transition_job(job_id, run_id, JobState.PROCESSING)

        # Retrieve target_classes from job metadata (stored in metadata_json)
        job_record = self.db.get_job(job_id, run_id)
        target_classes: Optional[List[str]] = None
        if job_record:
            try:
                meta = _json.loads(job_record.get("metadata_json", "{}"))
                tc = meta.get("target_classes")
                if isinstance(tc, list) and tc:
                    target_classes = [str(c) for c in tc]
            except Exception:
                pass
        if target_classes:
            logger.info("Job %s using target_classes=%s for YOLO inference", job_id, target_classes)
        else:
            logger.info("Job %s using all YOLO classes (no target_classes filter)", job_id)

        images = self.db.get_all_images(job_id, run_id)
        total_images = len(images)

        yield pb2.EvaluateResponse(
            progress=pb2.EvaluationProgress(
                job_id=job_id,
                run_id=run_id,
                job_status=pb2.JobStatus.PROCESSING,
                images_processed=0,
                images_total=total_images,
                message="Starting inference and evaluation pipeline",
            )
        )

        all_ann_evidences: List[ScoredAnnotationEvidence] = []
        all_missing_evidences: List[MissingCandidateEvidence] = []
        all_image_summaries: List[ImageRiskSummary] = []

        for idx, img_record in enumerate(images, start=1):
            image_id = img_record["image_id"]
            img_path = Path(img_record["file_path"])

            # 1. Image inference state
            self.lifecycle.transition_image(
                job_id, run_id, image_id, ImageState.YOLO_INFERENCE
            )

            # Get image annotations from DB
            raw_anns = self.db.get_annotations_for_image(job_id, run_id, image_id)
            normalized_anns = [
                NormalizedAnnotation(
                    ann_id=r["ann_id"],
                    image_id=r["image_id"],
                    class_name=r["class_name"],
                    class_id=0,
                    x=float(r["x"]),
                    y=float(r["y"]),
                    w=float(r["w"]),
                    h=float(r["h"]),
                    image_width=320,
                    image_height=180,
                    confidence=float(r["confidence"]),
                )
                for r in raw_anns
            ]

            # 2. YOLO local inference
            preds = self.yolo_adapter.predict(img_path, 320, 180, target_classes=target_classes)

            # 3. Matching
            self.lifecycle.transition_image(job_id, run_id, image_id, ImageState.MATCHING)
            match_res = self.matching_adapter.match_image_data(
                image_id, normalized_anns, preds
            )

            # 4. Error analysis & scoring
            self.lifecycle.transition_image(job_id, run_id, image_id, ImageState.SCORING)
            ann_evs, miss_evs = self.error_analyzer.analyze_image(
                match_res, image_path=img_path
            )
            img_summary = self.risk_scorer.compute_image_risk(
                image_id, ann_evs, miss_evs
            )

            all_ann_evidences.extend(ann_evs)
            all_missing_evidences.extend(miss_evs)
            all_image_summaries.append(img_summary)

            self.lifecycle.transition_image(
                job_id, run_id, image_id, ImageState.RESULT_PERSISTED
            )

            # 5. Persist and yield per-annotation results
            for a_ev in ann_evs:
                seq = self.result_storage.save_annotation_result(
                    job_id, run_id, {
                        "ann_id": a_ev.ann_id,
                        "image_id": a_ev.image_id,
                        "risk_score": a_ev.risk_score,
                        "risk_level": a_ev.risk_level,
                        "possible_error_type": a_ev.possible_error_type,
                        "suggested_action": a_ev.suggested_action,
                        "review_required": a_ev.review_required,
                    }
                )
                yield self.result_manager.make_annotation_response(
                    job_id, run_id, seq, a_ev
                )

            # 6. Persist and yield missing candidates
            for m_ev in miss_evs:
                seq = self.result_storage.save_missing_candidate(
                    job_id, run_id, {
                        "image_id": m_ev.image_id,
                        "item_id": m_ev.item_id,
                        "risk_score": m_ev.risk_score,
                        "risk_level": m_ev.risk_level,
                        "predicted_class": m_ev.predicted_class,
                        "prediction_confidence": m_ev.prediction_confidence,
                    }
                )
                yield self.result_manager.make_missing_candidate_response(
                    job_id, run_id, seq, m_ev
                )

            # 7. Persist and yield image result
            img_seq = self.result_storage.save_image_result(
                job_id, run_id, {
                    "image_id": img_summary.image_id,
                    "image_score": img_summary.image_score,
                    "risk_level": img_summary.risk_level,
                    "annotation_count": img_summary.annotation_count,
                    "review_required": img_summary.review_required,
                }
            )
            yield self.result_manager.make_image_response(
                job_id, run_id, img_seq, img_summary
            )

            self.lifecycle.transition_image(job_id, run_id, image_id, ImageState.DONE)

            # Stream progress
            yield pb2.EvaluateResponse(
                progress=pb2.EvaluationProgress(
                    job_id=job_id,
                    run_id=run_id,
                    image_id=image_id,
                    image_status=pb2.ImageStatus.IMAGE_DONE,
                    job_status=pb2.JobStatus.PROCESSING,
                    images_processed=idx,
                    images_total=total_images,
                    message=f"Evaluated image {image_id} ({idx}/{total_images})",
                )
            )

        # 8. Aggregation
        self.lifecycle.transition_job(job_id, run_id, JobState.AGGREGATING)
        job_summary = self.risk_scorer.compute_job_risk(
            job_id,
            run_id,
            all_image_summaries,
            all_ann_evidences,
            all_missing_evidences,
        )
        ranked_items = ResultAggregator.rank_items(
            all_ann_evidences, all_missing_evidences
        )
        job_payload = ResultAggregator.build_job_payload(job_summary, ranked_items)

        job_seq = self.result_storage.save_job_result(job_id, run_id, job_payload)
        yield self.result_manager.make_job_result_response(
            job_id, run_id, job_seq, job_payload
        )

        # 9. Completion
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        self.lifecycle.transition_job(job_id, run_id, JobState.COMPLETED)
        completed_seq = self.result_storage.save_job_completed(
            job_id, run_id, {
                "duration_ms": duration_ms,
                "final_status": "COMPLETED",
            }
        )
        yield self.result_manager.make_job_completed_response(
            job_id, run_id, completed_seq, duration_ms=duration_ms
        )
