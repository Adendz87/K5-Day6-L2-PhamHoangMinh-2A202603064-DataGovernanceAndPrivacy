"""gRPC Servicer implementing the single Evaluate bidirectional streaming RPC."""
from __future__ import annotations

import logging
from typing import Any, Dict, Generator, Iterator

import grpc

from service.core.job_lifecycle import JobState
from service.core.processing_engine import ProcessingEngine
from service.generated import pb2, pb2_grpc

logger = logging.getLogger("t11.grpc_handler")


class T11EvaluatorServicer(pb2_grpc.T11EvaluatorServicer):
    """Implementation of T11Evaluator service with single Evaluate bidirectional streaming RPC."""

    def __init__(self, engine: ProcessingEngine) -> None:
        self.engine = engine

    def Evaluate(
        self,
        request_iterator: Iterator[pb2.EvaluateRequest],
        context: grpc.ServicerContext,
    ) -> Generator[pb2.EvaluateResponse, None, None]:
        """Single business RPC for the entire evaluation lifecycle.

        Handles streaming inputs (metadata, chunks, annotations) and streams back
        acknowledgements, progress, and results.
        """
        job_id = ""
        run_id = ""
        input_completed = False

        try:
            for request in request_iterator:
                payload_case = request.WhichOneof("payload")

                if payload_case == "job_metadata":
                    meta = request.job_metadata
                    job_id = meta.job_id
                    run_id = meta.run_id
                    ack = self.engine.input_manager.handle_job_metadata(
                        {
                            "job_id": meta.job_id,
                            "run_id": meta.run_id,
                            "client_id": meta.client_id,
                            "dataset_name": meta.dataset_name,
                            "dataset_version": meta.dataset_version,
                            "total_images": meta.total_images,
                            "total_annotations": meta.total_annotations,
                            "metadata": dict(meta.metadata),
                            "target_classes": list(meta.target_classes),
                        }
                    )
                    yield pb2.EvaluateResponse(
                        acknowledgement=pb2.Acknowledgement(
                            job_id=ack.job_id,
                            run_id=ack.run_id,
                            ack_type=getattr(pb2.AckType, ack.ack_type),
                            detail=ack.detail,
                        )
                    )

                elif payload_case == "image_chunk":
                    chunk = request.image_chunk
                    if not job_id or not run_id:
                        active_job = self.engine.db.find_job_by_image_id(chunk.image_id)
                        if active_job:
                            job_id = active_job["job_id"]
                            run_id = active_job["run_id"]
                        else:
                            raise ValueError(
                                f"Unknown image_id '{chunk.image_id}' in stream without preceding JobMetadata or ResumeRequest"
                            )

                    ack = self.engine.input_manager.handle_image_chunk(
                        job_id,
                        run_id,
                        {
                            "image_id": chunk.image_id,
                            "chunk_index": chunk.chunk_index,
                            "chunk_data": chunk.chunk_data,
                            "is_last": chunk.is_last,
                            "total_chunks": chunk.total_chunks,
                            "sha256": chunk.sha256,
                        },
                    )
                    if ack:
                        yield pb2.EvaluateResponse(
                            acknowledgement=pb2.Acknowledgement(
                                job_id=ack.job_id,
                                run_id=ack.run_id,
                                ack_type=getattr(pb2.AckType, ack.ack_type),
                                detail=ack.detail,
                            )
                        )

                elif payload_case == "annotation_batch":
                    batch = request.annotation_batch
                    if (not job_id or not run_id) and batch.items:
                        active_job = self.engine.db.find_job_by_image_id(batch.items[0].image_id)
                        if active_job:
                            job_id = active_job["job_id"]
                            run_id = active_job["run_id"]

                    items = []
                    for it in batch.items:
                        items.append(
                            {
                                "ann_id": it.ann_id,
                                "image_id": it.image_id,
                                "class_name": it.class_name,
                                "confidence": it.confidence,
                                "bbox": {
                                    "x": it.bbox.x,
                                    "y": it.bbox.y,
                                    "w": it.bbox.w,
                                    "h": it.bbox.h,
                                },
                                "extra_fields": dict(it.extra_fields),
                            }
                        )
                    ack, errors = self.engine.input_manager.handle_annotation_batch(
                        job_id,
                        run_id,
                        {"batch_id": batch.batch_id, "items": items},
                    )
                    yield pb2.EvaluateResponse(
                        acknowledgement=pb2.Acknowledgement(
                            job_id=ack.job_id,
                            run_id=ack.run_id,
                            ack_type=getattr(pb2.AckType, ack.ack_type),
                            detail=ack.detail,
                        )
                    )

                elif payload_case == "end_of_input":
                    eoi = request.end_of_input
                    ack = self.engine.input_manager.handle_end_of_input(
                        job_id,
                        run_id,
                        {
                            "images_sent": eoi.images_sent,
                            "chunks_sent": eoi.chunks_sent,
                            "annotations_sent": eoi.annotations_sent,
                        },
                    )
                    yield pb2.EvaluateResponse(
                        acknowledgement=pb2.Acknowledgement(
                            job_id=ack.job_id,
                            run_id=ack.run_id,
                            ack_type=getattr(pb2.AckType, ack.ack_type),
                            detail=ack.detail,
                        )
                    )
                    input_completed = True
                    break

                elif payload_case == "resume_request":
                    rr = request.resume_request
                    job_id = rr.job_id
                    run_id = rr.run_id
                    st = self.engine.input_manager.handle_resume_request(
                        {"job_id": rr.job_id, "run_id": rr.run_id}
                    )
                    yield pb2.EvaluateResponse(
                        received_state=pb2.ReceivedState(
                            job_id=st.job_id,
                            run_id=st.run_id,
                            completed_image_ids=st.completed_image_ids,
                            pending_image_ids=st.pending_image_ids,
                            last_chunk_per_image=st.last_chunk_per_image,
                            confirmed_annotation_batch_ids=st.confirmed_annotation_batch_ids,
                            current_job_status=getattr(
                                pb2.JobStatus, st.current_job_status, pb2.JobStatus.CREATED
                            ),
                            last_persisted_result_sequence=st.last_persisted_result_sequence,
                        )
                    )

                elif payload_case == "cancel_request":
                    cr = request.cancel_request
                    self.engine.input_manager.handle_cancel_request(
                        {"job_id": cr.job_id, "run_id": run_id, "reason": cr.reason}
                    )
                    yield pb2.EvaluateResponse(
                        progress=pb2.EvaluationProgress(
                            job_id=cr.job_id,
                            run_id=run_id,
                            job_status=pb2.JobStatus.CANCELLED,
                            message=f"Job cancelled: {cr.reason}",
                        )
                    )
                    return

                elif payload_case == "result_ack":
                    rack = request.result_ack
                    self.engine.result_storage.mark_acknowledged(
                        rack.job_id, rack.run_id, rack.last_acknowledged_result_sequence
                    )

            # If input is complete, trigger the processing pipeline and stream results
            if input_completed and job_id and run_id:
                for response in self.engine.process_job(job_id, run_id):
                    yield response

        except Exception as e:
            logger.exception("Error processing Evaluate stream for job %s, run %s: %s", job_id, run_id, e)
            if job_id and run_id:
                try:
                    self.engine.lifecycle.transition_job(job_id, run_id, JobState.FAILED, reason=str(e))
                except Exception:
                    pass
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            yield pb2.EvaluateResponse(
                evaluation_error=pb2.EvaluationError(
                    job_id=job_id,
                    run_id=run_id,
                    error_code="INTERNAL_ERROR",
                    message=str(e),
                    is_fatal=True,
                )
            )
