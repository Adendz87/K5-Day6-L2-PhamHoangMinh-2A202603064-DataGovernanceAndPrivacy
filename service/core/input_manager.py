"""Input Manager coordinating binary stream reception, verification, and persistence."""
from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from service.core.annotation_normalizer import AnnotationNormalizer, ValidationError
from service.core.database import JobDatabase
from service.core.job_lifecycle import ImageState, JobLifecycleManager, JobState
from service.core.temporary_storage import TemporaryStorage
from service.core.transfer_session import TransferSession

logger = logging.getLogger("t11.input_manager")


@dataclass
class AckResult:
    ack_type: str
    job_id: str
    run_id: str
    detail: str = ""


@dataclass
class ReceivedStateResult:
    job_id: str
    run_id: str
    completed_image_ids: List[str]
    pending_image_ids: List[str]
    last_chunk_per_image: Dict[str, int]
    confirmed_annotation_batch_ids: List[str]
    current_job_status: str
    last_persisted_result_sequence: int


class InputManager:
    """Handles client input stream events without any gRPC coupling."""

    def __init__(
        self,
        storage: TemporaryStorage,
        database: JobDatabase,
        lifecycle: JobLifecycleManager,
        ack_window_chunks: int = 8,
    ) -> None:
        self.storage = storage
        self.db = database
        self.lifecycle = lifecycle
        self.ack_window_chunks = ack_window_chunks
        self.normalizer = AnnotationNormalizer()
        self._active_sessions: Dict[Tuple[str, str, str], TransferSession] = {}
        self._chunk_locks: Dict[Tuple[str, str, str, int], threading.Lock] = {}
        self._chunk_lock_guard = threading.Lock()

    def _get_chunk_lock(
        self, job_id: str, run_id: str, image_id: str, chunk_index: int
    ) -> threading.Lock:
        key = (job_id, run_id, image_id, chunk_index)
        with self._chunk_lock_guard:
            if key not in self._chunk_locks:
                self._chunk_locks[key] = threading.Lock()
            return self._chunk_locks[key]

    def _cleanup_chunk_locks(
        self, job_id: str, run_id: str, image_id: str
    ) -> None:
        with self._chunk_lock_guard:
            keys_to_del = [
                k for k in self._chunk_locks
                if k[0] == job_id and k[1] == run_id and k[2] == image_id
            ]
            for k in keys_to_del:
                del self._chunk_locks[k]

    def _get_session(self, job_id: str, run_id: str, image_id: str) -> TransferSession:
        key = (job_id, run_id, image_id)
        if key not in self._active_sessions:
            self._active_sessions[key] = TransferSession(
                job_id=job_id,
                run_id=run_id,
                image_id=image_id,
            )
        return self._active_sessions[key]

    def handle_job_metadata(self, metadata: Dict[str, Any]) -> AckResult:
        """Process JobMetadata message with strict idempotency."""
        job_id = str(metadata.get("job_id", "")).strip()
        run_id = str(metadata.get("run_id", "")).strip()
        if not job_id or not run_id:
            raise ValueError("job_id and run_id are required in JobMetadata")

        idempotency_key = metadata.get("metadata", {}).get("idempotency_key", "")
        if idempotency_key:
            existing_key = self.db.check_idempotency_key(idempotency_key)
            if existing_key:
                if existing_key["job_id"] != job_id or existing_key["run_id"] != run_id:
                    raise ValueError(
                        f"Idempotency key '{idempotency_key}' already used by run "
                        f"{existing_key['job_id']}/{existing_key['run_id']}"
                    )
            else:
                self.db.register_idempotency_key(idempotency_key, job_id, run_id)

        existing_job = self.db.get_job(job_id, run_id)
        if existing_job:
            # Idempotent re-submission: do not create new job or reset status
            return AckResult(
                ack_type="ACK_JOB_STARTED",
                job_id=job_id,
                run_id=run_id,
                detail=f"Job {job_id} run {run_id} acknowledged (existing status={existing_job['status']})",
            )

        # Build metadata dict: merge the 'metadata' map field + top-level target_classes
        meta_dict = dict(metadata.get("metadata", {}))
        target_classes = metadata.get("target_classes", [])
        if isinstance(target_classes, (list, tuple)) and target_classes:
            meta_dict["target_classes"] = list(target_classes)

        self.db.create_job(
            job_id=job_id,
            run_id=run_id,
            client_id=str(metadata.get("client_id", "")),
            dataset_name=str(metadata.get("dataset_name", "")),
            dataset_version=str(metadata.get("dataset_version", "")),
            total_images=int(metadata.get("total_images", 0)),
            total_annotations=int(metadata.get("total_annotations", 0)),
            metadata=meta_dict,
            initial_status=JobState.RECEIVING.value,
        )

        return AckResult(
            ack_type="ACK_JOB_STARTED",
            job_id=job_id,
            run_id=run_id,
            detail=f"Job {job_id} run {run_id} initialized",
        )

    def handle_image_chunk(
        self,
        job_id: str,
        run_id: str,
        chunk_dict: Dict[str, Any],
    ) -> Optional[AckResult]:
        """Process an incoming ImageChunk, supporting out-of-order arrival and idempotent retry."""
        image_id = str(chunk_dict.get("image_id", "")).strip()
        chunk_index = int(chunk_dict.get("chunk_index", 0))
        chunk_data = chunk_dict.get("chunk_data", b"")
        is_last = bool(chunk_dict.get("is_last", False))
        total_chunks = int(chunk_dict.get("total_chunks", 0))
        expected_image_sha256 = str(chunk_dict.get("sha256", "")).strip()

        if not isinstance(chunk_data, bytes):
            raise TypeError("chunk_data must be bytes (Base64 is strictly prohibited)")

        chunk_lock = self._get_chunk_lock(job_id, run_id, image_id, chunk_index)
        with chunk_lock:
            effective_total_chunks = total_chunks
            if is_last and (chunk_index + 1) > effective_total_chunks:
                effective_total_chunks = chunk_index + 1

            self.db.ensure_image(
                job_id,
                run_id,
                image_id,
                total_chunks=effective_total_chunks,
                expected_sha256=expected_image_sha256 if is_last else "",
            )
            chunk_sha = hashlib.sha256(chunk_data).hexdigest()

            # Pre-check chunk status in SQLite
            status, recorded_sha = self.db.check_chunk_status(
                job_id, run_id, image_id, chunk_index, chunk_sha
            )

            # Case C — Conflicting duplicate: reject immediately with ValueError
            if status == "DUPLICATE_CONFLICT":
                raise ValueError(
                    f"Conflicting chunk content for chunk {chunk_index} of image {image_id}: "
                    f"recorded={recorded_sha}, incoming={chunk_sha}"
                )

            session = self._get_session(job_id, run_id, image_id)
            if effective_total_chunks > 0:
                session.total_chunks = max(session.total_chunks, effective_total_chunks)
            if is_last:
                session.is_last_seen = True
            if expected_image_sha256:
                session.file_checksum = expected_image_sha256

            is_new_chunk = False
            if status == "DUPLICATE_IDENTICAL":
                # Case B — Identical duplicate: idempotent no-op
                pass
            else:
                # Case A — New chunk
                file_written = False
                try:
                    self.storage.write_chunk(
                        job_id, run_id, image_id, chunk_index, chunk_data, overwrite=False
                    )
                    file_written = True

                    is_new_chunk = self.db.record_chunk(
                        job_id, run_id, image_id, chunk_index, len(chunk_data), chunk_sha
                    )
                except Exception:
                    # If database recording fails or any exception occurs after writing chunk file,
                    # rollback by cleaning up the newly written chunk file so state remains consistent.
                    if file_written:
                        self.storage.delete_chunk(job_id, run_id, image_id, chunk_index)
                    raise

                session.add_chunk(chunk_index, offset=0, size=len(chunk_data), checksum=chunk_sha)

            # Check if image has already been assembled and verified
            if session.is_assembled and session.is_verified:
                return AckResult(
                    ack_type="ACK_IMAGE_RECEIVED",
                    job_id=job_id,
                    run_id=run_id,
                    detail=f"Image {image_id} fully received and verified ({session.total_chunks} chunks)",
                )

            # Check completion condition: all chunks 0..total_chunks-1 must be present AND is_last must have been seen
            is_complete, tot, exp_sha = self.db.is_image_complete(job_id, run_id, image_id)
            if not exp_sha and session.file_checksum:
                exp_sha = session.file_checksum
            if not exp_sha and expected_image_sha256:
                exp_sha = expected_image_sha256

            is_last_seen = session.is_last_seen or is_last or bool(exp_sha)
            if is_last_seen and is_complete and tot > 0:
                # All chunks from 0 to tot-1 have been received and verified present
                image_path = self.storage.assemble_image(
                    job_id, run_id, image_id, tot, expected_sha256=exp_sha
                )
                self.db.mark_image_received(
                    job_id, run_id, image_id, exp_sha, str(image_path)
                )
                session.is_assembled = True
                session.is_verified = True
                self._cleanup_chunk_locks(job_id, run_id, image_id)
                return AckResult(
                    ack_type="ACK_IMAGE_RECEIVED",
                    job_id=job_id,
                    run_id=run_id,
                    detail=f"Image {image_id} fully received and verified ({tot} chunks)",
                )

            # Windowed ACK for chunks
            if is_new_chunk and (chunk_index + 1) % self.ack_window_chunks == 0:
                return AckResult(
                    ack_type="ACK_CHUNK",
                    job_id=job_id,
                    run_id=run_id,
                    detail=f"Chunk {chunk_index} for image {image_id} received",
                )

            return None

    def handle_annotation_batch(
        self,
        job_id: str,
        run_id: str,
        batch_dict: Dict[str, Any],
    ) -> Tuple[AckResult, List[ValidationError]]:
        """Process and persist an AnnotationBatch."""
        batch_id = str(batch_dict.get("batch_id", "")).strip()
        items = batch_dict.get("items", [])

        normalized, errors = self.normalizer.normalize_grpc_batch(batch_dict)

        clean_items = []
        for n in normalized:
            clean_items.append(
                {
                    "ann_id": n.ann_id,
                    "image_id": n.image_id,
                    "class_name": n.class_name,
                    "confidence": n.confidence,
                    "bbox": {"x": n.x, "y": n.y, "w": n.w, "h": n.h},
                    "extra_fields": n.attributes,
                }
            )

        self.db.save_annotation_batch(job_id, run_id, batch_id, clean_items)

        ack = AckResult(
            ack_type="ACK_ANNOTATIONS_RECEIVED",
            job_id=job_id,
            run_id=run_id,
            detail=f"Batch {batch_id} with {len(clean_items)} annotations stored",
        )
        return ack, errors

    def handle_end_of_input(
        self,
        job_id: str,
        run_id: str,
        eoi_dict: Dict[str, Any],
    ) -> AckResult:
        """Process EndOfInput: transition job to INPUT_COMPLETE idempotently."""
        current_state = self.lifecycle.get_job_state(job_id, run_id)
        if current_state not in (JobState.INPUT_COMPLETE, JobState.PROCESSING, JobState.COMPLETED):
            self.lifecycle.transition_job(
                job_id, run_id, JobState.VALIDATING, reason="EndOfInput received"
            )
            self.lifecycle.transition_job(
                job_id, run_id, JobState.INPUT_COMPLETE, reason="Input validation passed"
            )

        images_sent = int(eoi_dict.get("images_sent", 0))
        annotations_sent = int(eoi_dict.get("annotations_sent", 0))
        return AckResult(
            ack_type="ACK_INPUT_COMPLETE",
            job_id=job_id,
            run_id=run_id,
            detail=f"Input complete confirmed: {images_sent} images, {annotations_sent} annotations",
        )

    def handle_resume_request(self, resume_dict: Dict[str, Any]) -> ReceivedStateResult:
        """Process ResumeRequest: compile current server state for client recovery."""
        job_id = str(resume_dict.get("job_id", "")).strip()
        run_id = str(resume_dict.get("run_id", "")).strip()

        job = self.db.get_job(job_id, run_id)
        current_status = job["status"] if job else "CREATED"

        completed = self.db.get_image_ids_by_status(
            job_id,
            run_id,
            [
                "IMAGE_RECEIVED",
                "IMAGE_DONE",
                "IMAGE_RESULT_PERSISTED",
                "IMAGE_SCORING",
                "IMAGE_MATCHING",
                "IMAGE_YOLO_INFERENCE",
            ],
        )
        pending = self.db.get_image_ids_by_status(job_id, run_id, "IMAGE_RECEIVING")
        last_chunks = self.db.get_last_chunk_indices(job_id, run_id)
        max_seq = self.db.get_max_persisted_result_sequence(job_id, run_id)

        return ReceivedStateResult(
            job_id=job_id,
            run_id=run_id,
            completed_image_ids=completed,
            pending_image_ids=pending,
            last_chunk_per_image=last_chunks,
            confirmed_annotation_batch_ids=[],
            current_job_status=current_status,
            last_persisted_result_sequence=max_seq,
        )

    def handle_cancel_request(self, cancel_dict: Dict[str, Any]) -> None:
        """Process CancelRequest."""
        job_id = str(cancel_dict.get("job_id", "")).strip()
        run_id = str(cancel_dict.get("run_id", "")).strip()
        reason = str(cancel_dict.get("reason", "Cancelled by client"))
        self.lifecycle.transition_job(job_id, run_id, JobState.CANCELLED, reason=reason)
