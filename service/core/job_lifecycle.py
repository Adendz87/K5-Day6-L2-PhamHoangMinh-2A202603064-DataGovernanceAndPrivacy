"""Lifecycle management and state machines for jobs, images, and results."""
from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Set

from service.core.database import JobDatabase


class JobState(str, Enum):
    CREATED = "CREATED"
    RECEIVING = "RECEIVING"
    VALIDATING = "VALIDATING"
    INPUT_COMPLETE = "INPUT_COMPLETE"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    AGGREGATING = "AGGREGATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ImageState(str, Enum):
    PENDING = "IMAGE_PENDING"
    RECEIVING = "IMAGE_RECEIVING"
    RECEIVED = "IMAGE_RECEIVED"
    VALIDATING = "IMAGE_VALIDATING"
    YOLO_INFERENCE = "IMAGE_YOLO_INFERENCE"
    MATCHING = "IMAGE_MATCHING"
    SAM3_INFERENCE = "IMAGE_SAM3_INFERENCE"
    SCORING = "IMAGE_SCORING"
    RESULT_PERSISTED = "IMAGE_RESULT_PERSISTED"
    DONE = "IMAGE_DONE"
    FAILED = "IMAGE_FAILED"


class ResultDeliveryState(str, Enum):
    PENDING = "PENDING"
    PERSISTED = "PERSISTED"
    SENT = "SENT"
    ACKNOWLEDGED = "ACKNOWLEDGED"


class JobLifecycleManager:
    """Enforces legal state transitions for jobs and images."""

    # Explicit legal job state transition table
    LEGAL_JOB_TRANSITIONS: Dict[JobState, Set[JobState]] = {
        JobState.CREATED: {JobState.RECEIVING, JobState.FAILED, JobState.CANCELLED},
        JobState.RECEIVING: {
            JobState.RECEIVING,
            JobState.VALIDATING,
            JobState.FAILED,
            JobState.CANCELLED,
        },
        JobState.VALIDATING: {
            JobState.INPUT_COMPLETE,
            JobState.FAILED,
            JobState.CANCELLED,
        },
        JobState.INPUT_COMPLETE: {
            JobState.QUEUED,
            JobState.PROCESSING,
            JobState.FAILED,
            JobState.CANCELLED,
        },
        JobState.QUEUED: {
            JobState.PROCESSING,
            JobState.FAILED,
            JobState.CANCELLED,
        },
        JobState.PROCESSING: {
            JobState.AGGREGATING,
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
        },
        JobState.AGGREGATING: {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
        },
        JobState.COMPLETED: set(),  # Terminal
        JobState.FAILED: set(),  # Terminal
        JobState.CANCELLED: set(),  # Terminal
    }

    def __init__(self, db: JobDatabase) -> None:
        self.db = db

    def can_transition_job(self, from_state: JobState, to_state: JobState) -> bool:
        if from_state == to_state:
            return True
        allowed = self.LEGAL_JOB_TRANSITIONS.get(from_state, set())
        return to_state in allowed

    def transition_job(
        self,
        job_id: str,
        run_id: str,
        to_state: JobState,
        reason: str = "",
    ) -> bool:
        """Attempt to transition job state. Returns True if successful."""
        job = self.db.get_job(job_id, run_id)
        if not job:
            return False

        current_state = JobState(job["status"])
        if not self.can_transition_job(current_state, to_state):
            return False

        self.db.update_job_status(job_id, run_id, to_state.value, reason)
        return True

    def transition_image(
        self,
        job_id: str,
        run_id: str,
        image_id: str,
        to_state: ImageState,
        error_msg: str = "",
    ) -> None:
        """Update per-image processing state."""
        self.db.update_image_status(job_id, run_id, image_id, to_state.value, error_msg)

    def get_job_state(self, job_id: str, run_id: str) -> Optional[JobState]:
        job = self.db.get_job(job_id, run_id)
        return JobState(job["status"]) if job else None
