"""Service crash and disconnect recovery manager."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from service.core.database import JobDatabase
from service.core.temporary_storage import TemporaryStorage


@dataclass
class JobRecoveryPlan:
    job_id: str
    run_id: str
    current_status: str
    pending_image_ids: List[str]
    completed_image_ids: List[str]
    undelivered_results_count: int


class RunRecovery:
    """Discovers interrupted runs and plans recovery on service startup."""

    def __init__(self, db: JobDatabase, storage: TemporaryStorage) -> None:
        self.db = db
        self.storage = storage

    def find_incomplete_jobs(self) -> List[Dict[str, Any]]:
        """Find jobs that were in progress when the service crashed or stopped."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status NOT IN ('COMPLETED', 'FAILED', 'CANCELLED');
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def create_recovery_plan(self, job_id: str, run_id: str) -> JobRecoveryPlan:
        """Inspect state and create a plan to resume processing."""
        job = self.db.get_job(job_id, run_id)
        status = job["status"] if job else "UNKNOWN"

        completed = self.db.get_image_ids_by_status(job_id, run_id, "IMAGE_RECEIVED")
        pending = self.db.get_image_ids_by_status(job_id, run_id, "IMAGE_RECEIVING")
        undelivered = self.db.get_results_from_sequence(job_id, run_id, 1)

        return JobRecoveryPlan(
            job_id=job_id,
            run_id=run_id,
            current_status=status,
            pending_image_ids=pending,
            completed_image_ids=completed,
            undelivered_results_count=len(undelivered),
        )
