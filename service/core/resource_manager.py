"""Resource manager for memory, disk, and concurrency quota enforcement."""
from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
from typing import Any, Dict


class ResourceManager:
    """Guards system resources: disk space, concurrent jobs, and inference slots."""

    def __init__(
        self,
        max_concurrent_jobs: int = 2,
        max_inference_slots: int = 4,
        min_free_disk_gb: float = 1.0,
    ) -> None:
        self.job_semaphore = threading.Semaphore(max_concurrent_jobs)
        self.inference_semaphore = threading.Semaphore(max_inference_slots)
        self.min_free_disk_bytes = int(min_free_disk_gb * 1024 * 1024 * 1024)
        self.max_jobs = max_concurrent_jobs
        self.max_inference = max_inference_slots

    def acquire_job_slot(self, blocking: bool = True, timeout: float = 5.0) -> bool:
        """Acquire a concurrent job slot."""
        return self.job_semaphore.acquire(blocking=blocking, timeout=timeout)

    def release_job_slot(self) -> None:
        """Release a concurrent job slot."""
        self.job_semaphore.release()

    def acquire_inference_slot(self, blocking: bool = True, timeout: float = 5.0) -> bool:
        """Acquire an inference execution slot."""
        return self.inference_semaphore.acquire(blocking=blocking, timeout=timeout)

    def release_inference_slot(self) -> None:
        """Release an inference execution slot."""
        self.inference_semaphore.release()

    def check_disk_space(self, path: Path, required_bytes: int = 0) -> bool:
        """Check if target disk has enough free space."""
        try:
            stat = shutil.disk_usage(str(path))
            return (stat.free - required_bytes) >= self.min_free_disk_bytes
        except OSError:
            return True  # Fallback if filesystem cannot be queried

    def get_stats(self) -> Dict[str, Any]:
        return {
            "max_jobs": self.max_jobs,
            "max_inference": self.max_inference,
            "min_free_disk_bytes": self.min_free_disk_bytes,
        }
