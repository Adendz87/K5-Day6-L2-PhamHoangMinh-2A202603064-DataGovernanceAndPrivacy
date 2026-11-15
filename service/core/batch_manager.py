"""Bounded queue and batch manager for backpressure and controlled processing."""
from __future__ import annotations

import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class QueueFullError(Exception):
    """Raised when queue is at maximum capacity (backpressure signal)."""
    pass


@dataclass
class ImageTask:
    job_id: str
    run_id: str
    image_id: str
    image_path: Path
    annotations: List[Dict[str, Any]]


class BatchManager:
    """Manages queueing of images and formation of batches for inference."""

    def __init__(self, batch_size: int = 16, queue_max_size: int = 256) -> None:
        self.batch_size = batch_size
        self.queue_max_size = queue_max_size
        self._queue: queue.Queue[ImageTask] = queue.Queue(maxsize=queue_max_size)

    def submit_image(
        self,
        job_id: str,
        run_id: str,
        image_id: str,
        image_path: Path,
        annotations: List[Dict[str, Any]],
    ) -> None:
        """Submit an assembled image and its annotations into the processing queue."""
        task = ImageTask(
            job_id=job_id,
            run_id=run_id,
            image_id=image_id,
            image_path=image_path,
            annotations=annotations,
        )
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            raise QueueFullError(
                f"Processing queue is full ({self.queue_max_size} items). Apply backpressure."
            )

    def get_batch(self, timeout: float = 0.5) -> List[ImageTask]:
        """Pull up to batch_size tasks from the queue."""
        batch: List[ImageTask] = []
        try:
            first = self._queue.get(timeout=timeout)
            batch.append(first)
        except queue.Empty:
            return []

        while len(batch) < self.batch_size:
            try:
                task = self._queue.get_nowait()
                batch.append(task)
            except queue.Empty:
                break

        return batch

    def mark_task_done(self, count: int = 1) -> None:
        for _ in range(count):
            self._queue.task_done()

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()
