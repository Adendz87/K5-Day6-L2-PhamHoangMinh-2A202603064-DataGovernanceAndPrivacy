"""Temporary storage manager for binary chunks and assembled image files."""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path


class TemporaryStorage:
    """Manages filesystem storage for incoming binary chunks and assembled images."""

    def __init__(self, base_path: str | Path, max_size_gb: float = 10.0) -> None:
        self.base_path = Path(base_path).resolve()
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, relative_path: Path) -> Path:
        """Resolve path and verify it stays inside base_path (prevents path traversal)."""
        resolved = (self.base_path / relative_path).resolve()
        if not str(resolved).startswith(str(self.base_path)):
            raise ValueError(f"Path traversal detected: {relative_path}")
        return resolved

    def get_job_dir(self, job_id: str, run_id: str) -> Path:
        """Get base directory for a specific job/run."""
        return self._safe_path(Path(job_id) / run_id)

    def get_chunks_dir(self, job_id: str, run_id: str, image_id: str) -> Path:
        """Get directory storing chunks for a specific image."""
        p = self.get_job_dir(job_id, run_id) / "chunks" / image_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_images_dir(self, job_id: str, run_id: str) -> Path:
        """Get directory storing assembled images."""
        p = self.get_job_dir(job_id, run_id) / "images"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_image_path(self, job_id: str, run_id: str, image_id: str, extension: str = "jpg") -> Path:
        """Get target path for an assembled image."""
        ext = extension.lstrip(".")
        return self.get_images_dir(job_id, run_id) / f"{image_id}.{ext}"

    def get_chunk_path(
        self, job_id: str, run_id: str, image_id: str, chunk_index: int
    ) -> Path:
        """Get path to a specific chunk file."""
        return self.get_chunks_dir(job_id, run_id, image_id) / f"chunk_{chunk_index:06d}.bin"

    def chunk_exists(
        self, job_id: str, run_id: str, image_id: str, chunk_index: int
    ) -> bool:
        """Check if a specific chunk file exists on disk."""
        return self.get_chunk_path(job_id, run_id, image_id, chunk_index).exists()

    def delete_chunk(
        self, job_id: str, run_id: str, image_id: str, chunk_index: int
    ) -> bool:
        """Delete a chunk file if it exists. Returns True if deleted."""
        target = self.get_chunk_path(job_id, run_id, image_id, chunk_index)
        if target.exists():
            try:
                target.unlink()
                return True
            except OSError:
                return False
        return False

    def write_chunk(
        self,
        job_id: str,
        run_id: str,
        image_id: str,
        chunk_index: int,
        data: bytes,
        overwrite: bool = True,
    ) -> Path:
        """Write an individual binary chunk atomically.
        
        If overwrite is False and target exists, FileExistsError is raised.
        """
        chunks_dir = self.get_chunks_dir(job_id, run_id, image_id)
        target = chunks_dir / f"chunk_{chunk_index:06d}.bin"

        if not overwrite and target.exists():
            raise FileExistsError(f"Chunk file {target} already exists and overwrite is disabled")

        # Atomic write via unique temp file in same directory to prevent concurrent collisions
        with tempfile.NamedTemporaryFile(
            dir=chunks_dir,
            prefix=f".chunk_{chunk_index:06d}_",
            suffix=".tmp",
            delete=False,
        ) as tf:
            tf.write(data)
            temp_path = Path(tf.name)

        try:
            if not overwrite and target.exists():
                raise FileExistsError(f"Chunk file {target} already exists and overwrite is disabled")
            os.replace(temp_path, target)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        return target

    def assemble_image(
        self,
        job_id: str,
        run_id: str,
        image_id: str,
        total_chunks: int,
        expected_sha256: str = "",
        extension: str = "jpg",
    ) -> Path:
        """Assemble all chunks into a final image file and verify SHA-256."""
        chunks_dir = self.get_chunks_dir(job_id, run_id, image_id)
        image_path = self.get_image_path(job_id, run_id, image_id, extension)

        hasher = hashlib.sha256()
        temp_target = image_path.parent / f".{image_id}.tmp"

        with open(temp_target, "wb") as out_f:
            for idx in range(total_chunks):
                chunk_file = chunks_dir / f"chunk_{idx:06d}.bin"
                if not chunk_file.exists():
                    if temp_target.exists():
                        temp_target.unlink()
                    raise FileNotFoundError(
                        f"Missing chunk {idx} for image {image_id} in {chunks_dir}"
                    )
                with open(chunk_file, "rb") as cf:
                    chunk_bytes = cf.read()
                    hasher.update(chunk_bytes)
                    out_f.write(chunk_bytes)

        actual_sha256 = hasher.hexdigest()
        if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
            if temp_target.exists():
                temp_target.unlink()
            raise ValueError(
                f"Checksum mismatch for image {image_id}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        os.replace(temp_target, image_path)

        # Cleanup chunks directory to free space
        try:
            shutil.rmtree(chunks_dir)
        except OSError:
            pass

        return image_path

    def image_exists(self, job_id: str, run_id: str, image_id: str, extension: str = "jpg") -> bool:
        """Check if assembled image exists."""
        return self.get_image_path(job_id, run_id, image_id, extension).exists()

    def get_used_bytes(self) -> int:
        """Calculate total bytes used by temporary storage."""
        total = 0
        for root, _, files in os.walk(self.base_path):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        return total

    def cleanup_job(self, job_id: str, run_id: str) -> None:
        """Clean up all temporary files for a specific run."""
        job_dir = self.get_job_dir(job_id, run_id)
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
