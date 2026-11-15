"""Unit tests for TemporaryStorage."""
import hashlib
from pathlib import Path
import pytest

from service.core.temporary_storage import TemporaryStorage


def test_write_chunk_and_assemble(tmp_path: Path):
    storage = TemporaryStorage(tmp_path / "temp_storage")
    job_id = "test_job_1"
    run_id = "run_01"
    image_id = "img_001"

    part1 = b"Hello "
    part2 = b"World!"
    full_data = part1 + part2
    full_sha = hashlib.sha256(full_data).hexdigest()

    storage.write_chunk(job_id, run_id, image_id, 0, part1)
    storage.write_chunk(job_id, run_id, image_id, 1, part2)

    assembled_path = storage.assemble_image(
        job_id, run_id, image_id, total_chunks=2, expected_sha256=full_sha
    )

    assert assembled_path.exists()
    assert assembled_path.read_bytes() == full_data
    assert storage.image_exists(job_id, run_id, image_id)


def test_checksum_mismatch_raises(tmp_path: Path):
    storage = TemporaryStorage(tmp_path / "temp_storage")
    job_id = "test_job_2"
    run_id = "run_01"
    image_id = "img_002"

    data = b"Corrupted data"
    storage.write_chunk(job_id, run_id, image_id, 0, data)

    with pytest.raises(ValueError, match="Checksum mismatch"):
        storage.assemble_image(
            job_id, run_id, image_id, total_chunks=1, expected_sha256="bad_hash"
        )


def test_path_traversal_protection(tmp_path: Path):
    storage = TemporaryStorage(tmp_path / "temp_storage")
    with pytest.raises(ValueError, match="Path traversal detected"):
        storage._safe_path(Path("../../../etc/passwd"))
