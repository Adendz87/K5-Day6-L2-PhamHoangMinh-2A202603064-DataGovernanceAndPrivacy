"""Unit tests for JobDatabase."""
from pathlib import Path
import pytest

from service.core.database import JobDatabase


def test_database_job_and_chunk_idempotency(tmp_path: Path):
    db = JobDatabase(tmp_path / "test.sqlite")
    job_id = "job_db_1"
    run_id = "run_01"

    # 1. Create job
    db.create_job(job_id, run_id, dataset_name="test_dataset", total_images=10)
    job = db.get_job(job_id, run_id)
    assert job is not None
    assert job["status"] == "CREATED"
    assert job["total_images"] == 10

    # 2. Update status
    db.update_job_status(job_id, run_id, "PROCESSING")
    job = db.get_job(job_id, run_id)
    assert job["status"] == "PROCESSING"

    # 3. Record chunk idempotency
    db.ensure_image(job_id, run_id, "img_1", total_chunks=2)
    is_new_1 = db.record_chunk(job_id, run_id, "img_1", 0, 1024, "sha_chunk_0")
    assert is_new_1 is True

    # Same chunk again
    is_new_2 = db.record_chunk(job_id, run_id, "img_1", 0, 1024, "sha_chunk_0")
    assert is_new_2 is False

    # 4. Result sequence
    seq1 = db.next_result_sequence(job_id, run_id)
    assert seq1 == 1
    db.persist_result(job_id, run_id, seq1, "ANNOTATION_RESULT", "ann_1", {"score": 0.9})
    seq2 = db.next_result_sequence(job_id, run_id)
    assert seq2 == 2

    # 5. Idempotency key registration
    assert db.check_idempotency_key("idem_key_1") is None
    db.register_idempotency_key("idem_key_1", job_id, run_id)
    assert db.check_idempotency_key("idem_key_1") is not None
