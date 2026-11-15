"""Reliability Phase 5 Tests: Database & Concurrency (TC26 - TC32)

TC26: Concurrent identical chunks (32 threads).
TC27: Concurrent conflicting chunks.
TC28: Concurrent different images within same job.
TC29: Concurrent jobs isolated from each other.
TC30: SQLite failure rollback & consistency.
TC31: Disk failure handling.
TC32: Lock cleanup and resource leak prevention.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import os
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from service.core.processing_engine import ProcessingEngine
from service.server.config import ServiceConfig


@pytest.fixture
def clean_engine(tmp_path: Path):
    cfg = ServiceConfig()
    cfg.storage.temp_path = str(tmp_path / "temp")
    cfg.storage.result_path = str(tmp_path / "results")
    cfg.storage.database_path = str(tmp_path / "db.sqlite")
    cfg.models.yolo.mock = True
    cfg.models.sam3.mock = True
    engine = ProcessingEngine(cfg)
    return engine


def test_tc26_concurrent_identical_chunks(clean_engine):
    """TC26: 32 threads concurrently upload identical chunk 0."""
    eng = clean_engine
    job_id = "j_tc26"
    run_id = "r01"
    image_id = "img_26"
    data = b"CONCURRENT_IDENTICAL_DATA" * 500

    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})
    payload = {"image_id": image_id, "chunk_index": 0, "chunk_data": data, "is_last": False, "total_chunks": 1}

    num_threads = 32
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(eng.input_manager.handle_image_chunk, job_id, run_id, payload) for _ in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    # Exact invariants
    img = eng.db.get_image(job_id, run_id, image_id)
    assert img["received_chunks"] == 1
    chunk_path = eng.storage.get_chunk_path(job_id, run_id, image_id, 0)
    assert chunk_path.exists()
    assert chunk_path.read_bytes() == data


def test_tc27_concurrent_conflicting_chunks(clean_engine):
    """TC27: Multiple threads send conflicting payloads for chunk 0; only one succeeds."""
    eng = clean_engine
    job_id = "j_tc27"
    run_id = "r01"
    image_id = "img_27"

    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})

    def send(idx):
        payload = {
            "image_id": image_id,
            "chunk_index": 0,
            "chunk_data": f"CONFLICT_{idx}".encode(),
            "is_last": False,
            "total_chunks": 1,
        }
        return eng.input_manager.handle_image_chunk(job_id, run_id, payload)

    results = []
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(send, i): i for i in range(8)}
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
            except ValueError as e:
                errors.append(e)

    # Exactly 1 payload committed, 7 rejected with ValueError
    assert len(errors) == 7
    img = eng.db.get_image(job_id, run_id, image_id)
    assert img["received_chunks"] == 1


def test_tc28_concurrent_different_images(clean_engine):
    """TC28: 10 images concurrently uploaded in same job."""
    eng = clean_engine
    job_id = "j_tc28"
    run_id = "r01"
    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 10})

    def upload_img(i):
        img_id = f"img_28_{i}"
        data = f"IMG_DATA_{i}".encode()
        sha = hashlib.sha256(data).hexdigest()
        return eng.input_manager.handle_image_chunk(
            job_id, run_id, {"image_id": img_id, "chunk_index": 0, "chunk_data": data, "is_last": True, "total_chunks": 1, "sha256": sha}
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(upload_img, i) for i in range(10)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    images = eng.db.get_all_images(job_id, run_id)
    assert len(images) == 10
    for img in images:
        assert img["status"] == "IMAGE_RECEIVED"


def test_tc29_concurrent_jobs(clean_engine):
    """TC29: Two distinct jobs executed concurrently without data cross-leakage."""
    eng = clean_engine

    def run_complete_job(job_id):
        run_id = "r01"
        eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})
        data = f"DATA_FOR_{job_id}".encode()
        sha = hashlib.sha256(data).hexdigest()
        eng.input_manager.handle_image_chunk(
            job_id, run_id, {"image_id": f"img_{job_id}", "chunk_index": 0, "chunk_data": data, "is_last": True, "total_chunks": 1, "sha256": sha}
        )
        eng.input_manager.handle_end_of_input(job_id, run_id, {"images_sent": 1})
        return list(eng.process_job(job_id, run_id))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(run_complete_job, "job_A")
        f2 = executor.submit(run_complete_job, "job_B")
        res_A = f1.result()
        res_B = f2.result()

    assert len(res_A) > 0
    assert len(res_B) > 0
    job_A_rec = eng.db.get_job("job_A", "r01")
    job_B_rec = eng.db.get_job("job_B", "r01")
    assert job_A_rec["status"] == "COMPLETED"
    assert job_B_rec["status"] == "COMPLETED"


def test_tc30_sqlite_failure(clean_engine):
    """TC30: SQLite failure during chunk record causes rollback and leaves no orphaned chunk file."""
    eng = clean_engine
    job_id = "j_tc30"
    run_id = "r01"
    image_id = "img_30"
    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})

    with patch.object(eng.db, "record_chunk", side_effect=sqlite3.OperationalError("Simulated locked DB")):
        with pytest.raises(sqlite3.OperationalError):
            eng.input_manager.handle_image_chunk(
                job_id, run_id, {"image_id": image_id, "chunk_index": 0, "chunk_data": b"FAIL", "is_last": False, "total_chunks": 1}
            )

    # Invariant: File on disk deleted by rollback
    assert not eng.storage.chunk_exists(job_id, run_id, image_id, 0)
    receipt = eng.db.get_chunk_receipt(job_id, run_id, image_id, 0)
    assert receipt is None


def test_tc31_disk_failure(clean_engine):
    """TC31: Disk failure raises error clearly without crashing process."""
    eng = clean_engine
    job_id = "j_tc31"
    run_id = "r01"
    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})

    with patch.object(eng.storage, "write_chunk", side_effect=OSError("No space left on device")):
        with pytest.raises(OSError) as excinfo:
            eng.input_manager.handle_image_chunk(
                job_id, run_id, {"image_id": "img_31", "chunk_index": 0, "chunk_data": b"FAIL", "is_last": False, "total_chunks": 1}
            )
        assert "No space left" in str(excinfo.value)


def test_tc32_lock_cleanup(clean_engine):
    """TC32: Chunk locks are cleaned up after image assembly, preventing unbounded memory growth."""
    eng = clean_engine
    job_id = "j_tc32"
    run_id = "r01"
    image_id = "img_32"
    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})

    data = b"LOCK_CLEANUP_TEST"
    sha = hashlib.sha256(data).hexdigest()
    eng.input_manager.handle_image_chunk(
        job_id, run_id, {"image_id": image_id, "chunk_index": 0, "chunk_data": data, "is_last": True, "total_chunks": 1, "sha256": sha}
    )

    # Locks for this image must be cleaned up from dict
    with eng.input_manager._chunk_lock_guard:
        matching_keys = [k for k in eng.input_manager._chunk_locks if k[0] == job_id and k[2] == image_id]
        assert len(matching_keys) == 0
