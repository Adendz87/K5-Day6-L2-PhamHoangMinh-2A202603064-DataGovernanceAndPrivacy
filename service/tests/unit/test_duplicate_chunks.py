"""Unit tests validating duplicate chunk handling and data integrity.

Tests coverage:
1. First chunk upload succeeds.
2. Identical duplicate chunk does not overwrite original file.
3. Conflicting duplicate chunk is rejected (raises ValueError).
4. Database SHA-256 remains consistent with actual chunk file.
5. received_chunks is not incremented for duplicate chunks.
6. Concurrent identical chunk submissions do not corrupt data or create duplicate receipts.
7. Failed database operation cleans up newly written file and leaves no inconsistent state.
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

from service.core.database import JobDatabase
from service.core.input_manager import InputManager
from service.core.job_lifecycle import JobLifecycleManager
from service.core.temporary_storage import TemporaryStorage


@pytest.fixture
def test_env(tmp_path: Path):
    """Set up temporary test environment with database and storage."""
    db_path = tmp_path / "test_jobs.db"
    storage_path = tmp_path / "storage"

    db = JobDatabase(db_path)
    storage = TemporaryStorage(storage_path)
    lifecycle = JobLifecycleManager(db)
    input_mgr = InputManager(storage=storage, database=db, lifecycle=lifecycle)

    job_id = "job_dup_test"
    run_id = "run_001"
    image_id = "img_001"

    # Initialize job metadata
    input_mgr.handle_job_metadata(
        {
            "job_id": job_id,
            "run_id": run_id,
            "client_id": "pytest_client",
            "dataset_name": "test_ds",
            "total_images": 1,
        }
    )

    return {
        "db": db,
        "storage": storage,
        "lifecycle": lifecycle,
        "input_mgr": input_mgr,
        "job_id": job_id,
        "run_id": run_id,
        "image_id": image_id,
    }


def test_first_chunk_upload_succeeds(test_env):
    """Test 1: First chunk upload succeeds and is recorded correctly."""
    mgr: InputManager = test_env["input_mgr"]
    db: JobDatabase = test_env["db"]
    storage: TemporaryStorage = test_env["storage"]
    job_id = test_env["job_id"]
    run_id = test_env["run_id"]
    image_id = test_env["image_id"]

    data = b"ORIGINAL_CHUNK_DATA_0"
    data_sha = hashlib.sha256(data).hexdigest()

    result = mgr.handle_image_chunk(
        job_id,
        run_id,
        {
            "image_id": image_id,
            "chunk_index": 0,
            "chunk_data": data,
            "is_last": False,
            "total_chunks": 2,
        },
    )

    # Verify chunk file exists on disk
    chunk_path = storage.get_chunk_path(job_id, run_id, image_id, 0)
    assert chunk_path.exists()
    assert chunk_path.read_bytes() == data

    # Verify DB receipt
    receipt = db.get_chunk_receipt(job_id, run_id, image_id, 0)
    assert receipt is not None
    assert receipt["sha256"] == data_sha
    assert receipt["chunk_size"] == len(data)

    # Verify received_chunks
    img = db.get_image(job_id, run_id, image_id)
    assert img is not None
    assert img["received_chunks"] == 1


def test_identical_duplicate_chunk_does_not_overwrite(test_env):
    """Test 2: Identical duplicate chunk does not overwrite original file."""
    mgr: InputManager = test_env["input_mgr"]
    db: JobDatabase = test_env["db"]
    storage: TemporaryStorage = test_env["storage"]
    job_id = test_env["job_id"]
    run_id = test_env["run_id"]
    image_id = test_env["image_id"]

    data = b"ORIGINAL_CHUNK_DATA_IDENTICAL"
    chunk_payload = {
        "image_id": image_id,
        "chunk_index": 0,
        "chunk_data": data,
        "is_last": False,
        "total_chunks": 1,
    }

    # First send
    mgr.handle_image_chunk(job_id, run_id, chunk_payload)
    chunk_path = storage.get_chunk_path(job_id, run_id, image_id, 0)
    assert chunk_path.exists()

    # Artificially set mtime in the past
    past_time = time.time() - 100.0
    os.utime(chunk_path, (past_time, past_time))
    initial_mtime_ns = chunk_path.stat().st_mtime_ns

    # Send identical chunk again
    mgr.handle_image_chunk(job_id, run_id, chunk_payload)

    # File must NOT have been overwritten or modified
    assert chunk_path.stat().st_mtime_ns == initial_mtime_ns
    assert chunk_path.read_bytes() == data


def test_conflicting_duplicate_chunk_is_rejected(test_env):
    """Test 3: Conflicting duplicate chunk is rejected with ValueError."""
    mgr: InputManager = test_env["input_mgr"]
    db: JobDatabase = test_env["db"]
    storage: TemporaryStorage = test_env["storage"]
    job_id = test_env["job_id"]
    run_id = test_env["run_id"]
    image_id = test_env["image_id"]

    data_original = b"ORIGINAL_BYTES"
    data_tampered = b"TAMPERED_MALICIOUS_BYTES"

    # Send original chunk
    mgr.handle_image_chunk(
        job_id,
        run_id,
        {
            "image_id": image_id,
            "chunk_index": 0,
            "chunk_data": data_original,
            "is_last": False,
            "total_chunks": 1,
        },
    )

    # Send conflicting chunk with same index
    with pytest.raises(ValueError) as excinfo:
        mgr.handle_image_chunk(
            job_id,
            run_id,
            {
                "image_id": image_id,
                "chunk_index": 0,
                "chunk_data": data_tampered,
                "is_last": False,
                "total_chunks": 1,
            },
        )

    assert "Conflicting chunk content" in str(excinfo.value)

    # Filesystem must retain original bytes
    chunk_path = storage.get_chunk_path(job_id, run_id, image_id, 0)
    assert chunk_path.read_bytes() == data_original

    # DB receipt must retain original hash
    receipt = db.get_chunk_receipt(job_id, run_id, image_id, 0)
    assert receipt is not None
    assert receipt["sha256"] == hashlib.sha256(data_original).hexdigest()


def test_database_sha256_remains_consistent_with_chunk_file(test_env):
    """Test 4: Database SHA-256 remains strictly consistent with chunk file."""
    mgr: InputManager = test_env["input_mgr"]
    db: JobDatabase = test_env["db"]
    storage: TemporaryStorage = test_env["storage"]
    job_id = test_env["job_id"]
    run_id = test_env["run_id"]
    image_id = test_env["image_id"]

    data_original = b"SECURE_BINARY_STREAM_CONTENT"
    mgr.handle_image_chunk(
        job_id,
        run_id,
        {
            "image_id": image_id,
            "chunk_index": 0,
            "chunk_data": data_original,
            "is_last": False,
            "total_chunks": 1,
        },
    )

    # Attempt tamper
    try:
        mgr.handle_image_chunk(
            job_id,
            run_id,
            {
                "image_id": image_id,
                "chunk_index": 0,
                "chunk_data": b"DIFFERENT_CONTENT",
                "is_last": False,
                "total_chunks": 1,
            },
        )
    except ValueError:
        pass

    # Read actual bytes from filesystem
    chunk_path = storage.get_chunk_path(job_id, run_id, image_id, 0)
    actual_disk_bytes = chunk_path.read_bytes()
    actual_disk_sha = hashlib.sha256(actual_disk_bytes).hexdigest()

    # Query DB
    receipt = db.get_chunk_receipt(job_id, run_id, image_id, 0)
    assert receipt is not None
    stored_db_sha = receipt["sha256"]

    # Invariant: DB SHA must equal filesystem SHA
    assert stored_db_sha == actual_disk_sha == hashlib.sha256(data_original).hexdigest()


def test_received_chunks_not_incremented_for_duplicates(test_env):
    """Test 5: received_chunks in images table is not incremented for duplicates."""
    mgr: InputManager = test_env["input_mgr"]
    db: JobDatabase = test_env["db"]
    job_id = test_env["job_id"]
    run_id = test_env["run_id"]
    image_id = test_env["image_id"]

    data = b"CHUNK_FOR_COUNTING"
    payload = {
        "image_id": image_id,
        "chunk_index": 0,
        "chunk_data": data,
        "is_last": False,
        "total_chunks": 3,
    }

    # Initial upload
    mgr.handle_image_chunk(job_id, run_id, payload)
    img_before = db.get_image(job_id, run_id, image_id)
    assert img_before["received_chunks"] == 1

    # Multiple duplicate identical submissions
    for _ in range(5):
        mgr.handle_image_chunk(job_id, run_id, payload)

    img_after_identical = db.get_image(job_id, run_id, image_id)
    assert img_after_identical["received_chunks"] == 1

    # Conflicting submission
    with pytest.raises(ValueError):
        mgr.handle_image_chunk(
            job_id,
            run_id,
            {
                "image_id": image_id,
                "chunk_index": 0,
                "chunk_data": b"CONFLICTING_BYTES",
                "is_last": False,
                "total_chunks": 3,
            },
        )

    img_after_conflict = db.get_image(job_id, run_id, image_id)
    assert img_after_conflict["received_chunks"] == 1


def test_concurrent_identical_chunks_do_not_corrupt(test_env):
    """Test 6: Concurrent identical chunk submissions do not corrupt data or duplicate receipts."""
    mgr: InputManager = test_env["input_mgr"]
    db: JobDatabase = test_env["db"]
    storage: TemporaryStorage = test_env["storage"]
    job_id = test_env["job_id"]
    run_id = test_env["run_id"]
    image_id = test_env["image_id"]

    data = b"CONCURRENT_CONTENT_" * 1024  # 19 KB chunk
    expected_sha = hashlib.sha256(data).hexdigest()

    payload = {
        "image_id": image_id,
        "chunk_index": 0,
        "chunk_data": data,
        "is_last": False,
        "total_chunks": 1,
    }

    num_threads = 16
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(mgr.handle_image_chunk, job_id, run_id, payload)
            for _ in range(num_threads)
        ]
        for f in concurrent.futures.as_completed(futures):
            # None of the threads should raise exceptions
            f.result()

    # Filesystem check: exactly one intact file
    chunk_path = storage.get_chunk_path(job_id, run_id, image_id, 0)
    assert chunk_path.exists()
    assert chunk_path.read_bytes() == data

    # Database check: exactly one receipt row in SQLite
    with db.get_connection() as conn:
        row_count = conn.execute(
            """
            SELECT COUNT(*) AS cnt FROM chunk_receipts
            WHERE job_id = ? AND run_id = ? AND image_id = ? AND chunk_index = 0;
            """,
            (job_id, run_id, image_id),
        ).fetchone()["cnt"]
        assert row_count == 1

    img = db.get_image(job_id, run_id, image_id)
    assert img["received_chunks"] == 1


def test_failed_database_operation_cleans_up_chunk_file(test_env):
    """Test 7: Failed database operation cleans up newly written file and leaves no inconsistent state."""
    mgr: InputManager = test_env["input_mgr"]
    db: JobDatabase = test_env["db"]
    storage: TemporaryStorage = test_env["storage"]
    job_id = test_env["job_id"]
    run_id = test_env["run_id"]
    image_id = test_env["image_id"]

    data = b"DATA_FAIL_SIMULATION"

    # Simulate a database failure during record_chunk
    with patch.object(db, "record_chunk", side_effect=sqlite3.OperationalError("Simulated disk error")):
        with pytest.raises(sqlite3.OperationalError) as excinfo:
            mgr.handle_image_chunk(
                job_id,
                run_id,
                {
                    "image_id": image_id,
                    "chunk_index": 0,
                    "chunk_data": data,
                    "is_last": False,
                    "total_chunks": 1,
                },
            )
        assert "Simulated disk error" in str(excinfo.value)

    # Invariant: Filesystem chunk file MUST be cleaned up (no orphaned/inconsistent file)
    chunk_path = storage.get_chunk_path(job_id, run_id, image_id, 0)
    assert not chunk_path.exists()

    # Invariant: DB receipt must not exist
    receipt = db.get_chunk_receipt(job_id, run_id, image_id, 0)
    assert receipt is None

    # Invariant: received_chunks must remain 0
    img = db.get_image(job_id, run_id, image_id)
    assert img["received_chunks"] == 0
