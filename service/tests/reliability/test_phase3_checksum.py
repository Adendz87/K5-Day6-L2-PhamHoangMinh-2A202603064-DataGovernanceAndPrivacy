"""Reliability Phase 3 Tests: Checksum & Data Integrity (TC13 - TC17)

TC13: Correct SHA-256 succeeds.
TC14: Wrong SHA-256 is rejected upon assembly.
TC15: Corrupted chunk (tampered byte) causes checksum mismatch.
TC16: Corrupted stored file on disk causes mismatch during assembly.
TC17: Database/file consistency invariants.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

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


def test_tc13_correct_sha256(clean_engine):
    """TC13: Correct SHA-256 verifies and marks image received."""
    eng = clean_engine
    job_id = "j_tc13"
    run_id = "r01"
    image_id = "img_13"
    data = b"VALID_IMAGE_BYTES_TC13"
    correct_sha = hashlib.sha256(data).hexdigest()

    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})
    ack = eng.input_manager.handle_image_chunk(
        job_id,
        run_id,
        {
            "image_id": image_id,
            "chunk_index": 0,
            "chunk_data": data,
            "is_last": True,
            "total_chunks": 1,
            "sha256": correct_sha,
        },
    )
    assert ack is not None
    assert ack.ack_type == "ACK_IMAGE_RECEIVED"
    img_rec = eng.db.get_image(job_id, run_id, image_id)
    assert img_rec["status"] == "IMAGE_RECEIVED"
    assert img_rec["sha256"] == correct_sha


def test_tc14_wrong_sha256(clean_engine):
    """TC14: Wrong SHA-256 provided by client is rejected upon assembly."""
    eng = clean_engine
    job_id = "j_tc14"
    run_id = "r01"
    image_id = "img_14"
    data = b"ACTUAL_DATA_14"
    wrong_sha = "0000000000000000000000000000000000000000000000000000000000000000"

    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})
    with pytest.raises(ValueError) as excinfo:
        eng.input_manager.handle_image_chunk(
            job_id,
            run_id,
            {
                "image_id": image_id,
                "chunk_index": 0,
                "chunk_data": data,
                "is_last": True,
                "total_chunks": 1,
                "sha256": wrong_sha,
            },
        )
    assert "Checksum mismatch" in str(excinfo.value)
    # Image must NOT be marked received
    img_rec = eng.db.get_image(job_id, run_id, image_id)
    assert img_rec["status"] != "IMAGE_RECEIVED"


def test_tc15_corrupted_chunk(clean_engine):
    """TC15: Corrupted byte in multi-chunk transfer causes assembly failure."""
    eng = clean_engine
    job_id = "j_tc15"
    run_id = "r01"
    image_id = "img_15"
    chunk0 = b"CHUNK_0_VALID"
    chunk1 = b"CHUNK_1_VALID"
    expected_full_sha = hashlib.sha256(chunk0 + chunk1).hexdigest()

    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})
    # Send chunk 0
    eng.input_manager.handle_image_chunk(
        job_id, run_id, {"image_id": image_id, "chunk_index": 0, "chunk_data": chunk0, "total_chunks": 2}
    )
    # Send corrupted chunk 1 (flipped byte)
    corrupted_chunk1 = b"CHUNK_1_VALIE"
    with pytest.raises(ValueError) as excinfo:
        eng.input_manager.handle_image_chunk(
            job_id,
            run_id,
            {
                "image_id": image_id,
                "chunk_index": 1,
                "chunk_data": corrupted_chunk1,
                "is_last": True,
                "total_chunks": 2,
                "sha256": expected_full_sha,
            },
        )
    assert "Checksum mismatch" in str(excinfo.value)


def test_tc16_corrupted_stored_file(clean_engine):
    """TC16: Tampering with stored chunk file on disk is caught during assemble_image."""
    eng = clean_engine
    job_id = "j_tc16"
    run_id = "r01"
    image_id = "img_16"
    data0 = b"CHUNK_ZERO_ORIGINAL"
    data1 = b"CHUNK_ONE_ORIGINAL"
    expected_full_sha = hashlib.sha256(data0 + data1).hexdigest()

    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})
    eng.input_manager.handle_image_chunk(
        job_id, run_id, {"image_id": image_id, "chunk_index": 0, "chunk_data": data0, "total_chunks": 2}
    )

    # Tamper with chunk 0 directly on filesystem!
    chunk0_path = eng.storage.get_chunk_path(job_id, run_id, image_id, 0)
    chunk0_path.write_bytes(b"TAMPERED_DIRECT_ON_DISK")

    # Send chunk 1 with expected sha
    with pytest.raises(ValueError) as excinfo:
        eng.input_manager.handle_image_chunk(
            job_id,
            run_id,
            {
                "image_id": image_id,
                "chunk_index": 1,
                "chunk_data": data1,
                "is_last": True,
                "total_chunks": 2,
                "sha256": expected_full_sha,
            },
        )
    assert "Checksum mismatch" in str(excinfo.value)


def test_tc17_database_filesystem_consistency(clean_engine):
    """TC17: Verify exact consistency between SQLite receipts, files, and image status."""
    eng = clean_engine
    job_id = "j_tc17"
    run_id = "r01"
    image_id = "img_17"
    data = b"INVARIANT_DATA_TEST"
    expected_sha = hashlib.sha256(data).hexdigest()

    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})
    eng.input_manager.handle_image_chunk(
        job_id,
        run_id,
        {
            "image_id": image_id,
            "chunk_index": 0,
            "chunk_data": data,
            "is_last": True,
            "total_chunks": 1,
            "sha256": expected_sha,
        },
    )

    # 1. Image record
    img = eng.db.get_image(job_id, run_id, image_id)
    assert img["status"] == "IMAGE_RECEIVED"
    assert img["received_chunks"] == 1
    assert img["sha256"] == expected_sha

    # 2. Assembled image on disk
    disk_path = Path(img["file_path"])
    assert disk_path.exists()
    assert hashlib.sha256(disk_path.read_bytes()).hexdigest() == expected_sha

    # 3. DB receipt for chunk 0
    receipt = eng.db.get_chunk_receipt(job_id, run_id, image_id, 0)
    assert receipt is not None
    assert receipt["sha256"] == expected_sha
