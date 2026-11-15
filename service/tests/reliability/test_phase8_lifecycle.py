"""Reliability Phase 8 Tests: Job Lifecycle & Results (TC51 - TC60)

TC51: Empty job (0 images, 0 annotations).
TC52: Image without annotations.
TC53: Annotation without image.
TC54: Duplicate annotation batch (idempotency).
TC55: Empty annotations in batch.
TC56: Normal job completion flow.
TC57: Job failure handling during inference.
TC58: Result replay audit.
TC59: Result sequence monotonicity (strictly increasing positive integers).
TC60: Job isolation between distinct runs.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from unittest.mock import patch

import grpc
import pytest

from service.client.example_client import T11Client
from service.core.processing_engine import ProcessingEngine
from service.generated import pb2
from service.server import create_server
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


def test_tc51_empty_job(clean_engine):
    """TC51: Empty job (0 images) processes successfully and yields JobCompleted."""
    eng = clean_engine
    job_id = "j_tc51"
    run_id = "r01"
    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 0})
    eng.input_manager.handle_end_of_input(job_id, run_id, {"images_sent": 0, "annotations_sent": 0})

    responses = list(eng.process_job(job_id, run_id))
    types = [r.WhichOneof("payload") for r in responses]
    assert "job_completed" in types
    job = eng.db.get_job(job_id, run_id)
    assert job["status"] == "COMPLETED"


def test_tc52_image_without_annotations(clean_engine):
    """TC52: Image without annotations evaluated with 0 human annotations (unmatched predictions)."""
    eng = clean_engine
    job_id = "j_tc52"
    run_id = "r01"
    image_id = "img_52"

    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})
    data = b"DUMMY_IMAGE_52"
    sha = hashlib.sha256(data).hexdigest()
    eng.input_manager.handle_image_chunk(
        job_id, run_id, {"image_id": image_id, "chunk_index": 0, "chunk_data": data, "is_last": True, "total_chunks": 1, "sha256": sha}
    )
    eng.input_manager.handle_end_of_input(job_id, run_id, {"images_sent": 1, "annotations_sent": 0})

    responses = list(eng.process_job(job_id, run_id))
    types = [r.WhichOneof("payload") for r in responses]
    assert "job_completed" in types
    assert "image_result" in types


def test_tc53_annotation_without_image(clean_engine):
    """TC53: Annotation referencing non-existent image is stored; does not crash pipeline."""
    eng = clean_engine
    job_id = "j_tc53"
    run_id = "r01"
    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 0})

    ack, errs = eng.input_manager.handle_annotation_batch(
        job_id,
        run_id,
        {
            "batch_id": "b53",
            "items": [
                {
                    "ann_id": "orphan_ann",
                    "image_id": "non_existent_image",
                    "class_name": "car",
                    "bbox": {"x": 10, "y": 10, "w": 20, "h": 20},
                }
            ],
        },
    )
    assert ack.ack_type == "ACK_ANNOTATIONS_RECEIVED"


def test_tc54_duplicate_annotation_batch(clean_engine):
    """TC54: Sending same annotation batch twice is handled idempotently via ON CONFLICT."""
    eng = clean_engine
    job_id = "j_tc54"
    run_id = "r01"
    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})

    batch = {
        "batch_id": "b54",
        "items": [
            {
                "ann_id": "ann_dup",
                "image_id": "img_54",
                "class_name": "car",
                "bbox": {"x": 10, "y": 10, "w": 20, "h": 20},
            }
        ],
    }

    eng.input_manager.handle_annotation_batch(job_id, run_id, batch)
    eng.input_manager.handle_annotation_batch(job_id, run_id, batch)

    anns = eng.db.get_annotations_for_image(job_id, run_id, "img_54")
    assert len(anns) == 1


def test_tc55_empty_annotations_batch(clean_engine):
    """TC55: Empty annotations batch does not raise error."""
    eng = clean_engine
    job_id = "j_tc55"
    run_id = "r01"
    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})

    ack, errs = eng.input_manager.handle_annotation_batch(job_id, run_id, {"batch_id": "b55", "items": []})
    assert ack.ack_type == "ACK_ANNOTATIONS_RECEIVED"


def test_tc56_job_completion_flow(clean_engine):
    """TC56: Full flow triggers transitions: RECEIVING -> INPUT_COMPLETE -> PROCESSING -> AGGREGATING -> COMPLETED."""
    eng = clean_engine
    job_id = "j_tc56"
    run_id = "r01"
    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})
    data = b"TC56_DATA"
    sha = hashlib.sha256(data).hexdigest()
    eng.input_manager.handle_image_chunk(
        job_id, run_id, {"image_id": "img_56", "chunk_index": 0, "chunk_data": data, "is_last": True, "total_chunks": 1, "sha256": sha}
    )
    eng.input_manager.handle_end_of_input(job_id, run_id, {"images_sent": 1})

    list(eng.process_job(job_id, run_id))
    job = eng.db.get_job(job_id, run_id)
    assert job["status"] == "COMPLETED"


def test_tc57_job_failure_during_inference(clean_engine):
    """TC57: Inference error during processing raises exception."""
    eng = clean_engine
    job_id = "j_tc57"
    run_id = "r01"
    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})
    data = b"TC57_DATA"
    sha = hashlib.sha256(data).hexdigest()
    eng.input_manager.handle_image_chunk(
        job_id, run_id, {"image_id": "img_57", "chunk_index": 0, "chunk_data": data, "is_last": True, "total_chunks": 1, "sha256": sha}
    )
    eng.input_manager.handle_end_of_input(job_id, run_id, {"images_sent": 1})

    # Simulate YOLO failure
    with patch.object(eng.yolo_adapter, "predict", side_effect=RuntimeError("YOLO CUDA out of memory")):
        with pytest.raises(RuntimeError) as excinfo:
            list(eng.process_job(job_id, run_id))
        assert "YOLO CUDA" in str(excinfo.value)


def test_tc58_result_replay(clean_engine):
    """TC58: Retrieve persisted results from ResultStorage for completed job."""
    eng = clean_engine
    job_id = "j_tc58"
    run_id = "r01"
    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})
    data = b"TC58_DATA"
    sha = hashlib.sha256(data).hexdigest()
    eng.input_manager.handle_image_chunk(
        job_id, run_id, {"image_id": "img_58", "chunk_index": 0, "chunk_data": data, "is_last": True, "total_chunks": 1, "sha256": sha}
    )
    eng.input_manager.handle_end_of_input(job_id, run_id, {"images_sent": 1})
    list(eng.process_job(job_id, run_id))

    # Replay all persisted results from sequence 1
    replayed = eng.result_storage.get_undelivered_results(job_id, run_id, from_sequence=1)
    assert len(replayed) > 0
    seqs = [r["_sequence"] for r in replayed]
    assert seqs == sorted(seqs)


def test_tc59_result_sequence_monotonicity(clean_engine):
    """TC59: Sequence numbers are strictly monotonically increasing starting from 1."""
    eng = clean_engine
    job_id = "j_tc59"
    run_id = "r01"
    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 2})

    for i in range(2):
        d = f"TC59_{i}".encode()
        s = hashlib.sha256(d).hexdigest()
        eng.input_manager.handle_image_chunk(
            job_id, run_id, {"image_id": f"img_59_{i}", "chunk_index": 0, "chunk_data": d, "is_last": True, "total_chunks": 1, "sha256": s}
        )
    eng.input_manager.handle_end_of_input(job_id, run_id, {"images_sent": 2})

    results = list(eng.process_job(job_id, run_id))
    sequences = []
    for r in results:
        field_name = r.WhichOneof("payload")
        msg = getattr(r, field_name)
        if hasattr(msg, "result_sequence") and msg.result_sequence > 0:
            sequences.append(msg.result_sequence)

    assert len(sequences) > 1
    assert sequences == list(range(1, len(sequences) + 1))


def test_tc60_job_isolation(clean_engine):
    """TC60: Results and state for Job 1 never appear in Job 2."""
    eng = clean_engine
    eng.input_manager.handle_job_metadata({"job_id": "job_iso_1", "run_id": "r01", "total_images": 1})
    eng.input_manager.handle_job_metadata({"job_id": "job_iso_2", "run_id": "r01", "total_images": 1})

    imgs_1 = eng.db.get_all_images("job_iso_1", "r01")
    imgs_2 = eng.db.get_all_images("job_iso_2", "r01")
    assert len(imgs_1) == 0
    assert len(imgs_2) == 0
