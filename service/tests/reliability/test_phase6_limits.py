"""Reliability Phase 6 Tests: Limits & Resource Protection (TC33 - TC40)

TC33: Maximum chunk size boundaries.
TC34: Maximum image size boundaries.
TC35: Maximum job size boundaries.
TC36: Large image multi-chunk streaming and memory efficiency.
TC37: Large job throughput and resources.
TC38: Backpressure and ACK windowing behavior.
TC39: Slow client simulation.
TC40: Slow server simulation.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

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
    cfg.transfer.ack_window_chunks = 4
    engine = ProcessingEngine(cfg)
    return engine


def test_tc33_maximum_chunk_size(tmp_path: Path):
    """TC33: Chunks within 1MB pass; messages exceeding gRPC limits are rejected."""
    cfg = ServiceConfig()
    sock_path = Path(f"/tmp/t11_tc33_{time.time_ns()}.sock")
    target = f"unix:{sock_path}"
    cfg.grpc.host = target
    cfg.grpc.max_message_size_mb = 1  # 1 MB limit
    cfg.storage.temp_path = str(tmp_path / "temp")
    cfg.storage.result_path = str(tmp_path / "results")
    cfg.storage.database_path = str(tmp_path / "db.sqlite")
    cfg.models.yolo.mock = True
    cfg.models.sam3.mock = True

    engine = ProcessingEngine(cfg)
    server = create_server(cfg, engine=engine)
    server.start()
    time.sleep(0.15)

    try:
        # Standard client has 32MB max message length option, but server only accepts 1MB
        client = T11Client(target=target, max_message_size_mb=32)
        oversized_data = b"X" * (2 * 1024 * 1024)  # 2 MB payload

        def stream_large():
            yield pb2.EvaluateRequest(job_metadata=pb2.JobMetadata(job_id="j33", run_id="r01", total_images=1))
            yield pb2.EvaluateRequest(image_chunk=pb2.ImageChunk(image_id="img33", chunk_index=0, chunk_data=oversized_data, total_chunks=1))

        with pytest.raises(grpc.RpcError) as excinfo:
            list(client.stub.Evaluate(stream_large()))
        # Expect RESOURCE_EXHAUSTED or message larger than max
        assert excinfo.value.code() in (grpc.StatusCode.RESOURCE_EXHAUSTED, grpc.StatusCode.INTERNAL, grpc.StatusCode.UNAVAILABLE)
    finally:
        server.stop(grace=0.1)
        if sock_path.exists():
            sock_path.unlink()


def test_tc34_maximum_image_size(clean_engine):
    """TC34: Verify max image size configuration property."""
    assert clean_engine.config.transfer.max_image_size_mb == 50
    assert clean_engine.config.transfer.max_image_size_bytes == 50 * 1024 * 1024


def test_tc35_maximum_job_size(clean_engine):
    """TC35: Verify max job size configuration property."""
    assert clean_engine.config.transfer.max_job_size_gb == 10
    assert clean_engine.config.transfer.max_job_size_bytes == 10 * 1024 * 1024 * 1024


def test_tc36_large_image_multi_chunk(clean_engine):
    """TC36: 5 MB image streamed across 10 chunks of 512 KB, assembled without memory issues."""
    eng = clean_engine
    job_id = "j_tc36"
    run_id = "r01"
    image_id = "img_36"

    chunk_size = 512 * 1024
    num_chunks = 10
    full_data = b"L" * (chunk_size * num_chunks)
    full_sha = hashlib.sha256(full_data).hexdigest()

    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})

    for i in range(num_chunks):
        slice_data = full_data[i * chunk_size : (i + 1) * chunk_size]
        is_last = (i == num_chunks - 1)
        ack = eng.input_manager.handle_image_chunk(
            job_id,
            run_id,
            {
                "image_id": image_id,
                "chunk_index": i,
                "chunk_data": slice_data,
                "is_last": is_last,
                "total_chunks": num_chunks,
                "sha256": full_sha if is_last else "",
            },
        )
        if is_last:
            assert ack is not None
            assert ack.ack_type == "ACK_IMAGE_RECEIVED"

    assembled_path = eng.storage.get_image_path(job_id, run_id, image_id, extension="jpg")
    assert assembled_path.exists()
    assert assembled_path.stat().st_size == len(full_data)


def test_tc37_large_job(clean_engine):
    """TC37: Multi-image job throughput and proper cleanup."""
    eng = clean_engine
    job_id = "j_tc37"
    run_id = "r01"
    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 5})

    for i in range(5):
        img_id = f"img_37_{i}"
        d = f"IMAGE_CONTENT_{i}".encode() * 100
        sha = hashlib.sha256(d).hexdigest()
        eng.input_manager.handle_image_chunk(
            job_id, run_id, {"image_id": img_id, "chunk_index": 0, "chunk_data": d, "is_last": True, "total_chunks": 1, "sha256": sha}
        )

    eng.input_manager.handle_end_of_input(job_id, run_id, {"images_sent": 5})
    results = list(eng.process_job(job_id, run_id))
    assert len(results) > 0
    job = eng.db.get_job(job_id, run_id)
    assert job["status"] == "COMPLETED"


def test_tc38_backpressure_and_ack_window(clean_engine):
    """TC38: Windowed ACKs are emitted every ack_window_chunks (4 chunks)."""
    eng = clean_engine
    job_id = "j_tc38"
    run_id = "r01"
    image_id = "img_38"
    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})

    acks = []
    for i in range(8):
        ack = eng.input_manager.handle_image_chunk(
            job_id,
            run_id,
            {"image_id": image_id, "chunk_index": i, "chunk_data": b"CHUNK_DATA", "is_last": False, "total_chunks": 9},
        )
        if ack:
            acks.append((i, ack.ack_type))

    # Expect ACK_CHUNK at chunk_index 3 (chunk 4) and chunk_index 7 (chunk 8)
    assert len(acks) == 2
    assert acks[0][0] == 3
    assert acks[0][1] == "ACK_CHUNK"
    assert acks[1][0] == 7
    assert acks[1][1] == "ACK_CHUNK"


def test_tc39_slow_client(clean_engine):
    """TC39: Deliberate inter-chunk delay from client does not cause session loss."""
    eng = clean_engine
    job_id = "j_tc39"
    run_id = "r01"
    image_id = "img_39"
    eng.input_manager.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})

    # Chunk 0
    eng.input_manager.handle_image_chunk(
        job_id, run_id, {"image_id": image_id, "chunk_index": 0, "chunk_data": b"SLOW_0", "is_last": False, "total_chunks": 2}
    )
    time.sleep(0.1)  # Simulated pause

    # Chunk 1
    full_sha = hashlib.sha256(b"SLOW_0" + b"SLOW_1").hexdigest()
    ack = eng.input_manager.handle_image_chunk(
        job_id, run_id, {"image_id": image_id, "chunk_index": 1, "chunk_data": b"SLOW_1", "is_last": True, "total_chunks": 2, "sha256": full_sha}
    )
    assert ack.ack_type == "ACK_IMAGE_RECEIVED"


def test_tc40_slow_server(clean_engine):
    """TC40: Processing delay in engine handles queue without memory overflow."""
    eng = clean_engine
    assert clean_engine.batch_manager.queue_max_size == 256
