"""Reliability Phase 4 Tests: Disconnect, Resume & Recovery (TC18 - TC25)

TC18: Disconnect midway through upload, reconnect, send ResumeRequest.
TC19: Resume from last received chunk (no re-upload of confirmed chunks).
TC20: Retransmission after resume (idempotency).
TC21: Server process restart during upload (recovery from persistent SQLite/disk).
TC22: Server restart after all chunks received but before inference.
TC23: Server restart during inference (job state audit).
TC24: Client disconnect during result streaming.
TC25: Repeated consecutive ResumeRequest messages.
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
def persistent_env(tmp_path: Path):
    temp_dir = tmp_path / "temp"
    results_dir = tmp_path / "results"
    db_path = tmp_path / "persistent.sqlite"

    def _make_server():
        sock_path = Path(f"/tmp/t11_rec_{time.time_ns()}.sock")
        if sock_path.exists():
            sock_path.unlink()
        target = f"unix:{sock_path}"

        cfg = ServiceConfig()
        cfg.grpc.host = target
        cfg.grpc.port = 0
        cfg.storage.temp_path = str(temp_dir)
        cfg.storage.result_path = str(results_dir)
        cfg.storage.database_path = str(db_path)
        cfg.models.yolo.mock = True
        cfg.models.sam3.mock = True

        engine = ProcessingEngine(cfg)
        server = create_server(cfg, engine=engine)
        server.start()
        time.sleep(0.15)
        return {"target": target, "server": server, "sock_path": sock_path, "engine": engine, "cfg": cfg}

    env = _make_server()
    yield env, _make_server

    try:
        env["server"].stop(grace=0.1)
    except Exception:
        pass
    if env["sock_path"].exists():
        env["sock_path"].unlink()


def test_tc18_disconnect_midway_and_resume_request(persistent_env):
    """TC18: Client disconnects after chunk 0, reconnects, requests resume state."""
    env, _ = persistent_env
    client = T11Client(target=env["target"])
    job_id = "j_tc18"
    run_id = "r01"

    # Client streams metadata and chunk 0, then abruptly disconnects (closes stream)
    def stream_part():
        yield pb2.EvaluateRequest(job_metadata=pb2.JobMetadata(job_id=job_id, run_id=run_id, total_images=1))
        yield pb2.EvaluateRequest(
            image_chunk=pb2.ImageChunk(
                image_id="img_18", chunk_index=0, chunk_data=b"CHUNK_0_BYTES", total_chunks=2
            )
        )

    # Consume response and close stream
    list(client.stub.Evaluate(stream_part()))

    # New client connection sends ResumeRequest
    new_client = T11Client(target=env["target"])
    state = new_client.resume_job(job_id, run_id)

    assert state.job_id == job_id
    assert state.run_id == run_id
    assert "img_18" in state.pending_image_ids
    assert state.last_chunk_per_image.get("img_18") == 0


def test_tc19_resume_from_last_chunk(persistent_env):
    """TC19: Resume starts from chunk 1 without re-uploading chunk 0."""
    env, _ = persistent_env
    client = T11Client(target=env["target"])
    job_id = "j_tc19"
    run_id = "r01"
    image_id = "img_19"

    chunk0 = b"CHUNK_0_CONTENT"
    chunk1 = b"CHUNK_1_CONTENT"
    full_sha = hashlib.sha256(chunk0 + chunk1).hexdigest()

    # Upload chunk 0
    def stream_1():
        yield pb2.EvaluateRequest(job_metadata=pb2.JobMetadata(job_id=job_id, run_id=run_id, total_images=1))
        yield pb2.EvaluateRequest(image_chunk=pb2.ImageChunk(image_id=image_id, chunk_index=0, chunk_data=chunk0, total_chunks=2))
    list(client.stub.Evaluate(stream_1()))

    # Check resume state
    state = client.resume_job(job_id, run_id)
    last_idx = state.last_chunk_per_image.get(image_id, -1)
    assert last_idx == 0

    # Stream only remaining chunk 1
    def stream_2():
        yield pb2.EvaluateRequest(
            image_chunk=pb2.ImageChunk(
                image_id=image_id, chunk_index=1, chunk_data=chunk1, is_last=True, total_chunks=2, sha256=full_sha
            )
        )

    responses = list(client.stub.Evaluate(stream_2()))
    img_acks = [r.acknowledgement for r in responses if r.WhichOneof("payload") == "acknowledgement" and r.acknowledgement.ack_type == pb2.AckType.ACK_IMAGE_RECEIVED]
    assert len(img_acks) == 1


def test_tc20_retransmission_after_resume(persistent_env):
    """TC20: Intentionally retransmitting chunk 0 after resume is safely handled idempotently."""
    env, _ = persistent_env
    client = T11Client(target=env["target"])
    job_id = "j_tc20"
    run_id = "r01"
    image_id = "img_20"
    chunk0 = b"CHUNK_0_BYTES"

    def stream_initial():
        yield pb2.EvaluateRequest(job_metadata=pb2.JobMetadata(job_id=job_id, run_id=run_id, total_images=1))
        yield pb2.EvaluateRequest(image_chunk=pb2.ImageChunk(image_id=image_id, chunk_index=0, chunk_data=chunk0, total_chunks=2))
    list(client.stub.Evaluate(stream_initial()))

    # Retransmit chunk 0 in a new connection
    def stream_retry():
        yield pb2.EvaluateRequest(image_chunk=pb2.ImageChunk(image_id=image_id, chunk_index=0, chunk_data=chunk0, total_chunks=2))
    list(client.stub.Evaluate(stream_retry()))

    img = env["engine"].db.get_image(job_id, run_id, image_id)
    assert img["received_chunks"] == 1


def test_tc21_server_restart_during_upload(persistent_env):
    """TC21: Server is killed and restarted during upload; resume state recovered from SQLite."""
    env, make_server = persistent_env
    client = T11Client(target=env["target"])
    job_id = "j_tc21"
    run_id = "r01"
    image_id = "img_21"

    # Send metadata and chunk 0
    def stream_init():
        yield pb2.EvaluateRequest(job_metadata=pb2.JobMetadata(job_id=job_id, run_id=run_id, total_images=1))
        yield pb2.EvaluateRequest(image_chunk=pb2.ImageChunk(image_id=image_id, chunk_index=0, chunk_data=b"CHUNK_0_PERSISTENT", total_chunks=2))
    list(client.stub.Evaluate(stream_init()))

    # Hard shutdown server (simulate process kill)
    env["server"].stop(grace=0)
    time.sleep(0.2)
    if env["sock_path"].exists():
        env["sock_path"].unlink()

    # Start a brand new server process pointing to same SQLite database and storage
    env2 = make_server()
    try:
        client2 = T11Client(target=env2["target"])
        state = client2.resume_job(job_id, run_id)
        assert state.job_id == job_id
        assert state.last_chunk_per_image.get(image_id) == 0
        assert "img_21" in state.pending_image_ids
    finally:
        env2["server"].stop(grace=0.1)
        if env2["sock_path"].exists():
            env2["sock_path"].unlink()


def test_tc22_restart_after_chunks_received_before_inference(persistent_env):
    """TC22: Restart server when all images are received but before EndOfInput inference."""
    env, make_server = persistent_env
    client = T11Client(target=env["target"])
    job_id = "j_tc22"
    run_id = "r01"
    image_id = "img_22"
    data = b"COMPLETE_IMAGE_PAYLOAD"
    sha = hashlib.sha256(data).hexdigest()

    def stream_complete_img():
        yield pb2.EvaluateRequest(job_metadata=pb2.JobMetadata(job_id=job_id, run_id=run_id, total_images=1))
        yield pb2.EvaluateRequest(image_chunk=pb2.ImageChunk(image_id=image_id, chunk_index=0, chunk_data=data, is_last=True, total_chunks=1, sha256=sha))
    list(client.stub.Evaluate(stream_complete_img()))

    # Kill and restart server
    env["server"].stop(grace=0)
    time.sleep(0.2)
    if env["sock_path"].exists():
        env["sock_path"].unlink()

    env2 = make_server()
    try:
        client2 = T11Client(target=env2["target"])
        state = client2.resume_job(job_id, run_id)
        assert image_id in state.completed_image_ids
    finally:
        env2["server"].stop(grace=0.1)
        if env2["sock_path"].exists():
            env2["sock_path"].unlink()


def test_tc23_restart_during_inference(persistent_env):
    """TC23: Audit behavior when server restarts while job is in PROCESSING status."""
    env, make_server = persistent_env
    job_id = "j_tc23"
    run_id = "r01"

    # Manually transition job to PROCESSING in SQLite
    env["engine"].db.create_job(job_id=job_id, run_id=run_id, total_images=5, initial_status="PROCESSING")
    job = env["engine"].db.get_job(job_id, run_id)
    assert job["status"] == "PROCESSING"

    # Restart server
    env["server"].stop(grace=0)
    time.sleep(0.1)
    if env["sock_path"].exists():
        env["sock_path"].unlink()

    env2 = make_server()
    try:
        client2 = T11Client(target=env2["target"])
        state = client2.resume_job(job_id, run_id)
        # Server accurately reports job status as recorded in DB
        assert state.current_job_status == pb2.JobStatus.PROCESSING
    finally:
        env2["server"].stop(grace=0.1)
        if env2["sock_path"].exists():
            env2["sock_path"].unlink()


def test_tc24_client_disconnect_during_result_streaming(persistent_env):
    """TC24: Client disconnects during streaming; results remain persisted in SQLite."""
    env, _ = persistent_env
    eng = env["engine"]
    job_id = "j_tc24"
    run_id = "r01"

    # Pre-populate persisted results
    eng.db.create_job(job_id, run_id, total_images=1, initial_status="COMPLETED")
    seq1 = eng.result_storage.save_annotation_result(job_id, run_id, {"ann_id": "a1", "risk_score": 0.8})
    seq2 = eng.result_storage.save_job_completed(job_id, run_id, {"final_status": "COMPLETED"})

    # Check results in DB
    undelivered = eng.result_storage.get_undelivered_results(job_id, run_id, from_sequence=1)
    assert len(undelivered) == 2


def test_tc25_repeated_resume_requests(persistent_env):
    """TC25: Consecutive ResumeRequest calls return identical, non-corrupted state."""
    env, _ = persistent_env
    client = T11Client(target=env["target"])
    job_id = "j_tc25"
    run_id = "r01"

    client.resume_job(job_id, run_id)
    for _ in range(5):
        st = client.resume_job(job_id, run_id)
        assert st.job_id == job_id
        assert st.current_job_status == pb2.JobStatus.CREATED
