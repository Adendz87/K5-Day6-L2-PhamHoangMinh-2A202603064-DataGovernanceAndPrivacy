"""Reliability Phase 1 Tests: Basic Communication (TC01 - TC04)

TC01: Server startup, port binding, request reception, shutdown, restart.
TC02: Client connection, metadata sending, protocol response, ACK, disconnect.
TC03: Binary transport, identical raw bytes, SHA-256 match, no Base64.
TC04: Complete job with multiple images and annotations, job COMPLETED.
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
def clean_server(tmp_path: Path):
    """Factory fixture for starting and stopping isolated servers."""
    servers = []
    socks = []

    def _start(port: int = 0, custom_sock: str = ""):
        sock_path = Path(custom_sock or f"/tmp/t11_comm_{len(servers)}_{time.time_ns()}.sock")
        if sock_path.exists():
            sock_path.unlink()
        target = f"unix:{sock_path}"

        cfg = ServiceConfig()
        cfg.grpc.host = target
        cfg.grpc.port = 0
        cfg.storage.temp_path = str(tmp_path / f"temp_{len(servers)}")
        cfg.storage.result_path = str(tmp_path / f"results_{len(servers)}")
        cfg.storage.database_path = str(tmp_path / f"db_{len(servers)}.sqlite")
        cfg.models.yolo.mock = True
        cfg.models.sam3.mock = True

        engine = ProcessingEngine(cfg)
        server = create_server(cfg, engine=engine)
        server.start()
        time.sleep(0.15)
        servers.append(server)
        socks.append(sock_path)
        return {"target": target, "server": server, "sock_path": sock_path, "engine": engine, "config": cfg}

    yield _start

    for s in servers:
        try:
            s.stop(grace=0.2)
        except Exception:
            pass
    for p in socks:
        if p.exists():
            p.unlink()


def test_tc01_server_startup_shutdown_restart(clean_server):
    """TC01: Verify server startup, binding, request reception, shutdown, restart."""
    inst1 = clean_server()
    client = T11Client(target=inst1["target"])

    # Test request reception
    state = client.resume_job("non_existent_job", "run_01")
    assert state.job_id == "non_existent_job"
    assert state.current_job_status == pb2.JobStatus.CREATED

    # Controlled shutdown
    inst1["server"].stop(grace=0.2)
    time.sleep(0.2)

    # Verify channel fails to connect after shutdown
    with pytest.raises(Exception):
        c_dead = T11Client(target=inst1["target"])
        c_dead.resume_job("job", "run")

    # Restart server on new instance
    inst2 = clean_server()
    client2 = T11Client(target=inst2["target"])
    st2 = client2.resume_job("restart_job", "run_01")
    assert st2.job_id == "restart_job"


def test_tc02_client_connection(clean_server):
    """TC02: Client connects, sends metadata, receives ACK, closes connection cleanly."""
    inst = clean_server()
    client = T11Client(target=inst["target"])

    def stream_meta():
        yield pb2.EvaluateRequest(
            job_metadata=pb2.JobMetadata(
                job_id="job_tc02",
                run_id="run_01",
                client_id="client_tc02",
                dataset_name="ds_tc02",
                total_images=2,
                total_annotations=5,
            )
        )

    responses = list(client.stub.Evaluate(stream_meta()))
    assert len(responses) == 1
    resp = responses[0]
    assert resp.WhichOneof("payload") == "acknowledgement"
    ack = resp.acknowledgement
    assert ack.job_id == "job_tc02"
    assert ack.ack_type == pb2.AckType.ACK_JOB_STARTED


def test_tc03_binary_transport(clean_server):
    """TC03: Raw binary bytes transfer, SHA-256 match, no Base64 corruption."""
    inst = clean_server()
    client = T11Client(target=inst["target"])
    raw_bytes = b"\x00\xff\xfe\x01\x42" * 5000  # 25,000 bytes binary with non-ASCII and null bytes
    expected_sha = hashlib.sha256(raw_bytes).hexdigest()

    def stream_binary():
        yield pb2.EvaluateRequest(
            job_metadata=pb2.JobMetadata(job_id="job_tc03", run_id="run_01", total_images=1)
        )
        yield pb2.EvaluateRequest(
            image_chunk=pb2.ImageChunk(
                image_id="img_tc03",
                chunk_index=0,
                chunk_data=raw_bytes,
                is_last=True,
                total_chunks=1,
                sha256=expected_sha,
            )
        )

    responses = list(client.stub.Evaluate(stream_binary()))
    ack_imgs = [r for r in responses if r.WhichOneof("payload") == "acknowledgement" and r.acknowledgement.ack_type == pb2.AckType.ACK_IMAGE_RECEIVED]
    assert len(ack_imgs) == 1

    # Verify on filesystem
    storage = inst["engine"].storage
    image_path = storage.get_image_path("job_tc03", "run_01", "img_tc03", extension="jpg")
    assert image_path.exists()
    disk_bytes = image_path.read_bytes()
    assert disk_bytes == raw_bytes
    assert hashlib.sha256(disk_bytes).hexdigest() == expected_sha


def test_tc04_complete_job(clean_server):
    """TC04: Complete evaluation job with multiple images and annotations."""
    inst = clean_server()
    client = T11Client(target=inst["target"])

    fixture_dir = Path("fixtures/t11_label_errors/images")
    img_files = sorted(fixture_dir.glob("*.jpg"))[:3]
    assert len(img_files) == 3

    images_payload = [{"image_id": f.stem, "path": f} for f in img_files]
    annotations_payload = [
        {
            "ann_id": f"ann_{i}",
            "image_id": img_files[i % 3].stem,
            "class_name": "car",
            "confidence": 0.85,
            "bbox": {"x": 10.0, "y": 10.0, "w": 40.0, "h": 30.0},
        }
        for i in range(6)
    ]

    responses = list(
        client.evaluate_job(
            job_id="job_tc04",
            run_id="run_01",
            images=images_payload,
            annotations=annotations_payload,
        )
    )

    types = [r.WhichOneof("payload") for r in responses]
    assert "acknowledgement" in types
    assert "image_result" in types
    assert "annotation_result" in types
    assert "job_result" in types
    assert "job_completed" in types

    completed_msgs = [r.job_completed for r in responses if r.WhichOneof("payload") == "job_completed"]
    assert len(completed_msgs) == 1
    assert completed_msgs[0].final_status == pb2.JobStatus.COMPLETED

    # Check database status
    job_rec = inst["engine"].db.get_job("job_tc04", "run_01")
    assert job_rec["status"] == "COMPLETED"
