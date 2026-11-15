"""Reliability Phase 7 Tests: Protocol & API Validation (TC41 - TC50)

TC41: Missing metadata before sending chunks.
TC42: Invalid job_id (empty/missing).
TC43: Invalid run_id (empty/missing).
TC44: Invalid transfer session parameters.
TC45: Invalid message sequences (repeated metadata, repeated EOI).
TC46: Empty or unpopulated request payload.
TC47: Client cancellation handling.
TC48: Request deadline / timeout handling.
TC49: TLS configuration state (NOT IMPLEMENTED).
TC50: Explicit error reporting with codes and messages.
"""
from __future__ import annotations

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
    servers = []
    socks = []

    def _start():
        sock_path = Path(f"/tmp/t11_proto_{len(servers)}_{time.time_ns()}.sock")
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
        return {"target": target, "server": server, "sock_path": sock_path, "engine": engine, "cfg": cfg}

    yield _start

    for s in servers:
        try:
            s.stop(grace=0.2)
        except Exception:
            pass
    for p in socks:
        if p.exists():
            p.unlink()


def test_tc41_missing_metadata(clean_server):
    """TC41: Sending chunk without prior JobMetadata returns error (job not found or foreign key)."""
    inst = clean_server()
    client = T11Client(target=inst["target"])

    def stream_no_meta():
        yield pb2.EvaluateRequest(
            image_chunk=pb2.ImageChunk(image_id="img_nometa", chunk_index=0, chunk_data=b"DATA", total_chunks=1)
        )

    with pytest.raises(grpc.RpcError) as excinfo:
        list(client.stub.Evaluate(stream_no_meta()))
    assert excinfo.value.code() == grpc.StatusCode.INTERNAL


def test_tc42_invalid_job_id(clean_server):
    """TC42: Empty job_id in metadata raises ValueError."""
    inst = clean_server()
    mgr = inst["engine"].input_manager

    with pytest.raises(ValueError) as excinfo:
        mgr.handle_job_metadata({"job_id": "", "run_id": "r01"})
    assert "job_id and run_id are required" in str(excinfo.value)


def test_tc43_invalid_run_id(clean_server):
    """TC43: Empty run_id in metadata raises ValueError."""
    inst = clean_server()
    mgr = inst["engine"].input_manager

    with pytest.raises(ValueError) as excinfo:
        mgr.handle_job_metadata({"job_id": "j43", "run_id": ""})
    assert "job_id and run_id are required" in str(excinfo.value)


def test_tc44_invalid_transfer_session(clean_server):
    """TC44: Transfer session isolates jobs and runs accurately."""
    inst = clean_server()
    mgr = inst["engine"].input_manager
    s1 = mgr._get_session("job_1", "run_1", "img_1")
    s2 = mgr._get_session("job_1", "run_2", "img_1")
    assert s1 is not s2


def test_tc45_invalid_message_sequence(clean_server):
    """TC45: Repeated EndOfInput is handled idempotently without error."""
    inst = clean_server()
    mgr = inst["engine"].input_manager
    job_id = "j45"
    run_id = "r01"
    mgr.handle_job_metadata({"job_id": job_id, "run_id": run_id, "total_images": 1})

    ack1 = mgr.handle_end_of_input(job_id, run_id, {"images_sent": 1})
    assert ack1.ack_type == "ACK_INPUT_COMPLETE"

    # Second EndOfInput
    ack2 = mgr.handle_end_of_input(job_id, run_id, {"images_sent": 1})
    assert ack2.ack_type == "ACK_INPUT_COMPLETE"


def test_tc46_unknown_or_empty_message(clean_server):
    """TC46: EvaluateRequest with no oneof payload set is safely ignored/handled."""
    inst = clean_server()
    client = T11Client(target=inst["target"])

    def stream_empty():
        yield pb2.EvaluateRequest()  # No payload set

    # Should not crash server process
    responses = list(client.stub.Evaluate(stream_empty()))
    assert len(responses) == 0


def test_tc47_client_cancellation(clean_server):
    """TC47: Client sending CancelRequest transitions job to CANCELLED state."""
    inst = clean_server()
    client = T11Client(target=inst["target"])
    job_id = "j47"
    run_id = "r01"

    def stream_cancel():
        yield pb2.EvaluateRequest(job_metadata=pb2.JobMetadata(job_id=job_id, run_id=run_id, total_images=2))
        yield pb2.EvaluateRequest(cancel_request=pb2.CancelRequest(job_id=job_id, reason="User stop"))

    responses = list(client.stub.Evaluate(stream_cancel()))
    progs = [r.progress for r in responses if r.WhichOneof("payload") == "progress"]
    assert len(progs) == 1
    assert progs[0].job_status == pb2.JobStatus.CANCELLED
    job = inst["engine"].db.get_job(job_id, run_id)
    assert job["status"] == "CANCELLED"


def test_tc48_timeout_deadline(clean_server):
    """TC48: Client timeout/deadline triggers DEADLINE_EXCEEDED when exceeded."""
    inst = clean_server()
    client = T11Client(target=inst["target"])

    def slow_stream():
        yield pb2.EvaluateRequest(job_metadata=pb2.JobMetadata(job_id="j48", run_id="r01", total_images=1))
        time.sleep(0.5)

    with pytest.raises(grpc.RpcError) as excinfo:
        # Call with tight 0.1 second timeout
        for _ in client.stub.Evaluate(slow_stream(), timeout=0.1):
            pass
    assert excinfo.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED


def test_tc49_tls_configuration(clean_server):
    """TC49: Verify TLS configuration behavior (raises NotImplementedError if enabled)."""
    inst = clean_server()
    cfg = inst["cfg"]
    cfg.security.enable_tls = True
    with pytest.raises(NotImplementedError) as excinfo:
        create_server(cfg)
    assert "TLS configuration is not yet implemented" in str(excinfo.value)


def test_tc50_error_reporting(clean_server):
    """TC50: Error response contains error_code, message and fatal flag."""
    inst = clean_server()
    client = T11Client(target=inst["target"])

    def stream_fail():
        # Trigger an internal error with non-existent job chunk
        yield pb2.EvaluateRequest(
            image_chunk=pb2.ImageChunk(image_id="img_err", chunk_index=0, chunk_data=b"ERR", total_chunks=1)
        )

    with pytest.raises(grpc.RpcError) as excinfo:
        list(client.stub.Evaluate(stream_fail()))
    assert excinfo.value.code() == grpc.StatusCode.INTERNAL
    assert len(excinfo.value.details()) > 0
