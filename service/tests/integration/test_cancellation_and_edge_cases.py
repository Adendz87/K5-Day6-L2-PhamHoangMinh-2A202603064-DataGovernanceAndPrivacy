"""Integration tests for Cancellation, Duplication, and Resume workflows."""
import time
from pathlib import Path
import pytest

from service.client.example_client import T11Client
from service.core.processing_engine import ProcessingEngine
from service.generated import pb2
from service.server import create_server
from service.server.config import ServiceConfig


@pytest.fixture(scope="module")
def cancel_grpc_service(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("grpc_cancel_test")
    sock_path = Path("/tmp/t11_cancel_test.sock")
    if sock_path.exists():
        sock_path.unlink()
    target = f"unix:{sock_path}"

    cfg = ServiceConfig()
    cfg.grpc.host = target
    cfg.grpc.port = 0
    cfg.storage.temp_path = str(tmp / "temp")
    cfg.storage.result_path = str(tmp / "results")
    cfg.storage.database_path = str(tmp / "cancel.sqlite")
    cfg.models.yolo.mock = True
    cfg.models.sam3.mock = True

    engine = ProcessingEngine(cfg)
    server = create_server(cfg, engine=engine)
    server.start()
    time.sleep(0.2)

    yield {"target": target, "engine": engine}

    server.stop(grace=0.5)
    if sock_path.exists():
        sock_path.unlink()


def test_job_cancellation(cancel_grpc_service):
    client = T11Client(target=cancel_grpc_service["target"])
    job_id = "job_cancel_01"

    # Start job metadata then immediately cancel
    def stream_cancel():
        yield pb2.EvaluateRequest(
            job_metadata=pb2.JobMetadata(job_id=job_id, run_id="run_01", total_images=5)
        )
        yield pb2.EvaluateRequest(
            cancel_request=pb2.CancelRequest(job_id=job_id, reason="User cancelled early")
        )

    responses = list(client.stub.Evaluate(stream_cancel()))
    assert len(responses) >= 2
    progress_responses = [r.progress for r in responses if r.WhichOneof("payload") == "progress"]
    assert any(p.job_status == pb2.JobStatus.CANCELLED for p in progress_responses)

    # Check status in database
    engine: ProcessingEngine = cancel_grpc_service["engine"]
    job = engine.db.get_job(job_id, "run_01")
    assert job["status"] == "CANCELLED"
    client.close()


def test_idempotent_duplicate_metadata(cancel_grpc_service):
    client = T11Client(target=cancel_grpc_service["target"])
    job_id = "job_idem_01"

    # Send same job metadata twice with same idempotency key
    def stream_duplicate():
        yield pb2.EvaluateRequest(
            job_metadata=pb2.JobMetadata(
                job_id=job_id,
                run_id="run_01",
                metadata={"idempotency_key": "unique_key_123"},
                total_images=2,
            )
        )
        yield pb2.EvaluateRequest(
            job_metadata=pb2.JobMetadata(
                job_id=job_id,
                run_id="run_01",
                metadata={"idempotency_key": "unique_key_123"},
                total_images=2,
            )
        )
        yield pb2.EvaluateRequest(end_of_input=pb2.EndOfInput(images_sent=0, annotations_sent=0))

    responses = list(client.stub.Evaluate(stream_duplicate()))
    acks = [r.acknowledgement for r in responses if r.WhichOneof("payload") == "acknowledgement"]
    assert len(acks) >= 2
    client.close()
