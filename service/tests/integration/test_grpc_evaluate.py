"""Integration tests for gRPC T11 Evaluator Service (Single RPC Evaluate)."""
from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import grpc
import pytest

from service.client.example_client import T11Client
from service.core.processing_engine import ProcessingEngine
from service.generated import pb2
from service.server import create_server
from service.server.config import ServiceConfig


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def grpc_service(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("grpc_test")
    sock_path = Path("/tmp/t11_test_eval.sock")
    if sock_path.exists():
        sock_path.unlink()
    target = f"unix:{sock_path}"

    cfg = ServiceConfig()
    cfg.grpc.host = target
    cfg.grpc.port = 0
    cfg.storage.temp_path = str(tmp / "temp")
    cfg.storage.result_path = str(tmp / "results")
    cfg.storage.database_path = str(tmp / "t11_test.sqlite")
    cfg.models.yolo.mock = True
    cfg.models.sam3.mock = True

    engine = ProcessingEngine(cfg)
    server = create_server(cfg, engine=engine)
    server.start()
    time.sleep(0.2)

    yield {"target": target, "engine": engine, "config": cfg}

    server.stop(grace=0.5)
    if sock_path.exists():
        sock_path.unlink()


def test_grpc_evaluate_end_to_end(grpc_service):
    client = T11Client(target=grpc_service["target"])

    fixture_dir = Path("fixtures/t11_label_errors")
    image_paths = sorted((fixture_dir / "images").glob("*.jpg"))[:3]
    assert len(image_paths) == 3

    images_payload = [{"image_id": p.stem, "path": p} for p in image_paths]
    annotations_payload = [
        {
            "ann_id": "ann_test_01",
            "image_id": image_paths[0].stem,
            "class_name": "car",
            "confidence": 0.9,
            "bbox": {"x": 20.0, "y": 30.0, "w": 50.0, "h": 30.0},
        },
        {
            "ann_id": "ann_test_02",
            "image_id": image_paths[1].stem,
            "class_name": "pedestrian",
            "confidence": 0.85,
            "bbox": {"x": 100.0, "y": 40.0, "w": 15.0, "h": 35.0},
        },
    ]

    job_id = "integration_job_01"
    run_id = "run_01"

    responses = list(
        client.evaluate_job(
            job_id=job_id,
            run_id=run_id,
            images=images_payload,
            annotations=annotations_payload,
        )
    )

    types_received = [r.WhichOneof("payload") for r in responses]

    # Verify key response types are received in stream
    assert "acknowledgement" in types_received
    assert "progress" in types_received
    assert "annotation_result" in types_received
    assert "image_result" in types_received
    assert "job_result" in types_received
    assert "job_completed" in types_received

    # Verify final result content
    job_completed = next(r.job_completed for r in responses if r.WhichOneof("payload") == "job_completed")
    assert job_completed.final_status == pb2.JobStatus.COMPLETED

    job_result = next(r.job_result for r in responses if r.WhichOneof("payload") == "job_result")
    assert job_result.total_images_evaluated == 3
    assert job_result.total_annotations_evaluated == 2
    assert len(job_result.top_item_ids) > 0

    client.close()


def test_grpc_resume_and_reconciliation(grpc_service):
    client = T11Client(target=grpc_service["target"])
    job_id = "integration_job_01"
    run_id = "run_01"

    state = client.resume_job(job_id, run_id)
    assert state.job_id == job_id
    assert state.run_id == run_id
    assert len(state.completed_image_ids) == 3
    assert state.current_job_status == pb2.JobStatus.COMPLETED

    client.close()
