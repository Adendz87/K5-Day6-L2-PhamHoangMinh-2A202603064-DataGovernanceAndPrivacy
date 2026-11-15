"""Full End-to-End Test with Real YOLOv8n Adapter.

Tests complete evaluation of all 55 fixture images and 205 annotations
over gRPC bidirectional streaming with 1024-byte chunks, SHA-256 verification,
and real model inference (mock=False, SAM3 disabled).
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import grpc
import pytest

from service.client.example_client import T11Client
from service.core.processing_engine import ProcessingEngine
from service.generated import pb2
from service.server import create_server
from service.server.config import ServiceConfig


def test_real_yolo_e2e_55_images_205_annotations(tmp_path: Path):
    fixture_dir = Path("fixtures/t11_label_errors")
    images_csv = fixture_dir / "images.csv"
    annotations_csv = fixture_dir / "annotations.csv"

    # 1. Load 55 images
    images_payload = []
    with open(images_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = fixture_dir / row["path"]
            assert img_path.exists(), f"Image not found: {img_path}"
            images_payload.append({
                "image_id": row["image_id"],
                "path": img_path,
            })
    assert len(images_payload) == 55, f"Expected 55 images, found {len(images_payload)}"

    # 2. Load 205 annotations
    annotations_payload = []
    with open(annotations_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations_payload.append({
                "ann_id": row["ann_id"],
                "image_id": row["image_id"],
                "class_name": row["class"],
                "confidence": 0.9,
                "bbox": {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "w": float(row["w"]),
                    "h": float(row["h"]),
                },
            })
    assert len(annotations_payload) == 205, f"Expected 205 annotations, found {len(annotations_payload)}"

    # 3. Configure Real YOLO server with mock=False and sam3 disabled
    sock_path = Path(f"/tmp/t11_real_yolo_{time.time_ns()}.sock")
    if sock_path.exists():
        sock_path.unlink()
    target = f"unix:{sock_path}"

    cfg = ServiceConfig()
    cfg.grpc.host = target
    cfg.grpc.port = 0
    cfg.storage.temp_path = str(tmp_path / "temp")
    cfg.storage.result_path = str(tmp_path / "results")
    cfg.storage.database_path = str(tmp_path / "t11_e2e.sqlite")

    # Real YOLO setup
    cfg.models.yolo.mock = False
    cfg.models.yolo.weights_path = "models/yolo/yolov8n.pt"
    assert Path(cfg.models.yolo.weights_path).exists(), "Local YOLO weights must exist"

    # SAM3 disabled
    cfg.models.sam3.enabled = False
    cfg.models.sam3.mock = True

    # 4. Start Server
    engine = ProcessingEngine(cfg)
    assert not engine.yolo_adapter.config.mock, "YOLO adapter must not be in mock mode"
    assert engine.sam3_adapter is None, "SAM3 must be disabled"

    server = create_server(cfg, engine=engine)
    server.start()
    time.sleep(0.2)

    # 5. Client with chunk_size = 1024 bytes (1 KB chunks)
    client = T11Client(target=target, chunk_size=1024)
    job_id = "real_yolo_e2e_job"
    run_id = "run_full_01"

    t0 = time.perf_counter()
    responses = list(
        client.evaluate_job(
            job_id=job_id,
            run_id=run_id,
            images=images_payload,
            annotations=annotations_payload,
            dataset_name="t11_label_errors",
        )
    )
    elapsed = time.perf_counter() - t0

    server.stop(grace=0.2)
    if sock_path.exists():
        sock_path.unlink()

    # 6. Verify Acceptance Criteria
    image_results = [r.image_result for r in responses if r.WhichOneof("payload") == "image_result"]
    ann_results = [r.annotation_result for r in responses if r.WhichOneof("payload") == "annotation_result"]
    job_results = [r.job_result for r in responses if r.WhichOneof("payload") == "job_result"]
    job_completed = [r.job_completed for r in responses if r.WhichOneof("payload") == "job_completed"]
    eval_errors = [r.evaluation_error for r in responses if r.WhichOneof("payload") == "evaluation_error"]
    img_acks = [
        r.acknowledgement for r in responses
        if r.WhichOneof("payload") == "acknowledgement" and r.acknowledgement.ack_type == pb2.AckType.ACK_IMAGE_RECEIVED
    ]

    print(f"\n[Real YOLO E2E Execution Results in {elapsed:.2f}s]")
    print(f"- Total responses: {len(responses)}")
    print(f"- Image ACKs verified: {len(img_acks)}/55")
    print(f"- Image results: {len(image_results)}/55")
    print(f"- Annotation results: {len(ann_results)}/205")
    print(f"- Job completed messages: {len(job_completed)}/1")
    print(f"- Evaluation errors: {len(eval_errors)}/0")

    assert len(eval_errors) == 0, f"Encountered evaluation errors: {eval_errors}"
    assert len(img_acks) == 55, f"Expected 55 image ACKs, got {len(img_acks)}"
    assert len(image_results) == 55, f"Expected 55 image results, got {len(image_results)}"
    assert len(ann_results) == 205, f"Expected 205 annotation results, got {len(ann_results)}"
    assert len(job_results) == 1, f"Expected 1 job result, got {len(job_results)}"
    assert len(job_completed) == 1, f"Expected 1 job completed, got {len(job_completed)}"
    assert job_completed[0].final_status == pb2.JobStatus.COMPLETED

    # Check database state
    job_rec = engine.db.get_job(job_id, run_id)
    assert job_rec["status"] == "COMPLETED"
    assert job_rec["total_images"] == 55
    assert job_rec["total_annotations"] == 205

    # Check images state in SQLite
    all_db_images = engine.db.get_all_images(job_id, run_id)
    assert len(all_db_images) == 55
    for img in all_db_images:
        assert img["status"] == "IMAGE_DONE"
        assert img["received_chunks"] > 0
        assert len(img["sha256"]) == 64
