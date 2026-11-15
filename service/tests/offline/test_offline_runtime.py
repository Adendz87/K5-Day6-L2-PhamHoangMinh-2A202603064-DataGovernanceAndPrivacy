"""Offline verification test guaranteeing runtime has zero external network calls."""
from __future__ import annotations

import socket
from pathlib import Path
import pytest

from service.core.processing_engine import ProcessingEngine
from service.server.config import ServiceConfig


def test_offline_initialization_and_processing(tmp_path: Path, monkeypatch):
    # Guard: monkeypatch socket.connect to block any external connection
    original_connect = socket.socket.connect

    def guarded_connect(self, address):
        # Allow AF_UNIX or localhost (127.0.0.1)
        if isinstance(address, tuple):
            host, _ = address
            if host not in ("127.0.0.1", "localhost", "::1"):
                raise ConnectionRefusedError(f"Offline guard: network access to {host} is blocked")
        return original_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

    cfg = ServiceConfig()
    cfg.storage.temp_path = str(tmp_path / "temp")
    cfg.storage.result_path = str(tmp_path / "results")
    cfg.storage.database_path = str(tmp_path / "offline.sqlite")
    cfg.models.yolo.mock = True
    cfg.models.sam3.mock = True

    # Initialize complete engine offline
    engine = ProcessingEngine(cfg)

    # Setup 1 synthetic image and annotation offline
    fixture_dir = Path("fixtures/t11_label_errors")
    img_path = next((fixture_dir / "images").glob("*.jpg"))
    img_bytes = img_path.read_bytes()

    job_id = "offline_job"
    run_id = "run_01"

    engine.input_manager.handle_job_metadata({
        "job_id": job_id,
        "run_id": run_id,
        "total_images": 1,
        "total_annotations": 1,
    })

    engine.input_manager.handle_image_chunk(job_id, run_id, {
        "image_id": "img_offline_1",
        "chunk_index": 0,
        "chunk_data": img_bytes,
        "is_last": True,
        "total_chunks": 1,
    })

    engine.input_manager.handle_annotation_batch(job_id, run_id, {
        "batch_id": "b1",
        "items": [{
            "ann_id": "ann_off_1",
            "image_id": "img_offline_1",
            "class_name": "car",
            "bbox": {"x": 10.0, "y": 10.0, "w": 30.0, "h": 20.0},
        }],
    })

    engine.input_manager.handle_end_of_input(job_id, run_id, {
        "images_sent": 1,
        "annotations_sent": 1,
    })

    # Execute complete evaluation pipeline offline
    results = list(engine.process_job(job_id, run_id))
    assert len(results) > 0

    job_res = [r.job_result for r in results if r.WhichOneof("payload") == "job_result"]
    assert len(job_res) == 1
    assert job_res[0].total_images_evaluated == 1
