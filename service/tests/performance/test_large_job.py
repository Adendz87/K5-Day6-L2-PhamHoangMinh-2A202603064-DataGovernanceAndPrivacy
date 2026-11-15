"""Performance benchmarks for Large Job processing."""
from __future__ import annotations

import time
from pathlib import Path

from service.core.processing_engine import ProcessingEngine
from service.server.config import ServiceConfig


def test_batch_processing_performance_100_images(tmp_path: Path):
    cfg = ServiceConfig()
    cfg.storage.temp_path = str(tmp_path / "temp")
    cfg.storage.result_path = str(tmp_path / "results")
    cfg.storage.database_path = str(tmp_path / "perf.sqlite")
    cfg.models.yolo.mock = True
    cfg.models.sam3.mock = True
    cfg.processing.batch_size = 32

    engine = ProcessingEngine(cfg)

    fixture_dir = Path("fixtures/t11_label_errors")
    sample_img = next((fixture_dir / "images").glob("*.jpg"))
    img_bytes = sample_img.read_bytes()

    job_id = "perf_job_100"
    run_id = "run_01"
    num_images = 100

    # 1. Measure Ingestion & Storage throughput
    t0_ingest = time.perf_counter()
    engine.input_manager.handle_job_metadata({
        "job_id": job_id,
        "run_id": run_id,
        "total_images": num_images,
        "total_annotations": num_images * 2,
    })

    for i in range(num_images):
        iid = f"img_{i:04d}"
        engine.input_manager.handle_image_chunk(job_id, run_id, {
            "image_id": iid,
            "chunk_index": 0,
            "chunk_data": img_bytes,
            "is_last": True,
            "total_chunks": 1,
        })
        engine.input_manager.handle_annotation_batch(job_id, run_id, {
            "batch_id": f"b_{i}",
            "items": [
                {"ann_id": f"a_{i}_1", "image_id": iid, "class_name": "car", "bbox": {"x": 10, "y": 10, "w": 40, "h": 30}},
                {"ann_id": f"a_{i}_2", "image_id": iid, "class_name": "pedestrian", "bbox": {"x": 60, "y": 20, "w": 15, "h": 35}},
            ],
        })

    engine.input_manager.handle_end_of_input(job_id, run_id, {
        "images_sent": num_images,
        "annotations_sent": num_images * 2,
    })
    ingest_time = time.perf_counter() - t0_ingest

    # 2. Measure Evaluation & Scoring throughput
    t0_eval = time.perf_counter()
    responses = list(engine.process_job(job_id, run_id))
    eval_time = time.perf_counter() - t0_eval

    total_time = ingest_time + eval_time
    fps = num_images / max(0.001, eval_time)

    print(
        f"\n[PERF REPORT - {num_images} images]: "
        f"Ingest: {ingest_time:.2f}s | Eval: {eval_time:.2f}s | Total: {total_time:.2f}s | "
        f"Throughput: {fps:.1f} img/sec"
    )

    assert len(responses) > num_images
    job_completed = [r.job_completed for r in responses if r.WhichOneof("payload") == "job_completed"]
    assert len(job_completed) == 1
    assert fps > 10.0  # Must achieve at least 10 images/sec on mock pipeline
