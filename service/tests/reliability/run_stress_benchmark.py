"""Performance & Stress Benchmark Script (Giai đoạn 9)

Benchmarks T11 gRPC Evaluate API across scales: 1, 10, 100 images.
Measures:
- Upload throughput (MB/s)
- Upload time (s)
- Processing time (s)
- End-to-end latency (s)
- Peak RAM (MB)
- Chunk throughput (chunks/s)
- Image throughput (images/s)
- Success rate (%)
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import resource
import shutil
import tempfile
import time
from pathlib import Path

from service.client.example_client import T11Client
from service.core.processing_engine import ProcessingEngine
from service.generated import pb2
from service.server import create_server
from service.server.config import ServiceConfig


def get_peak_memory_mb() -> float:
    # ru_maxrss is in bytes on macOS, in KB on Linux
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def run_scale_test(num_images: int, chunk_size_kb: int = 256) -> dict:
    temp_dir = Path(tempfile.mkdtemp(prefix=f"t11_perf_{num_images}_"))
    sock_path = Path(f"/tmp/t11_perf_{num_images}_{time.time_ns()}.sock")
    if sock_path.exists():
        sock_path.unlink()
    target = f"unix:{sock_path}"

    cfg = ServiceConfig()
    cfg.grpc.host = target
    cfg.grpc.port = 0
    cfg.storage.temp_path = str(temp_dir / "temp")
    cfg.storage.result_path = str(temp_dir / "results")
    cfg.storage.database_path = str(temp_dir / "db.sqlite")
    cfg.models.yolo.mock = True
    cfg.models.sam3.mock = True
    cfg.transfer.chunk_size_bytes = chunk_size_kb * 1024

    engine = ProcessingEngine(cfg)
    server = create_server(cfg, engine=engine)
    server.start()
    time.sleep(0.2)

    client = T11Client(target=target, chunk_size=cfg.transfer.chunk_size_bytes)
    job_id = f"job_perf_{num_images}"
    run_id = "run_01"

    # Prepare dummy image data
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * (chunk_size_kb * 1024 * 2)  # 2 chunks per image
    total_payload_bytes = len(image_bytes) * num_images
    total_payload_mb = total_payload_bytes / (1024 * 1024)
    img_sha = hashlib.sha256(image_bytes).hexdigest()

    # Track metrics
    start_rss = get_peak_memory_mb()
    t_start = time.perf_counter()

    def stream_requests():
        yield pb2.EvaluateRequest(
            job_metadata=pb2.JobMetadata(
                job_id=job_id,
                run_id=run_id,
                total_images=num_images,
                total_annotations=num_images * 2,
            )
        )
        for i in range(num_images):
            img_id = f"perf_img_{i:04d}"
            # 2 chunks per image
            half = len(image_bytes) // 2
            c0 = image_bytes[:half]
            c1 = image_bytes[half:]
            yield pb2.EvaluateRequest(
                image_chunk=pb2.ImageChunk(
                    image_id=img_id,
                    chunk_index=0,
                    chunk_data=c0,
                    is_last=False,
                    total_chunks=2,
                )
            )
            yield pb2.EvaluateRequest(
                image_chunk=pb2.ImageChunk(
                    image_id=img_id,
                    chunk_index=1,
                    chunk_data=c1,
                    is_last=True,
                    total_chunks=2,
                    sha256=img_sha,
                )
            )

        # Send annotation batch
        ann_items = []
        for i in range(num_images):
            ann_items.append(
                pb2.Annotation(
                    ann_id=f"ann_{i}_1",
                    image_id=f"perf_img_{i:04d}",
                    class_name="car",
                    bbox=pb2.BoundingBox(x=10, y=10, w=30, h=30),
                    confidence=0.9,
                )
            )
        yield pb2.EvaluateRequest(
            annotation_batch=pb2.AnnotationBatch(batch_id="batch_perf", items=ann_items)
        )

        yield pb2.EvaluateRequest(
            end_of_input=pb2.EndOfInput(
                images_sent=num_images,
                chunks_sent=num_images * 2,
                annotations_sent=len(ann_items),
            )
        )

    t_upload_start = time.perf_counter()
    responses = list(client.stub.Evaluate(stream_requests()))
    t_end = time.perf_counter()

    server.stop(grace=0.1)
    if sock_path.exists():
        sock_path.unlink()
    shutil.rmtree(temp_dir, ignore_errors=True)

    end_rss = get_peak_memory_mb()
    total_duration = t_end - t_start
    total_chunks = num_images * 2
    chunks_per_sec = total_chunks / max(total_duration, 0.001)
    images_per_sec = num_images / max(total_duration, 0.001)
    throughput_mb_s = total_payload_mb / max(total_duration, 0.001)

    completed = any(r.WhichOneof("payload") == "job_completed" for r in responses)

    return {
        "num_images": num_images,
        "total_payload_mb": round(total_payload_mb, 2),
        "total_chunks": total_chunks,
        "total_duration_s": round(total_duration, 4),
        "throughput_mb_s": round(throughput_mb_s, 2),
        "chunks_per_sec": round(chunks_per_sec, 1),
        "images_per_sec": round(images_per_sec, 1),
        "peak_ram_mb": round(end_rss, 2),
        "ram_growth_mb": round(max(0, end_rss - start_rss), 2),
        "success": completed,
        "num_responses": len(responses),
    }


if __name__ == "__main__":
    results = []
    print("Running performance benchmark on 1, 10, 100 images...")
    for n in [1, 10, 100]:
        res = run_scale_test(n)
        results.append(res)
        print(f"Scale {n:3d} images: {res['total_duration_s']:6.2f}s | {res['throughput_mb_s']:6.2f} MB/s | {res['images_per_sec']:5.1f} img/s | RAM: {res['peak_ram_mb']} MB | Success: {res['success']}")

    out_file = Path("service/reports/benchmark_metrics.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark results to {out_file}")
