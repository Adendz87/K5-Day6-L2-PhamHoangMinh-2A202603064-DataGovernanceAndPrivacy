# T11 Standalone Data Quality Assessment Service

T11 is a standalone post-label annotation quality assessment service designed to integrate with CVAT and other annotation platforms via high-throughput gRPC binary streaming.

---

## 1. Architecture Highlights

- **Single Business RPC (`Evaluate`)**: Entire communication (job initialization, binary chunk transfer, annotation ingestion, progress monitoring, hierarchical scoring, resume, and cancellation) occurs over a single bidirectional streaming RPC.
- **Zero Base64 Binary Streaming**: Raw image bytes are transferred directly in Protocol Buffer `bytes` fields via chunks with SHA-256 verification.
- **Hierarchical Multi-Level Risk Scoring**:
  - **Annotation level**: Suspicion scoring across 4 major error categories (`class_swap`, `box_shift`, `duplicate`, `missing`).
  - **Image level**: Weighted aggregation ($0.6 \times \text{max} + 0.3 \times p_{90} + 0.1 \times \text{missing}$).
  - **Job level**: Robust $p_{80}$ percentile score across all images.
- **Persistent State & Crash Recovery**: SQLite database in WAL mode tracks chunk receipts, idempotency keys, delivery sequences, and per-image lifecycle states. Interrupted transfers resume without re-sending completed images.
- **Complete Decoupling**: Processing engine is completely decoupled from gRPC handlers and works 100% offline.

---

## 2. Directory Structure

```text
service/
├── proto/
│   └── t11.proto                  # Protocol Buffers contract (Single RPC Evaluate)
├── generated/                     # Compiled Python gRPC stubs
│   ├── t11_pb2.py
│   └── t11_pb2_grpc.py
├── server/
│   ├── __init__.py                # Server factory (create_server)
│   ├── grpc_handler.py            # T11EvaluatorServicer implementation
│   └── config.py                  # Typed YAML configuration loader
├── core/
│   ├── temporary_storage.py       # Safe chunk assembly and disk manager
│   ├── database.py                # SQLite persistence (WAL mode)
│   ├── job_lifecycle.py           # Explicit state machines
│   ├── transfer_session.py        # Chunk receipts and window tracking
│   ├── input_manager.py           # Stream event ingestion & validation
│   ├── annotation_normalizer.py   # Format adapter & bounds validator
│   ├── batch_manager.py           # Bounded queue & backpressure
│   ├── resource_manager.py        # Concurrency & disk quota guards
│   ├── result_storage.py          # Results persistence & replay sequences
│   ├── run_recovery.py            # Startup crash recovery
│   └── processing_engine.py       # Pipeline orchestrator
├── adapters/
│   ├── yolo_adapter.py            # Local YOLO detector (Mock & Ultralytics)
│   └── sam3_adapter.py            # Local SAM 3 boundary evidence (Mock)
├── analysis/
│   ├── matching_adapter.py        # Safe bridge to src.t11.matching & geometry
│   ├── error_analysis.py          # Candidate detection & evidence packaging
│   └── risk_scorer.py             # Multi-level scoring engine
├── result/
│   ├── aggregator.py              # Review queue ranking
│   └── result_manager.py          # Protobuf response factory
├── client/
│   └── example_client.py          # Python reference client SDK
├── configs/
│   └── service.yaml               # Central service configuration
└── tests/
    ├── unit/                      # Unit tests (storage, db, lifecycle, normalizer)
    ├── integration/               # End-to-end gRPC stream tests
    ├── offline/                   # Offline verification test
    └── performance/               # Large job throughput benchmarks
```

---

## 3. Protobuf Contract

Defined in `service/proto/t11.proto`:

```protobuf
syntax = "proto3";

package t11service;

service T11Evaluator {
  rpc Evaluate(stream EvaluateRequest) returns (stream EvaluateResponse);
}
```

### Request Payloads (`EvaluateRequest`)
1. `JobMetadata`: Job metadata, expected image & annotation counts, idempotency key.
2. `ImageChunk`: Raw binary `bytes chunk_data`, sequence index, SHA-256 digest.
3. `AnnotationBatch`: Grouped annotations with pixel coordinates $(x, y, w, h)$.
4. `EndOfInput`: Signals completion of client data ingestion.
5. `ResumeRequest`: Requests server reconciliation state for interrupted transfers.
6. `CancelRequest`: Requests clean abort of active evaluation.
7. `ResultAcknowledgement`: Acknowledges received result sequences.

### Response Payloads (`EvaluateResponse`)
1. `Acknowledgement`: Immediate acknowledgment of metadata, chunks, and batches.
2. `ReceivedState`: Detailed state for resume and reconnection.
3. `EvaluationProgress`: Real-time status, processing percentage, and image milestones.
4. `AnnotationResult`: Scored annotations with evidence, IoU, and suggested actions.
5. `MissingCandidate`: Model-detected objects without corresponding annotations.
6. `ImageResult`: Aggregated image suspicion scores and risk levels.
7. `JobResult`: Final ranking of review queue items and overall job statistics.
8. `JobCompleted`: Completion signal with total runtime duration.
9. `EvaluationError`: Structured error reporting.

---

## 4. Running the Tests

### Service Tests
```bash
.venv/bin/python -m pytest service/tests/ -v
```

### Mode A Regression Tests (Protected Benchmark)
```bash
.venv/bin/python -m pytest src/t11/tests/ tests/ -v
```

### Full Repository Verification
```bash
.venv/bin/python -m pytest tests/ src/t11/tests/ service/tests/ -v
```

---

## 5. Quickstart Example Client

```python
from service.client.example_client import T11Client

client = T11Client(target="localhost:50051")

images = [{"image_id": "img_01", "path": "fixtures/t11_label_errors/images/lbl_00.jpg"}]
annotations = [{
    "ann_id": "ann_01",
    "image_id": "img_01",
    "class_name": "car",
    "bbox": {"x": 218.0, "y": 119.0, "w": 53.5, "h": 31.6},
}]

for response in client.evaluate_job("job_123", "run_01", images, annotations):
    payload_type = response.WhichOneof("payload")
    if payload_type == "job_result":
        print("Job Result:", response.job_result.top_item_ids)
    elif payload_type == "job_completed":
        print(f"Finished in {response.job_completed.duration_ms} ms")
```
