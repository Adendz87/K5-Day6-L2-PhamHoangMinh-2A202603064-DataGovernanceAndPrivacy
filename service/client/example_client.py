"""Python reference client demonstrating bidirectional streaming evaluation over a single RPC."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import grpc

from service.generated import pb2, pb2_grpc


class T11Client:
    """Reference Python Client for T11 Standalone Data Quality Assessment Service."""

    def __init__(
        self,
        target: str = "localhost:50051",
        chunk_size: int = 1048576,  # 1 MB
        max_message_size_mb: int = 32,
    ) -> None:
        self.target = target
        self.chunk_size = chunk_size
        max_bytes = max_message_size_mb * 1024 * 1024
        options = [
            ("grpc.max_receive_message_length", max_bytes),
            ("grpc.max_send_message_length", max_bytes),
        ]
        self.channel = grpc.insecure_channel(target, options=options)
        self.stub = pb2_grpc.T11EvaluatorStub(self.channel)

    def evaluate_job(
        self,
        job_id: str,
        run_id: str,
        images: List[Dict[str, Any]],  # list of {"image_id": str, "path": Path}
        annotations: List[Dict[str, Any]],  # list of annotations
        dataset_name: str = "evaluation_dataset",
    ) -> Generator[pb2.EvaluateResponse, None, None]:
        """Execute complete job evaluation over single bidirectional stream."""

        def request_generator() -> Generator[pb2.EvaluateRequest, None, None]:
            # Step 1: Send JobMetadata
            meta = pb2.JobMetadata(
                job_id=job_id,
                run_id=run_id,
                dataset_name=dataset_name,
                total_images=len(images),
                total_annotations=len(annotations),
            )
            yield pb2.EvaluateRequest(job_metadata=meta)

            # Step 2: Stream Image Chunks (Binary, no base64)
            total_chunks_sent = 0
            for img_info in images:
                img_id = img_info["image_id"]
                p = Path(img_info["path"])
                with open(p, "rb") as f:
                    file_bytes = f.read()

                full_sha = hashlib.sha256(file_bytes).hexdigest()
                total_len = len(file_bytes)
                num_chunks = max(1, (total_len + self.chunk_size - 1) // self.chunk_size)

                for chunk_idx in range(num_chunks):
                    start = chunk_idx * self.chunk_size
                    end = min(total_len, start + self.chunk_size)
                    chunk_slice = file_bytes[start:end]
                    is_last = (chunk_idx == num_chunks - 1)

                    chunk_msg = pb2.ImageChunk(
                        image_id=img_id,
                        chunk_index=chunk_idx,
                        chunk_data=chunk_slice,
                        is_last=is_last,
                        total_chunks=num_chunks,
                        sha256=full_sha if is_last else "",
                    )
                    total_chunks_sent += 1
                    yield pb2.EvaluateRequest(image_chunk=chunk_msg)

            # Step 3: Stream Annotations in Batches
            if annotations:
                ann_items = []
                for a in annotations:
                    bbox = a.get("bbox", {})
                    ann_items.append(
                        pb2.Annotation(
                            ann_id=str(a.get("ann_id", "")),
                            image_id=str(a.get("image_id", "")),
                            class_name=str(a.get("class_name", "")),
                            bbox=pb2.BoundingBox(
                                x=float(bbox.get("x", 0.0)),
                                y=float(bbox.get("y", 0.0)),
                                w=float(bbox.get("w", 0.0)),
                                h=float(bbox.get("h", 0.0)),
                            ),
                            confidence=float(a.get("confidence", 0.0)),
                        )
                    )
                batch_msg = pb2.AnnotationBatch(
                    batch_id=f"batch_{job_id}_01",
                    items=ann_items,
                )
                yield pb2.EvaluateRequest(annotation_batch=batch_msg)

            # Step 4: Send EndOfInput
            yield pb2.EvaluateRequest(
                end_of_input=pb2.EndOfInput(
                    images_sent=len(images),
                    chunks_sent=total_chunks_sent,
                    annotations_sent=len(annotations),
                )
            )

        # Step 5: Receive stream responses
        response_iterator = self.stub.Evaluate(request_generator())
        for response in response_iterator:
            yield response

    def resume_job(
        self,
        job_id: str,
        run_id: str,
    ) -> pb2.ReceivedState:
        """Send ResumeRequest and retrieve server-side reception state."""
        def resume_gen():
            yield pb2.EvaluateRequest(
                resume_request=pb2.ResumeRequest(
                    job_id=job_id,
                    run_id=run_id,
                )
            )

        responses = self.stub.Evaluate(resume_gen())
        for resp in responses:
            if resp.WhichOneof("payload") == "received_state":
                return resp.received_state
        raise RuntimeError("No ReceivedState response received from server")

    def cancel_job(
        self,
        job_id: str,
        reason: str = "Client requested cancellation",
    ) -> None:
        """Send CancelRequest."""
        def cancel_gen():
            yield pb2.EvaluateRequest(
                cancel_request=pb2.CancelRequest(
                    job_id=job_id,
                    reason=reason,
                )
            )

        responses = self.stub.Evaluate(cancel_gen())
        for _ in responses:
            pass

    def close(self) -> None:
        self.channel.close()
