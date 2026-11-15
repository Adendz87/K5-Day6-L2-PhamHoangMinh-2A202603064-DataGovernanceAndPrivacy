from pathlib import Path
from service.client.example_client import T11Client

fixture_dir = Path("fixtures/t11_label_errors")
image_paths = sorted((fixture_dir / "images").glob("*.jpg"))[:3]

assert len(image_paths) == 3, "Không tìm thấy đủ 3 ảnh"

images = [{"image_id": p.stem, "path": p} for p in image_paths]

annotations = [
    {
        "ann_id": "standalone_ann_01",
        "image_id": image_paths[0].stem,
        "class_name": "car",
        "confidence": 0.9,
        "bbox": {"x": 20.0, "y": 30.0, "w": 50.0, "h": 30.0},
    },
    {
        "ann_id": "standalone_ann_02",
        "image_id": image_paths[1].stem,
        "class_name": "pedestrian",
        "confidence": 0.85,
        "bbox": {"x": 100.0, "y": 40.0, "w": 15.0, "h": 35.0},
    },
]

client = T11Client(target="localhost:50051")

try:
    responses = list(client.evaluate_job(
        job_id="standalone_smoke_test",
        run_id="run_001",
        images=images,
        annotations=annotations,
    ))

    for response in responses:
        payload = response.WhichOneof("payload")
        print("RECEIVED:", payload)

        if payload == "job_completed":
            print("FINAL STATUS:", response.job_completed.final_status)

        elif payload == "job_result":
            print("IMAGES:", response.job_result.total_images_evaluated)
            print("ANNOTATIONS:", response.job_result.total_annotations_evaluated)
finally:
    client.close()
