"""T14 Diagnostic Script — Phase 1 Class-Conditioned YOLO Detection.

Standalone diagnostic: runs YOLOv8n with target_classes=["person"]
on Task 14 / Frame 5 / Image 000000027565.

Outputs:
  out/diagnostics/t14_raw_detections.json
  out/diagnostics/t14_filtered_detections.json
  out/diagnostics/t14_diagnostic_report.json
  out/diagnostics/t14_frame5_person_detected.jpg
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Add workspace root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

TARGET_IMAGE = ROOT / "fixtures" / "task14" / "000000027565.jpg"
WEIGHTS_PATH = ROOT / "models" / "yolo" / "yolov8n.pt"
OUT_DIR = ROOT / "out" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = 14
JOB_ID = 14
FRAME_INDEX = 5
IMAGE_ID = "000000027565"
TARGET_CLASSES = ["person"]
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


def compute_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main() -> None:
    print("=" * 60)
    print("T14 Diagnostic — Class-Conditioned YOLO Person Detection")
    print("=" * 60)

    if not TARGET_IMAGE.exists():
        print(f"ERROR: Image not found at {TARGET_IMAGE}")
        print("Run: docker cp cvat_server:/home/django/data/data/14/raw/000000027565.jpg fixtures/task14/")
        sys.exit(1)

    if not WEIGHTS_PATH.exists():
        print(f"ERROR: YOLO weights not found at {WEIGHTS_PATH}")
        sys.exit(1)

    print(f"\n[1] Loading YOLO model: {WEIGHTS_PATH}")
    from ultralytics import YOLO
    from PIL import Image as PILImage
    import cv2

    t_load = time.perf_counter()
    model = YOLO(str(WEIGHTS_PATH))
    load_ms = int((time.perf_counter() - t_load) * 1000)

    model_names = dict(model.names)
    name_to_id = {v.lower(): k for k, v in model_names.items()}

    weights_sha = compute_sha256(WEIGHTS_PATH)
    print(f"   SHA-256:   {weights_sha}")
    print(f"   Classes:   {len(model_names)}")
    print(f"   Load time: {load_ms} ms")

    # Resolve target class IDs
    resolved_ids = []
    for cls in TARGET_CLASSES:
        if cls.lower() not in name_to_id:
            raise ValueError(
                f"Unknown target class '{cls}'. "
                f"Valid classes: {sorted(model_names.values())[:20]}..."
            )
        cid = name_to_id[cls.lower()]
        resolved_ids.append(cid)
        print(f"   Resolved '{cls}' -> class_id {cid} (from model.names[{cid}]='{model_names[cid]}')")

    # Image dimensions
    pil_img = PILImage.open(TARGET_IMAGE)
    img_width, img_height = pil_img.size
    print(f"\n[2] Image: {TARGET_IMAGE.name}")
    print(f"   Dimensions: {img_width} x {img_height} px")

    # --- RAW inference (all classes, no filter) ---
    print(f"\n[3] Running RAW inference (all classes, conf={CONF_THRESHOLD})")
    t_raw = time.perf_counter()
    raw_results = model.predict(
        source=str(TARGET_IMAGE),
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
    )
    raw_ms = int((time.perf_counter() - t_raw) * 1000)
    raw_detections = []
    for r in raw_results:
        for b in r.boxes:
            xywh = b.xywh[0].tolist()
            xyxy = b.xyxy[0].tolist()
            cid = int(b.cls[0])
            raw_detections.append({
                "class_id": cid,
                "class_name": r.names.get(cid, str(cid)),
                "confidence": round(float(b.conf[0]), 6),
                "bbox_xyxy": [round(v, 2) for v in xyxy],
                "bbox_xywh_topleft": {
                    "x": round(xywh[0] - xywh[2] / 2.0, 2),
                    "y": round(xywh[1] - xywh[3] / 2.0, 2),
                    "w": round(xywh[2], 2),
                    "h": round(xywh[3], 2),
                },
            })
    print(f"   RAW detections: {len(raw_detections)} ({raw_ms} ms)")
    for d in raw_detections:
        print(f"     [{d['class_id']}] {d['class_name']:20s} conf={d['confidence']:.4f}  {d['bbox_xyxy']}")

    # --- FILTERED inference (target_classes=["person"]) ---
    print(f"\n[4] Running FILTERED inference (classes={resolved_ids}, conf={CONF_THRESHOLD})")
    t_filt = time.perf_counter()
    filt_results = model.predict(
        source=str(TARGET_IMAGE),
        classes=resolved_ids,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
    )
    filt_ms = int((time.perf_counter() - t_filt) * 1000)
    filtered_detections = []
    for r in filt_results:
        for b in r.boxes:
            xywh = b.xywh[0].tolist()
            xyxy = b.xyxy[0].tolist()
            cid = int(b.cls[0])
            filtered_detections.append({
                "class_id": cid,
                "class_name": r.names.get(cid, str(cid)),
                "confidence": round(float(b.conf[0]), 6),
                "bbox_xyxy": [round(v, 2) for v in xyxy],
                "bbox_xywh_topleft": {
                    "x": round(xywh[0] - xywh[2] / 2.0, 2),
                    "y": round(xywh[1] - xywh[3] / 2.0, 2),
                    "w": round(xywh[2], 2),
                    "h": round(xywh[3], 2),
                },
            })
    print(f"   FILTERED detections: {len(filtered_detections)} ({filt_ms} ms)")
    for d in filtered_detections:
        print(f"     [{d['class_id']}] {d['class_name']:20s} conf={d['confidence']:.4f}  {d['bbox_xyxy']}")

    # --- CVAT annotation for frame 5 (from shell query above) ---
    cvat_annotation = {
        "shape_id": 6778,
        "label": "person",
        "type": "rectangle",
        "frame": 5,
        "points_xyxy": [142.56, 59.30, 398.28, 609.24],
    }

    # --- Save JSON outputs ---
    raw_json_path = OUT_DIR / "t14_raw_detections.json"
    raw_json_path.write_text(json.dumps(raw_detections, indent=2))
    print(f"\n[5] Saved raw detections: {raw_json_path}")

    filt_json_path = OUT_DIR / "t14_filtered_detections.json"
    filt_json_path.write_text(json.dumps(filtered_detections, indent=2))
    print(f"    Saved filtered detections: {filt_json_path}")

    # --- Diagnostic image ---
    img_cv = cv2.imread(str(TARGET_IMAGE))
    # Draw CVAT annotation in blue
    x1, y1, x2, y2 = [int(v) for v in cvat_annotation["points_xyxy"]]
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img_cv, "CVAT:person", (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 1)

    # Draw YOLO filtered detections in green
    for d in filtered_detections:
        bx1, by1, bx2, by2 = [int(v) for v in d["bbox_xyxy"]]
        label = f"YOLO:{d['class_name']} {d['confidence']:.2f}"
        cv2.rectangle(img_cv, (bx1, by1), (bx2, by2), (0, 200, 0), 2)
        cv2.putText(img_cv, label, (bx1, max(by1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1)

    diag_img_path = OUT_DIR / "t14_frame5_person_detected.jpg"
    cv2.imwrite(str(diag_img_path), img_cv)
    print(f"    Saved diagnostic image: {diag_img_path}")

    # --- Full report ---
    report = {
        "task_id": TASK_ID,
        "job_id": JOB_ID,
        "frame_index": FRAME_INDEX,
        "image_id": IMAGE_ID,
        "image_path": str(TARGET_IMAGE),
        "image_dimensions": {"width": img_width, "height": img_height},
        "target_classes": TARGET_CLASSES,
        "resolved_class_ids": resolved_ids,
        "inference_config": {
            "weights_path": str(WEIGHTS_PATH),
            "weights_sha256": weights_sha,
            "conf_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "model_class_count": len(model_names),
            "class_0_name": model_names.get(0),
        },
        "raw_inference": {
            "detection_count": len(raw_detections),
            "inference_ms": raw_ms,
            "detections": raw_detections,
        },
        "filtered_inference": {
            "filter_applied": True,
            "filtered_classes": TARGET_CLASSES,
            "filtered_class_ids": resolved_ids,
            "detection_count": len(filtered_detections),
            "inference_ms": filt_ms,
            "detections": filtered_detections,
        },
        "cvat_annotation_frame5": cvat_annotation,
        "outputs": {
            "raw_detections_json": str(raw_json_path),
            "filtered_detections_json": str(filt_json_path),
            "diagnostic_image": str(diag_img_path),
        },
        "verdict": (
            f"YOLO detects {len(filtered_detections)} person(s) on frame 5"
            if filtered_detections else
            "YOLO detected ZERO persons on frame 5"
        ),
    }
    report_path = OUT_DIR / "t14_diagnostic_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"    Saved full report: {report_path}")

    print("\n" + "=" * 60)
    print(f"VERDICT: {report['verdict']}")
    print("=" * 60)
    return report


if __name__ == "__main__":
    main()
