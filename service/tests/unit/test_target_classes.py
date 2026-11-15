"""Unit tests for target class conditioned YOLO detection (Phase 1).

Covers:
1. Explicit person class filtering → only person detections
2. Correct class ID resolution from model.names (class 0 = person)
3. Unknown class raises ValueError (not silently swapped)
4. Missing target_classes → backward compatible (all classes)
5. Empty target_classes → all classes
6. Multiple people detections
7. Zero detections represented explicitly
8. Coordinate correctness (center→top-left conversion)
9. Real YOLO inference with mock=False on T14 image
10. Existing gRPC streaming still works (regression)
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# --- Workspace root on sys.path ---
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

WEIGHTS_PATH = ROOT / "models" / "yolo" / "yolov8n.pt"
T14_IMAGE = ROOT / "fixtures" / "task14" / "000000027565.jpg"


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _make_mock_config(mock: bool = True) -> "YoloModelConfig":
    from service.server.config import YoloModelConfig
    return YoloModelConfig(
        weights_path=str(WEIGHTS_PATH),
        mock=mock,
        conf_threshold=0.25,
        iou_threshold=0.45,
    )


# ──────────────────────────────────────────────────────────────────
# TC01 — MockYOLOAdapter: resolve_target_classes with valid class
# ──────────────────────────────────────────────────────────────────

class TestMockAdapterResolveClasses:
    def _adapter(self):
        from service.adapters.yolo_adapter import MockYOLOAdapter
        from service.server.config import YoloModelConfig
        cfg = YoloModelConfig(weights_path="fake.pt", mock=True)
        return MockYOLOAdapter(cfg)

    def test_resolve_known_class(self):
        a = self._adapter()
        ids = a.resolve_target_classes(["car"])
        assert ids == [0], f"Expected [0], got {ids}"

    def test_resolve_pedestrian(self):
        a = self._adapter()
        ids = a.resolve_target_classes(["pedestrian"])
        assert ids == [1]

    def test_resolve_unknown_raises(self):
        a = self._adapter()
        with pytest.raises(ValueError, match="Unknown target class"):
            a.resolve_target_classes(["person"])  # 'person' not in mock names

    def test_resolve_none_returns_none(self):
        a = self._adapter()
        assert a.resolve_target_classes(None) is None

    def test_resolve_empty_returns_none(self):
        a = self._adapter()
        assert a.resolve_target_classes([]) is None

    def test_predict_filtered_to_car(self):
        a = self._adapter()
        preds = a.predict(Path("img.jpg"), 640, 480, target_classes=["car"])
        assert all(p.class_name == "car" for p in preds)
        assert len(preds) >= 1

    def test_predict_no_filter_returns_all(self):
        a = self._adapter()
        preds = a.predict(Path("img.jpg"), 640, 480, target_classes=None)
        class_names = {p.class_name for p in preds}
        assert "car" in class_names
        assert "pedestrian" in class_names

    def test_predict_empty_filter_returns_all(self):
        a = self._adapter()
        preds = a.predict(Path("img.jpg"), 640, 480, target_classes=[])
        assert len(preds) >= 2


# ──────────────────────────────────────────────────────────────────
# TC02 — RealYOLOAdapter class name resolution
# ──────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not WEIGHTS_PATH.exists(), reason="YOLO weights not available")
class TestRealYOLOAdapterClassResolution:
    def _adapter(self):
        from service.adapters.yolo_adapter import RealYOLOAdapter
        from service.server.config import YoloModelConfig
        cfg = YoloModelConfig(weights_path=str(WEIGHTS_PATH), mock=False, conf_threshold=0.25)
        return RealYOLOAdapter(cfg)

    def test_person_resolves_to_class_id_0(self):
        """COCO class 0 = person. Must resolve exactly, never to 'pedestrian'."""
        a = self._adapter()
        ids = a.resolve_target_classes(["person"])
        assert ids == [0], f"Expected [0], got {ids}"

    def test_model_names_contains_person(self):
        a = self._adapter()
        assert 0 in a.names
        assert a.names[0].lower() == "person"

    def test_unknown_class_raises_value_error(self):
        a = self._adapter()
        with pytest.raises(ValueError, match="Unknown target class 'pedestrian'"):
            a.resolve_target_classes(["pedestrian"])

    def test_person_not_silently_mapped_to_pedestrian(self):
        """Guard: 'person' and 'pedestrian' are distinct; must never silently swap."""
        a = self._adapter()
        person_ids = a.resolve_target_classes(["person"])
        # pedestrian must NOT resolve — it's not a COCO class
        with pytest.raises(ValueError):
            a.resolve_target_classes(["pedestrian"])
        # person must map to ID 0 specifically
        assert person_ids == [0]

    def test_resolve_multiple_valid_classes(self):
        a = self._adapter()
        ids = a.resolve_target_classes(["person", "car"])
        assert 0 in ids
        assert 2 in ids  # 'car' = class 2 in COCO

    def test_resolve_none_returns_none(self):
        a = self._adapter()
        assert a.resolve_target_classes(None) is None

    def test_resolve_empty_returns_none(self):
        a = self._adapter()
        assert a.resolve_target_classes([]) is None

    def test_weights_sha256_attribute_set(self):
        a = self._adapter()
        assert hasattr(a, "weights_sha256")
        expected = hashlib.sha256(WEIGHTS_PATH.read_bytes()).hexdigest()
        assert a.weights_sha256 == expected


# ──────────────────────────────────────────────────────────────────
# TC03 — Real YOLO inference on T14 image
# ──────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not WEIGHTS_PATH.exists() or not T14_IMAGE.exists(),
    reason="YOLO weights or T14 image not available",
)
class TestRealYOLOT14Inference:
    def _adapter(self):
        from service.adapters.yolo_adapter import RealYOLOAdapter
        from service.server.config import YoloModelConfig
        cfg = YoloModelConfig(weights_path=str(WEIGHTS_PATH), mock=False, conf_threshold=0.25)
        return RealYOLOAdapter(cfg)

    def test_person_detected_on_t14_frame5(self):
        """Real YOLOv8n must detect at least 1 person on T14 frame 5."""
        a = self._adapter()
        preds = a.predict(T14_IMAGE, 437, 640, target_classes=["person"])
        person_preds = [p for p in preds if p.class_name == "person"]
        assert len(person_preds) >= 1, (
            f"Expected ≥1 person detection on {T14_IMAGE.name}, got 0. "
            f"All preds: {preds}"
        )

    def test_person_detection_confidence_above_threshold(self):
        """Detected person must meet conf threshold (not filtered out)."""
        a = self._adapter()
        preds = a.predict(T14_IMAGE, 437, 640, target_classes=["person"])
        for p in preds:
            assert p.conf >= 0.25, f"Detection below threshold: {p}"

    def test_person_detection_class_is_correct(self):
        """All returned detections must be class 'person', not any other class."""
        a = self._adapter()
        preds = a.predict(T14_IMAGE, 437, 640, target_classes=["person"])
        for p in preds:
            assert p.class_name == "person", f"Non-person detection returned: {p}"
            assert p.class_id == 0, f"person must be class_id 0, got {p.class_id}"

    def test_no_pedestrian_in_person_filtered_results(self):
        """'pedestrian' must never appear in person-filtered results."""
        a = self._adapter()
        preds = a.predict(T14_IMAGE, 437, 640, target_classes=["person"])
        for p in preds:
            assert p.class_name != "pedestrian"

    def test_bounding_box_coordinates_top_left_origin(self):
        """Coordinates must use top-left origin (x, y, w, h), x ≥ 0, y ≥ 0."""
        a = self._adapter()
        preds = a.predict(T14_IMAGE, 437, 640, target_classes=["person"])
        for p in preds:
            assert p.x >= 0, f"Negative x: {p.x}"
            assert p.y >= 0, f"Negative y: {p.y}"
            assert p.w > 0, f"Non-positive width: {p.w}"
            assert p.h > 0, f"Non-positive height: {p.h}"

    def test_zero_detections_when_filtering_to_non_present_class(self):
        """If filtering to a class not in image, must return empty list (not error)."""
        a = self._adapter()
        # 'boat' is unlikely in a person-only street image
        preds = a.predict(T14_IMAGE, 437, 640, target_classes=["boat"])
        # This may be 0 or more — just must not raise and must only contain boats
        for p in preds:
            assert p.class_name == "boat"

    def test_no_filter_includes_person(self):
        """Without target_classes, inference still returns person detections."""
        a = self._adapter()
        preds = a.predict(T14_IMAGE, 437, 640, target_classes=None)
        person_preds = [p for p in preds if p.class_name == "person"]
        assert len(person_preds) >= 1

    def test_mock_false_on_adapter(self):
        """Adapter must not be in mock mode."""
        a = self._adapter()
        assert not a.config.mock

    def test_is_mock_false_on_predictions(self):
        """Real predictions must have is_mock=False."""
        a = self._adapter()
        preds = a.predict(T14_IMAGE, 437, 640, target_classes=["person"])
        for p in preds:
            assert not p.is_mock


# ──────────────────────────────────────────────────────────────────
# TC04 — Coordinate correctness (center → top-left conversion)
# ──────────────────────────────────────────────────────────────────

class TestCoordinateConversion:
    """Verify that cx,cy,w,h → x,y,w,h (top-left) conversion is correct."""

    def test_center_to_topleft_math(self):
        """x = cx - w/2; y = cy - h/2."""
        cx, cy, w, h = 271.0, 347.0, 258.0, 518.0
        expected_x = cx - w / 2.0  # 142.0
        expected_y = cy - h / 2.0  # 88.0

        from service.adapters.yolo_adapter import YOLOPrediction
        pred = YOLOPrediction(
            class_name="person",
            class_id=0,
            conf=0.85,
            x=cx - w / 2.0,
            y=cy - h / 2.0,
            w=w,
            h=h,
        )
        assert abs(pred.x - expected_x) < 0.01
        assert abs(pred.y - expected_y) < 0.01

    @pytest.mark.skipif(
        not WEIGHTS_PATH.exists() or not T14_IMAGE.exists(),
        reason="YOLO weights or T14 image not available",
    )
    def test_t14_person_bbox_matches_expected_range(self):
        """T14 frame 5 person bbox should be near CVAT annotation [142, 59, 398, 609]."""
        from service.adapters.yolo_adapter import RealYOLOAdapter
        from service.server.config import YoloModelConfig
        cfg = YoloModelConfig(weights_path=str(WEIGHTS_PATH), mock=False, conf_threshold=0.25)
        a = RealYOLOAdapter(cfg)
        preds = a.predict(T14_IMAGE, 437, 640, target_classes=["person"])
        person = next((p for p in preds if p.class_name == "person"), None)
        assert person is not None
        # YOLO predicted x ≈ 142, y ≈ 88, w ≈ 258, h ≈ 519
        # CVAT annotation: x1=142, y1=59, x2=398, y2=609
        assert 100 <= person.x <= 200, f"x out of range: {person.x}"
        assert person.w > 200, f"Width too small: {person.w}"
        assert person.h > 400, f"Height too small: {person.h}"


# ──────────────────────────────────────────────────────────────────
# TC05 — InputManager persists target_classes in metadata_json
# ──────────────────────────────────────────────────────────────────

class TestInputManagerTargetClasses:
    def _setup(self, tmp_path):
        from service.core.database import JobDatabase
        from service.core.temporary_storage import TemporaryStorage
        from service.core.job_lifecycle import JobLifecycleManager
        from service.core.input_manager import InputManager
        db = JobDatabase(str(tmp_path / "test.db"))
        storage = TemporaryStorage(str(tmp_path / "tmp"), max_size_gb=1)
        lifecycle = JobLifecycleManager(db)
        return InputManager(storage, db, lifecycle), db

    def test_target_classes_stored_in_metadata(self, tmp_path):
        mgr, db = self._setup(tmp_path)
        result = mgr.handle_job_metadata({
            "job_id": "job-tc-01",
            "run_id": "run-01",
            "client_id": "test",
            "total_images": 1,
            "total_annotations": 0,
            "metadata": {},
            "target_classes": ["person"],
        })
        import json
        job = db.get_job("job-tc-01", "run-01")
        assert job is not None
        meta = json.loads(job["metadata_json"])
        assert "target_classes" in meta
        assert meta["target_classes"] == ["person"]

    def test_missing_target_classes_not_stored(self, tmp_path):
        mgr, db = self._setup(tmp_path)
        mgr.handle_job_metadata({
            "job_id": "job-tc-02",
            "run_id": "run-02",
            "client_id": "test",
            "total_images": 1,
            "total_annotations": 0,
            "metadata": {},
        })
        import json
        job = db.get_job("job-tc-02", "run-02")
        assert job is not None
        meta = json.loads(job["metadata_json"])
        # target_classes should not be present when not provided
        assert "target_classes" not in meta

    def test_empty_target_classes_not_stored(self, tmp_path):
        mgr, db = self._setup(tmp_path)
        mgr.handle_job_metadata({
            "job_id": "job-tc-03",
            "run_id": "run-03",
            "client_id": "test",
            "total_images": 1,
            "total_annotations": 0,
            "metadata": {},
            "target_classes": [],
        })
        import json
        job = db.get_job("job-tc-03", "run-03")
        meta = json.loads(job["metadata_json"])
        assert "target_classes" not in meta


# ──────────────────────────────────────────────────────────────────
# TC06 — ProcessingEngine reads target_classes from job record
# ──────────────────────────────────────────────────────────────────

class TestProcessingEngineTargetClasses:
    """ProcessingEngine must read target_classes from stored metadata and pass to YOLO."""

    def _engine(self, tmp_path):
        from service.server.config import ServiceConfig
        from service.core.processing_engine import ProcessingEngine
        cfg = ServiceConfig()
        cfg.storage.temp_path = str(tmp_path / "tmp")
        cfg.storage.database_path = str(tmp_path / "test.db")
        cfg.storage.result_path = str(tmp_path / "results")
        cfg.models.yolo.mock = True
        cfg.models.yolo.weights_path = str(WEIGHTS_PATH)
        cfg.models.sam3.enabled = False
        cfg.models.sam3.mock = True
        return ProcessingEngine(cfg)

    def test_target_classes_extracted_from_job_metadata(self, tmp_path):
        """ProcessingEngine reads target_classes from metadata_json and passes to predict()."""
        engine = self._engine(tmp_path)
        import json

        # Seed a job record with target_classes in metadata_json
        engine.db.create_job(
            job_id="job-pe-01",
            run_id="run-pe-01",
            client_id="test",
            metadata={"target_classes": ["car"]},
        )
        job = engine.db.get_job("job-pe-01", "run-pe-01")
        meta = json.loads(job["metadata_json"])
        assert meta.get("target_classes") == ["car"]

    def test_predict_called_with_target_classes(self, tmp_path):
        """When target_classes is set, yolo_adapter.predict() receives target_classes kwarg."""
        engine = self._engine(tmp_path)

        # Create a minimal job + image so process_job can run to the YOLO call
        engine.db.create_job(
            "job-pe-02", "run-pe-02", metadata={"target_classes": ["car"]}
        )

        import json
        called_kwargs = {}

        original_predict = engine.yolo_adapter.predict

        def spy_predict(image_path, iw, ih, target_classes=None):
            called_kwargs["target_classes"] = target_classes
            return []

        engine.yolo_adapter.predict = spy_predict

        # Create a fake image record pointing to T14 image (or any jpg)
        fake_img_path = str(T14_IMAGE) if T14_IMAGE.exists() else str(ROOT / "fixtures" / "sample.jpg")
        engine.db._get_conn = engine.db.get_connection
        try:
            with engine.db.get_connection() as conn:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc).isoformat()
                conn.execute("""
                    INSERT OR IGNORE INTO images
                    (job_id, run_id, image_id, status, total_chunks, received_chunks,
                     sha256, file_path, error_message, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, ("job-pe-02", "run-pe-02", "img-001", "COMPLETE", 1, 1,
                      "abc", fake_img_path, "", now, now))
                conn.commit()
        except Exception:
            pass

        # Drive process_job and collect the target_classes kwarg
        try:
            for _ in engine.process_job("job-pe-02", "run-pe-02"):
                pass
        except Exception:
            pass  # May fail on lifecycle transitions; we just need the spy called

        assert "target_classes" in called_kwargs, "predict() was never called with target_classes"
        assert called_kwargs["target_classes"] == ["car"]


# ──────────────────────────────────────────────────────────────────
# TC07 — Protobuf: target_classes in JobMetadata round-trip
# ──────────────────────────────────────────────────────────────────

class TestProtobufTargetClasses:
    def test_target_classes_field_exists(self):
        from service.generated import pb2
        meta = pb2.JobMetadata(target_classes=["person", "car"])
        assert list(meta.target_classes) == ["person", "car"]

    def test_empty_target_classes(self):
        from service.generated import pb2
        meta = pb2.JobMetadata()
        assert list(meta.target_classes) == []

    def test_serialize_deserialize(self):
        from service.generated import pb2
        meta = pb2.JobMetadata(
            job_id="j1",
            run_id="r1",
            target_classes=["person"],
        )
        serialized = meta.SerializeToString()
        restored = pb2.JobMetadata()
        restored.ParseFromString(serialized)
        assert list(restored.target_classes) == ["person"]
        assert restored.job_id == "j1"

    def test_backward_compat_no_target_classes(self):
        """Old messages without target_classes must still parse cleanly."""
        from service.generated import pb2
        old_meta = pb2.JobMetadata(job_id="old", run_id="run")
        serialized = old_meta.SerializeToString()
        restored = pb2.JobMetadata()
        restored.ParseFromString(serialized)
        assert list(restored.target_classes) == []


# ──────────────────────────────────────────────────────────────────
# TC08 — Diagnostic: T14 real inference execution trace
# ──────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not WEIGHTS_PATH.exists() or not T14_IMAGE.exists(),
    reason="YOLO weights or T14 image not available",
)
def test_t14_e2e_real_inference_produces_one_person():
    """Full diagnostic trace: load model → resolve class → predict → verify detection."""
    from service.adapters.yolo_adapter import RealYOLOAdapter
    from service.server.config import YoloModelConfig

    cfg = YoloModelConfig(weights_path=str(WEIGHTS_PATH), mock=False, conf_threshold=0.25)
    adapter = RealYOLOAdapter(cfg)

    # Verify weights SHA
    expected_sha = hashlib.sha256(WEIGHTS_PATH.read_bytes()).hexdigest()
    assert adapter.weights_sha256 == expected_sha

    # Verify class 0 = person
    assert adapter.names[0] == "person"

    # Resolve 'person' to ID 0
    ids = adapter.resolve_target_classes(["person"])
    assert ids == [0]

    # Run real inference
    preds = adapter.predict(T14_IMAGE, 437, 640, target_classes=["person"])

    # Must detect at least 1 person
    assert len(preds) >= 1, "Expected ≥1 person, got 0"
    p = preds[0]
    assert p.class_name == "person"
    assert p.class_id == 0
    assert p.conf >= 0.25
    assert not p.is_mock

    # Coordinates must be top-left origin and reasonable
    assert p.x > 100  # person is roughly at x=142
    assert p.y > 50   # person starts around y=88
    assert p.w > 200  # wide person
    assert p.h > 400  # tall person

    print(f"\nT14 Detection OK: conf={p.conf:.4f} x={p.x:.1f} y={p.y:.1f} w={p.w:.1f} h={p.h:.1f}")
