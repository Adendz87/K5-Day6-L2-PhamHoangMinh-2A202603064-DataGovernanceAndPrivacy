"""Unit tests for AnnotationNormalizer."""
from service.core.annotation_normalizer import AnnotationNormalizer


def test_valid_annotation_normalization():
    norm = AnnotationNormalizer(class_mapping={"car": 0, "pedestrian": 1})
    item = {
        "ann_id": "ann_101",
        "image_id": "img_01",
        "class_name": "car",
        "confidence": 0.95,
        "bbox": {"x": 50.0, "y": 60.0, "w": 40.0, "h": 30.0},
    }
    res, errors = norm.normalize_grpc_annotation(item, "img_01", 320, 180)
    assert len(errors) == 0
    assert res.ann_id == "ann_101"
    assert res.class_id == 0
    assert res.x == 50.0
    assert res.w == 40.0


def test_invalid_dimensions():
    norm = AnnotationNormalizer()
    item = {
        "ann_id": "ann_bad",
        "image_id": "img_01",
        "class_name": "car",
        "bbox": {"x": 50.0, "y": 60.0, "w": -5.0, "h": 0.0},
    }
    _, errors = norm.normalize_grpc_annotation(item, "img_01", 320, 180)
    assert any(e.error_code == "NON_POSITIVE_DIMENSIONS" for e in errors)


def test_batch_duplicate_ann_ids():
    norm = AnnotationNormalizer()
    batch = {
        "batch_id": "b1",
        "items": [
            {"ann_id": "dup_1", "image_id": "i1", "class_name": "car", "bbox": {"x": 1, "y": 1, "w": 10, "h": 10}},
            {"ann_id": "dup_1", "image_id": "i1", "class_name": "car", "bbox": {"x": 2, "y": 2, "w": 10, "h": 10}},
        ],
    }
    _, errors = norm.normalize_grpc_batch(batch, 320, 180)
    assert any(e.error_code == "DUPLICATE_ANN_ID" for e in errors)
