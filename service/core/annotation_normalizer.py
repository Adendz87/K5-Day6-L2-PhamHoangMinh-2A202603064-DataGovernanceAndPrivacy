"""Annotation normalization and validation layer.

Converts various external formats (gRPC batches, COCO, YOLO) to the standard internal format:
(x, y, w, h) with (x, y) as top-left corner in absolute pixel coordinates,
consistent with src/t11/geometry.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class NormalizedAnnotation:
    ann_id: str
    image_id: str
    class_name: str
    class_id: int
    x: float  # Top-left x in pixels
    y: float  # Top-left y in pixels
    w: float  # Width in pixels
    h: float  # Height in pixels
    image_width: int
    image_height: int
    confidence: float = 0.0
    segmentation: List[float] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)
    source_format: str = "grpc"


@dataclass
class ValidationError:
    ann_id: str
    image_id: str
    error_code: str
    message: str
    severity: str = "ERROR"  # "ERROR" or "WARNING"


class AnnotationNormalizer:
    """Normalizes and validates annotations from external inputs."""

    def __init__(self, class_mapping: Optional[Dict[str, int]] = None) -> None:
        self.class_mapping = class_mapping or {}

    def normalize_grpc_annotation(
        self,
        ann_dict: Dict[str, Any],
        image_id: str,
        image_width: int,
        image_height: int,
    ) -> Tuple[NormalizedAnnotation, List[ValidationError]]:
        """Normalize a single gRPC annotation dictionary and validate it."""
        errors: List[ValidationError] = []

        ann_id = str(ann_dict.get("ann_id", "")).strip()
        if not ann_id:
            errors.append(
                ValidationError(
                    ann_id="",
                    image_id=image_id,
                    error_code="EMPTY_ANN_ID",
                    message="Annotation ann_id is required",
                )
            )

        class_name = str(ann_dict.get("class_name", "")).strip()
        if not class_name:
            errors.append(
                ValidationError(
                    ann_id=ann_id,
                    image_id=image_id,
                    error_code="EMPTY_CLASS_NAME",
                    message=f"Annotation {ann_id} has empty class_name",
                )
            )

        class_id = int(ann_dict.get("class_id", 0))
        if self.class_mapping and class_name in self.class_mapping:
            class_id = self.class_mapping[class_name]

        bbox = ann_dict.get("bbox", {})
        x = float(bbox.get("x", 0.0))
        y = float(bbox.get("y", 0.0))
        w = float(bbox.get("w", 0.0))
        h = float(bbox.get("h", 0.0))

        if w <= 0.0 or h <= 0.0:
            errors.append(
                ValidationError(
                    ann_id=ann_id,
                    image_id=image_id,
                    error_code="NON_POSITIVE_DIMENSIONS",
                    message=f"Annotation {ann_id} has non-positive width/height: w={w}, h={h}",
                )
            )

        # Coordinate bounds check
        if image_width > 0 and image_height > 0:
            if x < 0.0 or y < 0.0 or (x + w) > image_width * 1.05 or (y + h) > image_height * 1.05:
                errors.append(
                    ValidationError(
                        ann_id=ann_id,
                        image_id=image_id,
                        error_code="OUT_OF_BOUNDS",
                        message=(
                            f"Annotation {ann_id} bbox [{x}, {y}, {w}, {h}] "
                            f"exceeds image dimensions [{image_width}, {image_height}]"
                        ),
                    )
                )
            elif (x + w) > image_width or (y + h) > image_height:
                # Minor boundary clamping with warning
                w = min(w, image_width - max(0.0, x))
                h = min(h, image_height - max(0.0, y))
                x = max(0.0, x)
                y = max(0.0, y)

        confidence = float(ann_dict.get("confidence", 0.0))
        attributes = dict(ann_dict.get("extra_fields", {}))

        norm = NormalizedAnnotation(
            ann_id=ann_id,
            image_id=image_id,
            class_name=class_name,
            class_id=class_id,
            x=x,
            y=y,
            w=w,
            h=h,
            image_width=image_width,
            image_height=image_height,
            confidence=confidence,
            attributes=attributes,
            source_format="grpc",
        )
        return norm, errors

    def normalize_grpc_batch(
        self,
        batch_dict: Dict[str, Any],
        image_width: int = 0,
        image_height: int = 0,
    ) -> Tuple[List[NormalizedAnnotation], List[ValidationError]]:
        """Normalize all annotations in a batch, checking for intra-batch duplicate IDs."""
        normalized: List[NormalizedAnnotation] = []
        all_errors: List[ValidationError] = []
        seen_ids: set[str] = set()

        items = batch_dict.get("items", [])
        for item in items:
            image_id = str(item.get("image_id", "")).strip()
            norm, errs = self.normalize_grpc_annotation(
                item, image_id, image_width, image_height
            )
            all_errors.extend(errs)

            if norm.ann_id in seen_ids:
                all_errors.append(
                    ValidationError(
                        ann_id=norm.ann_id,
                        image_id=image_id,
                        error_code="DUPLICATE_ANN_ID",
                        message=f"Duplicate ann_id '{norm.ann_id}' in batch",
                    )
                )
            seen_ids.add(norm.ann_id)
            normalized.append(norm)

        return normalized, all_errors
