"""YOLO Object Detection Adapter (Local Inference and Mock)."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from service.server.config import YoloModelConfig

logger = logging.getLogger("t11.yolo")


@dataclass
class YOLOPrediction:
    class_name: str
    class_id: int
    conf: float
    x: float  # Top-left x in pixels
    y: float  # Top-left y in pixels
    w: float  # Width in pixels
    h: float  # Height in pixels
    is_mock: bool = False


class BaseYOLOAdapter:
    """Base interface for YOLO detection."""

    def resolve_target_classes(self, target_classes: Optional[List[str]]) -> Optional[List[int]]:
        """Resolve human-readable class names to model class IDs.

        Args:
            target_classes: Optional list of class name strings (e.g. ["person"]).
                            None or empty list means no filtering (all classes).

        Returns:
            List of integer class IDs, or None if no filtering should be applied.

        Raises:
            ValueError: If any class name is not found in model names.
        """
        raise NotImplementedError

    def predict(
        self,
        image_path: Path,
        image_width: int,
        image_height: int,
        target_classes: Optional[List[str]] = None,
    ) -> List[YOLOPrediction]:
        raise NotImplementedError


class MockYOLOAdapter(BaseYOLOAdapter):
    """Mock YOLO detector generating synthetic detections for architecture testing."""

    # Mock class names (not COCO — deliberately simple for testing)
    _MOCK_NAMES: Dict[int, str] = {0: "car", 1: "pedestrian"}

    def __init__(self, config: Optional[YoloModelConfig] = None) -> None:
        self.config = config or YoloModelConfig()
        self.version = f"{self.config.version}-mock"
        # Expose names dict for compatibility
        self.names = self._MOCK_NAMES

    def resolve_target_classes(self, target_classes: Optional[List[str]]) -> Optional[List[int]]:
        if not target_classes:
            return None
        name_to_id = {v.lower(): k for k, v in self.names.items()}
        resolved = []
        for cls in target_classes:
            cls_lower = cls.lower()
            if cls_lower not in name_to_id:
                valid = sorted(self.names.values())
                raise ValueError(
                    f"Unknown target class '{cls}'. "
                    f"Valid classes for this model: {valid}"
                )
            resolved.append(name_to_id[cls_lower])
        return resolved

    def predict(
        self,
        image_path: Path,
        image_width: int,
        image_height: int,
        target_classes: Optional[List[str]] = None,
    ) -> List[YOLOPrediction]:
        """Generate mock predictions based on image path / dimensions."""
        resolved_ids = self.resolve_target_classes(target_classes)

        all_preds: List[YOLOPrediction] = []
        w = float(image_width if image_width > 0 else 320)
        h = float(image_height if image_height > 0 else 180)

        all_preds.append(
            YOLOPrediction(
                class_name="car",
                class_id=0,
                conf=0.88,
                x=w * 0.2,
                y=h * 0.3,
                w=w * 0.25,
                h=h * 0.2,
                is_mock=True,
            )
        )
        all_preds.append(
            YOLOPrediction(
                class_name="pedestrian",
                class_id=1,
                conf=0.75,
                x=w * 0.6,
                y=h * 0.4,
                w=w * 0.08,
                h=h * 0.25,
                is_mock=True,
            )
        )

        if resolved_ids is not None:
            return [p for p in all_preds if p.class_id in resolved_ids]
        return all_preds


class RealYOLOAdapter(BaseYOLOAdapter):
    """Real YOLO inference using local weights (Ultralytics)."""

    def __init__(self, config: YoloModelConfig) -> None:
        self.config = config
        self.version = config.version
        weights_path = Path(config.weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Local YOLO weights not found: {config.weights_path}")

        try:
            from ultralytics import YOLO
            self.model = YOLO(config.weights_path)
            # Cache class names for fast lookup
            self.names: Dict[int, str] = dict(self.model.names)
            self._name_to_id: Dict[str, int] = {
                v.lower(): k for k, v in self.names.items()
            }
            # Compute weights SHA-256 once at load time
            with open(weights_path, "rb") as f:
                self.weights_sha256 = hashlib.sha256(f.read()).hexdigest()
            logger.info(
                "Loaded real YOLO model from %s (SHA-256: %s, classes: %d)",
                config.weights_path,
                self.weights_sha256,
                len(self.names),
            )
        except ImportError as e:
            raise ImportError(
                "ultralytics is not installed. Run with mock: true or install ultralytics."
            ) from e

    def resolve_target_classes(self, target_classes: Optional[List[str]]) -> Optional[List[int]]:
        """Resolve class names to COCO class IDs using the loaded model's names dict.

        Never silently maps one class to another. Raises ValueError for unknown classes.
        """
        if not target_classes:
            return None
        resolved = []
        for cls in target_classes:
            cls_lower = cls.lower()
            if cls_lower not in self._name_to_id:
                valid = sorted(self.names.values())
                raise ValueError(
                    f"Unknown target class '{cls}'. "
                    f"Valid classes for this model ({len(self.names)} total): {valid[:20]}..."
                )
            class_id = self._name_to_id[cls_lower]
            logger.debug("Resolved target class '%s' -> class_id %d", cls, class_id)
            resolved.append(class_id)
        return resolved

    def predict(
        self,
        image_path: Path,
        image_width: int,
        image_height: int,
        target_classes: Optional[List[str]] = None,
    ) -> List[YOLOPrediction]:
        """Run YOLO inference, optionally filtering to specific class IDs.

        Class filtering is applied by Ultralytics at inference time (not post-filter),
        ensuring only the requested classes are detected.
        """
        resolved_ids = self.resolve_target_classes(target_classes)

        predict_kwargs: Dict[str, Any] = {
            "source": str(image_path),
            "conf": self.config.conf_threshold,
            "iou": self.config.iou_threshold,
            "verbose": False,
        }
        if resolved_ids is not None:
            predict_kwargs["classes"] = resolved_ids
            logger.debug(
                "YOLO inference on %s with classes=%s (resolved from %s)",
                image_path.name,
                resolved_ids,
                target_classes,
            )
        else:
            logger.debug("YOLO inference on %s (all classes)", image_path.name)

        results = self.model.predict(**predict_kwargs)
        preds: List[YOLOPrediction] = []
        for r in results:
            boxes = r.boxes
            for b in boxes:
                xywh = b.xywh[0].tolist()  # center x, center y, w, h
                conf = float(b.conf[0])
                cls_id = int(b.cls[0])
                cls_name = r.names.get(cls_id, str(cls_id))
                # Convert center (cx, cy) to top-left (x, y)
                x = xywh[0] - xywh[2] / 2.0
                y = xywh[1] - xywh[3] / 2.0
                preds.append(
                    YOLOPrediction(
                        class_name=cls_name,
                        class_id=cls_id,
                        conf=conf,
                        x=x,
                        y=y,
                        w=xywh[2],
                        h=xywh[3],
                        is_mock=False,
                    )
                )
        return preds


def create_yolo_adapter(config: YoloModelConfig) -> BaseYOLOAdapter:
    """Factory to instantiate mock or real YOLO adapter."""
    if config.mock or not Path(config.weights_path).exists():
        return MockYOLOAdapter(config)
    return RealYOLOAdapter(config)
