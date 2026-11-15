"""SAM 3 Segmentation Adapter (Local Inference and Mock)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from service.server.config import Sam3ModelConfig

logger = logging.getLogger("t11.sam3")


@dataclass
class SAM3MaskResult:
    mask_ref: str
    geometric_diff: float = 0.0
    boundary_points: List[float] = field(default_factory=list)
    is_mock: bool = True


class BaseSAM3Adapter:
    """Base interface for SAM 3 segmentation evidence generation."""

    def evaluate_box_evidence(
        self,
        image_path: Path,
        human_box: Tuple[float, float, float, float],
        model_box: Tuple[float, float, float, float],
    ) -> SAM3MaskResult:
        raise NotImplementedError


class MockSAM3Adapter(BaseSAM3Adapter):
    """Mock SAM 3 adapter providing synthetic boundary / mask difference evidence."""

    def __init__(self, config: Optional[Sam3ModelConfig] = None) -> None:
        self.config = config or Sam3ModelConfig()
        self.version = f"{self.config.version}-mock"

    def evaluate_box_evidence(
        self,
        image_path: Path,
        human_box: Tuple[float, float, float, float],
        model_box: Tuple[float, float, float, float],
    ) -> SAM3MaskResult:
        # Simple geometric difference between boxes
        dx = abs(human_box[0] - model_box[0])
        dy = abs(human_box[1] - model_box[1])
        diff = round((dx + dy) / 100.0, 4)
        mask_id = f"mask_mock_{int(human_box[0])}_{int(human_box[1])}"
        return SAM3MaskResult(
            mask_ref=mask_id,
            geometric_diff=diff,
            is_mock=True,
        )


def create_sam3_adapter(config: Sam3ModelConfig) -> BaseSAM3Adapter:
    return MockSAM3Adapter(config)
