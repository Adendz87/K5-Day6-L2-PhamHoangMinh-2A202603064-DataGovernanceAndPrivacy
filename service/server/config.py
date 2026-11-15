"""Configuration loader and validator for T11 Service."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class GrpcConfig:
    host: str = "0.0.0.0"
    port: int = 50051
    max_message_size_mb: int = 32
    max_workers: int = 4

    @property
    def max_message_size_bytes(self) -> int:
        return self.max_message_size_mb * 1024 * 1024


@dataclass
class TransferConfig:
    chunk_size_bytes: int = 1048576  # 1 MB
    max_image_size_mb: int = 50
    max_job_size_gb: int = 10
    max_concurrent_transfers: int = 8
    ack_window_chunks: int = 8

    @property
    def max_image_size_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024

    @property
    def max_job_size_bytes(self) -> int:
        return self.max_job_size_gb * 1024 * 1024 * 1024


@dataclass
class StorageConfig:
    temp_path: str = "/tmp/t11_service/temp"
    result_path: str = "/tmp/t11_service/results"
    database_path: str = "/tmp/t11_service/t11.sqlite"
    temp_retention_hours: int = 24
    result_retention_days: int = 90


@dataclass
class ProcessingConfig:
    batch_size: int = 16
    queue_max_size: int = 256
    max_concurrent_jobs: int = 2
    device: str = "cpu"
    max_ram_gb: float = 4.0
    max_vram_gb: float = 2.0


@dataclass
class YoloModelConfig:
    weights_path: str = "models/yolo/yolov8n.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    mock: bool = True
    version: str = "yolov8n"


@dataclass
class Sam3ModelConfig:
    weights_path: str = "models/sam3/sam3_base.pt"
    mock: bool = True
    enabled: bool = False
    conditional_execution: bool = True
    version: str = "sam3_base"


@dataclass
class ModelsConfig:
    yolo: YoloModelConfig = field(default_factory=YoloModelConfig)
    sam3: Sam3ModelConfig = field(default_factory=Sam3ModelConfig)


@dataclass
class ScoringConfig:
    version: str = "real-world-v1"
    risk_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"low": 0.2, "medium": 0.5, "high": 0.8}
    )
    match_iou: float = 0.30
    shift_gate: float = 2.0
    dup_iou: float = 0.50
    orphan_overlap: float = 0.30
    w_dup: float = 0.95
    w_swap: float = 0.90
    w_shift: float = 0.85
    w_missing: float = 0.85
    w_unmatched: float = 0.35
    missing_conf_floor: float = 0.40


@dataclass
class SecurityConfig:
    enable_tls: bool = False
    client_auth: bool = False
    request_timeout_seconds: int = 3600
    tls_cert_path: str = ""
    tls_key_path: str = ""


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"


@dataclass
class ServiceConfig:
    grpc: GrpcConfig = field(default_factory=GrpcConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        """Validate logical consistency of configuration values."""
        if not self.grpc.host.startswith("unix:"):
            if self.grpc.port <= 0 or self.grpc.port > 65535:
                raise ValueError(f"Invalid gRPC port: {self.grpc.port}")
        if self.transfer.chunk_size_bytes <= 0:
            raise ValueError("chunk_size_bytes must be positive")
        if self.processing.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.processing.queue_max_size <= 0:
            raise ValueError("queue_max_size must be positive")
        if self.processing.max_concurrent_jobs <= 0:
            raise ValueError("max_concurrent_jobs must be positive")

        # Check models config
        if not self.models.yolo.mock and not os.path.exists(self.models.yolo.weights_path):
            raise FileNotFoundError(
                f"YOLO weights not found at: {self.models.yolo.weights_path}. "
                "Set mock: true for development/testing."
            )
        if (
            self.models.sam3.enabled
            and not self.models.sam3.mock
            and not os.path.exists(self.models.sam3.weights_path)
        ):
            raise FileNotFoundError(
                f"SAM 3 weights not found at: {self.models.sam3.weights_path}. "
                "Set mock: true for development/testing."
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ServiceConfig:
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        grpc_data = data.get("grpc", {})
        transfer_data = data.get("transfer", {})
        storage_data = data.get("storage", {})
        processing_data = data.get("processing", {})
        models_data = data.get("models", {})
        scoring_data = data.get("scoring", {})
        security_data = data.get("security", {})
        logging_data = data.get("logging", {})

        yolo_data = models_data.get("yolo", {})
        sam3_data = models_data.get("sam3", {})

        cfg = cls(
            grpc=GrpcConfig(**grpc_data),
            transfer=TransferConfig(**transfer_data),
            storage=StorageConfig(**storage_data),
            processing=ProcessingConfig(**processing_data),
            models=ModelsConfig(
                yolo=YoloModelConfig(**yolo_data),
                sam3=Sam3ModelConfig(**sam3_data),
            ),
            scoring=ScoringConfig(**scoring_data),
            security=SecurityConfig(**security_data),
            logging=LoggingConfig(**logging_data),
        )
        cfg.validate()
        return cfg
