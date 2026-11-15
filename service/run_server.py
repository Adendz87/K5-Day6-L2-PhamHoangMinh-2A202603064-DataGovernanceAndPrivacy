"""Standalone runner for T11 Evaluator gRPC service with Real YOLOv8n."""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from pathlib import Path

from service.core.processing_engine import ProcessingEngine
from service.server import create_server
from service.server.config import ServiceConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("t11.main")


def main() -> None:
    t11_root = Path(__file__).resolve().parent.parent
    config_path = t11_root / "service" / "configs" / "service.yaml"
    cfg = ServiceConfig.from_yaml(config_path)

    # Force Real YOLO, offline, SAM3 disabled
    cfg.models.yolo.mock = False
    cfg.models.sam3.enabled = False
    cfg.models.sam3.mock = True

    # Allow environment override for host and port
    cfg.grpc.host = os.environ.get("T11_HOST", cfg.grpc.host)
    if "T11_PORT" in os.environ:
        cfg.grpc.port = int(os.environ["T11_PORT"])

    # Resolve weights path
    if not Path(cfg.models.yolo.weights_path).is_absolute():
        cfg.models.yolo.weights_path = str(t11_root / cfg.models.yolo.weights_path)

    engine = ProcessingEngine(cfg)
    server = create_server(cfg, engine=engine)
    server.start()
    logger.info("T11 gRPC Service running on %s:%s (mock=False, SAM3 disabled)", cfg.grpc.host, cfg.grpc.port)

    def handle_sig(sig, frame):
        logger.info("Stopping T11 gRPC Service...")
        server.stop(grace=2).wait(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    server.wait_for_termination()


if __name__ == "__main__":
    main()
