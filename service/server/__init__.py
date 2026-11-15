"""gRPC Server factory for T11 Evaluator Service."""
from __future__ import annotations

import logging
from concurrent import futures
from typing import Any, Optional

import grpc

from service.server.config import ServiceConfig

logger = logging.getLogger("t11.server")


def create_server(
    config: ServiceConfig,
    engine: Optional[Any] = None,
) -> grpc.Server:
    """Create and configure gRPC server instance according to ServiceConfig."""
    config.validate()

    from service.core.processing_engine import ProcessingEngine
    from service.generated import pb2_grpc
    from service.server.grpc_handler import T11EvaluatorServicer

    options = [
        ("grpc.max_receive_message_length", config.grpc.max_message_size_bytes),
        ("grpc.max_send_message_length", config.grpc.max_message_size_bytes),
    ]

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.grpc.max_workers),
        options=options,
    )

    if engine is None:
        engine = ProcessingEngine(config)

    servicer = T11EvaluatorServicer(engine)
    pb2_grpc.add_T11EvaluatorServicer_to_server(servicer, server)

    if config.grpc.host.startswith("unix:"):
        bind_addr = config.grpc.host
    else:
        bind_addr = f"{config.grpc.host}:{config.grpc.port}"
    if config.security.enable_tls:
        raise NotImplementedError("TLS configuration is not yet implemented for offline local run")
    else:
        server.add_insecure_port(bind_addr)
        logger.info("Bound insecure gRPC port on %s", bind_addr)

    return server
