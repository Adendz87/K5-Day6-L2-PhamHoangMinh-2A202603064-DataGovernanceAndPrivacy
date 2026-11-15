"""Generated Protocol Buffer & gRPC stubs for T11 Evaluator."""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure generated directory is in sys.path so t11_pb2 can be imported by t11_pb2_grpc
_gen_dir = str(Path(__file__).resolve().parent)
if _gen_dir not in sys.path:
    sys.path.insert(0, _gen_dir)

from . import t11_pb2 as pb2
from . import t11_pb2_grpc as pb2_grpc

__all__ = ["pb2", "pb2_grpc"]
