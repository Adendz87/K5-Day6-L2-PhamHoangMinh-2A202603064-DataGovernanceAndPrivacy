#!/bin/bash
# T11 Data Quality gRPC Server Startup Script
# Run this BEFORE starting CVAT or after system restart.
# T11 must run on the macOS host (outside Docker) so cvat_server can reach it via host.docker.internal:50051

set -e

T11_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$T11_ROOT"

echo "=================================================="
echo "  T11 Data Quality AI Service"
echo "  Real YOLOv8n | mock=False | SAM3 disabled"
echo "  Listening on 0.0.0.0:50051"
echo "=================================================="
echo ""

# Verify YOLO weights exist
if [ ! -f "models/yolo/yolov8n.pt" ]; then
    echo "ERROR: YOLO weights not found at models/yolo/yolov8n.pt"
    exit 1
fi

echo "YOLO weights: OK"
echo "Starting gRPC server..."
PYTHONPATH=. .venv/bin/python service/run_server.py
