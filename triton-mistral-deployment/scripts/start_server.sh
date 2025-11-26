#!/bin/bash

echo "Starting Triton Inference Server for Mistral-7B..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

# Check if NVIDIA runtime is available
if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA Docker runtime is not available. Please install nvidia-container-toolkit."
    exit 1
fi

# Navigate to docker directory
cd "$(dirname "$0")/../docker"

# Start the server
docker-compose up -d

echo "Server starting... Check logs with: docker-compose logs -f triton-server"
echo "Server will be available at:"
echo "  - HTTP: http://localhost:8000"
echo "  - gRPC: localhost:8001"
echo "  - Metrics: http://localhost:8002"

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -f http://localhost:8000/v2/health/ready >/dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "Server failed to start within 60 seconds"
        docker-compose logs triton-server
        exit 1
    fi
    sleep 1
done