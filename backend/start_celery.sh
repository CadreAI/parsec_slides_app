#!/bin/bash
# Script to start Celery worker with proper Python path setup

cd "$(dirname "$0")"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/python"

# Start Celery worker
celery -A celery_app worker --loglevel=info

