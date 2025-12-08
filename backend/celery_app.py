"""
Celery configuration for async task processing
"""
import os
from celery import Celery
from pathlib import Path

# Try loading environment variables
try:
    from dotenv import load_dotenv
    backend_env = Path(__file__).parent / '.env'
    root_env = Path(__file__).parent.parent / '.env'
    if backend_env.exists():
        load_dotenv(backend_env)
    elif root_env.exists():
        load_dotenv(root_env)
except ImportError:
    pass

# Redis URL - default to localhost if not set
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

# Set up Python path BEFORE creating Celery app
import sys
backend_dir = Path(__file__).parent
python_dir = backend_dir / 'python'
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Create Celery app
celery_app = Celery(
    'parsec_slides',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['python.tasks']  # This will import python/tasks.py
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600 * 2,  # 2 hours hard limit
    task_soft_time_limit=3600,  # 1 hour soft limit
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1,  # Restart worker after each task to prevent memory leaks
)

if __name__ == '__main__':
    celery_app.start()

