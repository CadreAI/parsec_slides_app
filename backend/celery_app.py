"""
Celery configuration for Parsec Slides App
"""
import os
import sys
from pathlib import Path
from celery import Celery

# Set matplotlib backend before any imports
import matplotlib
matplotlib.use('Agg')

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent / 'python'))

def make_celery() -> Celery:
    """
    Create a Celery app configured for Redis with defaults for long-running
    chart generation and slide creation tasks.
    """
    broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    result_backend = os.getenv("CELERY_RESULT_BACKEND", broker_url)

    celery_app = Celery(
        "parsec_slides_app",
        broker=broker_url,
        backend=result_backend,
        include=["tasks"],  # Import tasks module to register tasks
    )

    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "1500")),  # 25 minutes
        task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", "1800")),  # 30 minutes
        broker_transport_options={
            "visibility_timeout": int(os.getenv("CELERY_VISIBILITY_TIMEOUT", "1800"))  # 30 minutes
        },
        result_expires=int(os.getenv("CELERY_RESULT_EXPIRES", "3600")),  # 1 hour
    )

    @celery_app.task(name="celery.healthcheck")
    def healthcheck() -> str:
        return "ok"

    return celery_app


celery_app = make_celery()

# Import tasks to register them with Celery
# This must happen after celery_app is created to avoid circular imports
try:
    import tasks  # noqa: F401
except ImportError:
    # If tasks can't be imported, that's okay - they'll be imported when needed
    # But we log a warning for debugging
    import warnings
    warnings.warn("Could not import tasks module. Tasks may not be registered.", ImportWarning)

