import os
from celery import Celery

from dotenv import load_dotenv

load_dotenv()

def make_celery() -> Celery:
    """
    Create a Celery app configured for Redis with sane defaults for long-running tasks.
    """
    broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    result_backend = os.getenv("CELERY_RESULT_BACKEND", broker_url)

    app = Celery(
        "parsec_slides_backend",
        broker=broker_url,
        backend=result_backend,
        include=["python.tasks.slides"],
    )

    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        task_default_queue=os.getenv("CELERY_TASK_DEFAULT_QUEUE", "slides"),
        task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "1800")),  # 30 minutes
        task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", "2400")),  # 40 minutes
        broker_transport_options={
            "visibility_timeout": int(os.getenv("CELERY_VISIBILITY_TIMEOUT", "3600"))  # 1 hour
        },
        result_expires=int(os.getenv("CELERY_RESULT_EXPIRES", "14400")),
    )

    @app.task(name="celery.healthcheck")
    def healthcheck() -> str:
        return "ok"

    return app


celery_app = make_celery()

