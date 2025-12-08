# Celery Setup for Async Job Processing

This project uses Celery with Redis for async job processing to handle long-running tasks like processing 500k+ rows.

## Setup

### 1. Install Redis

**macOS:**

```bash
brew install redis
brew services start redis
```

**Linux (Ubuntu/Debian):**

```bash
sudo apt-get install redis-server
sudo systemctl start redis
```

**Docker:**

```bash
docker run -d -p 6379:6379 redis:latest
```

### 2. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Start Celery Worker

In a separate terminal, start the Celery worker:

**Option 1: Using the start script (recommended):**

```bash
cd backend
./start_celery.sh
```

**Option 2: Manual start:**

```bash
cd backend
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/python"
celery -A celery_app worker --loglevel=info
```

### 4. Start Flask App

In another terminal:

```bash
cd backend
python app.py
```

## Environment Variables

You can optionally set `REDIS_URL` in your `.env` file:

```bash
REDIS_URL=redis://localhost:6379/0
```

Default is `redis://localhost:6379/0` if not set.

## How It Works

1. Frontend calls `/ingest-and-generate` endpoint
2. Flask immediately returns a `jobId` (202 Accepted)
3. Frontend polls `/job-status/<jobId>` every 2 seconds
4. Celery worker processes the job in the background
5. Progress updates are available via the status endpoint
6. When complete, frontend receives charts and summary

## Monitoring

You can monitor Celery tasks using Flower:

```bash
pip install flower
celery -A celery_app flower
```

Then visit http://localhost:5555

## Troubleshooting

- **"Celery not available"**: Make sure Redis is running and Celery worker is started
- **Jobs stuck in PENDING**: Check that Celery worker is running and can connect to Redis
- **Import errors**: Make sure you're running from the `backend/` directory
