import os
from dotenv import load_dotenv

load_dotenv()

# Modern Celery 5.0+ lowercase settings
broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'Europe/Istanbul'
enable_utc = True

# Reliability & Performance
task_track_started = True
task_time_limit = 300
worker_prefetch_multiplier = 1
worker_max_tasks_per_child = 50

# MacOS Sustainability settings
worker_pool_restarts = True
