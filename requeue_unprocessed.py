"""
requeue_unprocessed.py
======================
Requeues any DailyIoTTelemetry readings that were saved when Redis
was down (is_processed=False). Run this once after starting Redis.

USAGE:
  python requeue_unprocessed.py
"""
import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from apps.telemetry.models import DailyIoTTelemetry
from apps.ingestion.tasks import process_telemetry_reading

pending = DailyIoTTelemetry.objects.filter(is_processed=False).order_by("time")
print(f"Found {pending.count()} unprocessed readings. Requeueing...")
for r in pending:
    process_telemetry_reading.delay(r.id)
    print(f"  Queued reading_id={r.id} device={r.device_id} time={r.time}")
print("Done — check Celery worker output for results.")
