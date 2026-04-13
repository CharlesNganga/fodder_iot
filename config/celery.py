"""
config/celery.py
================
Celery application instance for the Fodder IoT backend.

The agronomic pipeline is fully asynchronous:
  1. ESP32 hits POST /api/ingest/  →  Django returns 201 immediately
  2. Django enqueues  process_telemetry_reading.delay(reading_id)
  3. Celery worker picks up the task and runs:
       - Temperature compensation  (EC₂₅ formula)
       - Open-Meteo weather fetch
       - Feature engineering       (rolling averages, Δ-EC₂₅)
       - ML inference              (NPK prediction)
       - Agronomic rules engine    (intervention logic)
       - IDW spatial map update

This design keeps the SIM800L from timing out waiting for a slow response.
"""

import os
from celery import Celery

# Tell Celery which Django settings module to use.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("fodder_iot")

# Load config from Django settings, using the CELERY_ namespace prefix.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Auto-discover tasks.py in every INSTALLED_APPS entry.
app.autodiscover_tasks()
