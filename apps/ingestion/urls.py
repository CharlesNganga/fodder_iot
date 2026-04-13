# apps/ingestion/urls.py
from django.urls import path
from .views import ingest_telemetry

urlpatterns = [
    path('ingest/', ingest_telemetry, name='ingest_telemetry'),
]