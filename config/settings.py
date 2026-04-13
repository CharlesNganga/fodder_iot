"""
config/settings.py
==================
Central Django configuration for the Fodder IoT Precision Agriculture backend.

Key design decisions:
- django-environ reads all secrets from .env — zero hardcoded credentials.
- DATABASES uses dj-database-url-style parsing so PostGIS is auto-detected.
- TimescaleDB is registered as a third-party app so its migration operations work.
- Celery broker/backend both point to the same Redis instance (different DB indices
  can be used in production to isolate concerns).
- HMAC_SECRET is loaded here once and imported by the middleware — single source of truth.
"""

import os
from pathlib import Path
import environ

# ---------------------------------------------------------------------------
# Base directory & environment loader
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

env = environ.Env(
    DEBUG=(bool, False),
    ALLOWED_HOSTS=(list, ["localhost", "127.0.0.1"]),
)
environ.Env.read_env(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
SECRET_KEY = env("SECRET_KEY")
DEBUG = env("DEBUG")
ALLOWED_HOSTS = env("ALLOWED_HOSTS")

# ---------------------------------------------------------------------------
# Application registry
# ---------------------------------------------------------------------------
DJANGO_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.gis",          # Enables PostGIS / GeoQuerySet / PolygonField etc.
]

THIRD_PARTY_APPS = [
    "rest_framework",              # Django REST Framework
    "timescale",                   # django-timescaledb — migration support for hypertables
]

LOCAL_APPS = [
    "apps.farms",                  # Farm + field geofencing (PostGIS PolygonField)
    "apps.ingestion",              # ESP32 ingest endpoint + HMAC middleware
    "apps.telemetry",              # DailyIoTTelemetry hypertable
    "apps.agronomics",             # CompositeBaselineLabTest + NPK predictions + rules
]

INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

# ---------------------------------------------------------------------------
# Middleware stack — note HMACValidationMiddleware is applied only on the
# /api/ingest/ route via a decorator pattern (see ingestion/views.py),
# NOT globally here. This avoids running HMAC checks on admin/DRF browsable API.
# ---------------------------------------------------------------------------
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# ---------------------------------------------------------------------------
# Database — PostGIS backend (superset of standard PostgreSQL backend).
# TimescaleDB lives in the same database; PostGIS and TimescaleDB extensions
# are created via scripts/create_hypertable.py after the first migration.
# ---------------------------------------------------------------------------
DATABASES = {
    "default": env.db(
        "DATABASE_URL",
        default="postgres://fodder_user:change_this_password@localhost:5432/fodder_db",
    )
}
# Override the ENGINE to use the GIS-aware backend regardless of what
# django-environ detects from the URL scheme.
DATABASES["default"]["ENGINE"] = "django.contrib.gis.db.backends.postgis"

# ---------------------------------------------------------------------------
# Django REST Framework
# ---------------------------------------------------------------------------
REST_FRAMEWORK = {
    # Use JWT or session auth in production — keeping it simple for dev.
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        # Ingestion endpoint overrides this with AllowAny + HMAC validation.
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
    ],
    # ESP32 sends at most 1 reading per minute; 120/hour gives generous headroom.
    "DEFAULT_THROTTLE_RATES": {
        "anon": "120/hour",
    },
}

# ---------------------------------------------------------------------------
# Celery — async task queue for the agronomic pipeline
# ---------------------------------------------------------------------------
CELERY_BROKER_URL = env("REDIS_URL", default="redis://localhost:6379/0")
CELERY_RESULT_BACKEND = env("REDIS_URL", default="redis://localhost:6379/0")
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = "Africa/Nairobi"   # UTC+3 — matches farm operations

# ---------------------------------------------------------------------------
# HMAC shared secret (must match ESP32 firmware)
# ---------------------------------------------------------------------------
ESP32_HMAC_SECRET = env("ESP32_HMAC_SECRET", default="REPLACE_ME_IN_DOT_ENV")

# ---------------------------------------------------------------------------
# Open-Meteo (free, no API key required)
# ---------------------------------------------------------------------------
OPEN_METEO_BASE_URL = env(
    "OPEN_METEO_BASE_URL",
    default="https://api.open-meteo.com/v1/forecast",
)

# ---------------------------------------------------------------------------
# Internationalisation
# ---------------------------------------------------------------------------
LANGUAGE_CODE = "en-us"
TIME_ZONE = "Africa/Nairobi"
USE_I18N = True
USE_TZ = True   # Store all datetimes as UTC in DB; display in Africa/Nairobi

# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# IDW heat-map PNGs are stored here and served via /media/
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
