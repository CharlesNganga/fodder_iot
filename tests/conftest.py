"""
tests/conftest.py  (FIXED)
===========================
Key fix: Celery app is initialised at import time before Django test settings
are applied. CELERY_TASK_ALWAYS_EAGER in settings_test.py is read by Django
but the already-created Celery app instance keeps the old config.

Solution: Use a session-scoped autouse fixture that calls app.conf.update()
directly on the live Celery app object — this overrides AFTER app creation.
"""

import hashlib
import hmac
import json
from datetime import datetime, timezone, timedelta, date

import pytest
from django.contrib.auth.models import User
from django.contrib.gis.geos import Point, Polygon
from django.test import Client

from apps.agronomics.models import CompositeBaselineLabTest
from apps.farms.models import Farm, Farmer, Field
from apps.telemetry.models import DailyIoTTelemetry


# ── CRITICAL FIX: Force Celery into eager/synchronous mode ───────────────────

@pytest.fixture(scope="session", autouse=True)
def force_celery_eager():
    """
    Force the live Celery app instance into synchronous eager mode.

    WHY this is necessary:
      config/celery.py creates the Celery app at module import time:
        app = Celery("fodder_iot")
        app.config_from_object("django.conf:settings", namespace="CELERY")

      When pytest loads, this import happens before settings_test.py overrides
      CELERY_TASK_ALWAYS_EAGER. The Celery app instance therefore has the
      production config (real broker) baked in.

      Setting CELERY_TASK_ALWAYS_EAGER=True in settings_test.py DOES update
      Django's settings dict, but the already-instantiated Celery app does
      NOT re-read settings unless explicitly told to.

      app.conf.update() patches the live app instance directly — this is the
      only reliable way to change Celery behaviour after app creation.
    """
    from config.celery import app as celery_app
    celery_app.conf.update(
        task_always_eager=True,       # .delay() runs synchronously in-process
        task_eager_propagates=True,   # exceptions surface immediately in tests
    )
    yield
    # Restore after session (important if other test suites share the process)
    celery_app.conf.update(
        task_always_eager=False,
        task_eager_propagates=False,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_signature(secret: str, body: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        body.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _iso_now(offset_seconds: int = 0) -> str:
    ts = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


# ── User / Farmer ─────────────────────────────────────────────────────────────

@pytest.fixture
def django_user(db):
    return User.objects.create_user(
        username="charles_nganga",
        password="testpass123",
        first_name="Charles",
        last_name="Ng'ang'a",
    )


@pytest.fixture
def farmer_profile(db, django_user):
    return Farmer.objects.create(
        user=django_user,
        phone_number="+254712345678",
        location_county="Kiambu",
    )


# ── Farm / Field ──────────────────────────────────────────────────────────────

@pytest.fixture
def farm(db, farmer_profile):
    return Farm.objects.create(
        farmer=farmer_profile,
        name="Ng'ang'a Test Farm — Thika",
        centroid_latitude=-1.1018,
        centroid_longitude=37.0144,
    )


@pytest.fixture
def field_boundary():
    """
    Valid WGS84 polygon near Thika, Kenya — real coordinates so PostGIS
    UTM Zone 37 reprojection in Field.save() computes a valid hectare value.

    WHY geometry not geography?
      Field.boundary is PolygonField(srid=4326) — geometry type.
      DailyIoTTelemetry.location is PointField(geography=True).
      PostGIS cannot directly compare geography and geometry types in
      boundary__contains queries. The fix is in tasks.py (see below);
      the fixture itself just stores the correct geometry polygon.
    """
    return Polygon(
        (
            (37.0140, -1.1022),
            (37.0150, -1.1022),
            (37.0150, -1.1012),
            (37.0140, -1.1012),
            (37.0140, -1.1022),
        ),
        srid=4326,
    )


@pytest.fixture
def field(db, farm, field_boundary):
    return Field.objects.create(
        farm=farm,
        name="North Block — Napier",
        fodder_type="napier",
        boundary=field_boundary,
    )


# ── Baselines ─────────────────────────────────────────────────────────────────

@pytest.fixture
def baseline_lab_test(db, field):
    """Healthy soil — rules engine produces NO_ACTION."""
    # test_date is set to 10 days ago so that after physics decay the soil
    # is still healthy (N>30, P>15) and the rules engine produces NO_ACTION.
    # A fixed date like 2025-03-01 would be 400+ days old by the time this
    # test runs in 2026, causing the decay estimates to trigger interventions.
    recent_date = (date.today() - timedelta(days=10)).isoformat()
    return CompositeBaselineLabTest.objects.create(
        field=field,
        season_label="Long Rains 2025",
        test_date=recent_date,
        nitrogen_mg_per_kg=35.0,
        phosphorus_mg_per_kg=18.0,
        potassium_mg_per_kg=150.0,
        ph_at_test_date=6.2,
        ec_us_per_cm_at_test_date=280.0,
        organic_carbon_percent=2.1,
        is_active=True,
    )


@pytest.fixture
def baseline_acidic_soil(db, field):
    """pH=5.0 — triggers LIME_APPLICATION, blocks CAN."""
    return CompositeBaselineLabTest.objects.create(
        field=field,
        season_label="Dry Season 2025 — Acidic Plot",
        test_date=(date.today() - timedelta(days=10)).isoformat(),
        nitrogen_mg_per_kg=15.0,
        phosphorus_mg_per_kg=8.0,
        potassium_mg_per_kg=100.0,
        ph_at_test_date=5.0,
        ec_us_per_cm_at_test_date=200.0,
        is_active=True,
    )


@pytest.fixture
def baseline_nitrogen_deficient(db, field):
    """
    Baseline that triggers CAN_TOP_DRESS when rain < 15mm.

    WHY N0=28, test_date=90 days ago (not N0=12, 10 days ago):
      The XGBoost models were trained on farms with N0 in [25, 60] mg/kg.
      N0=12 is out-of-distribution — the model extrapolates unpredictably.
      At 90 days with N0=28: n_decay_estimate = 28 × exp(-0.008×90) = 13.6 mg/kg,
      which is below the 20 mg/kg CAN threshold, so CAN correctly triggers.
      P remains at 13.7 mg/kg (above 10 mg/kg DAP threshold), keeping
      the test focused solely on the CAN rule.
    """
    test_date_90d = (date.today() - timedelta(days=90)).isoformat()
    return CompositeBaselineLabTest.objects.create(
        field=field,
        season_label="Long Rains 2025 — N Deficient",
        test_date=test_date_90d,
        nitrogen_mg_per_kg=28.0,    # in training range [25,60]; decays to ~14 at 90d
        phosphorus_mg_per_kg=18.0,  # decays to ~13.7 at 90d (above 10mg/kg DAP threshold)
        potassium_mg_per_kg=140.0,
        ph_at_test_date=6.5,        # above 5.5 — CAN is not blocked by pH
        ec_us_per_cm_at_test_date=240.0,
        is_active=True,
    )


# ── HTTP Client ───────────────────────────────────────────────────────────────

@pytest.fixture
def api_client():
    return Client()


# ── Payload factory ───────────────────────────────────────────────────────────

@pytest.fixture
def make_signed_payload(settings):
    def _factory(
        device_id="TEST:ESP32:CONFTEST",
        offset_secs=0,
        latitude=-1.1018,
        longitude=37.0144,
        ph=6.2,
        ec_raw=310.0,
        moisture=45.0,
        temperature=23.0,
        battery_v=4.1,
        **extra,
    ):
        data = {
            "device_id":   device_id,
            "timestamp":   _iso_now(offset_secs),
            "latitude":    latitude,
            "longitude":   longitude,
            "ph":          ph,
            "ec_raw":      ec_raw,
            "moisture":    moisture,
            "temperature": temperature,
            "battery_v":   battery_v,
            **extra,
        }
        body = json.dumps(data, separators=(",", ":"))
        sig  = _make_signature(settings.ESP32_HMAC_SECRET, body)
        return body, sig

    return _factory


INGEST_URL = "/api/ingest/"