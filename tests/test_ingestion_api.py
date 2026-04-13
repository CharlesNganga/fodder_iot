"""
tests/test_ingestion_api.py  (FIXED)
=====================================
Fix: GET/PUT tests now expect 401, not 405.

Reason: The decorator order is @csrf_exempt → @validate_esp32_hmac → view.
HMAC validation runs BEFORE the method check inside the view body. A GET
request with no signature header correctly returns 401 — the endpoint does
not reveal its existence to unsigned callers. This is correct security
behaviour (fail-safe default). Tests updated to match.
"""

import hashlib
import hmac
import json

import pytest
from django.contrib.gis.geos import Point

from apps.telemetry.models import DailyIoTTelemetry
from tests.conftest import INGEST_URL


def _post(client, body: str, sig: str):
    return client.post(
        INGEST_URL,
        data=body,
        content_type="application/json",
        HTTP_X_ESP32_SIGNATURE=sig,
    )


def _post_no_sig(client, body: str):
    return client.post(INGEST_URL, data=body, content_type="application/json")


class TestHMACAuthentication:

    @pytest.mark.django_db
    def test_missing_hmac_header_returns_401(self, api_client, make_signed_payload):
        body, _ = make_signed_payload(offset_secs=0)
        response = _post_no_sig(api_client, body)
        assert response.status_code == 401
        assert "error" in response.json()
        assert DailyIoTTelemetry.objects.count() == 0

    @pytest.mark.django_db
    def test_wrong_hmac_returns_401(self, api_client, make_signed_payload):
        body, _ = make_signed_payload(offset_secs=1)
        response = _post(api_client, body, "deadbeef" * 8)
        assert response.status_code == 401
        assert response.json()["error"] == "Invalid HMAC signature."
        assert DailyIoTTelemetry.objects.count() == 0

    @pytest.mark.django_db
    def test_tampered_payload_rejected(self, api_client, make_signed_payload):
        body, sig = make_signed_payload(offset_secs=2, ph=6.2)
        tampered  = body.replace('"ph":6.2', '"ph":3.0')
        response  = _post(api_client, tampered, sig)
        assert response.status_code == 401
        assert DailyIoTTelemetry.objects.count() == 0

    @pytest.mark.django_db
    def test_valid_hmac_accepted(self, api_client, make_signed_payload):
        body, sig = make_signed_payload(offset_secs=3)
        response  = _post(api_client, body, sig)
        assert response.status_code in (200, 201)


class TestInputValidation:

    @pytest.mark.django_db
    @pytest.mark.parametrize("missing_field", [
        "device_id", "timestamp", "latitude", "longitude",
        "ph", "ec_raw", "moisture", "temperature",
    ])
    def test_missing_required_field_returns_400(
        self, api_client, make_signed_payload, settings, missing_field
    ):
        body, _ = make_signed_payload(offset_secs=10)
        data     = json.loads(body)
        del data[missing_field]
        new_body = json.dumps(data, separators=(",", ":"))
        new_sig  = hmac.new(
            settings.ESP32_HMAC_SECRET.encode(),
            new_body.encode(),
            hashlib.sha256,
        ).hexdigest()
        response = _post(api_client, new_body, new_sig)
        assert response.status_code == 400
        assert "error" in response.json()
        assert DailyIoTTelemetry.objects.count() == 0

    @pytest.mark.django_db
    @pytest.mark.parametrize("field,bad_value,description", [
        ("ph",          99.9,  "pH above 14"),
        ("ph",          -1.0,  "pH below 0"),
        ("moisture",   110.0,  "moisture above 100%"),
        ("moisture",    -5.0,  "moisture below 0%"),
        ("ec_raw",     -10.0,  "EC negative"),
        ("temperature", 95.0,  "temperature above 80°C"),
        ("temperature",-15.0,  "temperature below -10°C"),
        ("latitude",    95.0,  "latitude above 90°"),
        ("longitude",  200.0,  "longitude above 180°"),
    ])
    def test_out_of_range_sensor_value_returns_400(
        self, api_client, make_signed_payload, settings, field, bad_value, description
    ):
        body, _ = make_signed_payload(offset_secs=20)
        data     = json.loads(body)
        data[field] = bad_value
        data["timestamp"] = f"2025-06-01T{abs(int(bad_value)) % 24:02d}:00:00Z"
        new_body = json.dumps(data, separators=(",", ":"))
        new_sig  = hmac.new(
            settings.ESP32_HMAC_SECRET.encode(),
            new_body.encode(),
            hashlib.sha256,
        ).hexdigest()
        response = _post(api_client, new_body, new_sig)
        assert response.status_code == 400, f"Expected 400 for {description}"
        assert "violations" in response.json()
        assert DailyIoTTelemetry.objects.count() == 0

    @pytest.mark.django_db
    def test_invalid_json_body_returns_400(self, api_client, settings):
        bad_body = b'{"device_id": "TEST", "ph": NOT_A_NUMBER}'
        sig = hmac.new(
            settings.ESP32_HMAC_SECRET.encode(), bad_body, hashlib.sha256
        ).hexdigest()
        response = api_client.post(
            INGEST_URL, data=bad_body, content_type="application/json",
            HTTP_X_ESP32_SIGNATURE=sig,
        )
        assert response.status_code == 400
        assert "Invalid JSON" in response.json().get("error", "")

    @pytest.mark.django_db
    def test_get_without_signature_returns_401(self, api_client):
        """
        GET returns 401 (not 405) because HMAC check runs before method check.
        This is correct: the endpoint should not reveal its existence to
        unsigned callers — fail-safe security behaviour.
        """
        response = api_client.get(INGEST_URL)
        assert response.status_code == 401

    @pytest.mark.django_db
    def test_put_without_signature_returns_401(self, api_client):
        """Same as GET — HMAC gate fires before method check."""
        response = api_client.put(
            INGEST_URL, data="{}", content_type="application/json"
        )
        assert response.status_code == 401

    @pytest.mark.django_db
    def test_signed_get_returns_405(self, api_client, settings):
        """
        A GET with a valid HMAC passes the auth check and hits the method guard.
        This verifies the 405 path is reachable for signed non-POST requests.
        """
        body = b""
        sig  = hmac.new(
            settings.ESP32_HMAC_SECRET.encode(), body, hashlib.sha256
        ).hexdigest()
        response = api_client.get(
            INGEST_URL, HTTP_X_ESP32_SIGNATURE=sig,
        )
        assert response.status_code == 405


class TestIdempotency:

    @pytest.mark.django_db
    def test_first_request_returns_201(self, api_client, make_signed_payload):
        body, sig = make_signed_payload(offset_secs=100)
        response  = _post(api_client, body, sig)
        assert response.status_code == 201
        assert response.json()["status"] == "accepted"

    @pytest.mark.django_db
    def test_duplicate_request_returns_200(self, api_client, make_signed_payload):
        body, sig = make_signed_payload(offset_secs=200)
        r1 = _post(api_client, body, sig)
        assert r1.status_code == 201
        r2 = _post(api_client, body, sig)
        assert r2.status_code == 200
        assert r2.json()["status"] == "duplicate"

    @pytest.mark.django_db
    def test_duplicate_does_not_create_second_row(self, api_client, make_signed_payload):
        body, sig = make_signed_payload(offset_secs=300)
        _post(api_client, body, sig)
        _post(api_client, body, sig)
        assert DailyIoTTelemetry.objects.count() == 1

    @pytest.mark.django_db
    def test_different_timestamps_create_separate_rows(self, api_client, make_signed_payload):
        body1, sig1 = make_signed_payload(device_id="DEV:AA", offset_secs=400)
        body2, sig2 = make_signed_payload(device_id="DEV:AA", offset_secs=401)
        assert _post(api_client, body1, sig1).status_code == 201
        assert _post(api_client, body2, sig2).status_code == 201
        assert DailyIoTTelemetry.objects.count() == 2


class TestDatabaseWrite:

    @pytest.mark.django_db
    def test_reading_stored_with_correct_values(self, api_client, make_signed_payload):
        body, sig = make_signed_payload(
            offset_secs=500, device_id="DEV:STORE:TEST",
            ph=6.8, ec_raw=295.5, moisture=51.2, temperature=24.3, battery_v=7.1,
        )
        response   = _post(api_client, body, sig)
        assert response.status_code == 201
        reading    = DailyIoTTelemetry.objects.get(id=response.json()["reading_id"])
        assert reading.device_id == "DEV:STORE:TEST"
        assert abs(reading.ph - 6.8) < 0.001
        assert abs(reading.ec_raw_us_per_cm - 295.5) < 0.001
        assert abs(reading.moisture_percent - 51.2) < 0.001
        assert abs(reading.temperature_celsius - 24.3) < 0.001
        assert abs(reading.battery_voltage - 7.1) < 0.001
        assert reading.is_processed is False  # Celery not invoked by this test class

    @pytest.mark.django_db
    def test_reading_location_stored_as_postgis_point(self, api_client, make_signed_payload):
        body, sig = make_signed_payload(
            offset_secs=600, latitude=-1.1018, longitude=37.0144,
        )
        response = _post(api_client, body, sig)
        assert response.status_code == 201
        reading  = DailyIoTTelemetry.objects.get(id=response.json()["reading_id"])
        assert isinstance(reading.location, Point)
        assert abs(reading.location.x - 37.0144) < 0.0001  # longitude
        assert abs(reading.location.y - (-1.1018)) < 0.0001  # latitude

    @pytest.mark.django_db
    def test_battery_v_is_optional(self, api_client, make_signed_payload, settings):
        body, _ = make_signed_payload(offset_secs=700)
        data     = json.loads(body)
        data.pop("battery_v", None)
        new_body = json.dumps(data, separators=(",", ":"))
        new_sig  = hmac.new(
            settings.ESP32_HMAC_SECRET.encode(), new_body.encode(), hashlib.sha256
        ).hexdigest()
        response = _post(api_client, new_body, new_sig)
        assert response.status_code == 201
        reading  = DailyIoTTelemetry.objects.get(id=response.json()["reading_id"])
        assert reading.battery_voltage is None