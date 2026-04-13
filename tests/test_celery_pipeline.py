"""
tests/test_celery_pipeline.py
==============================
End-to-end integration tests for the async Celery pipeline.

These tests verify the full lifecycle:
  POST /api/ingest/
    → DailyIoTTelemetry created
    → process_telemetry_reading() runs synchronously (CELERY_TASK_ALWAYS_EAGER)
    → EC₂₅ temperature compensation computed
    → Open-Meteo mocked (no real HTTP calls)
    → NPKPrediction created
    → AgronomicRecommendation(s) created with correct intervention_type

Test matrix:
  ┌─────────────────────────────┬──────────────────────────┐
  │ Soil condition              │ Expected recommendation  │
  ├─────────────────────────────┼──────────────────────────┤
  │ Healthy (default baseline)  │ NO_ACTION                │
  │ Low pH (< 5.5)              │ LIME_APPLICATION         │
  │ Low N, good pH, no rain     │ CAN_TOP_DRESS            │
  │ Low moisture, no rain       │ IRRIGATE                 │
  │ Low moisture, rain forecast │ NO irrigation (suppressed│
  └─────────────────────────────┴──────────────────────────┘
"""

import hashlib
import hmac
import json
from unittest.mock import patch

import pytest

from apps.agronomics.models import AgronomicRecommendation, NPKPrediction
from apps.telemetry.models import DailyIoTTelemetry
from tests.conftest import INGEST_URL


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _post_signed(client, body: str, sig: str):
    """POST with correct content-type and HMAC header."""
    return client.post(
        INGEST_URL,
        data=body,
        content_type="application/json",
        HTTP_X_ESP32_SIGNATURE=sig,
    )


def _ingest_and_get_reading(client, make_signed_payload, **kwargs) -> DailyIoTTelemetry:
    """
    Helper: POST a signed payload, assert 201, return the created DB object.
    Because CELERY_TASK_ALWAYS_EAGER=True, the Celery task has already run
    by the time this function returns — the reading is fully processed.
    """
    body, sig = make_signed_payload(**kwargs)
    response = _post_signed(client, body, sig)
    assert response.status_code == 201, (
        f"Expected 201 but got {response.status_code}: {response.content}"
    )
    reading_id = response.json()["reading_id"]
    return DailyIoTTelemetry.objects.get(id=reading_id)


# ══════════════════════════════════════════════════════════════════════════════
# Open-Meteo mock strategy
# ══════════════════════════════════════════════════════════════════════════════

# WHY mock _fetch_open_meteo rather than the requests library directly?
#   tasks.py imports _fetch_open_meteo as a module-level function used
#   inside the Celery task.  Mocking at the task module level is more
#   specific and avoids accidentally suppressing other requests calls
#   (e.g., from django-environ or health checks).
#
# We patch at "apps.ingestion.tasks._fetch_open_meteo" — the location
# where it is USED, not where it is DEFINED.

def _no_rain_forecast():
    """Open-Meteo mock returning 0mm rain — irrigation conditions met."""
    return {"rain_next_24h_mm": 0.0, "rain_next_7d_mm": 0.0}

def _heavy_rain_forecast():
    """Open-Meteo mock returning 20mm rain — fertiliser suppressed, irrigation suppressed."""
    return {"rain_next_24h_mm": 20.0, "rain_next_7d_mm": 50.0}

def _light_rain_forecast():
    """
    14mm rain — above irrigation suppression threshold (5mm) but below
    CAN suppression threshold (15mm).  So CAN is still safe to apply.
    """
    return {"rain_next_24h_mm": 14.0, "rain_next_7d_mm": 30.0}


MOCK_PATH = "apps.ingestion.tasks._fetch_open_meteo"


# ══════════════════════════════════════════════════════════════════════════════
# EC₂₅ temperature compensation unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEC25TemperatureCompensation:
    """
    These unit tests verify the formula is implemented correctly in tasks.py.
    They call the Celery task via ingest to exercise the real code path.

    Formula: EC₂₅ = EC_raw / (1 + 0.019 × (T − 25))
    Reference: Rhoades et al., 1989.
    """

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_no_rain_forecast())
    def test_ec25_at_reference_temperature(
        self, mock_meteo, api_client, make_signed_payload, baseline_lab_test
    ):
        """At T=25°C, EC₂₅ must equal EC_raw exactly (no correction applied)."""
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload,
            offset_secs=1000, ec_raw=300.0, temperature=25.0,
        )
        reading.refresh_from_db()
        assert abs(reading.ec_25_us_per_cm - 300.0) < 0.01

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_no_rain_forecast())
    def test_ec25_hot_soil_lower_than_raw(
        self, mock_meteo, api_client, make_signed_payload, baseline_lab_test
    ):
        """
        At T=35°C, EC₂₅ must be LOWER than EC_raw.
        Warm soil has higher raw EC — compensation brings it down to 25°C baseline.
        Expected: 300 / (1 + 0.019 × 10) = 300 / 1.19 = 252.10
        """
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload,
            offset_secs=1001, ec_raw=300.0, temperature=35.0,
        )
        reading.refresh_from_db()
        assert reading.ec_25_us_per_cm < 300.0
        assert abs(reading.ec_25_us_per_cm - 252.10) < 0.5

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_no_rain_forecast())
    def test_ec25_cold_morning_higher_than_raw(
        self, mock_meteo, api_client, make_signed_payload, baseline_lab_test
    ):
        """
        At T=15°C, EC₂₅ must be HIGHER than EC_raw.
        Cold soil suppresses measured EC — compensation raises it to 25°C baseline.
        Expected: 300 / (1 + 0.019 × -10) = 300 / 0.81 = 370.37
        """
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload,
            offset_secs=1002, ec_raw=300.0, temperature=15.0,
        )
        reading.refresh_from_db()
        assert reading.ec_25_us_per_cm > 300.0
        assert abs(reading.ec_25_us_per_cm - 370.37) < 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Happy path: full end-to-end pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestFullPipelineHappyPath:
    """
    Verify the complete pipeline for a healthy soil reading:
      POST → save → EC₂₅ → weather → NPKPrediction → AgronomicRecommendation
    """

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_no_rain_forecast())
    def test_happy_path_creates_npk_prediction(
        self, mock_meteo, api_client, make_signed_payload, baseline_lab_test
    ):
        """
        A valid POST with a healthy-soil baseline must create an NPKPrediction.
        The Celery task requires an active baseline — baseline_lab_test fixture
        provides one for the field resolved by the GPS point-in-polygon query.

        WHY is baseline_lab_test needed here?
          The GPS point (-1.1018, 37.0144) falls inside field.boundary.
          The Celery task resolves field via PostGIS spatial query, then
          looks up the active baseline for that field. Without baseline_lab_test,
          the task logs a warning and skips NPK prediction.
        """
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload, offset_secs=2000,
            ph=6.2, ec_raw=310.0, moisture=50.0, temperature=23.0,
        )
        reading.refresh_from_db()

        # Pipeline must have run (CELERY_TASK_ALWAYS_EAGER=True)
        assert reading.is_processed is True
        assert reading.ec_25_us_per_cm is not None

        # NPKPrediction must exist
        assert hasattr(reading, "npk_prediction")
        pred = reading.npk_prediction
        assert pred.predicted_nitrogen_mg_per_kg > 0
        assert pred.predicted_phosphorus_mg_per_kg > 0
        assert pred.predicted_potassium_mg_per_kg > 0
        assert pred.baseline_used == baseline_lab_test

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_no_rain_forecast())
    def test_happy_path_healthy_soil_produces_no_action(
        self, mock_meteo, api_client, make_signed_payload, baseline_lab_test
    ):
        """
        baseline_lab_test has N=35, P=18, pH=6.2 — all above thresholds.
        moisture=50% — above 40% wilting threshold.
        Expected: NO_ACTION recommendation.
        """
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload, offset_secs=2001,
            ph=6.2, moisture=50.0, temperature=23.0,
        )
        reading.refresh_from_db()

        pred = reading.npk_prediction
        recs = list(pred.recommendations.values_list("intervention_type", flat=True))

        assert "NO_ACTION" in recs, (
            f"Expected NO_ACTION for healthy soil but got: {recs}"
        )
        assert "LIME_APPLICATION" not in recs
        assert "IRRIGATE" not in recs

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_no_rain_forecast())
    def test_open_meteo_was_called_with_correct_coordinates(
        self, mock_meteo, api_client, make_signed_payload, baseline_lab_test
    ):
        """
        Verify the task calls Open-Meteo with the GPS coordinates from the reading,
        not hardcoded values.
        """
        _ingest_and_get_reading(
            api_client, make_signed_payload, offset_secs=2002,
            latitude=-1.1018, longitude=37.0144,
        )
        mock_meteo.assert_called_once()
        call_args = mock_meteo.call_args
        lat, lon = call_args[0][0], call_args[0][1]
        assert abs(lat - (-1.1018)) < 0.001
        assert abs(lon - 37.0144) < 0.001


# ══════════════════════════════════════════════════════════════════════════════
# Rules engine: LIME_APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class TestLimeApplicationRule:
    """
    Rule: IF pH < 5.5 THEN LIME_APPLICATION.
    Also verifies CAN is NOT triggered (blocked by low pH).
    """

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_no_rain_forecast())
    def test_low_ph_triggers_lime_application(
        self, mock_meteo, api_client, make_signed_payload, baseline_acidic_soil
    ):
        """
        Payload with pH=4.8 (below 5.5 threshold) must produce LIME_APPLICATION.
        baseline_acidic_soil provides matching field + baseline.
        """
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload, offset_secs=3000,
            ph=4.8, moisture=50.0, temperature=23.0,
        )
        reading.refresh_from_db()

        recs = list(
            reading.npk_prediction.recommendations.values_list("intervention_type", flat=True)
        )
        assert "LIME_APPLICATION" in recs, (
            f"Expected LIME_APPLICATION for pH=4.8 but got: {recs}"
        )

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_no_rain_forecast())
    def test_low_ph_blocks_can_application(
        self, mock_meteo, api_client, make_signed_payload, baseline_acidic_soil
    ):
        """
        CAN Top-Dressing rule requires pH > 5.5.
        Even though N is low (baseline_acidic_soil has N=15), CAN must NOT appear
        when pH < 5.5 — applying N fertiliser to acidic soil is agronomically wrong.
        """
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload, offset_secs=3001,
            ph=4.8, moisture=50.0, temperature=23.0,
        )
        reading.refresh_from_db()

        recs = list(
            reading.npk_prediction.recommendations.values_list("intervention_type", flat=True)
        )
        assert "CAN_TOP_DRESS" not in recs, (
            f"CAN_TOP_DRESS must be blocked when pH=4.8, got: {recs}"
        )

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_no_rain_forecast())
    def test_lime_recommendation_includes_quantity(
        self, mock_meteo, api_client, make_signed_payload, baseline_acidic_soil
    ):
        """
        Lime recommendation must include a calculated quantity (kg/acre).
        Quantity = (6.0 - pH) × 800. For pH=4.8: (6.0-4.8)×800 = 960 kg/acre.
        """
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload, offset_secs=3002,
            ph=4.8, moisture=50.0, temperature=23.0,
        )
        reading.refresh_from_db()

        lime_rec = reading.npk_prediction.recommendations.get(
            intervention_type="LIME_APPLICATION"
        )
        expected_qty = (6.0 - 4.8) * 800  # = 960.0
        assert lime_rec.quantity_kg_per_acre is not None
        assert abs(lime_rec.quantity_kg_per_acre - expected_qty) < 1.0

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_no_rain_forecast())
    def test_borderline_ph_just_above_threshold_no_lime(
        self, mock_meteo, api_client, make_signed_payload, baseline_lab_test
    ):
        """
        pH=5.6 is just above the 5.5 threshold — LIME_APPLICATION must NOT trigger.
        """
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload, offset_secs=3003,
            ph=5.6, moisture=50.0, temperature=23.0,
        )
        reading.refresh_from_db()

        recs = list(
            reading.npk_prediction.recommendations.values_list("intervention_type", flat=True)
        )
        assert "LIME_APPLICATION" not in recs


# ══════════════════════════════════════════════════════════════════════════════
# Rules engine: CAN_TOP_DRESS
# ══════════════════════════════════════════════════════════════════════════════

class TestCANTopDressRule:
    """
    Rule: IF N < 20 AND rain_24h < 15mm AND pH > 5.5 THEN CAN_TOP_DRESS.
    """

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_no_rain_forecast())
    def test_low_nitrogen_triggers_can(
        self, mock_meteo, api_client, make_signed_payload, baseline_nitrogen_deficient
    ):
        """
        baseline_nitrogen_deficient: N=12 (below 20), pH=6.5 (above 5.5).
        0mm rain forecast → CAN is safe to apply.
        """
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload, offset_secs=4000,
            ph=6.5, moisture=50.0, temperature=23.0,
        )
        reading.refresh_from_db()

        recs = list(
            reading.npk_prediction.recommendations.values_list("intervention_type", flat=True)
        )
        assert "CAN_TOP_DRESS" in recs, (
            f"Expected CAN_TOP_DRESS for N=12, pH=6.5, 0mm rain but got: {recs}"
        )

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_heavy_rain_forecast())
    def test_heavy_rain_suppresses_can(
        self, mock_meteo, api_client, make_signed_payload, baseline_nitrogen_deficient
    ):
        """
        Heavy rain (20mm) must suppress CAN application to prevent nutrient leaching.
        Even though N is low and pH is fine, CAN_TOP_DRESS must NOT appear.
        """
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload, offset_secs=4001,
            ph=6.5, moisture=50.0, temperature=23.0,
        )
        reading.refresh_from_db()

        recs = list(
            reading.npk_prediction.recommendations.values_list("intervention_type", flat=True)
        )
        assert "CAN_TOP_DRESS" not in recs, (
            f"CAN must be suppressed when 20mm rain forecast but got: {recs}"
        )

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_light_rain_forecast())
    def test_light_rain_14mm_allows_can(
        self, mock_meteo, api_client, make_signed_payload, baseline_nitrogen_deficient
    ):
        """
        14mm rain is below the 15mm CAN suppression threshold.
        CAN_TOP_DRESS must still trigger (safe to apply before light rain).
        """
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload, offset_secs=4002,
            ph=6.5, moisture=50.0, temperature=23.0,
        )
        reading.refresh_from_db()

        recs = list(
            reading.npk_prediction.recommendations.values_list("intervention_type", flat=True)
        )
        assert "CAN_TOP_DRESS" in recs, (
            f"Expected CAN for 14mm rain (below 15mm threshold) but got: {recs}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Rules engine: IRRIGATE
# ══════════════════════════════════════════════════════════════════════════════

class TestIrrigateRule:
    """
    Rule: IF moisture < 40% AND rain_24h < 5mm THEN IRRIGATE.
    """

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_no_rain_forecast())
    def test_low_moisture_no_rain_triggers_irrigation(
        self, mock_meteo, api_client, make_signed_payload, baseline_lab_test
    ):
        """
        moisture=30% (below 40%), rain=0mm → IRRIGATE must trigger.
        """
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload, offset_secs=5000,
            ph=6.2, moisture=30.0, temperature=23.0,
        )
        reading.refresh_from_db()

        recs = list(
            reading.npk_prediction.recommendations.values_list("intervention_type", flat=True)
        )
        assert "IRRIGATE" in recs, (
            f"Expected IRRIGATE for moisture=30%, 0mm rain but got: {recs}"
        )

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_no_rain_forecast())
    def test_irrigation_relay_flag_is_set(
        self, mock_meteo, api_client, make_signed_payload, baseline_lab_test
    ):
        """
        When irrigation triggers, the relay_triggered flag on the recommendation
        must be True (this is what the firmware polls to switch the pump).
        """
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload, offset_secs=5001,
            ph=6.2, moisture=30.0, temperature=23.0,
        )
        reading.refresh_from_db()

        irrigate_rec = reading.npk_prediction.recommendations.get(
            intervention_type="IRRIGATE"
        )
        assert irrigate_rec.relay_triggered is True

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value={"rain_next_24h_mm": 10.0, "rain_next_7d_mm": 25.0})
    def test_rain_forecast_suppresses_irrigation(
        self, mock_meteo, api_client, make_signed_payload, baseline_lab_test
    ):
        """
        moisture=30% but 10mm rain is forecast (above 5mm threshold).
        Irrigation must be SUPPRESSED to conserve water and prevent waterlogging.
        """
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload, offset_secs=5002,
            ph=6.2, moisture=30.0, temperature=23.0,
        )
        reading.refresh_from_db()

        recs = list(
            reading.npk_prediction.recommendations.values_list("intervention_type", flat=True)
        )
        assert "IRRIGATE" not in recs, (
            f"IRRIGATE must be suppressed when 10mm rain forecast but got: {recs}"
        )

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_no_rain_forecast())
    def test_adequate_moisture_no_irrigation(
        self, mock_meteo, api_client, make_signed_payload, baseline_lab_test
    ):
        """
        moisture=55% (above 40%) — irrigation must NOT trigger even with no rain.
        """
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload, offset_secs=5003,
            ph=6.2, moisture=55.0, temperature=23.0,
        )
        reading.refresh_from_db()

        recs = list(
            reading.npk_prediction.recommendations.values_list("intervention_type", flat=True)
        )
        assert "IRRIGATE" not in recs


# ══════════════════════════════════════════════════════════════════════════════
# Field resolution (PostGIS spatial query)
# ══════════════════════════════════════════════════════════════════════════════

class TestFieldResolution:
    """
    The Celery task uses PostGIS boundary__contains= to resolve which field
    a GPS reading belongs to.  These tests verify that spatial matching works.
    """

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_no_rain_forecast())
    def test_gps_inside_boundary_resolves_field(
        self, mock_meteo, api_client, make_signed_payload, baseline_lab_test, field
    ):
        """
        The payload GPS coords (-1.1018, 37.0144) are inside field.boundary.
        After processing, reading.field must be set to the field fixture.

        WHY test this explicitly?
          If the PostGIS spatial query fails (wrong SRID, geography vs geometry
          mismatch), the field stays NULL and no NPKPrediction is created —
          silently degrading accuracy without an error.
        """
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload, offset_secs=6000,
            latitude=-1.1018, longitude=37.0144,
        )
        reading.refresh_from_db()

        assert reading.field is not None, (
            "field should be resolved via PostGIS spatial query"
        )
        assert reading.field.id == field.id

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_no_rain_forecast())
    def test_gps_outside_all_boundaries_field_is_null(
        self, mock_meteo, api_client, make_signed_payload, baseline_lab_test
    ):
        """
        GPS coordinates far outside all registered field boundaries.
        reading.field must remain NULL — no crash, graceful degradation.
        The task continues (EC₂₅ computed, weather fetched) but NPK prediction
        is skipped because no baseline can be found without a field.
        """
        # Nairobi CBD — far from our Thika test field
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload, offset_secs=6001,
            latitude=-1.2921, longitude=36.8219,
        )
        reading.refresh_from_db()

        assert reading.field is None
        # No NPKPrediction should exist (baseline can't be found without field)
        assert not hasattr(reading, "npk_prediction") or \
               NPKPrediction.objects.filter(telemetry_reading=reading).count() == 0


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline completeness
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineCompleteness:
    """
    Verify the pipeline marks readings as processed and populates all
    derived fields correctly.
    """

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_no_rain_forecast())
    def test_is_processed_set_to_true_after_pipeline(
        self, mock_meteo, api_client, make_signed_payload, baseline_lab_test
    ):
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload, offset_secs=7000,
        )
        reading.refresh_from_db()
        assert reading.is_processed is True

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_no_rain_forecast())
    def test_delta_ec_computed_relative_to_baseline(
        self, mock_meteo, api_client, make_signed_payload, baseline_lab_test
    ):
        """
        delta_ec_from_baseline = EC₂₅ − baseline.ec_us_per_cm_at_test_date (280.0).
        We send ec_raw=310, T=25 → EC₂₅=310.0 → Δ = 310.0 - 280.0 = +30.0
        Positive Δ means ions are HIGHER than at baseline (soil improving or
        fertiliser recently applied).
        """
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload, offset_secs=7001,
            ec_raw=310.0, temperature=25.0,
        )
        reading.refresh_from_db()

        # EC₂₅ = 310 (no correction at 25°C)
        assert abs(reading.ec_25_us_per_cm - 310.0) < 0.5
        # Δ = 310 - 280 = +30
        expected_delta = 310.0 - baseline_lab_test.ec_us_per_cm_at_test_date
        assert abs(reading.delta_ec_from_baseline - expected_delta) < 0.5

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_no_rain_forecast())
    def test_exactly_one_npk_prediction_per_reading(
        self, mock_meteo, api_client, make_signed_payload, baseline_lab_test
    ):
        """
        The OneToOneField from NPKPrediction to DailyIoTTelemetry must ensure
        exactly one prediction per reading — no duplicates from task retries.
        """
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload, offset_secs=7002,
        )
        reading.refresh_from_db()

        prediction_count = NPKPrediction.objects.filter(
            telemetry_reading=reading
        ).count()
        assert prediction_count == 1

    @pytest.mark.django_db
    @patch(MOCK_PATH, return_value=_no_rain_forecast())
    def test_at_least_one_recommendation_always_created(
        self, mock_meteo, api_client, make_signed_payload, baseline_lab_test
    ):
        """
        The rules engine always creates at least one AgronomicRecommendation —
        even for healthy soil (it creates NO_ACTION).  A reading with zero
        recommendations would silently fail to deliver advice.
        """
        reading = _ingest_and_get_reading(
            api_client, make_signed_payload, offset_secs=7003,
        )
        reading.refresh_from_db()

        rec_count = AgronomicRecommendation.objects.filter(
            npk_prediction__telemetry_reading=reading
        ).count()
        assert rec_count >= 1, "Rules engine must always produce at least one recommendation"
