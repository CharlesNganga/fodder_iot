"""
apps/ingestion/tasks.py  (FIXED)
=================================
FIX in STEP 1 — PostGIS geography vs geometry type mismatch.

DailyIoTTelemetry.location is PointField(geography=True) — stored as
PostGIS GEOGRAPHY type. Field.boundary is PolygonField(srid=4326) —
stored as PostGIS GEOMETRY type.

PostGIS cannot directly evaluate:
    ST_Contains(geometry_polygon, geography_point)

The __contains lookup raises an operator error or silently returns nothing.

Fix: cast reading.location to a geometry Point before the spatial query:
    from django.contrib.gis.geos import Point as GEOSPoint
    geom_point = Point(reading.location.x, reading.location.y, srid=4326)
    Field.objects.filter(boundary__contains=geom_point)

This works because:
  - reading.location.x/.y extract the raw WGS84 coordinates
  - A new Point(..., srid=4326) WITHOUT geography=True is a geometry type
  - boundary (geometry) __contains geometry_point → valid PostGIS operation
"""

import logging
import requests
from datetime import timedelta

from celery import shared_task
from django.conf import settings
import numpy as np

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    name="ingestion.process_telemetry_reading",
)
def process_telemetry_reading(self, reading_id: int):
    from apps.telemetry.models import DailyIoTTelemetry
    from apps.farms.models import Field
    from apps.agronomics.models import (
        CompositeBaselineLabTest, NPKPrediction, AgronomicRecommendation
    )

    try:
        reading = DailyIoTTelemetry.objects.select_related("field").get(id=reading_id)
    except DailyIoTTelemetry.DoesNotExist:
        logger.error("Task: reading_id=%s not found. Aborting.", reading_id)
        return

    if reading.is_processed:
        logger.info("Task: reading_id=%s already processed. Skipping.", reading_id)
        return

    logger.info("Task: processing reading_id=%s from device=%s", reading_id, reading.device_id)

    # ------------------------------------------------------------------
    # STEP 1 — Resolve field via PostGIS point-in-polygon
    # ------------------------------------------------------------------
    if reading.field is None:
        # FIX: Cast the geography Point to geometry Point before the spatial
        # query. reading.location is geography=True; boundary is geometry.
        # PostGIS cannot mix types in ST_Contains without an explicit cast.
        from django.contrib.gis.geos import Point as GeosPoint
        geom_point = GeosPoint(
            reading.location.x,   # longitude
            reading.location.y,   # latitude
            srid=4326,            # geometry type (no geography=True)
        )
        matched_field = Field.objects.filter(
            boundary__contains=geom_point
        ).first()

        if matched_field:
            reading.field = matched_field
            logger.info("Task: resolved reading to field '%s'", matched_field.name)
        else:
            logger.warning(
                "Task: GPS point (%.6f, %.6f) not inside any registered field boundary.",
                reading.location.y, reading.location.x,
            )

    # ------------------------------------------------------------------
    # STEP 2 — EC₂₅ Temperature Compensation
    # ------------------------------------------------------------------
    # Formula: EC₂₅ = EC_raw / (1 + 0.019 × (T − 25))
    # Reference: Rhoades et al., 1989
    T      = reading.temperature_celsius
    ec_raw = reading.ec_raw_us_per_cm
    ec_25  = ec_raw / (1 + 0.019 * (T - 25))
    reading.ec_25_us_per_cm = round(ec_25, 2)

    # ------------------------------------------------------------------
    # STEP 3 — Δ-EC₂₅ from KALRO baseline
    # ------------------------------------------------------------------
    baseline = None
    if reading.field:
        baseline = CompositeBaselineLabTest.objects.filter(
            field=reading.field,
            is_active=True,
        ).first()

    if baseline:
        delta_ec = ec_25 - baseline.ec_us_per_cm_at_test_date
        reading.delta_ec_from_baseline = round(delta_ec, 2)
    else:
        logger.warning("Task: no active baseline for field. Δ-EC₂₅ skipped.")

    # ------------------------------------------------------------------
    # STEP 4 — Open-Meteo weather forecast
    # ------------------------------------------------------------------
    weather_data         = _fetch_open_meteo(reading.location.y, reading.location.x)
    forecast_rain_24h_mm = weather_data.get("rain_next_24h_mm", None)

    # ------------------------------------------------------------------
    # STEP 5 — Feature engineering
    # ------------------------------------------------------------------
    features = _engineer_features(reading, baseline)

    # ------------------------------------------------------------------
    # STEP 6 — ML inference + rules engine
    # ------------------------------------------------------------------
    if baseline and features:
        npk_predicted = _run_ml_inference(features, baseline)
        if npk_predicted:
            prediction_record = NPKPrediction.objects.create(
                telemetry_reading              = reading,
                baseline_used                  = baseline,
                predicted_nitrogen_mg_per_kg   = npk_predicted["N"],
                predicted_phosphorus_mg_per_kg = npk_predicted["P"],
                predicted_potassium_mg_per_kg  = npk_predicted["K"],
                confidence_score               = npk_predicted.get("confidence"),
                model_version                  = npk_predicted.get("model_version", "v1.0"),
            )
            _run_rules_engine(
                prediction_record    = prediction_record,
                reading              = reading,
                baseline             = baseline,
                forecast_rain_24h_mm = forecast_rain_24h_mm,
            )

    # ------------------------------------------------------------------
    # STEP 7 — Mark as processed
    # ------------------------------------------------------------------
    reading.is_processed = True
    reading.save(update_fields=[
        "field", "ec_25_us_per_cm", "delta_ec_from_baseline", "is_processed"
    ])
    logger.info("Task: reading_id=%s pipeline complete.", reading_id)


# ── Open-Meteo ────────────────────────────────────────────────────────────────

def _fetch_open_meteo(latitude: float, longitude: float) -> dict:
    base_url = settings.OPEN_METEO_BASE_URL
    params = {
        "latitude":      latitude,
        "longitude":     longitude,
        "daily":         "precipitation_sum",
        "timezone":      "Africa/Nairobi",
        "forecast_days": 7,
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data         = response.json()
        daily_precip = data.get("daily", {}).get("precipitation_sum", [])
        return {
            "rain_next_24h_mm": daily_precip[0] if daily_precip else 0.0,
            "rain_next_7d_mm":  sum(daily_precip) if daily_precip else 0.0,
        }
    except (requests.RequestException, KeyError, IndexError) as exc:
        logger.warning("Open-Meteo fetch failed: %s — defaulting to 0mm rain", exc)
        return {"rain_next_24h_mm": 0.0, "rain_next_7d_mm": 0.0}


# ── Feature engineering ───────────────────────────────────────────────────────

def _engineer_features(reading, baseline) -> dict | None:
    from apps.telemetry.models import DailyIoTTelemetry
    import pandas as pd

    if not baseline:
        return None

    days_since_baseline = (reading.time.date() - baseline.test_date).days
    cutoff              = reading.time - timedelta(days=14)

    recent_readings = DailyIoTTelemetry.objects.filter(
        device_id=reading.device_id,
        time__gte=cutoff,
        time__lte=reading.time,
        is_processed=True,
    ).values("time", "ec_25_us_per_cm", "moisture_percent").order_by("time")

    if not recent_readings:
        return {
            "days_since_baseline": days_since_baseline,
            "ec_25":               reading.ec_25_us_per_cm or 0,
            "delta_ec":            reading.delta_ec_from_baseline or 0,
            "ph":                  reading.ph,
            "moisture_7d_avg":     reading.moisture_percent,
            "ec_7d_avg":           reading.ec_25_us_per_cm or 0,
            "ec_14d_avg":          reading.ec_25_us_per_cm or 0,
            "ec_delta_7d_14d":     0,
        }

    df = pd.DataFrame(list(recent_readings))
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()

    ec_7d_avg       = df["ec_25_us_per_cm"].last("7D").mean()
    ec_14d_avg      = df["ec_25_us_per_cm"].last("14D").mean()
    moisture_7d_avg = df["moisture_percent"].last("7D").mean()

    return {
        "days_since_baseline": days_since_baseline,
        "ec_25":               reading.ec_25_us_per_cm,
        "delta_ec":            reading.delta_ec_from_baseline or 0,
        "ph":                  reading.ph,
        "moisture_7d_avg":     moisture_7d_avg,
        "ec_7d_avg":           ec_7d_avg,
        "ec_14d_avg":          ec_14d_avg,
        "ec_delta_7d_14d":     ec_7d_avg - ec_14d_avg,
    }


# ── ML inference stub ─────────────────────────────────────────────────────────

def _run_ml_inference(features: dict, baseline) -> dict | None:
    days  = features["days_since_baseline"]
    delta = features["delta_ec"]

    k_N = 0.008
    k_P = 0.003
    k_K = 0.006

    depletion_factor = max(0.5, 1 + (delta / 500))

    N_pred = baseline.nitrogen_mg_per_kg   * np.exp(-k_N * days) * depletion_factor
    P_pred = baseline.phosphorus_mg_per_kg * np.exp(-k_P * days) * depletion_factor
    K_pred = baseline.potassium_mg_per_kg  * np.exp(-k_K * days) * depletion_factor

    return {
        "N": round(max(0, N_pred), 2),
        "P": round(max(0, P_pred), 2),
        "K": round(max(0, K_pred), 2),
        "confidence":    0.72,
        "model_version": "v0.1-decay-stub",
    }


# ── Agronomic Rules Engine ────────────────────────────────────────────────────

def _run_rules_engine(prediction_record, reading, baseline, forecast_rain_24h_mm):
    from apps.agronomics.models import AgronomicRecommendation

    N        = prediction_record.predicted_nitrogen_mg_per_kg
    P        = prediction_record.predicted_phosphorus_mg_per_kg
    K        = prediction_record.predicted_potassium_mg_per_kg
    ph       = reading.ph
    moisture = reading.moisture_percent
    rain_24h = forecast_rain_24h_mm or 0.0

    recommendations_created = 0

    # RULE 1 — pH Correction
    if ph < 5.5:
        lime_kg_per_acre = (6.0 - ph) * 800
        AgronomicRecommendation.objects.create(
            npk_prediction       = prediction_record,
            intervention_type    = "LIME_APPLICATION",
            message              = (
                f"CRITICAL: Soil pH is {ph:.1f}. "
                f"Apply {lime_kg_per_acre:.0f} kg of agricultural lime per acre."
            ),
            quantity_kg_per_acre = lime_kg_per_acre,
            forecast_rain_24h_mm = rain_24h,
            relay_triggered      = False,
        )
        recommendations_created += 1

    # RULE 2 — CAN Top-Dressing
    if N < 20 and rain_24h < 15 and ph > 5.5:
        n_deficit        = max(0, 30 - N)
        can_kg_per_acre  = (n_deficit / 0.26) * 0.1
        AgronomicRecommendation.objects.create(
            npk_prediction       = prediction_record,
            intervention_type    = "CAN_TOP_DRESS",
            message              = (
                f"LOW NITROGEN: N={N:.1f} mg/kg. "
                f"Apply {can_kg_per_acre:.1f} kg CAN per acre."
            ),
            quantity_kg_per_acre = can_kg_per_acre,
            forecast_rain_24h_mm = rain_24h,
            relay_triggered      = False,
        )
        recommendations_created += 1

    # RULE 3 — DAP
    if P < 10 and ph >= 5.5:
        dap_kg_per_acre = max(0, (10 - P) / 0.46) * 0.08
        AgronomicRecommendation.objects.create(
            npk_prediction       = prediction_record,
            intervention_type    = "DAP_BASAL",
            message              = f"LOW PHOSPHORUS: P={P:.1f} mg/kg. Apply {dap_kg_per_acre:.1f} kg DAP per acre.",
            quantity_kg_per_acre = dap_kg_per_acre,
            forecast_rain_24h_mm = rain_24h,
            relay_triggered      = False,
        )
        recommendations_created += 1

    # RULE 4 — Manure
    if N < 30 or P < 15:
        AgronomicRecommendation.objects.create(
            npk_prediction       = prediction_record,
            intervention_type    = "MANURE_BASAL",
            message              = "Apply 2 tonnes farmyard manure per acre as basal application.",
            quantity_kg_per_acre = 2000,
            forecast_rain_24h_mm = rain_24h,
            relay_triggered      = False,
        )
        recommendations_created += 1

    # RULE 5 — Irrigation
    if moisture < 40 and rain_24h < 5:
        AgronomicRecommendation.objects.create(
            npk_prediction       = prediction_record,
            intervention_type    = "IRRIGATE",
            message              = (
                f"IRRIGATION TRIGGERED: moisture={moisture:.1f}%, rain_24h={rain_24h:.1f}mm."
            ),
            quantity_kg_per_acre = None,
            forecast_rain_24h_mm = rain_24h,
            relay_triggered      = True,
        )
        recommendations_created += 1
        logger.info("Rules engine: IRRIGATION triggered for device %s", reading.device_id)

    # NO_ACTION fallback
    if recommendations_created == 0:
        AgronomicRecommendation.objects.create(
            npk_prediction       = prediction_record,
            intervention_type    = "NO_ACTION",
            message              = (
                f"All parameters optimal. N:{N:.1f} P:{P:.1f} K:{K:.1f} mg/kg | "
                f"pH:{ph:.1f} | Moisture:{moisture:.1f}% | Rain:{rain_24h:.1f}mm"
            ),
            forecast_rain_24h_mm = rain_24h,
            relay_triggered      = False,
        )

    logger.info(
        "Rules engine: %d recommendation(s) for reading_id=%s",
        recommendations_created, reading.id,
    )