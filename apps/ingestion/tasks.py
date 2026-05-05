"""
apps/ingestion/tasks.py  (v2 — XGBoost inference wired in)
============================================================
CHANGES FROM v1:
  Step 5 (_engineer_features): Added the three physics-informed decay
  estimate features that the v2 XGBoost models require:
    n_decay_estimate = N₀ × exp(-k_N_mean × days)
    p_decay_estimate = P₀ × exp(-k_P_mean × days)
    k_decay_estimate = K₀ × exp(-k_K_mean × days)
  These are computed from baseline.nitrogen_mg_per_kg etc.

  Step 6 (_run_ml_inference): Replaced the physics decay stub with a
  call to ml.predictor.predict_npk(). The predictor loads the three
  trained XGBoost models once at startup and reuses them. If model
  files are missing it falls back to the decay stub automatically.

  Step 1 (field resolution): PostGIS geography→geometry cast preserved.
"""

import logging
import requests
from datetime import timedelta

from celery import shared_task
from django.conf import settings
import numpy as np

logger = logging.getLogger(__name__)

try:
    from ml.constants import MODEL_VERSION, K_N_MEAN, K_P_MEAN, K_K_MEAN
except ImportError:
    MODEL_VERSION = "v2.0-xgb-physics-anchored"
    K_N_MEAN, K_P_MEAN, K_K_MEAN = 0.008, 0.003, 0.006


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

    logger.info("Task: processing reading_id=%s device=%s", reading_id, reading.device_id)

    # ------------------------------------------------------------------
    # STEP 1 — Field resolution (PostGIS geography→geometry cast fix)
    # ------------------------------------------------------------------
    if reading.field is None:
        from django.contrib.gis.geos import Point as GeosPoint
        geom_point = GeosPoint(reading.location.x, reading.location.y, srid=4326)
        matched_field = Field.objects.filter(boundary__contains=geom_point).first()
        if matched_field:
            reading.field = matched_field
            logger.info("Task: resolved field '%s'", matched_field.name)
        else:
            logger.warning(
                "Task: GPS (%.6f, %.6f) not inside any field boundary.",
                reading.location.y, reading.location.x,
            )

    # ------------------------------------------------------------------
    # STEP 2 — EC₂₅ Temperature Compensation
    # ------------------------------------------------------------------
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
            field=reading.field, is_active=True,
        ).first()

    if baseline:
        reading.delta_ec_from_baseline = round(ec_25 - baseline.ec_us_per_cm_at_test_date, 2)
    else:
        logger.warning("Task: no active baseline for field. Δ-EC₂₅ skipped.")

    # ------------------------------------------------------------------
    # STEP 4 — Open-Meteo weather forecast
    # ------------------------------------------------------------------
    weather_data         = _fetch_open_meteo(reading.location.y, reading.location.x)
    forecast_rain_24h_mm = weather_data.get("rain_next_24h_mm", None)

    # ------------------------------------------------------------------
    # STEP 5 — Feature engineering (13 features)
    # ------------------------------------------------------------------
    features = _engineer_features(reading, baseline)

    # ------------------------------------------------------------------
    # STEP 6 — XGBoost ML inference + rules engine
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
                model_version                  = npk_predicted.get("model_version", MODEL_VERSION),
            )
            logger.info(
                "Task: NPK N=%.1f P=%.1f K=%.1f conf=%.2f",
                npk_predicted["N"], npk_predicted["P"], npk_predicted["K"],
                npk_predicted.get("confidence", 0),
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


def _fetch_open_meteo(latitude: float, longitude: float) -> dict:
    params = {
        "latitude": latitude, "longitude": longitude,
        "daily": "precipitation_sum", "timezone": "Africa/Nairobi", "forecast_days": 7,
    }
    try:
        r = requests.get(settings.OPEN_METEO_BASE_URL, params=params, timeout=10)
        r.raise_for_status()
        daily = r.json().get("daily", {}).get("precipitation_sum", [])
        return {"rain_next_24h_mm": daily[0] if daily else 0.0,
                "rain_next_7d_mm":  sum(daily) if daily else 0.0}
    except (requests.RequestException, KeyError, IndexError) as exc:
        logger.warning("Open-Meteo failed: %s — defaulting to 0mm", exc)
        return {"rain_next_24h_mm": 0.0, "rain_next_7d_mm": 0.0}


def _engineer_features(reading, baseline) -> dict | None:
    """
    Build the 13-feature vector matching FEATURE_COLS in ml/constants.py.

    The three physics-informed decay estimates (n_decay_estimate,
    p_decay_estimate, k_decay_estimate) are the most important features
    in the v2 XGBoost models. They encode the KALRO baseline anchor
    explicitly so the model can distinguish farms with different starting
    N₀/P₀/K₀ values — without these, cross-farm generalisation fails.
    """
    from apps.telemetry.models import DailyIoTTelemetry
    import pandas as pd

    if not baseline:
        return None

    days = (reading.time.date() - baseline.test_date).days
    cutoff = reading.time - timedelta(days=14)

    recent = DailyIoTTelemetry.objects.filter(
        device_id=reading.device_id,
        time__gte=cutoff, time__lte=reading.time, is_processed=True,
    ).values("time", "ec_25_us_per_cm", "moisture_percent").order_by("time")

    ec_25 = reading.ec_25_us_per_cm or 0.0
    delta_ec = reading.delta_ec_from_baseline or 0.0
    ec0 = baseline.ec_us_per_cm_at_test_date

    if not recent:
        ec_7d = ec_14d = ec_25
        moist_7d = reading.moisture_percent
        ec_d7d14 = 0.0
    else:
        df = pd.DataFrame(list(recent))
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        cutoff_7d  = df.index.max() - pd.Timedelta("7D")
        cutoff_14d = df.index.max() - pd.Timedelta("14D")
        ec_7d    = float(df.loc[df.index >= cutoff_7d,  "ec_25_us_per_cm"].mean())
        ec_14d   = float(df.loc[df.index >= cutoff_14d, "ec_25_us_per_cm"].mean())
        moist_7d = float(df.loc[df.index >= cutoff_7d,  "moisture_percent"].mean())
        ec_d7d14 = float(ec_7d - ec_14d)

    return {
        # ── Sensor proxy features ──────────────────────────────────────────────
        "days_since_baseline":   days,
        "ec_25":                 ec_25,
        "delta_ec":              delta_ec,
        "ph":                    reading.ph,
        "moisture_7d_avg":       moist_7d,
        "ec_7d_avg":             ec_7d,
        "ec_14d_avg":            ec_14d,
        "ec_delta_7d_14d":       ec_d7d14,
        "baseline_ec_us_per_cm": float(ec0),
        "ec_depletion_pct":      float(delta_ec / ec0 * 100) if ec0 > 0 else 0.0,
        # ── Physics-informed decay estimates (v2 KEY features) ─────────────────
        # Computed from KALRO baseline values + literature decay constants.
        # Source: Smaling et al. 1993, Kenyan highland nutrient balance.
        "n_decay_estimate":      float(baseline.nitrogen_mg_per_kg   * np.exp(-K_N_MEAN * days)),
        "p_decay_estimate":      float(baseline.phosphorus_mg_per_kg * np.exp(-K_P_MEAN * days)),
        "k_decay_estimate":      float(baseline.potassium_mg_per_kg  * np.exp(-K_K_MEAN * days)),
    }


def _run_ml_inference(features: dict, baseline) -> dict | None:
    """Call ml.predictor.predict_npk(). Falls back to decay stub on any failure."""
    try:
        from ml.predictor import predict_npk
        return predict_npk(features, baseline)
    except ImportError as exc:
        logger.error("Task: ml.predictor import failed: %s", exc)
        return _decay_stub_fallback(features, baseline)
    except Exception as exc:
        logger.error("Task: ML inference error: %s", exc, exc_info=True)
        return _decay_stub_fallback(features, baseline)


def _decay_stub_fallback(features: dict, baseline) -> dict:
    days  = features.get("days_since_baseline", 0)
    delta = features.get("delta_ec", 0.0)
    f     = max(0.5, 1.0 + delta / 500.0)
    return {
        "N":             round(max(0, baseline.nitrogen_mg_per_kg   * np.exp(-K_N_MEAN * days) * f), 2),
        "P":             round(max(0, baseline.phosphorus_mg_per_kg * np.exp(-K_P_MEAN * days) * f), 2),
        "K":             round(max(0, baseline.potassium_mg_per_kg  * np.exp(-K_K_MEAN * days) * f), 2),
        "confidence":    0.50,
        "model_version": "v0.1-decay-stub-fallback",
    }


def _run_rules_engine(prediction_record, reading, baseline, forecast_rain_24h_mm):
    from apps.agronomics.models import AgronomicRecommendation
    N = prediction_record.predicted_nitrogen_mg_per_kg
    P = prediction_record.predicted_phosphorus_mg_per_kg
    K = prediction_record.predicted_potassium_mg_per_kg
    ph = reading.ph; moisture = reading.moisture_percent
    rain_24h = forecast_rain_24h_mm or 0.0; n_recs = 0

    if ph < 5.5:
        lime_kg = (6.0 - ph) * 800
        AgronomicRecommendation.objects.create(
            npk_prediction=prediction_record, intervention_type="LIME_APPLICATION",
            message=(f"CRITICAL: Soil pH is {ph:.1f}. Apply {lime_kg:.0f} kg agricultural lime per acre."),
            quantity_kg_per_acre=lime_kg, forecast_rain_24h_mm=rain_24h, relay_triggered=False)
        n_recs += 1

    if N < 20 and rain_24h < 15 and ph > 5.5:
        can_kg = (max(0, 30 - N) / 0.26) * 0.1
        AgronomicRecommendation.objects.create(
            npk_prediction=prediction_record, intervention_type="CAN_TOP_DRESS",
            message=(f"LOW NITROGEN: N={N:.1f} mg/kg. Apply {can_kg:.1f} kg CAN per acre. Rain={rain_24h:.1f}mm — safe."),
            quantity_kg_per_acre=can_kg, forecast_rain_24h_mm=rain_24h, relay_triggered=False)
        n_recs += 1

    if P < 10 and ph >= 5.5:
        dap_kg = max(0, (10 - P) / 0.46) * 0.08
        AgronomicRecommendation.objects.create(
            npk_prediction=prediction_record, intervention_type="DAP_BASAL",
            message=(f"LOW PHOSPHORUS: P={P:.1f} mg/kg. Apply {dap_kg:.1f} kg DAP per acre."),
            quantity_kg_per_acre=dap_kg, forecast_rain_24h_mm=rain_24h, relay_triggered=False)
        n_recs += 1

    if N < 30 or P < 15:
        AgronomicRecommendation.objects.create(
            npk_prediction=prediction_record, intervention_type="MANURE_BASAL",
            message="Apply 2 tonnes farmyard manure per acre as basal application.",
            quantity_kg_per_acre=2000, forecast_rain_24h_mm=rain_24h, relay_triggered=False)
        n_recs += 1

    if moisture < 40 and rain_24h < 5:
        AgronomicRecommendation.objects.create(
            npk_prediction=prediction_record, intervention_type="IRRIGATE",
            message=(f"IRRIGATION TRIGGERED: moisture={moisture:.1f}%, rain={rain_24h:.1f}mm. Pump relay activated."),
            quantity_kg_per_acre=None, forecast_rain_24h_mm=rain_24h, relay_triggered=True)
        n_recs += 1
        logger.info("Rules engine: IRRIGATION relay for device %s", reading.device_id)

    if n_recs == 0:
        AgronomicRecommendation.objects.create(
            npk_prediction=prediction_record, intervention_type="NO_ACTION",
            message=(f"All parameters optimal. N:{N:.1f} P:{P:.1f} K:{K:.1f} mg/kg | pH:{ph:.1f} | Moisture:{moisture:.1f}% | Rain:{rain_24h:.1f}mm"),
            forecast_rain_24h_mm=rain_24h, relay_triggered=False)

    logger.info("Rules engine: %d recommendation(s) for reading_id=%s", n_recs, reading.id)