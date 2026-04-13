import json
import logging
from datetime import datetime

from django.contrib.gis.geos import Point
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from apps.ingestion.middleware import validate_esp32_hmac
from apps.telemetry.models import DailyIoTTelemetry
from apps.ingestion.tasks import process_telemetry_reading

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {
    "device_id", "timestamp", "latitude", "longitude",
    "ph", "ec_raw", "moisture", "temperature",
}


@csrf_exempt
@validate_esp32_hmac
def ingest_telemetry(request):
    """
    Receives a signed JSON payload from the ESP32.
    Only accepts POST. Returns 201 immediately; all processing is async via Celery.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    # ── 1. Parse JSON ─────────────────────────────────────────────────────────
    try:
        payload = json.loads(request.body)
    except (json.JSONDecodeError, ValueError) as exc:
        return JsonResponse({"error": "Invalid JSON", "detail": str(exc)}, status=400)

    # ── 2. Required field check ───────────────────────────────────────────────
    missing = REQUIRED_FIELDS - set(payload.keys())
    if missing:
        return JsonResponse({"error": "Missing fields", "missing": list(missing)}, status=400)

    # ── 3. Type coercion ──────────────────────────────────────────────────────
    try:
        device_id    = str(payload["device_id"]).strip()
        latitude     = float(payload["latitude"])
        longitude    = float(payload["longitude"])
        ph           = float(payload["ph"])
        ec_raw       = float(payload["ec_raw"])
        moisture     = float(payload["moisture"])
        temperature  = float(payload["temperature"])
        battery_v    = float(payload["battery_v"]) if "battery_v" in payload else None
        reading_time = datetime.fromisoformat(
            payload["timestamp"].replace("Z", "+00:00")
        )
    except (TypeError, ValueError, KeyError) as exc:
        return JsonResponse({"error": "Field type error", "detail": str(exc)}, status=400)

    # ── 4. Sensor range validation ────────────────────────────────────────────
    errors = []
    if not (-90 <= latitude <= 90):     errors.append("latitude out of range")
    if not (-180 <= longitude <= 180):  errors.append("longitude out of range")
    if not (0 <= ph <= 14):             errors.append(f"pH {ph} invalid")
    if not (0 <= moisture <= 100):      errors.append(f"moisture {moisture}% invalid")
    if ec_raw < 0:                      errors.append(f"EC {ec_raw} negative")
    if not (-10 <= temperature <= 80):  errors.append(f"temperature {temperature}°C invalid")
    if errors:
        return JsonResponse({"error": "Sensor range violation", "violations": errors}, status=400)

    # ── 5. Idempotency check ──────────────────────────────────────────────────
    # Network drop safety to avoid duplicates
    if DailyIoTTelemetry.objects.filter(device_id=device_id, time=reading_time).exists():
        return JsonResponse({"status": "duplicate", "message": "Already recorded"}, status=200)

    # ── 6. Persist raw reading ────────────────────────────────────────────────
    reading = DailyIoTTelemetry.objects.create(
        time                = reading_time,
        device_id           = device_id,
        location            = Point(longitude, latitude, srid=4326),
        ph                  = ph,
        ec_raw_us_per_cm    = ec_raw,
        moisture_percent    = moisture,
        temperature_celsius = temperature,
        battery_voltage     = battery_v,
        hmac_signature      = request.headers.get("X-Esp32-Signature", ""),
        is_processed        = False,
    )

    # ── 7. Enqueue async pipeline ─────────────────────────────────────────────
    # transition from synchronous execution (blocking the user/device) to asynchronous execution (background processing)
    celery_queued = True
    try: #prevents code to crash if Redis server is down
        process_telemetry_reading.delay(reading.id)
    except Exception as exc:
        celery_queued = False
        logger.error(
            "Celery broker unreachable for reading_id=%s: %s. "
            "Reading saved — will need manual requeue or broker restart.",
            reading.id, exc,
        )

    logger.info("Ingestion: reading id=%s from device=%s saved.", reading.id, device_id)

    response_body = {
        "status": "accepted",
        "reading_id": reading.id,
        "message": "Telemetry received. Processing queued." if celery_queued
                   else "Telemetry saved. WARNING: Celery broker unreachable — processing delayed.",
    }
    return JsonResponse(response_body, status=201)