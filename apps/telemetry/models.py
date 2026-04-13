"""
apps/telemetry/models.py
========================
Time-series storage for all IoT sensor readings from the ESP32 edge node.

WHY TimescaleDB?
  The system is designed to receive one reading every ~60 seconds per device.
  At 1,440 readings/day per farm, standard PostgreSQL B-tree indexes degrade
  significantly over a full growing season (180 days = ~260,000 rows per device).
  TimescaleDB automatically partitions the table into time-ordered "chunks"
  (default: 7-day intervals), ensuring O(1) ingestion and fast range queries
  regardless of total dataset size.

WHY PostGIS PointField?
  Every reading is geo-tagged by the NEO-6M GPS module.  Storing locations as
  PostGIS Points (not just lat/lon floats) enables:
    - PostGIS spatial queries: find all readings within a field boundary
    - Distance calculations for the IDW interpolation worker
    - Future support for multi-device farms

The EC₂₅ Temperature Compensation:
  Raw EC from the RS485 sensor is temperature-dependent (diurnal thermal drift
  causes ±20% variation between morning and afternoon readings).
  EC₂₅ standardises all readings to 25°C using the formula:
    EC₂₅ = EC_raw / (1 + 0.019 × (T − 25))
  This is stored as a computed field — calculated at ingestion time in
  the Celery worker, NOT in the Django serialiser, to keep HTTP response fast.
"""

from django.db import models
from django.contrib.gis.db import models as gis_models
from timescale.db.models.models import TimescaleModel
from apps.farms.models import Field


class DailyIoTTelemetry(TimescaleModel):
    """
    Primary hypertable for all ESP32 sensor readings.

    Inherits from TimescaleModel which:
      1. Adds a 'time' DateTimeField (the TimescaleDB partition key)
      2. Registers the model so `python manage.py migrate` can call
         create_hypertable() via the django-timescaledb app

    After the first `python manage.py migrate`, run:
      python manage.py shell < scripts/create_hypertable.py
    to convert this table into a TimescaleDB hypertable.

    Field naming conventions:
      - Raw sensor fields use the sensor's native units
      - Derived/computed fields are prefixed with an underscore
    """

    # -----------------------------------------------------------------------
    # TimescaleDB partition key — MUST be the first field and named 'time'
    # -----------------------------------------------------------------------
    # Inherited from TimescaleModel as: time = models.DateTimeField(...)
    # We override it here to add help_text and ensure UTC storage.
    # Note: TimescaleModel already defines 'time'; we add metadata below.

    # -----------------------------------------------------------------------
    # Spatial — GPS coordinates from NEO-6M module
    # -----------------------------------------------------------------------
    location = gis_models.PointField(
        srid=4326,
        geography=True,            # geography=True → calculations in metres (not degrees)
        help_text="WGS84 GPS coordinates from NEO-6M. geography=True for metre-accurate IDW.",
    )

    # -----------------------------------------------------------------------
    # Foreign keys
    # -----------------------------------------------------------------------
    field = models.ForeignKey(
        Field,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="telemetry_readings",
        help_text="Auto-populated by Celery worker via PostGIS point-in-polygon query",
    )
    # Device identifier — allows multi-device farms in future.
    device_id = models.CharField(
        max_length=64,
        db_index=True,
        help_text="ESP32 device MAC address or provisioned UUID",
    )

    # -----------------------------------------------------------------------
    # Raw sensor readings (direct from RS485 4-in-1 probe)
    # -----------------------------------------------------------------------
    moisture_percent = models.FloatField(
        help_text="Volumetric Water Content (VWC) in % (0–100)"
    )
    temperature_celsius = models.FloatField(
        help_text="Soil temperature in °C at probe depth"
    )
    ph = models.FloatField(
        help_text="Soil pH (0–14 scale)"
    )
    ec_raw_us_per_cm = models.FloatField(
        help_text="Raw Electrical Conductivity in µS/cm (NOT temperature-compensated)"
    )

    # -----------------------------------------------------------------------
    # Derived fields — computed by Celery worker after ingestion
    # -----------------------------------------------------------------------
    ec_25_us_per_cm = models.FloatField(
        null=True,
        blank=True,
        help_text=(
            "Temperature-compensated EC at 25°C. Formula: "
            "EC₂₅ = EC_raw / (1 + 0.019 × (T − 25)). "
            "Null until the async Celery worker processes this reading."
        ),
    )
    # Δ-EC₂₅: change in EC₂₅ from the baseline lab test EC value.
    # Negative values indicate ion depletion (nutrient loss).
    delta_ec_from_baseline = models.FloatField(
        null=True,
        blank=True,
        help_text="EC₂₅ minus the KALRO baseline EC. Tracks ion depletion over time.",
    )

    # -----------------------------------------------------------------------
    # Payload metadata
    # -----------------------------------------------------------------------
    # HMAC signature from the ESP32 — stored for audit trail.
    hmac_signature = models.CharField(
        max_length=64,
        help_text="HMAC-SHA256 hex digest received from ESP32 (for audit)",
    )
    # Battery voltage — allows dashboard to warn farmer of low battery.
    battery_voltage = models.FloatField(
        null=True,
        blank=True,
        help_text="ESP32 battery voltage in V (from ADC pin) — for low-battery alerts",
    )
    # True once the Celery worker has finished processing this reading.
    is_processed = models.BooleanField(
        default=False,
        db_index=True,
        help_text="Set to True by Celery worker after EC₂₅, ML inference, and rules engine complete",
    )

    def __str__(self):
        return (
            f"[{self.time.strftime('%Y-%m-%d %H:%M')}] "
            f"Device:{self.device_id} | "
            f"pH:{self.ph} EC:{self.ec_raw_us_per_cm}µS | "
            f"Moisture:{self.moisture_percent}% T:{self.temperature_celsius}°C"
        )

    class Meta:
        # TimescaleDB requires the partition key (time) in all unique constraints
        # and indexes. Standard Django ordering by time is efficient due to chunking.
        ordering = ["-time"]
        verbose_name = "IoT Telemetry Reading"
        verbose_name_plural = "IoT Telemetry Readings"
        indexes = [
            # Composite index for the most common query pattern:
            # "give me all readings for device X in the last 14 days"
            models.Index(fields=["device_id", "-time"], name="idx_device_time"),
            # Index for the Celery worker's unprocessed-readings query
            models.Index(fields=["is_processed", "-time"], name="idx_unprocessed"),
        ]
