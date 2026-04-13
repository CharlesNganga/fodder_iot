"""
apps/agronomics/models.py  (FIXED)
===================================
FIX: Added db_constraint=False to NPKPrediction.telemetry_reading.
     TimescaleDB does not support SQL-level FK constraints referencing
     a hypertable. db_constraint=False tells Django not to create the
     SQL FK in the database, while the ORM relationship remains fully
     navigable (reading.npk_prediction, prediction.telemetry_reading, etc).
     Referential integrity is enforced at the application layer in tasks.py.
"""

from django.db import models
from apps.farms.models import Field


class CompositeBaselineLabTest(models.Model):
    field = models.ForeignKey(Field, on_delete=models.CASCADE, related_name="baseline_tests")
    season_label = models.CharField(max_length=100, help_text="e.g., 'Long Rains 2025'")
    test_date = models.DateField(help_text="Date the KALRO lab sample was taken")
    nitrogen_mg_per_kg   = models.FloatField(help_text="Total N in mg/kg from KALRO report")
    phosphorus_mg_per_kg = models.FloatField(help_text="Total P in mg/kg from KALRO report")
    potassium_mg_per_kg  = models.FloatField(help_text="Total K in mg/kg from KALRO report")
    ph_at_test_date = models.FloatField(help_text="Soil pH from KALRO lab")
    ec_us_per_cm_at_test_date = models.FloatField(help_text="EC in µS/cm from KALRO lab")
    organic_carbon_percent = models.FloatField(null=True, blank=True, help_text="Organic Carbon % (optional)")
    is_active = models.BooleanField(default=True, help_text="Only ONE active baseline per field at a time.")
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        if self.is_active:
            CompositeBaselineLabTest.objects.filter(
                field=self.field, is_active=True,
            ).exclude(pk=self.pk).update(is_active=False)
        super().save(*args, **kwargs)

    def __str__(self):
        return (f"{self.field.name} — {self.season_label} "
                f"(N:{self.nitrogen_mg_per_kg} P:{self.phosphorus_mg_per_kg} K:{self.potassium_mg_per_kg} mg/kg)")

    class Meta:
        ordering = ["-test_date"]
        verbose_name = "KALRO Composite Baseline Lab Test"


class NPKPrediction(models.Model):
    # FIX: db_constraint=False — Django ORM relationship works normally,
    # but NO SQL FK is created in PostgreSQL. Required for TimescaleDB hypertables.
    telemetry_reading = models.OneToOneField(
        "telemetry.DailyIoTTelemetry",
        on_delete=models.CASCADE,
        related_name="npk_prediction",
        db_constraint=False,  # ← THE FIX
        help_text="TimescaleDB does not support FK constraints on hypertables. Enforced in application layer.",
    )
    baseline_used = models.ForeignKey(
        CompositeBaselineLabTest,
        on_delete=models.SET_NULL,
        null=True,
        help_text="Which KALRO baseline anchor this prediction was derived from",
    )
    predicted_nitrogen_mg_per_kg   = models.FloatField()
    predicted_phosphorus_mg_per_kg = models.FloatField()
    predicted_potassium_mg_per_kg  = models.FloatField()
    confidence_score = models.FloatField(null=True, blank=True, help_text="Model R² score (0–1)")
    model_version = models.CharField(max_length=50, default="v1.0")
    predicted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return (f"Prediction @ {self.predicted_at.strftime('%Y-%m-%d %H:%M')} — "
                f"N:{self.predicted_nitrogen_mg_per_kg:.1f} "
                f"P:{self.predicted_phosphorus_mg_per_kg:.1f} "
                f"K:{self.predicted_potassium_mg_per_kg:.1f} mg/kg")

    class Meta:
        ordering = ["-predicted_at"]


class AgronomicRecommendation(models.Model):
    INTERVENTION_TYPES = [
        ("LIME_APPLICATION", "Apply Agricultural Lime (pH Correction)"),
        ("CAN_TOP_DRESS",    "Apply CAN Top-Dressing (Nitrogen Boost)"),
        ("DAP_BASAL",        "Apply DAP Basal (Phosphorus Correction)"),
        ("MANURE_BASAL",     "Apply Farmyard Manure (Organic Basal)"),
        ("IRRIGATE",         "Activate Irrigation Pump"),
        ("NO_ACTION",        "No Intervention Required"),
    ]
    npk_prediction = models.ForeignKey(NPKPrediction, on_delete=models.CASCADE, related_name="recommendations")
    intervention_type = models.CharField(max_length=30, choices=INTERVENTION_TYPES)
    message = models.TextField()
    quantity_kg_per_acre = models.FloatField(null=True, blank=True)
    forecast_rain_24h_mm = models.FloatField(null=True, blank=True)
    relay_triggered = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.get_intervention_type_display()} — {self.created_at.strftime('%Y-%m-%d')}"

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Agronomic Recommendation"