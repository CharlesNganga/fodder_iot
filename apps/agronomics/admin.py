from django.contrib import admin
from .models import CompositeBaselineLabTest, NPKPrediction, AgronomicRecommendation

@admin.register(CompositeBaselineLabTest)
class BaselineLabTestAdmin(admin.ModelAdmin):
    list_display = ["field", "season_label", "test_date", "nitrogen_mg_per_kg",
                    "phosphorus_mg_per_kg", "potassium_mg_per_kg", "ph_at_test_date", "is_active"]
    list_filter = ["is_active", "field__fodder_type"]
    search_fields = ["field__name", "season_label"]
    # This is the model farmers fill in at season start — make it easy to use
    fieldsets = [
        ("Field & Season", {"fields": ["field", "season_label", "test_date", "is_active"]}),
        ("NPK Values (mg/kg)", {"fields": ["nitrogen_mg_per_kg", "phosphorus_mg_per_kg", "potassium_mg_per_kg"]}),
        ("Supporting Chemistry", {"fields": ["ph_at_test_date", "ec_us_per_cm_at_test_date", "organic_carbon_percent"]}),
    ]

@admin.register(NPKPrediction)
class NPKPredictionAdmin(admin.ModelAdmin):
    list_display = ["predicted_at", "predicted_nitrogen_mg_per_kg", "predicted_phosphorus_mg_per_kg",
                    "predicted_potassium_mg_per_kg", "confidence_score", "model_version"]
    list_filter = ["model_version"]
    readonly_fields = ["predicted_at", "telemetry_reading", "baseline_used"]

@admin.register(AgronomicRecommendation)
class AgronomicRecommendationAdmin(admin.ModelAdmin):
    list_display = ["created_at", "intervention_type", "quantity_kg_per_acre",
                    "forecast_rain_24h_mm", "relay_triggered"]
    list_filter = ["intervention_type", "relay_triggered"]
    readonly_fields = ["created_at"]