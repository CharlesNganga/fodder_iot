from django.contrib import admin
from .models import DailyIoTTelemetry

@admin.register(DailyIoTTelemetry)
class DailyIoTTelemetryAdmin(admin.ModelAdmin):
    list_display = ["time", "device_id", "ph", "ec_raw_us_per_cm", "ec_25_us_per_cm",
                    "moisture_percent", "temperature_celsius", "is_processed", "field"]
    list_filter = ["is_processed", "device_id"]
    search_fields = ["device_id"]
    readonly_fields = ["ec_25_us_per_cm", "delta_ec_from_baseline", "hmac_signature"]
    ordering = ["-time"]