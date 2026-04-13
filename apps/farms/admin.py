from django.contrib import admin
from .models import Farmer, Farm, Field

@admin.register(Farmer)
class FarmerAdmin(admin.ModelAdmin):
    list_display = ["user", "location_county", "phone_number", "created_at"]
    search_fields = ["user__username", "user__first_name", "location_county"]

@admin.register(Farm)
class FarmAdmin(admin.ModelAdmin):
    list_display = ["name", "farmer", "centroid_latitude", "centroid_longitude", "created_at"]
    search_fields = ["name", "farmer__user__username"]

@admin.register(Field)
class FieldAdmin(admin.ModelAdmin):
    list_display = ["name", "farm", "fodder_type", "area_hectares", "created_at"]
    list_filter = ["fodder_type"]
    search_fields = ["name", "farm__name"]