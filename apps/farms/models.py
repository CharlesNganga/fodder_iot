"""
apps/farms/models.py
====================
Geospatial models for farm-level management.

WHY PostGIS PolygonField?
  The farmer draws their field boundary once in the React Native app (tapping
  corners on Google Maps). We store that polygon so the IDW spatial interpolation
  worker can clip the generated heat-map PNG to the exact field boundary — ensuring
  recommendations never bleed into neighbouring plots.

Models:
  Farmer   — the user account linked to one or more farms
  Farm     — the top-level geographic entity (e.g., "Ng'ang'a Farm, Thika")
  Field    — a named plot within a farm with a PostGIS polygon boundary
"""

from django.contrib.auth.models import User
from django.contrib.gis.db import models as gis_models
from django.db import models


class Farmer(models.Model):
    """
    Extends Django's built-in User with farmer-specific profile data.
    One User ↔ One Farmer (OneToOne).
    """
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name="farmer_profile",
    )
    phone_number = models.CharField(
        max_length=20,
        blank=True,
        help_text="Safaricom/Airtel number for SMS alerts (e.g., +254712345678)",
    )
    location_county = models.CharField(
        max_length=100,
        blank=True,
        help_text="e.g., Kiambu, Nyandarua, Nakuru",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.get_full_name()} ({self.location_county})"

    class Meta:
        verbose_name = "Farmer Profile"


class Farm(models.Model):
    """
    A named farm owned by one Farmer.
    A single farmer may own multiple farms (e.g., home plot + leased plot).
    """
    farmer = models.ForeignKey(
        Farmer,
        on_delete=models.CASCADE,
        related_name="farms",
    )
    name = models.CharField(
        max_length=200,
        help_text="Human-readable farm name, e.g., 'Gatundu Upper Plot'",
    )
    # Centroid for quick weather API calls — avoids computing polygon centroid
    # every time we hit Open-Meteo.
    centroid_latitude  = models.DecimalField(max_digits=10, decimal_places=7)
    centroid_longitude = models.DecimalField(max_digits=10, decimal_places=7)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} — {self.farmer}"

    class Meta:
        ordering = ["name"]


class Field(gis_models.Model):
    """
    A named plot within a Farm.

    WHY gis_models.Model (not models.Model)?
      We inherit from django.contrib.gis.db.models.Model to unlock
      PostGIS-backed field types like PolygonField and PointField.

    boundary — PostGIS PolygonField (SRID 4326 = WGS84 / standard GPS coordinates).
      The farmer traces their field on the React Native Google Maps view.
      The app sends the polygon vertices as GeoJSON; Django deserialises it into
      a GEOS Polygon and stores it natively in PostGIS.

    This enables:
      1. field.boundary.area         → approximate acreage calculation
      2. ORM spatial query: Field.objects.filter(boundary__contains=point)
         → instantly find which field a GPS reading belongs to
      3. IDW worker clips heat-map to this polygon boundary
    """
    FODDER_CHOICES = [
        ("napier",  "Napier Grass (Pennisetum purpureum)"),
        ("rhodes",  "Rhodes Grass (Chloris gayana)"),
        ("mixed",   "Mixed Fodder"),
        ("other",   "Other"),
    ]

    farm = models.ForeignKey(
        Farm,
        on_delete=models.CASCADE,
        related_name="fields",
    )
    name = models.CharField(
        max_length=200,
        help_text="e.g., 'North Block' or 'Section A'",
    )
    fodder_type = models.CharField(
        max_length=20,
        choices=FODDER_CHOICES,
        default="napier",
        help_text="Primary crop grown — determines KALRO threshold values in the rules engine",
    )
    # PostGIS polygon — the geofence for this field.
    # srid=4326 matches WGS84 GPS coordinates from the NEO-6M module.
    boundary = gis_models.PolygonField(
        srid=4326,
        null=True,
        blank=True,
        help_text="GeoJSON polygon drawn by farmer in the mobile app",
    )
    area_hectares = models.DecimalField(
        max_digits=8,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="Calculated area in hectares (auto-populated on save)",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        """
        Auto-calculate area in hectares whenever the boundary polygon is saved.

        PostGIS returns area in square degrees (SRID 4326).  We transform to
        SRID 21037 (Arc 1960 / UTM Zone 37S — standard for East Africa) to get
        metres², then convert to hectares.
        """
        if self.boundary:
            try:
                from django.contrib.gis.geos import GEOSGeometry
                # Transform polygon to UTM Zone 37 (metres) then compute area.
                utm_boundary = self.boundary.transform(21037, clone=True)
                self.area_hectares = round(utm_boundary.area / 10_000, 4)
            except Exception:
                pass  # Non-fatal — area can be re-calculated later
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.name} @ {self.farm.name} ({self.area_hectares} ha)"

    class Meta:
        ordering = ["farm", "name"]
