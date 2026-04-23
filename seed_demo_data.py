import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.contrib.auth.models import User
from apps.farms.models import Farmer, Farm, Field
from apps.agronomics.models import CompositeBaselineLabTest
from django.contrib.gis.geos import Polygon
from datetime import date, timedelta

print("🌱 Seeding live database for demo...")

# 1. Create a dummy user and farmer
user, _ = User.objects.get_or_create(username="demo_farmer", defaults={"first_name": "Charles"})
farmer, _ = Farmer.objects.get_or_create(user=user, defaults={"phone_number": "+254700000000", "location_county": "Kiambu"})

# 2. Create the Farm
farm, _ = Farm.objects.get_or_create(
    farmer=farmer, name="Demo Farm Thika", 
    defaults={"centroid_latitude": -1.1018, "centroid_longitude": 37.0144}
)

# 3. Create the Field with a PostGIS Polygon that perfectly wraps the Demo's GPS coordinates
# Coords wrap around: Lat -1.1018, Lon 37.0144
poly = Polygon(((37.0140, -1.1022), (37.0150, -1.1022), (37.0150, -1.1012), (37.0140, -1.1012), (37.0140, -1.1022)), srid=4326)
field, _ = Field.objects.get_or_create(
    farm=farm, name="North Block Demo", 
    defaults={"fodder_type": "napier", "boundary": poly}
)

# 4. Create the active KALRO Baseline Test (dated 14 days ago)
test_date = (date.today() - timedelta(days=14)).isoformat()
baseline, created = CompositeBaselineLabTest.objects.get_or_create(
    field=field, is_active=True,
    defaults={
        "season_label": "Demo Season",
        "test_date": test_date,
        "nitrogen_mg_per_kg": 35.0,
        "phosphorus_mg_per_kg": 18.0,
        "potassium_mg_per_kg": 150.0,
        "ph_at_test_date": 6.2,
        "ec_us_per_cm_at_test_date": 280.0
    }
)

print("✅ Database successfully seeded! Celery can now resolve the GPS points.")