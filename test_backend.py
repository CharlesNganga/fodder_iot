"""
test_backend.py  (FIXED v2)
============================
Fixes applied vs v1:
  - Test 2: Check settings.DATABASES dict instead of non-existent settings.DATABASE_URL
  - Test 7: Read from settings.CELERY_BROKER_URL instead of settings.REDIS_URL
  - Test 10: Corrected EC25 expected values (formula was correct, expectations were wrong)
             EC25(300, 35°C) = 252.10   (not 255.32)
             EC25(300, 15°C) = 370.37   (not 363.64)
"""

import sys
import json
import hmac
import hashlib
import os
from datetime import datetime, timezone, timedelta
import traceback

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):      print(f"  {GREEN}✅ PASS{RESET} — {msg}")
def fail(msg):    print(f"  {RED}❌ FAIL{RESET} — {msg}"); FAILURES.append(msg)
def info(msg):    print(f"  {BLUE}ℹ️  {RESET} {msg}")
def section(t):   print(f"\n{BOLD}{YELLOW}{'═'*60}{RESET}\n{BOLD} {t}{RESET}\n{BOLD}{YELLOW}{'═'*60}{RESET}")

FAILURES = []

# ═══════════════════════════════════════════════════════════════════════
# TEST 1 — Python dependencies
# ═══════════════════════════════════════════════════════════════════════
section("TEST 1 — Python dependencies")

deps = ["django","rest_framework","environ","psycopg2","celery",
        "redis","numpy","scipy","pandas","sklearn","xgboost","requests","timescale"]
for dep in deps:
    try:
        __import__(dep); ok(dep)
    except ImportError:
        fail(f"{dep} not installed — pip install -r requirements or see README")

# ═══════════════════════════════════════════════════════════════════════
# TEST 2 — Django configuration
# ═══════════════════════════════════════════════════════════════════════
section("TEST 2 — Django configuration")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

try:
    import django; django.setup(); ok("Django setup() succeeded")
except Exception as e:
    fail(f"Django setup failed: {e}"); sys.exit(1)

from django.conf import settings

# Check SECRET_KEY
sk = getattr(settings, "SECRET_KEY", "")
if len(sk) >= 20:
    ok(f"settings.SECRET_KEY = {sk[:6]}...")
else:
    fail("settings.SECRET_KEY too short or missing")

# FIX: Check DATABASES dict (django-environ parses DATABASE_URL into this)
db = settings.DATABASES.get("default", {})
db_name = db.get("NAME", "")
db_host = db.get("HOST", "")
db_engine = db.get("ENGINE", "")
if "postgis" in db_engine and db_name:
    ok(f"settings.DATABASES: engine=postgis, db={db_name}, host={db_host or 'localhost'}")
else:
    fail(f"DATABASES misconfigured: engine={db_engine}, name={db_name}")

# Check HMAC secret
hmac_secret = getattr(settings, "ESP32_HMAC_SECRET", "")
if len(hmac_secret) >= 16 and hmac_secret not in ("REPLACE_ME_IN_DOT_ENV", ""):
    ok(f"settings.ESP32_HMAC_SECRET = {hmac_secret[:6]}...")
else:
    fail("settings.ESP32_HMAC_SECRET looks unset — check .env")

# Check Celery broker URL
broker = getattr(settings, "CELERY_BROKER_URL", "")
if "redis" in broker:
    ok(f"settings.CELERY_BROKER_URL = {broker}")
else:
    fail(f"CELERY_BROKER_URL not pointing to Redis: '{broker}'\n"
         f"    Add to .env: REDIS_URL=redis://localhost:6379/0\n"
         f"    settings.py reads it as: CELERY_BROKER_URL = env('REDIS_URL', ...)")

# ═══════════════════════════════════════════════════════════════════════
# TEST 3 — Database connection
# ═══════════════════════════════════════════════════════════════════════
section("TEST 3 — Database connection")

try:
    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SELECT version();")
        ver = cur.fetchone()[0]
        ok(f"PostgreSQL connected: {ver[:50]}...")
except Exception as e:
    fail(f"Database connection failed: {e}\n    Try: sudo systemctl start postgresql")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════
# TEST 4 — Extensions
# ═══════════════════════════════════════════════════════════════════════
section("TEST 4 — PostGIS + TimescaleDB extensions")

with connection.cursor() as cur:
    cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname IN ('postgis','timescaledb') ORDER BY extname;")
    exts = {r[0]: r[1] for r in cur.fetchall()}

for ext in ["postgis", "timescaledb"]:
    if ext in exts:
        ok(f"{ext} v{exts[ext]}")
    else:
        fail(f"{ext} missing — run: sudo -u postgres psql -d fodder_db -c 'CREATE EXTENSION {ext};'")

# ═══════════════════════════════════════════════════════════════════════
# TEST 5 — TimescaleDB hypertable
# ═══════════════════════════════════════════════════════════════════════
section("TEST 5 — TimescaleDB hypertable")

with connection.cursor() as cur:
    cur.execute("""
        SELECT hypertable_name, num_dimensions, compression_enabled
        FROM timescaledb_information.hypertables
        WHERE hypertable_name = 'telemetry_dailyiottelemetry';
    """)
    row = cur.fetchone()
    if row:
        ok(f"Hypertable: {row[0]} | dims={row[1]} | compression={row[2]}")
    else:
        fail("Hypertable NOT created — run: python manage.py shell < scripts/create_hypertable.py")

    cur.execute("""
        SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint
        WHERE conrelid = 'telemetry_dailyiottelemetry'::regclass AND contype = 'p';
    """)
    pk = cur.fetchone()
    if pk and 'time' in pk[1]:
        ok(f"Composite PK: {pk[1]}")
    else:
        fail(f"PK wrong — expected (id, time), got: {pk}")

# ═══════════════════════════════════════════════════════════════════════
# TEST 6 — Models
# ═══════════════════════════════════════════════════════════════════════
section("TEST 6 — Django models")

import importlib
for module_path, names in [
    ("apps.farms.models",      ["Farmer","Farm","Field"]),
    ("apps.telemetry.models",  ["DailyIoTTelemetry"]),
    ("apps.agronomics.models", ["CompositeBaselineLabTest","NPKPrediction","AgronomicRecommendation"]),
]:
    try:
        mod = importlib.import_module(module_path)
        for name in names:
            cls = getattr(mod, name)
            count = cls.objects.count()
            ok(f"{name} — {count} rows")
    except Exception as e:
        fail(f"{module_path}.{name}: {e}")

# ═══════════════════════════════════════════════════════════════════════
# TEST 7 — Redis
# ═══════════════════════════════════════════════════════════════════════
section("TEST 7 — Redis connection")

try:
    import redis
    # FIX: read from CELERY_BROKER_URL (that's what settings.py actually sets)
    broker_url = getattr(settings, "CELERY_BROKER_URL", "redis://localhost:6379/0")
    r = redis.from_url(broker_url)
    r.ping()
    ok(f"Redis ping OK at {broker_url}")
except Exception as e:
    fail(f"Redis connection failed: {e}\n    Run: sudo systemctl start redis-server")

# ═══════════════════════════════════════════════════════════════════════
# TEST 8 — HMAC
# ═══════════════════════════════════════════════════════════════════════
section("TEST 8 — HMAC signature logic")

try:
    secret  = settings.ESP32_HMAC_SECRET
    payload = json.dumps({"device_id": "TEST:001", "ph": 6.2}, separators=(',', ':'))
    sig1    = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    sig2    = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    if hmac.compare_digest(sig1, sig2):
        ok(f"HMAC works (sig={sig1[:16]}...)")
    else:
        fail("HMAC mismatch on identical data")
except Exception as e:
    fail(f"HMAC error: {e}")

# ═══════════════════════════════════════════════════════════════════════
# TEST 9 — Live HTTP endpoint
# ═══════════════════════════════════════════════════════════════════════
section("TEST 9 — Live HTTP ingest endpoint")
info("Requires: python manage.py runserver running in another terminal")

try:
    import requests as req

    BASE   = "http://127.0.0.1:8000"
    secret = settings.ESP32_HMAC_SECRET

    # 9a: No HMAC → 401
    try:
        r = req.post(f"{BASE}/api/ingest/", json={"ph": 6.2}, timeout=3)
        if r.status_code == 401: ok("No HMAC → 401 ✓")
        else: fail(f"No HMAC → expected 401, got {r.status_code}")
    except req.exceptions.ConnectionError:
        fail("Cannot connect — is 'python manage.py runserver' running?")
        raise SystemExit(0)

    # 9b: Wrong HMAC → 401
    r = req.post(f"{BASE}/api/ingest/",
                 data=b'{"ph":6.2}',
                 headers={"Content-Type":"application/json","X-Esp32-Signature":"deadbeef"*8},
                 timeout=3)
    if r.status_code == 401: ok("Wrong HMAC → 401 ✓")
    else: fail(f"Wrong HMAC → expected 401, got {r.status_code}")

    # 9c: Valid payload → 201
    data = {
        "device_id":   "TEST:ESP32:001",
        "timestamp":   datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "latitude":    -1.1018, "longitude": 37.0144,
        "ph": 5.8, "ec_raw": 310.2, "moisture": 42.1,
        "temperature": 23.5, "battery_v": 4.0,
    }
    body = json.dumps(data, separators=(',', ':'))
    sig  = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
    hdrs = {"Content-Type": "application/json", "X-Esp32-Signature": sig}

    r = req.post(f"{BASE}/api/ingest/", data=body, headers=hdrs, timeout=5)
    if r.status_code == 201:
        resp = r.json()
        ok(f"Valid payload → 201 (reading_id={resp.get('reading_id')}, celery={'queued' if 'WARNING' not in resp.get('message','') else 'BROKER DOWN — fix .env'})")
        if "WARNING" in resp.get("message", ""):
            info("⚠️  Celery broker unreachable — add REDIS_URL=redis://localhost:6379/0 to .env")
    elif r.status_code == 200 and r.json().get("status") == "duplicate":
        ok("Valid payload → 200 duplicate (idempotency works ✓)")
    else:
        fail(f"Valid payload → expected 201, got {r.status_code}: {r.text[:200]}")

    # 9d: Duplicate → 200
    r2 = req.post(f"{BASE}/api/ingest/", data=body, headers=hdrs, timeout=5)
    if r2.status_code == 200 and r2.json().get("status") == "duplicate":
        ok("Duplicate → 200 idempotent ACK ✓")
    elif r2.status_code == 201:
        fail("Duplicate accepted as new reading — idempotency broken")
    else:
        fail(f"Duplicate test → unexpected {r2.status_code}")

    # 9e: Bad pH → 400
    bad = dict(data)
    bad["ph"] = 99.9
    bad["timestamp"] = (datetime.now(timezone.utc) + timedelta(seconds=5)).strftime('%Y-%m-%dT%H:%M:%SZ')
    bs  = json.dumps(bad, separators=(',', ':'))
    bsig= hmac.new(secret.encode(), bs.encode(), hashlib.sha256).hexdigest()
    r3  = req.post(f"{BASE}/api/ingest/", data=bs,
                   headers={"Content-Type":"application/json","X-Esp32-Signature":bsig}, timeout=3)
    if r3.status_code == 400: ok("Bad sensor value (pH=99.9) → 400 ✓")
    else: fail(f"Bad sensor → expected 400, got {r3.status_code}")

except SystemExit:
    pass
except Exception as e:
    fail(f"HTTP test error: {e}")

# ═══════════════════════════════════════════════════════════════════════
# TEST 10 — EC₂₅ formula (FIXED expected values)
# ═══════════════════════════════════════════════════════════════════════
section("TEST 10 — EC₂₅ temperature compensation formula")

def ec25(ec_raw, T):
    return ec_raw / (1 + 0.019 * (T - 25))

# FIX: correct expected values computed from the actual formula
# EC25(300, 35°C) = 300 / (1 + 0.019*10) = 300 / 1.19 = 252.10
# EC25(300, 15°C) = 300 / (1 + 0.019*-10) = 300 / 0.81 = 370.37
cases = [
    (300.0, 25.0, 300.00, "at 25°C — no correction"),
    (300.0, 35.0, 252.10, "hot soil — EC₂₅ lower than raw"),
    (300.0, 15.0, 370.37, "cold morning — EC₂₅ higher than raw"),
]
for ec_raw, T, expected, desc in cases:
    result = ec25(ec_raw, T)
    if abs(result - expected) < 0.5:
        ok(f"EC₂₅({ec_raw}µS, {T}°C) = {result:.2f}µS — {desc}")
    else:
        fail(f"EC₂₅ formula error: expected {expected}, got {result:.2f} — {desc}")

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{YELLOW}{'═'*60}{RESET}")
print(f"{BOLD} SUMMARY{RESET}")
print(f"{BOLD}{YELLOW}{'═'*60}{RESET}")

if not FAILURES:
    print(f"\n{GREEN}{BOLD}🎉 ALL TESTS PASSED — Backend fully operational!{RESET}")
    print(f"\n  Next: Phase 2 — synthetic dataset + XGBoost model training")
else:
    print(f"\n{RED}{BOLD}❌ {len(FAILURES)} test(s) failed:{RESET}")
    for i, f in enumerate(FAILURES, 1):
        print(f"  {i}. {f}")
    print(f"\n{YELLOW}Quick fixes:{RESET}")
    print(f"  • Celery/Redis 500 → add REDIS_URL=redis://localhost:6379/0 to .env")
    print(f"  • Then restart runserver")