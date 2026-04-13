"""
config/settings_test.py
========================
Test settings that override production settings.py.

Key overrides:
  - CELERY_TASK_ALWAYS_EAGER=True  — tasks run synchronously in-process,
    no worker needed, no Redis required for task dispatch during tests.
  - CELERY_TASK_EAGER_PROPAGATES=True — exceptions in tasks surface
    immediately in tests rather than being swallowed silently.
  - ESP32_HMAC_SECRET — fixed known value so test HMAC calculation is
    deterministic and independent of the developer's .env file.
  - DATABASES ENGINE — stays as PostGIS (TimescaleDB hypertable is used
    in tests; the test DB reuses the same hypertable setup).
"""

from config.settings import *   # noqa: F401, F403 — intentional wildcard import

# ── Fixed HMAC secret for tests ───────────────────────────────────────────────
# Using a known constant means test HMAC calculations don't depend on the
# developer's .env.  Never use this value in production.
ESP32_HMAC_SECRET = "test-hmac-secret-fodder-iot-1234567890abcdef"

# ── Celery: run tasks synchronously, no broker needed ─────────────────────────
CELERY_TASK_ALWAYS_EAGER    = True   # .delay() runs inline, no Redis/RabbitMQ
CELERY_TASK_EAGER_PROPAGATES = True  # exceptions propagate to test assertions

# ── Suppress logging noise during tests ───────────────────────────────────────
LOGGING = {
    "version": 1,
    "disable_existing_loggers": True,
    "handlers": {"null": {"class": "logging.NullHandler"}},
    "root": {"handlers": ["null"]},
}

# ── Use a separate test database name ─────────────────────────────────────────
# pytest-django appends "test_" automatically, but we set it explicitly
# to make it obvious which DB the test suite targets.
DATABASES["default"]["TEST"] = {  # noqa: F405
    "NAME": "test_fodder_db",
}
