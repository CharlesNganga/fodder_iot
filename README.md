# Fodder IoT Backend — JKUAT ENE212-0065/2020
# Precision Agriculture IoT & ML System for Kenyan Dairy Farmers

## Prerequisites
- Ubuntu 22.04+ / Debian 12+
- Python 3.10+
- PostgreSQL 14+ with PostGIS and TimescaleDB extensions
- Redis 7+

---

## 1. Database Setup

```bash
# Install PostgreSQL + PostGIS
sudo apt install postgresql postgresql-contrib postgis

# Install TimescaleDB (follow official docs for your distro)
# https://docs.timescale.com/self-hosted/latest/install/

sudo -u postgres psql << 'EOF'
CREATE USER fodder_user WITH PASSWORD 'change_this_password';
CREATE DATABASE fodder_db OWNER fodder_user;
\c fodder_db
CREATE EXTENSION postgis;
CREATE EXTENSION timescaledb;
GRANT ALL PRIVILEGES ON DATABASE fodder_db TO fodder_user;
EOF
```

---

## 2. Python Environment

```bash
cd fodder_iot/
python3 -m venv venv
source venv/bin/activate

pip install \
    django==5.0.* \
    djangorestframework==3.15.* \
    django-environ==0.11.* \
    psycopg2-binary==2.9.* \
    django-timescaledb==0.2.* \
    celery==5.3.* \
    redis==5.0.* \
    numpy==1.26.* \
    scipy==1.12.* \
    pandas==2.2.* \
    scikit-learn==1.4.* \
    xgboost==2.0.* \
    requests==2.31.*
```

---

## 3. Environment Variables

```bash
cp .env.example .env
# Edit .env with your actual credentials
```

---

## 4. Django Initialisation

```bash
source venv/bin/activate

python manage.py migrate
python manage.py createsuperuser

# Convert DailyIoTTelemetry into a TimescaleDB hypertable (run once after migrate)
python manage.py shell < scripts/create_hypertable.py
```

---

## 5. Run Services

```bash
# Terminal 1 — Django dev server
python manage.py runserver 0.0.0.0:8000

# Terminal 2 — Celery worker
celery -A config worker --loglevel=info

# Terminal 3 — Redis (if not running as a service)
redis-server
```

---

## Project Structure

```
fodder_iot/
├── config/                  # Django project config (settings, urls, celery)
├── apps/
│   ├── ingestion/           # ESP32 payload receiver + HMAC middleware
│   ├── farms/               # Farm & field geofencing models
│   ├── telemetry/           # TimescaleDB hypertable + IoT readings
│   └── agronomics/          # Baseline lab tests, NPK prediction, rules engine
├── scripts/                 # One-time DB setup scripts
└── manage.py
```
