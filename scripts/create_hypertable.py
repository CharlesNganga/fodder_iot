from django.db import connection

TABLE     = "telemetry_dailyiottelemetry"
NPK_TABLE = "agronomics_npkprediction"
FK        = "agronomics_npkpredic_telemetry_reading_id_2ae466a3_fk_telemetry"

print("=== Hypertable Final Setup ===\n")

with connection.cursor() as cur:

    cur.execute("SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint WHERE conrelid = 'telemetry_dailyiottelemetry'::regclass AND contype = 'p';")
    print(f"[DEBUG] PK: {cur.fetchall()}")
    cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'telemetry_dailyiottelemetry';")
    print(f"[DEBUG] Indexes: {cur.fetchall()}")
    cur.execute("SELECT count(*) FROM timescaledb_information.hypertables WHERE hypertable_name = 'telemetry_dailyiottelemetry';")
    already = bool(cur.fetchone()[0])
    print(f"[DEBUG] Already hypertable: {already}\n")

    print("Step 1: Removing indexes that block hypertable creation...")
    cur.execute(f"DROP INDEX IF EXISTS {TABLE}_id_unique;")
    print("✅ Step 1 done.")

    print("Step 2: Ensuring FK constraint is removed...")
    cur.execute(f"ALTER TABLE {NPK_TABLE} DROP CONSTRAINT IF EXISTS {FK};")
    print("✅ Step 2 done.")

    if not already:
        print("Step 3: Creating TimescaleDB hypertable...")
        cur.execute(f"""
            SELECT create_hypertable(
                '{TABLE}', 'time',
                if_not_exists => TRUE,
                migrate_data   => TRUE,
                chunk_time_interval => INTERVAL '7 days'
            );
        """)
        print("✅ Step 3: Hypertable created with 7-day chunk interval.")
    else:
        print("Step 3: Already a hypertable — skipping.")

    print("Step 4: Adding compression policy...")
    cur.execute(f"""
        ALTER TABLE {TABLE} SET (
            timescaledb.compress,
            timescaledb.compress_orderby   = 'time DESC',
            timescaledb.compress_segmentby = 'device_id'
        );
    """)
    cur.execute(f"SELECT add_compression_policy('{TABLE}', INTERVAL '30 days', if_not_exists => TRUE);")
    print("✅ Step 4 done.")

    print("Step 5: Adding unique index on id for efficient ORM lookups...")
    cur.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {TABLE}_id_unique ON {TABLE} (id);")
    print("✅ Step 5 done.")

    print("\n=== VERIFICATION ===")
    cur.execute("SELECT hypertable_name, num_dimensions, compression_enabled FROM timescaledb_information.hypertables WHERE hypertable_name = 'telemetry_dailyiottelemetry';")
    print(f"Hypertable info: {cur.fetchall()}")
    cur.execute(f"SELECT indexname FROM pg_indexes WHERE tablename = '{TABLE}';")
    print(f"All indexes: {cur.fetchall()}")
    print("\n🎉 Done! Run: python manage.py migrate && python manage.py createsuperuser && python manage.py runserver")