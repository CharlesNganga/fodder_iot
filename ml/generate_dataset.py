"""
ml/generate_dataset.py
=======================
Synthetic IoT Telemetry Dataset Generator
JKUAT ENE212-0065/2020 — Precision Agriculture IoT & ML System

PURPOSE:
  No real sensor data exists yet (the ESP32 hardware is not yet deployed).
  This script generates a physically realistic synthetic dataset that the
  XGBoost models in Phase 2b are trained on.

DESIGN PRINCIPLES:
  1. Physics-informed labels — N/P/K ground truth is derived from validated
     exponential decay constants sourced from published Kenyan highland soil
     depletion data (Smaling et al. 1993; Springer Nutrient Cycling 2020).
  2. Sensor-realistic features — EC_raw, pH, moisture, and temperature are
     simulated with the same noise characteristics as the RS485 4-in-1 probe.
  3. Pipeline-aligned columns — feature column names match the keys produced
     by _engineer_features() in tasks.py EXACTLY (enforced via ml/constants.py).
  4. No data leakage — decay constants k_N, k_P, k_K are used ONLY to generate
     labels and are never written to the CSV. The model must infer NPK from
     sensor proxies alone.

DATASET SCALE:
  5 farms × 2 seasons × 180 days × 24 readings/day = 43,200 raw rows
  Minus first 14 days per season (rolling window warmup) = ~41,000 clean rows

DECAY PARAMETERS (validated against literature):
  k_N = 0.008/day  — Nitrogen: fast depletion via leaching + Napier uptake
  k_P = 0.003/day  — Phosphorus: slow depletion, binds to soil particles
  k_K = 0.006/day  — Potassium: intermediate, moderate leaching
  Sources: Smaling et al. (1993) Kenya highland nutrient balance;
           Springer Nutrient Cycling in Agroecosystems (2020) Embu/Kiboko data

USAGE:
  cd fodder_iot/
  python ml/generate_dataset.py

OUTPUT:
  ml/data/synthetic_dataset.csv  (~41,000 rows, 12 feature+label columns)
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# ── Ensure project root is on path when running standalone ───────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.constants import (
    FEATURE_COLS, TARGET_COLS, META_COLS,
    TARGET_N, TARGET_P, TARGET_K,
    DATASET_PATH,
)

# ── Reproducibility ───────────────────────────────────────────────────────────
RNG = np.random.default_rng(seed=42)   # fixed seed → reproducible dataset

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SIMULATION CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

# Dataset scale
N_FARMS         = 5
N_SEASONS       = 2      # per farm: one long rains + one short rains
DAYS_PER_SEASON = 180    # one full Napier growing cycle
READINGS_PER_DAY = 24   # one reading per hour (matches ESP32 firmware interval)
ROLLING_WARMUP  = 14    # days to discard at start of each season (NaN rolling window)

# Decay constants (mean values; each farm gets ±noise around these)
K_N_MEAN = 0.008    # Nitrogen: fastest depletion
K_P_MEAN = 0.003    # Phosphorus: slowest depletion (binds to soil particles)
K_K_MEAN = 0.006    # Potassium: intermediate

# Decay constant noise (farm-to-farm soil variability)
K_N_STD  = 0.001
K_P_STD  = 0.0005
K_K_STD  = 0.001

# Rainfall acceleration of nitrogen leaching
# Each mm of rain the previous day multiplies the effective N decay by this factor.
# Based on: leaching loss ≈ 1.5% of soil N per 10mm rain (Smaling et al. 1993).
RAIN_LEACH_FACTOR = 0.015

# EC composition weights: EC correlates with total dissolved ions
# N contributes most (50%), K next (30%), P least (20%)
EC_WEIGHT_N = 0.50
EC_WEIGHT_P = 0.20
EC_WEIGHT_K = 0.30

# Sensor noise standard deviations (realistic for RS485 4-in-1 probe)
EC_SENSOR_NOISE_STD   = 8.0    # µS/cm — manufacturer spec ±1% of 800µS = ±8µS
PH_SENSOR_NOISE_STD   = 0.05   # pH units — RS485 probe accuracy ±0.1
TEMP_SENSOR_NOISE_STD = 0.3    # °C — probe accuracy ±0.5°C
MOIST_SENSOR_NOISE_STD = 0.8   # % — probe accuracy ±2%


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FARM PARAMETER SAMPLER
# ═════════════════════════════════════════════════════════════════════════════

def sample_farm_params(farm_id: int, season_id: int) -> dict:
    """
    Sample the soil baseline parameters for one farm-season combination.

    Each farm gets a randomised but agronomically realistic KALRO-style
    starting point. The ranges are drawn from KALRO Information Bulletin
    (2022) typical values for Napier grass fields in central Kenya highlands.

    Returns a dict used as the T₀ anchor throughout the season simulation.
    """
    season_type = "long_rains" if season_id == 0 else "short_rains"

    params = {
        "farm_id":    farm_id,
        "season_id":  season_id,
        "season_type": season_type,

        # KALRO baseline lab values (the T₀ anchor)
        "N0": float(RNG.uniform(25.0, 60.0)),    # mg/kg — typical highland range
        "P0": float(RNG.uniform(8.0,  22.0)),    # mg/kg — low-medium P common
        "K0": float(RNG.uniform(80.0, 200.0)),   # mg/kg — more variable
        "pH0": float(RNG.uniform(5.0,  6.8)),    # covers acidic to near-neutral
        "EC0": float(RNG.uniform(150.0, 400.0)), # µS/cm — RS485 probe range

        # Farm-specific decay constants (variability around literature means)
        "k_N": float(np.clip(RNG.normal(K_N_MEAN, K_N_STD), 0.004, 0.014)),
        "k_P": float(np.clip(RNG.normal(K_P_MEAN, K_P_STD), 0.001, 0.006)),
        "k_K": float(np.clip(RNG.normal(K_K_MEAN, K_K_STD), 0.003, 0.010)),

        # Diurnal temperature cycle parameters (Kenyan highlands)
        "T_mean": float(RNG.uniform(18.0, 24.0)),  # °C — mean daily soil temp
        "T_amp":  float(RNG.uniform(3.0,  6.0)),   # °C — diurnal amplitude

        # Evapotranspiration rate (moisture loss per day without rain)
        "ET_rate": float(RNG.uniform(0.5, 1.2)),   # %/day

        # Starting moisture
        "moisture0": float(RNG.uniform(35.0, 65.0)), # %
    }
    return params


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DAILY STATE SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════

def simulate_daily_rainfall(day: int, season_type: str) -> float:
    """
    Simulate daily rainfall (mm) based on Kenyan seasonal patterns.

    Long rains (March–May, days 0–89): high probability, higher volume
    Short rains (Oct–Dec, days 90–179): medium probability, lower volume
    Dry transition (days in between): very low probability

    Returns mm of rainfall for the day (0.0 if no rain event).
    """
    if season_type == "long_rains":
        # Long rains: 60% chance of rain, 5–30mm
        if RNG.random() < 0.60:
            return float(RNG.uniform(5.0, 30.0))
    else:
        # Short rains: 40% chance of rain, 3–20mm
        if RNG.random() < 0.40:
            return float(RNG.uniform(3.0, 20.0))
    return 0.0


def compute_ph_modifier(pH: float) -> float:
    """
    pH correction factor for nutrient availability.

    Below pH 5.5: nutrients lock up (particularly N and P).
    At pH 5.5–7.0: full availability.
    This models the real agronomic effect that motivates the LIME_APPLICATION rule.

    The factor smoothly ramps from 0.6 (very acidic) to 1.0 (neutral).
    """
    if pH >= 5.5:
        return 1.0
    elif pH <= 4.0:
        return 0.6
    else:
        # Linear interpolation between pH 4.0 (factor 0.6) and pH 5.5 (factor 1.0)
        return 0.6 + (pH - 4.0) / (5.5 - 4.0) * 0.4


def simulate_season(params: dict) -> pd.DataFrame:
    """
    Simulate one full growing season (180 days × 24 readings = 4,320 rows).

    This is the core simulation loop. It generates:
      - Ground truth NPK labels (physics-informed decay)
      - Sensor proxy variables (EC_25, pH, moisture, temperature)
      - All features that _engineer_features() in tasks.py computes

    The simulation advances day-by-day, with 24 hourly readings per day.
    Rolling statistics (7-day, 14-day averages) are computed AFTER the
    full time-series is built, then the warmup rows are dropped.
    """
    rows = []

    # ── Initialise state variables ────────────────────────────────────────────
    N_current   = params["N0"]
    P_current   = params["P0"]
    K_current   = params["K0"]
    pH_current  = params["pH0"]
    EC0         = params["EC0"]
    moisture    = params["moisture0"]
    prev_rain   = 0.0    # yesterday's rainfall (drives N leaching modifier)

    k_N = params["k_N"]
    k_P = params["k_P"]
    k_K = params["k_K"]

    for day in range(DAYS_PER_SEASON):

        # ── Daily events ──────────────────────────────────────────────────────

        # 1. Simulate today's rainfall
        rain_today = simulate_daily_rainfall(day, params["season_type"])

        # 2. Update moisture
        #    Evapotranspiration removes moisture; rain replenishes it.
        #    Capped at 80% (field capacity) and 10% (wilting floor).
        rain_recharge = min(rain_today * 0.6, 80.0 - moisture)
        moisture = float(np.clip(
            moisture - params["ET_rate"] + rain_recharge,
            10.0, 80.0
        ))

        # 3. Compute pH drift (slow acidification over the season)
        #    Napier grass and continuous cropping acidify soil ~0.003 pH units/day.
        pH_current = float(np.clip(
            pH_current - 0.003 + RNG.normal(0.0, 0.02),
            3.5, 8.0
        ))

        # 4. Compute pH modifier for nutrient availability
        pH_mod = compute_ph_modifier(pH_current)

        # 5. Rainfall leaching acceleration for N and K
        #    Previous day's rain drives today's leaching (water percolation lag).
        leach_factor = 1.0 + RAIN_LEACH_FACTOR * prev_rain

        # 6. Apply decay to compute ground truth NPK (the TARGET LABELS)
        #    These are what the XGBoost model must learn to predict.
        #    NOTE: the model never sees k_N, k_P, k_K — only the proxy variables.
        N_current = float(np.clip(
            params["N0"] * np.exp(-k_N * leach_factor * (day + 1)) * pH_mod,
            0.0, params["N0"] * 1.1  # small buffer for noise
        ))
        P_current = float(np.clip(
            params["P0"] * np.exp(-k_P * (day + 1)) * pH_mod,
            0.0, params["P0"] * 1.1
        ))
        K_current = float(np.clip(
            params["K0"] * np.exp(-k_K * leach_factor * (day + 1)),
            0.0, params["K0"] * 1.1
        ))

        # 7. Simulate raw EC for this day (proxy for total dissolved ions)
        #    EC correlates with the NPK ratios relative to their baselines.
        ec_ratio = (
            EC_WEIGHT_N * (N_current / max(params["N0"], 1e-9)) +
            EC_WEIGHT_P * (P_current / max(params["P0"], 1e-9)) +
            EC_WEIGHT_K * (K_current / max(params["K0"], 1e-9))
        )
        EC_raw_day = float(np.clip(
            EC0 * ec_ratio + RNG.normal(0.0, EC_SENSOR_NOISE_STD),
            10.0, 2000.0
        ))

        # 8. Advance prev_rain for next day's leaching
        prev_rain = rain_today

        # ── Hourly readings ───────────────────────────────────────────────────
        for hour in range(READINGS_PER_DAY):

            # Temperature: sinusoidal diurnal cycle peaking at 14:00
            T_raw = (params["T_mean"]
                     + params["T_amp"] * np.sin(2 * np.pi * (hour - 5) / 24)
                     + RNG.normal(0.0, TEMP_SENSOR_NOISE_STD))
            T = float(np.clip(T_raw, 5.0, 45.0))

            # pH with hourly sensor noise
            pH_reading = float(np.clip(
                pH_current + RNG.normal(0.0, PH_SENSOR_NOISE_STD),
                3.5, 8.5
            ))

            # Moisture with hourly sensor noise
            moisture_reading = float(np.clip(
                moisture + RNG.normal(0.0, MOIST_SENSOR_NOISE_STD),
                5.0, 85.0
            ))

            # EC_raw with additional per-reading noise (on top of daily noise)
            EC_raw_reading = float(np.clip(
                EC_raw_day + RNG.normal(0.0, EC_SENSOR_NOISE_STD * 0.3),
                10.0, 2000.0
            ))

            # Apply EC₂₅ temperature compensation
            # EXACT SAME FORMULA as tasks.py Step 2 — must match precisely.
            EC_25 = EC_raw_reading / (1.0 + 0.019 * (T - 25.0))

            # Δ-EC₂₅: deviation from KALRO baseline EC
            delta_EC = EC_25 - EC0

            # Add small label noise to prevent the model from achieving
            # perfect R²=1.0 (which would indicate data leakage / overfitting)
            N_label = float(np.clip(
                N_current + RNG.normal(0.0, 0.5), 0.0, 200.0
            ))
            P_label = float(np.clip(
                P_current + RNG.normal(0.0, 0.2), 0.0, 100.0
            ))
            K_label = float(np.clip(
                K_current + RNG.normal(0.0, 1.0), 0.0, 500.0
            ))

            rows.append({
                # ── Metadata (not fed to model) ──────────────────────────
                "farm_id":   params["farm_id"],
                "season_id": params["season_id"],
                "day":       day,
                "hour":      hour,

                # ── Raw sensor values (for reference and rolling) ─────────
                "temperature_celsius":  round(T, 2),
                "ec_raw_us_per_cm":     round(EC_raw_reading, 2),
                # _moisture_raw: dedicated column for rolling computation.
                # Stored separately so compute_rolling_features() can roll
                # it cleanly without the chain-assignment bug.
                "_moisture_raw":        round(moisture_reading, 2),

                # ── Feature columns (fed to XGBoost — matches FEATURE_COLS) ─
                "days_since_baseline":   day,
                "ec_25":                 round(EC_25, 2),
                "delta_ec":              round(delta_EC, 2),
                "ph":                    round(pH_reading, 3),
                "moisture_7d_avg":       round(moisture_reading, 2),        # recalculated below
                "ec_7d_avg":             round(EC_25, 2),                   # recalculated below
                "ec_14d_avg":            round(EC_25, 2),                   # recalculated below
                "ec_delta_7d_14d":       0.0,                               # recalculated below
                # FIX: baseline_ec stored per-row so the model can normalise
                # delta_ec against the farm's starting point.
                # At inference: comes from CompositeBaselineLabTest.ec_us_per_cm_at_test_date
                "baseline_ec_us_per_cm": round(EC0, 2),
                "ec_depletion_pct":      round(delta_EC / EC0 * 100, 3),  # recalculated below

                # ── Target labels ─────────────────────────────────────────
                TARGET_N: round(N_label, 3),
                TARGET_P: round(P_label, 3),
                TARGET_K: round(K_label, 3),
            })

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ROLLING FEATURE COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute the rolling average features using actual historical values.

    The simulation loop above used point-in-time EC_25 as a placeholder for
    ec_7d_avg and ec_14d_avg. This function replaces those with proper rolling
    window calculations, exactly matching what _engineer_features() computes
    in tasks.py at inference time.

    Rolling is computed PER FARM PER SEASON to avoid leakage across seasons.
    The first ROLLING_WARMUP days of each season are dropped to ensure all
    rolling windows have full 14-day history.
    """
    print("  Computing rolling features per farm-season...", flush=True)
    processed_chunks = []

    for (farm_id, season_id), chunk in df.groupby(["farm_id", "season_id"]):
        chunk = chunk.sort_values(["day", "hour"]).copy()

        # Synthetic datetime index (1 reading/hour)
        chunk["_ts"] = pd.date_range(
            start="2025-03-01", periods=len(chunk), freq="1h"
        )
        chunk = chunk.set_index("_ts")

        # moisture_7d_avg: roll the dedicated _moisture_raw column (not the
        # placeholder that was written during simulation)
        chunk["moisture_7d_avg"] = (
            chunk["_moisture_raw"]
            .rolling("7D", min_periods=24)
            .mean()
        )
        chunk["ec_7d_avg"] = (
            chunk["ec_25"]
            .rolling("7D", min_periods=24)
            .mean()
        )
        chunk["ec_14d_avg"] = (
            chunk["ec_25"]
            .rolling("14D", min_periods=48)
            .mean()
        )
        chunk["ec_delta_7d_14d"] = chunk["ec_7d_avg"] - chunk["ec_14d_avg"]
        # Recompute ec_depletion_pct using the stored baseline_ec (not delta_ec placeholder)
        chunk["ec_depletion_pct"] = chunk["delta_ec"] / chunk["baseline_ec_us_per_cm"] * 100

        # Drop warmup rows (first 14 days = 336 readings)
        warmup_rows = ROLLING_WARMUP * READINGS_PER_DAY
        chunk = chunk.iloc[warmup_rows:].reset_index(drop=True)

        # Drop any remaining NaN rows
        chunk = chunk.dropna(subset=["ec_7d_avg", "ec_14d_avg", "moisture_7d_avg"])

        processed_chunks.append(chunk)

    return pd.concat(processed_chunks, ignore_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DATASET VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Run 5 sanity checks on the generated dataset.

    Any failed check prints a clear error and returns False.
    All checks must pass before the CSV is saved.

    These checks are your first line of defence against data generation bugs
    that would silently corrupt the trained model.
    """
    print("\n  Running dataset validation...", flush=True)
    all_passed = True

    # Check 1: Row count in expected range
    min_rows, max_rows = 38_000, 48_000
    if min_rows <= len(df) <= max_rows:
        print(f"  ✅ Check 1 PASS: Row count = {len(df):,} (expected {min_rows:,}–{max_rows:,})")
    else:
        print(f"  ❌ Check 1 FAIL: Row count = {len(df):,} — outside [{min_rows:,}, {max_rows:,}]")
        all_passed = False

    # Check 2: Zero NaN in all feature + label columns
    check_cols = FEATURE_COLS + TARGET_COLS
    nan_counts = df[check_cols].isna().sum()
    total_nans = nan_counts.sum()
    if total_nans == 0:
        print(f"  ✅ Check 2 PASS: Zero NaN values in all {len(check_cols)} feature+label columns")
    else:
        print(f"  ❌ Check 2 FAIL: {total_nans} NaN values found:\n{nan_counts[nan_counts > 0]}")
        all_passed = False

    # Check 3: N_true always in physiologically valid range
    n_min, n_max = df[TARGET_N].min(), df[TARGET_N].max()
    if 0 <= n_min and n_max <= 80:
        print(f"  ✅ Check 3 PASS: N_true range [{n_min:.2f}, {n_max:.2f}] mg/kg — valid")
    else:
        print(f"  ❌ Check 3 FAIL: N_true range [{n_min:.2f}, {n_max:.2f}] — outside [0, 80]")
        all_passed = False

    # Check 4: EC_25 always positive (negative EC is physically impossible)
    ec_min = df["ec_25"].min()
    if ec_min > 0:
        print(f"  ✅ Check 4 PASS: ec_25 minimum = {ec_min:.2f} µS/cm — always positive")
    else:
        print(f"  ❌ Check 4 FAIL: ec_25 has non-positive values (min={ec_min:.2f})")
        all_passed = False

    # Check 5: delta_ec is negatively correlated with N_true
    # As EC drops (negative delta), nutrients are depleted (lower N).
    # This validates the core sensor-proxy relationship the model must learn.
    corr = df["delta_ec"].corr(df[TARGET_N])
    if corr > 0.20:
        print(f"  ✅ Check 5 PASS: Pearson r(delta_ec, N_true) = {corr:.3f} — positive correlation confirmed")
    else:
        print(f"  ❌ Check 5 FAIL: Pearson r(delta_ec, N_true) = {corr:.3f} — expected > 0.20")
        print(f"     (Positive r means: higher EC relative to baseline → more nutrients present)")
        all_passed = False

    return all_passed


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — STATISTICS SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

def print_statistics(df: pd.DataFrame) -> None:
    """
    Print a formatted statistics summary for all feature and label columns.

    This output should be visually inspected before training to confirm
    that distributions are realistic. It also serves as Table X in the
    thesis appendix (Section 3.3.1: Datasets and Agronomic Baselines).
    """
    print("\n" + "═" * 65)
    print("  DATASET STATISTICS SUMMARY")
    print("  (Inspect these ranges before running train_model.py)")
    print("═" * 65)

    stats_cols = FEATURE_COLS + TARGET_COLS
    stats = df[stats_cols].describe().T[["mean", "std", "min", "max"]]
    stats.columns = ["Mean", "Std", "Min", "Max"]

    col_w = 22
    print(f"\n  {'Column':<{col_w}} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}  Unit")
    print("  " + "-" * 65)

    units = {
        "days_since_baseline": "days",
        "ec_25":               "µS/cm",
        "delta_ec":            "µS/cm",
        "ph":                  "pH",
        "moisture_7d_avg":     "%",
        "ec_7d_avg":           "µS/cm",
        "ec_14d_avg":          "µS/cm",
        "ec_delta_7d_14d":     "µS/cm",
        TARGET_N:              "mg/kg",
        TARGET_P:              "mg/kg",
        TARGET_K:              "mg/kg",
    }

    for col, row in stats.iterrows():
        unit = units.get(col, "")
        print(f"  {col:<{col_w}} {row['Mean']:>8.2f} {row['Std']:>8.2f} {row['Min']:>8.2f} {row['Max']:>8.2f}  {unit}")

    print()
    print(f"  Total rows      : {len(df):,}")
    print(f"  Farms           : {df['farm_id'].nunique()}")
    print(f"  Season types    : {df['season_id'].nunique()} per farm")
    print(f"  Feature columns : {len(FEATURE_COLS)}")
    print(f"  Target columns  : {len(TARGET_COLS)} (N, P, K)")
    print("═" * 65)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Fodder IoT — Synthetic Dataset Generator")
    print("  JKUAT ENE212-0065/2020")
    print("=" * 65)
    print(f"\n  Config: {N_FARMS} farms × {N_SEASONS} seasons × "
          f"{DAYS_PER_SEASON} days × {READINGS_PER_DAY} readings/day")
    print(f"  Expected rows (before warmup drop): "
          f"{N_FARMS * N_SEASONS * DAYS_PER_SEASON * READINGS_PER_DAY:,}")
    print(f"  Expected rows (after warmup drop):  "
          f"~{N_FARMS * N_SEASONS * (DAYS_PER_SEASON - ROLLING_WARMUP) * READINGS_PER_DAY:,}")
    print()

    t_start = time.time()
    all_seasons = []

    # ── Generate raw time-series for all farms and seasons ───────────────────
    total = N_FARMS * N_SEASONS
    for farm_id in range(N_FARMS):
        for season_id in range(N_SEASONS):
            n = farm_id * N_SEASONS + season_id + 1
            params = sample_farm_params(farm_id, season_id)
            season_type = params["season_type"]
            print(f"  [{n:02d}/{total}] Simulating farm {farm_id} | {season_type} | "
                  f"N₀={params['N0']:.1f} P₀={params['P0']:.1f} K₀={params['K0']:.1f} "
                  f"pH₀={params['pH0']:.2f} k_N={params['k_N']:.4f}", flush=True)
            df_season = simulate_season(params)
            all_seasons.append(df_season)

    # ── Concatenate all seasons ───────────────────────────────────────────────
    print(f"\n  Concatenating {len(all_seasons)} season dataframes...", flush=True)
    df_raw = pd.concat(all_seasons, ignore_index=True)
    print(f"  Raw rows before rolling feature computation: {len(df_raw):,}")

    # ── Compute rolling features and drop warmup rows ────────────────────────
    df_clean = compute_rolling_features(df_raw)
    print(f"  Clean rows after warmup drop:                {len(df_clean):,}")

    # ── Finalise column selection ─────────────────────────────────────────────
    # Keep: metadata + raw sensor (for reference) + features + labels
    final_cols = META_COLS + ["temperature_celsius", "ec_raw_us_per_cm"] + FEATURE_COLS + TARGET_COLS
    # Only keep columns that exist (rolling computation may rename some)
    final_cols = [c for c in final_cols if c in df_clean.columns]
    df_final = df_clean[final_cols].copy()

    # Round all float columns to 4 dp for clean CSV storage
    float_cols = df_final.select_dtypes(include=[float]).columns
    df_final[float_cols] = df_final[float_cols].round(4)

    # ── Validate ──────────────────────────────────────────────────────────────
    passed = validate_dataset(df_final)
    if not passed:
        print("\n  ❌ VALIDATION FAILED — dataset not saved. Fix the issues above.")
        sys.exit(1)

    # ── Print statistics ──────────────────────────────────────────────────────
    print_statistics(df_final)

    # ── Save to CSV ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    df_final.to_csv(DATASET_PATH, index=False)
    elapsed = time.time() - t_start

    print(f"\n  ✅ Dataset saved to: {DATASET_PATH}")
    print(f"  ✅ Rows: {len(df_final):,}  |  Columns: {len(df_final.columns)}")
    print(f"  ✅ File size: {os.path.getsize(DATASET_PATH) / 1024:.1f} KB")
    print(f"  ✅ Time elapsed: {elapsed:.1f}s")
    print(f"\n  Next step: python ml/train_model.py")
    print("=" * 65)


if __name__ == "__main__":
    main()