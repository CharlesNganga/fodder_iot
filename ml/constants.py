"""
ml/constants.py  (v2 — physics-informed features added)
=========================================================
Single source of truth for feature names and model configuration.

ARCHITECTURE NOTE — Why N₀, K₀, P₀ are explicit features:
  The system is a Baseline-Anchored State Observer. The KALRO lab test
  provides the T₀ anchor: N₀, P₀, K₀. The model predicts N(t) from
  sensor proxies, but without knowing N₀ it cannot distinguish:
    • Farm A: N=15 mg/kg at day 100 (started at N₀=20 → nearly depleted)
    • Farm B: N=15 mg/kg at day 100 (started at N₀=60 → only 25% depleted)
  These two farms need completely different interventions.

  The physics-informed decay estimates encode this directly:
    n_decay_estimate = N₀ × exp(-k_N × days)  ← physics prior
  XGBoost then corrects the residuals between the physics estimate and
  actual sensor readings. This is the "Baseline-Anchored" part of the
  architecture — without N₀, the anchor is lost.

  N₀, K₀, P₀ are available at inference time from:
    baseline.nitrogen_mg_per_kg   (CompositeBaselineLabTest)
    baseline.potassium_mg_per_kg
    baseline.phosphorus_mg_per_kg
"""

import os

# ── Feature columns (10 sensor features + 3 physics priors = 13 total) ───────
FEATURE_COLS = [
    # ── Sensor proxy features ─────────────────────────────────────────────────
    "days_since_baseline",   # time elapsed since KALRO lab test
    "ec_25",                 # temperature-compensated EC at 25°C (µS/cm)
    "delta_ec",              # EC_25 − baseline EC (absolute depletion signal)
    "ph",                    # soil pH
    "moisture_7d_avg",       # 7-day rolling average moisture (%)
    "ec_7d_avg",             # 7-day rolling average EC_25 (µS/cm)
    "ec_14d_avg",            # 14-day rolling average EC_25 (µS/cm)
    "ec_delta_7d_14d",       # trend direction (7d − 14d avg)
    "baseline_ec_us_per_cm", # KALRO baseline EC (farm anchor for normalisation)
    "ec_depletion_pct",      # (delta_ec / baseline_ec) × 100 — farm-normalised %

    # ── Physics-informed decay estimates (the "Baseline-Anchored" features) ──
    # These encode the expected NPK value from the KALRO anchor + decay rate.
    # XGBoost corrects the residuals between this physics estimate and the
    # true sensor-derived value. Without these, the model cannot distinguish
    # farms with different starting baselines (GroupKFold CV R² was negative).
    "n_decay_estimate",      # N₀ × exp(-0.008 × days)  — N physics prior
    "p_decay_estimate",      # P₀ × exp(-0.003 × days)  — P physics prior
    "k_decay_estimate",      # K₀ × exp(-0.006 × days)  — K physics prior
]

# ── Target columns ─────────────────────────────────────────────────────────
TARGET_N    = "N_true"
TARGET_P    = "P_true"
TARGET_K    = "K_true"
TARGET_COLS = [TARGET_N, TARGET_P, TARGET_K]

# ── Metadata columns (NOT fed to model) ───────────────────────────────────
META_COLS = ["farm_id", "season_id", "day", "hour"]

# ── Mean decay constants (from Smaling et al. 1993, Kenyan highlands) ─────
# Used for physics-informed feature computation in both generate_dataset.py
# and _engineer_features() in tasks.py at inference time.
K_N_MEAN = 0.008
K_P_MEAN = 0.003
K_K_MEAN = 0.006

# ── File paths ──────────────────────────────────────────────────────────────
ML_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ML_DIR, "models")
DATA_DIR   = os.path.join(ML_DIR, "data")

MODEL_PATH_N = os.path.join(MODELS_DIR, "npk_model_N.joblib")
MODEL_PATH_P = os.path.join(MODELS_DIR, "npk_model_P.joblib")
MODEL_PATH_K = os.path.join(MODELS_DIR, "npk_model_K.joblib")
DATASET_PATH = os.path.join(DATA_DIR,   "synthetic_dataset.csv")

MODEL_VERSION = "v2.0-xgb-physics-anchored"