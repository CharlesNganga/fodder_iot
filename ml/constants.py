"""
ml/constants.py
===============
Single source of truth for feature names and model configuration.

WHY this file exists:
  The feature dict produced by tasks.py _engineer_features() at inference
  time must have IDENTICAL key names to the columns the XGBoost model was
  trained on. A single character mismatch causes silent wrong predictions
  or a crash at inference. By importing FEATURE_COLS in both
  generate_dataset.py and train_model.py, we guarantee consistency.

  If you ever add a feature, add it here first, then update
  _engineer_features() in tasks.py to match.
"""

# ── Feature columns (inputs to the XGBoost models) ───────────────────────────
# Order matters — XGBoost stores column order internally.
FEATURE_COLS = [
    "days_since_baseline",   # time elapsed since KALRO lab test (depletion proxy)
    "ec_25",                 # temperature-compensated EC at 25°C (µS/cm)
    "delta_ec",              # EC_25 minus baseline EC (absolute ion depletion signal)
    "ph",                    # soil pH
    "moisture_7d_avg",       # 7-day rolling average moisture (%)
    "ec_7d_avg",             # 7-day rolling average EC_25 (µS/cm)
    "ec_14d_avg",            # 14-day rolling average EC_25 (µS/cm)
    "ec_delta_7d_14d",       # ec_7d_avg − ec_14d_avg (trend direction)
    # FIX: two new features that normalise depletion across different farm baselines
    "baseline_ec_us_per_cm", # KALRO lab EC at T₀ — anchors the model to farm baseline
    "ec_depletion_pct",      # delta_ec / baseline_ec × 100 — % depletion (farm-normalised)
]

# ── Target columns (NPK labels — what the models predict) ────────────────────
TARGET_N = "N_true"
TARGET_P = "P_true"
TARGET_K = "K_true"
TARGET_COLS = [TARGET_N, TARGET_P, TARGET_K]

# ── Metadata columns (for debugging only — NOT fed to the model) ─────────────
META_COLS = ["farm_id", "season_id", "day", "hour"]

# ── Model file paths ─────────────────────────────────────────────────────────
import os
ML_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ML_DIR, "models")
DATA_DIR   = os.path.join(ML_DIR, "data")

MODEL_PATH_N = os.path.join(MODELS_DIR, "npk_model_N.joblib")
MODEL_PATH_P = os.path.join(MODELS_DIR, "npk_model_P.joblib")
MODEL_PATH_K = os.path.join(MODELS_DIR, "npk_model_K.joblib")
DATASET_PATH = os.path.join(DATA_DIR,   "synthetic_dataset.csv")

# ── Model version tag (stored on every NPKPrediction DB record) ───────────────
MODEL_VERSION = "v1.0-xgb-synthetic-kalro"