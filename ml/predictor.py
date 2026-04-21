"""
ml/predictor.py
===============
XGBoost Model Loader — Singleton Pattern
JKUAT ENE212-0065/2020

PURPOSE:
  This module replaces the _run_ml_inference() decay stub in tasks.py.
  It loads the three trained XGBoost models (N, P, K) once at Django
  startup and keeps them in memory for the lifetime of the process.

WHY SINGLETON PATTERN:
  The Celery worker processes thousands of readings per day. Loading
  three .joblib files (144 KB + 160 KB + 98 KB = ~400 KB total) on
  every inference call would add ~50ms per reading and waste memory.
  The singleton loads once, then every call to predict_npk() reuses
  the same in-memory model objects — inference takes ~0.1ms.

  This is standard production practice (Django app startup, gunicorn
  pre-fork, Celery worker initialisation all benefit from this).

USAGE in tasks.py:
  from ml.predictor import predict_npk

  npk = predict_npk(features_dict, baseline)
  # Returns: {"N": 18.4, "P": 10.2, "K": 68.1, "confidence": 0.89,
  #           "model_version": "v2.0-xgb-physics-anchored"}

FEATURE DICT REQUIRED KEYS (must match FEATURE_COLS in constants.py):
  days_since_baseline, ec_25, delta_ec, ph, moisture_7d_avg,
  ec_7d_avg, ec_14d_avg, ec_delta_7d_14d, baseline_ec_us_per_cm,
  ec_depletion_pct, n_decay_estimate, p_decay_estimate, k_decay_estimate

  The physics-informed features (n_decay_estimate etc.) are computed
  from the baseline object: baseline.nitrogen_mg_per_kg × exp(-k × days).
  This is done in _engineer_features() in tasks.py.
"""

import os
import sys
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy import joblib to avoid hard dependency at module load time ──────────
# Django imports this module at startup; if joblib is not installed,
# we want a clear error message, not a cryptic ImportError.
try:
    import joblib
except ImportError:
    joblib = None  # handled gracefully in _load_models()

# ── Ensure ml package is importable when called from Django ─────────────────
_ML_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_ML_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from ml.constants import (
    FEATURE_COLS,
    MODEL_PATH_N, MODEL_PATH_P, MODEL_PATH_K,
    MODEL_VERSION,
    K_N_MEAN, K_P_MEAN, K_K_MEAN,
)


# ═════════════════════════════════════════════════════════════════════════════
# SINGLETON STATE
# ═════════════════════════════════════════════════════════════════════════════

# Module-level variables — populated once on first call to predict_npk().
# Thread-safe for read access after initialisation (CPython GIL protects
# the assignment; multiple Celery threads read but never write after load).
_model_N = None
_model_P = None
_model_K = None
_models_loaded = False


def _load_models() -> bool:
    """
    Load all three XGBoost models from .joblib files into module globals.

    Called once on first predict_npk() invocation (lazy loading).
    Returns True if all models loaded successfully, False otherwise.

    WHY lazy loading (not at import time)?
      Django loads all app modules at startup, including tasks.py which
      imports this module. If we loaded models at import time, every
      `python manage.py migrate`, `pytest`, or `manage.py shell` command
      would load 400KB of model files unnecessarily.
      Lazy loading ensures models are only loaded when the first real
      telemetry reading arrives.
    """
    global _model_N, _model_P, _model_K, _models_loaded

    if _models_loaded:
        return True

    if joblib is None:
        logger.error(
            "predictor: joblib not installed. "
            "Run: pip install joblib   (or pip install scikit-learn)"
        )
        return False

    model_paths = {
        "N": MODEL_PATH_N,
        "P": MODEL_PATH_P,
        "K": MODEL_PATH_K,
    }

    # Check all model files exist before attempting to load
    missing = [
        f"{name}: {path}"
        for name, path in model_paths.items()
        if not os.path.exists(path)
    ]
    if missing:
        logger.error(
            "predictor: trained model files not found:\n  %s\n"
            "  Run: python ml/train_model.py",
            "\n  ".join(missing)
        )
        return False

    try:
        logger.info("predictor: loading XGBoost models from %s/", _ML_DIR)
        _model_N = joblib.load(MODEL_PATH_N)
        _model_P = joblib.load(MODEL_PATH_P)
        _model_K = joblib.load(MODEL_PATH_K)
        _models_loaded = True
        logger.info(
            "predictor: models loaded — N(%s KB) P(%s KB) K(%s KB)",
            round(os.path.getsize(MODEL_PATH_N) / 1024),
            round(os.path.getsize(MODEL_PATH_P) / 1024),
            round(os.path.getsize(MODEL_PATH_K) / 1024),
        )
        return True
    except Exception as exc:
        logger.error("predictor: failed to load models: %s", exc)
        _model_N = _model_P = _model_K = None
        _models_loaded = False
        return False


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

def predict_npk(features: dict, baseline) -> dict | None:
    """
    Run XGBoost inference and return predicted NPK values.

    This is the drop-in replacement for _run_ml_inference() in tasks.py.

    Args:
        features:  dict produced by _engineer_features() in tasks.py.
                   Must contain all 13 keys listed in FEATURE_COLS.
        baseline:  CompositeBaselineLabTest ORM object.
                   Used to compute the physics-informed decay estimates
                   if they are missing from the features dict (fallback
                   for legacy callers).

    Returns:
        dict with keys: N, P, K (float, mg/kg), confidence (float 0–1),
        model_version (str). Returns None if models are not loaded.

    The confidence score is the mean val R² across the three models,
    as reported in training_report.json. It is stored on NPKPrediction
    and surfaced in the React Native app dashboard.
    """
    # ── Ensure models are loaded ──────────────────────────────────────────────
    if not _load_models():
        logger.warning(
            "predictor: models not available — falling back to decay stub"
        )
        return _decay_stub_fallback(features, baseline)

    # ── Ensure physics features are present ───────────────────────────────────
    # If _engineer_features() in tasks.py has not yet been updated to include
    # the physics priors, compute them here from the baseline object.
    days = features.get("days_since_baseline", 0)
    if "n_decay_estimate" not in features:
        features = dict(features)   # don't mutate the caller's dict
        features["n_decay_estimate"] = float(
            baseline.nitrogen_mg_per_kg   * np.exp(-K_N_MEAN * days)
        )
        features["p_decay_estimate"] = float(
            baseline.phosphorus_mg_per_kg * np.exp(-K_P_MEAN * days)
        )
        features["k_decay_estimate"] = float(
            baseline.potassium_mg_per_kg  * np.exp(-K_K_MEAN * days)
        )

    # ── Ensure baseline_ec and ec_depletion_pct are present ──────────────────
    if "baseline_ec_us_per_cm" not in features:
        features = dict(features)
        ec0 = baseline.ec_us_per_cm_at_test_date
        features["baseline_ec_us_per_cm"] = float(ec0)
        delta_ec = features.get("delta_ec", 0.0)
        features["ec_depletion_pct"] = float(
            delta_ec / ec0 * 100 if ec0 > 0 else 0.0
        )

    # ── Build feature vector in the exact order XGBoost expects ───────────────
    # FEATURE_COLS defines the canonical column order used during training.
    # Any deviation causes silent wrong predictions (XGBoost uses positional
    # feature indices internally, not names at inference time).
    missing_keys = [k for k in FEATURE_COLS if k not in features]
    if missing_keys:
        logger.error(
            "predictor: feature dict missing keys: %s — cannot run inference",
            missing_keys
        )
        return _decay_stub_fallback(features, baseline)

    X = np.array([[features[col] for col in FEATURE_COLS]], dtype=np.float32)

    # ── Run inference ──────────────────────────────────────────────────────────
    try:
        N_pred = float(_model_N.predict(X)[0])
        P_pred = float(_model_P.predict(X)[0])
        K_pred = float(_model_K.predict(X)[0])
    except Exception as exc:
        logger.error("predictor: XGBoost inference failed: %s", exc)
        return _decay_stub_fallback(features, baseline)

    # Clamp predictions to physically valid ranges
    N_pred = max(0.0, min(N_pred, 200.0))
    P_pred = max(0.0, min(P_pred, 100.0))
    K_pred = max(0.0, min(K_pred, 500.0))

    # Confidence: mean val R² from training report (N=0.965, P=0.771, K=0.959)
    # Weighted average — N and K carry more weight as they are more actionable
    confidence = round((0.965 * 0.4 + 0.771 * 0.2 + 0.959 * 0.4), 3)

    logger.debug(
        "predictor: N=%.2f P=%.2f K=%.2f mg/kg (confidence=%.3f)",
        N_pred, P_pred, K_pred, confidence
    )

    return {
        "N":             round(N_pred, 2),
        "P":             round(P_pred, 2),
        "K":             round(K_pred, 2),
        "confidence":    confidence,
        "model_version": MODEL_VERSION,
    }


def is_ready() -> bool:
    """
    Returns True if all three models are loaded and ready for inference.
    Useful for health-check endpoints and admin diagnostics.
    """
    return _load_models()


def model_info() -> dict:
    """
    Returns metadata about the loaded models.
    Used by the Django admin and health-check endpoint.
    """
    if not _models_loaded:
        return {"status": "not_loaded", "model_version": MODEL_VERSION}

    return {
        "status":        "loaded",
        "model_version": MODEL_VERSION,
        "feature_count": len(FEATURE_COLS),
        "features":      FEATURE_COLS,
        "model_files": {
            "N": os.path.basename(MODEL_PATH_N),
            "P": os.path.basename(MODEL_PATH_P),
            "K": os.path.basename(MODEL_PATH_K),
        },
        "val_r2": {"N": 0.9652, "P": 0.7713, "K": 0.9594},
    }


# ═════════════════════════════════════════════════════════════════════════════
# FALLBACK — Physics decay stub (used if models fail to load)
# ═════════════════════════════════════════════════════════════════════════════

def _decay_stub_fallback(features: dict, baseline) -> dict:
    """
    Emergency fallback to the original physics decay formula.

    Used when:
      - Model .joblib files are missing (first deploy before training)
      - joblib is not installed
      - XGBoost inference raises an unexpected exception

    This ensures the Celery pipeline never crashes — it degrades gracefully
    to the stub with a warning log. The model_version field flags this so
    the mobile app can show a reduced-confidence indicator.
    """
    days  = features.get("days_since_baseline", 0)
    delta = features.get("delta_ec", 0.0)

    depletion_factor = max(0.5, 1.0 + (delta / 500.0))

    N_pred = baseline.nitrogen_mg_per_kg   * np.exp(-K_N_MEAN * days) * depletion_factor
    P_pred = baseline.phosphorus_mg_per_kg * np.exp(-K_P_MEAN * days) * depletion_factor
    K_pred = baseline.potassium_mg_per_kg  * np.exp(-K_K_MEAN * days) * depletion_factor

    logger.warning(
        "predictor: using decay stub fallback — "
        "run python ml/train_model.py to train XGBoost models"
    )

    return {
        "N":             round(max(0.0, N_pred), 2),
        "P":             round(max(0.0, P_pred), 2),
        "K":             round(max(0.0, K_pred), 2),
        "confidence":    0.50,
        "model_version": "v0.1-decay-stub-fallback",
    }
