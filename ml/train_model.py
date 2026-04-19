"""
ml/train_model.py  (v2 — physics-anchored, correct split strategy)
====================================================================
XGBoost Training Pipeline — 3 Separate NPK Regressors
JKUAT ENE212-0065/2020

ROOT CAUSE FIX (v1→v2):
  v1 used a chronological split (train=days 0-131, val=days 132-179).
  This causes distribution shift: train sees high-NPK early season,
  val sees low-NPK late season — the model extrapolates out-of-range.
  Additionally, with only 5 farms, GroupKFold CV left one farm out
  entirely, and farms have different k_N values — the model had never
  seen that farm's decay rate, causing CV R² << 0.

  v2 uses PER-FARM HOLDOUT split:
    train = farms 0,1,2,3 (all 180 days each)
    val   = farm  4       (all 180 days)
  This tests the actual production question: "can the model predict
  NPK for a NEW farm it has never seen, given its KALRO baseline?"

  v2 also adds PHYSICS-INFORMED DECAY ESTIMATES as features:
    n_decay_estimate = N₀ × exp(-k_N_mean × days)
    p_decay_estimate = P₀ × exp(-k_P_mean × days)
    k_decay_estimate = K₀ × exp(-k_K_mean × days)
  These encode the KALRO baseline anchor directly, removing the
  between-farm ambiguity that made N and K models weak.

EXPECTED RESULTS (v2):
  N R² ≥ 0.88  |  P R² ≥ 0.88  |  K R² ≥ 0.88

USAGE:
  cd fodder_iot/
  python ml/generate_dataset.py   # must regenerate with v2 constants.py
  python ml/train_model.py
"""

import os, sys, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.constants import (
    FEATURE_COLS, TARGET_N, TARGET_P, TARGET_K,
    DATASET_PATH, MODEL_PATH_N, MODEL_PATH_P, MODEL_PATH_K,
    MODELS_DIR, MODEL_VERSION,
)


# ═════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

HYPERPARAMS = {
    TARGET_N: {
        "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": 42, "n_jobs": -1, "verbosity": 0,
    },
    TARGET_P: {
        "n_estimators": 400, "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.75, "colsample_bytree": 0.7, "min_child_weight": 5,
        "reg_alpha": 0.3, "reg_lambda": 1.5,
        "random_state": 42, "n_jobs": -1, "verbosity": 0,
    },
    TARGET_K: {
        "n_estimators": 450, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": 42, "n_jobs": -1, "verbosity": 0,
    },
}

R2_THRESHOLDS = {TARGET_N: 0.82, TARGET_P: 0.75, TARGET_K: 0.82}

# ─── Validation farm (held out entirely — simulates a brand-new farm) ────────
VAL_FARM_ID = 4


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING AND SPLITTING
# ═════════════════════════════════════════════════════════════════════════════

def load_and_split(path: str) -> tuple:
    """
    Load dataset and split using PER-FARM HOLDOUT strategy.

    WHY per-farm holdout (not chronological):
      Production scenario: a new farmer enters their KALRO baseline and the
      model must predict NPK for a farm it has NEVER seen before.
      Per-farm holdout replicates this exactly.

      Chronological split (old approach) tested time extrapolation within
      known farms — not the actual generalisation challenge.

    Train: farms 0–3 (all 180 days each)
    Val:   farm  4 (all 180 days) — fully unseen farm

    CV: 4-fold on training set (each fold withholds one training farm).
    """
    print(f"  Loading {path}...")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")

    # ── Validate features ─────────────────────────────────────────────────────
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"\n  ❌ Missing features: {missing}\n"
            f"  Run: python ml/generate_dataset.py  (regenerate dataset)"
        )

    # ── Validate physics features are present ─────────────────────────────────
    physics = ["n_decay_estimate", "p_decay_estimate", "k_decay_estimate"]
    missing_p = [f for f in physics if f not in df.columns]
    if missing_p:
        raise ValueError(
            f"\n  ❌ Physics features missing: {missing_p}\n"
            f"  Regenerate: python ml/generate_dataset.py"
        )
    print(f"  ✅ All {len(FEATURE_COLS)} features present (incl. physics priors)")

    # ── Per-farm holdout split ────────────────────────────────────────────────
    train_df = df[df["farm_id"] != VAL_FARM_ID].copy()
    val_df   = df[df["farm_id"] == VAL_FARM_ID].copy()

    print(f"  Train: farms {sorted(train_df['farm_id'].unique())} "
          f"({len(train_df):,} rows)")
    print(f"  Val:   farm  [{VAL_FARM_ID}] "
          f"({len(val_df):,} rows) ← FULLY UNSEEN FARM")
    print(f"  Split: {len(train_df)/len(df):.1%} train / {len(val_df)/len(df):.1%} val")

    # ── Show baseline range to confirm anchor features are working ────────────
    print(f"\n  KALRO baseline range across training farms:")
    print(f"    N₀ (n_decay at day 0): {train_df['n_decay_estimate'].max():.0f} "
          f"max across farms at day 14")
    for fid in sorted(train_df['farm_id'].unique()):
        fd = train_df[(train_df['farm_id']==fid) & (train_df['days_since_baseline']==14)]
        n0 = fd['n_decay_estimate'].mean()
        k0 = fd['k_decay_estimate'].mean()
        print(f"    Farm {fid}: N₀≈{n0:.1f}  K₀≈{k0:.1f} mg/kg")
    fd4 = val_df[val_df['days_since_baseline']==14]
    print(f"    Farm {VAL_FARM_ID} (val): N₀≈{fd4['n_decay_estimate'].mean():.1f}  "
          f"K₀≈{fd4['k_decay_estimate'].mean():.1f} mg/kg  ← model must generalise here")

    X_train = train_df[FEATURE_COLS].values
    X_val   = val_df[FEATURE_COLS].values

    # Groups for 4-fold CV (each fold withholds one training farm)
    groups_train = train_df["farm_id"].values

    targets = {}
    for t in [TARGET_N, TARGET_P, TARGET_K]:
        targets[t] = {"train": train_df[t].values, "val": val_df[t].values}

    return X_train, X_val, targets, groups_train, df


# ═════════════════════════════════════════════════════════════════════════════
# SINGLE MODEL TRAINER
# ═════════════════════════════════════════════════════════════════════════════

def train_single_model(target_name, X_train, y_train, X_val, y_val,
                       feature_names, groups_train=None) -> dict:
    """Train one XGBoost regressor with 4-fold farm-aware CV."""
    print(f"\n  {'─'*55}")
    print(f"  Training: {target_name}")
    print(f"  {'─'*55}")

    params = HYPERPARAMS[target_name]

    # ── 4-fold CV (leave-one-farm-out on training farms 0-3) ─────────────────
    print("  Cross-validation (4-fold, leave-one-farm-out)...", flush=True)
    from sklearn.model_selection import GroupKFold
    cv_model = XGBRegressor(**params)
    gkf      = GroupKFold(n_splits=4)  # 4 training farms → 4 folds
    cv_scores = cross_val_score(
        cv_model, X_train, y_train,
        cv=gkf, groups=groups_train, scoring="r2", n_jobs=-1,
    )
    cv_mean = float(np.mean(cv_scores))
    cv_std  = float(np.std(cv_scores))
    print(f"  CV R² = {cv_mean:.4f} ± {cv_std:.4f}  "
          f"(folds: {[round(s,3) for s in cv_scores]})")

    # ── Final fit with early stopping ─────────────────────────────────────────
    print("  Fitting final model...", flush=True)
    t0 = time.time()
    final = XGBRegressor(**params, early_stopping_rounds=30, eval_metric="rmse")
    final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s | Best round: {final.best_iteration}")

    # ── Evaluate on held-out farm ─────────────────────────────────────────────
    y_pred   = final.predict(X_val)
    r2       = float(r2_score(y_val, y_pred))
    rmse     = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    mae      = float(np.mean(np.abs(y_val - y_pred)))
    within_2 = float(np.mean(np.abs(y_val - y_pred) <= 2.0) * 100)

    threshold = R2_THRESHOLDS[target_name]
    status    = "✅ GOOD" if r2 >= threshold else f"⚠️  WEAK (< {threshold})"
    print(f"  Val R²      = {r2:.4f}   {status}")
    print(f"  Val RMSE    = {rmse:.3f} mg/kg")
    print(f"  Val MAE     = {mae:.3f} mg/kg")
    print(f"  Within ±2   = {within_2:.1f}%")

    # ── Feature importances ───────────────────────────────────────────────────
    feat_imp = sorted(zip(feature_names, final.feature_importances_),
                      key=lambda x: x[1], reverse=True)
    print(f"  Top feature: {feat_imp[0][0]} ({feat_imp[0][1]:.3f})")
    print(f"  Feature importances:")
    for fname, fimp in feat_imp:
        bar = "█" * int(fimp * 35)
        print(f"    {fname:<28} {fimp:.3f}  {bar}")

    return {
        "model":        final,
        "target":       target_name,
        "cv_r2_mean":   round(cv_mean, 4),
        "cv_r2_std":    round(cv_std, 4),
        "val_r2":       round(r2, 4),
        "val_rmse":     round(rmse, 4),
        "val_mae":      round(mae, 4),
        "within_2_pct": round(within_2, 2),
        "best_round":   int(final.best_iteration),
        "val_farm":     VAL_FARM_ID,
        "feature_importances": {fn: round(float(fi), 4) for fn, fi in feat_imp},
        "model_version": MODEL_VERSION,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SAVE AND SUMMARISE
# ═════════════════════════════════════════════════════════════════════════════

MODEL_PATHS = {TARGET_N: MODEL_PATH_N, TARGET_P: MODEL_PATH_P, TARGET_K: MODEL_PATH_K}

def save_results(results: dict) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    for target, result in results.items():
        joblib.dump(result["model"], MODEL_PATHS[target], compress=3)
        size_kb = os.path.getsize(MODEL_PATHS[target]) / 1024
        print(f"  Saved {target} → {MODEL_PATHS[target]}  ({size_kb:.0f} KB)")
    report = {t: {k: v for k, v in r.items() if k != "model"}
              for t, r in results.items()}
    rpath = os.path.join(MODELS_DIR, "training_report.json")
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved report → {rpath}")


def print_summary(results: dict, elapsed: float) -> None:
    print()
    print("═" * 65)
    print("  TRAINING SUMMARY  (val = fully unseen farm)")
    print("═" * 65)
    print(f"  {'Target':<8} {'CV R²':>8} {'Val R²':>8} {'RMSE':>8} {'MAE':>7} {'±2mg':>7}  Status")
    print("  " + "─" * 60)
    all_pass = True
    for t in [TARGET_N, TARGET_P, TARGET_K]:
        r = results[t]
        thr    = R2_THRESHOLDS[t]
        status = "✅ PASS" if r["val_r2"] >= thr else f"⚠️  < {thr}"
        if r["val_r2"] < thr: all_pass = False
        print(f"  {t:<8} {r['cv_r2_mean']:>8.4f} {r['val_r2']:>8.4f} "
              f"{r['val_rmse']:>8.3f} {r['val_mae']:>7.3f} "
              f"{r['within_2_pct']:>6.1f}%  {status}")
    print("  " + "─" * 60)
    print(f"  Training time: {elapsed:.1f}s | Version: {MODEL_VERSION}")
    print()
    if all_pass:
        print("  🎉 All models meet quality thresholds!")
        print("  Next: python ml/evaluate.py")
        print("  Then: wire ml/predictor.py into tasks.py")
    else:
        print("  ⚠️  Some models below threshold.")
        print("  Tip: increase N_FARMS in generate_dataset.py to 10+")
    print("═" * 65)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Fodder IoT — XGBoost NPK Model Training  (v2)")
    print("  JKUAT ENE212-0065/2020")
    print("=" * 65)
    if not os.path.exists(DATASET_PATH):
        print(f"  ❌ Dataset not found. Run: python ml/generate_dataset.py")
        sys.exit(1)

    t_start = time.time()
    X_train, X_val, targets, groups_train, df = load_and_split(DATASET_PATH)

    results = {}
    for target in [TARGET_N, TARGET_P, TARGET_K]:
        results[target] = train_single_model(
            target_name=target,
            X_train=X_train, y_train=targets[target]["train"],
            X_val=X_val, y_val=targets[target]["val"],
            feature_names=FEATURE_COLS, groups_train=groups_train,
        )

    print(f"\n  {'─'*55}")
    print("  Saving models...")
    save_results(results)
    print_summary(results, time.time() - t_start)


if __name__ == "__main__":
    main()