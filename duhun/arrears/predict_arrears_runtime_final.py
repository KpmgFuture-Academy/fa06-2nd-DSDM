# ============================================
# predict_arrears_runtime_final.py
# (CSV feature list → inference, clean logs)
# --------------------------------------------
# Required files:
#   1) arrears_step5_final_features.csv  ← feature list (col: feature)
#   2) base_clean_test.csv               ← test data
#   3) final_catboost_arrears.cbm        ← trained CatBoost model
#
# Output:
#   - predictions_arrears.csv
# ============================================

import os
import pandas as pd
from catboost import CatBoostClassifier

FEATURE_LIST_CSV = "arrears_step5_final_features.csv"
TEST_PATH        = "base_clean_test.csv"
MODEL_PATH       = "final_catboost_arrears.cbm"
OUT_CSV          = "predictions_arrears.csv"

ID_COL   = "ID"
DATE_COL = "기준년월"

RISK_MAP = {0: "Low", 1: "Medium", 2: "High"}

def log(msg): 
    print(msg, flush=True)

def main():
    # 1) feature list
    feats_df = pd.read_csv(FEATURE_LIST_CSV)
    features = feats_df["feature"].astype(str).tolist()
    log(f"[INFO] feature list loaded: {len(features)} cols")

    # 2) test data
    test_df = pd.read_csv(TEST_PATH, dtype=str, low_memory=False)
    log(f"[INFO] test raw shape: {test_df.shape[0]} rows, {test_df.shape[1]} cols")

    # 3) align schema
    aligned = pd.DataFrame(index=test_df.index)
    missing = []
    for c in features:
        if c in test_df.columns:
            aligned[c] = test_df[c]
        else:
            aligned[c] = pd.NA
            missing.append(c)
    extra = [c for c in test_df.columns if c not in features]
    if extra:
        log(f"[WARN] extra test columns ignored: {len(extra)}")
    if missing:
        log(f"[WARN] missing feature columns filled as NaN: {len(missing)} → {missing[:5]}")

    # 4) cast: object/string as string, numeric-like to float
    for c in aligned.columns:
        aligned[c] = pd.to_numeric(aligned[c], errors="ignore")
    aligned = aligned.fillna("Unknown")

    # 5) load model
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    log(f"[INFO] model loaded: {MODEL_PATH}")

    # 6) predict
    proba = model.predict_proba(aligned[features])
    preds = proba.argmax(axis=1)

    out = pd.DataFrame(index=test_df.index)
    if ID_COL in test_df.columns:   out[ID_COL] = test_df[ID_COL]
    if DATE_COL in test_df.columns: out[DATE_COL] = test_df[DATE_COL]

    for c in features:
        out[c] = aligned[c]

    out["p0"] = proba[:, 0]
    if proba.shape[1] > 1: out["p1"] = proba[:, 1]
    if proba.shape[1] > 2: out["p2"] = proba[:, 2]
    out["pred_class"] = preds
    out["risk_label"] = out["pred_class"].map(RISK_MAP)

    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    log(f"[DONE] predictions saved: {OUT_CSV}")

    # 7) summary
    dist = out["risk_label"].value_counts(normalize=True)
    log("[INFO] risk_label distribution (ratio): " + 
        ", ".join([f"{k}={v:.4f}" for k, v in dist.items()]))

    # 8) clean sample preview
    preview_cols = [c for c in [ID_COL, DATE_COL, "risk_label"] if c in out.columns]
    log("[INFO] sample preview:")
    log(out[preview_cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
