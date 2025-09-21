# ============================================
# predict_runtime_final.py
# --------------------------------------------
# 🔑 필요 파일 (모두 동일한 경로에 있어야 함):
#   1) schema_config.yaml    : Feature/라벨 설정 및 모델 경로 정의
#   2) base_clean_test.csv   : 예측 대상 테스트 데이터 (입력)
#   3) final_model.cbm       : 학습된 CatBoost 모델 (YAML에 경로 지정)
#
# 📂 출력 파일:
#   - test_predictions_with_labels.csv : 최종 예측 결과 (항상 생성됨)
#   - test_quarantined.csv (선택적)    : YAML 검증에서 이상 행이 발견될 경우 자동 생성
#
# 📝 안내:
#   - 위 필요 파일들은 모두 이 실행기(predict_runtime_final.py)와 같은 폴더에 존재해야 함
#   - test_quarantined.csv 파일은 사전에 존재할 필요 없음
#     → 데이터 값이 규칙 위반(예: 범위 초과, 음수값 등)일 때만 자동 생성됨
#   - 콘솔 출력 메시지는 인코딩 문제를 방지하기 위해 영어로 표기
#
# 주요 기능:
#   - YAML 규칙 적용 (특수값 -999999 등은 결측치로 처리)
#   - 예측 결과에 risk_label_text (High/Medium/Low) 추가
#   - 전체 데이터 라벨 분포(count / %)를 콘솔에 요약 출력
# ============================================

import os
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier

LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}

def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def to_numeric_safe(s: pd.Series, want_float: bool) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.astype("float64") if want_float else x.fillna(0).astype("int64")

def validate_with_yaml(df: pd.DataFrame, cfg: dict):
    id_col = cfg["meta"]["id_col"]
    date_col = cfg["meta"]["date_col"]
    passthru = cfg["meta"].get("passthru_cols", [id_col, date_col])

    feats_cfg = cfg["features"]
    feat_names = [f["name"] for f in feats_cfg]

    # 누락 피처 생성
    df = df.copy()
    for c in feat_names:
        if c not in df.columns:
            df[c] = np.nan

    # 전역 룰
    g = cfg.get("globals", {})
    sentinel = set([float(s) for s in g.get("sentinel_nulls", [])])
    clip_neg = bool(g.get("clip_negative_to_zero", True))

    bad_mask = pd.Series(False, index=df.index)
    for f in feats_cfg:
        c = f["name"]
        want_float = f.get("type", "float") != "int"
        allow_null = bool(f.get("allow_null", True))
        fillna_val = f.get("fillna", 0)
        nonneg = bool(f.get("nonneg", False))
        rng = f.get("range", None)

        # 숫자 변환
        df[c] = to_numeric_safe(df[c], want_float=True)

        # 특수 결측치 감지(-999999 등) -> NaN
        if sentinel:
            s = df[c].astype("float64")
            mask_sentinel = s.isin(sentinel)
            if mask_sentinel.any():
                df.loc[mask_sentinel, c] = np.nan

        # 결측 -> 지정값(기본 0)
        if (not allow_null) or df[c].isna().any():
            df[c] = df[c].fillna(fillna_val)

        # 음수 클램프
        if nonneg or clip_neg:
            neg_m = df[c] < 0
            if neg_m.any():
                df.loc[neg_m, c] = 0

        # 범위 체크 -> 벗어난 행은 bad
        if rng is not None:
            lo, hi = rng[0], rng[1]
            if lo is not None:
                bad_mask |= (df[c] < float(lo))
            if hi is not None:
                bad_mask |= (df[c] > float(hi))

    good = df.loc[~bad_mask].copy()
    bad  = df.loc[ bad_mask].copy()

    out_cols = [col for col in passthru if col in df.columns] + feat_names
    return good[out_cols], bad[out_cols]

def infer(df_ok: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    feats = [f["name"] for f in cfg["features"]]
    proba_cols = cfg["output"]["proba_cols"]
    label_pred_col = cfg["output"]["label_pred"]
    label_text_col = cfg["output"]["label_text"]
    th_high = cfg.get("thresholds", {}).get("high")

    model = CatBoostClassifier()
    model.load_model(cfg["model"]["path"])

    proba = model.predict_proba(df_ok[feats])
    pred = proba.argmax(axis=1)

    # High 임계값 승격
    if th_high is not None:
        mask_high = proba[:, 2] >= float(th_high)
        pred[mask_high] = 2

    out = df_ok.copy()
    out[label_pred_col] = pred
    out[label_text_col] = [LABEL_MAP[i] for i in pred]
    for i, c in enumerate(proba_cols):
        out[c] = proba[:, i]
    return out

def print_summary(df_pred: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    label_text_col = cfg["output"]["label_text"]
    total = len(df_pred)
    cnts = df_pred[label_text_col].value_counts().reindex(["Low","Medium","High"], fill_value=0)
    ratios = (cnts / total * 100).round(2)

    summary = pd.DataFrame({
        "count": cnts.astype(int),
        "ratio(%)": ratios
    })
    print("\n[SUMMARY] label distribution")
    print(summary)
    return summary

if __name__ == "__main__":
    # 0) 설정 파일 경로
    CFG_PATH = "schema_config.yaml"
    # 1) 테스트 CSV 경로
    CSV_PATH = "base_clean_test.csv"
    # 2) 결과 저장 경로
    OUT_PATH = "test_predictions_with_labels.csv"

    # A) 설정 로드
    cfg = load_cfg(CFG_PATH)

    # B) 테스트 데이터 로드
    df = pd.read_csv(CSV_PATH)

    # C) YAML 검증/보정
    ok_df, bad_df = validate_with_yaml(df, cfg)
    if len(bad_df) > 0:
        quarantine_path = "test_quarantined.csv"
        bad_df.to_csv(quarantine_path, index=False, encoding="utf-8-sig")
        print(f"[WARN] 격리 행: {len(bad_df)}건 → {quarantine_path}")

    # D) 예측 + 라벨 텍스트 부여
    pred_df = infer(ok_df, cfg)

    # E) 요약(라벨 분포) 출력
    _summary = print_summary(pred_df, cfg)

    # F) 결과 저장
    pred_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[DONE] Prediction result saved → {OUT_PATH}")
