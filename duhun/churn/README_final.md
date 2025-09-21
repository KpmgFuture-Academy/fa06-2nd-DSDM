# Churn Prediction – 최종 운영 패키지

본 문서는 **이탈 예측 자동화 파이프라인**의 최종 산출물을 정리합니다.
이번 업데이트에서는 다음이 반영되었습니다.

- 특수 결측치(-999999 등) 자동 인식 및 보정 (YAML `globals.sentinel_nulls`)
- 음수값 0 클램프 옵션 (`globals.clip_negative_to_zero`)
- 예측 결과에 `risk_label_text`(High/Medium/Low) 추가
- 전체 데이터에 대한 라벨 `count / %` 요약 출력

## 파일 구성
1) `final_catboost_model.cbm` — 학습 완료 모델(엔진)  
2) `schema_config.yaml` — 런타임 검증/보정 규칙(유연하게 수정 가능)  
3) `predict_runtime_final.py` — YAML+CBM 실행기(결과 저장/요약 포함)

## 실행 요령
- 모델 경로는 YAML의 `model.path`에 설정되어 있습니다
  - 예: `final_catboost_model.cbm`
  - 모델 경로는 YAML의 `model.path`에 설정되어 있으며,
  - 실행기와 같은 폴더에 `final_catboost_model.cbm` 파일이 존재해야 합니다.
- 테스트 파일 예: `base_clean_test.csv`
- 실행 결과:
  - `test_predictions_with_labels.csv`: 라벨/확률 포함 결과
  - (있을 경우) `test_quarantined.csv`: 규칙 위반(범위 초과 등) 행
  - 콘솔에 라벨 분포 요약(count / %)

## 라벨 사전
- 0 → Low
- 1 → Medium
- 2 → High

## 요약 메시지(발표용)
> YAML 규칙으로 입력 데이터 품질을 자동 검증/보정하고, 학습된 CatBoost 모델(cbm)로 예측합니다.  
> 특수 결측치(-999999 등)도 안전하게 처리하며, 최종 결과에 High/Medium/Low 라벨 텍스트와 분포(건수/비율)를 함께 제공합니다.
