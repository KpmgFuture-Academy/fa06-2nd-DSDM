# 📌 연체 예측 파이프라인 (최종)

## 📂 구성 파일
- `schema_config_arrears.yaml` : 데이터/피처 스키마 정의
- `step2_build_arrears.py`     : Step2, shortlist 적용
- `step4_importance_arrears.py`: Step4, 중요도 산출 (GPU)
- `step5_feature_select_arrears.py`: Step5, 중요도 기반 최종 피처 확정
- `step6_train_arrears.py`     : Step6, 최종 학습/검증 (GPU)
- `step7_inference_align_to_training.py` : Step7, 학습 스키마 정렬 → 테스트 데이터 예측
- `final_catboost_arrears.cbm` (또는 `final_catboost_arrears_multi.cbm`) : 학습된 모델 파일 *(사용자 생성본을 같은 경로에 두세요)*

---

## 🚀 실행 순서
1. **Step2**
   ```bash
   python step2_build_arrears.py
   ```

2. **Step3 (라벨 생성)**  
   - 라벨링은 사용자 제공본(`arrears_train_step3_with_label_fromcsv.parquet`)을 활용합니다.  
   - 별도의 실행 스크립트는 포함되지 않음.

3. **Step4**
   ```bash
   python step4_importance_arrears.py
   ```

4. **Step5**
   ```bash
   python step5_feature_select_arrears.py
   ```

5. **Step6 (GPU 학습)**
   ```bash
   python step6_train_arrears.py
   ```

6. **Step7 (추론 실행기)**
   ```bash
   python step7_inference_align_to_training.py
   ```

---

## 📝 비고
- Step7은 **학습 parquet의 dtype/순서를 기준**으로 테스트 CSV를 강제 정렬/캐스팅하여 CatBoost의 categorical 타입 오류를 방지합니다.
- Step7 실행 시 입력은 반드시 **테스트 데이터**(`test_features_aligned.csv` 등 feature alignment 완료본)여야 합니다.  
  원본 `base_clean_test.csv`를 그대로 넣으면 오류가 발생할 수 있습니다.
- 모델 파일 이름은 학습 단계에서 저장된 이름(`final_catboost_arrears.cbm`)에 맞춰 `MODEL_PATH`를 수정하세요.
