# 이식 가능한 예측 파이프라인 사용 가이드

## 📁 파일 구조

```
프로젝트 폴더/
├── portable_prediction_pipeline.py    # 메인 파이프라인 코드
├── prediction_config.json             # 설정 파일
├── README_PORTABLE.md                 # 사용 가이드
├── requirements.txt                   # 필요한 패키지
└── fintech/
    └── fintech/
        ├── base_test.csv              # Test 데이터
        └── base_train.csv             # Train 데이터
```

## 🚀 사용 방법

### 1. 환경 설정

```bash
# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

- `fintech/fintech/base_test.csv` 파일을 준비
- `fintech/fintech/base_train.csv` 파일을 준비

### 3. 경로 설정 (선택사항)

`prediction_config.json` 파일에서 경로를 수정할 수 있습니다:

```json
{
  "data_paths": {
    "base_test_csv": "fintech/fintech/base_test.csv",
    "base_train_csv": "fintech/fintech/base_train.csv"
  },
  "output_paths": {
    "predictions_csv": "predictions.csv",
    "model_pkl": "trained_model.pkl"
  },
  "model_settings": {
    "sample_size": 50000,
    "chunk_size": 10000,
    "max_features": 100
  }
}
```

**경로 설정 방법:**
- 상대 경로: `"fintech/fintech/base_test.csv"`
- 절대 경로: `"/Users/username/data/base_test.csv"`

### 4. 실행

```bash
python portable_prediction_pipeline.py
```

## 📊 출력 결과

### 생성되는 파일:
- `predictions.csv`: 예측 결과 (ID, Segment)
- `trained_model.pkl`: 학습된 모델

### 예측 결과 형태:
```csv
ID,Segment
0,D
1,E
2,D
3,E
4,E
...
```

## ⚙️ 설정 옵션

### data_paths
- `base_test_csv`: Test 데이터 파일 경로
- `base_train_csv`: Train 데이터 파일 경로

### output_paths
- `predictions_csv`: 예측 결과 저장 경로
- `model_pkl`: 학습된 모델 저장 경로

### model_settings
- `sample_size`: 사용할 샘플 수 (기본: 50000)
- `chunk_size`: 청크 크기 (기본: 10000)
- `max_features`: 선택할 특성 수 (기본: 100)

## 🔧 다른 컴퓨터에서 사용하기

### 1. 파일 복사
```bash
# 필요한 파일들을 새 컴퓨터로 복사
cp portable_prediction_pipeline.py /path/to/new/computer/
cp prediction_config.json /path/to/new/computer/
cp requirements.txt /path/to/new/computer/
```

### 2. 데이터 준비
```bash
# 데이터 파일을 해당 경로에 배치
mkdir -p fintech/fintech/
cp base_test.csv fintech/fintech/
cp base_train.csv fintech/fintech/
```

### 3. 경로 수정 (필요시)
`prediction_config.json`에서 경로를 새 컴퓨터에 맞게 수정

### 4. 실행
```bash
python portable_prediction_pipeline.py
```

## 📋 필요한 패키지

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
imbalanced-learn>=0.9.0
catboost>=1.1.0
joblib>=1.2.0
```

## ⚠️ 주의사항

1. **메모리 요구사항**: 최소 4GB RAM 권장
2. **실행 시간**: 약 10-15분 (데이터 크기에 따라)
3. **파일 경로**: 설정 파일에서 경로를 정확히 지정
4. **Python 버전**: 3.8 이상 권장

## 🐛 문제 해결

### 경로 오류
- `prediction_config.json`에서 경로 확인
- 파일이 실제로 존재하는지 확인

### 메모리 부족
- `model_settings`에서 `sample_size` 줄이기
- `chunk_size` 줄이기

### 패키지 오류
- `requirements.txt`로 모든 패키지 설치 확인
- Python 버전 확인

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. 파일 경로가 올바른지
2. 필요한 패키지가 설치되었는지
3. 데이터 파일이 존재하는지
4. Python 버전이 호환되는지
