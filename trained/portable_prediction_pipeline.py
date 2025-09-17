"""
이식 가능한 예측 파이프라인
- 경로를 유연하게 처리
- 다른 컴퓨터에서도 실행 가능
- 설정 파일로 경로 관리
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 현재 스크립트의 디렉토리
SCRIPT_DIR = Path(__file__).parent.absolute()

def load_config():
    """설정 파일 로드"""
    config_file = SCRIPT_DIR / 'prediction_config.json'
    
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        # 기본 설정 생성
        config = {
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
        
        # 설정 파일 저장
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"설정 파일이 생성되었습니다: {config_file}")
        print("경로를 수정한 후 다시 실행하세요.")
        return None
    
    return config

def check_data_files(config):
    """데이터 파일 존재 확인"""
    print("=== 데이터 파일 확인 ===")
    
    base_test_path = Path(config['data_paths']['base_test_csv'])
    base_train_path = Path(config['data_paths']['base_train_csv'])
    
    # 절대 경로가 아니면 현재 디렉토리 기준으로 처리
    if not base_test_path.is_absolute():
        base_test_path = SCRIPT_DIR / base_test_path
    if not base_train_path.is_absolute():
        base_train_path = SCRIPT_DIR / base_train_path
    
    print(f"Test 데이터 경로: {base_test_path}")
    print(f"Train 데이터 경로: {base_train_path}")
    
    if not base_test_path.exists():
        print(f"❌ Test 데이터 파일을 찾을 수 없습니다: {base_test_path}")
        return False
    
    if not base_train_path.exists():
        print(f"❌ Train 데이터 파일을 찾을 수 없습니다: {base_train_path}")
        return False
    
    print("✅ 모든 데이터 파일이 존재합니다.")
    return True

def optimize_memory(df):
    """메모리 최적화"""
    print("메모리 최적화 중...")
    
    # 1. 데이터 타입 최적화
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= 0:
            if df[col].max() < 255:
                df[col] = df[col].astype('uint8')
            elif df[col].max() < 65535:
                df[col] = df[col].astype('uint16')
            else:
                df[col] = df[col].astype('uint32')
        else:
            if df[col].min() > -128 and df[col].max() < 127:
                df[col] = df[col].astype('int8')
            elif df[col].min() > -32768 and df[col].max() < 32767:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('int32')
    
    # 2. float64 → float32
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    print(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df

def load_data_in_chunks(filepath, chunk_size=10000, max_rows=50000):
    """청크 단위로 데이터 로드"""
    print(f"청크 단위로 데이터 로드 중... (청크 크기: {chunk_size})")
    
    chunks = []
    total_rows = 0
    
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        # 메모리 최적화
        chunk = optimize_memory(chunk)
        chunks.append(chunk)
        total_rows += len(chunk)
        
        if total_rows >= max_rows:
            break
    
    # 청크들을 하나로 합치기
    df = pd.concat(chunks, ignore_index=True)
    print(f"로드 완료: {df.shape}, 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

def create_features(df):
    """새로운 특성 생성"""
    df_new = df.copy()
    
    # 1. 카드 이용 효율성 특성
    if '카드이용한도금액' in df_new.columns and '이용금액_R3M_신용' in df_new.columns:
        df_new['한도소진율'] = df_new['이용금액_R3M_신용'] / (df_new['카드이용한도금액'] + 1)
    
    # 2. 거래 빈도 특성
    if '이용건수_신용_R3M' in df_new.columns and '이용금액_R3M_신용' in df_new.columns:
        df_new['평균거래금액'] = df_new['이용금액_R3M_신용'] / (df_new['이용건수_신용_R3M'] + 1)
    
    # 3. 카드 다양성 특성
    if '소지카드수_유효_신용' in df_new.columns and '소지카드수_체크' in df_new.columns:
        df_new['총카드수'] = df_new['소지카드수_유효_신용'] + df_new['소지카드수_체크']
    
    # 4. 연체 위험도 특성
    if '연체잔액_B0M' in df_new.columns and '카드이용한도금액' in df_new.columns:
        df_new['연체위험도'] = df_new['연체잔액_B0M'] / (df_new['카드이용한도금액'] + 1)
    
    return df_new

def preprocess_data(df, label_encoders=None, is_train=True):
    """데이터 전처리"""
    df_processed = df.copy()
    
    # 1. 특이값 처리 (99999999 같은 값들을 NaN으로 변경)
    for col in df_processed.select_dtypes(include=[np.number]).columns:
        if df_processed[col].max() > 9999999:
            df_processed[col] = df_processed[col].replace(99999999, np.nan)
            df_processed[col] = df_processed[col].replace(999999, np.nan)
    
    # 2. 새로운 특성 생성
    df_processed = create_features(df_processed)
    
    # 3. 범주형 변수 인코딩
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    if is_train and 'Segment' in categorical_cols:
        categorical_cols.remove('Segment')
    
    if label_encoders is None:
        label_encoders = {}
    
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna('Unknown')
            df_processed[col] = df_processed[col].astype(str)
            
            if col not in label_encoders:
                from sklearn.preprocessing import LabelEncoder
                label_encoders[col] = LabelEncoder()
                df_processed[col] = label_encoders[col].fit_transform(df_processed[col])
            else:
                try:
                    df_processed[col] = label_encoders[col].transform(df_processed[col])
                except ValueError:
                    # 새로운 값이 있으면 'Unknown'으로 처리
                    unseen_labels = set(df_processed[col]) - set(label_encoders[col].classes_)
                    if unseen_labels:
                        df_processed[col] = df_processed[col].replace(list(unseen_labels), 'Unknown')
                        if 'Unknown' not in label_encoders[col].classes_:
                            from sklearn.preprocessing import LabelEncoder
                            all_values = list(label_encoders[col].classes_) + ['Unknown']
                            new_encoder = LabelEncoder()
                            new_encoder.fit(all_values)
                            label_encoders[col] = new_encoder
                        df_processed[col] = label_encoders[col].transform(df_processed[col])
    
    return df_processed, label_encoders

def train_model(config):
    """모델 학습"""
    print("=== 모델 학습 ===")
    
    # 데이터 경로 설정
    base_train_path = Path(config['data_paths']['base_train_csv'])
    if not base_train_path.is_absolute():
        base_train_path = SCRIPT_DIR / base_train_path
    
    # 데이터 로드
    print("Train 데이터 로드 중...")
    df_train = load_data_in_chunks(
        base_train_path, 
        chunk_size=config['model_settings']['chunk_size'],
        max_rows=config['model_settings']['sample_size']
    )
    
    # 데이터 전처리
    print("데이터 전처리 중...")
    df_processed, label_encoders = preprocess_data(df_train, is_train=True)
    
    # 특성과 타겟 분리
    X = df_processed.drop('Segment', axis=1)
    y = df_processed['Segment']
    
    # 결측값 및 무한대 값 처리
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # 특성 선택
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(score_func=f_classif, k=config['model_settings']['max_features'])
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    # 언더샘플링
    from imblearn.under_sampling import RandomUnderSampler
    class_counts = y.value_counts()
    sampling_strategy = {}
    for segment in ['A', 'B', 'C', 'D', 'E']:
        if segment in class_counts:
            original_count = class_counts[segment]
            target_count = min(int(original_count * 0.8), 5000)
            target_count = min(target_count, original_count)
            target_count = max(target_count, 1)
            sampling_strategy[segment] = target_count
    
    undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    X_balanced, y_balanced = undersampler.fit_resample(X_selected, y)
    
    # 스케일링
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)
    
    # 모델 학습
    print("CatBoost 모델 학습 중...")
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
        random_state=42,
        verbose=False,
        iterations=1000,
        depth=6,
        learning_rate=0.1
    )
    model.fit(X_scaled, y_balanced)
    
    # 모델 저장
    model_data = {
        'model': model,
        'scaler': scaler,
        'selector': selector,
        'undersampler': undersampler,
        'label_encoders': label_encoders,
        'selected_features': selected_features
    }
    
    model_path = SCRIPT_DIR / config['output_paths']['model_pkl']
    import joblib
    joblib.dump(model_data, model_path)
    print(f"모델이 저장되었습니다: {model_path}")
    
    return model_data

def predict_test_data(config, model_data):
    """Test 데이터 예측"""
    print("=== Test 데이터 예측 ===")
    
    # 데이터 경로 설정
    base_test_path = Path(config['data_paths']['base_test_csv'])
    if not base_test_path.is_absolute():
        base_test_path = SCRIPT_DIR / base_test_path
    
    # 데이터 로드
    print("Test 데이터 로드 중...")
    df_test = load_data_in_chunks(
        base_test_path,
        chunk_size=config['model_settings']['chunk_size'],
        max_rows=config['model_settings']['sample_size']
    )
    
    # 데이터 전처리
    print("Test 데이터 전처리 중...")
    df_test_processed, _ = preprocess_data(df_test, model_data['label_encoders'], is_train=False)
    
    # 특성 선택 및 스케일링
    X_test = df_test_processed[model_data['selected_features']]
    X_test = X_test.fillna(X_test.median())
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.fillna(X_test.median())
    
    X_test_scaled = model_data['scaler'].transform(X_test)
    
    # 예측 수행
    print("예측 수행 중...")
    predictions = model_data['model'].predict(X_test_scaled)
    
    # 결과 분석
    unique, counts = np.unique(predictions, return_counts=True)
    print("\n예측된 세그먼트 분포:")
    for segment, count in zip(unique, counts):
        percentage = (count / len(predictions)) * 100
        print(f"  {segment}: {count:,}개 ({percentage:.2f}%)")
    
    # 결과 저장
    # 세그먼트 데이터 정리 (['E'] 형태를 E로 변환)
    segments_clean = [str(seg).strip("[]'") for seg in predictions]
    
    result_df = pd.DataFrame({
        'ID': list(range(len(predictions))),
        'Segment': segments_clean
    })
    
    output_path = SCRIPT_DIR / config['output_paths']['predictions_csv']
    result_df.to_csv(output_path, index=False)
    print(f"\n결과가 저장되었습니다: {output_path}")
    
    return result_df

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("이식 가능한 예측 파이프라인")
    print("=" * 80)
    
    try:
        # 1. 설정 로드
        config = load_config()
        if config is None:
            return
        
        # 2. 데이터 파일 확인
        if not check_data_files(config):
            return
        
        # 3. 모델 학습
        model_data = train_model(config)
        
        # 4. Test 데이터 예측
        result_df = predict_test_data(config, model_data)
        
        print("\n" + "=" * 80)
        print("파이프라인 실행 완료!")
        print("=" * 80)
        print(f"생성된 파일:")
        print(f"  - {config['output_paths']['model_pkl']}: 학습된 모델")
        print(f"  - {config['output_paths']['predictions_csv']}: 예측 결과")
        print(f"\n예측 결과 요약:")
        print(f"  - 총 샘플 수: {len(result_df):,}")
        print(f"  - 세그먼트 수: {len(result_df['Segment'].unique())}")
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
