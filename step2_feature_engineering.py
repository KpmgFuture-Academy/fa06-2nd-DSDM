"""
2단계: 특성 엔지니어링 및 전처리 전략
- 데이터 전처리 파이프라인
- 새로운 특성 생성
- 특성 선택
- 클래스 불균형 해결
- 데이터 스케일링
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

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

def preprocess_data(df, is_train=True):
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
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    if is_train and 'Segment' in categorical_cols:
        categorical_cols.remove('Segment')
    
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna('Unknown')
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    return df_processed

def select_features(X, y, k=100):
    """중요 특성 선택"""
    print(f'\n중요 특성 선택 (상위 {k}개)...')
    
    # F-test를 사용한 특성 선택
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # 선택된 특성 이름
    selected_features = X.columns[selector.get_support()].tolist()
    print(f'선택된 특성 수: {len(selected_features)}')
    
    # 상위 15개 중요 특성 출력
    feature_scores = selector.scores_
    feature_names = X.columns
    feature_importance = list(zip(feature_names, feature_scores))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print('\n상위 15개 중요 특성:')
    for i, (feature, score) in enumerate(feature_importance[:15]):
        print(f'  {i+1:2d}. {feature}: {score:.2f}')
    
    return X_selected, selected_features, selector

def balance_classes(X_selected, y):
    """클래스 불균형 해결"""
    print('\n클래스 불균형 해결 (언더샘플링)...')
    
    # 원본 분포
    print('원본 분포:')
    segment_counts = y.value_counts()
    for segment, count in segment_counts.items():
        percentage = (count / len(y)) * 100
        print(f'  {segment}: {count:,}개 ({percentage:.2f}%)')
    
    # 언더샘플링 전략
    undersampler = RandomUnderSampler(sampling_strategy={'E': 10000, 'D': 10000, 'C': 5000, 'A': 46, 'B': 4}, random_state=42)
    X_balanced, y_balanced = undersampler.fit_resample(X_selected, y)
    
    print('\n언더샘플링 후 분포:')
    segment_counts_balanced = pd.Series(y_balanced).value_counts()
    for segment, count in segment_counts_balanced.items():
        percentage = (count / len(y_balanced)) * 100
        print(f'  {segment}: {count:,}개 ({percentage:.2f}%)')
    
    return X_balanced, y_balanced, undersampler

def scale_data(X_balanced, X_test_selected):
    """데이터 스케일링"""
    print('\n데이터 스케일링...')
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)
    X_test_scaled = scaler.transform(X_test_selected)
    
    print(f'스케일링 후 Train 데이터: {X_scaled.shape}')
    print(f'스케일링 후 Test 데이터: {X_test_scaled.shape}')
    
    return X_scaled, X_test_scaled, scaler

def main():
    """메인 실행 함수"""
    print('=== 2단계: 특성 엔지니어링 및 전처리 전략 ===\n')
    
    # 데이터 로드
    print('1. 데이터 로드...')
    df_train = pd.read_csv('fintech/fintech/base_train.csv', nrows=100000)
    df_test = pd.read_csv('fintech/fintech/base_test.csv', nrows=100000)
    
    print(f'Train 데이터: {df_train.shape}')
    print(f'Test 데이터: {df_test.shape}')
    
    # 전처리 실행
    print('\n2. 데이터 전처리 실행...')
    df_train_processed = preprocess_data(df_train, is_train=True)
    df_test_processed = preprocess_data(df_test, is_train=False)
    
    print(f'전처리 후 Train 데이터: {df_train_processed.shape}')
    print(f'전처리 후 Test 데이터: {df_test_processed.shape}')
    
    # 특성과 타겟 분리
    X = df_train_processed.drop('Segment', axis=1)
    y = df_train_processed['Segment']
    
    # 결측값 및 무한대 값 처리
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Test 데이터도 동일하게 처리
    X_test = df_test_processed.fillna(df_test_processed.median())
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.fillna(X_test.median())
    
    # 특성 선택
    X_selected, selected_features, selector = select_features(X, y, k=100)
    X_test_selected = X_test[selected_features]
    
    # 클래스 불균형 해결
    X_balanced, y_balanced, undersampler = balance_classes(X_selected, y)
    
    # 스케일링
    X_scaled, X_test_scaled, scaler = scale_data(X_balanced, X_test_selected)
    
    print('\n=== 2단계 완료 ===')
    print('전처리 결과:')
    print(f'- 원본 특성: 857개 → 선택된 특성: 100개')
    print(f'- 샘플 수: 100,000개 → 25,050개 (언더샘플링)')
    print(f'- 새로운 특성 4개 생성')
    print(f'- 스케일링 적용 완료')
    
    return {
        'X_scaled': X_scaled,
        'y_balanced': y_balanced,
        'X_test_scaled': X_test_scaled,
        'selected_features': selected_features,
        'scaler': scaler,
        'selector': selector,
        'undersampler': undersampler
    }

if __name__ == "__main__":
    result = main()
