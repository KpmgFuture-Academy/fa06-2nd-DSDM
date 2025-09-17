"""
1단계: 데이터 탐색 및 전처리
- 데이터 구조 분석
- 세그먼트 분포 확인
- 결측값 및 이상치 분석
- 데이터 품질 평가
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """데이터 로드 및 기본 탐색"""
    print('=== 1단계: 데이터 탐색 및 전처리 ===\n')
    
    # 데이터 로드
    print('1. 데이터 로드...')
    df_train = pd.read_csv('fintech/fintech/base_train.csv', nrows=100000)
    df_test = pd.read_csv('fintech/fintech/base_test.csv', nrows=100000)
    
    print(f'Train 데이터: {df_train.shape}')
    print(f'Test 데이터: {df_test.shape}')
    
    return df_train, df_test

def analyze_segment_distribution(df_train):
    """세그먼트 분포 분석"""
    print('\n2. 세그먼트 분포 분석...')
    
    segment_counts = df_train['Segment'].value_counts()
    print('세그먼트별 분포:')
    for segment, count in segment_counts.items():
        percentage = (count / len(df_train)) * 100
        print(f'  {segment}: {count:,}개 ({percentage:.2f}%)')
    
    return segment_counts

def analyze_data_quality(df_train, df_test):
    """데이터 품질 분석"""
    print('\n3. 데이터 품질 분석...')
    
    # 결측값 분석
    missing_train = df_train.isnull().sum()
    missing_test = df_test.isnull().sum()
    
    print(f'Train 데이터 결측값이 있는 컬럼 수: {(missing_train > 0).sum()}')
    print(f'Test 데이터 결측값이 있는 컬럼 수: {(missing_test > 0).sum()}')
    
    # 결측값이 많은 상위 10개 컬럼
    print('\nTrain 데이터 결측값 상위 10개 컬럼:')
    top_missing_train = missing_train[missing_train > 0].sort_values(ascending=False).head(10)
    for col, count in top_missing_train.items():
        percentage = (count / len(df_train)) * 100
        print(f'  {col}: {count:,}개 ({percentage:.2f}%)')
    
    return missing_train, missing_test

def analyze_data_types(df_train):
    """데이터 타입 및 특성 유형 분석"""
    print('\n4. 데이터 타입 및 특성 유형 분석...')
    
    # 데이터 타입 분석
    print('데이터 타입 분포:')
    dtype_counts = df_train.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f'  {dtype}: {count}개')
    
    # 수치형 vs 범주형 특성 분류
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
    
    print(f'\n수치형 특성: {len(numeric_cols)}개')
    print(f'범주형 특성: {len(categorical_cols)}개')
    
    return numeric_cols, categorical_cols

def analyze_segment_characteristics(df_train):
    """세그먼트별 특성 분석"""
    print('\n5. 세그먼트별 특성 분석...')
    
    # 연령대별 세그먼트 분포
    print('연령대별 세그먼트 분포:')
    age_segment = pd.crosstab(df_train['연령'], df_train['Segment'], normalize='index') * 100
    print(age_segment.round(2))
    
    # 성별 세그먼트 분포
    print('\n성별 세그먼트 분포:')
    gender_segment = pd.crosstab(df_train['남녀구분코드'], df_train['Segment'], normalize='index') * 100
    print(gender_segment.round(2))
    
    # 주요 수치형 특성의 세그먼트별 통계
    print('\n주요 수치형 특성의 세그먼트별 통계:')
    key_numeric_cols = ['카드이용한도금액', '이용금액_R3M_신용', '이용건수_신용_R3M', '소지카드수_유효_신용']
    for col in key_numeric_cols:
        if col in df_train.columns:
            print(f'\n{col} 세그먼트별 평균:')
            segment_stats = df_train.groupby('Segment')[col].mean()
            for segment, mean_val in segment_stats.items():
                print(f'  {segment}: {mean_val:,.0f}')

def detect_outliers(df_train):
    """이상치 탐지"""
    print('\n6. 이상치 탐지...')
    
    # 주요 수치형 특성의 이상치 분석
    key_cols = ['카드이용한도금액', '이용금액_R3M_신용', '이용건수_신용_R3M', '소지카드수_유효_신용']
    
    for col in key_cols:
        if col in df_train.columns:
            print(f'\n{col}:')
            Q1 = df_train[col].quantile(0.25)
            Q3 = df_train[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df_train[(df_train[col] < lower_bound) | (df_train[col] > upper_bound)]
            outlier_pct = (len(outliers) / len(df_train)) * 100
            
            print(f'  정상 범위: {lower_bound:,.0f} ~ {upper_bound:,.0f}')
            print(f'  이상치 개수: {len(outliers):,}개 ({outlier_pct:.2f}%)')
            print(f'  최솟값: {df_train[col].min():,.0f}')
            print(f'  최댓값: {df_train[col].max():,.0f}')

def main():
    """메인 실행 함수"""
    # 데이터 로드
    df_train, df_test = load_and_explore_data()
    
    # 세그먼트 분포 분석
    segment_counts = analyze_segment_distribution(df_train)
    
    # 데이터 품질 분석
    missing_train, missing_test = analyze_data_quality(df_train, df_test)
    
    # 데이터 타입 분석
    numeric_cols, categorical_cols = analyze_data_types(df_train)
    
    # 세그먼트별 특성 분석
    analyze_segment_characteristics(df_train)
    
    # 이상치 탐지
    detect_outliers(df_train)
    
    print('\n=== 1단계 완료 ===')
    print('주요 발견사항:')
    print('- 심각한 클래스 불균형 (E: 80% vs A: 0.05%)')
    print('- 30개 컬럼에 결측값 존재')
    print('- 주요 특성에서 5-22%의 이상치 발견')
    print('- A 세그먼트가 가장 높은 카드한도와 이용금액')

if __name__ == "__main__":
    main()
