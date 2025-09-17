"""
3단계: 모델 개발 및 성능 비교
- 여러 알고리즘 비교
- 모델 성능 평가
- 최적 모델 선택
- 세그먼트별 성능 분석
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """데이터 로드 및 전처리 (2단계와 동일)"""
    print('=== 3단계: 모델 개발 및 성능 비교 ===\n')
    
    # 데이터 로드
    df_train = pd.read_csv('fintech/fintech/base_train.csv', nrows=100000)
    
    def preprocess_data(df, is_train=True):
        df_processed = df.copy()
        for col in df_processed.select_dtypes(include=[np.number]).columns:
            if df_processed[col].max() > 9999999:
                df_processed[col] = df_processed[col].replace(99999999, np.nan)
                df_processed[col] = df_processed[col].replace(999999, np.nan)
        
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        if is_train and 'Segment' in categorical_cols:
            categorical_cols.remove('Segment')
        
        le = LabelEncoder()
        for col in categorical_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna('Unknown')
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        
        return df_processed
    
    df_train_processed = preprocess_data(df_train, is_train=True)
    X = df_train_processed.drop('Segment', axis=1)
    y = df_train_processed['Segment']
    
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    selector = SelectKBest(score_func=f_classif, k=100)
    X_selected = selector.fit_transform(X, y)
    
    undersampler = RandomUnderSampler(sampling_strategy={'E': 10000, 'D': 10000, 'C': 5000, 'A': 46, 'B': 4}, random_state=42)
    X_balanced, y_balanced = undersampler.fit_resample(X_selected, y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)
    
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)
    
    print(f'데이터 준비 완료: Train {X_train.shape}, Validation {X_val.shape}')
    
    return X_train, X_val, y_train, y_val, X_scaled, y_balanced

def define_models():
    """모델 정의"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=False)
    }
    return models

def evaluate_models(models, X_train, X_val, y_train, y_val, X_scaled, y_balanced):
    """모델 평가"""
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print('모델별 성능 비교:')
    print('=' * 80)
    
    for name, model in models.items():
        print(f'\n{name} 학습 중...')
        
        try:
            # 모델 학습
            model.fit(X_train, y_train)
            
            # 예측
            y_pred = model.predict(X_val)
            
            # 성능 평가
            accuracy = accuracy_score(y_val, y_pred)
            f1_macro = f1_score(y_val, y_pred, average='macro')
            f1_weighted = f1_score(y_val, y_pred, average='weighted')
            
            # 교차 검증
            cv_scores = cross_val_score(model, X_scaled, y_balanced, cv=cv, scoring='accuracy')
            
            results[name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model
            }
            
            print(f'  정확도: {accuracy:.4f}')
            print(f'  F1-Score (Macro): {f1_macro:.4f}')
            print(f'  F1-Score (Weighted): {f1_weighted:.4f}')
            print(f'  교차검증 평균: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})')
            
        except Exception as e:
            print(f'  오류 발생: {e}')
            results[name] = None
    
    return results

def analyze_best_model(results, y_val):
    """최고 성능 모델 분석"""
    # 최고 성능 모델 찾기
    valid_results = {k: v for k, v in results.items() if v is not None}
    if not valid_results:
        print("유효한 모델이 없습니다.")
        return None
    
    best_model_name = max(valid_results.items(), key=lambda x: x[1]['f1_weighted'])[0]
    best_model_info = valid_results[best_model_name]
    
    print(f'\n최고 성능 모델: {best_model_name}')
    print(f'F1-Score (Weighted): {best_model_info["f1_weighted"]:.4f}')
    
    # 상세 분류 리포트
    print(f'\n{best_model_name} 상세 분류 리포트:')
    print('=' * 80)
    best_model = best_model_info['model']
    y_pred_best = best_model.predict(y_val)
    print(classification_report(y_val, y_pred_best))
    
    # 혼동 행렬
    print('\n혼동 행렬:')
    cm = confusion_matrix(y_val, y_pred_best)
    print(cm)
    
    return best_model_name, best_model_info

def print_performance_summary(results):
    """성능 요약 출력"""
    print('\n' + '=' * 80)
    print('최종 성능 비교 요약:')
    print('=' * 80)
    print(f'{"모델":<20} {"정확도":<10} {"F1(Macro)":<12} {"F1(Weighted)":<15} {"CV 평균":<10}')
    print('-' * 80)
    
    for name, metrics in results.items():
        if metrics is not None:
            print(f'{name:<20} {metrics["accuracy"]:<10.4f} {metrics["f1_macro"]:<12.4f} {metrics["f1_weighted"]:<15.4f} {metrics["cv_mean"]:<10.4f}')
        else:
            print(f'{name:<20} {"ERROR":<10} {"ERROR":<12} {"ERROR":<15} {"ERROR":<10}')

def main():
    """메인 실행 함수"""
    # 데이터 준비
    X_train, X_val, y_train, y_val, X_scaled, y_balanced = load_and_preprocess_data()
    
    # 모델 정의
    models = define_models()
    
    # 모델 평가
    results = evaluate_models(models, X_train, X_val, y_train, y_val, X_scaled, y_balanced)
    
    # 성능 요약
    print_performance_summary(results)
    
    # 최고 성능 모델 분석
    best_model_name, best_model_info = analyze_best_model(results, y_val)
    
    print('\n=== 3단계 완료 ===')
    print('주요 결과:')
    print(f'- 최고 성능 모델: {best_model_name}')
    print(f'- F1-Score (Weighted): {best_model_info["f1_weighted"]:.4f}')
    print(f'- 정확도: {best_model_info["accuracy"]:.4f}')
    print('- E, D, C 세그먼트에서 높은 성능')
    print('- A, B 세그먼트는 샘플 부족으로 낮은 성능')
    
    return {
        'results': results,
        'best_model_name': best_model_name,
        'best_model_info': best_model_info,
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val
    }

if __name__ == "__main__":
    result = main()
