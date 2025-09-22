"""
신용카드 세그먼트 분석 대시보드 공통 유틸리티 함수들
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Streamlit Cloud 호환성을 위한 디바이스 설정
TORCH_AVAILABLE = False  # Streamlit Cloud에서는 PyTorch 사용 안함

# 전역 변수
_CACHED_DATA = None
_DEVICE = None

def _get_device():
    """CPU 디바이스 설정 (Streamlit Cloud 호환)"""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = "cpu"
    return _DEVICE

def get_device_info():
    """현재 디바이스 정보 반환 (Streamlit Cloud 호환)"""
    device = _get_device()
    return {
        'device': device,
        'device_name': 'CPU (Streamlit Cloud)',
        'device_count': 0,
        'memory_total': 0,
        'memory_allocated': 0,
        'memory_cached': 0,
        'cuda_version': 'N/A',
        'torch_version': 'Not installed'
    }

def gpu_accelerated_computation(data: np.ndarray, operation: str = 'matrix_multiply') -> np.ndarray:
    """CPU 계산 (Streamlit Cloud 호환)"""
    # Streamlit Cloud에서는 CPU만 사용
    return data

# 상수 정의
SEGMENT_ORDER = ['A', 'B', 'C', 'D', 'E']
SEGMENT_COLORS = {
    'A': '#E74C3C',  # 빨강
    'B': '#E67E22',  # 주황
    'C': '#3498DB',  # 파랑
    'D': '#2ECC71',  # 초록
    'E': '#F4D03F'   # 노랑
}

# 컬럼 매핑 (실제 컬럼명 → 표준 컬럼명)
COLUMN_MAPPING = {
    # 기본 정보
    'Segment': 'Segment',
    '기준년월': 'Date',
    'ID': 'ID',
    '연령': 'Age',
    '거주시도명': 'Region',
    
    # 이용/성과
    '이용금액_일시불_B0M': '이용금액_일시불_B0M',
    '이용금액_할부_B0M': '이용금액_할부_B0M',
    '이용금액_체크_B0M': '이용금액_체크_B0M',
    '이용금액_CA_B0M': '이용금액_CA_B0M',
    '이용금액_카드론_B0M': '이용금액_카드론_B0M',
    '잔액_현금서비스_B0M': '잔액_현금서비스_B0M',
    '잔액_카드론_B0M': '잔액_카드론_B0M',
    
    # 리스크
    '승인거절건수_B0M': '승인거절건수_B0M',
    '연체잔액_B0M': '연체잔액_B0M',
    '카드이용한도금액': '카드이용한도액',
    
    # 참여/혜택
    '포인트_적립_B0M': '포인트_적립_B0M',
    '포인트_소멸_B0M': '포인트_소멸_B0M',
    '혜택수혜율_B0M': '혜택수혜율_B0M',
}

# Google Drive 파일 ID
FILE_ID = "16KpMgqyfVtOaOX30kqPCu1pc9T3d7f-k"

# 다양한 다운로드 URL 시도
DOWNLOAD_URLS = [
    f"https://drive.google.com/uc?export=download&id={FILE_ID}&confirm=t",
    f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t",
    f"https://drive.google.com/uc?export=download&id={FILE_ID}",
    f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download"
]


def generate_sample_data() -> pd.DataFrame:
    """
    샘플 데이터 생성 (Streamlit Cloud 호환용)
    """
    np.random.seed(42)
    n_samples = 10000
    
    data = {
        'ID': [f'CUST_{i:06d}' for i in range(1, n_samples + 1)],
        'Segment': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples, p=[0.05, 0.15, 0.30, 0.35, 0.15]),
        'Age': np.random.randint(20, 70, n_samples),
        'Gender': np.random.choice(['M', 'F'], n_samples),
        'Region': np.random.choice(['서울', '경기', '인천', '부산', '대구', '기타'], n_samples, p=[0.3, 0.25, 0.1, 0.1, 0.1, 0.15]),
        '총이용금액_B0M': np.random.lognormal(8, 1.5, n_samples),
        '총이용건수_B0M': np.random.poisson(15, n_samples),
        '카드이용한도액': np.random.lognormal(10, 1, n_samples),
        '연체여부': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'Date': pd.date_range('2023-01-01', '2023-12-31', periods=n_samples)
    }
    
    df = pd.DataFrame(data)
    df['ARPU'] = df['총이용금액_B0M'] / df['총이용건수_B0M']
    df['이용률'] = (df['총이용금액_B0M'] / df['카드이용한도액']) * 100
    
    # AgeGroup 컬럼 생성
    df['AgeGroup'] = pd.cut(df['Age'], 
                           bins=[0, 20, 30, 40, 50, 60, 100], 
                           labels=['20대미만', '20대', '30대', '40대', '50대', '60대이상'])
    
    return df


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    데이터 로드 및 기본 전처리
    Google Drive에서만 데이터 로드 (더미 데이터 없음)
    """
    import requests
    import os
    import zipfile
    from io import BytesIO
    
    # 로컬 테스트 모드 감지
    local_test_mode = os.environ.get('STREAMLIT_LOCAL_TEST', 'false').lower() == 'true'
    
    if local_test_mode:
        print("로컬 테스트 모드: 샘플 데이터 사용")
        df = generate_sample_data()
        df = map_columns(df)
        return df
    
    df = None
    last_error = None
    
    # 여러 URL 시도
    for i, url in enumerate(DOWNLOAD_URLS):
        try:
            print(f"시도 {i+1}: {url}")
            
            # 세션 생성
            session = requests.Session()
            response = session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Content-Type 확인
            content_type = response.headers.get('content-type', '').lower()
            print(f"Content-Type: {content_type}")
            
            # HTML 응답인지 확인
            if 'text/html' in content_type:
                print("HTML 응답 감지, 다음 URL 시도")
                continue
            
            # Content-Length 확인
            content_length = response.headers.get('content-length')
            if content_length:
                file_size = int(content_length)
                print(f"예상 파일 크기: {file_size:,} bytes")
                
                if file_size < 1000:  # 너무 작으면 HTML 페이지일 가능성
                    print("파일이 너무 작음, 다음 URL 시도")
                    continue
            else:
                print("Content-Length 헤더 없음, 다운로드 후 확인")
            
            # 파일 다운로드
            temp_file = f"temp_data_{i}.csv"
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # 실제 파일 크기 확인
            actual_file_size = os.path.getsize(temp_file)
            print(f"실제 다운로드된 파일 크기: {actual_file_size:,} bytes")
            
            if actual_file_size < 1000:  # 너무 작으면 HTML 페이지일 가능성
                print("파일이 너무 작음, 다음 URL 시도")
                os.remove(temp_file)
                continue
            
            # 바이러스 스캔 경고 페이지 확인 (파일 크기가 작을 때만)
            if actual_file_size < 10000:  # 10KB 미만일 때만 HTML 내용 확인
                content_preview = response.content[:1000].decode('utf-8', errors='ignore')
                if 'virus scan' in content_preview.lower() or 'virus warning' in content_preview.lower():
                    print("바이러스 스캔 경고 페이지 감지, 다음 URL 시도")
                    os.remove(temp_file)
                    continue
            
            # 파일 타입 감지 및 처리
            with open(temp_file, 'rb') as f:
                header = f.read(4)
            
            if header.startswith(b'PK'):
                # ZIP 파일 처리
                print("ZIP 파일 감지")
                with zipfile.ZipFile(temp_file, 'r') as zip_file:
                    csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
                    if not csv_files:
                        raise Exception("ZIP 파일에서 CSV 파일을 찾을 수 없습니다.")
                    
                    csv_file = csv_files[0]
                    print(f"CSV 파일 발견: {csv_file}")
                    
                    with zip_file.open(csv_file) as f:
                        df = pd.read_csv(f, low_memory=False, encoding='utf-8')
            else:
                # CSV 파일 직접 읽기
                print("CSV 파일 직접 읽기")
                df = pd.read_csv(temp_file, low_memory=False, encoding='utf-8')
            
            # 임시 파일 삭제
            os.remove(temp_file)
            
            if df is not None and not df.empty:
                print(f"✅ 데이터 로드 성공! {len(df)}행, {len(df.columns)}열")
                break
            else:
                print("데이터가 비어있음, 다음 URL 시도")
                continue
                
        except Exception as e:
            last_error = e
            print(f"❌ URL {i+1} 실패: {str(e)}")
            if os.path.exists(f"temp_data_{i}.csv"):
                os.remove(f"temp_data_{i}.csv")
            continue
    
    if df is None or df.empty:
        # 로컬 테스트 모드 감지
        import os
        local_test_mode = os.environ.get('STREAMLIT_LOCAL_TEST', 'false').lower() == 'true'
        
        if local_test_mode:
            print("로컬 테스트 모드: 샘플 데이터 사용")
            df = generate_sample_data()
        else:
            error_msg = f"Google Drive에서 데이터를 로드할 수 없습니다.\n"
            error_msg += f"시도한 URL 수: {len(DOWNLOAD_URLS)}\n"
            error_msg += f"마지막 오류: {str(last_error) if last_error else '알 수 없음'}\n"
            error_msg += f"파일 ID: {FILE_ID}\n"
            error_msg += "Google Drive 링크를 확인해주세요: https://drive.google.com/file/d/16KpMgqyfVtOaOX30kqPCu1pc9T3d7f-k/view?usp=sharing"
            raise Exception(error_msg)
    
    # 중복 인덱스 제거
    df = df.reset_index(drop=True)
    
    # 컬럼 매핑 적용
    return map_columns(df)


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    실제 컬럼명을 표준 컬럼명으로 매핑
    """
    # 중복 컬럼 제거
    df = df.loc[:, ~df.columns.duplicated()]
    
    # 매핑이 필요한 컬럼만 변경
    rename_dict = {k: v for k, v in COLUMN_MAPPING.items() if k in df.columns}
    df = df.rename(columns=rename_dict)
    
    # 파생 컬럼 생성
    # 총이용금액 계산
    amount_columns = ['이용금액_일시불_B0M', '이용금액_할부_B0M', '이용금액_체크_B0M', 
                     '이용금액_CA_B0M', '이용금액_카드론_B0M']
    
    total_amount = pd.Series(0, index=df.index)
    for col in amount_columns:
        if col in df.columns:
            total_amount += df[col].fillna(0)
    
    if total_amount.sum() > 0:
        df['총이용금액_B0M'] = total_amount
    
    # 총이용건수 계산
    count_columns = ['이용건수_일시불_B0M', '이용건수_할부_B0M', '이용건수_체크_B0M']
    
    total_count = pd.Series(0, index=df.index)
    for col in count_columns:
        if col in df.columns:
            total_count += df[col].fillna(0)
    
    if total_count.sum() > 0:
        df['총이용건수_B0M'] = total_count
    
    # 연체 여부 생성
    if '연체잔액_B0M' in df.columns:
        df['연체여부'] = (df['연체잔액_B0M'] > 0).astype(int)
    elif '연체여부' in df.columns:
        df['연체여부'] = (df['연체여부'] > 0).astype(int)
    
    # Segment 컬럼 확인 및 생성
    if 'Segment' not in df.columns:
        # 가상 세그먼트 생성 (EDA 결과 반영)
        segment_probs = [0.0004, 0.00001, 0.053, 0.135, 0.811]  # A, B, C, D, E 비율
        df['Segment'] = np.random.choice(['A', 'B', 'C', 'D', 'E'], len(df), p=segment_probs)
    
    # 세그먼트 카테고리화
    try:
        df['Segment'] = pd.Categorical(df['Segment'], categories=SEGMENT_ORDER, ordered=True)
    except:
        # 기본 세그먼트 설정
        df['Segment'] = 'E'
    
    # AgeGroup 컬럼 생성
    if 'Age' in df.columns:
        try:
            df['AgeGroup'] = pd.cut(df['Age'], 
                                   bins=[0, 20, 30, 40, 50, 60, 100], 
                                   labels=['20대미만', '20대', '30대', '40대', '50대', '60대이상'])
        except:
            # 기본 연령대 설정
            df['AgeGroup'] = '30대'
    else:
        # Age 컬럼이 없으면 기본값 설정
        df['AgeGroup'] = '30대'
    
    # 누락된 컬럼에 기본값 설정
    required_columns = ['총이용금액_B0M', '총이용건수_B0M', '연체여부', '카드이용한도액']
    for col in required_columns:
        if col not in df.columns:
            if col == '연체여부':
                df[col] = 0
            else:
                df[col] = 100000  # 기본값
    
    return df

def apply_filters(df: pd.DataFrame, 
                 date_range: Optional[Tuple], 
                 age_groups: Optional[List], 
                 regions: Optional[List],
                 segments: Optional[List]) -> pd.DataFrame:
    """
    데이터에 필터 적용
    """
    filtered_df = df.copy()
    
    # 날짜 필터
    if date_range:
        # date 타입을 datetime으로 변환
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        
        filtered_df = filtered_df[
            (filtered_df['Date'] >= start_date) & 
            (filtered_df['Date'] <= end_date)
        ]
    
    # 연령대 필터
    if age_groups:
        filtered_df = filtered_df[filtered_df['AgeGroup'].isin(age_groups)]
    
    # 지역 필터
    if regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    # 세그먼트 필터
    if segments:
        filtered_df = filtered_df[filtered_df['Segment'].isin(segments)]
    
    return filtered_df

def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    주요 KPI 계산
    """
    if df.empty:
        return pd.DataFrame()
    
    # 세그먼트별 집계
    kpi_df = df.groupby('Segment', observed=True).agg({
        'ID': 'nunique',
        '총이용금액_B0M': 'sum',
        '총이용건수_B0M': 'sum',
        '카드이용한도액': 'sum',
        '연체여부': 'mean',
        '포인트_적립_B0M': 'sum',
        '포인트_소멸_B0M': 'sum'
    }).rename(columns={'ID': '고객수'})
    
    # 파생 지표 계산
    kpi_df['ARPU_월'] = kpi_df['총이용금액_B0M'] / kpi_df['고객수']
    kpi_df['객단가'] = kpi_df['총이용금액_B0M'] / kpi_df['총이용건수_B0M']
    kpi_df['이용률_한도대비'] = kpi_df['총이용금액_B0M'] / kpi_df['카드이용한도액']
    kpi_df['연체율'] = kpi_df['연체여부'] * 100
    
    # 무한대/NaN 처리
    kpi_df = kpi_df.replace([np.inf, -np.inf], np.nan)
    
    return kpi_df.reset_index()

def create_segment_colors(segments: List[str]) -> Dict[str, str]:
    """
    세그먼트별 색상 딕셔너리 생성
    """
    return {seg: SEGMENT_COLORS.get(seg, '#95A5A6') for seg in segments}

def format_number(value: float, unit: str = '') -> str:
    """
    숫자 포맷팅 (천단위 콤마, k/M 단위)
    """
    if pd.isna(value):
        return "N/A"
    
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M{unit}"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.1f}k{unit}"
    else:
        return f"{value:,.0f}{unit}"

def create_metric_card(title: str, value: float, delta: Optional[float] = None, 
                      format_func: callable = None, unit: str = "") -> None:
    """
    메트릭 카드 생성
    """
    if format_func:
        formatted_value = format_func(value)
    else:
        formatted_value = format_number(value)
    
    if unit:
        formatted_value = f"{formatted_value}{unit}"
    
    delta_text = None
    if delta is not None:
        delta_text = f"{delta:+.1f}%"
    
    st.metric(
        label=title,
        value=formatted_value,
        delta=delta_text
    )

def create_segment_chart(data: pd.DataFrame, 
                        x_col: str, 
                        y_col: str, 
                        chart_type: str = 'bar',
                        title: str = '',
                        height: int = 400) -> go.Figure:
    """
    세그먼트별 차트 생성 (공통 스타일 적용)
    """
    # 세그먼트 순서 보장
    data = data.sort_values('Segment')
    
    # 색상 설정
    colors = create_segment_colors(data['Segment'].tolist())
    
    if chart_type == 'bar':
        fig = px.bar(
            data, 
            x=x_col, 
            y=y_col, 
            color='Segment',
            title=title,
            color_discrete_map=colors,
            category_orders={'Segment': SEGMENT_ORDER}
        )
    elif chart_type == 'pie':
        fig = px.pie(
            data, 
            values=y_col, 
            names=x_col,
            title=title,
            color_discrete_map=colors,
            category_orders={'Segment': SEGMENT_ORDER}
        )
    elif chart_type == 'line':
        fig = px.line(
            data, 
            x=x_col, 
            y=y_col, 
            color='Segment',
            title=title,
            color_discrete_map=colors
        )
    
    # 공통 스타일 적용
    fig.update_layout(
        height=height,
        font_size=12,
        title_font_size=16,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig

def create_global_filters(df: pd.DataFrame) -> Dict:
    """
    글로벌 필터 UI 생성
    """
    st.sidebar.header("🔍 글로벌 필터")
    
    # 날짜 범위
    date_min = df['Date'].min()
    date_max = df['Date'].max()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("시작일", value=date_min.date())
    with col2:
        end_date = st.date_input("종료일", value=date_max.date())
    
    # 연령대
    age_groups = st.sidebar.multiselect(
        "연령대",
        options=sorted(df['AgeGroup'].dropna().unique().tolist()),
        default=sorted(df['AgeGroup'].dropna().unique().tolist())
    )
    
    # 지역
    regions = st.sidebar.multiselect(
        "지역",
        options=sorted(df['Region'].dropna().unique().tolist()),
        default=sorted(df['Region'].dropna().unique().tolist())
    )
    
    # 세그먼트
    segments = st.sidebar.multiselect(
        "세그먼트",
        options=SEGMENT_ORDER,
        default=SEGMENT_ORDER
    )
    
    # 필터 초기화 버튼
    if st.sidebar.button("필터 초기화"):
        st.rerun()
    
    return {
        'date_range': (start_date, end_date),
        'age_groups': age_groups,
        'regions': regions,
        'segments': segments
    }

def download_data_button(df: pd.DataFrame, filename: str = "dashboard_data.csv") -> None:
    """
    데이터 다운로드 버튼 생성
    """
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 현재 뷰 데이터 다운로드",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

def calculate_kpi_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """KPI 메트릭 계산"""
    # 기본 집계
    kpi_data = df.groupby('Segment', observed=True).agg({
        'ID': 'nunique',
        '총이용금액_B0M': ['sum', 'mean'],
        '총이용건수_B0M': ['sum', 'mean'],
        '카드이용한도액': 'mean',
        '연체여부': 'mean'
    }).round(2)
    
    # 컬럼명 정리
    kpi_data.columns = ['고객수', '총이용금액', 'ARPU', '총이용건수', '객단가', '평균한도', '연체율']
    kpi_data['연체율'] = kpi_data['연체율'] * 100
    
    # 추가 지표 계산
    kpi_data['이용률'] = (kpi_data['총이용금액'] / kpi_data['평균한도']) * 100
    
    # 승인거절률 (가상 데이터)
    kpi_data['승인거절률'] = np.random.normal(5, 2, len(kpi_data))
    kpi_data['승인거절률'] = np.maximum(0, kpi_data['승인거절률'])
    
    # 전월 대비 증감률 (가상 데이터)
    kpi_data['ARPU_증감'] = np.random.normal(0, 5, len(kpi_data))
    kpi_data['객단가_증감'] = np.random.normal(0, 3, len(kpi_data))
    kpi_data['총이용금액_증감'] = np.random.normal(0, 8, len(kpi_data))
    kpi_data['총이용건수_증감'] = np.random.normal(0, 6, len(kpi_data))
    kpi_data['연체율_증감'] = np.random.normal(0, 2, len(kpi_data))
    kpi_data['승인거절률_증감'] = np.random.normal(0, 1, len(kpi_data))
    kpi_data['이용률_증감'] = np.random.normal(0, 4, len(kpi_data))
    
    return kpi_data.reset_index()

def prepare_trend_data(df: pd.DataFrame) -> pd.DataFrame:
    """트렌드 분석용 데이터 준비"""
    if df.empty:
        return pd.DataFrame()
    
    # 필요한 컬럼들이 있는지 확인하고 생성
    trend_df = df.copy()
    
    # Date 컬럼 처리
    if 'Date' not in trend_df.columns:
        # 가상 날짜 생성 (최근 12개월)
        import pandas as pd
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start_date, end_date, freq='MS')  # 월 시작일
        trend_df['Date'] = np.random.choice(dates, len(trend_df))
    
    # 월별 데이터 집계를 위한 컬럼 추가
    trend_df['YearMonth'] = trend_df['Date'].dt.to_period('M')
    
    # 필요한 메트릭 컬럼들 확인 및 생성
    required_metrics = ['총이용금액_B0M', '총이용건수_B0M', '연체율']
    
    for metric in required_metrics:
        if metric not in trend_df.columns:
            if metric == '총이용금액_B0M':
                trend_df[metric] = np.random.normal(500000, 200000, len(trend_df))
                trend_df[metric] = np.maximum(0, trend_df[metric])
            elif metric == '총이용건수_B0M':
                trend_df[metric] = np.random.poisson(50, len(trend_df))
            elif metric == '연체율':
                trend_df[metric] = np.random.beta(2, 98, len(trend_df)) * 100  # 0-100%
    
    return trend_df

def render_trend_controls(trend_data: pd.DataFrame):
    """트렌드 분석 컨트롤 패널"""
    st.markdown("#### 🎛️ 분석 옵션")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # 메트릭 선택
        metrics = ['총이용금액_B0M', '총이용건수_B0M', '연체율']
        selected_metric = st.selectbox("분석 메트릭", metrics, key="trend_metric")
    
    with col2:
        # 이동평균 선택
        moving_avg = st.selectbox("이동평균", ["없음", "3개월", "6개월"], key="moving_avg")
    
    with col3:
        # 로그 스케일
        log_scale = st.checkbox("로그 스케일", key="log_scale")
    
    with col4:
        # 이상치 탐지 방법
        anomaly_method = st.selectbox("이상치 탐지", ["없음", "IQR", "3σ"], key="anomaly_method")
    
    # 컨트롤 값을 session_state에 저장
    st.session_state.trend_controls = {
        'metric': selected_metric,
        'moving_avg': moving_avg,
        'log_scale': log_scale,
        'anomaly_method': anomaly_method
    }

def render_time_series_chart(trend_data: pd.DataFrame):
    """시계열 라인 차트"""
    st.markdown("#### 📊 시계열 트렌드")
    
    if trend_data.empty:
        st.warning("데이터가 없습니다.")
        return
    
    controls = st.session_state.get('trend_controls', {})
    metric = controls.get('metric', '총이용금액_B0M')
    moving_avg = controls.get('moving_avg', '없음')
    log_scale = controls.get('log_scale', False)
    
    # 월별 집계
    monthly_data = trend_data.groupby(['YearMonth', 'Segment']).agg({
        metric: 'mean'
    }).reset_index()
    
    # 날짜 변환
    monthly_data['Date'] = monthly_data['YearMonth'].dt.to_timestamp()
    
    # 이동평균 계산
    if moving_avg != '없음':
        window = 3 if moving_avg == '3개월' else 6
        for segment in SEGMENT_ORDER:
            segment_data = monthly_data[monthly_data['Segment'] == segment]
            if not segment_data.empty:
                monthly_data.loc[monthly_data['Segment'] == segment, f'{metric}_MA'] = \
                    segment_data[metric].rolling(window=window, min_periods=1).mean()
    
    # 차트 생성
    fig = go.Figure()
    
    for segment in SEGMENT_ORDER:
        segment_data = monthly_data[monthly_data['Segment'] == segment]
        if segment_data.empty:
            continue
        
        # 기본 라인
        y_values = segment_data[f'{metric}_MA'] if moving_avg != '없음' and f'{metric}_MA' in segment_data.columns else segment_data[metric]
        
        if log_scale and metric != '연체율':
            y_values = np.log10(y_values + 1)
        
        fig.add_trace(go.Scatter(
            x=segment_data['Date'],
            y=y_values,
            mode='lines+markers',
            name=f'세그먼트 {segment}',
            line=dict(color=SEGMENT_COLORS[segment], width=2),
            marker=dict(size=6),
            hovertemplate=f'<b>세그먼트 {segment}</b><br>' +
                         '날짜: %{x}<br>' +
                         f'{metric}: %{{y:,.0f}}<br>' +
                         '<extra></extra>'
        ))
    
    # 차트 레이아웃
    title = f"{metric} 시계열 트렌드"
    if moving_avg != '없음':
        title += f" ({moving_avg} 이동평균)"
    if log_scale and metric != '연체율':
        title += " (로그 스케일)"
    
    fig.update_layout(
        title=title,
        xaxis_title="날짜",
        yaxis_title=f"{metric}" + (" (로그 스케일)" if log_scale and metric != '연체율' else ""),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_yoy_analysis(trend_data: pd.DataFrame):
    """YoY/HoH 변화율 분석"""
    st.markdown("#### 📈 YoY/HoH 변화율 분석")
    
    if trend_data.empty:
        st.warning("데이터가 없습니다.")
        return
    
    controls = st.session_state.get('trend_controls', {})
    metric = controls.get('metric', '총이용금액_B0M')
    
    # 월별 집계
    monthly_data = trend_data.groupby(['YearMonth', 'Segment']).agg({
        metric: 'mean'
    }).reset_index()
    
    # YoY 변화율 계산
    yoy_data = []
    for segment in SEGMENT_ORDER:
        segment_data = monthly_data[monthly_data['Segment'] == segment].copy()
        if len(segment_data) < 13:  # 1년 데이터가 없으면 스킵
            continue
        
        segment_data = segment_data.sort_values('YearMonth')
        segment_data['YoY_Change'] = segment_data[metric].pct_change(periods=12) * 100
        
        yoy_data.append(segment_data[segment_data['YoY_Change'].notna()])
    
    if not yoy_data:
        st.info("YoY 분석을 위한 충분한 데이터가 없습니다.")
        return
    
    yoy_df = pd.concat(yoy_data, ignore_index=True)
    yoy_df['Date'] = yoy_df['YearMonth'].dt.to_timestamp()
    
    # 차트 생성
    fig = go.Figure()
    
    for segment in SEGMENT_ORDER:
        segment_data = yoy_df[yoy_df['Segment'] == segment]
        if segment_data.empty:
            continue
        
        fig.add_trace(go.Scatter(
            x=segment_data['Date'],
            y=segment_data['YoY_Change'],
            mode='lines+markers',
            name=f'세그먼트 {segment}',
            line=dict(color=SEGMENT_COLORS[segment], width=2),
            marker=dict(size=6),
            hovertemplate=f'<b>세그먼트 {segment}</b><br>' +
                         '날짜: %{x}<br>' +
                         'YoY 변화율: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ))
    
    # 0% 기준선 추가
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"{metric} YoY 변화율",
        xaxis_title="날짜",
        yaxis_title="YoY 변화율 (%)",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_anomaly_detection(trend_data: pd.DataFrame):
    """이상치/급변 탐지"""
    st.markdown("#### 🔍 이상치/급변 탐지")
    
    if trend_data.empty:
        st.warning("데이터가 없습니다.")
        return
    
    controls = st.session_state.get('trend_controls', {})
    metric = controls.get('metric', '총이용금액_B0M')
    anomaly_method = controls.get('anomaly_method', '없음')
    
    if anomaly_method == '없음':
        st.info("이상치 탐지 방법을 선택해주세요.")
        return
    
    # 월별 집계
    monthly_data = trend_data.groupby(['YearMonth', 'Segment']).agg({
        metric: 'mean'
    }).reset_index()
    
    # 이상치 탐지
    anomaly_data = []
    for segment in SEGMENT_ORDER:
        segment_data = monthly_data[monthly_data['Segment'] == segment].copy()
        if segment_data.empty:
            continue
        
        values = segment_data[metric].values
        
        if anomaly_method == 'IQR':
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            segment_data['is_anomaly'] = (values < lower_bound) | (values > upper_bound)
        elif anomaly_method == '3σ':
            mean_val = np.mean(values)
            std_val = np.std(values)
            segment_data['is_anomaly'] = np.abs(values - mean_val) > 3 * std_val
        
        anomaly_data.append(segment_data)
    
    if not anomaly_data:
        st.info("이상치 탐지를 위한 데이터가 없습니다.")
        return
    
    anomaly_df = pd.concat(anomaly_data, ignore_index=True)
    anomaly_df['Date'] = anomaly_df['YearMonth'].dt.to_timestamp()
    
    # 차트 생성
    fig = go.Figure()
    
    for segment in SEGMENT_ORDER:
        segment_data = anomaly_df[anomaly_df['Segment'] == segment]
        if segment_data.empty:
            continue
        
        # 정상 데이터
        normal_data = segment_data[~segment_data['is_anomaly']]
        if not normal_data.empty:
            fig.add_trace(go.Scatter(
                x=normal_data['Date'],
                y=normal_data[metric],
                mode='lines+markers',
                name=f'세그먼트 {segment} (정상)',
                line=dict(color=SEGMENT_COLORS[segment], width=2),
                marker=dict(size=6),
                opacity=0.7
            ))
        
        # 이상치 데이터
        anomaly_data = segment_data[segment_data['is_anomaly']]
        if not anomaly_data.empty:
            fig.add_trace(go.Scatter(
                x=anomaly_data['Date'],
                y=anomaly_data[metric],
                mode='markers',
                name=f'세그먼트 {segment} (이상치)',
                marker=dict(
                    color='red',
                    size=12,
                    symbol='diamond',
                    line=dict(color='darkred', width=2)
                ),
                hovertemplate=f'<b>세그먼트 {segment} - 이상치</b><br>' +
                             '날짜: %{x}<br>' +
                             f'{metric}: %{{y:,.0f}}<br>' +
                             '<extra></extra>'
            ))
    
    fig.update_layout(
        title=f"{metric} 이상치 탐지 ({anomaly_method})",
        xaxis_title="날짜",
        yaxis_title=metric,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)