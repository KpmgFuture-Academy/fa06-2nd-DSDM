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

# GPU/CPU 디바이스 설정
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# 전역 변수
_CACHED_DATA = None
_DEVICE = None

def _get_device():
    """GPU 사용 가능 여부 확인 및 디바이스 설정"""
    global _DEVICE
    if _DEVICE is None:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            _DEVICE = torch.device('cuda')
            print("🚀 GPU 사용: CUDA")
        else:
            _DEVICE = torch.device('cpu') if TORCH_AVAILABLE else 'cpu'
            print("💻 CPU 사용")
    return _DEVICE

def get_device_info():
    """현재 디바이스 정보 반환"""
    device = _get_device()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            return {
                'device': device,
                'device_name': torch.cuda.get_device_name(0),
                'device_count': torch.cuda.device_count(),
                'memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,  # GB
                'memory_allocated': torch.cuda.memory_allocated(0) / 1024**3,  # GB
                'memory_cached': torch.cuda.memory_reserved(0) / 1024**3,  # GB
                'cuda_version': torch.version.cuda,
                'torch_version': torch.__version__
            }
        except Exception as e:
            return {
                'device': device,
                'device_name': f'CUDA Error: {str(e)}',
                'device_count': 0,
                'memory_total': 0,
                'memory_allocated': 0,
                'memory_cached': 0,
                'cuda_version': 'Unknown',
                'torch_version': torch.__version__ if TORCH_AVAILABLE else 'Not installed'
            }
    else:
        return {
            'device': device,
            'device_name': 'CPU',
            'device_count': 0,
            'memory_total': 0,
            'memory_allocated': 0,
            'memory_cached': 0,
            'cuda_version': 'N/A',
            'torch_version': torch.__version__ if TORCH_AVAILABLE else 'Not installed'
        }

def gpu_accelerated_computation(data: np.ndarray, operation: str = 'matrix_multiply') -> np.ndarray:
    """GPU 가속 계산 예시"""
    if not TORCH_AVAILABLE:
        st.warning("⚠️ PyTorch가 설치되지 않았습니다. CPU로 계산합니다.")
        return data
    
    device = _get_device()
    
    try:
        # NumPy 배열을 PyTorch 텐서로 변환
        tensor = torch.from_numpy(data.astype(np.float32)).to(device)
        
        if operation == 'matrix_multiply':
            # 행렬 곱셈 (GPU 가속)
            result = torch.mm(tensor, tensor.T)
        elif operation == 'sum':
            # 합계 계산
            result = torch.sum(tensor)
        elif operation == 'mean':
            # 평균 계산
            result = torch.mean(tensor)
        else:
            result = tensor
        
        # 결과를 CPU로 다시 이동하여 NumPy 배열로 변환
        return result.cpu().numpy()
        
    except Exception as e:
        st.error(f"❌ GPU 계산 중 오류 발생: {str(e)}")
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

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    데이터 로드 및 기본 전처리
    """
    try:
        # 기본 데이터 로드
        df = pd.read_csv('base_test_merged_seg.csv', low_memory=False)
        
        # 중복 인덱스 제거
        df = df.reset_index(drop=True)
        
        # 컬럼 매핑 적용
        df = map_columns(df)
        
        # 필수 컬럼 확인 및 생성
        if 'Date' not in df.columns:
            if '기준년월' in df.columns:
                df['Date'] = df['기준년월']
            else:
                # 가상 날짜 생성
                df['Date'] = pd.date_range('2023-01-01', periods=len(df), freq='M')
        
        # 날짜 변환
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m')
        except:
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except:
                # 가상 날짜 생성
                df['Date'] = pd.date_range('2023-01-01', periods=len(df), freq='M')
        
        # 연령 컬럼 확인 및 생성
        if 'Age' not in df.columns:
            if '연령' in df.columns:
                df['Age'] = df['연령']
            else:
                # 가상 연령 생성
                df['Age'] = np.random.randint(20, 70, len(df))
        
        # 연령대 생성
        try:
            df['AgeGroup'] = pd.cut(df['Age'], 
                                   bins=[0, 20, 30, 40, 50, 60, 100], 
                                   labels=['20대미만', '20대', '30대', '40대', '50대', '60대이상'])
        except:
            # 기본 연령대 설정
            df['AgeGroup'] = '30대'
        
        # 지역 컬럼 확인 및 생성
        if 'Region' not in df.columns:
            if '거주시도명' in df.columns:
                df['Region'] = df['거주시도명']
            else:
                # 가상 지역 생성
                regions = ['서울', '경기', '부산', '대구', '인천', '광주', '대전', '울산']
                df['Region'] = np.random.choice(regions, len(df))
        
        # 세그먼트 컬럼 확인 및 생성
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
        
        # ID 컬럼 확인 및 생성
        if 'ID' not in df.columns:
            df['ID'] = range(len(df))
        
        return df
        
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        # 빈 데이터프레임 반환
        return pd.DataFrame()


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
    total_amount = 0
    amount_columns = ['이용금액_일시불_B0M', '이용금액_할부_B0M', '이용금액_체크_B0M', 
                     '이용금액_CA_B0M', '이용금액_카드론_B0M']
    
    for col in amount_columns:
        if col in df.columns:
            total_amount += df[col].fillna(0)
    
    if total_amount.sum() > 0:
        df['총이용금액_B0M'] = total_amount
    
    # 총이용건수 계산
    total_count = 0
    count_columns = ['이용건수_일시불_B0M', '이용건수_할부_B0M', '이용건수_체크_B0M']
    
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
    kpi_df = df.groupby('Segment', observed=False).agg({
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
