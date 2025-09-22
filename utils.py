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
    '카드이용한도금액': '카드이용한도금액',
    
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
        
        # 날짜 컬럼 처리 (2018년 7월~12월 데이터)
        if '기준년월' in df.columns:
            # 기준년월에서 년과 월 추출
            df['Year'] = pd.to_numeric(df['기준년월'].astype(str).str[:4], errors='coerce')
            df['Month'] = pd.to_numeric(df['기준년월'].astype(str).str[4:6], errors='coerce')
            # Date 컬럼은 년월 문자열로 유지 (201807~201812 형태)
            df['Date'] = df['기준년월'].astype(str)
            
            # 실제 데이터 범위 확인
            unique_months = sorted(df['기준년월'].unique())
            # st.info(f"ℹ️ 데이터 기간: {unique_months[0]} ~ {unique_months[-1]} ({len(unique_months)}개월)")
        elif 'Date' not in df.columns:
            st.warning("⚠️ 날짜 컬럼이 없습니다. 기준년월 컬럼을 확인해주세요.")
            df['Year'] = 2018  # 기본값
            df['Month'] = 7    # 기본값
            df['Date'] = '201807'
        
        # 연령 컬럼 처리 (기존 데이터셋 그대로 사용)
        age_column = None
        
        # 연령 관련 컬럼 찾기
        age_candidates = [col for col in df.columns if '연령' in col or 'age' in col.lower() or 'Age' in col]
        
        if age_candidates:
            age_column = age_candidates[0]
            # st.info(f"ℹ️ 연령 컬럼 발견: '{age_column}'")
            # 기존 연령 컬럼을 AgeGroup으로 직접 사용
            df['AgeGroup'] = df[age_column].astype(str)
            # NaN이나 빈 값 처리
            df['AgeGroup'] = df['AgeGroup'].replace(['nan', 'NaN', 'None', ''], '30대')
        else:
            st.warning("⚠️ 연령 관련 컬럼을 찾을 수 없습니다.")
            # st.info(f"ℹ️ 사용 가능한 컬럼들: {list(df.columns)[:20]}...")
            df['AgeGroup'] = '30대'
        
        # 지역 컬럼 확인 및 생성
        if 'Region' not in df.columns:
            if '거주시도명' in df.columns:
                df['Region'] = df['거주시도명']
            else:
                st.warning("⚠️ 지역 컬럼이 없습니다. 거주시도명 컬럼을 확인해주세요.")
                df['Region'] = pd.NA
        
        # 세그먼트 컬럼 확인 및 생성
        if 'Segment' not in df.columns:
            st.warning("⚠️ Segment 컬럼이 없습니다. 세그먼트 정보를 확인해주세요.")
            df['Segment'] = pd.NA
        
        # 세그먼트 카테고리화
        try:
            df['Segment'] = pd.Categorical(df['Segment'], categories=SEGMENT_ORDER, ordered=True)
        except:
            # 기본 세그먼트 설정
            df['Segment'] = 'E'
        
        # ID 컬럼 확인 및 생성
        if 'ID' not in df.columns:
            st.warning("⚠️ ID 컬럼이 없습니다. 고객 ID 정보를 확인해주세요.")
            df['ID'] = pd.NA
        
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
    
    # 카드이용한도금액 컬럼 처리 (누락 시 대체 컬럼 찾기)
    if '카드이용한도금액' not in df.columns:
        # 유사한 컬럼명 찾기
        limit_candidates = [col for col in df.columns if any(keyword in col.lower() for keyword in ['한도', 'limit', 'credit_limit', 'card_limit'])]
        if limit_candidates:
            df['카드이용한도금액'] = pd.to_numeric(df[limit_candidates[0]], errors='coerce').fillna(100000)
            # st.info(f"ℹ️ 카드이용한도금액을 '{limit_candidates[0]}' 컬럼에서 매핑했습니다.")
        else:
            # 기본값으로 설정 (실제 데이터 기반 추정)
            if '총이용금액_B0M' in df.columns:
                # 총이용금액의 3배를 한도로 추정
                df['카드이용한도금액'] = (pd.to_numeric(df['총이용금액_B0M'], errors='coerce') * 3).fillna(100000)
                # st.info("ℹ️ 카드이용한도금액을 총이용금액 기반으로 추정 생성했습니다.")
            else:
                df['카드이용한도금액'] = 100000  # 기본값
                st.warning("⚠️ 카드이용한도금액 컬럼이 없어 기본값(100,000)으로 설정했습니다.")
    
    # 기타 누락된 컬럼 처리
    other_required_columns = ['총이용금액_B0M', '총이용건수_B0M', '연체여부']
    missing_columns = [col for col in other_required_columns if col not in df.columns]
    
    if missing_columns:
        st.warning(f"⚠️ 다음 컬럼들이 누락되었습니다: {missing_columns}")
        for col in missing_columns:
            if col == '연체여부':
                df[col] = 0  # 기본값: 연체 없음
            else:
                df[col] = 0  # 기본값
    
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
    
    # 날짜 필터 (2018년 7월~12월 범위)
    if date_range:
        # date_range를 년월로 변환 (2018년 7월~12월 범위로 제한)
        start_year = max(2018, date_range[0].year)
        start_month = max(7, date_range[0].month) if start_year == 2018 else date_range[0].month
        end_year = min(2018, date_range[1].year)
        end_month = min(12, date_range[1].month) if end_year == 2018 else date_range[1].month
        
        # Year와 Month 컬럼이 있는 경우 사용
        if 'Year' in filtered_df.columns and 'Month' in filtered_df.columns:
            filtered_df = filtered_df[
                ((filtered_df['Year'] > start_year) | 
                 ((filtered_df['Year'] == start_year) & (filtered_df['Month'] >= start_month))) &
                ((filtered_df['Year'] < end_year) | 
                 ((filtered_df['Year'] == end_year) & (filtered_df['Month'] <= end_month)))
            ]
        else:
            # Date 컬럼이 문자열인 경우 (201807~201812 형태)
            start_ym = start_year * 100 + start_month
            end_ym = end_year * 100 + end_month
            
            date_numeric = pd.to_numeric(filtered_df['Date'].astype(str).str.replace('[^0-9]', ''), errors='coerce')
            filtered_df = filtered_df[
                (date_numeric >= start_ym) & (date_numeric <= end_ym)
            ]
    
    # 연령 필터
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
    
    # 세그먼트별 집계 (실제 데이터셋 컬럼 기반)
    agg_dict = {'ID': 'nunique'}
    
    # 실제 존재하는 컬럼만 집계에 포함
    if '총이용금액_B0M' in df.columns:
        agg_dict['총이용금액_B0M'] = 'sum'
    if '총이용건수_B0M' in df.columns:
        agg_dict['총이용건수_B0M'] = 'sum'
    if '카드이용한도금액' in df.columns:
        agg_dict['카드이용한도금액'] = 'sum'
    if '연체여부' in df.columns:
        agg_dict['연체여부'] = 'mean'
    if '포인트_적립_B0M' in df.columns:
        agg_dict['포인트_적립_B0M'] = 'sum'
    if '포인트_소멸_B0M' in df.columns:
        agg_dict['포인트_소멸_B0M'] = 'sum'
    
    kpi_df = df.groupby('Segment', observed=False).agg(agg_dict).rename(columns={'ID': '고객수'})
    
    # 파생 지표 계산 (타입 안전하게)
    if '총이용금액_B0M' in kpi_df.columns:
        # 0으로 나누기 방지
        safe_customers = kpi_df['고객수'].replace(0, 1)
        kpi_df['ARPU_월'] = kpi_df['총이용금액_B0M'] / safe_customers
    
    if '총이용금액_B0M' in kpi_df.columns and '총이용건수_B0M' in kpi_df.columns:
        # 0으로 나누기 방지
        safe_count = kpi_df['총이용건수_B0M'].replace(0, 1)
        kpi_df['객단가'] = kpi_df['총이용금액_B0M'] / safe_count
    
    if '총이용금액_B0M' in kpi_df.columns and '카드이용한도금액' in kpi_df.columns:
        # 0으로 나누기 방지
        safe_limit = kpi_df['카드이용한도금액'].replace(0, 1)
        kpi_df['이용률_한도대비'] = kpi_df['총이용금액_B0M'] / safe_limit
    
    if '연체여부' in kpi_df.columns:
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
    
    # 연령
    age_groups = st.sidebar.multiselect(
        "연령",
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

def safe_csv_encode(df: pd.DataFrame, **kwargs) -> str:
    """
    한글 깨짐 방지를 위한 안전한 CSV 인코딩 함수
    """
    try:
        # Windows 환경에서 한글 깨짐 방지를 위해 cp949 인코딩 시도
        return df.to_csv(encoding='cp949', **kwargs)
    except UnicodeEncodeError:
        # cp949로 인코딩 실패 시 utf-8-sig로 fallback
        return df.to_csv(encoding='utf-8-sig', **kwargs)

def download_data_button(df: pd.DataFrame, filename: str = "dashboard_data.csv") -> None:
    """
    데이터 다운로드 버튼 생성 (Windows 환경에서 한글 깨짐 방지)
    """
    csv = safe_csv_encode(df, index=False)
    
    st.download_button(
        label="📥 현재 뷰 데이터 다운로드",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )
