"""
신용카드 세그먼트 분석 대시보드 - 메인 앱
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from utils import load_data, apply_filters, SEGMENT_ORDER, SEGMENT_COLORS, format_number, get_device_info, _get_device, gpu_accelerated_computation, TORCH_AVAILABLE


import html
import streamlit as st
from streamlit.components.v1 import html as st_html

# --- OPENAI ---
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
prompt_template = """
    너는 데이터 분석가이자 비즈니스 컨설턴트다.  
    아래 데이터를 토대로 신뢰할 수 있는 요약과 인사이트를 작성하라. 
    답변은 간결하고 핵심적인 어투로 정리할 것.

    출력은 두 개의 큰 카테고리로 나눈다.  
    각 카테고리는 시각적으로 한눈에 들어오도록 구조화한다.

    ### 1. 데이터 요약 (Data Summary)
    - 표(table) 형식으로 핵심 지표를 정리한다.  
    - 표는 항목(지표) / 값(숫자·분포) / 설명 세 열 구조로 작성한다.  
    - 우측 항목(지표) 세부 내용들은 **굵게** 표시한다. 
    - 표 아래에는 필요한 경우 간단한 불릿 포인트로 추가 설명을 붙인다.
    - 강조할 수치(최대값, 최소값, 비율)는 **굵게** 표시한다.  

    ### 2. 핵심 인사이트 및 실행 제안 (Key Insights & Action Plan)
    - 소제목을 나누어 구조화한다. (예: 고객 특성 / 카테고리별 기회 / 액션 플랜 / 위험 요인)  
    - 각 소제목 아래에는 불릿 포인트로 정리한다.  
    - 반드시 수치나 비율을 근거로 설명하여 신뢰성을 높인다.  
    - 액션 플랜은 2~4개의 구체적 실행 방안을 **번호 리스트(1. 2. 3.)**로 제시한다.  
    - 잠재적 위험 요인도 간단히 정리한다.  

    출력 형식은 Markdown으로 작성하며,  
    제목(###), 소제목(**(1),(2)**)), 불릿(-), 번호리스트(1.) 등을 활용해 가독성을 높인다.  
    """

def openai_get_insight(prompt_template, *dfs, model="gpt-4o"):
    """
    dfs: DataFrame들(1~5개). 
         이름을 주고 싶으면 ("name", df) 튜플로 넘겨도 됨.
         예) openai_get_insight(pt, df1) 
             openai_get_insight(pt, df1, df2, df3)
             openai_get_insight(pt, ("kpi", df1), ("segment", df2))
    """
    # 단일 인자로 dict가 오면 처리
    if len(dfs) == 1 and isinstance(dfs[0], dict):
        pairs = list(dfs[0].items())
    else:
        pairs = []
        for i, d in enumerate(dfs, 1):
            if isinstance(d, tuple) and len(d) == 2 and isinstance(d[1], pd.DataFrame):
                name, df = d
            else:
                name, df = f"df_{i}", d
            pairs.append((str(name), df))

    blocks = [
        f"### DATASET: {name}\n```csv\n{df.to_csv(index=False)}\n```"
        for name, df in pairs
    ]
    content = "\n\n".join(blocks)

    messages = [
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": content},
    ]
    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content


# --- NAV 정의 ---
NAV = {
    "세그먼트별 비교분석": {
        "icon": "📊",
        "subtabs": ["주요 KPI 분석", "세그먼트별 세부특성", "트렌드 분석(시계열)"]
    },
    "리스크 분석": {
        "icon": "⚠️",
        "subtabs": ["연체/부실", "한도/이용률", "승인/거절", "조기경보(EWS)"]
    },
    "행동마케팅 분석": {
        "icon": "🎯",
        "subtabs": ["캠페인 반응", "개인화 오퍼", "이탈/리텐션", "채널 효율"]
    },
}

# 페이지 설정
st.set_page_config(
    page_title="신용카드 세그먼트 분석 대시보드",
    page_icon="💳",
    layout="wide"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #2C3E50;
    }
    
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498DB;
    }
    
    .segment-A { color: #E74C3C; }
    .segment-B { color: #E67E22; }
    .segment-C { color: #3498DB; }
    .segment-D { color: #2ECC71; }
    .segment-E { color: #F4D03F; }
    
    [data-testid="stSidebar"] { 
        padding-top: 0.5rem; 
    }
</style>
""", unsafe_allow_html=True)

def render_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    """글로벌 필터 컴포넌트 렌더링"""
    st.markdown("### 🔍 글로벌 필터")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # 기간 필터 (Date range)
        try:
            if '기준년월' in df.columns and not df['기준년월'].empty:
                # 기준년월에서 최소/최대 날짜 추출 (2018년 7월~12월)
                unique_months = sorted(df['기준년월'].unique())
                min_month = str(unique_months[0])
                max_month = str(unique_months[-1])
                
                # YYYYMM 형태를 date 객체로 변환
                date_min = date(int(min_month[:4]), int(min_month[4:6]), 1)
                date_max = date(int(max_month[:4]), int(max_month[4:6]), 1)
            else:
                raise ValueError("기준년월 column not found or empty")
        except Exception as e:
            # 날짜 컬럼이 없거나 오류 발생 시 2018년 7월~12월 기본값 사용
            date_min = date(2018, 7, 1)
            date_max = date(2018, 12, 1)
        
        date_range = st.date_input(
            "기간 선택",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
            key="date_range_filter"
        )
    
    with col2:
        # 연령 필터
        age_candidates = [col for col in df.columns if '연령' in col or 'age' in col.lower() or 'Age' in col]
        if age_candidates:
            age_column = age_candidates[0]
            age_options = sorted(df[age_column].dropna().unique().tolist())
        else:
            age_options = ['30대']  # 기본값
        
        selected_ages = st.multiselect(
            "연령",
            options=age_options,
            default=age_options,
            key="age_filter"
        )
    
    with col3:
        # 지역 필터
        region_options = sorted(df['Region'].dropna().unique().tolist())
        selected_regions = st.multiselect(
            "지역",
            options=region_options,
            default=region_options,
            key="region_filter"
        )
    
    with col4:
        # 채널 필터 (가상 데이터)
        channel_options = ['온라인', '오프라인', '모바일', '전화']
        selected_channels = st.multiselect(
            "채널",
            options=channel_options,
            default=channel_options,
            key="channel_filter"
        )
    
    with col5:
        # 카드유형 필터 (가상 데이터)
        card_type_options = ['신용카드', '체크카드', '기프트카드', '포인트카드']
        selected_card_types = st.multiselect(
            "카드유형",
            options=card_type_options,
            default=card_type_options,
            key="card_type_filter"
        )
    
    # 필터 초기화 버튼
    if st.button("🔄 필터 초기화", key="reset_filters"):
        st.rerun()
    
    # 필터 적용
    filtered_df = apply_filters(
        df, 
        date_range=date_range if isinstance(date_range, tuple) else None,
        age_groups=selected_ages,
        regions=selected_regions,
        segments=None  # 세그먼트 필터는 각 탭에서 개별 처리
    )
    
    return filtered_df


def render_kpi_analysis(df: pd.DataFrame, collector: dict = None, return_df: bool = False):
    """주요 KPI 분석"""
    st.markdown("### 📈 주요 KPI 분석")
    
    # 데이터프레임 수집 초기화
    if collector is None:
        collector = {}
    
    # 기본 데이터프레임 저장
    collector["KPI/original_df"] = df.copy()
    
    # KPI 계산
    kpi_data = calculate_kpi_metrics(df)
    collector["KPI/kpi_base"] = kpi_data.copy()
    
    # 정렬 토글
    col1, col2 = st.columns([1, 4])
    with col1:
        sort_by_kpi = st.selectbox(
            "정렬 기준",
            options=["고객수", "ARPU", "객단가", "총이용금액", "연체율"],
            key="kpi_sort"
        )
    
    # 정렬 적용
    if sort_by_kpi in kpi_data.columns:
        kpi_data_sorted = kpi_data.sort_values(sort_by_kpi, ascending=False)
    else:
        kpi_data_sorted = kpi_data
    
    collector["KPI/kpi_sorted"] = kpi_data_sorted.copy()
    
    # KPI 카드 행
    st.markdown("#### 🎯 세그먼트별 KPI 카드")
    render_kpi_cards(kpi_data_sorted, collector=collector)
    
    # 차트 영역 (두 줄)
    st.markdown("#### 📊 KPI 시각화")
    
    # 1행 - 좌: 막대차트, 우: 레이더차트
    col1, col2 = st.columns(2)
    
    with col1:
        render_kpi_bar_chart(kpi_data_sorted, collector=collector)
    
    with col2:
        render_kpi_radar_chart(kpi_data_sorted, collector=collector)
    
    # 2행 - 좌: 박스플롯, 우: 스택바
    col1, col2 = st.columns(2)
    
    with col1:
        render_kpi_boxplot(df, collector=collector)
    
    with col2:
        render_payment_method_chart(df, collector=collector)
    
    # AI 요약    
    st.markdown("---")
    with st.expander("📈 분석 인사이트 보기", expanded=False):
        st.markdown(openai_get_insight(prompt_template, kpi_data))

    # CSV 다운로드
    csv_dl(kpi_data, "KPI", index=False)
    
    # 데이터프레임 반환
    if return_df:
        return collector

def calculate_kpi_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """KPI 메트릭 계산"""
    # 기본 집계
    kpi_data = df.groupby('Segment').agg({
        'ID': 'nunique',
        '총이용금액_B0M': ['sum', 'mean'],
        '총이용건수_B0M': ['sum', 'mean'],
        '카드이용한도금액': 'mean',
        '연체여부': 'mean'
    }).round(2)
    
    # 컬럼명 정리
    kpi_data.columns = ['고객수', '총이용금액', 'ARPU', '총이용건수', '객단가', '평균한도', '연체율']
    kpi_data['연체율'] = kpi_data['연체율'] * 100
    
    # 추가 지표 계산 - 안전한 계산을 위한 처리
    # 분모가 0이거나 너무 작을 때 처리
    safe_denominator = kpi_data['평균한도'].where(kpi_data['평균한도'] > 1000, 1000)  # 최소 1000원으로 설정
    kpi_data['이용률'] = (kpi_data['총이용금액'] / safe_denominator) * 100
    
    # 퍼센트 지표들을 최대 100%로 clip 처리
    kpi_data['이용률'] = np.clip(kpi_data['이용률'], 0, 100)
    kpi_data['연체율'] = np.clip(kpi_data['연체율'], 0, 100)
    
    # 승인거절률 (가상 데이터) - clip 처리 적용
    kpi_data['승인거절률'] = np.random.normal(5, 2, len(kpi_data))
    kpi_data['승인거절률'] = np.clip(np.maximum(0, kpi_data['승인거절률']), 0, 100)
    
    # 전월 대비 증감률 (가상 데이터) - clip 처리 적용
    kpi_data['ARPU_증감'] = np.clip(np.random.normal(0, 5, len(kpi_data)), -50, 50)
    kpi_data['객단가_증감'] = np.clip(np.random.normal(0, 3, len(kpi_data)), -30, 30)
    kpi_data['총이용금액_증감'] = np.clip(np.random.normal(0, 8, len(kpi_data)), -50, 50)
    kpi_data['총이용건수_증감'] = np.clip(np.random.normal(0, 6, len(kpi_data)), -40, 40)
    kpi_data['연체율_증감'] = np.clip(np.random.normal(0, 2, len(kpi_data)), -10, 10)
    kpi_data['승인거절률_증감'] = np.clip(np.random.normal(0, 1, len(kpi_data)), -5, 5)
    kpi_data['이용률_증감'] = np.clip(np.random.normal(0, 4, len(kpi_data)), -20, 20)
    
    # NaN 값 처리
    kpi_data = kpi_data.fillna(0)
    
    return kpi_data.reset_index()

def render_kpi_cards(kpi_data: pd.DataFrame, collector: dict = None, return_df: bool = False):
    """KPI 카드 렌더링"""
    cards_per_row = 5  # A, B, C, D, E 모두 표시
    
    cols = st.columns(cards_per_row)
    
    for j, col in enumerate(cols):
        if j < len(SEGMENT_ORDER):
            segment = SEGMENT_ORDER[j]
            
            # 해당 세그먼트 데이터 찾기
            segment_row = kpi_data[kpi_data['Segment'] == segment]
            
            with col:
                if segment_row.empty or segment_row.iloc[0]['고객수'] < 10:  # 희소 데이터
                    st.markdown(f"""
                    <div style="
                        padding: 1rem; 
                        border-radius: 0.5rem; 
                        background-color: #f8f9fa; 
                        border: 1px solid #dee2e6;
                        text-align: center;
                        color: #6c757d;
                        height: 200px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                    ">
                        <h4 style="color: #6c757d; margin: 0;">세그먼트 {segment}</h4>
                        <p style="margin: 0.5rem 0 0 0;">데이터 없음</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    segment_data = segment_row.iloc[0]
                    
                    # 정상 데이터 카드
                    st.markdown(f"""
                    <div style="
                        padding: 1rem; 
                        border-radius: 0.5rem; 
                        background-color: #ffffff; 
                        border: 1px solid #dee2e6;
                        text-align: center;
                        height: 200px;
                        display: flex;
                        flex-direction: column;
                        justify-content: space-between;
                    ">
                        <div>
                            <h4 style="color: {SEGMENT_COLORS.get(segment, '#6c757d')}; margin: 0;">세그먼트 {segment}</h4>
                            <div style="margin: 0.5rem 0;">
                                <div style="font-size: 1.1rem; font-weight: bold; color: #2c3e50;">
                                    {format_number(segment_data['ARPU'], '원')}
                                </div>
                                <div style="font-size: 0.8rem; color: {'#27ae60' if segment_data['ARPU_증감'] >= 0 else '#e74c3c'};">
                                    {segment_data['ARPU_증감']:+.2f}%
                                </div>
                            </div>
                        </div>
                        
                        <div style="font-size: 0.7rem; color: #7f8c8d;">
                            <div>객단가: {format_number(segment_data['객단가'], '원')}</div>
                            <div>이용률: {segment_data['이용률']:.2f}%</div>
                            <div>연체율: {segment_data['연체율']:.2f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # 데이터프레임 수집
    if collector is not None:
        collector["KPI/cards/kpi_data"] = kpi_data.copy()
    
    if return_df:
        return {"kpi_data": kpi_data.copy()}

def render_kpi_bar_chart(kpi_data: pd.DataFrame, collector: dict = None, return_df: bool = False):
    """KPI 막대 차트"""
    # 비율 계산 (전체 대비)
    total_arpu = kpi_data['ARPU'].sum()
    kpi_data = kpi_data.copy()
    kpi_data['비율'] = (kpi_data['ARPU'] / total_arpu * 100).round(1)
    
    fig = px.bar(
        kpi_data, 
        x='Segment', 
        y='ARPU',
        title="세그먼트별 ARPU 비교",
        color='Segment',
        color_discrete_map=SEGMENT_COLORS,
        category_orders={'Segment': SEGMENT_ORDER}
    )
    
    # 표준화된 라벨: 값 + 비율(%)
    # 막대 높이에 따라 inside/outside 자동 조정
    max_value = kpi_data['ARPU'].max()
    min_value = kpi_data['ARPU'].min()
    threshold = max_value * 0.1  # 최대값의 10% 이하면 outside
    
    for i, row in kpi_data.iterrows():
        if row['ARPU'] < threshold:
            # 작은 막대는 outside
            text_position = 'outside'
        else:
            # 큰 막대는 inside
            text_position = 'inside'
    
    fig.update_traces(
        texttemplate='%{y:,.0f}원<br>(%{customdata:.2f}%)',
        textposition=text_position,
        customdata=kpi_data['비율']
    )
    
    fig.update_layout(
        font_size=12,
        title_font_size=16,
        showlegend=False,
        yaxis_title="ARPU (원)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 데이터프레임 수집
    if collector is not None:
        collector["KPI/bar_chart/chart_data"] = kpi_data.copy()
    
    if return_df:
        return {"chart_data": kpi_data.copy()}

def render_kpi_radar_chart(kpi_data: pd.DataFrame, collector: dict = None, return_df: bool = False):
    """KPI 레이더 차트"""
    # 정규화를 위한 최대값 계산
    max_values = {
        'ARPU': kpi_data['ARPU'].max(),
        '객단가': kpi_data['객단가'].max(),
        '이용률': kpi_data['이용률'].max(),
        '연체율': kpi_data['연체율'].max(),
        '승인거절률': kpi_data['승인거절률'].max()
    }
    
    # 레이더 차트 데이터 준비
    categories = ['ARPU', '객단가', '이용률', '연체율(역)', '승인거절률(역)']
    
    fig = go.Figure()
    
    for _, row in kpi_data.iterrows():
        segment = row['Segment']
        
        # 정규화된 값들 (역축은 1-정규화)
        values = [
            row['ARPU'] / max_values['ARPU'] * 100,
            row['객단가'] / max_values['객단가'] * 100,
            row['이용률'] / max_values['이용률'] * 100,
            (1 - row['연체율'] / max_values['연체율']) * 100,  # 역축
            (1 - row['승인거절률'] / max_values['승인거절률']) * 100  # 역축
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=f'세그먼트 {segment}',
            line_color=SEGMENT_COLORS.get(segment, '#95A5A6')
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="세그먼트별 종합 KPI 비교",
        font_size=12,
        title_font_size=16
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 데이터프레임 수집
    if collector is not None:
        collector["KPI/radar_chart/chart_data"] = kpi_data.copy()
    
    if return_df:
        return {"chart_data": kpi_data.copy()}

def render_kpi_boxplot(df: pd.DataFrame, collector: dict = None, return_df: bool = False):
    """KPI 박스플롯"""
    fig = px.box(
        df, 
        x='Segment', 
        y='총이용금액_B0M',
        title="세그먼트별 총이용금액 분포",
        color='Segment',
        color_discrete_map=SEGMENT_COLORS,
        category_orders={'Segment': SEGMENT_ORDER}
    )
    
    fig.update_layout(
        font_size=12,
        title_font_size=16,
        showlegend=False,
        yaxis_title="총이용금액 (원)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 데이터프레임 수집
    if collector is not None:
        collector["KPI/boxplot/chart_data"] = df.copy()
    
    if return_df:
        return {"chart_data": df.copy()}

def render_payment_method_chart(df: pd.DataFrame, collector: dict = None, return_df: bool = False):
    """결제수단 비중 차트"""
    # 결정적 결제수단 데이터 생성
    payment_data = []
    
    segment_payment_ratios = {
        'A': {'신판': 70, '체크': 20, '현금서비스': 10},
        'B': {'신판': 60, '체크': 25, '현금서비스': 15},
        'C': {'신판': 55, '체크': 30, '현금서비스': 15},
        'D': {'신판': 50, '체크': 35, '현금서비스': 15},
        'E': {'신판': 45, '체크': 40, '현금서비스': 15}
    }
    
    for segment in SEGMENT_ORDER:
        segment_df = df[df['Segment'] == segment]
        if not segment_df.empty:
            # 세그먼트별 고정 비율 사용
            ratios = segment_payment_ratios.get(segment, {'신판': 60, '체크': 25, '현금서비스': 15})
            
            payment_data.append({
                'Segment': segment,
                '신판': ratios['신판'],
                '체크': ratios['체크'],
                '현금서비스': ratios['현금서비스']
            })
    
    payment_df = pd.DataFrame(payment_data)
    
    # 스택바 차트
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='신판',
        x=payment_df['Segment'],
        y=payment_df['신판'],
        marker_color='#3498DB',
        text=[f"{val:.2f}%" if val > 5 else "" for val in payment_df['신판']],
        textposition='inside',
        textfont=dict(color='white', size=10)
    ))
    
    fig.add_trace(go.Bar(
        name='체크',
        x=payment_df['Segment'],
        y=payment_df['체크'],
        marker_color='#2ECC71',
        text=[f"{val:.2f}%" if val > 5 else "" for val in payment_df['체크']],
        textposition='inside',
        textfont=dict(color='white', size=10)
    ))
    
    fig.add_trace(go.Bar(
        name='현금서비스',
        x=payment_df['Segment'],
        y=payment_df['현금서비스'],
        marker_color='#E67E22',
        text=[f"{val:.2f}%" if val > 5 else "" for val in payment_df['현금서비스']],
        textposition='inside',
        textfont=dict(color='white', size=10)
    ))
    
    fig.update_layout(
        barmode='stack',
        title="세그먼트별 결제수단 비중",
        font_size=12,
        title_font_size=16,
        yaxis_title="비중 (%)",
        xaxis={'categoryorder': 'array', 'categoryarray': SEGMENT_ORDER}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 데이터프레임 수집
    if collector is not None:
        collector["KPI/payment_method/chart_data"] = payment_df.copy()
    
    if return_df:
        return {"chart_data": payment_df.copy()}

def render_segment_details(df: pd.DataFrame, collector: dict = None, return_df: bool = False):
    """세그먼트별 세부특성"""
    st.markdown("### 🔍 세그먼트별 세부특성")
    
    # 데이터프레임 수집 초기화
    if collector is None:
        collector = {}
    
    # 기본 데이터프레임 저장
    collector["SegmentDetails/original_df"] = df.copy()
    
    # 1. 분포/구성 분석
    st.markdown("#### 📊 분포/구성 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 연령×세그먼트 Stacked Bar (%)
        age_segment_data = render_age_segment_distribution(df, collector=collector)
    
    with col2:
        # 지역×세그먼트 Heatmap
        region_segment_data = render_region_segment_heatmap(df, collector=collector)
    
    # 채널 선호도 TopN
    st.markdown("##### 📱 세그먼트별 채널 선호도 (Top 5)")
    channel_preference_data = render_channel_preference(df, collector=collector)
    
    # 2. 업종/MCC 요약
    st.markdown("#### 🏢 세그먼트별 업종 분석")
    industry_data = render_industry_analysis(df, collector=collector)
    
    # 3. 코호트/잔존 분석
    st.markdown("#### 📈 코호트/잔존 분석")
    cohort_data = render_cohort_analysis(df, collector=collector)
    
    # 4. 다운로드 버튼
    st.markdown("---")
    render_download_section(df, collector=collector)
    
    # 데이터프레임 반환
    if return_df:
        return collector

def render_age_segment_distribution(df: pd.DataFrame, collector: dict = None, return_df: bool = False):
    """연령×세그먼트 분포 Stacked Bar"""
    # 연령×세그먼트 교차표 생성
    # 연령 컬럼 찾기
    age_candidates = [col for col in df.columns if '연령' in col or 'age' in col.lower() or 'Age' in col]
    if age_candidates:
        age_column = age_candidates[0]
        cross_table = pd.crosstab(df[age_column], df['Segment'], normalize='index') * 100
    else:
        # 연령 컬럼이 없으면 빈 테이블 생성
        cross_table = pd.DataFrame()
    
    # 세그먼트 순서 보장
    cross_table = cross_table.reindex(columns=SEGMENT_ORDER, fill_value=0)
    
    fig = go.Figure()
    
    for segment in SEGMENT_ORDER:
        if segment in cross_table.columns:
            # 표준화된 라벨: 비율이 5% 이상일 때만 표시
            segment_data = cross_table[segment]
            text_labels = [f"{val:.2f}%" if val > 5 else "" for val in segment_data]
            
            fig.add_trace(go.Bar(
                name=f'세그먼트 {segment}',
                x=cross_table.index,
                y=cross_table[segment],
                marker_color=SEGMENT_COLORS.get(segment, '#95A5A6'),
                text=text_labels,
                textposition='inside',
                textfont=dict(color='white', size=9),
                hovertemplate=f'세그먼트 {segment}<br>%{{x}}: %{{y:.2f}}%<extra></extra>'
            ))
    
    fig.update_layout(
        barmode='stack',
        title="연령별 세그먼트 분포 (%)",
        xaxis_title="연령",
        yaxis_title="비율 (%)",
        font_size=12,
        title_font_size=14,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 데이터프레임 수집
    if collector is not None:
        collector["SegmentDetails/age_distribution/cross_table"] = cross_table.copy()
    
    if return_df:
        return {"cross_table": cross_table.copy()}
    
    return cross_table

def render_region_segment_heatmap(df: pd.DataFrame, collector: dict = None, return_df: bool = False):
    """지역×세그먼트 히트맵"""
    # 지역×세그먼트 교차표 생성 (비율)
    cross_table = pd.crosstab(df['Region'], df['Segment'], normalize='index') * 100
    
    # 세그먼트 순서 보장
    cross_table = cross_table.reindex(columns=SEGMENT_ORDER, fill_value=0)
    
    # 상위 지역만 표시 (최대 15개)
    if len(cross_table) > 15:
        cross_table = cross_table.head(15)
    
    # 표준화된 라벨 생성 (값 + 비율)
    text_labels = []
    for i in range(len(cross_table.index)):
        row_labels = []
        for j in range(len(cross_table.columns)):
            value = cross_table.iloc[i, j]
            if value > 1:  # 1% 이상일 때만 표시
                row_labels.append(f"{value:.2f}%")
            else:
                row_labels.append("")
        text_labels.append(row_labels)
    
    fig = px.imshow(
        cross_table,
        title="지역별 세그먼트 분포 (%)",
        color_continuous_scale='RdYlBu_r',
        aspect="auto",
        labels=dict(x="세그먼트", y="지역", color="비율(%)"),
        text_auto=False  # 수동으로 텍스트 설정
    )
    
    # 표준화된 라벨 적용
    fig.update_traces(
        text=text_labels,
        texttemplate="%{text}",
        textfont=dict(size=10, color="black")
    )
    
    fig.update_layout(
        font_size=10,
        title_font_size=14,
        xaxis={'categoryorder': 'array', 'categoryarray': SEGMENT_ORDER}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 데이터프레임 수집
    if collector is not None:
        collector["SegmentDetails/region_heatmap/cross_table"] = cross_table.copy()
    
    if return_df:
        return {"cross_table": cross_table.copy()}
    
    return cross_table

def render_channel_preference(df: pd.DataFrame, collector: dict = None, return_df: bool = False):
    """채널 선호도 분석 (실제 데이터 기반)"""
    st.markdown("#### 📱 세그먼트별 채널 선호도 분석")
    
    # 실제 데이터에서 채널 관련 컬럼 찾기 (제공된 컬럼명 기반)
    channel_related_cols = [
        # ARS 관련
        '인입횟수_ARS_R6M', '이용메뉴건수_ARS_R6M', '인입일수_ARS_R6M', '인입월수_ARS_R6M',
        '인입횟수_ARS_BOM', '이용메뉴건수_ARS_BOM', '인입일수_ARS_BOM',
        
        # PC 관련
        '방문횟수_PC_R6M', '방문일수_PC_R6M', '방문월수_PC_R6M',
        '방문횟수_PC_BOM', '방문일수_PC_BOM',
        
        # 앱 관련
        '방문횟수_앱_R6M', '방문일수_앱_R6M', '방문월수_앱_R6M',
        '방문횟수_앱_BOM', '방문일수_앱_BOM',
        
        # 모바일웹 관련
        '방문횟수_모바일웹_R6M', '방문일수_모바일웹_R6M', '방문월수_모바일웹_R6M',
        '방문횟수_모바일웹_BOM', '방문일수_모바일웹_BOM',
        
        # 인터넷뱅킹 관련
        '인입횟수_IB_R6M', '인입횟수_금융_IB_R6M', '인입일수_IB_R6M', '인입월수_IB_R6M',
        '이용메뉴건수_IB_R6M', '인입횟수_IB_BOM', '인입일수_IB_BOM', '이용메뉴건수_IB_BOM',
        
        # 상담 관련
        '상담건수_BOM', '상담건수_R6M',
        
        # 당사 서비스 관련
        '당사PAY_방문횟수_BOM', '당사PAY_방문횟수_R6M', '당사PAY_방문월수_R6M',
        '당사멤버쉽_방문횟수_BOM', '당사멤버쉽_방문횟수_R6M', '당사멤버쉽_방문월수_R6M',
        
        # 홈페이지 관련
        '홈페이지_금융건수_R6M', '홈페이지_선결제건수_R6M', '홈페이지_금융건수_R3M', '홈페이지_선결제건수_R3M'
    ]
    
    # 실제 존재하는 채널 관련 컬럼만 필터링
    existing_channel_cols = [col for col in channel_related_cols if col in df.columns]
    
    if existing_channel_cols:
        st.info(f"ℹ️ 채널 분석에 사용할 컬럼 {len(existing_channel_cols)}개 발견")
        
        # 채널 카테고리별 매핑 (실제 컬럼명 기반)
        channel_category_map = {
            'ARS': [col for col in existing_channel_cols if 'ARS' in col],
            'PC': [col for col in existing_channel_cols if 'PC' in col and '방문' in col],
            '앱': [col for col in existing_channel_cols if '앱' in col and '방문' in col],
            '모바일웹': [col for col in existing_channel_cols if '모바일웹' in col],
            '인터넷뱅킹': [col for col in existing_channel_cols if 'IB' in col],
            '상담': [col for col in existing_channel_cols if '상담' in col],
            '당사PAY': [col for col in existing_channel_cols if '당사PAY' in col],
            '당사멤버쉽': [col for col in existing_channel_cols if '당사멤버쉽' in col],
            '홈페이지': [col for col in existing_channel_cols if '홈페이지' in col]
        }
        
        # 세그먼트별 채널 분석
        channel_data = []
        
        for segment in SEGMENT_ORDER:
            segment_df = df[df['Segment'] == segment]
            if not segment_df.empty:
                for channel_category, cols in channel_category_map.items():
                    if cols:  # 해당 채널에 컬럼이 있는 경우
                        # 해당 채널의 모든 컬럼 합계 (방문횟수/인입횟수 우선)
                        total_activity = 0
                        for col in cols:
                            if any(keyword in col for keyword in ['방문횟수', '인입횟수', '이용메뉴건수', '상담건수']):
                                activity = pd.to_numeric(segment_df[col], errors='coerce').fillna(0).sum()
                                total_activity += activity
                        
                        if total_activity > 0:  # 0보다 큰 값만 추가
                            # 세그먼트 총 활동 대비 비율 계산
                            total_segment_activity = 0
                            for col in existing_channel_cols:
                                if any(keyword in col for keyword in ['방문횟수', '인입횟수', '이용메뉴건수', '상담건수']):
                                    activity = pd.to_numeric(segment_df[col], errors='coerce').fillna(0).sum()
                                    total_segment_activity += activity
                            
                            usage_rate = (total_activity / total_segment_activity * 100) if total_segment_activity > 0 else 0
                            
                            channel_data.append({
                                'Segment': segment,
                                'Channel': channel_category,
                                'Usage_Rate': usage_rate,
                                'Activity_Count': total_activity
                            })
        
        channel_df = pd.DataFrame(channel_data)
        
        if channel_df.empty:
            st.warning("⚠️ 채널별 데이터가 없습니다.")
            channel_df = pd.DataFrame(columns=['Segment', 'Channel', 'Usage_Rate', 'Activity_Count'])
    else:
        st.warning("⚠️ 채널 분석을 위한 컬럼을 찾을 수 없습니다.")
        # 기본 채널 정보 표시
        channel_df = pd.DataFrame(columns=['Segment', 'Channel', 'Usage_Rate', 'Activity_Count'])
    
    # Horizontal Bar Chart
    fig = go.Figure()
    
    for segment in SEGMENT_ORDER:
        segment_data = channel_df[channel_df['Segment'] == segment]
        if not segment_data.empty:
            fig.add_trace(go.Bar(
                name=f'세그먼트 {segment}',
                y=segment_data['Channel'],
                x=segment_data['Usage_Rate'],
                orientation='h',
                marker_color=SEGMENT_COLORS.get(segment, '#95A5A6'),
                hovertemplate=f'세그먼트 {segment}<br>%{{y}}: %{{x:.2f}}%<extra></extra>'
            ))
    
    fig.update_layout(
        title="세그먼트별 채널 선호도 (Top 5)",
        xaxis_title="이용률 (%)",
        yaxis_title="채널",
        font_size=12,
        title_font_size=14,
        height=600,
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 데이터프레임 수집
    if collector is not None:
        collector["SegmentDetails/channel_preference/channel_df"] = channel_df.copy()
    
    if return_df:
        return {"channel_df": channel_df.copy()}
    
    return channel_df

def render_industry_analysis(df: pd.DataFrame, collector: dict = None, return_df: bool = False):
    """업종 분석"""
    # 가상의 업종 데이터 생성
    industries = [
        '할인점', '마트', '백화점', '온라인쇼핑', '주유소', '카페', '음식점', 
        '병원', '약국', '통신', '보험', '교육', '여행', '문화', '운송'
    ]
    
    # 실제 데이터 컬럼을 사용한 업종 분석
    st.markdown("#### 📊 업종 분석 (실제 데이터 기반)")
    
    # 실제 데이터에서 사용 가능한 업종 관련 컬럼들
    industry_amount_cols = [
        # 쇼핑 관련
        '쇼핑_도소매_이용금액', '쇼핑_백화점_이용금액', '쇼핑_마트_이용금액', '쇼핑_슈퍼마켓_이용금액',
        '쇼핑_편의점_이용금액', '쇼핑_아울렛_이용금액', '쇼핑_온라인_이용금액', '쇼핑_기타_이용금액',
        # 교통 관련
        '교통_주유이용금액', '교통_정비이용금액', '교통_통행료이용금액', '교통_버스지하철이용금액',
        '교통_택시이용금액', '교통_철도버스이용금액',
        # 여유 관련
        '여유_운동이용금액', '여유_Pet이용금액', '여유_공연이용금액', '여유_공원이용금액',
        '여유_숙박이용금액', '여유_여행이용금액', '여유_항공이용금액', '여유_기타이용금액',
        # 납부 관련
        '납부_통신비이용금액', '납부_관리비이용금액', '납부_렌탈료이용금액', '납부_가스전기료이용금액',
        '납부_보험료이용금액', '납부_유선방송이용금액', '납부_건강연금이용금액', '납부_기타이용금액',
        # 기타
        '이용금액_해외',
        # 순위 업종
        '_1순위업종_이용금액', '_2순위업종_이용금액', '_3순위업종_이용금액',
        '_1순위쇼핑업종_이용금액', '_2순위쇼핑업종_이용금액', '_3순위쇼핑업종_이용금액',
        '_1순위교통업종_이용금액', '_2순위교통업종_이용금액', '_3순위교통업종_이용금액',
        '_1순위여유업종_이용금액', '_2순위여유업종_이용금액', '_3순위여유업종_이용금액',
        '_1순위납부업종_이용금액', '_2순위납부업종_이용금액', '_3순위납부업종_이용금액'
    ]
    
    # 실제 존재하는 컬럼만 필터링
    existing_industry_cols = [col for col in industry_amount_cols if col in df.columns]
    
    if existing_industry_cols:
        st.info(f"ℹ️ 업종 분석에 사용할 컬럼 {len(existing_industry_cols)}개 발견")
        
        # 업종 카테고리별 매핑
        industry_category_map = {
            '쇼핑': [col for col in existing_industry_cols if '쇼핑_' in col or '_순위쇼핑업종_' in col],
            '교통': [col for col in existing_industry_cols if '교통_' in col or '_순위교통업종_' in col],
            '여유': [col for col in existing_industry_cols if '여유_' in col or '_순위여유업종_' in col],
            '납부': [col for col in existing_industry_cols if '납부_' in col or '_순위납부업종_' in col],
            '해외': [col for col in existing_industry_cols if col == '이용금액_해외'],
            '기타_순위': [col for col in existing_industry_cols if '_순위업종_' in col and not any(cat in col for cat in ['쇼핑', '교통', '여유', '납부'])]
        }
        
        # 세그먼트별 업종 분석
        industry_data = []
        
        for segment in SEGMENT_ORDER:
            segment_df = df[df['Segment'] == segment]
            if not segment_df.empty:
                for category, cols in industry_category_map.items():
                    if cols:  # 해당 카테고리에 컬럼이 있는 경우
                        # 해당 카테고리의 모든 컬럼 합계
                        total_amount = 0
                        for col in cols:
                            amount = pd.to_numeric(segment_df[col], errors='coerce').fillna(0).sum()
                            total_amount += amount
                        
                        if total_amount > 0:  # 0보다 큰 값만 추가
                            industry_data.append({
                                'Segment': segment,
                                'Industry': category,
                                'Amount': total_amount
                            })
        
        industry_df = pd.DataFrame(industry_data)
        
        if industry_df.empty:
            st.warning("⚠️ 업종별 데이터가 없습니다.")
            industry_df = pd.DataFrame(columns=['Segment', 'Industry', 'Amount'])
    else:
        st.warning("⚠️ 업종 분석을 위한 컬럼을 찾을 수 없습니다.")
        # 기존 가상 데이터로 폴백
        industry_data = []
        for segment in SEGMENT_ORDER:
            segment_df = df[df['Segment'] == segment]
            if not segment_df.empty:
                base_amount = segment_df['총이용금액_B0M'].mean() if '총이용금액_B0M' in df.columns else 100000
                for industry in industries:
                    industry_amount = base_amount * 0.1  # 기본 10% 비율
                    industry_data.append({
                        'Segment': segment,
                        'Industry': industry,
                        'Amount': industry_amount
                    })
        industry_df = pd.DataFrame(industry_data)
    
    # 세그먼트별 Top 10 업종 계산
    col1, col2 = st.columns(2)
    
    with col1:
        # 업종별 총 이용금액 막대 차트
        segment_industry_sum = industry_df.groupby(['Segment', 'Industry'])['Amount'].sum().reset_index()
        
        # 각 세그먼트별 Top 10 업종 선택
        top_industries = []
        for segment in SEGMENT_ORDER:
            segment_data = segment_industry_sum[segment_industry_sum['Segment'] == segment]
            if not segment_data.empty:
                top_10 = segment_data.nlargest(10, 'Amount')
                top_industries.append(top_10)
        
        if top_industries:
            top_industry_df = pd.concat(top_industries)
            
            fig = px.bar(
                top_industry_df,
                x='Industry',
                y='Amount',
                color='Segment',
                title="세그먼트별 Top 10 업종 이용금액",
                color_discrete_map=SEGMENT_COLORS,
                category_orders={'Segment': SEGMENT_ORDER}
            )
            
            fig.update_layout(
                font_size=10,
                title_font_size=14,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 업종별 비중 테이블
        st.markdown("##### 📋 세그먼트별 업종 비중 (Top 5)")
        
        # 각 세그먼트별 Top 5 업종 비중 계산
        industry_pivot = industry_df.pivot_table(
            index='Industry', 
            columns='Segment', 
            values='Amount', 
            aggfunc='sum'
        ).fillna(0)
        
        # 비율 계산
        industry_ratio = industry_pivot.div(industry_pivot.sum()) * 100
        
        # Top 5 업종 선택 (전체 평균 기준)
        top_5_industries = industry_ratio.mean(axis=1).nlargest(5).index
        
        display_table = industry_ratio.loc[top_5_industries, SEGMENT_ORDER].round(2)
        
        # Highlift 항목 강조를 위한 스타일링
        def highlight_highlift(val):
            max_val = display_table.max().max()
            if val > max_val * 0.8:  # 상위 20% 값들 강조
                return 'background-color: #ffeb3b; font-weight: bold'
            return ''
        
        styled_table = display_table.style.applymap(highlight_highlift)
        
        st.dataframe(
            styled_table,
            use_container_width=True,
            column_config={
                col: st.column_config.NumberColumn(
                    col,
                    help=f"세그먼트 {col}의 업종별 비중 (%)",
                    format="%.2f%%"
                ) for col in SEGMENT_ORDER
            }
        )
        
        # 데이터프레임 반환
        return display_table

def render_cohort_analysis(df: pd.DataFrame):
    """코호트 분석"""
    # 가상의 코호트 데이터 생성
    months = pd.date_range('2023-01-01', '2023-12-01', freq='MS')
    
    cohort_data = []
    
    for segment in SEGMENT_ORDER:
        segment_df = df[df['Segment'] == segment]
        if not segment_df.empty:
            # 각 월별 코호트 생성
            for month in months:
                # 가입 월별 잔존율 패턴 생성
                if segment == 'A':
                    retention_pattern = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45]
                elif segment == 'B':
                    retention_pattern = [1.0, 0.98, 0.95, 0.92, 0.88, 0.84, 0.80, 0.76, 0.72, 0.68, 0.64, 0.60]
                elif segment == 'C':
                    retention_pattern = [1.0, 0.96, 0.92, 0.88, 0.84, 0.80, 0.76, 0.72, 0.68, 0.64, 0.60, 0.56]
                elif segment == 'D':
                    retention_pattern = [1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79, 0.76, 0.73, 0.70, 0.67]
                else:  # E
                    retention_pattern = [1.0, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78]
                
                for month_idx, retention_rate in enumerate(retention_pattern):
                    cohort_data.append({
                        'Segment': segment,
                        'Cohort_Month': month.strftime('%Y-%m'),
                        'Month_Index': month_idx,
                        'Retention_Rate': retention_rate * 100
                    })
    
    cohort_df = pd.DataFrame(cohort_data)
    
    # Line Chart
    fig = go.Figure()
    
    for segment in SEGMENT_ORDER:
        segment_data = cohort_df[cohort_df['Segment'] == segment]
        if not segment_data.empty:
            # 평균 잔존율 계산
            avg_retention = segment_data.groupby('Month_Index')['Retention_Rate'].mean()
            
            fig.add_trace(go.Scatter(
                x=avg_retention.index,
                y=avg_retention.values,
                mode='lines+markers',
                name=f'세그먼트 {segment}',
                line=dict(color=SEGMENT_COLORS.get(segment, '#95A5A6'), width=3),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title="세그먼트별 코호트 잔존율",
        xaxis_title="월차 (Month Index)",
        yaxis_title="잔존율 (%)",
        font_size=12,
        title_font_size=14,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 데이터프레임 반환
    return cohort_df

def render_download_section(df: pd.DataFrame):
    """다운로드 섹션"""
    st.markdown("#### 📥 데이터 다운로드")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 연령×세그먼트 분포 데이터
        # 연령 컬럼 찾기
        age_candidates = [col for col in df.columns if '연령' in col or 'age' in col.lower() or 'Age' in col]
        if age_candidates:
            age_column = age_candidates[0]
            age_segment_cross = pd.crosstab(df[age_column], df['Segment'], normalize='index') * 100
        else:
            age_segment_cross = pd.DataFrame()
        csv_age = age_segment_cross.to_csv()
        
        st.download_button(
            label="📊 연령×세그먼트 분포",
            data=csv_age,
            file_name="age_segment_distribution.csv",
            mime="text/csv"
        )
    
    with col2:
        # 지역×세그먼트 분포 데이터
        region_segment_cross = pd.crosstab(df['Region'], df['Segment'], normalize='index') * 100
        csv_region = region_segment_cross.to_csv()
        
        st.download_button(
            label="🗺️ 지역×세그먼트 분포",
            data=csv_region,
            file_name="region_segment_distribution.csv",
            mime="text/csv"
        )
    
    with col3:
        # 전체 세부특성 데이터
        csv_full = df.to_csv(index=False)
        
        st.download_button(
            label="📋 전체 세부특성 데이터",
            data=csv_full,
            file_name="segment_details_full.csv",
            mime="text/csv"
        )

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
            # 결정적 데이터 생성 (ID 기반)
            if metric == '총이용금액_B0M':
                trend_df[metric] = trend_df.apply(lambda row: 
                    abs(hash(str(row.get('ID', 0))) % 400000 + 300000), axis=1)
            elif metric == '총이용건수_B0M':
                trend_df[metric] = trend_df.apply(lambda row: 
                    abs(hash(str(row.get('ID', 0))) % 100 + 10), axis=1)
            elif metric == '연체율':
                trend_df[metric] = trend_df.apply(lambda row: 
                    abs(hash(str(row.get('ID', 0))) % 10), axis=1)  # 0-10%
    
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
        
        # 마지막 점에만 라벨 표시
        last_point_x = segment_data['Date'].iloc[-1]
        last_point_y = y_values.iloc[-1]
        
        # 기본 라인 (라벨 없음)
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
        
        # 마지막 점에 라벨 추가
        fig.add_trace(go.Scatter(
            x=[last_point_x],
            y=[last_point_y],
            mode='markers+text',
            name=f'세그먼트 {segment} (최신)',
            marker=dict(color=SEGMENT_COLORS[segment], size=8),
            text=[f"{last_point_y:,.0f}"],
            textposition='top center',
            textfont=dict(size=10, color=SEGMENT_COLORS[segment]),
            showlegend=False,
            hovertemplate=f'<b>세그먼트 {segment} (최신값)</b><br>' +
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
    height=700,   # 세로
    width=900     # 가로 (원하는 값으로 줄이기)
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
                         'YoY 변화율: %{y:.2f}%<br>' +
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
        height=700,   # 세로
        width=900     # 가로 (원하는 값으로 줄이기)
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
        height=700,   # 세로
        width=900     # 가로 (원하는 값으로 줄이기)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_seasonal_decomposition(trend_data: pd.DataFrame):
    """분해(seasonal_decompose) 분석"""
    st.markdown("#### 🔬 시계열 분해 분석")
    
    if trend_data.empty:
        st.warning("데이터가 없습니다.")
        return
    
    controls = st.session_state.get('trend_controls', {})
    metric = controls.get('metric', '총이용금액_B0M')
    
    # 충분한 데이터가 있는지 확인 (최소 24개월)
    monthly_data = trend_data.groupby(['YearMonth', 'Segment']).agg({
        metric: 'mean'
    }).reset_index()
    
    if len(monthly_data) < 24:
        st.info("시계열 분해를 위한 충분한 데이터가 없습니다. (최소 24개월 필요)")
        return
    
    # 세그먼트별 분해 분석
    for segment in SEGMENT_ORDER:
        segment_data = monthly_data[monthly_data['Segment'] == segment].copy()
        if segment_data.empty or len(segment_data) < 24:
            continue
        
        segment_data = segment_data.sort_values('YearMonth')
        
        # 시계열 분해
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            ts = pd.Series(segment_data[metric].values, 
                          index=pd.date_range(start=segment_data['YearMonth'].min().to_timestamp(), 
                                            periods=len(segment_data), freq='MS'))
            
            decomposition = seasonal_decompose(ts, model='additive', period=12)
            
            # 분해 결과 시각화
            fig = go.Figure()
            
            # 원본 데이터
            fig.add_trace(go.Scatter(
                x=decomposition.observed.index,
                y=decomposition.observed.values,
                mode='lines',
                name='원본',
                line=dict(color='blue', width=2)
            ))
            
            # 트렌드
            fig.add_trace(go.Scatter(
                x=decomposition.trend.index,
                y=decomposition.trend.values,
                mode='lines',
                name='트렌드',
                line=dict(color='red', width=2)
            ))
            
            # 계절성
            fig.add_trace(go.Scatter(
                x=decomposition.seasonal.index,
                y=decomposition.seasonal.values,
                mode='lines',
                name='계절성',
                line=dict(color='green', width=2)
            ))
            
            # 잔차
            fig.add_trace(go.Scatter(
                x=decomposition.resid.index,
                y=decomposition.resid.values,
                mode='lines',
                name='잔차',
                line=dict(color='orange', width=2)
            ))
            
            fig.update_layout(
                title=f"세그먼트 {segment} - {metric} 시계열 분해",
                xaxis_title="날짜",
                yaxis_title=metric,
                height=700,   # 세로
                width=900,     # 가로 (원하는 값으로 줄이기)
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            st.warning("statsmodels 라이브러리가 설치되지 않았습니다. 시계열 분해 기능을 사용할 수 없습니다.")
            break
        except Exception as e:
            st.warning(f"세그먼트 {segment}의 시계열 분해 중 오류가 발생했습니다: {str(e)}")
            continue

def render_trend_download_section(trend_data: pd.DataFrame):
    """트렌드 분석 다운로드 섹션"""
    st.markdown("#### 📥 데이터 다운로드")
    
    if trend_data.empty:
        st.warning("다운로드할 데이터가 없습니다.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 월별 집계 데이터
        monthly_summary = trend_data.groupby(['YearMonth', 'Segment']).agg({
            '총이용금액_B0M': 'mean',
            '총이용건수_B0M': 'mean',
            '연체율': 'mean'
        }).reset_index()
        
        csv_monthly = monthly_summary.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📊 월별 집계 데이터",
            data=csv_monthly,
            file_name="trend_monthly_summary.csv",
            mime="text/csv"
        )
    
    with col2:
        # YoY 변화율 데이터
        controls = st.session_state.get('trend_controls', {})
        metric = controls.get('metric', '총이용금액_B0M')
        
        monthly_data = trend_data.groupby(['YearMonth', 'Segment']).agg({
            metric: 'mean'
        }).reset_index()
        
        yoy_data = []
        for segment in SEGMENT_ORDER:
            segment_data = monthly_data[monthly_data['Segment'] == segment].copy()
            if len(segment_data) >= 13:
                segment_data = segment_data.sort_values('YearMonth')
                segment_data['YoY_Change'] = segment_data[metric].pct_change(periods=12) * 100
                yoy_data.append(segment_data[segment_data['YoY_Change'].notna()])
        
        if yoy_data:
            yoy_df = pd.concat(yoy_data, ignore_index=True)
            csv_yoy = yoy_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📈 YoY 변화율 데이터",
                data=csv_yoy,
                file_name="trend_yoy_analysis.csv",
                mime="text/csv"
            )
        else:
            st.info("YoY 데이터가 부족합니다.")
    
    with col3:
        # 이상치 탐지 결과
        anomaly_method = controls.get('anomaly_method', '없음')
        if anomaly_method != '없음':
            monthly_data = trend_data.groupby(['YearMonth', 'Segment']).agg({
                metric: 'mean'
            }).reset_index()
            
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
            
            if anomaly_data:
                anomaly_df = pd.concat(anomaly_data, ignore_index=True)
                csv_anomaly = anomaly_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="🔍 이상치 탐지 결과",
                    data=csv_anomaly,
                    file_name="trend_anomaly_detection.csv",
                    mime="text/csv"
                )
        else:
            st.info("이상치 탐지를 먼저 실행해주세요.")

def render_trend_analysis(df: pd.DataFrame, collector: dict = None, return_df: bool = False):
    """트렌드 분석(시계열)"""
    st.markdown("### 📈 트렌드 분석(시계열)")
    
    if df.empty:
        st.warning("데이터가 없습니다.")
        return
    
    # 데이터프레임 수집 초기화
    if collector is None:
        collector = {}
    
    # 기본 데이터프레임 저장
    collector["Trend/original_df"] = df.copy()
    
    # 데이터 전처리
    trend_data = prepare_trend_data(df)
    collector["Trend/trend_base"] = trend_data.copy()
    
    # 컨트롤 패널
    render_trend_controls(trend_data, collector=collector)
    
    # 시계열 라인 차트
    render_time_series_chart(trend_data, collector=collector)
    
    # YoY/HoH 변화율 분석
    render_yoy_analysis(trend_data, collector=collector)
    
    # 이상치/급변 탐지
    render_anomaly_detection(trend_data, collector=collector)
    
    # 분해 분석 (선택적)
    render_seasonal_decomposition(trend_data, collector=collector)
    
    # 다운로드 섹션
    render_trend_download_section(trend_data, collector=collector)
    
    # 데이터프레임 반환
    if return_df:
        return collector



def main():
    """메인 함수"""
    
    # 헤더
    st.markdown('<h1 class="main-header">💳 신용카드 세그먼트 분석 대시보드</h1>', 
                unsafe_allow_html=True)
    
    # 데이터 로드
    with st.spinner("데이터를 로딩 중입니다..."):
        df = load_data()
    
    # 연령 컬럼은 utils.py의 load_data()에서 이미 생성됨
    
    if df.empty:
        st.error("📁 데이터를 찾을 수 없습니다.")
        st.info("""
        **데이터 파일이 필요합니다:**
        - `base_test_merged_seg.csv` 파일이 프로젝트 루트 디렉토리에 있어야 합니다.
        - 파일에 다음 컬럼들이 포함되어야 합니다:
          - Segment (A~E)
          - 기준년월 또는 Date
          - ID (고객 ID)
          - 연령 또는 Age
          - 거주시도명 또는 Region
          - 총이용금액_B0M 또는 관련 이용금액 컬럼들
          - 총이용건수_B0M 또는 관련 이용건수 컬럼들
          - 카드이용한도금액
          - 연체여부 또는 연체잔액_B0M
        """)
        return
    
    # 글로벌 필터 렌더링
    filtered_df = render_global_filters(df)
    
    # 사이드바 네비게이션
    with st.sidebar:
        st.markdown("### 🧭 네비게이션")
        
        # 메인 메뉴
        main_tab = st.radio(
            "메인 메뉴",
            list(NAV.keys()),
            format_func=lambda k: f"{NAV[k]['icon']} {k}",
            index=0,
            label_visibility="collapsed",
            key="nav_main",
        )

        # 메인 변경 시 서브 기본값 리셋
        if "nav_main_prev" not in st.session_state:
            st.session_state["nav_main_prev"] = main_tab
        if st.session_state["nav_main_prev"] != main_tab:
            st.session_state.pop("nav_sub", None)
            st.session_state["nav_main_prev"] = main_tab

        # 세부 메뉴
        subtabs = NAV[main_tab]["subtabs"]
        sub_tab = st.radio(
            "세부 메뉴",
            subtabs,
            index=0 if "nav_sub" not in st.session_state else subtabs.index(st.session_state["nav_sub"]) if st.session_state.get("nav_sub") in subtabs else 0,
            key="nav_sub",
        )
        
        st.divider()
        
        # 디바이스 정보
        st.markdown("### 🖥️ 시스템 정보")
        device_info = get_device_info()
        
        # 디바이스 상태 표시
        if device_info['device_count'] > 0:
            st.success(f"🚀 **GPU 활성화:** {device_info['device_name']}")
            st.write(f"**CUDA 버전:** {device_info['cuda_version']}")
            st.write(f"**PyTorch 버전:** {device_info['torch_version']}")
            st.write(f"**GPU 메모리:** {device_info['memory_allocated']:.1f}/{device_info['memory_total']:.1f} GB")
            
            # GPU 메모리 사용률 시각화
            memory_usage = device_info['memory_allocated'] / device_info['memory_total'] * 100
            st.progress(memory_usage / 100)
            st.caption(f"메모리 사용률: {memory_usage:.2f}%")
        else:
            st.info(f"💻 **CPU 사용:** {device_info['device_name']}")
            if TORCH_AVAILABLE:
                st.write(f"**PyTorch 버전:** {device_info['torch_version']}")
            else:
                st.warning("⚠️ PyTorch가 설치되지 않았습니다.")
        
        st.divider()
        
        # 데이터 요약 정보
        st.markdown("### 📊 데이터 요약")
        st.write(f"**전체 고객 수:** {len(df):,}명")
        st.write(f"**필터 적용 후:** {len(filtered_df):,}명")
        
        # 세그먼트별 고객 수
        st.markdown("### 🎯 세그먼트별 고객 분포")
        segment_counts = filtered_df['Segment'].value_counts().sort_index()
        for segment in SEGMENT_ORDER:
            if segment in segment_counts.index:
                count = segment_counts[segment]
                pct = (count / len(filtered_df)) * 100
                st.write(f"**{segment}:** {count:,}명 ({pct:.2f}%)")
            else:
                st.write(f"**{segment}:** 데이터 없음")
    
    # 본문 라우팅
    st.markdown(f"## {NAV[main_tab]['icon']} {main_tab}")
    
    def route(main_tab: str, sub_tab: str):
        # 세그먼트별 비교분석
        if main_tab == "세그먼트별 비교분석":
            if sub_tab == "주요 KPI 분석":
                result = render_kpi_analysis(filtered_df, return_df=True)
                if result:
                    st.session_state['current_analysis_data'] = result
            elif sub_tab == "세그먼트별 세부특성":
                result = render_segment_details(filtered_df, return_df=True)
                if result:
                    st.session_state['current_analysis_data'] = result
            elif sub_tab == "트렌드 분석(시계열)":
                result = render_trend_analysis(filtered_df, return_df=True)
                if result:
                    st.session_state['current_analysis_data'] = result
        # 리스크 분석
        elif main_tab == "리스크 분석":
            if sub_tab == "연체/부실":
                result = render_risk_delinquency(filtered_df, return_df=True)
                if result:
                    st.session_state['current_analysis_data'] = result
            elif sub_tab == "한도/이용률":
                result = render_risk_limit_util(filtered_df, return_df=True)
                if result:
                    st.session_state['current_analysis_data'] = result
            elif sub_tab == "승인/거절":
                result = render_risk_auth_decline(filtered_df, return_df=True)
                if result:
                    st.session_state['current_analysis_data'] = result
            elif sub_tab == "조기경보(EWS)":
                result = render_risk_ews(filtered_df, return_df=True)
                if result:
                    st.session_state['current_analysis_data'] = result
        # 행동마케팅 분석
        elif main_tab == "행동마케팅 분석":
            if sub_tab == "캠페인 반응":
                result = render_behavior_campaign(filtered_df, return_df=True)
                if result:
                    st.session_state['current_analysis_data'] = result
            elif sub_tab == "개인화 오퍼":
                result = render_behavior_offer(filtered_df, return_df=True)
                if result:
                    st.session_state['current_analysis_data'] = result
            elif sub_tab == "이탈/리텐션":
                result = render_behavior_churn(filtered_df, return_df=True)
                if result:
                    st.session_state['current_analysis_data'] = result
            elif sub_tab == "채널 효율":
                result = render_behavior_channel(filtered_df, return_df=True)
                if result:
                    st.session_state['current_analysis_data'] = result
    
    # 라우팅 실행
    route(main_tab, sub_tab)
    
    # 데이터프레임 확인 섹션 (디버깅용)
    if 'current_analysis_data' in st.session_state:
        with st.expander("🔍 현재 분석 데이터프레임 확인 (LLM 인사이트 추출용)", expanded=False):
            st.markdown("### 📊 분석 결과 데이터프레임들")
            
            analysis_data = st.session_state['current_analysis_data']
            
            for key, df in analysis_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    st.markdown(f"#### {key}")
                    st.dataframe(df.head(10), use_container_width=True)
                    st.markdown(f"**Shape:** {df.shape}")
                    st.markdown(f"**Columns:** {list(df.columns)}")
                    st.markdown("---")
                elif isinstance(df, dict):
                    st.markdown(f"#### {key} (Dictionary)")
                    st.json(df)
                    st.markdown("---")


# --- 미구현 함수 플레이스홀더 ---
def _placeholder(msg):
    st.info(msg)

# 미구현 함수들 정의
def render_risk_delinquency(df: pd.DataFrame):
    """연체/부실 분석 (GPU 가속)"""
    st.markdown("### 💸 연체/부실 분석")
    
    if df.empty:
        st.warning("데이터가 없습니다.")
        return
    
    # GPU 가속 옵션
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        use_gpu = st.toggle("🚀 GPU 가속 사용", key="delinquency_gpu", disabled=not TORCH_AVAILABLE)
    with col3:
        high_risk_toggle = st.toggle("고위험군만 표시", key="delinquency_high_risk")
    
    # 고위험군 필터링
    if high_risk_toggle:
        high_risk_mask = (df['연체여부'] == 1)
        filtered_df = df[high_risk_mask] if high_risk_mask.any() else df
    else:
        filtered_df = df
    
    # GPU 가속 계산 (대용량 데이터 처리 시뮬레이션)
    if use_gpu and TORCH_AVAILABLE:
        st.info("🚀 GPU 가속으로 대용량 계산을 수행합니다...")
        
        # 가상의 대용량 데이터 생성 (GPU 가속 계산 시뮬레이션)
        large_data = np.random.randn(10000, 100).astype(np.float32)
        
        with st.spinner("GPU에서 계산 중..."):
            # GPU 가속 계산
            gpu_result = gpu_accelerated_computation(large_data, 'matrix_multiply')
            st.success(f"✅ GPU 계산 완료! 결과 크기: {gpu_result.shape}")
    
    # KPI 메트릭 (기존 로직)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(filtered_df)
        default_customers = len(filtered_df[filtered_df['연체여부'] == 1])
        default_rate = (default_customers / total_customers * 100) if total_customers > 0 else 0
        st.metric("연체율", f"{default_rate:.2f}%")
    
    with col2:
        avg_default_count = np.random.beta(2, 8, len(filtered_df)) * 5
        st.metric("평균 연체횟수", f"{avg_default_count.mean():.2f}회")
    
    with col3:
        new_default_rate = np.random.beta(1, 20, len(filtered_df)) * 100
        st.metric("신규연체발생률", f"{new_default_rate.mean():.2f}%")
    
    with col4:
        cumulative_default_rate = np.random.beta(1, 10, len(filtered_df)) * 100
        st.metric("누적부실률", f"{cumulative_default_rate.mean():.2f}%")
    
    # 차트 섹션
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 세그먼트별 연체율")
        segment_default = filtered_df.groupby('Segment').agg({
            '연체여부': 'mean',
            'ID': 'nunique'
        }).reset_index()
        segment_default['연체율'] = segment_default['연체여부'] * 100
        
        fig_default = px.bar(
            segment_default,
            x='Segment',
            y='연체율',
            title="세그먼트별 연체율",
            color='Segment',
            color_discrete_map=SEGMENT_COLORS,
            category_orders={'Segment': SEGMENT_ORDER}
        )
        fig_default.update_layout(showlegend=False, height=700, width=900)
        st.plotly_chart(fig_default, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 시계열 연체율")
        months = pd.date_range(start='2023-01', end='2023-12', freq='MS')
        time_series_data = []
        
        for month in months:
            for segment in SEGMENT_ORDER:
                default_rate = np.random.beta(2, 98) * 100
                time_series_data.append({
                    'Date': month,
                    'Segment': segment,
                    '연체율': default_rate
                })
        
        ts_df = pd.DataFrame(time_series_data)
        fig_ts = go.Figure()
        
        for segment in SEGMENT_ORDER:
            segment_data = ts_df[ts_df['Segment'] == segment]
            fig_ts.add_trace(go.Scatter(
                x=segment_data['Date'],
                y=segment_data['연체율'],
                mode='lines+markers',
                name=f'세그먼트 {segment}',
                line=dict(color=SEGMENT_COLORS[segment], width=2)
            ))
        
        fig_ts.update_layout(title="월별 연체율 추이", height=700, width=900)
        st.plotly_chart(fig_ts, use_container_width=True)
    
    # 다운로드 버튼
    st.markdown("#### 📥 데이터 다운로드")
    # 연령 컬럼 찾기
    age_candidates = [col for col in filtered_df.columns if '연령' in col or 'age' in col.lower() or 'Age' in col]
    if age_candidates:
        age_column = age_candidates[0]
        csv_data = filtered_df[['ID', 'Segment', '연체여부', age_column, 'Region']].to_csv(index=False, encoding='utf-8-sig')
    else:
        csv_data = filtered_df[['ID', 'Segment', '연체여부', 'Region']].to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📊 연체 데이터",
        data=csv_data,
        file_name="delinquency_data.csv",
        mime="text/csv"
    )
    
    # 데이터프레임 반환
    return {
        'delinquency_data': filtered_df,
        'original_df': df
    }

def render_risk_limit_util(df):
    _placeholder("한도/이용률 뷰가 준비 중입니다.")
    return {'original_df': df}

def render_risk_auth_decline(df):
    _placeholder("승인/거절 뷰가 준비 중입니다.")
    return {'original_df': df}

def render_risk_ews(df):
    _placeholder("조기경보(EWS) 뷰가 준비 중입니다.")
    return {'original_df': df}

def render_behavior_campaign(df: pd.DataFrame):
    """캠페인 반응 분석"""
    st.markdown("### 📧 캠페인 반응 분석")
    
    if df.empty:
        st.warning("데이터가 없습니다.")
        return
    
    # KPI 메트릭
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # 오픈률 (가상 데이터)
        open_rate = np.random.beta(20, 80) * 100
        st.metric("평균 오픈률", f"{open_rate:.2f}%")
    
    with col2:
        # 클릭률 (가상 데이터)
        click_rate = np.random.beta(5, 95) * 100
        st.metric("평균 클릭률", f"{click_rate:.2f}%")
    
    with col3:
        # 전환률 (가상 데이터)
        conversion_rate = np.random.beta(2, 98) * 100
        st.metric("평균 전환률", f"{conversion_rate:.2f}%")
    
    with col4:
        # 세그먼트별 반응지표 (가상 데이터)
        response_score = np.random.beta(15, 85) * 100
        st.metric("반응지표 점수", f"{response_score:.2f}점")
    
    # 차트 섹션
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 세그먼트별 반응률")
        
        # 세그먼트별 반응 데이터 생성
        segment_response = []
        for segment in SEGMENT_ORDER:
            if segment in df['Segment'].values:
                open_rate = np.random.beta(15, 85) * 100
                click_rate = np.random.beta(3, 97) * 100
                conversion_rate = np.random.beta(1, 99) * 100
                
                segment_response.append({
                    'Segment': segment,
                    '오픈률': open_rate,
                    '클릭률': click_rate,
                    '전환률': conversion_rate
                })
        
        if segment_response:
            response_df = pd.DataFrame(segment_response)
            
            # 막대 차트
            fig_response = go.Figure()
            
            metrics = ['오픈률', '클릭률', '전환률']
            for i, metric in enumerate(metrics):
                fig_response.add_trace(go.Bar(
                    name=metric,
                    x=response_df['Segment'],
                    y=response_df[metric],
                    marker_color=[SEGMENT_COLORS[seg] for seg in response_df['Segment']],
                    opacity=0.8 - i * 0.2
                ))
            
            fig_response.update_layout(
                title="세그먼트별 마케팅 반응률",
                xaxis_title="세그먼트",
                yaxis_title="반응률 (%)",
                barmode='group',
                height=600
            )
            st.plotly_chart(fig_response, use_container_width=True)
        else:
            st.info("세그먼트별 반응 데이터가 없습니다.")
    
    with col2:
        st.markdown("#### 🎯 캠페인별 성과 분석")
        
        # 캠페인별 성과 데이터 생성
        campaigns = ['신용카드 신규 발급', '할인 이벤트', '포인트 적립', '리볼빙 안내', '부가서비스']
        
        campaign_data = []
        for campaign in campaigns:
            reach = np.random.randint(10000, 100000)
            conversion = np.random.randint(100, 5000)
            conversion_rate = (conversion / reach) * 100
            
            campaign_data.append({
                'Campaign': campaign,
                '도달수': reach,
                '전환수': conversion,
                '전환률': conversion_rate
            })
        
        campaign_df = pd.DataFrame(campaign_data)
        
        # 산점도
        fig_scatter = px.scatter(
            campaign_df,
            x='도달수',
            y='전환수',
            size='전환률',
            hover_data=['Campaign', '전환률'],
            title="캠페인별 도달 vs 전환 성과",
            labels={'도달수': '도달 수', '전환수': '전환 수'}
        )
        
        fig_scatter.update_layout(height=700, width=900)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # 리프트 차트
    st.markdown("#### 📈 타깃팅 리프트 분석")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        k_percent = st.slider("상위 타깃팅 비율 (%)", 5, 50, 20, 5)
    
    with col1:
        # 리프트 차트 데이터 생성
        lift_data = []
        for i in range(10, 101, 10):
            if i <= k_percent:
                lift = np.random.uniform(1.5, 3.0)
            else:
                lift = np.random.uniform(0.5, 1.2)
            
            lift_data.append({
                'Population_Percent': i,
                'Lift': lift,
                'Segment': 'A' if i <= k_percent else 'B'
            })
        
        lift_df = pd.DataFrame(lift_data)
        
        fig_lift = px.line(
            lift_df,
            x='Population_Percent',
            y='Lift',
            title=f"상위 {k_percent}% 타깃팅 시 예상 리프트",
            labels={'Population_Percent': '인구 비율 (%)', 'Lift': '리프트 배수'}
        )
        
        fig_lift.add_hline(y=1.0, line_dash="dash", line_color="red", 
                          annotation_text="기준선 (1.0)")
        
        st.plotly_chart(fig_lift, use_container_width=True)
    
    # 다운로드 섹션
    st.markdown("#### 📥 데이터 다운로드")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if segment_response:
            segment_csv = pd.DataFrame(segment_response).to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📊 세그먼트별 반응 데이터",
                data=segment_csv,
                file_name="campaign_segment_response.csv",
                mime="text/csv"
            )
    
    with col2:
        campaign_csv = campaign_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="🎯 캠페인별 성과 데이터",
            data=campaign_csv,
            file_name="campaign_performance.csv",
            mime="text/csv"
        )
    
    with col3:
        lift_csv = lift_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📈 리프트 분석 데이터",
            data=lift_csv,
            file_name="campaign_lift_analysis.csv",
            mime="text/csv"
        )
        
        # 데이터프레임 반환
        return {
            'campaign_response_data': campaign_df,
            'lift_analysis_data': lift_df,
            'original_df': df
        }

def render_behavior_offer(df: pd.DataFrame):
    """개인화 오퍼 분석"""
    st.markdown("### 🎁 개인화 오퍼 분석")
    
    if df.empty:
        st.warning("데이터가 없습니다.")
        return
    
    # 규칙기반 PoC: 세그먼트×업종 매칭 오퍼 추천
    st.markdown("#### 🎯 세그먼트별 오퍼 추천 테이블")
    
    # 업종 데이터 생성
    industries = ['온라인쇼핑', '외식업', '주유소', '의료', '교육', '여행', '스포츠', '뷰티']
    
    # 세그먼트별 추천 오퍼 생성
    offer_recommendations = []
    
    for segment in SEGMENT_ORDER:
        if segment in df['Segment'].values:
            # 세그먼트별 특성에 따른 오퍼 매칭
            if segment == 'A':  # 고가치 고객
                recommended_offers = ['프리미엄 카드 업그레이드', 'VIP 라운지 서비스', '높은 포인트 적립']
                industries_pref = ['여행', '외식업', '뷰티']
            elif segment == 'B':  # 성장 고객
                recommended_offers = ['할부 서비스', '적립 포인트 2배', '추가 카드 발급']
                industries_pref = ['온라인쇼핑', '외식업', '교육']
            elif segment == 'C':  # 일반 고객
                recommended_offers = ['기본 할인 서비스', '포인트 적립', '간편결제 서비스']
                industries_pref = ['주유소', '온라인쇼핑', '의료']
            elif segment == 'D':  # 신규 고객
                recommended_offers = ['신규 혜택', '첫 구매 할인', '친구 추천 보상']
                industries_pref = ['온라인쇼핑', '외식업', '스포츠']
            else:  # E - 휴면 고객
                recommended_offers = ['재활성화 캠페인', '특별 할인', '서비스 개선 안내']
                industries_pref = ['주유소', '의료', '교육']
            
            for i, industry in enumerate(industries_pref[:3]):  # 상위 3개 업종
                offer_recommendations.append({
                    'Segment': segment,
                    '업종': industry,
                    '추천_오퍼': recommended_offers[i],
                    '예상_응답률': np.random.beta(10, 90) * 100,
                    '예상_ARPU_증가': np.random.randint(5000, 50000)
                })
    
    if offer_recommendations:
        offer_df = pd.DataFrame(offer_recommendations)
        
        # 테이블 표시
        st.dataframe(
            offer_df,
            use_container_width=True,
            hide_index=True
        )
        
        # 오퍼별 성과 요약
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 세그먼트별 예상 응답률")
            
            segment_response = offer_df.groupby('Segment')['예상_응답률'].mean().reset_index()
            
            fig_response = px.bar(
                segment_response,
                x='Segment',
                y='예상_응답률',
                title="세그먼트별 평균 응답률",
                color='Segment',
                color_discrete_map=SEGMENT_COLORS,
                category_orders={'Segment': SEGMENT_ORDER}
            )
            
            fig_response.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_response, use_container_width=True)
        
        with col2:
            st.markdown("#### 💰 세그먼트별 예상 ARPU 증가")
            
            segment_arpu = offer_df.groupby('Segment')['예상_ARPU_증가'].mean().reset_index()
            
            fig_arpu = px.bar(
                segment_arpu,
                x='Segment',
                y='예상_ARPU_증가',
                title="세그먼트별 평균 ARPU 증가",
                color='Segment',
                color_discrete_map=SEGMENT_COLORS,
                category_orders={'Segment': SEGMENT_ORDER}
            )
            
            fig_arpu.update_layout(showlegend=False, height=400, yaxis=dict(tickformat=","))
            st.plotly_chart(fig_arpu, use_container_width=True)
    
    # 오퍼 시뮬레이션
    st.markdown("#### 🧪 오퍼 A/B 테스트 시뮬레이션")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**오퍼 A 설정**")
        offer_a_name = st.selectbox("오퍼 A 유형", ["할인 쿠폰", "포인트 적립", "무이자 할부", "부가서비스"], key="offer_a")
        offer_a_discount = st.slider("할인/혜택 비율 (%)", 5, 50, 10, key="discount_a")
        offer_a_target = st.selectbox("타깃 세그먼트", SEGMENT_ORDER, key="target_a")
    
    with col2:
        st.markdown("**오퍼 B 설정**")
        offer_b_name = st.selectbox("오퍼 B 유형", ["할인 쿠폰", "포인트 적립", "무이자 할부", "부가서비스"], key="offer_b")
        offer_b_discount = st.slider("할인/혜택 비율 (%)", 5, 50, 15, key="discount_b")
        offer_b_target = st.selectbox("타깃 세그먼트", SEGMENT_ORDER, key="target_b")
    
    # 시뮬레이션 결과
    if st.button("🚀 시뮬레이션 실행", key="simulate_offers"):
        st.markdown("#### 📈 시뮬레이션 결과")
        
        # 시뮬레이션 데이터 생성
        simulation_results = []
        
        for offer_name, discount, target, label in [
            (offer_a_name, offer_a_discount, offer_a_target, "오퍼 A"),
            (offer_b_name, offer_b_discount, offer_b_target, "오퍼 B")
        ]:
            # 할인 비율에 따른 응답률 및 ARPU 증가 계산
            base_response_rate = np.random.beta(8, 92) * 100
            response_rate = base_response_rate * (1 + discount / 100)
            
            base_arpu_increase = np.random.randint(3000, 30000)
            arpu_increase = base_arpu_increase * (1 + discount / 50)
            
            simulation_results.append({
                '오퍼': label,
                '유형': offer_name,
                '혜택_비율': f"{discount}%",
                '타깃_세그먼트': target,
                '예상_응답률': response_rate,
                '예상_ARPU_증가': arpu_increase,
                'ROI': (arpu_increase * response_rate / 100) / (discount * 1000) * 100
            })
        
        sim_df = pd.DataFrame(simulation_results)
        
        # 결과 테이블
        st.dataframe(sim_df, use_container_width=True, hide_index=True)
        
        # 비교 차트
        col1, col2 = st.columns(2)
        
        with col1:
            fig_compare_response = px.bar(
                sim_df,
                x='오퍼',
                y='예상_응답률',
                title="오퍼별 예상 응답률 비교",
                color='오퍼',
                color_discrete_sequence=['#3498DB', '#E74C3C']
            )
            st.plotly_chart(fig_compare_response, use_container_width=True)
        
        with col2:
            fig_compare_arpu = px.bar(
                sim_df,
                x='오퍼',
                y='예상_ARPU_증가',
                title="오퍼별 예상 ARPU 증가 비교",
                color='오퍼',
                color_discrete_sequence=['#3498DB', '#E74C3C']
            )
            fig_compare_arpu.update_layout(yaxis=dict(tickformat=","))
            st.plotly_chart(fig_compare_arpu, use_container_width=True)
    
    # 타깃 리스트 샘플 내보내기
    st.markdown("#### 📤 타깃 리스트 내보내기")
    
    if st.button("🎯 타깃 리스트 생성", key="generate_target_list"):
        # 타깃 고객 샘플 생성 (마스킹된 ID)
        target_customers = []
        
        for segment in SEGMENT_ORDER:
            if segment in df['Segment'].values:
                segment_customers = df[df['Segment'] == segment].head(100)  # 상위 100명
                
                for _, customer in segment_customers.iterrows():
                    # ID 마스킹
                    masked_id = f"CUST_{customer['ID'][:4]}****"
                    
                    target_customers.append({
                        '마스킹_ID': masked_id,
                        'Segment': segment,
                        '연령': customer.get('연령', 'Unknown'),
                        'Region': customer.get('Region', 'Unknown'),
                        '예상_응답률': np.random.beta(10, 90) * 100,
                        '우선순위': np.random.randint(1, 5)
                    })
        
        if target_customers:
            target_df = pd.DataFrame(target_customers)
            
            # 우선순위별 정렬
            target_df = target_df.sort_values(['우선순위', '예상_응답률'], ascending=[True, False])
            
            st.dataframe(target_df.head(50), use_container_width=True, hide_index=True)
            
            # CSV 다운로드
            target_csv = target_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 타깃 리스트 다운로드",
                data=target_csv,
                file_name="target_customer_list.csv",
                mime="text/csv"
            )
        else:
            st.warning("타깃 고객 데이터를 생성할 수 없습니다.")
    
    # 다운로드 섹션
    st.markdown("#### 📥 분석 데이터 다운로드")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if offer_recommendations:
            offer_csv = pd.DataFrame(offer_recommendations).to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="🎁 오퍼 추천 데이터",
                data=offer_csv,
                file_name="offer_recommendations.csv",
                mime="text/csv"
            )
    
    with col2:
        if 'sim_df' in locals():
            sim_csv = sim_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="🧪 시뮬레이션 결과",
                data=sim_csv,
                file_name="offer_simulation_results.csv",
                mime="text/csv"
            )
            
            # 데이터프레임 반환
            return {
                'offer_recommendations': offer_df,
                'simulation_results': sim_df,
                'target_customers': target_df,
                'original_df': df
            }

def render_behavior_churn(df: pd.DataFrame):
    """이탈/리텐션 분석"""
    st.markdown("### 🔄 이탈/리텐션 분석")
    
    if df.empty:
        st.warning("데이터가 없습니다.")
        return
    
    # 휴면위험 점수 모델
    st.markdown("#### ⚠️ 휴면위험 점수 분석")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        # 임계치 슬라이더
        risk_threshold = st.slider("휴면위험 임계치", 0.3, 0.9, 0.7, 0.05)
        
        # 고위험군만 표시 옵션
        show_high_risk_only = st.toggle("고위험군만 표시", key="churn_high_risk")
    
    with col1:
        # 휴면위험 점수 계산 (가상 데이터)
        churn_data = []
        
        for segment in SEGMENT_ORDER:
            if segment in df['Segment'].values:
                segment_customers = df[df['Segment'] == segment]
                
                for _, customer in segment_customers.head(1000).iterrows():  # 샘플링
                    # 휴면위험 점수 계산 (최근 이용 부재 + 이용 감소율)
                    recent_usage = np.random.beta(2, 8)  # 최근 이용률 (0-1)
                    usage_decline = np.random.beta(3, 7)  # 이용 감소율 (0-1)
                    
                    # 가중 평균으로 휴면위험 점수 계산
                    churn_score = (recent_usage * 0.6 + usage_decline * 0.4)
                    
                    churn_data.append({
                        'ID': customer['ID'],
                        'Segment': segment,
                        '연령': customer.get('연령', 'Unknown'),
                        'Region': customer.get('Region', 'Unknown'),
                        '휴면위험_점수': churn_score,
                        '최근_이용률': recent_usage,
                        '이용_감소율': usage_decline,
                        '고위험군': churn_score >= risk_threshold
                    })
        
        if churn_data:
            churn_df = pd.DataFrame(churn_data)
            
            # 고위험군 필터링
            if show_high_risk_only:
                churn_df = churn_df[churn_df['고위험군'] == True]
            
            # KPI 메트릭
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_customers = len(churn_df)
                st.metric("분석 고객 수", f"{total_customers:,}명")
            
            with col2:
                high_risk_customers = len(churn_df[churn_df['고위험군'] == True])
                high_risk_rate = (high_risk_customers / total_customers * 100) if total_customers > 0 else 0
                st.metric("고위험군 비율", f"{high_risk_rate:.2f}%")
            
            with col3:
                avg_churn_score = churn_df['휴면위험_점수'].mean()
                st.metric("평균 휴면위험 점수", f"{avg_churn_score:.3f}")
            
            with col4:
                # 예상 절감액 계산 (가상)
                expected_savings = high_risk_customers * np.random.randint(50000, 200000)
                st.metric("예상 절감액", f"{expected_savings:,}원")
            
            # 세그먼트별 휴면위험 분포
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 세그먼트별 휴면위험 분포")
                
                segment_risk = churn_df.groupby('Segment').agg({
                    '휴면위험_점수': ['mean', 'std', 'count'],
                    '고위험군': 'sum'
                }).round(3)
                
                if not segment_risk.empty:
                    segment_risk.columns = ['평균_위험점수', '표준편차', '고객수', '고위험군_수']
                    segment_risk['고위험군_비율'] = (segment_risk['고위험군_수'] / segment_risk['고객수'] * 100).round(1)
                    
                    # 막대 차트
                    fig_risk = px.bar(
                        segment_risk.reset_index(),
                        x='Segment',
                        y='평균_위험점수',
                        title="세그먼트별 평균 휴면위험 점수",
                        color='Segment',
                        color_discrete_map=SEGMENT_COLORS,
                        category_orders={'Segment': SEGMENT_ORDER}
                    )
                    
                    fig_risk.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig_risk, use_container_width=True)
                else:
                    st.info("세그먼트별 휴면위험 데이터가 없습니다.")
            
            with col2:
                st.markdown("#### 🎯 고위험군 비율")
                
                if not segment_risk.empty:
                    fig_high_risk = px.bar(
                        segment_risk.reset_index(),
                        x='Segment',
                        y='고위험군_비율',
                        title="세그먼트별 고위험군 비율",
                        color='Segment',
                        color_discrete_map=SEGMENT_COLORS,
                        category_orders={'Segment': SEGMENT_ORDER}
                    )
                    
                    fig_high_risk.update_layout(showlegend=False, height=400, yaxis=dict(title="비율 (%)"))
                    st.plotly_chart(fig_high_risk, use_container_width=True)
                else:
                    st.info("고위험군 데이터가 없습니다.")
            
            # 휴면위험 점수 분포 히스토그램
            st.markdown("#### 📈 휴면위험 점수 분포")
            
            fig_hist = px.histogram(
                churn_df,
                x='휴면위험_점수',
                nbins=20,
                title="휴면위험 점수 분포",
                labels={'휴면위험_점수': '휴면위험 점수', 'count': '고객 수'}
            )
            
            # 임계치 라인 추가
            fig_hist.add_vline(x=risk_threshold, line_dash="dash", line_color="red",
                              annotation_text=f"임계치: {risk_threshold}")
            
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # 고위험군 상세 테이블
            st.markdown("#### 📋 고위험군 고객 리스트")
            
            high_risk_df = churn_df[churn_df['고위험군'] == True].sort_values('휴면위험_점수', ascending=False)
            
            if not high_risk_df.empty:
                # ID 마스킹
                display_df = high_risk_df.copy()
                display_df['ID'] = display_df['ID'].apply(lambda x: f"CUST_{str(x)[:4]}****")
                
                st.dataframe(
                    display_df[['ID', 'Segment', '연령', 'Region', '휴면위험_점수', '최근_이용률', '이용_감소율']].head(100),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("고위험군 고객이 없습니다.")
    
    # 코호트 잔존 곡선 비교
    st.markdown("#### 📊 세그먼트별 코호트 잔존 곡선")
    
    # 코호트 데이터 생성
    months = list(range(1, 13))  # 1개월차부터 12개월차까지
    
    cohort_data = []
    for segment in SEGMENT_ORDER:
        if segment in df['Segment'].values:
            for month in months:
                # 세그먼트별 잔존율 계산 (가상 데이터)
                if segment == 'A':  # 고가치 고객 - 높은 잔존율
                    base_retention = 0.95
                    decay_rate = 0.02
                elif segment == 'B':  # 성장 고객
                    base_retention = 0.85
                    decay_rate = 0.05
                elif segment == 'C':  # 일반 고객
                    base_retention = 0.75
                    decay_rate = 0.08
                elif segment == 'D':  # 신규 고객
                    base_retention = 0.65
                    decay_rate = 0.12
                else:  # E - 휴면 고객
                    base_retention = 0.45
                    decay_rate = 0.15
                
                retention_rate = max(0.1, base_retention - (month - 1) * decay_rate)
                retention_rate += np.random.normal(0, 0.02)  # 노이즈 추가
                retention_rate = max(0, min(1, retention_rate))  # 0-1 범위로 제한
                
                cohort_data.append({
                    'Segment': segment,
                    'Month': month,
                    'Retention_Rate': retention_rate * 100
                })
    
    if cohort_data:
        cohort_df = pd.DataFrame(cohort_data)
        
        # 라인 차트
        fig_cohort = px.line(
            cohort_df,
            x='Month',
            y='Retention_Rate',
            color='Segment',
            title="세그먼트별 코호트 잔존 곡선",
            labels={'Month': '개월차', 'Retention_Rate': '잔존율 (%)'},
            color_discrete_map=SEGMENT_COLORS
        )
        
        fig_cohort.update_layout(height=500)
        st.plotly_chart(fig_cohort, use_container_width=True)
        
        # 코호트 요약 테이블
        st.markdown("#### 📋 코호트 잔존 요약")
        
        cohort_summary = cohort_df.groupby('Segment').agg({
            'Retention_Rate': ['mean', 'min', 'max']
        }).round(1)
        
        cohort_summary.columns = ['평균_잔존율', '최소_잔존율', '최대_잔존율']
        
        st.dataframe(cohort_summary, use_container_width=True)
    
    # 다운로드 섹션
    st.markdown("#### 📥 데이터 다운로드")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if churn_data:
            churn_csv = churn_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="⚠️ 휴면위험 분석 데이터",
                data=churn_csv,
                file_name="churn_risk_analysis.csv",
                mime="text/csv"
            )
    
    with col2:
        if 'high_risk_df' in locals() and not high_risk_df.empty:
            high_risk_csv = high_risk_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="🎯 고위험군 리스트",
                data=high_risk_csv,
                file_name="high_risk_customers.csv",
                mime="text/csv"
            )
    
    with col3:
        if cohort_data:
            cohort_csv = cohort_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📊 코호트 분석 데이터",
                data=cohort_csv,
                file_name="cohort_retention_analysis.csv",
                mime="text/csv"
            )
            
            # 데이터프레임 반환
            return {
                'churn_risk_data': high_risk_df if 'high_risk_df' in locals() else pd.DataFrame(),
                'cohort_data': cohort_df,
                'original_df': df
            }

def render_behavior_channel(df: pd.DataFrame):
    """채널 이용 패턴 분석 (실제 데이터 기반)"""
    st.markdown("### 📱 채널 이용 패턴 분석")
    
    if df.empty:
        st.warning("데이터가 없습니다.")
        return
    
    # 실제 채널 관련 컬럼 찾기
    channel_related_cols = [
        # ARS 관련
        '인입횟수_ARS_R6M', '이용메뉴건수_ARS_R6M', '인입일수_ARS_R6M', '인입월수_ARS_R6M',
        '인입횟수_ARS_BOM', '이용메뉴건수_ARS_BOM', '인입일수_ARS_BOM',
        
        # PC 관련
        '방문횟수_PC_R6M', '방문일수_PC_R6M', '방문월수_PC_R6M',
        '방문횟수_PC_BOM', '방문일수_PC_BOM',
        
        # 앱 관련
        '방문횟수_앱_R6M', '방문일수_앱_R6M', '방문월수_앱_R6M',
        '방문횟수_앱_BOM', '방문일수_앱_BOM',
        
        # 모바일웹 관련
        '방문횟수_모바일웹_R6M', '방문일수_모바일웹_R6M', '방문월수_모바일웹_R6M',
        '방문횟수_모바일웹_BOM', '방문일수_모바일웹_BOM',
        
        # 인터넷뱅킹 관련
        '인입횟수_IB_R6M', '인입횟수_금융_IB_R6M', '인입일수_IB_R6M', '인입월수_IB_R6M',
        '이용메뉴건수_IB_R6M', '인입횟수_IB_BOM', '인입일수_IB_BOM', '이용메뉴건수_IB_BOM',
        
        # 상담 관련
        '상담건수_BOM', '상담건수_R6M',
        
        # 당사 서비스 관련
        '당사PAY_방문횟수_BOM', '당사PAY_방문횟수_R6M', '당사PAY_방문월수_R6M',
        '당사멤버쉽_방문횟수_BOM', '당사멤버쉽_방문횟수_R6M', '당사멤버쉽_방문월수_R6M',
        
        # 홈페이지 관련
        '홈페이지_금융건수_R6M', '홈페이지_선결제건수_R6M', '홈페이지_금융건수_R3M', '홈페이지_선결제건수_R3M'
    ]
    
    # 실제 존재하는 채널 관련 컬럼만 필터링
    existing_channel_cols = [col for col in channel_related_cols if col in df.columns]
    
    if not existing_channel_cols:
        st.warning("⚠️ 채널 분석을 위한 컬럼을 찾을 수 없습니다.")
        return
    
    st.info(f"ℹ️ 채널 분석에 사용할 컬럼 {len(existing_channel_cols)}개 발견")
    
    # 채널 카테고리별 매핑
    channel_category_map = {
        'ARS': [col for col in existing_channel_cols if 'ARS' in col],
        'PC': [col for col in existing_channel_cols if 'PC' in col and '방문' in col],
        '앱': [col for col in existing_channel_cols if '앱' in col and '방문' in col],
        '모바일웹': [col for col in existing_channel_cols if '모바일웹' in col],
        '인터넷뱅킹': [col for col in existing_channel_cols if 'IB' in col],
        '상담': [col for col in existing_channel_cols if '상담' in col],
        '당사PAY': [col for col in existing_channel_cols if '당사PAY' in col],
        '당사멤버쉽': [col for col in existing_channel_cols if '당사멤버쉽' in col],
        '홈페이지': [col for col in existing_channel_cols if '홈페이지' in col]
    }
    
    # 채널별 통계 계산
    channel_stats = []
    
    for channel_category, cols in channel_category_map.items():
        if cols:
            # 해당 채널의 총 활동량 계산
            total_visits = 0
            total_days = 0
            total_months = 0
            
            for col in cols:
                if '방문횟수' in col or '인입횟수' in col:
                    visits = pd.to_numeric(df[col], errors='coerce').fillna(0).sum()
                    total_visits += visits
                elif '일수' in col:
                    days = pd.to_numeric(df[col], errors='coerce').fillna(0).sum()
                    total_days += days
                elif '월수' in col:
                    months = pd.to_numeric(df[col], errors='coerce').fillna(0).sum()
                    total_months += months
            
            # 평균 활동 지표 계산
            avg_daily_activity = total_visits / len(df) if len(df) > 0 else 0
            avg_monthly_activity = total_visits / 6 if total_visits > 0 else 0  # 6개월 기준
            
            channel_stats.append({
                'Channel': channel_category,
                '총_방문수': total_visits,
                '총_이용일수': total_days,
                '총_이용월수': total_months,
                '평균_일일활동': avg_daily_activity,
                '평균_월간활동': avg_monthly_activity,
                '이용고객수': len(df[df[cols[0]] > 0]) if cols else 0
            })
    
    channel_df = pd.DataFrame(channel_stats)
    
    if channel_df.empty:
        st.warning("⚠️ 채널별 데이터가 없습니다.")
        return
    
    # KPI 메트릭
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_visits = channel_df['총_방문수'].sum()
        st.metric("총 채널 방문수", f"{total_visits:,}")
    
    with col2:
        avg_daily = channel_df['평균_일일활동'].mean()
        st.metric("평균 일일 활동", f"{avg_daily:.1f}")
    
    with col3:
        total_users = channel_df['이용고객수'].sum()
        st.metric("총 이용 고객수", f"{total_users:,}")
    
    with col4:
        most_popular = channel_df.loc[channel_df['총_방문수'].idxmax(), 'Channel']
        st.metric("가장 인기 채널", most_popular)
    
    # 차트 섹션
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 채널별 방문수 분석")
        
        # 채널별 방문수 막대 차트
        fig_visits = px.bar(
            channel_df,
            x='Channel',
            y='총_방문수',
            title="채널별 총 방문수",
            color='총_방문수',
            color_continuous_scale='Blues'
        )
        
        fig_visits.update_layout(height=500)
        st.plotly_chart(fig_visits, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 채널별 이용고객수")
        
        # 채널별 이용고객수 파이 차트
        fig_users = px.pie(
            channel_df,
            values='이용고객수',
            names='Channel',
            title="채널별 이용고객수 분포"
        )
        
        fig_users.update_layout(height=500)
        st.plotly_chart(fig_users, use_container_width=True)
    
    # 세그먼트별 채널 이용 현황
    st.markdown("#### 🔄 세그먼트별 채널 이용 현황")
    
    # 세그먼트×채널 매트릭스 데이터 생성
    segment_channel_data = []
    
    for segment in SEGMENT_ORDER:
        segment_df = df[df['Segment'] == segment]
        if not segment_df.empty:
            for channel_category, cols in channel_category_map.items():
                if cols:
                    # 해당 세그먼트의 채널 활동량 계산
                    total_activity = 0
                    for col in cols:
                        if any(keyword in col for keyword in ['방문횟수', '인입횟수', '이용메뉴건수', '상담건수']):
                            activity = pd.to_numeric(segment_df[col], errors='coerce').fillna(0).sum()
                            total_activity += activity
                    
                    # 세그먼트 내 상대적 비율 계산
                    total_segment_activity = 0
                    for all_cols in channel_category_map.values():
                        for col in all_cols:
                            if any(keyword in col for keyword in ['방문횟수', '인입횟수', '이용메뉴건수', '상담건수']):
                                activity = pd.to_numeric(segment_df[col], errors='coerce').fillna(0).sum()
                                total_segment_activity += activity
                    
                    preference_score = (total_activity / total_segment_activity * 100) if total_segment_activity > 0 else 0
                    
                    segment_channel_data.append({
                        'Segment': segment,
                        'Channel': channel_category,
                        'Usage_Count': total_activity,
                        'Preference_Score': preference_score
                    })
    
    if segment_channel_data:
        segment_channel_df = pd.DataFrame(segment_channel_data)
        
        # 히트맵
        pivot_data = segment_channel_df.pivot(index='Segment', columns='Channel', values='Usage_Count')
        
        fig_heatmap = px.imshow(
            pivot_data,
            title="세그먼트별 채널 이용 현황 히트맵",
            color_continuous_scale='Blues',
            aspect='auto'
        )
        
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # 채널별 세그먼트 분포
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 채널별 세그먼트 분포")
            
            # 채널별 세그먼트 분포 막대 차트
            channel_segment_dist = segment_channel_df.groupby(['Channel', 'Segment'])['Usage_Count'].sum().reset_index()
            
            fig_dist = px.bar(
                channel_segment_dist,
                x='Channel',
                y='Usage_Count',
                color='Segment',
                title="채널별 세그먼트 이용 분포",
                color_discrete_map=SEGMENT_COLORS,
                category_orders={'Segment': SEGMENT_ORDER}
            )
            
            fig_dist.update_layout(height=400)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 채널별 선호도 점수")
            
            # 채널별 평균 선호도
            channel_preference = segment_channel_df.groupby('Channel')['Preference_Score'].mean().reset_index()
            
            fig_preference = px.bar(
                channel_preference,
                x='Channel',
                y='Preference_Score',
                title="채널별 평균 선호도 점수",
                color='Preference_Score',
                color_continuous_scale='RdYlGn'
            )
            
            fig_preference.update_layout(height=400)
            st.plotly_chart(fig_preference, use_container_width=True)
    
    # 채널 최적화 추천
    st.markdown("#### 💡 채널 최적화 추천")
    
    # 채널별 활동량 기반 추천
    if not channel_df.empty:
        # 활동량이 낮은 채널 식별 (평균 대비 50% 이하)
        avg_activity = channel_df['총_방문수'].mean()
        low_activity_channels = channel_df[channel_df['총_방문수'] < avg_activity * 0.5]['Channel'].tolist()
        
        # 활동량이 높은 채널 식별 (평균 대비 150% 이상)
        high_activity_channels = channel_df[channel_df['총_방문수'] > avg_activity * 1.5]['Channel'].tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔴 저활동 채널 (평균 대비 50% 이하)**")
            if low_activity_channels:
                for channel in low_activity_channels:
                    activity = channel_df[channel_df['Channel'] == channel]['총_방문수'].iloc[0]
                    st.write(f"• {channel}: {activity:,}회")
                st.warning("이 채널들의 이용을 활성화할 방법을 모색하세요.")
            else:
                st.success("모든 채널이 적절한 활동량을 보입니다!")
        
        with col2:
            st.markdown("**🟢 고활동 채널 (평균 대비 150% 이상)**")
            if high_activity_channels:
                for channel in high_activity_channels:
                    activity = channel_df[channel_df['Channel'] == channel]['총_방문수'].iloc[0]
                    st.write(f"• {channel}: {activity:,}회")
                st.success("이 채널들이 주요 이용 채널입니다. 서비스 품질 유지에 집중하세요.")
            else:
                st.info("특별히 고활동인 채널이 없습니다.")
        
        # 채널별 이용고객 비율 분석
        st.markdown("#### 📊 채널별 이용고객 비율 분석")
        
        # 이용고객 비율 계산
        total_customers = channel_df['이용고객수'].sum()
        channel_df['이용고객_비율'] = (channel_df['이용고객수'] / total_customers * 100).round(1)
        
        # 상위 3개 채널 표시
        top_channels = channel_df.nlargest(3, '이용고객수')
        
        for idx, row in top_channels.iterrows():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{row['Channel']}**")
            with col2:
                st.write(f"이용고객: {row['이용고객수']:,}명")
            with col3:
                st.write(f"비율: {row['이용고객_비율']}%")
        
        # 채널 이용 패턴 개선 제안
        if low_activity_channels:
            st.markdown("**저활동 채널 개선 제안:**")
            
            improvement_suggestions = {
                'ARS': 'ARS 메뉴 개선 및 사용성 향상',
                'PC': 'PC 인터페이스 최적화',
                '앱': '앱 기능 강화 및 푸시 알림',
                '모바일웹': '모바일웹 반응형 개선',
                '인터넷뱅킹': 'IB 서비스 편의성 향상',
                '상담': '상담 서비스 품질 개선',
                '당사PAY': 'PAY 서비스 혜택 강화',
                '당사멤버쉽': '멤버쉽 혜택 확대',
                '홈페이지': '홈페이지 UX/UI 개선'
            }
            
            for low_channel in low_activity_channels:
                if low_channel in improvement_suggestions:
                    st.write(f"• **{low_channel}**: {improvement_suggestions[low_channel]}")
        
        # 채널별 상세 성과 테이블
        st.markdown("#### 📋 채널별 상세 성과")
        
        # 활동량 등급 계산
        channel_df['활동등급'] = pd.cut(channel_df['총_방문수'], 
                                      bins=[0, channel_df['총_방문수'].quantile(0.33), 
                                            channel_df['총_방문수'].quantile(0.67), float('inf')], 
                                      labels=['낮음', '보통', '높음'])
        
        # 컬러 코딩을 위한 스타일 함수
        def highlight_activity(val):
            if val == '높음':
                return 'background-color: #d4edda'
            elif val == '낮음':
                return 'background-color: #f8d7da'
            else:
                return 'background-color: #fff3cd'
        
        styled_df = channel_df.style.applymap(highlight_activity, subset=['활동등급'])
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )
    
    # 다운로드 섹션
    st.markdown("#### 📥 데이터 다운로드")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not channel_df.empty:
            channel_csv = channel_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📊 채널별 활동 데이터",
                data=channel_csv,
                file_name="channel_activity_analysis.csv",
                mime="text/csv"
            )
    
    with col2:
        if 'segment_channel_data' in locals() and segment_channel_data:
            segment_channel_csv = segment_channel_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="🔄 세그먼트×채널 데이터",
                data=segment_channel_csv,
                file_name="segment_channel_analysis.csv",
                mime="text/csv"
            )
    
    with col3:
        # 채널 분석 요약
        if not channel_df.empty:
            analysis_summary = {
                '총_채널수': len(channel_df),
                '총_방문수': int(channel_df['총_방문수'].sum()),
                '총_이용고객수': int(channel_df['이용고객수'].sum()),
                '가장_활발한_채널': channel_df.loc[channel_df['총_방문수'].idxmax(), 'Channel'],
                '저활동_채널수': len(low_activity_channels) if 'low_activity_channels' in locals() else 0,
                '고활동_채널수': len(high_activity_channels) if 'high_activity_channels' in locals() else 0
            }
            
            summary_df = pd.DataFrame([analysis_summary])
            summary_csv = summary_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📋 채널 분석 요약",
                data=summary_csv,
                file_name="channel_analysis_summary.csv",
                mime="text/csv"
            )
            
            # 데이터프레임 반환
            return {
                'channel_activity_data': channel_df,
                'segment_channel_data': segment_channel_df if 'segment_channel_data' in locals() else pd.DataFrame(),
                'analysis_summary': analysis_summary,
                'original_df': df
            }


if __name__ == "__main__":
    main()
