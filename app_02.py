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
            if 'Date' in df.columns and not df['Date'].empty:
                # Date 컬럼이 datetime 타입인지 확인하고 변환
                if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                    df['Date'] = pd.to_datetime(df['Date'])
                date_min = df['Date'].min().date()
                date_max = df['Date'].max().date()
            else:
                raise ValueError("Date column not found or empty")
        except Exception as e:
            # 날짜 컬럼이 없거나 오류 발생 시 기본값 사용
            date_min = date(2023, 1, 1)
            date_max = date(2023, 12, 31)
        
        date_range = st.date_input(
            "기간 선택",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
            key="date_range_filter"
        )
    
    with col2:
        # 연령대 필터
        age_options = sorted(df['AgeGroup'].dropna().unique().tolist())
        selected_ages = st.multiselect(
            "연령대",
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


def render_kpi_analysis(df: pd.DataFrame):
    """주요 KPI 분석"""
    st.markdown("### 📈 주요 KPI 분석")
    
    # KPI 계산
    kpi_data = calculate_kpi_metrics(df)
    
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
    
    # KPI 카드 행
    st.markdown("#### 🎯 세그먼트별 KPI 카드")
    render_kpi_cards(kpi_data_sorted)
    
    # 차트 영역 (두 줄)
    st.markdown("#### 📊 KPI 시각화")
    
    # 1행 - 좌: 막대차트, 우: 레이더차트
    col1, col2 = st.columns(2)
    
    with col1:
        render_kpi_bar_chart(kpi_data_sorted)
    
    with col2:
        render_kpi_radar_chart(kpi_data_sorted)
    
    # 2행 - 좌: 박스플롯, 우: 스택바
    col1, col2 = st.columns(2)
    
    with col1:
        render_kpi_boxplot(df)
    
    with col2:
        render_payment_method_chart(df)
    
    # CSV 다운로드
    st.markdown("---")
    csv_data = kpi_data_sorted.to_csv(index=False)
    st.download_button(
        label="📥 KPI 데이터 다운로드",
        data=csv_data,
        file_name="kpi_analysis.csv",
        mime="text/csv"
    )

def calculate_kpi_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """KPI 메트릭 계산"""
    # 기본 집계
    kpi_data = df.groupby('Segment').agg({
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

def render_kpi_cards(kpi_data: pd.DataFrame):
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
                        height: 250px;
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
                                    {segment_data['ARPU_증감']:+.1f}%
                                </div>
                            </div>
                        </div>
                        
                        <div style="font-size: 0.7rem; color: #7f8c8d;">
                            <div>객단가: {format_number(segment_data['객단가'], '원')}</div>
                            <div>이용률: {segment_data['이용률']:.1f}%</div>
                            <div>연체율: {segment_data['연체율']:.2f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 스파크라인 데이터 생성 (가상)
                    sparkline_data = np.random.normal(100, 10, 12).cumsum()
                    sparkline_data = sparkline_data / sparkline_data[0] * 100
                    
                    # 스파크라인 차트 생성
                    sparkline_fig = go.Figure()
                    sparkline_fig.add_trace(go.Scatter(
                        x=list(range(12)),
                        y=sparkline_data,
                        mode='lines',
                        line=dict(color=SEGMENT_COLORS.get(segment, '#6c757d'), width=2),
                        showlegend=False,
                        hovertemplate='%{y:.1f}%<extra></extra>'
                    ))
                    
                    sparkline_fig.update_layout(
                        width=120,
                        height=30,
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(showgrid=False, showticklabels=False),
                        yaxis=dict(showgrid=False, showticklabels=False),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    # 스파크라인 차트 표시
                    st.plotly_chart(sparkline_fig, use_container_width=False, config={'displayModeBar': False})

def render_kpi_bar_chart(kpi_data: pd.DataFrame):
    """KPI 막대 차트"""
    fig = px.bar(
        kpi_data, 
        x='Segment', 
        y='ARPU',
        title="세그먼트별 ARPU 비교",
        color='Segment',
        color_discrete_map=SEGMENT_COLORS,
        category_orders={'Segment': SEGMENT_ORDER}
    )
    
    # 막대 위 수치 라벨
    fig.update_traces(
        texttemplate='%{y:,.0f}원',
        textposition='outside'
    )
    
    fig.update_layout(
        font_size=12,
        title_font_size=16,
        showlegend=False,
        yaxis_title="ARPU (원)"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_kpi_radar_chart(kpi_data: pd.DataFrame):
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

def render_kpi_boxplot(df: pd.DataFrame):
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

def render_payment_method_chart(df: pd.DataFrame):
    """결제수단 비중 차트"""
    # 가상의 결제수단 데이터 생성
    payment_data = []
    
    for segment in SEGMENT_ORDER:
        segment_df = df[df['Segment'] == segment]
        if not segment_df.empty:
            # 가상 데이터 생성
            신판_비율 = np.random.normal(60, 15, len(segment_df))
            체크_비율 = np.random.normal(25, 10, len(segment_df))
            현금서비스_비율 = np.random.normal(15, 8, len(segment_df))
            
            # 비율 정규화
            total = 신판_비율 + 체크_비율 + 현금서비스_비율
            신판_비율 = 신판_비율 / total * 100
            체크_비율 = 체크_비율 / total * 100
            현금서비스_비율 = 현금서비스_비율 / total * 100
            
            payment_data.append({
                'Segment': segment,
                '신판': 신판_비율.mean(),
                '체크': 체크_비율.mean(),
                '현금서비스': 현금서비스_비율.mean()
            })
    
    payment_df = pd.DataFrame(payment_data)
    
    # 스택바 차트
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='신판',
        x=payment_df['Segment'],
        y=payment_df['신판'],
        marker_color='#3498DB'
    ))
    
    fig.add_trace(go.Bar(
        name='체크',
        x=payment_df['Segment'],
        y=payment_df['체크'],
        marker_color='#2ECC71'
    ))
    
    fig.add_trace(go.Bar(
        name='현금서비스',
        x=payment_df['Segment'],
        y=payment_df['현금서비스'],
        marker_color='#E67E22'
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

def render_segment_details(df: pd.DataFrame):
    """세그먼트별 세부특성"""
    st.markdown("### 🔍 세그먼트별 세부특성")
    
    # 1. 분포/구성 분석
    st.markdown("#### 📊 분포/구성 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 연령대×세그먼트 Stacked Bar (%)
        render_age_segment_distribution(df)
    
    with col2:
        # 지역×세그먼트 Heatmap
        render_region_segment_heatmap(df)
    
    # 채널 선호도 TopN
    st.markdown("##### 📱 세그먼트별 채널 선호도 (Top 5)")
    render_channel_preference(df)
    
    # 2. 업종/MCC 요약
    st.markdown("#### 🏢 세그먼트별 업종 분석")
    render_industry_analysis(df)
    
    # 3. 코호트/잔존 분석
    st.markdown("#### 📈 코호트/잔존 분석")
    render_cohort_analysis(df)
    
    # 4. 다운로드 버튼
    st.markdown("---")
    render_download_section(df)

def render_age_segment_distribution(df: pd.DataFrame):
    """연령대×세그먼트 분포 Stacked Bar"""
    # 연령대×세그먼트 교차표 생성
    cross_table = pd.crosstab(df['AgeGroup'], df['Segment'], normalize='index') * 100
    
    # 세그먼트 순서 보장
    cross_table = cross_table.reindex(columns=SEGMENT_ORDER, fill_value=0)
    
    fig = go.Figure()
    
    for segment in SEGMENT_ORDER:
        if segment in cross_table.columns:
            fig.add_trace(go.Bar(
                name=f'세그먼트 {segment}',
                x=cross_table.index,
                y=cross_table[segment],
                marker_color=SEGMENT_COLORS.get(segment, '#95A5A6'),
                hovertemplate=f'세그먼트 {segment}<br>%{{x}}: %{{y:.1f}}%<extra></extra>'
            ))
    
    fig.update_layout(
        barmode='stack',
        title="연령대별 세그먼트 분포 (%)",
        xaxis_title="연령대",
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

def render_region_segment_heatmap(df: pd.DataFrame):
    """지역×세그먼트 히트맵"""
    # 지역×세그먼트 교차표 생성 (비율)
    cross_table = pd.crosstab(df['Region'], df['Segment'], normalize='index') * 100
    
    # 세그먼트 순서 보장
    cross_table = cross_table.reindex(columns=SEGMENT_ORDER, fill_value=0)
    
    # 상위 지역만 표시 (최대 15개)
    if len(cross_table) > 15:
        cross_table = cross_table.head(15)
    
    fig = px.imshow(
        cross_table,
        title="지역별 세그먼트 분포 (%)",
        color_continuous_scale='RdYlBu_r',
        aspect="auto",
        labels=dict(x="세그먼트", y="지역", color="비율(%)")
    )
    
    fig.update_layout(
        font_size=10,
        title_font_size=14,
        xaxis={'categoryorder': 'array', 'categoryarray': SEGMENT_ORDER}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_channel_preference(df: pd.DataFrame):
    """채널 선호도 TopN"""
    # 가상의 채널 데이터 생성
    channels = ['모바일앱', '온라인', '오프라인', 'ATM', '전화', '인터넷뱅킹', 'QR결제', '간편결제']
    
    # 세그먼트별 채널 선호도 생성
    channel_data = []
    
    for segment in SEGMENT_ORDER:
        segment_df = df[df['Segment'] == segment]
        if not segment_df.empty:
            # 세그먼트별로 다른 채널 선호도 패턴
            if segment == 'A':
                channel_probs = [0.1, 0.2, 0.4, 0.1, 0.1, 0.05, 0.03, 0.02]
            elif segment == 'B':
                channel_probs = [0.15, 0.25, 0.35, 0.1, 0.1, 0.03, 0.01, 0.01]
            elif segment == 'C':
                channel_probs = [0.3, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01]
            elif segment == 'D':
                channel_probs = [0.4, 0.35, 0.15, 0.05, 0.03, 0.01, 0.005, 0.005]
            else:  # E
                channel_probs = [0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005]
            
            # Top 5 채널 선택
            if len(channels) >= 5:
                top_channels = np.random.choice(channels, 5, replace=False, p=channel_probs[:5])
            else:
                # 채널이 5개 미만인 경우
                top_channels = channels
                if len(channels) < 5:
                    # 부족한 채널은 '기타'로 채움
                    top_channels = list(top_channels) + ['기타'] * (5 - len(channels))
            channel_usage = np.random.uniform(10, 50, 5)
            channel_usage = channel_usage / channel_usage.sum() * 100
            
            for channel, usage in zip(top_channels, channel_usage):
                channel_data.append({
                    'Segment': segment,
                    'Channel': channel,
                    'Usage_Rate': usage
                })
    
    channel_df = pd.DataFrame(channel_data)
    
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
                hovertemplate=f'세그먼트 {segment}<br>%{{y}}: %{{x:.1f}}%<extra></extra>'
            ))
    
    fig.update_layout(
        title="세그먼트별 채널 선호도 (Top 5)",
        xaxis_title="이용률 (%)",
        yaxis_title="채널",
        font_size=12,
        title_font_size=14,
        height=400,
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_industry_analysis(df: pd.DataFrame):
    """업종 분석"""
    # 가상의 업종 데이터 생성
    industries = [
        '할인점', '마트', '백화점', '온라인쇼핑', '주유소', '카페', '음식점', 
        '병원', '약국', '통신', '보험', '교육', '여행', '문화', '운송'
    ]
    
    # 세그먼트별 업종 데이터 생성
    industry_data = []
    
    for segment in SEGMENT_ORDER:
        segment_df = df[df['Segment'] == segment]
        if not segment_df.empty:
            # 세그먼트별로 다른 업종 선호도
            base_amount = segment_df['총이용금액_B0M'].mean()
            
            for industry in industries:
                # 업종별 이용금액 생성
                if segment == 'A':
                    industry_amount = base_amount * np.random.uniform(0.01, 0.05)
                elif segment == 'B':
                    industry_amount = base_amount * np.random.uniform(0.02, 0.08)
                elif segment == 'C':
                    industry_amount = base_amount * np.random.uniform(0.03, 0.12)
                elif segment == 'D':
                    industry_amount = base_amount * np.random.uniform(0.04, 0.15)
                else:  # E
                    industry_amount = base_amount * np.random.uniform(0.05, 0.20)
                
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
                    format="%.1f%%"
                ) for col in SEGMENT_ORDER
            }
        )

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

def render_download_section(df: pd.DataFrame):
    """다운로드 섹션"""
    st.markdown("#### 📥 데이터 다운로드")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 연령대×세그먼트 분포 데이터
        age_segment_cross = pd.crosstab(df['AgeGroup'], df['Segment'], normalize='index') * 100
        csv_age = age_segment_cross.to_csv()
        
        st.download_button(
            label="📊 연령대×세그먼트 분포",
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
                height=600,
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

def render_trend_analysis(df: pd.DataFrame):
    """트렌드 분석(시계열)"""
    st.markdown("### 📈 트렌드 분석(시계열)")
    
    if df.empty:
        st.warning("데이터가 없습니다.")
        return
    
    # 데이터 전처리
    trend_data = prepare_trend_data(df)
    
    # 컨트롤 패널
    render_trend_controls(trend_data)
    
    # 시계열 라인 차트
    render_time_series_chart(trend_data)
    
    # YoY/HoH 변화율 분석
    render_yoy_analysis(trend_data)
    
    # 이상치/급변 탐지
    render_anomaly_detection(trend_data)
    
    # 분해 분석 (선택적)
    render_seasonal_decomposition(trend_data)
    
    # 다운로드 섹션
    render_trend_download_section(trend_data)



def main():
    """메인 함수"""
    
    # 헤더
    st.markdown('<h1 class="main-header">💳 신용카드 세그먼트 분석 대시보드</h1>', 
                unsafe_allow_html=True)
    
    # 데이터 로드
    with st.spinner("데이터를 로딩 중입니다..."):
        df = load_data()
    
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
          - 카드이용한도액
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
            st.caption(f"메모리 사용률: {memory_usage:.1f}%")
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
                st.write(f"**{segment}:** {count:,}명 ({pct:.1f}%)")
            else:
                st.write(f"**{segment}:** 데이터 없음")
    
    # 본문 라우팅
    st.markdown(f"## {NAV[main_tab]['icon']} {main_tab}")
    
    def route(main_tab: str, sub_tab: str):
        # 세그먼트별 비교분석
        if main_tab == "세그먼트별 비교분석":
            if sub_tab == "주요 KPI 분석":
                render_kpi_analysis(filtered_df)
            elif sub_tab == "세그먼트별 세부특성":
                render_segment_details(filtered_df)
            elif sub_tab == "트렌드 분석(시계열)":
                render_trend_analysis(filtered_df)
        # 리스크 분석
        elif main_tab == "리스크 분석":
            if sub_tab == "연체/부실":
                render_risk_delinquency(filtered_df)
            elif sub_tab == "한도/이용률":
                render_risk_limit_util(filtered_df)
            elif sub_tab == "승인/거절":
                render_risk_auth_decline(filtered_df)
            elif sub_tab == "조기경보(EWS)":
                render_risk_ews(filtered_df)
        # 행동마케팅 분석
        elif main_tab == "행동마케팅 분석":
            if sub_tab == "캠페인 반응":
                render_behavior_campaign(filtered_df)
            elif sub_tab == "개인화 오퍼":
                render_behavior_offer(filtered_df)
            elif sub_tab == "이탈/리텐션":
                render_behavior_churn(filtered_df)
            elif sub_tab == "채널 효율":
                render_behavior_channel(filtered_df)
    
    # 라우팅 실행
    route(main_tab, sub_tab)


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
        st.metric("평균 연체횟수", f"{avg_default_count.mean():.1f}회")
    
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
        fig_default.update_layout(showlegend=False, height=400)
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
        
        fig_ts.update_layout(title="월별 연체율 추이", height=400)
        st.plotly_chart(fig_ts, use_container_width=True)
    
    # 다운로드 버튼
    st.markdown("#### 📥 데이터 다운로드")
    csv_data = filtered_df[['ID', 'Segment', '연체여부', 'AgeGroup', 'Region']].to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📊 연체 데이터",
        data=csv_data,
        file_name="delinquency_data.csv",
        mime="text/csv"
    )

def render_risk_limit_util(df):
    _placeholder("한도/이용률 뷰가 준비 중입니다.")

def render_risk_auth_decline(df):
    _placeholder("승인/거절 뷰가 준비 중입니다.")

def render_risk_ews(df):
    _placeholder("조기경보(EWS) 뷰가 준비 중입니다.")

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
        st.metric("평균 오픈률", f"{open_rate:.1f}%")
    
    with col2:
        # 클릭률 (가상 데이터)
        click_rate = np.random.beta(5, 95) * 100
        st.metric("평균 클릭률", f"{click_rate:.1f}%")
    
    with col3:
        # 전환률 (가상 데이터)
        conversion_rate = np.random.beta(2, 98) * 100
        st.metric("평균 전환률", f"{conversion_rate:.1f}%")
    
    with col4:
        # 세그먼트별 반응지표 (가상 데이터)
        response_score = np.random.beta(15, 85) * 100
        st.metric("반응지표 점수", f"{response_score:.1f}점")
    
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
                height=400
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
        
        fig_scatter.update_layout(height=400)
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
            
            fig_response.update_layout(showlegend=False, height=300)
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
            
            fig_arpu.update_layout(showlegend=False, height=300)
            fig_arpu.update_yaxis(tickformat=",")
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
            fig_compare_arpu.update_yaxis(tickformat=",")
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
                        'AgeGroup': customer.get('AgeGroup', 'Unknown'),
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
                        'AgeGroup': customer.get('AgeGroup', 'Unknown'),
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
                st.metric("고위험군 비율", f"{high_risk_rate:.1f}%")
            
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
                
                fig_risk.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_risk, use_container_width=True)
            
            with col2:
                st.markdown("#### 🎯 고위험군 비율")
                
                fig_high_risk = px.bar(
                    segment_risk.reset_index(),
                    x='Segment',
                    y='고위험군_비율',
                    title="세그먼트별 고위험군 비율",
                    color='Segment',
                    color_discrete_map=SEGMENT_COLORS,
                    category_orders={'Segment': SEGMENT_ORDER}
                )
                
                fig_high_risk.update_layout(showlegend=False, height=300)
                fig_high_risk.update_yaxis(title="비율 (%)")
                st.plotly_chart(fig_high_risk, use_container_width=True)
            
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
                    display_df[['ID', 'Segment', 'AgeGroup', 'Region', '휴면위험_점수', '최근_이용률', '이용_감소율']].head(100),
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
        
        fig_cohort.update_layout(height=400)
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

def render_behavior_channel(df: pd.DataFrame):
    """채널 효율 분석"""
    st.markdown("### 📱 채널 효율 분석")
    
    if df.empty:
        st.warning("데이터가 없습니다.")
        return
    
    # 채널별 KPI 메트릭
    st.markdown("#### 📊 채널별 성과 KPI")
    
    # 채널 데이터 생성
    channels = ['온라인', '모바일앱', 'ATM', '전화상담', '오프라인', '카드단말기']
    
    channel_kpi_data = []
    for channel in channels:
        # 채널별 성과 지표 (가상 데이터)
        impressions = np.random.randint(10000, 100000)
        clicks = np.random.randint(100, 5000)
        conversions = np.random.randint(10, 500)
        
        ctr = (clicks / impressions * 100) if impressions > 0 else 0
        cvr = (conversions / clicks * 100) if clicks > 0 else 0
        cpa = np.random.randint(5000, 50000)  # Cost Per Acquisition
        
        channel_kpi_data.append({
            'Channel': channel,
            '도달수': impressions,
            '클릭수': clicks,
            '전환수': conversions,
            'CTR': ctr,
            'CVR': cvr,
            'CPA': cpa,
            'ROI': np.random.uniform(0.5, 3.0)
        })
    
    channel_df = pd.DataFrame(channel_kpi_data)
    
    # KPI 메트릭 카드
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_ctr = channel_df['CTR'].mean()
        st.metric("평균 CTR", f"{avg_ctr:.2f}%")
    
    with col2:
        avg_cvr = channel_df['CVR'].mean()
        st.metric("평균 CVR", f"{avg_cvr:.2f}%")
    
    with col3:
        avg_cpa = channel_df['CPA'].mean()
        st.metric("평균 CPA", f"{avg_cpa:,}원")
    
    with col4:
        avg_roi = channel_df['ROI'].mean()
        st.metric("평균 ROI", f"{avg_roi:.2f}x")
    
    # 차트 섹션
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 채널별 퍼널 분석")
        
        # 퍼널 차트
        fig_funnel = go.Figure(go.Funnel(
            y=channel_df['Channel'],
            x=channel_df['전환수'],
            textinfo="value+percent initial",
            marker=dict(color=channel_df['전환수'], colorscale='Blues')
        ))
        
        fig_funnel.update_layout(
            title="채널별 전환 퍼널",
            height=400
        )
        st.plotly_chart(fig_funnel, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 채널별 효율성")
        
        # 효율성 산점도 (CPA vs CVR)
        fig_efficiency = px.scatter(
            channel_df,
            x='CPA',
            y='CVR',
            size='전환수',
            color='ROI',
            hover_data=['Channel', 'CTR'],
            title="채널별 효율성 분석 (CPA vs CVR)",
            labels={'CPA': 'CPA (원)', 'CVR': 'CVR (%)'},
            color_continuous_scale='RdYlGn'
        )
        
        fig_efficiency.update_layout(height=400)
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    # 세그먼트×채널 교차 성과
    st.markdown("#### 🔄 세그먼트별 채널 성과")
    
    # 세그먼트×채널 매트릭스 데이터 생성
    segment_channel_data = []
    
    for segment in SEGMENT_ORDER:
        if segment in df['Segment'].values:
            for channel in channels:
                # 세그먼트별 채널 선호도 (가상 데이터)
                if segment == 'A':  # 고가치 고객
                    channel_preference = {'온라인': 0.8, '모바일앱': 0.9, 'ATM': 0.3, '전화상담': 0.7, '오프라인': 0.6, '카드단말기': 0.4}
                elif segment == 'B':  # 성장 고객
                    channel_preference = {'온라인': 0.7, '모바일앱': 0.8, 'ATM': 0.5, '전화상담': 0.4, '오프라인': 0.6, '카드단말기': 0.7}
                elif segment == 'C':  # 일반 고객
                    channel_preference = {'온라인': 0.5, '모바일앱': 0.6, 'ATM': 0.8, '전화상담': 0.3, '오프라인': 0.7, '카드단말기': 0.9}
                elif segment == 'D':  # 신규 고객
                    channel_preference = {'온라인': 0.9, '모바일앱': 0.7, 'ATM': 0.2, '전화상담': 0.8, '오프라인': 0.3, '카드단말기': 0.2}
                else:  # E - 휴면 고객
                    channel_preference = {'온라인': 0.3, '모바일앱': 0.2, 'ATM': 0.9, '전화상담': 0.1, '오프라인': 0.8, '카드단말기': 0.1}
                
                preference = channel_preference.get(channel, 0.5)
                usage_count = int(preference * np.random.randint(100, 1000))
                
                segment_channel_data.append({
                    'Segment': segment,
                    'Channel': channel,
                    'Usage_Count': usage_count,
                    'Preference_Score': preference
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
        
        fig_heatmap.update_layout(height=400)
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
            
            fig_dist.update_layout(height=300)
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
            
            fig_preference.update_layout(height=300)
            st.plotly_chart(fig_preference, use_container_width=True)
    
    # 채널 최적화 추천
    st.markdown("#### 💡 채널 최적화 추천")
    
    # 비효율 채널 식별
    inefficient_channels = channel_df[channel_df['ROI'] < 1.0]['Channel'].tolist()
    efficient_channels = channel_df[channel_df['ROI'] > 2.0]['Channel'].tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔴 비효율 채널 (ROI < 1.0)**")
        if inefficient_channels:
            for channel in inefficient_channels:
                st.write(f"• {channel}")
            st.warning("이 채널들의 투자를 재검토하거나 중단을 고려하세요.")
        else:
            st.success("모든 채널이 효율적입니다!")
    
    with col2:
        st.markdown("**🟢 고효율 채널 (ROI > 2.0)**")
        if efficient_channels:
            for channel in efficient_channels:
                st.write(f"• {channel}")
            st.success("이 채널들에 더 많은 투자를 고려하세요.")
        else:
            st.info("고효율 채널이 없습니다.")
    
    # 대체 채널 제안 (룰 베이스)
    st.markdown("#### 🔄 대체 채널 제안")
    
    if inefficient_channels:
        st.markdown("**비효율 채널 대체 제안:**")
        
        replacement_suggestions = {
            '전화상담': '온라인 챗봇',
            'ATM': '모바일앱',
            '오프라인': '온라인',
            '카드단말기': '모바일앱'
        }
        
        for inefficient_channel in inefficient_channels:
            if inefficient_channel in replacement_suggestions:
                st.write(f"• **{inefficient_channel}** → **{replacement_suggestions[inefficient_channel]}**")
                st.caption(f"예상 비용 절감: {np.random.randint(20, 50)}%")
    
    # 채널별 상세 성과 테이블
    st.markdown("#### 📋 채널별 상세 성과")
    
    # 성과 등급 계산
    channel_df['성과등급'] = pd.cut(channel_df['ROI'], 
                                  bins=[0, 1.0, 2.0, float('inf')], 
                                  labels=['낮음', '보통', '높음'])
    
    # 컬러 코딩을 위한 스타일 함수
    def highlight_performance(val):
        if val == '높음':
            return 'background-color: #d4edda'
        elif val == '낮음':
            return 'background-color: #f8d7da'
        else:
            return 'background-color: #fff3cd'
    
    styled_df = channel_df.style.applymap(highlight_performance, subset=['성과등급'])
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )
    
    # 다운로드 섹션
    st.markdown("#### 📥 데이터 다운로드")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        channel_csv = channel_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📊 채널별 KPI 데이터",
            data=channel_csv,
            file_name="channel_kpi_analysis.csv",
            mime="text/csv"
        )
    
    with col2:
        if segment_channel_data:
            segment_channel_csv = segment_channel_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="🔄 세그먼트×채널 데이터",
                data=segment_channel_csv,
                file_name="segment_channel_analysis.csv",
                mime="text/csv"
            )
    
    with col3:
        # 최적화 추천 요약
        optimization_summary = {
            '비효율_채널': ', '.join(inefficient_channels) if inefficient_channels else '없음',
            '고효율_채널': ', '.join(efficient_channels) if efficient_channels else '없음',
            '추천_대체채널': ', '.join([replacement_suggestions.get(ch, 'N/A') for ch in inefficient_channels]) if inefficient_channels else '없음'
        }
        
        optimization_df = pd.DataFrame([optimization_summary])
        optimization_csv = optimization_df.to_csv(index=False, encoding='utf-8-sig')
        
        st.download_button(
            label="💡 최적화 추천 요약",
            data=optimization_csv,
            file_name="channel_optimization_recommendations.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
