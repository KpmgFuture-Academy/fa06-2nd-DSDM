"""
신용카드 세그먼트 분석 대시보드 - 메인 앱
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from utils import load_data, apply_filters, SEGMENT_ORDER, SEGMENT_COLORS, format_number

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
</style>
""", unsafe_allow_html=True)

def render_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    """글로벌 필터 컴포넌트 렌더링"""
    st.markdown("### 🔍 글로벌 필터")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # 기간 필터 (Date range)
        try:
            date_min = df['Date'].min().date()
            date_max = df['Date'].max().date()
        except:
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

def render_compare_tab(df: pd.DataFrame):
    """세그먼트별 비교분석 탭"""
    st.markdown("## 📊 세그먼트별 비교분석")
    
    # 세부 탭 생성
    subtabs = st.tabs(["주요 KPI 분석", "세그먼트별 세부특성", "트렌드 분석(시계열)"])
    
    with subtabs[0]:
        render_kpi_analysis(df)
    
    with subtabs[1]:
        render_segment_details(df)
    
    with subtabs[2]:
        render_trend_analysis(df)

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

def render_risk_tab(df: pd.DataFrame):
    """리스크 분석 탭"""
    st.markdown("## ⚠️ 리스크 분석")
    
    # 리스크 KPI 메트릭
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(df)
        default_customers = len(df[df['연체여부'] == 1])
        default_rate = (default_customers / total_customers * 100) if total_customers > 0 else 0
        st.metric("연체율", f"{default_rate:.2f}%")
    
    with col2:
        high_risk_customers = len(df[df['연체여부'] == 1])
        st.metric("연체 고객 수", f"{high_risk_customers:,}명")
    
    with col3:
        avg_limit = df['카드이용한도액'].mean()
        st.metric("평균 한도액", f"{format_number(avg_limit, '원')}")
    
    with col4:
        utilization_rate = (df['총이용금액_B0M'] / df['카드이용한도액']).mean() * 100
        st.metric("평균 이용률", f"{utilization_rate:.1f}%")
    
    # 세그먼트별 리스크 지표
    st.markdown("### 🎯 세그먼트별 리스크 지표")
    
    # 리스크 지표 계산
    risk_metrics = df.groupby('Segment').agg({
        '연체여부': 'mean',
        '카드이용한도액': 'mean',
        '총이용금액_B0M': 'mean',
        'ID': 'nunique'
    }).reset_index()
    
    risk_metrics.columns = ['Segment', '연체율', '평균한도', '평균이용금액', '고객수']
    risk_metrics['연체율'] = risk_metrics['연체율'] * 100
    risk_metrics['이용률'] = (risk_metrics['평균이용금액'] / risk_metrics['평균한도']) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 연체율 차트
        fig_default = px.bar(
            risk_metrics, 
            x='Segment', 
            y='연체율',
            title="세그먼트별 연체율",
            color='Segment',
            color_discrete_map=SEGMENT_COLORS,
            category_orders={'Segment': SEGMENT_ORDER}
        )
        fig_default.update_traces(
            texttemplate='%{y:.2f}%',
            textposition='outside'
        )
        fig_default.update_layout(
            font_size=14,
            title_font_size=18,
            showlegend=False
        )
        st.plotly_chart(fig_default, use_container_width=True)
    
    with col2:
        # 이용률 차트
        fig_util = px.bar(
            risk_metrics, 
            x='Segment', 
            y='이용률',
            title="세그먼트별 한도 이용률",
            color='Segment',
            color_discrete_map=SEGMENT_COLORS,
            category_orders={'Segment': SEGMENT_ORDER}
        )
        fig_util.update_traces(
            texttemplate='%{y:.1f}%',
            textposition='outside'
        )
        fig_util.update_layout(
            font_size=14,
            title_font_size=18,
            showlegend=False
        )
        st.plotly_chart(fig_util, use_container_width=True)
    
    # 신용한도 분석
    st.markdown("### 💳 신용한도 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 한도 분포 히스토그램
        fig_hist = px.histogram(
            df, 
            x='카드이용한도액',
            nbins=50,
            title="신용한도 분포",
            labels={'카드이용한도액': '신용한도 (원)', 'count': '고객 수'}
        )
        fig_hist.update_layout(
            font_size=14,
            title_font_size=18
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # 세그먼트별 평균 한도
        limit_by_segment = df.groupby('Segment')['카드이용한도액'].mean().reset_index()
        
        fig_limit = px.bar(
            limit_by_segment, 
            x='Segment', 
            y='카드이용한도액',
            title="세그먼트별 평균 신용한도",
            color='Segment',
            color_discrete_map=SEGMENT_COLORS,
            category_orders={'Segment': SEGMENT_ORDER}
        )
        fig_limit.update_traces(
            texttemplate='%{y:,.0f}원',
            textposition='outside'
        )
        fig_limit.update_layout(
            font_size=14,
            title_font_size=18,
            showlegend=False
        )
        st.plotly_chart(fig_limit, use_container_width=True)

def render_behavior_tab(df: pd.DataFrame):
    """행동마케팅 분석 탭"""
    st.markdown("## 🎯 행동마케팅 분석")
    
    # 가상의 마케팅 데이터 생성
    behavior_data = df.copy()
    behavior_data['모바일_로그인'] = np.random.choice([0, 1], len(behavior_data), p=[0.35, 0.65])
    behavior_data['앱_사용'] = np.random.choice([0, 1], len(behavior_data), p=[0.28, 0.72])
    behavior_data['포인트_적립'] = np.random.normal(5000, 2000, len(behavior_data))
    behavior_data['포인트_적립'] = np.maximum(0, behavior_data['포인트_적립'])
    behavior_data['혜택수혜율'] = np.random.normal(15, 5, len(behavior_data))
    behavior_data['혜택수혜율'] = np.maximum(0, behavior_data['혜택수혜율'])
    
    # 마케팅 KPI 메트릭
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mobile_users = behavior_data['모바일_로그인'].sum()
        st.metric("모바일 이용자", f"{mobile_users:,}명")
    
    with col2:
        total_points = behavior_data['포인트_적립'].sum()
        st.metric("총 포인트 적립", f"{format_number(total_points, 'P')}")
    
    with col3:
        avg_benefit = behavior_data['혜택수혜율'].mean()
        st.metric("평균 혜택수혜율", f"{avg_benefit:.2f}%")
    
    with col4:
        app_users = behavior_data['앱_사용'].sum()
        app_rate = (app_users / len(behavior_data)) * 100
        st.metric("앱 이용률", f"{app_rate:.1f}%")
    
    # 세그먼트별 마케팅 지표
    st.markdown("### 📱 세그먼트별 마케팅 지표")
    
    # 세그먼트별 마케팅 지표
    marketing_metrics = behavior_data.groupby('Segment').agg({
        '모바일_로그인': 'mean',
        '앱_사용': 'mean',
        '포인트_적립': 'sum',
        '혜택수혜율': 'mean',
        'ID': 'nunique'
    }).reset_index()
    
    marketing_metrics.columns = ['Segment', '모바일_로그인율', '앱_사용율', '포인트_적립', '혜택수혜율', '고객수']
    marketing_metrics['모바일_로그인율'] = marketing_metrics['모바일_로그인율'] * 100
    marketing_metrics['앱_사용율'] = marketing_metrics['앱_사용율'] * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 모바일 로그인율
        fig_mobile = px.bar(
            marketing_metrics, 
            x='Segment', 
            y='모바일_로그인율',
            title="세그먼트별 모바일 로그인율",
            color='Segment',
            color_discrete_map=SEGMENT_COLORS,
            category_orders={'Segment': SEGMENT_ORDER}
        )
        fig_mobile.update_traces(
            texttemplate='%{y:.1f}%',
            textposition='outside'
        )
        fig_mobile.update_layout(
            font_size=14,
            title_font_size=18,
            showlegend=False
        )
        st.plotly_chart(fig_mobile, use_container_width=True)
    
    with col2:
        # 포인트 적립
        fig_points = px.bar(
            marketing_metrics, 
            x='Segment', 
            y='포인트_적립',
            title="세그먼트별 포인트 적립",
            color='Segment',
            color_discrete_map=SEGMENT_COLORS,
            category_orders={'Segment': SEGMENT_ORDER}
        )
        fig_points.update_traces(
            texttemplate='%{y:,.0f}P',
            textposition='outside'
        )
        fig_points.update_layout(
            font_size=14,
            title_font_size=18,
            showlegend=False
        )
        st.plotly_chart(fig_points, use_container_width=True)
    
    # 마케팅 캠페인 분석
    st.markdown("### 🎪 마케팅 캠페인 분석")
    
    # 캠페인 타입별 성과 (가상 데이터)
    campaign_types = ['할인쿠폰', '적립이벤트', '무이자할부', '포인트배수', '신규가입혜택']
    campaign_data = []
    
    for segment in SEGMENT_ORDER:
        segment_df = behavior_data[behavior_data['Segment'] == segment]
        if not segment_df.empty:
            for campaign in campaign_types:
                # 가상의 캠페인 성과 데이터
                participation = np.random.normal(0.25, 0.1, len(segment_df))
                participation = np.maximum(0, np.minimum(1, participation))
                
                conversion = np.random.normal(0.15, 0.05, len(segment_df))
                conversion = np.maximum(0, np.minimum(1, conversion))
                
                campaign_data.append({
                    'Segment': segment,
                    'Campaign': campaign,
                    'Participation_Rate': participation.mean() * 100,
                    'Conversion_Rate': conversion.mean() * 100,
                    'Customers': len(segment_df)
                })
    
    campaign_df = pd.DataFrame(campaign_data)
    
    # 캠페인별 참여율
    campaign_summary = campaign_df.groupby('Campaign').agg({
        'Participation_Rate': 'mean',
        'Conversion_Rate': 'mean'
    }).reset_index()
    
    fig_campaign = px.bar(
        campaign_summary, 
        x='Campaign', 
        y='Participation_Rate',
        title="캠페인별 평균 참여율",
        color='Participation_Rate',
        color_continuous_scale='Blues'
    )
    fig_campaign.update_traces(
        texttemplate='%{y:.1f}%',
        textposition='outside'
    )
    fig_campaign.update_layout(
        font_size=14,
        title_font_size=18
    )
    st.plotly_chart(fig_campaign, use_container_width=True)

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
    
    # 사이드바에 데이터 요약 정보 표시
    with st.sidebar:
        st.markdown("---")
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
    
    # 메인 탭
    tabs = st.tabs(["세그먼트별 비교분석", "리스크 분석", "행동마케팅 분석"])
    
    with tabs[0]:
        render_compare_tab(filtered_df)
    
    with tabs[1]:
        render_risk_tab(filtered_df)
    
    with tabs[2]:
        render_behavior_tab(filtered_df)


if __name__ == "__main__":
    main()
