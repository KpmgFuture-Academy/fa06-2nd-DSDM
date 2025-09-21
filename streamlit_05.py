import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="신용카드 세그먼트 분석 대시보드",
    page_icon="💳",
    layout="wide"
)

# CSS 스타일 추가
st.markdown("""
<style>
    /* 사이드바 탭 스타일 */
    .sidebar .stButton > button {
        background-color: transparent;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        margin: 2px 0;
        width: 100%;
        text-align: left;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: none;
    }
    
    .sidebar .stButton > button:hover {
        background-color: #f0f2f6;
        border-left: 3px solid #1f77b4;
        transform: translateX(2px);
    }
    
    .sidebar .stButton > button:active {
        background-color: #e6f3ff;
        border-left: 3px solid #1f77b4;
    }
    
    /* 활성 탭 스타일 */
    .sidebar .stMarkdown {
        margin: 4px 0;
    }
    
    .sidebar .stMarkdown strong {
        background-color: #e6f3ff;
        padding: 8px 16px;
        border-radius: 8px;
        border-left: 3px solid #1f77b4;
        display: block;
        margin: 2px 0;
    }
    
    /* 섹션 제목 스타일 */
    .sidebar h3 {
        color: #1f77b4;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    /* 메인 컨텐츠 상단 탭 컴팩트 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 6px 10px;
        font-size: 13px;
        min-width: auto;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        height: 2px;
    }
</style>
""", unsafe_allow_html=True)

# 데이터 로드 함수
@st.cache_data
def load_data():
    try:
        # 필요한 컬럼만 로드하여 메모리 사용량 감소
        required_columns = [
            'Segment', 'ID', '연령', '남녀구분코드', '거주시도명',
            '유효카드수_신용', '유효카드수_체크', '카드이용한도금액', '입회경과개월수_신용',
            
            # 이용 관련 컬럼 (Usage)
            '이용금액_일시불_B0M', '이용금액_할부_B0M', 
            '이용건수_신용_B0M', '이용건수_신판_B0M', '이용건수_일시불_B0M', '이용건수_할부_B0M',
            '이용건수_CA_B0M', '이용건수_체크_B0M',
            '잔액_일시불_B0M', '잔액_할부_B0M', '잔액_현금서비스_B0M', '잔액_카드론_B0M',
            
            # 수익/혜택 관련 컬럼 (Profit & Benefit)
            '청구금액_B0', '포인트_마일리지_건별_B0M', '연체잔액_B0M',
            
            # 기타 컬럼
            '_1순위업종', '_1순위업종_이용금액', '_2순위업종', '_2순위업종_이용금액',
            '이용금액_온라인_R6M', '이용금액_오프라인_R6M', '이용건수_온라인_R6M', '이용건수_오프라인_R6M'
        ]
        
        df = pd.read_csv('base_test_merged_seg.csv', 
                        usecols=required_columns,
                        low_memory=False)
        
        # 데이터 타입 최적화
        df['Segment'] = df['Segment'].astype('category')
        df['남녀구분코드'] = df['남녀구분코드'].astype('category')
        df['거주시도명'] = df['거주시도명'].astype('category')
        
        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
        return None

# 메인 타이틀
st.title("💳 신용카드 세그먼트 분석 대시보드")

# 데이터 로드
df = load_data()

# 공통 시각화 자료 설명 컴포넌트
def show_visualization_guide():
    """모든 페이지 하단에 표시할 시각화 자료 설명"""
    with st.expander("📖 시각화 자료 설명", expanded=False):
        st.markdown("""
        **데이터 활용 및 계산 방법:**
        
        **1. KPI 카드 지표:**
        - **고객수**: `Segment` 컬럼으로 필터링한 해당 세그먼트 고객 수
        - **주요 연령대**: `연령` 컬럼에서 가장 많은 비율을 차지하는 연령대
        - **성별비**: `남녀구분코드` 컬럼에서 여성(2) 비율 계산
        - **평균가입기간**: `입회경과개월수_신용` 컬럼의 평균값을 12로 나누어 년수로 변환
        - **유효카드수**: `유효카드수_신용`, `유효카드수_체크` 컬럼의 평균값
        - **활성비율**: `이용금액_일시불_B0M > 0` 조건으로 계산
        - **지역 커버리지**: `거주시도명` 컬럼의 고유값 개수
        
        **2. 분포 차트:**
        - **연령대 분포**: `연령` 컬럼의 문자열 값별 고객수 막대그래프
        - **성별 분포**: `남녀구분코드` 컬럼으로 남성(1), 여성(2) 구분
        - **가입기간 분포**: `입회경과개월수_신용` 컬럼을 12로 나누어 년수로 변환한 히스토그램
        - **카드 보유 구성**: `유효카드수_신용`, `유효카드수_체크` 평균값
        
        **3. 인덱스 분석:**
        - **주요연령대비율**: 세그먼트 내 주요 연령대 비율 vs 전체 주요 연령대 비율
        - **지수 계산**: (세그먼트 평균 / 전체 평균 - 1) × 100
        - **과대표/과소표**: 지수 > 100이면 과대표, < 100이면 과소표
        - **전체 평균**: 실제 전체 데이터 기반 계산 (추정값 없음)
        
        **4. 안전 처리:**
        - 결측치 제거 후 계산
        - 표본수 n 표시
        - 전체 대비 지수로 왜곡 방지
        - 실제 데이터 기반 정확한 통계 계산
        
        **5. 데이터 소스:**
        - **파일**: `base_test_merged_seg.csv`
        - **기간**: 2018년 7월-12월
        - **총 고객 수**: 전체 테스트 데이터셋 기준
        - **세그먼트**: A, B, C, D, E (5개 세그먼트)
        """)

if df is not None:
    # 사이드바
    st.sidebar.title("📊 대시보드 메뉴")
    
    # 기본 탭 설정
    if 'current_main_tab' not in st.session_state:
        st.session_state.current_main_tab = "세그먼트 분석"
    
    # 메인 탭 선택
    main_tab = st.sidebar.selectbox(
        "메인 분석",
        ["세그먼트 분석", "리스크 분석", "행동·마케팅"],
        key="main_tab_selector"
    )
    
    if main_tab != st.session_state.current_main_tab:
        st.session_state.current_main_tab = main_tab
        # 메인 탭이 변경되면 첫 번째 세부탭으로 리셋
        if main_tab == "세그먼트 분석":
            st.session_state.current_sub_tab = "주요 KPI 비교"
        elif main_tab == "리스크 분석":
            st.session_state.current_sub_tab = "승인거절 분석"
        elif main_tab == "행동·마케팅":
            st.session_state.current_sub_tab = "앱/웹 이용행태"
    
    # 기본 세부탭 설정
    if 'current_sub_tab' not in st.session_state:
        st.session_state.current_sub_tab = "주요 KPI 비교"
    
    current_main_tab = st.session_state.current_main_tab
    
    # 세그먼트별 고객 수 계산
    segment_counts = df['Segment'].value_counts().sort_index()
    total_customers = len(df)
    
    # 메인 컨텐츠 영역
    # st.markdown(f"## {current_main_tab}")  # 제목 제거
    
    # 세부탭 생성
    if current_main_tab == "세그먼트 분석":
        sub_tabs = st.tabs(["주요 KPI 비교", "세그먼트별 분석", "트렌드 분석"])
        
        with sub_tabs[0]:  # 주요 KPI 비교
            st.markdown("### 📊 주요 KPI 비교 분석")
            
            # 상단 필터 바
            st.markdown("#### 🔍 분석 설정")
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                st.markdown("**세그먼트**")
                selected_segments = st.multiselect("세그먼트 선택", ['A', 'B', 'C', 'D', 'E'], 
                                                 default=['A', 'B', 'C', 'D', 'E'], key="kpi_analysis_segments")
            
            with filter_col2:
                st.markdown("**지역**")
                regions = df['거주시도명'].unique().tolist()
                selected_regions = st.multiselect("지역 선택", regions, 
                                                default=regions[:5] if len(regions) > 5 else regions, 
                                                key="kpi_analysis_regions")
            
            with filter_col3:
                st.markdown("**통계 유형**")
                stat_type = st.selectbox("통계 유형 선택", 
                                       ["합계", "평균", "중앙값"], 
                                       key="kpi_stat_type")
            
            # 금액 단위는 천원으로 고정
            amount_unit = "천원"
            
            # 데이터 필터링
            filtered_df = df[df['Segment'].isin(selected_segments)]
            if selected_regions:
                filtered_df = filtered_df[filtered_df['거주시도명'].isin(selected_regions)]
            
            # 헤더 고정 요약 (필터와 무관, 항상 전체 데이터 기준)
            st.markdown("#### 🏢 전사 기준 고객 분포 (필터 무관)")
            
            # 전체 데이터 기준 고객 분포 계산 (필터 적용 전)
            total_customers_all = len(df)
            segment_counts_all = df['Segment'].value_counts().to_dict()
            
            # 차트용 데이터 준비 (A, B, C, D, E 순서 보장)
            segments = ['A', 'B', 'C', 'D', 'E']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            
            df_segments_all = pd.DataFrame({
                '세그먼트': segments,
                '고객수': [segment_counts_all.get(seg, 0) for seg in segments],
                '비율': [(segment_counts_all.get(seg, 0) / total_customers_all * 100) for seg in segments]
            })
            
            # 세그먼트를 카테고리 타입으로 변환하여 순서 보장
            df_segments_all['세그먼트'] = pd.Categorical(df_segments_all['세그먼트'], categories=['A', 'B', 'C', 'D', 'E'], ordered=True)
            df_segments_all = df_segments_all.sort_values('세그먼트')
            
            # 상위 2개 세그먼트 찾기
            top2_segments = df_segments_all.nlargest(2, '고객수')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 파이 차트 (상위 2개에 값 표기)
                fig_pie = px.pie(
                    df_segments_all, 
                    values='고객수', 
                    names='세그먼트',
                    title="A~E 고객 분포 (전사 기준)",
                    color='세그먼트',
                    color_discrete_map={
                        'A': colors[0], 'B': colors[1], 'C': colors[2], 
                        'D': colors[3], 'E': colors[4]
                    }
                )
                
                # 상위 2개 세그먼트에만 값 표시
                pie_text = []
                for i, row in df_segments_all.iterrows():
                    if row['세그먼트'] in top2_segments['세그먼트'].values:
                        pie_text.append(f"{row['세그먼트']}<br>{row['비율']:.1f}%")
                    else:
                        pie_text.append(row['세그먼트'])
                
                fig_pie.update_traces(
                    textposition='inside',
                    text=pie_text,
                    textfont_size=14,
                    hovertemplate='<b>%{label}</b><br>고객수: %{value:,}<br>비율: %{percent}<extra></extra>'
                )
                fig_pie.update_layout(
                    font_size=14,
                    title_font_size=16
                )
                st.plotly_chart(fig_pie, width='stretch')
            
            with col2:
                # 바 차트 (로그 옵션 토글)
                use_log_scale = st.checkbox("Y축 로그 스케일 사용", key="log_scale_toggle")
                
                fig_bar = px.bar(
                    df_segments_all,
                    x='세그먼트',
                    y='고객수',
                    title="A~E 고객 수 (전사 기준)",
                    color='세그먼트',
                    color_discrete_map={
                        'A': colors[0], 'B': colors[1], 'C': colors[2], 
                        'D': colors[3], 'E': colors[4]
                    }
                )
                
                # 고객수를 천 단위로 표시
                fig_bar.update_traces(
                    texttemplate='%{customdata}천명<br>(%{text})',
                    customdata=df_segments_all['고객수'].apply(lambda x: f"{x/1000:.0f}"),
                    text=df_segments_all['비율'].apply(lambda x: f"{x:.1f}%"),
                    textposition='outside',
                    textfont_size=12
                )
                
                # 로그 스케일 옵션
                if use_log_scale:
                    fig_bar.update_layout(
                        yaxis_type="log",
                        yaxis_title="고객 수 (로그 스케일)",
                        xaxis_title="세그먼트"
                    )
                else:
                    fig_bar.update_layout(
                        yaxis_title="고객 수",
                        xaxis_title="세그먼트"
                    )
                
                fig_bar.update_layout(
                    font_size=14,
                    title_font_size=16
                )
                st.plotly_chart(fig_bar, width='stretch')
            
            st.markdown("---")
            
            # KPI 계산 함수 (실제 존재하는 컬럼명으로 수정)
            @st.cache_data
            def calculate_kpi_metrics(df_filtered, stat_type, amount_unit):
                # 세그먼트별 집계 (실제 존재하는 컬럼명 사용)
                g = df_filtered.groupby("Segment", as_index=False).agg(
                    고객수=("ID", "nunique"),
                    일시불=("이용금액_일시불_B0M", "sum"),
                    할부=("이용금액_할부_B0M", "sum"),
                    현금서비스=("잔액_현금서비스_B0M", "sum"),
                    카드론=("잔액_카드론_B0M", "sum"),
                )
                
                # 총이용금액 계산 (실제 존재하는 컬럼들만 사용)
                g["총이용금액"] = g["일시불"] + g["할부"] + g["현금서비스"] + g["카드론"]
                
                # 총이용건수 계산 (실제 존재하는 컬럼들만 사용)
                total_count_by_segment = df_filtered.groupby("Segment").agg({
                    "이용건수_일시불_B0M": "sum",
                    "이용건수_할부_B0M": "sum", 
                    "이용건수_신용_B0M": "sum",
                    "이용건수_체크_B0M": "sum"
                })
                g["총이용건수"] = total_count_by_segment.sum(axis=1).values
                
                # ARPU 계산 (청구금액 기준)
                billing_by_segment = df_filtered.groupby("Segment")["청구금액_B0"].sum()
                g["ARPU"] = billing_by_segment.values / g["고객수"]
                
                # 객단가 계산
                g["객단가"] = g["총이용금액"] / g["총이용건수"]
                g["객단가"] = g["객단가"].fillna(0)  # 0으로 나누기 방지
                
                # 비중 계산
                g["일시불비중"] = g["일시불"] / g["총이용금액"]
                g["할부비중"] = g["할부"] / g["총이용금액"]
                g["일시불비중"] = g["일시불비중"].fillna(0)
                g["할부비중"] = g["할부비중"].fillna(0)
                
                # 금액 단위 변환 (큰 값은 백만원, 작은 값은 천원)
                g["총이용금액"] = g["총이용금액"] / 1000000  # 백만원 단위
                g["일시불"] = g["일시불"] / 1000000  # 백만원 단위
                g["할부"] = g["할부"] / 1000000  # 백만원 단위
                g["현금서비스"] = g["현금서비스"] / 1000000  # 백만원 단위
                g["카드론"] = g["카드론"] / 1000000  # 백만원 단위
                
                # 작은 금액들 (ARPU, 객단가)은 천원 단위
                g["ARPU"] = g["ARPU"] / 1000  # 천원 단위
                g["객단가"] = g["객단가"] / 1000  # 천원 단위
                
                # 컬럼명 정리
                g = g.rename(columns={
                    "Segment": "세그먼트",
                    "일시불": "일시불금액",
                    "할부": "할부금액",
                    "현금서비스": "현금서비스금액",
                    "카드론": "카드론금액"
                })
                
                # 세그먼트 순서 보장 (A, B, C, D, E)
                g['세그먼트'] = pd.Categorical(g['세그먼트'], categories=['A', 'B', 'C', 'D', 'E'], ordered=True)
                g = g.sort_values('세그먼트')
                
                return g
            
            # KPI 메트릭 계산
            if not filtered_df.empty:
                kpi_df = calculate_kpi_metrics(filtered_df, stat_type, amount_unit)
                
                # 상단 KPI 카드 4개
                st.markdown("#### 📈 주요 KPI 지표")
                col1, col2, col3, col4 = st.columns(4)
                
                total_amount = kpi_df['총이용금액'].sum()
                total_count = kpi_df['총이용건수'].sum()
                avg_arpu = kpi_df['ARPU'].mean()
                avg_transaction = kpi_df['객단가'].mean()
                
                with col1:
                    st.metric(
                        label="총 이용금액 (백만원)",
                        value=f"{total_amount:,.1f}"
                    )
                
                with col2:
                    st.metric(
                        label="총 이용건수",
                        value=f"{total_count:,.0f}"
                    )
                
                with col3:
                    st.metric(
                        label="평균 ARPU (천원)",
                        value=f"{avg_arpu:,.0f}"
                    )
                
                with col4:
                    st.metric(
                        label="평균 객단가 (천원)",
                        value=f"{avg_transaction:,.0f}"
                    )
                
                st.markdown("---")
                
                # 이용 금액 비교 (세그먼트 상대 비교)
                st.markdown("#### 💰 이용 금액 비교 (세그먼트 상대 비교)")
                
                # 보조 지표: 객단가 배지 표시
                st.markdown("##### 📊 보조 지표: 객단가 (소액다건 vs 고액소건 감)")
                col1, col2, col3, col4, col5 = st.columns(5)
                for i, segment in enumerate(['A', 'B', 'C', 'D', 'E']):
                    with [col1, col2, col3, col4, col5][i]:
                        segment_data = kpi_df[kpi_df['세그먼트'] == segment]
                        if not segment_data.empty:
                            avg_transaction = segment_data['객단가'].iloc[0]
                            st.metric(
                                label=f"세그먼트 {segment}",
                                value=f"{avg_transaction:,.0f}천원",
                                help="총이용금액/총이용건수"
                            )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 세그먼트별 총 이용금액 막대그래프
                    # 금액 단위 자동 환산 (억/만원)
                    kpi_df['총이용금액_억원'] = kpi_df['총이용금액'] / 100  # 백만원 -> 억원
                    
                    fig_amount = px.bar(
                        kpi_df, 
                        x='세그먼트', 
                        y='총이용금액_억원',
                        title="세그먼트별 총 이용금액 (억원)",
                        color='세그먼트',
                        color_discrete_map={
                            'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 
                            'D': '#96CEB4', 'E': '#FFEAA7'
                        }
                    )
                    
                    # 인사이트 포인트를 위한 텍스트 생성
                    insight_text = []
                    for _, row in kpi_df.iterrows():
                        customer_count = row['고객수']
                        total_amount = row['총이용금액_억원']
                        
                        if customer_count > 1000 and total_amount < 100:  # 고객수 큰데 금액 낮음
                            insight_text.append(f"저활성 대규모<br>(고객 {customer_count:,}명)")
                        elif customer_count < 500 and total_amount > 50:  # 고객수 적은데 금액 높음
                            insight_text.append(f"VIP 소수정예<br>(고객 {customer_count:,}명)")
                        else:
                            insight_text.append(f"{total_amount:.1f}억원<br>(고객 {customer_count:,}명)")
                    
                    fig_amount.update_traces(
                        texttemplate='%{text}',
                        text=insight_text,
                        textposition='inside',
                        textfont_size=10
                    )
                    
                    fig_amount.update_layout(
                        height=400, 
                        width=500,
                        font_size=14,
                        title_font_size=16,
                        yaxis_title="총 이용금액 (억원)",
                        xaxis_title="세그먼트"
                    )
                    st.plotly_chart(fig_amount)
                
                with col2:
                    # 세그먼트별 1인당 이용금액(ARPU) 막대그래프
                    fig_arpu = px.bar(
                        kpi_df, 
                        x='세그먼트', 
                        y='ARPU',
                        title="세그먼트별 1인당 이용금액 (ARPU)",
                        color='세그먼트',
                        color_discrete_map={
                            'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 
                            'D': '#96CEB4', 'E': '#FFEAA7'
                        }
                    )
                    
                    # ARPU 인사이트 포인트를 위한 텍스트 생성
                    arpu_insight_text = []
                    for _, row in kpi_df.iterrows():
                        arpu = row['ARPU']
                        customer_count = row['고객수']
                        
                        if arpu > 500 and customer_count > 1000:  # ARPU 높고 고객수 많음
                            arpu_insight_text.append(f"핵심 성장엔진<br>{arpu:,.0f}천원")
                        elif arpu > 500 and customer_count < 500:  # ARPU 높고 고객수 적음
                            arpu_insight_text.append(f"VIP 육성타깃<br>{arpu:,.0f}천원")
                        elif arpu < 200 and customer_count > 1000:  # ARPU 낮고 고객수 많음
                            arpu_insight_text.append(f"대중형 프로모션<br>{arpu:,.0f}천원")
                        else:
                            arpu_insight_text.append(f"{arpu:,.0f}천원<br>(고객 {customer_count:,}명)")
                    
                    fig_arpu.update_traces(
                        texttemplate='%{text}',
                        text=arpu_insight_text,
                        textposition='inside',
                        textfont_size=10
                    )
                    
                    fig_arpu.update_layout(
                        height=400, 
                        width=500,
                        font_size=14,
                        title_font_size=16,
                        yaxis_title="ARPU (천원)",
                        xaxis_title="세그먼트"
                    )
                    st.plotly_chart(fig_arpu)
                
                # 인사이트 포인트 설명
                st.markdown("##### 💡 인사이트 포인트")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **총 이용금액 분석:**
                    - 🟢 **저활성 대규모 군**: 고객수 많은데 금액 낮음 → 리텐션/활성화 과제
                    - 🔴 **VIP 소수 정예 군**: 고객수 적은데 금액 높음 → 프리미엄 베네핏/고객관리
                    """)
                
                with col2:
                    st.markdown("""
                    **ARPU 분석:**
                    - 🚀 **핵심 성장 엔진**: ARPU↑ + 고객수↑
                    - 👑 **VIP 육성 타깃**: ARPU↑ + 고객수↓  
                    - 📈 **대중형 프로모션**: ARPU↓ + 고객수↑ → 빈도 강화
                    """)
                
                # 신판 Stacked Bar와 ARPU vs 객단가 Scatter를 한 줄에 배치
                st.markdown("#### 💳 신판 이용 패턴 & ARPU vs 객단가 분석")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_stacked = px.bar(
                        kpi_df, 
                        x='세그먼트', 
                        y=['일시불금액', '할부금액'],
                        title="세그먼트별 신판 이용금액 (백만원)",
                        color_discrete_map={
                            '일시불금액': '#FF6B6B', '할부금액': '#4ECDC4'
                        }
                    )
                    fig_stacked.update_traces(
                        texttemplate='%{y:,.1f}',
                        textposition='inside',
                        textfont_size=10
                    )
                    fig_stacked.update_layout(
                        height=400, 
                        width=500,
                        font_size=14,
                        title_font_size=16,
                        yaxis_title="이용금액 (백만원)",
                        xaxis_title="세그먼트"
                    )
                    st.plotly_chart(fig_stacked)
                
                with col2:
                    fig_scatter = px.scatter(
                        kpi_df, 
                        x='ARPU', 
                        y='객단가',
                        size='고객수',
                        color='세그먼트',
                        title="ARPU vs 객단가 산점도 (버블 크기: 고객수)",
                        color_discrete_map={
                            'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 
                            'D': '#96CEB4', 'E': '#FFEAA7'
                        },
                        hover_data=['고객수']
                    )
                    fig_scatter.update_traces(
                        text=kpi_df['세그먼트'],
                        textposition="top center",
                        textfont_size=14
                    )
                    fig_scatter.update_layout(
                        height=400, 
                        width=500,
                        font_size=14,
                        title_font_size=16,
                        xaxis_title="ARPU (천원)",
                        yaxis_title="객단가 (천원)"
                    )
                    st.plotly_chart(fig_scatter)
                
                # CA/카드론 Box Plot (또는 Violin)
                st.markdown("#### 📦 현금서비스/카드론 분포 분석")
                col1, col2 = st.columns(2)
                
                with col1:
                    # 현금서비스 Box Plot (순서 보장)
                    box_data = filtered_df[filtered_df['Segment'].isin(selected_segments)].copy()
                    box_data['Segment'] = pd.Categorical(box_data['Segment'], categories=['A', 'B', 'C', 'D', 'E'], ordered=True)
                    box_data = box_data.sort_values('Segment')
                    
                    fig_cash_box = px.box(
                        box_data,
                        x='Segment',
                        y='잔액_현금서비스_B0M' if '잔액_현금서비스_B0M' in filtered_df.columns else '이용금액_일시불_B0M',
                        title=f"세그먼트별 현금서비스 분포 ({amount_unit})",
                        color='Segment',
                        color_discrete_map={
                            'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 
                            'D': '#96CEB4', 'E': '#FFEAA7'
                        }
                    )
                    fig_cash_box.update_layout(
                        height=400, 
                        width=500,
                        font_size=14,
                        title_font_size=16,
                        yaxis_title="현금서비스 (백만원)",
                        xaxis_title="세그먼트"
                    )
                    st.plotly_chart(fig_cash_box)
                
                with col2:
                    # 카드론 Box Plot (순서 보장)
                    fig_loan_box = px.box(
                        box_data,
                        x='Segment',
                        y='잔액_카드론_B0M' if '잔액_카드론_B0M' in filtered_df.columns else '이용금액_할부_B0M',
                        title=f"세그먼트별 카드론 분포 ({amount_unit})",
                        color='Segment',
                        color_discrete_map={
                            'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 
                            'D': '#96CEB4', 'E': '#FFEAA7'
                        }
                    )
                    fig_loan_box.update_layout(
                        height=400, 
                        width=500,
                        font_size=14,
                        title_font_size=16,
                        yaxis_title="카드론 (백만원)",
                        xaxis_title="세그먼트"
                    )
                    st.plotly_chart(fig_loan_box)
                
                # 세그먼트×지역 분석 (개선된 시각화)
                st.markdown("#### 🗺️ 세그먼트×지역 분석")
                
                # 시각화 타입 선택
                viz_type = st.selectbox(
                    "분석 방법 선택:",
                    ["지역별 세그먼트 분포 (스택바)", "세그먼트별 지역 집중도 (도넛차트)", "지역별 ARPU 비교 (버블차트)"],
                    key="region_segment_viz"
                )
                
                if viz_type == "지역별 세그먼트 분포 (스택바)":
                    # 지역별 세그먼트 분포 스택바 차트
                    region_segment_data = filtered_df.groupby(['거주시도명', 'Segment']).size().reset_index(name='고객수')
                    region_segment_data['Segment'] = pd.Categorical(region_segment_data['Segment'], categories=['A', 'B', 'C', 'D', 'E'], ordered=True)
                    
                    fig_stack = px.bar(
                        region_segment_data,
                        x='거주시도명',
                        y='고객수',
                        color='Segment',
                        title="지역별 세그먼트 분포 (스택바)",
                        color_discrete_map={
                            'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 
                            'D': '#96CEB4', 'E': '#FFEAA7'
                        },
                        text_auto=True
                    )
                    fig_stack.update_traces(
                        texttemplate='%{y:,}',
                        textposition='inside',
                        textfont_size=10
                    )
                    fig_stack.update_layout(
                        height=500,
                        width=900,
                        font_size=14,
                        title_font_size=16,
                        xaxis_title="지역",
                        yaxis_title="고객수",
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig_stack)
                
                elif viz_type == "세그먼트별 지역 집중도 (도넛차트)":
                    # 세그먼트별로 도넛차트 생성
                    segments = ['A', 'B', 'C', 'D', 'E']
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    for i, segment in enumerate(segments):
                        segment_data = filtered_df[filtered_df['Segment'] == segment]
                        if not segment_data.empty:
                            region_counts = segment_data['거주시도명'].value_counts()
                            
                            # 상위 5개 지역만 표시, 나머지는 '기타'로 합침
                            if len(region_counts) > 5:
                                top_regions = region_counts.head(5)
                                others_count = region_counts.tail(-5).sum()
                                region_counts = pd.concat([top_regions, pd.Series({'기타': others_count})])
                            
                            with [col1, col2, col3][i % 3]:
                                fig_donut = px.pie(
                                    values=region_counts.values,
                                    names=region_counts.index,
                                    title=f"세그먼트 {segment} 지역 분포",
                                    hole=0.4,
                                    color_discrete_sequence=px.colors.qualitative.Set3[:len(region_counts)]
                                )
                                fig_donut.update_traces(
                                    textposition='inside',
                                    textinfo='percent+label',
                                    textfont_size=10
                                )
                                fig_donut.update_layout(
                                    font_size=12,
                                    title_font_size=14,
                                    showlegend=False
                                )
                                st.plotly_chart(fig_donut, use_container_width=True)
                
                else:  # 지역별 ARPU 비교 (버블차트)
                    # 지역별 ARPU 계산
                    region_arpu = filtered_df.groupby('거주시도명').agg({
                        '청구금액_B0': 'sum',
                        'ID': 'nunique'
                    }).reset_index()
                    region_arpu['ARPU'] = region_arpu['청구금액_B0'] / region_arpu['ID']
                    region_arpu['ARPU_천원'] = region_arpu['ARPU'] / 1000
                    
                    # 세그먼트 비율 계산 (각 지역에서 가장 많은 세그먼트)
                    region_dominant_segment = filtered_df.groupby('거주시도명')['Segment'].apply(
                        lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else 'N/A'
                    ).reset_index()
                    region_dominant_segment.columns = ['거주시도명', '주요세그먼트']
                    
                    # 데이터 병합
                    bubble_data = region_arpu.merge(region_dominant_segment, on='거주시도명')
                    
                    fig_bubble = px.scatter(
                        bubble_data,
                        x='ID',
                        y='ARPU_천원',
                        size='ID',
                        color='주요세그먼트',
                        hover_name='거주시도명',
                        hover_data={'ID': ':,.0f', 'ARPU_천원': ':,.1f'},
                        title="지역별 ARPU vs 고객수 (버블 크기: 고객수)",
                        color_discrete_map={
                            'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 
                            'D': '#96CEB4', 'E': '#FFEAA7'
                        }
                    )
                    
                    # 지역명을 버블 위에 표시
                    fig_bubble.update_traces(
                        text=bubble_data['거주시도명'],
                        textposition="top center",
                        textfont_size=11
                    )
                    
                    fig_bubble.update_layout(
                        height=500,
                        width=900,
                        font_size=14,
                        title_font_size=16,
                        xaxis_title="고객수",
                        yaxis_title="ARPU (천원)",
                        showlegend=True
                    )
                    st.plotly_chart(fig_bubble)
                
                # 추가 인사이트
                st.markdown("##### 💡 지역별 세그먼트 인사이트")
                col1, col2 = st.columns(2)
                
                with col1:
                    # 지역별 주요 세그먼트
                    region_insights = filtered_df.groupby('거주시도명')['Segment'].apply(
                        lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else 'N/A'
                    ).value_counts()
                    
                    st.markdown("**지역별 주요 세그먼트 분포:**")
                    for segment, count in region_insights.items():
                        st.write(f"• 세그먼트 {segment}: {count}개 지역")
                
                with col2:
                    # 지역별 고객 밀도
                    region_density = filtered_df['거주시도명'].value_counts().head(5)
                    st.markdown("**고객 밀도 상위 지역:**")
                    for region, count in region_density.items():
                        st.write(f"• {region}: {count:,}명")
            
            else:
                st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
            
            # 시각화 자료 설명
            show_visualization_guide()
        
        with sub_tabs[1]:  # 세그먼트별 분석
            st.markdown("### 📊 세그먼트별 상세 분석")
            st.info("세그먼트별 상세 분석 페이지입니다. (추후 구현 예정)")
            
            # 세그먼트별 상세 정보
            st.markdown("### 📋 세그먼트별 상세 정보")
            
            # 세그먼트 선택
            selected_segment = st.selectbox(
                "분석할 세그먼트를 선택하세요",
                segments,
                key="segment_selector"
            )
            
            # 선택된 세그먼트 데이터 필터링
            segment_data = df[df['Segment'] == selected_segment].copy()
            segment_count = len(segment_data)
            
            if segment_count > 0:
                # 1) 상단 KPI 카드 (프로필 핵심 6~8개)
                st.markdown("#### 📊 핵심 프로필 지표")
                
                # 전체 평균 계산 (비교 기준) - 연령은 문자열이므로 제외
                total_stats = {
                    'female_ratio': (df['남녀구분코드'] == 2).sum() / len(df) * 100,
                    'avg_credit_cards': df['유효카드수_신용'].mean(),
                    'avg_check_cards': df['유효카드수_체크'].mean(),
                    'active_ratio': (df['이용금액_일시불_B0M'] > 0).sum() / len(df) * 100,
                    'region_coverage': df['거주시도명'].nunique()
                }
                
                # 세그먼트별 통계 계산 (연령은 문자열이므로 연령대 분포로 처리)
                segment_stats = {
                    'age_distribution': segment_data['연령'].value_counts(),
                    'female_ratio': (segment_data['남녀구분코드'] == 2).sum() / len(segment_data) * 100,
                    'avg_credit_cards': segment_data['유효카드수_신용'].mean(),
                    'avg_check_cards': segment_data['유효카드수_체크'].mean(),
                    'active_ratio': (segment_data['이용금액_일시불_B0M'] > 0).sum() / len(segment_data) * 100,
                    'region_coverage': segment_data['거주시도명'].nunique()
                }
                
                # 가입기간 계산 (실제 데이터 사용, 년수로 변환)
                segment_stats['avg_membership_years'] = segment_data['입회경과개월수_신용'].mean() / 12
                total_stats['avg_membership_years'] = df['입회경과개월수_신용'].mean() / 12
                
                # 가입기간 분포 차트용 데이터 (년수로 변환)
                segment_data['가입기간_년'] = segment_data['입회경과개월수_신용'] / 12
                
                # KPI 카드 생성
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # 고객수
                    index_vs_total = (segment_count / total_customers - 1) * 100
                    st.metric(
                        label="고객수",
                        value=f"{segment_count:,}",
                        help=f"전체 대비 지수: {index_vs_total:+.1f}%"
                    )
                    st.caption(f"n={segment_count}")
                    
                    # 주요 연령대 (가장 많은 연령대)
                    top_age = segment_stats['age_distribution'].index[0]
                    top_age_count = segment_stats['age_distribution'].iloc[0]
                    top_age_ratio = (top_age_count / segment_count) * 100
                    st.metric(
                        label="주요 연령대",
                        value=f"{top_age}",
                        help=f"비율: {top_age_ratio:.1f}%"
                    )
                    st.caption(f"고객수: {top_age_count:,}명")
                
                with col2:
                    # 성별비 (여성%)
                    female_index = (segment_stats['female_ratio'] / total_stats['female_ratio'] - 1) * 100
                    st.metric(
                        label="성별비 (여성%)",
                        value=f"{segment_stats['female_ratio']:.1f}%",
                        help=f"전체 대비 지수: {female_index:+.1f}%"
                    )
                    st.caption(f"전체 평균: {total_stats['female_ratio']:.1f}%")
                    
                    # 평균가입기간
                    membership_index = (segment_stats['avg_membership_years'] / total_stats['avg_membership_years'] - 1) * 100
                    st.metric(
                        label="평균가입기간",
                        value=f"{segment_stats['avg_membership_years']:.1f}년",
                        help=f"전체 대비 지수: {membership_index:+.1f}%"
                    )
                    st.caption(f"전체 평균: {total_stats['avg_membership_years']:.1f}년")
                
                with col3:
                    # 유효카드수_신용
                    credit_index = (segment_stats['avg_credit_cards'] / total_stats['avg_credit_cards'] - 1) * 100
                    st.metric(
                        label="유효카드수_신용",
                        value=f"{segment_stats['avg_credit_cards']:.1f}개",
                        help=f"전체 대비 지수: {credit_index:+.1f}%"
                    )
                    st.caption(f"전체 평균: {total_stats['avg_credit_cards']:.1f}개")
                    
                    # 유효카드수_체크
                    check_index = (segment_stats['avg_check_cards'] / total_stats['avg_check_cards'] - 1) * 100
                    st.metric(
                        label="유효카드수_체크",
                        value=f"{segment_stats['avg_check_cards']:.1f}개",
                        help=f"전체 대비 지수: {check_index:+.1f}%"
                    )
                    st.caption(f"전체 평균: {total_stats['avg_check_cards']:.1f}개")
                
                with col4:
                    # 활성비율
                    active_index = (segment_stats['active_ratio'] / total_stats['active_ratio'] - 1) * 100
                    st.metric(
                        label="활성비율",
                        value=f"{segment_stats['active_ratio']:.1f}%",
                        help=f"전체 대비 지수: {active_index:+.1f}%"
                    )
                    st.caption(f"전체 평균: {total_stats['active_ratio']:.1f}%")
                    
                    # 지역 커버리지
                    region_index = (segment_stats['region_coverage'] / total_stats['region_coverage'] - 1) * 100
                    st.metric(
                        label="지역 커버리지",
                        value=f"{segment_stats['region_coverage']}개 시도",
                        help=f"전체 대비 지수: {region_index:+.1f}%"
                    )
                    st.caption(f"전체 평균: {total_stats['region_coverage']}개 시도")
                
                # 2) 중단 분포/구성 차트
                st.markdown("#### 📊 분포 및 구성 분석")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 연령대 분포 (막대그래프) - 문자열 연령 데이터 활용
                    age_dist = segment_stats['age_distribution'].sort_index()
                    
                    # 연령대별 비율 계산
                    age_percentages = (age_dist.values / age_dist.sum() * 100).round(1)
                    
                    fig_age_dist = px.bar(
                        x=age_dist.index,
                        y=age_dist.values,
                        title=f"세그먼트 {selected_segment} 연령대 분포",
                        color=age_dist.values,
                        color_continuous_scale='Blues',
                        labels={'x': '연령대', 'y': '고객수'},
                        text=[f"{count:,}명<br>({pct:.1f}%)" for count, pct in zip(age_dist.values, age_percentages)]
                    )
                    fig_age_dist.update_traces(textposition='inside', textfont_color='white')
                    fig_age_dist.update_layout(showlegend=False)
                    st.plotly_chart(fig_age_dist, width='stretch')
                    
                    # 성별비 도넛 차트
                    gender_counts = segment_data['남녀구분코드'].value_counts()
                    gender_labels = ['남성' if x == 1 else '여성' for x in gender_counts.index]
                    
                    fig_gender = px.pie(
                        values=gender_counts.values,
                        names=gender_labels,
                        title=f"세그먼트 {selected_segment} 성별 분포",
                        hole=0.4
                    )
                    fig_gender.update_traces(
                        textinfo='label+percent+value',
                        texttemplate='%{label}<br>%{value:,}명<br>(%{percent})'
                    )
                    st.plotly_chart(fig_gender, width='stretch')
                
                with col2:
                    # 가입기간 분포 히스토그램 (년수로 표시)
                    fig_membership = px.histogram(
                        segment_data,
                        x='가입기간_년',
                        title=f"세그먼트 {selected_segment} 가입기간 분포",
                        nbins=20,
                        color_discrete_sequence=['#4ECDC4'],
                        text_auto=True
                    )
                    fig_membership.update_traces(
                        texttemplate='%{y:,}명',
                        textposition='inside',
                        textfont_color='white'
                    )
                    st.plotly_chart(fig_membership, width='stretch')
                    
                    # 카드 보유 구성
                    card_data = {
                        '카드 유형': ['신용카드', '체크카드'],
                        '평균 카드수': [segment_stats['avg_credit_cards'], segment_stats['avg_check_cards']]
                    }
                    
                    # 카드 보유 비율 계산 (세그먼트 내에서의 비율)
                    total_cards = segment_stats['avg_credit_cards'] + segment_stats['avg_check_cards']
                    credit_ratio = (segment_stats['avg_credit_cards'] / total_cards * 100).round(1) if total_cards > 0 else 0
                    check_ratio = (segment_stats['avg_check_cards'] / total_cards * 100).round(1) if total_cards > 0 else 0
                    
                    fig_cards = px.bar(
                        x=card_data['카드 유형'],
                        y=card_data['평균 카드수'],
                        title=f"세그먼트 {selected_segment} 카드 보유 구성",
                        color=card_data['평균 카드수'],
                        color_continuous_scale='Greens',
                        text=[f"{count:.1f}개<br>({ratio:.1f}%)" for count, ratio in zip(card_data['평균 카드수'], [credit_ratio, check_ratio])]
                    )
                    fig_cards.update_traces(textposition='inside', textfont_color='white')
                    st.plotly_chart(fig_cards, width='stretch')
                
                # 3) 하단 하이라이트
                st.markdown("#### 🎯 세그먼트 인덱스 분석")
                
                # 세그먼트 인덱스 표 (전체=100 기준) - 연령대 분포로 변경
                # 주요 연령대 비율 계산
                top_age_ratio = (segment_stats['age_distribution'].iloc[0] / segment_count) * 100
                
                # 전체 주요 연령대 비율 계산 (실제 데이터 사용)
                total_age_distribution = df['연령'].value_counts()
                total_top_age_ratio = (total_age_distribution.iloc[0] / len(df) * 100).round(1)
                
                index_data = {
                    '지표': ['주요연령대비율', '여성%', '가입기간', '신용카드수', '체크카드수'],
                    '세그먼트 값': [
                        top_age_ratio,
                        segment_stats['female_ratio'],
                        segment_stats['avg_membership_years'],
                        segment_stats['avg_credit_cards'],
                        segment_stats['avg_check_cards']
                    ],
                    '전체 평균': [
                        total_top_age_ratio,  # 실제 전체 주요 연령대 비율
                        total_stats['female_ratio'],
                        total_stats['avg_membership_years'],  # 실제 전체 평균 가입기간 (년)
                        total_stats['avg_credit_cards'],
                        total_stats['avg_check_cards']
                    ]
                }
                
                index_df = pd.DataFrame(index_data)
                index_df['지수'] = (index_df['세그먼트 값'] / index_df['전체 평균'] * 100).round(1)
                
                # 더 직관적인 표현으로 변경
                def get_index_label(index):
                    if index >= 120:
                        return f"🔴 매우 높음 ({index:.0f})"
                    elif index >= 110:
                        return f"🟠 높음 ({index:.0f})"
                    elif index >= 95:
                        return f"🟡 비슷함 ({index:.0f})"
                    elif index >= 85:
                        return f"🔵 낮음 ({index:.0f})"
                    else:
                        return f"🟣 매우 낮음 ({index:.0f})"
                
                index_df['특성 강도'] = index_df['지수'].apply(get_index_label)
                
                # 컬럼 순서 조정
                display_df = index_df[['지표', '세그먼트 값', '전체 평균', '특성 강도']].copy()
                
                # 컬럼명을 더 직관적으로 변경
                display_df.columns = ['지표', f'세그먼트 {selected_segment} 값', '전체 평균', '특성 강도']
                
                st.dataframe(display_df, width='stretch')
                
                # 추가 분석 옵션들
                st.markdown("##### 🔍 추가 분석 옵션")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    show_region_age = st.checkbox("🌍 지역/연령 교차 분석", key="region_age_analysis")
                    show_card_usage = st.checkbox("💳 카드 이용 패턴 분석", key="card_usage_analysis")
                    show_age_gender = st.checkbox("👥 연령대/성별 상세 분석", key="age_gender_analysis")
                
                with col2:
                    show_membership = st.checkbox("📅 가입기간별 세분화", key="membership_analysis")
                    show_comparison = st.checkbox("📊 다른 세그먼트와 비교", key="segment_comparison")
                    show_trend = st.checkbox("📈 주요 지표 트렌드", key="trend_analysis")
                
                # 지역/연령 교차 분석
                if show_region_age:
                    st.markdown("###### 🌍 지역별 주요 연령대")
                    
                    # 지역별 주요 연령대 계산 (안전한 처리)
                    region_age_dist = segment_data.groupby('거주시도명')['연령'].apply(
                        lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else 'N/A'
                    ).reset_index()
                    region_age_dist.columns = ['거주시도명', '주요연령대']
                    
                    # 지역별 고객수도 함께 표시
                    region_counts = segment_data['거주시도명'].value_counts().reset_index()
                    region_counts.columns = ['거주시도명', '고객수']
                    
                    region_data = region_age_dist.merge(region_counts, on='거주시도명')
                    region_data = region_data.sort_values('고객수', ascending=False)
                    
                    # 지역별 비율 계산
                    total_region_customers = region_data['고객수'].sum()
                    region_data['비율(%)'] = (region_data['고객수'] / total_region_customers * 100).round(1)
                    
                    fig_region_age = px.bar(
                        region_data,
                        x='거주시도명',
                        y='고객수',
                        color='주요연령대',
                        title=f"세그먼트 {selected_segment} 지역별 고객수 및 주요 연령대",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        text=[f"{count:,}명<br>({pct:.1f}%)" for count, pct in zip(region_data['고객수'], region_data['비율(%)'])]
                    )
                    fig_region_age.update_traces(textposition='inside', textfont_color='white')
                    fig_region_age.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_region_age, width='stretch')
                
                # 카드 이용 패턴 분석
                if show_card_usage:
                    st.markdown("###### 💳 카드 이용 패턴 분석")
                    
                    # 카드 이용 패턴 데이터 생성
                    card_pattern_data = {
                        '이용 패턴': ['신용카드 위주', '체크카드 위주', '균형 이용', '낮은 이용'],
                        '고객수': [
                            len(segment_data[(segment_data['유효카드수_신용'] > segment_data['유효카드수_체크']) & 
                                            (segment_data['유효카드수_신용'] >= 2)]),
                            len(segment_data[(segment_data['유효카드수_체크'] > segment_data['유효카드수_신용']) & 
                                            (segment_data['유효카드수_체크'] >= 2)]),
                            len(segment_data[(abs(segment_data['유효카드수_신용'] - segment_data['유효카드수_체크']) <= 1) & 
                                            (segment_data['유효카드수_신용'] >= 1) & (segment_data['유효카드수_체크'] >= 1)]),
                            len(segment_data[(segment_data['유효카드수_신용'] + segment_data['유효카드수_체크']) <= 1])
                        ]
                    }
                    
                    pattern_df = pd.DataFrame(card_pattern_data)
                    pattern_df['비율(%)'] = (pattern_df['고객수'] / pattern_df['고객수'].sum() * 100).round(1)
                    
                    fig_card_pattern = px.pie(
                        pattern_df,
                        values='고객수',
                        names='이용 패턴',
                        title=f"세그먼트 {selected_segment} 카드 이용 패턴",
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                    )
                    fig_card_pattern.update_traces(textinfo='label+percent+value')
                    st.plotly_chart(fig_card_pattern, width='stretch')
                
                # 연령대/성별 상세 분석
                if show_age_gender:
                    st.markdown("###### 👥 연령대/성별 상세 분석")
                    
                    # 연령대-성별 교차표 생성
                    age_gender_cross = pd.crosstab(segment_data['연령'], 
                                                  segment_data['남녀구분코드'].map({1: '남성', 2: '여성'}))
                    
                    fig_age_gender = px.bar(
                        age_gender_cross.reset_index(),
                        x='연령',
                        y=['남성', '여성'],
                        title=f"세그먼트 {selected_segment} 연령대별 성별 분포",
                        color_discrete_map={'남성': '#4ECDC4', '여성': '#FF6B6B'}
                    )
                    fig_age_gender.update_layout(barmode='group')
                    st.plotly_chart(fig_age_gender, width='stretch')
                
                # 가입기간별 세분화
                if show_membership:
                    st.markdown("###### 📅 가입기간별 세분화")
                    
                    # 가입기간 구간 생성
                    segment_data_copy = segment_data.copy()
                    segment_data_copy['가입기간_년'] = segment_data_copy['입회경과개월수_신용'] / 12
                    
                    def categorize_membership(years):
                        if years < 2:
                            return "신규 (2년 미만)"
                        elif years < 5:
                            return "중간 (2-5년)"
                        elif years < 10:
                            return "기존 (5-10년)"
                        else:
                            return "우수 (10년 이상)"
                    
                    segment_data_copy['가입기간_구간'] = segment_data_copy['가입기간_년'].apply(categorize_membership)
                    membership_dist = segment_data_copy['가입기간_구간'].value_counts()
                    
                    fig_membership = px.bar(
                        x=membership_dist.index,
                        y=membership_dist.values,
                        title=f"세그먼트 {selected_segment} 가입기간별 분포",
                        color=membership_dist.values,
                        color_continuous_scale='Blues',
                        text=[f"{count:,}명" for count in membership_dist.values]
                    )
                    fig_membership.update_traces(textposition='inside', textfont_color='white')
                    st.plotly_chart(fig_membership, width='stretch')
                
                # 다른 세그먼트와 비교
                if show_comparison:
                    st.markdown("###### 📊 다른 세그먼트와 비교")
                    
                    # 비교할 세그먼트 선택
                    other_segments = [s for s in segments if s != selected_segment]
                    compare_segments = st.multiselect(
                        f"세그먼트 {selected_segment}와 비교할 세그먼트 선택",
                        other_segments,
                        default=other_segments[:2] if len(other_segments) >= 2 else other_segments,
                        key="compare_segments"
                    )
                    
                    if compare_segments:
                        # 비교 데이터 생성
                        compare_data = []
                        all_compare_segments = [selected_segment] + compare_segments
                        
                        for seg in all_compare_segments:
                            seg_data = df[df['Segment'] == seg]
                            if len(seg_data) > 0:
                                # 가장 많은 연령대 계산 (안전한 처리)
                                age_counts = seg_data['연령'].value_counts()
                                most_common_age = age_counts.index[0] if len(age_counts) > 0 else 'N/A'
                                
                                compare_data.append({
                                    '세그먼트': seg,
                                    '평균연령': most_common_age,  # 가장 많은 연령대
                                    '여성비율': (seg_data['남녀구분코드'] == 2).sum() / len(seg_data) * 100,
                                    '평균가입기간': (seg_data['입회경과개월수_신용'].mean() / 12).round(1),
                                    '평균신용카드': seg_data['유효카드수_신용'].mean().round(1),
                                    '평균체크카드': seg_data['유효카드수_체크'].mean().round(1)
                                })
                        
                        compare_df = pd.DataFrame(compare_data)
                        # 세그먼트 순서 보장 (A, B, C, D, E)
                        compare_df['세그먼트'] = pd.Categorical(compare_df['세그먼트'], categories=['A', 'B', 'C', 'D', 'E'], ordered=True)
                        compare_df = compare_df.sort_values('세그먼트')
                        st.dataframe(compare_df, width='stretch')
                
                # 주요 지표 트렌드
                if show_trend:
                    st.markdown("###### 📈 주요 지표 트렌드")
                    st.info("현재 데이터는 단일 시점 데이터이므로, 시계열 트렌드 분석은 추후 월별 데이터가 추가되면 구현 예정입니다.")
                    
                    # 현재는 주요 지표 요약만 표시
                    trend_summary = {
                        '지표': ['고객수', '평균연령', '여성비율', '가입기간', '신용카드수', '체크카드수'],
                        f'세그먼트 {selected_segment}': [
                            f"{segment_count:,}명",
                            segment_stats['age_distribution'].index[0],
                            f"{segment_stats['female_ratio']:.1f}%",
                            f"{segment_stats['avg_membership_years']:.1f}년",
                            f"{segment_stats['avg_credit_cards']:.1f}개",
                            f"{segment_stats['avg_check_cards']:.1f}개"
                        ]
                    }
                    
                    trend_df = pd.DataFrame(trend_summary)
                    st.dataframe(trend_df, width='stretch')
                
                else:
                    st.warning(f"세그먼트 {selected_segment}에 대한 데이터가 없습니다.")
            
            # 시각화 자료 설명
            show_visualization_guide()
            
            # AI 인사이트 (비워둠)
            st.markdown("---")
            st.markdown("#### 🤖 AI 인사이트")
            st.info("AI 인사이트가 여기에 표시됩니다.")
        
        with sub_tabs[2]:  # 트렌드 분석
            st.markdown("### 📊 세그먼트별 트렌드 분석")
            st.info("트렌드 분석 페이지입니다. (추후 구현 예정)")
            
            # 시각화 자료 설명
            show_visualization_guide()
    
    elif current_main_tab == "리스크 분석":
        sub_tabs = st.tabs(["승인거절 분석", "연체 현황", "리볼빙/현금서비스", "한도/FDS"])
        
        with sub_tabs[0]:  # 승인거절 분석
            st.markdown("### 🚫 승인거절 분석")
            st.info("승인거절 분석 페이지입니다. (추후 구현 예정)")
            
            # 시각화 자료 설명
            show_visualization_guide()
        
        with sub_tabs[1]:  # 연체 현황
            st.markdown("### 💸 연체 현황")
            st.info("연체 현황 페이지입니다. (추후 구현 예정)")
            
            # 시각화 자료 설명
            show_visualization_guide()
        
        with sub_tabs[2]:  # 리볼빙/현금서비스
            st.markdown("### 🔄 리볼빙/현금서비스")
            st.info("리볼빙/현금서비스 페이지입니다. (추후 구현 예정)")
            
            # 시각화 자료 설명
            show_visualization_guide()
        
        with sub_tabs[3]:  # 한도/FDS
            st.markdown("### 🛡️ 한도/FDS")
            st.info("한도/FDS 페이지입니다. (추후 구현 예정)")
            
            # 시각화 자료 설명
            show_visualization_guide()
    
    elif current_main_tab == "행동·마케팅":
        sub_tabs = st.tabs(["앱/웹 이용행태", "마케팅 채널 반응", "캠페인 참여 & 쿠폰 사용", "업종(MCC) 소비 패턴", "VOC/CS"])
        
        with sub_tabs[0]:  # 앱/웹 이용행태
            st.markdown("### 📱 앱/웹 이용행태")
            st.info("앱/웹 이용행태 페이지입니다. (추후 구현 예정)")
            
            # 시각화 자료 설명
            show_visualization_guide()
        
        with sub_tabs[1]:  # 마케팅 채널 반응
            st.markdown("### 📢 마케팅 채널 반응")
            st.info("마케팅 채널 반응 페이지입니다. (추후 구현 예정)")
            
            # 시각화 자료 설명
            show_visualization_guide()
        
        with sub_tabs[2]:  # 캠페인 참여 & 쿠폰 사용
            st.markdown("### 🎁 캠페인 참여 & 쿠폰 사용")
            st.info("캠페인 참여 & 쿠폰 사용 페이지입니다. (추후 구현 예정)")
            
            # 시각화 자료 설명
            show_visualization_guide()
        
        with sub_tabs[3]:  # 업종(MCC) 소비 패턴
            st.markdown("### 🏪 업종(MCC) 소비 패턴")
            st.info("업종(MCC) 소비 패턴 페이지입니다. (추후 구현 예정)")
            
            # 시각화 자료 설명
            show_visualization_guide()
        
        with sub_tabs[4]:  # VOC/CS
            st.markdown("### 📞 VOC/CS")
            st.info("VOC/CS 페이지입니다. (추후 구현 예정)")
            
            # 시각화 자료 설명
            show_visualization_guide()
    
    # 사이드바 하단 정보 (모든 탭에서 공통)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 데이터 정보")
    st.sidebar.info(f"""
    - 데이터 기간: 2018년 7월-12월
    - 총 고객 수: {total_customers:,}명
    - 세그먼트 수: {len(segments)}개
    - 총 컬럼 수: {len(df.columns)}개
    """)
    
    # 데이터 미리보기
    st.sidebar.markdown("### 🔍 데이터 미리보기")
    if st.sidebar.checkbox("원본 데이터 보기"):
        st.sidebar.dataframe(df.head(10))

else:
    st.error("데이터를 로드할 수 없습니다. 파일 경로를 확인해주세요.")
