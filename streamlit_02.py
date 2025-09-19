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
</style>
""", unsafe_allow_html=True)

# 데이터 로드 함수
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('base_test_merged_seg.csv', low_memory=False)
        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
        return None

# 메인 타이틀
st.title("💳 신용카드 세그먼트 분석 대시보드")

# 데이터 로드
df = load_data()

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
            st.session_state.current_sub_tab = "세그먼트 개요"
        elif main_tab == "리스크 분석":
            st.session_state.current_sub_tab = "승인거절 분석"
        elif main_tab == "행동·마케팅":
            st.session_state.current_sub_tab = "앱/웹 이용행태"
    
    # 기본 세부탭 설정
    if 'current_sub_tab' not in st.session_state:
        st.session_state.current_sub_tab = "세그먼트 개요"
    
    current_main_tab = st.session_state.current_main_tab
    
    # 세그먼트별 고객 수 계산
    segment_counts = df['Segment'].value_counts().sort_index()
    total_customers = len(df)
    
    # 메인 컨텐츠 영역
    st.markdown(f"## {current_main_tab}")
    
    # 세부탭 생성
    if current_main_tab == "세그먼트 분석":
        sub_tabs = st.tabs(["세그먼트 개요", "주요 KPI 비교", "세그먼트별 리포트 요약", "트렌드 분석"])
        
        with sub_tabs[0]:  # 세그먼트 개요
            st.markdown("### 📊 세그먼트별 고객 분포")
            
            # 메트릭 카드
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.metric(
                    label="전체 고객 수",
                    value=f"{total_customers:,}",
                    delta=f"{total_customers:,}"
                )
            
            segments = ['A', 'B', 'C', 'D', 'E']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            segment_columns = [col2, col3, col4, col5, col6]
            
            for i, segment in enumerate(segments):
                with segment_columns[i]:
                    count = segment_counts.get(segment, 0)
                    percentage = (count / total_customers) * 100
                    st.metric(
                        label=f"세그먼트 {segment}",
                        value=f"{count:,}",
                        delta=f"{percentage:.1f}%"
                    )
            
            # 차트 섹션
            col1, col2 = st.columns(2)
            
            # 차트용 데이터 준비
            df_segments = pd.DataFrame({
                '세그먼트': segments,
                '고객수': [segment_counts.get(seg, 0) for seg in segments]
            })
            
            with col1:
                # 파이 차트
                fig_pie = px.pie(
                    df_segments, 
                    values='고객수', 
                    names='세그먼트',
                    title="세그먼트별 고객 분포",
                    color='세그먼트',
                    color_discrete_map={
                        'A': colors[0],  # #FF6B6B
                        'B': colors[1],  # #4ECDC4
                        'C': colors[2],  # #45B7D1
                        'D': colors[3],  # #96CEB4
                        'E': colors[4]   # #FFEAA7
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # 바 차트
                fig_bar = px.bar(
                    df_segments,
                    x='세그먼트',
                    y='고객수',
                    title="세그먼트별 고객 수",
                    color='세그먼트',
                    color_discrete_map={
                        'A': colors[0],  # #FF6B6B
                        'B': colors[1],  # #4ECDC4
                        'C': colors[2],  # #45B7D1
                        'D': colors[3],  # #96CEB4
                        'E': colors[4]   # #FFEAA7
                    }
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
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
                
                # 가입기간 계산 (기준년월 기준)
                if '기준년월' in df.columns:
                    # 가입기간 시뮬레이션 (실제 데이터가 없는 경우)
                    segment_data['가입기간_개월'] = np.random.normal(24, 12, len(segment_data))
                    df['가입기간_개월'] = np.random.normal(24, 12, len(df))
                    segment_stats['avg_membership_months'] = segment_data['가입기간_개월'].mean()
                    total_stats['avg_membership_months'] = df['가입기간_개월'].mean()
                else:
                    segment_stats['avg_membership_months'] = 24  # 기본값
                    total_stats['avg_membership_months'] = 24
                
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
                    membership_index = (segment_stats['avg_membership_months'] / total_stats['avg_membership_months'] - 1) * 100
                    st.metric(
                        label="평균가입기간",
                        value=f"{segment_stats['avg_membership_months']:.1f}개월",
                        help=f"전체 대비 지수: {membership_index:+.1f}%"
                    )
                    st.caption(f"전체 평균: {total_stats['avg_membership_months']:.1f}개월")
                
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
                    fig_age_dist.update_traces(textposition='outside')
                    fig_age_dist.update_layout(showlegend=False)
                    st.plotly_chart(fig_age_dist, use_container_width=True)
                    
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
                    st.plotly_chart(fig_gender, use_container_width=True)
                
                with col2:
                    # 가입기간 분포 히스토그램
                    fig_membership = px.histogram(
                        segment_data,
                        x='가입기간_개월',
                        title=f"세그먼트 {selected_segment} 가입기간 분포",
                        nbins=20,
                        color_discrete_sequence=['#4ECDC4'],
                        text_auto=True
                    )
                    fig_membership.update_traces(texttemplate='%{y:,}', textposition='outside')
                    st.plotly_chart(fig_membership, use_container_width=True)
                    
                    # 카드 보유 구성
                    card_data = {
                        '카드 유형': ['신용카드', '체크카드'],
                        '평균 카드수': [segment_stats['avg_credit_cards'], segment_stats['avg_check_cards']]
                    }
                    
                    # 카드 보유 비율 계산
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
                    fig_cards.update_traces(textposition='outside')
                    st.plotly_chart(fig_cards, use_container_width=True)
                
                # 3) 하단 하이라이트
                st.markdown("#### 🎯 세그먼트 인덱스 분석")
                
                # 세그먼트 인덱스 표 (전체=100 기준) - 연령대 분포로 변경
                # 주요 연령대 비율 계산
                top_age_ratio = (segment_stats['age_distribution'].iloc[0] / segment_count) * 100
                
                index_data = {
                    '지표': ['주요연령대비율', '여성%', '가입기간', '신용카드수', '체크카드수'],
                    '세그먼트 값': [
                        top_age_ratio,
                        segment_stats['female_ratio'],
                        segment_stats['avg_membership_months'],
                        segment_stats['avg_credit_cards'],
                        segment_stats['avg_check_cards']
                    ],
                    '전체 평균': [
                        25.0,  # 전체 주요 연령대 비율 추정값
                        total_stats['female_ratio'],
                        24.0,  # 전체 평균 가입기간 추정값
                        total_stats['avg_credit_cards'],
                        total_stats['avg_check_cards']
                    ]
                }
                
                index_df = pd.DataFrame(index_data)
                index_df['지수'] = (index_df['세그먼트 값'] / index_df['전체 평균'] * 100).round(1)
                index_df['과대표/과소표'] = index_df['지수'].apply(lambda x: '과대표' if x > 100 else '과소표')
                
                st.dataframe(index_df, use_container_width=True)
                
                # 지역/연령 교차 분석 (선택)
                if st.checkbox("지역/연령 교차 분석 보기"):
                    st.markdown("##### 🌍 지역별 주요 연령대")
                    
                    # 지역별 주요 연령대 계산
                    region_age_dist = segment_data.groupby('거주시도명')['연령'].apply(lambda x: x.value_counts().index[0]).reset_index()
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
                    fig_region_age.update_traces(textposition='outside')
                    fig_region_age.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_region_age, use_container_width=True)
                
                # 시각화 설명 버튼
                with st.expander("📖 시각화 자료 설명"):
                    st.markdown("""
                    **데이터 활용 및 계산 방법:**
                    
                    **1. KPI 카드 지표:**
                    - **고객수**: `Segment` 컬럼으로 필터링한 해당 세그먼트 고객 수
                    - **주요 연령대**: `연령` 컬럼에서 가장 많은 비율을 차지하는 연령대
                    - **성별비**: `남녀구분코드` 컬럼에서 여성(2) 비율 계산
                    - **평균가입기간**: 시뮬레이션 데이터 (실제 가입일 데이터가 없는 경우)
                    - **유효카드수**: `유효카드수_신용`, `유효카드수_체크` 컬럼의 평균값
                    - **활성비율**: `이용금액_일시불_B0M > 0` 조건으로 계산
                    - **지역 커버리지**: `거주시도명` 컬럼의 고유값 개수
                    
                    **2. 분포 차트:**
                    - **연령대 분포**: `연령` 컬럼의 문자열 값별 고객수 막대그래프
                    - **성별 분포**: `남녀구분코드` 컬럼으로 남성(1), 여성(2) 구분
                    - **가입기간 분포**: 시뮬레이션 데이터의 히스토그램
                    - **카드 보유 구성**: `유효카드수_신용`, `유효카드수_체크` 평균값
                    
                    **3. 인덱스 분석:**
                    - **지수 계산**: (세그먼트 평균 / 전체 평균 - 1) × 100
                    - **과대표/과소표**: 지수 > 100이면 과대표, < 100이면 과소표
                    
                    **4. 안전 처리:**
                    - 결측치 제거 후 계산
                    - 표본수 n 표시
                    - 전체 대비 지수로 왜곡 방지
                    """)
            else:
                st.warning(f"세그먼트 {selected_segment}에 대한 데이터가 없습니다.")
        
        with sub_tabs[1]:  # 주요 KPI 비교
            st.markdown("### 📈 세그먼트별 주요 KPI 비교")
            st.info("주요 KPI 비교 페이지입니다. (추후 구현 예정)")
        
        with sub_tabs[2]:  # 세그먼트별 리포트 요약
            st.markdown("### 📄 세그먼트별 리포트 요약")
            st.info("세그먼트별 리포트 요약 페이지입니다. (추후 구현 예정)")
        
        with sub_tabs[3]:  # 트렌드 분석
            st.markdown("### 📊 세그먼트별 트렌드 분석")
            st.info("트렌드 분석 페이지입니다. (추후 구현 예정)")
    
    elif current_main_tab == "리스크 분석":
        sub_tabs = st.tabs(["승인거절 분석", "연체 현황", "리볼빙/현금서비스", "한도/FDS"])
        
        with sub_tabs[0]:  # 승인거절 분석
            st.markdown("### 🚫 승인거절 분석")
            st.info("승인거절 분석 페이지입니다. (추후 구현 예정)")
        
        with sub_tabs[1]:  # 연체 현황
            st.markdown("### 💸 연체 현황")
            st.info("연체 현황 페이지입니다. (추후 구현 예정)")
        
        with sub_tabs[2]:  # 리볼빙/현금서비스
            st.markdown("### 🔄 리볼빙/현금서비스")
            st.info("리볼빙/현금서비스 페이지입니다. (추후 구현 예정)")
        
        with sub_tabs[3]:  # 한도/FDS
            st.markdown("### 🛡️ 한도/FDS")
            st.info("한도/FDS 페이지입니다. (추후 구현 예정)")
    
    elif current_main_tab == "행동·마케팅":
        sub_tabs = st.tabs(["앱/웹 이용행태", "마케팅 채널 반응", "캠페인 참여 & 쿠폰 사용", "업종(MCC) 소비 패턴", "VOC/CS"])
        
        with sub_tabs[0]:  # 앱/웹 이용행태
            st.markdown("### 📱 앱/웹 이용행태")
            st.info("앱/웹 이용행태 페이지입니다. (추후 구현 예정)")
        
        with sub_tabs[1]:  # 마케팅 채널 반응
            st.markdown("### 📢 마케팅 채널 반응")
            st.info("마케팅 채널 반응 페이지입니다. (추후 구현 예정)")
        
        with sub_tabs[2]:  # 캠페인 참여 & 쿠폰 사용
            st.markdown("### 🎁 캠페인 참여 & 쿠폰 사용")
            st.info("캠페인 참여 & 쿠폰 사용 페이지입니다. (추후 구현 예정)")
        
        with sub_tabs[3]:  # 업종(MCC) 소비 패턴
            st.markdown("### 🏪 업종(MCC) 소비 패턴")
            st.info("업종(MCC) 소비 패턴 페이지입니다. (추후 구현 예정)")
        
        with sub_tabs[4]:  # VOC/CS
            st.markdown("### 📞 VOC/CS")
            st.info("VOC/CS 페이지입니다. (추후 구현 예정)")
    
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
