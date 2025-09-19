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
            st.info("세그먼트별 상세 정보 페이지입니다. (추후 구현 예정)")
        
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