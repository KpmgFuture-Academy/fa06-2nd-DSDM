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
    page_title="카드사 고객 세그먼트",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 메인 제목
st.title("💳 카드사 고객 세그먼트")
st.markdown("---")

# 데이터 로드
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('base_test_merged_seg.csv', low_memory=False)
        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
        return None

# 데이터 로드
df = load_data()

if df is not None:
    # 사이드바
    st.sidebar.title("📊 대시보드 메뉴")
    st.sidebar.markdown("### 고객 세그먼트 분석")
    
    # 세그먼트별 고객 수 계산
    segment_counts = df['Segment'].value_counts().sort_index()
    total_customers = len(df)
    
    # 메인 컨텐츠 영역
    st.markdown("## 🎯 고객 세그먼트 개요")
    
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
    
    st.markdown("---")
    
    # 차트 영역
    st.markdown("## 📈 세그먼트별 분석")
    
    # 세그먼트별 데이터 준비
    segment_data = []
    for segment in segments:
        count = segment_counts.get(segment, 0)
        percentage = (count / total_customers) * 100
        segment_data.append({
            '세그먼트': segment,
            '고객수': count,
            '비율(%)': round(percentage, 1)
        })
    
    df_segments = pd.DataFrame(segment_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 세그먼트별 고객 수 파이차트
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
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # 세그먼트별 고객 수 바차트
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
            },
            text='고객수'
        )
        fig_bar.update_traces(texttemplate='%{text:,}', textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # 추가 분석 섹션
    st.markdown("## 📊 세그먼트별 상세 분석")
    
    # 의미 있는 변수들 정의
    analysis_cols = {
        '총이용금액_R6M': '최근 6개월 총 이용금액',
        '총이용건수_R6M': '최근 6개월 총 이용건수', 
        '단가_R6M': '평균 단가'
    }
    
    # 존재하는 컬럼만 필터링
    available_cols = {col: desc for col, desc in analysis_cols.items() if col in df.columns}
    
    if len(available_cols) > 0:
        # 세그먼트별 통계 계산
        segment_stats_list = []
        
        for col, desc in available_cols.items():
            stats = df.groupby('Segment')[col].agg(['count', 'mean', 'std']).round(2)
            stats.columns = ['고객수', f'{desc}_평균', f'{desc}_표준편차']
            segment_stats_list.append(stats)
        
        # 통계 테이블 결합
        segment_stats = pd.concat(segment_stats_list, axis=1)
        
        # 중복된 고객수 컬럼 제거 (첫 번째만 유지)
        customer_count_cols = [col for col in segment_stats.columns if col == '고객수']
        if len(customer_count_cols) > 1:
            cols_to_drop = customer_count_cols[1:]
            segment_stats = segment_stats.drop(columns=cols_to_drop)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 세그먼트별 통계")
            st.dataframe(segment_stats, use_container_width=True)
        
        with col2:
            # 세그먼트별 평균값 비교 (첫 번째 변수 기준)
            first_col = list(available_cols.keys())[0]
            first_desc = available_cols[first_col]
            
            fig_avg = px.bar(
                segment_stats.reset_index(),
                x='Segment',
                y=f'{first_desc}_평균',
                title=f"세그먼트별 {first_desc}",
                color='Segment',
                color_discrete_map={
                    'A': colors[0],  # #FF6B6B
                    'B': colors[1],  # #4ECDC4
                    'C': colors[2],  # #45B7D1
                    'D': colors[3],  # #96CEB4
                    'E': colors[4]   # #FFEAA7
                }
            )
            st.plotly_chart(fig_avg, use_container_width=True)
    
    else:
        st.warning("분석할 수 있는 컬럼이 없습니다.")
    
    # 하단 영역
    st.markdown("## 📋 세그먼트별 요약")
    
    # 세그먼트별 요약 정보
    summary_data = []
    for segment in segments:
        count = segment_counts.get(segment, 0)
        percentage = (count / total_customers) * 100
        summary_data.append({
            '세그먼트': segment,
            '고객수': f"{count:,}",
            '비율(%)': f"{percentage:.1f}%",
            '특징': f"세그먼트 {segment} 고객군"
        })
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True)
    
    # 푸터
    st.markdown("---")
    st.markdown("### 💡 인사이트")
    
    # 동적 인사이트 생성
    largest_segment = segment_counts.idxmax()
    largest_count = segment_counts.max()
    largest_percentage = (largest_count / total_customers) * 100
    
    smallest_segment = segment_counts.idxmin()
    smallest_count = segment_counts.min()
    smallest_percentage = (smallest_count / total_customers) * 100
    
    st.info(f"""
    - **가장 큰 세그먼트**: 세그먼트 {largest_segment} ({largest_count:,}명, {largest_percentage:.1f}%)
    - **가장 작은 세그먼트**: 세그먼트 {smallest_segment} ({smallest_count:,}명, {smallest_percentage:.1f}%)
    - **전체 고객 수**: {total_customers:,}명
    - **세그먼트 수**: {len(segments)}개 (A, B, C, D, E)
    """)
    
    # 사이드바 하단 정보
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
