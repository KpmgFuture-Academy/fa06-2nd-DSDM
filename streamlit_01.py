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
    page_title="카드사 고객 세그먼트 분석",
    page_icon="💳",
    layout="wide"
)

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
st.title("💳 카드사 고객 세그먼트 분석")

# 데이터 로드
df = load_data()

if df is not None:
    # 사이드바
    st.sidebar.title("📊 대시보드 메뉴")
    
    # 메인 탭 선택
    main_tab = st.sidebar.selectbox(
        "메인 분석",
        ["고객 세그먼트 분석", "Starter (경영 개요)"]
    )
    
    # 각 탭별 서브탭 설정
    if main_tab == "고객 세그먼트 분석":
        sub_tab = st.sidebar.selectbox(
            "세부 분석",
            ["개요", "세그먼트별 상세", "트렌드 분석", "비교 분석"]
        )
    elif main_tab == "Starter (경영 개요)":
        sub_tab = st.sidebar.selectbox(
            "경영 개요",
            ["개요", "지역", "업종", "채널"]
        )
    
    # 세그먼트별 고객 수 계산
    segment_counts = df['Segment'].value_counts().sort_index()
    total_customers = len(df)
    
    # 탭별 컨텐츠 렌더링
    if main_tab == "고객 세그먼트 분석":
        st.markdown(f"## 🎯 고객 세그먼트 분석 - {sub_tab}")
        
        if sub_tab == "개요":
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
            st.markdown("### 📊 세그먼트별 고객 분포")
            
            # 차트용 데이터 준비
            df_segments = pd.DataFrame({
                '세그먼트': segments,
                '고객수': [segment_counts.get(seg, 0) for seg in segments]
            })
            
            col1, col2 = st.columns(2)
            
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
        
        elif sub_tab == "세그먼트별 상세":
            st.markdown("### 📈 세그먼트별 상세 분석")
            
            # 색상 정의
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            
            # 의미 있는 변수들 정의
            analysis_cols = {
                '이용금액_일시불_B0M': '당월 일시불 이용금액',
                '이용금액_할부_B0M': '당월 할부 이용금액', 
                '이용건수_일시불_B0M': '당월 일시불 이용건수'
            }
            
            # 존재하는 컬럼만 필터링
            available_cols = {col: desc for col, desc in analysis_cols.items() if col in df.columns}
            
            if len(available_cols) > 0:
                segment_stats_list = []
                for col, desc in available_cols.items():
                    stats = df.groupby('Segment')[col].agg(['count', 'mean', 'std']).round(2)
                    stats.columns = ['고객수', f'{desc}_평균', f'{desc}_표준편차']
                    segment_stats_list.append(stats)
                
                segment_stats = pd.concat(segment_stats_list, axis=1)
                
                # 중복된 고객수 컬럼 제거
                customer_count_cols = [col for col in segment_stats.columns if col == '고객수']
                if len(customer_count_cols) > 1:
                    cols_to_drop = customer_count_cols[1:]
                    segment_stats = segment_stats.drop(columns=cols_to_drop)
                
                # 첫 번째 컬럼으로 차트 생성
                first_col = list(available_cols.keys())[0]
                first_desc = available_cols[first_col]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 세그먼트별 평균값 차트
                    fig_avg = px.bar(
                        segment_stats.reset_index(),
                        x='Segment',
                        y=f'{first_desc}_평균',
                        title=f"세그먼트별 {first_desc} 평균",
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
                
                with col2:
                    # 상세 통계 테이블
                    st.markdown("### 📋 세그먼트별 상세 통계")
                    st.dataframe(segment_stats, use_container_width=True)
            else:
                st.warning("분석할 수 있는 컬럼이 없습니다.")
        
        elif sub_tab == "트렌드 분석":
            st.markdown("### 📈 세그먼트별 트렌드 분석")
            st.info("세그먼트별 트렌드 분석 페이지입니다. (추후 구현 예정)")
        
        elif sub_tab == "비교 분석":
            st.markdown("### 📊 세그먼트별 비교 분석")
            st.info("세그먼트별 비교 분석 페이지입니다. (추후 구현 예정)")
    
    elif main_tab == "Starter (경영 개요)":
        st.markdown(f"## 📈 경영 개요 - {sub_tab}")
        
        if sub_tab == "개요":
            st.markdown("### 📊 KPI 대시보드")
            
            # 실제 데이터 기반 KPI 계산
            # 당월 이용금액 합계 (일시불 + 할부)
            total_amount = 0
            if '이용금액_일시불_B0M' in df.columns:
                total_amount += df['이용금액_일시불_B0M'].sum()
            if '이용금액_할부_B0M' in df.columns:
                total_amount += df['이용금액_할부_B0M'].sum()
            
            # 당월 이용건수 합계
            total_count = 0
            if '이용건수_일시불_B0M' in df.columns:
                total_count += df['이용건수_일시불_B0M'].sum()
            
            # 객단가 계산 (이용금액 / 이용건수)
            avg_amount_per_transaction = total_amount / total_count if total_count > 0 else 0
            
            # ARPU 계산 (총 이용금액 / 총 고객수)
            arpu = total_amount / total_customers if total_customers > 0 else 0
            
            # KPI 카드들
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="총 매출",
                    value=f"₩{total_amount:,.0f}",
                    delta=f"{total_amount/100000000:.1f}억원"
                )
            
            with col2:
                st.metric(
                    label="총 거래건수",
                    value=f"{total_count:,}건",
                    delta=f"{total_count/10000:.1f}만건"
                )
            
            with col3:
                st.metric(
                    label="객단가",
                    value=f"₩{avg_amount_per_transaction:,.0f}",
                    delta=f"{avg_amount_per_transaction/1000:.0f}천원"
                )
            
            with col4:
                st.metric(
                    label="ARPU",
                    value=f"₩{arpu:,.0f}",
                    delta=f"{arpu/1000:.0f}천원"
                )
            
            # 이탈 위험 고객수 (실제 데이터 기반)
            st.markdown("### ⚠️ 이탈 위험 고객 분석")
            
            # 이용금액이 0인 고객을 High Risk로 분류
            high_risk_customers = len(df[(df['이용금액_일시불_B0M'] == 0) & (df['이용금액_할부_B0M'] == 0)])
            high_risk_percentage = (high_risk_customers / total_customers) * 100
            
            # 이용금액이 낮은 고객을 Medium Risk로 분류 (하위 30%)
            if '이용금액_일시불_B0M' in df.columns and '이용금액_할부_B0M' in df.columns:
                df['총이용금액'] = df['이용금액_일시불_B0M'] + df['이용금액_할부_B0M']
                medium_risk_threshold = df['총이용금액'].quantile(0.3)
                medium_risk_customers = len(df[(df['총이용금액'] > 0) & (df['총이용금액'] <= medium_risk_threshold)])
                medium_risk_percentage = (medium_risk_customers / total_customers) * 100
                
                # 나머지를 Low Risk로 분류
                low_risk_customers = total_customers - high_risk_customers - medium_risk_customers
                low_risk_percentage = (low_risk_customers / total_customers) * 100
            else:
                medium_risk_customers = 0
                medium_risk_percentage = 0
                low_risk_customers = total_customers - high_risk_customers
                low_risk_percentage = (low_risk_customers / total_customers) * 100
            
            risk_col1, risk_col2, risk_col3 = st.columns(3)
            
            with risk_col1:
                st.metric(
                    label="High Risk",
                    value=f"{high_risk_customers:,}명",
                    delta=f"{high_risk_percentage:.1f}%"
                )
            
            with risk_col2:
                st.metric(
                    label="Medium Risk", 
                    value=f"{medium_risk_customers:,}명",
                    delta=f"{medium_risk_percentage:.1f}%"
                )
            
            with risk_col3:
                st.metric(
                    label="Low Risk",
                    value=f"{low_risk_customers:,}명",
                    delta=f"{low_risk_percentage:.1f}%"
                )
            
            # LLM 설명 (실제 데이터 기반)
            st.markdown("### 🤖 AI 인사이트")
            st.info(f"""
            **이번 달 활성 고객 중 {high_risk_percentage:.1f}%가 High Risk로 분류.**
            
            주로 당월 이용금액이 0원인 고객군이 해당됩니다.
            이들 고객에게 맞춤형 리텐션 캠페인을 추천합니다.
            
            - 총 매출: ₩{total_amount:,.0f}원
            - 총 거래건수: {total_count:,}건
            - 평균 객단가: ₩{avg_amount_per_transaction:,.0f}원
            """)
        
        elif sub_tab == "지역":
            st.markdown("### 🗺️ 지역별 분석")
            
            # 실제 데이터 기반 지역별 분석
            if '거주시도명' in df.columns:
                # 지역별 데이터 계산
                region_stats = df.groupby('거주시도명').agg({
                    '이용금액_일시불_B0M': 'sum',
                    '이용금액_할부_B0M': 'sum',
                    '이용건수_일시불_B0M': 'sum',
                    'Segment': 'count'
                }).reset_index()
                
                # 총 이용금액 계산
                region_stats['총이용금액'] = region_stats['이용금액_일시불_B0M'] + region_stats['이용금액_할부_B0M']
                region_stats['ARPU'] = region_stats['총이용금액'] / region_stats['Segment']
                
                # High Risk 고객 비율 계산
                high_risk_by_region = df.groupby('거주시도명').apply(
                    lambda x: len(x[(x['이용금액_일시불_B0M'] == 0) & (x['이용금액_할부_B0M'] == 0)]) / len(x) * 100
                ).reset_index()
                high_risk_by_region.columns = ['거주시도명', 'High Risk 비율(%)']
                
                # 데이터 병합
                region_data = region_stats.merge(high_risk_by_region, on='거주시도명')
                region_data = region_data.rename(columns={
                    '거주시도명': '지역',
                    'Segment': '고객수',
                    '총이용금액': '매출(원)'
                })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_region_sales = px.bar(
                        region_data,
                        x='지역',
                        y='매출(원)',
                        title="지역별 매출",
                        color='매출(원)',
                        color_continuous_scale='Blues'
                    )
                    fig_region_sales.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_region_sales, use_container_width=True)
                
                with col2:
                    fig_region_risk = px.bar(
                        region_data,
                        x='지역',
                        y='High Risk 비율(%)',
                        title="지역별 이탈 위험도",
                        color='High Risk 비율(%)',
                        color_continuous_scale='Reds'
                    )
                    fig_region_risk.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_region_risk, use_container_width=True)
                
                # 지역별 상세 테이블
                st.markdown("### 📊 지역별 상세 통계")
                region_display = region_data[['지역', '고객수', '매출(원)', 'ARPU', 'High Risk 비율(%)']].copy()
                region_display['매출(억원)'] = region_display['매출(원)'] / 100000000
                region_display['ARPU(만원)'] = region_display['ARPU'] / 10000
                region_display = region_display.drop(['매출(원)', 'ARPU'], axis=1)
                region_display = region_display.round(2)
                st.dataframe(region_display, use_container_width=True)
                
                # LLM 설명
                st.markdown("### 🤖 AI 인사이트")
                highest_risk_region = region_data.loc[region_data['High Risk 비율(%)'].idxmax()]
                st.warning(f"""
                **{highest_risk_region['지역']} 거주 고객의 이탈 High Risk 비율이 {highest_risk_region['High Risk 비율(%)']:.1f}%로 가장 높음.**
                
                지역별 맞춤형 마케팅 전략이 필요합니다.
                """)
            else:
                st.warning("지역 정보 컬럼이 없습니다.")
        
        elif sub_tab == "업종":
            st.markdown("### 🏪 업종별 분석")
            
            # 실제 데이터 기반 업종별 분석
            if '_1순위업종' in df.columns:
                # 1순위 업종별 데이터 계산
                category_stats = df.groupby('_1순위업종').agg({
                    '_1순위업종_이용금액': 'sum',
                    'Segment': 'count'
                }).reset_index()
                
                # High Risk 고객의 업종별 분포
                high_risk_df = df[(df['이용금액_일시불_B0M'] == 0) & (df['이용금액_할부_B0M'] == 0)]
                high_risk_category = high_risk_df.groupby('_1순위업종').size().reset_index()
                high_risk_category.columns = ['_1순위업종', 'High Risk 고객수']
                
                # 데이터 병합
                category_data = category_stats.merge(high_risk_category, on='_1순위업종', how='left')
                category_data['High Risk 고객수'] = category_data['High Risk 고객수'].fillna(0)
                category_data['High Risk 비중(%)'] = (category_data['High Risk 고객수'] / category_data['Segment']) * 100
                
                # 상위 10개 업종만 표시
                category_data = category_data.nlargest(10, '_1순위업종_이용금액')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_category_treemap = px.treemap(
                        category_data,
                        path=['_1순위업종'],
                        values='_1순위업종_이용금액',
                        title="업종별 매출 트리맵",
                        color='_1순위업종_이용금액',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_category_treemap, use_container_width=True)
                
                with col2:
                    fig_category_risk = px.bar(
                        category_data,
                        x='High Risk 비중(%)',
                        y='_1순위업종',
                        orientation='h',
                        title="High Risk 고객이 많이 이용한 업종",
                        color='High Risk 비중(%)',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig_category_risk, use_container_width=True)
                
                # 업종별 상세 테이블
                st.markdown("### 📊 업종별 상세 통계")
                category_display = category_data[['_1순위업종', 'Segment', '_1순위업종_이용금액', 'High Risk 비중(%)']].copy()
                category_display['매출(억원)'] = category_display['_1순위업종_이용금액'] / 100000000
                category_display = category_display.rename(columns={
                    '_1순위업종': '업종',
                    'Segment': '고객수',
                    '_1순위업종_이용금액': '매출(원)'
                })
                category_display = category_display.drop(['매출(원)'], axis=1)
                category_display = category_display.round(2)
                st.dataframe(category_display, use_container_width=True)
                
                # LLM 설명
                st.markdown("### 🤖 AI 인사이트")
                highest_risk_category = category_data.loc[category_data['High Risk 비중(%)'].idxmax()]
                st.error(f"""
                **이탈 위험 고객이 가장 많이 이용한 업종은 '{highest_risk_category['_1순위업종']}'으로, High Risk 비중이 {highest_risk_category['High Risk 비중(%)']:.1f}%입니다.**
                
                해당 업종에 대한 특별한 리텐션 전략이 필요합니다.
                """)
            else:
                st.warning("업종 정보 컬럼이 없습니다.")
        
        elif sub_tab == "채널":
            st.markdown("### 📱 채널별 분석")
            
            # 실제 데이터 기반 채널별 분석
            if '이용금액_온라인_R6M' in df.columns and '이용금액_오프라인_R6M' in df.columns:
                # 채널별 총 이용금액 계산
                online_total = df['이용금액_온라인_R6M'].sum()
                offline_total = df['이용금액_오프라인_R6M'].sum()
                total_channel_amount = online_total + offline_total
                
                # 채널별 비율 계산
                online_ratio = (online_total / total_channel_amount) * 100 if total_channel_amount > 0 else 0
                offline_ratio = (offline_total / total_channel_amount) * 100 if total_channel_amount > 0 else 0
                
                # High Risk 고객의 채널별 분포
                high_risk_df = df[(df['이용금액_일시불_B0M'] == 0) & (df['이용금액_할부_B0M'] == 0)]
                high_risk_online = high_risk_df['이용금액_온라인_R6M'].sum()
                high_risk_offline = high_risk_df['이용금액_오프라인_R6M'].sum()
                high_risk_total = high_risk_online + high_risk_offline
                
                high_risk_online_ratio = (high_risk_online / high_risk_total) * 100 if high_risk_total > 0 else 0
                high_risk_offline_ratio = (high_risk_offline / high_risk_total) * 100 if high_risk_total > 0 else 0
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 전체 채널별 이용 추이 (월별 데이터가 없으므로 현재 비율로 표시)
                    channel_data = pd.DataFrame({
                        '채널': ['온라인', '오프라인'],
                        '비율(%)': [online_ratio, offline_ratio],
                        '이용금액(억원)': [online_total/100000000, offline_total/100000000]
                    })
                    
                    fig_channel_pie = px.pie(
                        channel_data,
                        values='비율(%)',
                        names='채널',
                        title="채널별 이용 비율",
                        color='채널',
                        color_discrete_map={'온라인': '#3498db', '오프라인': '#e74c3c'}
                    )
                    st.plotly_chart(fig_channel_pie, use_container_width=True)
                
                with col2:
                    # High Risk 고객 채널별 데이터
                    risk_channel_data = pd.DataFrame({
                        '채널': ['온라인', '오프라인'],
                        '비중(%)': [high_risk_online_ratio, high_risk_offline_ratio]
                    })
                    
                    fig_risk_channel = px.pie(
                        risk_channel_data,
                        values='비중(%)',
                        names='채널',
                        title="High Risk 고객 채널별 이용 패턴",
                        color='채널',
                        color_discrete_map={'온라인': '#3498db', '오프라인': '#e74c3c'}
                    )
                    st.plotly_chart(fig_risk_channel, use_container_width=True)
                
                # 채널별 상세 테이블
                st.markdown("### 📊 채널별 상세 통계")
                channel_display = pd.DataFrame({
                    '채널': ['온라인', '오프라인', '전체'],
                    '이용금액(억원)': [online_total/100000000, offline_total/100000000, total_channel_amount/100000000],
                    '비율(%)': [online_ratio, offline_ratio, 100],
                    'High Risk 비중(%)': [high_risk_online_ratio, high_risk_offline_ratio, 100]
                })
                channel_display = channel_display.round(2)
                st.dataframe(channel_display, use_container_width=True)
                
                # LLM 설명
                st.markdown("### 🤖 AI 인사이트")
                dominant_channel = "온라인" if online_ratio > offline_ratio else "오프라인"
                dominant_risk_channel = "온라인" if high_risk_online_ratio > high_risk_offline_ratio else "오프라인"
                
                st.info(f"""
                **전체 고객은 {dominant_channel} 채널을 주로 이용하지만, High Risk 고객은 {dominant_risk_channel} 채널을 더 많이 이용합니다.**
                
                채널 전환을 유도하는 프로모션과 간편결제 도입 인센티브가 필요합니다.
                
                - 온라인 채널 비율: {online_ratio:.1f}%
                - 오프라인 채널 비율: {offline_ratio:.1f}%
                """)
            else:
                st.warning("채널 정보 컬럼이 없습니다.")
    
    
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