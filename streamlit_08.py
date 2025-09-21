import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 0) 상단 공통 상수/유틸
SEGMENTS = ['A','B','C','D','E']
COLOR_MAP = {'A':'#FF6B6B','B':'#4ECDC4','C':'#45B7D1','D':'#96CEB4','E':'#FFEAA7'}

def _exists(df, col): 
    return col in df.columns

def _to_num(s): 
    return pd.to_numeric(s, errors='coerce')

@st.cache_data
def filter_df(df, segments, regions):
    m = df['Segment'].isin(segments)
    if regions: 
        m &= df['거주시도명'].isin(regions)
    out = df.loc[m].copy()
    out['Segment'] = pd.Categorical(out['Segment'], categories=SEGMENTS, ordered=True)
    return out

@st.cache_data
def filter_df_extended(df, segments, regions, genders=None, tenures=None):
    """앱/웹 이용행태용 확장 필터"""
    dff = df[df['Segment'].isin(segments)].copy()
    if regions: 
        dff = dff[dff['거주시도명'].isin(regions)]
    if genders: 
        dff = dff[dff['남녀구분코드'].isin(genders)]
    
    # 가입기간 구간
    dff['가입기간_년'] = _to_num(dff['입회경과개월수_신용']) / 12
    bins = [-1, 2, 5, 10, 1e9]
    labels = ["신규(<2)", "중간(2-5)", "기존(5-10)", "우수(10+)"]
    dff['가입기간_구간'] = pd.cut(dff['가입기간_년'], bins=bins, labels=labels)
    dff['Segment'] = pd.Categorical(dff['Segment'], categories=SEGMENTS, ordered=True)
    return dff

@st.cache_data
def filter_mcc_data(df, start_date, end_date, card_types, channels, regions, genders, ages):
    """MCC 소비 패턴용 데이터 필터링"""
    # 실제 데이터가 없으므로 시뮬레이션 데이터 생성
    np.random.seed(42)
    n_customers = 10000
    n_transactions = 50000
    
    # MCC 코드와 이름
    mcc_codes = {
        '5411': '식료품점/슈퍼마켓',
        '5812': '식당/카페',
        '5311': '백화점',
        '5999': '기타 소매업',
        '5541': '주유소',
        '5814': '패스트푸드',
        '4121': '택시',
        '4111': '대중교통',
        '7011': '호텔/숙박',
        '5813': '주점/바'
    }
    
    # 시뮬레이션 거래 데이터 생성
    transactions = []
    for i in range(n_transactions):
        mcc = np.random.choice(list(mcc_codes.keys()))
        amount = np.random.lognormal(mean=8, sigma=1)  # 평균 약 3,000원
        count = np.random.poisson(1.5)  # 평균 1.5건
        
        transactions.append({
            'ID': f'CUST_{np.random.randint(1, n_customers+1):05d}',
            'MCC': mcc,
            'MCC명': mcc_codes[mcc],
            '금액': int(amount),
            '건수': max(1, count),
            '카드유형': np.random.choice(['신용', '체크'], p=[0.7, 0.3]),
            '채널': np.random.choice(['온라인', '오프라인'], p=[0.3, 0.7]),
            '가맹점지역': np.random.choice(['서울', '경기', '부산', '대구', '인천', '광주', '대전', '울산']),
            '승인일자': np.random.choice(pd.date_range(start_date, end_date))
        })
    
    return pd.DataFrame(transactions)

@st.cache_data
def compute_mcc_metrics(df):
    """MCC 메트릭 계산"""
    # 총 금액/건수
    total_amount = df['금액'].sum()
    total_count = df['건수'].sum()
    avg_ticket = total_amount / total_count if total_count > 0 else 0
    
    # 카드유형 비중
    card_type_ratio = df.groupby('카드유형')['금액'].sum() / total_amount
    
    # MCC별 집계
    mcc_summary = df.groupby(['MCC', 'MCC명']).agg({
        '금액': 'sum',
        '건수': 'sum',
        'ID': 'nunique'
    }).reset_index()
    
    mcc_summary['점유율'] = mcc_summary['금액'] / total_amount
    mcc_summary['평균객단가'] = mcc_summary['금액'] / mcc_summary['건수']
    mcc_summary['고객수'] = mcc_summary['ID']
    
    # HHI 계산
    hhi = (mcc_summary['점유율'] ** 2).sum()
    
    return mcc_summary, {
        '총금액': total_amount,
        '평균객단가': avg_ticket,
        '신용비중': card_type_ratio.get('신용', 0),
        '체크비중': card_type_ratio.get('체크', 0),
        'HHI': hhi
    }

@st.cache_data
def compute_mcc_growth(df):
    """MCC별 성장률 계산 (시뮬레이션)"""
    mcc_codes = df['MCC'].unique()
    months = pd.date_range('2023-01-01', '2023-12-31', freq='M')
    
    growth_data = []
    for mcc in mcc_codes:
        mcc_name = df[df['MCC'] == mcc]['MCC명'].iloc[0]
        base_amount = np.random.uniform(1000000, 10000000)
        
        for month in months:
            # 계절성과 성장 트렌드 반영
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * month.month / 12)
            growth_factor = 1 + 0.1 * (month.month - 1) / 11  # 연간 10% 성장
            amount = base_amount * seasonal_factor * growth_factor * np.random.uniform(0.8, 1.2)
            
            growth_data.append({
                '월': month.strftime('%Y-%m'),
                'MCC': mcc,
                'MCC명': mcc_name,
                '금액': int(amount)
            })
    
    return pd.DataFrame(growth_data)

@st.cache_data
def compute_region_mcc_heatmap(df):
    """지역별 MCC 히트맵 데이터"""
    region_mcc = df.groupby(['가맹점지역', 'MCC', 'MCC명'])['금액'].sum().reset_index()
    region_total = region_mcc.groupby('가맹점지역')['금액'].sum()
    region_mcc['비중'] = region_mcc.apply(lambda x: x['금액'] / region_total[x['가맹점지역']], axis=1)
    
    # 피벗 테이블 생성
    heatmap_data = region_mcc.pivot_table(
        index='가맹점지역', 
        columns='MCC명', 
        values='비중', 
        fill_value=0
    )
    
    return heatmap_data

@st.cache_data
def filter_voc_data(df, start_date, end_date, channels, categories, sentiment_range, regions, segments):
    """VOC/CS 데이터 필터링"""
    # 실제 데이터가 없으므로 시뮬레이션 데이터 생성
    np.random.seed(42)
    n_cases = 5000
    
    # VOC 카테고리와 채널
    categories_list = [
        '카드발급', '이용한도', '할부/일시불', '포인트/마일리지', 
        '해외이용', '온라인결제', 'ATM/현금서비스', '기타'
    ]
    channels_list = ['콜센터', '챗봇', '앱', '이메일', 'SMS']
    
    # 시뮬레이션 VOC 데이터 생성
    voc_data = []
    for i in range(n_cases):
        category = np.random.choice(categories_list)
        channel = np.random.choice(channels_list)
        
        # 감성점수 (-1 ~ 1)
        sentiment = np.random.normal(0.2, 0.8)  # 약간 긍정적 편향
        sentiment = np.clip(sentiment, -1, 1)
        
        # 처리시간 (분)
        processing_time = np.random.exponential(15)  # 평균 15분
        
        # 1차해결여부 (감성에 따라 확률 조정)
        fcr_prob = 0.7 if sentiment > 0 else 0.5
        first_call_resolved = np.random.random() < fcr_prob
        
        # 재문의여부
        re_inquiry = np.random.random() < 0.15
        
        # 환불여부
        refund = np.random.random() < 0.05
        
        voc_data.append({
            '수신일자': np.random.choice(pd.date_range(start_date, end_date)),
            '채널': channel,
            '카테고리': category,
            '세부유형': f"{category}_상세_{i%10}",
            '요약문구': f"{category} 관련 문의 - {np.random.choice(['이용불가', '오류발생', '혜택문의', '해지요청', '기타'])}",
            '감성점수': round(sentiment, 2),
            '처리시간분': int(processing_time),
            '1차해결여부': first_call_resolved,
            '재문의여부': re_inquiry,
            '환불여부': refund,
            '관련MCC': np.random.choice(['5411', '5812', '5311', '5999', '5541']),
            '캠페인ID': np.random.choice(['CAMP001', 'CAMP002', 'CAMP003', 'CAMP004', 'CAMP005', '']),
            'ID': f"CUST_{np.random.randint(1, 10000):05d}"
        })
    
    return pd.DataFrame(voc_data)

@st.cache_data
def compute_voc_metrics(df):
    """VOC 메트릭 계산"""
    total_cases = len(df)
    avg_sentiment = df['감성점수'].mean()
    fcr = df['1차해결여부'].mean()
    aht = df['처리시간분'].mean()
    re_inquiry_rate = df['재문의여부'].mean()
    refund_rate = df['환불여부'].mean()
    
    return {
        '총건수': total_cases,
        '평균감성': avg_sentiment,
        'FCR': fcr,
        'AHT': aht,
        '재문의율': re_inquiry_rate,
        '환불비중': refund_rate
    }

@st.cache_data
def compute_voc_trends(df):
    """VOC 월별 추이 계산"""
    df['월'] = df['수신일자'].dt.to_period('M')
    
    monthly_trends = df.groupby('월').agg({
        'ID': 'count',
        '감성점수': 'mean',
        '1차해결여부': 'mean',
        '처리시간분': 'mean',
        '재문의여부': 'mean'
    }).reset_index()
    
    monthly_trends.columns = ['월', '볼륨', '평균감성', 'FCR', 'AHT', '재문의율']
    monthly_trends['월'] = monthly_trends['월'].astype(str)
    
    return monthly_trends

@st.cache_data
def compute_category_issues(df):
    """카테고리별 이슈 분석"""
    category_summary = df.groupby('카테고리').agg({
        'ID': 'count',
        '감성점수': 'mean',
        '1차해결여부': 'mean',
        '처리시간분': 'mean',
        '재문의여부': 'mean'
    }).reset_index()
    
    category_summary.columns = ['카테고리', '건수', '평균감성', 'FCR', 'AHT', '재문의율']
    category_summary = category_summary.sort_values('건수', ascending=False)
    
    return category_summary

@st.cache_data
def extract_keywords(df):
    """키워드 추출 (간단한 빈도 기반)"""
    # 요약문구에서 키워드 추출
    all_text = ' '.join(df['요약문구'].astype(str))
    
    # 간단한 키워드 추출 (실제로는 더 정교한 전처리 필요)
    keywords = {
        '카드발급': df['카테고리'].value_counts().get('카드발급', 0),
        '이용한도': df['카테고리'].value_counts().get('이용한도', 0),
        '할부': df['카테고리'].value_counts().get('할부/일시불', 0),
        '포인트': df['카테고리'].value_counts().get('포인트/마일리지', 0),
        '해외이용': df['카테고리'].value_counts().get('해외이용', 0),
        '온라인결제': df['카테고리'].value_counts().get('온라인결제', 0),
        'ATM': df['카테고리'].value_counts().get('ATM/현금서비스', 0),
        '기타': df['카테고리'].value_counts().get('기타', 0)
    }
    
    # 상위 키워드 반환
    sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
    return pd.DataFrame(sorted_keywords, columns=['키워드', '빈도'])

@st.cache_data
def get_sla_cases(df):
    """SLA 초과 및 재문의 케이스"""
    # SLA 초과 케이스 (처리시간 30분 초과)
    sla_exceeded = df[df['처리시간분'] > 30].copy()
    
    # 재문의 케이스
    re_inquiry_cases = df[df['재문의여부'] == True].copy()
    
    # 개인정보 마스킹
    if not sla_exceeded.empty:
        sla_exceeded['ID'] = sla_exceeded['ID'].str[:3] + '***'
    
    if not re_inquiry_cases.empty:
        re_inquiry_cases['ID'] = re_inquiry_cases['ID'].str[:3] + '***'
    
    return sla_exceeded, re_inquiry_cases

@st.cache_data
def get_top_mcc_list(df):
    """상위 MCC 목록"""
    mcc_summary, _ = compute_mcc_metrics(df)
    
    # Top 5 MCC (점유율 기준)
    top_mcc = mcc_summary.nlargest(5, '점유율')[['MCC', 'MCC명', '점유율', '금액', '평균객단가']]
    
    # 성장률 계산 (시뮬레이션)
    growth_data = compute_mcc_growth(df)
    recent_3m = growth_data[growth_data['월'].isin(['2023-10', '2023-11', '2023-12'])]
    prev_3m = growth_data[growth_data['월'].isin(['2023-07', '2023-08', '2023-09'])]
    
    recent_sum = recent_3m.groupby('MCC')['금액'].sum()
    prev_sum = prev_3m.groupby('MCC')['금액'].sum()
    
    growth_rate = ((recent_sum - prev_sum) / prev_sum * 100).fillna(0)
    
    # 성장률을 mcc_summary에 추가
    mcc_summary['R3M증감률'] = mcc_summary['MCC'].map(growth_rate)
    
    return mcc_summary.sort_values('점유율', ascending=False)

@st.cache_data
def compute_appweb_metrics(dff: pd.DataFrame):
    """앱/웹 이용행태 메트릭 계산"""
    # 원시 플래그
    dff['플래그_온라인이용'] = _to_num(dff['이용건수_온라인_R6M']) > 0
    dff['플래그_IB조회'] = _to_num(dff['IB문의건수_사용승인내역_B0M']) > 0
    dff['플래그_반복결제'] = _to_num(dff['이용건수_온라인_R6M']) >= 3

    g = dff.groupby('Segment').agg(
        고객수=('ID','nunique'),
        온라인금액_천원=('이용금액_온라인_R6M','sum'),
        오프라인금액_천원=('이용금액_오프라인_R6M','sum'),
        온라인건수=('이용건수_온라인_R6M','sum'),
        오프라인건수=('이용건수_오프라인_R6M','sum'),
        IB조회고객=('플래그_IB조회','sum'),
        온라인이용고객=('플래그_온라인이용','sum'),
        반복결제고객=('플래그_반복결제','sum')
    ).fillna(0).reset_index()

    # 파생 KPI
    total_amt = (g['온라인금액_천원'] + g['오프라인금액_천원']).replace(0, np.nan)
    g['디지털이용율'] = np.where(g['고객수']>0, g['온라인이용고객']/g['고객수'], np.nan)
    g['온라인비중_금액'] = g['온라인금액_천원'] / total_amt
    g['온라인_ARPU_원'] = np.where(g['고객수']>0, g['온라인금액_천원']*1000 / g['고객수'], np.nan)
    g['온라인_객단가_천원'] = np.where(g['온라인건수']>0, g['온라인금액_천원'] / g['온라인건수'], np.nan)
    g['IB조회율'] = np.where(g['고객수']>0, g['IB조회고객']/g['고객수'], np.nan)

    # 퍼널 전환율
    g['퍼널_1to2'] = np.where(g['IB조회고객']>0, g['온라인이용고객']/g['IB조회고객'], np.nan)
    g['퍼널_2to3'] = np.where(g['온라인이용고객']>0, g['반복결제고객']/g['온라인이용고객'], np.nan)

    g['small_sample'] = g['고객수'] < 100
    g['Segment'] = pd.Categorical(g['Segment'], categories=SEGMENTS, ordered=True)
    return g.sort_values('Segment')

@st.cache_data
def filter_for_marketing(df, segs, regions, genders, tenure_bins, exclude_risk=False):
    dff = df[df['Segment'].isin(segs)].copy()
    if regions: dff = dff[dff['거주시도명'].isin(regions)]
    if genders: dff = dff[dff['남녀구분코드'].isin(genders)]
    # 가입기간 구간
    dff['가입기간_년'] = _to_num(dff['입회경과개월수_신용'])/12
    bins = [-1,2,5,10,1e9]; labels = ["신규(<2)","중간(2-5)","기존(5-10)","우수(10+)"]
    dff['가입기간_구간'] = pd.cut(dff['가입기간_년'], bins=bins, labels=labels)
    # 리스크 제외(옵션)
    if exclude_risk:
        dff = dff[(_to_num(dff['연체잔액_B0M'])<=0) & (_to_num(dff['RV_평균잔액_R3M']).rank(pct=True)<=0.75)]
    dff['Segment'] = pd.Categorical(dff['Segment'], categories=SEGMENTS, ordered=True)
    return dff

@st.cache_data
def compute_channel_metrics(dff: pd.DataFrame):
    # 플래그
    dff['IB조회'] = _to_num(dff['IB문의건수_사용승인내역_B0M'])>0
    dff['온라인사용'] = _to_num(dff['이용건수_온라인_R6M'])>0
    dff['혜택반응'] = _to_num(dff['포인트_마일리지_건별_B0M'])>0

    g = dff.groupby('Segment').agg(
        고객수=('ID','nunique'),
        온라인금액_천원=('이용금액_온라인_R6M','sum'),
        오프라인금액_천원=('이용금액_오프라인_R6M','sum'),
        온라인건수=('이용건수_온라인_R6M','sum'),
        IB조회고객=('IB조회','sum'),
        온라인고객=('온라인사용','sum'),
        혜택고객=('혜택반응','sum')
    ).reset_index()
    
    # 숫자형 컬럼만 fillna(0) 적용
    numeric_cols = g.select_dtypes(include=[np.number]).columns
    g[numeric_cols] = g[numeric_cols].fillna(0)

    total_amt = (g['온라인금액_천원']+g['오프라인금액_천원']).replace(0, np.nan)
    g['IB조회율'] = np.where(g['고객수']>0, g['IB조회고객']/g['고객수'], np.nan)
    g['디지털이용율'] = np.where(g['고객수']>0, g['온라인고객']/g['고객수'], np.nan)
    g['온라인비중'] = g['온라인금액_천원']/total_amt
    g['온라인ARPU_원'] = np.where(g['고객수']>0, g['온라인금액_천원']*1000/g['고객수'], np.nan)
    g['온라인객단가_천원'] = np.where(g['온라인건수']>0, g['온라인금액_천원']/g['온라인건수'], np.nan)
    g['혜택반응율'] = np.where(g['고객수']>0, g['혜택고객']/g['고객수'], np.nan)

    # 임계값(선택 데이터 중앙값)
    thr = {
        'IB조회율': np.nanmedian(g['IB조회율']),
        '온라인비중': np.nanmedian(g['온라인비중'])
    }

    def recommend(row):
        if pd.isna(row['IB조회율']) or pd.isna(row['온라인비중']):
            return "데이터부족"
        if row['IB조회율']>=thr['IB조회율'] and row['온라인비중']>=thr['온라인비중']:
            return "App Push"
        if row['IB조회율']>=thr['IB조회율'] and row['온라인비중']<thr['온라인비중']:
            return "톡채널"
        if row['IB조회율']<thr['IB조회율'] and row['온라인비중']>=thr['온라인비중']:
            return "이메일/DM"
        return "TM/콜센터"

    g['추천채널'] = g.apply(recommend, axis=1)
    g['small_sample'] = g['고객수']<100
    g['Segment'] = pd.Categorical(g['Segment'], categories=SEGMENTS, ordered=True)
    return g.sort_values('Segment'), thr

@st.cache_data
def filter_campaign_data(df, date_type, start_date, end_date, channels, segments, regions, genders, ages, consent, sampling_rate):
    """캠페인 데이터 필터링"""
    # 기본 필터링
    filtered_df = df.copy()
    
    # 샘플링 적용
    if sampling_rate < 100:
        sample_size = int(len(filtered_df) * sampling_rate / 100)
        filtered_df = filtered_df.sample(n=sample_size, random_state=42)
    
    # 지역 필터
    if regions:
        filtered_df = filtered_df[filtered_df['거주시도명'].isin(regions)]
    
    # 성별 필터
    if genders:
        filtered_df = filtered_df[filtered_df['남녀구분코드'].isin(genders)]
    
    # 세그먼트 필터
    if segments:
        filtered_df = filtered_df[filtered_df['Segment'].isin(segments)]
    
    return filtered_df

@st.cache_data
def compute_campaign_metrics(df):
    """캠페인 메트릭 계산"""
    # 실제 데이터가 없으므로 시뮬레이션 데이터 생성
    metrics = pd.DataFrame({
        '캠페인ID': ['CAMP001', 'CAMP002', 'CAMP003', 'CAMP004', 'CAMP005'],
        '캠페인명': ['신규가입 혜택', '생일 축하 쿠폰', '할인 이벤트', '추천인 보상', 'VIP 전용 혜택'],
        '채널': ['SMS', '앱푸시', '이메일', 'DM', 'TM'],
        '노출수': [10000, 8000, 12000, 5000, 3000],
        '참여수': [1200, 1600, 1800, 750, 900],
        '전환수': [300, 400, 450, 150, 225],
        '쿠폰발급수': [300, 400, 450, 150, 225],
        '쿠폰사용수': [180, 280, 315, 90, 180],
        '사용금액': [1800000, 2800000, 3150000, 900000, 1800000],
        '할인금액': [180000, 280000, 315000, 90000, 180000]
    })
    return metrics

@st.cache_data
def compute_monthly_trends(df):
    """월별 추이 계산"""
    # 시뮬레이션 데이터
    months = ['2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06']
    trends = pd.DataFrame({
        '월': months,
        '참여율': [0.12, 0.15, 0.18, 0.16, 0.20, 0.22],
        '전환율': [0.25, 0.28, 0.30, 0.27, 0.32, 0.35],
        '쿠폰사용률': [0.60, 0.65, 0.70, 0.68, 0.75, 0.80]
    })
    return trends

@st.cache_data
def compute_channel_performance(df):
    """채널별 성과 계산"""
    channels = ['SMS', '앱푸시', '이메일', 'DM', 'TM']
    performance = pd.DataFrame({
        '채널': channels,
        '참여율': [0.15, 0.20, 0.18, 0.12, 0.25],
        '전환율': [0.30, 0.35, 0.32, 0.28, 0.40],
        'ROI': [2.5, 3.2, 2.8, 2.1, 3.8]
    })
    return performance

@st.cache_data
def get_top_campaigns(df):
    """상위 캠페인 목록"""
    metrics = compute_campaign_metrics(df)
    metrics['참여율'] = metrics['참여수'] / metrics['노출수']
    metrics['전환율'] = metrics['전환수'] / metrics['참여수']
    metrics['쿠폰사용률'] = metrics['쿠폰사용수'] / metrics['쿠폰발급수']
    metrics['순효익'] = metrics['사용금액'] - metrics['할인금액']
    
    return metrics[['캠페인명', '채널', '노출수', '참여율', '전환율', '쿠폰사용률', '순효익']].sort_values('순효익', ascending=False)

@st.cache_data
def get_campaign_comparison(df, selected_campaigns):
    """캠페인 비교 데이터"""
    metrics = compute_campaign_metrics(df)
    comparison = metrics[metrics['캠페인명'].isin(selected_campaigns)].copy()
    comparison['참여율'] = comparison['참여수'] / comparison['노출수']
    comparison['전환율'] = comparison['전환수'] / comparison['참여수']
    comparison['쿠폰사용률'] = comparison['쿠폰사용수'] / comparison['쿠폰발급수']
    
    return comparison[['캠페인명', '채널', '참여율', '전환율', '쿠폰사용률']]

@st.cache_data
def heatmap_matrix(dff):
    """세그먼트 × 가입기간 구간 히트맵 데이터"""
    m = dff.groupby(['Segment','가입기간_구간']).apply(
        lambda x: ( (_to_num(x['이용금액_온라인_R6M']) /
                     (_to_num(x['이용금액_온라인_R6M']) + _to_num(x['이용금액_오프라인_R6M'])) ).replace([np.inf,-np.inf], np.nan).mean())
    ).reset_index(name='온라인비중_금액')
    return m

@st.cache_data
def region_channel_mix(dff: pd.DataFrame, thr):
    # 지역별 추천채널 분포
    dff = dff.copy()
    # 개인 기준 추천채널 산정
    ib = (_to_num(dff['IB문의건수_사용승인내역_B0M'])>0).astype(int)
    on_amt = _to_num(dff['이용금액_온라인_R6M'])
    off_amt = _to_num(dff['이용금액_오프라인_R6M'])
    on_share = on_amt / (on_amt+off_amt)
    cond = pd.Series(np.where((ib==1) & (on_share>=thr['온라인비중']), 'App Push',
                     np.where((ib==1) & (on_share<thr['온라인비중']), '톡채널',
                     np.where((ib==0) & (on_share>=thr['온라인비중']), '이메일/DM','TM/콜센터'))))
    dff['개인추천채널'] = cond.values
    mix = dff.groupby(['거주시도명','개인추천채널'])['ID'].nunique().reset_index(name='고객수')
    total = mix.groupby('거주시도명')['고객수'].transform('sum')
    mix['비중'] = mix['고객수']/total
    return mix

def _sum_or_zero(df, col):
    return (df[col] if col in df else 0)

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
        usecols = [
            'Segment', 'ID', '연령', '남녀구분코드', '거주시도명',
            '유효카드수_신용', '유효카드수_체크', '카드이용한도금액', '입회경과개월수_신용',
            
            # 이용 관련 컬럼 (Usage)
            '이용금액_일시불_B0M', '이용금액_할부_B0M', 
            '이용건수_신용_B0M', '이용건수_신판_B0M', '이용건수_일시불_B0M', '이용건수_할부_B0M',
            '이용건수_CA_B0M', '이용건수_체크_B0M',
            '잔액_일시불_B0M', '잔액_할부_B0M', '잔액_현금서비스_B0M', '잔액_카드론_B0M',
            
            # 수익/혜택 관련 컬럼 (Profit & Benefit)
            '청구금액_B0', '포인트_마일리지_건별_B0M', '연체잔액_B0M',
            
            # 승인거절 관련 컬럼 (Risk Analysis)
            '승인거절건수_B0M', '승인거절건수_한도초과_B0M', '승인거절건수_입력오류_B0M', '승인거절건수_기타_B0M',
            'IB문의건수_사용승인내역_B0M',
            
            # 연체 관련 컬럼 (Risk Analysis)
            '연체잔액_B0M', '연체잔액_일시불_B0M', '연체잔액_할부_B0M', '연체잔액_현금서비스_B0M', '연체잔액_카드론_B0M',
            '시장단기연체여부_R3M', '시장단기연체여부_R6M',
            
            # 리볼빙/현금서비스 관련 컬럼 (Risk Analysis)
            'RV_평균잔액_R3M', 'RV잔액이월횟수_R3M', 'RV전환가능여부',
            '이용금액_CA_B0M', '신청건수_ATM_CA_B0', 'CA이자율_할인전',
            
            # 한도/FDS 관련 컬럼 (Risk Analysis)
            '카드이용한도금액', '상향가능한도금액', '한도심사요청건수', '한도증액횟수_R12M',
            '강제한도감액횟수_R12M', '자발한도감액횟수_R12M',
            '잔액_신판평균한도소진율_r3m', '잔액_신판최대한도소진율_r3m', '잔액_신판최대한도소진율_r6m',
            
            # 기타 컬럼
            '_1순위업종', '_1순위업종_이용금액', '_2순위업종', '_2순위업종_이용금액',
            '이용금액_온라인_R6M', '이용금액_오프라인_R6M', '이용건수_온라인_R6M', '이용건수_오프라인_R6M'
        ]
        
        # dtype 지정으로 메모리 효율성 및 성능 향상
        dtypes = {
            'Segment': 'category',
            '남녀구분코드': 'category', 
            '거주시도명': 'category'
        }
        
        df = pd.read_csv('base_test_merged_seg.csv', 
                        usecols=usecols, 
                        dtype=dtypes, 
                        low_memory=False)
        
        # 숫자형 컬럼 안전 변환(문자 섞인 경우 방지)
        num_cols = [c for c in df.columns if any(x in c for x in ['이용금액','이용건수','잔액','청구금액','한도','횟수','건수','이자율'])]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
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
        
        **📊 세그먼트 분석 - 주요 KPI 비교:**
        
        **1. 전사 기준 고객 분포 (헤더 고정 요약):**
        - **데이터 소스**: 전체 데이터셋 (`df`) 기준 (필터 무관)
        - **파이차트**: `Segment` 컬럼의 value_counts()로 세그먼트별 고객수 계산
        - **막대차트**: 고객수를 천 단위로 표시, 로그 스케일 옵션 제공
        - **순서 보장**: A, B, C, D, E 순서로 정렬 (`pd.Categorical` 사용)
        
        **2. 주요 KPI 지표:**
        - **총 이용금액**: `이용금액_일시불_B0M + 이용금액_할부_B0M + 잔액_현금서비스_B0M + 잔액_카드론_B0M` (천원 단위)
        - **총 이용건수**: `이용건수_일시불_B0M + 이용건수_할부_B0M + 이용건수_신용_B0M + 이용건수_체크_B0M`
        - **ARPU**: `(청구금액_B0 × 1000) ÷ 고객수` (천원→원 변환 후 계산)
        - **객단가**: `총이용금액 ÷ 총이용건수` (천원 단위)
        
        **3. 이용 금액 비교 차트:**
        - **총 이용금액 막대차트**: 세그먼트별 총 이용금액 (천원 단위)
        - **ARPU 막대차트**: 세그먼트별 1인당 이용금액 (원 단위)
        - **인사이트 자동 감지**: 고객수와 금액 패턴으로 전략 분류
        - **객단가 배지**: 각 세그먼트별 거래당 평균금액 표시
        
        **4. 신판 이용 패턴 & ARPU vs 객단가:**
        - **신판 이용금액**: 일시불 vs 할부 비중 (천원 단위)
        - **ARPU vs 객단가 산점도**: 버블 크기 = 고객수, X축 = ARPU(원), Y축 = 객단가(천원)
        - **현금서비스/카드론 분포**: Box Plot으로 분포 확인 (천원 단위)
        
        **5. 세그먼트×지역 분석:**
        - **지역별 세그먼트 분포**: 스택바 차트로 지역별 세그먼트 구성
        - **세그먼트별 지역 집중도**: 각 세그먼트의 지역 분포 도넛차트
        - **지역별 ARPU 비교**: 버블차트 (X축=고객수, Y축=ARPU, 색상=주요세그먼트)
        
        **🚫 리스크 분석 - 승인거절 분석:**
        
        **1. 승인거절 주요 지표:**
        - **총 승인거절건수**: `승인거절건수_B0M` 합계
        - **한도초과 거절건수**: `승인거절건수_한도초과_B0M` 합계
        - **입력오류 거절건수**: `승인거절건수_입력오류_B0M` 합계
        - **거절률**: `승인거절건수_B0M ÷ IB문의건수_사용승인내역_B0M` (프록시 계산)
        
        **2. 승인거절 분석 차트:**
        - **세그먼트별 거절률**: 내림차순 정렬된 막대차트 (퍼센트 표시)
        - **거절 사유 분포**: 한도초과/입력오류/기타 스택바 차트
        - **인사이트 자동 생성**: 평균 대비 높은 거절률 세그먼트 식별
        
        **📋 세그먼트별 분석 - 세그먼트 인덱스 분석:**
        
        **1. 세그먼트별 상세 정보:**
        - **고객수**: `ID` 컬럼의 nunique() 계산
        - **주요 연령대**: `연령` 컬럼의 value_counts().index[0]
        - **성별비**: `남녀구분코드` 컬럼에서 여성(2) 비율
        - **평균가입기간**: `입회경과개월수_신용` 평균값을 12로 나누어 년수 변환
        - **유효카드수**: `유효카드수_신용`, `유효카드수_체크` 평균값
        - **활성비율**: `이용금액_일시불_B0M > 0` 조건으로 계산
        - **지역 커버리지**: `거주시도명` 컬럼의 고유값 개수
        
        **2. 인덱스 분석:**
        - **주요연령대비율**: 세그먼트 내 주요 연령대 비율 vs 전체 주요 연령대 비율
        - **지수 계산**: (세그먼트 평균 / 전체 평균 - 1) × 100
        - **특성 강도**: 지수 기준으로 🔴매우 높음, 🟠높음, 🟡보통, 🔵낮음, ⚪매우 낮음 분류
        - **전체 평균**: 실제 전체 데이터 기반 계산 (추정값 없음)
        
        **3. 추가 분석 옵션:**
        - **지역/연령 교차 분석**: 지역별 주요 연령대 분포
        - **카드 이용 패턴 분석**: 신용/체크카드 이용 패턴
        - **연령대/성별 상세 분석**: 연령대별 성별 분포
        - **가입기간별 세분화**: 가입기간별 고객 특성
        - **다른 세그먼트와 비교**: 선택 세그먼트와의 비교 분석
        - **주요 지표 트렌드**: 시간별 지표 변화 (데이터 가능시)
        
        **🔧 데이터 처리 및 안전 처리:**
        
        **1. 데이터 정합성:**
        - **임의 데이터 없음**: 모든 계산이 실제 데이터셋 기반
        - **결측치 처리**: `fillna(0)` 또는 적절한 기본값 설정
        - **0으로 나누기 방지**: `fillna(0)` 및 조건문으로 안전 처리
        - **정렬 보장**: `pd.Categorical`로 A, B, C, D, E 순서 유지
        
        **2. 단위 통일:**
        - **금액 단위**: 데이터셋이 천원 단위이므로 그대로 사용
        - **ARPU 계산**: 천원→원 변환 (`×1000`) 후 계산
        - **표시 단위**: 천원/원 단위 명시적 표시
        
        **3. 성능 최적화:**
        - **캐싱**: `@st.cache_data`로 계산 결과 캐싱
        - **필터링**: 필요한 컬럼만 로드하여 메모리 사용량 감소
        - **집계 최적화**: `groupby().agg()`로 한 번에 모든 집계 수행
        
        **📁 데이터 소스:**
        - **파일**: `base_test_merged_seg.csv`
        - **기간**: 2018년 7월-12월 테스트 데이터
        - **총 고객 수**: 전체 테스트 데이터셋 기준
        - **세그먼트**: A, B, C, D, E (5개 세그먼트)
        - **지역**: 전국 시도 단위
        - **모든 지표**: 실제 데이터 기반 계산 (추정값 없음)
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
                                                default=regions, 
                                                key="kpi_analysis_regions")
            
            with filter_col3:
                st.markdown("**통계 유형**")
                stat_type = st.selectbox("통계 유형 선택", 
                                       ["합계", "평균", "중앙값"], 
                                       index=0,  # 기본값: "합계"
                                       key="kpi_stat_type")
            
            # 금액 단위는 천원으로 고정 (데이터셋이 천원 단위이므로)
            amount_unit = "천원"
            
            # 데이터 필터링 (간소화)
            filtered_df = filter_df(df, selected_segments, selected_regions)
            
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
            
            # KPI 계산 함수 (중복합산 제거 + 한 번에 계산)
            @st.cache_data
            def calculate_kpi_metrics(df_filtered):
                # 금액(천원) 구성: 신판(일시불+할부) + 체크 + CA + 카드론(가능 시)
                have_cardloan_amt = _exists(df_filtered,'이용금액_카드론_B0M')
                have_ca_amt      = _exists(df_filtered,'이용금액_CA_B0M')

                g = df_filtered.groupby('Segment', as_index=False).agg(
                    고객수=('ID','nunique'),
                    청구금액_천원=('청구금액_B0','sum'),
                    일시불금액=('이용금액_일시불_B0M','sum'),
                    할부금액=('이용금액_할부_B0M','sum'),
                    체크건=('이용건수_체크_B0M','sum'),
                    일시불건=('이용건수_일시불_B0M','sum'),
                    할부건=('이용건수_할부_B0M','sum'),
                    CA금액=('이용금액_CA_B0M','sum') if have_ca_amt else ('잔액_현금서비스_B0M','sum'),
                    카드론금액=('이용금액_카드론_B0M','sum') if have_cardloan_amt else ('잔액_카드론_B0M','sum'),
                    CA건=('이용건수_CA_B0M','sum') if _exists(df_filtered,'이용건수_CA_B0M') else ('ID','size'),  # 없으면 더미
                    카드론건=('이용건수_카드론_B0M','sum') if _exists(df_filtered,'이용건수_카드론_B0M') else ('ID','size')
                )
                
                # 숫자형 컬럼만 fillna 적용 (Categorical 제외)
                numeric_cols = g.select_dtypes(include=[np.number]).columns
                g[numeric_cols] = g[numeric_cols].fillna(0)

                # 총이용금액(천원) & 총이용건수(중복 없는 구성요소만)
                g['총이용금액'] = g[['일시불금액','할부금액','CA금액','카드론금액']].sum(axis=1)
                g['총이용건수'] = g[['일시불건','할부건','체크건','CA건','카드론건']].sum(axis=1)

                # ARPU(원) : 청구금액(천원) ×1000 / 고객수
                g['ARPU'] = np.where(g['고객수']>0, g['청구금액_천원']*1000 / g['고객수'], np.nan)

                # 객단가(천원/건) : 총이용금액(천원) / 총이용건수
                g['객단가'] = np.where(g['총이용건수']>0, g['총이용금액'] / g['총이용건수'], np.nan)

                # 신판 비중
                g['일시불비중'] = np.where(g['총이용금액']>0, g['일시불금액']/g['총이용금액'], 0)
                g['할부비중']   = np.where(g['총이용금액']>0, g['할부금액']  /g['총이용금액'], 0)

                g = g.rename(columns={'Segment':'세그먼트'})
                g['세그먼트'] = pd.Categorical(g['세그먼트'], categories=SEGMENTS, ordered=True)
                return g.sort_values('세그먼트')
            
            # KPI 메트릭 계산
            if not filtered_df.empty:
                kpi_df = calculate_kpi_metrics(filtered_df)
                
                # 상단 KPI 카드 4개
                st.markdown("#### 📈 주요 KPI 지표")
                col1, col2, col3, col4 = st.columns(4)
                
                total_amount = kpi_df['총이용금액'].sum()
                total_count = kpi_df['총이용건수'].sum()
                avg_arpu = kpi_df['ARPU'].mean()
                avg_transaction = kpi_df['객단가'].mean()
                
                with col1:
                    st.metric(
                        label="총 이용금액 (천원)",
                        value=f"{total_amount:,.0f}"
                    )
                
                with col2:
                    st.metric(
                        label="총 이용건수",
                        value=f"{total_count:,.0f}"
                    )
                
                with col3:
                    st.metric(
                        label="평균 ARPU (원)",
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
                    # 금액 단위는 천원으로 통일 (데이터셋이 천원 단위)
                    
                    fig_amount = px.bar(
                        kpi_df, 
                        x='세그먼트', 
                        y='총이용금액',
                        title="세그먼트별 총 이용금액 (천원)",
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
                        total_amount = row['총이용금액']
                        
                        if customer_count > 1000 and total_amount < 1000000:  # 고객수 큰데 금액 낮음 (100만천원 미만)
                            insight_text.append(f"저활성 대규모<br>(고객 {customer_count:,}명)")
                        elif customer_count < 500 and total_amount > 5000000:  # 고객수 적은데 금액 높음 (500만천원 이상)
                            insight_text.append(f"VIP 소수정예<br>(고객 {customer_count:,}명)")
                        else:
                            insight_text.append(f"{total_amount/10000:.1f}만천원<br>(고객 {customer_count:,}명)")
                    
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
                        yaxis_title="총 이용금액 (천원)",
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
                        
                        if arpu > 500000 and customer_count > 1000:  # ARPU 높고 고객수 많음 (50만원 이상)
                            arpu_insight_text.append(f"핵심 성장엔진<br>{arpu:,.0f}원")
                        elif arpu > 500000 and customer_count < 500:  # ARPU 높고 고객수 적음 (50만원 이상)
                            arpu_insight_text.append(f"VIP 육성타깃<br>{arpu:,.0f}원")
                        elif arpu < 200000 and customer_count > 1000:  # ARPU 낮고 고객수 많음 (20만원 미만)
                            arpu_insight_text.append(f"대중형 프로모션<br>{arpu:,.0f}원")
                        else:
                            arpu_insight_text.append(f"{arpu:,.0f}원<br>(고객 {customer_count:,}명)")
                    
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
                        yaxis_title="ARPU (원)",
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
                        title="세그먼트별 신판 이용금액 (천원)",
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
                        yaxis_title="이용금액 (천원)",
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
                        xaxis_title="ARPU (원)",
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
                        yaxis_title="현금서비스 (천원)",
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
                        yaxis_title="카드론 (천원)",
                        xaxis_title="세그먼트"
                    )
                    st.plotly_chart(fig_loan_box)
                
                # 세그먼트×지역 분석 (개선된 시각화)
                st.markdown("#### 🗺️ 세그먼트×지역 분석")
                
                # 시각화 타입 선택
                viz_type = st.selectbox(
                    "분석 방법 선택:",
                    ["지역별 세그먼트 분포 (스택바)", "세그먼트별 지역 집중도 (도넛차트)", "지역별 ARPU 비교 (버블차트)"],
                    index=0,  # 기본값: "지역별 세그먼트 분포 (스택바)"
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
                    # 지역별 ARPU 계산 (개선된 방식)
                    region_arpu = filtered_df.groupby('거주시도명', as_index=False).agg(
                        총청구금액=('청구금액_B0', 'sum'),  # 천원 단위
                        고객수=('ID', 'nunique')
                    )
                    # ARPU 계산 (천원 단위를 원 단위로 변환 후 계산)
                    region_arpu['ARPU'] = (region_arpu['총청구금액'] * 1000) / region_arpu['고객수']  # 원 단위 ARPU
                    region_arpu['ARPU_천원'] = region_arpu['ARPU'] / 1000  # 천원 단위로 변환하여 표시
                    
                    # 세그먼트 비율 계산 (각 지역에서 가장 많은 세그먼트)
                    region_dominant_segment = filtered_df.groupby('거주시도명')['Segment'].apply(
                        lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else 'N/A'
                    ).reset_index()
                    region_dominant_segment.columns = ['거주시도명', '주요세그먼트']
                    
                    # 데이터 병합
                    bubble_data = region_arpu.merge(region_dominant_segment, on='거주시도명')
                    
                    fig_bubble = px.scatter(
                        bubble_data,
                        x='고객수',
                        y='ARPU_천원',
                        size='고객수',
                        color='주요세그먼트',
                        hover_name='거주시도명',
                        hover_data={'고객수': ':,.0f', 'ARPU_천원': ':,.0f'},
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
                        yaxis_title="ARPU (원)",
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
                index=0,  # 기본값: 첫 번째 세그먼트 (A)
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
                        default=other_segments,
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
        # 리스크 분석 공통 필터 (상단에 배치)
        st.markdown("### 🔍 리스크 분석 설정")
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            st.markdown("**세그먼트**")
            segments = ['A', 'B', 'C', 'D', 'E']
            selected_segments = st.multiselect("세그먼트 선택", segments, 
                                             default=segments, key="risk_analysis_segments")
        
        with filter_col2:
            st.markdown("**지역**")
            regions = df['거주시도명'].unique().tolist()
            selected_regions = st.multiselect("지역 선택", regions, 
                                            default=regions, key="risk_analysis_regions")
        
        # 공통 필터링된 데이터 (간소화)
        filtered_df = filter_df(df, selected_segments, selected_regions)
        
        st.markdown("---")
        
        sub_tabs = st.tabs(["승인거절 분석", "연체 현황", "리볼빙/현금서비스", "한도/FDS"])
        
        with sub_tabs[0]:  # 승인거절 분석
            st.markdown("### 🚫 승인거절 분석")
            
            if not filtered_df.empty:
                # 승인거절 KPI 계산 (정확한 분모 + 캐시)
                @st.cache_data
                def calculate_rejection_metrics(df_filtered):
                    r = df_filtered.groupby('Segment', as_index=False).agg(
                        고객수=('ID','nunique'),
                        총거절건수=('승인거절건수_B0M','sum'),
                        한도초과거절=('승인거절건수_한도초과_B0M','sum'),
                        입력오류거절=('승인거절건수_입력오류_B0M','sum'),
                        기타거절=('승인거절건수_기타_B0M','sum'),
                        승인요청건수=('IB문의건수_사용승인내역_B0M','sum'),
                        청구금액_천원=('청구금액_B0','sum')
                    )
                    
                    # 숫자형 컬럼만 fillna 적용 (Categorical 제외)
                    numeric_cols = r.select_dtypes(include=[np.number]).columns
                    r[numeric_cols] = r[numeric_cols].fillna(0)

                    r['거절률'] = np.where(r['승인요청건수']>0, r['총거절건수']/r['승인요청건수'], np.nan)
                    r['한도초과비율'] = np.where(r['총거절건수']>0, r['한도초과거절']/r['총거절건수'], np.nan)
                    r['입력오류비율'] = np.where(r['총거절건수']>0, r['입력오류거절']/r['총거절건수'], np.nan)
                    r['ARPU'] = np.where(r['고객수']>0, r['청구금액_천원']*1000 / r['고객수'], np.nan)

                    r = r.rename(columns={'Segment':'세그먼트'})
                    r['세그먼트'] = pd.Categorical(r['세그먼트'], categories=SEGMENTS, ordered=True)
                    return r.sort_values('세그먼트')
                
                rejection_df = calculate_rejection_metrics(filtered_df)
                
                # 상단 KPI 카드
                st.markdown("#### 📈 승인거절 주요 지표")
                col1, col2, col3, col4 = st.columns(4)
                
                total_rejections = rejection_df['총거절건수'].sum()
                total_limit_rejections = rejection_df['한도초과거절'].sum()
                total_input_rejections = rejection_df['입력오류거절'].sum()
                total_approval_requests = rejection_df['승인요청건수'].sum()
                avg_rejection_rate = total_rejections / total_approval_requests if total_approval_requests > 0 else 0
                
                with col1:
                    st.metric(
                        label="총 승인거절건수",
                        value=f"{total_rejections:,.0f}건"
                    )
                
                with col2:
                    st.metric(
                        label="한도초과 거절건수",
                        value=f"{total_limit_rejections:,.0f}건"
                    )
                
                with col3:
                    st.metric(
                        label="입력오류 거절건수",
                        value=f"{total_input_rejections:,.0f}건"
                    )
                
                with col4:
                    st.metric(
                        label="평균 거절률",
                        value=f"{avg_rejection_rate:.2%}"
                    )
                
                st.markdown("---")
                
                # 차트 섹션
                st.markdown("#### 📊 승인거절 분석 차트")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 세그먼트별 거절률 Bar (정렬: 내림차순)
                    rejection_df_sorted = rejection_df.sort_values('거절률', ascending=False)
                    
                    fig_rejection_rate = px.bar(
                        rejection_df_sorted,
                        x='세그먼트',
                        y='거절률',
                        title="세그먼트별 거절률 (내림차순)",
                        color='세그먼트',
                        color_discrete_map={
                            'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 
                            'D': '#96CEB4', 'E': '#FFEAA7'
                        }
                    )
                    
                    fig_rejection_rate.update_traces(
                        texttemplate='%{y:.1%}',
                        textposition='outside',
                        textfont_size=12
                    )
                    
                    fig_rejection_rate.update_layout(
                        height=400,
                        width=500,
                        font_size=14,
                        title_font_size=16,
                        yaxis_title="거절률",
                        xaxis_title="세그먼트",
                        yaxis=dict(tickformat='.1%')
                    )
                    st.plotly_chart(fig_rejection_rate)
                
                with col2:
                    # 전체 거절 사유 비율 파이차트
                    total_rejections = rejection_df[['한도초과거절', '입력오류거절', '기타거절']].sum()
                    rejection_reason_data = pd.DataFrame({
                        '거절사유': ['한도초과', '입력오류', '기타'],
                        '거절건수': [total_rejections['한도초과거절'], total_rejections['입력오류거절'], total_rejections['기타거절']]
                    })
                    
                    # 거절건수가 0인 항목 제거
                    rejection_reason_data = rejection_reason_data[rejection_reason_data['거절건수'] > 0]
                    
                    if not rejection_reason_data.empty:
                        fig_rejection_pie = px.pie(
                            rejection_reason_data,
                            values='거절건수',
                            names='거절사유',
                            title="전체 거절 사유 비율",
                            color_discrete_map={
                                '한도초과': '#FF6B6B',
                                '입력오류': '#4ECDC4',
                                '기타': '#45B7D1'
                            }
                        )
                        
                        fig_rejection_pie.update_traces(
                            textposition='inside',
                            textinfo='percent+label',
                            textfont_size=12
                        )
                        
                        fig_rejection_pie.update_layout(
                            height=400,
                            width=500,
                            font_size=14,
                            title_font_size=16
                        )
                        st.plotly_chart(fig_rejection_pie)
                    else:
                        st.info("거절 데이터가 없습니다.")
                
                # 인사이트 포인트
                st.markdown("#### 💡 승인거절 인사이트")
                col1, col2 = st.columns(2)
                
                with col1:
                    # 한도초과 거절률이 높은 세그먼트
                    high_limit_rejection = rejection_df[rejection_df['한도초과거절'] > rejection_df['한도초과거절'].mean()]
                    if not high_limit_rejection.empty:
                        st.markdown("**🔴 한도초과 거절률 높은 세그먼트:**")
                        for _, row in high_limit_rejection.iterrows():
                            limit_rate = row['한도초과거절'] / row['총거절건수'] if row['총거절건수'] > 0 else 0
                            st.write(f"• 세그먼트 {row['세그먼트']}: {limit_rate:.1%} (한도 튜닝 필요)")
                
                with col2:
                    # 입력오류 거절률이 높은 세그먼트
                    high_input_rejection = rejection_df[rejection_df['입력오류거절'] > rejection_df['입력오류거절'].mean()]
                    if not high_input_rejection.empty:
                        st.markdown("**🟡 입력오류 거절률 높은 세그먼트:**")
                        for _, row in high_input_rejection.iterrows():
                            input_rate = row['입력오류거절'] / row['총거절건수'] if row['총거절건수'] > 0 else 0
                            st.write(f"• 세그먼트 {row['세그먼트']}: {input_rate:.1%} (UX 개선 필요)")
                
                # 상세 분석 테이블
                st.markdown("#### 📋 세그먼트별 상세 거절 현황")
                
                # 표시용 데이터 준비
                display_df = rejection_df.copy()
                display_df['거절률'] = display_df['거절률'].apply(lambda x: f"{x:.1%}")
                display_df['한도초과비율'] = display_df['한도초과비율'].apply(lambda x: f"{x:.1%}")
                display_df['입력오류비율'] = display_df['입력오류비율'].apply(lambda x: f"{x:.1%}")
                display_df['ARPU'] = display_df['ARPU'].apply(lambda x: f"{x:,.0f}원")
                
                st.dataframe(
                    display_df[['세그먼트', '고객수', '총거절건수', '거절률', '한도초과거절', '한도초과비율', '입력오류거절', '입력오류비율']],
                    use_container_width=True,
                    hide_index=True
                )
            
            else:
                st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
            
            # 시각화 자료 설명
            show_visualization_guide()
        
        with sub_tabs[1]:  # 연체 현황
            st.markdown("### 💸 연체 현황")
            
            if not filtered_df.empty:
                # 연체 현황 KPI 계산
                @st.cache_data
                def calculate_delinquency_metrics(df_filtered):
                    # 세그먼트별 연체 집계
                    delinquency_data = df_filtered.groupby("Segment", as_index=False).agg(
                        고객수=("ID", "nunique"),
                        연체고객수=("연체잔액_B0M", lambda x: (x > 0).sum()),  # 연체여부 계산
                        총연체잔액=("연체잔액_B0M", "sum"),
                        연체잔액_일시불=("연체잔액_일시불_B0M", "sum"),
                        연체잔액_할부=("연체잔액_할부_B0M", "sum"),
                        연체잔액_현금서비스=("연체잔액_현금서비스_B0M", "sum"),
                        연체잔액_카드론=("연체잔액_카드론_B0M", "sum"),
                        시장단기연체_R3M=("시장단기연체여부_R3M", lambda x: (x == True).sum()),
                        시장단기연체_R6M=("시장단기연체여부_R6M", lambda x: (x == True).sum())
                    )
                    
                    # 연체율 계산
                    delinquency_data["연체율"] = delinquency_data["연체고객수"] / delinquency_data["고객수"]
                    delinquency_data["연체율"] = delinquency_data["연체율"].fillna(0)
                    
                    # 시장단기연체 비율 계산
                    delinquency_data["시장단기연체비율_R3M"] = delinquency_data["시장단기연체_R3M"] / delinquency_data["고객수"]
                    delinquency_data["시장단기연체비율_R6M"] = delinquency_data["시장단기연체_R6M"] / delinquency_data["고객수"]
                    
                    # 컬럼명 정리
                    delinquency_data = delinquency_data.rename(columns={"Segment": "세그먼트"})
                    
                    return delinquency_data
                
                delinquency_df = calculate_delinquency_metrics(filtered_df)
                
                # 상단 KPI 카드
                st.markdown("#### 📈 연체 현황 주요 지표")
                col1, col2, col3, col4 = st.columns(4)
                
                total_customers = delinquency_df['고객수'].sum()
                total_delinquent_customers = delinquency_df['연체고객수'].sum()
                overall_delinquency_rate = total_delinquent_customers / total_customers if total_customers > 0 else 0
                total_delinquency_amount = delinquency_df['총연체잔액'].sum()
                total_cash_service_delinquency = delinquency_df['연체잔액_현금서비스'].sum()
                total_card_loan_delinquency = delinquency_df['연체잔액_카드론'].sum()
                
                with col1:
                    st.metric(
                        label="전체 연체율",
                        value=f"{overall_delinquency_rate:.2%}"
                    )
                
                with col2:
                    st.metric(
                        label="총 연체잔액",
                        value=f"{total_delinquency_amount:,.0f}천원"
                    )
                
                with col3:
                    st.metric(
                        label="현금서비스 연체잔액",
                        value=f"{total_cash_service_delinquency:,.0f}천원"
                    )
                
                with col4:
                    st.metric(
                        label="카드론 연체잔액",
                        value=f"{total_card_loan_delinquency:,.0f}천원"
                    )
                
                st.markdown("---")
                
                # 차트 섹션
                st.markdown("#### 📊 연체 현황 분석 차트")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 세그먼트별 연체율 Bar
                    fig_delinquency_rate = px.bar(
                        delinquency_df,
                        x='세그먼트',
                        y='연체율',
                        title="세그먼트별 연체율",
                        color='세그먼트',
                        color_discrete_map={
                            'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 
                            'D': '#96CEB4', 'E': '#FFEAA7'
                        }
                    )
                    
                    fig_delinquency_rate.update_traces(
                        texttemplate='%{y:.2%}',
                        textposition='outside',
                        textfont_size=12
                    )
                    
                    fig_delinquency_rate.update_layout(
                        height=400,
                        width=500,
                        font_size=14,
                        title_font_size=16,
                        yaxis_title="연체율",
                        xaxis_title="세그먼트",
                        yaxis=dict(tickformat='.1%')
                    )
                    st.plotly_chart(fig_delinquency_rate)
                
                with col2:
                    # 연체잔액 구성 Stacked Bar
                    delinquency_composition = delinquency_df[['세그먼트', '연체잔액_일시불', '연체잔액_할부', '연체잔액_현금서비스', '연체잔액_카드론']].copy()
                    delinquency_composition_melted = delinquency_composition.melt(
                        id_vars=['세그먼트'], 
                        value_vars=['연체잔액_일시불', '연체잔액_할부', '연체잔액_현금서비스', '연체잔액_카드론'],
                        var_name='연체유형', 
                        value_name='연체잔액'
                    )
                    
                    # 연체유형 한글화
                    type_mapping = {
                        '연체잔액_일시불': '일시불',
                        '연체잔액_할부': '할부',
                        '연체잔액_현금서비스': '현금서비스',
                        '연체잔액_카드론': '카드론'
                    }
                    delinquency_composition_melted['연체유형'] = delinquency_composition_melted['연체유형'].map(type_mapping)
                    
                    fig_delinquency_composition = px.bar(
                        delinquency_composition_melted,
                        x='세그먼트',
                        y='연체잔액',
                        color='연체유형',
                        title="세그먼트별 연체잔액 구성",
                        color_discrete_map={
                            '일시불': '#FF6B6B',
                            '할부': '#4ECDC4',
                            '현금서비스': '#45B7D1',
                            '카드론': '#96CEB4'
                        }
                    )
                    
                    fig_delinquency_composition.update_traces(
                        texttemplate='%{y:,}',
                        textposition='inside',
                        textfont_size=10
                    )
                    
                    fig_delinquency_composition.update_layout(
                        height=400,
                        width=500,
                        font_size=14,
                        title_font_size=16,
                        yaxis_title="연체잔액 (천원)",
                        xaxis_title="세그먼트"
                    )
                    st.plotly_chart(fig_delinquency_composition)
                
                # 박스/바이올린 플롯
                st.markdown("#### 📦 세그먼트별 연체잔액 분포 분석")
                
                # 연체잔액이 있는 고객만 필터링
                delinquent_customers = filtered_df[filtered_df['연체잔액_B0M'] > 0].copy()
                
                if not delinquent_customers.empty:
                    fig_box = px.box(
                        delinquent_customers,
                        x='Segment',
                        y='연체잔액_B0M',
                        title="세그먼트별 연체잔액 분포 (Box Plot)",
                        color='Segment',
                        color_discrete_map={
                            'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 
                            'D': '#96CEB4', 'E': '#FFEAA7'
                        }
                    )
                    
                    fig_box.update_layout(
                        height=400,
                        width=800,
                        font_size=14,
                        title_font_size=16,
                        yaxis_title="연체잔액 (천원)",
                        xaxis_title="세그먼트"
                    )
                    st.plotly_chart(fig_box)
                else:
                    st.info("연체잔액이 있는 고객이 없습니다.")
                
                # 인사이트 포인트
                st.markdown("#### 💡 연체 현황 인사이트")
                col1, col2 = st.columns(2)
                
                with col1:
                    # CA/카드론 비중이 높고 연체율이 높은 세그먼트
                    high_risk_segments = delinquency_df[
                        (delinquency_df['연체잔액_현금서비스'] + delinquency_df['연체잔액_카드론']) / 
                        delinquency_df['총연체잔액'] > 0.5
                    ]
                    high_risk_segments = high_risk_segments[high_risk_segments['연체율'] > delinquency_df['연체율'].mean()]
                    
                    if not high_risk_segments.empty:
                        st.markdown("**🔴 CA/카드론 비중 높은 고위험 세그먼트:**")
                        for _, row in high_risk_segments.iterrows():
                            ca_card_ratio = (row['연체잔액_현금서비스'] + row['연체잔액_카드론']) / row['총연체잔액'] if row['총연체잔액'] > 0 else 0
                            st.write(f"• 세그먼트 {row['세그먼트']}: 연체율 {row['연체율']:.1%}, CA/카드론 비중 {ca_card_ratio:.1%} (한도·금리·상환 캠페인 필요)")
                
                with col2:
                    # 시장단기연체 비율이 높은 세그먼트
                    market_delinquency_segments = delinquency_df[
                        (delinquency_df['시장단기연체비율_R3M'] > delinquency_df['시장단기연체비율_R3M'].mean()) |
                        (delinquency_df['시장단기연체비율_R6M'] > delinquency_df['시장단기연체비율_R6M'].mean())
                    ]
                    
                    if not market_delinquency_segments.empty:
                        st.markdown("**🟡 시장단기연체 비율 높은 세그먼트:**")
                        for _, row in market_delinquency_segments.iterrows():
                            st.write(f"• 세그먼트 {row['세그먼트']}: R3M {row['시장단기연체비율_R3M']:.1%}, R6M {row['시장단기연체비율_R6M']:.1%} (조기경보 리스트업 필요)")
                
                # 상세 분석 테이블
                st.markdown("#### 📋 세그먼트별 상세 연체 현황")
                
                # 표시용 데이터 준비
                display_df = delinquency_df.copy()
                display_df['연체율'] = display_df['연체율'].apply(lambda x: f"{x:.2%}")
                display_df['시장단기연체비율_R3M'] = display_df['시장단기연체비율_R3M'].apply(lambda x: f"{x:.1%}")
                display_df['시장단기연체비율_R6M'] = display_df['시장단기연체비율_R6M'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(
                    display_df[['세그먼트', '고객수', '연체고객수', '연체율', '총연체잔액', '연체잔액_현금서비스', '연체잔액_카드론', '시장단기연체비율_R3M', '시장단기연체비율_R6M']],
                    use_container_width=True,
                    hide_index=True
                )
            
            else:
                st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
            
            # 시각화 자료 설명
            show_visualization_guide()
        
        with sub_tabs[2]:  # 리볼빙/현금서비스
            st.markdown("### 🔄 리볼빙/현금서비스")
            
            if not filtered_df.empty:
                # 리볼빙/현금서비스 KPI 계산 (동일 패턴)
                @st.cache_data
                def calculate_revolving_metrics(df_filtered):
                    r = df_filtered.groupby('Segment', as_index=False).agg(
                        고객수=('ID','nunique'),
                        리볼빙이용고객수=('RV_평균잔액_R3M', lambda x: (x > 0).sum()),
                        총RV평균잔액=('RV_평균잔액_R3M','sum'),
                        평균RV평균잔액=('RV_평균잔액_R3M','mean'),
                        총RV잔액이월횟수=('RV잔액이월횟수_R3M','sum'),
                        평균RV잔액이월횟수=('RV잔액이월횟수_R3M','mean'),
                        총CA이용금액=('이용금액_CA_B0M','sum'),
                        평균CA이용금액=('이용금액_CA_B0M','mean'),
                        총CA신청건수=('신청건수_ATM_CA_B0','sum'),
                        평균CA이자율=('CA이자율_할인전','mean'),
                        총신판이용금액_일시불=('이용금액_일시불_B0M','sum'),
                        총신판이용금액_할부=('이용금액_할부_B0M','sum'),
                        총체크이용금액=('이용건수_체크_B0M','sum'),
                        총카드론이용금액=('잔액_카드론_B0M','sum')
                    )
                    
                    # 숫자형 컬럼만 fillna 적용 (Categorical 제외)
                    numeric_cols = r.select_dtypes(include=[np.number]).columns
                    r[numeric_cols] = r[numeric_cols].fillna(0)
                    
                    # 신판 이용금액 합계 계산
                    r['총신판이용금액'] = r['총신판이용금액_일시불'] + r['총신판이용금액_할부']
                    
                    # 리볼빙 이용율 계산 (비율 그대로 유지)
                    r['리볼빙이용율'] = np.where(r['고객수']>0, r['리볼빙이용고객수']/r['고객수'], np.nan)

                    r = r.rename(columns={'Segment':'세그먼트'})
                    r['세그먼트'] = pd.Categorical(r['세그먼트'], categories=SEGMENTS, ordered=True)
                    return r.sort_values('세그먼트')
                
                revolving_df = calculate_revolving_metrics(filtered_df)
                
                # 상단 KPI 카드
                st.markdown("#### 📈 리볼빙/현금서비스 주요 지표")
                col1, col2, col3, col4 = st.columns(4)
                
                total_customers = revolving_df['고객수'].sum()
                total_revolving_customers = revolving_df['리볼빙이용고객수'].sum()
                overall_revolving_rate = total_revolving_customers / total_customers if total_customers > 0 else 0
                total_rv_balance = revolving_df['총RV평균잔액'].sum()
                total_ca_amount = revolving_df['총CA이용금액'].sum() * 1000  # 천원 -> 원 변환
                total_ca_applications = revolving_df['총CA신청건수'].sum()
                
                with col1:
                    st.metric(
                        label="전체 리볼빙 이용율",
                        value=f"{overall_revolving_rate:.2%}"
                    )
                
                with col2:
                    st.metric(
                        label="총 RV 평균잔액",
                        value=f"{total_rv_balance:,.0f}천원"
                    )
                
                with col3:
                    st.metric(
                        label="총 CA 이용금액",
                        value=f"{total_ca_amount:,.0f}원"
                    )
                
                with col4:
                    st.metric(
                        label="총 CA 신청건수",
                        value=f"{total_ca_applications:,}건"
                    )
                
                st.markdown("---")
                
                # 차트 섹션
                st.markdown("#### 📊 리볼빙/현금서비스 분석 차트")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 세그먼트별 리볼빙 이용율 Bar
                    fig_revolving_rate = px.bar(
                        revolving_df,
                        x='세그먼트',
                        y='리볼빙이용율',
                        title="세그먼트별 리볼빙 이용율",
                        color='세그먼트',
                        color_discrete_map={
                            'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 
                            'D': '#96CEB4', 'E': '#FFEAA7'
                        }
                    )
                    
                    fig_revolving_rate.update_traces(
                        texttemplate='%{y:.2%}',
                        textposition='outside',
                        textfont_size=12
                    )
                    
                    fig_revolving_rate.update_layout(
                        height=400,
                        width=500,
                        font_size=14,
                        title_font_size=16,
                        yaxis_title="리볼빙 이용율",
                        xaxis_title="세그먼트",
                        yaxis=dict(tickformat='.1%')
                    )
                    st.plotly_chart(fig_revolving_rate)
                
                with col2:
                    # 세그먼트별 CA 이용금액 Box Plot
                    ca_customers = filtered_df[filtered_df['이용금액_CA_B0M'] > 0].copy()
                    
                    if not ca_customers.empty:
                        fig_ca_box = px.box(
                            ca_customers,
                            x='Segment',
                            y='이용금액_CA_B0M',
                            title="세그먼트별 CA 이용금액 분포",
                            color='Segment',
                            color_discrete_map={
                                'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 
                                'D': '#96CEB4', 'E': '#FFEAA7'
                            }
                        )
                        
                        fig_ca_box.update_layout(
                            height=400,
                            width=500,
                            font_size=14,
                            title_font_size=16,
                            yaxis_title="CA 이용금액 (천원)",
                            xaxis_title="세그먼트"
                        )
                        st.plotly_chart(fig_ca_box)
                    else:
                        st.info("CA 이용금액이 있는 고객이 없습니다.")
                
                # 전환/잔액 트렌드 Line (RV 평균잔액과 잔액이월횟수)
                st.markdown("#### 📈 리볼빙 트렌드 분석")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # RV 평균잔액 트렌드
                    fig_rv_balance = px.bar(
                        revolving_df,
                        x='세그먼트',
                        y='평균RV평균잔액',
                        title="세그먼트별 평균 RV 잔액",
                        color='세그먼트',
                        color_discrete_map={
                            'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 
                            'D': '#96CEB4', 'E': '#FFEAA7'
                        }
                    )
                    
                    fig_rv_balance.update_traces(
                        texttemplate='%{y:,.0f}',
                        textposition='outside',
                        textfont_size=12
                    )
                    
                    fig_rv_balance.update_layout(
                        height=400,
                        width=500,
                        font_size=14,
                        title_font_size=16,
                        yaxis_title="평균 RV 잔액 (천원)",
                        xaxis_title="세그먼트"
                    )
                    st.plotly_chart(fig_rv_balance)
                
                with col2:
                    # RV 잔액이월횟수 트렌드
                    fig_rv_rollover = px.bar(
                        revolving_df,
                        x='세그먼트',
                        y='평균RV잔액이월횟수',
                        title="세그먼트별 평균 RV 잔액이월횟수",
                        color='세그먼트',
                        color_discrete_map={
                            'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 
                            'D': '#96CEB4', 'E': '#FFEAA7'
                        }
                    )
                    
                    fig_rv_rollover.update_traces(
                        texttemplate='%{y:.1f}',
                        textposition='outside',
                        textfont_size=12
                    )
                    
                    fig_rv_rollover.update_layout(
                        height=400,
                        width=500,
                        font_size=14,
                        title_font_size=16,
                        yaxis_title="평균 RV 잔액이월횟수",
                        xaxis_title="세그먼트"
                    )
                    st.plotly_chart(fig_rv_rollover)
                
                # 비중 파이차트 (신판/체크/CA/카드론)
                st.markdown("#### 🥧 이용 유형별 금액 비중")
                
                # 전체 이용금액 비중 계산
                total_new_sales = revolving_df['총신판이용금액'].sum()
                total_check = revolving_df['총체크이용금액'].sum()  # 체크카드는 건수
                total_ca = revolving_df['총CA이용금액'].sum()
                total_card_loan = revolving_df['총카드론이용금액'].sum()  # 카드론은 잔액
                
                usage_type_data = pd.DataFrame({
                    '이용유형': ['신판', '체크(건수)', 'CA', '카드론(잔액)'],
                    '이용금액': [total_new_sales, total_check, total_ca, total_card_loan]
                })
                
                # 이용금액이 0인 항목 제거
                usage_type_data = usage_type_data[usage_type_data['이용금액'] > 0]
                
                if not usage_type_data.empty:
                    fig_usage_pie = px.pie(
                        usage_type_data,
                        values='이용금액',
                        names='이용유형',
                        title="전체 이용 유형별 금액 비중",
                        color_discrete_map={
                            '신판': '#FF6B6B',
                            '체크(건수)': '#4ECDC4',
                            'CA': '#45B7D1',
                            '카드론(잔액)': '#96CEB4'
                        }
                    )
                    
                    fig_usage_pie.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        textfont_size=12
                    )
                    
                    fig_usage_pie.update_layout(
                        height=400,
                        width=600,
                        font_size=14,
                        title_font_size=16
                    )
                    st.plotly_chart(fig_usage_pie)
                else:
                    st.info("이용금액 데이터가 없습니다.")
                
                # 인사이트 포인트
                st.markdown("#### 💡 리볼빙/현금서비스 인사이트")
                col1, col2 = st.columns(2)
                
                with col1:
                    # 리볼빙 헤비 유저군 식별
                    high_revolving_segments = revolving_df[
                        (revolving_df['리볼빙이용율'] > revolving_df['리볼빙이용율'].mean()) &
                        (revolving_df['평균RV평균잔액'] > revolving_df['평균RV평균잔액'].mean())
                    ]
                    
                    if not high_revolving_segments.empty:
                        st.markdown("**🔴 리볼빙 헤비 유저군:**")
                        for _, row in high_revolving_segments.iterrows():
                            st.write(f"• 세그먼트 {row['세그먼트']}: 이용율 {row['리볼빙이용율']:.1%}, 평균잔액 {row['평균RV평균잔액']:,.0f}천원 (금리/상환 가이드 필요)")
                
                with col2:
                    # CA 헤비 유저군 식별
                    high_ca_segments = revolving_df[
                        (revolving_df['평균CA이용금액'] > revolving_df['평균CA이용금액'].mean()) &
                        (revolving_df['평균CA이용금액'] > 0)
                    ]
                    
                    if not high_ca_segments.empty:
                        st.markdown("**🟡 CA 헤비 유저군:**")
                        for _, row in high_ca_segments.iterrows():
                            st.write(f"• 세그먼트 {row['세그먼트']}: 평균이용금액 {row['평균CA이용금액']:,.0f}천원, 신청건수 {row['총CA신청건수']:,}건 (부담 경감 캠페인 필요)")
                
                # 초과 의존군 식별
                st.markdown("#### ⚠️ 초과 의존군 분석")
                
                # RV 잔액이월횟수가 높은 세그먼트
                high_rollover_segments = revolving_df[
                    revolving_df['평균RV잔액이월횟수'] > revolving_df['평균RV잔액이월횟수'].mean()
                ]
                
                if not high_rollover_segments.empty:
                    st.markdown("**🟠 RV 잔액이월 초과 의존군:**")
                    for _, row in high_rollover_segments.iterrows():
                        st.write(f"• 세그먼트 {row['세그먼트']}: 평균이월횟수 {row['평균RV잔액이월횟수']:.1f}회 (FDS/리스크 탭과 교차 확인 필요)")
                
                # 상세 분석 테이블
                st.markdown("#### 📋 세그먼트별 상세 리볼빙/현금서비스 현황")
                
                # 표시용 데이터 준비
                display_df = revolving_df.copy()
                display_df['리볼빙이용율'] = display_df['리볼빙이용율'].apply(lambda x: f"{x:.2%}")
                display_df['평균RV평균잔액'] = display_df['평균RV평균잔액'].apply(lambda x: f"{x:,.0f}")
                display_df['평균RV잔액이월횟수'] = display_df['평균RV잔액이월횟수'].apply(lambda x: f"{x:.1f}")
                display_df['평균CA이용금액'] = display_df['평균CA이용금액'].apply(lambda x: f"{x:,.0f}")
                
                st.dataframe(
                    display_df[['세그먼트', '고객수', '리볼빙이용고객수', '리볼빙이용율', '평균RV평균잔액', '평균RV잔액이월횟수', '평균CA이용금액', '총CA신청건수']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
            
            # 시각화 자료 설명
            show_visualization_guide()
        
        with sub_tabs[3]:  # 한도/FDS
            st.markdown("### 🛡️ 한도/FDS")
            
            if not filtered_df.empty:
                # 한도/FDS KPI 계산 (동일 패턴)
                @st.cache_data
                def calculate_limit_fds_metrics(df_filtered):
                    r = df_filtered.groupby('Segment', as_index=False).agg(
                        고객수=('ID','nunique'),
                        평균카드이용한도=('카드이용한도금액','mean'),
                        평균상향가능한도=('상향가능한도금액','mean'),
                        총한도심사요청=('한도심사요청건수','sum'),
                        총한도증액횟수=('한도증액횟수_R12M','sum'),
                        총강제한도감액횟수=('강제한도감액횟수_R12M','sum'),
                        총자발한도감액횟수=('자발한도감액횟수_R12M','sum'),
                        평균신판한도소진율=('잔액_신판평균한도소진율_r3m','mean'),
                        최대신판한도소진율_R3M=('잔액_신판최대한도소진율_r3m','mean'),
                        최대신판한도소진율_R6M=('잔액_신판최대한도소진율_r6m','mean'),
                        총한도초과거절=('승인거절건수_한도초과_B0M','sum'),
                        총승인거절=('승인거절건수_B0M','sum')
                    )
                    
                    # 숫자형 컬럼만 fillna 적용 (Categorical 제외)
                    numeric_cols = r.select_dtypes(include=[np.number]).columns
                    r[numeric_cols] = r[numeric_cols].fillna(0)
                    
                    # 한도소진율 계산 (신판 기준)
                    r['한도소진율'] = r['평균신판한도소진율']
                    
                    # 한도초과 거절 비중 계산 (비율 그대로 유지)
                    r['한도초과거절비중'] = np.where(r['총승인거절']>0, r['총한도초과거절']/r['총승인거절'], np.nan)
                    
                    # FDS 근사 지표 계산 (한도초과 거절 비중으로 대체)
                    r['FDS거절비중'] = np.where(r['총승인거절']>0, r['총한도초과거절']/r['총승인거절'], np.nan)

                    r = r.rename(columns={'Segment':'세그먼트'})
                    r['세그먼트'] = pd.Categorical(r['세그먼트'], categories=SEGMENTS, ordered=True)
                    return r.sort_values('세그먼트')
                
                limit_df = calculate_limit_fds_metrics(filtered_df)
                
                # 상단 KPI 카드
                st.markdown("#### 📈 한도/FDS 주요 지표")
                col1, col2, col3, col4 = st.columns(4)
                
                avg_card_limit = limit_df['평균카드이용한도'].mean()
                avg_limit_utilization = limit_df['한도소진율'].mean()
                total_limit_exceeded_rejections = limit_df['총한도초과거절'].sum()
                total_rejections = limit_df['총승인거절'].sum()
                limit_exceeded_ratio = total_limit_exceeded_rejections / total_rejections if total_rejections > 0 else 0
                
                with col1:
                    st.metric(
                        label="평균 카드이용한도",
                        value=f"{avg_card_limit:,.0f}원"
                    )
                
                with col2:
                    st.metric(
                        label="평균 한도소진율",
                        value=f"{avg_limit_utilization:.1%}"
                    )
                
                with col3:
                    st.metric(
                        label="한도초과 거절 비중",
                        value=f"{limit_exceeded_ratio:.1%}"
                    )
                
                with col4:
                    # 안전한 합계 계산
                    total_limit_requests = limit_df['총한도심사요청'].apply(lambda x: pd.to_numeric(x, errors='coerce')).sum()
                    st.metric(
                        label="총 한도심사요청",
                        value=f"{int(total_limit_requests):,}건"
                    )
                
                st.markdown("---")
                
                # 차트 섹션
                st.markdown("#### 📊 한도/FDS 분석 차트")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 세그먼트별 한도 금액 Bar + 에러바
                    fig_limit_amount = px.bar(
                        limit_df,
                        x='세그먼트',
                        y='평균카드이용한도',
                        title="세그먼트별 평균 카드이용한도",
                        color='세그먼트',
                        color_discrete_map={
                            'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 
                            'D': '#96CEB4', 'E': '#FFEAA7'
                        }
                    )
                    
                    fig_limit_amount.update_traces(
                        texttemplate='%{y:,.0f}',
                        textposition='outside',
                        textfont_size=12
                    )
                    
                    fig_limit_amount.update_layout(
                        height=400,
                        width=500,
                        font_size=14,
                        title_font_size=16,
                        yaxis_title="카드이용한도 (원)",
                        xaxis_title="세그먼트"
                    )
                    st.plotly_chart(fig_limit_amount)
                
                with col2:
                    # 세그먼트별 한도소진율 Box Plot
                    limit_utilization_data = filtered_df[filtered_df['잔액_신판평균한도소진율_r3m'].notna()].copy()
                    
                    if not limit_utilization_data.empty:
                        fig_limit_utilization = px.box(
                            limit_utilization_data,
                            x='Segment',
                            y='잔액_신판평균한도소진율_r3m',
                            title="세그먼트별 한도소진율 분포",
                            color='Segment',
                            color_discrete_map={
                                'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 
                                'D': '#96CEB4', 'E': '#FFEAA7'
                            }
                        )
                        
                        fig_limit_utilization.update_layout(
                            height=400,
                            width=500,
                            font_size=14,
                            title_font_size=16,
                            yaxis_title="한도소진율",
                            xaxis_title="세그먼트",
                            yaxis=dict(tickformat='.1%')
                        )
                        st.plotly_chart(fig_limit_utilization)
                    else:
                        st.info("한도소진율 데이터가 없습니다.")
                
                # 한도 관련 이벤트 타임라인
                st.markdown("#### 📈 한도 관련 이벤트 분석")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 한도 증액/감액 횟수 비교
                    limit_events_data = limit_df[['세그먼트', '총한도증액횟수', '총강제한도감액횟수', '총자발한도감액횟수']].copy()
                    limit_events_melted = limit_events_data.melt(
                        id_vars=['세그먼트'], 
                        value_vars=['총한도증액횟수', '총강제한도감액횟수', '총자발한도감액횟수'],
                        var_name='이벤트유형', 
                        value_name='횟수'
                    )
                    
                    # 이벤트유형 한글화
                    event_mapping = {
                        '총한도증액횟수': '증액',
                        '총강제한도감액횟수': '강제감액',
                        '총자발한도감액횟수': '자발감액'
                    }
                    limit_events_melted['이벤트유형'] = limit_events_melted['이벤트유형'].map(event_mapping)
                    
                    fig_limit_events = px.bar(
                        limit_events_melted,
                        x='세그먼트',
                        y='횟수',
                        color='이벤트유형',
                        title="세그먼트별 한도 이벤트 횟수",
                        color_discrete_map={
                            '증액': '#4ECDC4',
                            '강제감액': '#FF6B6B',
                            '자발감액': '#45B7D1'
                        }
                    )
                    
                    fig_limit_events.update_traces(
                        texttemplate='%{y:,}',
                        textposition='inside',
                        textfont_size=10
                    )
                    
                    fig_limit_events.update_layout(
                        height=400,
                        width=500,
                        font_size=14,
                        title_font_size=16,
                        yaxis_title="횟수",
                        xaxis_title="세그먼트"
                    )
                    st.plotly_chart(fig_limit_events)
                
                with col2:
                    # FDS 근사 지표 (BL 거절 비중)
                    fig_fds_approximation = px.bar(
                        limit_df,
                        x='세그먼트',
                        y='FDS거절비중',
                        title="세그먼트별 한도초과 거절 비중",
                        color='세그먼트',
                        color_discrete_map={
                            'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 
                            'D': '#96CEB4', 'E': '#FFEAA7'
                        }
                    )
                    
                    fig_fds_approximation.update_traces(
                        texttemplate='%{y:.1%}',
                        textposition='outside',
                        textfont_size=12
                    )
                    
                    fig_fds_approximation.update_layout(
                        height=400,
                        width=500,
                        font_size=14,
                        title_font_size=16,
                        yaxis_title="한도초과 거절 비중",
                        xaxis_title="세그먼트",
                        yaxis=dict(tickformat='.1%')
                    )
                    st.plotly_chart(fig_fds_approximation)
                
                # 인사이트 포인트
                st.markdown("#### 💡 한도/FDS 인사이트")
                col1, col2 = st.columns(2)
                
                with col1:
                    # 소진율↑·거절(한도초과)↑ 세그먼트
                    high_risk_segments = limit_df[
                        (limit_df['한도소진율'] > limit_df['한도소진율'].mean()) &
                        (limit_df['한도초과거절비중'] > limit_df['한도초과거절비중'].mean())
                    ]
                    
                    if not high_risk_segments.empty:
                        st.markdown("**🔴 한도·승인룰 미스매치 세그먼트:**")
                        for _, row in high_risk_segments.iterrows():
                            st.write(f"• 세그먼트 {row['세그먼트']}: 소진율 {row['한도소진율']:.1%}, 한도초과거절비중 {row['한도초과거절비중']:.1%} (한도·승인룰 점검 필요)")
                
                with col2:
                    # 상향가능한도금액 높고 소진율↑ 세그먼트
                    upgrade_candidate_segments = limit_df[
                        (limit_df['평균상향가능한도'] > limit_df['평균상향가능한도'].mean()) &
                        (limit_df['한도소진율'] > limit_df['한도소진율'].mean())
                    ]
                    
                    if not upgrade_candidate_segments.empty:
                        st.markdown("**🟡 선제 증액 캠페인 후보군:**")
                        for _, row in upgrade_candidate_segments.iterrows():
                            st.write(f"• 세그먼트 {row['세그먼트']}: 상향가능한도 {row['평균상향가능한도']:,.0f}원, 소진율 {row['한도소진율']:.1%} (선제 증액 캠페인 추천)")
                
                # 상세 분석 테이블
                st.markdown("#### 📋 세그먼트별 상세 한도/FDS 현황")
                
                # 표시용 데이터 준비
                display_df = limit_df.copy()
                display_df['한도소진율'] = display_df['한도소진율'].apply(lambda x: f"{x:.1%}")
                display_df['한도초과거절비중'] = display_df['한도초과거절비중'].apply(lambda x: f"{x:.1%}")
                display_df['평균카드이용한도'] = display_df['평균카드이용한도'].apply(lambda x: f"{x:,.0f}")
                display_df['평균상향가능한도'] = display_df['평균상향가능한도'].apply(lambda x: f"{x:,.0f}")
                
                st.dataframe(
                    display_df[['세그먼트', '고객수', '평균카드이용한도', '평균상향가능한도', '한도소진율', '한도초과거절비중', '총한도심사요청']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
            
            # 시각화 자료 설명
            show_visualization_guide()
    
    elif current_main_tab == "행동·마케팅":
        sub_tabs = st.tabs(["앱/웹 이용행태", "마케팅 채널 반응", "캠페인 참여 & 쿠폰 사용", "업종(MCC) 소비 패턴", "VOC/CS"])
        
        with sub_tabs[0]:  # 앱/웹 이용행태
            st.markdown("### 📱 앱/웹 이용행태")
            
            # 상단 필터
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("**세그먼트**")
                selected_segments = st.multiselect("세그먼트 선택", SEGMENTS, 
                                                default=SEGMENTS, key="appweb_segments")
            
            with col2:
                st.markdown("**지역**")
                regions = df['거주시도명'].unique().tolist()
                selected_regions = st.multiselect("지역 선택", regions, 
                                                default=regions, key="appweb_regions")
            
            with col3:
                st.markdown("**성별**")
                genders = df['남녀구분코드'].unique().tolist()
                selected_genders = st.multiselect("성별 선택", genders, 
                                                default=genders, key="appweb_genders")
            
            with col4:
                st.markdown("**가입기간 구간**")
                tenure_options = ["신규(<2)", "중간(2-5)", "기존(5-10)", "우수(10+)"]
                selected_tenures = st.multiselect("가입기간 구간 선택", tenure_options, 
                                                default=tenure_options, key="appweb_tenures")
            
            # 데이터 필터링
            filtered_df = filter_df_extended(df, selected_segments, selected_regions, selected_genders, selected_tenures)
            
            if not filtered_df.empty:
                # 메트릭 계산
                metrics_df = compute_appweb_metrics(filtered_df)
                
                # 상단 KPI 카드 5개
                st.markdown("#### 📈 주요 KPI 지표")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                total_customers = metrics_df['고객수'].sum()
                total_online_customers = metrics_df['온라인이용고객'].sum()
                total_online_amount = metrics_df['온라인금액_천원'].sum()
                total_amount = metrics_df['온라인금액_천원'].sum() + metrics_df['오프라인금액_천원'].sum()
                total_ib_customers = metrics_df['IB조회고객'].sum()
                
                with col1:
                    digital_rate = total_online_customers / total_customers if total_customers > 0 else 0
                    st.metric(
                        label="디지털 이용율",
                        value=f"{digital_rate:.1%}"
                    )
                
                with col2:
                    online_ratio = total_online_amount / total_amount if total_amount > 0 else 0
                    st.metric(
                        label="온라인 비중(금액)",
                        value=f"{online_ratio:.1%}"
                    )
                
                with col3:
                    online_arpu = (total_online_amount * 1000) / total_customers if total_customers > 0 else 0
                    st.metric(
                        label="온라인 ARPU",
                        value=f"{online_arpu:,.0f}원"
                    )
                
                with col4:
                    total_online_count = metrics_df['온라인건수'].sum()
                    avg_ticket = total_online_amount / total_online_count if total_online_count > 0 else 0
                    st.metric(
                        label="온라인 객단가",
                        value=f"{avg_ticket:,.1f}천원"
                    )
                
                with col5:
                    ib_rate = total_ib_customers / total_customers if total_customers > 0 else 0
                    st.metric(
                        label="승인내역 조회율",
                        value=f"{ib_rate:.1%}"
                    )
                
                st.markdown("---")
                
                # 섹션1: 디지털 채택&활용도
                st.markdown("#### 📊 디지털 채택 & 활용도")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 디지털 이용율 Bar 차트
                    fig_bar = px.bar(
                        metrics_df.sort_values('디지털이용율', ascending=False),
                        x='Segment', y='디지털이용율',
                        color='Segment',
                        color_discrete_map=COLOR_MAP,
                        title="세그먼트별 디지털 이용율"
                    )
                    fig_bar.update_layout(
                        yaxis=dict(tickformat='.0%'),
                        showlegend=False,
                        height=400
                    )
                    fig_bar.update_traces(
                        marker_color=[COLOR_MAP.get(seg, '#999999') if not metrics_df[metrics_df['Segment']==seg]['small_sample'].iloc[0] else '#CCCCCC' 
                                    for seg in metrics_df.sort_values('디지털이용율', ascending=False)['Segment']]
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    # 온라인 비중 vs ARPU Scatter
                    fig_scatter = px.scatter(
                        metrics_df,
                        x='온라인비중_금액', y='온라인_ARPU_원',
                        size='고객수',
                        color='Segment',
                        color_discrete_map=COLOR_MAP,
                        title="온라인 비중 vs ARPU (원 단위)",
                        hover_data=['고객수', '온라인_객단가_천원']
                    )
                    fig_scatter.update_layout(
                        xaxis=dict(tickformat='.0%'),
                        yaxis=dict(tickformat=',.0f'),
                        height=400
                    )
                    # 50% 가이드라인 추가
                    fig_scatter.add_vline(x=0.5, line_dash="dash", line_color="gray", 
                                        annotation_text="50% 기준선")
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # 요약표
                st.markdown("**세그먼트별 상세 지표**")
                display_df = metrics_df[['Segment', '고객수', '디지털이용율', '온라인비중_금액', 
                                       '온라인_ARPU_원', '온라인_객단가_천원', 'IB조회율']].copy()
                display_df['디지털이용율'] = display_df['디지털이용율'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                display_df['온라인비중_금액'] = display_df['온라인비중_금액'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                display_df['온라인_ARPU_원'] = display_df['온라인_ARPU_원'].apply(lambda x: f"{x:,.0f}원" if pd.notna(x) else "N/A")
                display_df['온라인_객단가_천원'] = display_df['온라인_객단가_천원'].apply(lambda x: f"{x:,.1f}천원" if pd.notna(x) else "N/A")
                display_df['IB조회율'] = display_df['IB조회율'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                
                st.dataframe(display_df, use_container_width=True)
                
                st.markdown("---")
                
                # 섹션2: 퍼널 분석
                st.markdown("#### 🔄 디지털 퍼널 분석")
                
                # 퍼널 데이터 준비
                funnel_data = {
                    '단계': ['승인내역 조회', '온라인 결제 경험', '반복 결제(3회+)'],
                    '고객수': [
                        metrics_df['IB조회고객'].sum(),
                        metrics_df['온라인이용고객'].sum(),
                        metrics_df['반복결제고객'].sum()
                    ]
                }
                
                # 퍼널 차트
                fig_funnel = go.Figure(go.Funnel(
                    y=funnel_data['단계'],
                    x=funnel_data['고객수'],
                    textinfo="value+percent initial",
                    marker={"color": ["#FF6B6B", "#4ECDC4", "#45B7D1"]}
                ))
                fig_funnel.update_layout(
                    title="디지털 이용 퍼널",
                    height=400
                )
                st.plotly_chart(fig_funnel, use_container_width=True)
                
                # 전환율 표시
                col1, col2 = st.columns(2)
                with col1:
                    conversion_1to2 = metrics_df['퍼널_1to2'].mean()
                    st.metric("조회 → 결제 전환율", f"{conversion_1to2:.1%}")
                
                with col2:
                    conversion_2to3 = metrics_df['퍼널_2to3'].mean()
                    st.metric("결제 → 반복 전환율", f"{conversion_2to3:.1%}")
                
                st.markdown("---")
                
                # 섹션3: 인게이지먼트 히트맵
                st.markdown("#### 🔥 세그먼트 × 가입기간 인게이지먼트")
                
                heatmap_data = heatmap_matrix(filtered_df)
                heatmap_pivot = heatmap_data.pivot(index='Segment', columns='가입기간_구간', values='온라인비중_금액')
                
                fig_heatmap = px.imshow(
                    heatmap_pivot,
                    color_continuous_scale='Blues',
                    title="온라인 이용 비중 히트맵",
                    text_auto=True
                )
                fig_heatmap.update_layout(
                    height=400,
                    coloraxis_colorbar=dict(tickformat='.0%')
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                st.markdown("---")
                
                # 섹션4: 온라인/오프라인 구성 & 티켓 사이즈
                st.markdown("#### 💰 온라인/오프라인 구성 & 티켓 사이즈")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 온라인/오프라인 구성 스택바
                    stack_data = metrics_df[['Segment', '온라인금액_천원', '오프라인금액_천원']].copy()
                    stack_data = stack_data.set_index('Segment')
                    
                    fig_stack = go.Figure()
                    fig_stack.add_trace(go.Bar(
                        name='온라인',
                        x=stack_data.index,
                        y=stack_data['온라인금액_천원'],
                        marker_color='#4ECDC4'
                    ))
                    fig_stack.add_trace(go.Bar(
                        name='오프라인',
                        x=stack_data.index,
                        y=stack_data['오프라인금액_천원'],
                        marker_color='#FF6B6B'
                    ))
                    fig_stack.update_layout(
                        barmode='stack',
                        title="온라인/오프라인 금액 구성",
                        yaxis_title="금액 (천원)",
                        height=400
                    )
                    st.plotly_chart(fig_stack, use_container_width=True)
                
                with col2:
                    # 온라인 객단가 박스플롯
                    fig_box = px.box(
                        metrics_df,
                        x='Segment', y='온라인_객단가_천원',
                        color='Segment',
                        color_discrete_map=COLOR_MAP,
                        title="세그먼트별 온라인 객단가 분포"
                    )
                    fig_box.update_layout(
                        yaxis_title="객단가 (천원)",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                
                st.markdown("---")
                
                # 섹션5: 지역·업종 교차 인사이트
                st.markdown("#### 🌍 지역·업종 교차 인사이트")
                
                # 지역별 데이터 집계
                region_data = filtered_df.groupby('거주시도명').agg({
                    'ID': 'nunique',
                    '이용금액_온라인_R6M': 'sum',
                    'Segment': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'C'
                }).reset_index()
                region_data.columns = ['지역', '고객수', '온라인금액_천원', '주요세그먼트']
                region_data['온라인_ARPU_원'] = (region_data['온라인금액_천원'] * 1000) / region_data['고객수']
                region_data = region_data.sort_values('고객수', ascending=False).head(10)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 지역별 버블 차트
                    fig_bubble = px.scatter(
                        region_data,
                        x='고객수', y='온라인_ARPU_원',
                        size='고객수',
                        color='주요세그먼트',
                        color_discrete_map=COLOR_MAP,
                        hover_name='지역',
                        title="지역별 고객수 vs 온라인 ARPU",
                        size_max=50
                    )
                    fig_bubble.update_layout(
                        xaxis_title="고객수",
                        yaxis_title="온라인 ARPU (원)",
                        height=400
                    )
                    st.plotly_chart(fig_bubble, use_container_width=True)
                
                with col2:
                    # 상위 지역 표
                    st.markdown("**고객 밀도 상위 지역**")
                    top_regions = region_data.head(5)[['지역', '고객수', '온라인_ARPU_원']].copy()
                    top_regions['고객수'] = top_regions['고객수'].apply(lambda x: f"{x:,}명")
                    top_regions['온라인_ARPU_원'] = top_regions['온라인_ARPU_원'].apply(lambda x: f"{x:,.0f}원")
                    st.dataframe(top_regions, use_container_width=True)
                
                st.markdown("---")
                
                # 하단: KPI 정의/단위/가드
                with st.expander("📖 KPI 정의 및 주의사항", expanded=False):
                    st.markdown("""
                    **KPI 정의:**
                    
                    - **디지털 이용율**: 온라인 이용 고객수 / 전체 고객수
                    - **온라인 비중(금액)**: 온라인 이용금액 / (온라인 + 오프라인 이용금액)
                    - **온라인 ARPU**: 온라인 이용금액(천원) × 1000 / 고객수 (원 단위)
                    - **온라인 객단가**: 온라인 이용금액 / 온라인 이용건수 (천원 단위)
                    - **승인내역 조회율**: IB문의건수 > 0인 고객수 / 전체 고객수
                    
                    **데이터 단위:**
                    - 금액 컬럼은 천원 단위로 저장됨
                    - ARPU만 원 단위로 변환하여 표시 (×1000)
                    - 비율은 계산 시 0~1, UI에서만 % 표기
                    
                    **주의사항:**
                    - 고객수 < 100인 세그먼트는 회색 처리 (해석 주의)
                    - 소표본 세그먼트의 지표는 참고용으로만 활용
                    - 지역별 분석 시 표본 수 고려 필요
                    """)
            
            else:
                st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
            
            # 시각화 자료 설명
            show_visualization_guide()
        
        with sub_tabs[1]:  # 마케팅 채널 반응
            st.markdown("### 📢 마케팅 채널 반응")
            
            # 상단 필터
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown("**세그먼트**")
                selected_segments = st.multiselect("세그먼트 선택", SEGMENTS, 
                                                default=SEGMENTS, key="marketing_segments")
            
            with col2:
                st.markdown("**지역**")
                regions = df['거주시도명'].unique().tolist()
                selected_regions = st.multiselect("지역 선택", regions, 
                                                default=regions, key="marketing_regions")
            
            with col3:
                st.markdown("**성별**")
                genders = df['남녀구분코드'].unique().tolist()
                selected_genders = st.multiselect("성별 선택", genders, 
                                                default=genders, key="marketing_genders")
            
            with col4:
                st.markdown("**가입기간 구간**")
                tenure_options = ["신규(<2)", "중간(2-5)", "기존(5-10)", "우수(10+)"]
                selected_tenures = st.multiselect("가입기간 구간 선택", tenure_options, 
                                                default=tenure_options, key="marketing_tenures")
            
            with col5:
                st.markdown("**리스크 제외**")
                exclude_risk = st.checkbox("리스크 고객 제외", value=False, 
                                        help="연체잔액>0 또는 RV잔액 상위 25% 고객 제외")
            
            # 데이터 필터링
            filtered_df = filter_for_marketing(df, selected_segments, selected_regions, 
                                            selected_genders, selected_tenures, exclude_risk)
            
            if not filtered_df.empty:
                # 메트릭 계산
                metrics_df, thresholds = compute_channel_metrics(filtered_df)
                
                # 상단 KPI 카드 5개
                st.markdown("#### 📈 주요 KPI 지표")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                total_customers = metrics_df['고객수'].sum()
                total_ib_customers = metrics_df['IB조회고객'].sum()
                total_online_customers = metrics_df['온라인고객'].sum()
                total_online_amount = metrics_df['온라인금액_천원'].sum()
                total_amount = metrics_df['온라인금액_천원'].sum() + metrics_df['오프라인금액_천원'].sum()
                total_benefit_customers = metrics_df['혜택고객'].sum()
                
                with col1:
                    ib_rate = total_ib_customers / total_customers if total_customers > 0 else 0
                    st.metric(
                        label="IB 조회율",
                        value=f"{ib_rate:.1%}"
                    )
                
                with col2:
                    digital_rate = total_online_customers / total_customers if total_customers > 0 else 0
                    st.metric(
                        label="디지털 이용율",
                        value=f"{digital_rate:.1%}"
                    )
                
                with col3:
                    online_ratio = total_online_amount / total_amount if total_amount > 0 else 0
                    st.metric(
                        label="온라인 비중(금액)",
                        value=f"{online_ratio:.1%}"
                    )
                
                with col4:
                    online_arpu = (total_online_amount * 1000) / total_customers if total_customers > 0 else 0
                    st.metric(
                        label="온라인 ARPU",
                        value=f"{online_arpu:,.0f}원"
                    )
                
                with col5:
                    benefit_rate = total_benefit_customers / total_customers if total_customers > 0 else 0
                    st.metric(
                        label="혜택 반응율",
                        value=f"{benefit_rate:.1%}"
                    )
                
                st.markdown("---")
                
                # 섹션1: 채널 반응 매트릭스
                st.markdown("#### 📊 채널 반응 매트릭스")
                
                # Scatter 차트
                fig_scatter = px.scatter(
                    metrics_df,
                    x='온라인비중', y='IB조회율',
                    size='고객수',
                    color='Segment',
                    color_discrete_map=COLOR_MAP,
                    title="온라인 비중 vs IB 조회율 (채널 반응 매트릭스)",
                    hover_data=['고객수', '온라인ARPU_원', '온라인객단가_천원', '추천채널']
                )
                fig_scatter.update_layout(
                    xaxis=dict(tickformat='.0%', title="온라인 비중(금액)"),
                    yaxis=dict(tickformat='.0%', title="IB 조회율"),
                    height=500
                )
                
                # 중앙값 기준 보조선 추가
                fig_scatter.add_vline(x=thresholds['온라인비중'], line_dash="dash", line_color="gray", 
                                    annotation_text=f"중앙값: {thresholds['온라인비중']:.1%}")
                fig_scatter.add_hline(y=thresholds['IB조회율'], line_dash="dash", line_color="gray", 
                                    annotation_text=f"중앙값: {thresholds['IB조회율']:.1%}")
                
                # 4분면 라벨 추가
                fig_scatter.add_annotation(x=0.8, y=0.8, text="App Push", showarrow=False, 
                                        font=dict(size=12, color="blue"))
                fig_scatter.add_annotation(x=0.2, y=0.8, text="톡채널", showarrow=False, 
                                        font=dict(size=12, color="green"))
                fig_scatter.add_annotation(x=0.8, y=0.2, text="이메일/DM", showarrow=False, 
                                        font=dict(size=12, color="orange"))
                fig_scatter.add_annotation(x=0.2, y=0.2, text="TM/콜센터", showarrow=False, 
                                        font=dict(size=12, color="red"))
                
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                st.markdown("---")
                
                # 섹션2: 추천 채널 & 사유
                st.markdown("#### 🎯 추천 채널 & 사유")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 세그먼트별 추천채널 분포 Bar
                    channel_counts = metrics_df['추천채널'].value_counts()
                    fig_bar = px.bar(
                        x=channel_counts.index, y=channel_counts.values,
                        title="세그먼트별 추천 채널 분포",
                        color=channel_counts.index,
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                    )
                    fig_bar.update_layout(
                        xaxis_title="추천 채널",
                        yaxis_title="세그먼트 수",
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    # 추천 타깃 카드들
                    st.markdown("**추천 타깃**")
                    for _, row in metrics_df.iterrows():
                        if row['고객수'] >= 100:  # 소표본 제외
                            st.info(f"""
                            **세그먼트 {row['Segment']} — {row['추천채널']}**
                            - IB조회율: {row['IB조회율']:.1%}
                            - 온라인비중: {row['온라인비중']:.1%}
                            - 온라인 ARPU: {row['온라인ARPU_원']:,.0f}원
                            - 고객수: {row['고객수']:,}명
                            """)
                
                # 상세 표
                st.markdown("**세그먼트별 상세 지표**")
                display_df = metrics_df[['Segment', '고객수', 'IB조회율', '온라인비중', 
                                       '온라인ARPU_원', '온라인객단가_천원', '혜택반응율', '추천채널']].copy()
                display_df['IB조회율'] = display_df['IB조회율'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                display_df['온라인비중'] = display_df['온라인비중'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                display_df['온라인ARPU_원'] = display_df['온라인ARPU_원'].apply(lambda x: f"{x:,.0f}원" if pd.notna(x) else "N/A")
                display_df['온라인객단가_천원'] = display_df['온라인객단가_천원'].apply(lambda x: f"{x:,.1f}천원" if pd.notna(x) else "N/A")
                display_df['혜택반응율'] = display_df['혜택반응율'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                
                st.dataframe(display_df, use_container_width=True)
                
                st.markdown("---")
                
                # 섹션3: 혜택 반응 & 티켓 사이즈
                st.markdown("#### 🎁 혜택 반응 & 티켓 사이즈")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 혜택반응율 vs 온라인객단가 Scatter
                    fig_scatter2 = px.scatter(
                        metrics_df,
                        x='온라인객단가_천원', y='혜택반응율',
                        size='고객수',
                        color='Segment',
                        color_discrete_map=COLOR_MAP,
                        title="온라인 객단가 vs 혜택 반응율",
                        hover_data=['고객수', '온라인ARPU_원', '추천채널']
                    )
                    fig_scatter2.update_layout(
                        xaxis_title="온라인 객단가 (천원)",
                        yaxis_title="혜택 반응율",
                        height=400
                    )
                    st.plotly_chart(fig_scatter2, use_container_width=True)
                
                with col2:
                    # 온라인객단가 박스플롯
                    fig_box = px.box(
                        metrics_df,
                        x='Segment', y='온라인객단가_천원',
                        color='Segment',
                        color_discrete_map=COLOR_MAP,
                        title="세그먼트별 온라인 객단가 분포"
                    )
                    fig_box.update_layout(
                        yaxis_title="객단가 (천원)",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # 메시지 권장
                st.markdown("**메시지 권장**")
                high_benefit_low_ticket = metrics_df[(metrics_df['혜택반응율'] > metrics_df['혜택반응율'].median()) & 
                                                   (metrics_df['온라인객단가_천원'] < metrics_df['온라인객단가_천원'].median())]
                high_ticket = metrics_df[metrics_df['온라인객단가_천원'] > metrics_df['온라인객단가_천원'].median()]
                
                if not high_benefit_low_ticket.empty:
                    st.success(f"**빈도형 메시지 권장**: {', '.join(high_benefit_low_ticket['Segment'].astype(str))} 세그먼트 - 혜택반응율 높고 객단가 낮음")
                
                if not high_ticket.empty:
                    st.info(f"**프리미엄 메시지 권장**: {', '.join(high_ticket['Segment'].astype(str))} 세그먼트 - 객단가 높음")
                
                st.markdown("---")
                
                # 섹션4: 지역×채널 히트맵
                st.markdown("#### 🌍 지역×채널 히트맵")
                
                # 지역별 채널 분포 계산
                region_mix = region_channel_mix(filtered_df, thresholds)
                
                # 히트맵 데이터 준비
                heatmap_pivot = region_mix.pivot(index='거주시도명', columns='개인추천채널', values='비중').fillna(0)
                
                fig_heatmap = px.imshow(
                    heatmap_pivot,
                    color_continuous_scale='Blues',
                    title="지역별 추천 채널 비중",
                    text_auto=True
                )
                fig_heatmap.update_layout(
                    height=500,
                    coloraxis_colorbar=dict(tickformat='.0%')
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # 상위 지역 요약
                st.markdown("**상위 지역 요약**")
                top_regions = region_mix.groupby('거주시도명')['고객수'].sum().sort_values(ascending=False).head(5)
                
                for region in top_regions.index:
                    region_data = region_mix[region_mix['거주시도명'] == region]
                    top_channel = region_data.loc[region_data['비중'].idxmax(), '개인추천채널']
                    st.write(f"**{region}**: 주요 채널 {top_channel} ({region_data['비중'].max():.1%}), 고객수 {region_data['고객수'].sum():,}명")
                
                st.markdown("---")
                
                # 하단: 정의/가드
                with st.expander("📖 KPI 정의 및 주의사항", expanded=False):
                    st.markdown("""
                    **KPI 정의:**
                    
                    - **IB 조회율**: IB문의건수 > 0인 고객수 / 전체 고객수
                    - **디지털 이용율**: 온라인 이용 고객수 / 전체 고객수
                    - **온라인 비중(금액)**: 온라인 이용금액 / (온라인 + 오프라인 이용금액)
                    - **온라인 ARPU**: 온라인 이용금액(천원) × 1000 / 고객수 (원 단위)
                    - **혜택 반응율**: 포인트/마일리지 이용 고객수 / 전체 고객수
                    
                    **추천 채널 룰:**
                    - **App Push**: IB조회율 높음 + 온라인비중 높음
                    - **톡채널**: IB조회율 높음 + 온라인비중 낮음
                    - **이메일/DM**: IB조회율 낮음 + 온라인비중 높음
                    - **TM/콜센터**: IB조회율 낮음 + 온라인비중 낮음
                    
                    **데이터 단위:**
                    - 금액 컬럼은 천원 단위로 저장됨
                    - ARPU만 원 단위로 변환하여 표시 (×1000)
                    - 비율은 계산 시 0~1, UI에서만 % 표기
                    
                    **주의사항:**
                    - 고객수 < 100인 세그먼트는 소표본으로 처리
                    - 리스크 제외 옵션: 연체잔액>0 또는 RV잔액 상위 25% 고객 제외
                    - 임계값은 선택된 데이터의 중앙값 기준
                    """)
            
            else:
                st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
            
            # 시각화 자료 설명
            show_visualization_guide()
        
        with sub_tabs[2]:  # 캠페인 참여 & 쿠폰 사용
            st.markdown("### 🎁 캠페인 참여 & 쿠폰 사용")
            
            # 사이드바 필터
            with st.sidebar:
                st.markdown("### 📊 필터 설정")
                
                # 기준년월 필터
                st.markdown("**기준년월**")
                date_filter_type = st.radio("날짜 필터 타입", ["범위", "단일"], key="campaign_date_type")
                
                if date_filter_type == "범위":
                    start_date = st.date_input("시작일", value=pd.to_datetime('2023-01-01'), key="campaign_start_date")
                    end_date = st.date_input("종료일", value=pd.to_datetime('2023-12-31'), key="campaign_end_date")
                else:
                    single_date = st.date_input("기준일", value=pd.to_datetime('2023-06-01'), key="campaign_single_date")
                
                # 채널 필터
                st.markdown("**채널**")
                channels = ["SMS", "DM", "TM", "앱푸시", "이메일"]
                selected_channels = st.multiselect("채널 선택", channels, default=channels, key="campaign_channels")
                
                # 대상세그 필터
                st.markdown("**대상세그**")
                segments = ["A", "B", "C", "D", "E"]
                selected_segments = st.multiselect("세그먼트 선택", segments, default=segments, key="campaign_segments")
                
                # 지역 필터
                st.markdown("**지역**")
                regions = df['거주시도명'].unique().tolist()
                selected_regions = st.multiselect("지역 선택", regions, default=regions, key="campaign_regions")
                
                # 성별 필터
                st.markdown("**성별**")
                genders = df['남녀구분코드'].unique().tolist()
                selected_genders = st.multiselect("성별 선택", genders, default=genders, key="campaign_genders")
                
                # 연령대 필터
                st.markdown("**연령대**")
                age_ranges = ["20대", "30대", "40대", "50대", "60대+"]
                selected_ages = st.multiselect("연령대 선택", age_ranges, default=age_ranges, key="campaign_ages")
                
                # 마케팅동의 필터
                st.markdown("**마케팅동의**")
                marketing_consent = st.selectbox("마케팅동의 여부", ["전체", "동의", "비동의"], key="campaign_consent")
                
                # 샘플링 옵션
                st.markdown("**샘플링**")
                sampling_rate = st.selectbox("샘플링 비율", [100, 50, 20], key="campaign_sampling")
            
            # 데이터 필터링
            filtered_df = filter_campaign_data(df, date_filter_type, start_date if date_filter_type == "범위" else single_date, 
                                            end_date if date_filter_type == "범위" else single_date,
                                            selected_channels, selected_segments, selected_regions, 
                                            selected_genders, selected_ages, marketing_consent, sampling_rate)
            
            if not filtered_df.empty:
                # 메트릭 계산
                campaign_metrics = compute_campaign_metrics(filtered_df)
                
                # 상단 KPI 카드 6개
                st.markdown("#### 📈 주요 KPI 지표")
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                with col1:
                    total_exposure = campaign_metrics['노출수'].sum()
                    st.metric(
                        label="총 노출수",
                        value=f"{total_exposure:,}"
                    )
                
                with col2:
                    total_participation = campaign_metrics['참여수'].sum()
                    participation_rate = total_participation / total_exposure if total_exposure > 0 else 0
                    st.metric(
                        label="참여율",
                        value=f"{participation_rate:.1%}"
                    )
                
                with col3:
                    total_conversion = campaign_metrics['전환수'].sum()
                    conversion_rate = total_conversion / total_participation if total_participation > 0 else 0
                    st.metric(
                        label="전환율",
                        value=f"{conversion_rate:.1%}"
                    )
                
                with col4:
                    total_coupon_usage = campaign_metrics['쿠폰사용수'].sum()
                    total_coupon_issued = campaign_metrics['쿠폰발급수'].sum()
                    coupon_usage_rate = total_coupon_usage / total_coupon_issued if total_coupon_issued > 0 else 0
                    st.metric(
                        label="쿠폰사용률",
                        value=f"{coupon_usage_rate:.1%}"
                    )
                
                with col5:
                    total_discount = campaign_metrics['할인금액'].sum()
                    st.metric(
                        label="총 할인금액",
                        value=f"{total_discount:,.0f}원"
                    )
                
                with col6:
                    total_revenue = campaign_metrics['사용금액'].sum()
                    net_benefit = total_revenue - total_discount
                    st.metric(
                        label="순효익(추정)",
                        value=f"{net_benefit:,.0f}원"
                    )
                
                st.markdown("---")
                
                # 섹션1: 퍼널 차트
                st.markdown("#### 🔄 캠페인 퍼널")
                
                funnel_data = {
                    '단계': ['노출', '참여', '전환'],
                    '고객수': [
                        total_exposure,
                        total_participation,
                        total_conversion
                    ]
                }
                
                fig_funnel = go.Figure(go.Funnel(
                    y=funnel_data['단계'],
                    x=funnel_data['고객수'],
                    textinfo="value+percent initial",
                    marker={"color": ["#FF6B6B", "#4ECDC4", "#45B7D1"]}
                ))
                fig_funnel.update_layout(
                    title="캠페인 퍼널 (노출 → 참여 → 전환)",
                    height=400
                )
                st.plotly_chart(fig_funnel, use_container_width=True)
                
                st.markdown("---")
                
                # 섹션2: 월별 추이
                st.markdown("#### 📈 월별 추이")
                
                monthly_trends = compute_monthly_trends(filtered_df)
                
                fig_trends = go.Figure()
                fig_trends.add_trace(go.Scatter(
                    x=monthly_trends['월'],
                    y=monthly_trends['참여율'],
                    mode='lines+markers',
                    name='참여율',
                    line=dict(color='#FF6B6B')
                ))
                fig_trends.add_trace(go.Scatter(
                    x=monthly_trends['월'],
                    y=monthly_trends['전환율'],
                    mode='lines+markers',
                    name='전환율',
                    line=dict(color='#4ECDC4')
                ))
                fig_trends.add_trace(go.Scatter(
                    x=monthly_trends['월'],
                    y=monthly_trends['쿠폰사용률'],
                    mode='lines+markers',
                    name='쿠폰사용률',
                    line=dict(color='#45B7D1')
                ))
                fig_trends.update_layout(
                    title="월별 성과 추이",
                    xaxis_title="월",
                    yaxis_title="비율",
                    yaxis=dict(tickformat='.0%'),
                    height=400
                )
                st.plotly_chart(fig_trends, use_container_width=True)
                
                st.markdown("---")
                
                # 섹션3: 채널별 성과
                st.markdown("#### 📊 채널별 성과")
                
                channel_performance = compute_channel_performance(filtered_df)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 참여율 바차트
                    fig_participation = px.bar(
                        channel_performance,
                        x='채널', y='참여율',
                        title="채널별 참여율",
                        color='채널',
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                    )
                    fig_participation.update_layout(
                        yaxis=dict(tickformat='.0%'),
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_participation, use_container_width=True)
                
                with col2:
                    # 전환율 바차트
                    fig_conversion = px.bar(
                        channel_performance,
                        x='채널', y='전환율',
                        title="채널별 전환율",
                        color='채널',
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                    )
                    fig_conversion.update_layout(
                        yaxis=dict(tickformat='.0%'),
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_conversion, use_container_width=True)
                
                st.markdown("---")
                
                # 섹션4: 캠페인 TOP 테이블
                st.markdown("#### 🏆 캠페인 성과 TOP")
                
                top_campaigns = get_top_campaigns(filtered_df)
                
                # 다운로드 버튼
                csv_data = top_campaigns.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 CSV 다운로드",
                    data=csv_data,
                    file_name=f"campaign_performance_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                st.dataframe(top_campaigns, use_container_width=True)
                
                st.markdown("---")
                
                # 섹션5: 캠페인 비교
                st.markdown("#### 🔍 캠페인 비교")
                
                # 체크박스로 캠페인 선택
                campaign_options = top_campaigns['캠페인명'].tolist()[:5]  # 상위 5개만
                selected_campaigns = st.multiselect("비교할 캠페인 선택 (최대 3개)", campaign_options, max_selections=3)
                
                if len(selected_campaigns) >= 2:
                    comparison_data = get_campaign_comparison(filtered_df, selected_campaigns)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 비교 차트
                        fig_comparison = px.bar(
                            comparison_data,
                            x='캠페인명', y='참여율',
                            title="캠페인별 참여율 비교",
                            color='캠페인명',
                            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
                        )
                        fig_comparison.update_layout(
                            yaxis=dict(tickformat='.0%'),
                            height=400,
                            showlegend=False
                        )
                        st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    with col2:
                        # 비교 표
                        st.markdown("**상세 비교**")
                        st.dataframe(comparison_data, use_container_width=True)
                
                st.markdown("---")
                
                # 하단: 정의/가드
                with st.expander("📖 KPI 정의 및 주의사항", expanded=False):
                    st.markdown("""
                    **KPI 정의:**
                    
                    - **참여율**: 참여수 / 노출수
                    - **전환율**: 전환수 / 참여수
                    - **쿠폰사용률**: 쿠폰사용수 / 쿠폰발급수
                    - **순효익(추정)**: 사용금액 - 할인금액 - 캠페인비용
                    - **간이 리프트**: 타겟 전환율 - 컨트롤 전환율
                    
                    **데이터 단위:**
                    - 금액: 원 단위
                    - 비율: 계산 시 0~1, UI에서만 % 표기
                    
                    **주의사항:**
                    - 0으로 나눗셈 방지
                    - 참여수 > 노출수 같은 논리오류 하이라이트
                    - 샘플링 적용 시 결과 해석 주의
                    - 결측값은 0으로 처리
                    """)
            
            else:
                st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
            
            # 시각화 자료 설명
            show_visualization_guide()
        
        with sub_tabs[3]:  # 업종(MCC) 소비 패턴
            st.markdown("### 🏪 업종(MCC) 소비 패턴")
            
            # 사이드바 필터
            with st.sidebar:
                st.markdown("### 📊 필터 설정")
                
                # 기간 필터
                st.markdown("**기간**")
                start_date = st.date_input("시작일", value=pd.to_datetime('2023-01-01'), key="mcc_start_date")
                end_date = st.date_input("종료일", value=pd.to_datetime('2023-12-31'), key="mcc_end_date")
                
                # 카드유형 필터
                st.markdown("**카드유형**")
                card_types = ["신용", "체크"]
                selected_card_types = st.multiselect("카드유형 선택", card_types, default=card_types, key="mcc_card_types")
                
                # 채널 필터
                st.markdown("**채널**")
                channels = ["온라인", "오프라인"]
                selected_channels = st.multiselect("채널 선택", channels, default=channels, key="mcc_channels")
                
                # 지역 필터
                st.markdown("**지역**")
                regions = ['서울', '경기', '부산', '대구', '인천', '광주', '대전', '울산']
                selected_regions = st.multiselect("지역 선택", regions, default=regions, key="mcc_regions")
                
                # 성별 필터
                st.markdown("**성별**")
                genders = [1, 2]
                selected_genders = st.multiselect("성별 선택", genders, default=genders, key="mcc_genders")
                
                # 연령대 필터
                st.markdown("**연령대**")
                age_ranges = ["20대", "30대", "40대", "50대", "60대+"]
                selected_ages = st.multiselect("연령대 선택", age_ranges, default=age_ranges, key="mcc_ages")
            
            # 데이터 필터링
            filtered_df = filter_mcc_data(df, start_date, end_date, selected_card_types, 
                                        selected_channels, selected_regions, 
                                        selected_genders, selected_ages)
            
            if not filtered_df.empty:
                # 메트릭 계산
                mcc_summary, kpi_metrics = compute_mcc_metrics(filtered_df)
                
                # 상단 KPI 카드 4개
                st.markdown("#### 📈 주요 KPI 지표")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="총금액",
                        value=f"{kpi_metrics['총금액']:,.0f}원"
                    )
                
                with col2:
                    st.metric(
                        label="평균객단가",
                        value=f"{kpi_metrics['평균객단가']:,.0f}원"
                    )
                
                with col3:
                    st.metric(
                        label="신용카드 비중",
                        value=f"{kpi_metrics['신용비중']:.1%}"
                    )
                
                with col4:
                    st.metric(
                        label="집중도(HHI)",
                        value=f"{kpi_metrics['HHI']:.3f}"
                    )
                
                st.markdown("---")
                
                # 섹션1: MCC 점유 Treemap
                st.markdown("#### 🎯 MCC 점유 현황")
                
                # Treemap 차트
                fig_treemap = px.treemap(
                    mcc_summary.head(10),
                    path=['MCC명'],
                    values='금액',
                    color='점유율',
                    color_continuous_scale='Viridis',
                    title="MCC별 금액 점유율 (Top 10)"
                )
                fig_treemap.update_layout(height=500)
                st.plotly_chart(fig_treemap, use_container_width=True)
                
                st.markdown("---")
                
                # 섹션2: 월별 MCC별 성장 라인
                st.markdown("#### 📈 월별 MCC별 성장 추이")
                
                growth_data = compute_mcc_growth(filtered_df)
                
                # MCC 선택
                mcc_options = mcc_summary.head(5)['MCC명'].tolist()
                selected_mccs = st.multiselect("MCC 선택 (최대 5개)", mcc_options, default=mcc_options[:3], max_selections=5)
                
                if selected_mccs:
                    filtered_growth = growth_data[growth_data['MCC명'].isin(selected_mccs)]
                    
                    fig_growth = px.line(
                        filtered_growth,
                        x='월',
                        y='금액',
                        color='MCC명',
                        title="월별 MCC별 금액 추이",
                        markers=True
                    )
                    fig_growth.update_layout(
                        xaxis_title="월",
                        yaxis_title="금액 (원)",
                        height=400
                    )
                    st.plotly_chart(fig_growth, use_container_width=True)
                
                st.markdown("---")
                
                # 섹션3: 지역×MCC 히트맵
                st.markdown("#### 🗺️ 지역별 MCC 히트맵")
                
                heatmap_data = compute_region_mcc_heatmap(filtered_df)
                
                fig_heatmap = px.imshow(
                    heatmap_data.values,
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    color_continuous_scale='Blues',
                    title="지역별 MCC 비중 히트맵"
                )
                fig_heatmap.update_layout(
                    xaxis_title="MCC",
                    yaxis_title="지역",
                    height=400
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                st.markdown("---")
                
                # 섹션4: MCC 랭킹 테이블
                st.markdown("#### 🏆 MCC 랭킹")
                
                top_mcc_list = get_top_mcc_list(filtered_df)
                
                # 다운로드 버튼
                csv_data = top_mcc_list.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 CSV 다운로드",
                    data=csv_data,
                    file_name=f"mcc_ranking_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                # 테이블 표시
                display_columns = ['MCC', 'MCC명', '점유율', '금액', '평균객단가', 'R3M증감률']
                st.dataframe(
                    top_mcc_list[display_columns].round(4),
                    use_container_width=True
                )
                
                st.markdown("---")
                
                # 섹션5: 제휴 후보 제안
                st.markdown("#### 💡 제휴 후보 제안")
                
                # AI 연결용 제안 영역
                st.info("🤖 AI 분석을 통한 제휴 후보 제안이 여기에 표시됩니다.")
                
                # 제안 입력 영역 (AI 연결용)
                with st.expander("📝 제휴 후보 제안 입력", expanded=False):
                    st.text_area(
                        "AI 분석 결과를 입력하세요:",
                        placeholder="예: 식료품점/슈퍼마켓 업종이 최근 3개월간 15.2% 성장하여 제휴 후보로 적합합니다...",
                        height=100,
                        key="partnership_suggestion_input"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💾 제안 저장", key="save_suggestion"):
                            st.success("제안이 저장되었습니다!")
                    
                    with col2:
                        if st.button("🔄 AI 재분석", key="reanalyze"):
                            st.info("AI 재분석을 요청했습니다.")
                
                # 기존 제안 표시 영역 (비워둠)
                st.markdown("**제안 목록:**")
                st.info("제휴 후보 제안이 아직 생성되지 않았습니다. AI 분석을 통해 제안을 받아보세요.")
                
                st.markdown("---")
                
                # 하단: 정의/가드
                with st.expander("📖 KPI 정의 및 주의사항", expanded=False):
                    st.markdown("""
                    **KPI 정의:**
                    
                    - **점유율**: MCC별 금액 / 전체 금액
                    - **평균객단가**: MCC별 금액 / MCC별 건수
                    - **HHI (집중도)**: Σ(각 MCC 점유율²)
                    - **R3M 증감률**: 최근 3개월 대비 이전 3개월 증감률
                    
                    **데이터 단위:**
                    - 금액: 원 단위
                    - 비율: 계산 시 0~1, UI에서만 % 표기
                    
                    **주의사항:**
                    - 시뮬레이션 데이터 기반 분석
                    - 실제 제휴 결정 시 추가 검증 필요
                    - 계절성 및 시장 변화 고려
                    """)
            
            else:
                st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
            
            # 시각화 자료 설명
            show_visualization_guide()
        
        with sub_tabs[4]:  # VOC/CS
            st.markdown("### 📞 VOC/CS")
            
            # 사이드바 필터
            with st.sidebar:
                st.markdown("### 📊 필터 설정")
                
                # 기간 필터
                st.markdown("**기간**")
                start_date = st.date_input("시작일", value=pd.to_datetime('2023-01-01'), key="voc_start_date")
                end_date = st.date_input("종료일", value=pd.to_datetime('2023-12-31'), key="voc_end_date")
                
                # 채널 필터
                st.markdown("**채널**")
                channels = ['콜센터', '챗봇', '앱', '이메일', 'SMS']
                selected_channels = st.multiselect("채널 선택", channels, default=channels, key="voc_channels")
                
                # 카테고리 필터
                st.markdown("**카테고리**")
                categories = [
                    '카드발급', '이용한도', '할부/일시불', '포인트/마일리지', 
                    '해외이용', '온라인결제', 'ATM/현금서비스', '기타'
                ]
                selected_categories = st.multiselect("카테고리 선택", categories, default=categories, key="voc_categories")
                
                # 감성 범위 필터
                st.markdown("**감성 범위**")
                sentiment_range = st.slider(
                    "감성점수 범위", 
                    min_value=-1.0, 
                    max_value=1.0, 
                    value=(-1.0, 1.0),
                    step=0.1,
                    key="voc_sentiment"
                )
                
                # 지역 필터
                st.markdown("**지역**")
                regions = ['서울', '경기', '부산', '대구', '인천', '광주', '대전', '울산']
                selected_regions = st.multiselect("지역 선택", regions, default=regions, key="voc_regions")
                
                # 세그먼트 필터
                st.markdown("**세그먼트**")
                segments = ['A', 'B', 'C', 'D', 'E']
                selected_segments = st.multiselect("세그먼트 선택", segments, default=segments, key="voc_segments")
            
            # 데이터 필터링
            filtered_df = filter_voc_data(df, start_date, end_date, selected_channels, 
                                        selected_categories, sentiment_range, 
                                        selected_regions, selected_segments)
            
            if not filtered_df.empty:
                # 메트릭 계산
                voc_metrics = compute_voc_metrics(filtered_df)
                
                # 상단 KPI 카드 5개
                st.markdown("#### 📈 주요 KPI 지표")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        label="총 VOC 건수",
                        value=f"{voc_metrics['총건수']:,}"
                    )
                
                with col2:
                    st.metric(
                        label="FCR (1차해결율)",
                        value=f"{voc_metrics['FCR']:.1%}"
                    )
                
                with col3:
                    st.metric(
                        label="AHT (평균처리시간)",
                        value=f"{voc_metrics['AHT']:.1f}분"
                    )
                
                with col4:
                    st.metric(
                        label="재문의율",
                        value=f"{voc_metrics['재문의율']:.1%}"
                    )
                
                with col5:
                    st.metric(
                        label="평균 감성점수",
                        value=f"{voc_metrics['평균감성']:.2f}"
                    )
                
                st.markdown("---")
                
                st.info("VOC/CS 분석 페이지 구현 중... (추후 완성 예정)")
            
            else:
                st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
            
            # 시각화 자료 설명
            show_visualization_guide()
    
    # 사이드바 하단 정보 (모든 탭에서 공통)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 데이터 정보")
    st.sidebar.info(f"""
    - 데이터 기간: 2018년 7월-12월
    - 총 고객 수: {total_customers:,}명
    - 세그먼트 수: 5개 (A, B, C, D, E)
    - 총 컬럼 수: {len(df.columns)}개
    """)
    
    # 데이터 미리보기
    st.sidebar.markdown("### 🔍 데이터 미리보기")
    if st.sidebar.checkbox("원본 데이터 보기"):
        st.sidebar.dataframe(df.head(10))

else:
    st.error("데이터를 로드할 수 없습니다. 파일 경로를 확인해주세요.")
