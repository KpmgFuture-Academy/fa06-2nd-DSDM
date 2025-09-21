# app_arrears.py
# -----------------------------------------------------------------------------
# 연체 예측 결과 시각화 대시보드 (CSV 직접 로드 버전)
# - 실행기(YAML+모델) 불필요
# - 이미 생성된 결과 CSV (pred_class, risk_label, p0/p1/p2 포함)를 로드
# - 라벨 분포 요약 + Feature별 분석 시각화 + 데이터 미리보기
# -----------------------------------------------------------------------------

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="연체 예측 결과 대시보드", page_icon="💳", layout="wide")

# =========================
# 사이드바 입력
# =========================
st.sidebar.title("💳 연체 예측 결과 뷰어")
csv_path = st.sidebar.text_input("결과 CSV 경로", value="predictions_arrears.csv")
run_btn = st.sidebar.button("📂 데이터 불러오기")

# =========================
# 세션 상태
# =========================
if "pred_df" not in st.session_state:
    st.session_state.pred_df = None

# =========================
# 메인 탭
# =========================
st.title("연체 예측 결과 대시보드")
tabs = st.tabs(["① 라벨 요약", "② Feature 시각화", "③ 데이터 미리보기"])

# ① 라벨 요약
with tabs[0]:
    st.subheader("라벨 분포 요약")
    if run_btn:
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            st.session_state.pred_df = df
            st.success("데이터 로드 성공 ✅")
        except Exception as e:
            st.error(f"CSV 로드 실패: {e}")

    if st.session_state.pred_df is not None:
        df = st.session_state.pred_df
        if "risk_label" not in df.columns:
            st.error("CSV에 'risk_label' 컬럼이 없습니다.")
        else:
            summary = df["risk_label"].value_counts().reindex(
                ["Low", "Medium", "High"], fill_value=0
            )
            ratios = (summary / len(df) * 100).round(2)

            st.dataframe(
                pd.DataFrame({"count": summary, "ratio(%)": ratios}), width="stretch"
            )

            df_plot = summary.reset_index(name="count")
            x_col = df_plot.columns[0]
            fig = px.bar(df_plot, x=x_col, y="count", text="count", title="라벨 분포 (건수)")
            fig.update_layout(xaxis_title="Label", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

# ② Feature 시각화
with tabs[1]:
    st.subheader("Feature별 시각화")
    if st.session_state.pred_df is not None:
        df = st.session_state.pred_df
        label_col = "risk_label"
        num_feats = [
            c
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in ["pred_class"]
        ]

        if not num_feats:
            st.warning("수치형 Feature가 없어 시각화를 표시할 수 없습니다.")
        else:
            # 단일 Feature 분포
            c1, c2 = st.columns([2, 2])
            f_sel = c1.selectbox("분포 비교 Feature", num_feats, index=0)
            kind = c2.radio("그래프 유형", ["Boxplot", "Histogram"], horizontal=True)

            if kind == "Boxplot":
                fig_box = px.box(
                    df,
                    x=label_col,
                    y=f_sel,
                    points=False,
                    title=f"[Boxplot] {f_sel} by {label_col}",
                )
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                fig_hist = px.histogram(
                    df,
                    x=f_sel,
                    color=label_col,
                    barmode="overlay",
                    nbins=40,
                    opacity=0.6,
                    title=f"[Histogram] {f_sel} distribution by {label_col}",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("---")
            st.markdown("**라벨별 평균 비교 (Top-K Feature)**")
            k = st.slider("Top-K Feature", 3, min(10, len(num_feats)), 5, 1)
            grp_mean = df.groupby(label_col)[num_feats].mean().T
            diff_score = (grp_mean.max(axis=1) - grp_mean.min(axis=1)).sort_values(
                ascending=False
            )
            topk = list(diff_score.head(k).index)

            fig_bar = go.Figure()
            for lab in ["Low", "Medium", "High"]:
                if lab in grp_mean.columns:
                    fig_bar.add_bar(name=lab, x=topk, y=grp_mean[lab].loc[topk])
            fig_bar.update_layout(
                title="라벨별 평균값 비교 (Top-K)", barmode="group", xaxis_tickangle=-30
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ③ 데이터 미리보기
with tabs[2]:
    st.subheader("예측 결과 데이터 미리보기")
    if st.session_state.pred_df is not None:
        st.dataframe(st.session_state.pred_df.head(1000), use_container_width=True, height=420)
        st.write("**기술 통계 (수치형)**")
        st.dataframe(st.session_state.pred_df.describe().T, use_container_width=True)
