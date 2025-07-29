import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
import joblib

st.set_page_config(page_title="ROA 시뮬레이터", layout="wide")

# 제목
st.title("iM뱅크 비이자수익 기반 ROA 시뮬레이터")
st.markdown("*2025년 1분기 기준 데이터를 바탕으로 전략적 조정 시뮬레이션*")

# 📌 입력 및 조정
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("1. 기본 수익/비용 항목 설정")
    interest_ratio = st.slider("이자수익 비중 (%)", 0.0, 100.0, 90.8, step=0.5)
    admin_ratio = st.slider("판관비 비중 (%)", 0.0, 100.0, 18.7, step=0.5)

    st.subheader("2. 조정할 비이자수익 항목 선택")
    options = ["수수료수익", "외환이익", "신탁수익", "유가증권손익"]
    selected = st.selectbox("조정할 항목", options)

    default_ratios = {
        "수수료수익": 41.0,
        "외환이익": 15.0,
        "신탁수익": 6.0,
        "유가증권손익": 38.0
    }

    selected_value = st.slider(f"{selected} 비중 (%)", 0.0, 100.0, value=default_ratios[selected], step=0.5)
    change = selected_value - default_ratios[selected]
    change_str = f"{change:+.1f}%p 변화 (기준: {default_ratios[selected]}%)"

    st.markdown(f"""
    <div style="font-size:16px; color:#00C7A9; font-weight:bold; margin-top:10px;">
    조정 결과: {change_str}
    </div>
    """, unsafe_allow_html=True)

# 📐 비중 조정 계산
delta = selected_value - default_ratios[selected]
reduction = round(delta / 3, 2)

ratios = {}
for item in options:
    if item == selected:
        ratios[item] = selected_value
    else:
        new_val = round(default_ratios[item] - reduction, 2)
        ratios[item] = max(0.0, new_val)

# 🔍 예측 준비
model = joblib.load("xgb_roa_model.pkl")
poly = joblib.load("poly_features.pkl")
scaler = joblib.load("scaler.pkl")

input_data = pd.DataFrame({
    "이자수익_비중": [interest_ratio / 100],
    "수수료수익_비중": [ratios["수수료수익"] / 100],
    "외환수익_비중": [ratios["외환이익"] / 100],
    "유가증권손익_비중": [ratios["유가증권손익"] / 100],
    "신탁수익_비중": [ratios["신탁수익"] / 100],
    "판관비_비중": [admin_ratio / 100]
})

X_poly = poly.transform(input_data)
X_scaled = scaler.transform(X_poly)
pred_roa = model.predict(X_scaled)[0]


default_input = pd.DataFrame({
    "이자수익_비중": [90.8 / 100],
    "수수료수익_비중": [41.0 / 100],
    "외환수익_비중": [15.0 / 100],
    "유가증권손익_비중": [38.0 / 100],
    "신탁수익_비중": [6.0 / 100],
    "판관비_비중": [18.7 / 100]
})
X_poly_base = poly.transform(default_input)
X_scaled_base = scaler.transform(X_poly_base)
baseline_roa = model.predict(X_scaled_base)[0]


# 변화량 계산
delta_roa = pred_roa - baseline_roa
delta_percent = (delta_roa / baseline_roa) * 100 if baseline_roa != 0 else 0

# 포맷
delta_str = f"{delta_roa:+.4f}"
percent_str = f"{delta_percent:+.1f}%"

# 👉 표시
with col_right:
    st.subheader("3. 예측된 ROA 결과")
    
    st.markdown(f"""
    <div style='
        background-color:#00C7A9;
        border-left:6px solid #E2F15E;
        padding: 25px 35px;
        border-radius: 8px;
        margin-top: 30px;
        width: 100%;
    '>
        <h3 style='margin-bottom:10px; color:#003c3c;'>예측 ROA</h3>
        <p style='font-size:42px; font-weight:bold; color:#003c3c; margin:0 0 20px 0;'>
            {pred_roa:.4f}
        </p>
        <table style="
            width:100%;
            font-size:18px;
            color:#111;
            border-collapse:collapse;
            border: 2px solid black;
            text-align: center;
        ">
            <tr style="border-bottom:2px solid black;">
                <th style="padding:10px;">기준값</th>
                <th style="padding:10px;">{baseline_roa:.4f}</th>
            </tr>
            <tr style="border-bottom:2px solid black;">
                <td style="padding:10px;">차이</td>
                <td style="padding:10px;">{delta_str} ({percent_str})</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# 📊 시각화
st.markdown("---")
st.subheader("4. 비이자수익 비중 시각화")

df = pd.DataFrame({"항목": list(ratios.keys()), "비중": list(ratios.values())})

col1, col2 = st.columns(2)

# 📊 파이차트
with col1:
    st.markdown("**파이차트**")
    fig_pie = px.pie(
        df,
        names="항목",
        values="비중",
        color_discrete_sequence=px.colors.sequential.RdBu,
        hole=0.3
    )
    fig_pie.update_traces(
        textfont_size=16,  # 숫자 라벨 크기
        textinfo='percent+label'
    )
    fig_pie.update_layout(
        font=dict(size=18, color="white"),
        paper_bgcolor="black",
        plot_bgcolor="black",
        title=dict(text="비이자수익 구성 비율", font=dict(size=22))
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# 📊 막대그래프 (Altair)
with col2:
    st.markdown("**막대그래프**")
    bar_chart = alt.Chart(df).mark_bar(size=40).encode(
        x=alt.X("항목:N", sort="-y", title="항목", axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
        y=alt.Y("비중:Q", title="비중 (%)", axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
        color=alt.Color("항목:N", legend=None),
        tooltip=[alt.Tooltip("항목:N", title="항목"), alt.Tooltip("비중:Q", title="비중", format=".1f")]
    ).properties(
        width=500,
        height=500,
        title=alt.TitleParams(
            text="비이자수익 항목별 비중 막대그래프",
            fontSize=20,
            anchor='start',
            font='sans-serif'
        )
    )
    st.altair_chart(bar_chart, use_container_width=True)
# 📋 테이블
st.markdown("---")
st.subheader("5. 비이자수익 항목별 비중 테이블")
st.dataframe(df.set_index("항목"))
