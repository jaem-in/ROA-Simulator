
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

st.set_page_config(page_title="ROA 민감도 분석 (Altair)", layout="wide")
st.title("ROA 민감도 분석: 비이자수익 항목 변화 (Altair)")

# ✅ 기본값
interest_ratio = 60.0  # %
admin_ratio = 20.0     # %

default_ratios = {
    "수수료수익": 41.0,
    "외환이익": 15.0,
    "신탁수익": 6.0,
    "유가증권손익": 38.0
}
# 0.414381	0.144409	0.056451	0.384758
# ✅ 모델 불러오기
model = joblib.load("xgb_roa_model.pkl")
poly = joblib.load("poly_features.pkl")
scaler = joblib.load("scaler.pkl")

sim_targets = ["수수료수익", "외환이익", "신탁수익", "유가증권손익"]

st.markdown("### 각 항목별 비중 변화에 따른 ROA 민감도 (Altair)")

for target in sim_targets:
    base_val = default_ratios[target]
    x_changes = []
    y_preds = []

    for diff in np.arange(-15, 15.5, 0.5):
        new_val = base_val + diff
        if not (0 <= new_val <= 100):
            continue

        delta = new_val - base_val
        reduction = round(delta / 3, 2)

        sim_ratios = {}
        for item in sim_targets:
            if item == target:
                sim_ratios[item] = new_val
            else:
                sim_ratios[item] = max(0.0, round(default_ratios[item] - reduction, 2))

        sim_input = pd.DataFrame({
            "이자수익_비중": [interest_ratio / 100],
            "수수료수익_비중": [sim_ratios.get("수수료수익", 0) / 100],
            "외환수익_비중": [sim_ratios.get("외환이익", 0) / 100],
            "유가증권손익_비중": [sim_ratios.get("유가증권손익", 0) / 100],
            "신탁수익_비중": [sim_ratios.get("신탁수익", 0) / 100],
            "판관비_비중": [admin_ratio / 100]
        })

        X_poly_sim = poly.transform(sim_input)
        X_scaled_sim = scaler.transform(X_poly_sim)
        pred_sim = model.predict(X_scaled_sim)[0]

        x_changes.append(diff)
        y_preds.append(pred_sim)

    df_altair = pd.DataFrame({
        "Δ비중 (%p)": x_changes,
        "예측 ROA": y_preds
    })

    y_min = min(y_preds)
    y_max = max(y_preds)
    y_margin = (y_max - y_min) * 0.05 if (y_max - y_min) > 0 else 0.01
    y_range = [y_min - y_margin, y_max + y_margin]

    st.markdown(f"#### 🔹 {target} 변화에 따른 ROA")

    chart = alt.Chart(df_altair).mark_line(point=True).encode(
        x=alt.X("Δ비중 (%p):Q", title="비중 변화 (기준 대비 %p)", scale=alt.Scale(domain=[-15, 15])),
        y=alt.Y("예측 ROA:Q", title="예측 ROA", scale=alt.Scale(domain=y_range)),
        tooltip=["Δ비중 (%p)", "예측 ROA"]
    ).properties(
        width=600,
        height=300
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )

    st.altair_chart(chart, use_container_width=True)