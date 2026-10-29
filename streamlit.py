import streamlit as st
import pandas as pd
from main import (
    get_iqr_mask,
    get_zscore_mask,
    get_ma_mask,
    get_grubbs_mask,
)
from chart import build_mine_chart
from pdf_generator import generate_anomaly_pdf

st.set_page_config(page_title="Mining Data Dashboard", layout="wide")
st.title("Mining Output Analysis Dashboard")

URL = "https://docs.google.com/spreadsheets/d/1FIXlCHaGddNVS9xkLC_pnWso8RDQT2J6GbZ6DGmLpXk/export?format=csv&gid=1028194232"

df = pd.read_csv(URL)
mines = df.drop(columns=["Date"])

st.sidebar.header("Anomaly Test Parameters")

k_val = st.sidebar.slider("IQR multiplier (k)", min_value=0.1, max_value=3.0, value=1.5, step=0.1)
z_threshold = st.sidebar.slider("Z-score threshold", min_value=0.5, max_value=4.0, value=2.0, step=0.1)
ma_window = st.sidebar.slider("Moving average window (days)", min_value=2, max_value=15, value=5, step=1)
ma_pct = st.sidebar.slider("Moving average distance (%)", min_value=5.0, max_value=60.0, value=20.0, step=1.0)
grubbs_alpha = st.sidebar.slider("Grubbs test alpha", min_value=0.01, max_value=0.50, value=0.05, step=0.01)

iqr_mask = get_iqr_mask(mines, k=k_val)
z_mask = get_zscore_mask(mines, threshold=z_threshold)
ma_mask = get_ma_mask(mines, window=int(ma_window), threshold_pct=ma_pct)
grubbs_mask = get_grubbs_mask(mines, alpha=grubbs_alpha)

data = {
    "Mean daily output": mines.mean(),
    "Standard deviation": mines.std(),
    "Median": mines.median(),
    "Interquartile range": mines.quantile(0.75) - mines.quantile(0.25),
    "IQR Rule": iqr_mask.sum(),
    "Z-score": z_mask.sum(),
    "Moving average anomaly": ma_mask.sum(),
    "Grubbs' test": grubbs_mask.sum(),
}

summary_df = pd.DataFrame(data).T.round(2)

st.subheader("Summary Statistics and Detected Anomalies")
st.dataframe(summary_df, width="stretch")

st.subheader("Visualization")

col1, col2, col3, col4 = st.columns([2, 1, 1, 1.2], vertical_alignment="bottom")

with col1:
    selected_cols = st.multiselect(
        "Mines",
        options=list(mines.columns),
        default=[mines.columns[0]],
    )
with col2:
    chart_type = st.selectbox("Chart type", options=["Line", "Bar", "Stacked Bar"])
with col3:
    trend_degree = st.selectbox("Trend degree", options=[1, 2, 3, 4])
with col4:
    highlight_method = st.selectbox(
        "Outliers method",
        options=["Moving average", "IQR Rule", "Z-score", "Grubbs' test"],
    )

masks_map = {
    "IQR Rule": iqr_mask,
    "Z-score": z_mask,
    "Moving average": ma_mask,
    "Grubbs' test": grubbs_mask,
}
active_mask = masks_map[highlight_method]

if selected_cols:
    fig = build_mine_chart(
        df=df,
        anomalies_mask=active_mask,
        selected_cols=selected_cols,
        chart_type=chart_type,
        trend_degree=trend_degree,
    )
    st.plotly_chart(fig, width="stretch")

    pdf_bytes = generate_anomaly_pdf(
        df=df,
        summary_df=summary_df,
        anomalies_mask=active_mask,
        active_method=highlight_method
    )

    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="mining_anomaly_report.pdf",
        mime="application/pdf"
    )
else:
    st.warning("Please select at least one column to display the chart.")