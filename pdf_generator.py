import os
import matplotlib.pyplot as plt
from fpdf import FPDF


def generate_anomaly_pdf(df, summary_df, anomalies_mask, active_method):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)

    pdf.set_font("Helvetica", style="B", size=14)
    pdf.cell(0, 10, "Mining Anomaly Report", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 8, f"Detection method: {active_method}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", style="B", size=10)
    pdf.cell(0, 8, "Summary Statistics", ln=True)
    pdf.set_font("Helvetica", size=8)

    cols = ["Metric"] + list(summary_df.columns)
    col_w = 190 / len(cols)
    for c in cols:
        pdf.cell(col_w, 6, str(c)[:12], border=1)
    pdf.ln()

    for idx, row in summary_df.iterrows():
        pdf.cell(col_w, 6, str(idx)[:15], border=1)
        for val in row:
            pdf.cell(col_w, 6, f"{val:.1f}" if isinstance(val, (int, float)) else str(val), border=1)
        pdf.ln()
    pdf.ln(6)

    plt.figure(figsize=(7, 3))
    for c in summary_df.columns:
        if c in df.columns:
            plt.plot(df.index, df[c], label=c)
    plt.title("Output Trend")
    plt.tight_layout()
    plt.savefig("temp_chart.png")
    plt.close()

    pdf.image("temp_chart.png", w=180)
    pdf.ln(6)
    if os.path.exists("temp_chart.png"):
        os.remove("temp_chart.png")

    pdf.set_font("Helvetica", style="B", size=10)
    pdf.cell(0, 8, "Detected Anomalies (Spikes / Drops)", ln=True)
    pdf.set_font("Helvetica", size=8)

    found_any = False
    mine_cols = [c for c in anomalies_mask.columns if c in df.columns]

    for col in mine_cols:
        mask = anomalies_mask[col].fillna(False).astype(bool)
        if not mask.any():
            continue

        found_any = True
        mean_val = df[col].mean()
        pdf.set_font("Helvetica", style="B", size=9)
        pdf.cell(0, 6, f"Mine: {col}", ln=True)
        pdf.set_font("Helvetica", size=8)

        for date_idx, row in df[mask].iterrows():
            date_val = row["Date"] if "Date" in df.columns else str(date_idx)
            val = row[col]
            anom_type = "SPIKE" if val > mean_val else "DROP"
            pdf.cell(0, 5, f"- Date: {date_val} | Val: {val:.2f} (Mean: {mean_val:.2f}) | Type: {anom_type}", ln=True)

    if not found_any:
        pdf.cell(0, 5, "No anomalies detected with the current parameters.", ln=True)

    return bytes(pdf.output())