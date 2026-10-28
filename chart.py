import numpy as np
import pandas as pd
import plotly.graph_objects as go


def build_mine_chart(
    df: pd.DataFrame,
    anomalies_mask: pd.DataFrame,
    selected_cols: list,
    chart_type: str = "Line",
    trend_degree: int = 1,
):
    fig = go.Figure()

    if "Date" in df.columns:
        x_axis = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    else:
        x_axis = df.index

    x_numeric = np.arange(len(df))

    cols_to_plot = (
        [c for c in selected_cols if c != "Total"]
        if chart_type == "Stacked Bar" and len(selected_cols) > 1
        else selected_cols
    )

    for col in cols_to_plot:
        y_vals = df[col].values

        if chart_type == "Line":
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=y_vals,
                    mode="lines+markers",
                    name=col,
                    marker=dict(size=5),
                )
            )
        elif chart_type in ["Bar", "Stacked Bar"]:
            fig.add_trace(go.Bar(x=x_axis, y=y_vals, name=col))

        if col in anomalies_mask.columns:
            outliers = anomalies_mask[col].values.astype(bool)
            if np.any(outliers):
                fig.add_trace(
                    go.Scatter(
                        x=x_axis[outliers],
                        y=y_vals[outliers],
                        mode="markers",
                        name=f"{col} Outliers",
                        marker=dict(
                            color="red",
                            size=11,
                            symbol="x",
                        ),
                    )
                )

        if trend_degree in [1, 2, 3, 4]:
            valid_idx = ~np.isnan(y_vals)
            if np.sum(valid_idx) > trend_degree:
                coeffs = np.polyfit(
                    x_numeric[valid_idx], y_vals[valid_idx], deg=trend_degree
                )
                trend_y = np.polyval(coeffs, x_numeric)

                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=trend_y,
                        mode="lines",
                        name=f"{col} Trend (deg {trend_degree})",
                        line=dict(dash="dash", width=2),
                    )
                )

    barmode = "stack" if chart_type == "Stacked Bar" else "group"
    fig.update_layout(
        title=f"Mines Output ({chart_type}) - Trendline Degree {trend_degree}",
        barmode=barmode,
        xaxis_title="Date",
        yaxis_title="Output",
        hovermode="x unified",
        height=550,
    )

    return fig