import pandas as pd
import numpy as np
from scipy import stats

def get_iqr_mask(mines_df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    q1 = mines_df.quantile(0.25)
    q3 = mines_df.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - k * iqr
    higher = q3 + k * iqr

    return (mines_df < lower) | (mines_df > higher)


def get_zscore_mask(mines_df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    mean = mines_df.mean()
    std = mines_df.std()
    z = (mines_df - mean) / std
    return z.abs() > threshold


def get_ma_mask(mines_df: pd.DataFrame, window: int = 5, threshold_pct: float = 20.0) -> pd.DataFrame:
    ma = mines_df.rolling(window=window).mean()
    pct_diff = ((mines_df - ma).abs() / ma) * 100
    return pct_diff > threshold_pct


def get_grubbs_mask(mines_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=mines_df.index, columns=mines_df.columns)

    for col in mines_df.columns:
        series = mines_df[col].dropna().copy()

        while len(series) > 2:
            n = len(series)
            mean = series.mean()
            std = series.std(ddof=1)

            if std == 0:
                break

            diffs = (series - mean).abs()
            max_idx = diffs.idxmax()
            g_stat = diffs.loc[max_idx] / std

            t_crit = stats.t.ppf(1 - alpha / (2 * n), df=n - 2)
            g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(
                (t_crit**2) / (n - 2 + t_crit**2)
            )

            if g_stat > g_crit:
                mask.loc[max_idx, col] = True
                series = series.drop(index=max_idx)
            else:
                break

    return mask