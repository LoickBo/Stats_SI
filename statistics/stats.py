import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import statsmodels.api as sm
from typing import List, Optional, Tuple
import os
from tqdm import tqdm

def _pairwise_clean(x: pd.Series, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(x.values) & np.isfinite(y.values)
    return x.values[m], y.values[m]

def _safe_pearson(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    xv, yv = _pairwise_clean(x, y)
    n = xv.size
    if n < 3:
        return np.nan, np.nan, n
    if np.all(xv == xv[0]) or np.all(yv == yv[0]):
        # One of the arrays is constant
        return np.nan, np.nan, n
    r, p = stats.pearsonr(xv, yv)
    return r, p, n

def _safe_spearman(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    xv, yv = _pairwise_clean(x, y)
    n = xv.size
    if n < 3:
        return np.nan, np.nan, n
    if np.all(xv == xv[0]) or np.all(yv == yv[0]):
        return np.nan, np.nan, n
    r, p = stats.spearmanr(xv, yv)
    return r, p, n

def _safe_kendall(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    xv, yv = _pairwise_clean(x, y)
    n = xv.size
    if n < 3:
        return np.nan, np.nan, n
    if np.all(xv == xv[0]) or np.all(yv == yv[0]):
        return np.nan, np.nan, n
    r, p = stats.kendalltau(xv, yv)
    return r, p, n

def _safe_mi(x: pd.Series, y: pd.Series) -> float:
    xv, yv = _pairwise_clean(x, y)
    if xv.size < 3:
        return np.nan
    X = xv.reshape(-1, 1)
    mi = mutual_info_regression(X, yv, discrete_features=False, random_state=0)
    return float(mi[0])

def _safe_ols(x: pd.Series, y: pd.Series):
    xv, yv = _pairwise_clean(x, y)
    n = xv.size
    if n < 3 or np.all(xv == xv[0]):
        return np.nan, np.nan, np.nan, np.nan, np.nan, n
    X = sm.add_constant(xv)
    model = sm.OLS(yv, X).fit()
    slope = model.params[1]
    intercept = model.params[0]
    r2 = model.rsquared
    adj_r2 = model.rsquared_adj
    pval = model.pvalues[1]
    return slope, intercept, r2, adj_r2, pval, n

def _shift_series(s: pd.Series, lag: int) -> pd.Series:
    return s.shift(lag) if lag else s

def compute_all_impacts(
    input_file_path: str,
    output_file_path: str,
    ticker: str,
    date_col: str = "DATE",
    lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    df = pd.read_csv(input_file_path)

    # Coerce numerics except date col
    for c in df.columns:
        if c != date_col:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    target_col = f"{ticker}_PX_LAST"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {input_file_path}.")

    if not lags:
        lags = [0]

    # numeric features excluding the target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c != target_col]

    rows = []
    y = df[target_col]

    for feat in features:
        x0 = df[feat]
        # skip degenerate columns (all NA or constant)
        if x0.notna().sum() < 3 or x0.dropna().nunique() <= 1:
            continue
        for lag in lags:
            x = _shift_series(x0, lag)
            pearson_r, pearson_p, n_p = _safe_pearson(x, y)
            spearman_r, spearman_p, n_s = _safe_spearman(x, y)
            kendall_t, kendall_p, n_k = _safe_kendall(x, y)
            mi = _safe_mi(x, y)
            slope, intercept, r2, adj_r2, slope_p, n_o = _safe_ols(x, y)
            n_pairs = np.nanmax([n_p, n_s, n_k, n_o])

            rows.append({
                "feature": feat,
                "lag": lag,
                "n_pairs": int(n_pairs) if np.isfinite(n_pairs) else np.nan,
                "pearson_r": pearson_r, "pearson_p": pearson_p,
                "spearman_rho": spearman_r, "spearman_p": spearman_p,
                "kendall_tau": kendall_t, "kendall_p": kendall_p,
                "mutual_info": mi,
                "ols_slope": slope, "ols_intercept": intercept,
                "ols_r2": r2, "ols_adj_r2": adj_r2, "ols_slope_p": slope_p,
                "abs_pearson": abs(pearson_r) if np.isfinite(pearson_r) else np.nan,
                "abs_spearman": abs(spearman_r) if np.isfinite(spearman_r) else np.nan,
                "abs_kendall": abs(kendall_t) if np.isfinite(kendall_t) else np.nan,
            })

    # Build DataFrame
    cols = [
        "feature","lag","n_pairs",
        "pearson_r","pearson_p","spearman_rho","spearman_p",
        "kendall_tau","kendall_p","mutual_info",
        "ols_slope","ols_intercept","ols_r2","ols_adj_r2","ols_slope_p",
        "abs_pearson","abs_spearman","abs_kendall",
    ]
    out = pd.DataFrame(rows, columns=cols)

    # Handle empty results (no usable features)
    if out.empty:
        print(f"ℹ️  {ticker}: no usable numeric features besides {target_col}. Writing empty stats.")
        out.to_csv(output_file_path, index=False)
        return out

    # Sort by |Pearson| at the primary sort lag
    sort_lag = 0 if 0 in lags else lags[0]
    sort_mask = (out["lag"] == sort_lag)
    out = pd.concat([
        out.loc[sort_mask].sort_values("abs_pearson", ascending=False),
        out.loc[~sort_mask]
    ], ignore_index=True)

    print(f"✅ {ticker} stats computed successfully!")
    out.to_csv(output_file_path, index=False)
    return out


tickers = pd.read_csv('files_creation/data/input/tickers_cleaned.csv')['Ticker'].tolist()

output_dir = 'statistics/data/output_lags0'
os.makedirs(output_dir, exist_ok=True)

input_dir = 'files_creation/data/output_cleaned'
os.makedirs(input_dir, exist_ok=True)

for ticker in tqdm(tickers, desc="Processing tickers"):
    output_file = os.path.join(output_dir, f'{ticker}_stats.csv')
    input_file = os.path.join(input_dir, f'{ticker}_cleaned.csv')

    if os.path.exists(output_file):
        print(f"✅ Skipping {ticker} (already computed stats).")
        continue
    else :
        print(f"\nComputing stats data for {ticker}...")

    compute_all_impacts(input_file, output_file, ticker)


