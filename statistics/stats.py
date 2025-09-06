import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import statsmodels.api as sm
from typing import List, Optional, Tuple

def _pairwise_clean(x: pd.Series, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Return aligned finite numpy arrays for pairwise computations."""
    m = np.isfinite(x.values) & np.isfinite(y.values)
    return x.values[m], y.values[m]

def _safe_pearson(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    xv, yv = _pairwise_clean(x, y)
    if xv.size < 3:
        return np.nan, np.nan, xv.size
    r, p = stats.pearsonr(xv, yv)
    return r, p, xv.size

def _safe_spearman(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    xv, yv = _pairwise_clean(x, y)
    if xv.size < 3:
        return np.nan, np.nan, xv.size
    r, p = stats.spearmanr(xv, yv)
    return r, p, xv.size

def _safe_kendall(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    xv, yv = _pairwise_clean(x, y)
    if xv.size < 3:
        return np.nan, np.nan, xv.size
    r, p = stats.kendalltau(xv, yv)
    return r, p, xv.size

def _safe_mi(x: pd.Series, y: pd.Series) -> float:
    """Mutual information (continuous target). Returns np.nan if <3 samples."""
    xv, yv = _pairwise_clean(x, y)
    if xv.size < 3:
        return np.nan
    # reshape for sklearn
    X = xv.reshape(-1, 1)
    mi = mutual_info_regression(X, yv, discrete_features=False, random_state=0)
    return float(mi[0])

def _safe_ols(x: pd.Series, y: pd.Series) -> Tuple[float, float, float, float, float, int]:
    """
    Simple OLS: y = a + b*x. Returns slope b, intercept a, R2, adj_R2, pval(b), n.
    """
    xv, yv = _pairwise_clean(x, y)
    n = xv.size
    if n < 3 or np.all(xv == xv[0]):
        return np.nan, np.nan, np.nan, np.nan, np.nan, n
    X = sm.add_constant(xv)  # [1, x]
    model = sm.OLS(yv, X).fit()
    slope = model.params[1]
    intercept = model.params[0]
    r2 = model.rsquared
    adj_r2 = model.rsquared_adj
    pval = model.pvalues[1]
    return slope, intercept, r2, adj_r2, pval, n

def _shift_series(s: pd.Series, lag: int) -> pd.Series:
    """Positive lag means feature is shifted backward (feature at t-lag vs price at t)."""
    return s.shift(lag) if lag != 0 else s

def compute_all_impacts(
    input_file_path: str,
    output_file_path: str,
    ticker: str,
    date_col: str = "DATE",
    lags: Optional[List[int]] = None,  # e.g., [0, 1, 5]
) -> pd.DataFrame:
    """
    Compute multiple dependence/impact indicators of each numeric field on {ticker}_PX_LAST.
    
    Metrics per feature (for each lag):
      - Pearson r, p-value
      - Spearman rho, p-value
      - Kendall tau, p-value
      - Mutual Information (MI)
      - OLS slope, intercept, R^2, Adj. R^2, slope p-value
      - n_pairs (sample size used for each metric)
    
    Parameters
    ----------
    input_file_path : str
        CSV path.
    output_file_path : str
        CSV to write results.
    ticker : str
        Ticker prefix. Target column is f"{ticker}_PX_LAST".
    date_col : str
        Date column name (optional).
    lags : list[int] | None
        Lags to evaluate; default [0] (contemporaneous). lag=1 means X_{t-1} vs Y_t.
    
    Returns
    -------
    pd.DataFrame
        Results table sorted by |Pearson r| (lag 0 if present, else first lag).
    """
    df = pd.read_csv(input_file_path)

    # Coerce numerics; leave date as is
    for c in df.columns:
        if c != date_col:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    target_col = f"{ticker}_PX_LAST"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    # Default to contemporaneous metrics only
    if lags is None or len(lags) == 0:
        lags = [0]

    # Candidate features = all numeric except the target itself
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c != target_col]

    # Prepare container for results
    rows = []

    y = df[target_col]

    for feat in features:
        x0 = df[feat]
        for lag in lags:
            x = _shift_series(x0, lag)

            # Metrics
            pearson_r, pearson_p, n_p = _safe_pearson(x, y)
            spearman_r, spearman_p, n_s = _safe_spearman(x, y)
            kendall_t, kendall_p, n_k = _safe_kendall(x, y)
            mi = _safe_mi(x, y)
            slope, intercept, r2, adj_r2, slope_p, n_o = _safe_ols(x, y)

            n_pairs = np.nanmax([n_p, n_s, n_k, n_o])  # mostly identical, but take max valid

            rows.append({
                "feature": feat,
                "lag": lag,  # 0 = contemporaneous; 1 = feature leads price by 1 period
                "n_pairs": int(n_pairs) if np.isfinite(n_pairs) else np.nan,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_r,
                "spearman_p": spearman_p,
                "kendall_tau": kendall_t,
                "kendall_p": kendall_p,
                "mutual_info": mi,
                "ols_slope": slope,
                "ols_intercept": intercept,
                "ols_r2": r2,
                "ols_adj_r2": adj_r2,
                "ols_slope_p": slope_p,
                # Useful for sorting/quick scan:
                "abs_pearson": abs(pearson_r) if np.isfinite(pearson_r) else np.nan,
                "abs_spearman": abs(spearman_r) if np.isfinite(spearman_r) else np.nan,
                "abs_kendall": abs(kendall_t) if np.isfinite(kendall_t) else np.nan,
            })

    out = pd.DataFrame(rows)

    # Sort by |Pearson| at lag 0 if present, else first lag provided
    sort_lag = 0 if 0 in lags else lags[0]
    sort_mask = out["lag"] == sort_lag
    # Place lag==sort_lag rows first and sort by abs_pearson within them
    out = pd.concat([
        out.loc[sort_mask].sort_values("abs_pearson", ascending=False),
        out.loc[~sort_mask]
    ], ignore_index=True)

    out.to_csv(output_file_path, index=False)
    return out


