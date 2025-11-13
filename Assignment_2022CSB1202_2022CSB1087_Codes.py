# Author: Prashant Kumar, 2022CSB1202
"""
Build and backtest GMV, Tangency, EW, and Active (Treynor–Black) portfolios vs NIFTY 50
with a 6M formation / 3M holding window from 2009–2022, and 99% historical VaR (3M).
Outputs: rolling 3M returns, cumulative growth plot, performance table, and VaR vs realized.
"""
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
import statsmodels.api as sm
import warnings

#

# File names
STOCKS_FILE = "Stocks_data.csv"                # provided daily prices (levels)
FACTORS_FILE = "market_Factor_risk_Free.csv"   # MF (percent), RF (decimal)

# Output artifact names
OUTPUT_RETURNS_FILE = "portfolio_holding_returns_3m.csv"
OUTPUT_CUMRET_PLOT = "cumulative_growth_series.png"
OUTPUT_PERF_FILE = "performance_summary.csv"
OUTPUT_VAR_PLOT = "historical_var_vs_realized_returns.png"

# Helpers
def _to_datetime_series(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if dt.isna().mean() > 0.3:
        dt = pd.to_datetime(series, errors="coerce", dayfirst=False)
    return dt


def compound_holding_return(returns: np.ndarray) -> float:
    # Stable compounding of simple returns
    if returns is None or len(returns) == 0:
        return float("nan")
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.expm1(np.nansum(np.log1p(arr))))


def rolling_compound_simple(series: np.ndarray, L: int) -> np.ndarray:
    # Vectorized rolling compounding over L days
    arr = np.asarray(series, dtype=float)
    n = arr.shape[0]
    if L is None or L <= 0 or n < L:
        return np.array([], dtype=float)
    base = 1.0 + arr
    cp = np.cumprod(base)
    denom = np.concatenate(([1.0], cp[:-L]))
    window_prod = cp[L - 1:] / denom
    return window_prod - 1.0


def clean_universe(prices: pd.DataFrame, max_missing_frac: float = 0.0) -> pd.DataFrame:
    """Drop series with missing values (over threshold) and near-constant returns."""
    prices_clean = prices.copy()
    n_rows = len(prices_clean)
    missing_frac = prices_clean.isna().sum(axis=0) / float(max(1, n_rows))
    keep_cols = missing_frac[missing_frac <= max_missing_frac].index.tolist()

    idx_cols = [c for c in prices_clean.columns if "NIFTY" in c.upper()]
    for idx in idx_cols:
        if idx not in keep_cols:
            keep_cols.append(idx)

    prices_clean = prices_clean.loc[:, [c for c in keep_cols if c in prices_clean.columns]]

    rets = prices_clean.pct_change()
    rets_std = rets.std(axis=0, ddof=1, skipna=True)
    nonconstant = rets_std[rets_std > 1e-12].index.tolist()

    nonconstant = list(dict.fromkeys(nonconstant + [c for c in idx_cols if c in prices_clean.columns]))

    prices_clean = prices_clean.loc[:, [c for c in prices_clean.columns if c in nonconstant]]

    return prices_clean


def pseudo_inverse_safe(mat: np.ndarray, ridge: float = 1e-8, max_iters: int = 6, cond_threshold: float = 1e12) -> np.ndarray:
    """Pseudo-inverse with adaptive ridge if ill-conditioned."""
    m = np.array(mat, dtype=float, copy=True)

    if not np.isfinite(m).all():
        m = np.where(np.isfinite(m), m, 0.0)
        ridge = max(ridge, 1e-6)

    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return linalg.pinv(m)

    cur_ridge = float(ridge)
    inv = None
    for _ in range(max_iters):
        try:
            reg = m + cur_ridge * np.eye(m.shape[0], dtype=float)
            inv = linalg.pinv(reg)
            cond = np.linalg.cond(reg)
            if np.isfinite(cond) and cond < cond_threshold and np.isfinite(inv).all():
                return inv
            cur_ridge *= 10.0
        except Exception:
            cur_ridge *= 10.0
            continue

    inv = np.linalg.pinv(m)
    if not np.isfinite(inv).all():
        n = m.shape[0]
        diag = np.diag(m)
        diag_safe = np.where(np.isfinite(diag) & (np.abs(diag) > 1e-12), diag, 1e-6)
        return np.diag(1.0 / diag_safe)
    return inv


def stabilise_covariance(cov: np.ndarray, min_eig: float = 1e-8, ridge: float = 1e-8) -> np.ndarray:
    """Make covariance numeric, symmetric, PSD, and well-conditioned."""
    A = np.array(cov, dtype=float, copy=True)

    if not np.isfinite(A).all():
        A = np.nan_to_num(A, nan=0.0, posinf=1e6, neginf=-1e6)

    A = 0.5 * (A + A.T)

    try:
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals_clipped = np.clip(eigvals, min_eig, None)
        A = (eigvecs * eigvals_clipped) @ eigvecs.T
    except Exception:
        A = A + ridge * np.eye(A.shape[0], dtype=float)

    A = A + ridge * np.eye(A.shape[0], dtype=float)

    A = 0.5 * (A + A.T)

    if not np.isfinite(A).all():
        n = A.shape[0]
        A = np.eye(n, dtype=float) * max(np.nanmedian(np.diag(A[np.isfinite(np.diag(A))])) if np.isfinite(A).any() else 1e-6, 1e-6)

    return A


def weights_gmv(cov: np.ndarray) -> np.ndarray:
    """GMV weights under sum(weights)=1."""
    try:
        inv_cov = pseudo_inverse_safe(cov)
        ones = np.ones((cov.shape[0], 1), dtype=float)
        numer = inv_cov @ ones
        denom_arr = ones.T @ inv_cov @ ones
        try:
            denom = float(denom_arr.item())
        except Exception:
            denom = float(np.asarray(denom_arr).ravel()[0])
        if not np.isfinite(denom) or abs(denom) < 1e-16:
            return weights_equal(cov.shape[0])
        w = (numer / denom).ravel()
        if not np.isfinite(w).all():
            return weights_equal(cov.shape[0])
        return w
    except Exception:
        return weights_equal(cov.shape[0])


def weights_tangency(mu: np.ndarray, cov: np.ndarray, rf: float) -> np.ndarray:
    """Tangency weights (sum=1), fallback to GMV if unstable."""
    try:
        inv_cov = pseudo_inverse_safe(cov)
        ones = np.ones_like(mu, dtype=float)
        targ = mu - rf * ones
        if not np.isfinite(targ).all():
            return weights_gmv(cov)
        k = inv_cov @ targ
        if not np.isfinite(k).all():
            return weights_gmv(cov)
        s = float(np.sum(k))
        if abs(s) < 1e-12 or not np.isfinite(s):
            return weights_gmv(cov)
        w = (k / s).ravel()
        if not np.isfinite(w).all():
            return weights_gmv(cov)
        return w
    except Exception:
        return weights_gmv(cov)


def weights_equal(n: int) -> np.ndarray:
    return np.ones(n) / float(n)


@dataclass
class BacktestSpan:
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    holding_start: pd.Timestamp
    holding_end: pd.Timestamp


def generate_windows(start_year: int = 2009, end_year: int = 2022) -> List[BacktestSpan]:
    windows: List[BacktestSpan] = []
    current = pd.Timestamp(year=start_year, month=1, day=1)
    last_holding_end = pd.Timestamp(year=end_year, month=12, day=31)
    while True:
        form_start = current.normalize().replace(day=1)
        form_end = (form_start + pd.DateOffset(months=6) - pd.DateOffset(days=1)).normalize()
        hold_start = (form_end + pd.DateOffset(days=1)).normalize()
        hold_end = (hold_start + pd.DateOffset(months=3) - pd.DateOffset(days=1)).normalize()
        if hold_end > last_holding_end:
            break
        windows.append(BacktestSpan(
            formation_start=form_start,
            formation_end=form_end,
            holding_start=hold_start,
            holding_end=hold_end,
        ))
        current = current + pd.DateOffset(months=3)
    return windows


def load_data(stocks_path: str, factors_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prices_raw = pd.read_csv(stocks_path)
    date_col = [c for c in prices_raw.columns if c.lower().startswith("date") or c.lower().startswith("dates")]
    if not date_col:
        raise ValueError("Could not find 'Date' column in stocks file.")
    date_col = date_col[0]
    prices_raw[date_col] = _to_datetime_series(prices_raw[date_col])
    prices_raw = prices_raw.dropna(subset=[date_col])
    prices_raw = prices_raw.set_index(date_col).sort_index()
    for c in prices_raw.columns:
        prices_raw[c] = pd.to_numeric(prices_raw[c], errors="coerce")
    idx_cols = [c for c in prices_raw.columns if "NIFTY" in c.upper()]
    if not idx_cols:
        raise ValueError("Could not find NIFTY index column in stocks file.")
    nifty_col = idx_cols[-1]
    prices_raw = clean_universe(prices_raw, max_missing_frac=0.0)
    idx_cols_left = [c for c in prices_raw.columns if "NIFTY" in c.upper()]
    if not idx_cols_left:
        raise ValueError("NIFTY index column removed by cleaning. Check input data.")
    nifty_col = idx_cols_left[-1]

    factors = pd.read_csv(factors_path)
    if "Date" not in factors.columns:
        f_date_col = [c for c in factors.columns if c.lower() == "date"]
        if not f_date_col:
            raise ValueError("Could not find 'Date' in factors file.")
        factors = factors.rename(columns={f_date_col[0]: "Date"})
    factors["Date"] = _to_datetime_series(factors["Date"]).dt.normalize()
    factors = factors.dropna(subset=["Date"]).set_index("Date").sort_index()
    if "MF" not in factors.columns or "RF" not in factors.columns:
        raise ValueError("Factors file must contain columns 'MF' and 'RF'.")
    factors["MF"] = pd.to_numeric(factors["MF"], errors="coerce") / 100.0
    factors["RF"] = pd.to_numeric(factors["RF"], errors="coerce")
    return prices_raw, factors


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change().dropna(how="all")
    return rets


def align_on_common_dates(stock_rets: pd.DataFrame, factors: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common = stock_rets.index.intersection(factors.index)
    stock_rets = stock_rets.loc[common].sort_index()
    factors = factors.loc[common].sort_index()
    return stock_rets, factors


def filter_by_dates(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df.loc[(df.index >= start) & (df.index <= end)]


#


def rolling_window_portfolio_returns(
    weights: Optional[np.ndarray],
    rets_window: pd.DataFrame,
    L: int,
    market_ret: Optional[pd.Series] = None,
) -> np.ndarray:
    if weights is None:
        if market_ret is None:
            raise ValueError("market_ret required when weights is None")
        series = market_ret.values
        return rolling_compound_simple(series, L)
    port_daily = np.dot(rets_window.values, weights)
    return rolling_compound_simple(port_daily, L)


def nan_safe_quantile(arr: np.ndarray, q: float) -> Optional[float]:
    """Return quantile or None if arr empty/invalid."""
    if arr is None or len(arr) == 0:
        return None
    try:
        return float(np.quantile(arr, q))
    except Exception:
        return None


def main() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    stocks_path = os.path.join(here, STOCKS_FILE)
    factors_path = os.path.join(here, FACTORS_FILE)

    prices_raw, factors = load_data(stocks_path, factors_path)
    all_stock_returns = compute_simple_returns(prices_raw)

    index_cols = [c for c in all_stock_returns.columns if "NIFTY" in c.upper()]
    if not index_cols:
        raise RuntimeError("NIFTY index missing after cleaning/returns. Check input data.")
    market_col = index_cols[-1]

    all_stock_returns, factors = align_on_common_dates(all_stock_returns, factors)
    spans = generate_windows(2009, 2022)

    results_rows: List[Dict[str, object]] = []
    portfolio_var_estimates: Dict[str, List[float]] = {"GMV": [], "MV": [], "EW": [], "Active": []}
    portfolio_realized_returns: Dict[str, List[float]] = {"GMV": [], "MV": [], "EW": [], "Active": []}
    window_endpoints: List[pd.Timestamp] = []

    for ws in spans:
        formation_returns = filter_by_dates(all_stock_returns, ws.formation_start, ws.formation_end)
        holding_returns = filter_by_dates(all_stock_returns, ws.holding_start, ws.holding_end)
        formation_factors = filter_by_dates(factors, ws.formation_start, ws.formation_end)
        holding_factors = filter_by_dates(factors, ws.holding_start, ws.holding_end)

        if len(formation_returns) < 20 or len(holding_returns) < 20:
            continue

        stock_cols = [c for c in formation_returns.columns if c != market_col]
        formation_stock_returns = formation_returns[stock_cols].dropna(axis=1, how="any")

        # Remove near-constant series
        stds = formation_stock_returns.std(axis=0, ddof=1, skipna=True)
        non_const_cols = stds[stds > 1e-12].index.tolist()
        formation_stock_returns = formation_stock_returns[non_const_cols]
        if formation_stock_returns.shape[1] < 3:
            continue

        holding_stock_returns = holding_returns[formation_stock_returns.columns].dropna(how="any")
        formation_market = formation_returns[market_col].dropna()
        holding_market = holding_returns[market_col].dropna()

        mean_return = formation_stock_returns.mean(axis=0).values
        cov_matrix = np.cov(formation_stock_returns.values, rowvar=False)

        # --- PREPROCESS COVARIANCE for numerical stability ---
        cov_matrix = stabilise_covariance(cov_matrix, min_eig=1e-8, ridge=1e-8)

        rf_mean_daily = float(formation_factors["RF"].mean()) if "RF" in formation_factors.columns else 0.0

        # Passive portfolios (weights validated and labeled)
        w_gmv = weights_gmv(cov_matrix)
        w_mv = weights_tangency(mean_return, cov_matrix, rf_mean_daily)
        w_ew = weights_equal(formation_stock_returns.shape[1])

        n_assets = formation_stock_returns.shape[1]

        def _validate_and_normalize(w: np.ndarray, n: int) -> np.ndarray:
            if not isinstance(w, np.ndarray):
                w = np.asarray(w, dtype=float)
            if w.ndim > 1:
                w = w.ravel()
            if w.size != n:
                return weights_equal(n)
            if not np.isfinite(w).all():
                w = np.where(np.isfinite(w), w, 0.0)
            s = float(np.sum(w))
            if abs(s) < 1e-12:
                return weights_equal(n)
            w = w / s
            if not np.isfinite(w).all():
                return weights_equal(n)
            return w

        w_gmv = _validate_and_normalize(w_gmv, n_assets)
        w_mv = _validate_and_normalize(w_mv, n_assets)
        w_ew = _validate_and_normalize(w_ew, n_assets)

        w_gmv_s = pd.Series(w_gmv, index=formation_stock_returns.columns)
        w_mv_s = pd.Series(w_mv, index=formation_stock_returns.columns)
        w_ew_s = pd.Series(w_ew, index=formation_stock_returns.columns)

        # ---------------------------
        # Active sleeve (Treynor-Black)
        # ---------------------------
        active_names: List[str] = []
        alpha_by_name: Dict[str, float] = {}
        beta_by_name: Dict[str, float] = {}
        resid_var_by_name: Dict[str, float] = {}

        for col in formation_stock_returns.columns:
            y_excess_full = formation_stock_returns[col] - formation_factors["RF"]
            y_excess = y_excess_full.dropna()
            mf_for_y = formation_factors.loc[y_excess.index, "MF"]

            if len(y_excess) < 10:
                continue

            X = sm.add_constant(mf_for_y.values)
            model = sm.OLS(y_excess.values, X, missing="drop")
            res = model.fit()

            alpha = float(res.params[0])
            p_alpha = float(res.pvalues[0])
            beta = float(res.params[1]) if len(res.params) > 1 else 0.0
            sigma2 = float(res.mse_resid) if hasattr(res, "mse_resid") else float(np.var(res.resid, ddof=0))
            sigma2 = max(sigma2, 1e-12)

            if p_alpha < 0.05 and abs(alpha) > 1e-12:
                active_names.append(col)
                alpha_by_name[col] = alpha
                beta_by_name[col] = beta
                resid_var_by_name[col] = sigma2

        is_tb_active = len(active_names) > 0

        # TB placeholders
        w_sleeve_unit: Optional[np.ndarray] = None
        w0A = 0.0
        betaA = 0.0
        alphaA = 0.0
        sigma2_eA = 0.0
        wstarA = 0.0
        wstarM = 1.0
        final_active_asset_weights: Optional[np.ndarray] = None

        if is_tb_active:
            w0_vec = np.array([alpha_by_name[t] / resid_var_by_name[t] for t in active_names], dtype=float)
            if not np.isfinite(w0_vec).all() or np.allclose(w0_vec, 0.0):
                w_sleeve_unit = weights_equal(len(active_names))
            else:
                total_w0 = float(np.sum(w0_vec))
                if abs(total_w0) < 1e-12:
                    w_sleeve_unit = weights_equal(len(active_names))
                else:
                    w_sleeve_unit = w0_vec / total_w0
                    if not np.isfinite(w_sleeve_unit).all():
                        w_sleeve_unit = weights_equal(len(active_names))

            alphas_arr = np.array([alpha_by_name[t] for t in active_names], dtype=float)
            betas_arr = np.array([beta_by_name[t] for t in active_names], dtype=float)
            sigma2_arr = np.array([resid_var_by_name[t] for t in active_names], dtype=float)

            alphaA = float(np.sum(w_sleeve_unit * alphas_arr))
            sigma2_eA = float(np.sum((w_sleeve_unit ** 2) * sigma2_arr))
            sigma2_eA = max(sigma2_eA, 1e-12)

            market_excess = formation_market - formation_factors.loc[formation_market.index, "RF"] if len(formation_market) > 0 else np.array([])
            if len(market_excess) < 5:
                E_RM = float(np.nanmean(market_excess)) if len(market_excess) > 0 else 0.0
                sigma2_M = float(np.nanvar(market_excess, ddof=0)) if len(market_excess) > 0 else 1e-6
            else:
                E_RM = float(np.nanmean(market_excess))
                sigma2_M = float(np.nanvar(market_excess, ddof=0))
            sigma2_M = max(sigma2_M, 1e-12)

            w0A = (alphaA / sigma2_eA) * (E_RM / sigma2_M) if sigma2_eA > 0 and sigma2_M > 0 else 0.0
            betaA = float(np.sum(w_sleeve_unit * betas_arr))

            denom_adj = 1.0 + (1.0 - betaA) * w0A
            if not np.isfinite(denom_adj) or abs(denom_adj) < 1e-12:
                wstarA = 0.0
            else:
                wstarA = float(w0A / denom_adj)
            if not np.isfinite(wstarA):
                wstarA = 0.0
            wstarA = float(np.clip(wstarA, -10.0, 10.0))
            wstarM = 1.0 - wstarA
            final_active_asset_weights = (wstarA * w_sleeve_unit).astype(float)

        # ---------------------------
        # Holding-period realized returns
        # ---------------------------
        L = len(holding_stock_returns)
        if L < 20:
            continue

        # Use labeled .dot() with alignment for safety
        rets_hold_aligned = holding_stock_returns.reindex(columns=formation_stock_returns.columns).dropna(how="any")

        if rets_hold_aligned.shape[1] != n_assets or len(rets_hold_aligned) < 1:
            try:
                port_gmv_daily = np.dot(holding_stock_returns.values, w_gmv_s.values)
                port_mv_daily = np.dot(holding_stock_returns.values, w_mv_s.values)
                port_ew_daily = np.dot(holding_stock_returns.values, w_ew_s.values)
            except Exception:
                port_gmv_daily = holding_market.loc[holding_stock_returns.index].values
                port_mv_daily = holding_market.loc[holding_stock_returns.index].values
                port_ew_daily = holding_market.loc[holding_stock_returns.index].values
        else:
            port_gmv_daily = rets_hold_aligned.dot(w_gmv_s).values
            port_mv_daily = rets_hold_aligned.dot(w_mv_s).values
            port_ew_daily = rets_hold_aligned.dot(w_ew_s).values

        # small sanity fallback
        if not (len(port_gmv_daily) > 0 and np.isfinite(port_gmv_daily).all()):
            port_gmv_daily = holding_market.loc[holding_stock_returns.index].values
        if not (len(port_mv_daily) > 0 and np.isfinite(port_mv_daily).all()):
            port_mv_daily = holding_market.loc[holding_stock_returns.index].values
        if not (len(port_ew_daily) > 0 and np.isfinite(port_ew_daily).all()):
            port_ew_daily = holding_market.loc[holding_stock_returns.index].values

        # Active daily returns
        if (not is_tb_active) or final_active_asset_weights is None:
            port_active_daily = holding_market.loc[holding_stock_returns.index].values
        else:
            sel_cols = active_names
            rets_hold_sel = holding_returns[sel_cols].dropna(how="any")
            idx_common = holding_stock_returns.index.intersection(rets_hold_sel.index)
            rets_hold_sel = rets_hold_sel.loc[idx_common]

            nifty_hold_aligned = holding_market.loc[idx_common]

            if len(rets_hold_sel) < 20:
                port_active_daily = holding_market.loc[holding_stock_returns.index].values
            else:
                sleeve_daily = np.dot(rets_hold_sel.values, final_active_asset_weights)
                port_active_daily = (wstarM * nifty_hold_aligned.values) + sleeve_daily

                port_gmv_daily = pd.Series(port_gmv_daily, index=holding_stock_returns.index).loc[idx_common].values
                port_mv_daily = pd.Series(port_mv_daily, index=holding_stock_returns.index).loc[idx_common].values
                port_ew_daily = pd.Series(port_ew_daily, index=holding_stock_returns.index).loc[idx_common].values
                nifty_hold_aligned = nifty_hold_aligned.loc[idx_common]
                holding_factors = holding_factors.loc[idx_common]
                L = len(idx_common)

        ret_gmv = compound_holding_return(port_gmv_daily)
        ret_mv = compound_holding_return(port_mv_daily)
        ret_ew = compound_holding_return(port_ew_daily)
        ret_active = compound_holding_return(port_active_daily)
        ret_nifty = compound_holding_return(holding_market.loc[holding_stock_returns.index].values)

        results_rows.append({
            "window_end": ws.holding_end,
            "GMV": ret_gmv,
            "MV": ret_mv,
            "EW": ret_ew,
            "Active": ret_active,
            "NIFTY50": ret_nifty,
        })

        # ---------------------------
        # Historical VaR (99%) based on formation window with L-day rolls
        # ---------------------------
        market_form = formation_market

        var_input_map: Dict[str, Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]] = {
            "GMV": (w_gmv_s.values, formation_stock_returns),
            "MV": (w_mv_s.values, formation_stock_returns),
            "EW": (w_ew_s.values, formation_stock_returns),
            "Active": (None, None),
        }

        for k, (wts, form_df) in var_input_map.items():
            if wts is None:
                portfolio_var_estimates[k].append(np.nan)
                portfolio_realized_returns[k].append({"GMV": ret_gmv, "MV": ret_mv, "EW": ret_ew, "Active": ret_active}[k])
                continue

            form_df2 = form_df.dropna(how="any")
            if form_df2.empty or len(form_df2) < L + 5:
                portfolio_var_estimates[k].append(np.nan)
                portfolio_realized_returns[k].append({"GMV": ret_gmv, "MV": ret_mv, "EW": ret_ew}[k])
                continue

            l_block = rolling_window_portfolio_returns(wts, form_df2, L)
            if len(l_block) == 0:
                portfolio_var_estimates[k].append(np.nan)
            else:
                q01 = nan_safe_quantile(l_block, 0.01)
                var_neg = -(-q01) if q01 is not None else np.nan
                portfolio_var_estimates[k].append(var_neg)
            portfolio_realized_returns[k].append({"GMV": ret_gmv, "MV": ret_mv, "EW": ret_ew}[k])

        # Active VaR
        if (not is_tb_active) or final_active_asset_weights is None:
            l_block_active = rolling_window_portfolio_returns(None, formation_stock_returns, L, market_ret=market_form)
        else:
            form_df_sel = formation_returns[active_names].dropna(how="any")
            idx_common_form = market_form.index.intersection(form_df_sel.index)
            form_df_sel = form_df_sel.loc[idx_common_form]
            market_form_aligned = market_form.loc[idx_common_form]

            if form_df_sel.empty or len(form_df_sel) < L + 5:
                l_block_active = np.array([])
            else:
                combined_daily_form = (wstarM * market_form_aligned.values) + np.dot(form_df_sel.values, final_active_asset_weights)
                l_block_active = rolling_compound_simple(combined_daily_form, L)

        if len(l_block_active) == 0:
            portfolio_var_estimates["Active"].append(np.nan)
        else:
            q01 = nan_safe_quantile(l_block_active, 0.01)
            var_neg = -(-q01) if q01 is not None else np.nan
            portfolio_var_estimates["Active"].append(var_neg)
        portfolio_realized_returns["Active"].append(ret_active)

        window_endpoints.append(ws.holding_end)

    # Build return matrix and write CSV
    if not results_rows:
        raise RuntimeError("No windows produced results. Check data alignment and file contents.")

    ret_df = pd.DataFrame(results_rows).sort_values("window_end").reset_index(drop=True)
    out_csv = os.path.join(here, OUTPUT_RETURNS_FILE)
    ret_df[["GMV", "MV", "EW", "Active", "NIFTY50"]].to_csv(out_csv, index=False)

    # Robust indexing for var/real dfs
    var_df = pd.DataFrame({k: pd.Series(v) for k, v in portfolio_var_estimates.items()})
    real_df = pd.DataFrame({k: pd.Series(v) for k, v in portfolio_realized_returns.items()})
    win_idx = pd.Index(window_endpoints)

    def _align_index(df: pd.DataFrame, win_idx: pd.Index) -> pd.DataFrame:
        if df.empty:
            return df
        n = len(df)
        m = len(win_idx)
        if m >= n:
            df.index = win_idx[:n]
            return df
        if m > 0:
            pad = [win_idx[-1]] * (n - m)
        else:
            pad = [pd.NaT] * (n - m)
        new_idx = pd.Index(list(win_idx) + pad)
        df.index = new_idx[:n]
        return df

    var_df = _align_index(var_df, win_idx)
    real_df = _align_index(real_df, win_idx)

    # Cumulative returns plot
    growth = (1.0 + ret_df[["GMV", "MV", "EW", "Active", "NIFTY50"]]).cumprod()
    growth.index = ret_df["window_end"]
    plt.figure(figsize=(11, 6))
    sns.set_style("whitegrid")
    for col in growth.columns:
        plt.plot(growth.index, growth[col], label=col)
    plt.title("Cumulative Growth ($1 base) across 3M windows")
    plt.xlabel("Window End")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(here, OUTPUT_CUMRET_PLOT), dpi=160)
    plt.close()

    # Performance table
    summary_records = []
    windows_per_year = 4.0
    sqrt_f = math.sqrt(windows_per_year)

    for col in ["GMV", "MV", "EW", "Active", "NIFTY50"]:
        r = ret_df[col].values.astype(float)
        log_r = np.log1p(r)
        mean_ann = float(np.expm1(np.nanmean(log_r) * windows_per_year))
        std_win = float(np.nanstd(r, ddof=1))
        mean_win = float(np.nanmean(r))
        sharpe_win = mean_win / (std_win + 1e-12)
        sharpe_ann = sharpe_win * sqrt_f

        if col != "NIFTY50":
            spread = (ret_df[col] - ret_df["NIFTY50"]).values.astype(float)
            mean_spread = float(np.nanmean(spread))
            te_win = float(np.nanstd(spread, ddof=1))
            ir_ann = (mean_spread / (te_win + 1e-12)) * sqrt_f
        else:
            ir_ann = np.nan

        summary_records.append({
            "Portfolio": col,
            "Mean_Ann": mean_ann,
            "Std_Win": std_win,
            "Sharpe_Ann": sharpe_ann,
            "InfoRatio_Ann_vs_NIFTY": ir_ann,
        })

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(here, OUTPUT_PERF_FILE), index=False)

    # VaR vs realized plots + violations
    violation_counts = {}
    for k in ["GMV", "MV", "EW", "Active"]:
        aligned = var_df[k].dropna().index.intersection(real_df[k].dropna().index)
        v = var_df.loc[aligned, k]
        r = real_df.loc[aligned, k]
        viol = (r < v).sum()
        violation_counts[k] = int(viol)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    portfolio_list = ["GMV", "MV", "EW", "Active"]
    for ax, k in zip(axes.ravel(), portfolio_list):
        ax.plot(var_df.index, var_df[k], label=f"VaR p99 (neg)")
        ax.plot(real_df.index, real_df[k], label="Realized 3M Return")
        ax.set_title(f"{k} (violations={violation_counts.get(k, 0)})")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("3M Historical VaR (99%) vs Realized Returns")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(here, OUTPUT_VAR_PLOT), dpi=160)
    plt.close()

    print("Done. Files created:")
    print(f" - {OUTPUT_RETURNS_FILE}")
    print(f" - {OUTPUT_CUMRET_PLOT}")
    print(f" - {OUTPUT_PERF_FILE}")
    print(f" - {OUTPUT_VAR_PLOT}")


if __name__ == "__main__":
    main()
