"""Estimators shared by the simulation check and the real-data pipeline."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit


def sigma2_from_mid(t_ms, mid, step_s=10.0):
    """Price variance per second from a mid series sampled on a regular grid
    (last value), using step_s returns to damp microstructure noise."""
    s = pd.Series(np.asarray(mid, float), index=pd.to_datetime(np.asarray(t_ms), unit="ms"))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    g = s.resample(f"{int(step_s * 1000)}ms").last().ffill().dropna()
    d = g.diff().dropna().values
    if len(d) < 20:
        return np.nan
    # robust: scale MAD to std, mean-square otherwise; use plain variance of
    # demeaned increments (drift is negligible at this horizon)
    return float(np.var(d) / step_s)


def intensity_fit(depth, weight, T, q_lo=0.10, q_hi=0.98, n_grid=40):
    """Fit Lambda(delta) = A exp(-k delta) to the empirical reach curve.

    depth  : how far (price units) each aggressive order reached from the mid
    weight : contribution of each order (1 for counts, size/Delta for lots)
    T      : observation time in seconds
    Returns dict with A (per second, at delta=0), k (1/price), the grid,
    the empirical curve, and a power-law comparison.
    """
    depth = np.asarray(depth, float)
    weight = np.asarray(weight, float)
    ok = np.isfinite(depth) & (depth > 0) & np.isfinite(weight)
    depth, weight = depth[ok], weight[ok]
    if len(depth) < 30:
        return None
    lo, hi = np.quantile(depth, [q_lo, q_hi])
    grid = np.linspace(lo, hi, n_grid)
    order = np.argsort(depth)
    d_sorted, w_sorted = depth[order], weight[order]
    cum = np.cumsum(w_sorted[::-1])[::-1]               # weight with depth >= d_sorted[i]
    idx = np.searchsorted(d_sorted, grid, side="left")
    lam = np.where(idx < len(d_sorted), cum[np.minimum(idx, len(cum) - 1)], 0.0) / T
    m = lam > 0
    g, y = grid[m], np.log(lam[m])
    X = sm.add_constant(g)
    ols = sm.OLS(y, X).fit()
    logA, negk = ols.params
    # power-law alternative on the same grid: log lam = c - alpha log delta
    Xp = sm.add_constant(np.log(g))
    olsp = sm.OLS(y, Xp).fit()
    return {"A": float(np.exp(logA)), "k": float(-negk), "r2_exp": float(ols.rsquared),
            "r2_pow": float(olsp.rsquared), "alpha_pow": float(-olsp.params[1]),
            "aic_exp": float(ols.aic), "aic_pow": float(olsp.aic),
            "grid": g, "lam": np.exp(y), "n": int(len(depth))}


def skew_regression(center_offset, q, controls=None, maxlags=30):
    """center_offset = alpha - b * q (+ controls) ; returns b and HAC s.e."""
    y = np.asarray(center_offset, float)
    cols = {"q": np.asarray(q, float)}
    if controls:
        cols.update({k: np.asarray(v, float) for k, v in controls.items()})
    X = pd.DataFrame(cols)
    ok = np.isfinite(y) & np.all(np.isfinite(X.values), axis=1)
    if ok.sum() < 50 or np.nanstd(X["q"][ok]) == 0:
        return None
    Xc = sm.add_constant(X[ok], has_constant="add")
    res = sm.OLS(y[ok], Xc).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {"b": float(-res.params["q"]), "b_se": float(res.bse["q"]),
            "t": float(-res.tvalues["q"]), "r2": float(res.rsquared), "n": int(ok.sum()),
            "params": res.params.to_dict(), "tvalues": res.tvalues.to_dict()}
