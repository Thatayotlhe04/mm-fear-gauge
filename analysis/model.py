"""Per-coin structural estimation of HLP's risk aversion and the time-varying fear gauge."""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(__file__))
import glft  # noqa: E402
from estimators import intensity_fit, sigma2_from_mid, skew_regression  # noqa: E402
from panel import load  # noqa: E402

CONTROLS = ["basis", "imb5", "ret10", "ret60"]


def own_fill_intensity(pc, fills_c, D, k_fixed=None):
    """HLP's own fill curve. For every poll interval and side, the number of passive
    fills HLP received is Poisson with mean dt * A * exp(-k * delta), delta = HLP's
    distance from the mid at the start of the interval. Bid and ask pooled.
    Returns A in lots per second per side, k in 1/price."""
    tcol = "t_snap" if "t_snap" in pc else "t_recv"
    pc = pc.sort_values(tcol)
    t = pc[tcol].values
    dt = np.diff(t) / 1000.0
    if len(t) < 50 or len(fills_c) < 10:
        return None
    f = fills_c[~fills_c.crossed]
    rows = []
    for side, dcol in (("B", "db"), ("A", "da")):
        ff = f[f.side == side]
        idx = np.searchsorted(t, ff.time.values, side="right") - 1
        cnt = np.bincount(idx[(idx >= 0) & (idx < len(dt))], minlength=len(dt))[: len(dt)]
        d = pc[dcol].values[:-1]
        rows.append(pd.DataFrame({"n": cnt, "delta": d, "dt": dt}))
    z = pd.concat(rows)
    z = z[(z.dt > 0) & (z.dt < 10) & np.isfinite(z.delta) & (z.delta > 0)]
    if z.n.sum() < 10:
        return None
    try:
        res = smf.glm("n ~ delta", z, family=sm.families.Poisson(), offset=np.log(z.dt)).fit()
    except Exception:
        return None
    lots_per_fill = float((f.sz / D).mean()) if len(f) else 1.0
    out = {"A": float(np.exp(res.params["Intercept"])) * lots_per_fill, "k": float(-res.params["delta"]),
           "k_se": float(res.bse["delta"]), "n_fills": int(z.n.sum())}
    if k_fixed is not None:
        # Poisson MLE for A with the fill-curve slope fixed: A = N / sum(dt * exp(-k delta))
        out["A_given_k"] = float(z.n.sum() / (z.dt * np.exp(-k_fixed * z.delta)).sum()) * lots_per_fill
    return out


def coin_params(label, p, aggr, min_aggr=150, min_obs=300, events=None):
    """One row per coin: sigma^2, two fill-curve calibrations (market trades, HLP's own
    fills), skew slopes (levels with/without controls, fill-event), half spread,
    implied gammas, and the GLFT over-identifying restrictions."""
    ctx = load(label, "ctx")
    fills = load(label, "fills").drop_duplicates("tid")
    T = (p.t_recv.max() - p.t_recv.min()) / 1000
    fills = fills[(fills.time >= p.t_recv.min()) & (fills.time <= p.t_recv.max())]
    rows = []
    for coin, pc in p[p.two_sided].groupby("coin"):
        pc = pc.dropna(subset=["c", "q", "S"])
        if len(pc) < min_obs:
            continue
        cc = ctx[ctx.coin == coin]
        sig2 = sigma2_from_mid(cc.t_recv.values, cc.mid.values, step_s=10)
        D = float(np.nanmedian(np.r_[pc.bs1.values, pc.as1.values]))
        ag = aggr[(aggr.coin == coin) & (aggr.reach > 0)]
        if len(ag) < min_aggr or not np.isfinite(sig2) or sig2 <= 0:
            continue
        fit = intensity_fit(ag.reach.values, ag.sz.values / D, T)
        if fit is None or fit["k"] <= 0:
            continue
        A_m, k_m = fit["A"] / 2, fit["k"]
        own = own_fill_intensity(pc, fills[fills.coin == coin], D, k_fixed=k_m)
        has_own = own is not None and own.get("A_given_k", 0) > 0
        A_o, k_o = (own["A_given_k"], own["k"]) if has_own else (np.nan, np.nan)
        S = float(pc.S.median())
        sk = skew_regression(pc.c.values, pc.q.values)
        ska = skew_regression(pc.c.values, pc.q.values,
                              controls={c: pc[c].fillna(0).values for c in CONTROLS})
        nan_fit = {"b": np.nan, "b_se": np.nan, "t": np.nan, "r2": np.nan, "params": {}, "tvalues": {}}
        sk, ska = sk or nan_fit, ska or nan_fit
        psi = float(pc.psi.mean())
        r = {"coin": coin, "n": len(pc), "S": S, "sigma2": sig2, "sigma_bps_s": 1e4 * np.sqrt(sig2) / S,
             "A_mkt": A_m, "k_mkt": k_m, "A_own": A_o, "k_own": k_o,
             "k_own_se": own["k_se"] if has_own else np.nan, "n_own_fills": own["n_fills"] if own else 0,
             "D": D, "D_usd": D * S,
             "r2_exp": fit["r2_exp"], "r2_pow": fit["r2_pow"], "aic_exp": fit["aic_exp"], "aic_pow": fit["aic_pow"],
             "alpha_pow": fit["alpha_pow"], "n_aggr": len(ag), "psi": psi, "psi_bps": 1e4 * psi / S,
             "q_std": float(pc.q.std()), "Q_std_usd": float((pc.q * pc.S).std()),
             "Q_mean_usd": float((pc.q * pc.S).mean())}
        # primary calibration: slope k from the market-wide depth profile, level A from
        # HLP's own passive fills at its own quote distances (captures its flow share)
        A, k = (A_o, k_m) if has_own else (A_m, k_m)
        r["flow_share"] = A_o / A_m if has_own else np.nan
        r["A"], r["k"], r["calib"] = A, k, "own" if has_own else "mkt"
        slopes = [("tb", sk["b"], sk["b_se"], sk["t"]), ("aug", ska["b"], ska["b_se"], ska["t"])]
        if events is not None and len(events):
            eb = event_b(events, coin, h=5)
            if eb:
                slopes.append(("ev", eb["b"], eb["se"], eb["b"] / eb["se"] if eb["se"] > 0 else np.nan))
                r["n_ev"] = eb["n"]
        r["r2_tb"], r["r2_aug"] = sk["r2"], ska["r2"]
        for tag, b, se, tt in slopes:
            r[f"b_{tag}"], r[f"b_se_{tag}"], r[f"t_{tag}"] = b, se, tt
            r[f"gamma_{tag}"] = glft.gamma_from_skew(b, sig2, A, k, D)
            r[f"gamma_{tag}_lo"] = glft.gamma_from_skew(b - 1.96 * se, sig2, A, k, D)
            r[f"gamma_{tag}_hi"] = glft.gamma_from_skew(b + 1.96 * se, sig2, A, k, D)
            r[f"gamma_{tag}_mkt"] = glft.gamma_from_skew(b, sig2, A_m, k_m, D)
            r[f"skew_bps_10k_{tag}"] = 1e4 * b * (1e4 / S) / S      # bps of price per $10k inventory
        r["phi_basis"] = ska["params"].get("basis", np.nan)
        r["t_basis"] = ska["tvalues"].get("basis", np.nan)
        r["gamma_spread"] = glft.gamma_from_spread(psi, sig2, A, k, D)
        r["gamma_spread_mkt"] = glft.gamma_from_spread(psi, sig2, A_m, k_m, D)
        # over-identifying restriction (small gamma*D/k limit): psi = 1/k + (D/2) b
        bpos = max(r["b_aug"], 0) if np.isfinite(r["b_aug"]) else 0.0
        for tag, kk in (("", k), ("_mkt", k_m)):
            pred = 1 / kk + 0.5 * D * bpos
            r[f"psi_pred{tag}"] = pred
            r[f"psi_ratio{tag}"] = psi / pred
            r[f"markup_bps{tag}"] = 1e4 / kk / S
        r["excess_bps"] = r["psi_bps"] - r["markup_bps"] - 1e4 * 0.5 * D * bpos / S
        rows.append(r)
    return pd.DataFrame(rows)


def fill_event_skew(label, p, horizons=(2, 5, 15), pre_ms=3000, fill_label=None, crossed=False):
    """Identification from inventory shocks HLP did not choose: a passive (maker) fill
    changes q by the fill size. Compare HLP's quote centre in the last poll before the
    fill with the first poll >= h seconds after it:
        d c = -b * d q + phi * d basis + rho * d ret + e
    Returns per-coin and pooled estimates of b at each horizon."""
    fl = load(fill_label or label, "fills").drop_duplicates("tid")
    fl = fl[(fl.crossed == crossed) & (fl.time >= p.t_recv.min()) & (fl.time <= p.t_recv.max())]
    out = []
    for coin, f in fl.groupby("coin"):
        pc = p[(p.coin == coin) & p.two_sided].dropna(subset=["c", "basis"]).sort_values("t_snap")
        if len(pc) < 100 or len(f) < 5:
            continue
        t = pc.t_snap.values
        i0 = np.searchsorted(t, f.time.values) - 1
        ok0 = (i0 >= 0)
        for h in horizons:
            i1 = np.searchsorted(t, f.time.values + h * 1000)
            ok = ok0 & (i1 < len(t))
            ok &= (f.time.values - t[np.clip(i0, 0, None)]) < pre_ms
            if ok.sum() == 0:
                continue
            a, b_ = pc.iloc[i0[ok]], pc.iloc[i1[ok]]
            fs = np.where(f.side.values[ok] == "B", 1.0, -1.0) * f.sz.values[ok]
            out.append(pd.DataFrame({
                "coin": coin, "h": h, "time": f.time.values[ok], "S": a.S.values, "fill_dq": fs,
                "dc": b_.c.values - a.c.values,
                "dq": (b_.q.values - a.q.values) if fill_label in (None, label) else fs,
                "dbasis": b_.basis.values - a.basis.values,
                "dret": np.log(b_.S.values / a.S.values)}))
    ev = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return ev


def event_b(ev, coin=None, h=5):
    e = ev[(ev.h == h) & ((ev.coin == coin) if coin else True)].dropna()
    e = e[e.dq != 0]
    if len(e) < 15:
        return None
    X = sm.add_constant(e[["dq", "dbasis", "dret"]], has_constant="add")
    r = sm.OLS(e.dc, X).fit(cov_type="HC1")
    return {"b": -r.params["dq"], "se": r.bse["dq"], "n": len(e)}


def maker_fill_rates(label, p, params):
    """Realised maker fills per side per second versus Lambda(delta) at HLP's quotes."""
    fl = load(label, "fills").drop_duplicates("tid")
    T = (p.t_recv.max() - p.t_recv.min()) / 1000
    t0, t1 = p.t_recv.min(), p.t_recv.max()
    fl = fl[(fl.time >= t0) & (fl.time <= t1) & (~fl.crossed)]
    out = []
    for _, r in params.iterrows():
        f = fl[fl.coin == r.coin]
        lots = f.sz.sum() / r.D / 2 / T
        pc = p[(p.coin == r.coin) & p.two_sided]
        lam_b = (r.A_mkt * np.exp(-r.k_mkt * pc.db.clip(lower=0))).mean()
        lam_a = (r.A_mkt * np.exp(-r.k_mkt * pc.da.clip(lower=0))).mean()
        out.append({"coin": r.coin, "maker_lots_per_s": lots, "model_lots_per_s": (lam_b + lam_a) / 2})
    return pd.DataFrame(out)


def risk_scale(pr):
    """GLFT skew per unit inventory per sqrt(gamma), in price units: sqrt(sigma^2 e / (2 A k D))."""
    return np.sqrt(pr.sigma2 * np.e / (2 * pr.A * pr.k * pr.D))


def fear_gauge(label, p, params, window_min=10, min_coins=15):
    """Between-coin structural gauge. At each window, across coins i:
        cbar_i = a + phi * basisbar_i - theta_w * xbar_i + e_i,
        xbar_i = 1e4 * qbar_i * sqrt(sigma_i^2 e / (2 A_i k_i D_i)) / S_i   (bps per sqrt(gamma))
    GLFT says theta_w = sqrt(gamma_w) and is common to all coins. Identification comes
    from the cross-section of inventories the market maker carries at that moment, so it
    does not need inventory to move inside the window. Coins with an own-fill calibration only."""
    pr = params[params.calib == "own"].set_index("coin") if (params.calib == "own").sum() >= min_coins else params.set_index("coin")
    scale = risk_scale(pr)
    d = p[p.two_sided & p.coin.isin(pr.index)].dropna(subset=["c", "q", "basis"]).copy()
    d["w"] = pd.to_datetime(d.t_recv, unit="ms").dt.floor(f"{window_min}min")
    d["xb"] = 1e4 * d.q * scale.reindex(d.coin).values / d.S
    d["cb"] = d.c_bps
    d["bb"] = d.basis_bps
    acct = load(label, "pos")
    acct = acct[acct.coin == "__ACCOUNT__"][["t_recv", "szi"]].rename(columns={"szi": "acct"})
    acct["w"] = pd.to_datetime(acct.t_recv, unit="ms").dt.floor(f"{window_min}min")
    acct = acct.groupby("w").acct.last()
    out = []
    for w, g in d.groupby("w"):
        a = g.groupby("coin")[["cb", "bb", "xb"]].mean()
        a = a[np.isfinite(a).all(axis=1)]
        if len(a) < min_coins:
            continue
        lo, hi = a.xb.quantile([0.01, 0.99])
        a = a[(a.xb >= lo) & (a.xb <= hi)]
        X = sm.add_constant(pd.DataFrame({"bb": a.bb, "negx": -a.xb}), has_constant="add")
        res = sm.OLS(a.cb, X).fit(cov_type="HC1")
        th, se = float(res.params["negx"]), float(res.bse["negx"])
        out.append({"w": w, "theta": th, "theta_se": se, "gamma": np.sign(th) * th ** 2, "phi": float(res.params["bb"]),
                    "r2": res.rsquared, "coins": len(a), "acct": acct.get(w, np.nan)})
    fg = pd.DataFrame(out, columns=["w", "theta", "theta_se", "gamma", "phi", "r2", "coins", "acct"])
    if len(fg):
        lo = fg.theta - 1.96 * fg.theta_se
        fg["gamma_lo"] = np.where(lo > 0, lo ** 2, 0.0)
        fg["gamma_hi"] = (fg.theta + 1.96 * fg.theta_se).clip(lower=0) ** 2
        fg["rra"] = fg.gamma * fg.acct
    return fg, d


# ---------------------------------------------------------------- the levers HLP actually uses
def size_skew(p, label_note="", window_min=10):
    """Top-of-ladder size imbalance (bs1 - as1)/(bs1 + as1) against inventory ($1k).
    GLFT skews prices and keeps size fixed; a negative slope means the market maker
    leans on size: less on the side that would add to inventory."""
    d = p[p.two_sided].dropna(subset=["q", "bs1", "as1"]).iloc[::7].copy()
    d["Q1k"] = d.q * d.S / 1e3
    d["simb"] = (d.bs1 - d.as1) / (d.bs1 + d.as1)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["simb", "Q1k"])
    g = pd.factorize(d.coin)[0]
    rw = smf.ols("simb ~ C(coin) + Q1k", d).fit(cov_type="cluster", cov_kwds={"groups": g})
    a = d.groupby("coin")[["simb", "Q1k"]].mean()
    rb = smf.ols("simb ~ Q1k", a).fit(cov_type="HC1")
    d["bin"] = pd.qcut(d.Q1k.clip(-15, 15), 10, duplicates="drop")
    bins = d.groupby("bin", observed=True).agg(Q1k=("Q1k", "median"), simb=("simb", "mean"), se=("simb", "sem"),
                                               n=("simb", "size")).reset_index()
    d["w"] = pd.to_datetime(d.t_recv, unit="ms").dt.floor(f"{window_min}min")
    ws = []
    for w, gw in d.groupby("w"):
        aw = gw.groupby("coin")[["simb", "Q1k"]].mean()
        if len(aw) < 20:
            continue
        r = smf.ols("simb ~ Q1k", aw).fit(cov_type="HC1")
        ws.append({"w": w, "size_skew": float(-r.params["Q1k"]), "se": float(r.bse["Q1k"]), "coins": len(aw)})
    return ({"within": float(rw.params["Q1k"]), "within_t": float(rw.tvalues["Q1k"]),
             "between": float(rb.params["Q1k"]), "between_t": float(rb.tvalues["Q1k"]), "n": int(rw.nobs)},
            bins, pd.DataFrame(ws))


def _fit_step(z, y, grid_q=(0.05, 0.95), n_grid=120):
    """Hard-step threshold by profile likelihood: P(y=1) = p_below for z < z_bar and
    p_above for z >= z_bar. Returns ((p_below, p_above, z_bar), nll)."""
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    order = np.argsort(z)
    z, y = z[order], y[order]
    cy = np.cumsum(y)
    n = len(z)
    best = (None, np.inf)
    for zb in np.unique(np.quantile(z, np.linspace(*grid_q, n_grid))):
        i = np.searchsorted(z, zb)
        if i < 20 or n - i < 20:
            continue
        p0 = cy[i - 1] / i
        p1 = (cy[-1] - cy[i - 1]) / (n - i)
        p0c, p1c = np.clip([p0, p1], 1e-6, 1 - 1e-6)
        nll = -(cy[i - 1] * np.log(p0c) + (i - cy[i - 1]) * np.log(1 - p0c)
                + (cy[-1] - cy[i - 1]) * np.log(p1c) + (n - i - cy[-1] + cy[i - 1]) * np.log(1 - p1c))
        if nll < best[1]:
            best = ((p0, p1, zb), nll)
    return best


def taker_threshold(label, p, params, n_boot=200, seed=0):
    """GLFT with market orders: cross the spread when the reservation-price deviation
    omega*q exceeds the crossing cost c. With omega_i = sqrt(gamma) * s_i,
        z = |q| * s_i / c_i   and the switch happens at  z_bar = 1 / sqrt(gamma).
    Fit P(taker fill reduces inventory | z) as a logistic step across all coins,
    then gamma_T = 1 / z_bar^2. Waiting for a passive fill has option value, so the
    true boundary sits above c/omega: gamma_T is a lower bound."""
    from panel import inventory_events
    pr = params[params.calib == "own"].set_index("coin")
    sc = risk_scale(pr)
    ev, _ = inventory_events(label)
    fl = load(label, "fills").drop_duplicates("tid").sort_values("time").reset_index(drop=True)
    fl = fl[(fl.time >= max(ev.t.min(), p.t_recv.min())) & (fl.time <= p.t_recv.max()) & fl.coin.isin(pr.index)].reset_index(drop=True)
    ctx = load(label, "ctx")[["t_recv", "coin", "mid"]].sort_values("t_recv")
    k = fl[["time", "coin"]].copy()
    k["t"] = k.time - 1
    fl["qv"] = pd.merge_asof(k, ev[["t", "coin", "q"]], on="t", by="coin").q.values
    fl["mid"] = pd.merge_asof(fl[["time", "coin"]], ctx.rename(columns={"t_recv": "time"}), on="time", by="coin",
                              direction="backward", tolerance=3000).mid.values
    fl["sgn"] = np.where(fl.side == "B", 1.0, -1.0)
    fl["red"] = ((np.sign(fl.qv) == -fl.sgn) & (fl.qv != 0)).astype(int)
    tk = fl[fl.crossed].dropna(subset=["qv", "mid"]).copy()
    tk["cost"] = tk.sgn * (tk.px - tk.mid) + tk.fee / tk.sz           # price units per unit
    c = tk.groupby("coin").cost.median().clip(lower=1e-12)
    tk = tk[tk.coin.isin(c.index)]
    tk["z"] = np.abs(tk.qv) * sc.reindex(tk.coin).values / c.reindex(tk.coin).values
    tk = tk[np.isfinite(tk.z)]
    tk["cost_bps"] = 1e4 * tk.cost / tk.mid
    out = {"n_taker": len(tk), "coins": int(tk.coin.nunique()), "median_cost_bps": float(tk.cost_bps.median())}
    if len(tk) < 300:
        return out, tk
    th, nll = _fit_step(tk.z.values, tk.red.values)
    p_flat = tk.red.mean()
    nll_flat = -np.sum(tk.red * np.log(p_flat) + (1 - tk.red) * np.log(1 - p_flat))
    out.update({"p_below": float(th[0]), "p_above": float(th[1]), "z_bar": float(th[2]),
                "lr_stat_vs_flat": float(2 * (nll_flat - nll))})
    rng = np.random.default_rng(seed)
    coins = tk.coin.unique()
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(coins, len(coins), replace=True)
        b = pd.concat([tk[tk.coin == cc] for cc in pick])
        boots.append(_fit_step(b.z.values, b.red.values)[0][2])
    boots = np.array(boots)
    out["z_bar_lo"], out["z_bar_hi"] = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    out["gamma_T"] = 1 / out["z_bar"] ** 2
    out["gamma_T_lo"], out["gamma_T_hi"] = 1 / out["z_bar_hi"] ** 2, 1 / out["z_bar_lo"] ** 2
    # the same threshold expressed in dollars per coin: |Q| at which the switch happens
    tk["Qbar_usd"] = out["z_bar"] * c.reindex(tk.coin).values / sc.reindex(tk.coin).values * tk.mid
    out["median_threshold_usd"] = float(tk.groupby("coin").Qbar_usd.median().median())
    # reservation-price skew the threshold implies, bps per $10k, typical coin: omega_i = s_i / z_bar
    S_i = tk.groupby("coin").mid.median()
    om = sc.reindex(S_i.index) / out["z_bar"]
    out["reservation_skew_bps_per_10k"] = float((1e4 * om * (1e4 / S_i) / S_i).median())
    # 30-minute thresholds for the gauge
    tk["h"] = pd.to_datetime(tk.time, unit="ms").dt.floor("30min")
    hrs = []
    for h, g in tk.groupby("h"):
        if len(g) < 250:
            continue
        t_, _ = _fit_step(g.z.values, g.red.values)
        hrs.append({"h": h, "z_bar": float(t_[2]), "gamma_T": float(1 / t_[2] ** 2), "p_below": float(t_[0]),
                    "p_above": float(t_[1]), "n": len(g)})
    out["windows"] = hrs
    return out, tk


def event_horizon_profile(label, p, params, horizons=(5, 30, 120, 300, 900, 1800), n_boot=500, seed=0):
    """Quote response to inventory shocks, by horizon, after passive fills (shocks HLP did
    not choose) and after HLP's own taker fills. For each coin, a winsorised OLS of the
    quote-centre change (bps) on the model-scaled fill, controlling for the change in basis
    and the mid return; the cross-coin median theta with a bootstrap CI is robust to the
    handful of coins that dominate a pooled regression."""
    pr = params[params.calib == "own"].set_index("coin")
    sc = risk_scale(pr)
    rng = np.random.default_rng(seed)
    rows = []
    for kind, crossed in (("passive", False), ("own_taker", True)):
        ev = fill_event_skew(label, p, horizons=horizons, crossed=crossed)
        if not len(ev):
            continue
        ev = ev[ev.coin.isin(pr.index)].dropna()
        for h in horizons:
            e = ev[(ev.h == h) & (ev.fill_dq != 0)].copy()
            if len(e) < 50:
                continue
            e["x"] = -1e4 * sc.reindex(e.coin).values * e.fill_dq / e.S
            e["y"] = 1e4 * e.dc / e.S
            e["db"] = 1e4 * e.dbasis / e.S
            e["dr"] = 1e4 * e.dret
            for c in ("x", "y", "db", "dr"):
                lo, hi = e[c].quantile([0.01, 0.99])
                e[c] = e[c].clip(lo, hi)
            th = np.array([sm.OLS(g.y, g[["x", "db", "dr"]]).fit().params["x"]
                           for _, g in e.groupby("coin") if len(g) >= 15 and g.x.std() > 0])
            if len(th) < 10:
                continue
            bs = np.array([np.median(rng.choice(th, len(th))) for _ in range(n_boot)])
            rows.append({"kind": kind, "h": h, "theta": float(np.median(th)), "lo": float(np.quantile(bs, 0.025)),
                         "hi": float(np.quantile(bs, 0.975)), "share_positive": float((th > 0).mean()),
                         "coins": len(th), "n": len(e)})
    return pd.DataFrame(rows)


def coin_median_within(params, n_boot=500, seed=0):
    """Naive textbook estimator: per-coin levels regression of the quote centre on the
    inventory stock (coins whose inventory moved), theta_i = b_i / s_i, cross-coin median."""
    pr = params[(params.calib == "own") & (params.n_own_fills >= 10)].copy()
    pr = pr[np.isfinite(pr.b_aug)]
    th = (pr.b_aug / risk_scale(pr)).values
    if len(th) < 10:
        return {}
    rng = np.random.default_rng(seed)
    bs = np.array([np.median(rng.choice(th, len(th))) for _ in range(n_boot)])
    return {"theta": float(np.median(th)), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)),
            "share_positive": float((th > 0).mean()), "coins": len(th)}
