"""Where the textbook model breaks. Each function returns tidy results used by the
report and figures."""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(__file__))
from panel import load, USERS, HLP_LIQUIDATORS  # noqa: E402


# ------------------------------------------------------------------ 1. reference price
def reference_price(p, params):
    """Pooled: c_bps = a_i + phi * basis_bps + b * Q + theta * imb + rho * ret ...
    Textbook AS/GLFT: phi = 0 (quotes centre on the traded mid)."""
    d = p[p.two_sided & p.coin.isin(params.coin)].dropna(subset=["c_bps", "basis_bps", "Qusd"]).iloc[::5].copy()
    d["Q10k"] = d.Qusd / 1e4
    d["sig"] = d.coin.map(params.set_index("coin").sigma_bps_s)
    d["Q10k_sig2"] = d.Q10k  # inventory in $10k (kept name for table compatibility)
    d["imb5"] = d.imb5.fillna(0)
    d["ret10_bps"] = 1e4 * d.ret10.fillna(0)
    d["ret60_bps"] = 1e4 * d.ret60.fillna(0)
    out = {}
    specs = {
        "textbook": "c_bps ~ C(coin) + Q10k_sig2",
        "plus_basis": "c_bps ~ C(coin) + Q10k_sig2 + basis_bps",
        "full": "c_bps ~ C(coin) + Q10k_sig2 + basis_bps + imb5 + ret10_bps + ret60_bps",
    }
    groups = pd.factorize(d.coin)[0]
    for name, f in specs.items():
        r = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": groups})
        keep = [k for k in r.params.index if not k.startswith("C(")]
        out[name] = {"r2": r.rsquared, "params": r.params[keep].to_dict(), "t": r.tvalues[keep].to_dict(),
                     "n": int(r.nobs)}
    # within-coin R^2 contributions (demeaned)
    dm = d.copy()
    for c in ["c_bps", "Q10k_sig2", "basis_bps", "imb5", "ret10_bps", "ret60_bps"]:
        dm[c] = dm[c] - dm.groupby("coin")[c].transform("mean")
    r_inv = smf.ols("c_bps ~ Q10k_sig2 - 1", dm).fit()
    r_all = smf.ols("c_bps ~ Q10k_sig2 + basis_bps + imb5 + ret10_bps + ret60_bps - 1", dm).fit()
    r_noinv = smf.ols("c_bps ~ basis_bps + imb5 + ret10_bps + ret60_bps - 1", dm).fit()
    out["within_r2"] = {"inventory_only": r_inv.rsquared, "all": r_all.rsquared,
                        "all_but_inventory": r_noinv.rsquared,
                        "partial_inventory": r_all.rsquared - r_noinv.rsquared}
    return out, d


# ------------------------------------------------------------------ 2. linearity / asymmetry
def inventory_shape(p, params, nbins=12):
    """Binned residual skew vs inventory normalised by each coin's inventory scale."""
    pr = params.set_index("coin")
    d = p[p.two_sided & p.coin.isin(pr.index)].dropna(subset=["c_bps", "basis_bps", "q"]).iloc[::5].copy()
    # strip the reference-price component coin by coin
    res = []
    for coin, g in d.groupby("coin"):
        X = sm.add_constant(g[["basis_bps"]].assign(imb5=g.imb5.fillna(0)), has_constant="add")
        r = sm.OLS(g.c_bps, X).fit()
        e = g.c_bps - r.params["basis_bps"] * g.basis_bps - r.params["imb5"] * g.imb5.fillna(0)
        qs = np.nanquantile(np.abs(g.q), 0.95)
        if qs <= 0:
            continue
        res.append(pd.DataFrame({"coin": coin, "z": g.q / qs, "e": e - e.mean(), "Qusd": g.Qusd}))
    r = pd.concat(res)
    r = r[np.abs(r.z) <= 1.5]
    r["bin"] = pd.cut(r.z, np.linspace(-1.5, 1.5, nbins + 1))
    b = r.groupby("bin", observed=True).agg(z=("z", "mean"), e=("e", "mean"), se=("e", "sem"), n=("e", "size")).reset_index()
    # linear vs cubic, long vs short slopes (pooled, coin-clustered)
    r["z_pos"] = r.z.clip(lower=0)
    r["z_neg"] = r.z.clip(upper=0)
    g = pd.factorize(r.coin)[0]
    lin = smf.ols("e ~ z", r).fit(cov_type="cluster", cov_kwds={"groups": g})
    cub = smf.ols("e ~ z + I(z**2) + I(z**3)", r).fit(cov_type="cluster", cov_kwds={"groups": g})
    asym = smf.ols("e ~ z_pos + z_neg", r).fit(cov_type="cluster", cov_kwds={"groups": g})
    wt = asym.t_test("z_pos - z_neg = 0")
    return {"bins": b, "lin_slope": lin.params["z"], "lin_t": lin.tvalues["z"], "lin_r2": lin.rsquared,
            "cub_r2": cub.rsquared, "cub_t3": cub.tvalues["I(z ** 3)"], "cub_t2": cub.tvalues["I(z ** 2)"],
            "slope_long": asym.params["z_pos"], "slope_short": asym.params["z_neg"],
            "t_long": asym.tvalues["z_pos"], "t_short": asym.tvalues["z_neg"],
            "asym_t": float(np.squeeze(wt.tvalue)), "n": len(r)}


# ------------------------------------------------------------------ 3. markouts
def markouts(label, horizons=(1, 5, 10, 30, 60, 300)):
    """Signed mid move after each HLP fill, in bps, from HLP's point of view:
    positive = the fill made money at horizon h. Textbook: maker markout = the
    half spread captured at every horizon (fills carry no information)."""
    fl = load(label, "fills").drop_duplicates("tid")
    ctx = load(label, "ctx")[["t_recv", "coin", "mid"]].sort_values("t_recv")
    fl = fl.sort_values("time")
    sgn = np.where(fl.side == "B", 1.0, -1.0)
    out = fl[["time", "coin", "px", "sz", "side", "crossed", "startPosition"]].copy()
    out["sgn"] = sgn
    base = pd.merge_asof(out, ctx.rename(columns={"t_recv": "time"}), on="time", by="coin",
                         direction="backward", tolerance=3000)
    out["mid0"] = base.mid.values
    out["edge_bps"] = 1e4 * out.sgn * (out.mid0 - out.px) / out.mid0
    for h in horizons:
        k = out[["time", "coin"]].copy()
        k["time"] = k.time + h * 1000
        m = pd.merge_asof(k.reset_index().sort_values("time"), ctx.rename(columns={"t_recv": "time"}), on="time",
                          by="coin", direction="backward", tolerance=3000).set_index("index").sort_index()
        out[f"mo{h}"] = 1e4 * out.sgn * (m.mid - out.px) / out.px
    # only keep fills whose horizon fits within the sample
    out["usd"] = out.px * out.sz
    out["reduces"] = np.sign(out.startPosition) == -out.sgn
    out.loc[out.startPosition == 0, "reduces"] = False
    return out


def markout_table(mo, horizons=(1, 5, 10, 30, 60, 300)):
    rows = []
    for crossed, g in mo.groupby("crossed"):
        row = {"type": "taker" if crossed else "maker", "n": len(g), "usd": g.usd.sum(),
               "edge_bps": np.average(g.edge_bps.dropna(), weights=g.loc[g.edge_bps.notna(), "usd"])}
        for h in horizons:
            v = g[[f"mo{h}", "usd"]].dropna()
            row[f"mo{h}"] = np.average(v[f"mo{h}"], weights=v.usd) if len(v) else np.nan
            row[f"mo{h}_se"] = v[f"mo{h}"].std() / np.sqrt(len(v)) if len(v) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 4. taking liquidity
def taker_analysis(label, p):
    """Who triggers HLP's aggressive orders? Inventory (Guilbaud-Pham: market orders
    only to dump inventory) or the oracle/mid basis (stale-quote arbitrage)?"""
    fl = load(label, "fills").drop_duplicates("tid").sort_values("time")
    fl = fl[(fl.time >= p.t_recv.min()) & (fl.time <= p.t_recv.max())]
    ctx = load(label, "ctx")[["t_recv", "coin", "mid", "oracle"]].sort_values("t_recv")
    m = pd.merge_asof(fl, ctx.rename(columns={"t_recv": "time"}), on="time", by="coin",
                      direction="backward", tolerance=3000)
    m["sgn"] = np.where(m.side == "B", 1.0, -1.0)
    if label == "V":
        # judge inventory reduction against the vault's net position just before the fill
        from panel import inventory_events
        ev, _ = inventory_events("V")
        k = m[["time", "coin"]].copy()
        k["t"] = k.time - 1
        vq = pd.merge_asof(k.reset_index().sort_values("t"), ev[["t", "coin", "q"]], on="t", by="coin",
                           direction="backward").set_index("index").sort_index()
        m["startPosition"] = vq.q.fillna(0).values
    m["reduces"] = (np.sign(m.startPosition) == -m.sgn) & (m.startPosition != 0)
    m["basis_bps"] = 1e4 * (m.oracle - m.mid) / m.mid
    m["with_basis"] = np.sign(m.basis_bps) == m.sgn          # buys when oracle above mid
    m["usd"] = m.px * m.sz
    m["crossed"] = m.crossed.astype(int)
    summ = m.groupby("crossed").apply(lambda g: pd.Series({
        "n": len(g), "usd": g.usd.sum(), "share_reducing": np.average(g.reduces, weights=g.usd),
        "share_with_basis": np.average(g.with_basis, weights=g.usd),
        "mean_signed_basis_bps": np.average(g.sgn * g.basis_bps.fillna(0), weights=g.usd)}), include_groups=False)
    # logit: P(taker) on |basis| aligned with direction and inventory-reduction flag
    m["sb"] = (m.sgn * m.basis_bps).clip(-50, 50)
    m["red"] = m.reduces.astype(int)
    lg = smf.logit("crossed ~ sb + red", m.dropna(subset=["sb"])).fit(disp=0)
    return summ, {"params": lg.params.to_dict(), "t": lg.tvalues.to_dict(), "n": int(lg.nobs)}, m


# ------------------------------------------------------------------ 5. inventory limits & quote pulling
def quote_pulling(p, params):
    pr = params.set_index("coin")
    d = p[p.coin.isin(pr.index)].dropna(subset=["q"]).copy()
    d["bid_off"] = d.nb == 0
    d["ask_off"] = d.na == 0
    d["Qusd"] = d.q * d.S
    rows = []
    for coin, g in d.groupby("coin"):
        rows.append({"coin": coin, "share_bid_off": g.bid_off.mean(), "share_ask_off": g.ask_off.mean(),
                     "Q_when_bid_off": g.loc[g.bid_off, "Qusd"].median() if g.bid_off.any() else np.nan,
                     "Q_when_ask_off": g.loc[g.ask_off, "Qusd"].median() if g.ask_off.any() else np.nan,
                     "absQ_p95": np.nanquantile(np.abs(g.Qusd), 0.95), "absQ_max": np.nanmax(np.abs(g.Qusd))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 6. volatility scaling
def vol_scaling(label, p, params, window_min=15):
    """Within-coin elasticity of the half spread and of the skew slope to local volatility.
    GLFT: skew ~ sigma, (psi - 1/k) ~ sigma. Finite-horizon AS: both ~ sigma^2."""
    ctx = load(label, "ctx")
    pr = params.set_index("coin")
    rows = []
    for coin, g in p[p.two_sided & p.coin.isin(pr.index)].groupby("coin"):
        g = g.dropna(subset=["c", "q"])
        g = g.assign(w=pd.to_datetime(g.t_recv, unit="ms").dt.floor(f"{window_min}min"))
        cc = ctx[ctx.coin == coin].assign(w=lambda x: pd.to_datetime(x.t_recv, unit="ms").dt.floor(f"{window_min}min"))
        for w, gw in g.groupby("w"):
            cw = cc[cc.w == w]
            if len(gw) < 100 or len(cw) < 200:
                continue
            m = cw.set_index(pd.to_datetime(cw.t_recv, unit="ms")).mid.resample("10s").last().ffill()
            s2 = np.var(np.diff(m.values)) / 10
            rows.append({"coin": coin, "w": w, "sigma": np.sqrt(s2) / gw.S.mean(), "psi_rel": gw.psi.mean() / gw.S.mean(),
                         "excess_rel": gw.psi.mean() / gw.S.mean() - pr.loc[coin, "markup_bps"] / 1e4})
    v = pd.DataFrame(rows)
    v = v[(v.sigma > 0) & (v.psi_rel > 0)]
    v["ls"], v["lp"] = np.log(v.sigma), np.log(v.psi_rel)
    r = smf.ols("lp ~ ls + C(coin)", v).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(v.coin)[0]})
    return {"elasticity_psi_sigma": r.params["ls"], "se": r.bse["ls"], "n": int(r.nobs)}, v


# ------------------------------------------------------------------ 7. portfolio inventory
def portfolio_inventory(label, p, params):
    """Does coin i's skew respond to the rest of the book? Single-asset GLFT: no."""
    pos = load(label, "pos")
    pos = pos[pos.coin != "__ACCOUNT__"].copy()
    net = pos.groupby("t_recv").pos_value.sum()  # pos_value is |notional|
    signed = pos.assign(sv=np.sign(pos.szi) * pos.pos_value).groupby("t_recv").sv.sum().rename("net_usd")
    gross = net.rename("gross_usd")
    tot = pd.concat([signed, gross], axis=1).reset_index().sort_values("t_recv")
    d = p[p.two_sided & p.coin.isin(params.coin)].dropna(subset=["c_bps", "Qusd", "basis_bps"]).iloc[::5].sort_values("t_recv")
    d = pd.merge_asof(d, tot, on="t_recv", direction="backward")
    d["other_net_100k"] = (d.net_usd - d.Qusd) / 1e5
    d["Q10k"] = d.Qusd / 1e4
    d["gross_1m"] = d.gross_usd / 1e6
    d = d.dropna(subset=["c_bps", "Q10k", "other_net_100k", "gross_1m", "basis_bps"])
    r = smf.ols("c_bps ~ C(coin) + Q10k + other_net_100k + gross_1m + basis_bps", d).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.coin)[0]})
    keep = ["Q10k", "other_net_100k", "gross_1m", "basis_bps"]
    return {"params": r.params[keep].to_dict(), "t": r.tvalues[keep].to_dict(), "n": int(r.nobs)}, tot


# ------------------------------------------------------------------ 8. A vs B
def twin_strategies(pA, pB, prA, prB):
    m = prA.merge(prB, on="coin", suffixes=("_A", "_B"))
    qa = pA[["t_recv", "coin", "b1", "a1", "q"]].sort_values("t_recv")
    qb = pB[["t_recv", "coin", "b1", "a1", "q"]].sort_values("t_recv")
    j = pd.merge_asof(qa, qb, on="t_recv", by="coin", direction="nearest", tolerance=700, suffixes=("_A", "_B")).dropna()
    same_bid = (np.isclose(j.b1_A, j.b1_B)).mean()
    same_ask = (np.isclose(j.a1_A, j.a1_B)).mean()
    qcorr = j.groupby("coin").apply(lambda g: g.q_A.corr(g.q_B), include_groups=False).median()
    return m, {"share_same_best_bid": same_bid, "share_same_best_ask": same_ask, "median_inventory_corr": qcorr}


# ------------------------------------------------------------------ 9. ladder
def ladder_profile(label, p):
    lad = load(label, "ladder")
    ctx = load(label, "ctx")[["t_recv", "coin", "mid"]].sort_values("t_recv")
    lad = pd.merge_asof(lad.sort_values("t_recv"), ctx, on="t_recv", by="coin", direction="backward", tolerance=3000)
    lad["dist_bps"] = 1e4 * np.abs(lad.px - lad.mid) / lad.mid
    lad["usd"] = lad.px * lad.sz
    lad["rank"] = lad.groupby(["t_recv", "coin", "side"]).dist_bps.rank(method="first")
    prof = lad.groupby("rank").agg(dist_bps=("dist_bps", "median"), usd=("usd", "median"), n=("usd", "size")).reset_index()
    return prof[prof.n > 100]


# ------------------------------------------------------------------ 10. liquidations & stress
def liquidation_activity(label):
    tr = load(label, "trades").drop_duplicates("tid")
    liq = tr[tr.u0.isin(HLP_LIQUIDATORS) | tr.u1.isin(HLP_LIQUIDATORS)].copy()
    liq["usd"] = liq.px * liq.sz
    liq["w"] = pd.to_datetime(liq.t, unit="ms").dt.floor("10min")
    return liq.groupby("w").usd.sum()


# ------------------------------------------------------------------ 11. whose inventory is priced?
def inventory_attribution(pA, pV, paramsV, window_min=10):
    """A's quote centre on its own inventory and on the vault's net inventory (A + B).
    With mirrored books the two are close to orthogonal, so the test discriminates:
    a vault-level risk manager loads on q_vault, an account-level one on q_own."""
    from model import risk_scale
    pr = paramsV[paramsV.calib == "own"].set_index("coin")
    sc = risk_scale(pr)
    a = pA[pA.two_sided & pA.coin.isin(pr.index)][["t_recv", "coin", "c_bps", "basis_bps", "q", "S"]].dropna()
    v = pV[["t_recv", "coin", "q"]].rename(columns={"q": "q_v"})
    d = a.merge(v, on=["t_recv", "coin"], how="inner")
    d["x_own"] = 1e4 * d.q * sc.reindex(d.coin).values / d.S
    d["x_vault"] = 1e4 * d.q_v * sc.reindex(d.coin).values / d.S
    d = d[np.isfinite(d[["x_own", "x_vault"]]).all(axis=1)]
    out = {"corr_x_own_x_vault": float(d.x_own.corr(d.x_vault))}
    # within coin (coin fixed effects), sub-sampled to ~10 s
    s = d.iloc[::7].copy()
    g = pd.factorize(s.coin)[0]
    r = smf.ols("c_bps ~ C(coin) + basis_bps + x_own + x_vault", s).fit(cov_type="cluster", cov_kwds={"groups": g})
    out["within"] = {k: {"coef": float(-r.params[k]), "t": float(-r.tvalues[k])} for k in ("x_own", "x_vault")}
    out["within"]["n"] = int(r.nobs)
    # between coins, window means, one regression per window
    d["w"] = pd.to_datetime(d.t_recv, unit="ms").dt.floor(f"{window_min}min")
    rows = []
    for w, gw in d.groupby("w"):
        m = gw.groupby("coin")[["c_bps", "basis_bps", "x_own", "x_vault"]].mean()
        if len(m) < 15:
            continue
        rr = smf.ols("c_bps ~ basis_bps + x_own + x_vault", m).fit(cov_type="HC1")
        rows.append({"w": w, "theta_own": -rr.params["x_own"], "theta_vault": -rr.params["x_vault"],
                     "t_own": -rr.tvalues["x_own"], "t_vault": -rr.tvalues["x_vault"], "coins": len(m)})
    # size lean: A's top-of-ladder size imbalance on own vs vault inventory ($1k)
    sz = pA[pA.two_sided][["t_recv", "coin", "q", "S", "bs1", "as1"]].dropna().merge(v, on=["t_recv", "coin"], how="inner")
    sz["simb"] = (sz.bs1 - sz.as1) / (sz.bs1 + sz.as1)
    sz["Q_own_1k"] = sz.q * sz.S / 1e3
    sz["Q_vault_1k"] = sz.q_v * sz.S / 1e3
    sz = sz.replace([np.inf, -np.inf], np.nan).dropna().iloc[::7]
    rs = smf.ols("simb ~ C(coin) + Q_own_1k + Q_vault_1k", sz).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(sz.coin)[0]})
    out["size_lean"] = {k: {"coef": float(rs.params[k]), "t": float(rs.tvalues[k])} for k in ("Q_own_1k", "Q_vault_1k")}
    out["size_lean"]["n"] = int(rs.nobs)
    bw = pd.DataFrame(rows)
    if len(bw):
        out["between"] = {"median_theta_own": float(bw.theta_own.median()), "median_theta_vault": float(bw.theta_vault.median()),
                          "share_windows_vault_positive": float((bw.theta_vault > 0).mean()),
                          "share_windows_own_positive": float((bw.theta_own > 0).mean()), "windows": len(bw)}
    return out, bw
