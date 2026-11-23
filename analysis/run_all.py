"""End-to-end pipeline: panel -> structural estimates -> fear gauge -> breakdown tests.
Writes results/*.csv|json used by figures.py and the report.

    bash analysis/sync_data.sh && python analysis/run_all.py
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import breaks  # noqa: E402
import glft  # noqa: E402
from model import (coin_params, event_b, fear_gauge, fill_event_skew, maker_fill_rates,  # noqa: E402
                   size_skew, taker_threshold)
from panel import ROOT, aggressor_orders, build_panel, load  # noqa: E402

RES = os.path.join(ROOT, "results")


def jdump(obj, name):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (pd.Timestamp,)):
            return str(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")
        raise TypeError(type(o))
    with open(os.path.join(RES, name), "w") as fh:
        json.dump(obj, fh, indent=1, default=conv)


def pooled_structural(params, ev, label_acct):
    """Pool per-coin skews into one sqrt(gamma). GLFT: b_i = theta * s_i with
    s_i = sqrt(sigma_i^2 e / (2 A_i k_i D_i)) and theta = sqrt(gamma) common to all coins.
      within : inverse-variance mean of b_i / s_i from the per-coin levels regression
               (coins whose inventory actually moved: >= 10 own fills)
      event  : one regression of the 5 s quote-centre change on s_i * dq across all fill events
    """
    from model import risk_scale
    pr = params[(params.calib == "own")].copy()
    pr["s"] = risk_scale(pr)
    out = {}
    w = pr[(pr.n_own_fills >= 10) & np.isfinite(pr.b_aug) & (pr.b_se_aug > 0)].copy()
    if len(w) >= 5:
        th = w.b_aug / w.s
        se = w.b_se_aug / w.s
        lo, hi = th.quantile([0.05, 0.95])
        keep = (th >= lo) & (th <= hi)
        wt = 1 / se[keep] ** 2
        t = float((th[keep] * wt).sum() / wt.sum())
        s = float(np.sqrt(1 / wt.sum()))
        out["within"] = {"theta": t, "theta_se": s, "coins": int(keep.sum()), "median_theta": float(th.median()),
                         "share_positive": float((th > 0).mean())}
    e = ev[(ev.h == 5) & (ev.dq != 0) & ev.coin.isin(pr.coin)].dropna()
    if len(e) >= 30:
        sc = pr.set_index("coin").s
        X = pd.DataFrame({"negx": -1e4 * sc.reindex(e.coin).values * e.dq.values / e.S.values,
                          "db": 1e4 * e.dbasis.values / e.S.values, "dr": 1e4 * e.dret.values})
        y = 1e4 * e.dc.values / e.S.values
        r = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(e.coin)[0]})
        out["event"] = {"theta": float(r.params["negx"]), "theta_se": float(r.bse["negx"]), "coins": int(e.coin.nunique()),
                        "n": int(r.nobs), "phi_dbasis": float(r.params["db"])}
    for k, v in out.items():
        th, se = v["theta"], v["theta_se"]
        v["gamma"] = float(np.sign(th) * th ** 2)
        v["gamma_lo"] = float(max(th - 1.96 * se, 0) ** 2)
        v["gamma_hi"] = float(max(th + 1.96 * se, 0) ** 2)
        v["rra_trading_capital"] = v["gamma"] * label_acct
    return out


def run_label(label):
    t0 = time.time()
    p = build_panel(label, cache=False)
    aggr, _ = aggressor_orders(label)
    ev = fill_event_skew(label, p)
    params = coin_params(label, p, aggr, events=ev)
    dayvlm = load(label, "ctx").groupby("coin").dayvlm.median()
    params["dayvlm"] = params.coin.map(dayvlm)
    acct = load(label, "pos")
    acct = acct[acct.coin == "__ACCOUNT__"].szi
    acct_value = float(acct.median())
    params["acct"] = acct_value
    mfr = maker_fill_rates(label, p, params)
    params = params.merge(mfr, on="coin", how="left")
    fg, dfg = fear_gauge(label, p, params)
    out = {"label": label, "minutes": (p.t_recv.max() - p.t_recv.min()) / 60000,
           "start": str(pd.to_datetime(p.t_recv.min(), unit="ms")), "end": str(pd.to_datetime(p.t_recv.max(), unit="ms")),
           "n_quotes": len(p), "n_coins_quoted": int(p.coin.nunique()), "n_coins_estimated": len(params),
           "account_value": acct_value}
    out["pooled"] = pooled_structural(params, ev, acct_value)
    if len(fg):
        out["pooled"]["between"] = {"theta": float(fg.theta.median()), "gamma": float(np.sign(fg.theta.median()) * fg.theta.median() ** 2),
                                    "share_windows_positive": float((fg.theta > 0).mean()), "windows": len(fg),
                                    "median_coins": float(fg.coins.median()), "median_phi": float(fg.phi.median())}
        out["pooled"]["between"]["rra_trading_capital"] = out["pooled"]["between"]["gamma"] * acct_value
    good = params[(params.t_aug > 2)]
    out["coin_gamma"] = {
        "share_skew_correct_sign_aug": float((params.b_aug > 0).mean()),
        "share_skew_signif_aug": float((params.t_aug > 1.96).mean()),
        "share_skew_correct_sign_tb": float((params.b_tb > 0).mean()),
        "share_event_correct_sign": float((params.b_ev > 0).mean()) if "b_ev" in params else np.nan,
        "median_gamma_aug_signif": float(good.gamma_aug.median()) if len(good) else np.nan,
        "iqr_gamma_aug_signif": [float(good.gamma_aug.quantile(.25)), float(good.gamma_aug.quantile(.75))] if len(good) else [],
        "median_rra_aug_signif": float(good.gamma_aug.median() * acct_value) if len(good) else np.nan,
        "n_signif": len(good),
        "median_skew_bps_per_10k_aug": float(params.skew_bps_10k_aug.median()),
    }
    # heterogeneity: one CARA agent => one gamma for every coin
    h = good[(good.gamma_aug > 0)].dropna(subset=["dayvlm"])
    if len(h) > 8:
        X = sm.add_constant(pd.DataFrame({"log_vlm": np.log(h.dayvlm), "log_sigma": np.log(h.sigma_bps_s)}))
        r = sm.OLS(np.log(h.gamma_aug), X).fit(cov_type="HC1")
        out["heterogeneity"] = {"params": r.params.to_dict(), "t": r.tvalues.to_dict(), "r2": r.rsquared, "n": len(h),
                                "log10_gamma_range_p10_p90": float(np.log10(h.gamma_aug.quantile(.9) / h.gamma_aug.quantile(.1)))}
    # over-identification
    out["overid"] = {"median_psi_ratio": float(params.psi_ratio.median()),
                     "share_psi_ratio_gt1": float((params.psi_ratio > 1).mean()),
                     "share_spread_unrationalisable": float(params.gamma_spread.isna().mean()),
                     "median_psi_bps": float(params.psi_bps.median()),
                     "median_markup_bps": float(params.markup_bps.median())}
    out["intensity"] = {"share_exp_beats_pow_aic": float((params.aic_exp < params.aic_pow).mean()),
                        "median_r2_exp": float(params.r2_exp.median()), "median_r2_pow": float(params.r2_pow.median()),
                        "median_maker_over_model_fill": float((params.maker_lots_per_s / params.model_lots_per_s).median()),
                        "median_flow_share": float(params.flow_share.median()),
                        "share_coins_own_calib": float((params.calib == "own").mean())}
    rp, _ = breaks.reference_price(p, params)
    out["reference_price"] = rp
    shape = breaks.inventory_shape(p, params)
    out["inventory_shape"] = {k: v for k, v in shape.items() if k != "bins"}
    shape["bins"].assign(bin=shape["bins"].bin.astype(str)).to_csv(os.path.join(RES, f"shape_bins_{label}.csv"), index=False)
    mo = breaks.markouts(label)
    mo = mo[mo.time <= p.t_recv.max() - 300_000]
    mt = breaks.markout_table(mo)
    # per-coin adverse selection on passive fills, to set against the excess spread
    mm = mo[~mo.crossed].copy()
    pc_mo = mm.groupby("coin").apply(lambda g: pd.Series({
        "n_maker_mo": len(g), "edge_bps_m": np.average(g.edge_bps.fillna(0), weights=g.usd),
        **{f"as{h}_bps": np.average((g.edge_bps - g[f"mo{h}"]).fillna(0), weights=g.usd) for h in (5, 30, 60, 300)}}),
        include_groups=False).reset_index()
    params = params.merge(pc_mo, on="coin", how="left")
    ok = params[(params.n_maker_mo >= 20) & np.isfinite(params.excess_bps)]
    out["spread_vs_adverse_selection"] = {f"corr_excess_as{h}": float(ok.excess_bps.corr(ok[f"as{h}_bps"])) for h in (5, 30, 60, 300)} if len(ok) > 8 else {}
    if len(ok) > 8:
        out["spread_vs_adverse_selection"].update({"n": len(ok), "median_excess_bps": float(ok.excess_bps.median()),
                                                   **{f"median_as{h}_bps": float(ok[f"as{h}_bps"].median()) for h in (5, 30, 60, 300)}})
    mt.to_csv(os.path.join(RES, f"markouts_{label}.csv"), index=False)
    summ, lg, _ = breaks.taker_analysis(label, p)
    out["taker"] = {"summary": summ.reset_index().to_dict(orient="records"), "logit": lg}
    qp = breaks.quote_pulling(p, params)
    qp.to_csv(os.path.join(RES, f"quote_pulling_{label}.csv"), index=False)
    out["quote_pulling"] = {"median_share_bid_off": float(qp.share_bid_off.median()),
                            "median_share_ask_off": float(qp.share_ask_off.median()),
                            "share_coins_ever_one_sided": float(((qp.share_bid_off > 0) | (qp.share_ask_off > 0)).mean())}
    vs, vdf = breaks.vol_scaling(label, p, params)
    out["vol_scaling"] = vs
    vdf.to_csv(os.path.join(RES, f"vol_scaling_{label}.csv"), index=False)
    pi, tot = breaks.portfolio_inventory(label, p, params)
    out["portfolio"] = pi
    tot.to_csv(os.path.join(RES, f"portfolio_{label}.csv"), index=False)
    lp = breaks.ladder_profile(label, p)
    lp.to_csv(os.path.join(RES, f"ladder_{label}.csv"), index=False)
    liq = breaks.liquidation_activity(label)
    fg["w"] = pd.to_datetime(fg.w)
    liq = liq.rename("liq_usd").reset_index() if len(liq) else pd.DataFrame({"w": pd.to_datetime([]), "liq_usd": []})
    fg = fg.merge(liq, on="w", how="left").fillna({"liq_usd": 0})
    # market state per window for the gauge
    ctx = load(label, "ctx")
    ctx["w"] = pd.to_datetime(ctx.t_recv, unit="ms").dt.floor("10min")
    ctx = ctx[ctx.coin.isin(params.coin)].sort_values("t_recv")
    ctx["r"] = ctx.groupby("coin").mid.transform(lambda s: np.log(s).diff())
    vol = ctx.groupby(["w", "coin"]).r.std().groupby("w").median().rename("mkt_vol")
    btc = ctx[ctx.coin == "BTC"].groupby("w").mid.last().rename("btc")
    fg = fg.merge(vol.reset_index(), on="w", how="left").merge(btc.reset_index(), on="w", how="left")
    if len(fg) > 5:
        out["fear_corr"] = {"corr_gamma_vol": float(fg[["gamma", "mkt_vol"]].corr().iloc[0, 1]),
                            "corr_gamma_liq": float(fg[["gamma", "liq_usd"]].corr().iloc[0, 1]),
                            "corr_gamma_acct_change": float(fg.gamma.corr(fg.acct.diff())),
                            "gamma_p10_p90": [float(fg.gamma.quantile(.1)), float(fg.gamma.quantile(.9))]}
    fg.to_csv(os.path.join(RES, f"fear_gauge_{label}.csv"), index=False)
    params.to_csv(os.path.join(RES, f"params_{label}.csv"), index=False)
    ev.to_parquet(os.path.join(ROOT, "work", f"events_{label}.parquet"))
    # the levers HLP actually uses: size lean at the touch, and taker unwinds past a threshold
    ss, ss_bins, ss_w = size_skew(p)
    out["size_skew"] = ss
    ss_bins.assign(bin=ss_bins.bin.astype(str)).to_csv(os.path.join(RES, f"size_skew_bins_{label}.csv"), index=False)
    ss_w.to_csv(os.path.join(RES, f"size_skew_windows_{label}.csv"), index=False)
    tt, tk = taker_threshold(label, p, params, n_boot=300 if label == "V" else 100)
    out["taker_threshold"] = {k: v for k, v in tt.items() if k != "windows"}
    pd.DataFrame(tt.get("windows", [])).to_csv(os.path.join(RES, f"taker_windows_{label}.csv"), index=False)
    if len(tk):
        tk["zbin"] = pd.qcut(tk.z, 12, duplicates="drop")
        tk.groupby("zbin", observed=True).agg(z=("z", "median"), red=("red", "mean"), n=("red", "size")).reset_index().assign(
            zbin=lambda x: x.zbin.astype(str)).to_csv(os.path.join(RES, f"taker_bins_{label}.csv"), index=False)
    # cross-account response: does this account reprice when its sibling is filled?
    if label in ("A", "B"):
        sib = "B" if label == "A" else "A"
        evx = fill_event_skew(label, p, horizons=(5,), fill_label=sib)
        evo = ev[ev.h == 5].assign(dq=lambda x: x.fill_dq)
        out["cross_account"] = {}
        for tag, e in (("own_fills", evo), ("sibling_fills", evx)):
            if len(e) < 30:
                continue
            e = e.dropna()
            # regressor: minus the fill in $10k, so a positive coefficient is the textbook sign
            X = pd.DataFrame({"fill_bps_per_10k": -(e.fill_dq * e.S) / 1e4,
                              "db": 1e4 * e.dbasis / e.S, "dr": 1e4 * e.dret})
            y = 1e4 * e.dc / e.S
            r = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(e.coin)[0]})
            out["cross_account"][tag] = {"bps_per_10k_fill": float(r.params["fill_bps_per_10k"]),
                                         "t": float(r.tvalues["fill_bps_per_10k"]), "n": int(r.nobs), "coins": int(e.coin.nunique())}
    out["seconds"] = time.time() - t0
    print(f"[{label}] {out['minutes']:.0f} min, {len(params)} coins, {out['seconds']:.0f}s", flush=True)
    return out, p, params


def main():
    os.makedirs(RES, exist_ok=True)
    summary = {}
    panels, params = {}, {}
    for label in ("A", "B", "V"):
        summary[label], panels[label], params[label] = run_label(label)
    m, twin = breaks.twin_strategies(panels["A"], panels["B"], params["A"], params["B"])
    m.to_csv(os.path.join(RES, "twin_params.csv"), index=False)
    ok = m[(m.t_aug_A > 2) & (m.t_aug_B > 2)]
    twin["n_coins_both_signif"] = len(ok)
    if len(ok) > 5:
        twin["corr_log_gamma"] = float(np.corrcoef(np.log(ok.gamma_aug_A), np.log(ok.gamma_aug_B))[0, 1])
        twin["median_ratio_gamma"] = float((ok.gamma_aug_A / ok.gamma_aug_B).median())
    summary["twin"] = twin
    attr, bw = breaks.inventory_attribution(panels["A"], panels["V"], params["V"])
    summary["attribution"] = attr
    bw.to_csv(os.path.join(RES, "attribution_windows.csv"), index=False)
    jdump(summary, "summary.json")
    print(json.dumps({k: ({kk: v[kk] for kk in ("pooled", "coin_gamma", "overid")} if k in ("A", "B", "V") else v)
                      for k, v in summary.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
