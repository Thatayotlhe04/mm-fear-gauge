"""How much of HLP's post-fill repricing is inventory and how much is news?

A passive fill changes inventory, but it also tells the market maker its quote was stale.
Two separations:
  1. size vs direction: per coin, regress the 5 s quote-centre change on the GLFT-scaled
     fill (inventory: scales with size) and on the fill's sign alone (news: direction).
  2. HLP's own market orders: inventory shocks HLP chose, so they carry no stale-quote news.
The size coefficient under (1), for passive fills and for HLP's own market orders, brackets
the inventory component. Assumption: the news a fill carries is in its direction, not its size.

    python analysis/decompose.py      (after run_all.py)
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from model import fill_event_skew, risk_scale  # noqa: E402
from panel import ROOT, build_panel  # noqa: E402

RES = os.path.join(ROOT, "results")


def main(n_boot=1000, seed=0):
    path = os.path.join(ROOT, "work", "panel_V.parquet")
    p = pd.read_parquet(path) if os.path.exists(path) else build_panel("V")
    pr = pd.read_csv(os.path.join(RES, "params_V.csv"))
    pr = pr[pr.calib == "own"].set_index("coin")
    sc = risk_scale(pr)
    S = json.load(open(os.path.join(RES, "summary.json")))["V"]
    scale, sig = S["scale"]["bps_per_10k_per_theta"], S["scale"]["median_sigma_bps_s"] / 1e4
    rng = np.random.default_rng(seed)

    def med(v):
        v = np.asarray(v)
        bs = [np.median(rng.choice(v, len(v))) for _ in range(n_boot)]
        return float(np.median(v)), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))

    def conv(th):
        g = np.sign(th) * th ** 2
        return {"gamma": float(g), "bps_per_10k": float(th * scale),
                "hold_cost_bps_10k_1h": float(1e4 * 0.5 * g * sig ** 2 * 1e4 * 3600)}

    out = {"assumption": "news is carried by a fill's direction, inventory by its size"}
    for kind, crossed in (("passive", False), ("own_taker", True)):
        ev = fill_event_skew("V", p, horizons=(5,), crossed=crossed)
        e = ev[ev.coin.isin(pr.index) & (ev.fill_dq != 0)].dropna().copy()
        e["x"] = -1e4 * sc.reindex(e.coin).values * e.fill_dq / e.S
        e["sgn"] = -np.sign(e.fill_dq)
        e["y"] = 1e4 * e.dc / e.S
        e["db"] = 1e4 * e.dbasis / e.S
        e["dr"] = 1e4 * e.dret
        for c in ("x", "y", "db", "dr"):
            lo, hi = e[c].quantile([0.01, 0.99])
            e[c] = e[c].clip(lo, hi)
        size_only, size, news = [], [], []
        for _, g in e.groupby("coin"):
            if len(g) < 15 or g.x.std() == 0:
                continue
            size_only.append(sm.OLS(g.y, g[["x", "db", "dr"]]).fit().params["x"])
            if g.sgn.nunique() == 2 and g.x.abs().std() > 0:
                r = sm.OLS(g.y, sm.add_constant(g[["x", "sgn", "db", "dr"]], has_constant="add")).fit()
                size.append(r.params["x"])
                news.append(r.params["sgn"])
        a, b, c = med(size_only), med(size), med(news)
        out[kind] = {"coins": len(size), "theta_size_only": a[0], "theta_size_only_ci": a[1:],
                     "theta_inventory": b[0], "theta_inventory_ci": b[1:], "share_positive": float(np.mean(np.array(size) > 0)),
                     "news_bps": c[0], "news_bps_ci": c[1:], **conv(b[0])}
    lo, hi = sorted([out["own_taker"]["theta_inventory"], out["passive"]["theta_inventory"]])
    out["bracket"] = {"theta": [lo, hi], "gamma": [conv(lo)["gamma"], conv(hi)["gamma"]],
                      "bps_per_10k": [conv(lo)["bps_per_10k"], conv(hi)["bps_per_10k"]],
                      "hold_cost_bps_10k_1h": [conv(lo)["hold_cost_bps_10k_1h"], conv(hi)["hold_cost_bps_10k_1h"]]}
    json.dump(out, open(os.path.join(RES, "decomposition.json"), "w"), indent=1)
    print(json.dumps(out["bracket"]), "| passive news", round(out["passive"]["news_bps"], 2), "bps")


if __name__ == "__main__":
    main()
