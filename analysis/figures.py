"""Figures for the report. Reads results/ and work/ written by run_all.py."""
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import style  # noqa: E402
from estimators import intensity_fit  # noqa: E402
from panel import ROOT, aggressor_orders  # noqa: E402

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
LAB = {"V": "HLP vault (A+B)", "A": "Strategy A", "B": "Strategy B"}
MK = {"V": "o", "A": "s", "B": "^"}
LS = {"V": "-", "A": "--", "B": ":"}
COL = {"V": style.SERIES[0], "A": style.SERIES[1], "B": style.SERIES[2]}


def rd(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("wrote", name)


def fig_fear_gauge(summary):
    ss = rd("size_skew_windows_V.csv")
    tw = rd("taker_windows_V.csv")
    fg = rd("fear_gauge_V.csv")
    if ss is None or not len(ss):
        return
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.4), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.4, 1]})
    ax = axes[0]
    t = pd.to_datetime(ss.w)
    ax.fill_between(t, 100 * (ss.size_skew - 1.96 * ss.se), 100 * (ss.size_skew + 1.96 * ss.se), color=style.SERIES[0], alpha=0.14, lw=0)
    ax.plot(t, 100 * ss.size_skew, "-o", ms=3, color=style.SERIES[0])
    ax.axhline(0, color=style.MUTED, lw=0.8)
    ax.set_ylabel("size lean\n(pp of top-level imbalance\nper $1k vault inventory)")
    style.title(ax, "The fear gauge",
                "Top: how hard HLP leans its touch size against vault inventory (10-min cross-sections of ~170 coins, 95% CI)")
    ax = axes[1]
    if tw is not None and len(tw):
        th = pd.to_datetime(tw.h) + pd.Timedelta(minutes=15)
        S = summary["V"]["taker_threshold"]
        scale = S["reservation_skew_bps_per_10k"] * S["z_bar"]
        ax.plot(th, scale / tw.z_bar, "-s", ms=4, color=style.SERIES[0], label="unwind threshold, 30-min windows (lower bound)")
        ax.axhline(S["reservation_skew_bps_per_10k"], color=style.MUTED, lw=0.8, ls=":", label="unwind threshold, full sample")
    g5 = summary["V"]["price_skew"].get("event_5s")
    if g5:
        ax.axhspan(g5["bps_per_10k_lo"], g5["bps_per_10k_hi"], color=style.SERIES[1], alpha=0.10, lw=0)
        ax.axhline(g5["bps_per_10k"], color=style.SERIES[1], lw=1, ls="--", label="quote response to fill shocks (full sample)")
    ax.axhline(0, color=style.GRID, lw=0.8)
    ax.set_ylabel("bps of price\nper $10k inventory")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.text(0, 1.02, "Middle: reservation-price skew implied by when HLP starts dumping inventory with market orders",
            transform=ax.transAxes, color=style.INK2, fontsize=8.5)
    ax = axes[2]
    if fg is not None and len(fg):
        ax.plot(pd.to_datetime(fg.w), 1e4 * fg.mkt_vol, color=style.INK2, lw=1.3)
    ax.set_ylabel("median 1 s vol (bps)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("UTC")
    save(fig, "01_fear_gauge.png")


def fig_levers(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    for l in ("V", "A", "B"):
        b = rd(f"taker_bins_{l}.csv")
        if b is None:
            continue
        ax.plot(b.z, b.red, LS[l], marker=MK[l], ms=4, color=COL[l],
                label=f"{LAB[l]} inventory" + ("" if l == "V" else " (own account)"))
    S = summary["V"]["taker_threshold"]
    ax.axvline(S["z_bar"], color=style.MUTED, lw=0.8, ls=":")
    ax.text(S["z_bar"], 0.97, f" threshold ~${S['median_threshold_usd']:,.0f} per coin", color=style.INK2, fontsize=7.5, va="top")
    ax.axhline(0.5, color=style.GRID, lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(0.2, 1.0)
    ax.set_xlabel("inventory / crossing cost, in GLFT units  (z = |q| s / c)")
    ax.set_ylabel("share of taker fills that reduce inventory")
    ax.legend(loc="lower left", fontsize=7.5)
    style.title(ax, "Lever 1: unwind with market orders past a threshold",
                "Only vault inventory switches the behaviour on")
    ax = axes[1]
    b = rd("size_skew_bins_V.csv")
    if b is not None:
        ax.errorbar(b.Q1k, 100 * b.simb, yerr=196 * b.se, fmt="o-", ms=4, color=style.SERIES[0], lw=1.2, capsize=0)
    ax.axhline(0, color=style.MUTED, lw=0.8); ax.axvline(0, color=style.MUTED, lw=0.8)
    ax.set_xlabel("vault inventory in the coin ($k)")
    ax.set_ylabel("top-of-ladder size imbalance, bid - ask (%)")
    style.title(ax, "Lever 2: lean the size, not the price",
                "Less size on the side that would add to inventory")
    save(fig, "02_two_levers.png")


def fig_show_vs_do(summary):
    S = summary["V"]
    ps, tt = S["price_skew"], S["taker_threshold"]
    rows = []
    if ps.get("naive_levels"):
        v = ps["naive_levels"]
        rows.append(("naive textbook regression\nquote centre on inventory stock", v["bps_per_10k"],
                     v["theta"] * S["scale"]["bps_per_10k_per_theta"] if False else v["bps_per_10k"],
                     v["lo"] * S["scale"]["bps_per_10k_per_theta"], v["hi"] * S["scale"]["bps_per_10k_per_theta"], style.SERIES[2]))
    if ps.get("event_5s"):
        v = ps["event_5s"]
        rows.append(("identified: quote response to\nfills HLP did not choose (5 s)", v["bps_per_10k"], None, v["bps_per_10k_lo"],
                     v["bps_per_10k_hi"], style.SERIES[0]))
    rows.append(("identified: market-order unwind\nthreshold (lower bound)", tt["bps_per_10k"], None, tt["bps_per_10k_lo"],
                 tt["bps_per_10k_hi"], style.SERIES[1]))
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    for i, (name, x, _, lo, hi, col) in enumerate(rows):
        ax.errorbar(x, i, xerr=[[x - lo], [hi - x]], fmt="o", color=col, ms=7, capsize=0, lw=1.6)
        ax.text(x, i + 0.2, f"{x:+.0f}", ha="center", color=style.INK, fontsize=8.5)
    ax.axvline(0, color=style.MUTED, lw=0.8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("reservation-price skew, bps per $10k of vault inventory (typical coin, 95% CI)")
    g5 = ps.get("event_5s", {})
    style.title(ax, "The textbook regression says risk-neutral. Identified, it isn't.",
                f"gamma ~ {g5.get('gamma', float('nan')):.4f} per $ from fill shocks; at least {tt['gamma']:.4f} per $ from the unwind threshold")
    save(fig, "03_three_estimators.png")


def fig_event_profile():
    e = rd("event_profile_V.csv")
    if e is None or not len(e):
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for kind, col, mk, lab in (("passive", style.SERIES[0], "o", "after passive fills (shock HLP did not choose)"),
                               ("own_taker", style.SERIES[1], "s", "after HLP's own market orders")):
        g = e[e.kind == kind]
        ax.fill_between(g.h, g.lo, g.hi, color=col, alpha=0.12, lw=0)
        ax.plot(g.h, g.theta, "-", marker=mk, ms=4, color=col, label=lab)
    ax.axhline(0, color=style.MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([5, 30, 120, 300, 900, 1800]); ax.set_xticklabels(["5s", "30s", "2m", "5m", "15m", "30m"])
    ax.set_xlabel("time after the inventory shock")
    ax.set_ylabel("theta = sqrt(gamma), median coin")
    ax.legend(loc="upper left", fontsize=8)
    style.title(ax, "The lean is real, and it builds for minutes",
                "Quote-centre response per unit of GLFT-scaled inventory shock; GLFT re-centres instantly")
    save(fig, "10_event_profile.png")


def fig_overid():
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    allv = []
    for l in ("V",):
        p = rd(f"params_{l}.csv")
        if p is None:
            continue
        pred = 1e4 * p.psi_pred / p.S
        ax.scatter(pred, p.psi_bps, s=16, marker=MK[l], facecolor="none" if l == "B" else COL[l],
                   edgecolor=COL[l], lw=0.8, label=f"{LAB[l]}  (median ratio {np.median(p.psi_bps / pred):.2f})")
        allv += list(pred) + list(p.psi_bps)
    allv = np.array([v for v in allv if np.isfinite(v) and v > 0])
    lo, hi = allv.min() * 0.8, allv.max() * 1.2
    ax.plot([lo, hi], [lo, hi], color=style.MUTED, lw=1, ls=":")
    ax.text(hi * 0.6, hi * 0.5, "GLFT restriction\npsi = 1/k + (D/2) b", color=style.MUTED, fontsize=8, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("half spread the model allows given flow and skew (bps)")
    ax.set_ylabel("half spread HLP actually quotes (bps)")
    ax.legend(loc="upper left")
    style.title(ax, "Over-identification: HLP quotes wider than GLFT permits",
                "One point per coin. On the dotted line the textbook model is internally consistent.")
    save(fig, "04_overid_spread.png")


def fig_shape():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for l in ("V", "A", "B"):
        b = rd(f"shape_bins_{l}.csv")
        if b is None:
            continue
        ax.errorbar(b.z, b.e, yerr=1.96 * b.se, fmt=MK[l] + LS[l], color=COL[l], ms=4, lw=1.2,
                    capsize=0, label=LAB[l])
    ax.axhline(0, color=style.MUTED, lw=0.8)
    ax.axvline(0, color=style.MUTED, lw=0.8)
    ax.set_xlabel("inventory / coin's 95th-percentile |inventory|")
    ax.set_ylabel("quote-centre offset after removing reference price (bps)")
    ax.legend(loc="upper right")
    style.title(ax, "Inventory skew: not the straight line the model predicts",
                "Binned within-coin residuals, all coins pooled, 95% CI. GLFT: a line through the origin.")
    save(fig, "03_inventory_shape.png")


def fig_markouts():
    hs = [1, 5, 10, 30, 60, 300]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for l in ("V",):
        m = rd(f"markouts_{l}.csv")
        if m is None:
            continue
        for _, r in m.iterrows():
            y = [r.edge_bps] + [r[f"mo{h}"] for h in hs]
            x = [0.3] + hs
            ls = "-" if r.type == "maker" else ":"
            ax.plot(x, y, ls, marker="o" if r.type == "maker" else "s", ms=3.5,
                    color=style.SERIES[0] if r.type == "maker" else style.SERIES[1], label=f"{r.type} fills (n={int(r.n):,})")
    ax.axhline(0, color=style.MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.3] + hs)
    ax.set_xticklabels(["fill"] + [f"{h}s" for h in hs])
    ax.set_ylabel("markout from HLP's side (bps, $-weighted)")
    ax.legend(loc="lower left", fontsize=7.5)
    style.title(ax, "Adverse selection: maker fills lose their edge, taker fills gain",
                "Signed mid move after each fill. Textbook GLFT: maker curve flat at the captured half spread.")
    save(fig, "05_markouts.png")


def fig_reference(summary):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names = ["inventory_only", "all_but_inventory", "all"]
    labels = ["inventory only\n(textbook)", "reference price,\nflow, momentum", "everything"]
    x = np.arange(len(names))
    for i, l in enumerate(("V", "A", "B")):
        if l not in summary:
            continue
        w = summary[l]["reference_price"]["within_r2"]
        v = [w[n] for n in names]
        ax.bar(x + (i - 1) * 0.27, v, width=0.25, color=COL[l], label=LAB[l])
        for xi, vi in zip(x, v):
            ax.text(xi + (i - 1) * 0.27, vi + 0.005, f"{vi:.2f}", ha="center", va="bottom", fontsize=7.5, color=style.INK2)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("within-coin R^2 of the quote centre")
    ax.legend(loc="upper left")
    style.title(ax, "What moves HLP's quotes is mostly not inventory",
                "Share of within-coin variance in the quote-centre offset explained by each block")
    save(fig, "05_reference_price.png")


def fig_intensity():
    p = rd("params_V.csv")
    if p is None or not len(p):
        return
    coin = p.sort_values("n_aggr", ascending=False).coin.iloc[0]
    aggr, _ = aggressor_orders("V")
    pr = p.set_index("coin").loc[coin]
    ag = aggr[(aggr.coin == coin) & (aggr.reach > 0)]
    T = 1.0
    fit = intensity_fit(ag.reach.values, ag.sz.values / pr.D, T)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    g = 1e4 * fit["grid"] / pr.S
    ax.plot(g, fit["lam"] / fit["lam"].max(), "o", ms=3.5, color=style.INK, label="empirical")
    ax.plot(g, np.exp(-fit["k"] * fit["grid"]) * fit["A"] / fit["lam"].max(), "-", color=style.SERIES[1],
            label=f"exponential  R^2 {fit['r2_exp']:.2f}")
    c = np.exp(np.polyfit(np.log(fit["grid"]), np.log(fit["lam"]), 1))
    ax.plot(g, (fit["grid"] ** (-fit["alpha_pow"])) * (fit["lam"] / fit["grid"] ** (-fit["alpha_pow"])).mean() / fit["lam"].max(),
            "--", color=style.SERIES[2], label=f"power law  R^2 {fit['r2_pow']:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("distance from mid (bps)")
    ax.set_ylabel("flow reaching that distance (relative)")
    ax.legend(loc="lower left")
    style.title(ax, f"Fill curve, {coin}", "Aggressive orders collapsed to their deepest level")
    ax = axes[1]
    fs = []
    for l in ("V",):
        q = rd(f"params_{l}.csv")
        if q is not None:
            fs.append(q.flow_share.dropna())
    if fs:
        v = pd.concat(fs)
        v = v[(v > 0) & np.isfinite(v)]
        ax.hist(np.log10(v), bins=30, color=style.SERIES[1], edgecolor=style.BG, lw=0.5)
        ax.axvline(np.log10(v.median()), color=style.INK, lw=1)
        ax.text(np.log10(v.median()), ax.get_ylim()[1] * 0.92, f"  median {v.median():.2f}", color=style.INK, fontsize=8.5)
        ax.set_xlabel("log10 HLP flow share  (A_own / A_market)")
        ax.set_ylabel("coins")
        style.title(ax, "HLP is not the monopolist the model assumes",
                    "Share of the flow reaching HLP's price that HLP actually fills")
    save(fig, "06_intensity.png")


def fig_heterogeneity():
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for l in ("V",):
        p = rd(f"params_{l}.csv")
        if p is None:
            continue
        q = p[(p.t_aug > 2) & (p.gamma_aug > 0)].dropna(subset=["dayvlm"])
        ax.scatter(q.dayvlm / 1e6, q.gamma_aug, s=18, marker=MK[l], facecolor="none" if l == "B" else COL[l],
                   edgecolor=COL[l], lw=0.8, label=LAB[l])
        for _, r in q.nlargest(4, "dayvlm").iterrows():
            ax.annotate(r.coin, (r.dayvlm / 1e6, r.gamma_aug), fontsize=7, color=style.INK2,
                        xytext=(4, 2), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("24h notional volume ($M)")
    ax.set_ylabel("implied gamma (1/$), skew with controls")
    ax.legend(loc="upper right")
    style.title(ax, "One agent, many risk aversions",
                "A single CARA market maker would have one gamma for every coin. Coins with t > 2 shown.")
    save(fig, "07_gamma_heterogeneity.png")


def fig_twin():
    m = rd("twin_params.csv")
    if m is None:
        return
    q = m[(m.t_aug_A > 2) & (m.t_aug_B > 2) & (m.gamma_aug_A > 0) & (m.gamma_aug_B > 0)]
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(q.gamma_aug_A, q.gamma_aug_B, s=18, color=style.INK)
    if len(q):
        lo, hi = min(q.gamma_aug_A.min(), q.gamma_aug_B.min()) * 0.5, max(q.gamma_aug_A.max(), q.gamma_aug_B.max()) * 2
        ax.plot([lo, hi], [lo, hi], ls=":", color=style.MUTED)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("gamma, Strategy A"); ax.set_ylabel("gamma, Strategy B")
    style.title(ax, "Twin strategies, same coins", "Per-coin implied gamma; dotted = identical")
    save(fig, "08_twin_strategies.png")


def fig_simulation():
    s = rd("simulation_check.csv")
    if s is None:
        return
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for share, mk in ((1.0, "o"), (0.3, "s")):
        g = s[s.share == share]
        ax.scatter(g.gamma_true * (1 + 0.04 * (g.seed - 1)), g.gamma_own, marker=mk, s=26, color=style.SERIES[0],
                   label=f"own-fill calibration, MM gets {int(share*100)}% of flow")
        ax.scatter(g.gamma_true * (1 + 0.04 * (g.seed - 1)), g.gamma_mkt, marker=mk, s=26, facecolor="none",
                   edgecolor=style.SERIES[1], label=f"market-trade calibration, {int(share*100)}% of flow")
    lo, hi = 1e-4, 3e-2
    ax.plot([lo, hi], [lo, hi], ":", color=style.MUTED)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("true gamma"); ax.set_ylabel("recovered gamma")
    ax.legend(loc="upper left", fontsize=7.5)
    style.title(ax, "The estimator recovers a known gamma",
                "Simulated GLFT market maker, 4 h, polled like HLP; 3 seeds each")
    save(fig, "07_simulation_check.png")


def fig_ladder():
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for l in ("A",):
        lp = rd(f"ladder_{l}.csv")
        if lp is None:
            continue
        ax.plot(lp.dist_bps, lp.usd, LS[l], marker=MK[l], ms=4, color=COL[l], label=LAB[l])
        for _, r in lp.iterrows():
            if l == "A":
                ax.annotate(f"L{int(r['rank'])}", (r.dist_bps, r.usd), fontsize=7, color=style.INK2,
                            xytext=(3, 4), textcoords="offset points")
    ax.set_xlabel("median distance from mid (bps)")
    ax.set_ylabel("median order size ($)")
    ax.legend(loc="upper left")
    style.title(ax, "Eight quotes a side, not one", "HLP's resting ladder by level; GLFT posts a single quote per side")
    save(fig, "08_ladder.png")


def fig_mirror():
    from panel import load
    pa, pb = load("A", "pos"), load("B", "pos")
    la = pa[pa.t_recv == pa.t_recv.max()].set_index("coin")
    lb = pb[pb.t_recv == pb.t_recv.max()].set_index("coin")
    j = la[["szi", "pos_value"]].join(lb[["szi", "pos_value"]], lsuffix="_A", rsuffix="_B", how="outer").fillna(0)
    j = j.drop("__ACCOUNT__", errors="ignore")
    xa, xb = np.sign(j.szi_A) * j.pos_value_A / 1e3, np.sign(j.szi_B) * j.pos_value_B / 1e3
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axes[0]
    lim = max(np.abs(xa).max(), np.abs(xb).max()) * 1.1
    ax.plot([-lim, lim], [lim, -lim], ":", color=style.MUTED, lw=1)
    ax.scatter(xa, xb, s=12, color=style.INK, lw=0)
    for c in (np.abs(xa) + np.abs(xb)).nlargest(5).index:
        ax.annotate(c, (xa[c], xb[c]), fontsize=7, color=style.INK2, xytext=(4, 2), textcoords="offset points")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color=style.GRID, lw=0.8); ax.axvline(0, color=style.GRID, lw=0.8)
    ax.set_xlabel("Strategy A position ($k, signed)"); ax.set_ylabel("Strategy B position ($k, signed)")
    rho = np.corrcoef(xa, xb)[0, 1]
    style.title(ax, "Two accounts, one book", f"Every open position, end of sample. Correlation {rho:.3f}; dotted = exact mirror")
    ax = axes[1]
    try:
        v = json.load(open(os.path.join(ROOT, "data", "r1", "X", "vaults.json")))
        for l, ls in (("A", "-"), ("B", "--")):
            ph = dict(v[l]["portfolio"])["allTime"]["pnlHistory"]
            t = pd.to_datetime([x[0] for x in ph], unit="ms")
            y = np.array([float(x[1]) for x in ph]) / 1e6
            ax.plot(t, y, ls, color=COL[l], label=f"{LAB[l]}  ({y[-1]:+.2f}M)")
        ax.axhline(0, color=style.MUTED, lw=0.8)
        ax.set_ylabel("cumulative PnL ($M)")
        ax.legend(loc="upper left")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        style.title(ax, "and mirrored PnL", "All-time PnL history reported by Hyperliquid for each sub-vault")
    except Exception as e:
        print("pnl panel skipped", e)
    save(fig, "00_mirror_book.png")


def fig_exposure():
    from panel import load
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for l in ("A", "B", "V"):
        pos = load(l, "pos")
        pos = pos[pos.coin != "__ACCOUNT__"]
        g = pos.groupby("t_recv").pos_value.sum() / 1e6
        t = pd.to_datetime(g.index, unit="ms")
        ax.plot(t, g.values, LS[l], color=COL[l], label=LAB[l])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 3])
    ax.set_ylabel("gross open notional ($M, log)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(loc="center right")
    style.title(ax, "Each account carries ~$2.9M; the vault carries a fraction of that",
                "Sum over coins of |position| x price. Vault = A + B netted coin by coin.")
    save(fig, "09_exposure.png")


def fig_excess_vs_as():
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    p = rd("params_V.csv")
    if p is None or "as60_bps" not in p:
        return
    q = p[(p.n_maker_mo >= 20) & np.isfinite(p.excess_bps)]
    ax.scatter(q.as60_bps, q.excess_bps, s=16, color=style.INK)
    lim = [min(q.as60_bps.min(), q.excess_bps.min(), 0) - 2, max(q.as60_bps.max(), q.excess_bps.max()) + 2]
    ax.plot(lim, lim, ":", color=style.MUTED)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("adverse selection on passive fills, 60 s (bps)")
    ax.set_ylabel("excess half spread over GLFT (bps)")
    rho = q.excess_bps.corr(q.as60_bps)
    style.title(ax, "The missing term is adverse selection",
                f"Per coin. Excess = quoted half spread - 1/k - (D/2) b. Correlation {rho:.2f}; dotted = one-for-one")
    save(fig, "12_excess_vs_adverse_selection.png")


def main():
    style.apply()
    os.makedirs(FIG, exist_ok=True)
    summary = json.load(open(os.path.join(RES, "summary.json")))
    for f in (fig_mirror, fig_exposure, lambda: fig_fear_gauge(summary), lambda: fig_levers(summary),
              lambda: fig_show_vs_do(summary), fig_event_profile, fig_overid, fig_markouts, fig_intensity, fig_simulation,
              fig_ladder):
        try:
            f()
        except Exception as e:  # keep going: one broken figure should not stop the rest
            print("figure failed:", getattr(f, "__name__", f), repr(e))


if __name__ == "__main__":
    main()
