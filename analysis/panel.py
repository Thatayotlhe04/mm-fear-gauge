"""Load collector output and build the per-(coin, poll) panel used everywhere else."""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
WORK = os.path.join(ROOT, "work")
USERS = {"A": "0x010461c14e146ac35fe42271bdc1134ee31c703a",
         "B": "0x31ca8395cf837de08b24da3f660e77761dfb974b"}
HLP_LIQUIDATORS = {"0x2e3d94f0562703b25c83308a05046ddaf9a8dd14", "0x2ed5c4484ea3ff8b57d5f2fb152a40d9f2b68308",
                   "0x5e177e5e39c0f4e421f5865a6d8beed8d921cb70", "0xb0a55f13d22f66e6d495ac98113841b2326e9540"}


def runs():
    return sorted(d for d in os.listdir(DATA) if d.startswith("r"))


def load(label, stream, run=None):
    if label == "V":
        return _load_vault(stream, run)
    pats = [os.path.join(DATA, r, label, stream, "*.parquet") for r in ([run] if run else runs())]
    files = sorted(f for p in pats for f in glob.glob(p))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "coin" in df:
        df["coin"] = df.coin.astype("str")
    return df


def _load_vault(stream, run=None):
    """The HLP vault as one agent: Strategy A + Strategy B. Market streams and the
    quote ladder come from A's collector (A and B post near-identical quotes); fills
    are the union; positions are summed coin by coin at A's snapshot times."""
    if stream == "fills":
        a, b = load("A", "fills", run), load("B", "fills", run)
        return pd.concat([a.assign(acct="A"), b.assign(acct="B")], ignore_index=True)
    if stream == "pos":
        a, b = load("A", "pos", run), load("B", "pos", run)
        if a.empty or b.empty:
            return a
        a = a.sort_values("t_recv")
        bt = np.sort(b.t_recv.unique())
        out = []
        for t, ga in a.groupby("t_recv"):
            i = np.searchsorted(bt, t, side="right") - 1
            if i < 0:
                continue
            gb = b[b.t_recv == bt[i]]
            m = ga.set_index("coin")[["szi", "pos_value", "upnl", "margin"]].add(
                gb.set_index("coin")[["szi", "pos_value", "upnl", "margin"]], fill_value=0)
            px = pd.concat([ga.set_index("coin").pos_value / ga.set_index("coin").szi.abs(),
                            gb.set_index("coin").pos_value / gb.set_index("coin").szi.abs()], axis=1).bfill(axis=1).iloc[:, 0]
            acct = m.loc["__ACCOUNT__"].copy() if "__ACCOUNT__" in m.index else None
            m["pos_value"] = (m.szi.abs() * px.reindex(m.index)).fillna(0)
            if acct is not None:
                m.loc["__ACCOUNT__", ["szi", "pos_value"]] = acct[["szi", "pos_value"]].values
            m = m.reset_index().assign(t_send=t, t_recv=t, entry=np.nan)
            out.append(m)
        return pd.concat(out, ignore_index=True)[["t_send", "t_recv", "coin", "szi", "entry", "pos_value", "upnl", "margin"]]
    return load("A", stream, run)


def inventory_events(label):
    if label == "V":
        ea, fa = inventory_events("A")
        eb, fb = inventory_events("B")
        tgrid = pd.concat([ea[["t", "coin"]], eb[["t", "coin"]]]).drop_duplicates().sort_values("t")
        ma = pd.merge_asof(tgrid, ea[["t", "coin", "q"]], on="t", by="coin", direction="backward")
        mb = pd.merge_asof(tgrid, eb[["t", "coin", "q"]], on="t", by="coin", direction="backward")
        ev = tgrid.assign(q=ma.q.fillna(0).values + mb.q.fillna(0).values, qa=ma.q.values, qb=mb.q.values)
        # the vault state is only known once both accounts have reported a full position snapshot
        t_start = max(ea[ea.src == "snap"].t.min(), eb[eb.src == "snap"].t.min())
        ev = ev[ev.t >= t_start]
        return ev.reset_index(drop=True), pd.concat([fa.assign(acct="A"), fb.assign(acct="B")], ignore_index=True)
    return _inventory_events_single(label)


def _inventory_events_single(label):
    """Inventory path per coin from fills (exact: startPosition +/- size) and
    clearinghouse snapshots (fills gaps for coins with no fills)."""
    fl = load(label, "fills").drop_duplicates("tid")
    fl["q_after"] = fl.startPosition + np.where(fl.side == "B", fl.sz, -fl.sz)
    ev_f = fl[["time", "coin", "q_after"]].rename(columns={"time": "t", "q_after": "q"})
    pos = load(label, "pos")
    pos = pos[pos.coin != "__ACCOUNT__"]
    ev_p = pos[["t_recv", "coin", "szi"]].rename(columns={"t_recv": "t", "szi": "q"})
    # coins absent from a snapshot are flat
    snaps = load(label, "pos")
    allc = set(fl.coin) | set(pos.coin)
    flat = []
    for t, grp in snaps.groupby("t_recv"):
        for c in allc - set(grp.coin):
            flat.append((t, c, 0.0))
    ev_flat = pd.DataFrame(flat, columns=["t", "coin", "q"])
    ev = pd.concat([ev_f.assign(src="fill"), ev_p.assign(src="snap"), ev_flat.assign(src="snap")])
    ev["t"] = ev.t.astype("int64"); ev["q"] = ev.q.astype(float); ev["coin"] = ev.coin.astype("str")
    return ev.sort_values("t").reset_index(drop=True), fl


def build_panel(label, cache=True):
    os.makedirs(WORK, exist_ok=True)
    path = os.path.join(WORK, f"panel_{label}.parquet")
    q = load(label, "quotes")
    if cache and os.path.exists(path):
        old = pd.read_parquet(path)
        if len(old) and old.t_recv.max() >= q.t_recv.max():
            return old
    ctx = load(label, "ctx").sort_values("t_recv")
    book = load(label, "book").sort_values("t_recv")
    q = q.sort_values("t_recv")
    p = pd.merge_asof(q, ctx[["t_recv", "coin", "mid", "oracle", "mark", "funding", "oi", "impb", "impa"]],
                      on="t_recv", by="coin", direction="backward", tolerance=3000)
    book["imb5"] = (book.bdep5 - book.adep5) / (book.bdep5 + book.adep5)
    p = pd.merge_asof(p, book[["t_recv", "coin", "imb5"]], on="t_recv", by="coin",
                      direction="backward", tolerance=20000)
    ev, _ = inventory_events(label)
    p = pd.merge_asof(p, ev[["t", "coin", "q"]].rename(columns={"t": "t_recv"}), on="t_recv", by="coin",
                      direction="backward")
    # short-horizon mid momentum (from the 1 Hz ctx mid)
    for h in (10, 60):
        lag = ctx[["t_recv", "coin", "mid"]].copy()
        lag["t_recv"] = lag.t_recv + h * 1000
        p = pd.merge_asof(p.sort_values("t_recv"), lag.rename(columns={"mid": f"mid_lag{h}"}).sort_values("t_recv"),
                          on="t_recv", by="coin", direction="backward", tolerance=3000)
        p[f"ret{h}"] = np.log(p.mid / p[f"mid_lag{h}"])
    p["S"] = p.mid
    p["t_snap"] = (p.t_send + p.t_recv) // 2     # the open-orders snapshot reflects state mid-request
    p["two_sided"] = p.b1.notna() & p.a1.notna()
    p["center"] = (p.b1 + p.a1) / 2
    p["c"] = p.center - p.S
    p["psi"] = (p.a1 - p.b1) / 2
    p["db"] = p.S - p.b1
    p["da"] = p.a1 - p.S
    p["basis"] = p.oracle - p.S
    p["Qusd"] = p.q * p.S
    p["c_bps"] = 1e4 * p.c / p.S
    p["psi_bps"] = 1e4 * p.psi / p.S
    p["basis_bps"] = 1e4 * p.basis / p.S
    p["imp_bps"] = 1e4 * ((p.impb + p.impa) / 2 - p.S) / p.S
    p = p.drop(columns=[c for c in p.columns if c.startswith("mid_lag")])
    p.to_parquet(path)
    return p


def aggressor_orders(label):
    """Collapse public trades into aggressive orders: one row per (coin, time,
    taker side, taker address) with the deepest price reached and total size."""
    tr = load(label, "trades").drop_duplicates("tid")
    tr["taker"] = np.where(tr.side == "B", tr.u0, tr.u1)
    tr["maker"] = np.where(tr.side == "B", tr.u1, tr.u0)
    ctx = load(label, "ctx")[["t_recv", "coin", "mid"]].sort_values("t_recv")
    tr = tr.sort_values("t")
    tr["t_key"] = tr.t - 1     # mid strictly before the trade
    m = pd.merge_asof(tr, ctx.rename(columns={"t_recv": "t_key"}), on="t_key", by="coin",
                      direction="backward", tolerance=3000)
    m["sgn"] = np.where(m.side == "B", 1.0, -1.0)
    m["reach"] = m.sgn * (m.px - m.mid)
    g = m.groupby(["coin", "t", "side", "taker"], sort=False).agg(
        reach=("reach", "max"), sz=("sz", "sum"), px=("px", "last"), mid=("mid", "first"),
        n=("tid", "size")).reset_index()
    return g, m
