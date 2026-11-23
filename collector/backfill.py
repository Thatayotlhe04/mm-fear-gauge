#!/usr/bin/env python3
"""One-shot historical backfill: HLP vault history, recent fills for both
strategies, 1-minute candles and hourly funding for every quoted coin."""
import json
import os
import subprocess
import time

import pyarrow as pa
import pyarrow.parquet as pq
import requests

API = "https://api.hyperliquid.xyz/info"
HLP = "0xdfc24b077bc1425ad1dea75bcb6f8158e10df303"
USERS = {"A": "0x010461c14e146ac35fe42271bdc1134ee31c703a",
         "B": "0x31ca8395cf837de08b24da3f660e77761dfb974b"}
OUT = "data/X"
S = requests.Session()
_budget = []  # (t, weight) for a rolling 60 s window


def post(body, weight=20):
    while True:
        now = time.time()
        while _budget and now - _budget[0][0] > 60:
            _budget.pop(0)
        if sum(w for _, w in _budget) + weight <= 1000:
            break
        time.sleep(1)
    _budget.append((time.time(), weight))
    for attempt in range(6):
        r = S.post(API, json=body, timeout=30)
        if r.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("rate limited repeatedly")


def save(name, rows, cols):
    if not rows:
        return
    os.makedirs(OUT, exist_ok=True)
    t = pa.table({c: [r[i] for r in rows] for i, c in enumerate(cols)})
    pq.write_table(t, os.path.join(OUT, f"{name}.parquet"), compression="zstd")
    print(name, len(rows), flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    now = int(time.time() * 1000)
    meta = post({"type": "metaAndAssetCtxs"})
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"))
    vd = {"HLP": post({"type": "vaultDetails", "vaultAddress": HLP})}
    for k, u in USERS.items():
        vd[k] = post({"type": "vaultDetails", "vaultAddress": u})
    json.dump(vd, open(os.path.join(OUT, "vaults.json"), "w"))

    cols = ["user", "time", "coin", "px", "sz", "side", "startPosition", "dir", "closedPnl",
            "oid", "crossed", "fee", "tid", "liq"]
    rows = []
    for k, u in USERS.items():
        start = now - 30 * 86400_000
        seen = set()
        while True:
            fl = post({"type": "userFillsByTime", "user": u, "startTime": start, "endTime": now,
                       "aggregateByTime": False}, weight=120)
            new = [x for x in fl if x["tid"] not in seen]
            for x in new:
                seen.add(x["tid"])
                rows.append((k, x["time"], x["coin"], float(x["px"]), float(x["sz"]), x["side"],
                             float(x["startPosition"]), x.get("dir"), float(x.get("closedPnl") or 0),
                             x["oid"], bool(x["crossed"]), float(x.get("fee") or 0), x["tid"],
                             bool(x.get("liquidation"))))
            if len(fl) < 2000 or not new:
                break
            start = max(x["time"] for x in fl)
    save("fills_backfill", rows, cols)

    oo = post({"type": "openOrders", "user": USERS["A"]})
    coins = sorted({o["coin"] for o in oo})
    crows, frows = [], []
    for i, c in enumerate(coins):
        try:
            cs = post({"type": "candleSnapshot", "req": {"coin": c, "interval": "1m",
                       "startTime": now - 5000 * 60_000, "endTime": now}}, weight=105)
            for x in cs:
                crows.append((c, x["t"], float(x["o"]), float(x["h"]), float(x["l"]), float(x["c"]),
                              float(x["v"]), x["n"]))
            fh = post({"type": "fundingHistory", "coin": c, "startTime": now - 7 * 86400_000}, weight=30)
            for x in fh:
                frows.append((c, x["time"], float(x["fundingRate"]), float(x["premium"])))
        except Exception as e:
            print("skip", c, e, flush=True)
        if i % 20 == 0:
            print(f"{i}/{len(coins)}", flush=True)
    save("candles_1m", crows, ["coin", "t", "o", "h", "l", "c", "v", "n"])
    save("funding", frows, ["coin", "time", "rate", "premium"])

    branch = f"data-X-r{os.environ.get('GITHUB_RUN_NUMBER', 'local')}"
    subprocess.run(f"git add -A {OUT} && git -c user.name=collector -c user.email=collector@users.noreply.github.com "
                   f"commit -qm 'backfill' && git push -q origin HEAD:refs/heads/{branch}", shell=True, check=True)


if __name__ == "__main__":
    main()
