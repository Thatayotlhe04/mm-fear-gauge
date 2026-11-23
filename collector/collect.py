#!/usr/bin/env python3
"""Hyperliquid market-maker collector.

Streams the public market state for every coin a target account quotes and
reconstructs that account's quote ladder by polling its open orders. Writes
compressed parquet chunks and pushes them to a per-run data branch.

Streams (one parquet directory each):
  quotes   per poll, per coin: MM top-3 ladder both sides + market BBO at receipt
  ladder   every Nth poll: the MM's full ladder (long format)
  book     every l2Book push: top of book, level-2 prices, cumulative depth
  trades   every public trade (with buyer/seller addresses)
  fills    the target account's fills (websocket, includes taker/maker flag)
  mids     allMids changes for universe coins
  ctx      activeAssetCtx: oracle, mark, funding, open interest, impact prices
  pos      clearinghouseState: per-coin inventory + account value
  horders  historicalOrders snapshots: order lifetimes, IOC usage
"""
import argparse
import asyncio
import os
import subprocess
import sys
import time
import traceback

import aiohttp
import orjson
import pyarrow as pa
import pyarrow.parquet as pq

API = "https://api.hyperliquid.xyz/info"
WS_URL = "wss://api.hyperliquid.xyz/ws"

SCHEMAS = {
    "quotes": ["t_send", "t_recv", "coin", "nb", "na",
               "b1", "bs1", "b2", "bs2", "b3", "bs3",
               "a1", "as1", "a2", "as2", "a3", "as3",
               "btot", "atot", "bt1", "at1",
               "mb1", "mbs1", "ma1", "mas1", "mbt", "amid", "amid_t"],
    "ladder": ["t_recv", "coin", "side", "px", "sz", "ts", "oid"],
    "book": ["t", "t_recv", "coin", "b1", "bs1", "a1", "as1", "b2", "a2",
             "bdep5", "adep5", "bdep20", "adep20", "bn1", "an1"],
    "trades": ["t", "t_recv", "coin", "side", "px", "sz", "tid", "u0", "u1"],
    "fills": ["time", "t_recv", "coin", "px", "sz", "side", "startPosition", "dir",
              "closedPnl", "oid", "crossed", "fee", "tid", "hash", "snapshot", "liq"],
    "mids": ["t_recv", "coin", "mid"],
    "ctx": ["t_recv", "coin", "oracle", "mark", "mid", "funding", "oi", "premium",
            "impb", "impa", "dayvlm"],
    "pos": ["t_send", "t_recv", "coin", "szi", "entry", "pos_value", "upnl", "margin"],
    "horders": ["t_recv", "coin", "side", "px", "sz", "orig_sz", "oid", "ts", "tif",
                "status", "status_t", "reduce_only"],
}


def now_ms():
    return int(time.time() * 1000)


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class Sink:
    def __init__(self, root, label):
        self.root, self.label = root, label
        self.buf = {k: [] for k in SCHEMAS}
        self.total = {k: 0 for k in SCHEMAS}

    def add(self, stream, row):
        self.buf[stream].append(row)
        self.total[stream] += 1

    def flush(self):
        ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        n = 0
        for stream, rows in self.buf.items():
            if not rows:
                continue
            self.buf[stream] = []
            cols = list(zip(*rows))
            table = pa.table({name: list(c) for name, c in zip(SCHEMAS[stream], cols)})
            d = os.path.join(self.root, stream)
            os.makedirs(d, exist_ok=True)
            pq.write_table(table, os.path.join(d, f"{stream}_{self.label}_{ts}.parquet"),
                           compression="zstd", compression_level=9)
            n += len(rows)
        return n


class Collector:
    def __init__(self, args):
        self.a = args
        self.user = args.user.lower()
        self.sink = Sink(args.out, args.label)
        self.book = {}      # coin -> (b1, bs1, a1, as1, t)
        self.mid = {}       # coin -> (mid, t_recv)
        self.universe = set()
        self.poll_interval = args.poll
        self.stats = {"polls": 0, "poll_ms": [], "http429": 0, "http_err": 0,
                      "ws_reconnect": 0, "ws_msgs": 0, "task_crash": 0}
        self.stop_at = time.time() + args.minutes * 60
        self.log_path = os.path.join(args.out, "status.log")
        os.makedirs(args.out, exist_ok=True)

    def log(self, msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as fh:
            fh.write(line + "\n")

    # ---------------------------------------------------------------- REST
    async def post(self, body, timeout=10):
        async with self.http.post(API, data=orjson.dumps(body),
                                  headers={"Content-Type": "application/json"},
                                  timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status == 429:
                self.stats["http429"] += 1
                raise RateLimited()
            if r.status != 200:
                self.stats["http_err"] += 1
                raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:200]}")
            return orjson.loads(await r.read())

    async def poll_orders(self):
        k = 0
        while time.time() < self.stop_at:
            t0 = time.time()
            t_send = now_ms()
            try:
                oo = await self.post({"type": "openOrders", "user": self.user})
            except RateLimited:
                self.poll_interval = min(self.poll_interval * 1.5, 10.0)
                self.log(f"429 on openOrders; interval -> {self.poll_interval:.2f}s")
                await asyncio.sleep(5)
                continue
            except Exception as e:
                self.log(f"openOrders error: {e!r}")
                await asyncio.sleep(2)
                continue
            t_recv = now_ms()
            self.stats["polls"] += 1
            self.stats["poll_ms"].append(t_recv - t_send)
            self.poll_interval = max(self.a.poll, self.poll_interval * 0.98)
            by = {}
            for o in oo:
                by.setdefault(o["coin"], ([], []))[0 if o["side"] == "B" else 1].append(o)
            self.universe.update(by.keys())
            ladder = (k % self.a.ladder_every == 0)
            for coin in self.universe:
                bids, asks = by.get(coin, ([], []))
                bids.sort(key=lambda o: -float(o["limitPx"]))
                asks.sort(key=lambda o: float(o["limitPx"]))
                bl = [(f(o["limitPx"]), f(o["sz"])) for o in bids[:3]] + [(None, None)] * 3
                al = [(f(o["limitPx"]), f(o["sz"])) for o in asks[:3]] + [(None, None)] * 3
                mb = self.book.get(coin, (None, None, None, None, None))
                am = self.mid.get(coin, (None, None))
                self.sink.add("quotes", (
                    t_send, t_recv, coin, len(bids), len(asks),
                    bl[0][0], bl[0][1], bl[1][0], bl[1][1], bl[2][0], bl[2][1],
                    al[0][0], al[0][1], al[1][0], al[1][1], al[2][0], al[2][1],
                    sum(f(o["sz"]) for o in bids), sum(f(o["sz"]) for o in asks),
                    bids[0]["timestamp"] if bids else None,
                    asks[0]["timestamp"] if asks else None,
                    mb[0], mb[1], mb[2], mb[3], mb[4], am[0], am[1]))
                if ladder:
                    for o in bids + asks:
                        self.sink.add("ladder", (t_recv, coin, o["side"], f(o["limitPx"]),
                                                 f(o["sz"]), o["timestamp"], o["oid"]))
            k += 1
            await asyncio.sleep(max(0.0, self.poll_interval - (time.time() - t0)))

    async def poll_positions(self):
        while time.time() < self.stop_at:
            t_send = now_ms()
            try:
                ch = await self.post({"type": "clearinghouseState", "user": self.user})
                t_recv = now_ms()
                ms = ch.get("marginSummary", {})
                self.sink.add("pos", (t_send, t_recv, "__ACCOUNT__", f(ms.get("accountValue")),
                                      None, f(ms.get("totalNtlPos")), None,
                                      f(ms.get("totalMarginUsed"))))
                for ap in ch.get("assetPositions", []):
                    p = ap["position"]
                    self.sink.add("pos", (t_send, t_recv, p["coin"], f(p["szi"]), f(p.get("entryPx")),
                                          f(p.get("positionValue")), f(p.get("unrealizedPnl")),
                                          f(p.get("marginUsed"))))
            except RateLimited:
                await asyncio.sleep(10)
            except Exception as e:
                self.log(f"clearinghouseState error: {e!r}")
            await asyncio.sleep(self.a.pos_every)

    async def poll_historical_orders(self):
        while time.time() < self.stop_at:
            try:
                ho = await self.post({"type": "historicalOrders", "user": self.user}, timeout=20)
                t_recv = now_ms()
                for h in ho:
                    o = h["order"]
                    self.sink.add("horders", (t_recv, o["coin"], o["side"], f(o["limitPx"]), f(o["sz"]),
                                              f(o.get("origSz")), o["oid"], o["timestamp"],
                                              o.get("tif"), h.get("status"), h.get("statusTimestamp"),
                                              bool(o.get("reduceOnly"))))
            except RateLimited:
                await asyncio.sleep(30)
            except Exception as e:
                self.log(f"historicalOrders error: {e!r}")
            await asyncio.sleep(self.a.horders_every)

    # ---------------------------------------------------------------- WS
    def on_msg(self, m, t_recv):
        ch = m.get("channel")
        d = m.get("data")
        self.stats["ws_msgs"] += 1
        if ch == "l2Book":
            coin = d["coin"]
            bids, asks = d["levels"]
            b1 = bids[0] if bids else None
            a1 = asks[0] if asks else None
            row = (d.get("time"), t_recv, coin,
                   f(b1["px"]) if b1 else None, f(b1["sz"]) if b1 else None,
                   f(a1["px"]) if a1 else None, f(a1["sz"]) if a1 else None,
                   f(bids[1]["px"]) if len(bids) > 1 else None,
                   f(asks[1]["px"]) if len(asks) > 1 else None,
                   sum(float(x["sz"]) for x in bids[:5]), sum(float(x["sz"]) for x in asks[:5]),
                   sum(float(x["sz"]) for x in bids[:20]), sum(float(x["sz"]) for x in asks[:20]),
                   b1["n"] if b1 else None, a1["n"] if a1 else None)
            self.sink.add("book", row)
            self.book[coin] = (row[3], row[4], row[5], row[6], row[0])
        elif ch == "trades":
            for t in d:
                u = t.get("users") or [None, None]
                self.sink.add("trades", (t["time"], t_recv, t["coin"], t["side"], f(t["px"]),
                                         f(t["sz"]), t["tid"], u[0], u[1]))
        elif ch == "userFills":
            snap = bool(d.get("isSnapshot"))
            for x in d.get("fills", []):
                self.sink.add("fills", (x["time"], t_recv, x["coin"], f(x["px"]), f(x["sz"]), x["side"],
                                        f(x["startPosition"]), x.get("dir"), f(x.get("closedPnl")),
                                        x["oid"], bool(x["crossed"]), f(x.get("fee")), x["tid"],
                                        x.get("hash"), snap, bool(x.get("liquidation"))))
        elif ch == "allMids":
            for coin, px in d["mids"].items():
                if coin not in self.universe:
                    continue
                v = float(px)
                old = self.mid.get(coin)
                if old is None or old[0] != v:
                    self.mid[coin] = (v, t_recv)
                    self.sink.add("mids", (t_recv, coin, v))
        elif ch == "activeAssetCtx":
            c = d["ctx"]
            imp = c.get("impactPxs") or [None, None]
            self.sink.add("ctx", (t_recv, d["coin"], f(c.get("oraclePx")), f(c.get("markPx")),
                                  f(c.get("midPx")), f(c.get("funding")), f(c.get("openInterest")),
                                  f(c.get("premium")), f(imp[0]), f(imp[1]), f(c.get("dayNtlVlm"))))
        elif ch == "error":
            self.log(f"ws error message: {str(d)[:200]}")

    async def ws_loop(self, name, subs):
        backoff = 1
        while time.time() < self.stop_at:
            try:
                async with self.http.ws_connect(WS_URL, max_msg_size=0, heartbeat=None) as ws:
                    for s in subs:
                        await ws.send_str(orjson.dumps({"method": "subscribe", "subscription": s}).decode())
                        await asyncio.sleep(0.02)
                    self.log(f"ws {name}: subscribed {len(subs)}")
                    backoff = 1
                    last_ping = time.time()
                    while time.time() < self.stop_at:
                        if time.time() - last_ping > 30:
                            await ws.send_str('{"method":"ping"}')
                            last_ping = time.time()
                        try:
                            msg = await ws.receive(timeout=35)
                        except asyncio.TimeoutError:
                            raise RuntimeError("ws receive timeout")
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self.on_msg(orjson.loads(msg.data), now_ms())
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR,
                                          aiohttp.WSMsgType.CLOSE):
                            raise RuntimeError(f"ws closed: {msg.type}")
            except Exception as e:
                self.stats["ws_reconnect"] += 1
                self.log(f"ws {name} reconnect after: {e!r}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    # ---------------------------------------------------------------- housekeeping
    def git_push(self):
        branch = f"data-{self.a.label}-r{os.environ.get('GITHUB_RUN_NUMBER', 'local')}"
        cmd = (f"git add -A {self.a.out} && git -c user.name=collector -c user.email=collector@users.noreply.github.com "
               f"commit -qm 'data {self.a.label} {time.strftime('%H:%M:%S', time.gmtime())}' && "
               f"git push -q origin HEAD:refs/heads/{branch}")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            self.log(f"git push failed: {r.stderr.strip()[:300]}")
        return r.returncode == 0

    async def housekeeping(self):
        start = time.time()
        next_flush = start + self.a.first_flush * 60
        next_status = start + 60
        while True:
            await asyncio.sleep(5)
            done = time.time() >= self.stop_at
            if time.time() >= next_status or done:
                pm = sorted(self.stats["poll_ms"]) or [0]
                self.log("status " + " ".join(f"{k}={v}" for k, v in self.sink.total.items()) +
                         f" | polls={self.stats['polls']} poll_p50={pm[len(pm)//2]}ms poll_p90={pm[int(len(pm)*.9)]}ms"
                         f" interval={self.poll_interval:.2f}s 429={self.stats['http429']} http_err={self.stats['http_err']}"
                         f" ws_msgs={self.stats['ws_msgs']} ws_reconnect={self.stats['ws_reconnect']}"
                         f" universe={len(self.universe)}")
                self.stats["poll_ms"] = []
                next_status = time.time() + 60
            if time.time() >= next_flush or done:
                n = await asyncio.get_running_loop().run_in_executor(None, self.sink.flush)
                ok = await asyncio.get_running_loop().run_in_executor(None, self.git_push)
                self.log(f"flushed {n} rows, push ok={ok}")
                next_flush = time.time() + self.a.flush_every * 60
            if done:
                return

    async def supervise(self, name, factory):
        while time.time() < self.stop_at:
            try:
                await factory()
            except Exception:
                self.stats["task_crash"] += 1
                self.log(f"task {name} crashed:\n{traceback.format_exc()[-1500:]}")
                await asyncio.sleep(2)

    async def run(self):
        conn = aiohttp.TCPConnector(limit=20, ttl_dns_cache=300)
        async with aiohttp.ClientSession(connector=conn) as http:
            self.http = http
            oo = await self.post({"type": "openOrders", "user": self.user})
            self.universe = {o["coin"] for o in oo}
            coins = sorted(self.universe)
            self.log(f"start user={self.user} label={self.a.label} coins={len(coins)} orders={len(oo)}")
            with open(os.path.join(self.a.out, "universe.txt"), "w") as fh:
                fh.write("\n".join(coins) + "\n")
            half = len(coins) // 2
            ws_groups = {
                "book1": [{"type": "l2Book", "coin": c} for c in coins[:half]],
                "book2": [{"type": "l2Book", "coin": c} for c in coins[half:]],
                "trades": [{"type": "trades", "coin": c} for c in coins],
                "ctx": [{"type": "activeAssetCtx", "coin": c} for c in coins],
                "user": [{"type": "userFills", "user": self.user}, {"type": "allMids"}],
            }
            tasks = [asyncio.create_task(self.supervise(n, lambda s=s, n=n: self.ws_loop(n, s)))
                     for n, s in ws_groups.items()]
            tasks += [asyncio.create_task(self.supervise("orders", self.poll_orders)),
                      asyncio.create_task(self.supervise("pos", self.poll_positions)),
                      asyncio.create_task(self.supervise("horders", self.poll_historical_orders))]
            await self.housekeeping()
            for t in tasks:
                t.cancel()
        self.log("done")


class RateLimited(Exception):
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--minutes", type=float, default=340)
    p.add_argument("--out", default=None)
    p.add_argument("--poll", type=float, default=1.4, help="openOrders poll interval (s)")
    p.add_argument("--ladder-every", type=int, default=10)
    p.add_argument("--pos-every", type=float, default=10)
    p.add_argument("--horders-every", type=float, default=120)
    p.add_argument("--first-flush", type=float, default=3)
    p.add_argument("--flush-every", type=float, default=10)
    a = p.parse_args()
    a.out = a.out or os.path.join("data", a.label)
    asyncio.run(Collector(a).run())


if __name__ == "__main__":
    sys.exit(main())
