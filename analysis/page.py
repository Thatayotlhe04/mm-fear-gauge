"""Render the shareable one-page report (report/page/index.html) from the same values as the README."""
import html
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))
from panel import ROOT  # noqa: E402
from report import readme_values  # noqa: E402

OUT = os.path.join(ROOT, "report", "page")
REPO = "https://github.com/Thatayotlhe04/mm-fear-gauge"


def fig(name, caption):
    return (f'<figure><img src="figures/{name}" alt="{html.escape(caption)}" loading="lazy">'
            f'<figcaption>{caption}</figcaption></figure>')


def main():
    v, S = readme_values()
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    for f in os.listdir(os.path.join(ROOT, "figures")):
        if f.endswith(".png"):
            shutil.copy(os.path.join(ROOT, "figures", f), os.path.join(OUT, "figures", f))
    rows = [
        ("One agent, one book", "Two mirrored sub-vaults, risk managed on the net", f"position correlation {v['CORR_POS']}"),
        ("Inventory managed by price skew", "Size lean and market-order unwinds carry much of it", "size lean t " + v["SS_BETWEEN_T"] + "; unwind LR " + v["LR"]),
        ("Only limit orders", f"{v['TAKER_SHARE']} of fills are taker", f"{v['RED_SHARE']} of taker volume reduces vault inventory"),
        ("Quotes re-centre instantly", "The price response builds for minutes", "fill-event profile"),
        ("Fills carry no information", f"Passive fills lose {v['AS60']} of {v['EDGE']} bps within 60 s", "markouts"),
        ("Spread = 1/k + (D/2) skew", f"{v['PSI_RATIO']}x the GLFT value, median coin", f"{v['UNRAT']} of coins tighter than any gamma allows"),
        ("Gets all flow at its price", f"Fills {v['FLOW_SHARE']} of it", "own-fill Poisson calibration"),
        ("One quote per side", "Eight geometric levels", f"{v['LADDER_D']} bps"),
        ("Exponential fill curve", "Holds up", f"beats a power law in {v['EXP_SHARE']} of coins"),
    ]
    table = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)
    kpis = [
        (v["CORR_POS"], "correlation of Strategy A and B positions across " + v["HELD"] + " coins"),
        (v["EV_GAMMA"], "gamma per $, identified from fill shocks (CI " + v["EV_GAMMA_CI"] + ")"),
        (v["EV_HOLD"].split(" bps")[0] + " bps", "what holding $10k of a typical coin for an hour costs it, as priced"),
        ("$" + v["THRESH"], "vault inventory per coin at which it starts dumping with market orders"),
    ]
    kpi_html = "".join(f'<div class="k"><b>{a}</b><span>{b}</span></div>' for a, b in kpis)
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Maker Fear Gauge</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@300;500&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
/* VOICE  a quant desk's research note: clinical, exact, no decoration
   IDEA   a printed research note with figures from the trading screen
   PICTURE the mirror-book scatter: two accounts, one book
   COLOUR and TYPE  near-black ground, off-white ink, a grey ramp; Inter Tight 300 for figures, Inter for text
   LAYOUT finding as title, four figures, the picture, then one reading column with wide figures */
:root{{color-scheme:dark;--bg:#0d0d0d;--ink:#ededed;--ink2:#a3a3a3;--mute:#6b6b6b;--rule:#262626}}
:root[data-theme=light],:root[data-theme=dark]{{color-scheme:dark}}
*{{box-sizing:border-box;min-width:0}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16.5px/1.6 Inter,system-ui,sans-serif;overflow-x:clip}}
.band{{padding:clamp(28px,4vw,64px) max(5vw,16px)}}
.wrap{{max-width:1180px;margin:0 auto}}
.col{{max-width:720px}}
h1{{font:300 clamp(32px,4.6vw,52px)/1.05 "Inter Tight",Inter,sans-serif;letter-spacing:-.01em;margin:0 0 14px;text-wrap:balance;max-width:20ch}}
h2{{font:500 clamp(22px,2.4vw,28px)/1.2 "Inter Tight",Inter,sans-serif;margin:0 0 12px;text-wrap:balance}}
p{{margin:0 0 14px;color:var(--ink)}}
.lede{{color:var(--ink2);font-size:18px;max-width:62ch}}
.kpis{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--rule);border-block:1px solid var(--rule);margin:28px 0}}
.k{{background:var(--bg);padding:18px 18px 18px 0}}
.k b{{display:block;font:300 clamp(28px,3.4vw,42px)/1.05 "Inter Tight",Inter,sans-serif;white-space:nowrap}}
.k span{{display:block;color:var(--ink2);font-size:13.5px;margin-top:6px}}
figure{{margin:22px 0 30px}}
figure img{{width:100%;height:auto;display:block;border:1px solid var(--rule)}}
figcaption{{color:var(--mute);font-size:13.5px;margin-top:8px;max-width:72ch}}
section{{border-top:1px solid var(--rule)}}
.grid2{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:28px}}
table{{border-collapse:collapse;width:100%;font-size:14.5px}}
th,td{{text-align:left;padding:10px 12px 10px 0;border-bottom:1px solid var(--rule);vertical-align:top}}
th{{color:var(--ink2);font-weight:500}}
td:first-child{{color:var(--ink2)}}
a{{color:var(--ink);text-underline-offset:3px}}
code{{font-size:14px;overflow-wrap:anywhere;color:var(--ink2)}}
.nav{{display:flex;gap:20px;overflow-x:auto;font-size:14px;color:var(--ink2)}}
.nav a{{flex:none;color:var(--ink2);text-decoration:none}}
.foot{{color:var(--mute);font-size:13.5px}}
@media(max-width:900px){{.kpis{{grid-template-columns:repeat(2,minmax(0,1fr))}}.grid2{{grid-template-columns:minmax(0,1fr)}}}}
@media(max-width:600px){{table{{font-size:13px}}.stack thead{{display:none}}.stack tr{{display:grid;padding-block:10px}}.stack td{{display:flow-root;border:0;padding:3px 0;padding-inline-start:8em}}.stack td::before{{content:attr(data-l);float:inline-start;width:7.5em;margin-inline-start:-8em;color:var(--mute)}}}}
</style>
<script>new MutationObserver(()=>{{for(const c of document.querySelectorAll('.stack td:not([data-l])')){{let i=0,p=c;while(p=p.previousElementSibling)i+=p.colSpan;c.dataset.l=c.closest('table').rows[0].cells[i]?.textContent||''}}}}).observe(document,{{childList:true,subtree:true}})</script>
</head><body>
<header class="band"><div class="wrap">
<nav class="nav"><a href="#vault">The vault</a><a href="#gamma">Risk aversion</a><a href="#levers">Two levers</a><a href="#gauge">Fear gauge</a><a href="#breaks">Where it breaks</a><a href="#method">Method</a><a href="{REPO}">Code and data</a></nav>
<div style="height:28px"></div>
<h1>Hyperliquid's market maker is one risk-averse vault split across two mirrored accounts</h1>
<p class="lede">Recovered from {v['HOURS']} hours of HLP's public quotes, fills and inventory on {v['COINS']} coins. The textbook regression says HLP is risk-neutral. Identified from shocks it did not choose, it prices inventory at {v['EV_BPS']} bps per $10k, and it manages the rest with size and market orders the model has no room for.</p>
<div class="kpis">{kpi_html}</div>
{fig("00_mirror_book.png", "Every open position of HLP Strategy A against Strategy B at the end of the sample, and the two sub-vaults' all-time PnL as reported by Hyperliquid.")}
</div></header>

<section id="vault" class="band"><div class="wrap grid2"><div class="col">
<h2>The market maker is the vault, not an account</h2>
<p>Strategy A and Strategy B quote the same coins at nearly the same prices and hold opposite books: they agree in sign on {v['SAME_SIGN']} of {v['HELD']} positions. Each carries about ${v['GROSS_A']}M gross; netted coin by coin the vault carries ${v['GROSS_V']}k. They trade the same flow (dollar-flow correlation {v['FLOW_CORR']} per coin-minute), take liquidity together ({v['SAME_SIDE']} same side within 250 ms), never trade with each other, and report mirrored all-time PnL of {v['PNL_A']} and {v['PNL_B']}.</p>
<p>Against either account's own inventory, HLP shows no unwind threshold and no size lean. Against the vault's, it shows both. Everything below is the vault.</p>
</div><div>{fig("09_exposure.png", "Gross open notional of each account and of the vault netted coin by coin.")}</div></div></section>

<section id="gamma" class="band"><div class="wrap">
<div class="col"><h2>The textbook regression says risk-neutral. Identified, it is not.</h2>
<p>In GLFT the quote centre sits at the reservation price S − ωq, so its slope on inventory reveals γ. Regressed on the inventory stock, the median coin gives {v['NAIVE_BPS']} bps per $10k: nothing, because inventory piles up where the quotes already lean. Five seconds after a passive fill, a shock HLP did not choose, the quote centre has moved against the new inventory in {v['EV_SHARE']} of coins, by {v['EV_BPS']} bps per $10k in the median coin (CI {v['EV_CI']}): γ ≈ {v['EV_GAMMA']} per $. Independently, the threshold at which HLP starts unwinding with market orders implies at least {v['TT_BPS']} bps per $10k, γ ≥ {v['TT_GAMMA']}.</p></div>
<div class="grid2">{fig("03_three_estimators.png", "Three estimators of the same reservation-price skew, 95% intervals.")}{fig("10_event_profile.png", "Median-coin quote response to inventory shocks by horizon. Later horizons also absorb subsequent fills and unwinds.")}</div>
</div></section>

<section id="levers" class="band"><div class="wrap">
<div class="col"><h2>Two levers the model does not have</h2>
<p>{v['TAKER_SHARE']} of HLP's fills are aggressive. While vault inventory in a coin is small, about half of its taker fills reduce it; as inventory grows the share ramps up to {v['P_TOP']}, with a fitted switch near ${v['THRESH']} per coin (likelihood ratio {v['LR']} against no switch). Each account reprices after its sibling's fills about as much as after its own. It also leans size: top-of-ladder imbalance runs from {v['SS_SHORT']} when the vault is short to {v['SS_LONG']} when long (t {v['SS_BETWEEN_T']} across coins). Jointly on own and vault inventory, the size lean loads on the vault (t {v['ATTR_V_T']}) and not the account (t {v['ATTR_O_T']}).</p></div>
{fig("02_two_levers.png", "Left: share of taker fills that reduce inventory against inventory in GLFT units, for the vault and each account. Right: top-level size imbalance against vault inventory.")}
</div></section>

<section id="gauge" class="band"><div class="wrap">
<div class="col"><h2>The fear gauge</h2><p>{v['FEAR_SENTENCE']}</p></div>
{fig("01_fear_gauge.png", "Size lean per 10-minute cross-section, unwind-implied reservation skew per 30 minutes, and market volatility.")}
</div></section>

<section id="breaks" class="band"><div class="wrap">
<h2>Where else the textbook breaks</h2>
<table class="stack"><thead><tr><th>Textbook assumption</th><th>What HLP does</th><th>Evidence</th></tr></thead><tbody>{table}</tbody></table>
<div class="grid2" style="margin-top:22px">{fig("04_overid_spread.png", "Quoted half spread against the GLFT restriction, one point per coin.")}{fig("05_markouts.png", "Markouts of the vault's passive and aggressive fills.")}</div>
</div></section>

<section id="method" class="band"><div class="wrap grid2"><div class="col">
<h2>Method</h2>
<p>GLFT closed form with order size (Guéant 2017), inverted for γ with brentq. σ² from 10-second mid changes; the fill-curve slope k from the reach of every public aggressive order; its level A from HLP's own passive fills at its own quote distances (Poisson MLE), so competition is priced in. Three skew estimators: within coin (HAC), fill events (median coin, bootstrap), between coins (the gauge's cross-sections). The unwind threshold is a profile-likelihood step in GLFT units z = |q|s/c, so z̄ = 1/√γ, a lower bound.</p>
<p>A simulated GLFT market maker with known γ, observed the way HLP is observed, is recovered within a few per cent whether it gets all of the flow at its price or 30%; calibrating A on market flow instead overstates γ by the inverse of the flow share.</p>
<p>Data: two GitHub Actions runners polling HLP's open orders every 1.4 s and streaming books, trades, oracle and mark prices, fills and inventory, {v['ERR429']} rate-limit errors, {v['RECONNECTS']} websocket reconnects. Code, data branches and every table: <a href="{REPO}">{REPO.replace('https://', '')}</a>.</p>
</div><div>{fig("07_simulation_check.png", "Recovered against true γ for a simulated GLFT market maker.")}</div></div></section>

<footer class="band"><div class="wrap foot">Independent research by Thatayotlhe Tsenang. Not affiliated with Hyperliquid. Not investment advice. Sample: {v['START']}–{v['END']} UTC, {v['DATE']}.</div></footer>
</body></html>"""
    open(os.path.join(OUT, "index.html"), "w").write(page)
    print("wrote", os.path.join(OUT, "index.html"), len(page), "bytes")


if __name__ == "__main__":
    main()
