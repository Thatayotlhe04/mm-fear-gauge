# Market Maker Fear Gauge

Recovering the risk aversion of Hyperliquid's protocol market maker, HLP, from its public quotes, fills and inventory, and finding where the textbook market-making model (Avellaneda–Stoikov, Guéant–Lehalle–Fernandez-Tapia) breaks.

![Two accounts, one book](figures/00_mirror_book.png)

**Data.** {HOURS} hours of live Hyperliquid data ({START}–{END} UTC, {DATE}), collected by two GitHub Actions runners from the public API: {QUOTES} snapshots of HLP's quote ladders on {COINS} coins, {FILLS} HLP fills, {TRADES} public trades, the 1 Hz oracle/mark/mid stream and 10-second inventory snapshots, plus a backfill of 1-minute candles, funding and vault history.

**Stack.** Python, Hyperliquid API, SciPy (root finding, profile likelihood), statsmodels (HAC / clustered OLS, Poisson GLM, logit), pandas, pyarrow, matplotlib. Collection runs on GitHub Actions; analysis is reproducible from the data branches with one command.

---

## Findings

**1. The market maker is the vault, not an account.** HLP trades through two sub-vaults, *Strategy A* and *Strategy B*, that quote the same {COINS} coins at nearly the same prices. Their books are mirror images: across the {HELD} coins both hold, the correlation of their signed positions is **{CORR_POS}** and they agree in sign on {SAME_SIGN} of them. Each carries about **${GROSS_A}M** of gross exposure; netted coin by coin, the vault carries **${GROSS_V}k**. They trade the same flow (dollar-flow correlation {FLOW_CORR} per coin-minute), execute together (when both take liquidity on a coin within 250 ms, {SAME_SIDE} are on the same side), never trade with each other ({AB_TRADES} prints), and their reported all-time PnLs mirror each other: **{PNL_A}** and **{PNL_B}**. Each account reprices after its sibling's passive fills about as much as after its own ({CROSS}). Measured against either account's own inventory, HLP shows no unwind threshold and no size lean (findings 4 and 5); measured against the vault's, it shows both. Every estimate below is for the vault.

**2. The textbook regression says HLP is risk-neutral. That regression is biased.** In GLFT the quote centre sits at the reservation price `S − ω·q`, so the slope of the quote centre on inventory reveals ω and hence γ. Run coin by coin, that regression gives a median slope of **{NAIVE_BPS} bps per $10k** of vault inventory (95% CI {NAIVE_CI}): statistically indistinguishable from zero, and a small fraction of what the identified estimators below find. The problem is that inventory is not exogenous. Where HLP's quotes lean, it gets filled and inventory piles up, which pushes the levels relationship towards zero.

**3. Identified from inventory shocks HLP did not choose, it leans.** A passive fill changes HLP's inventory through someone else's decision. Five seconds after one, the vault's quote centre has moved against the new inventory in **{EV_SHARE}** of coins; the median coin's response is a reservation-price skew of **{EV_BPS} bps per $10k** (95% CI {EV_CI}), which inverts to **γ ≈ {EV_GAMMA} per $** (CI {EV_GAMMA_CI}). In plain terms: HLP prices inventory as if holding $10k of a typical coin for an hour cost it **{EV_HOLD}**. The response keeps building for about fifteen minutes and then holds (figure 10); GLFT re-centres instantly.

**4. A second, independent identification from market orders.** The textbook market maker never crosses the spread. HLP does: {TAKER_SHARE} of its fills are aggressive. In GLFT with market orders, the market maker crosses when its reservation-price deviation `ω·q` exceeds the cost of crossing. HLP behaves like that, and only at vault level: while vault inventory in a coin is small, about half of its taker fills reduce it, as you would expect if they were driven by something else; as inventory grows the share ramps up to {P_TOP}. A step fitted by profile likelihood puts the switch at about **${THRESH}** of vault inventory per coin, with the share going from **{P_BELOW} to {P_ABOVE}** (likelihood ratio {LR} against no switch). Using its measured crossing cost of {COST} bps, the threshold implies a reservation skew of at least **{TT_BPS} bps per $10k** and **γ ≥ {TT_GAMMA} per $** (CI {TT_GAMMA_CI}). It is a lower bound, because waiting for a passive fill has option value; it sits below the fill-shock estimate, as it should, and it is a loose one. Fitting the same threshold on either account's own inventory finds nothing (LR {LR_A} and {LR_B}).

![Three estimators](figures/03_three_estimators.png)

**5. HLP also leans size, which GLFT has no room for.** GLFT keeps quote size fixed and moves prices. HLP moves size too: its top-of-ladder size imbalance swings from {SS_SHORT} when the vault is short to {SS_LONG} when it is long, {SS_WITHIN} per $1k of vault inventory within coin (t {SS_WITHIN_T}) and {SS_BETWEEN} across coins (t {SS_BETWEEN_T}). Regressing A's size imbalance jointly on A's own inventory and the vault's: vault **t = {ATTR_V_T}**, own **t = {ATTR_O_T}**. The risk manager sits above the accounts.

![Two levers](figures/02_two_levers.png)

**6. The fear gauge.** Two series track HLP's appetite for inventory through the session: how hard it leans touch size against vault inventory (10-minute cross-sections of ~170 coins) and the reservation skew implied by its unwind threshold (30-minute windows). {FEAR_SENTENCE}

![Fear gauge](figures/01_fear_gauge.png)

**7. Where else the textbook breaks.**

| Textbook assumption | What HLP does | Evidence |
|---|---|---|
| One agent, one book | Two mirrored sub-vaults; risk is managed on the net | position correlation {CORR_POS}; size lean and unwinds load only on vault inventory |
| Inventory managed by price skew | Price skew is only part of it; size lean and market-order unwinds carry the rest | figures 02, 03 |
| Only limit orders | {TAKER_SHARE} of fills are taker; {RED_SHARE} of taker volume reduces vault inventory | `results/RESULTS.md` |
| Quotes re-centre instantly (Markov in q) | Price response builds for minutes after a shock | figure 10 |
| Fills carry no information | Passive fills start {EDGE} bps in the money and are worth {MO60} bps a minute later | figure 05 |
| Spread = 1/k + (Δ/2)·skew | Quoted half spread is {PSI_RATIO}x the GLFT value in the median coin; {UNRAT} of coins quote tighter than any γ allows | figure 04 |
| Monopolist: gets all flow reaching its price | Fills {FLOW_SHARE} of it (median coin) | figure 06 |
| Spread insensitive to vol when skew is small | Elasticity of the half spread to local vol {VOL_EL} | `results/RESULTS.md` |
| One quote per side | Eight levels, geometric spacing ({LADDER_D} bps from mid), ~${LADDER_TOP} at the touch, ${LADDER_DEEP} behind | figure 08 |
| Exponential fill curve | Holds up: exponential beats a power law in {EXP_SHARE} of coins (median R² {R2_EXP} vs {R2_POW}) | figure 06 |

---

## What HLP is

HLP is Hyperliquid's protocol vault: about **${HLP_EQ}M** of depositor capital, of which ${TRADE_EQ}M sits in the two trading sub-vaults studied here. They rest roughly 2,900 orders each across {COINS} perpetuals, cancel and replace about 1,250 orders a second, live a median of 1.7 s per order, and post with `Alo` (post-only) for quotes and IOC for aggressive orders. Liquidator sub-vaults are separate and were excluded.

## How the data was collected

Hyperliquid's API is public, but the order-update stream is only pushed to the account that owns it, so HLP's quotes have to be reconstructed by polling its open orders. That costs 20 of the 1,200 weight a minute each IP gets, so each strategy has its own runner, polling every 1.4 s (`collector/collect.py`, `.github/workflows/collect.yml`). Each runner also streams order books, trades, asset contexts (oracle, mark, mid, funding, open interest), the strategy's fills and `allMids` over websockets, snapshots positions every 10 s and historical orders every 2 minutes, and pushes zstd-compressed parquet to its own `data-*` branch every 10 minutes. A third job backfills candles, funding, recent fills and vault history (`collector/backfill.py`). Over the run the two collectors logged {ERR429} rate-limit errors and {RECONNECTS} websocket reconnects, each recovered automatically within seconds.

{METHOD}

## Reproduce

```bash
pip install -r requirements.txt
bash analysis/sync_data.sh          # pulls every data-* branch into ./data
python analysis/simulate.py         # ground-truth check of the estimators
python analysis/run_all.py          # panel -> estimates -> tests -> results/
python analysis/figures.py          # figures/
python analysis/report.py           # results/RESULTS.md
```

The `analyze` workflow runs the same five steps on GitHub Actions with pinned versions. To collect a new sample, run the `collect` workflow (default 340 minutes).

## Layout

```
collector/collect.py     live collector: quote ladders, books, trades, fills, contexts, inventory
collector/backfill.py    candles, funding, recent fills, vault history
analysis/glft.py         GLFT closed form, inversion for gamma (brentq)
analysis/estimators.py   volatility, fill-curve fit, HAC skew regression
analysis/panel.py        loads the data; builds the per-(coin, snapshot) panel; the vault as one agent
analysis/model.py        calibration, three skew estimators, fill-event profile, size lean, unwind threshold, gauge
analysis/breaks.py       tests of each textbook assumption, mirror statistics, inventory attribution
analysis/simulate.py     simulated GLFT market maker with known gamma
analysis/run_all.py      the pipeline
results/                 summary.json, RESULTS.md, per-coin parameters, every table behind a figure
figures/                 the figures above
```

## Limitations

* The analysis window is fixed at {START}–{END} UTC in `analysis/config.py`; the collector ran past it. {HOURS} hours on one day is a sample of HLP's behaviour, not its policy. The fear gauge has {N_WIN} windows; it shows what the gauge measures rather than a long history.
* Quotes are snapshots every 1.4 s of a book that changes every ~1.7 s, so the fastest repricing is not observed.
* The fill-event estimate uses the median coin; a pooled regression is dominated by a handful of coins with very large scaled inventory changes and is reported but not headlined. Responses at long horizons mix in later fills and unwinds.
* The unwind threshold is a lower bound on γ under GLFT with market orders; mapping it exactly would need the full quasi-variational inequality solved per coin.
* Positions of A and B are mirrored, but I have no way to see why; concealing the vault's net position is one hypothesis, not a finding.

---

*Independent research. Not affiliated with Hyperliquid. Nothing here is investment advice.*
