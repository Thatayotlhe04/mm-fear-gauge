## Method

### The model

Guéant, Lehalle and Fernandez-Tapia (2013), in the order-size form of Guéant (2017). The market maker has CARA utility with absolute risk aversion γ (1/$), the mid follows dS = σ dW, and a quote at distance δ from the mid is filled, one lot of size Δ at a time, with intensity Λ(δ) = A e^(−kδ). The closed-form quotes are

```
δ_bid(q) = (1/γΔ) ln(1 + γΔ/k) + (2q + Δ)/2 · ω
δ_ask(q) = (1/γΔ) ln(1 + γΔ/k) − (2q − Δ)/2 · ω
ω        = sqrt( γσ² / (2AkΔ) · (1 + γΔ/k)^(1 + k/γΔ) )
```

Two observable consequences, one unknown:

* **skew**: the quote centre is `S − ω·q`, so the slope of the centre on inventory is ω, which is monotone in γ and inverted with `scipy.optimize.brentq` (`analysis/glft.py`);
* **spread**: the half spread is `(1/γΔ) ln(1 + γΔ/k) + (Δ/2)·ω`. With γΔ/k ≪ 1, as it is here, this is `1/k + (Δ/2)·ω` whatever γ is. That is an over-identifying restriction: half spread, fill-curve slope and skew must line up.

### Calibration

* **σ²**: variance of 10-second mid changes from the 1 Hz asset-context stream.
* **k**: slope of the reach curve of aggressive orders. Public trades are collapsed into aggressive orders (same coin, time, taker), each order's deepest price is measured against the mid just before it, and `log Λ(δ) = log A − kδ` is fitted on the empirical curve (`analysis/estimators.py`). A power law is fitted on the same grid for comparison.
* **A**: the level of the fill curve **HLP itself faces**. For every poll interval and side, HLP's passive fills are Poisson with mean `dt · A · exp(−k δ)`, with δ HLP's own distance from the mid; with k fixed from the market curve, the MLE is `A = N / Σ dt·exp(−kδ)`. Calibrating A on market-wide flow instead assumes the market maker is a monopolist that receives every order reaching its price; the simulation shows what that does to γ.
* **Δ**: median size of HLP's top-of-ladder orders.

### Three ways to identify the skew

1. **Within coin.** Regress HLP's quote-centre offset on its inventory over time, coin by coin, controlling for the oracle–mid basis, top-of-book imbalance and 10 s / 60 s momentum, with Newey–West errors. Needs inventory to actually move.
2. **Fill events.** A passive fill changes HLP's inventory through someone else's decision. Compare the quote centre in the last snapshot before each passive fill with the first snapshot five seconds after, controlling for the change in basis and the mid return. This is the cleanest design and the headline estimator.
3. **Between coins.** At any moment HLP carries some 170 positions of very different size. One cross-sectional regression per 10-minute window of the window-mean quote centre on model-scaled inventory, `x_i = q_i · sqrt(σ_i² e / (2 A_i k_i Δ_i))`, gives θ_w = sqrt(γ_w) directly. That series is the **fear gauge**.

Coin-level slopes are pooled into one γ by inverse-variance weighting of `θ_i = b_i / s_i`, where s_i is the coin's GLFT scale; GLFT says θ is common to every coin.

### Who is the market maker?

HLP trades through two sub-vaults, Strategy A and Strategy B, which quote the same 174 coins at near-identical prices. Their positions are mirror images (see the first figure). Every estimator is therefore run three times: on A, on B, and on **the vault**, V = A + B, with inventory netted coin by coin, fills pooled and equity summed. A further test regresses A's quote centre jointly on A's own inventory and on the vault's net inventory: a vault-level risk manager loads on the second, an account-level one on the first. A cross-account event study asks whether A reprices when B is filled.

### Validation

`analysis/simulate.py` simulates a GLFT market maker with known γ for four hours, observes it the way HLP is observed (1.4 s polls, its own fills, public trades), and runs the same estimators. With k from the market reach curve and A from the market maker's own fills, γ is recovered within a few per cent across three orders of magnitude, whether the simulated market maker receives all of the flow at its price or only 30 % of it. Calibrating A on market flow instead overstates γ by the inverse of the flow share. The spread restriction holds to within ±5 % in simulation, so deviations in the real data are not estimator artefacts.
