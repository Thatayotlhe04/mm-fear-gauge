"""Ground-truth check: simulate a GLFT market maker with a known gamma, observe it
the way we observe HLP (polled quotes, its own fills, public trades that reach some
depth), and verify the estimators recover gamma.

Scenario 'competition' gives the simulated MM only a share pi of the flow that
reaches its price: calibrating the fill curve on market trades (textbook) then
overstates A by 1/pi and gamma with it; the own-fill Poisson calibration does not.

Run:  python analysis/simulate.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import glft  # noqa: E402
from estimators import intensity_fit, sigma2_from_mid, skew_regression  # noqa: E402
from model import own_fill_intensity  # noqa: E402


def simulate(gamma, sigma=0.00425, A=0.08, k=80.0, D=10.0, S0=25.0, hours=4.0, dt=0.05,
             poll=1.4, qmax=400.0, share=1.0, seed=0):
    """A, k describe the flow that reaches depth delta; the MM gets a share of it."""
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 / dt)
    S = S0 + np.cumsum(rng.normal(0, sigma * np.sqrt(dt), n))
    q = 0.0
    polls, trades, fills = [], [], []
    next_poll = 0.0
    A_mm = A * share
    for i in range(n):
        t = i * dt
        db, da = glft.quotes(q, gamma, sigma ** 2, A_mm, k, D)   # the MM optimises against its own flow
        bid, ask = S[i] - db, S[i] + da
        if t >= next_poll:
            polls.append((t * 1000, S[i], bid, ask, q, db, da))
            next_poll += poll
        for side in (-1, 1):
            if rng.random() < A * dt:
                depth = rng.exponential(1 / k)
                trades.append((t, side, depth))
                if rng.random() > share:
                    continue
                if side == -1 and depth >= db and q + D <= qmax:
                    q += D
                    fills.append((t * 1000, "B", D))
                elif side == 1 and depth >= da and q - D >= -qmax:
                    q -= D
                    fills.append((t * 1000, "A", D))
    pc = pd.DataFrame(polls, columns=["t_recv", "S", "b1", "a1", "q", "db", "da"])
    fl = pd.DataFrame(fills, columns=["time", "side", "sz"]).assign(crossed=False)
    return pc, np.array(trades), S, fl


def recover(pc, trades, S, fl, D, hours, dt=0.05):
    T = hours * 3600
    sig2 = sigma2_from_mid(np.arange(len(S)) * dt * 1000, S, step_s=10)
    fit = intensity_fit(trades[:, 2], np.ones(len(trades)), T, q_lo=0.02, q_hi=0.98)
    A_m, k_m = fit["A"] / 2, fit["k"]
    own = own_fill_intensity(pc, fl, D, k_fixed=k_m)
    sk = skew_regression(((pc.b1 + pc.a1) / 2 - pc.S).values, pc.q.values)
    psi = float(((pc.a1 - pc.b1) / 2).mean())
    return {"sigma2": sig2, "A_mkt": A_m, "k_mkt": k_m, "A_own": own["A"], "k_own": own["k"],
            "b": sk["b"], "b_se": sk["b_se"], "psi": psi,
            "gamma_mkt": glft.gamma_from_skew(sk["b"], sig2, A_m, k_m, D),
            "A_own_k": own["A_given_k"],
            "gamma_own": glft.gamma_from_skew(sk["b"], sig2, own["A_given_k"], k_m, D),
            "psi_ratio": psi / (1 / k_m + 0.5 * D * sk["b"])}


def main():
    out = []
    for share in (1.0, 0.3):
        for gamma in [2e-4, 1e-3, 5e-3]:
            for seed in range(3):
                pc, trades, S, fl = simulate(gamma, share=share, seed=seed)
                r = recover(pc, trades, S, fl, D=10.0, hours=4.0)
                r.update({"gamma_true": gamma, "seed": seed, "share": share})
                out.append(r)
                print(f"share={share} gamma={gamma:.1e} seed={seed}  own-calib={r['gamma_own']:.2e}  "
                      f"market-calib={r['gamma_mkt']:.2e}  A_own|k={r['A_own_k']:.3f} "
                      f"(true {0.08*share:.3f})  k={r['k_mkt']:.1f} glm_k={r['k_own']:.0f}  psi/(1/k+Db/2)={r['psi_ratio']:.3f}", flush=True)
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(out).to_csv("results/simulation_check.csv", index=False)


if __name__ == "__main__":
    main()
