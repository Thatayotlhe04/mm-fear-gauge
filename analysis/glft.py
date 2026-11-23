"""Guéant-Lehalle-Fernandez-Tapia closed-form market-making approximations and
their inversion for the risk-aversion parameter gamma.

Model (Guéant 2017, Model A, order size Delta):
    mid            dS = sigma dW
    fill intensity Lambda(delta) = A exp(-k delta)   (fills of one lot, size Delta, per second)
    (derived from the unit-lot GLFT 2013 formulas by rescaling q -> q/D, sigma -> D sigma, k -> k/D)
    utility        CARA with absolute risk aversion gamma (1/$)

Closed-form quotes (distance from mid, price units):
    delta_b(q) = (1/(g D)) ln(1 + g D / k) + (2q + D)/2 * omega
    delta_a(q) = (1/(g D)) ln(1 + g D / k) - (2q - D)/2 * omega
    omega      = sqrt( g sigma^2 / (2 A k D) * (1 + g D / k)^(1 + k / (g D)) )

Implications used for estimation:
    quote centre  c(q)  = S - omega * q            (skew slope  = omega)
    half spread   psi   = (1/(g D)) ln(1 + g D / k) + (D/2) * omega
Two observables, one unknown gamma: the model is over-identified.
"""
import numpy as np
from scipy.optimize import brentq


def _g(x):
    """(1 + x)^(1 + 1/x), continuous at 0 (limit e)."""
    x = np.asarray(x, dtype=float)
    out = np.where(x < 1e-8, np.e * (1 - x / 2), np.exp((1 + 1 / np.maximum(x, 1e-300)) * np.log1p(x)))
    return out


def _markup(gamma, k, D):
    """(1/(gD)) ln(1 + gD/k): the monopoly mark-up term, -> 1/k as gamma -> 0."""
    x = gamma * D
    return np.where(x / k < 1e-10, 1 / k, np.log1p(x / k) / np.maximum(x, 1e-300))


def omega(gamma, sigma2, A, k, D):
    return np.sqrt(gamma * sigma2 / (2 * A * k * D) * _g(gamma * D / k))


def half_spread(gamma, sigma2, A, k, D):
    return _markup(gamma, k, D) + 0.5 * D * omega(gamma, sigma2, A, k, D)


def quotes(q, gamma, sigma2, A, k, D):
    w = omega(gamma, sigma2, A, k, D)
    m = _markup(gamma, k, D)
    return m + (2 * q + D) / 2 * w, m - (2 * q - D) / 2 * w   # delta_b, delta_a


def gamma_from_skew(b, sigma2, A, k, D, lo=1e-14, hi=1e6):
    """Invert the skew slope b (price per unit of inventory) for gamma."""
    if not np.isfinite(b) or b <= 0:
        return np.nan
    f = lambda lg: np.log(omega(np.exp(lg), sigma2, A, k, D)) - np.log(b)
    a, z = np.log(lo), np.log(hi)
    if f(a) > 0 or f(z) < 0:
        return np.nan
    return float(np.exp(brentq(f, a, z, xtol=1e-10)))


def gamma_from_spread(psi, sigma2, A, k, D, lo=1e-14, hi=1e6):
    """Invert the observed half spread for gamma. Returns nan when psi < 1/k:
    no positive risk aversion can produce a spread tighter than the monopoly mark-up."""
    if not np.isfinite(psi) or psi <= 0:
        return np.nan
    f = lambda lg: half_spread(np.exp(lg), sigma2, A, k, D) - psi
    grid = np.linspace(np.log(lo), np.log(hi), 400)
    vals = np.array([f(x) for x in grid])
    idx = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    return float(np.exp(brentq(f, grid[i], grid[i + 1], xtol=1e-10)))


def gamma_skew_smallx(b, sigma2, A, k, D):
    """Closed form when gamma*D/k -> 0:  gamma = 2 A k D b^2 / (e sigma^2)."""
    return 2 * A * k * D * b ** 2 / (np.e * sigma2)
