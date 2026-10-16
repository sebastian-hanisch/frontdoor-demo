"""Schätzer: naiver Vergleich, Kontrolle für den Mediator, Frontdoor (zwei Regressionen, Produkt, Delta-Methode), Orakel mit dem verborgenen Confounder; dazu die Population
(exakte Grenzwerte aus der Kovarianz) und die binäre Frontdoor-Formel. Nur numpy, kein Streamlit."""

from __future__ import annotations

import numpy as np

import frd_world as W


def ols(y: np.ndarray, columns) -> tuple[np.ndarray, np.ndarray]:
    """Kleinste Quadrate mit Achsenabschnitt: (Koeffizienten, Kovarianzmatrix der Koeffizienten); `columns` sind die Regressoren ohne Achsenabschnitt."""
    d = np.column_stack([np.ones(len(y))] + list(columns))
    beta = np.linalg.solve(d.T @ d, d.T @ y)
    res = y - d @ beta
    return beta, np.linalg.inv(d.T @ d) * (res @ res) / (len(y) - d.shape[1])


def naive(x, y) -> tuple[float, float]:
    beta, cov = ols(y, [x])
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def control_mediator(x, m, y) -> tuple[float, float]:
    """Koeffizient von X in Y ~ X + M (die schlechte Kontrolle: blockiert den Weg über den Mediator)."""
    beta, cov = ols(y, [x, m])
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def oracle(x, u, y) -> tuple[float, float]:
    """Koeffizient von X in Y ~ X + U: das Backdoor-Orakel, das den verborgenen Confounder kennt (in der Praxis nicht möglich)."""
    beta, cov = ols(y, [x, u])
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def frontdoor(x, m, y) -> dict:
    """Frontdoor, linear: ĉ aus M ~ X, d̂ aus Y ~ M + X; Schätzung ĉ·d̂ mit Delta-Methoden-Varianz d̂² Var(ĉ) + ĉ² Var(d̂)."""
    (b1, c1), (b2, c2) = ols(m, [x]), ols(y, [m, x])
    c_hat, d_hat = float(b1[1]), float(b2[1])
    se_c, se_d = float(np.sqrt(c1[1, 1])), float(np.sqrt(c2[1, 1]))
    return {"estimate": c_hat * d_hat, "se": float(np.sqrt(d_hat ** 2 * se_c ** 2 + c_hat ** 2 * se_d ** 2)), "c": c_hat, "d": d_hat, "se_c": se_c, "se_d": se_d}


# ----------------------------------------------------------------------------- Population

def observed_covariance(w: W.World) -> np.ndarray:
    """Kovarianz der Knoten, wobei der Mediator mit Messrauschen beobachtet wird (Varianz plus noise²)."""
    cov = W.population(w)[0].copy()
    cov[W.M, W.M] += w.noise ** 2
    return cov


def coefficients(cov: np.ndarray, y: int, xs: list[int]) -> np.ndarray:
    return np.linalg.solve(cov[np.ix_(xs, xs)], cov[xs, y])


def population_estimates(w: W.World) -> dict:
    """Grenzwerte (n → ∞) der Schätzer und die Wahrheit, exakt aus der Kovarianz."""
    cov = observed_covariance(w)
    c_hat = coefficients(cov, W.M, [W.X])[0]
    d_hat = coefficients(cov, W.Y, [W.M, W.X])[0]
    return {"truth": W.truth(w), "naive": float(coefficients(cov, W.Y, [W.X])[0]), "control": float(coefficients(cov, W.Y, [W.X, W.M])[0]),
            "oracle": float(coefficients(cov, W.Y, [W.X, W.U])[0]), "frontdoor": float(c_hat * d_hat), "c": float(c_hat), "d": float(d_hat)}


# ----------------------------------------------------------------------------- binär

def frontdoor_binary(x: np.ndarray, m: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """Frontdoor-Formel mit Zellhäufigkeiten: P(y | do(x)) = Σ_m P(m | x) Σ_x′ P(y | m, x′) P(x′). Rückgabe (P(Y=1 | do(X=1)), P(Y=1 | do(X=0))); None, wenn eine
    Zelle (m, x′) leer ist."""
    def p(xv):
        out = 0.0
        for mv in (0, 1):
            p_m = float(np.mean(m[x == xv] == mv))
            inner = 0.0
            for xp in (0, 1):
                cell = (m == mv) & (x == xp)
                if not cell.any():
                    return None
                inner += float(y[cell].mean()) * float(np.mean(x == xp))
            out += p_m * inner
        return out

    p1, p0 = p(1), p(0)
    return None if p1 is None or p0 is None else (p1, p0)
