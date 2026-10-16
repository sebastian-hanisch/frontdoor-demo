"""Vehikel: ein Standort wählt die neue Regel B (X) abhängig von der verborgenen Standortqualität U; die Regel wirkt über den Bestand (M, beobachtet) auf den Servicegrad (Y).
Lineares Modell mit Einheitsrauschen; die Wahrheit ist die Gesamtwirkung. Nur numpy und der Generator der Linie, kein Streamlit.

Knoten (topologisch): 0 U (verborgen), 1 U2 (zweiter verborgener Confounder), 2 X, 3 M, 4 Y.
Kanten: U -> X und U -> Y (je a), X -> M (c), M -> Y (d); Verletzungen: X -> Y (e), U -> M (f), U2 -> M und U2 -> Y (je g).
Der Mediator wird mit Rauschen `noise` gemessen: M_beobachtet = M + noise · ε."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import frd_constants as C
from kausal_generator import SplitMix64

U, U2, X, M, Y = 0, 1, 2, 3, 4
NAMES = ("Standortqualität (verborgen)", "zweite Ursache (verborgen)", "Regel B", "Bestand", "Servicegrad")


@dataclass(frozen=True)
class World:
    a: float = C.DEFAULT_A
    c: float = C.DEFAULT_C
    d: float = C.DEFAULT_D
    e: float = 0.0
    f: float = 0.0
    g: float = 0.0
    noise: float = 0.0


def coefficient_matrix(w: World) -> np.ndarray:
    """B[j, i] = Koeffizient der Kante i -> j."""
    b = np.zeros((5, 5))
    b[X, U], b[Y, U], b[M, X], b[Y, M], b[Y, X], b[M, U], b[M, U2], b[Y, U2] = w.a, w.a, w.c, w.d, w.e, w.f, w.g, w.g
    return b


def parents(w: World) -> tuple:
    b = coefficient_matrix(w)
    return tuple(tuple(i for i in range(5) if b[j, i] != 0.0) for j in range(5))


def population(w: World):
    """Kovarianz Σ der fünf Knoten und Gesamtwirkungen T = (I − B)⁻¹ − I bei Einheitsrauschen; der gemessene Mediator hat Var(M) + noise², sonst dieselben Kovarianzen."""
    inv = np.linalg.inv(np.eye(5) - coefficient_matrix(w))
    return inv @ inv.T, inv - np.eye(5)


def truth(w: World) -> float:
    """Gesamtwirkung von X auf Y: Summe der Pfadprodukte (c·d + e)."""
    return float(population(w)[1][Y, X])


def simulate(w: World, n: int, rng: SplitMix64, reps: int = 1):
    """(v, m_obs) mit Formen (reps, n, 5) und (reps, n): alle Knoten, der Mediator mit Messrauschen."""
    z = rng.normal(reps * n * 5).reshape(reps, n, 5)
    v = z @ np.linalg.inv(np.eye(5) - coefficient_matrix(w)).T
    noise = rng.normal(reps * n).reshape(reps, n)
    return v, v[..., M] + w.noise * noise


# ----------------------------------------------------------------------------- binäres Beispiel

def binary_probabilities():
    """Exakte Verteilung des binären Beispiels: (P(U=1), P(X=1 | U), P(M=1 | X), P(Y=1 | M, U) als Funktion)."""
    return C.BIN_PU, C.BIN_PX, C.BIN_PM, lambda m, u: C.BIN_Y0 + C.BIN_YM * m + C.BIN_YU * u


def binary_truth() -> tuple[float, float]:
    """(P(Y=1 | do(X=1)), P(Y=1 | do(X=0))) nach der Frontdoor-Formel, aus den bekannten Wahrscheinlichkeiten."""
    pu, _, pm, py = binary_probabilities()
    out = []
    for x in (1, 0):
        total = 0.0
        for m in (0, 1):
            p_m = pm[x] if m == 1 else 1.0 - pm[x]
            total += p_m * sum((pu if u == 1 else 1.0 - pu) * py(m, u) for u in (0, 1))
        out.append(total)
    return out[0], out[1]


def binary_observed() -> tuple[float, float]:
    """(P(Y=1 | X=1), P(Y=1 | X=0)) im Beobachtungsmodell (mit dem verborgenen Confounder): was der naive Vergleich sieht."""
    pu, px, pm, py = binary_probabilities()
    out = []
    for x in (1, 0):
        joint_u = [(pu if u == 1 else 1.0 - pu) * (px[u] if x == 1 else 1.0 - px[u]) for u in (0, 1)]
        total = 0.0
        for u in (0, 1):
            p_m1 = pm[x]
            total += joint_u[u] * (p_m1 * py(1, u) + (1.0 - p_m1) * py(0, u))
        out.append(total / sum(joint_u))
    return out[0], out[1]


def simulate_binary(n: int, rng: SplitMix64, reps: int = 1):
    """(u, x, m, y) als 0/1-Felder mit Form (reps, n)."""
    pu, px, pm, py = binary_probabilities()
    size = reps * n
    u = (rng.uniform(size).reshape(reps, n) < pu).astype(int)
    x = (rng.uniform(size).reshape(reps, n) < np.where(u == 1, px[1], px[0])).astype(int)
    m = (rng.uniform(size).reshape(reps, n) < np.where(x == 1, pm[1], pm[0])).astype(int)
    y = (rng.uniform(size).reshape(reps, n) < py(m, u)).astype(int)
    return u, x, m, y
