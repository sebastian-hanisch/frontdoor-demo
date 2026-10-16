"""Orakel: ein unabhängiger Rechenweg für die Frontdoor-Wirkung — die Formel über Kovarianzen wird gegen die Pfadsummen und gegen die gewichtete Formel mit Zellmitteln geprüft."""

import numpy as np
import pytest

import frd_estimators as E
import frd_world as W
from kausal_generator import SplitMix64


@pytest.mark.parametrize("kw", [dict(), dict(a=2.0, c=0.5, d=1.5), dict(a=0.0, c=1.2, d=0.3), dict(a=1.5, c=0.1, d=1.0)])
def test_population_frontdoor_equals_the_path_product_without_violations(kw):
    """Bei gültiger Frontdoor ist ĉ·d̂ genau c·d: der Pfad X → M → Y."""
    w = W.World(**kw)
    p = E.population_estimates(w)
    assert p["frontdoor"] == pytest.approx(w.c * w.d) and p["frontdoor"] == pytest.approx(p["truth"])


@pytest.mark.parametrize("g", [0.1, 0.3, 0.5, 0.8])
def test_second_confounder_bias_is_g_squared_over_one_plus_g_squared(g):
    p = E.population_estimates(W.World(g=g))
    assert p["frontdoor"] - p["truth"] == pytest.approx(g * g / (1 + g * g))


@pytest.mark.parametrize("noise", [0.2, 0.7, 1.0, 1.5])
def test_noise_attenuates_the_second_step_by_one_over_one_plus_sigma_squared(noise):
    p = E.population_estimates(W.World(noise=noise))
    assert p["d"] == pytest.approx(1.0 / (1 + noise ** 2)) and p["c"] == pytest.approx(1.0)


def test_linear_frontdoor_equals_the_adjustment_formula_on_large_data():
    """Zweiter Rechenweg im Linearen: Σ_m P(m|x) Σ_x′ P(y|m,x′) P(x′) wird zur Differenz der Erwartungen; hier über die Erwartung bei festem m: ĉ · d̂ = ĉ · E[Y | m+1, x] − E[Y | m, x]."""
    v, m = W.simulate(W.World(), 100000, SplitMix64(21))
    x, y = v[0, :, W.X], v[0, :, W.Y]
    c, d = E.frontdoor(x, m[0], y)["c"], E.frontdoor(x, m[0], y)["d"]
    design = np.column_stack([np.ones_like(x), m[0], x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    shifted = design.copy()
    shifted[:, 1] += 1.0
    step = np.mean(shifted @ beta - design @ beta)                                              # eine Einheit mehr Mediator, gemittelt über die beobachteten x
    assert step == pytest.approx(d) and c * step == pytest.approx(1.0, abs=0.02)


def test_binary_formula_against_exact_probabilities():
    """Exakter Rechenweg über die vier Zustände: aus den wahren Wahrscheinlichkeiten die beobachtete Verteilung bilden und die Frontdoor-Formel darauf anwenden."""
    pu, px, pm = 0.5, (0.2, 0.8), (0.1, 0.9)
    y0, ym, yu = 0.1, 0.5, 0.3
    joint = {}
    for u in (0, 1):
        for x in (0, 1):
            for m in (0, 1):
                joint[u, x, m] = (pu if u else 1 - pu) * (px[u] if x else 1 - px[u]) * (pm[x] if m else 1 - pm[x])

    def p_x(x):
        return sum(p for (u, xx, m), p in joint.items() if xx == x)

    def p_m_given_x(m, x):
        return sum(p for (u, xx, mm), p in joint.items() if xx == x and mm == m) / p_x(x)

    def e_y_given(m, x):                                                                        # E[Y | m, x] über die Verteilung von U bei (m, x)
        cell = [(u, p) for (u, xx, mm), p in joint.items() if xx == x and mm == m]
        return sum(p * (y0 + ym * m + yu * u) for u, p in cell) / sum(p for _, p in cell)

    def do(x):
        return sum(p_m_given_x(m, x) * sum(e_y_given(m, xs) * p_x(xs) for xs in (0, 1)) for m in (0, 1))

    t1, t0 = W.binary_truth()
    assert do(1) == pytest.approx(t1) and do(0) == pytest.approx(t0)
    o1, o0 = W.binary_observed()
    assert sum(p_m_given_x(m, 1) * e_y_given(m, 1) for m in (0, 1)) == pytest.approx(o1)
