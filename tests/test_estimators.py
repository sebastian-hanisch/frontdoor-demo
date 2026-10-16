"""Schätzer: Regression, Frontdoor, Population und binäre Formel; an den Handrechnungen aus conftest.py."""

import numpy as np
import pytest

import frd_estimators as E
import frd_world as W
from kausal_generator import SplitMix64


def test_ols_against_numpy_lstsq():
    rng = SplitMix64(1)
    x1, x2 = rng.normal(80), rng.normal(80)
    y = 1.0 + 2.0 * x1 - x2 + 0.5 * rng.normal(80)
    beta, cov = E.ols(y, [x1, x2])
    d = np.column_stack([np.ones(80), x1, x2])
    ref, res, *_ = np.linalg.lstsq(d, y, rcond=None)
    assert beta == pytest.approx(ref) and cov == pytest.approx(np.linalg.inv(d.T @ d) * res[0] / (80 - 3))


def test_naive_control_and_oracle_on_exact_data():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    m = np.array([1.0, 0.0, 1.0, 0.0])
    y = 2.0 * x + m                                                                 # exakt linear: kein Rest, Koeffizienten exakt
    assert E.naive(x, y)[0] == pytest.approx(np.polyfit(x, y, 1)[0])
    assert E.control_mediator(x, m, y)[0] == pytest.approx(2.0) and E.oracle(x, m, y)[0] == pytest.approx(2.0)


def test_frontdoor_on_exact_data():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    m = 2.0 * x + 0.3 * np.array([1.0, -2.0, 1.0, 1.0, -2.0, 1.0])                 # Rest ist orthogonal zu 1 und x: die Steigung ist genau 2
    y = 3.0 * m + 1.0 * x                                                           # d = 3 bei festem x, c = 2: Produkt 6 (die direkte Wirkung 1 fehlt)
    out = E.frontdoor(x, m, y)
    assert out["c"] == pytest.approx(2.0, abs=1e-9) and out["d"] == pytest.approx(3.0, abs=1e-9) and out["estimate"] == pytest.approx(6.0, abs=1e-8)
    assert out["se_d"] == pytest.approx(0.0, abs=1e-7) and out["se"] == pytest.approx(out["d"] * out["se_c"], rel=1e-6)


def test_population_estimates_by_hand(unit_world):
    p = E.population_estimates(unit_world)
    assert p["truth"] == pytest.approx(1.0) and p["naive"] == pytest.approx(1.5) and p["control"] == pytest.approx(0.5)
    assert p["oracle"] == pytest.approx(1.0) and p["frontdoor"] == pytest.approx(1.0) and p["c"] == pytest.approx(1.0) and p["d"] == pytest.approx(1.0)


def test_population_under_each_violation():
    assert E.population_estimates(W.World(e=0.3))["frontdoor"] == pytest.approx(1.0) and E.population_estimates(W.World(e=0.3))["truth"] == pytest.approx(1.3)
    assert E.population_estimates(W.World(f=0.3))["frontdoor"] == pytest.approx(1.315, abs=1e-3)
    assert E.population_estimates(W.World(g=0.5))["frontdoor"] == pytest.approx(1.0 + 0.25 / 1.25)               # Verzerrung g² / (1 + g²)
    assert E.population_estimates(W.World(noise=0.5))["frontdoor"] == pytest.approx(1.0 / (1.0 + 0.25))             # Stauchung 1 / (1 + σ²)
    assert E.population_estimates(W.World(noise=1.0))["frontdoor"] == pytest.approx(0.5)
    assert E.population_estimates(W.World(a=2.0))["frontdoor"] == pytest.approx(1.0)
    p = E.population_estimates(W.World(c=0.5))
    assert p["control"] == pytest.approx(p["truth"])                                                              # die Koinzidenz bei c = 0,5


def test_observed_covariance_adds_only_the_noise_variance():
    w = W.World(noise=0.8)
    cov, obs = W.population(w)[0], E.observed_covariance(w)
    diff = obs - cov
    assert diff[W.M, W.M] == pytest.approx(0.64) and np.abs(diff).sum() == pytest.approx(0.64)


def test_coefficients_solve_the_normal_equations():
    cov = np.array([[2.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 7.0]])
    assert E.coefficients(cov, 2, [0, 1]) == pytest.approx([0.5, 1.0])


def test_frontdoor_converges_to_the_truth_and_the_delta_interval_covers():
    w = W.World()
    v, m = W.simulate(w, 500, SplitMix64(9), reps=300)
    hits, ests, ses = 0, [], []
    for r in range(300):
        out = E.frontdoor(v[r, :, W.X], m[r], v[r, :, W.Y])
        ests.append(out["estimate"])
        ses.append(out["se"])
        hits += abs(out["estimate"] - 1.0) <= 1.96 * out["se"]
    assert np.mean(ests) == pytest.approx(1.0, abs=0.02) and 0.91 <= hits / 300 <= 0.99
    assert np.mean(ses) == pytest.approx(np.std(ests), rel=0.12)


def test_naive_and_control_are_biased_in_a_large_sample(unit_world):
    v, m = W.simulate(unit_world, 200000, SplitMix64(10))
    x, y = v[0, :, W.X], v[0, :, W.Y]
    assert E.naive(x, y)[0] == pytest.approx(1.5, abs=0.02) and E.control_mediator(x, m[0], y)[0] == pytest.approx(0.5, abs=0.02)
    assert E.oracle(x, v[0, :, W.U], y)[0] == pytest.approx(1.0, abs=0.02) and E.frontdoor(x, m[0], y)["estimate"] == pytest.approx(1.0, abs=0.02)


def test_binary_frontdoor_by_hand(binary_sample):
    p1, p0 = E.frontdoor_binary(*binary_sample)
    assert p1 == pytest.approx(11 / 16) and p0 == pytest.approx(13 / 24)


def test_delta_standard_error_is_the_formula_of_the_two_steps():
    rng = SplitMix64(14)
    x = rng.normal(60)
    m = 0.8 * x + rng.normal(60)
    y = 1.2 * m + 0.3 * x + rng.normal(60)
    out = E.frontdoor(x, m, y)
    assert out["se"] ** 2 == pytest.approx(out["d"] ** 2 * out["se_c"] ** 2 + out["c"] ** 2 * out["se_d"] ** 2, rel=1e-9) and out["se_c"] > 0 and out["se_d"] > 0
    sxx = np.sum((x - x.mean()) ** 2)
    resid = m - m.mean() - (np.cov(x, m)[0, 1] / np.cov(x)) * (x - x.mean())
    assert out["se_c"] == pytest.approx(np.sqrt(np.sum(resid ** 2) / 58 / sxx), rel=1e-9)


def test_binary_frontdoor_weights_by_the_unbalanced_distribution_of_x():
    """Acht Einheiten mit P(x′ = 0) = 3/8: Zellmittel (m=0, x′=0) 1, (m=1, x′=0) 1/2, (m=0, x′=1) 1/2, (m=1, x′=1) 2/3;
    innen m = 0: 3/8 + 5/16 = 11/16, m = 1: 3/16 + 5/12 = 29/48; P(M=1 | X=1) = 3/5, P(M=1 | X=0) = 2/3: p1 = 51/80, p0 = 91/144."""
    x = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    m = np.array([0, 1, 1, 0, 0, 1, 1, 1])
    y = np.array([1, 0, 1, 0, 1, 1, 1, 0])
    p1, p0 = E.frontdoor_binary(x, m, y)
    assert p1 == pytest.approx(51 / 80) and p0 == pytest.approx(91 / 144)


def test_binary_frontdoor_returns_none_for_an_empty_cell(binary_sample):
    x, m, y = (np.array(a) for a in binary_sample)
    keep = ~((m == 1) & (x == 0))                                                    # Zelle (m = 1, x′ = 0) leer
    assert E.frontdoor_binary(x[keep], m[keep], y[keep]) is None
    keep = ~((m == 0) & (x == 1))
    assert E.frontdoor_binary(x[keep], m[keep], y[keep]) is None


def test_binary_frontdoor_recovers_the_truth_in_a_large_sample():
    u, x, m, y = (a[0] for a in W.simulate_binary(300000, SplitMix64(11)))
    p1, p0 = E.frontdoor_binary(x, m, y)
    t1, t0 = W.binary_truth()
    assert p1 == pytest.approx(t1, abs=0.006) and p0 == pytest.approx(t0, abs=0.006)
