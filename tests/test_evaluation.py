"""Auswertung: Einstellungen, eine Stichprobe, Zusammenfassung, die vier Experimente (klein gefahren) und die Zufallsdiagramme."""

import numpy as np
import pytest

import bdr_backdoor as B
import frd_constants as C
import frd_criterion as K
import frd_evaluation as ev
import frd_world as W
from kausal_generator import SplitMix64


def test_number_formats():
    assert ev.de(-1.234) == "−1,23" and ev.de(0.5, 1) == "0,5" and ev.de(3.0, 0) == "3"
    assert ev.signed(0.3) == "+0,30" and ev.signed(-0.3) == "−0,30" and ev.signed(1e-12) == "+0,00" and ev.signed(-1e-12) == "+0,00"


def test_settings_build_the_world_and_default_to_the_base_case():
    s = ev.Settings(a=0.5, c=0.7, d=0.9, e=0.1, f=0.2, g=0.3, noise=0.4)
    assert s.world == W.World(0.5, 0.7, 0.9, 0.1, 0.2, 0.3, 0.4)
    d = ev.Settings()
    assert (d.a, d.c, d.d, d.e, d.f, d.g, d.noise, d.n, d.seed) == (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, C.DEFAULT_N, C.DEFAULT_SEED)


def test_analyse_rows_and_limits():
    a = ev.analyse(ev.Settings())
    assert [r.name for r in a.rows] == list(ev.ROW_NAMES) and a.truth == pytest.approx(1.0)
    by = {r.name: r for r in a.rows}
    assert by["Naiver Vergleich"].limit == pytest.approx(1.5) and by["Kontrolle für den Mediator"].limit == pytest.approx(0.5)
    assert by["Frontdoor"].limit == pytest.approx(1.0) and by["Orakel mit U (nicht beobachtbar)"].limit == pytest.approx(1.0)
    for r in a.rows:
        assert r.lo == pytest.approx(r.estimate - C.Z_CRIT * r.se) and r.hi == pytest.approx(r.estimate + C.Z_CRIT * r.se) and r.error == pytest.approx(r.estimate - a.truth)
        assert r.covers == (r.lo <= a.truth <= r.hi)
    assert a.steps["c_limit"] == pytest.approx(1.0) and a.steps["d_limit"] == pytest.approx(1.0)
    assert by["Frontdoor"].estimate == pytest.approx(a.steps["c"] * a.steps["d"])


def test_analyse_is_deterministic_and_seed_dependent():
    assert ev.analyse(ev.Settings()).rows[2].estimate == ev.analyse(ev.Settings()).rows[2].estimate
    assert ev.analyse(ev.Settings(seed=1)).rows[2].estimate != ev.analyse(ev.Settings(seed=2)).rows[2].estimate


def test_summarize_by_hand():
    out = ev.summarize([1.0, 3.0], 1.0, covers=[True, False], widths=[0.5, 1.5])
    assert out["mean"] == 2.0 and out["bias"] == 1.0 and out["sd"] == 1.0 and out["rmse"] == pytest.approx(np.sqrt(2.0))
    assert out["cover"] == 0.5 and out["width"] == 1.0 and "cover" not in ev.summarize([1.0, 2.0], 0.0)


def test_base_experiment_small_run():
    rows = ev.base_experiment(c_values=(1.0, 0.2), n=300, reps=60)
    assert [r["c"] for r in rows] == [1.0, 0.2] and rows[0]["truth"] == pytest.approx(1.0) and rows[1]["truth"] == pytest.approx(0.2)
    e = rows[0]["estimators"]
    assert set(e) == set(ev.ROW_NAMES)
    assert abs(e["Frontdoor"]["bias"]) < 0.08 and e["Naiver Vergleich"]["bias"] == pytest.approx(0.5, abs=0.1) and e["Kontrolle für den Mediator"]["bias"] == pytest.approx(-0.5, abs=0.1)
    assert 0.8 <= e["Frontdoor"]["cover"] <= 1.0 and rows[1]["estimators"]["Frontdoor"]["width"] > 0
    assert rows == ev.base_experiment(c_values=(1.0, 0.2), n=300, reps=60)


def test_violations_experiment_matches_the_closed_forms():
    by = {r["label"]: r for r in ev.violations_experiment()}
    assert len(by) == len(C.VIOLATIONS) and by["Grundfall"]["error"] == pytest.approx(0.0, abs=1e-12)
    assert by["direkte Wirkung X → Y (0,3)"]["error"] == pytest.approx(-0.3) and by["U wirkt auf den Mediator (0,3)"]["error"] == pytest.approx(0.315, abs=1e-3)
    assert by["zweiter Confounder zwischen Mediator und Y (0,5)"]["error"] == pytest.approx(0.2)
    assert by["Messrauschen im Mediator (0,5)"]["error"] == pytest.approx(-0.2) and by["Messrauschen im Mediator (1,0)"]["error"] == pytest.approx(-0.5)
    assert by["starker Confounder (U wirkt mit 2,0)"]["error"] == pytest.approx(0.0, abs=1e-9) and by["starker Confounder (U wirkt mit 2,0)"]["naive"] == pytest.approx(1.8)
    assert by["schwacher erster Schritt (c = 0,2)"]["truth"] == pytest.approx(0.2)


def test_binary_experiment_small_run():
    res = ev.binary_experiment(ns=(100, 1000), reps=80)
    assert res["truth"] == pytest.approx(0.4) and res["do1"] == pytest.approx(0.7) and res["do0"] == pytest.approx(0.3) and res["observed"] == pytest.approx(0.58)
    small, big = res["rows"]
    assert small["n"] == 100 and big["n"] == 1000 and big["estimates"]["sd"] < small["estimates"]["sd"]
    assert abs(big["estimates"]["bias"]) < 0.03 and big["naive"] == pytest.approx(0.58, abs=0.02) and small["empty"] >= 0 and small["reps"] == 80


def test_random_dag_is_topological_and_reproducible():
    parents = ev.random_dag(7, 0.4, SplitMix64(4))
    assert len(parents) == 7 and all(all(p < j for p in ps) for j, ps in enumerate(parents)) and parents == ev.random_dag(7, 0.4, SplitMix64(4))
    assert ev.random_dag(5, 0.0, SplitMix64(1)) == ((),) * 5 and sum(len(p) for p in ev.random_dag(5, 1.0, SplitMix64(1))) == 10


def test_diagram_experiment_small_run():
    res = ev.diagram_experiment(diagrams=200, nodes=7)
    assert res["diagrams"] == 200 and res["backdoor"] == 0 and 0 < res["frontdoor"] < 100 and res["checked"] == res["frontdoor"] and res["wrong"] == 0
    assert len(res["sizes"]) == res["frontdoor"] and min(res["sizes"]) >= 1


def test_a_hidden_cause_of_x_and_y_leaves_no_backdoor_set_and_a_blocked_frontdoor():
    """Knoten 0 (verborgen) wirkt auf x = 1 und y = 4; Knoten 2 hängt auch an 0, deshalb ist der Weg 3 <- 2 <- 0 -> 4 nicht durch x blockiert: keine Frontdoor-Menge."""
    parents = ((), (0,), (0, 1), (1, 2), (0, 3))
    assert not any(B.adjustment_valid(parents, 1, 4, z) for z in B.all_sets([2, 3]))
    assert K.frontdoor_sets(parents, 1, 4, [2, 3]) == []
    assert K.frontdoor_sets(((), (0,), (1,), (0, 2)), 1, 3, [2]) == [(2,)]
