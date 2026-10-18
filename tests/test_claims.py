"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über SplitMix64-Ströme (ganzzahlig, plattformstabil); Fließkomma-Reihenfolgen können je Bibliotheksversion die letzte Stelle ändern, deshalb
Toleranzen von einer halben Einheit der gedruckten Stelle.
"""

import pytest

import frd_constants as C
import frd_evaluation as ev


@pytest.fixture(scope="module")
def base():
    return ev.base_experiment()


@pytest.fixture(scope="module")
def viol():
    return {r["label"]: r for r in ev.violations_experiment()}


@pytest.fixture(scope="module")
def binary():
    return ev.binary_experiment()


@pytest.fixture(scope="module")
def diagrams():
    return ev.diagram_experiment()


def test_default_sample():
    a = ev.analyse(ev.Settings())
    by = {r.name: r for r in a.rows}
    assert a.truth == pytest.approx(1.0) and by["Naiver Vergleich"].estimate == pytest.approx(1.54, abs=0.005) and by["Kontrolle für den Mediator"].estimate == pytest.approx(0.52, abs=0.005)
    assert by["Frontdoor"].estimate == pytest.approx(1.02, abs=0.005) and by["Frontdoor"].covers is True and by["Orakel mit U (nicht beobachtbar)"].estimate == pytest.approx(1.11, abs=0.005)
    assert by["Naiver Vergleich"].covers is False and by["Kontrolle für den Mediator"].covers is False
    assert a.steps["c"] == pytest.approx(1.05, abs=0.005) and a.steps["se_c"] == pytest.approx(0.032, abs=0.0006)
    assert a.steps["d"] == pytest.approx(0.97, abs=0.005) and a.steps["se_d"] == pytest.approx(0.057, abs=0.0006)
    assert ev.verdict_instance(a)[0] == "success"


def test_experiment_1_base_case(base):
    e = base[0]["estimators"]
    assert base[0]["c"] == 1.0 and C.EXP_N == 500 and C.EXP_REPS == 500
    fd, orc = e["Frontdoor"], e["Orakel mit U (nicht beobachtbar)"]
    assert fd["bias"] == pytest.approx(-0.001, abs=0.0006) and fd["sd"] == pytest.approx(0.065, abs=0.0006) and fd["cover"] == pytest.approx(0.95, abs=0.005)
    assert orc["bias"] == pytest.approx(-0.003, abs=0.0006) and orc["sd"] == pytest.approx(0.066, abs=0.0006) and orc["cover"] == pytest.approx(0.95, abs=0.005)
    assert e["Naiver Vergleich"]["bias"] == pytest.approx(0.50, abs=0.005) and e["Kontrolle für den Mediator"]["bias"] == pytest.approx(-0.50, abs=0.005)
    assert e["Naiver Vergleich"]["cover"] == 0.0 and e["Kontrolle für den Mediator"]["cover"] == 0.0


def test_experiment_1_weak_first_step(base):
    by = {r["c"]: r["estimators"] for r in base}
    assert [r["c"] for r in base] == [1.0, 0.5, 0.2, 0.1, 0.05]
    for c, bias in ((0.5, 0.001), (0.2, -0.002), (0.1, 0.000), (0.05, 0.001)):
        assert by[c]["Frontdoor"]["bias"] == pytest.approx(bias, abs=0.0006), c
        assert 0.94 <= by[c]["Frontdoor"]["cover"] <= 0.9625, c
    for c, sd in ((0.5, 0.044), (0.2, 0.034), (0.1, 0.0325), (0.05, 0.031)):
        assert by[c]["Frontdoor"]["sd"] == pytest.approx(sd, abs=0.0006), c
    assert all(0.059 <= by[c]["Orakel mit U (nicht beobachtbar)"]["sd"] <= 0.068 for c in (0.5, 0.2, 0.1, 0.05))
    assert by[0.05]["Frontdoor"]["width"] == pytest.approx(0.125, abs=0.0006)
    assert by[0.5]["Kontrolle für den Mediator"]["bias"] == pytest.approx(-0.004, abs=0.0006)
    assert by[0.2]["Kontrolle für den Mediator"]["bias"] == pytest.approx(0.30, abs=0.005) and by[0.1]["Kontrolle für den Mediator"]["bias"] == pytest.approx(0.40, abs=0.005)


def test_experiment_2_violations(viol):
    assert viol["direkte Wirkung X → Y (0,3)"]["error"] == pytest.approx(-0.30, abs=0.005)
    assert viol["U wirkt auf den Mediator (0,3)"]["error"] == pytest.approx(0.32, abs=0.005)
    assert viol["zweiter Confounder zwischen Mediator und Y (0,5)"]["error"] == pytest.approx(0.20, abs=0.005)
    assert viol["Messrauschen im Mediator (0,5)"]["error"] == pytest.approx(-0.20, abs=0.005) and viol["Messrauschen im Mediator (1,0)"]["error"] == pytest.approx(-0.50, abs=0.005)
    assert viol["Messrauschen im Mediator (1,0)"]["frontdoor"] == pytest.approx(0.50, abs=0.005) and viol["Messrauschen im Mediator (1,0)"]["truth"] == pytest.approx(1.0)
    assert viol["starker Confounder (U wirkt mit 2,0)"]["error"] == pytest.approx(0.0, abs=0.005) and viol["starker Confounder (U wirkt mit 2,0)"]["naive"] == pytest.approx(1.80, abs=0.005)


def test_experiment_3_binary(binary):
    assert binary["truth"] == pytest.approx(0.40) and binary["observed"] == pytest.approx(0.58)
    rows = {r["n"]: r for r in binary["rows"]}
    assert list(rows) == [100, 500, 2000] and rows[100]["reps"] == 500
    for n, bias in ((100, 0.002), (500, 0.001), (2000, 0.001)):
        assert rows[n]["estimates"]["bias"] == pytest.approx(bias, abs=0.0006), n
    for n, sd in ((100, 0.142), (500, 0.054), (2000, 0.028)):
        assert rows[n]["estimates"]["sd"] == pytest.approx(sd, abs=0.0006), n
    assert rows[100]["empty"] == 6 and rows[500]["empty"] == 0 and rows[2000]["empty"] == 0


def test_experiment_4_diagrams(diagrams):
    assert diagrams["diagrams"] == 3000 and diagrams["nodes"] == 7 and diagrams["backdoor"] == 0 and diagrams["frontdoor"] == 312
    assert diagrams["frontdoor"] / diagrams["diagrams"] == pytest.approx(0.104, abs=0.0005)
    s = diagrams["sizes"]
    assert s.count(1) == 265 and s.count(2) == 43 and s.count(3) == 4 and len(s) == 312 and max(s) == 3
    assert diagrams["checked"] == 312 and diagrams["wrong"] == 0


def test_the_text_numbers_of_the_app_follow_the_findings(viol, diagrams):
    assert "Messrauschen 1,0 schätzt die Frontdoor 0,50 statt 1,00" in ev.verdict_violations(list(viol.values()))[1]
    assert "(0)" in ev.verdict_diagrams(diagrams)[1] and "10,4 %" in ev.verdict_diagrams(diagrams)[1]
