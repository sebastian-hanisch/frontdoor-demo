"""Verdikte: Art und Text je Lage; jede Zahl im Text stammt aus dem Befund, nicht aus einer Konstanten."""

import frd_evaluation as ev


def test_instance_verdict_in_the_base_case_is_a_success():
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings()))
    assert kind == "success" and "Die Frontdoor schätzt" in text and "deckt" in text and "verfehlt die Wahrheit" not in text
    assert "naive Vergleich liegt um +0," in text and "Kontrolle für den Mediator um −0," in text


def test_instance_verdict_names_every_active_violation():
    for kw, phrase in ((dict(e=0.3), "direkten Pfeil"), (dict(f=0.3), "Standortqualität"), (dict(g=0.5), "zweiten Confounder"), (dict(noise=0.5), "Messrauschen")):
        kind, text = ev.verdict_instance(ev.analyse(ev.Settings(**kw)))
        assert kind == "warning" and "Eine Annahme ist verletzt" in text and phrase in text, kw
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings(e=0.3, noise=0.5)))
    assert "direkten Pfeil" in text and "Messrauschen" in text and "zweiten Confounder" not in text


def test_instance_verdict_is_info_when_the_interval_misses_without_a_violation():
    for seed in range(60):
        a = ev.analyse(ev.Settings(n=100, seed=seed))
        if not next(r for r in a.rows if r.name == "Frontdoor").covers:
            kind, text = ev.verdict_instance(a)
            assert kind == "info" and "verfehlt" in text
            return
    raise AssertionError("bei n = 100 und 60 Seeds sollte mindestens ein Intervall verfehlen")


def test_base_verdict_reads_the_first_row():
    rows = ev.base_experiment(c_values=(1.0, 0.5), n=200, reps=40)
    kind, text = ev.verdict_base(rows)
    f = rows[0]["estimators"]["Frontdoor"]
    assert kind == "info" and ev.de(f["sd"], 3) in text and ev.de(100 * f["cover"], 0) + " %" in text and "(c = 1)" in text


def test_violation_verdict_quotes_the_exact_values():
    kind, text = ev.verdict_violations(ev.violations_experiment())
    assert kind == "info" and "schätzt die Frontdoor 0,50 statt 1,00" in text and "−0,30" in text and "+0,00" in text


def test_binary_verdict_quotes_the_rows():
    res = ev.binary_experiment(ns=(100, 400), reps=40)
    kind, text = ev.verdict_binary(res)
    assert kind == "info" and "Wahrheit 0,40" in text and "0,58" in text and "n = 400" in text and "n = 100" in text and f"in {res['rows'][0]['empty']} von 40" in text


def test_diagram_verdict_counts():
    res = ev.diagram_experiment(diagrams=100, nodes=7)
    kind, text = ev.verdict_diagrams(res)
    assert kind == "info" and "In 100 Zufallsdiagrammen" in text and f"({res['backdoor']})" in text and f"in {res['frontdoor']} " in text
    assert f"{res['checked'] - res['wrong']} von {res['checked']}" in text
