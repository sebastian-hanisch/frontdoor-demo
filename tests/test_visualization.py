"""Figuren und Tabellen: bauen ohne Fehler, und die gezeichneten Zahlen sind die der Auswertung."""

import pytest

import frd_evaluation as ev
import frd_visualization as V


def test_diagram_chart_bars_add_up_to_the_frontdoor_share():
    res = ev.diagram_experiment(diagrams=300, nodes=7)
    fig = V.build_diagrams_chart(res)
    y = list(fig.data[0].y)
    assert len(y) == 5 and y[0] == 0.0 and y[1] == pytest.approx(100 * res["frontdoor"] / 300)
    assert sum(y[2:]) == pytest.approx(y[1]) and len(fig.data[0].marker.color) == 5


def test_estimate_figures_and_tables_build_for_every_setting():
    for kw in (dict(), dict(e=0.3), dict(f=0.3), dict(g=0.5), dict(noise=1.0), dict(c=0.1, n=100)):
        a = ev.analyse(ev.Settings(**kw))
        assert V.build_diagram(a.settings) is not None and V.build_estimates_chart(a) is not None
        assert len(V.build_estimates_table(a)) == 4 and len(V.build_steps_table(a)) == 3
        V.style_estimates_table(V.build_estimates_table(a))


def test_experiment_figures_and_tables_build():
    rows = ev.base_experiment(c_values=(1.0, 0.5), n=200, reps=30)
    assert V.build_base_chart(rows) is not None and len(V.build_base_table(rows)) >= 2
    viol = ev.violations_experiment()
    assert V.build_violations_chart(viol) is not None and len(V.build_violations_table(viol)) == len(viol)
    res = ev.binary_experiment(ns=(100, 300), reps=30)
    assert V.build_binary_chart(res) is not None and len(V.build_binary_table(res)) == 2
