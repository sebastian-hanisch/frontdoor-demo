"""Presets, Reglergrenzen, Permalink-Raster und Konstanten."""

import pytest

import frd_constants as C
import frd_evaluation as ev
import frd_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for name, p in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            assert spec.lo <= p[key] <= spec.hi, (name, key)
            assert P.snap(p[key], spec) == pytest.approx(p[key]), (name, key)


def test_first_preset_is_the_default_setting():
    p = C.PRESETS["Grundfall (Voreinstellung)"]
    assert ev.Settings() == ev.Settings(p["a"], p["c"], p["d"], p["e"], p["f"], p["g"], p["noise"], p["n"], p["seed"])
    assert all(P.SETTING_SPECS[sk].default == p[k] for k, sk in P.PRESET_KEYS.items())


def test_presets_differ_and_each_breaks_exactly_one_assumption():
    assert len({tuple(sorted(p.items())) for p in C.PRESETS.values()}) == len(C.PRESETS)
    base = C.PRESETS["Grundfall (Voreinstellung)"]
    for name, p in C.PRESETS.items():
        changed = [k for k in p if p[k] != base[k]]
        assert len(changed) <= 1, (name, changed)
    assert C.PRESETS["Direkte Wirkung"]["e"] > 0 and C.PRESETS["Confounder wirkt auf den Mediator"]["f"] > 0 and C.PRESETS["Zweiter Confounder"]["g"] > 0
    assert C.PRESETS["Messfehler im Mediator"]["noise"] > 0 and C.PRESETS["Schwacher erster Schritt"]["c"] < C.DEFAULT_C


def test_default_does_not_sit_on_the_coincidence_of_the_mediator_control():
    """Bei c = 0,5 trifft die Kontrolle für den Mediator zufällig (Fehler 0,5 − c = 0); der Grundfall liegt dort nicht."""
    assert C.DEFAULT_C != 0.5 and all(p["c"] != 0.5 for p in C.PRESETS.values())


def test_snap_clamps_and_rounds_to_the_grid():
    a = P.SETTING_SPECS["a_slider"]
    assert P.snap(5.0, a) == C.A_MAX and P.snap(-1.0, a) == C.A_MIN and P.snap(0.43, a) == pytest.approx(0.4)
    n = P.SETTING_SPECS["n_slider"]
    assert P.snap(50, n) == C.N_MIN and P.snap(10 ** 6, n) == C.N_MAX and P.snap(1049, n) == 1000 and P.snap(1051, n) == 1100
    assert P.snap(999999, P.SETTING_SPECS["seed_input"]) == 999999 and P.snap(0.07, P.SETTING_SPECS["noise_slider"]) == pytest.approx(0.1)


def test_bounds_defaults_and_params():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("g_slider") == (C.G_MIN, C.G_MAX)
    assert C.DEFAULT_N in range(C.N_MIN, C.N_MAX + 1, C.N_STEP) and {s.url_param for s in P.SETTING_SPECS.values()} == {"a", "c", "d", "e", "f", "g", "nz", "n", "seed"}


def test_experiment_constants_are_consistent():
    assert C.EXP_C_VALUES[0] == C.DEFAULT_C and C.EXP_REPS >= 200 and C.EXP_DIAGRAMS >= 1000 and C.EXP_NODES >= 5 and C.EXP_BINARY_NS == tuple(sorted(C.EXP_BINARY_NS))
    assert C.COEF_LO > 0 and C.COEF_HI > C.COEF_LO and 0 < C.RANDOM_P_LO and C.RANDOM_P_LO + C.RANDOM_P_WIDTH < 1
    assert next(iter(C.VIOLATIONS)) == "Grundfall" and C.VIOLATIONS["Grundfall"] == {}
