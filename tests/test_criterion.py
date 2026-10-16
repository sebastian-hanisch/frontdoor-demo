"""Frontdoor-Kriterium: Hand-Diagramme, Zerlegung in die drei Bedingungen und Brute-Force-Gegenproben."""

import itertools

import numpy as np
import pytest

import frd_criterion as K

nx = pytest.importorskip("networkx")


def test_classic_diagram(classic):
    chk = K.frontdoor_check(classic, 1, 3, (2,))
    assert chk == {"intercepts": True, "no_backdoor": True, "blocked": True, "valid": True}
    assert K.frontdoor_sets(classic, 1, 3, [0, 1, 2, 3]) == [(2,)] and K.frontdoor_sets(classic, 1, 3, [1, 2, 3]) == [(2,)]
    assert K.frontdoor_sets(classic, 1, 3, [0]) == []                                          # nur der verborgene Knoten steht zur Wahl: keine Menge


def test_direct_edge_breaks_interception():
    parents = ((), (0,), (1,), (0, 1, 2))                                                      # zusätzlich X -> Y
    chk = K.frontdoor_check(parents, 1, 3, (2,))
    assert chk["intercepts"] is False and chk["valid"] is False and chk["no_backdoor"] and chk["blocked"]


def test_confounder_of_x_and_the_mediator_breaks_the_first_step():
    parents = ((), (0,), (0, 1), (0, 2))                                                       # U -> M
    chk = K.frontdoor_check(parents, 1, 3, (2,))
    assert chk["no_backdoor"] is False and chk["valid"] is False and chk["intercepts"] and not chk["blocked"]


def test_second_confounder_of_mediator_and_outcome_breaks_the_second_step():
    parents = ((), (), (0,), (1, 2), (0, 1, 3))                                                # U2 (1) -> M, U2 -> Y; X = 2, M = 3, Y = 4
    chk = K.frontdoor_check(parents, 2, 4, (3,))
    assert chk["blocked"] is False and chk["valid"] is False and chk["intercepts"] and chk["no_backdoor"]


def test_two_mediators_in_a_row_and_in_parallel():
    chain = ((), (0,), (1,), (2,), (0, 3))                                                     # U -> X -> M1 -> M2 -> Y
    assert K.frontdoor_sets(chain, 1, 4, [2, 3]) == [(2,), (3,), (2, 3)]
    parallel = ((), (0,), (1,), (1,), (0, 2, 3))                                               # X -> M1 -> Y und X -> M2 -> Y
    assert K.frontdoor_sets(parallel, 1, 4, [2, 3]) == [(2, 3)]                                # einzeln fängt keiner beide Pfade ab


def test_empty_set_and_endpoints_are_rejected_or_invalid(classic):
    assert K.frontdoor_valid(classic, 1, 3, ()) is False
    with pytest.raises(ValueError):
        K.frontdoor_check(classic, 1, 3, (1,))
    with pytest.raises(ValueError):
        K.frontdoor_check(classic, 1, 1, (2,))


def test_no_causal_path_means_no_interception():
    parents = ((), (0,), (0,), (0, 2))                                                         # X (1) hat keinen Pfad zu Y (3)
    assert K.frontdoor_check(parents, 1, 3, (2,))["intercepts"] is False


def surgery(parents, x, y, z):
    """Zweiter Rechenweg über networkx: dieselben drei Bedingungen mit nx.is_d_separator."""
    n = len(parents)

    def graph(edges):
        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        g.add_edges_from(edges)
        return g

    full = [(p, c) for c, ps in enumerate(parents) for p in ps]
    g = graph(full)
    intercepts = bool(z) and nx.has_path(g, x, y) and all(set(path[1:-1]) & set(z) for path in nx.all_simple_paths(g, x, y))
    no_back = bool(z) and all(nx.is_d_separator(graph([(p, c) for p, c in full if p != x]), {x}, {v}, set()) for v in z)
    blocked = bool(z) and all(nx.is_d_separator(graph([(p, c) for p, c in full if p not in z]), {v}, {y}, {x}) for v in z)
    return intercepts and no_back and blocked


def test_criterion_against_networkx_on_random_diagrams():
    rng = np.random.default_rng(5)
    checked = valid = 0
    for _ in range(200):
        n = 6
        p = rng.uniform(0.25, 0.55)
        parents = tuple(tuple(i for i in range(j) if rng.random() < p) for j in range(n))
        x, y = sorted(int(v) for v in rng.choice(n, 2, replace=False))
        others = [v for v in range(n) if v not in (x, y)]
        for k in range(1, len(others) + 1):
            for z in itertools.combinations(others, k):
                ok = K.frontdoor_valid(parents, x, y, z)
                assert ok == surgery(parents, x, y, z), (parents, x, y, z)
                checked += 1
                valid += ok
    assert checked == 3000 and valid > 20
