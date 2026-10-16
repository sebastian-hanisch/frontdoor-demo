"""Frontdoor-Kriterium (Pearl) für Diagramme als Tupel von Elternmengen. Nur Standardbibliothek und dsep_graph / bdr_backdoor (wortgleich aus den Vorgängern)."""

from __future__ import annotations

import itertools

import bdr_backdoor as B
import dsep_graph as G


def frontdoor_check(parents, x: int, y: int, z) -> dict:
    """Die drei Bedingungen für die Menge z: (1) z fängt alle gerichteten Pfade von x nach y ab, (2) kein offener Rückweg von x zu irgendeinem Knoten von z,
    (3) alle Rückwege von z nach y sind durch x blockiert."""
    z = tuple(sorted(set(z)))
    if x == y or x in z or y in z:
        raise ValueError("x und y dürfen nicht in z liegen und müssen verschieden sein")
    paths = B.directed_paths(parents, x, y)
    intercepts = bool(z) and bool(paths) and all(set(p[1:-1]) & set(z) for p in paths)
    cut_x = tuple(tuple(q for q in ps if q != x) for ps in parents)                  # Kanten aus x entfernt: übrig bleiben nur Rückwege von x
    no_backdoor = bool(z) and all(G.d_separated(cut_x, x, v, ()) for v in z)
    cut_z = tuple(tuple(q for q in ps if q not in z) for ps in parents)             # Kanten aus z entfernt
    blocked = bool(z) and all(G.d_separated(cut_z, v, y, (x,)) for v in z)
    return {"intercepts": intercepts, "no_backdoor": no_backdoor, "blocked": blocked, "valid": intercepts and no_backdoor and blocked}


def frontdoor_valid(parents, x: int, y: int, z) -> bool:
    return frontdoor_check(parents, x, y, z)["valid"]


def frontdoor_sets(parents, x: int, y: int, observed) -> list[tuple]:
    """Alle gültigen Frontdoor-Mengen unter den beobachteten Knoten (ohne x und y), nach Größe sortiert."""
    cands = sorted(v for v in observed if v not in (x, y))
    return [z for k in range(1, len(cands) + 1) for z in itertools.combinations(cands, k) if frontdoor_valid(parents, x, y, z)]
