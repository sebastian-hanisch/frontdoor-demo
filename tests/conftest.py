"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def unit_world():
    """Mini-Instanz 1 (a = b = c = d = 1, Einheitsrauschen): U -> X, U -> Y, X -> M, M -> Y, kein weiterer Pfeil.
    Handrechnung: Var X = a² + 1 = 2; Var M = c² Var X + 1 = 3; Cov(X, M) = c Var X = 2; Cov(X, Y) = ab + cd Var X = 1 + 2 = 3; Cov(M, Y) = d Var M + b·c·a = 3 + 1 = 4.
    Naiv 3/2 = 1,5. Regression von Y auf X und M: [[2, 2], [2, 3]] β = (3, 4): det 2, β_X = (9 − 8)/2 = 0,5, β_M = (8 − 6)/2 = 1.
    Frontdoor: ĉ = Cov(X, M)/Var X = 1, d̂ = β_M = 1, Produkt 1 = Wahrheit c·d."""
    from frd_world import World

    return World()


@pytest.fixture
def classic():
    """Diagramm 0 = U (verborgen), 1 = X, 2 = M, 3 = Y: U -> X, U -> Y, X -> M, M -> Y. Gültige Frontdoor-Menge {M}."""
    return ((), (0,), (1,), (0, 2))


@pytest.fixture
def binary_sample():
    """Mini-Instanz 2 (binär, acht Einheiten): (x, m, y) = (0,0,0) (0,0,1) (0,1,1) (0,1,1) (1,0,0) (1,1,1) (1,1,0) (1,1,1).
    P(x′ = 0) = P(x′ = 1) = 1/2; P(M=1 | X=0) = 1/2, P(M=1 | X=1) = 3/4. Zellmittel y: (m=0, x′=0) 1/2; (m=0, x′=1) 0; (m=1, x′=0) 1; (m=1, x′=1) 2/3.
    Innen: m = 0: 1/4, m = 1: 5/6. P(Y=1 | do(X=1)) = 1/4·1/4 + 3/4·5/6 = 11/16; P(Y=1 | do(X=0)) = 1/2·1/4 + 1/2·5/6 = 13/24."""
    x = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    m = np.array([0, 0, 1, 1, 0, 1, 1, 1])
    y = np.array([0, 1, 1, 1, 0, 1, 0, 1])
    return x, m, y
