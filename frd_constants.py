"""Konstanten: Regler-Grenzen, feste Experimente, Farben und Presets."""

# Rechenaufwand am Maximum: n = 5 000 und fünf Knoten, zwei Regressionen: wenige Millisekunden.
N_MIN, N_MAX, N_STEP, DEFAULT_N = 100, 5000, 100, 500
SEED_MAX, DEFAULT_SEED = 999999, 35
Z_CRIT = 1.959964

# Stärken (Einheitsrauschen in jedem Knoten)
A_MIN, A_MAX, A_STEP, DEFAULT_A = 0.0, 2.5, 0.1, 1.0       # Stärke des verborgenen Confounders (U -> X und U -> Y gleich stark)
C_MIN, C_MAX, C_STEP, DEFAULT_C = 0.1, 1.5, 0.1, 1.0       # erster Schritt X -> M
D_MIN, D_MAX, D_STEP, DEFAULT_D = 0.1, 1.5, 0.1, 1.0       # zweiter Schritt M -> Y
E_MIN, E_MAX, E_STEP = 0.0, 0.6, 0.1                       # Verletzung: direkte Wirkung X -> Y
F_MIN, F_MAX, F_STEP = 0.0, 0.6, 0.1                       # Verletzung: U -> M
G_MIN, G_MAX, G_STEP = 0.0, 0.8, 0.1                       # Verletzung: zweiter Confounder U2 -> M und U2 -> Y
NOISE_MIN, NOISE_MAX, NOISE_STEP = 0.0, 1.5, 0.1           # Messrauschen im Mediator (Standardabweichung)

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 980000
EXP_N, EXP_REPS = 500, 500
EXP_C_VALUES = (1.0, 0.5, 0.2, 0.1, 0.05)                  # Experiment 1: Stärke des ersten Schritts (der erste Wert ist der Grundfall)
EXP_BINARY_NS, EXP_BINARY_REPS = (100, 500, 2000), 500     # Experiment 3: binäres Beispiel
EXP_DIAGRAMS, EXP_NODES = 3000, 7                          # Experiment 4: Zufallsdiagramme mit verborgener gemeinsamer Ursache
RANDOM_P_LO, RANDOM_P_WIDTH = 0.25, 0.25                   # Kantenwahrscheinlichkeit 0,25 bis 0,5
COEF_LO, COEF_HI = 0.4, 1.2

# Experiment 2: Verletzungen (Name -> Parameter der Welt; Rest Voreinstellung)
VIOLATIONS = {
    "Grundfall": {},
    "starker Confounder (U wirkt mit 2,0)": dict(a=2.0),
    "schwacher erster Schritt (c = 0,2)": dict(c=0.2),
    "direkte Wirkung X → Y (0,3)": dict(e=0.3),
    "U wirkt auf den Mediator (0,3)": dict(f=0.3),
    "zweiter Confounder zwischen Mediator und Y (0,5)": dict(g=0.5),
    "Messrauschen im Mediator (0,5)": dict(noise=0.5),
    "Messrauschen im Mediator (1,0)": dict(noise=1.0),
}

# Binäres Beispiel: U ~ B(0,5); P(X=1 | U) = 0,2 / 0,8; P(M=1 | X) = 0,1 / 0,9; P(Y=1 | M, U) = 0,1 + 0,5 M + 0,3 U
BIN_PU, BIN_PX, BIN_PM = 0.5, (0.2, 0.8), (0.1, 0.9)
BIN_Y0, BIN_YM, BIN_YU = 0.1, 0.5, 0.3

COLOR_TRUTH = "#D68A2E"
COLOR_GOOD = "#2F6B65"
COLOR_BAD = "#c0392b"
COLOR_NEUTRAL = "#8a8a8a"
COLOR_HIDDEN = "#cfcfcf"
COLOR_CONTROL = "#4c78a8"


def preset(**kw) -> dict:
    base = dict(a=DEFAULT_A, c=DEFAULT_C, d=DEFAULT_D, e=0.0, f=0.0, g=0.0, noise=0.0, n=DEFAULT_N, seed=DEFAULT_SEED)
    base.update(kw)
    return base


PRESETS = {
    "Grundfall (Voreinstellung)": preset(),
    "Direkte Wirkung": preset(e=0.3),
    "Confounder wirkt auf den Mediator": preset(f=0.3),
    "Zweiter Confounder": preset(g=0.5),
    "Messfehler im Mediator": preset(noise=1.0),
    "Schwacher erster Schritt": preset(c=0.2),
}
PRESET_HELP = {
    "Grundfall (Voreinstellung)": "Alle Annahmen gelten: Der Confounder ist verborgen, die Wirkung läuft ganz über den Mediator. Die Frontdoor trifft, so genau wie das unmögliche Backdoor-Orakel.",
    "Direkte Wirkung": "Ein Pfeil von der Regel direkt auf den Servicegrad: Die Frontdoor schätzt nur den Weg über den Bestand und verfehlt die Gesamtwirkung um den direkten Teil.",
    "Confounder wirkt auf den Mediator": "Die Standortqualität beeinflusst auch den Bestand: Der Rückweg zum Mediator ist offen, die Frontdoor ist verzerrt.",
    "Zweiter Confounder": "Eine zweite verborgene Ursache wirkt auf Bestand und Servicegrad: Der Rückweg vom Mediator zum Ergebnis ist nicht mehr durch x blockiert.",
    "Messfehler im Mediator": "Der Bestand ist nur mit Rauschen gemessen: Der zweite Schritt wird zum Nullpunkt gezogen, die Frontdoor unterschätzt die Wirkung.",
    "Schwacher erster Schritt": "Die Regel bewegt den Bestand kaum: Die Wirkung ist klein, das Intervall in Relation zu ihr breit. Die Kontrolle für den Bestand trifft bei c = 0,5 zufällig, hier nicht.",
}
