"""Auswertung: Einstellungen, eine Stichprobe mit allen Schätzern, die vier Experimente über feste Ströme und die Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import bdr_backdoor as B
import frd_constants as C
import frd_criterion as K
import frd_estimators as E
import frd_world as W
from kausal_generator import SplitMix64


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def signed(x: float, digits: int = 2) -> str:
    x = 0.0 if abs(x) < 0.5 * 10 ** -digits * 1e-3 else x          # −0,00 aus Rundungsresten vermeiden
    return ("+" if x >= 0 else "−") + de(abs(x), digits)


@dataclass(frozen=True)
class Settings:
    a: float = C.DEFAULT_A
    c: float = C.DEFAULT_C
    d: float = C.DEFAULT_D
    e: float = 0.0
    f: float = 0.0
    g: float = 0.0
    noise: float = 0.0
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED

    @property
    def world(self) -> W.World:
        return W.World(self.a, self.c, self.d, self.e, self.f, self.g, self.noise)


ROW_NAMES = ("Naiver Vergleich", "Kontrolle für den Mediator", "Frontdoor", "Orakel mit U (nicht beobachtbar)")


@dataclass
class Row:
    name: str
    estimate: float
    se: float
    lo: float
    hi: float
    error: float
    covers: bool
    limit: float          # Grenzwert bei n → ∞


@dataclass
class Analysis:
    settings: Settings
    rows: list
    truth: float
    steps: dict           # Frontdoor: ĉ, d̂ mit Standardfehlern und ihre Grenzwerte
    population: dict


def _row(name, est, se, truth, limit) -> Row:
    return Row(name, est, se, est - C.Z_CRIT * se, est + C.Z_CRIT * se, est - truth, bool(abs(est - truth) <= C.Z_CRIT * se), limit)


def analyse(s: Settings) -> Analysis:
    w = s.world
    v, m = (a[0] for a in W.simulate(w, s.n, SplitMix64(s.seed)))
    x, y, u = v[:, W.X], v[:, W.Y], v[:, W.U]
    truth = W.truth(w)
    pop = E.population_estimates(w)
    fd = E.frontdoor(x, m, y)
    rows = [_row("Naiver Vergleich", *E.naive(x, y), truth, pop["naive"]), _row("Kontrolle für den Mediator", *E.control_mediator(x, m, y), truth, pop["control"]),
            _row("Frontdoor", fd["estimate"], fd["se"], truth, pop["frontdoor"]), _row("Orakel mit U (nicht beobachtbar)", *E.oracle(x, u, y), truth, pop["oracle"])]
    return Analysis(s, rows, truth, {"c": fd["c"], "d": fd["d"], "se_c": fd["se_c"], "se_d": fd["se_d"], "c_limit": pop["c"], "d_limit": pop["d"]}, pop)


def summarize(values, truth: float, covers=None, widths=None) -> dict:
    v = np.asarray(values, dtype=float)
    out = {"mean": float(v.mean()), "bias": float(v.mean() - truth), "sd": float(v.std()), "rmse": float(np.sqrt(((v - truth) ** 2).mean()))}
    if covers is not None:
        out["cover"] = float(np.mean(covers))
        out["width"] = float(np.mean(widths))
    return out


# ----------------------------------------------------------------------------- Experiment 1: Grundfall und Stärke des ersten Schritts

def base_experiment(c_values=C.EXP_C_VALUES, n: int = C.EXP_N, reps: int = C.EXP_REPS) -> list[dict]:
    """Je Stärke c des ersten Schritts: naiver Vergleich, Kontrolle für den Mediator, Frontdoor und das Orakel über feste Wiederholungen (Grundfall: c = 1)."""
    rows = []
    for i, c in enumerate(c_values):
        w = W.World(c=c)
        truth = W.truth(w)
        v, m = W.simulate(w, n, SplitMix64(C.SEED_BASE + i), reps)
        vals = {name: [] for name in ROW_NAMES}
        cover = {name: [] for name in ROW_NAMES}
        width = {name: [] for name in ROW_NAMES}
        for r in range(reps):
            x, y, u = v[r, :, W.X], v[r, :, W.Y], v[r, :, W.U]
            fd = E.frontdoor(x, m[r], y)
            for name, (est, se) in (("Naiver Vergleich", E.naive(x, y)), ("Kontrolle für den Mediator", E.control_mediator(x, m[r], y)), ("Frontdoor", (fd["estimate"], fd["se"])),
                                    ("Orakel mit U (nicht beobachtbar)", E.oracle(x, u, y))):
                vals[name].append(est)
                cover[name].append(abs(est - truth) <= C.Z_CRIT * se)
                width[name].append(2 * C.Z_CRIT * se)
        rows.append({"c": c, "truth": truth, "estimators": {name: summarize(vals[name], truth, cover[name], width[name]) for name in ROW_NAMES}})
    return rows


# ----------------------------------------------------------------------------- Experiment 2: Verletzungen (Population)

def violations_experiment() -> list[dict]:
    """Exakte Grenzwerte der Schätzer, wenn eine Annahme verletzt ist; die Wahrheit schließt die Verletzung ein (Gesamtwirkung)."""
    out = []
    for label, kw in C.VIOLATIONS.items():
        pop = E.population_estimates(W.World(**kw))
        out.append({"label": label, "truth": pop["truth"], "naive": pop["naive"], "control": pop["control"], "frontdoor": pop["frontdoor"], "error": pop["frontdoor"] - pop["truth"]})
    return out


# ----------------------------------------------------------------------------- Experiment 3: binäres Beispiel

def binary_experiment(ns=C.EXP_BINARY_NS, reps: int = C.EXP_BINARY_REPS) -> dict:
    t1, t0 = W.binary_truth()
    o1, o0 = W.binary_observed()
    rows = []
    for i, n in enumerate(ns):
        u, x, m, y = W.simulate_binary(n, SplitMix64(C.SEED_BASE + 100 + i), reps)
        est, naive, empty = [], [], 0
        for r in range(reps):
            naive.append(float(y[r][x[r] == 1].mean() - y[r][x[r] == 0].mean()))
            res = E.frontdoor_binary(x[r], m[r], y[r])
            if res is None:
                empty += 1
            else:
                est.append(res[0] - res[1])
        rows.append({"n": n, "naive": float(np.mean(naive)), "estimates": summarize(est, t1 - t0), "empty": empty, "reps": reps})
    return {"truth": t1 - t0, "do1": t1, "do0": t0, "observed": o1 - o0, "rows": rows}


# ----------------------------------------------------------------------------- Experiment 4: Zufallsdiagramme mit verborgener gemeinsamer Ursache

def random_dag(n: int, p: float, rng: SplitMix64) -> tuple:
    """Zufälliges Diagramm in der Knotenreihenfolge 0..n−1 mit Kantenwahrscheinlichkeit p (ein Block Gleichverteilte, feste Reihenfolge)."""
    u = rng.uniform(n * (n - 1) // 2)
    k = 0
    parents = []
    for j in range(n):
        ps = []
        for i in range(j):
            if u[k] < p:
                ps.append(i)
            k += 1
        parents.append(tuple(ps))
    return tuple(parents)


def diagram_experiment(diagrams: int = C.EXP_DIAGRAMS, nodes: int = C.EXP_NODES) -> dict:
    """Zufallsdiagramme mit der verborgenen gemeinsamen Ursache u (Knoten 0) von x und y und mindestens einem kausalen Pfad; alle anderen Knoten sind beobachtet.
    Gezählt wird, wie oft eine gültige Frontdoor-Menge existiert; Gegenprobe: In einer linearen Population mit zufälligen Koeffizienten stimmt die Formel mit der Wirkung."""
    rng = SplitMix64(C.SEED_BASE + 200)
    done = fd = bd = checked = wrong = 0
    sizes = []
    while done < diagrams:
        p = C.RANDOM_P_LO + C.RANDOM_P_WIDTH * float(rng.uniform(1)[0])
        base = [list(ps) for ps in random_dag(nodes, p, rng)]
        draw = rng.uniform(2)
        x = 1 + int(draw[0] * (nodes - 1))
        y = 1 + int(draw[1] * (nodes - 1))
        if x == y:
            continue
        x, y = sorted((x, y))
        for v in (x, y):
            if 0 not in base[v]:
                base[v].append(0)
        parents = tuple(tuple(sorted(ps)) for ps in base)
        if not B.directed_paths(parents, x, y):
            continue
        done += 1
        observed = [v for v in range(1, nodes) if v not in (x, y)]
        sets = K.frontdoor_sets(parents, x, y, observed)
        bd += any(B.adjustment_valid(parents, x, y, z) for z in B.all_sets(observed))
        if sets:
            fd += 1
            smallest = min(sets, key=len)
            sizes.append(len(smallest))
            bm = np.zeros((nodes, nodes))
            for j, ps in enumerate(parents):
                if ps:
                    dr = rng.uniform(2 * len(ps))
                    for k, q in enumerate(ps):
                        bm[j, q] = (C.COEF_LO + (C.COEF_HI - C.COEF_LO) * dr[k]) * (1.0 if dr[len(ps) + k] < 0.5 else -1.0)
            inv = np.linalg.inv(np.eye(nodes) - bm)
            cov, total = inv @ inv.T, inv - np.eye(nodes)
            z = list(smallest)
            b = np.array([E.coefficients(cov, v, [x])[0] for v in z])
            theta = E.coefficients(cov, y, z + [x])[:len(z)]
            checked += 1
            wrong += abs(b @ theta - total[y, x]) > 1e-8
    return {"diagrams": diagrams, "nodes": nodes, "frontdoor": fd, "backdoor": bd, "sizes": sizes, "checked": checked, "wrong": int(wrong)}


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def verdict_instance(a: Analysis) -> tuple[str, str]:
    by = {r.name: r for r in a.rows}
    fd, naive, ctrl = by["Frontdoor"], by["Naiver Vergleich"], by["Kontrolle für den Mediator"]
    text = (f"Die Frontdoor schätzt {de(fd.estimate)} (Wahrheit {de(a.truth)}, Fehler {signed(fd.error)}), das Intervall {'deckt' if fd.covers else 'verfehlt'} die Wahrheit. "
            f"Der naive Vergleich liegt um {signed(naive.error)} daneben, die Kontrolle für den Mediator um {signed(ctrl.error)}.")
    s = a.settings
    notes = [("direkten Pfeil von der Regel auf den Servicegrad", s.e), ("Einfluss der Standortqualität auf den Bestand", s.f), ("zweiten Confounder", s.g), ("Messrauschen im Bestand", s.noise)]
    active = [name for name, v in notes if v > 0]
    if active:
        return "warning", text + " Eine Annahme ist verletzt: " + ", ".join(active) + "."
    if abs(fd.limit - a.truth) < 1e-9 and fd.covers:
        return "success", text
    return "info", text


def verdict_base(rows: list) -> tuple[str, str]:
    r = rows[0]["estimators"]
    return "info", (f"Im Grundfall (c = 1) ist die Frontdoor unverzerrt ({signed(r['Frontdoor']['bias'], 3)}) mit Streuung {de(r['Frontdoor']['sd'], 3)} und Deckung "
                    f"{de(100 * r['Frontdoor']['cover'], 0)} %; das Orakel, das U kennt, streut {de(r['Orakel mit U (nicht beobachtbar)']['sd'], 3)}. Der naive Vergleich liegt um "
                    f"{signed(r['Naiver Vergleich']['bias'])} daneben, die Kontrolle für den Mediator um {signed(r['Kontrolle für den Mediator']['bias'])}.")


def verdict_violations(rows: list) -> tuple[str, str]:
    by = {r["label"]: r for r in rows}
    return "info", (f"Bei Messrauschen 1,0 schätzt die Frontdoor {de(by['Messrauschen im Mediator (1,0)']['frontdoor'])} statt {de(by['Messrauschen im Mediator (1,0)']['truth'])}; "
                    f"eine direkte Wirkung von 0,3 verfehlt sie um {signed(by['direkte Wirkung X → Y (0,3)']['error'])}. Die Stärke des Confounders ändert nichts "
                    f"({signed(by['starker Confounder (U wirkt mit 2,0)']['error'])}).")


def verdict_binary(res: dict) -> tuple[str, str]:
    last = res["rows"][-1]
    first = res["rows"][0]
    return "info", (f"Wahrheit {de(res['truth'], 2)}, der naive Vergleich zeigt {de(res['observed'], 2)}. Die Frontdoor-Formel trifft im Mittel ({signed(last['estimates']['bias'], 3)} bei "
                    f"n = {last['n']}), streut aber bei n = {first['n']} mit {de(first['estimates']['sd'], 2)}; dort ist in {first['empty']} von {first['reps']} Stichproben eine Zelle leer.")


def verdict_diagrams(res: dict) -> tuple[str, str]:
    share = 100 * res["frontdoor"] / res["diagrams"]
    return "info", (f"In {res['diagrams']} Zufallsdiagrammen mit verborgener gemeinsamer Ursache gibt es per Konstruktion keine gültige Backdoor-Menge ({res['backdoor']}), aber in {res['frontdoor']} "
                    f"({de(share, 1)} %) eine Frontdoor-Menge. In allen {res['checked'] - res['wrong']} von {res['checked']} geprüften linearen Populationen stimmt die Formel mit der Wirkung.")
