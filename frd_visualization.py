"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

import frd_constants as C
import frd_world as W
from frd_evaluation import ROW_NAMES, Analysis, Settings, de, signed

SHORT = {"Naiver Vergleich": "naiv", "Kontrolle für den Mediator": "Kontrolle M", "Frontdoor": "Frontdoor", "Orakel mit U (nicht beobachtbar)": "Orakel mit U"}
COLORS = {"Naiver Vergleich": C.COLOR_NEUTRAL, "Kontrolle für den Mediator": C.COLOR_CONTROL, "Frontdoor": C.COLOR_GOOD, "Orakel mit U (nicht beobachtbar)": C.COLOR_TRUTH}


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.22), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _arrow(fig, p0, p1, color, width=2, dash=None, shrink=0.11):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    ln = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / ln, dy / ln
    fig.add_annotation(x=x1 - ux * shrink, y=y1 - uy * shrink, ax=x0 + ux * shrink, ay=y0 + uy * shrink, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3,
                       arrowsize=1.3, arrowwidth=width, arrowcolor=color, text="")


def build_diagram(s: Settings) -> go.Figure:
    """Das Diagramm der Einstellung: verborgene Knoten grau, der Weg über den Mediator grün, verletzte Annahmen als rote Pfeile (nur wenn der Regler > 0 ist)."""
    pos = {W.U: (0.5, 1.05), W.U2: (0.5, -0.65), W.X: (0.0, 0.0), W.M: (0.5, 0.0), W.Y: (1.0, 0.0)}
    fig = go.Figure()
    base = [(W.U, W.X, f"{de(s.a, 1)}"), (W.U, W.Y, f"{de(s.a, 1)}")]
    for a, b, label in base:
        _arrow(fig, pos[a], pos[b], "#6b6b6b", 2)
    _arrow(fig, pos[W.X], pos[W.M], C.COLOR_GOOD, 4)
    _arrow(fig, pos[W.M], pos[W.Y], C.COLOR_GOOD, 4)
    fig.add_annotation(x=0.25, y=0.12, text=f"c = {de(s.c, 1)}", showarrow=False, font=dict(size=11, color="#555555"), bgcolor="rgba(255,255,255,0.75)")
    fig.add_annotation(x=0.75, y=0.12, text=f"d = {de(s.d, 1)}", showarrow=False, font=dict(size=11, color="#555555"), bgcolor="rgba(255,255,255,0.75)")
    bad = [(s.e, W.X, W.Y, f"direkt {de(s.e, 1)}"), (s.f, W.U, W.M, f"{de(s.f, 1)}")]
    for strength, a, b, label in bad:
        if strength > 0:
            _arrow(fig, pos[a], pos[b], C.COLOR_BAD, 3)
    if s.g > 0:
        _arrow(fig, pos[W.U2], pos[W.M], C.COLOR_BAD, 3)
        _arrow(fig, pos[W.U2], pos[W.Y], C.COLOR_BAD, 3)
    nodes = [W.U, W.X, W.M, W.Y] + ([W.U2] if s.g > 0 else [])
    names = {W.U: "Standortqualität<br>(verborgen)", W.U2: "zweite Ursache<br>(verborgen)", W.X: "Regel B<br>(X)", W.M: "Bestand<br>(M)" + ("<br>mit Rauschen" if s.noise > 0 else ""), W.Y: "Servicegrad<br>(Y)"}
    hidden = {W.U, W.U2}
    for group, is_hidden in (([v for v in nodes if v not in hidden], False), ([v for v in nodes if v in hidden], True)):
        fig.add_trace(go.Scatter(x=[pos[v][0] for v in group], y=[pos[v][1] for v in group], mode="markers+text", text=[names[v] for v in group], textposition="middle center",
                                 textfont=dict(size=10, color="#1b1b1b") if is_hidden else dict(size=10),
                                 marker=dict(size=74, color=C.COLOR_HIDDEN if is_hidden else "rgba(47,107,101,0.12)", line=dict(width=1.5 if is_hidden else 4, color="#8a8a8a" if is_hidden else "#1b1b1b")),
                                 hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False, range=[-0.3, 1.3])
    fig.update_yaxes(visible=False, range=[-1.0, 1.4])
    return _fixed(fig, 360, showlegend=False)


def build_estimates_chart(a: Analysis) -> go.Figure:
    """Schätzung mit 95-%-Intervall je Schätzer (grün: deckt die Wahrheit, rot: deckt nicht); Raute: Grenzwert bei n → ∞."""
    fig = go.Figure()
    for r in a.rows:
        color = C.COLOR_GOOD if r.covers else C.COLOR_BAD
        fig.add_trace(go.Scatter(x=[r.estimate], y=[SHORT[r.name]], mode="markers", marker=dict(color=color, size=10), showlegend=False,
                                 error_x=dict(type="data", array=[r.hi - r.estimate], arrayminus=[r.estimate - r.lo], color=color, thickness=2),
                                 hovertemplate=f"{r.name}: {de(r.estimate, 3)} [{de(r.lo, 2)}; {de(r.hi, 2)}]<extra></extra>"))
    fig.add_trace(go.Scatter(x=[r.limit for r in a.rows], y=[SHORT[r.name] for r in a.rows], mode="markers", marker=dict(color="#1b1b1b", size=8, symbol="diamond"), name="Grenzwert (n → ∞)"))
    fig.add_vline(x=a.truth, line=dict(color=C.COLOR_TRUTH, width=2), annotation_text=f"Wahrheit {de(a.truth, 2)}", annotation_position="top")
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=[SHORT[n] for n in ROW_NAMES])
    fig.update_xaxes(title="geschätzte Wirkung der Regel B")
    return _fixed(fig, 320)


def build_estimates_table(a: Analysis) -> pd.DataFrame:
    return pd.DataFrame([{"Schätzer": r.name, "Schätzung": signed(r.estimate, 3), "Fehler": signed(r.error, 3), "Standardfehler": de(r.se, 3), "95-%-Intervall": f"{de(r.lo, 2)} bis {de(r.hi, 2)}",
                          "Deckt": "ja" if r.covers else "nein", "Grenzwert": de(r.limit, 3)} for r in a.rows])


def style_estimates_table(df: pd.DataFrame):
    def row_style(row):
        return ["background-color: rgba(192,57,43,0.15)" if row["Deckt"] == "nein" else ""] * len(row)

    return df.style.apply(row_style, axis=1)


def build_steps_table(a: Analysis) -> pd.DataFrame:
    st_ = a.steps
    return pd.DataFrame([
        {"Schritt": "1: Wirkung der Regel auf den Bestand (c)", "Schätzung": de(st_["c"], 3), "Standardfehler": de(st_["se_c"], 3), "Grenzwert": de(st_["c_limit"], 3), "wahr": de(a.settings.c, 2)},
        {"Schritt": "2: Wirkung des Bestands auf den Servicegrad bei festem x (d)", "Schätzung": de(st_["d"], 3), "Standardfehler": de(st_["se_d"], 3), "Grenzwert": de(st_["d_limit"], 3),
         "wahr": de(a.settings.d, 2)},
        {"Schritt": "Produkt c · d", "Schätzung": de(st_["c"] * st_["d"], 3), "Standardfehler": de(a.rows[2].se, 3), "Grenzwert": de(st_["c_limit"] * st_["d_limit"], 3), "wahr": de(a.truth, 2)},
    ])


# ----------------------------------------------------------------------------- Experiment 1

def build_base_chart(rows: list) -> go.Figure:
    fig = go.Figure()
    labels = [f"c = {de(r['c'], 2)}" for r in rows]
    for name in ROW_NAMES:
        fig.add_trace(go.Bar(x=labels, y=[r["estimators"][name]["bias"] for r in rows], name=SHORT[name], marker_color=COLORS[name]))
    fig.add_hline(y=0.0, line=dict(color="#888888", width=1))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Verzerrung (Mittel minus Wahrheit)")
    return _fixed(fig, 400)


def build_base_table(rows: list) -> pd.DataFrame:
    out = []
    for r in rows:
        for name in ROW_NAMES:
            v = r["estimators"][name]
            out.append({"c": de(r["c"], 2), "Wahrheit": de(r["truth"], 2), "Schätzer": name, "Verzerrung": signed(v["bias"], 3), "Streuung": de(v["sd"], 3), "Deckung": f"{de(100 * v['cover'], 0)} %",
                        "Breite des Intervalls": de(v["width"], 2)})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- Experiment 2

def build_violations_chart(rows: list) -> go.Figure:
    labels = [r["label"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=[r["control"] - r["truth"] for r in rows], name="Kontrolle für den Mediator", marker_color=C.COLOR_CONTROL, orientation="h"))
    fig.add_trace(go.Bar(y=labels, x=[r["error"] for r in rows], name="Frontdoor", marker_color=C.COLOR_GOOD, orientation="h"))
    fig.add_vline(x=0.0, line=dict(color="#888888", width=1))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Fehler im Grenzfall (Schätzung minus Gesamtwirkung)")
    fig.update_yaxes(autorange="reversed")
    return _fixed(fig, 120 + 56 * len(rows))


def build_violations_table(rows: list) -> pd.DataFrame:
    return pd.DataFrame([{"Fall": r["label"], "Wahrheit": de(r["truth"], 2), "naiv": de(r["naive"], 2), "Kontrolle M": de(r["control"], 2), "Frontdoor": de(r["frontdoor"], 2),
                          "Fehler der Frontdoor": signed(r["error"], 2)} for r in rows])


# ----------------------------------------------------------------------------- Experiment 3

def build_binary_table(res: dict) -> pd.DataFrame:
    out = []
    for r in res["rows"]:
        e = r["estimates"]
        out.append({"n": r["n"], "naiver Vergleich": de(r["naive"], 3), "Frontdoor (Mittel)": de(e["mean"], 3), "Verzerrung": signed(e["bias"], 3), "Streuung": de(e["sd"], 3),
                    "Stichproben mit leerer Zelle": f"{r['empty']} von {r['reps']}"})
    return pd.DataFrame(out)


def build_binary_chart(res: dict) -> go.Figure:
    ns = [str(r["n"]) for r in res["rows"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ns, y=[r["estimates"]["mean"] for r in res["rows"]], name="Frontdoor (Mittel)", marker_color=C.COLOR_GOOD, error_y=dict(type="data", array=[r["estimates"]["sd"] for r in res["rows"]],
                                                                                                                                                    visible=True)))
    fig.add_trace(go.Bar(x=ns, y=[r["naive"] for r in res["rows"]], name="naiver Vergleich", marker_color=C.COLOR_NEUTRAL))
    fig.add_hline(y=res["truth"], line=dict(color=C.COLOR_TRUTH, width=2), annotation_text=f"Wahrheit {de(res['truth'], 2)}", annotation_position="top left")
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Beobachtungen n", type="category")
    fig.update_yaxes(title="Effekt (Differenz der Wahrscheinlichkeiten)", rangemode="tozero")
    return _fixed(fig, 360)


# ----------------------------------------------------------------------------- Experiment 4

def build_diagrams_chart(res: dict) -> go.Figure:
    sizes = res["sizes"]
    labels = ["gültige Backdoor-Menge", "gültige Frontdoor-Menge", "Frontdoor: kleinste Menge hat 1 Knoten", "Frontdoor: kleinste Menge hat 2 Knoten", "Frontdoor: kleinste Menge hat 3 Knoten"]
    vals = [res["backdoor"], res["frontdoor"], sizes.count(1), sizes.count(2), sizes.count(3)]
    fig = go.Figure(go.Bar(x=labels, y=[100 * v / res["diagrams"] for v in vals], marker_color=[C.COLOR_BAD, C.COLOR_GOOD, C.COLOR_CONTROL, C.COLOR_CONTROL, C.COLOR_CONTROL],
                           text=[de(100 * v / res["diagrams"], 1) + " %" for v in vals], textposition="outside"))
    fig.update_yaxes(title="Anteil der Diagramme (%)", range=[0, 14])
    fig.update_xaxes(tickangle=-12)
    return _fixed(fig, 380, showlegend=False)
