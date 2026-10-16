"""Frontdoor: die Wirkung über den Mediator, wenn der Confounder fehlt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Achtes Stück der Linie Kausale Diagramme und Adjustierung. Der Confounder ist verborgen, die Hintertür bleibt offen; läuft die Wirkung ganz über einen beobachteten Mediator,
liefert das Produkt zweier Regressionen die Gesamtwirkung. Kern: frd_estimators.py, frd_criterion.py; Diagramme: dsep_graph.py und bdr_backdoor.py (wortgleich aus den Vorgängern).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import frd_constants as C
from frd_evaluation import (
    Settings,
    analyse,
    base_experiment,
    binary_experiment,
    de,
    diagram_experiment,
    signed,
    verdict_base,
    verdict_binary,
    verdict_diagrams,
    verdict_instance,
    verdict_violations,
    violations_experiment,
)
from frd_pdf_export import build_pdf
from frd_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from frd_visualization import (
    build_base_chart,
    build_base_table,
    build_binary_chart,
    build_binary_table,
    build_diagram,
    build_diagrams_chart,
    build_estimates_chart,
    build_estimates_table,
    build_steps_table,
    build_violations_chart,
    build_violations_table,
    style_estimates_table,
)

st.set_page_config(page_title="Frontdoor – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _base():
    return base_experiment()


@st.cache_data(show_spinner=False)
def _violations():
    return violations_experiment()


@st.cache_data(show_spinner=SPINNER)
def _binary():
    return binary_experiment()


@st.cache_data(show_spinner=SPINNER)
def _diagrams():
    return diagram_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🚪 Frontdoor: die Wirkung über den Mediator")
st.markdown(
    """
**Wo die Hintertür verschlossen bleibt, hilft die Vordertür.** Dies ist das achte Stück der Linie **Kausale Diagramme und Adjustierung**. Das dritte Stück hat gezeigt, dass es bei einer
verborgenen gemeinsamen Ursache keine gültige Menge zum Festhalten gibt; das vierte, dass der Mediator als Kontrolle schadet. Hier ist derselbe Mediator der Schlüssel: Läuft die
Wirkung der Regel **ganz** über den Bestand, schätzt man in zwei Schritten erst, was die Regel mit dem Bestand macht, dann, was der Bestand bei festem x mit dem Servicegrad macht. Das
**Produkt** ist die Gesamtwirkung, ohne dass man den verborgenen Confounder kennt.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Vordertür", expanded=True):
    st.markdown(
        """
1. **Die Struktur.** Die Standortqualität U (verborgen) bewegt die Regelwahl X und den Servicegrad Y. Die Regel wirkt nur über den Bestand M auf Y: kein direkter Pfeil X → Y.
2. **Erster Schritt.** Regression von M auf X: Es gibt keinen offenen Rückweg von X zu M (der Weg X ← U → Y ← M ist durch den Collider Y blockiert), also ist ĉ ohne Verzerrung.
3. **Zweiter Schritt.** Regression von Y auf M **und X**: Festhalten von X blockiert den Rückweg M ← X ← U → Y, also ist d̂ ohne Verzerrung.
4. **Das Produkt ĉ · d̂** ist die Gesamtwirkung. Die Kontrolle für M allein dagegen sperrt den Weg, den man messen will (Stück 4).
5. **Nicht testbar.** Keine dieser Annahmen lässt sich aus den Daten prüfen; das Experiment zeigt, was bei Verletzung passiert.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Struktur (Annahmen erfüllt)**")
    a_ = st.slider("Stärke des Confounders", *bounds("a_slider"), step=C.A_STEP, key="a_slider", help="Wie stark die verborgene Standortqualität U die Regelwahl und den Servicegrad bewegt (beide gleich stark).")
    c_ = st.slider("Erster Schritt c (Regel → Bestand)", *bounds("c_slider"), step=C.C_STEP, key="c_slider", help="Wirkung der Regel auf den Bestand.")
    d_ = st.slider("Zweiter Schritt d (Bestand → Servicegrad)", *bounds("d_slider"), step=C.D_STEP, key="d_slider", help="Wirkung des Bestands auf den Servicegrad. Die Gesamtwirkung der Regel ist c · d.")
    st.markdown("**Verletzte Annahmen**")
    e_ = st.slider("Direkte Wirkung Regel → Servicegrad", *bounds("e_slider"), step=C.E_STEP, key="e_slider", help="Ein Pfeil an der Vordertür vorbei: Die Frontdoor sieht ihn nicht.")
    f_ = st.slider("Standortqualität wirkt auf den Bestand", *bounds("f_slider"), step=C.F_STEP, key="f_slider", help="Der verborgene Confounder erreicht den Mediator: Der erste Schritt ist verzerrt.")
    g_ = st.slider("Zweiter Confounder (Bestand und Servicegrad)", *bounds("g_slider"), step=C.G_STEP, key="g_slider", help="Eine zweite verborgene Ursache für Bestand und Servicegrad: Der zweite Schritt ist verzerrt.")
    noise = st.slider("Messrauschen im Bestand", *bounds("noise_slider"), step=C.NOISE_STEP, key="noise_slider", help="Standardabweichung des Messfehlers im Mediator (die Streuung des Bestands ohne Fehler ist etwa 1,7).")
    st.markdown("**Daten**")
    n = st.slider("Beobachtungen n", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="Stichprobengröße der simulierten Daten.")
    seed = st.number_input("Zufalls-Seed der Stichprobe", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Stichprobe würfeln", width="stretch", on_click=randomize_seed)

values = {"a_slider": a_, "c_slider": c_, "d_slider": d_, "e_slider": e_, "f_slider": f_, "g_slider": g_, "noise_slider": noise}
sync_query_params({**{k: round(float(v), 1) for k, v in values.items()}, "n_slider": int(n), "seed_input": int(seed)})

settings = Settings(*(round(float(v), 1) for v in values.values()), int(n), int(seed))
a = _analysis(settings)
by = {r.name: r for r in a.rows}
fd = by["Frontdoor"]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
(c1, c2), (c3, c4) = st.columns(2), st.columns(2)
c1.metric("Wahre Gesamtwirkung", de(a.truth, 2), delta="aus dem Modell, nicht aus den Daten", delta_color="off")
c2.metric("Naiver Vergleich", signed(by["Naiver Vergleich"].estimate, 3), delta=f"Fehler {signed(by['Naiver Vergleich'].error, 2)}", delta_color="off",
          help="Koeffizient von x in der Regression von y auf x, ohne Adjustierung.")
c3.metric("Frontdoor", signed(fd.estimate, 3), delta=f"Fehler {signed(fd.error, 2)}, Intervall {de(fd.lo, 2)} bis {de(fd.hi, 2)}", delta_color="off",
          help="Produkt ĉ · d̂ der zwei Regressionen; Intervall nach der Delta-Methode.")
c4.metric("Kontrolle für den Mediator", signed(by["Kontrolle für den Mediator"].estimate, 3), delta=f"Fehler {signed(by['Kontrolle für den Mediator'].error, 2)}", delta_color="off",
          help="Die schlechte Kontrolle aus Stück 4: Festhalten des Mediators sperrt den Weg, den man messen will.")

left, right = st.columns([1, 1.2])
with left:
    st.plotly_chart(build_diagram(settings), width="stretch", key="diagram_chart")
    st.caption("Grün: der Weg über den Mediator. Grau: verborgene Knoten und der Confounder-Pfad. Rot: verletzte Annahmen (nur wenn der Regler über 0 liegt).")
with right:
    st.plotly_chart(build_estimates_chart(a), width="stretch", key="estimates_chart")
    st.caption("Punkt mit Intervall: grün, wenn es die Wahrheit deckt, rot, wenn nicht. Raute: Grenzwert bei unendlich vielen Beobachtungen. Das Orakel kennt U und ist in der Praxis nicht möglich.")
verdict(verdict_instance(a))
st.dataframe(style_estimates_table(build_estimates_table(a)), hide_index=True, width="stretch")
st.markdown("**Die zwei Schritte der Frontdoor**")
st.dataframe(build_steps_table(a), hide_index=True, width="stretch")

if st.button("📄 PDF der aktuellen Auswertung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="frontdoor.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Die Vordertür gegen Naiv, Kontrolle und Orakel")
res1 = _base()
st.caption(f"Alle Annahmen erfüllt, n = {C.EXP_N}, {C.EXP_REPS} feste Wiederholungen je Stärke c des ersten Schritts (c = 1 ist der Grundfall); die Wahrheit ist c · d. Das Orakel kennt U.")
st.plotly_chart(build_base_chart(res1), width="stretch", key="base_chart")
verdict(verdict_base(res1))
st.dataframe(build_base_table(res1), hide_index=True, width="stretch")
st.caption("Die Kontrolle für den Mediator trifft bei c = 0,5 zufällig (Verzerrung 0,00): Ihre Verzerrung ist 0,5 − c. Das ist eine Koinzidenz, kein Beleg.")

st.subheader("🔬 Experiment 2: Wenn eine Annahme verletzt ist")
res2 = _violations()
st.caption("Exakte Grenzwerte (unendlich viele Beobachtungen), keine Stichprobe. Die Wahrheit schließt die Verletzung ein: Gesamtwirkung von x auf y.")
st.plotly_chart(build_violations_chart(res2), width="stretch", key="violations_chart")
verdict(verdict_violations(res2))
st.dataframe(build_violations_table(res2), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 3: Die Formel mit Zellen (binäres Beispiel)")
res3 = _binary()
st.caption(f"Standortqualität, Regel, Bestand und Servicegrad sind 0/1. P(Y = 1 | do(X = 1)) = {de(res3['do1'], 2)} und P(Y = 1 | do(X = 0)) = {de(res3['do0'], 2)}: Effekt {de(res3['truth'], 2)}. "
           f"{C.EXP_BINARY_REPS} feste Wiederholungen je n.")
st.plotly_chart(build_binary_chart(res3), width="stretch", key="binary_chart")
verdict(verdict_binary(res3))
st.dataframe(build_binary_table(res3), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Wie oft gibt es eine Vordertür?")
res4 = _diagrams()
st.caption(f"{res4['diagrams']} Zufallsdiagramme mit {res4['nodes']} Knoten, in denen ein verborgener Knoten die Regel und das Ergebnis gemeinsam verursacht und es mindestens einen kausalen Pfad "
           "gibt; alle anderen Knoten sind beobachtet. Das Kriterium: die Menge fängt alle kausalen Pfade ab, kein offener Rückweg von x zur Menge, alle Rückwege von der Menge zu y sind durch x blockiert.")
st.plotly_chart(build_diagrams_chart(res4), width="stretch", key="diagrams_chart")
verdict(verdict_diagrams(res4))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Kein direkter Pfeil von der Regel auf das Ergebnis** | Die Frontdoor schätzt nur den Weg über den Mediator und verfehlt den direkten Teil (Experiment 2). | Wann geht es gar nicht? |
| **Der Confounder erreicht den Mediator nicht** | Der erste Schritt ist verzerrt, die Frontdoor ebenfalls (Experiment 2). | Wann geht es gar nicht? |
| **Kein zweiter Confounder zwischen Mediator und Ergebnis** | Der Rückweg vom Mediator zum Ergebnis ist nicht blockiert (Experiment 2). | Sensitivität und Falsifikation |
| **Der Mediator ist genau gemessen** | Messrauschen zieht den zweiten Schritt zum Nullpunkt: bei Rauschen 1,0 halbiert sich die Schätzung (Experiment 2). | Sensitivität und Falsifikation |
| **Die Struktur ist bekannt** | Keine der Annahmen lässt sich aus den Daten prüfen; ein falsches Diagramm gibt eine sichere, aber falsche Zahl. | Wie falsch darf das Diagramm sein? |
| **Es gibt überhaupt eine Vordertür** | In nur etwa jedem zehnten Zufallsdiagramm mit verborgener Ursache existiert eine gültige Menge (Experiment 4). | Instrumente |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Frontdoor-Kriterium** (Pearl 1995). Eine Menge $Z$ erfüllt das Kriterium für $(X, Y)$, wenn (i) $Z$ alle gerichteten Pfade von $X$ nach $Y$ abfängt, (ii) es keinen offenen Rückweg von $X$ zu
$Z$ gibt und (iii) alle Rückwege von $Z$ nach $Y$ durch $X$ blockiert sind.

**Frontdoor-Formel.**
$$P(y \mid \mathrm{do}(x)) = \sum_{z} P(z \mid x) \sum_{x'} P(y \mid z, x')\,P(x').$$

**Lineare Fassung.** Mit $\hat c$ aus $M \sim X$ und $\hat d$ aus $Y \sim M + X$ ist die Gesamtwirkung $\hat c\,\hat d$. Delta-Methode:
$\widehat{\mathrm{Var}}(\hat c \hat d) \approx \hat d^2\,\widehat{\mathrm{Var}}(\hat c) + \hat c^2\,\widehat{\mathrm{Var}}(\hat d)$. Bei mehreren Mediatoren: $\sum_k \hat c_k \hat\theta_k$ mit $\hat c_k$ aus
$M_k \sim X$ und $\hat\theta$ aus $Y \sim M + X$.

**Verzerrung ohne die Annahmen.** Mit dem Messfehler $\sigma$ im Mediator wird $\hat d$ um den Faktor $\mathrm{Var}(M \mid X)/(\mathrm{Var}(M \mid X) + \sigma^2)$ gestaucht; mit einer direkten Wirkung $e$ fehlt $e$ in
der Schätzung ($\hat c \hat d = cd$ statt $cd + e$).

**Literatur.** J. Pearl, *Causal diagrams for empirical research*, Biometrika 82, 1995; J. Pearl, *Causality*, 2. Auflage, 2009; I. Shpitser, J. Pearl, *Identification of joint interventional
distributions in recursive semi-Markovian causal models*, AAAI 2006 (alle aus dem Gedächtnis, nicht geprüft).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
