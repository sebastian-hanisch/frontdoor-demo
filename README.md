# Frontdoor: die Wirkung über den Mediator – Streamlit-Demo

Achtes Stück der Linie **Kausale Diagramme und Adjustierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Nachfolger von [doppelt-robust-demo](https://github.com/sebastian-hanisch/doppelt-robust-demo): Dort waren alle Störfaktoren beobachtet. Hier ist die gemeinsame Ursache von Regel und Ergebnis **verborgen**, und die Wirkung wird trotzdem bestimmt, weil sie ganz über einen beobachteten Mediator läuft.

## Kernfrage

Wenn ein Confounder nicht zu messen ist, scheitert jede Backdoor-Adjustierung. Läuft die Wirkung aber vollständig über einen beobachteten Mediator, den der Confounder nicht erreicht, schätzt man sie in zwei Schritten: erst die Wirkung der Regel auf den Mediator, dann die des Mediators auf das Ergebnis bei festem x. Was trägt das, wie genau ist es, und wie schnell geht es kaputt, wenn eine der Annahmen nicht stimmt?

## Modell

- **Vehikel:** ein Standort wählt die neue Regel B (X) abhängig von der verborgenen Standortqualität U; die Regel wirkt über den Bestand (M, beobachtet) auf den Servicegrad (Y). Linear-gaußsches Strukturmodell mit Einheitsrauschen in jedem Knoten: U → X, U → Y (Stärke a), X → M (c), M → Y (d). Die wahre Gesamtwirkung ist **c · d** (Grundfall 1,0).
- **Vier Verletzungen** als Regler: ein direkter Pfeil X → Y (e), U → M (f), ein zweiter verborgener Confounder U₂ → M und U₂ → Y (g), Messrauschen im Mediator (Standardabweichung σ). Die Wahrheit ist stets die Gesamtwirkung des Modells, einschließlich der Verletzung.
- **Binäres Beispiel:** vier 0/1-Größen mit verborgener Ursache (Wirkung 0,40, naiver Vergleich 0,58) für die allgemeine Formel.
- **Daten:** `SplitMix64`-Ströme (Generator `kausal_generator.py`, wortgleich in allen Stücken der Linie).

## Methodik

- **Schätzer** (`frd_estimators.py`): naiver Vergleich; Kontrolle für den Mediator (Y ~ X + M, der Koeffizient von X); **Frontdoor** (ĉ aus M ~ X mal d̂ aus Y ~ M + X); das Orakel Y ~ X + U (U ist nicht beobachtbar, nur als Maßstab).
- **Intervalle:** Delta-Methode, d̂² Var(ĉ) + ĉ² Var(d̂); kein Bootstrap. Die Deckung wird in Experiment 1 gemessen.
- **Kriterium** (`frd_criterion.py`): die drei Bedingungen der Frontdoor, gerechnet über `dsep_graph.py` und `bdr_backdoor.py` (wortgleich aus Stück 2 und 3), und eine Suche über alle Mengen beobachteter Knoten.
- **Experimente** laufen über feste Zufallsströme; die Verletzungen (Experiment 2) sind exakte Grenzwerte aus der Population, keine Stichproben.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Wahrheit 1,00 im Grundfall, Verzerrung = Mittel minus Wahrheit.

| Frage | Befund | Test |
|---|---|---|
| Standard-Stichprobe (n = 500, Seed 35) | Wahr 1,00; naiv **1,54**; Kontrolle für den Mediator **0,52**; Frontdoor **1,02** (deckt); das Orakel mit U 1,11. Schritte: ĉ 1,05 (Standardfehler 0,032), d̂ 0,97 (0,057). | `test_default_sample` |
| Trifft die Frontdoor im Grundfall? (n = 500, 500 Wiederholungen) | Frontdoor: Verzerrung **−0,001**, Streuung **0,065**, Deckung **95 %**. Das Orakel, das U kennt: −0,003, Streuung 0,066, Deckung 95 %. Naiv **+0,50**, Kontrolle **−0,50** (Deckung beide 0 %). | `test_experiment_1_base_case` |
| Und bei schwachem ersten Schritt? (c = 0,5 / 0,2 / 0,1 / 0,05) | Frontdoor bleibt unverzerrt (+0,001, −0,002, +0,000, +0,001), Deckung 94 bis 96 %, die Streuung sinkt auf 0,044 / 0,034 / 0,033 / 0,031 (das Orakel streut 0,060 bis 0,067). Aber das Intervall ist im Verhältnis zur Wirkung breit: Breite **0,125** bei Wirkung 0,05. Die Kontrolle für den Mediator trifft bei c = 0,5 **zufällig** (−0,004), bei c = 0,2 steht sie bei **+0,30**, bei 0,1 bei +0,40. | `test_experiment_1_weak_first_step` |
| Was kostet jede Verletzung? (exakte Grenzwerte) | Direkte Wirkung 0,3: Fehler **−0,30** (die Frontdoor sieht nur den Weg über M). U → M mit 0,3: **+0,32**. Zweiter Confounder 0,5: **+0,20** (= g² / (1 + g²)). Messrauschen 0,5: **−0,20**, Rauschen 1,0: **−0,50** (Schätzung 0,50 statt 1,00; Faktor 1 / (1 + σ²)). Ein starker Confounder (a = 2) ändert nichts (0,00), der naive Vergleich geht auf 1,80. | `test_experiment_2_violations` |
| Allgemeine Formel im binären Beispiel (n = 100 / 500 / 2 000, 500 Wiederholungen) | Wahr 0,40, naiv 0,58. Die Formel trifft im Mittel (+0,002 / +0,001 / +0,001), streut aber **0,142 / 0,054 / 0,028**; bei n = 100 ist in **6 von 500** Stichproben eine Zelle leer. | `test_experiment_3_binary` |
| Wie oft gibt es eine Vordertür? (3 000 Zufallsdiagramme, sieben Knoten, verborgene Ursache von x und y) | Nie eine gültige Backdoor-Menge (**0**), aber in **312** (10,4 %) eine Frontdoor-Menge: 265-mal mit einem, 43-mal mit zwei und 4-mal mit drei Mediatoren. In allen 312 linearen Populationen stimmt die Formel mit der Wirkung (Fehler unter 10⁻⁸). | `test_experiment_4_diagrams` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Kein direkter Pfeil von der Regel auf das Ergebnis** | Die Frontdoor schätzt nur den Weg über den Mediator und verfehlt den direkten Teil (Experiment 2). | Wann geht es gar nicht? |
| **Der Confounder erreicht den Mediator nicht** | Der erste Schritt ist verzerrt, die Frontdoor ebenfalls (Experiment 2). | Wann geht es gar nicht? |
| **Kein zweiter Confounder zwischen Mediator und Ergebnis** | Der Rückweg vom Mediator zum Ergebnis ist nicht blockiert (Experiment 2). | Sensitivität und Falsifikation |
| **Der Mediator ist genau gemessen** | Messrauschen zieht den zweiten Schritt zum Nullpunkt: bei Rauschen 1,0 halbiert sich die Schätzung (Experiment 2). | Sensitivität und Falsifikation |
| **Die Struktur ist bekannt** | Keine der Annahmen lässt sich aus den Daten prüfen; ein falsches Diagramm gibt eine sichere, aber falsche Zahl. | Wie falsch darf das Diagramm sein? |
| **Es gibt überhaupt eine Vordertür** | In nur etwa jedem zehnten Zufallsdiagramm mit verborgener Ursache existiert eine gültige Menge (Experiment 4). | Instrumente |

Die Zahlen gelten für dieses erfundene Vehikel (linear-gaußsch, ein Mediator, Einheitsrauschen); die Frontdoor-Formel selbst gilt allgemein, ihre linearen Zahlen hier nicht. „Wahrheit“ ist das simulierte Modell, nicht ein Befund über echte Standorte. Die Quellen im 📐-Expander sind aus dem Gedächtnis, nicht geprüft.

## Bezug zu den Nachbarn im Portfolio

- [backdoor-demo](https://github.com/sebastian-hanisch/backdoor-demo) und [adjustierungsmenge-demo](https://github.com/sebastian-hanisch/adjustierungsmenge-demo): Adjustierung über beobachtete Mengen. Hier ist der Confounder verborgen; die Backdoor-Suche findet in den Zufallsdiagrammen nie eine Menge.
- [schlechte-kontrollen-demo](https://github.com/sebastian-hanisch/schlechte-kontrollen-demo): Dort verzerrt die Kontrolle für einen Collider oder Mediator. Hier ist die Kontrolle für den Mediator der naive Fehlgriff, der den Weg zur Wirkung wegrechnet (Fehler −0,50 im Grundfall).
- [konfundierung-demo](https://github.com/sebastian-hanisch/konfundierung-demo): der verborgene Confounder und sein naiver Vergleich; hier wird er umgangen, nicht gemessen.
- [off-policy-demo](https://github.com/sebastian-hanisch/off-policy-demo): Gewichten mit bekannter Protokoll-Wahrscheinlichkeit; hier ist keine Zuteilung zu modellieren.
- **Abgrenzung zur Mediationsanalyse:** Die Frontdoor schätzt die **Gesamtwirkung** unter voller Vermittlung (kein direkter Pfeil); direkte und indirekte Wirkung (NDE, NIE) und die sequentielle Ignorierbarkeit sind ein anderes Thema und nicht gebaut.

## Tests

Mini-Instanzen von Hand (Varianzen 2 und 3, naiv 1,5, Kontrolle 0,5, Frontdoor 1; acht binäre Einheiten mit 11/16 und 13/24), das Kriterium an Hand-Diagrammen und gegen ein zweites Verfahren (networkx, 3 000 Mengen in 200 Zufallsdiagrammen), Orakel für die Formeln der Verletzungen, Delta-Intervalle gegen die empirische Streuung, Auswertung und Verdikte, jede README-Zahl in `test_claims.py`, AppTest (Presets, Regler, Permalink, PDF), Footer und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `kausal_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Normalverteilung, Strukturmodell) |
| `dsep_graph.py`, `bdr_backdoor.py` | d-Separation und Backdoor/Adjustierungskriterium (wortgleich aus Stück 2 und 3) |
| `frd_world.py` | Vehikel: Strukturmodell, Wahrheit, Population, Stichprobe, binäres Beispiel |
| `frd_estimators.py` | naiv, Kontrolle, Orakel, Frontdoor mit Delta-Intervall, binäre Formel |
| `frd_criterion.py` | Frontdoor-Kriterium und Suche nach gültigen Mengen |
| `frd_evaluation.py` | Einstellungen, Analyse, vier Experimente, Verdikte |
| `frd_visualization.py`, `frd_pdf_export.py` | Figuren, Tabellen, PDF |
| `frd_constants.py`, `frd_presets.py` | Regler, Experimente, Presets, Permalink |
| `tests/` | Mini-Instanzen, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Mediationsanalyse (NDE, NIE, sequentielle Ignorierbarkeit):** eigenes Thema; die Linie führt es nicht.
- **Bootstrap-Intervalle:** die Delta-Methode genügt für das lineare Produkt; ihre Deckung ist gemessen.
- **Mehrere Mediatoren, nichtlineare Mediatoren im Regler:** das Kriterium rechnet sie (Experiment 4), der Schätzer der Demo hat einen Mediator.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
