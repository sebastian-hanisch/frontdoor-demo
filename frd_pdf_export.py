"""PDF-Export der aktuellen Auswertung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from frd_evaluation import Analysis, de, signed, verdict_instance

_REPLACE = {"→": "->", "←": "<-", "–": "-", "—": "-", "−": "-", "€": "EUR", "≥": ">=", "≤": "<=", "„": '"', "“": '"', "×": "x", "·": "*", "**": "", "ĉ": "c^", "d̂": "d^", "∞": "inf"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Frontdoor: die Wirkung über den Mediator"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Stärke des Confounders {de(s.a, 1)}, erster Schritt c = {de(s.c, 1)}, zweiter Schritt d = {de(s.d, 1)}; Verletzungen: direkt {de(s.e, 1)}, U -> M {de(s.f, 1)}, "
                            f"zweiter Confounder {de(s.g, 1)}, Messrauschen {de(s.noise, 1)}; n = {s.n}, Seed {s.seed}. Alles simuliert. Wahre Gesamtwirkung {de(a.truth, 2)}."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    for r in a.rows:
        pdf.multi_cell(0, 4.6, _s(f"{r.name}: {de(r.estimate, 3)} [{de(r.lo, 2)}; {de(r.hi, 2)}], Fehler {signed(r.error, 3)}, {'deckt' if r.covers else 'deckt nicht'}; Grenzwert {de(r.limit, 3)}."),
                       new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.multi_cell(0, 4.6, _s(f"Schritte der Frontdoor: c = {de(a.steps['c'], 3)} (Grenzwert {de(a.steps['c_limit'], 3)}), d = {de(a.steps['d'], 3)} (Grenzwert {de(a.steps['d_limit'], 3)})."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 4.5, _s(verdict_instance(a)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: frontdoor-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
