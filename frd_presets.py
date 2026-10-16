"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio). Alle Regler sind in jeder Einstellung wirksam und werden immer
gezeichnet; es gibt weder KEPT-Regler noch berechnete Grenzen."""

import random
from dataclasses import dataclass

import streamlit as st

import frd_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float
    hi: float
    step: float


SETTING_SPECS = {
    "a_slider": SettingSpec("a", float, C.DEFAULT_A, C.A_MIN, C.A_MAX, C.A_STEP),
    "c_slider": SettingSpec("c", float, C.DEFAULT_C, C.C_MIN, C.C_MAX, C.C_STEP),
    "d_slider": SettingSpec("d", float, C.DEFAULT_D, C.D_MIN, C.D_MAX, C.D_STEP),
    "e_slider": SettingSpec("e", float, 0.0, C.E_MIN, C.E_MAX, C.E_STEP),
    "f_slider": SettingSpec("f", float, 0.0, C.F_MIN, C.F_MAX, C.F_STEP),
    "g_slider": SettingSpec("g", float, 0.0, C.G_MIN, C.G_MAX, C.G_STEP),
    "noise_slider": SettingSpec("nz", float, 0.0, C.NOISE_MIN, C.NOISE_MAX, C.NOISE_STEP),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX, C.N_STEP),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX, 1),
}
PRESET_KEYS = {"a": "a_slider", "c": "c_slider", "d": "d_slider", "e": "e_slider", "f": "f_slider", "g": "g_slider", "noise": "noise_slider", "n": "n_slider", "seed": "seed_input"}


def snap(value, spec: SettingSpec):
    """Auf das Raster des Reglers legen und in die Grenzen klemmen."""
    value = max(spec.lo, min(spec.hi, value))
    if spec.step > 0 and (spec.caster is float or spec.step > 1):
        value = round((value - spec.lo) / spec.step) * spec.step + spec.lo
        value = round(value, 6) if spec.caster is float else int(value)
    return value


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    init_session_state_defaults()
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = snap(spec.caster(float(qp[spec.url_param])), spec)
        except (ValueError, TypeError):
            continue
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values: dict):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
