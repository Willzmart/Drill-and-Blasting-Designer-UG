"""Diseño de mallas de perforación y voladura subterránea — avance, chimeneas, piques y producción.

Fuentes: Carpio Chávez (2025), López Jimeno et al. (IGME, 1987), Konya (Manual de voladura), material del
curso de taladros largos (Llontop; Salas), Guía del explosivista de Enaex y papers de voladura en anillos.
Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import energy as en
from core import fragmentation as fr
from core import report
from core import sequencing as sq
from core import viz
from core import viz_energy as ve
from core.avance import CHARGE_METHODS, DESIGNS, AvanceOptions, Drill, run_avance
from core.common import (CARTRIDGES, EXPLOSIVES, EXSANEL_LP, EXSANEL_MS, FANEL_MS, KONYA_EQS, KONYA_KD, KONYA_KR,
                         KONYA_KS, P80_REFERENCIAS, Rock, make_explosive, ms_por_metro)
from core.cuts import CUT_NOTES, CUT_TYPES
from core.domains import EDGE_MODES, GRADE_UNITS, DomainOptions
from core.drilling import DrillSetup
from core.slot import SLOT_BLAST, SLOT_CUTS, SLOT_DIR, SlotOptions, run_slot

SLOT_METHOD = "Chimenea slot con Simba (taladros largos)"
from core.produccion import CE0_TABLE, CRITERIA, DEFAULTS, FAN_STEMMING, PROD_METHODS, ProdOptions, run_produccion
from core.sections import GALLERY_TYPES, PRESETS, VERTICAL_TYPES, build_section, build_vertical_section
from core.vertical import CHIMENEA_METHODS, PIQUE_METHODS, VerticalOptions, run_vertical

st.set_page_config(page_title="Diseño de voladura subterránea", layout="wide")
ASSETS = Path(__file__).parent / "assets"
ss = st.session_state
ss.setdefault("overrides", {})
ss.setdefault("seq", None)
ss.setdefault("seq_sig", None)
ss.setdefault("emem", {})

# =============================================================================
# BARRA LATERAL
# =============================================================================
st.sidebar.title("Datos de entrada")
labor = st.sidebar.radio("Tipo de labor", ["Avance", "Chimenea", "Pique", "Producción"], horizontal=True)


def explosive_picker(label, default, cart_default, key, allow_bulk=True):
    names = [n for n in EXPLOSIVES if allow_bulk or not EXPLOSIVES[n]["granel"]] + ["Personalizado"]
    name = st.selectbox(label, names, index=names.index(default), key=key + "n")
    if name == "Personalizado":
        c1, c2 = st.columns(2)
        custom = dict(name="Personalizado", vod=c1.number_input("VOD (m/s)", 1500, 8000, 5000, key=key + "v"),
                      rho=c2.number_input("Densidad (g/cm³)", 0.6, 1.6, 1.10, key=key + "r"),
                      rws=c1.number_input("RWS (% ANFO)", 40, 180, 90, key=key + "w") / 100,
                      E_MJkg=c2.number_input("Energía (MJ/kg; 0 = RWS·3.82)", 0.0, 8.0, 0.0, 0.05, key=key + "e"))
        bulk = c2.checkbox("A granel", False, key=key + "g")
        if bulk:
            custom.update(cart="granel", w_cart=0.0, L_cart=1.0, granel=True)
        else:
            custom.update(cart="—", w_cart=c1.number_input("Peso cartucho (kg)", 0.02, 2.0, 0.25, key=key + "c"),
                          L_cart=c2.number_input("Long. cartucho (m)", 0.1, 0.8, 0.3048, key=key + "l"), granel=False)
        return make_explosive("Personalizado", custom=custom)
    if EXPLOSIVES[name]["granel"]:
        ex = make_explosive(name)
    else:
        cart = st.selectbox("Cartucho", list(CARTRIDGES), index=list(CARTRIDGES).index(cart_default), key=key + "k")
        ex = make_explosive(name, cart)
    st.caption(f"ρ = {ex.rho:.2f} g/cm³ · VOD = {ex.vod:.0f} m/s · RWS = {ex.rws * 100:.0f} % · "
               f"energía ≈ {ex.energy:.2f} MJ/kg")
    return ex


ROCK_MODES = ["Un solo dominio litológico", "Por dominios: mineral, caja techo y caja piso"]


def rock_domain_inputs(prefix, d):
    """Propiedades geomecánicas completas de un dominio (caja techo o caja piso)."""
    c1, c2 = st.columns(2)
    rmr = c1.number_input("RMR", 5, 100, d["RMR"], key=prefix + "rmr")
    gsi = c2.number_input("GSI", 5, 100, d["GSI"], key=prefix + "gsi")
    rqd = c1.number_input("RQD (%)", 1, 100, d["RQD"], key=prefix + "rqd")
    ucs = c2.number_input("σci (MPa)", 5.0, 350.0, d["UCS"], 5.0, key=prefix + "ucs")
    rho = c1.number_input("Densidad (t/m³)", 1.8, 4.5, d["rho"], 0.01, key=prefix + "rho")
    q = c2.number_input("Q de Barton", 0.001, 1000.0, d["Q"], key=prefix + "q")
    e = c1.number_input("E (GPa)", 1.0, 120.0, d["E"], key=prefix + "e")
    stt = c2.number_input("σt tracción (MPa)", 0.3, 40.0, round(ucs / 10, 1), 0.1, key=prefix + "st")
    vp = c1.number_input("Vp (m/s; 0 = con Q)", 0.0, 7000.0, 0.0, 50.0, key=prefix + "vp")
    jsf = 1.0 if rmr > 60 else 0.9 if rmr > 40 else 0.8 if rmr > 20 else 0.7
    r = Rock(RMR=rmr, GSI=gsi, RQD=rqd, UCS=ucs, rho=rho, Q=q, JSF=jsf, E=e, sigma_t=stt, Vp_user=vp or None)
    c2.metric("VPP crítica", f"{r.ppv_crit:,.0f} mm/s")
    return r


with st.sidebar.expander("Macizo rocoso", expanded=False):
    rock_mode = st.radio("Modelo del macizo", ROCK_MODES, key="rock_mode",
                         help="Un solo dominio: todo el diseño usa las mismas propiedades (cuerpos masivos, irregulares "
                              "o sin datos de las cajas). Por dominios: el mineral, la caja techo y la caja piso tienen "
                              "propiedades propias y cada método usa la que le corresponde (cuerpos tabulares: vetas, "
                              "mantos).")
    multi = rock_mode == ROCK_MODES[1]
    if multi:
        st.markdown("**Mineral**")
    c1, c2 = st.columns(2)
    RMR = c1.number_input("RMR", 5, 100, 50)
    GSI = c2.number_input("GSI", 5, 100, 45)
    RQD = c1.number_input("RQD (%)", 1, 100, 55)
    UCS = c2.number_input("σci (MPa)", 5.0, 350.0, 75.0, 5.0)
    rho_r = c1.number_input("Densidad (t/m³)", 1.8, 4.5, 2.75, 0.01)
    Qb = c2.number_input("Q de Barton", 0.001, 1000.0, 1.95)
    Emod = c1.number_input("E (GPa)", 1.0, 120.0, 35.0)
    jsf_auto = 1.0 if RMR > 60 else 0.9 if RMR > 40 else 0.8 if RMR > 20 else 0.7
    JSF = c2.number_input("JSF", 0.5, 1.0, jsf_auto, 0.05, help="Carpio Tabla 01: 1.0 / 0.9 / 0.8 / 0.7")
    sig_t = c1.number_input("σt tracción (MPa)", 0.5, 40.0, round(UCS / 10, 1), 0.5,
                            help="Por defecto σci/10. Se usa en Torbica y Lapcevic, Pearse del curso y la VPP crítica.")
    nu = c2.number_input("Poisson ν", 0.05, 0.45, 0.25, 0.01)
    vp_m = st.number_input("Vp medida (m/s; 0 = estimar con Q)", 0.0, 7000.0, 0.0, 50.0)
    ore = Rock(RMR=RMR, GSI=GSI, RQD=RQD, UCS=UCS, rho=rho_r, Q=Qb, JSF=JSF, E=Emod, sigma_t=sig_t, nu=nu,
               Vp_user=vp_m or None)
    hw_rock = fw_rock = None
    rock_where = "Mineral"
    if multi:
        st.caption(f"VPP crítica del mineral: {ore.ppv_crit:,.0f} mm/s")
        st.markdown("**Caja techo**")
        hw_rock = rock_domain_inputs("hw_", dict(RMR=40, GSI=35, RQD=40, UCS=45.0, rho=2.70, Q=0.6, E=20.0))
        st.markdown("**Caja piso**")
        fw_rock = rock_domain_inputs("fw_", dict(RMR=55, GSI=50, RQD=65, UCS=80.0, rho=2.70, Q=3.0, E=35.0))
        if labor != "Producción":
            rock_where = st.selectbox("La labor se excava en", ["Mineral", "Caja techo", "Caja piso"], key="rock_where",
                                      help="Avances, chimeneas y piques usan las propiedades del dominio donde se "
                                           "excavan (p. ej., una rampa o un by-pass en la caja piso).")
        else:
            st.caption("Producción: el burden y la carga se diseñan con el mineral; las cajas se usan para la "
                       "sobrerotura, la dilución y la estabilidad (pilares, techo).")
rock = {"Mineral": ore, "Caja techo": hw_rock, "Caja piso": fw_rock}[rock_where] or ore

with st.sidebar.expander("Iniciación", expanded=False):
    system = st.selectbox("Sistema de iniciación", sq.SYSTEMS, index=2 if labor != "Producción" else 4)
    primer = st.radio("Posición del cebo", ["Fondo", "Boca"], horizontal=True)
with st.sidebar.expander("Costos unitarios", expanded=False):
    cost = dict(drill=st.number_input("Perforación (US$/m)", 0.0, 100.0, 2.5, 0.1),
                exp=st.number_input("Explosivo (US$/kg)", 0.0, 50.0, 4.0, 0.1),
                acc=st.number_input("Accesorios por taladro (US$)", 0.0, 100.0, 3.0, 0.1))

def slot_sidebar():
    """Barra lateral y cálculo de la chimenea slot con Simba (producción)."""
    so = SlotOptions()
    with st.sidebar.expander("Chimenea slot", expanded=True):
        so.kind = st.selectbox("Sección", VERTICAL_TYPES, index=0, key="sl_kind")
        c1, c2 = st.columns(2)
        so.a = c1.number_input("Lado x / diámetro (m)", 1.2, 6.0, 2.0, 0.1, key="sl_a")
        so.b = c2.number_input("Lado y (m)", 1.2, 6.0, so.a, 0.1, key="sl_b") if so.kind.startswith("Rect") else so.a
        so.L = c1.number_input("Longitud (altura entre subniveles, m)", 5.0, 80.0, 20.0, 0.5, key="sl_L")
        so.direction = st.radio("Perforación", SLOT_DIR, key="sl_dir")
        so.cut = st.selectbox("Diseño del cuele", SLOT_CUTS, key="sl_cut")
        so.blast = st.selectbox("Voladura", SLOT_BLAST, key="sl_bl",
                                help="Sesión 16 del curso: VCR/por tramos, integral o integral en 2–3 tramos con tacos "
                                     "intermedios.")
        c1, c2 = st.columns(2)
        if so.blast.startswith("Por tramos"):
            so.tramo = c1.number_input("Altura del tramo (m)", 1.0, 8.0, 3.0, 0.5, key="sl_tr")
            so.T_collar = c2.number_input("Retacado (m)", 0.2, 3.0, 0.5, 0.1, key="sl_tc")
        else:
            so.T_fondo = c1.number_input("Taco de fondo y boca (m)", 0.3, 4.0, 1.0, 0.1, key="sl_tf")
            if "espaciadas" in so.blast:
                so.n_tramos = c2.number_input("N° de cargas", 2, 4, 2, key="sl_nt")
                so.T_inter = c1.number_input("Taco intermedio (m)", 0.5, 5.0, 2.0, 0.1, key="sl_ti")
        so.booster_kg = c2.number_input("Booster por carga (kg)", 0.0, 1.0, 0.15, 0.01, key="sl_bo")
    with st.sidebar.expander("Perforación (Simba) y explosivo", expanded=False):
        c1, c2 = st.columns(2)
        so.D1_mm = c1.number_input("Ø cargados (mm)", 45.0, 115.0, 63.5, 0.5, key="sl_d1")
        so.D2_mm = c2.number_input("Ø vacíos (mm)", 64.0, 203.0, 101.6, 0.5, key="sl_d2")
        so.NB = c1.number_input("N° de vacíos", 1, 4, 2, key="sl_nb")
        so.alpha = c2.number_input("Desviación (m/m)", 0.0, 0.05, 0.01, 0.005, key="sl_al")
        so.Lmax_ft = c1.number_input("Capacidad del equipo (pies)", 30.0, 200.0, 102.0, 1.0, key="sl_lm",
                                     help="Simba: hasta 102 pies (≈ 31.1 m) con barras de extensión.")
        so.rod_ft = c2.number_input("Barra (pies)", 4.0, 6.0, 5.0, 1.0, key="sl_rod")
        sx = explosive_picker("Explosivo", "ANFO (granel)", '1 1/4" x 12"', "slx")
    with st.sidebar.expander("Esponjamiento y ensanche", expanded=False):
        c1, c2 = st.columns(2)
        so.swell = c1.slider("Esponjamiento (%)", 10, 60, 30, key="sl_sw") / 100
        so.L_exp = c2.number_input("Long. de expansión (m)", 1.0, 30.0, 4.0, 0.5, key="sl_le",
                                   help="Tramo de la labor inferior que recibe el material volado.")
        so.drift_w = c1.number_input("Ancho labor inferior (m)", 2.0, 8.0, 4.0, 0.1, key="sl_dw")
        so.drift_h = c2.number_input("Alto labor inferior (m)", 2.0, 8.0, 4.0, 0.1, key="sl_dh")
        so.widen = st.checkbox("Ensanchar a cámara slot (sesión 8)", False, key="sl_wd")
        if so.widen:
            so.slot_len = c1.number_input("Largo del slot (m)", 3.0, 40.0, 8.0, 0.5, key="sl_sl")
            so.slot_w = c2.number_input("Ancho del slot (m)", 2.0, 10.0, 4.0, 0.5, key="sl_sw2")
            so.SB = st.slider("S/B del ensanche", 1.0, 1.2, 1.1, 0.05, key="sl_sb")
    r = run_slot(rock, sx, so, cost)
    params = [("Método", SLOT_METHOD), ("Sección", f"{so.kind} {so.a:.2f} × {so.b:.2f} m"), ("Longitud", f"{so.L:.1f} m"),
              ("Perforación", so.direction), ("Cuele", so.cut), ("Voladura", so.blast),
              ("Ø cargados / vacíos", f"{so.D1_mm:.1f} / {so.D2_mm:.1f} mm × {so.NB}"),
              ("Equipo", f"Simba, capacidad {so.Lmax_ft:.0f} pies, barra {so.rod_ft:.0f} pies"), ("Explosivo", sx.label())]
    return r, sx, [sx], params


# ---------------------------------------------------------------------------
res = None
err = None
try:
    if labor == "Avance":
        with st.sidebar.expander("Sección de la galería", expanded=True):
            kind = st.selectbox("Tipo de galería", GALLERY_TYPES)
            preset = st.selectbox("Sección típica", ["1.00 x 1.50 (labor artesanal)", "1.50 x 1.50"] + list(PRESETS),
                                  index=8)
            dims = {"1.00 x 1.50 (labor artesanal)": (1.0, 1.5), "1.50 x 1.50": (1.5, 1.5)}.get(preset) or \
                PRESETS.get(preset) or (3.5, 3.5)
            c1, c2 = st.columns(2)
            A = c1.number_input("Ancho A (m)", 0.8, 12.0, float(dims[0]), 0.05)
            Hs = c2.number_input("Alto H (m)", 1.0, 12.0, float(dims[1]), 0.05)
            r_arch = st.number_input("Radio del baúl r (m)", 0.1, A / 2, min(1.25, A / 2), 0.05) \
                if kind.startswith("Baúl") else None
            top_w = st.number_input("Ancho en el techo (m)", 0.3, A, round(0.8 * A, 2), 0.05) \
                if kind.startswith("Trapezoidal") else None
            narrow = st.checkbox("Modo labor estrecha", value=A * Hs <= 4.0,
                                 help="Limita el cuele al ancho disponible y compara con el predimensionamiento de Carpio.")
            if A < 1.5 or Hs < 1.5:
                st.warning("Sección menor a 1.5 × 1.5 m: revise el estándar de sostenimiento y evacuación.")
        with st.sidebar.expander("Cuele", expanded=True):
            design = st.radio("Criterio de diseño de la malla", DESIGNS,
                              help="Konya: B = 0.012·(2ρe/ρr + 1.5)·De; auxiliares S = 1.1B, T = 0.5B; piso T = 0.2B; "
                                   "recorte S = 0.45–0.60 m, B = 1.3S, q = D²/12 g/m; look-out 0.1 + H·tan 2°.")
            default_cut = "Quemado simple" if narrow else (
                "Cuña quemada — Konya (barreno vacío grande)" if design.startswith("Konya") else CUT_TYPES[0])
            cut_type = st.selectbox("Tipo de cuele", CUT_TYPES, index=CUT_TYPES.index(default_cut),
                                    key=f"cut{design[:3]}{narrow}")
            st.caption(CUT_NOTES.get(cut_type, ""))
            cut_method, cut_factor, n_quad, burn_sp = "Holmberg", 1.7, None, None
            vertex, advf, dcr, Et, dlo = 60.0, 0.475, 200.0, 1.5, 0.5
            if cut_type.startswith("Cilíndrico 4"):
                cut_method = st.radio("B_max del arranque", ["Holmberg", "Pearse"], horizontal=True)
                if cut_method == "Holmberg":
                    cut_factor = st.radio("Factor", [1.7, 1.5], horizontal=True, format_func=lambda v: f"{v}·Ø_equiv")
                if st.radio("N° de cuadrantes", ["Automático", "Manual"], horizontal=True) == "Manual":
                    n_quad = st.slider("Cuadrantes", 1, 3, 3)
            elif cut_type in ("Quemado simple", "Quemado Sarrois"):
                if st.checkbox("Fijar distancia entre ejes", False):
                    burn_sp = st.number_input("Distancia (m)", 0.06, 0.30, 0.12, 0.01)
            elif cut_type in ("En V / cuña", "En abanico", "Instantáneo piramidal"):
                vertex = st.slider("Ángulo interior de la cuña (°)", 40, 90, 60)
                if cut_type != "Instantáneo piramidal":
                    advf = st.slider("Avance / ancho", 0.40, 0.55, 0.475, 0.005)
            elif cut_type.startswith("Cráter"):
                dcr = st.number_input("Ø barreno central (mm)", 76.0, 250.0, 200.0, 1.0)
                Et = st.number_input("Factor energía-tensión E_t", 0.8, 3.0, 1.5, 0.05)
                dlo = st.number_input("Relación óptima Δ_o", 0.3, 0.8, 0.5, 0.05)
        with st.sidebar.expander("Perforación", expanded=False):
            c1, c2 = st.columns(2)
            d1 = c1.number_input("Ø producción (mm)", 22.0, 76.0, 32.0 if narrow else 45.0, 1.0)
            d2 = c2.number_input("Ø vacío / rimado (mm)", 22.0, 203.0, 38.0 if narrow else 102.0, 1.0)
            NB = c1.number_input("N° de vacíos", 1, 4, 2)
            bar_ft = c2.number_input("Barra (pies)", 3.0, 20.0, 4.0 if narrow else 11.0, 0.5)
            eff = st.slider("Eficiencia de perforación (%)", 70, 100, 95)
            Fm = st.selectbox("Error de perforación F", ["Dato directo", "Ec. 26", "Ec. 27"])
            Fv, alpha, ecol = 0.04, 10.0, 20.0
            if Fm == "Dato directo":
                Fv = st.number_input("F (m)", 0.0, 0.5, 0.04, 0.005)
            elif Fm == "Ec. 26":
                alpha = c1.number_input("α (mm/m)", 0.0, 50.0, 10.0)
                ecol = c2.number_input("e (mm)", 0.0, 100.0, 20.0)
        with st.sidebar.expander("Explosivos y carga", expanded=False):
            exm = explosive_picker("Explosivo principal", "Emulnor 1000", '7/8" x 7"' if narrow else '1 1/4" x 12"', "am")
            exc = explosive_picker("Explosivo de corona", "Emulnor 500", '1" x 12"', "ac")
            charge_method = st.selectbox("Método de carga", CHARGE_METHODS)
            atac = st.number_input("Factor de atacado del cartucho", 0.6, 1.0, 1.0, 0.05,
                                   help="Longitud que ocupa el cartucho atacado / longitud nominal. Las plantillas del "
                                        "curso usan 0.8 (el cartucho se comprime y aumenta la carga por metro).")
            exm.atacado = exc.atacado = atac
        with st.sidebar.expander("Contorno y otros", expanded=False):
            contour_mode = st.selectbox("Contorno", ["Voladura controlada (S = K·Ø1, q = 90·Ø1²)",
                                                     "Sin voladura controlada (f = 1.2, S/B = 1.25)"],
                                        index=1 if narrow else 0)
            c1, c2 = st.columns(2)
            gamma = c1.number_input("γ (°)", 0.0, 10.0, 3.0, 0.5)
            Kc = c2.number_input("K corona", 10.0, 20.0, 15.0, 0.5)
            c_rule = st.selectbox("Ĉ", ["Iterativa", "c + 0.05"])
            offset = st.number_input("Separación del contorno (m)", 0.0, 0.3, 0.10, 0.01)
            cut_h = st.number_input("Altura del cuele (m)", 0.3, 8.0, 0.6, 0.05) if st.checkbox("Fijar altura del cuele") else None
            beff = st.slider("Eficiencia de voladura (%)", 70, 100, 95) / 100
        key = f"Avance|{cut_type}"
        ov = ss.overrides.get(key, (None, None))
        section = build_section(kind, A, Hs, r_arch, top_w)
        drill = Drill(d1_mm=d1, d2_mm=d2, NB=int(NB), bar_m=bar_ft * 0.3048, eff=eff / 100, F_method=Fm, F_value=Fv,
                      alpha_mm_m=alpha, e_mm=ecol)
        opts = AvanceOptions(cut_type=cut_type, cut_method=cut_method, cut_factor=float(cut_factor), n_quadrants=n_quad,
                             burn_spacing=burn_sp, vertex_deg=vertex, adv_factor=advf, d_crater_mm=dcr, Et=Et,
                             delta_o=dlo, gamma_deg=gamma, K_corona=Kc, contour_mode=contour_mode, c_hat_rule=c_rule,
                             contour_offset=offset, cut_height=cut_h, blast_eff=beff, charge_method=charge_method,
                             narrow=narrow, B_prod_override=ov[0], SB_prod=ov[1] or (1.1 if design.startswith("Konya") else 1.25),
                             design=design)
        res = run_avance(section, rock, drill, exm, exc, opts, cost)
        main_exp = exm
        used_exps = [exm, exc]
        pdf_params = [("Sección", f"{kind} {A:.2f} × {Hs:.2f} m"), ("Criterio de diseño", design),
                      ("Cuele", cut_type), ("Ø producción / vacío", f"{d1:.0f} / {d2:.0f} mm × {int(NB)}"),
                      ("Barra · eficiencia", f"{bar_ft:.1f} pies · {eff} %"), ("Explosivo", exm.label()),
                      ("Explosivo de corona", exc.label()), ("Método de carga", charge_method), ("Contorno", contour_mode)]
    elif labor in ("Chimenea", "Pique"):
        methods = CHIMENEA_METHODS if labor == "Chimenea" else PIQUE_METHODS
        with st.sidebar.expander("Sección y método", expanded=True):
            method = st.selectbox("Método", methods)
            vkind = st.selectbox("Sección", VERTICAL_TYPES, index=0 if labor == "Chimenea" else 1)
            c1, c2 = st.columns(2)
            a = c1.number_input("Lado x / diámetro (m)", 1.0, 12.0, 1.8 if labor == "Chimenea" else 6.0, 0.1)
            b = c2.number_input("Lado y (m)", 1.0, 12.0, a, 0.1) if vkind.startswith("Rect") else a
            long_m = labor == "Chimenea" and method in CHIMENEA_METHODS[3:]
            length = st.number_input("Longitud de la chimenea (m)", 5.0, 150.0, 30.0, 1.0) if long_m else 30.0
            tramo = st.number_input("Altura de cada pega (m)", 1.0, 6.0, 3.0, 0.5) if long_m else 3.0
            lmax_v = st.number_input("Capacidad del equipo (pies)", 30.0, 200.0, 102.0, 1.0,
                                     help="Simba: hasta 102 pies (≈ 31.1 m).") if long_m else 102.0
            cut_o = None
            if not long_m and method not in (PIQUE_METHODS[0], PIQUE_METHODS[1], PIQUE_METHODS[5]):
                if st.checkbox("Elegir otro cuele", False):
                    cut_o = st.selectbox("Cuele", CUT_TYPES)
            dcen = st.number_input("Ø chimenea piloto / barreno central (m)", 0.1, 3.0, 1.5 if labor == "Chimenea" else 1.2,
                                   0.05) if method in (CHIMENEA_METHODS[6], PIQUE_METHODS[5]) else 1.5
            incl = st.slider("Inclinación de barrenos del cuele (° desde la horizontal)", 50, 75, 60) \
                if method in (PIQUE_METHODS[2], PIQUE_METHODS[3]) else 60
        with st.sidebar.expander("Perforación y explosivo", expanded=False):
            dflt = {CHIMENEA_METHODS[3]: (64, 150, 1), CHIMENEA_METHODS[4]: (89, 102, 1), CHIMENEA_METHODS[5]: (114, 102, 1),
                    CHIMENEA_METHODS[6]: (64, 102, 1), CHIMENEA_METHODS[1]: (38, 127, 1),
                    CHIMENEA_METHODS[2]: (38, 64, 3)}.get(method, (38 if labor == "Chimenea" else 45, 102, 2))
            c1, c2 = st.columns(2)
            vd1 = c1.number_input("Ø producción (mm)", 22.0, 200.0, float(dflt[0]), 1.0)
            vd2 = c2.number_input("Ø vacío (mm)", 22.0, 250.0, float(dflt[1]), 1.0)
            vNB = c1.number_input("N° de vacíos", 1, 4, int(dflt[2]))
            vbar = c2.number_input("Barra (pies)", 3.0, 20.0, 8.0 if labor == "Chimenea" else 12.0, 0.5)
            veff = st.slider("Eficiencia (%)", 70, 100, 95, key="veff")
            vF = st.number_input("Error de perforación F (m)", 0.0, 0.5, 0.05, 0.005)
            vexp = explosive_picker("Explosivo", "Emulsión a granel" if vd1 >= 60 else "Emulnor 1000",
                                    '1 1/4" x 12"', f"ve{labor}{methods.index(method)}")
            contour = st.selectbox("Contorno", ["Recorte (S ≈ 13·D)", "Voladura controlada (S = K·Ø, q = 90·Ø²)"])
            ratio_blk = st.number_input("Dimensión mínima / bloque máximo", 2.0, 8.0, 3.0, 0.5,
                                        help="Verificación de atascos: el bloque mayor debe ser varias veces menor que la sección.")
            Et_v, dlo_v, kv = 1.5, 0.5, 0.8
            if method in (CHIMENEA_METHODS[4], CHIMENEA_METHODS[5]):
                Et_v = st.number_input("E_t (Livingston)", 0.8, 3.0, 1.5, 0.05)
                dlo_v = st.number_input("Δ_o", 0.3, 0.8, 0.5, 0.05)
                if method == CHIMENEA_METHODS[5]:
                    kv = st.number_input("S / D_o", 0.5, 1.5, 0.8, 0.05)
        key = f"{labor}|{method}"
        ov = ss.overrides.get(key, (None, None))
        vs = build_vertical_section(vkind, a, b)
        vo = VerticalOptions(method=method, cut_type=cut_o, d1_mm=vd1, d2_mm=vd2, NB=int(vNB), bar_m=vbar * 0.3048,
                             eff=veff / 100, F=vF, length=length, tramo=tramo, contour=contour, d_central=dcen, Et=Et_v,
                             delta_o=dlo_v, k_vcr=kv, ratio_bloque=ratio_blk, inclin_deg=incl, B_override=ov[0],
                             SB=ov[1] or 1.25, Lmax_ft=lmax_v)
        res = run_vertical(labor, vs, rock, vexp, vo, cost)
        main_exp = vexp
        used_exps = [vexp]
        pdf_params = [("Método", method), ("Sección", f"{vkind} {a:.2f} × {b:.2f} m"),
                      ("Ø producción / vacío", f"{vd1:.0f} / {vd2:.0f} mm × {int(vNB)}"), ("Barra", f"{vbar:.1f} pies"),
                      ("Explosivo", vexp.label()), ("Contorno", contour)]
    else:
        method = st.sidebar.selectbox("Método", PROD_METHODS + [SLOT_METHOD])
        if method == SLOT_METHOD:
            res, main_exp, used_exps, pdf_params = slot_sidebar()
            key = f"Producción|{method}"
        else:
            with st.sidebar.expander("Método de explotación", expanded=True):
                Dd, crit_d = DEFAULTS[method]
                criterion = st.selectbox("Criterio de burden", CRITERIA, index=CRITERIA.index(crit_d))
                po = ProdOptions(method=method, criterion=criterion)
                c1, c2 = st.columns(2)
                po.D_mm = c1.number_input("Ø perforación (mm)", 25.0, 250.0, float(Dd), 1.0)
                geo = {PROD_METHODS[0]: (30.0, 15.0, 30.0), PROD_METHODS[1]: (20.0, 1.5, 15.0),
                       PROD_METHODS[2]: (20.0, 12.0, 15.0), PROD_METHODS[3]: (20.0, 8.0, 30.0),
                       PROD_METHODS[4]: (20.0, 8.0, 6.0), PROD_METHODS[5]: (4.0, 8.0, 5.0),
                       PROD_METHODS[6]: (15.0, 2.0, 3.0), PROD_METHODS[7]: (4.0, 2.0, 4.0),
                       PROD_METHODS[8]: (20.0, 15.0, 30.0)}[method]
                lab_len = "Longitud de barreno (m)" if method in (PROD_METHODS[5], PROD_METHODS[7]) else "Largo del tajeo (m)"
                po.length = c2.number_input(lab_len, 1.0, 300.0, geo[0], 0.5)
                lab_w = {PROD_METHODS[4]: "Ancho de cámara (m)", PROD_METHODS[5]: "Ancho de cámara (m)"}.get(
                    method, "Potencia / ancho AV (m)")
                wv = c1.number_input(lab_w, 0.5, 100.0, geo[1], 0.1)
                if method in (PROD_METHODS[4], PROD_METHODS[5]):
                    po.room = wv
                po.width = wv
                lab_h = {PROD_METHODS[2]: "Altura del abanico ascendente (m)", PROD_METHODS[4]: "Altura del banco (m)",
                         PROD_METHODS[5]: "Altura del banco (m)", PROD_METHODS[6]: "Altura de la tajada (m)",
                         PROD_METHODS[7]: "Altura de la tajada (m)", PROD_METHODS[8]: "Altura total de la cámara (m)"}.get(
                    method, "Altura entre niveles (m)")
                hv = c2.number_input(lab_h, 1.0, 120.0, geo[2], 0.5)
                po.height = hv
                po.up_h = hv
                if method in (PROD_METHODS[0], PROD_METHODS[1]):
                    lab_dip = "Buzamiento de la veta / inclinación de los taladros (°)" if method == PROD_METHODS[1] \
                        else "Inclinación de los taladros paralelos (°)"
                    po.dip_deg = st.slider(lab_dip, 45, 90, 90, help="90° = verticales. El carrusel del Simba gira en "
                                           "'dip' para perforar los taladros paralelos inclinados.")
                    if po.dip_deg < 90:
                        po.dip_side = 1.0 if st.radio("Sentido de la inclinación", ["Hacia −y (izquierda)",
                                                      "Hacia +y (derecha)"], horizontal=True).startswith("Hacia −") else -1.0
                    tipo = st.radio("Perforación", ["Pasante (conecta las dos labores)", "Ciega"], horizontal=True,
                                    help="Pasante: el taladro sale a la labor inferior. Ciega: se deja un puente de roca "
                                         "sobre la labor inferior.")
                    po.breakthrough = tipo.startswith("Pasante")
                    if not po.breakthrough:
                        po.toe_gap = st.number_input("Puente sobre la labor inferior (m)", 0.3, 5.0, 1.0, 0.1)
                if method == PROD_METHODS[2]:
                    po.fan_type = st.radio("Abanico", ["Ascendente", "Descendente", "Ambos sentidos"], horizontal=True)
                    if po.fan_type != "Ascendente":
                        po.down_h = st.number_input("Altura del abanico descendente (m)", 1.0, 40.0, 10.0, 0.5)
                    po.ce0_key = st.selectbox("Tipo de roca (Tabla 24.4)", list(CE0_TABLE), index=1)
                    po.SB = st.slider("S/B (1.3–2)", 1.3, 2.0, 2.0, 0.05)
                if method == PROD_METHODS[3]:
                    po.k_vcr = st.number_input("S / D_o", 0.6, 1.5, 1.0, 0.05)
                    po.Et = st.number_input("E_t", 0.8, 3.0, 1.5, 0.05)
                    po.delta_o = st.number_input("Δ_o", 0.3, 0.8, 0.5, 0.05)
                    po.corona = st.number_input("Altura del pilar corona (m)", 1.0, 20.0, 6.0, 0.5)
                if method == PROD_METHODS[4]:
                    po.pillar = st.number_input("Lado del pilar (m)", 2.0, 30.0, 6.0, 0.5)
                if method == PROD_METHODS[6]:
                    po.incl_deg = st.slider("Inclinación de los barrenos (°)", 50, 65, 60)
                if method == PROD_METHODS[8]:
                    po.phase_h = st.number_input("Altura de cada banco (m)", 2.0, 20.0, 8.0, 0.5)
                    po.orient_h = st.radio("Banqueo", ["Vertical (50–65 mm)", "Horizontal (32–45 mm)"]) .startswith("Horiz")
                if method == PROD_METHODS[2]:
                    po.fan_stem = st.selectbox("Retacado de los anillos", FAN_STEMMING)
                    po.ring_dip = st.slider("Inclinación del anillo — dip (°)", 60, 90, 90,
                                            help="90° = anillo vertical. El carrusel gira en 'dip' e inclina el plano "
                                                 "del abanico hacia el slot.")
            if criterion.startswith("Konya"):
                with st.sidebar.expander("Konya: ecuación y correcciones", expanded=True):
                    po.konya_eq = st.selectbox("Ecuación del burden", KONYA_EQS)
                    if "6.3" in po.konya_eq:
                        po.stv = st.number_input("Potencia relativa por volumen Stv (ANFO = 100)", 50.0, 250.0, 100.0, 5.0)
                    po.Kd = KONYA_KD[st.selectbox("Kd — orientación de los estratos (Tabla 6.3)", list(KONYA_KD), index=2)]
                    po.Ks = KONYA_KS[st.selectbox("Ks — estructura geológica (Tabla 6.4)", list(KONYA_KS), index=1)]
                    po.Kr = KONYA_KR[st.selectbox("Kr — número de hileras (Tabla 6.2)", list(KONYA_KR), index=0)]
                    po.konya_delayed = st.radio("Iniciación entre taladros de una fila", ["Retardada", "Instantánea"],
                                                horizontal=True) == "Retardada"
                    po.konya_S = st.checkbox("Espaciamiento por la relación de rigidez L/B (ecs. 7.1–7.4)", True)
                    st.caption("Taco T = 0.7·B (ec. 6.4); sobreperforación J = 0.3·B (ec. 6.6) en bancos.")
            elif criterion.startswith("Ash"):
                with st.sidebar.expander("Ash", expanded=True):
                    po.Kb = st.number_input("Constante Kb", 15.0, 45.0, 30.0, 1.0,
                                            help="Plantilla del curso: 25 (ANFO) y 30 (emulsión).")
            if method in PROD_METHODS[:4]:
                with st.sidebar.expander("Labor de perforación y equipo", expanded=False):
                    parallel = method != PROD_METHODS[2]
                    full = False
                    if parallel:
                        full = st.radio("Perforación desde", ["Galería(s) de perforación", "Cámara de ancho completo"],
                                        index=0 if method == PROD_METHODS[1] else 1,
                                        help="LBH y VCR suelen perforarse desde un corte superior de ancho completo; las "
                                             "vetas angostas, desde una galería sobre la veta.") == "Cámara de ancho completo"
                    dk = st.selectbox("Sección de la labor", GALLERY_TYPES, index=0 if not full else 3, disabled=full,
                                      key=f"dk{full}")
                    dpre = st.selectbox("Sección típica", list(PRESETS), index=3 if method == PROD_METHODS[1] else 5,
                                        key=f"dpre{method[:6]}")
                    ddim = PRESETS.get(dpre) or (4.0, 4.0)
                    c1, c2 = st.columns(2)
                    dw = c1.number_input("Ancho (m)", 2.0, 8.0, float(ddim[0]), 0.1, disabled=full, key=f"dw{dpre}")
                    dh = c2.number_input("Alto (m)", 2.0, 8.0, float(ddim[1]), 0.1, key=f"dh{dpre}")
                    dr = st.number_input("Radio del baúl (m)", 0.1, dw / 2, min(1.0, dw / 2), 0.05) \
                        if dk.startswith("Baúl") and not full else None
                    dtw = st.number_input("Ancho en el techo (m)", 0.5, dw, round(0.8 * dw, 2), 0.05) \
                        if dk.startswith("Trap") and not full else None
                    st.markdown("**Equipo de perforación**")
                    c1, c2 = st.columns(2)
                    ph = c1.number_input("Altura del pivote (m)", 0.8, 2.5, 1.4, 0.05,
                                         help="Punto de giro del equipo sobre el piso de la labor: 1.3–1.5 m típico.")
                    py_ = c2.number_input("Pivote: desplazamiento lateral (m)", -3.0, 3.0, 0.0, 0.05) if not parallel else 0.0
                    cw = c1.number_input("Carrusel: ancho (m)", 0.3, 4.0, 1.2, 0.05,
                                         help="Rectángulo que encierra el carrusel en la sección; varía por equipo "
                                              "(consulte el catálogo del fabricante).")
                    chh = c2.number_input("Carrusel: alto (m)", 0.2, 3.0, 0.8, 0.05)
                    cdz = c1.number_input("Carrusel: desfase vertical (m)", -1.5, 1.5, 0.0, 0.05,
                                          help="Posición del centro del rectángulo respecto al pivote.")
                    nd, dy_ = 1, 0.0
                    if parallel and not full:
                        dy_ = st.number_input("Eje de la galería respecto al eje del tajeo (m)", -10.0, 10.0, 0.0, 0.05,
                                              help="Se perfora desde una sola galería por nivel.")
                    st.markdown("**Capacidad del equipo de taladros largos**")
                    c1, c2 = st.columns(2)
                    po.Lmax_ft = c1.number_input("Longitud máxima (pies)", 30.0, 200.0, 102.0, 1.0,
                                                 help="Simba: hasta 102 pies (≈ 31.1 m) con barras de extensión.")
                    po.rod_ft = c2.number_input("Barra (pies)", 4.0, 6.0, 5.0, 1.0)
                    po.drill = DrillSetup(kind=dk if not full else "Rectangular", width=dw, height=dh, radius=dr, top_w=dtw,
                                          pivot_h=ph, pivot_y=py_, car_w=cw, car_h=chh, car_dz=cdz, n_drifts=int(nd),
                                          drift_y=dy_, full_width=full)
                    lvl = "inferior" if parallel else {"Ascendente": "superior", "Descendente": "inferior"}.get(
                        po.fan_type, "superior e inferior")
                    st.markdown(f"**Galería del otro nivel ({lvl})**")
                    po.other_on = st.checkbox(f"Dibujar la galería {lvl}", True, key="oth_on",
                                              help="Galería de extracción / undercut bajo el tajeo (taladros paralelos) o "
                                                   "galería del nivel siguiente hacia donde llegan los abanicos.")
                    if po.other_on:
                        ok_ = st.selectbox("Sección de la galería", GALLERY_TYPES, index=0, key="oth_k")
                        c1, c2 = st.columns(2)
                        ow = c1.number_input("Ancho (m) ", 2.0, 10.0, 4.0, 0.1, key="oth_w")
                        oh = c2.number_input("Alto (m) ", 2.0, 10.0, 4.0, 0.1, key="oth_h")
                        orr = min(1.0, ow / 2) if ok_.startswith("Baúl") else None
                        otw = round(0.8 * ow, 2) if ok_.startswith("Trap") else None
                        ody = c1.number_input("Desplazamiento lateral (m)", -10.0, 10.0, 0.0, 0.1, key="oth_dy")
                        po.other = DrillSetup(kind=ok_, width=ow, height=oh, radius=orr, top_w=otw, drift_y=ody)
                    st.caption("Una sola labor de perforación por nivel; el carrusel gira con el 'dip' de cada taladro.")
            if method == PROD_METHODS[3]:
                po.slot_on = False          # VCR: la cara libre es el corte inferior (undercut), sin chimenea slot
            if method in PROD_METHODS[:3]:
                with st.sidebar.expander("Chimenea slot (cara libre inicial)", expanded=False):
                    po.slot_on = st.checkbox("Incluir chimenea slot", True, key=f"so{method[:6]}",
                                             help="En taladros largos la producción arranca hacia una chimenea slot "
                                                  "ensanchada de caja a caja.")
                    if po.slot_on:
                        so = SlotOptions(widen=True)
                        so.kind = st.selectbox("Sección", VERTICAL_TYPES, index=0, key="ps_kind")
                        c1, c2 = st.columns(2)
                        so.a = c1.number_input("Lado transversal (m)", 1.2, 6.0, 2.0, 0.1, key="ps_a")
                        so.b = c2.number_input("Lado longitudinal (m)", 1.2, 6.0, 2.0, 0.1, key="ps_b") \
                            if so.kind.startswith("Rect") else so.a
                        so.direction = st.radio("Perforación del slot", SLOT_DIR, key="ps_dir")
                        so.cut = st.selectbox("Cuele", SLOT_CUTS, key="ps_cut")
                        so.blast = st.selectbox("Voladura del slot", SLOT_BLAST, key="ps_bl")
                        c1, c2 = st.columns(2)
                        so.D1_mm = c1.number_input("Ø cargados (mm)", 45.0, 115.0, 63.5, 0.5, key="ps_d1")
                        so.D2_mm = c2.number_input("Ø vacíos (mm)", 64.0, 203.0, 101.6, 0.5, key="ps_d2")
                        so.NB = c1.number_input("N° de vacíos", 1, 4, 2, key="ps_nb")
                        so.booster_kg = c2.number_input("Booster (kg)", 0.0, 1.0, 0.15, 0.01, key="ps_bo")
                        if so.blast.startswith("Por tramos"):
                            so.tramo = c1.number_input("Altura del tramo (m)", 1.0, 8.0, 3.0, 0.5, key="ps_tr")
                        so.blind = st.radio("Tipo", ["Pasante", "Ciega"], horizontal=True, key="ps_bd") == "Ciega"
                        if so.blind:
                            so.toe_gap = st.number_input("Puente (m)", 0.3, 5.0, 1.0, 0.1, key="ps_tg")
                        so.widen = st.checkbox("Ensanchar de caja a caja (cámara slot)", True, key="ps_wd")
                        if so.widen:
                            so.slot_w = c1.number_input("Espesor de la cámara slot (m)", 2.0, 10.0, 4.0, 0.5, key="ps_sw")
                            so.SB = c2.slider("S/B ensanche", 1.0, 1.2, 1.1, 0.05, key="ps_sb")
                        so.alpha = 0.01
                        po.slot = so
                        st.caption("La chimenea se perfora con la misma inclinación de los taladros paralelos y su ensanche "
                                   "abarca todo el ancho del tajeo; la primera fila de producción sale hacia ella con el "
                                   "burden y espaciamiento de producción.")
            if method in PROD_METHODS[:8] and multi:
                rp_m = method in (PROD_METHODS[4], PROD_METHODS[5])
                with st.sidebar.expander("Sobrerotura y estabilidad por dominios (pilares, techo y piso)" if rp_m else
                                         "Sobrerotura y dilución por dominios (cajas)", expanded=False):
                    dmo = DomainOptions()
                    dmo.on = st.checkbox("Evaluar sobrerotura, dilución y estabilidad", True, key="dm_on",
                                         help="Usa el mineral, la caja techo y la caja piso del apartado Macizo rocoso.")
                    if dmo.on:
                        st.caption(f"Rocas del apartado Macizo rocoso — VPP crítica: mineral {ore.ppv_crit:,.0f}, caja "
                                   f"techo {hw_rock.ppv_crit:,.0f}, caja piso {fw_rock.ppv_crit:,.0f} mm/s")
                        inclined = method in (PROD_METHODS[0], PROD_METHODS[1]) and po.dip_deg < 90
                        if rp_m:
                            st.caption("Cámaras y pilares: los pilares son mineral (apartado Macizo rocoso); se evalúan "
                                       "su ancho efectivo y su FS, el techo (abertura de Barton) y el piso.")
                        elif inclined:
                            st.caption("Con taladros inclinados, la caja techo es la pared que queda sobre el mineral "
                                       "(lado hacia el que buza la veta).")
                        else:
                            dmo.hw_side = -1.0 if st.radio("Caja techo al lado", ["−y (izquierda)", "+y (derecha)"],
                                                           horizontal=True, key="dm_side").startswith("−") else 1.0

                        dmo.hw, dmo.fw = hw_rock, fw_rock
                        if rp_m:
                            st.caption("Manto: el techo es la caja techo y el piso es la caja piso.")
                            c1, c2 = st.columns(2)
                            roof_is = c1.selectbox("Techo", ["Caja techo (estéril)", "Mineral"], key="dm_rw")
                            floor_is = c2.selectbox("Piso", ["Mineral", "Caja piso (estéril)"], key="dm_fw")
                            dmo.roof_waste = roof_is.startswith("Caja")
                            dmo.floor_waste = floor_is.startswith("Caja")
                            dmo.back = hw_rock if dmo.roof_waste else None
                            dmo.floor = fw_rock if dmo.floor_waste else None
                            st.markdown("**Pilares y techo**")
                            c1, c2 = st.columns(2)
                            if method == PROD_METHODS[5]:
                                po.pillar = c1.number_input("Lado del pilar (m)", 2.0, 30.0, 6.0, 0.5, key="dm_pw")
                            dmo.depth = c2.number_input("Profundidad (m)", 10.0, 2500.0, 200.0, 10.0, key="dm_H")
                            dmo.h_head = c1.number_input("Altura de la galería piloto (m)", 0.0, 10.0, 4.0, 0.1,
                                                         key="dm_hh", help="El techo queda a banco + galería piloto; "
                                                                           "la altura del pilar es la suma de ambas.")
                            dmo.rho_over = c2.number_input("ρ recubrimiento (t/m³)", 1.8, 4.0, 2.7, 0.05, key="dm_ro")
                            dmo.esr = c1.number_input("ESR (Barton)", 0.8, 5.0, 3.0, 0.1, key="dm_esr",
                                                      help="Labores mineras temporales: 3–5; permanentes: 1.6.")
                            dmo.K_hg = c2.number_input("K Hedley–Grant (MPa)", 20.0, 300.0, 133.0, 1.0, key="dm_khg")
                        if method == PROD_METHODS[2] and po.fan_type != "Descendente":
                            bk = st.selectbox("Techo / corona sobre los abanicos", ["Mineral", "Caja techo"], key="dm_bk")
                            dmo.back = ore if bk == "Mineral" else hw_rock
                        if method in (PROD_METHODS[6], PROD_METHODS[7]):
                            st.caption("Corte y relleno: el techo de la tajada es mineral (apartado Macizo rocoso) y el "
                                       "piso es relleno. El daño al techo no diluye, pero afecta la seguridad del "
                                       "frente; las cajas sí diluyen.")
                        st.markdown("**Vibración y sobrerotura**")
                        c1, c2, c3 = st.columns(3)
                        dmo.K = c1.number_input("K (H–P)", 50.0, 2000.0, 700.0, 10.0, key="dm_K")
                        dmo.alpha = c2.number_input("α (H–P)", 0.3, 1.5, 0.7, 0.01, key="dm_a")
                        dmo.n_ob = c3.number_input("n × VPP crít.", 1.0, 8.0, 4.0, 0.5, key="dm_n",
                                                   help="Umbral de sobrerotura = n × VPP crítica de la caja (curso: 4).")
                        dmo.edge_mode = st.selectbox("Taladros de borde", EDGE_MODES, key="dm_em")
                        if dmo.edge_mode == EDGE_MODES[1]:
                            dmo.edge_factor = st.slider("Factor de carga en el borde", 0.15, 1.0, 0.6, 0.05, key="dm_f")
                        elif dmo.edge_mode == EDGE_MODES[2]:
                            dmo.elos_target = st.number_input("ELOS objetivo (m)", 0.0, 3.0, 0.3, 0.05, key="dm_t")
                        c1, c2 = st.columns(2)
                        dmo.grade = c1.number_input("Ley in situ del mineral", 0.0, 1000.0, 0.0, 0.1, key="dm_g")
                        dmo.grade_unit = c2.selectbox("Unidad", GRADE_UNITS, key="dm_gu")
                        st.caption("K y α de Holmberg–Persson son propios de cada sitio: calíbrelos con monitoreo de "
                                   "vibraciones para que el ELOS estimado sea confiable.")
                    po.domains = dmo
            with st.sidebar.expander("Parámetros del esquema", expanded=False):
                c1, c2 = st.columns(2)
                fdef = {PROD_METHODS[5]: 1.45, PROD_METHODS[6]: 0.85, PROD_METHODS[7]: 1.20}.get(method, 1.0)
                po.f = c1.number_input("Factor de fijación f", 0.8, 1.6, fdef, 0.05,
                                       help="Langefors: vertical 1.0; 3:1 0.9; 2:1 0.85. Holmberg: arriba/horizontal 1.45; abajo 1.20")
                if method != PROD_METHODS[2]:
                    po.SB = c2.number_input("S/B", 1.0, 2.0, 1.25, 0.05)
                po.rows_blast = c1.number_input("Filas por voladura", 1, 50, 3)
                po.col_ratio = c2.number_input("Columna / fondo", 0.3, 1.0, 0.5 if method != PROD_METHODS[2] else 0.6, 0.05)
                if method in PROD_METHODS[:3]:
                    po.full_column = st.checkbox("Carga continua en toda la longitud del taladro", True,
                                                 help="Taladros largos: la columna llega a todo el taladro descontando "
                                                      "el taco de boca y el iniciador va en el fondo. Desmarque para "
                                                      "cargas espaciadas (decks).")
                if not po.full_column or method not in PROD_METHODS[:3]:
                    po.q_deck_max = c1.number_input("Carga máx. por sección de taladro (kg)", 20.0, 500.0, 150.0, 10.0)
                po.k_T = c2.number_input("Factor de retacado", 0.5, 2.0, 1.0, 0.1)
                po.alpha = c1.number_input("Desviación (m/m)", 0.0, 0.05, 0.01, 0.005)
                po.e_coll = c2.number_input("Error de emboquille (m)", 0.0, 0.3, 0.05, 0.01)
                po.staggered = st.checkbox("Malla al tresbolillo", True)
                pexp = explosive_picker("Explosivo", "ANFO (granel)" if po.D_mm >= 51 else "Emulnor 1000",
                                        '1 1/4" x 12"', f"pe{PROD_METHODS.index(method)}")
            key = f"Producción|{method}"
            ov = ss.overrides.get(key, (None, None))
            po.B_override = ov[0]
            if ov[1]:
                po.SB = ov[1]
            res = run_produccion(rock, pexp, po, cost)
            main_exp = pexp
            used_exps = [pexp]
            pdf_params = [("Método", method), ("Criterio de burden", criterion), ("Ø perforación", f"{po.D_mm:.0f} mm"),
                          ("Geometría (largo · ancho · alto)", f"{po.length:.1f} · {po.width:.1f} · {po.height:.1f} m"),
                          ("Explosivo", pexp.label()), ("Filas por voladura", po.rows_blast),
                          ("Desviación · emboquille", f"{po.alpha * 100:.1f} % · {po.e_coll:.2f} m")]
            if multi:
                pdf_params.append(("Caja techo (RMR · σci · VPP crít.)",
                                   f"{hw_rock.RMR} · {hw_rock.UCS:.0f} MPa · {hw_rock.ppv_crit:,.0f} mm/s"))
                pdf_params.append(("Caja piso (RMR · σci · VPP crít.)",
                                   f"{fw_rock.RMR} · {fw_rock.UCS:.0f} MPa · {fw_rock.ppv_crit:,.0f} mm/s"))
            if method in (PROD_METHODS[0], PROD_METHODS[1]):
                pdf_params.append(("Taladros paralelos", f"{po.dip_deg:.0f}° · " + ("pasantes" if po.breakthrough
                                   else f"ciegos (puente {po.toe_gap:.1f} m)")))
            if method == PROD_METHODS[2]:
                pdf_params.append(("Inclinación del anillo", f"{po.ring_dip:.0f}°"))
            if method in PROD_METHODS[:3]:
                pdf_params.append(("Chimenea slot", "sí — " + (po.slot.direction if po.slot else "")
                                   if po.slot_on else "no"))
            if method == PROD_METHODS[3]:
                pdf_params.append(("Cara libre", "corte inferior (undercut), sin chimenea slot"))
except Exception as e:  # noqa: BLE001
    err = e

if res is None:
    st.error(f"No se pudo generar el diseño con estos datos: {err}")
    st.stop()

H = res.holes
sig = (labor, res.metodo, len(H), round(float(H.x.sum()), 3), round(float(H.kg.sum()), 3), system)
if ss.seq_sig != sig:
    ss.seq = None

# =============================================================================
# ENCABEZADO
# =============================================================================
st.title(f"Diseño de voladura — {labor}")
st.caption(f"{res.metodo}")
S = res.summary
m = st.columns(6)
m[0].metric("Taladros + vacíos", f"{len(H)} + {0 if res.relief is None else len(res.relief)}")
adv_key = "Avance por disparo (m)" if "Avance por disparo (m)" in S else "Avance por pega (m)" if "Avance por pega (m)" in S else None
if adv_key:
    m[1].metric("Avance", f"{S[adv_key]:.2f} m")
else:
    m[1].metric("Volumen por voladura", f"{S['Volumen roto (m³)']:.0f} m³")
m[2].metric("Explosivo por voladura", f"{S['Explosivo total (kg)']:.1f} kg")
m[3].metric("Factor de carga", f"{S['Factor de carga (kg/m³)']:.2f} kg/m³")
m[4].metric("Consumo", f"{S['Factor de carga (g/t)']:.0f} g/t")
m[5].metric("Costo (US$/m)" if "Costo por metro de avance (US$/m)" in S else "Costo (US$/t)",
            f"{S.get('Costo por metro de avance (US$/m)', S.get('Costo por tonelada (US$/t)', 0)):.1f}")
if ss.overrides.get(key, (None, None))[0]:
    st.info(f"Burden impuesto por la optimización de P80: B = {ss.overrides[key][0]:.2f} m "
            f"(S/B = {ss.overrides[key][1] or 1.25:.2f}). Puede restablecerlo en la pestaña de fragmentación.")
for w in dict.fromkeys(res.warnings):
    st.warning(w)

tabs = st.tabs(["Memoria de cálculo", "Fragmentación (P80)", "Vista 2D", "Vista 3D y carguío", "Energía",
                "Secuenciamiento", "Resumen y exportación"])
emap = {e.label(): e.energy for e in used_exps}

# =============================================================================
with tabs[0]:
    note = res.p.get("note") or ""
    cnote = res.p.get("cut_note") or ""
    if note:
        st.info(note)
    if cnote:
        st.caption("Cuele: " + cnote)
    if labor == "Avance":
        c1, c2 = st.columns(2)
        if (ASSETS / "diagrama_flujo.png").exists():
            with c1.expander("Diagrama de flujo del método Holmberg"):
                st.image(str(ASSETS / "diagrama_flujo.png"), width="stretch")
        if (ASSETS / "cueles_1x1.5.png").exists():
            with c2.expander("Cueles quemados para labores de 1.0 × 1.5 m"):
                st.image(str(ASSETS / "cueles_1x1.5.png"), width="stretch")
    log = res.log.df()
    for fase in sorted(log.Fase.unique(), key=lambda f: f.split(" ")[0]):
        with st.expander(fase, expanded=True):
            st.dataframe(log[log.Fase == fase].drop(columns="Fase"), hide_index=True, width="stretch")
    if res.p.get("burden_table") is not None:
        with st.expander("Comparación de criterios de burden (mismo Ø y explosivo)", expanded=False):
            st.dataframe(res.p["burden_table"].round(3), hide_index=True, width="stretch")
            st.caption("Konya, Ash, Torbica y Lapcevic, Pearse (PD de Cook) y McKenzie/Andersen: curso de taladros "
                       "largos; Rustan: Onederra y Chitombo (2007). Úselos como primera aproximación y valide con "
                       "pruebas de campo (Konya: ±10 %).")
    st.subheader("Distribución por zona")
    zt = H.groupby("zona", sort=False).agg(N=("id", "count"), B_m=("B", "mean"), S_m=("S", "mean"),
                                          L_m=("L", "mean"), kg_tal=("kg", "mean"), kg=("kg", "sum"))
    st.dataframe(zt.reindex([z for z in res.zone_order if z in zt.index]).round(3), width="stretch")

# =============================================================================
with tabs[1]:
    fp = res.frag
    if fp is None:
        st.info("Este diseño no tiene parámetros de fragmentación.")
    else:
        verify = fp["modo"] == "verificar"
        st.markdown(f"Se analiza la zona de **{fp['zona']}**. " + (
            "En chimeneas el P80 no se optimiza: se verifica que el bloque mayor no atasque la labor."
            if verify else "El burden puede optimizarse para alcanzar el P80 objetivo."))
        route = st.radio("Ruta", ["Ruta A — Kuz-Ram", "Ruta B — Modelo ML híbrido"], horizontal=True)
        c1, c2, c3 = st.columns(3)
        ref = c1.selectbox("Referencia (Carpio 6.1.7)", list(P80_REFERENCIAS))
        rng = P80_REFERENCIAS[ref]
        P80_t = c2.number_input("P80 objetivo (cm)", 2.0, 200.0, float(np.mean(rng)) if rng else 20.0, 0.5)
        B_lim = float(fp.get("B_limit") or fp["B"] * 1.5)
        SB_now = fp["S"] / fp["B"] if fp["B"] > 0 else 1.25
        B_min = c3.number_input("Burden mínimo a evaluar (m)", 0.1, max(0.15, B_lim - 0.05),
                                float(min(max(0.1, 0.4 * B_lim), B_lim - 0.1)), 0.05)
        best_B = best_SB = None
        if route.startswith("Ruta A"):
            c1, c2 = st.columns(2)
            fmodel = c1.radio("Distribución", ["Kuz-Ram (Rosin-Rammler)", "KCO (Swebrec, Ouchterlony 2005)"],
                              horizontal=True, help="KCO corrige la subestimación de finos de Kuz-Ram (paper "
                                                    "'Comparative analysis of rock fragmentation models').")
            xmax_in = None
            if fmodel.startswith("KCO"):
                xmax_in = c2.number_input("Tamaño máximo in situ x_max (cm; 0 = mín(B, S))", 0.0, 500.0, 0.0, 5.0) or None
            with st.expander("Exponentes de Kuznetsov (calibración local)", expanded=False):
                st.caption("X50 = A·K^(−e₁)·Q^(e₂)·(115/RWS)^(e₃). Valores clásicos 0.8, 1/6 y 19/30; el paper "
                           "'Reparametrización de la ecuación de Kuznetsov' ajusta estos exponentes con datos de mina.")
                cc = st.columns(3)
                ea = cc[0].number_input("e₁", 0.3, 1.5, 0.8, 0.01)
                eb = cc[1].number_input("e₂", 0.0, 0.6, round(1 / 6, 4), 0.01)
                ec = cc[2].number_input("e₃", 0.2, 1.2, round(19 / 30, 4), 0.01)
            with st.expander("Factor de roca A (Lilly)", expanded=False):
                mode = st.radio("Cálculo de A", ["Lilly", "Valor directo"], horizontal=True)
                if mode == "Lilly":
                    c1, c2, c3 = st.columns(3)
                    rmd = c1.selectbox("RMD", list(fr.RMD_OPTS), index=1)
                    jps = c2.selectbox("JPS", list(fr.JPS_OPTS), index=1)
                    jpa = c3.selectbox("JPA", list(fr.JPA_OPTS), index=1)
                    Arf, parts = fr.rock_factor(rmd, jps, jpa, rock.rho, rock.E, rock.UCS)
                    st.caption(f"A = {Arf:.2f} · " + " · ".join(f"{k} = {v:.1f}" for k, v in parts.items()))
                else:
                    Arf = st.number_input("A", 1.0, 22.0, 7.0, 0.1)
            if mode != "Lilly":
                pass
            k = fr.KuzRamInput(A=Arf, q=fp["q"], Lc=fp["Lc"], H=fp["H"], d_mm=fp["d_mm"], W=fp["W"],
                               rws_pct=fp["rws"], SB=SB_now, K_ref=fp.get("K_override"),
                               B_ref=fp["B"] if fp.get("K_override") else None,
                               model="KCO" if fmodel.startswith("KCO") else "Kuz-Ram", xmax_cm=xmax_in, ea=ea, eb=eb,
                               ec=ec)
            cur = fr.kuzram(fp["B"], k, fp["S"])
            p100 = fr.size_model(cur, 99)
            c = st.columns(6)
            c[0].metric("B actual", f"{fp['B']:.2f} m")
            c[1].metric("K", f"{cur['K']:.2f} kg/m³")
            c[2].metric("X50", f"{cur['X50']:.1f} cm")
            c[3].metric("n", f"{cur['n']:.2f}")
            c[4].metric("P80", f"{cur['P80']:.1f} cm", f"{cur['P80'] - P80_t:+.1f}", delta_color="inverse")
            c[5].metric("P99 (bloque mayor)", f"{p100:.0f} cm")
            if "min_dim" in fp:
                lim = 100 * fp["min_dim"] / fp["ratio"]
                ok = p100 <= lim
                (st.success if ok else st.error)(
                    f"Verificación de atascos: P99 = {p100:.0f} cm {'≤' if ok else '>'} dimensión mínima / "
                    f"{fp['ratio']:.0f} = {lim:.0f} cm.")
            st.latex(r"X_{50}=A\,K^{-e_1}Q^{e_2}\left(\tfrac{115}{RWS}\right)^{e_3}\;\;"
                     r"n=\left(2.2-14\tfrac{B}{d}\right)\sqrt{\tfrac{1+S/B}{2}}\left(1-\tfrac{W}{B}\right)\tfrac{L_c}{H}")
            if cur.get("model") == "KCO":
                st.latex(r"P(x)=\frac{1}{1+\left[\ln(x_{max}/x)/\ln(x_{max}/X_{50})\right]^{b}},\quad "
                         r"b=2\ln 2\,\ln(x_{max}/X_{50})\,n")
                st.caption(f"x_max = {cur['xmax']:.0f} cm · b = {cur['b']:.2f}")
            else:
                st.latex(r"P_{80}=X_{50}\left(\tfrac{\ln 5}{\ln 2}\right)^{1/n}")
            if res.p.get("fan_note"):
                st.caption(res.p["fan_note"])
            opt_r, curve = fr.optimize_burden_kuzram(k, P80_t, B_min, B_lim)
            g1, g2 = st.columns(2)
            f1 = go.Figure()
            f1.add_trace(go.Scatter(x=curve.B, y=curve.P80, name="P80(B)", line=dict(width=3, color="#c2416f")))
            f1.add_trace(go.Scatter(x=curve.B, y=curve.X50, name="X50(B)", line=dict(dash="dot", color="#888")))
            f1.add_hline(y=P80_t, line=dict(dash="dash", color="#2ca02c"), annotation_text="Objetivo")
            if rng:
                f1.add_hrect(y0=rng[0], y1=rng[1], fillcolor="#2ca02c", opacity=0.08, line_width=0)
            f1.add_vline(x=B_lim, line=dict(dash="dash", color="#d62728"), annotation_text="Límite")
            f1.add_trace(go.Scatter(x=[opt_r["B"]], y=[opt_r["P80"]], mode="markers", name="Óptimo",
                                    marker=dict(size=14, symbol="star", color="#ff7f0e")))
            f1.update_layout(height=360, xaxis_title="Burden B (m)", yaxis_title="cm", plot_bgcolor="white",
                             margin=dict(l=10, r=10, t=30, b=10))
            g1.plotly_chart(f1, width="stretch")
            f2 = go.Figure()
            for lab, rr, col in [("Actual", cur, "#888"), ("Óptimo", opt_r, "#c2416f")]:
                x, R = fr.model_curve(rr, xmax=max(cur["P80"], opt_r["P80"]) * 2.2)
                f2.add_trace(go.Scatter(x=x, y=R, name=f"{lab}: P80 = {rr['P80']:.1f} cm", line=dict(color=col, width=3)))
            f2.add_hline(y=80, line=dict(dash="dot"))
            f2.update_layout(height=360, xaxis_title="Tamaño (cm)", yaxis_title="% pasante", plot_bgcolor="white",
                             margin=dict(l=10, r=10, t=30, b=10))
            g2.plotly_chart(f2, width="stretch")
            if not verify:
                if opt_r["converged"]:
                    st.success(f"B óptimo = {opt_r['B']:.2f} m · S = {opt_r['S']:.2f} m → P80 = {opt_r['P80']:.1f} cm.")
                else:
                    st.warning(f"El objetivo no se alcanza en [{B_min:.2f}, {B_lim:.2f}] m; lo más cercano es "
                               f"B = {opt_r['B']:.2f} m (P80 = {opt_r['P80']:.1f} cm).")
                best_B, best_SB = opt_r["B"], SB_now
        else:
            st.markdown("Cargue su modelo (.joblib / .pkl). Sólo archivos propios o de confianza.")
            up = st.file_uploader("Modelo", type=["joblib", "pkl", "pickle"])
            demo = st.checkbox("Usar el sustituto de DEMOSTRACIÓN (no es un modelo real)", value=up is None)
            model = None
            if up is not None and not demo:
                try:
                    import joblib
                    model = joblib.load(io.BytesIO(up.getvalue()))
                except Exception:  # noqa: BLE001
                    try:
                        model = pickle.loads(up.getvalue())
                    except Exception as e:  # noqa: BLE001
                        st.error(f"No se pudo cargar el modelo: {e}")
            elif demo:
                model = fr.DemoSurrogate()
            if model is not None:
                c1, c2, c3 = st.columns(3)
                target_kind = c1.radio("Predice", ["P80", "X50"], horizontal=True)
                unit = c2.selectbox("Unidad", ["cm", "mm", "m"])
                scale = {"cm": 1.0, "mm": 0.1, "m": 100.0}[unit]
                n_u = c3.number_input("n (si predice X50)", 0.5, 3.0, 1.5, 0.05)
                st.caption(f"Use un modelo entrenado con datos de **{labor.lower()}**: no mezcle avance y producción.")
                base_vals = dict(d_mm=fp["d_mm"], q=fp["q"], H=fp["H"], Lc=fp["Lc"], Lt=fp["H"] - fp["Lc"], RMR=rock.RMR,
                                 GSI=rock.GSI, RQD=rock.RQD, UCS=rock.UCS, rho_r=rock.rho, VOD=main_exp.vod, rho_e=main_exp.rho,
                                 RWS=main_exp.rws * 100, A_sec=S.get("Área de la sección (m²)", 0.0))
                names = fr.model_feature_names(model)
                if names and [f for f in names if f not in fr.ML_VARIABLES]:
                    st.error("Variables no disponibles: " + ", ".join(f for f in names if f not in fr.ML_VARIABLES))
                    names = None
                if not names:
                    names = st.multiselect("Orden de las variables", list(fr.ML_VARIABLES),
                                           default=["B", "S", "d_mm", "q", "Lc", "H", "RMR", "RWS"])
                if names:
                    v_now = fr.ml_features(fp["B"], SB_now, base_vals)
                    p_now, _ = fr.ml_predict(model, names, v_now, scale, target_kind, n_u)
                    st.metric("P80 con el diseño actual", f"{p_now:.1f} cm", f"{p_now - P80_t:+.1f}",
                              delta_color="inverse")
                    if not verify:
                        c1, c2 = st.columns(2)
                        SBr = c1.slider("Rango S/B", 1.0, 2.0, (1.0, 1.6), 0.05)
                        lam = c2.slider("Peso del costo λ", 0.0, 1.0, 0.15, 0.05)
                        mkey = (labor, res.metodo, P80_t, B_min, round(B_lim, 3), SBr, lam, tuple(names), unit, target_kind)
                        if st.button("Optimizar con Evolución Diferencial"):
                            with st.spinner("Optimizando…"):
                                ss.ml_res = (mkey,) + fr.optimize_ml(model, names, base_vals, P80_t, (B_min, B_lim), SBr,
                                                                     scale, target_kind, n_u, lam, 0)
                        if ss.get("ml_res") and ss.ml_res[0] == mkey:
                            _, rr, hist = ss.ml_res
                            c = st.columns(3)
                            c[0].metric("B óptimo", f"{rr['B']:.2f} m")
                            c[1].metric("S/B", f"{rr['SB']:.2f}")
                            c[2].metric("P80", f"{rr['P80']:.1f} cm")
                            best_B, best_SB = rr["B"], rr["SB"]
        if not verify:
            c1, c2 = st.columns(2)
            if c1.button("Aplicar burden óptimo al diseño", type="primary", disabled=best_B is None):
                ss.overrides[key] = (float(best_B), float(best_SB))
                st.rerun()
            if c2.button("Restablecer burden del método", disabled=key not in ss.overrides):
                ss.overrides.pop(key, None)
                st.rerun()

# =============================================================================
with tabs[2]:
    c1, c2, c3 = st.columns(3)
    label = c1.selectbox("Etiqueta", ["id", "periodo", "kg", "ninguna"], format_func={
        "id": "N° de taladro", "periodo": "N° de retardo", "kg": "Carga (kg)", "ninguna": "Sin etiqueta"}.get)
    vec = c2.checkbox("Vectores de salida (con secuencia)", False)
    opn = c3.checkbox("Aberturas del cuele", True)
    st.plotly_chart(viz.view_2d(res, label, ss.seq, vec and ss.seq is not None, opn), width="stretch")
    fsl = viz.slot_plan(res)
    if fsl is not None:
        st.plotly_chart(fsl, width="stretch")
    fsv = viz.section_view(res)
    if fsv is not None:
        st.plotly_chart(fsv, width="stretch")
        st.caption("Cada taladro paralelo se perfora con el pivote sobre la prolongación de su eje, a la altura del "
                   "pivote sobre el piso de la labor; el carrusel gira con el 'dip' del taladro y, si el rectángulo "
                   "girado no cabe en la labor (choca con paredes o techo), el taladro queda fuera del alcance (rojo).")

with tabs[3]:
    c1, c2, c3, c4 = st.columns(4)
    sh = c1.checkbox("Labor / bloque", True)
    si = c2.checkbox("Iniciación", True)
    st_ = c3.checkbox("Taco / retacado", True)
    zf = c4.multiselect("Zonas", res.zone_order, default=[], placeholder="Todas")
    st.plotly_chart(viz.view_3d(res, system, primer, sh, si, st_, zf or None, ss.seq), width="stretch")
    st.plotly_chart(viz.hole_profiles(res, primer), width="stretch")

# =============================================================================
with tabs[4]:
    st.markdown("Energía de cada taladro y su distribución en el macizo. La concentración 3D de explosivo sigue a "
                "**Kleine (1988)** como la presentan **Onederra y Chitombo (2007)** —es la base de los planos de energía "
                "de JKSimBlast/2D Ring que pide el curso—; la PPV de campo cercano es la de **Holmberg y Persson (1980)**.")
    etab = en.hole_energy_table(res, emap, main_exp.energy, rock.rho)
    vol_s = S.get("Volumen roto (m³)", np.nan)
    c = st.columns(4)
    c[0].metric("Energía total de la malla", f"{etab.E_MJ.sum():,.0f} MJ")
    c[1].metric("Energía media por taladro", f"{etab.E_MJ.mean():.1f} MJ")
    c[2].metric("Energía lineal media", f"{etab.E_lineal_MJm.mean():.2f} MJ/m")
    c[3].metric("Factor de energía (voladura)",
                f"{S['Explosivo total (kg)'] * main_exp.energy / (vol_s * rock.rho):.2f} MJ/t" if vol_s == vol_s else "—")
    g1, g2 = st.columns([1.25, 1])
    g1.plotly_chart(ve.energy_bars(etab, res.zone_colors), width="stretch")
    g2.dataframe(etab.round(3), hide_index=True, width="stretch", height=360)
    st.divider()
    scope = "voladura"
    if labor == "Producción" and not res.p.get("blast_all"):
        scope = st.radio("Taladros considerados", ["voladura", "malla"], horizontal=True,
                         format_func={"voladura": f"Una voladura ({res.p.get('rows_blast', 1)} fila(s) o anillo(s))",
                                      "malla": "Toda la malla dibujada"}.get,
                         help="La concentración de Kleine suma todas las cargas: con toda la malla crece con el tamaño "
                              "del patrón. Onederra la evalúa sobre el anillo o la voladura.")
    seg = en.segments(res, emap, main_exp.energy, en.blast_holes(res, scope))
    frame = en.plane_frame(res)
    c1, c2, c3 = st.columns([1.3, 1, 1])
    field = c1.radio("Variable", ["Concentración de explosivo (kg/t)", "Concentración de energía (MJ/t)",
                                  "Factor de carga 3D (kg/m³)", "PPV de campo cercano — Holmberg–Persson (mm/s)"])
    pos = c2.slider(frame["name"] + " (m)", float(frame["lo"]), float(frame["hi"]), float(frame["default"]),
                    float(max((frame["hi"] - frame["lo"]) / 100, 0.01)))
    res_n = c3.select_slider("Resolución", [80, 110, 140, 180], value=110)
    thr = []
    pf_kgt = S["Factor de carga (g/t)"] / 1000 if S.get("Factor de carga (g/t)") == S.get("Factor de carga (g/t)") else 0.5
    if field.startswith("PPV"):
        cc = st.columns(4)
        Khp = cc[0].number_input("K (H–P)", 50.0, 2000.0, 700.0, 10.0, help="Holmberg–Persson: K = 700, α = 0.7 (roca "
                                                                           "dura sueca); la plantilla del curso usa K = 470, α = 0.94.")
        ahp = cc[1].number_input("α (H–P)", 0.3, 1.5, 0.7, 0.01)
        ppvc = rock.ppv_crit
        cc[2].metric("VPP crítica = σt·Vp/E", f"{ppvc:,.0f} mm/s")
        brk = cc[3].number_input("Umbral de rotura (mm/s)", 100.0, 20000.0, float(round(4 * ppvc, -1)), 50.0,
                                 help="Plantilla del curso: ≈ 4 × VPP crítica.")
        thr = [(ppvc, "Inicio de daño", "#1f3b73"), (brk, "Rotura", "#2ca02c")]
        for c in (res.p.get("domains") or {}).get("contacts", []):
            thr.append((c["thr"], f"Sobrerotura en {c['name'].lower()}", c["color"]))
        unit, title = "mm/s", "PPV de Holmberg–Persson (envolvente)"
    else:
        unit = {"Concentración de explosivo (kg/t)": "kg/t", "Concentración de energía (MJ/t)": "MJ/t",
                "Factor de carga 3D (kg/m³)": "kg/m³"}[field]
        base = {"kg/t": pf_kgt, "MJ/t": pf_kgt * main_exp.energy, "kg/m³": pf_kgt * rock.rho}[unit]
        tv = st.number_input(f"Umbral a resaltar ({unit})", 0.001, 1000.0, float(round(base, 3)), float(round(base / 20, 4) or 0.01),
                             help="Por defecto, el factor de carga del diseño. Onederra y Chitombo: rotura en el pie de "
                                  "los anillos con 0.25–0.35 kg/t.")
        thr = [(tv, "Umbral", "#1f3b73")]
        title = field.split(" (")[0]
    ekey = (sig, scope, field, round(pos, 3), res_n, tuple(round(t[0], 4) for t in thr),
            (Khp, ahp) if field.startswith("PPV") else None)
    if ekey not in ss.emem:
        uu, vv, PP = en.plane_grid(res, frame, pos, nx=res_n)
        if field.startswith("PPV"):
            F = en.hp_ppv(PP, seg, Khp, ahp)
        else:
            F = en.kleine(PP, seg, rock.rho, energy=(unit == "MJ/t")) * (rock.rho if unit == "kg/m³" else 1.0)
        if len(ss.emem) > 12:
            ss.emem.clear()
        ss.emem[ekey] = (uu, vv, F)
    uu, vv, F = ss.emem[ekey]
    vmax = None if field.startswith("PPV") else float(min(np.nanpercentile(F, 97), 4 * thr[0][0]))
    st.plotly_chart(ve.energy_map_2d(res, uu, vv, F, frame, pos, title, unit, thr, vmax=vmax,
                                     log=field.startswith("PPV")), width="stretch")
    view_ax = (0, 2) if res.labor == "Avance" else ((0, 1) if (res.labor in ("Chimenea", "Pique") or
                                                               res.view2d.get("kind") == "plan") else (1, 2))
    ss.energy_last = dict(uu=uu, vv=vv, F=F, title=f"{title} — {frame['name']} = {pos:.2f} m", unit=unit,
                          thresholds=[(t[0], t[1], t[2]) for t in thr], labels=frame["labels"], vmax=vmax,
                          outline=res.view2d.get("outline") if frame["axes"] == view_ax else None, sig=sig)
    within = (F >= thr[0][0]).mean() * 100
    if field.startswith("PPV"):
        st.caption(f"{within:.0f} % del plano supera la VPP crítica y {(F >= thr[min(1, len(thr) - 1)][0]).mean() * 100:.0f} % el umbral "
                   "de rotura. Se grafica la envolvente de la PPV de cada taladro (salen en retardos distintos). Calibre "
                   "K y α con monitoreos de campo cercano.")
    else:
        st.caption(f"{within:.0f} % del plano supera el umbral de {thr[0][0]:.3g} {unit}. El valor es una concentración "
                   "equivalente a una esfera centrada en cada elemento de carga (Kleine): en el plano del burden se "
                   "aproxima al factor de carga. No considera los tiempos de retardo (análisis estático 3D).")
    st.subheader("Vista 3D de la energía")
    c1, c2 = st.columns(2)
    show_iso = c1.checkbox("Mostrar la zona que supera el umbral (isosuperficie)", False)
    iso = None
    if show_iso:
        ikey = ("iso", sig, scope, field, tuple(round(t[0], 4) for t in thr))
        if ikey not in ss.emem:
            X, Y, Z, PV = en.volume_grid(seg, n=26)
            FV = (en.hp_ppv(PV, seg, Khp, ahp) if field.startswith("PPV")
                  else en.kleine(PV, seg, rock.rho, energy=(unit == "MJ/t")) * (rock.rho if unit == "kg/m³" else 1.0))
            ss.emem[ikey] = (X, Y, Z, FV)
        X, Y, Z, FV = ss.emem[ikey]
        iso = (X, Y, Z, FV, thr[min(1, len(thr) - 1)][0], unit)
    st.plotly_chart(ve.energy_3d(res, seg, iso, c2.checkbox("Contorno de la labor", True)), width="stretch")

# =============================================================================
with tabs[5]:
    prod = labor == "Producción"
    c1, c2, c3 = st.columns(3)
    with c1:
        meth = st.radio("Método", ["Por filas", "MILP", "Heurística"] if prod else ["MILP", "Heurística"],
                        format_func={"Por filas": "Por filas (Du Pont)", "MILP": "MILP exacto (HiGHS)",
                                     "Heurística": "Heurística"}.get)
        tl = st.slider("Tiempo límite MILP (s)", 5, 120, 20) if meth == "MILP" else 20
        classic = st.checkbox("Respetar el orden por zonas / filas", True)
        cut_single = st.checkbox("Un taladro por retardo en el cuele", not prod)
        sim_on = st.checkbox("Zonas simultáneas (corona controlada, cuele instantáneo)", True)
    with c2:
        nmax = st.number_input("Máx. taladros por retardo", 1, 100, 4 if not prod else 10)
        qm = st.radio("Carga máxima por retardo", ["Directa", "Por vibraciones (PPV)"], horizontal=True)
        if qm == "Directa":
            qmax = st.number_input("Q máx (kg)", 0.5, 5000.0, 25.0 if not prod else 400.0, 0.5)
        else:
            Ks = st.number_input("K", 50.0, 3000.0, 267.57, 1.0, help="Carpio cap. V: K = 267.57, α = 1.62")
            beta = st.number_input("α / β", 0.8, 2.5, 1.62, 0.01)
            Dm = st.number_input("Distancia (m)", 2.0, 1000.0, 30.0)
            ppv = st.number_input("PPV admisible (mm/s)", 1.0, 1000.0, 340.0)
            qmax = sq.qmax_from_ppv(ppv, Dm, Ks, beta)
            st.caption(f"Q máx = {qmax:.1f} kg")
        link = st.slider("Radio de alivio (× B)", 1.2, 3.0, 1.6, 0.1)
    with c3:
        cfg = dict(ms_m=ms_por_metro(rock.RMR))
        carpio_gap = False
        if system.startswith("Carmex"):
            cfg.update(burn=st.number_input("Combustión de la mecha (s/m)", 80.0, 250.0, 150.0, 5.0),
                       dt_s=st.number_input("Retardo entre salidas (s)", 0.5, 10.0, 2.0, 0.5),
                       tail=st.number_input("Mecha fuera del taladro (m)", 0.1, 1.5, 0.35, 0.05),
                       primer=primer, Lt=float(H.Lt.median()))
        elif system == "Electrónico":
            cfg.update(t0=st.number_input("t inicial (ms)", 0.0, 1000.0, 0.0),
                       dt_ms=st.number_input("Intervalo (ms)", 1.0, 1000.0, 25.0 if prod else 50.0))
        elif system.endswith("tabla propia"):
            txt = st.text_area("Tiempos (ms)", ", ".join(str(t) for t in EXSANEL_LP))
            cfg["table"] = [float(t) for t in txt.replace(";", ",").split(",") if t.strip()]
        else:
            st.dataframe(pd.DataFrame({"N°": range(1, 21), "Exsanel MS": EXSANEL_MS, "Exsanel LP": EXSANEL_LP,
                                       "Fanel MS": FANEL_MS}), hide_index=True, height=200)
        if not prod and not system.startswith(("Carmex", "Electrónico")):
            carpio_gap = st.checkbox(f"Tiempo mínimo entre taladros del arranque (Carpio Tabla 04: "
                                     f"{cfg['ms_m']} ms/m × L)", True)
        if prod:
            cfg.update(row_ms=st.number_input("Entre filas (ms)", 1.0, 500.0, 75.0),
                       hole_ms=st.number_input("Entre barrenos de una fila (ms)", 0.0, 200.0, 10.0))
    q_unit = H.cargas.map(sq.deck_max) if meth == "Por filas" else H.kg
    if st.button("Calcular secuencia", type="primary"):
        if q_unit.max() > qmax:
            st.error(f"Una carga de {q_unit.max():.1f} kg supera Q máx ({qmax:.1f} kg)"
                     + (" — reduzca la carga por sección del taladro." if meth == "Por filas" else "."))
        else:
            opt = dict(classic=classic, qmax=qmax, nmax=int(nmax), cut_single=cut_single, method=meth, time_limit=tl,
                       link=link, carpio_gap=carpio_gap, simultaneous=res.simultaneous if sim_on else set())
            with st.spinner("Calculando…"):
                seq, info = sq.run_sequencing(H, res.sources, opt, system, cfg, rock.RMR)
            ss.seq, ss.seq_sig, ss.seq_info, ss.seq_qmax = seq, sig, info, qmax
            st.rerun()
    if ss.seq is not None:
        seq, info = ss.seq, ss.seq_info
        c = st.columns(5)
        c[0].metric("Retardos", info["n_periods"])
        c[1].metric("Duración", f"{seq.t_ms.max() / 1000:.2f} s")
        per_delay = (seq.groupby("t_ms").kg_carga.sum() if "Por filas" in info["method"]
                     else seq.groupby("periodo").kg.sum())
        c[2].metric("Carga máx./retardo", f"{per_delay.max():.1f} kg")
        c[3].metric("Radio de alivio", f"{info['link']:.2f}·B")
        c[4].metric("Mín. entre cueles", f"{info['gap_ms']:.0f} ms" if carpio_gap else "—")
        st.caption("Método: " + info["method"] + (f" · {info.get('milp_status', '')}" if "milp_status" in info else ""))
        g1, g2 = st.columns([1.1, 1])
        g1.plotly_chart(viz.sequence_animation(res, seq), width="stretch")
        g2.plotly_chart(viz.view_2d(res, "periodo", seq, True, False, height=620), width="stretch")
        if "Por filas" in info["method"]:
            dd = seq.assign(periodo=seq.t_ms.rank(method="dense").astype(int), kg=seq.kg_carga)
            st.caption("Por filas: cada barreno (y cada sección de carga, +50 ms) sale en un tiempo propio; "
                       "se grafica la carga por instante de detonación.")
            st.plotly_chart(viz.charge_per_delay(dd, ss.seq_qmax), width="stretch")
        else:
            st.plotly_chart(viz.charge_per_delay(seq, ss.seq_qmax), width="stretch")
        cols = ["orden", "id", "zona", "periodo", "n_detonador", "t_ms", "predecesor", "kg", "x", "y"] + \
               (["mecha_m"] if "mecha_m" in seq else [])
        st.dataframe(seq[cols].round(3), hide_index=True, width="stretch")
        st.download_button("Descargar secuencia (CSV)", seq[cols].to_csv(index=False).encode("utf-8"),
                           "secuencia.csv", "text/csv")
        with st.expander("Daño al macizo: VPP crítica y ley de Devine (curso de taladros largos)"):
            c1, c2, c3 = st.columns(3)
            Kd_ = c1.number_input("K (Devine)", 10.0, 5000.0, 420.14, 1.0, key="kdev",
                                  help="PPV = K·(D/√Q)^(−α). Plantilla del curso (Ejemplo 3): K = 420.14, α = 2.081.")
            ad_ = c2.number_input("α (Devine)", 0.5, 3.5, 2.081, 0.01, key="adev")
            ppv_c = c3.number_input("VPP crítica (mm/s)", 10.0, 20000.0, float(round(rock.ppv_crit, 0)), 10.0,
                                    help="σt·Vp/E con los datos del macizo (barra lateral).")
            Qd = float(per_delay.max())
            Ddam = math.sqrt(Qd) * (Kd_ / ppv_c) ** (1 / ad_)
            st.write(f"Con Q = {Qd:.1f} kg por retardo, la PPV supera la VPP crítica ({ppv_c:,.0f} mm/s) hasta "
                     f"**{Ddam:.2f} m** del disparo; a 4 × VPP crítica (rotura) hasta "
                     f"**{math.sqrt(Qd) * (Kd_ / (4 * ppv_c)) ** (1 / ad_):.2f} m**.")
            dd = np.linspace(max(0.3, 0.1 * Ddam), max(4 * Ddam, 5), 60)
            figd = go.Figure(go.Scatter(x=dd, y=Kd_ * (dd / math.sqrt(Qd)) ** (-ad_), name="PPV"))
            figd.add_hline(y=ppv_c, line=dict(dash="dash", color="#d62728"), annotation_text="VPP crítica")
            figd.update_layout(height=280, xaxis_title="Distancia (m)", yaxis_title="PPV (mm/s)", yaxis_type="log",
                               plot_bgcolor="white", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(figd, width="stretch")
        with st.expander("Daño al shotcrete (Carpio cap. V)"):
            c1, c2, c3 = st.columns(3)
            fc = c1.number_input("f'c del shotcrete (MPa)", 10.0, 60.0, 23.5, 0.5)
            Ksh = c2.number_input("K del sitio", 50.0, 2000.0, 267.57)
            ash = c3.number_input("α del sitio", 0.8, 2.5, 1.62, 0.01)
            Qd = float(per_delay.max())
            r_sh = sq.shotcrete(fc, None, 0.2, 2400, Ksh, ash, Qd)
            st.write(f"E = {r_sh['E']:.1f} GPa · Vp = {r_sh['Vp']:.0f} m/s · V crítica = {r_sh['Vcrit']:.0f} mm/s · "
                     f"con Q = {Qd:.1f} kg por retardo, el shotcrete se daña hasta **d = {r_sh['d']:.2f} m** del disparo.")

# =============================================================================
with tabs[6]:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("Indicadores")
        st.dataframe(pd.DataFrame({"Indicador": list(S), "Valor": [round(float(v), 3) for v in S.values()]}),
                     hide_index=True, width="stretch")
    with c2:
        st.subheader("Carga por zona")
        zt = H.groupby("zona", sort=False).agg(Taladros=("id", "count"), Cartuchos=("n_cart", "sum"),
                                              Explosivo=("explosivo", "first"), kg=("kg", "sum"))
        st.dataframe(zt.round(2), width="stretch")
    cols = ["id", "zona", "x", "y", "X0", "Y0", "Z0", "ux", "uy", "uz", "L", "B", "S", "explosivo", "n_cart", "kg"] + \
        (["alcance"] if "alcance" in H else [])
    st.dataframe(H[cols].round(3), hide_index=True, width="stretch")
    d1c, d2c, d3c = st.columns(3)
    d1c.download_button("Taladros (CSV)", H[cols].to_csv(index=False).encode("utf-8"), "taladros.csv", "text/csv")
    d2c.download_button("Memoria de cálculo (CSV)", res.log.df().to_csv(index=False).encode("utf-8"),
                        "memoria_calculo.csv", "text/csv")
    d3c.download_button("Resumen (JSON)", json.dumps({"labor": labor, "metodo": res.metodo,
                                                      "resumen": {k: float(v) for k, v in S.items()}},
                                                     ensure_ascii=False, indent=2).encode("utf-8"),
                        "resumen.json", "application/json")
    st.divider()
    st.subheader("Exportar el diseño a PDF")
    st.caption("Plano acotado de la malla con el origen de coordenadas, columnas de carga por zona, mapa de energía y "
               "cuadro de taladros (ubicación, orientación, longitud, taco, carga, explosivo y retardo).")
    c1, c2, c3 = st.columns(3)
    info = dict(proyecto=c1.text_input("Proyecto", "Diseño de perforación y voladura"),
                mina=c2.text_input("Mina / unidad", ""), responsable=c3.text_input("Elaborado por", ""),
                revisor=c1.text_input("Revisado por", ""), labor=labor,
                observaciones=c2.text_input("Observaciones", ""))
    inc_e = c3.checkbox("Incluir el mapa de energía (pestaña Energía)", True)
    inc_s = c3.checkbox("Incluir la malla de tiempos y retardos", True,
                        help="Si no se calculó la secuencia, se usa una por defecto (heurística; por filas en producción).")
    if st.button("Generar PDF", type="primary"):
        el = ss.get("energy_last")
        seq_pdf = ss.seq if ss.seq is not None else None
        if seq_pdf is None and inc_s:
            try:      # secuencia por defecto (heurística o por filas) para la malla de tiempos del PDF
                opt_d = dict(classic=classic, qmax=max(float(qmax), float(q_unit.max()) + 1e-6), nmax=int(nmax),
                             cut_single=cut_single, method="Por filas" if prod else "Heurística", time_limit=tl,
                             link=link, carpio_gap=carpio_gap, simultaneous=res.simultaneous if sim_on else set())
                seq_pdf, _ = sq.run_sequencing(H, res.sources, opt_d, system, cfg, rock.RMR)
            except Exception:  # noqa: BLE001
                seq_pdf = None
        with st.spinner("Generando el PDF…"):
            pdf = report.build_pdf(res, info, pdf_params + [("Modelo del macizo", rock_mode + ("" if not multi else
                                                                               f" — labor en: {rock_where}")),
                                                            ("Macizo de diseño (RMR · RQD · σci · ρ)",
                                                             f"{rock.RMR} · {rock.RQD} % · {rock.UCS:.0f} MPa · "
                                                             f"{rock.rho:.2f} t/m³"),
                                                            ("Iniciación", system)],
                                   seq=seq_pdf if inc_s else None,
                                   energy=el if (inc_e and el and el.get("sig") == sig) else None, system=system)
        ss.pdf_bytes = pdf
    if ss.get("pdf_bytes"):
        st.download_button("Descargar PDF del diseño", ss.pdf_bytes, f"diseno_{labor.lower()}.pdf", "application/pdf")
