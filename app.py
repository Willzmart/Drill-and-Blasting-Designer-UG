"""Diseño de mallas de perforación y voladura subterránea — avance, chimeneas, piques y producción.

Fuentes: Carpio Chávez (2025) y López Jimeno et al. (IGME, 1987).
Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import fragmentation as fr
from core import sequencing as sq
from core import viz
from core.avance import CHARGE_METHODS, AvanceOptions, Drill, run_avance
from core.common import (CARTRIDGES, EXPLOSIVES, EXSANEL_LP, EXSANEL_MS, P80_REFERENCIAS, Rock, make_explosive,
                         ms_por_metro)
from core.cuts import ANGLED_CUTS, CUT_NOTES, CUT_TYPES, PARALLEL_CUTS
from core.produccion import CE0_TABLE, CRITERIA, DEFAULTS, PROD_METHODS, ProdOptions, run_produccion
from core.sections import GALLERY_TYPES, PRESETS, VERTICAL_TYPES, build_section, build_vertical_section
from core.vertical import CHIMENEA_METHODS, PIQUE_METHODS, VerticalOptions, run_vertical

st.set_page_config(page_title="Diseño de voladura subterránea", layout="wide")
ASSETS = Path(__file__).parent / "assets"
ss = st.session_state
ss.setdefault("overrides", {})
ss.setdefault("seq", None)
ss.setdefault("seq_sig", None)

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
                      rws=c1.number_input("RWS (% ANFO)", 40, 180, 90, key=key + "w") / 100)
        bulk = c2.checkbox("A granel", False, key=key + "g")
        if bulk:
            custom.update(cart="granel", w_cart=0.0, L_cart=1.0, granel=True)
        else:
            custom.update(cart="—", w_cart=c1.number_input("Peso cartucho (kg)", 0.02, 2.0, 0.25, key=key + "c"),
                          L_cart=c2.number_input("Long. cartucho (m)", 0.1, 0.8, 0.3048, key=key + "l"), granel=False)
        return make_explosive("Personalizado", custom=custom)
    if EXPLOSIVES[name]["granel"]:
        return make_explosive(name)
    cart = st.selectbox("Cartucho", list(CARTRIDGES), index=list(CARTRIDGES).index(cart_default), key=key + "k")
    return make_explosive(name, cart)


with st.sidebar.expander("Macizo rocoso", expanded=False):
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
rock = Rock(RMR=RMR, GSI=GSI, RQD=RQD, UCS=UCS, rho=rho_r, Q=Qb, JSF=JSF, E=Emod)

with st.sidebar.expander("Iniciación", expanded=False):
    system = st.selectbox("Sistema de iniciación", sq.SYSTEMS, index=2 if labor != "Producción" else 4)
    primer = st.radio("Posición del cebo", ["Fondo", "Boca"], horizontal=True)
with st.sidebar.expander("Costos unitarios", expanded=False):
    cost = dict(drill=st.number_input("Perforación (US$/m)", 0.0, 100.0, 2.5, 0.1),
                exp=st.number_input("Explosivo (US$/kg)", 0.0, 50.0, 4.0, 0.1),
                acc=st.number_input("Accesorios por taladro (US$)", 0.0, 100.0, 3.0, 0.1))

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
            default_cut = "Quemado simple" if narrow else CUT_TYPES[0]
            cut_type = st.selectbox("Tipo de cuele", CUT_TYPES, index=CUT_TYPES.index(default_cut))
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
                             narrow=narrow, B_prod_override=ov[0], SB_prod=ov[1] or 1.25)
        res = run_avance(section, rock, drill, exm, exc, opts, cost)
        main_exp = exm
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
                             SB=ov[1] or 1.25)
        res = run_vertical(labor, vs, rock, vexp, vo, cost)
        main_exp = vexp
    else:
        with st.sidebar.expander("Método de explotación", expanded=True):
            method = st.selectbox("Método", PROD_METHODS)
            Dd, crit_d = DEFAULTS[method]
            criterion = st.selectbox("Criterio de burden", CRITERIA, index=CRITERIA.index(crit_d))
            po = ProdOptions(method=method, criterion=criterion)
            c1, c2 = st.columns(2)
            po.D_mm = c1.number_input("Ø perforación (mm)", 25.0, 250.0, float(Dd), 1.0)
            geo = {PROD_METHODS[0]: (30.0, 15.0, 40.0), PROD_METHODS[1]: (20.0, 1.5, 15.0),
                   PROD_METHODS[2]: (20.0, 12.0, 15.0), PROD_METHODS[3]: (20.0, 8.0, 40.0),
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
            if method == PROD_METHODS[1]:
                po.dip_deg = st.slider("Buzamiento de la veta (°)", 45, 90, 90)
            if method == PROD_METHODS[2]:
                po.fan_type = st.radio("Abanico", ["Ascendente", "Descendente", "Ambos sentidos"], horizontal=True)
                if po.fan_type != "Ascendente":
                    po.down_h = st.number_input("Altura del abanico descendente (m)", 1.0, 40.0, 10.0, 0.5)
                c1, c2 = st.columns(2)
                po.drift_w = c1.number_input("Ancho de galería (m)", 2.0, 6.0, 4.0, 0.1)
                po.drift_h = c2.number_input("Alto de galería (m)", 2.0, 6.0, 4.0, 0.1)
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
        with st.sidebar.expander("Parámetros del esquema", expanded=False):
            c1, c2 = st.columns(2)
            fdef = {PROD_METHODS[5]: 1.45, PROD_METHODS[6]: 0.85, PROD_METHODS[7]: 1.20}.get(method, 1.0)
            po.f = c1.number_input("Factor de fijación f", 0.8, 1.6, fdef, 0.05,
                                   help="Langefors: vertical 1.0; 3:1 0.9; 2:1 0.85. Holmberg: arriba/horizontal 1.45; abajo 1.20")
            if method != PROD_METHODS[2]:
                po.SB = c2.number_input("S/B", 1.0, 2.0, 1.25, 0.05)
            po.rows_blast = c1.number_input("Filas por voladura", 1, 50, 3)
            po.col_ratio = c2.number_input("Columna / fondo", 0.3, 1.0, 0.5 if method != PROD_METHODS[2] else 0.6, 0.05)
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

tabs = st.tabs(["Memoria de cálculo", "Fragmentación (P80)", "Vista 2D", "Vista 3D y carguío", "Secuenciamiento",
                "Resumen y exportación"])

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
    for fase in log.Fase.unique():
        with st.expander(fase, expanded=True):
            st.dataframe(log[log.Fase == fase].drop(columns="Fase"), hide_index=True, width="stretch")
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
                               B_ref=fp["B"] if fp.get("K_override") else None)
            cur = fr.kuzram(fp["B"], k, fp["S"])
            p100 = fr.size_at(cur["X50"], cur["n"], 99)
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
            st.latex(r"X_{50}=A\,K^{-0.8}Q^{1/6}\left(\tfrac{115}{RWS}\right)^{19/30}\;\;"
                     r"n=\left(2.2-14\tfrac{B}{d}\right)\sqrt{\tfrac{1+S/B}{2}}\left(1-\tfrac{W}{B}\right)\tfrac{L_c}{H}"
                     r"\;\;P_{80}=X_{50}\left(\tfrac{\ln 5}{\ln 2}\right)^{1/n}")
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
                x, R = fr.rosin_rammler_curve(rr["X50"], rr["n"], xmax=max(cur["P80"], opt_r["P80"]) * 2.2)
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
                base_vals = dict(d_mm=fp["d_mm"], q=fp["q"], H=fp["H"], Lc=fp["Lc"], Lt=fp["H"] - fp["Lc"], RMR=RMR,
                                 GSI=GSI, RQD=RQD, UCS=UCS, rho_r=rho_r, VOD=main_exp.vod, rho_e=main_exp.rho,
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
        cfg = dict(ms_m=ms_por_metro(RMR))
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
            st.dataframe(pd.DataFrame({"N°": range(1, 21), "MS (ms)": EXSANEL_MS, "LP (ms)": EXSANEL_LP}),
                         hide_index=True, height=200)
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
                seq, info = sq.run_sequencing(H, res.sources, opt, system, cfg, RMR)
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
with tabs[5]:
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
    cols = ["id", "zona", "x", "y", "X0", "Y0", "Z0", "ux", "uy", "uz", "L", "B", "S", "explosivo", "n_cart", "kg"]
    st.dataframe(H[cols].round(3), hide_index=True, width="stretch")
    d1c, d2c, d3c = st.columns(3)
    d1c.download_button("Taladros (CSV)", H[cols].to_csv(index=False).encode("utf-8"), "taladros.csv", "text/csv")
    d2c.download_button("Memoria de cálculo (CSV)", res.log.df().to_csv(index=False).encode("utf-8"),
                        "memoria_calculo.csv", "text/csv")
    d3c.download_button("Resumen (JSON)", json.dumps({"labor": labor, "metodo": res.metodo,
                                                      "resumen": {k: float(v) for k, v in S.items()}},
                                                     ensure_ascii=False, indent=2).encode("utf-8"),
                        "resumen.json", "application/json")
