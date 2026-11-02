"""Labores verticales: chimeneas (ascendentes / descendentes) y piques — López Jimeno (1987) cap. 23 y
diseño de anillos con barrenos verticales de Konya (Manual de voladura, §9.2)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from shapely import affinity
from shapely.geometry import Point, Polygon

from .common import (KONYA_EQS, Explosive, Result, Rock, bmax_holmberg, bmax_langefors, colors_for,
                     finalize_holes, konya_burden, new_hole, relief_row, summary_common)
from .cuts import CUT_NOTES, CutCtx, make_cut
from .facelayout import fill_helpers, uniform_contour

CHIMENEA_METHODS = [
    "Manual clásico (cuele en V / abanico)",
    "Jaula Jora (barreno central)",
    "Plataforma Alimak (barrenos paralelos)",
    "Taladros largos — cuele de taladros vacíos (descendente)",
    "Cuele cráter (Livingston)",
    "VCR (cráteres invertidos)",
    "Combinado (raise boring + destroza)",
]
PIQUE_METHODS = [
    "Banqueo (media sección)",
    "Espiral",
    "Sección completa — cuele en V",
    "Sección completa — cuele cónico",
    "Sección completa — barrenos paralelos",
    "Sección completa — barreno o chimenea central",
    "Anillos con barrenos verticales — Konya (§9.2)",
]
METHOD_NOTES = {
    CHIMENEA_METHODS[0]: "Fig. 23.10–23.11 LJ: estructura de madera como plataforma; barrenos en V o abanico; "
                         "avance 1.5–2 m por pega. Competitivo en minas pequeñas; requiere personal experimentado.",
    CHIMENEA_METHODS[1]: "Fig. 23.12 LJ: jaula colgada de un cable que pasa por un taladro central previo, que sirve "
                         "de cuele; barrenos paralelos; avances de 3 a 4 m.",
    CHIMENEA_METHODS[2]: "Fig. 23.13 LJ: plataforma sobre guías; martillos y empujadores; barrenos paralelos con "
                         "vacíos; avances de hasta 3 m.",
    CHIMENEA_METHODS[3]: "§3.2-A LJ: perforación descendente de toda la chimenea; cuele S = D1 + 1.25·D2, franqueo "
                         "S = 10·D1 + 500 (mm); q = 0.03·D1 − 0.85 y 0.0125·D1 + 0.26 kg/m; voladura por tramos de 2–4 m.",
    CHIMENEA_METHODS[4]: "§3.2-B LJ: cavidad de ≈ 1 m² con cinco barrenos de 65–102 mm; cargas esféricas L = 6D a la "
                         "profundidad óptima (Livingston, D_o ≈ 13.7·D con ρe = 1.3).",
    CHIMENEA_METHODS[5]: "§3.2-C y Tabla 23.1 LJ: barrenos de 114–165 mm en mallas cuadradas con cargas esféricas a la "
                         "misma altura.",
    CHIMENEA_METHODS[6]: "§3.2-D LJ: chimenea piloto de 1–2 m por raise boring como hueco de expansión y destroza.",
    PIQUE_METHODS[0]: "§2.1 LJ: se perfora la mitad del fondo; la otra mitad, más baja, es la cara libre (pozos "
                      "cuadrados o rectangulares; perforación manual).",
    PIQUE_METHODS[1]: "§2.2 LJ: el fondo se excava en espiral; cada sector se vuela con una cara libre que desciende.",
    PIQUE_METHODS[2]: "§2.3 LJ: cuele en V para pozos rectangulares; barrenos inclinados 50–75° paralelos a las "
                      "discontinuidades.",
    PIQUE_METHODS[3]: "§2.3 y Fig. 23.4 LJ: superficies cónicas invertidas; el más usado en pozos circulares.",
    PIQUE_METHODS[4]: "§2.3 y Fig. 23.7 LJ: cuele de barrenos paralelos, perforación más sencilla.",
    PIQUE_METHODS[5]: "Fig. 23.8 LJ: barreno central de gran diámetro o chimenea como cara libre más efectiva.",
    PIQUE_METHODS[6]: "Konya §9.2.1 (Fig. 9.4): cuña quemada central y anillos concéntricos; B de la ec. 6.2; "
                      "N_R = R_tiro/B; S = B; N_H = 2π·R_R/S; avance L = 2B; J = 0.3B; T = 0.5B; "
                      "ajuste 0.1 + H·tan 2°; 100–150 ms (o LP) entre anillos, en espiral hacia afuera.",
}

# Figuras 23.5 y 23.6 LJ (digitalizadas): avance y consumo específico vs superficie del pozo
FIG_23_5 = ([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [0, 1.3, 2.2, 3.3, 4.0, 4.5, 4.9, 5.3, 5.6, 5.9, 6.1, 6.2])
FIG_23_6 = ([10, 15, 20, 30, 40, 50, 60, 70, 80], [3.8, 2.6, 2.0, 1.3, 1.0, 0.85, 0.75, 0.68, 0.62])


def avance_pozo(area):
    return float(np.interp(area, *FIG_23_5))


def ce_pozo(area):
    return float(np.interp(area, *FIG_23_6))


@dataclass
class VerticalOptions:
    method: str = CHIMENEA_METHODS[1]
    cut_type: str | None = None
    d1_mm: float = 45
    d2_mm: float = 102
    NB: int = 1
    bar_m: float = 3.05
    eff: float = 0.95
    F: float = 0.05
    length: float = 30.0          # longitud total de la chimenea / profundidad de pique a excavar
    tramo: float = 3.0            # chimenea por taladros largos: altura de cada pega
    contour: str = "Recorte (S ≈ 13·D)"
    K_corona: float = 15.0
    c_hat_rule: str = "Iterativa"
    offset: float = 0.10
    blast_eff: float = 0.95
    d_central: float = 1.5        # m: chimenea piloto / barreno central de pique
    Et: float = 1.5
    delta_o: float = 0.5
    k_vcr: float = 0.8            # S = k·D_o en VCR
    ratio_bloque: float = 3.0     # dimensión mínima / tamaño máximo de bloque
    inclin_deg: float = 60.0      # inclinación de barrenos del cuele (desde la horizontal) en piques
    B_override: float | None = None
    SB: float = 1.25


def _layout_face(vs, opening, cut_holes, relief, B_ring, SB, contour_S, contour_B, offset, center=(0, 0)):
    cont, cpoly = uniform_contour(vs.polygon, offset, contour_S)
    existing = [(h[1], h[2]) for h in cut_holes] + [(x, y) for x, y, *_ in cont]
    helpers = fill_helpers(vs.polygon, cpoly, opening, existing, [(r[0], r[1]) for r in relief], B_ring, SB,
                           max(contour_B, 0.25), center=center, mirror_axis=center[0])
    return cont, helpers


def _franqueo_rings(opening_half, D1_mm, outer_limit, max_n=6, rot0=45.0):
    """Secciones de franqueo para chimeneas por taladros largos: S = 10·D1 + 500 (mm), B ≤ anchura del hueco."""
    S = (10 * D1_mm + 500) / 1000
    secs = []
    Ah = 2 * opening_half
    rot = rot0
    for k in range(max_n):
        B = min(S, Ah)
        dist_new = Ah / 2 + B
        if dist_new > outer_limit:
            break
        secs.append(dict(dist=dist_new, rot=rot, B=B))
        Ah = math.sqrt(2) * dist_new
        rot = 45.0 - rot
    return secs, S


def run_vertical(labor, vs, rock: Rock, exm: Explosive, o: VerticalOptions, cost=None) -> Result:
    res = Result(labor=labor, metodo=o.method)
    lg = res.log
    P = res.p
    d1 = o.d1_mm / 1000
    up = labor == "Chimenea"
    sign = 1.0 if up else -1.0
    q_av = exm.q_lineal(d1)
    c = rock.c
    F0 = "1. Datos generales"
    lg.add(F0, "Área de la sección", "S", vs.area, "m²")
    lg.add(F0, "Constante sueca de roca", "c", c, "kg/m³", "Carpio: 0.8784·C.E + 0.0052")
    lg.add(F0, "Densidad lineal del explosivo", "q", q_av, "kg/m")
    P["note"] = METHOD_NOTES.get(o.method, "")
    rows = []
    rel_rows = []
    order = ["Arranque", "Cuadrante 1", "Cuadrante 2", "Cuadrante 3", "Cuadrante 4", "Franqueo", "Destroza",
             "Cráter", "Contorno"]
    group = {z: i for i, z in enumerate(order)}
    contour_S = 13 * d1 if o.contour.startswith("Recorte") else o.K_corona * d1
    Bm_d, _ = bmax_holmberg(q_av, exm.rws, c, 1.20 if up else 1.45, o.SB, o.c_hat_rule)
    B_lim = Bm_d - o.F
    B_ring = min(o.B_override, B_lim) if o.B_override else B_lim
    if o.B_override and o.B_override > B_lim:
        res.warnings.append(f"Burden impuesto mayor que el límite ({B_lim:.2f} m): se usa el límite.")
    lg.add(F0, "Burden de destroza (Holmberg)", "B", B_ring, "m",
           f"0.9·√(q·RWS/(Ĉ·f·S/B)) − F, f = {1.20 if up else 1.45}")
    contour_B = contour_S / 0.8 if not o.contour.startswith("Recorte") else 1.25 * contour_S
    Lt = 10 * d1

    def hole(z, x, y, L, cargas, T, B=np.nan, S=np.nan, q=np.nan, u=None, d_mm=None, Z0=0.0, tacos=None):
        u = u if u is not None else (0.0, 0.0, sign)
        rows.append(new_hole(z, group.get(z, 5), x, y, x, y, Z0, u, L, T, cargas, d_mm or o.d1_mm, exm.label(),
                             B, S, q, cuele=z in ("Arranque", "Cuadrante 1", "Cuadrante 2", "Cuadrante 3"),
                             w_cart=exm.w_cart, tacos=tacos))

    def long_hole(z, x, y, L, s0, s1, q, T_st, B, S):
        """Taladro descendente desde el nivel superior: carga entre s0 y s1 (medido desde el collar),
        retacado T_st sobre la carga y tapón de 0.3 m en el fondo."""
        tacos = [(max(0.0, s0 - T_st), s0)] + ([(L - 0.3, L)] if s1 <= L - 0.3 + 1e-6 else [])
        hole(z, x, y, L, [(s0, s1, q)], 0.0, B, S, q, u=(0, 0, -1), Z0=L, tacos=tacos)

    vol = None
    blk_frag = None
    long_methods = (CHIMENEA_METHODS[3], CHIMENEA_METHODS[4], CHIMENEA_METHODS[5], CHIMENEA_METHODS[6])
    if o.method in long_methods:
        L = o.length
        F1 = "2. " + o.method
        if o.method == CHIMENEA_METHODS[5]:  # VCR
            D = o.d1_mm / 1000
            Q = 1.5 * math.pi * D ** 3 * exm.rho * 1000
            Dc = o.Et * Q ** (1 / 3)
            Do = o.delta_o * Dc
            X = 4 / 3 * Do
            S = o.k_vcr * Do
            T = 12 * D
            lg.add(F1, "Carga esférica por barreno (L = 6D)", "Q", Q, "kg", "3πD³ρe/2")
            lg.add(F1, "Profundidad crítica", "D_c", Dc, "m", "E_t·Q^(1/3)")
            lg.add(F1, "Profundidad óptima", "D_o", Do, "m", "Δ_o·D_c (≈13.7·D con ρe = 1.3)")
            lg.add(F1, "Espaciamiento", "S", S, "m", f"{o.k_vcr}·D_o (Tabla 23.1: 1.2 m con 114 mm; 1.8 m con 165 mm)")
            lg.add(F1, "Avance por pega", "X", X, "m", "4/3·D_o (Tabla 23.1: 2.1 m / 3.0 m)")
            lg.add(F1, "Retacado", "T", T, "m", "12·D (arena o agua)")
            nx = max(2, round(vs.a / S) + 1)
            ny = max(2, round(vs.b / S) + 1)
            xs = np.linspace(-vs.a / 2 + 0.15, vs.a / 2 - 0.15, nx)
            ys = np.linspace(-vs.b / 2 + 0.15, vs.b / 2 - 0.15, ny)
            for x in xs:
                for y in ys:
                    if vs.polygon.buffer(-0.1).contains(Point(x, y)):
                        z0 = L - Do
                        long_hole("Cráter", x, y, L, z0 - 3 * D, z0 + 3 * D, Q / (6 * D), T, S, S)
            res.sources = [(0.0, 0.0)]
            vol = vs.area * X
            blk_frag = dict(B=S, S=S, q=Q / (6 * D), Lc=6 * D, H=X, Q=Q)
            P["adv"] = X
        else:
            D1, D2 = o.d1_mm, o.d2_mm
            S_cut = (D1 + 1.25 * D2) / 1000
            q_cut = max(0.03 * D1 - 0.85, 0.2)
            q_dest = 0.0125 * D1 + 0.26
            lg.add(F1, "Distancia cuele–vacío", "S", S_cut, "m", "D1 + 1.25·D2 (mm)")
            lg.add(F1, "Carga lineal del cuele", "q₁", q_cut, "kg/m", "0.03·D1 − 0.85")
            lg.add(F1, "Carga lineal de destroza", "q_i", q_dest, "kg/m", "0.0125·D1 + 0.26")
            tr = o.tramo
            lg.add(F1, "Altura de cada pega (tramo)", "h", tr, "m", "2–4 m (Fig. 23.14)")
            if o.method == CHIMENEA_METHODS[6]:
                half = o.d_central / 2
                rel_rows.append(relief_row(0, 0, o.d_central * 1000, 0, 0, L, (0, 0, -1), L))
                opening = Point(0, 0).buffer(half, 32)
                lg.add(F1, "Chimenea piloto (raise boring)", "Ø", o.d_central, "m", "1–2 m")
                cut_pts = []
            elif o.method == CHIMENEA_METHODS[4]:  # cráter: 5 barrenos
                D = D1 / 1000
                Q = 1.5 * math.pi * D ** 3 * exm.rho * 1000
                Do = o.delta_o * o.Et * Q ** (1 / 3)
                lg.add(F1, "Profundidad óptima de carga", "D_o", Do, "m", "Livingston (≈13.7·D)")
                lg.add(F1, "Carga por barreno (L = 6D)", "Q", Q, "kg", "3πD³ρe/2")
                a = 0.5
                cut_pts = [(0, 0)] + [(a * sx, a * sy) for sx in (-1, 1) for sy in (-1, 1)]
                for k, (x, y) in enumerate(cut_pts):
                    z0 = L - Do - (0.15 * k if k else 0.0)
                    long_hole("Cráter", x, y, L, z0 - 3 * D, z0 + 3 * D, Q / (6 * D), 12 * D, a, a)
                opening = Polygon([(-a, -a), (a, -a), (a, a), (-a, a)])
                half = a
                P["adv"] = 4 / 3 * Do
            else:  # taladros largos con vacíos
                rel = [(0.0, 0.0)] if o.NB == 1 else [((D2 / 1000 + 0.02) / 2 * sx, 0) for sx in (-1, 1)]
                for (x, y) in rel:
                    rel_rows.append(relief_row(x, y, D2, x, y, L, (0, 0, -1), L))
                cut_pts = [(S_cut * math.cos(math.radians(45 + 90 * k)), S_cut * math.sin(math.radians(45 + 90 * k)))
                           for k in range(4)]
                for (x, y) in cut_pts:
                    long_hole("Arranque", x, y, L, L - tr, L - 0.3, q_cut, 12 * D1 / 1000, S_cut, S_cut)
                opening = Polygon(cut_pts)
                half = S_cut
            outer = min(vs.a, vs.b) / 2 - max(contour_B, 0.3)
            secs, S_fr = _franqueo_rings(half / math.sqrt(2) if o.method == CHIMENEA_METHODS[3] else half, D1, outer,
                                         rot0=0.0 if o.method == CHIMENEA_METHODS[3] else 45.0)
            lg.add(F1, "Espaciamiento de franqueo", "S", S_fr, "m", "10·D1 + 500 (mm)")
            for k, sc in enumerate(secs):
                for j in range(4):
                    a_ = math.radians(sc["rot"] + 90 * j)
                    x, y = sc["dist"] * math.cos(a_), sc["dist"] * math.sin(a_)
                    long_hole("Franqueo", x, y, L, L - tr, L - 0.3, q_dest, 12 * D1 / 1000, sc["B"], S_fr)
                opening = opening.union(Polygon([(sc["dist"] * math.cos(math.radians(sc["rot"] + 90 * j)),
                                                  sc["dist"] * math.sin(math.radians(sc["rot"] + 90 * j)))
                                                 for j in range(4)]))
            cont, cpoly = uniform_contour(vs.polygon, o.offset, contour_S)
            for (x, y, nx_, ny_) in cont:
                long_hole("Contorno", x, y, L, L - tr, L - 0.3, 0.5 * q_dest, 12 * D1 / 1000, contour_B, contour_S)
            existing = [(r["x"], r["y"]) for r in rows]
            helpers = fill_helpers(vs.polygon, cpoly, opening.convex_hull, existing,
                                   [(r["x"], r["y"]) for r in rel_rows], S_fr, 1.0, max(contour_B, 0.3))
            for (x, y) in helpers:
                long_hole("Destroza", x, y, L, L - tr, L - 0.3, q_dest, 12 * D1 / 1000, S_fr, S_fr)
            res.sources = [(r["x"], r["y"]) for r in rel_rows] or [(0.0, 0.0)]
            vol = vs.area * tr
            P.setdefault("adv", tr)
            blk_frag = dict(B=S_fr, S=S_fr, q=q_dest, Lc=tr - 0.3, H=tr, Q=q_dest * (tr - 0.3))
            res.warnings.append("Desviación de perforación: mantener < 2 % (martillo en cabeza) o < 1 % (fondo) — LJ.")
        Lh = L
    else:
        # métodos con frente (ascendentes manuales/Jora/Alimak y piques)
        Hp = o.bar_m * o.eff
        F1 = "2. " + o.method
        if labor == "Pique":
            curve = avance_pozo(vs.area)
            NB_emp = 2 * (vs.a if vs.kind.startswith("Circular") else math.sqrt(4 * vs.area / math.pi)) ** 2 + 20
            lg.add(F1, "Avance recomendado (Fig. 23.5)", "X", curve, "m", "Wild (1984)")
            lg.add(F1, "Consumo específico orientativo (Fig. 23.6)", "CE", ce_pozo(vs.area), "kg/m³", "Wild (1984)")
            lg.add(F1, "N° de barrenos (sin contorno)", "NB", NB_emp, "", "2·Dp² + 20 (Espley-Jones y Wilson, 1979)")
            P["NB_emp"] = NB_emp
        cut_type = o.cut_type
        if cut_type is None:
            cut_type = {PIQUE_METHODS[6]: "Cuña quemada — Konya (barreno vacío grande)",
                        CHIMENEA_METHODS[0]: "En V / cuña", CHIMENEA_METHODS[1]:
                        "Cilíndrico 4 secciones — Holmberg (López Jimeno, con desviación)",
                        CHIMENEA_METHODS[2]: "Quemado simple", PIQUE_METHODS[2]: "En V / cuña",
                        PIQUE_METHODS[3]: "Instantáneo piramidal",
                        PIQUE_METHODS[4]: "Cilíndrico 4 secciones — Holmberg (López Jimeno, con desviación)"
                        }.get(o.method, "Quemado simple")
        L_face = Hp
        adv = Hp * o.blast_eff
        if o.method == PIQUE_METHODS[6]:
            De_k = exm.De_mm or o.d1_mm
            Bk = konya_burden(De_k, exm.rho, rock.rho, KONYA_EQS[1])
            R_sh = vs.min_dim / 2
            NR = max(1, round(R_sh / Bk))
            Bk_r = R_sh / NR
            Hk = min(Hp, 2.3 * Bk_r)
            Jk = 0.3 * Bk_r
            FK = "2. Anillos — Konya"
            lg.add(FK, "Bordo ideal", "B", Bk, "m", f"(6.2) 0.012·(2ρe/ρr + 1.5)·De; De = {De_k:.1f} mm")
            lg.add(FK, "N° de anillos", "N_R", NR, "", f"(9.1) R_tiro/B = {R_sh:.2f}/{Bk:.2f}")
            lg.add(FK, "Bordo real", "B_R", Bk_r, "m", "(9.2) R_tiro/N_R")
            lg.add(FK, "Avance", "L", 2 * Bk_r, "m", "(9.6) 2B")
            lg.add(FK, "Sub-barrenación", "J", Jk, "m", "(9.7) 0.3B")
            lg.add(FK, "Profundidad del barreno", "H", Hk, "m", "L + J (limitada por la barra)")
            lg.add(FK, "Taco", "T", 0.5 * Bk_r, "m", "(9.8) 0.5B")
            lg.add(FK, "Ángulo de ajuste", "LO", 0.1 + Hk * math.tan(math.radians(2)), "m", "(9.9) 0.1 + H·tan 2°")
            lg.add(FK, "Tiempo de retardo", "—", "100–150 ms o LP por anillo, en espiral hacia afuera", "", "§9.2.1.11")
            L_face = Hk
            adv = Hk - Jk
            P["konya_rings"] = (NR, Bk_r, Hk, R_sh)
        if o.method in (PIQUE_METHODS[0], PIQUE_METHODS[1], PIQUE_METHODS[5]):
            cut = None
        else:
            ctx = CutCtx(d1=d1, d2=o.d2_mm / 1000, NB=o.NB, H=L_face if "konya_rings" in P else Hp, F=o.F, c=c,
                         rws=exm.rws, q_avail=q_av,
                         UCS=rock.UCS, width=vs.min_dim - 2 * o.offset, height=vs.min_dim - 2 * o.offset,
                         vertex_deg=2 * (90 - o.inclin_deg) if labor == "Pique" else 60.0,
                         adv_factor=0.475 if labor == "Chimenea" else 0.6, rho_e=exm.rho, rho_r=rock.rho,
                         De_mm=exm.De_mm,
                         max_open=(1.6 * P["konya_rings"][1]) if "konya_rings" in P else None)
            cut = make_cut(cut_type, ctx)
            for row in cut.log:
                lg.add(F1 + " — " + cut_type, *row)
            res.warnings += cut.warnings
            if cut.advance and "konya_rings" not in P:
                adv = cut.advance if labor == "Pique" else min(cut.advance, 2.0)
                L_face = min(Hp, adv + 0.1)
            if o.method == CHIMENEA_METHODS[0]:
                adv = min(adv, 2.0)
                L_face = min(L_face, 2.1)
            P["cut_note"] = CUT_NOTES.get(cut_type, "")
            res.simultaneous |= set(cut.simultaneous)
        P["adv"] = adv
        lg.add(F1, "Avance por pega", "X", adv, "m")
        if cut is not None:
            for h in cut.holes:
                ux, uy = h.get("tu", 0.0), h.get("tv", 0.0)
                uz = sign * math.sqrt(max(1e-6, 1 - ux * ux - uy * uy))
                L = h.get("L", L_face)
                if "qf" in h:
                    cargas = [(h["T"], L - h["lf"], h["qc"]), (L - h["lf"], L, h["qf"])]
                    T = h["T"]
                elif "Tk" in h:
                    T = min(h["Tk"], 0.5 * L)
                    cargas = [(T, L, h.get("q", q_av))]
                else:
                    cargas = [(Lt, L, h.get("q", q_av))]
                    T = Lt
                hole(h["zona"], h["u"], h["v"], L, cargas, T, h.get("B"), h.get("S"), h.get("q", q_av), u=(ux, uy, uz))
            for (u, v, dd) in cut.relief:
                rel_rows.append(relief_row(u, v, dd * 1000, u, v, 0, (0, 0, sign), L_face + 0.1))
            opening = cut.opening
        elif o.method == PIQUE_METHODS[5]:
            rad = o.d_central / 2
            rel_rows.append(relief_row(0, 0, o.d_central * 1000, 0, 0, 0, (0, 0, -1), L_face + 0.3))
            opening = Point(0, 0).buffer(rad, 32)
            lg.add(F1, "Barreno / chimenea central", "Ø", o.d_central, "m", "Fig. 23.8")
        else:
            opening = None
        if o.method == PIQUE_METHODS[0]:  # banqueo: media sección más alta
            Bm, ch = bmax_langefors(o.d1_mm, exm.rho, exm.rws, c, 1.0, o.SB)
            B = min(o.B_override, Bm) if o.B_override else min(Bm - o.F, 0.6 * L_face)
            lg.add(F1, "Burden de banqueo (Langefors)", "B", B, "m", "D/33·√(ρe·PRP/(ĉ·f·S/B)), f = 1")
            half = vs.polygon.intersection(Polygon([(-50, 0), (50, 0), (50, 50), (-50, 50)]))
            S = o.SB * B
            ys = np.arange(B * 0.8, vs.b / 2 - 0.1, B)
            for y in ys:
                xs = np.arange(-vs.a / 2 + 0.2, vs.a / 2 - 0.1, S)
                for x in xs:
                    if half.buffer(-0.1).contains(Point(x, y)):
                        hole("Destroza", x, y, L_face, [(Lt, L_face, q_av)], Lt, B, S, q_av)
            res.sources = [(x, 0.0) for x in np.linspace(-vs.a / 2, vs.a / 2, 7)]
            vol = half.area * adv
            B_ring = B
            res.warnings.append("Banqueo: la mitad inferior del fondo sirve como cara libre y para recoger el agua.")
        elif o.method == PIQUE_METHODS[1]:  # espiral
            R = vs.a / 2
            nsec = 8
            radii = np.arange(R - max(contour_B, 0.3), 0.2, -B_ring)[::-1]
            for k in range(nsec):
                ang0 = 2 * math.pi * k / nsec
                for rr in radii:
                    narc = max(1, round(2 * math.pi * rr / nsec / (o.SB * B_ring)))
                    for j in range(narc):
                        ang = ang0 + (j + 0.5) * 2 * math.pi / nsec / narc
                        hole("Destroza", rr * math.cos(ang), rr * math.sin(ang), L_face, [(Lt, L_face, q_av)], Lt,
                             B_ring, o.SB * B_ring, q_av)
                        rows[-1]["grupo"] = k
            res.sources = [(r_ * 1.0, 0.0) for r_ in np.linspace(0, R, 5)]
            vol = vs.area * adv
            lg.add(F1, "Sectores de la espiral", "n", nsec, "", "un sector por pega (los totales son por vuelta)")
        if o.method == PIQUE_METHODS[6]:
            NR, Bk_r, Hk, R_sh = P["konya_rings"]
            Tk = 0.5 * Bk_r
            gk = math.atan((0.1 + Hk * math.tan(math.radians(2))) / Hk)
            r_open = max(math.hypot(*xy) for xy in opening.exterior.coords) if opening is not None else 0.0
            for k in range(1, NR + 1):
                RR = R_sh - o.offset if k == NR else k * Bk_r
                if RR <= r_open + 0.5 * Bk_r:
                    continue
                NH = max(4, math.ceil(2 * math.pi * RR / Bk_r))
                Sr = 2 * math.pi * RR / NH
                zona = "Contorno" if k == NR else "Destroza"
                lg.add("2. Anillos — Konya", f"Anillo {k}: radio, N° de barrenos y espaciamiento real", "R / N_H / S",
                       f"{RR:.2f} / {NH} / {Sr:.2f}", "m", "(9.4–9.5) N_H = 2π·R_R/S; S_real = 2π·R_R/N_H")
                for j in range(NH):
                    a_ = 2 * math.pi * j / NH + (math.pi / NH if k % 2 else 0.0)
                    x, y = RR * math.cos(a_), RR * math.sin(a_)
                    u = (math.sin(gk) * math.cos(a_), math.sin(gk) * math.sin(a_), -math.cos(gk)) if k == NR else None
                    hole(zona, x, y, Hk, [(Tk, Hk, q_av)], Tk, Bk_r, Sr, q_av, u=u)
                    rows[-1]["grupo"] = 4 + k
            res.sources = [(r["x"], r["y"]) for r in rel_rows] if rel_rows else [(0.0, 0.0)]
            vol = vs.area * adv
        elif o.method not in (PIQUE_METHODS[0], PIQUE_METHODS[1]):
            cont, helpers = _layout_face(vs, opening, [(None, r["x"], r["y"]) for r in rows],
                                         [(r["x"], r["y"]) for r in rel_rows], B_ring, o.SB, contour_S, contour_B,
                                         o.offset)
            for x, y in helpers:
                hole("Destroza", x, y, L_face, [(Lt, L_face, q_av)], Lt, B_ring, o.SB * B_ring, q_av)
            cq = 90 * d1 ** 2 if not o.contour.startswith("Recorte") else 0.5 * q_av
            for (x, y, nx_, ny_) in cont:
                hole("Contorno", x, y, L_face, [(Lt, L_face, cq)], Lt, contour_B, contour_S, cq)
            res.sources = [(r["x"], r["y"]) for r in rel_rows] if rel_rows else [(0.0, 0.0)]
            vol = vs.area * adv
        elif o.method == PIQUE_METHODS[0]:
            cont, _ = uniform_contour(vs.polygon, o.offset, contour_S)
            cq = 0.5 * q_av
            for (x, y, nx_, ny_) in cont:
                if y >= 0:
                    hole("Contorno", x, y, L_face, [(Lt, L_face, cq)], Lt, contour_B, contour_S, cq)
        Lh = L_face
        blk_frag = dict(B=B_ring, S=o.SB * B_ring, q=q_av, Lc=L_face - Lt, H=L_face, Q=q_av * (L_face - Lt))
    res.holes = finalize_holes(rows, order)
    res.relief = pd.DataFrame(rel_rows, columns=["x", "y", "d_mm", "X0", "Y0", "Z0", "ux", "uy", "uz", "L"])
    res.zone_order = [z for z in order if z in set(res.holes.zona)]
    res.zone_colors = colors_for(order)
    if labor == "Pique" and "NB_emp" in P and o.method.startswith("Sección completa"):
        n_in = int((res.holes.zona != "Contorno").sum())
        if not (0.6 * P["NB_emp"] <= n_in <= 1.5 * P["NB_emp"]):
            res.warnings.append(f"N° de barrenos sin contorno = {n_in}; la correlación de Espley-Jones y Wilson "
                                f"(cartuchos de 32 mm) da ≈ {P['NB_emp']:.0f}. Compare también el consumo específico "
                                f"con la Fig. 23.6 en la memoria de cálculo.")
    res.view2d = dict(kind="plan", outline=list(vs.polygon.exterior.coords), openings=[],
                      xlabel="x (m)", ylabel="y (m)", dims=(vs.a, vs.b))
    res.scene3d = dict(kind="vertical", outline=list(vs.polygon.exterior.coords), up=up, advance=P.get("adv", 1.0),
                       length=o.length if o.method in long_methods else None)
    vol = vol or vs.area * P.get("adv", 1.0)
    res.summary = summary_common(res, vol, rock.rho, cost, extra={"Avance por pega (m)": P.get("adv", np.nan),
                                                                   "Área de la sección (m²)": vs.area})
    rws = exm.rws * 100
    if labor == "Chimenea":
        res.frag = dict(B=blk_frag["B"], S=blk_frag["S"], q=blk_frag["q"], Lc=blk_frag["Lc"], H=blk_frag["H"],
                        d_mm=o.d1_mm, W=o.F, rws=rws, modo="verificar", min_dim=vs.min_dim, ratio=o.ratio_bloque,
                        B_limit=blk_frag["B"], zona="destroza")
    else:
        res.frag = dict(B=blk_frag["B"], S=blk_frag["S"], q=blk_frag["q"], Lc=blk_frag["Lc"], H=blk_frag["H"],
                        d_mm=o.d1_mm, W=o.F, rws=rws, modo="optimizar", B_limit=B_lim, min_dim=vs.min_dim,
                        ratio=o.ratio_bloque, zona="destroza")
    return res
