"""Chimenea slot perforada con Simba (taladros largos) como cara libre de los tajeos por subniveles, con
ensanche opcional a cámara slot.

Fuentes:
  - Curso "Cálculos para la perforación y voladura en minería subterránea" (Salas), plantillas de las
    sesiones 7 (chimenea slot), 8 (cámara slot) y 16 (voladura de chimeneas por tramos e integral).
    El cuele sigue las proporciones de Konya: B1 = 1.5·D_H, R1 = B1, S_c = √2·R; cuadros siguientes
    B_n = S_c(n−1), R_n = S_c(n−1)/2 + B_n; taco por cuadro T = (1.5, 1.06, 2.25, 4.77)·D_H.
    El esponjamiento (30 %) se compara con el volumen disponible en la labor inferior.
  - López Jimeno (1987) cap. 23 §3.2-A como alternativa (cuele S = D1 + 1.25·D2; franqueo S = 10·D1 + 500 mm;
    q = 0.03·D1 − 0.85 y 0.0125·D1 + 0.26 kg/m).
  - Konya, Manual de voladura: B = 0.012·(2ρe/ρr + 1.5)·De para el ensanche (cámara slot).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from .common import (KONYA_EQS, Explosive, Result, Rock, colors_for, finalize_holes, konya_burden, new_hole,
                     raise_dip_log, relief_row, summary_common)
from .cuts import _relief_positions, _sq
from .facelayout import fill_helpers, uniform_contour
from .sections import build_vertical_section

FT = 0.3048
SLOT_CUTS = ["Konya / plantilla del curso (B = 1.5·D_H; B_n = S_c(n−1))", "López Jimeno §3.2-A (S = D1 + 1.25·D2)"]
SLOT_BLAST = ["Por tramos ascendentes", "Integral (un disparo)", "Integral con cargas espaciadas (2–3 tramos)"]
SLOT_DIR = ["Descendente (desde el subnivel superior)", "Ascendente (desde el subnivel inferior)"]
KONYA_T = [1.5, 1.06, 2.25, 4.77]   # taco por cuadro / D_H (plantilla de la sesión 7)


@dataclass
class SlotOptions:
    kind: str = "Rectangular"
    a: float = 2.0                  # sección de la chimenea (m)
    b: float = 2.0
    L: float = 20.0                 # longitud de la chimenea = altura entre subniveles (m)
    direction: str = SLOT_DIR[0]
    D1_mm: float = 63.5             # taladros cargados (2.5")
    D2_mm: float = 101.6            # vacíos (4")
    NB: int = 2
    cut: str = SLOT_CUTS[0]
    blast: str = SLOT_BLAST[0]
    tramo: float = 3.0              # altura de cada pega (por tramos)
    n_tramos: int = 2               # integral con cargas espaciadas
    T_fondo: float = 1.0            # taco de fondo (m)
    T_inter: float = 2.0            # taco intermedio (m)
    T_collar: float = 0.5           # taco en la boca (por tramos)
    booster_kg: float = 0.15        # booster APD por carga
    offset: float = 0.15            # separación del contorno respecto a la sección
    swell: float = 0.30             # esponjamiento
    drift_w: float = 4.0            # labor inferior que recibe el material
    drift_h: float = 4.0
    L_exp: float = 4.0              # longitud de expansión disponible en la labor inferior
    Lmax_ft: float = 102.0          # capacidad de perforación del equipo
    rod_ft: float = 5.0             # longitud de barra
    alpha: float = 0.01             # desviación (m/m)
    blind: bool = False             # perforación ciega (no conecta con la otra labor)
    toe_gap: float = 1.0            # ciega: distancia del fondo del taladro a la otra labor (m)
    widen: bool = False             # ensanche a cámara slot
    dip_deg: float = 90.0           # inclinación de la chimenea slot (sola); en producción la da el método
    slot_len: float = 8.0           # cámara slot: dimensión transversal al tajeo (m)
    slot_w: float = 4.0             # cámara slot: dimensión a lo largo del tajeo (m)
    SB: float = 1.1
    B_override: float | None = None


def run_slot(rock: Rock, ex: Explosive, o: SlotOptions, cost=None) -> Result:
    res = Result(labor="Producción", metodo="Chimenea slot con Simba (taladros largos)")
    lg = res.log
    P = res.p
    P["note"] = ("Chimenea slot perforada con Simba: cara libre para los tajeos por subniveles (curso de cálculos de "
                 "perforación y voladura, sesiones 7, 8 y 16). Cuele de vacíos de gran diámetro con proporciones de "
                 "Konya; voladura por tramos o integral; el material esponjado debe caber en la labor inferior.")
    vs = build_vertical_section(o.kind, o.a, o.b)
    D1, D2 = o.D1_mm / 1000, o.D2_mm / 1000
    q = ex.q_lineal(D1)
    Ltot = o.L
    L = o.L - (o.toe_gap if o.blind else 0.0)
    down = o.direction.startswith("Desc")
    F0, F1, F2, F3 = ("1. Datos de la chimenea slot", "2. Cuele y distribución", "2a. Equipo y longitud de perforación",
                      "3. Carguío y voladura")
    lg.add(F0, "Sección de la chimenea", "a × b", f"{o.a:.2f} × {o.b:.2f}", "m", f"área {vs.area:.2f} m²")
    lg.add(F0, "Altura entre labores", "H", Ltot, "m", "")
    lg.add(F0, "Longitud de perforación", "L", L, "m",
           f"ciega: queda un puente de {o.toe_gap:.2f} m hasta la otra labor" if o.blind
           else "pasante: conecta las dos labores")
    lg.add(F0, "Volumen in situ", "V", vs.area * Ltot, "m³")
    lg.add(F0, "Tonelaje", "T", vs.area * Ltot * rock.rho, "t", f"ρ = {rock.rho:.2f} t/m³")
    lg.add(F0, "Densidad lineal de carga", "q", q, "kg/m", ex.label())
    # equipo
    Lmax = o.Lmax_ft * FT
    n_rod = math.ceil(L / (o.rod_ft * FT))
    lg.add(F2, "Capacidad de perforación del equipo", "L_máx", Lmax, "m", f"{o.Lmax_ft:.0f} pies")
    lg.add(F2, "Barras por taladro", "n", n_rod, "", f"⌈L / barra⌉, barra de {o.rod_ft:.0f} pies")
    lg.add(F2, "Desviación al fondo", "α·L", o.alpha * L, "m", f"α = {o.alpha * 100:.1f} %")
    if L > Lmax:
        res.warnings.append(f"La chimenea de {L:.1f} m supera la capacidad del equipo ({o.Lmax_ft:.0f} pies = {Lmax:.1f} m): "
                            "use un subnivel intermedio o perfore desde ambos niveles (ascendente y descendente).")
    if o.alpha * L > 0.5 * max(D2 * math.sqrt(o.NB), 0.1) + 0.15:
        res.warnings.append(f"La desviación al fondo ({o.alpha * L:.2f} m) es grande frente al cuele: el arranque puede "
                            "cruzarse con los vacíos o perder la cara libre. Controle la desviación (< 1–2 %).")
    rows, rel = [], []
    order = ["Arranque", "Cuadrante 1", "Cuadrante 2", "Cuadrante 3", "Franqueo", "Destroza", "Contorno",
             "Cámara slot"]
    group = {z: i for i, z in enumerate(order)}
    Z0, u = (Ltot, (0, 0, -1)) if down else (0.0, (0, 0, 1))
    # tacos y cargas según el modo de voladura (s se mide desde la boca)
    if o.blast.startswith("Por tramos"):
        def charge(Th):
            tr = min(o.tramo, L - 0.6)
            if down:   # el primer tramo es el del fondo (sobre la labor inferior)
                s0 = L - tr
                return [(s0, L - 0.3, q)], [(max(0.0, s0 - Th), s0), (L - 0.3, L)]
            return [(Th, Th + tr, q)], [(0.0, Th), (Th + tr, min(L, Th + tr + 0.3))]
        mode_txt = f"primer tramo de {o.tramo:.1f} m cargado (retacado {o.T_collar:.1f} m)"
    elif o.blast.startswith("Integral (un"):
        def charge(Th):
            return [(Th, L - o.T_fondo, q)], [(0.0, Th), (L - o.T_fondo, L)]
        mode_txt = f"columna completa; taco de boca y de fondo {o.T_fondo:.1f} m"
    else:
        def charge(Th):
            n = max(2, o.n_tramos)
            Lc = (L - Th - o.T_fondo - (n - 1) * o.T_inter) / n
            cs, ts, s = [], [(0.0, Th)], Th
            for k in range(n):
                cs.append((s, s + Lc, q))
                if k < n - 1:
                    ts.append((s + Lc, s + Lc + o.T_inter))
                s += Lc + o.T_inter
            ts.append((L - o.T_fondo, L))
            return cs, ts
        mode_txt = f"{max(2, o.n_tramos)} cargas con tacos intermedios de {o.T_inter:.1f} m"

    def hole(z, x, y, B, S, Th, qq=None):
        cs, ts = charge(Th)
        if qq:
            cs = [(a, b, qq) for a, b, _ in cs]
        rows.append(new_hole(z, group[z], x, y, x, y, Z0, u, L, ts[0][1] - ts[0][0], cs, o.D1_mm, ex.label(),
                             B, S, cs[0][2] if cs else q, cuele=z.startswith(("Arranque", "Cuadrante")),
                             w_cart=ex.w_cart, tacos=ts))

    DH = D2 * math.sqrt(o.NB)
    lg.add(F1, "Diámetro equivalente de los vacíos", "D_H", DH, "m", f"{o.D2_mm:.1f} mm · √{o.NB}")
    Hk = 0.15 + 34.1 * DH - 39.4 * DH ** 2
    lg.add(F1, "Avance por tramo que asegura el vacío", "L_tramo,máx", Hk, "m", "0.15 + 34.1·D_H − 39.4·D_H²")
    if o.blast.startswith("Por tramos") and o.tramo > Hk:
        res.warnings.append(f"Tramo de {o.tramo:.1f} m mayor que el que asegura el alivio ({Hk:.2f} m).")
    De = ex.De_mm or o.D1_mm
    Bk = konya_burden(De, ex.rho, rock.rho, KONYA_EQS[1])
    lg.add(F1, "Bordo que rompe la carga (Konya ec. 6.2)", "B", Bk, "m", f"0.012·(2ρe/ρr + 1.5)·De, De = {De:.1f} mm")
    inner = Polygon(vs.polygon.buffer(-o.offset - 0.15, join_style=2))
    for (x, y) in _relief_positions(o.NB, D2):
        rel.append(relief_row(x, y, o.D2_mm, x, y, Z0, u, L))
    if o.cut.startswith("Konya"):
        R, B, rot, opening = 1.5 * DH, 1.5 * DH, 0.0, None
        for n in range(1, 5):
            Sc = math.sqrt(2) * R
            sq = Polygon(_sq(R, rot))
            if n > 1 and (B > Bk or not inner.contains(sq)):
                break
            zona = "Arranque" if n == 1 else f"Cuadrante {n - 1}"
            Th = max(KONYA_T[n - 1] * DH, 0.3)
            lg.add(F1, f"Cuadro {n}: B / R / S_c / taco", "—", f"{B:.3f} / {R:.3f} / {Sc:.3f} / {Th:.2f}", "m",
                   "B1 = 1.5·D_H" if n == 1 else f"B{n} = S_c{n - 1}; R{n} = S_c{n - 1}/2 + B{n}")
            for (x, y) in _sq(R, rot):
                hole(zona, x, y, B, Sc, Th if not o.blast.startswith("Por tramos") else o.T_collar)
            opening = sq
            B = Sc
            R = Sc / 2 + B
            rot = 45.0 - rot
        Bd = o.B_override or min(Bk, 0.5 * vs.min_dim)
    else:
        S_cut = (o.D1_mm + 1.25 * o.D2_mm) / 1000
        lg.add(F1, "Distancia cuele–vacío", "S", S_cut, "m", "D1 + 1.25·D2 (LJ)")
        pts = _sq(S_cut, 45.0)
        for (x, y) in pts:
            hole("Arranque", x, y, S_cut, S_cut, o.T_collar, max(0.03 * o.D1_mm - 0.85, 0.2))
        opening = Polygon(pts)
        Bd = o.B_override or min((10 * o.D1_mm + 500) / 1000, 0.5 * vs.min_dim)
        lg.add(F1, "Espaciamiento de franqueo", "S", Bd, "m", "10·D1 + 500 (mm) (LJ)")
    # contorno (cuadradores) y destroza
    cS = min(Bd, 1.0)
    per = vs.polygon.buffer(-o.offset, join_style=2).length
    if not o.kind.startswith("Circular"):
        cS = per / (4 * math.ceil(per / (4 * cS)))   # esquinas y lados simétricos (cuadradores)
    cont, cpoly = uniform_contour(vs.polygon, o.offset, cS * 1.001)
    for (x, y, nx_, ny_) in cont:
        if opening is not None and opening.buffer(0.15).contains(Point(x, y)):
            continue
        hole("Contorno", x, y, Bd, cS, o.T_collar)
    existing = [(r["x"], r["y"]) for r in rows]
    helpers = fill_helpers(vs.polygon, cpoly, opening.convex_hull if opening is not None else Point(0, 0).buffer(0.2),
                           existing, [(r["x"], r["y"]) for r in rel], Bd, o.SB, max(Bd, 0.3))
    for (x, y) in helpers:
        hole("Destroza", x, y, Bd, o.SB * Bd, o.T_collar)
    lg.add(F1, "Taladros cargados / vacíos", "N", f"{len(rows)} / {len(rel)}", "",
           "plantilla del curso: arranque 4 + ayudas 8 + cuadradores 8 = 20 cargados, 2–3 vacíos (2 × 2 m)")
    lg.add(F3, "Modo de voladura", "—", o.blast, "", mode_txt)
    # cámara slot (ensanche)
    if o.widen:
        Bw = o.B_override or Bk
        Sw = o.SB * Bw
        lg.add("2b. Ensanche a cámara slot", "Dimensiones", "—", f"{o.slot_len:.1f} × {o.slot_w:.1f} m", "",
               "plantilla de la sesión 8")
        lg.add("2b. Ensanche a cámara slot", "Burden y espaciamiento", "B / S", f"{Bw:.2f} / {Sw:.2f}", "m",
               "Konya ec. 6.2; S = 1.0–1.2·B")
        nx, ny = max(1, math.ceil(o.slot_len / Bw)), max(1, math.ceil(o.slot_w / Sw))
        xs = np.linspace(-o.slot_len / 2 + o.slot_len / (2 * nx), o.slot_len / 2 - o.slot_len / (2 * nx), nx)
        ys = np.linspace(-o.slot_w / 2 + o.slot_w / (2 * ny), o.slot_w / 2 - o.slot_w / (2 * ny), ny)
        keep_out = vs.polygon.buffer(0.35 * Bw)
        nw = 0
        for x in xs:
            for y in ys:
                if keep_out.contains(Point(x, y)):
                    continue
                hole("Cámara slot", x, y, Bw, Sw, 0.7 * Bw)
                nw += 1
        lg.add("2b. Ensanche a cámara slot", "Taladros del ensanche", "N", nw, "")
    res.holes = finalize_holes(rows, order)
    res.relief = pd.DataFrame(rel, columns=["x", "y", "d_mm", "X0", "Y0", "Z0", "ux", "uy", "uz", "L"])
    res.zone_order = [z for z in order if z in set(res.holes.zona)]
    res.zone_colors = colors_for(order)
    res.sources = [(r["x"], r["y"]) for r in rel] or [(0.0, 0.0)]
    # esponjamiento vs volumen disponible
    V_tr = vs.area * (o.tramo if o.blast.startswith("Por tramos") else L)
    V_esp = V_tr * (1 + o.swell)
    V_disp = o.drift_w * o.drift_h * o.L_exp
    lg.add(F3, "Volumen roto por disparo (esponjado)", "V_e", V_esp, "m³", f"V·(1 + {o.swell:.0%})")
    lg.add(F3, "Volumen disponible en la labor inferior", "V_d", V_disp, "m³", "ancho × alto × longitud de expansión")
    if V_esp > V_disp:
        res.warnings.append(f"El material esponjado ({V_esp:.0f} m³) no cabe en la labor inferior ({V_disp:.0f} m³): "
                            "limpie entre tramos, vuele por tramos o amplíe la zona de expansión.")
    if not down and o.blast.startswith("Integral"):
        res.warnings.append("Voladura integral con perforación ascendente: verifique el carguío de taladros "
                            "ascendentes (bombeo de emulsión o tapones de retención).")
    vol = vs.area * (o.tramo if o.blast.startswith("Por tramos") else L)
    if o.widen:
        vol += max(o.slot_len * o.slot_w - vs.area, 0) * (o.tramo if o.blast.startswith("Por tramos") else L)
    kg = float(res.holes.kg.sum()) + o.booster_kg * sum(len(c) for c in res.holes.cargas)
    drill_m = float(res.holes.L.sum() + res.relief.L.sum())
    res.summary = summary_common(res, vol, rock.rho, cost, extra={"Volumen por voladura (m³)": vol})
    res.summary["Explosivo total (kg)"] = kg
    res.summary["Factor de carga (kg/m³)"] = kg / vol if vol else np.nan
    res.summary["Factor de carga (g/t)"] = 1000 * kg / (vol * rock.rho) if vol else np.nan
    res.summary["Índice de perforación (t/m)"] = vs.area * Ltot * rock.rho / drill_m if drill_m else np.nan
    res.summary["Barras por taladro"] = n_rod
    lg.add(F3, "Explosivo por voladura (incluye boosters)", "Q", kg, "kg", f"{o.booster_kg:.2f} kg de booster por carga")
    lg.add(F3, "Índice de perforación", "IP", res.summary["Índice de perforación (t/m)"], "t/m",
           "tonelaje de la chimenea / metros perforados (incluye vacíos)")
    outline = list(vs.polygon.exterior.coords)
    lines = []
    if o.widen:
        lines.append(([-o.slot_len / 2, o.slot_len / 2, o.slot_len / 2, -o.slot_len / 2, -o.slot_len / 2],
                      [-o.slot_w / 2, -o.slot_w / 2, o.slot_w / 2, o.slot_w / 2, -o.slot_w / 2]))
    res.view2d = dict(kind="plan", outline=outline, openings=[("Cuele", list(opening.exterior.coords))]
                      if opening is not None else [], lines=lines, xlabel="x (m)", ylabel="y (m)",
                      dims=(o.a, o.b))
    res.scene3d = dict(kind="vertical", outline=outline, up=not down, advance=o.tramo, length=Ltot)
    if o.dip_deg < 89.99:
        raise_dip_log(res, o.dip_deg, Ltot, "largos")
    res.frag = dict(B=Bd, S=o.SB * Bd, q=q, Lc=max((o.tramo if o.blast.startswith("Por tramos") else L) - 0.5, 0.2),
                    H=o.tramo if o.blast.startswith("Por tramos") else L, d_mm=o.D1_mm, W=o.alpha * L,
                    rws=ex.rws * 100, modo="verificar", min_dim=vs.min_dim, ratio=3.0, B_limit=Bd, zona="destroza")
    P["vol"] = vol
    P["rows_blast"] = 99
    P["blast_all"] = True
    return res


def place_slot(slot_res: Result, xc: float, yc: float, dip_deg: float = 90.0, side: float = 1.0, down=True,
               H: float | None = None):
    """Lleva los taladros de la chimenea slot (coordenadas locales: x transversal, y longitudinal) al sistema del
    tajeo (X a lo largo del tajeo, Y transversal) y los inclina con el buzamiento de los taladros paralelos.
    Devuelve (taladros, vacíos) como DataFrames."""
    dip = math.radians(dip_deg)
    cot = math.cos(dip) / math.sin(dip) if dip_deg < 89.99 else 0.0
    ud = np.array([0.0, -side * math.cos(dip), -math.sin(dip)])  # dirección descendente

    def tr(df):
        df = df.copy()
        if df.empty:
            return df
        xl, yl = df.x.values.copy(), df.y.values.copy()
        X, Y = xc + yl, yc + xl
        df["X0"], df["Y0"] = X, Y
        df["x"], df["y"] = X, Y
        if dip_deg < 89.99:
            L_h = df.L.values
            if down:
                df["ux"], df["uy"], df["uz"] = ud[0], ud[1], ud[2]
                df["L"] = L_h / math.sin(dip)
            else:
                df["ux"], df["uy"], df["uz"] = -ud[0], -ud[1], -ud[2]
                df["L"] = L_h / math.sin(dip)
                df["Y0"] = Y - side * (H or 0.0) * cot
            if "cargas" in df:
                k = 1 / math.sin(dip)
                df["cargas"] = [[(a * k, b * k, q) for a, b, q in c] for c in df.cargas]
                df["tacos"] = [[(a * k, b * k) for a, b in (t or [])] for t in df.tacos]
                df["kg"] = [sum((b - a) * q for a, b, q in c) for c in df.cargas]
        return df

    return tr(slot_res.holes), tr(slot_res.relief)
