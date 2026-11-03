"""Voladuras subterráneas de producción por método de explotación — López Jimeno (1987) cap. 24,
fórmula de Langefors (cap. 20), tajeo de Holmberg, Pearse modificado con Kv (Carpio, ec. 06) y los
criterios del material de capacitación: Konya (caps. 6–7), Ash, Torbica y Lapcevic (2018), Rustan,
Pearse del curso de taladros largos y el retacado escalonado de anillos (AEL / Onederra y Chitombo)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

from .drilling import DrillSetup, carousel, collar_from_pivot, drift_centers, drift_polygon, rig_fits
from .common import (KONYA_EQS, Explosive, Result, Rock, ash_burden, b_pearse_kv, bmax_holmberg, bmax_langefors,
                     colors_for, finalize_holes, konya_burden, konya_spacing, new_hole, pearse_course_burden,
                     ring_stemming, rustan_burden, summary_common, torbica_burden)

PROD_METHODS = [
    "Taladros largos — LBH (paralelos, gran diámetro)",
    "Taladros largos en vetas angostas (paralelos, Ø 51–89 mm)",
    "Subniveles con taladros en abanico",
    "VCR — cráteres invertidos",
    "Cámaras y pilares — banqueo vertical",
    "Cámaras y pilares — banqueo frontal (horizontal)",
    "Corte y relleno — realce (perforación vertical/inclinada)",
    "Corte y relleno — breasting (perforación horizontal)",
    "Cámaras de obra pública — banqueo por fases",
]
CRITERIA = ["Langefors (LJ cap. 20 y 24)", "Holmberg — tajeo (Carpio / LJ cap. 22)",
            "Pearse modificado con Kv (Carpio ec. 06)", "Consumo específico (LJ §4, subniveles)",
            "Konya (Konya caps. 6–7; curso de taladros largos)", "Ash: B = Kb·De/12 (curso)",
            "Torbica y Lapcevic (2018) (curso)", "Pearse con PD de Cook (plantilla del curso)",
            "Rustan: B = 11.8·D^0.63 (Onederra 2007)"]
FAN_STEMMING = ["Ts–Tl–Tm–Tl: 20D / 125D / 50D (AEL, Onederra; curso)", "Alternado largo/corto (geométrico)"]
NOTES = {
    PROD_METHODS[0]: "LJ §3: corte inferior + corte lateral (chimenea de 1.8–3.5 m) + zona de barrenos largos de gran "
                     "diámetro (85–90 % del tonelaje). B_max de Langefors; B = B_max − 2D − 0.02L; S = 1.25B. Sólo "
                     "carga de columna; seccionado con retacado intermedio de 12–15D (ANFO) o 16–18D (hidrogeles); "
                     "cargas de 100–200 kg por retardo; L_carga/D > 20. Retardos: 50 ms en el mismo barreno, 10 ms "
                     "entre barrenos de una fila y 75 ms entre filas (Du Pont).",
    PROD_METHODS[1]: "Variante de taladros largos para vetas angostas (especialidad citada por Carpio): 1–2 taladros "
                     "por fila según la potencia; el burden puede calcularse con Pearse modificado con Kv (ec. 06).",
    PROD_METHODS[2]: "LJ §4: CE = CE₀ + 0.03·L + 0.40/AV (kg/m³ de explosivo gelatinoso) en el fondo y en un quinto "
                     "de la longitud; A_e = S·B = q₁/CE; S/B = 1.3–2 (S = 2B da buenos resultados); separación entre "
                     "secciones 1.2–1.8 m; bocas de 51–64 mm; L ≤ 25 m; columna al 50–75 % del fondo.",
    PROD_METHODS[3]: "LJ §2: cargas esféricas L ≤ 6D a la profundidad óptima (Livingston D_c = E_t·Q^(1/3), "
                     "D_o = Δ_o·D_c); retacado 12D; pegas ascendentes; corona según Tabla 24.2 (< 1.5X una voladura; "
                     "1.6–2.0X cargas espaciadas simultáneas; > 2X dos voladuras). Yacimientos de 3 a 10 m de potencia.",
    PROD_METHODS[4]: "LJ §5: yacimientos horizontales o poco inclinados; en depósitos potentes el nivel inferior se "
                     "recupera por banqueo con barrenos verticales (carros) — fórmula de banco de Langefors.",
    PROD_METHODS[5]: "LJ §5 (Fig. 24.17): banqueo frontal con jumbos y barrenos horizontales, como un banco girado 90°.",
    PROD_METHODS[6]: "LJ §6 (Fig. 24.19): perforación en el techo a modo de banco invertido, barrenos inclinados "
                     "50–65°, ≤ 4 m; martillos de 29–33 mm o jumbos de 33–64 mm; 200–260 g/t; retacado con arcilla.",
    PROD_METHODS[7]: "LJ §6 (Fig. 24.19): barrenos horizontales de longitud igual a la varilla (< 4 m) disparados "
                     "hacia la superficie libre inferior; techo más regular y mejor selectividad.",
    PROD_METHODS[8]: "LJ §7: cámaras pequeñas (< 10 m: galería piloto, destrozas laterales y banqueo; horizontales "
                     "32–45 mm, verticales 50 mm) y grandes (hasta 60 m, fases descendentes; horizontales 32–45 mm, "
                     "verticales 50–65 mm). Las fases de galería se diseñan en el modo 'Avance'.",
}
DEFAULTS = {  # D (mm), criterio, explosivo sugerido
    PROD_METHODS[0]: (165, CRITERIA[0]), PROD_METHODS[1]: (64, CRITERIA[0]), PROD_METHODS[2]: (64, CRITERIA[3]),
    PROD_METHODS[3]: (165, CRITERIA[0]), PROD_METHODS[4]: (76, CRITERIA[0]), PROD_METHODS[5]: (45, CRITERIA[1]),
    PROD_METHODS[6]: (38, CRITERIA[0]), PROD_METHODS[7]: (38, CRITERIA[1]), PROD_METHODS[8]: (51, CRITERIA[0]),
}
CE0_TABLE = {"Fisurada y dura": 0.60, "Con juntas": 0.55, "Fracturada": 0.50, "Relativamente homogénea": 0.45,
             "Homogénea y dura": 0.40, "Blanda y homogénea": 0.35}  # Tabla 24.4 LJ


@dataclass
class ProdOptions:
    method: str = PROD_METHODS[0]
    criterion: str = CRITERIA[0]
    D_mm: float = 165
    length: float = 30.0       # longitud del tajeo/cámara a lo largo del avance (m)
    width: float = 15.0        # ancho / potencia AV (m)
    height: float = 40.0       # altura entre niveles / altura del banco o de la tajada (m)
    dip_deg: float = 90.0      # buzamiento (vetas angostas)
    incl_deg: float = 60.0     # inclinación de realce (desde la horizontal)
    rows_blast: int = 3        # filas/secciones por voladura
    SB: float = 1.25
    f: float = 1.0
    alpha: float = 0.01        # desviación angular (m/m)
    e_coll: float = 0.05       # error de emboquille (m)
    k_T: float = 1.0           # factor del retacado
    col_ratio: float = 0.5     # carga de columna / fondo
    q_deck_max: float = 150.0  # kg por carga (LBH)
    staggered: bool = True
    fan_type: str = "Ascendente"
    drift_w: float = 4.0
    drift_h: float = 4.0
    up_h: float = 15.0
    down_h: float = 0.0
    ce0_key: str = "Con juntas"
    k_vcr: float = 1.0
    Et: float = 1.5
    delta_o: float = 0.5
    corona: float = 6.0
    pillar: float = 6.0
    room: float = 8.0
    phase_h: float = 8.0
    orient_h: bool = False     # cámaras de obra pública: banqueo horizontal
    B_override: float | None = None
    # Konya
    konya_eq: str = KONYA_EQS[0]
    Kd: float = 1.0
    Ks: float = 1.0
    Kr: float = 1.0
    konya_delayed: bool = True
    konya_S: bool = True          # espaciamiento por la relación de rigidez (ecs. 7.1–7.4)
    stv: float = 100.0            # potencia relativa por volumen (ec. 6.3)
    Kb: float = 30.0              # Ash
    fan_stem: str = FAN_STEMMING[0]
    drill: DrillSetup | None = None   # labor de perforación, pivote y carrusel (taladros largos)
    Lmax_ft: float = 102.0        # capacidad de perforación del equipo de taladros largos (Simba)
    rod_ft: float = 5.0           # longitud de barra


def _burden(o: ProdOptions, rock: Rock, ex: Explosive, L, q, f, log, fase):
    D = o.D_mm
    if o.criterion.startswith("Langefors"):
        Bm, ch = bmax_langefors(D, ex.rho, ex.rws, rock.c, f, o.SB)
        long_ = L > 10
        B = Bm - (2 * D / 1000 + 0.02 * L if long_ else o.e_coll + o.alpha * 3 * L)
        log.add(fase, "Piedra máxima (Langefors)", "B_max", Bm, "m", f"D/33·√(ρe·PRP/(ĉ·f·S/B)), ĉ = {ch:.3f}, f = {f}")
        log.add(fase, "Piedra práctica", "B", B, "m", "B_max − 2D − 0.02L" if long_ else "B_max − e' − d_b·L")
        return Bm, B
    if o.criterion.startswith("Holmberg"):
        Bm, ch = bmax_holmberg(q, ex.rws, rock.c, f, o.SB)
        F = o.e_coll + o.alpha * L
        log.add(fase, "Burden máximo (Holmberg, tajeo)", "B_max", Bm, "m", f"0.9·√(q·RWS/(Ĉ·f·S/B)), Ĉ = {ch:.3f}")
        log.add(fase, "Burden práctico", "B", Bm - F, "m", "B_max − (e' + α·L)")
        return Bm, Bm - F
    if o.criterion.startswith("Pearse"):
        Bm = b_pearse_kv(rock.Kv, D, ex.PD, rock.Std)
        F = o.e_coll + o.alpha * L
        log.add(fase, "Burden (Pearse modificado con Kv)", "B_max", Bm, "m",
                f"Kv·D/1000·√(PD/Std); Kv = {rock.Kv:.3f}, PD = {ex.PD:.0f} MPa, Std = {rock.Std:.1f} MPa")
        log.add(fase, "Burden práctico", "B", Bm - F, "m", "B_max − (e' + α·L)")
        if Bm - F < 0.5:
            log.add(fase, "Advertencia", "—", "Pearse da piedras propias del arranque; en producción se usa B ≥ 0.5 m",
                    "", "")
        return Bm, max(Bm - F, 0.25)
    De = ex.De_mm or D
    if o.criterion.startswith("Konya"):
        B0 = konya_burden(De, ex.rho, rock.rho, o.konya_eq, o.stv)
        B = B0 * o.Kd * o.Ks * o.Kr
        log.add(fase, "Burden de Konya", "B₀", B0, "m", o.konya_eq.split(":")[1].strip() + f"; De = {De:.1f} mm")
        log.add(fase, "Burden corregido", "B", B, "m", f"B₀·Kd·Ks·Kr = B₀·{o.Kd}·{o.Ks}·{o.Kr} (Tablas 6.2–6.4)")
        return B, B
    if o.criterion.startswith("Ash"):
        B = ash_burden(De, o.Kb)
        log.add(fase, "Burden de Ash", "B", B, "m", f"Kb·De/12 (pies, pulg); Kb = {o.Kb:.0f}, De = {De:.1f} mm")
        return B, B
    if o.criterion.startswith("Torbica"):
        B, K = torbica_burden(ex.PD_cook, D, rock.st, rock.nu)
        log.add(fase, "Burden de Torbica y Lapcevic", "B", B, "m",
                f"0.17·PD·r_h/(K·σt); PD = {ex.PD_cook:.0f} MPa (Cook), σt = {rock.st:.1f} MPa, K = {K:.3f}")
        return B, B
    if o.criterion.startswith("Pearse con PD"):
        Bm = pearse_course_burden(rock.Kv, D, ex.PD_cook, rock.st)
        F = o.e_coll + o.alpha * L
        log.add(fase, "Burden (Pearse, plantilla del curso)", "B_max", Bm, "m",
                f"Kv·D·√(PD/Std); Kv = {rock.Kv:.3f}, PD = {ex.PD_cook:.0f} MPa, Std = 4.5·σt = {4.5 * rock.st:.1f} MPa")
        log.add(fase, "Burden práctico", "B", Bm - F, "m", "B_max − (e' + α·L)")
        return Bm, max(Bm - F, 0.25)
    if o.criterion.startswith("Rustan"):
        B = rustan_burden(D)
        log.add(fase, "Burden de Rustan", "B", B, "m", "11.8·D^0.63 (D en m)")
        return B, B
    # consumo específico (subniveles)
    CE0 = CE0_TABLE[o.ce0_key]
    CE = CE0 + 0.03 * L + 0.40 / o.width
    B = math.sqrt(q / (o.SB * CE))
    log.add(fase, "Consumo específico de diseño", "CE", CE, "kg/m³", f"CE₀ + 0.03·L + 0.40/AV (CE₀ = {CE0})")
    log.add(fase, "Piedra en el fondo", "B", B, "m", "A_e = S·B = q₁/CE, S = (S/B)·B")
    return B, B


def burden_table(rock: Rock, ex: Explosive, o: ProdOptions, L: float, q: float) -> pd.DataFrame:
    """Compara el burden de todos los criterios para el mismo diámetro y explosivo."""
    D = o.D_mm
    De = ex.De_mm or D
    rows = []
    try:
        rows.append(("Langefors (LJ)", bmax_langefors(D, ex.rho, ex.rws, rock.c, o.f, o.SB)[0], "D/33·√(ρe·PRP/(ĉ·f·S/B))"))
    except Exception:  # noqa: BLE001
        pass
    rows.append(("Holmberg — tajeo", bmax_holmberg(q, ex.rws, rock.c, o.f, o.SB)[0], "0.9·√(q·RWS/(Ĉ·f·S/B))"))
    rows.append(("Pearse con Kv (Carpio)", b_pearse_kv(rock.Kv, D, ex.PD, rock.Std), "Kv·D·√(PD/Std)"))
    rows.append(("Pearse con PD de Cook (curso)", pearse_course_burden(rock.Kv, D, ex.PD_cook, rock.st),
                 "Kv·D·√(PD/(4.5·σt))"))
    for eq in KONYA_EQS:
        rows.append((eq.split(":")[0] + " × Kd·Ks·Kr", konya_burden(De, ex.rho, rock.rho, eq, o.stv) * o.Kd * o.Ks * o.Kr,
                     eq.split(":")[1].strip()))
    rows.append((f"Ash (Kb = {o.Kb:.0f})", ash_burden(De, o.Kb), "Kb·De/12"))
    rows.append(("Torbica y Lapcevic (2018)", torbica_burden(ex.PD_cook, D, rock.st, rock.nu)[0], "0.17·PD·r_h/(K·σt)"))
    rows.append(("Rustan", rustan_burden(D), "11.8·D^0.63"))
    rows.append(("McKenzie (25·D, mínimo del rango)", 25 * D / 1000, f"25–40·D = {25 * D / 1000:.2f}–{40 * D / 1000:.2f} m"))
    rows.append(("Andersen", math.sqrt(L * D / 1000), "√(L·D)"))
    return pd.DataFrame(rows, columns=["Criterio", "B (m)", "Ecuación"])


def run_produccion(rock: Rock, ex: Explosive, o: ProdOptions, cost=None) -> Result:
    res = _run(rock, ex, o, cost)
    if res.frag and res.frag["B"] < 0.5 and o.criterion.startswith("Pearse"):
        res.warnings.append("Pearse modificado con Kv entrega piedras muy pequeñas para taladros de producción "
                            "(su ejemplo en Carpio es el arranque); compare con Langefors.")
    return res


def _run(rock: Rock, ex: Explosive, o: ProdOptions, cost=None) -> Result:
    res = Result(labor="Producción", metodo=o.method)
    lg = res.log
    P = res.p
    P["note"] = NOTES.get(o.method, "")
    D = o.D_mm / 1000
    q = ex.q_lineal(D)
    F0 = "1. Datos generales"
    lg.add(F0, "Diámetro de perforación", "D", o.D_mm, "mm")
    lg.add(F0, "Densidad lineal de carga", "q", q, "kg/m", "ρe·π·D²/4 (granel) o peso/longitud (cartucho)")
    lg.add(F0, "Constante de roca", "c", rock.c, "kg/m³", "Carpio: 0.8784·C.E + 0.0052")
    rows = []
    order = ["Cráter", "Primera fila", "Producción", "Contorno", "Zapatera"]
    m = o.method
    fase = "2. Esquema — " + m
    lines2d = []
    outline = None
    xlabel, ylabel = "x (m)", "y (m)"
    scene = dict(kind="box")
    sources = []

    def add(z, grupo, x2, y2, X0, Y0, Z0, u, L, cargas, tacos, B, S):
        rows.append(new_hole(z, grupo, x2, y2, X0, Y0, Z0, u, L, tacos[0][1] if tacos and tacos[0][0] == 0 else 0.0,
                             cargas, o.D_mm, ex.label(), B, S, q, w_cart=ex.w_cart, tacos=tacos))

    konya = o.criterion.startswith("Konya")

    def spacing(B, Lh, label=""):
        if konya and o.konya_S:
            S_, eq = konya_spacing(Lh, B, o.konya_delayed)
            lg.add(fase, "Espaciamiento (Konya)" + label, "S", S_, "m", eq)
            return S_
        return o.SB * B

    def charge_column(L, B, bottom=True, plug=0.0, decks=False):
        T = o.k_T * (0.7 * B if konya else max(10 * D, min(0.7 * B, 25 * D)))
        s_top, s_bot = T, L - plug
        tacos = [(0.0, T)] + ([(L - plug, L)] if plug > 0 else [])
        Lch = s_bot - s_top
        if Lch <= 0.1:
            return [], tacos, T, 0.0
        if decks:
            Qh = q * Lch
            nd = max(1, math.ceil(Qh / o.q_deck_max))
            ti = (14 if ex.name.startswith("ANFO") else 17) * D
            ld = (Lch - (nd - 1) * ti) / nd
            if nd > 1 and ld / D < 20:
                res.warnings.append(f"Cargas seccionadas de {ld:.1f} m: L/D = {ld / D:.0f} < 20 (mala fragmentación).")
            cargas = []
            s = s_top
            for k in range(nd):
                cargas.append((s, s + ld, q))
                if k < nd - 1:
                    tacos.append((s + ld, s + ld + ti))
                s += ld + ti
            return cargas, tacos, T, Lch
        if bottom:
            lf = min(1.3 * B, 0.5 * Lch)
            return [(s_top, s_bot - lf, o.col_ratio * q), (s_bot - lf, s_bot, q)], tacos, T, Lch
        return [(s_top, s_bot, q)], tacos, T, Lch

    if m in (PROD_METHODS[0], PROD_METHODS[1]):
        H = o.height
        dip = math.radians(o.dip_deg)
        L = H / math.sin(dip)
        Bm, B = _burden(o, rock, ex, L, q, o.f, lg, fase)
        B = min(B, o.B_override) if o.B_override else B
        if o.B_override and o.B_override > Bm:
            res.warnings.append("El burden impuesto supera la piedra máxima; se usa la práctica.")
        S = spacing(B, L)
        if not (konya and o.konya_S):
            lg.add(fase, "Espaciamiento", "S", S, "m", "S = 1.25·B" if o.SB == 1.25 else "S = (S/B)·B")
        nrow_y = max(1, math.ceil(o.width / S)) if m == PROD_METHODS[1] else max(1, round(o.width / S))
        ys = np.linspace(-o.width / 2 + o.width / (2 * nrow_y), o.width / 2 - o.width / (2 * nrow_y), nrow_y)
        nx = max(1, int((o.length - 0.5 * B) / B))
        u = (0.0, -math.cos(dip), -math.sin(dip))
        for i in range(nx):
            x = (i + 0.5) * B + 0.5 * B
            shift = (S / 2 if (o.staggered and i % 2 and nrow_y > 1) else 0.0)
            for y in ys:
                yy = y + shift
                if abs(yy) > o.width / 2 + 1e-6:
                    continue
                cargas, tacos, T, Lch = charge_column(L, B, bottom=False, plug=0.3, decks=True)
                add("Primera fila" if i == 0 else "Producción", i, x, yy, x, yy + 0.0, H, u, L, cargas, tacos, B, S)
        sources = [(0.0, y) for y in np.linspace(-o.width / 2, o.width / 2, 5)]
        outline = [(0, -o.width / 2), (o.length, -o.width / 2), (o.length, o.width / 2), (0, o.width / 2),
                   (0, -o.width / 2)]
        vol = min(o.rows_blast, nx) * B * o.width * H
        scene.update(box=(0, o.length, -o.width / 2, o.width / 2, 0, H), slot=True)
        xlabel, ylabel = "Avance desde el corte lateral x (m)", "Ancho / potencia y (m)"
        frag = dict(B=B, S=S, q=q, Lc=L - 0.3 - o.k_T * max(10 * D, min(0.7 * B, 25 * D)), H=L, limit=Bm)
        if m == PROD_METHODS[0]:
            lg.add(fase, "Corte lateral (chimenea)", "—", "1.8–3.5 m", "", "Raise borer o VCR")
            lg.add(fase, "Corte inferior", "—", "tiros en abanico 65 mm, malla 1.5 × 2 m", "", "≈ 800 g/t")
        if m == PROD_METHODS[1] and o.width < 1.0:
            res.warnings.append("Potencias menores a 1 m: la dilución puede ser alta con taladros largos.")
    elif m == PROD_METHODS[2]:  # abanicos
        AV = o.width
        ds = o.drill or DrillSetup(width=o.drift_w, height=o.drift_h)
        o.drift_w, o.drift_h = ds.width, ds.height
        z_lo, z_hi = (-o.down_h if o.fan_type != "Ascendente" else 0.0), (o.up_h if o.fan_type != "Descendente" else 0.0)
        drift = drift_polygon(ds)
        dp = (ds.pivot_y, ds.pivot_h)
        ring = Polygon([(-AV / 2, z_lo), (AV / 2, z_lo), (AV / 2, o.drift_h + z_hi), (-AV / 2, o.drift_h + z_hi)])
        F_d = "2a. Labor de perforación y equipo"
        lg.add(F_d, "Labor de perforación", "—", f"{ds.kind.split(' (')[0]} {ds.width:.2f} × {ds.height:.2f} m", "",
               f"área {drift.area:.2f} m²")
        lg.add(F_d, "Punto de pivote del equipo", "(y, z)", f"({dp[0]:.2f}, {dp[1]:.2f})", "m",
               "altura típica 1.3–1.5 m sobre el piso")
        lg.add(F_d, "Rectángulo del carrusel", "a × h", f"{ds.car_w:.2f} × {ds.car_h:.2f}", "m", "depende del equipo")
        if not rig_fits(drift, ds, dp):
            res.warnings.append("El carrusel del equipo no cabe en la labor de perforación con el pivote indicado: "
                                "amplíe la sección, baje el pivote o use un equipo más compacto.")
        if not (1.3 <= ds.pivot_h <= 1.5):
            res.warnings.append(f"Pivote a {ds.pivot_h:.2f} m del piso: lo habitual es 1.3–1.5 m.")
        Lavg = 0.5 * (math.hypot(AV / 2, z_hi + o.drift_h - dp[1]) + max(z_hi, abs(z_lo), 1.0))
        Bm, B = _burden(o, rock, ex, Lavg, q, o.f, lg, fase)
        B = o.B_override if o.B_override else B
        if not (1.2 <= B <= 1.8):
            res.warnings.append(f"Separación entre secciones B = {B:.2f} m fuera del rango típico 1.2–1.8 m (LJ).")
        S_toe = o.SB * B
        lg.add(fase, "Espaciamiento en el fondo", "S", S_toe, "m", "S/B = 1.3–2")
        ring_out = LineString(ring.exterior.coords)
        n_t = max(3, math.ceil(ring_out.length / S_toe))
        toes = []
        for k in range(n_t):
            p = ring_out.interpolate(k * ring_out.length / n_t)
            if drift.buffer(0.3).contains(p):
                continue
            if o.fan_type == "Ascendente" and p.y < o.drift_h * 0.5:
                continue
            if o.fan_type == "Descendente" and p.y > o.drift_h * 0.5:
                continue
            toes.append((p.x, p.y))
        toes.sort(key=lambda t: math.atan2(t[1] - dp[1], t[0] - dp[0]))
        angs = [math.atan2(ty - dp[1], tx - dp[0]) for tx, ty in toes]
        nr = max(1, int(o.length / B))
        Ls = []
        feed2d, dead = [], []
        for j, (tx, ty) in enumerate(toes):
            (cy, cz), dpc = collar_from_pivot(drift, dp, (tx, ty))
            dead.append(dpc)
            L = math.hypot(tx - cy, ty - cz)
            if L > 25:
                res.warnings.append("Hay taladros de más de 25 m: LJ recomienda sistemas de orientación para controlar "
                                    "la desviación.")
            da = min([abs(angs[j] - angs[k]) for k in (j - 1, j + 1) if 0 <= k < len(angs)] or [0.5])
            if o.fan_stem.startswith("Ts"):
                s0, stem_lab = ring_stemming(j, D, L)
                s0 = max(s0, 0.5)
            else:
                stem_lab = "geométrico"
                s0 = max(1.0, min(0.6 * L, 0.5 * S_toe / max(da, 1e-3)))
                if j % 2:  # alternar retacado largo/corto
                    s0 = min(0.75 * L, s0 * 1.35)
            lb = L / 5
            cargas = [(s0, max(s0, L - lb), o.col_ratio * q), (max(s0, L - lb), L, q)]
            Ls.append((L, L - s0, sum((b - a) * qq for a, b, qq in cargas)))
            ux, uz = (tx - cy) / L, (ty - cz) / L
            for i in range(nr):
                X0 = (i + 0.5) * B
                if i == 0:
                    lines2d.append(([cy, tx], [cz, ty]))
                    feed2d.append(([dp[0], cy], [dp[1], cz]))
                add("Primera fila" if i == 0 else "Producción", i, tx if i == 0 else tx, ty, X0, cy, cz,
                    (uz * 0 + 0.0, ux, uz), L, cargas, [(0.0, s0)], B, S_toe)
                rows[-1]["X0"], rows[-1]["Y0"], rows[-1]["Z0"] = X0, cy, cz
                rows[-1]["ux"], rows[-1]["uy"], rows[-1]["uz"] = 0.0, ux, uz
                rows[-1]["x"], rows[-1]["y"] = tx + (0.0 if i == 0 else 0.0), ty
        # 2D: vista de la sección (anillo); sólo el primer anillo se dibuja en 2D
        P["fan_rows"] = nr
        if o.fan_stem.startswith("Ts"):
            lg.add(fase, "Retacado escalonado de anillos", "T", f"Ts = {20 * D:.2f} · Tl = {125 * D:.2f} · Tm = {50 * D:.2f}",
                   "m", "secuencia Ts–Tl–Tm–Tl; T ≤ 2/3·L (AEL / Onederra y Chitombo 2007)")
        outline = list(ring.exterior.coords)
        if dead:
            lg.add(F_d, "Distancia pivote–boca (media / máx.)", "—", f"{np.mean(dead):.2f} / {max(dead):.2f}", "m",
                   "las bocas están donde la recta pivote–fondo corta el contorno de la labor")
        P["drill2d"] = dict(drift=list(drift.exterior.coords), pivots=[dp], carousels=[list(carousel(ds, dp).exterior.coords)],
                            feed=feed2d)
        xlabel, ylabel = "Ancho del tajeo y (m)", "Altura z (m)"
        ring_area = ring.area - drift.area
        vol = min(o.rows_blast, nr) * B * ring_area
        sources = [(t[0], t[1]) for t in toes]
        scene.update(box=(0, o.length, -AV / 2, AV / 2, z_lo, o.drift_h + z_hi),
                     drifts3d=[(list(drift.exterior.coords), 0.0, o.length)])
        Lm = np.mean([a for a, _, _ in Ls]) if Ls else 10
        Lcm = np.mean([b for _, b, _ in Ls]) if Ls else 8
        Qm = np.mean([c for _, _, c in Ls]) if Ls else 10
        lg.add(fase, "Longitud media de taladro", "L", Lm, "m")
        lg.add(fase, "N° de taladros por sección", "n", len(toes), "")
        lg.add(fase, "Carga por sección", "Q", Qm * len(toes), "kg")
        frag = dict(B=B, S=S_toe, q=Qm / max(Lcm, 0.1), Lc=Lcm, H=Lm, limit=2.0,
                    K_override=Qm * len(toes) / (B * ring_area))
        res.p["fan_note"] = ("En abanicos la piedra y el espaciamiento varían a lo largo del taladro; Kuz-Ram usa el "
                             "espaciamiento de fondo y la carga media por sección.")
    elif m == PROD_METHODS[3]:  # VCR
        Qs = 1.5 * math.pi * D ** 3 * ex.rho * 1000
        Dc = o.Et * Qs ** (1 / 3)
        Do = o.delta_o * Dc
        X = 4 / 3 * Do
        S = o.B_override if o.B_override else o.k_vcr * Do
        T = 12 * D
        for lab, sym, v, un, eq in [("Carga esférica (L = 6D)", "Q", Qs, "kg", "3πD³ρe/2"),
                                    ("Profundidad crítica", "D_c", Dc, "m", "E_t·Q^(1/3)"),
                                    ("Profundidad óptima", "D_o", Do, "m", "Δ_o·D_c"),
                                    ("Avance por pega", "X", X, "m", "4/3·D_o"),
                                    ("Espaciamiento", "S", S, "m", f"{o.k_vcr}·D_o (Fig. 24.9: 2.4 m con 165 mm)"),
                                    ("Retacado", "T", T, "m", "12·D")]:
            lg.add(fase, lab, sym, v, un, eq)
        n_bl = max(1, int((o.height - o.corona) / X))
        lg.add(fase, "N° de pegas ascendentes", "n", n_bl, "", "(H − corona)/X")
        rule = "una voladura, carga simétrica" if o.corona < 1.5 * X else (
            "una voladura, cargas espaciadas simultáneas" if o.corona <= 2.0 * X else "dos voladuras separadas")
        lg.add(fase, "Voladura de la corona (Tabla 24.2)", "—", rule, "", f"corona = {o.corona:.1f} m = {o.corona / X:.2f}·X")
        nx = max(1, int(o.length / S))
        ny = max(1, int(o.width / S))
        xs = np.linspace(S / 2, o.length - S / 2, nx)
        ys = np.linspace(-o.width / 2 + S / 2, o.width / 2 - S / 2, ny)
        H = o.height
        for x in xs:
            for y in ys:
                s0 = H - Do - 3 * D
                add("Cráter", 0, x, y, x, y, H, (0, 0, -1), H, [(s0, s0 + 6 * D, Qs / (6 * D))],
                    [(s0 - T, s0), (H - 0.3, H)], S, S)
        sources = [((xs.min() + xs.max()) / 2, 0.0)]
        outline = [(0, -o.width / 2), (o.length, -o.width / 2), (o.length, o.width / 2), (0, o.width / 2),
                   (0, -o.width / 2)]
        vol = o.length * o.width * X
        scene.update(box=(0, o.length, -o.width / 2, o.width / 2, 0, H))
        xlabel, ylabel = "Largo x (m)", "Potencia y (m)"
        frag = dict(B=S, S=S, q=Qs / (6 * D), Lc=6 * D, H=X, limit=1.6 * Do)
        if not (3 <= o.width <= 10):
            res.warnings.append("VCR se adapta bien a yacimientos de 3 a 10 m de potencia (LJ).")
    elif m in (PROD_METHODS[4], PROD_METHODS[8]) and not (m == PROD_METHODS[8] and o.orient_h):
        Hb = o.phase_h if m == PROD_METHODS[8] else o.height
        J = 0.3
        Bm, B = _burden(o, rock, ex, Hb, q, o.f, lg, fase)
        B = min(B, o.B_override) if o.B_override else B
        J = 0.3 * B
        L = Hb + J
        S = spacing(B, Hb)
        lg.add(fase, "Sobreperforación", "J", J, "m", "0.3·B" + (" (Konya ec. 6.6)" if konya else ""))
        if not (konya and o.konya_S):
            lg.add(fase, "Espaciamiento", "S", S, "m")
        W = o.room if m == PROD_METHODS[4] else o.width
        nx = max(1, int(o.length / B))
        ny = max(1, round(W / S))
        ys = np.linspace(-W / 2 + W / (2 * ny), W / 2 - W / (2 * ny), ny)
        for i in range(nx):
            x = (i + 1) * B
            for y in ys:
                yy = y + (S / 2 if (o.staggered and i % 2 and ny > 1) else 0)
                if abs(yy) > W / 2:
                    continue
                cargas, tacos, T, Lch = charge_column(L, B, bottom=True)
                add("Primera fila" if i == 0 else "Producción", i, x, yy, x, yy, Hb, (0, 0, -1), L, cargas, tacos, B, S)
        sources = [(0.0, y) for y in np.linspace(-W / 2, W / 2, 5)]
        outline = [(0, -W / 2), (o.length, -W / 2), (o.length, W / 2), (0, W / 2), (0, -W / 2)]
        vol = min(o.rows_blast, nx) * B * W * Hb
        scene.update(box=(0, o.length, -W / 2, W / 2, 0, Hb))
        xlabel, ylabel = "Avance del banco x (m)", "Ancho y (m)"
        frag = dict(B=B, S=S, q=q, Lc=L - o.k_T * max(10 * D, min(0.7 * B, 25 * D)), H=L, limit=Bm)
        if m == PROD_METHODS[4]:
            rec = 1 - o.pillar ** 2 / (o.pillar + o.room) ** 2
            lg.add(fase, "Recuperación (pilares cuadrados)", "R", rec * 100, "%", "1 − Wp²/(Wp + Wo)²")
        else:
            lg.add(fase, "Fases de la cámara", "—", f"{math.ceil(o.height / o.phase_h)} bancos de {o.phase_h} m",
                   "", "galería piloto + destrozas (modo Avance) y banqueo")
    else:  # frontales horizontales: cámaras y pilares frontal, breasting, obra pública horizontal
        if m == PROD_METHODS[6]:  # realce inclinado
            hs = o.height
            inc = math.radians(o.incl_deg)
            L = hs / math.sin(inc)
            Bm, B = _burden(o, rock, ex, L, q, o.f, lg, fase)
            B = min(B, o.B_override) if o.B_override else B
            S = spacing(B, L)
            lg.add(fase, "Longitud del barreno inclinado", "L", L, "m", f"h/sen({o.incl_deg:.0f}°)")
            if L > 4.5:
                res.warnings.append("En corte y relleno la perforación no suele superar 4 m (LJ).")
            W = o.width
            ny = max(1, math.ceil(W / S))
            ys = np.linspace(-W / 2 + W / (2 * ny), W / 2 - W / (2 * ny), ny)
            nx = max(1, int(o.length / B))
            for i in range(nx):
                x = (i + 1) * B
                for y in ys:
                    cargas, tacos, T, Lch = charge_column(L, B, bottom=True)
                    add("Primera fila" if i == 0 else "Producción", i, x, y, x, y, 0.0,
                        (math.cos(inc), 0, math.sin(inc)), L, cargas, tacos, B, S)
                    if y == ys[0]:
                        lines2d.append(([x, x + L * math.cos(inc)], [y, y]))
            sources = [(0.0, y) for y in np.linspace(-W / 2, W / 2, 5)]
            outline = [(0, -W / 2), (o.length, -W / 2), (o.length, W / 2), (0, W / 2), (0, -W / 2)]
            vol = min(o.rows_blast, nx) * B * W * hs
            scene.update(box=(0, o.length + L * math.cos(inc), -W / 2, W / 2, 0, hs))
            xlabel, ylabel = "Avance x (m)", "Potencia de la veta y (m)"
            frag = dict(B=B, S=S, q=q, Lc=L - o.k_T * max(10 * D, min(0.7 * B, 25 * D)), H=L, limit=Bm)
            ce_gt = None
        else:
            Hf = o.height if m != PROD_METHODS[8] else o.phase_h
            W = o.room if m == PROD_METHODS[5] else o.width
            Lh = min(o.length, 4.0 if m == PROD_METHODS[7] else o.length)
            L = Lh
            Bm, B = _burden(o, rock, ex, L, q, o.f, lg, fase)
            B = min(B, o.B_override) if o.B_override else B
            S = spacing(B, L)
            lg.add(fase, "Longitud de barreno", "L", L, "m", "≤ varilla / brazo del jumbo")
            nz = max(1, int(Hf / B))
            from_top = m != PROD_METHODS[7]  # breasting: cara libre inferior
            zs = [Hf - B * (k + 1) for k in range(nz)] if from_top else [B * (k + 1) for k in range(nz)]
            zs = [z for z in zs if 0.2 <= z <= Hf - 0.2] + ([0.15] if from_top else [Hf - 0.15])
            ny = max(1, round(W / S))
            ys = np.linspace(-W / 2 + W / (2 * ny), W / 2 - W / (2 * ny), ny)
            for k, z in enumerate(zs):
                last = k == len(zs) - 1
                for y in ys:
                    cargas, tacos, T, Lch = charge_column(L, B, bottom=True)
                    zona = "Zapatera" if (last and from_top) else ("Contorno" if last else
                                                                   ("Primera fila" if k == 0 else "Producción"))
                    add(zona, k, y, z, 0.0, y, z, (1, 0, 0), L, cargas, tacos, B, S)
            sources = [(y, Hf if from_top else 0.0) for y in np.linspace(-W / 2, W / 2, 5)]
            outline = [(-W / 2, 0), (W / 2, 0), (W / 2, Hf), (-W / 2, Hf), (-W / 2, 0)]
            vol = W * Hf * L * 0.95
            scene.update(box=(0, L, -W / 2, W / 2, 0, Hf), front=True)
            xlabel, ylabel = "Ancho y (m)", "Altura z (m)"
            frag = dict(B=B, S=S, q=q, Lc=L - o.k_T * max(10 * D, min(0.7 * B, 25 * D)), H=L, limit=Bm)
    res.holes = finalize_holes(rows, order)
    Lmax = float(res.holes.L.max()) if len(res.holes) else 0.0
    if Lmax > 0:
        lg.add("2b. Control de desviación", "Desviación al fondo del taladro más largo", "α·L", o.alpha * Lmax, "m",
               f"α = {o.alpha * 100:.1f} %, L = {Lmax:.1f} m")
        if Lmax >= 8 and o.alpha > 0.03:
            res.warnings.append(f"Desviación de {o.alpha * 100:.1f} % en taladros de {Lmax:.0f} m: el curso de taladros "
                                f"largos considera aceptable 2–3 % (en frentes se tolera hasta 5 %).")
        res.p["burden_table"] = burden_table(rock, ex, o, Lmax, q)
        if m in PROD_METHODS[:4]:
            Lcap = o.Lmax_ft * 0.3048
            n_over = int((res.holes.L > Lcap + 1e-6).sum())
            lg.add("2b. Control de desviación", "Capacidad de perforación del equipo", "L_máx", Lcap, "m",
                   f"{o.Lmax_ft:.0f} pies (Simba)")
            lg.add("2b. Control de desviación", "Barras para el taladro más largo", "n", math.ceil(Lmax / (o.rod_ft * 0.3048)),
                   "", f"barras de {o.rod_ft:.0f} pies")
            if n_over:
                res.warnings.append(f"{n_over} taladro(s) superan la capacidad del equipo ({o.Lmax_ft:.0f} pies = {Lcap:.1f} m; "
                                    f"el más largo mide {Lmax:.1f} m): reduzca la altura entre subniveles, agregue un "
                                    f"subnivel intermedio o perfore desde ambos niveles.")
    if m == PROD_METHODS[2]:
        lg.add("2c. Tiempos (Onederra y Chitombo 2007)", "Retardo entre taladros de un anillo", "Δt",
               "3–15 ms/m de burden; cooperación óptima 5–25 ms", "", "")
    res.relief = pd.DataFrame(columns=["x", "y", "d_mm", "X0", "Y0", "Z0", "ux", "uy", "uz", "L"])
    res.zone_order = [z for z in order if z in set(res.holes.zona)]
    res.zone_colors = colors_for(order)
    res.sources = sources
    res.view2d = dict(kind="plan" if outline and m not in (PROD_METHODS[2], PROD_METHODS[5], PROD_METHODS[7]) else "front",
                      outline=outline, openings=[], lines=lines2d, xlabel=xlabel, ylabel=ylabel)
    res.scene3d = scene
    if P.get("drill2d"):
        res.view2d["drill"] = P["drill2d"]
    if m in (PROD_METHODS[0], PROD_METHODS[1], PROD_METHODS[3]) and len(res.holes):
        _parallel_drilling(res, o, scene, lg)
    P["vol"] = vol
    if m == PROD_METHODS[2]:  # en 2D sólo el primer anillo
        res.p["view_filter"] = "Primera fila"
    kg_blast = _kg_per_blast(res, o)
    P["rows_blast"] = o.rows_blast
    P["blast_all"] = (m in (PROD_METHODS[3], PROD_METHODS[5], PROD_METHODS[7]) or (m == PROD_METHODS[8] and o.orient_h))
    ce_gt = 1000 * kg_blast / (vol * rock.rho) if vol else np.nan
    lg.add("3. Resultados", "Explosivo por voladura", "Q", kg_blast, "kg", f"{o.rows_blast} fila(s)/sección(es)")
    lg.add("3. Resultados", "Consumo específico", "CE", ce_gt, "g/t")
    if m in (PROD_METHODS[6], PROD_METHODS[7]) and not (150 <= ce_gt <= 330):
        res.warnings.append(f"Consumo de {ce_gt:.0f} g/t; LJ indica 200–260 g/t en corte y relleno.")
    res.summary = summary_common(res, vol, rock.rho, cost, extra={"Volumen por voladura (m³)": vol})
    res.summary["Explosivo total (kg)"] = kg_blast
    res.summary["Factor de carga (kg/m³)"] = kg_blast / vol if vol else np.nan
    res.summary["Factor de carga (g/t)"] = ce_gt
    res.summary["Explosivo de toda la malla dibujada (kg)"] = float(res.holes.kg.sum())
    res.frag = dict(B=frag["B"], S=frag["S"], q=frag["q"], Lc=max(frag["Lc"], 0.2), H=frag["H"], d_mm=o.D_mm,
                    W=o.e_coll + o.alpha * frag["H"], rws=ex.rws * 100, modo="optimizar", B_limit=frag["limit"],
                    K_override=frag.get("K_override"), zona="producción")
    return res


def _parallel_drilling(res, o: ProdOptions, scene, lg):
    """Taladros paralelos desde una labor de perforación superior: ubicación de las labores, pivote del equipo
    sobre la prolongación del eje de cada taladro y verificación de que el carrusel quepa en la labor."""
    ds = o.drill or DrillSetup()
    F_d = "2a. Labor de perforación y equipo"
    h = res.holes
    H = float(h.Z0.max())
    W = o.width
    narrow = res.metodo == PROD_METHODS[1]
    if ds.full_width:
        from dataclasses import replace as _rep
        ds = _rep(ds, kind="Rectangular", width=W + 0.4, n_drifts=1, drift_y=0.0)
    centers, span = drift_centers(ds, W, narrow)
    sec = ds.section()
    lg.add(F_d, "Labor de perforación", "—", ("Cámara de ancho completo " if (o.drill and o.drill.full_width) else "")
           + f"{ds.kind.split(' (')[0]} {ds.width:.2f} × {ds.height:.2f} m", "",
           f"piso en la cota del techo del tajeo (z = {H:.1f} m)")
    lg.add(F_d, "N° de labores de perforación", "n", len(centers), "",
           "dato" if ds.n_drifts else f"automático: ancho del tajeo / (recorrido útil del pivote {span:.2f} m + carrusel)")
    lg.add(F_d, "Pivote del equipo", "h_p", ds.pivot_h, "m", "sobre el piso de la labor (1.3–1.5 m)")
    lg.add(F_d, "Rectángulo del carrusel", "a × h", f"{ds.car_w:.2f} × {ds.car_h:.2f}", "m", "depende del equipo")
    drifts = [drift_polygon(ds, c, H) for c in centers]
    reach = []
    piv = []
    for _, r in h.iterrows():
        k = int(np.argmin([abs(r.Y0 - c) for c in centers]))
        uz = r.uz if abs(r.uz) > 1e-6 else -1e-6
        t = ds.pivot_h / (-uz) if uz < 0 else ds.pivot_h
        py, pz = r.Y0 - r.uy * t, r.Z0 - r.uz * t if uz < 0 else r.Z0 + ds.pivot_h
        ok = rig_fits(drifts[k], ds, (py, pz)) and abs(r.Y0 - centers[k]) <= sec.floor_width / 2 + 1e-6
        reach.append(ok)
        piv.append((py, pz))
    h["alcance"] = reach
    res.holes = h
    n_bad = int((~h.alcance).sum())
    first = h[h.grupo == h.grupo.min()]
    if n_bad:
        res.warnings.append(f"{n_bad} taladro(s) quedan fuera del alcance del equipo: el carrusel no cabe en la labor "
                            f"al alinear el pivote con el eje del taladro. Agregue labores de perforación, ensánchelas, "
                            f"o cambie a taladros en abanico.")
    lg.add(F_d, "Taladros fuera del alcance del equipo", "—", n_bad, "", "carrusel centrado en el pivote fuera de la labor")
    if o.width > len(centers) * sec.floor_width + 1e-6:
        lg.add(F_d, "Ancho cubierto por las labores", "—", f"{len(centers) * sec.floor_width:.1f} de {o.width:.1f} m", "", "")
    # vista en sección (primera fila) y planta
    dip = math.radians(o.dip_deg) if res.metodo == PROD_METHODS[1] else math.pi / 2
    sh = H / math.tan(dip) if dip < math.pi / 2 - 1e-6 else 0.0
    stope = [(-W / 2, H), (W / 2, H), (W / 2 - sh, 0.0), (-W / 2 - sh, 0.0), (-W / 2, H)]
    holes2d, ch2d, pv2d, car2d = [], [], [], []
    for (_, r), (py, pz) in zip(h.iterrows(), piv):
        if r.grupo != first.grupo.iloc[0]:
            continue
        y1, z1 = r.Y0 + r.uy * r.L, r.Z0 + r.uz * r.L
        holes2d.append(([r.Y0, y1], [r.Z0, z1], bool(r.alcance), int(r.id)))
        for s0, s1, _ in r.cargas:
            ch2d.append(([r.Y0 + r.uy * s0, r.Y0 + r.uy * s1], [r.Z0 + r.uz * s0, r.Z0 + r.uz * s1]))
        pv2d.append((py, pz, bool(r.alcance)))
    if pv2d:
        ys = [p[0] for p in pv2d]
        for k in (int(np.argmin(ys)), int(np.argmax(ys))):
            car2d.append(list(carousel(ds, pv2d[k][:2]).exterior.coords))
    res.p["section_view"] = dict(stope=stope, drifts=[list(d.exterior.coords) for d in drifts], holes=holes2d,
                                 charges=ch2d, pivots=pv2d, carousels=car2d, H=H)
    for c in centers:
        res.view2d.setdefault("lines", [])
        hw = sec.floor_width / 2
        L0 = scene.get("box", (0, o.length))[1] if scene.get("box") else o.length
        res.view2d["lines"] += [([0, L0], [c - hw, c - hw]), ([0, L0], [c + hw, c + hw])]
    x0, x1 = (scene["box"][0], scene["box"][1]) if scene.get("box") else (0.0, o.length)
    scene["drifts3d"] = [(list(d.exterior.coords), x0, x1) for d in drifts]


def _kg_per_blast(res, o):
    h = res.holes
    if h.empty:
        return 0.0
    if res.metodo == PROD_METHODS[3]:
        return float(h.kg.sum())
    if res.metodo in (PROD_METHODS[5], PROD_METHODS[7]) or (res.metodo == PROD_METHODS[8] and o.orient_h):
        return float(h.kg.sum())
    return float(h[h.grupo < o.rows_blast].kg.sum())
