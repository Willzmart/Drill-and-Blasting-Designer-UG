"""Macizo rocoso por dominios en producción: mineral, caja techo, caja piso y roca del techo (corona).

Cada dominio tiene sus propiedades (σt, E, Vp o Q, densidad) y, con ellas, su VPP crítica σt·Vp/E (curso de
taladros largos, sesión 08). Con el modelo de Holmberg–Persson de la app (PPV = K·[q/r·(atan(L2/r) − atan(L1/r))]^α)
se calcula, para la carga de cada taladro de borde, el radio hasta el que se alcanza el umbral de sobrerotura de la
caja (n × VPP crítica; n = 4 por defecto, como el umbral de rotura de la plantilla del curso). La sobrerotura del
taladro es ese radio menos su distancia efectiva al contacto (distancia geométrica − emboquille − desviación·s), y
la sobrerotura equivalente de la pared (ELOS, Clark y Pakalnis) es el promedio en la fila de borde.

Dilución (por metro de corrida): Σ ELOS_i·L_pared_i·ρ_i / (A_mineral·ρ_mineral + Σ ELOS_i·L_pared_i·ρ_i).
Ley diluida = ley · t_mineral / (t_mineral + t_estéril).

Medidas de control en los taladros de borde: factor de carga manual o automático (reduce la carga lineal en
taladros paralelos al contacto o recorta la carga del extremo que se acerca al contacto en abanicos, hasta que la
sobrerotura estimada no supere el ELOS objetivo).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .common import Rock

EDGE_MODES = ["Sin ajuste", "Factor manual en taladros de borde", "Automático (ELOS objetivo)"]
GRADE_UNITS = ["g/t", "%", "oz/t"]


@dataclass
class DomainOptions:
    on: bool = True
    hw: Rock = field(default_factory=lambda: Rock(RMR=40, UCS=45, rho=2.70, Q=0.6, E=20.0, sigma_t=4.0))
    fw: Rock = field(default_factory=lambda: Rock(RMR=55, UCS=80, rho=2.70, Q=3.0, E=35.0, sigma_t=7.0))
    back: Rock | None = None          # roca del techo / corona (abanicos ascendentes)
    hw_side: float = -1.0             # cuerpos verticales y abanicos: caja techo al lado −y (−1) o +y (+1)
    K: float = 700.0                  # Holmberg–Persson
    alpha: float = 0.7
    n_ob: float = 4.0                 # umbral de sobrerotura = n × VPP crítica
    grade: float = 0.0
    grade_unit: str = GRADE_UNITS[0]
    edge_mode: str = EDGE_MODES[0]
    edge_factor: float = 0.6
    elos_target: float = 0.3
    # cámaras y pilares
    floor: Rock | None = None         # roca del piso (None = mineral)
    roof_waste: bool = True           # el techo es estéril (diluye) o mineral
    floor_waste: bool = False         # el piso es estéril (diluye) o mineral
    depth: float = 200.0              # profundidad bajo superficie (m)
    h_head: float = 4.0               # altura de la galería piloto sobre el banco (m): techo a Hb + h_head
    esr: float = 3.0                  # Barton: labores mineras temporales 3–5
    K_hg: float = 133.0               # Hedley y Grant (MPa)
    rho_over: float = 2.7             # densidad media del recubrimiento (t/m³)


def ppv_hp(q, Lc, r, K, alpha):
    """PPV (mm/s) a la distancia perpendicular r del centro de una carga de longitud Lc y carga lineal q."""
    r = max(r, 1e-3)
    return K * (q / r * 2 * math.atan(Lc / (2 * r))) ** alpha


def damage_radius(q, Lc, ppv_lim, K, alpha, rmax=200.0):
    """Distancia a la que la PPV de Holmberg–Persson baja a ppv_lim (bisección)."""
    if q <= 0 or Lc <= 0:
        return 0.0
    if ppv_hp(q, Lc, rmax, K, alpha) >= ppv_lim:
        return rmax
    lo, hi = 1e-3, rmax
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ppv_hp(q, Lc, mid, K, alpha) > ppv_lim:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def contacts(o, methods, dom: DomainOptions, ore: Rock):
    """Contactos del tajeo en la sección (y, z): caja techo, caja piso y techo. Devuelve la lista de contactos y
    el área de mineral por metro de corrida."""
    m = o.method
    W = o.width
    out = []
    if m in (methods[4], methods[5]):     # cámaras y pilares: paredes = pilares (mineral), techo y piso
        Hb = o.height
        W = o.room
        left, right = ((-W / 2, Hb), (-W / 2, 0.0)), ((W / 2, Hb), (W / 2, 0.0))
        center = (0.0, Hb / 2)
        area = W * Hb
        roof = dom.back or ore
        flo = dom.floor or ore
        zt = Hb + dom.h_head
        out = [dict(name="Pilar izquierdo", rock=ore, a=left[0], b=left[1], color="#ff7f0e", dilutes=False),
               dict(name="Pilar derecho", rock=ore, a=right[0], b=right[1], color="#ff7f0e", dilutes=False),
               dict(name="Techo" + (" (estéril)" if dom.roof_waste else " (mineral)"), rock=roof, a=(-W / 2, zt),
                    b=(W / 2, zt), color="#8c564b", dilutes=dom.roof_waste),
               dict(name="Piso" + (" (estéril)" if dom.floor_waste else " (mineral)"), rock=flo, a=(-W / 2, 0.0),
                    b=(W / 2, 0.0), color="#7f7f7f", dilutes=dom.floor_waste, no_inside=True)]
        return _finish_contacts(out, center), area, center
    if m in (methods[6], methods[7]):     # corte y relleno: tajada de potencia W y altura h sobre el relleno
        hs = o.height
        left, right = ((-W / 2, hs), (-W / 2, 0.0)), ((W / 2, hs), (W / 2, 0.0))
        hw_left = dom.hw_side < 0
        center = (0.0, hs / 2)
        area = W * hs
        out.append(dict(name="Techo de la tajada (mineral)", rock=ore, a=(-W / 2, hs), b=(W / 2, hs),
                        color="#8c564b", dilutes=False))
    elif m == methods[2]:
        z_lo = -o.down_h if o.fan_type != "Ascendente" else 0.0
        z_top = o.drift_h + (o.up_h if o.fan_type != "Descendente" else 0.0)
        left, right = ((-W / 2, z_top), (-W / 2, z_lo)), ((W / 2, z_top), (W / 2, z_lo))
        hw_left = dom.hw_side < 0
        center = (0.0, 0.5 * (z_top + z_lo))
        area = W * (z_top - z_lo)
        if o.fan_type != "Descendente" and dom.back is not None:
            out.append(dict(name="Techo (corona)", rock=dom.back, a=(-W / 2, z_top), b=(W / 2, z_top), color="#8c564b"))
    else:
        H = o.height
        dip = o.dip_deg if m in (methods[0], methods[1]) else 90.0
        sh = o.dip_side * H / math.tan(math.radians(dip)) if dip < 89.99 else 0.0
        left, right = ((-W / 2, H), (-W / 2 - sh, 0.0)), ((W / 2, H), (W / 2 - sh, 0.0))
        hw_left = (o.dip_side > 0) if dip < 89.99 else (dom.hw_side < 0)
        center = (-sh / 2, H / 2)
        area = W * H
    hw, fw = (left, right) if hw_left else (right, left)
    out = [dict(name="Caja techo", rock=dom.hw, a=hw[0], b=hw[1], color="#d62728"),
           dict(name="Caja piso", rock=dom.fw, a=fw[0], b=fw[1], color="#9467bd")] + out
    return _finish_contacts(out, center), area, center


def _finish_contacts(out, center, dom=None):
    for c in out:
        a, b = np.array(c["a"], float), np.array(c["b"], float)
        t = b - a
        Lw = float(np.linalg.norm(t))
        n = np.array([-t[1], t[0]]) / Lw
        if np.dot(np.array(center) - a, n) < 0:
            n = -n
        c.setdefault("dilutes", True)
        c.update(n=n, Lw=Lw, ppv_c=c["rock"].ppv_crit)
    return out


def _dist(row, c, s, o):
    y = row["Y0"] + row["uy"] * s
    z = row["Z0"] + row["uz"] * s
    d = float(np.dot(np.array([y, z]) - np.array(c["a"]), c["n"]))
    return d, d - (o.e_coll + o.alpha * s)


def _hole_vs(row, c, o):
    """Distancia mínima (sin y con desviación) de las cargas del taladro al contacto y la carga más cercana."""
    best = None
    for s0, s1, q in row["cargas"]:
        for s in (s0, s1):
            d, de = _dist(row, c, s, o)
            if best is None or de < best[1]:
                best = (d, de, q, s1 - s0, s0, s1)
    return best


def _analyze(rows, cons, o, dom, S_med):
    res = []
    for c in cons:
        info = []
        for k, r in enumerate(rows):
            if not r["cargas"]:
                continue
            b = _hole_vs(r, c, o)
            if b is not None:
                info.append((k,) + b)
        if not info:
            res.append(dict(c=c, edge=[], elos=0.0, inside=0, under=False, dmin=np.nan))
            continue
        dmin = min(i[1] for i in info)
        # fila de borde: en cada fila / anillo (grupo), los taladros más cercanos al contacto
        groups = {}
        for i in info:
            groups.setdefault(rows[i[0]].get("grupo", 0), []).append(i)
        edge = []
        for g in groups.values():
            gm = min(i[1] for i in g)
            if gm <= dmin + 1.5 * S_med:
                edge += [i for i in g if i[1] <= gm + 0.5 * S_med]
        obs = []
        for k, d, de, q, Lc, s0, s1 in edge:
            R = damage_radius(q, Lc, c["thr"], dom.K, dom.alpha)
            obs.append(max(0.0, R - de))
        inside = sum(1 for i in info if i[1] < -1e-3)
        res.append(dict(c=c, edge=[i[0] for i in edge], elos=float(np.mean(obs)) if obs else 0.0, inside=inside,
                        dmin=dmin, obs=obs, info=edge))
    return res


def _recalc(row, w_cart):
    row["kg"] = float(sum((s1 - s0) * q for s0, s1, q in row["cargas"]))
    row["n_cart"] = int(max(1, round(row["kg"] / w_cart))) if (w_cart and row["kg"] > 0) else 0
    if row["cargas"]:
        row["primer_s"] = max(s1 for _, s1, _ in row["cargas"]) - 0.12


def _mitigate(rows, an, o, dom, w_cart):
    """Ajusta la carga de los taladros de borde. Devuelve el n° de taladros ajustados y los que no alcanzan."""
    done, short = set(), 0
    for a in an:
        c = a["c"]
        for k, d, de, q, Lc, s0, s1 in a.get("info", []):
            if k in done:
                continue
            r = rows[k]
            if dom.edge_mode == EDGE_MODES[1]:
                r["cargas"] = [(x0, x1, qq * dom.edge_factor) for x0, x1, qq in r["cargas"]]
                _recalc(r, w_cart)
                done.add(k)
                continue
            allow = de + dom.elos_target
            if damage_radius(q, Lc, c["thr"], dom.K, dom.alpha) <= allow:
                continue
            ds = [_dist(r, c, s, o)[1] for s in (s0, s1)]
            if abs(ds[1] - ds[0]) < 0.3:            # taladro paralelo al contacto: bajar la carga lineal
                lo, hi = 0.15, 1.0
                if damage_radius(lo * q, Lc, c["thr"], dom.K, dom.alpha) > allow:
                    f = lo
                    short += 1
                else:
                    for _ in range(40):
                        mid = 0.5 * (lo + hi)
                        if damage_radius(mid * q, Lc, c["thr"], dom.K, dom.alpha) > allow:
                            hi = mid
                        else:
                            lo = mid
                    f = lo
                r["cargas"] = [(x0, x1, qq * f) for x0, x1, qq in r["cargas"]]
            else:                                   # abanico: recortar la carga del extremo cercano al contacto
                L0 = sum(x1 - x0 for x0, x1, _ in r["cargas"])
                new = []
                for x0, x1, qq in r["cargas"]:
                    R = damage_radius(qq, x1 - x0, c["thr"], dom.K, dom.alpha)
                    need = R - dom.elos_target
                    e0, e1 = _dist(r, c, x0, o)[1], _dist(r, c, x1, o)[1]
                    if min(e0, e1) >= need:
                        new.append((x0, x1, qq))
                        continue
                    if max(e0, e1) <= need:
                        continue
                    sc = x0 + (need - e0) / (e1 - e0) * (x1 - x0)
                    new.append((sc, x1, qq) if e0 < e1 else (x0, sc, qq))
                kept = sum(x1 - x0 for x0, x1, _ in new)
                if kept < 0.3 * L0:                 # no recortar más del 70 %: se baja la carga lineal
                    short += 1
                    new = [(x0, x1, qq * 0.5) for x0, x1, qq in r["cargas"]]
                r["cargas"] = new
            _recalc(r, w_cart)
            done.add(k)
    return len(done), short


def apply_domains(rows, o, methods, dom: DomainOptions, ore: Rock, w_cart, lg, warnings):
    """Analiza (y, si se pide, ajusta) los taladros de borde frente a cada contacto. Marca la columna 'borde' en
    los taladros y devuelve el resumen para vistas, memoria y PDF."""
    cons, area, center = contacts(o, methods, dom, ore)
    for c in cons:
        c["thr"] = dom.n_ob * c["ppv_c"]
    Ss = [r.get("S") for r in rows if r.get("S") == r.get("S") and r.get("S")]
    S_med = float(np.median(Ss)) if Ss else 1.5
    an0 = _analyze(rows, cons, o, dom, S_med)
    n_adj, n_short = 0, 0
    if dom.edge_mode != EDGE_MODES[0]:
        n_adj, n_short = _mitigate(rows, an0, o, dom, w_cart)
    an = _analyze(rows, cons, o, dom, S_med) if n_adj else an0
    for r in rows:
        r["borde"] = ""
    for a in an:
        for k in a["edge"]:
            rows[k]["borde"] = (rows[k]["borde"] + " · " if rows[k]["borde"] else "") + a["c"]["name"]
    F = "2c. Macizo por dominios"
    lg.add(F, "Mineral: VPP crítica", "VPP_c", ore.ppv_crit, "mm/s",
           f"σt·Vp/E = {ore.st:.1f}·{ore.Vp:.0f}/{ore.E:.0f}; ρ = {ore.rho:.2f} t/m³")
    lg.add(F, "Modelo de vibración", "—", f"Holmberg–Persson K = {dom.K:.0f}, α = {dom.alpha:.2f}", "",
           f"umbral de sobrerotura = {dom.n_ob:.1f} × VPP crítica de cada roca")
    t_ore = area * ore.rho
    t_w, tab = 0.0, []
    for a, a0 in zip(an, an0):
        c = a["c"]
        rk = c["rock"]
        lg.add(F, f"{c['name']}: VPP crítica / umbral", "—", f"{c['ppv_c']:,.0f} / {c['thr']:,.0f}", "mm/s",
               f"σt = {rk.st:.1f} MPa, Vp = {rk.Vp:.0f} m/s, E = {rk.E:.0f} GPa, ρ = {rk.rho:.2f} t/m³")
        lg.add(F, f"{c['name']}: distancia mínima de la carga al contacto", "d_ef", a["dmin"], "m",
               "geométrica − emboquille − desviación·s")
        lg.add(F, f"{c['name']}: taladros de borde", "n", len(a["edge"]), "",
               "en cada fila / anillo, los más cercanos al contacto")
        if dom.edge_mode != EDGE_MODES[0]:
            lg.add(F, f"{c['name']}: ELOS sin ajuste", "ELOS₀", a0["elos"], "m", "")
        lg.add(F, f"{c['name']}: sobrerotura equivalente", "ELOS", a["elos"], "m",
               "promedio de (radio de sobrerotura − d_ef) en la fila de borde")
        tw = a["elos"] * c["Lw"] * rk.rho if c["dilutes"] else 0.0
        t_w += tw
        tab.append([c["name"], f"{c['ppv_c']:,.0f}", f"{c['thr']:,.0f}", f"{a['dmin']:.2f}", len(a["edge"]),
                    f"{a0['elos']:.2f}", f"{a['elos']:.2f}", f"{tw:.1f}" if c["dilutes"] else "no diluye"])
        if not c["dilutes"]:
            eff = ("adelgaza el pilar (ver FS)" if c["name"].startswith("Pilar") else
                   "daño bajo la cota del piso" if c["name"].startswith("Piso") else
                   "daño al techo de trabajo (mineral de la siguiente tajada)")
            lg.add(F, f"{c['name']}: efecto", "—", eff, "", "es mineral: no diluye")
        if a["inside"] and not c.get("no_inside"):
            warnings.append(f"{a['inside']} taladro(s) tienen carga dentro de la {c['name'].lower()} considerando la "
                            f"desviación: retírelos del contacto o controle la desviación.")
        if a["elos"] > 0.5 and not c["dilutes"]:
            if c["name"].startswith("Pilar"):
                warnings.append(f"{c['name']}: sobrerotura estimada de {a['elos']:.2f} m: adelgaza el pilar; use voladura "
                                f"controlada (carga reducida o desacoplada) en los taladros junto a los pilares.")
            elif c["name"].startswith("Piso"):
                warnings.append(f"{c['name']}: daño estimado de {a['elos']:.2f} m bajo la cota del piso: reduzca la "
                                f"sobreperforación o la carga de fondo (modo automático).")
            else:
                warnings.append(f"{c['name']}: daño estimado de {a['elos']:.2f} m sobre el techo de trabajo: use taco de "
                                f"fondo o recorte la carga en el extremo de los taladros (modo automático) y refuerce el "
                                f"sostenimiento.")
        elif a["elos"] > 0.5:
            warnings.append(f"{c['name']}: sobrerotura estimada de {a['elos']:.2f} m (ELOS > 0.5 m): reduzca la carga de "
                            f"los taladros de borde, aléjelos del contacto o use un diámetro menor.")
        if c["name"].startswith("Caja") and a["info"]:
            br = [damage_radius(q, Lc, dom.n_ob * ore.ppv_crit, dom.K, dom.alpha) for _, d, de, q, Lc, *_ in a["info"]]
            if min(i[1] for i in a["info"]) > max(br) + 1e-6:
                warnings.append(f"{c['name']}: la fila de borde no alcanza a romper hasta el contacto (radio de rotura "
                                f"{max(br):.2f} m): riesgo de mineral pegado a la pared (subrotura).")
    dil = t_w / (t_ore + t_w) if (t_ore + t_w) > 0 else 0.0
    lg.add(F, "Mineral por metro de corrida", "t_m", t_ore, "t/m", "área de mineral de la sección × ρ")
    lg.add(F, "Estéril por sobrerotura", "t_e", t_w, "t/m", "Σ ELOS·L_pared·ρ_caja")
    lg.add(F, "Dilución estimada", "D", 100 * dil, "%", "t_e / (t_m + t_e)")
    g_dil = dom.grade * (1 - dil) if dom.grade else None
    if g_dil is not None:
        lg.add(F, "Ley diluida", "ley", g_dil, dom.grade_unit, f"ley in situ {dom.grade:.2f} {dom.grade_unit} × (1 − D)")
    if n_adj:
        lg.add(F, "Taladros de borde con carga ajustada", "n", n_adj, "", dom.edge_mode)
    if n_short:
        warnings.append(f"{n_short} taladro(s) de borde no llegan al ELOS objetivo ni con la carga mínima: aléjelos del "
                        f"contacto o reduzca el diámetro.")
    rp = None
    if o.method in (methods[4], methods[5]):
        rp = pillar_roof(o, dom, ore, an, lg, warnings)
    return dict(rp=rp, contacts=[dict(name=a["c"]["name"], a=a["c"]["a"], b=a["c"]["b"], n=list(a["c"]["n"]),
                               color=a["c"]["color"], elos=a["elos"], elos0=a0["elos"], ppv_c=a["c"]["ppv_c"],
                               thr=a["c"]["thr"]) for a, a0 in zip(an, an0)],
                dilution=dil, t_ore=t_ore, t_waste=t_w, grade=dom.grade, grade_dil=g_dil, unit=dom.grade_unit,
                table=tab, n_adj=n_adj, ore_ppv=ore.ppv_crit, n_ob=dom.n_ob)


def domain_shapes(dinfo, band=2.0, kz=1.0):
    """Polígonos de las cajas (franja exterior a cada contacto) y líneas de sobrerotura (contacto desplazado
    ELOS hacia afuera), en coordenadas de la sección (y, z·kz)."""
    regs, obs = [], []
    for c in dinfo["contacts"]:
        a, b, n = np.array(c["a"]), np.array(c["b"]), -np.array(c["n"])
        poly = [a, b, b + n * band, a + n * band, a]
        regs.append(([p[0] for p in poly], [p[1] * kz for p in poly], c["name"], c["color"]))
        if c["elos"] > 1e-3:
            p, q = a + n * c["elos"], b + n * c["elos"]
            obs.append(([p[0], q[0]], [p[1] * kz, q[1] * kz], f"Sobrerotura {c['name'].lower()} ({c['elos']:.2f} m)",
                        c["color"]))
    return regs, obs


def _lunder_pakalnis(ucs, w, h):
    """Lunder y Pakalnis (1997). Para pilares muy esbeltos (w/h ≤ 0.25) log(w/h + 0.75) ≤ 0: el confinamiento
    medio es nulo (Cpav = 0, κ = 0) y la resistencia queda en 0.44·σci·0.68."""
    r = max(w / h, 1e-3)
    base = math.log10(r + 0.75)
    cpav = 0.46 * base ** (1.4 / r) if base > 0 else 0.0
    cpav = min(max(cpav, 0.0), 0.999)
    kappa = math.tan(math.acos((1 - cpav) / (1 + cpav)))
    return 0.44 * ucs * (0.68 + 0.52 * kappa)


def pillar_roof(o, dom, ore, an, lg, warnings):
    """Pilares cuadrados por área tributaria (Hedley y Grant; Lunder y Pakalnis) con el ancho efectivo tras la
    sobrerotura, y abertura máxima del techo de Barton (2·ESR·Q^0.4) frente a la abertura real."""
    F = "2d. Cámaras y pilares: pilares y techo"
    el = [a["elos"] for a in an if a["c"]["name"].startswith("Pilar")]
    elos_p = float(np.mean(el)) if el else 0.0
    w, Wc = o.pillar, o.room
    hp = o.height + dom.h_head
    w_ef = max(w - 2 * elos_p, 0.1)
    gamma = dom.rho_over * 9.81 / 1000            # MN/m³
    sv = gamma * dom.depth
    out = {}
    lg.add(F, "Esfuerzo vertical in situ", "σv", sv, "MPa", f"γ·H = {gamma:.4f}·{dom.depth:.0f}")
    lg.add(F, "Pilar: ancho de diseño / altura", "w / h", f"{w:.2f} / {hp:.2f}", "m",
           f"altura = banco {o.height:.1f} + galería piloto {dom.h_head:.1f}")
    lg.add(F, "Sobrerotura media en los pilares", "ELOS_p", elos_p, "m", "de los taladros junto a los pilares")
    lg.add(F, "Ancho efectivo del pilar", "w_ef", w_ef, "m", "w − 2·ELOS_p")
    if w / hp < 0.5:
        warnings.append(f"Pilar muy esbelto (w/h = {w / hp:.2f} < 0.5): fuera del rango de calibración de Lunder–Pakalnis "
                        f"y Hedley–Grant; aumente el ancho del pilar o reduzca su altura.")
    rows = []
    for lab, ww in (("diseño", w), ("tras la voladura", w_ef)):
        sp = sv * ((ww + Wc + (w - ww)) / ww) ** 2      # el área tributaria no cambia; el pilar se adelgaza
        shg = dom.K_hg * ww ** 0.5 / hp ** 0.75
        slp = _lunder_pakalnis(ore.UCS, ww, hp)
        rows.append((lab, ww, sp, shg, slp))
        lg.add(F, f"Esfuerzo en el pilar ({lab})", "σp", sp, "MPa", "σv·((w + Wc)/w_ef)² (área tributaria)")
        lg.add(F, f"Resistencia Hedley–Grant ({lab})", "σs", shg, "MPa", f"K·w^0.5/h^0.75, K = {dom.K_hg:.0f} MPa")
        lg.add(F, f"Resistencia Lunder–Pakalnis ({lab})", "σs", slp, "MPa", f"0.44·σci·(0.68 + 0.52·κ), σci = {ore.UCS:.0f}")
        lg.add(F, f"FS del pilar ({lab})", "FS", f"{shg / sp:.2f} (H–G) · {slp / sp:.2f} (L–P)", "", "")
    fs_hg, fs_lp = rows[1][3] / rows[1][2], rows[1][4] / rows[1][2]
    out["FS pilar Hedley–Grant (diseño → tras voladura)"] = f"{rows[0][3] / rows[0][2]:.2f} → {fs_hg:.2f}"
    if fs_lp < 1.4 or fs_hg < 1.5:
        warnings.append(f"Pilar: FS tras la voladura {fs_hg:.2f} (Hedley–Grant) / {fs_lp:.2f} (Lunder–Pakalnis) por "
                        f"debajo de 1.5 / 1.4: use voladura controlada junto a los pilares o aumente su ancho.")
    roof = dom.back or ore
    wmax = 2 * dom.esr * max(roof.Q, 1e-3) ** 0.4
    w_real = Wc + 2 * elos_p
    lg.add(F, "Abertura máxima sin sostenimiento (Barton)", "W_máx", wmax, "m",
           f"2·ESR·Q^0.4; ESR = {dom.esr:.1f}, Q del techo = {roof.Q:.2f}")
    lg.add(F, "Abertura real de la cámara", "W_real", w_real, "m", "Wc + 2·ELOS_p")
    lg.add(F, "Abertura en la intersección", "√2·W_real", math.sqrt(2) * w_real, "m", "diagonal entre pilares")
    if w_real > wmax:
        warnings.append(f"Techo: abertura real {w_real:.1f} m mayor que la máxima sin sostenimiento ({wmax:.1f} m, "
                        f"Barton): instale sostenimiento sistemático o reduzca el ancho de la cámara.")
    elif math.sqrt(2) * w_real > wmax:
        warnings.append(f"Techo: en las intersecciones la abertura llega a {math.sqrt(2) * w_real:.1f} m (> {wmax:.1f} m): "
                        f"refuerce el sostenimiento en los cruces.")
    rec0 = 1 - w ** 2 / (w + Wc) ** 2
    rec1 = 1 - w_ef ** 2 / (w + Wc) ** 2
    lg.add(F, "Recuperación (diseño → efectiva)", "R", f"{100 * rec0:.1f} → {100 * rec1:.1f}", "%",
           "la sobrerotura de los pilares sube la recuperación pero los debilita")
    summ = {"Ancho efectivo del pilar (m)": w_ef, "FS pilar Hedley–Grant tras voladura": fs_hg,
            "FS pilar Lunder–Pakalnis tras voladura": fs_lp, "Abertura real de la cámara (m)": w_real,
            "Abertura máxima Barton (m)": wmax}
    return dict(summary=summ, w=w, w_ef=w_ef, elos_p=elos_p, rows=rows, wmax=wmax, w_real=w_real)
