"""Avance de producción: galería o subnivel que avanza dentro de la veta (el material volado se cuenta como
producción). La malla del frente se diseña como un avance (Holmberg o Konya) y aquí se superpone la veta en el
frente: área de mineral y de desmonte, toneladas por disparo, dilución, ley diluida, contenido metálico y el
dominio (mineral / caja techo / caja piso) donde queda cada taladro."""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .common import Result, Rock

VEIN_COLOR = "#d4a017"
HW_COLOR = "#c0392b"
FW_COLOR = "#2c7fb8"
HW_SIDES = ["Derecha", "Izquierda"]
SEL_ORDERS = ["Mineral primero, luego el desmonte (desquinche)", "Desmonte primero, luego el mineral (circado)"]


@dataclass
class VeinOptions:
    potencia: float = 1.2         # potencia (ancho verdadero) de la veta, m
    dip_deg: float = 70.0         # buzamiento de la veta, °
    hw_side: str = HW_SIDES[0]    # lado del frente hacia donde buza la veta (caja techo)
    offset: float = 0.0           # desplazamiento horizontal del eje de la veta a media altura (+ a la derecha), m
    grade: float = 0.0            # ley in situ del mineral
    grade_unit: str = "g/t"
    waste_grade: float = 0.0      # ley del desmonte (cajas)
    selective: bool = False       # voladura selectiva: mineral y desmonte en disparos separados
    order: str = SEL_ORDERS[0]
    e_contact: float = 0.10       # sobrerotura del disparo hacia el dominio que aún no se vuela, m
    residual: float = 0.05        # fracción del material de un disparo que queda y se mezcla con el siguiente


def _pieces(geom):
    if geom.is_empty:
        return []
    return [g for g in getattr(geom, "geoms", [geom]) if g.area > 1e-6]


def vein_polygons(section, o: VeinOptions):
    """Polígonos (en el plano del frente) de la veta, la caja techo y la caja piso, recortados a la sección."""
    s = section.polygon
    x0, y0, x1, y1 = s.bounds
    ym = (y0 + y1) / 2
    big = 4 * max(x1 - x0, y1 - y0) + 10
    sgn = 1.0 if o.hw_side == HW_SIDES[0] else -1.0
    dip = math.radians(min(max(o.dip_deg, 5.0), 90.0))
    cot = math.cos(dip) / math.sin(dip)
    half = o.potencia / (2 * math.sin(dip))       # semiancho horizontal de la veta

    def xc(y):
        return o.offset - sgn * (y - ym) * cot

    ya, yb = ym - big, ym + big
    left = [(xc(ya) - half, ya), (xc(yb) - half, yb)]
    right = [(xc(ya) + half, ya), (xc(yb) + half, yb)]
    vein = Polygon(left + right[::-1])
    far_l = Polygon([(xc(ya) - half - 2 * big, ya), (xc(yb) - half - 2 * big, yb)] + left[::-1])
    far_r = Polygon(right + [(xc(yb) + half + 2 * big, yb), (xc(ya) + half + 2 * big, ya)])
    hw, fw = (far_r, far_l) if sgn > 0 else (far_l, far_r)
    return dict(vein=s.intersection(vein), hw=s.intersection(hw), fw=s.intersection(fw), half=half, xc=xc, ym=ym)


def apply_vein(res: Result, section, o: VeinOptions, ore: Rock, hw: Rock | None = None, fw: Rock | None = None,
               cost=None):
    """Agrega al resultado del avance el balance mineral / desmonte de la veta en el frente."""
    lg = res.log
    F = "8. Veta en el frente (avance de producción)"
    g = vein_polygons(section, o)
    A = section.area
    a_ore, a_hw, a_fw = g["vein"].area, g["hw"].area, g["fw"].area
    rho_hw = hw.rho if hw else ore.rho
    rho_fw = fw.rho if fw else ore.rho
    adv = float(res.summary.get("Avance por disparo (m)", res.p.get("adv", 0.0)))
    lg.add(F, "Potencia de la veta", "w", o.potencia, "m", "ancho verdadero")
    lg.add(F, "Buzamiento", "β", o.dip_deg, "°", f"caja techo a la {o.hw_side.lower()} del frente")
    lg.add(F, "Ancho horizontal de la veta en el frente", "w_h", 2 * g["half"], "m", "w / sen β")
    lg.add(F, "Área de mineral en el frente", "A_min", a_ore, "m²", "sección ∩ veta")
    lg.add(F, "Área de desmonte — caja techo", "A_ct", a_hw, "m²")
    lg.add(F, "Área de desmonte — caja piso", "A_cp", a_fw, "m²")
    t_ore = a_ore * adv * ore.rho
    t_hw, t_fw = a_hw * adv * rho_hw, a_fw * adv * rho_fw
    t_w = t_hw + t_fw
    t_tot = t_ore + t_w
    lg.add(F, "Mineral por disparo", "T_min", t_ore, "t", "A_min · avance · ρ_mineral")
    lg.add(F, "Desmonte por disparo", "T_des", t_w, "t", "Σ A_caja · avance · ρ_caja")
    dil = t_w / t_tot if t_tot > 0 else 0.0
    lg.add(F, "Dilución", "D", 100 * dil, "%", "T_des / (T_min + T_des)")
    lg.add(F, "Dilución (respecto al mineral)", "D'", 100 * t_w / t_ore if t_ore > 0 else float("nan"), "%",
           "T_des / T_min")
    g_dil = (t_ore * o.grade + t_w * o.waste_grade) / t_tot if (o.grade and t_tot > 0) else None
    metal = None
    if o.grade:
        lg.add(F, "Ley diluida", "ley_d", g_dil, o.grade_unit,
               f"(T_min·{o.grade:g} + T_des·{o.waste_grade:g}) / (T_min + T_des)")
        metal = t_tot * g_dil * (1 / 100 if o.grade_unit == "%" else 1.0)
        mu = {"g/t": "g", "%": "t de metal", "oz/t": "oz"}.get(o.grade_unit, "")
        lg.add(F, "Contenido metálico por disparo", "M", metal, mu, "(T_min + T_des) · ley diluida")
        if o.grade_unit == "g/t":
            lg.add(F, "Contenido metálico por metro de avance", "M/m", metal / adv if adv else float("nan"), "g/m")

    # dominio de cada taladro (por la boca)
    def dom(x, y):
        p = Point(x, y)
        for name, key in (("Mineral", "vein"), ("Caja techo", "hw"), ("Caja piso", "fw")):
            if g[key].buffer(1e-6).contains(p):
                return name
        return "Mineral"

    H = res.holes
    H["dominio"] = [dom(x, y) for x, y in zip(H.x, H.y)]
    kg_ore = float(H.loc[H.dominio == "Mineral", "kg"].sum())
    n_ore = int((H.dominio == "Mineral").sum())
    lg.add(F, "Taladros en mineral / en cajas", "N", f"{n_ore} / {len(H) - n_ore}", "", "según la boca del taladro")
    lg.add(F, "Explosivo en mineral", "kg_min", kg_ore, "kg")

    # avisos
    w_lab = section.polygon.bounds[2] - section.polygon.bounds[0]
    if a_ore < 1e-3:
        res.warnings.append("La veta no corta la sección del frente: revise la posición y el buzamiento.")
    elif a_ore > 0.97 * A and 2 * g["half"] > w_lab:
        res.warnings.append(f"La veta (ancho horizontal {2 * g['half']:.2f} m) es más ancha que la labor "
                            f"({w_lab:.2f} m): queda mineral en los hastiales, que se recupera con el tajeo.")
    if dil > 0.5:
        res.warnings.append(f"Más de la mitad del frente es desmonte (dilución {100 * dil:.0f} %): la labor es más "
                            "ancha que la veta. Considere reducir la sección"
                            + ("." if o.selective else " o la voladura selectiva (circado)."))
    if dil > 0.5 and (hw or fw):
        res.warnings.append("La malla se diseñó con las propiedades del mineral pero el frente es mayormente caja; "
                            "verifique el burden con la caja.")

    # resumen
    S = res.summary
    kg = S.get("Explosivo total (kg)", 0.0)
    S["Tonelaje roto (t)"] = t_tot
    if t_tot > 0:
        S["Factor de carga (g/t)"] = 1000 * kg / t_tot
    S.update({"Potencia de la veta (m)": o.potencia, "Área de mineral en el frente (m²)": a_ore,
              "Mineral por disparo (t)": t_ore, "Desmonte por disparo (t)": t_w, "Dilución (%)": 100 * dil,
              "Mineral por metro de avance (t/m)": t_ore / adv if adv else float("nan")})
    if g_dil is not None:
        S[f"Ley diluida ({o.grade_unit})"] = g_dil
    if cost and "Costo total por disparo (US$)" in S and t_ore > 0:
        S["Costo por tonelada (US$/t)"] = S["Costo total por disparo (US$)"] / t_tot
        S["Costo por tonelada de mineral (US$/t)"] = S["Costo total por disparo (US$)"] / t_ore

    # vistas
    regs = []
    for key, name, col in (("vein", "Veta (mineral)", VEIN_COLOR), ("hw", "Caja techo", HW_COLOR),
                           ("fw", "Caja piso", FW_COLOR)):
        for i, pc in enumerate(_pieces(g[key])):
            xs, ys = pc.exterior.coords.xy
            regs.append((list(xs), list(ys), name if i == 0 else name + f" ({i + 1})", col))
    res.view2d["domain_regions"] = regs
    res.view2d.setdefault("domain_lines", [])
    res.scene3d["vein"] = [list(pc.exterior.coords) for pc in _pieces(g["vein"])]
    res.metodo = res.metodo + " — avance en veta"
    res.p["vein"] = dict(t_ore=t_ore, t_waste=t_w, dilution=dil, grade_dil=g_dil, metal=metal, area_ore=a_ore,
                         potencia=o.potencia, dip=o.dip_deg)
    return res


# =============================================================================
# Voladura selectiva (circado / desquinche)
# =============================================================================
DOM_KEYS = (("vein", "Mineral"), ("hw", "Caja techo"), ("fw", "Caja piso"))


def selective_regions(section, o: VeinOptions):
    """Regiones del frente en el orden en que se vuelan. Cada una: dict(dom, key, geom)."""
    g = vein_polygons(section, o)
    pcs = {k: sorted(_pieces(g[k]), key=lambda p: -p.area) for k, _ in DOM_KEYS}
    name = dict(DOM_KEYS)
    vein = [dict(dom="Mineral", key="vein", geom=p) for p in pcs["vein"]]
    waste = sorted([dict(dom=name[k], key=k, geom=p) for k in ("hw", "fw") for p in pcs[k]], key=lambda r: -r["geom"].area)
    if not vein or not waste:
        return [], g
    if o.order == SEL_ORDERS[0]:
        regs = vein + waste
    else:
        regs = waste[:1] + vein + waste[1:]
    return regs, g


def selective_cut_x(section, o: VeinOptions, y_cut: float):
    """Centro horizontal, a la altura del cuele, de la región que sale en el primer disparo (allí va el cuele) y
    ancho disponible en esa altura."""
    regs, _ = selective_regions(section, o)
    if not regs:
        return None, 0.0
    x0, _, x1, _ = section.polygon.bounds
    seg = regs[0]["geom"].intersection(LineString([(x0 - 1, y_cut), (x1 + 1, y_cut)]))
    segs = [gg for gg in getattr(seg, "geoms", [seg]) if gg.length > 0]
    if not segs:
        c = regs[0]["geom"].representative_point()
        return c.x, 0.0
    best = max(segs, key=lambda q: q.length)
    return best.centroid.x, best.length


def apply_selective(res: Result, section, o: VeinOptions, ore: Rock, hw: Rock | None = None, fw: Rock | None = None,
                    cut_width: float = 0.0):
    """Divide el frente en disparos (mineral y desmonte por separado) y calcula lo que va a planta y a desmonte.

    Cada disparo vuela su región más una franja de sobrerotura e_c hacia las regiones que aún no se vuelan
    (la carga del borde rompe más allá del contacto); además, una fracción r del material de cada disparo no se
    limpia y sale con el disparo siguiente. Los disparos de mineral van a planta; los de desmonte, al botadero
    (o como relleno)."""
    lg = res.log
    F = "9. Voladura selectiva"
    regs, g = selective_regions(section, o)
    if not regs:
        res.warnings.append("Voladura selectiva: el frente es todo mineral o todo desmonte; se dispara en una sola pega.")
        return res
    H = res.holes
    adv = float(res.summary.get("Avance por disparo (m)", res.p.get("adv", 0.0)))
    rho = {"Mineral": ore.rho, "Caja techo": (hw or ore).rho, "Caja piso": (fw or ore).rho}

    # taladros por región (por la boca); el cuele siempre en el primer disparo
    def region_of(x, y):
        p = Point(x, y)
        d = [r["geom"].distance(p) for r in regs]
        return int(min(range(len(regs)), key=lambda i: d[i]))

    idx = [0 if c else region_of(x, y) for x, y, c in zip(H.x, H.y, H.cuele)]
    # regiones sin taladros: su roca sale con el disparo anterior (o con el siguiente si es la primera)
    groups = []
    for i, r in enumerate(regs):
        if i in idx or not groups:
            groups.append(dict(dom=r["dom"], geoms=[r["geom"]], regions=[i]))
        else:
            groups[-1]["geoms"].append(r["geom"])
            groups[-1]["regions"].append(i)
            res.warnings.append(f"Voladura selectiva: la franja de {r['dom'].lower()} ({r['geom'].area:.2f} m²) no "
                                "tiene taladros propios y sale con el disparo anterior.")
    reg2blast = {i: k for k, gp in enumerate(groups) for i in gp["regions"]}
    H["disparo"] = [reg2blast[i] + 1 for i in idx]
    rem = [unary_union(gp["geoms"]) for gp in groups]
    # el cuele sale en el primer disparo: la roca de su abertura va con él aunque sea de otro dominio
    ops = [Polygon(c).buffer(0) for _, c in res.view2d.get("openings", []) if len(c) >= 4]
    if ops:
        cut_poly = unary_union(ops).intersection(section.polygon)
        extra = cut_poly.difference(rem[0])
        if extra.area > 0.05 * max(cut_poly.area, 1e-6):
            doms = sorted({name for key, name in DOM_KEYS if extra.intersection(g[key]).area > 1e-4})
            cw_ = cut_poly.bounds[2] - cut_poly.bounds[0]
            res.warnings.append(f"Voladura selectiva: el cuele ({cw_:.2f} m de ancho) sobresale del primer disparo y "
                                f"rompe {extra.area:.2f} m² de {', '.join(d.lower() for d in doms)} que salen con él. "
                                "Use un cuele más angosto (quemado) o perforación de menor diámetro.")
        rem[0] = rem[0].union(cut_poly)
        for j in range(1, len(rem)):
            rem[j] = rem[j].difference(cut_poly)
    full = {k: g[k] for k, _ in DOM_KEYS}
    blasts, prev, sources = [], None, {}
    for k, gp in enumerate(groups):
        geom = rem[k]
        later = unary_union(rem[k + 1:]) if k + 1 < len(rem) else None
        if later is not None and not later.is_empty and o.e_contact > 0:
            band = geom.buffer(o.e_contact, join_style=2).intersection(later)
            geom = geom.union(band)
            for j in range(k + 1, len(rem)):
                rem[j] = rem[j].difference(band)
        comp = {name: geom.intersection(full[key]).area * adv * rho[name] for key, name in DOM_KEYS}
        if prev is not None:
            touch = rem[k].distance(prev) < 1e-3
            if not touch:
                res.warnings.append(f"Voladura selectiva: el disparo {k + 1} no tiene cara libre hacia el vacío del "
                                    "disparo anterior; revise el orden.")
            line = unary_union(gp["geoms"]).boundary.intersection(prev.buffer(1e-3))
            pts = []
            for ln in getattr(line, "geoms", [line]):
                if ln.length > 0:
                    n = max(2, int(ln.length / 0.2) + 1)
                    pts += [(ln.interpolate(t * ln.length / (n - 1)).x, ln.interpolate(t * ln.length / (n - 1)).y)
                            for t in range(n)]
            if not pts:
                c = prev.representative_point()
                pts = [(c.x, c.y)]
            sources[k + 1] = pts
        blasts.append(dict(dom=gp["dom"], comp=comp, geom=geom))
        prev = geom if prev is None else prev.union(geom)
    # material que no se limpia y pasa al disparo siguiente
    for k in range(len(blasts) - 1):
        carry = {n: o.residual * t for n, t in blasts[k]["comp"].items()}
        for n in carry:
            blasts[k]["comp"][n] -= carry[n]
            blasts[k + 1]["comp"][n] += carry[n]
    rows = []
    plant_ore = plant_w = dump_ore = dump_w = 0.0
    for k, b in enumerate(blasts):
        t_o = b["comp"]["Mineral"]
        t_w = b["comp"]["Caja techo"] + b["comp"]["Caja piso"]
        to_plant = b["dom"] == "Mineral"
        if to_plant:
            plant_ore += t_o
            plant_w += t_w
        else:
            dump_ore += t_o
            dump_w += t_w
        hk = H[H.disparo == k + 1]
        rows.append(dict(Disparo=k + 1, Dominio=b["dom"], Taladros=len(hk), Explosivo_kg=round(float(hk.kg.sum()), 2),
                         Mineral_t=round(t_o, 2), Desmonte_t=round(t_w, 2),
                         Destino="Planta" if to_plant else "Desmonte / relleno"))
    tab = pd.DataFrame(rows)
    ore_insitu = g["vein"].area * adv * ore.rho
    plant = plant_ore + plant_w
    dil_sel = plant_w / plant if plant > 0 else 0.0
    rec = plant_ore / ore_insitu if ore_insitu > 0 else 0.0
    dil_conv = res.p.get("vein", {}).get("dilution", 0.0)
    lg.add(F, "Orden de los disparos", "—", " → ".join(f"{k + 1}: {b['dom'].lower()}" for k, b in enumerate(blasts)), "",
           o.order)
    lg.add(F, "Sobrerotura hacia el dominio aún no volado", "e_c", o.e_contact, "m", "franja que sale con el disparo")
    lg.add(F, "Material no limpiado que pasa al disparo siguiente", "r", 100 * o.residual, "%")
    if cut_width:
        lg.add(F, "Ancho del primer disparo a la altura del cuele", "a₁", cut_width, "m", "el cuele va dentro")
    lg.add(F, "Mineral a planta", "T_p,min", plant_ore, "t")
    lg.add(F, "Desmonte que va a planta (dilución)", "T_p,des", plant_w, "t")
    lg.add(F, "Mineral que se pierde con el desmonte", "T_pérd", dump_ore, "t")
    lg.add(F, "Dilución con voladura selectiva", "D_sel", 100 * dil_sel, "%", "T_p,des / (T_p,min + T_p,des)")
    lg.add(F, "Dilución con un solo disparo", "D", 100 * dil_conv, "%", "fase 8")
    lg.add(F, "Recuperación minera", "R", 100 * rec, "%", "T_p,min / mineral in situ del frente")
    g_pl = None
    if o.grade and plant > 0:
        g_pl = (plant_ore * o.grade + plant_w * o.waste_grade) / plant
        lg.add(F, "Ley que va a planta", "ley_p", g_pl, o.grade_unit, "(T_p,min·ley + T_p,des·ley_des) / T_p")
        mu = {"g/t": "g", "%": "t de metal", "oz/t": "oz"}.get(o.grade_unit, "")
        lg.add(F, "Contenido metálico a planta", "M_p", plant * g_pl * (1 / 100 if o.grade_unit == "%" else 1.0), mu)
    lg.add(F, "Disparos por avance", "N", len(blasts), "", "cada uno con su propia pega, limpieza y ventilación")
    S = res.summary
    S.update({"Disparos por avance (selectiva)": len(blasts), "Mineral a planta (t)": plant_ore,
              "Desmonte a planta (t)": plant_w, "Dilución selectiva (%)": 100 * dil_sel,
              "Recuperación minera (%)": 100 * rec})
    if g_pl is not None:
        S[f"Ley a planta ({o.grade_unit})"] = g_pl
    if "Costo total por disparo (US$)" in S and plant > 0:
        S["Costo por tonelada a planta (US$/t)"] = S["Costo total por disparo (US$)"] / plant
    if dil_sel >= dil_conv - 0.02 and dil_conv > 0:
        res.warnings.append("La voladura selectiva casi no reduce la dilución con estos datos (franja de sobrerotura "
                            "o material no limpiado altos).")
    w_first = cut_width
    if 0 < w_first < 0.6:
        res.warnings.append(f"El primer disparo ({blasts[0]['dom'].lower()}) mide {w_first:.2f} m a la altura del "
                            "cuele: un cuele quemado de pocos taladros (labor estrecha) o perforación de menor "
                            "diámetro.")
    res.p["selective"] = dict(table=tab, sources=sources, n=len(blasts), dil_sel=dil_sel, dil_conv=dil_conv, rec=rec,
                              grade_plant=g_pl, unit=o.grade_unit, plant_ore=plant_ore, plant_w=plant_w, dump_ore=dump_ore)
    res.metodo = res.metodo.replace("avance en veta", "avance en veta, voladura selectiva")
    res.zone_colors = dict(res.zone_colors)
    return res
