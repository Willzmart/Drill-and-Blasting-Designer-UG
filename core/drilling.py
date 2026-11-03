"""Labor de perforación, punto de pivote del equipo y carrusel para taladros largos (abanicos y paralelos).

Convención en la sección transversal del tajeo: y horizontal (m, transversal a la labor), z vertical (m),
piso de la labor de perforación en z = 0 (abanicos) o en el techo del tajeo (taladros paralelos
descendentes). El equipo se representa, como en el diseño 2D de anillos, por un punto de pivote a 1.3–1.5 m
del piso y por un rectángulo que encierra el carrusel; el rectángulo debe caber dentro de la labor.
Las bocas de los taladros se ubican donde la línea pivote–fondo corta el contorno de la labor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon

from .sections import GALLERY_TYPES, build_section


@dataclass
class DrillSetup:
    kind: str = GALLERY_TYPES[0]
    width: float = 4.0
    height: float = 4.0
    radius: float | None = None
    top_w: float | None = None
    pivot_h: float = 1.4          # altura del pivote sobre el piso (m), 1.3–1.5 m típico
    pivot_y: float = 0.0          # desplazamiento lateral del pivote respecto al eje de la labor (m)
    car_w: float = 1.2            # ancho del rectángulo del carrusel (m) — varía por equipo
    car_h: float = 0.8            # alto del rectángulo del carrusel (m)
    car_dz: float = 0.0           # desplazamiento vertical del rectángulo respecto al pivote (m)
    n_drifts: int = 0             # taladros paralelos: n° de labores de perforación (0 = automático)
    drift_y: float = 0.0          # taladros paralelos: eje de la labor respecto al eje del tajeo (m)
    full_width: bool = False      # taladros paralelos: cámara de perforación de ancho completo (corte superior)

    def section(self):
        r = self.radius if self.kind.startswith(("Baúl",)) else None
        tw = self.top_w if self.kind.startswith("Trap") else None
        return build_section(self.kind, self.width, self.height, r, tw)


def drift_polygon(ds: DrillSetup, dy=0.0, dz=0.0) -> Polygon:
    return affinity.translate(ds.section().polygon, dy, dz)


def carousel(ds: DrillSetup, pivot) -> Polygon:
    py, pz = pivot
    zc = pz + ds.car_dz
    return Polygon([(py - ds.car_w / 2, zc - ds.car_h / 2), (py + ds.car_w / 2, zc - ds.car_h / 2),
                    (py + ds.car_w / 2, zc + ds.car_h / 2), (py - ds.car_w / 2, zc + ds.car_h / 2)])


def rig_fits(drift: Polygon, ds: DrillSetup, pivot) -> bool:
    return drift.buffer(1e-6).contains(carousel(ds, pivot)) and drift.contains(Point(pivot))


def collar_from_pivot(drift: Polygon, pivot, toe):
    """Boca del taladro: salida de la recta pivote → fondo por el contorno de la labor.
    Devuelve (boca, distancia pivote–boca). Si el pivote está fuera de la labor, la boca es el pivote."""
    P = Point(pivot)
    if not drift.contains(P):
        return tuple(pivot), 0.0
    d = math.hypot(toe[0] - pivot[0], toe[1] - pivot[1])
    if d < 1e-9:
        return tuple(pivot), 0.0
    far = (pivot[0] + (toe[0] - pivot[0]) / d * (d + 50), pivot[1] + (toe[1] - pivot[1]) / d * (d + 50))
    inter = LineString([pivot, far]).intersection(drift.exterior)
    pts = []
    for g in getattr(inter, "geoms", [inter]):
        if g.is_empty:
            continue
        if g.geom_type == "Point":
            pts.append((g.x, g.y))
        else:
            pts += list(g.coords)
    if not pts:
        return tuple(pivot), 0.0
    c = min(pts, key=lambda q: math.hypot(q[0] - pivot[0], q[1] - pivot[1]))
    return c, math.hypot(c[0] - pivot[0], c[1] - pivot[1])


def drift_centers(ds: DrillSetup, stope_w: float, narrow: bool):
    """Ejes de las labores de perforación para taladros paralelos."""
    sec = ds.section()
    span = max(sec.floor_width - ds.car_w, 0.5)   # recorrido lateral útil del pivote
    if ds.n_drifts > 0:
        n = ds.n_drifts
    elif narrow:
        n = 1
    else:
        n = max(1, math.ceil(stope_w / span))
    if n == 1:
        return [ds.drift_y], span
    return list(np.linspace(-stope_w / 2 + stope_w / (2 * n), stope_w / 2 - stope_w / (2 * n), n)), span
