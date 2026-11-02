"""Distribución de energía del explosivo.

1. Energía por taladro: E = Q·e (MJ), con e en MJ/kg (RWS·3.82 si no hay dato del fabricante; ANFO = 3.82
   MJ/kg según la Guía del explosivista de Enaex).
2. Concentración 3D de explosivo de Kleine (1988), tal como la presentan Onederra y Chitombo (2007,
   "Design methodology for underground ring blasting", apéndice): para cada carga (columna de densidad
   lineal q entre las abscisas L1 y L2 medidas desde el pie de la perpendicular, a distancia h del eje):

        P = 3·q / (4·π·ρr) · (1/h²) · (L2/r2 − L1/r1)      [kg/t]

   que con q = 1000·ρe·π·D²/4 se reduce a la ec. (3) del paper, P = 187.5·D²·(ρe/ρr)·(L2/r2 − L1/r1)/h².
   Multiplicada por la energía específica del explosivo da MJ/t; por ρr, kg/m³. En el plano del burden
   aproxima el factor de carga (Onederra: umbral de rotura en el pie de los anillos ≈ 0.25–0.35 kg/t).
3. Velocidad pico de partícula de campo cercano de Holmberg y Persson (1980):

        PPV = K · [ q · ∫ dx / (h² + (x − x0)²) ]^α = K · [ (q/h)·(atan(L2/h) − atan(L1/h)) ]^α

   por taladro (los taladros salen en retardos distintos: se toma la envolvente). Umbrales del curso de
   taladros largos: VPP crítica = σt·Vp/E (inicio del daño) y ≈ 4·VPP crítica (rotura), como en la
   plantilla "Cálculo_Burden_Espaciamiento" (hoja 1).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

AX = "XYZ"


def blast_holes(res, scope="voladura"):
    """Taladros de una voladura (producción: filas o anillos por voladura) o toda la malla."""
    h = res.holes
    if scope != "voladura" or res.labor != "Producción" or res.p.get("blast_all"):
        return h
    return h[h.grupo < res.p.get("rows_blast", 10 ** 6)]


def segments(res, emap: dict, e_default: float, holes=None):
    """Cargas de los taladros como segmentos: origen a, dirección u, longitud, q, e, id, radio."""
    A, U, Ls, Q, E, I, R = [], [], [], [], [], [], []
    for _, r in (res.holes if holes is None else holes).iterrows():
        u = np.array([r.ux, r.uy, r.uz], dtype=float)
        c = np.array([r.X0, r.Y0, r.Z0], dtype=float)
        e = emap.get(r.explosivo, e_default)
        for s0, s1, qq in r.cargas:
            if s1 - s0 <= 1e-3 or qq <= 0:
                continue
            A.append(c + u * s0)
            U.append(u)
            Ls.append(s1 - s0)
            Q.append(qq)
            E.append(e)
            I.append(int(r.id))
            R.append(r.d_mm / 2000)
    return dict(a=np.array(A).reshape(-1, 3), u=np.array(U).reshape(-1, 3), L=np.array(Ls), q=np.array(Q),
                e=np.array(E), id=np.array(I), r=np.array(R))


def _geom(P, seg, k):
    w = P - seg["a"][k]
    t = w @ seg["u"][k]
    h2 = np.maximum(np.einsum("ij,ij->i", w, w) - t * t, 0.0)
    h = np.maximum(np.sqrt(h2), max(seg["r"][k], 0.05))
    return h, -t, seg["L"][k] - t


def kleine(P, seg, rho_r, energy=False, chunk=6000):
    """Concentración 3D (kg/t, o MJ/t si energy=True) en los puntos P (N, 3)."""
    P = np.asarray(P, dtype=float)
    out = np.zeros(len(P))
    for i0 in range(0, len(P), chunk):
        Pc = P[i0:i0 + chunk]
        acc = np.zeros(len(Pc))
        for k in range(len(seg["L"])):
            h, L1, L2 = _geom(Pc, seg, k)
            r1, r2 = np.sqrt(h * h + L1 * L1), np.sqrt(h * h + L2 * L2)
            coef = 3 * seg["q"][k] / (4 * math.pi * rho_r) * (seg["e"][k] if energy else 1.0)
            acc += coef * (L2 / r2 - L1 / r1) / (h * h)
        out[i0:i0 + chunk] = acc
    return out


def hp_ppv(P, seg, K=700.0, alpha=0.7, chunk=4000):
    """PPV de Holmberg–Persson (mm/s): envolvente de los taladros individuales."""
    P = np.asarray(P, dtype=float)
    ids = np.unique(seg["id"])
    col = {v: i for i, v in enumerate(ids)}
    cidx = np.array([col[v] for v in seg["id"]])
    out = np.zeros(len(P))
    for i0 in range(0, len(P), chunk):
        Pc = P[i0:i0 + chunk]
        I = np.zeros((len(Pc), len(ids)))
        for k in range(len(seg["L"])):
            h, L1, L2 = _geom(Pc, seg, k)
            I[:, cidx[k]] += seg["q"][k] / h * (np.arctan(L2 / h) - np.arctan(L1 / h))
        out[i0:i0 + chunk] = K * np.max(np.maximum(I, 0.0), axis=1) ** alpha
    return out


def plane_frame(res):
    """Plano de análisis 2D según la vista del diseño: normal, ejes del plano, rango y valor por defecto."""
    h = res.holes
    ends = []
    for _, r in h.iterrows():
        for s0, s1, _ in r.cargas:
            for s in (s0, s1):
                ends.append((r.X0 + r.ux * s, r.Y0 + r.uy * s, r.Z0 + r.uz * s))
    ends = np.array(ends) if ends else np.zeros((1, 3))
    if res.labor == "Avance":
        n, ax = 1, (0, 2)
        name = "Plano paralelo al frente a la profundidad y"
        lo, hi = 0.0, float(ends[:, 1].max())
        default = 0.5 * hi
    elif res.labor in ("Chimenea", "Pique"):
        n, ax = 2, (0, 1)
        name = "Plano horizontal a la cota z"
        lo, hi = float(ends[:, 2].min()), float(ends[:, 2].max())
        default = 0.5 * (lo + hi)
    elif res.view2d.get("kind") == "plan":
        n, ax = 2, (0, 1)
        name = "Plano horizontal a la cota z"
        lo, hi = float(ends[:, 2].min()), float(ends[:, 2].max())
        default = 0.5 * (lo + hi)
    else:  # vistas frontales (anillos, breasting, banqueo frontal)
        n, ax = 0, (1, 2)
        name = "Plano vertical (sección) en x"
        lo, hi = float(ends[:, 0].min()), float(ends[:, 0].max())
        if res.p.get("view_filter"):
            first = h[h.zona == res.p["view_filter"]]
            Bf = (res.frag or {}).get("B", 0.0)
            default = float(first.X0.mean()) + 0.5 * Bf if len(first) else 0.5 * (lo + hi)
            name = "Plano del burden (Onederra) en x"
        else:
            default = 0.5 * (lo + hi)
    if hi - lo < 1e-6:
        lo, hi = lo - 0.5, hi + 0.5
    return dict(normal=n, axes=ax, name=name, lo=lo, hi=hi, default=default,
                labels=(f"{AX[ax[0]]} (m)", f"{AX[ax[1]]} (m)"), nlabel=AX[n])


def plane_grid(res, frame, pos, nx=140, margin=1.0):
    """Rejilla en el plano: devuelve (u, v, puntos 3D)."""
    i, j = frame["axes"]
    out = res.view2d.get("outline")
    pts2 = []
    if out and frame["axes"] == _view_axes(res):
        pts2 = np.array(out)
    ends = []
    for _, r in res.holes.iterrows():
        for s in (0.0, r.L):
            p = np.array([r.X0 + r.ux * s, r.Y0 + r.uy * s, r.Z0 + r.uz * s])
            ends.append((p[i], p[j]))
    ends = np.array(ends)
    allp = np.vstack([ends, pts2]) if len(pts2) else ends
    u0, v0 = allp.min(axis=0) - margin
    u1, v1 = allp.max(axis=0) + margin
    asp = (v1 - v0) / max(u1 - u0, 1e-6)
    ny = int(np.clip(nx * asp, 40, 220))
    uu, vv = np.linspace(u0, u1, nx), np.linspace(v0, v1, ny)
    U, V = np.meshgrid(uu, vv)
    P = np.zeros((U.size, 3))
    P[:, i], P[:, j], P[:, frame["normal"]] = U.ravel(), V.ravel(), pos
    return uu, vv, P


def _view_axes(res):
    if res.labor == "Avance":
        return (0, 2)
    if res.labor in ("Chimenea", "Pique") or res.view2d.get("kind") == "plan":
        return (0, 1)
    return (1, 2)


def volume_grid(seg, n=30, margin=1.0):
    ends = np.vstack([seg["a"], seg["a"] + seg["u"] * seg["L"][:, None]])
    lo, hi = ends.min(axis=0) - margin, ends.max(axis=0) + margin
    axes = [np.linspace(lo[k], hi[k], n) for k in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    return X, Y, Z, np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def hole_energy_table(res, emap: dict, e_default: float, rho_r: float) -> pd.DataFrame:
    h = res.holes
    rows = []
    for _, r in h.iterrows():
        e = emap.get(r.explosivo, e_default)
        Lc = sum(s1 - s0 for s0, s1, _ in r.cargas)
        Emj = r.kg * e
        B, S = r.B, r.S
        vol = B * S * r.L if (B == B and S == S and B > 0 and S > 0) else np.nan
        rows.append(dict(id=int(r.id), zona=r.zona, kg=r.kg, e_MJkg=e, E_MJ=Emj,
                         E_lineal_MJm=Emj / Lc if Lc > 0 else 0.0, L_carga=Lc,
                         MJ_t=Emj / (vol * rho_r) if vol == vol else np.nan,
                         kg_t=r.kg / (vol * rho_r) if vol == vol else np.nan))
    return pd.DataFrame(rows)
