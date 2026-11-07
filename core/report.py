"""Exportación del diseño a PDF: plano acotado de la malla, columnas de carga, mapa de energía y tabla de
taladros (ubicación, orientación, longitud, taco, carga, explosivo, retardo).

Usa matplotlib para los dibujos y reportlab para la maquetación (A4 apaisado). Las fuentes DejaVu se toman
de la instalación de matplotlib, de modo que el PDF admite tildes y símbolos (Ø, ², ³, ≤).
"""
from __future__ import annotations

import io
import math
import os
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Polygon as MplPolygon  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4, landscape  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, LongTable, PageBreak, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

_FONT = "Helvetica"
_FONT_B = "Helvetica-Bold"


def _fonts():
    global _FONT, _FONT_B
    if _FONT == "DejaVu":
        return
    base = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    try:
        pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(base, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(base, "DejaVuSans-Bold.ttf")))
        _FONT, _FONT_B = "DejaVu", "DejaVu-Bold"
    except Exception:  # noqa: BLE001
        pass


def _hex(c):
    return c if isinstance(c, str) and c.startswith("#") else "#555555"


# ----------------------------------------------------------------------------
# Dibujos
# ----------------------------------------------------------------------------
def _dim(ax, p0, p1, off, text, horizontal=True, color="#1f3b73", fs=8):
    """Cota con flechas entre p0 y p1, desplazada 'off' perpendicularmente."""
    (x0, y0), (x1, y1) = p0, p1
    if horizontal:
        yy = y0 + off
        ax.plot([x0, x0], [y0, yy], color=color, lw=0.6)
        ax.plot([x1, x1], [y1, yy], color=color, lw=0.6)
        ax.annotate("", xy=(x0, yy), xytext=(x1, yy), arrowprops=dict(arrowstyle="<->", color=color, lw=0.8))
        ax.text((x0 + x1) / 2, yy, text, ha="center", va="bottom" if off > 0 else "top", fontsize=fs, color=color,
                bbox=dict(fc="white", ec="none", pad=0.6))
    else:
        xx = x0 + off
        ax.plot([x0, xx], [y0, y0], color=color, lw=0.6)
        ax.plot([x1, xx], [y1, y1], color=color, lw=0.6)
        ax.annotate("", xy=(xx, y0), xytext=(xx, y1), arrowprops=dict(arrowstyle="<->", color=color, lw=0.8))
        ax.text(xx, (y0 + y1) / 2, text, ha="right" if off < 0 else "left", va="center", fontsize=fs, color=color,
                rotation=90, bbox=dict(fc="white", ec="none", pad=0.6))


def design_drawing(res, title, max_label=400):
    """Plano 2D acotado de la malla. Devuelve PNG (bytes) y el factor de escala aproximado."""
    v = res.view2d
    h = res.holes
    if res.p.get("view_filter"):
        h = h[h.zona == res.p["view_filter"]]
    fig, ax = plt.subplots(figsize=(10.8, 6.6), dpi=220)
    out = np.asarray(v["outline"]) if v.get("outline") else None
    pts = [h[["x", "y"]].to_numpy()]
    if out is not None:
        ax.add_patch(MplPolygon(out, closed=True, fc="#eeeeee", ec="#222222", lw=1.6, zorder=1))
        pts.append(out)
    for xs, ys in v.get("lines", []):
        ax.plot(xs, ys, color="#888888", lw=0.6, zorder=2)
    d = v.get("drill")
    if d:
        a = np.asarray(d["drift"])
        ax.add_patch(MplPolygon(a, closed=True, fc="#dbe9f6", ec="#1f77b4", lw=1.2, zorder=3))
        pts.append(a)
        for oco in d.get("others", []):
            a2 = np.asarray(oco)
            ax.add_patch(MplPolygon(a2, closed=True, fc="#dff2df", ec="#2ca02c", lw=1.2, zorder=3))
            pts.append(a2)
        for fx, fy in d.get("feed", []):
            ax.plot(fx, fy, ls=":", lw=0.6, color="#1f77b4", zorder=3)
        for cco in d.get("carousels", []):
            c = np.asarray(cco)
            ax.plot(c[:, 0], c[:, 1], ls="--", lw=1.0, color="#c2187a", zorder=4)
        for (pyy, pzz) in d.get("pivots", []):
            ax.plot([pyy], [pzz], marker="x", ms=8, mew=1.6, color="#c2187a", zorder=7)
            ax.annotate(f"Pivote h = {pzz:.2f} m", (pyy, pzz), textcoords="offset points", xytext=(6, 6), fontsize=7,
                        color="#c2187a")
    for name, coords in v.get("openings", []):
        a = np.asarray(coords)
        ax.plot(a[:, 0], a[:, 1], ls=":", lw=0.8, color=_hex(res.zone_colors.get(name, "#999999")), zorder=2)
    rel = res.relief
    if rel is not None and len(rel) and not res.p.get("view_filter"):
        for _, r in rel.iterrows():
            ax.add_patch(Circle((r.x, r.y), max(r.d_mm / 2000, 0.02), fc="white", ec="#111111", lw=1.0, zorder=4))
        pts.append(rel[["x", "y"]].to_numpy())
    allp = np.vstack(pts)
    span = float(max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1]), 0.5))
    rmark = max(0.012 * span, 0.025)
    handles = {}
    for z in res.zone_order:
        g = h[h.zona == z]
        if g.empty:
            continue
        col = _hex(res.zone_colors.get(z, "#555555"))
        for _, r in g.iterrows():
            ax.add_patch(Circle((r.x, r.y), rmark, fc=col, ec="#111111", lw=0.5, zorder=5))
        handles[z] = ax.scatter([], [], s=30, c=col, edgecolors="#111111", linewidths=0.5, label=f"{z} ({len(g)})")
    if len(h) <= max_label:
        fs = 6.5 if len(h) < 80 else 5
        for _, r in h.iterrows():
            ax.text(r.x + rmark * 0.9, r.y + rmark * 0.9, str(int(r.id)), fontsize=fs, zorder=6, color="#111111")
    # cotas generales
    for xs, ys in v.get("lines", []):
        allp = np.vstack([allp, np.c_[xs, ys]])
    span = float(max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1]), 0.5))
    X0, Y0 = allp.min(axis=0)
    X1, Y1 = allp.max(axis=0)
    x0, y0, x1, y1 = X0, Y0, X1, Y1
    if out is not None:
        x0, y0 = out.min(axis=0)
        x1, y1 = out.max(axis=0)
    off = min(0.07 * span, 0.18 * max(x1 - x0, y1 - y0))
    _dim(ax, (x0, y0), (x1, y0), -off, f"{x1 - x0:.2f} m")
    _dim(ax, (x0, y0), (x0, y1), -off, f"{y1 - y0:.2f} m", horizontal=False)
    # cotas del cuele (avance) o de la primera fila
    cc = res.p.get("cut_center")
    if cc is not None and res.labor == "Avance":
        cx, cy = cc
        ax.plot([cx], [cy], marker="+", color="#c0392b", ms=10, mew=1.2, zorder=7)
        _dim(ax, (x1, y0), (x1, cy), off * 0.6, f"{cy - y0:.2f} m (centro del cuele)", horizontal=False, color="#c0392b")
        if abs(cx - (x0 + x1) / 2) > 1e-3:
            _dim(ax, ((x0 + x1) / 2, y1), (cx, y1), off * 0.6, f"{cx - (x0 + x1) / 2:+.2f} m", color="#c0392b")
    # origen de coordenadas
    ax.plot([0], [0], marker="o", ms=5, mfc="white", mec="#1f3b73", zorder=8)
    ax.annotate("(0, 0)", (0, 0), textcoords="offset points", xytext=(4, -10), fontsize=7, color="#1f3b73")
    ax.set_aspect("equal")
    ax.set_xlim(min(x0, X0) - 2 * off, max(x1, X1) + 2.2 * off)
    ax.set_ylim(min(y0, Y0) - 2 * off, max(y1, Y1) + 1.5 * off)
    step = _nice_step(span)
    ax.set_xticks(np.arange(math.floor(min(x0, X0) / step) * step, max(x1, X1) + step, step))
    ax.set_yticks(np.arange(math.floor(min(y0, Y0) / step) * step, max(y1, Y1) + step, step))
    ax.grid(True, lw=0.3, color="#cccccc")
    ax.tick_params(labelsize=7)
    ax.set_xlabel(v.get("xlabel", "x (m)"), fontsize=8)
    ax.set_ylabel(v.get("ylabel", "y (m)"), fontsize=8)
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(handles=list(handles.values()), fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, title="Zonas", title_fontsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    # escala: ancho útil del dibujo en el papel ≈ 200 mm
    scale = (max(x1, X1) - min(x0, X0) + 4.2 * off) * 1000 / 200
    return buf.getvalue(), _round_scale(scale)


def _nice_step(span):
    for s in (0.25, 0.5, 1, 2, 5, 10, 20):
        if span / s <= 14:
            return s
    return 50


def _round_scale(s):
    for v in (5, 10, 20, 25, 50, 75, 100, 125, 150, 200, 250, 500, 1000):
        if s <= v:
            return v
    return int(round(s, -2))


def _plan_holes(res):
    """Taladros visibles en el plano del dibujo (sin la chimenea slot si es producción con slot)."""
    h = res.holes
    if res.p.get("view_filter"):
        h = h[h.zona == res.p["view_filter"]]
    return h


def _base_axes(res, h, figsize=(10.8, 6.6), dpi=220):
    v = res.view2d
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    out = np.asarray(v["outline"]) if v.get("outline") else None
    pts = [h[["x", "y"]].to_numpy()]
    if out is not None:
        ax.add_patch(MplPolygon(out, closed=True, fc="#f6f6f6", ec="#222222", lw=1.6, zorder=1))
        pts.append(out)
    for xs, ys in v.get("lines", []):
        ax.plot(xs, ys, color="#9a9a9a", lw=0.6, zorder=2)
    d = v.get("drill")
    if d:
        a = np.asarray(d["drift"])
        ax.add_patch(MplPolygon(a, closed=True, fc="#dbe9f6", ec="#1f77b4", lw=1.0, zorder=2))
        pts.append(a)
        for (pyy, pzz) in d.get("pivots", []):
            ax.plot([pyy], [pzz], marker="x", ms=7, mew=1.4, color="#c2187a", zorder=7)
    rel = res.relief
    if rel is not None and len(rel) and not res.p.get("view_filter"):
        for _, r in rel.iterrows():
            ax.add_patch(Circle((r.x, r.y), max(r.d_mm / 2000, 0.05), fc="#e8531f", ec="#8a2a08", lw=0.6, zorder=5))
        pts.append(rel[["x", "y"]].to_numpy())
    A = np.vstack(pts)
    span = float(max(np.ptp(A[:, 0]), np.ptp(A[:, 1]), 0.5))
    return fig, ax, out, A, span


def _finish(fig, ax, res, A, span, title, pad=0.10):
    v = res.view2d
    x0, y0 = A.min(axis=0)
    x1, y1 = A.max(axis=0)
    ax.set_aspect("equal")
    ax.set_xlim(x0 - pad * span, x1 + pad * span)
    ax.set_ylim(y0 - pad * span, y1 + pad * 0.8 * span)
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.25, color="#dddddd")
    ax.set_xlabel(v.get("xlabel", "x (m)"), fontsize=8)
    ax.set_ylabel(v.get("ylabel", "y (m)"), fontsize=8)
    ax.set_title(title, fontsize=10.5, loc="center", fontweight="bold")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def timing_drawing(res, seq, system=""):
    """Malla con el tiempo de iniciación de cada taladro (distribución del accesorio) y flechas de la secuencia:
    cada flecha va del taladro (o vacío) que le abre la cara libre al taladro que sale después."""
    if seq is None or "t_ms" not in seq:
        return None
    h = _plan_holes(res).merge(seq[["id", "t_ms", "periodo", "predecesor"] +
                                   (["n_detonador"] if "n_detonador" in seq else [])], on="id", how="left")
    if h.t_ms.isna().all():
        return None
    fig, ax, out, A, span = _base_axes(res, h)
    rmark = max(0.011 * span, 0.03)
    pos = h.set_index("id")[["x", "y"]]
    src = np.asarray(res.sources) if res.sources else np.empty((0, 2))
    tmax = float(np.nanmax(h.t_ms)) or 1.0
    cmap = plt.get_cmap("viridis")
    chain = res.labor == "Producción" and bool(res.p.get("view_filter"))
    if chain:      # anillo: flechas en el orden de salida de los taladros
        q = h.dropna(subset=["t_ms"]).sort_values("t_ms")[["x", "y", "t_ms"]].to_numpy(float)
        for (ax_, ay, _), (bx, by, tb) in zip(q[:-1], q[1:]):
            ax.annotate("", xy=(bx, by), xytext=(ax_, ay), zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=cmap(0.15 + 0.7 * tb / tmax), lw=0.9, shrinkA=4,
                                        shrinkB=5, mutation_scale=8, alpha=0.85))
    for _, r in h.iterrows():
        if chain or r.t_ms != r.t_ms:
            continue
        if r.predecesor and int(r.predecesor) in pos.index:
            px, py = pos.loc[int(r.predecesor)]
        elif len(src):
            k = int(np.argmin(np.hypot(src[:, 0] - r.x, src[:, 1] - r.y)))
            px, py = src[k]
        else:
            continue
        if math.hypot(r.x - px, r.y - py) < 1e-6:
            continue
        ax.annotate("", xy=(r.x, r.y), xytext=(px, py), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=cmap(0.15 + 0.7 * r.t_ms / tmax), lw=0.9,
                                    shrinkA=4, shrinkB=5, mutation_scale=8, alpha=0.85))
    for _, r in h.iterrows():
        ax.add_patch(Circle((r.x, r.y), rmark, fc="#1531c9", ec="#0b1b6e", lw=0.5, zorder=6))
        if r.t_ms == r.t_ms:
            ax.text(r.x, r.y + rmark * 1.25, f"{r.t_ms:.0f}", fontsize=6.8 if len(h) < 90 else 5.2,
                    color="#c2187a", ha="center", va="bottom", zorder=7, fontweight="bold",
                    bbox=dict(fc="white", ec="none", pad=0.15, alpha=0.75))
    ax.scatter([], [], s=28, c="#1531c9", label="Taladro cargado (tiempo en ms, magenta)")
    if res.relief is not None and len(res.relief) and not res.p.get("view_filter"):
        ax.scatter([], [], s=28, c="#e8531f", label="Taladro de alivio (vacío)")
    ax.plot([], [], color=cmap(0.5), lw=1, label="Secuencia de salida (flecha → taladro siguiente)")
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    nper = int(np.nanmax(h.periodo)) if "periodo" in h else 0
    ax.text(1.01, 0.02, f"Sistema: {system}\nRetardos: {nper}\nÚltima salida: {tmax:.0f} ms", transform=ax.transAxes,
            fontsize=7, va="bottom")
    return _finish(fig, ax, res, A, span, f"DISTRIBUCIÓN DEL ACCESORIO DE INICIACIÓN ({system or 'retardos'})")


def _chain(ax, vals, fixed, off, horizontal, color="#111111", fs=7):
    """Cadena de cotas entre valores consecutivos (sobre una línea fija)."""
    vals = sorted(vals)
    for a, b in zip(vals[:-1], vals[1:]):
        if b - a < 0.05:
            continue
        if horizontal:
            _dim(ax, (a, fixed), (b, fixed), off, f"{b - a:.2f}", True, color, fs)
        else:
            _dim(ax, (fixed, a), (fixed, b), off, f"{b - a:.2f}", False, color, fs)


def _cluster(vals, tol):
    out = []
    for v in sorted(vals):
        if not out or v - out[-1][-1] > tol:
            out.append([v])
        else:
            out[-1].append(v)
    return [float(np.mean(c)) for c in out]


RED = "#d62728"


def _ring_edges(P, center, closed_gap=2.2):
    """Ordena un anillo de taladros por ángulo alrededor del centro y une los consecutivos; si el anillo no
    rodea al centro queda abierto (se corta en el mayor vacío angular)."""
    n = len(P)
    if n < 2:
        return []
    ang = np.arctan2(P[:, 1] - center[1], P[:, 0] - center[0])
    order = np.argsort(ang)
    a = ang[order]
    gaps = np.diff(np.r_[a, a[0] + 2 * np.pi])
    k = int(np.argmax(gaps))
    closed = gaps[k] < np.radians(100) and n >= 3
    if not closed:
        order = np.r_[order[k + 1:], order[:k + 1]]
    E = [(order[t], order[t + 1]) for t in range(n - 1)]
    if closed:
        E.append((order[-1], order[0]))
    return E


def _grid_edges(P, cone=30.0):
    """Malla en planta: cada taladro se une a su vecino más cercano hacia +x y hacia +y."""
    E = set()
    for i in range(len(P)):
        for ux, uy in ((1, 0), (0, 1)):
            d = P - P[i]
            L = np.hypot(d[:, 0], d[:, 1])
            cosang = (d[:, 0] * ux + d[:, 1] * uy) / np.where(L > 0, L, np.inf)
            ok = (L > 1e-6) & (cosang >= math.cos(math.radians(cone)))
            if ok.any():
                j = int(np.argmin(np.where(ok, L, np.inf)))
                E.add(tuple(sorted((i, j))))
    return sorted(E)


def _edge_label(ax, p, q, center, span, fs=6.3, color="#111111", lw=1.0, line_color=RED, ls="-", z=4):
    (ax_, ay), (bx, by) = p, q
    d = math.hypot(bx - ax_, by - ay)
    if d < 0.03:
        return
    ax.plot([ax_, bx], [ay, by], color=line_color, lw=lw, ls=ls, zorder=z)
    mx, my = (ax_ + bx) / 2, (ay + by) / 2
    nx_, ny_ = -(by - ay) / d, (bx - ax_) / d                      # normal al tramo
    if (mx - center[0]) * nx_ + (my - center[1]) * ny_ < 0:      # rótulo hacia afuera del centro
        nx_, ny_ = -nx_, -ny_
    off = 0.012 * span
    rot = math.degrees(math.atan2(by - ay, bx - ax_))
    rot = rot - 180 if rot > 90 else (rot + 180 if rot <= -90 else rot)
    ax.text(mx + nx_ * off, my + ny_ * off, f"{d:.2f}", fontsize=fs, fontweight="bold", color=color, rotation=rot,
            rotation_mode="anchor", ha="center", va="center", zorder=8,
            bbox=dict(fc="white", ec="none", pad=0.05, alpha=0.6))


def _mesh_edges(res, h, center, plan):
    """Tramos a acotar: anillos por zona (frentes) o filas y columnas (planta), más un tramo representativo
    entre cada anillo y el anillo siguiente hacia afuera (burden entre anillos)."""
    P = h[["x", "y"]].to_numpy(float)
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    np.fill_diagonal(D, np.inf)
    nn_med = float(np.median(D.min(axis=1)))
    if plan:     # filas de la voladura (x constante) con su espaciamiento; filas en y solo si no es tresbolillo
        tol = 0.25 * float(D.min(axis=1).min())
        E = []
        xc = _cluster(P[:, 0], tol)
        for c in xc:
            ii = np.where(np.abs(P[:, 0] - c) <= tol + 1e-9)[0]
            ii = ii[np.argsort(P[ii, 1])]
            E += [(a, b) for a, b in zip(ii[:-1], ii[1:]) if D[a, b] <= 3.0 * nn_med]
        dxm = float(np.median(np.diff(xc))) if len(xc) > 1 else np.inf
        for c in _cluster(P[:, 1], tol):
            ii = np.where(np.abs(P[:, 1] - c) <= tol + 1e-9)[0]
            ii = ii[np.argsort(P[ii, 0])]
            ez = [(a, b) for a, b in zip(ii[:-1], ii[1:])]
            if ez and np.median([D[a, b] for a, b in ez]) <= 1.3 * dxm:
                E += ez
        return P, E, xc
    zones = list(dict.fromkeys(h.zona))
    idx = {z: np.where(h.zona.values == z)[0] for z in zones}
    E = []
    for z in zones:
        ii = idx[z]
        ez = [(ii[a], ii[b]) for a, b in _ring_edges(P[ii], center)]
        if not ez:
            continue
        Lz = np.array([D[a, b] for a, b in ez])
        lim = max(2.3 * float(np.median(Lz)), 1.2 * nn_med)
        E += [(a, b) for (a, b), L in zip(ez, Lz) if L <= min(lim, 4.0 * nn_med)]
    # tramo entre anillos: zona -> la siguiente hacia afuera (radio medio mayor)
    rad = {z: float(np.mean(np.hypot(P[idx[z], 0] - center[0], P[idx[z], 1] - center[1]))) for z in zones}
    zs = sorted(zones, key=lambda z: rad[z])
    C = []
    for k, z in enumerate(zs[:-1]):
        outer = [w for w in zs[k + 1:] if rad[w] > rad[z] + 1e-6][:1]
        for w in outer:
            sub = D[np.ix_(idx[z], idx[w])]
            a, b = np.unravel_index(int(np.argmin(sub)), sub.shape)
            i, j = idx[z][a], idx[w][b]
            if (i, j) not in E and (j, i) not in E and D[i, j] <= 4.0 * nn_med:
                C.append((i, j))
    return P, E, C


def dims_drawing(res):
    """Estándar de malla de perforación con las distancias entre taladros escritas sobre los tramos que los
    unen (anillos del cuele, ayudas y contorno; filas y columnas en planta), como en los estándares de mina."""
    h = _plan_holes(res)
    slot_alone = res.metodo.startswith("Chimenea slot")
    if res.labor == "Producción" and "grupo" in h and (h.grupo >= 0).any() and not slot_alone:
        h = h[h.grupo >= 0]
    if len(h) < 2:
        return None
    h = h.reset_index(drop=True)
    ring_view = bool(res.p.get("view_filter"))
    plan = res.labor == "Producción" and not slot_alone and not ring_view
    crop = None
    vo = np.asarray(res.view2d["outline"]) if res.view2d.get("outline") else None
    if plan and vo is not None:        # tajeos muy alargados: se acota un tramo típico (la malla se repite)
        ex_, ey_ = np.ptp(vo[:, 0]), np.ptp(vo[:, 1])
        if ey_ > 0 and ex_ / ey_ > 3.2:
            crop = float(vo[:, 0].min() + max(3.2 * ey_, 3.5 * float(h.B.median() if "B" in h else 1.0)))
            h = h[h.x <= crop].reset_index(drop=True)
    fig, ax, out, A, span = _base_axes(res, h)
    if crop is not None:
        A = A[A[:, 0] <= crop + 0.05 * span]
        span = float(max(np.ptp(A[:, 0]), np.ptp(A[:, 1]), 0.5))
        A = np.vstack([A, [[crop + 0.32 * span, float(A[:, 1].mean())]]])
    cc = res.p.get("cut_center")
    rel = res.relief
    if ring_view and res.view2d.get("drill"):
        center = tuple(res.view2d["drill"]["pivots"][0])
    elif cc is not None:
        center = tuple(cc)
    elif rel is not None and len(rel) and not plan:
        center = (float(rel.x.mean()), float(rel.y.mean()))
    elif out is not None:
        center = (float(out[:, 0].mean()), float(out[:, 1].mean()))
    else:
        center = (float(h.x.mean()), float(h.y.mean()))
    P, E, C = _mesh_edges(res, h, center, plan)
    fs = 6.4 if len(h) < 70 else (5.4 if len(h) < 140 else 4.6)
    # ejes por el centro del cuele (frentes con alivios) o de la sección
    if not plan and not ring_view:
        x0, y0 = A.min(axis=0)
        x1, y1 = A.max(axis=0)
        if rel is not None and len(rel):
            axc = (float(rel.x.mean()), float(rel.y.mean()))
        elif out is not None:
            axc = ((out[:, 0].min() + out[:, 0].max()) / 2, (out[:, 1].min() + out[:, 1].max()) / 2)
        else:
            axc = center
        ax.plot([axc[0], axc[0]], [y0, y1], color="#1f77b4", lw=0.8, zorder=2)
        ax.plot([x0, x1], [axc[1], axc[1]], color="#222222", lw=0.8, zorder=2)
    for i, j in E:
        _edge_label(ax, P[i], P[j], center, span, fs)
    if plan:      # cadena de burden entre filas, debajo de la malla
        yb = float(P[:, 1].min()) - 0.05 * span
        for xa, xb in zip(C[:-1], C[1:]):
            _edge_label(ax, (xa, yb), (xb, yb), (0.0, 1e9), span, fs, line_color=RED, lw=0.7, ls=(0, (3, 2)))
            for xx in (xa, xb):
                ax.plot([xx, xx], [yb - 0.008 * span, yb + 0.008 * span], color=RED, lw=0.7, zorder=4)
        if len(C) > 1:
            ax.text(C[0], yb - 0.03 * span, "B (burden entre filas)", fontsize=6, color=RED, va="top")
        C = []
    for i, j in C:
        _edge_label(ax, P[i], P[j], center, span, fs, line_color=RED, lw=0.7, ls=(0, (3, 2)))
    # taladros (negros) y alivios (círculos abiertos), como el estándar de mina
    rmark = max(0.009 * span, 0.028)
    for x, y in P:
        ax.add_patch(Circle((x, y), rmark, fc="#111111", ec="#111111", lw=0.4, zorder=6))
    if rel is not None and len(rel) and not ring_view:
        for _, r in rel.iterrows():
            ax.add_patch(Circle((r.x, r.y), max(r.d_mm / 2000, rmark), fc="white", ec="#111111", lw=0.9, zorder=6))
    # cotas generales de la sección
    if out is not None:
        xo0, yo0 = out.min(axis=0)
        xo1, yo1 = out.max(axis=0)
        off = 0.06 * span
        if crop is not None:
            ax.text(crop, (yo0 + yo1) / 2, f"  … la malla se repite\n  hasta {xo1:.1f} m", fontsize=7, va="center",
                    color="#1f3b73", zorder=8, bbox=dict(fc="white", ec="none", pad=0.3))
            xo1 = crop
        _dim(ax, (xo0, yo0), (xo1, yo0), -off, f"{xo1 - xo0:.2f} m" + (" (tramo típico)" if crop else ""), True,
             "#1f3b73", 7.5)
        _dim(ax, (xo0, yo0), (xo0, yo1), -off, f"{yo1 - yo0:.2f} m", False, "#1f3b73", 7.5)
        A = np.vstack([A, [[xo0 - 1.6 * off, yo0 - 1.6 * off]]])
    # detalle ampliado del cuele cuando es pequeño frente a la sección
    cutz = [z for z in res.zone_order if z.startswith(("Arranque", "Cuadrante", "Slot: arranque", "Slot: cuadrante"))]
    inner = [z for z in cutz if z in ("Arranque", "Cuadrante 1", "Cuadrante 2", "Slot: arranque", "Slot: cuadrante 1")]
    g = h[h.zona.isin(inner)]
    if len(g) >= 3 and not plan:
        cext = float(max(np.ptp(g.x), np.ptp(g.y)))
        if 0 < cext < 0.3 * span:
            ia = ax.inset_axes([1.03, 0.04, 0.36, 0.50])
            gi = g.reset_index(drop=True)
            Pg, Eg, Cg = _mesh_edges(res, gi, center, False)
            sp_i = max(cext, 0.3)
            for i, j in Eg:
                _edge_label(ia, Pg[i], Pg[j], center, sp_i, 5.8)
            for i, j in Cg:
                _edge_label(ia, Pg[i], Pg[j], center, sp_i, 5.8, lw=0.7, ls=(0, (3, 2)))
            m_ = 0.2 * sp_i
            ia.plot([center[0]] * 2, [gi.y.min() - m_, gi.y.max() + m_], color="#1f77b4", lw=0.7)
            ia.plot([gi.x.min() - m_, gi.x.max() + m_], [center[1]] * 2, color="#222222", lw=0.7)
            if rel is not None and len(rel):
                for _, r in rel.iterrows():
                    ia.add_patch(Circle((r.x, r.y), max(r.d_mm / 2000, 0.012 * sp_i), fc="white", ec="#111111",
                                        lw=0.8, zorder=6))
                    ia.text(r.x, r.y - 0.06 * sp_i, f"Ø{r.d_mm:.0f}", fontsize=4.8, ha="center", va="top")
            for x, y in Pg:
                ia.add_patch(Circle((x, y), 0.018 * sp_i, fc="#111111", ec="#111111", zorder=6))
            ia.set_xlim(gi.x.min() - m_, gi.x.max() + m_)
            ia.set_ylim(gi.y.min() - m_, gi.y.max() + m_)
            ia.set_aspect("equal")
            ia.tick_params(labelsize=5)
            ia.set_title("Detalle del cuele (m)", fontsize=7)
            ax.add_patch(plt.Rectangle((gi.x.min() - m_, gi.y.min() - m_), np.ptp(gi.x) + 2 * m_,
                                       np.ptp(gi.y) + 2 * m_, fill=False, ls="--", lw=0.6, ec="#555555", zorder=3))
    ax.scatter([], [], s=24, c="#111111", label=f"Taladros cargados ({len(h)})")
    if rel is not None and len(rel) and not ring_view:
        ax.scatter([], [], s=30, facecolors="white", edgecolors="#111111", label=f"Taladros de alivio ({len(rel)})")
    ax.plot([], [], color=RED, lw=1.0, label="Distancia entre taladros (m)")
    ax.plot([], [], color=RED, lw=0.7, ls=(0, (3, 2)), label="Burden entre anillos / filas (m)")
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, title="Leyenda",
              title_fontsize=7)
    return _finish(fig, ax, res, A, span, f"ESTÁNDAR DE MALLA DE PERFORACIÓN — {res.metodo}".upper()[:95], pad=0.06)


def section_drawing(res):
    """Sección transversal de taladros paralelos con labores de perforación, pivotes y carrusel."""
    sv = res.p.get("section_view")
    if not sv:
        return None
    fig, ax = plt.subplots(figsize=(10.8, 6.4), dpi=200)
    st = np.asarray(sv["stope"])
    ax.add_patch(MplPolygon(st, closed=True, fc="#eeeeee", ec="#222222", lw=1.4))
    allp = [st]
    for dco in sv["drifts"]:
        a = np.asarray(dco)
        ax.add_patch(MplPolygon(a, closed=True, fc="#dbe9f6", ec="#1f77b4", lw=1.1))
        allp.append(a)
    for oco in sv.get("others", []):
        a = np.asarray(oco)
        ax.add_patch(MplPolygon(a, closed=True, fc="#dff2df", ec="#2ca02c", lw=1.1))
        ax.text(a[:, 0].mean(), a[:, 1].min() + 0.25 * np.ptp(a[:, 1]), "Galería inferior", fontsize=6.5,
                ha="center", va="center", color="#2ca02c")
        allp.append(a)
    for hx, hy, ok, hid in sv["holes"]:
        ax.plot(hx, hy, color="#555555" if ok else "#d62728", lw=0.8, ls="-" if ok else "--")
        ax.text(hx[0], hy[0] + 0.15, str(hid), fontsize=6, ha="center")
    for cx, cy in sv["charges"]:
        ax.plot(cx, cy, color="#ff7f0e", lw=2.5)
    for cco in sv["carousels"]:
        c = np.asarray(cco)
        ax.plot(c[:, 0], c[:, 1], ls="--", lw=1.0, color="#c2187a")
    for pyy, pzz, ok in sv["pivots"]:
        ax.plot([pyy], [pzz], marker="x", ms=5, mew=1.1, color="#c2187a" if ok else "#d62728")
    A = np.vstack(allp)
    x0, z0 = A.min(axis=0)
    x1, z1 = A.max(axis=0)
    span = max(x1 - x0, z1 - z0)
    _dim(ax, tuple(st[3]), tuple(st[2]), -0.05 * span, f"{st[2][0] - st[3][0]:.2f} m")
    _dim(ax, (st[:, 0].min(), 0), (st[:, 0].min(), sv["H"]), -0.06 * span, f"{sv['H']:.2f} m", horizontal=False)
    ax.set_aspect("equal")
    ax.set_xlim(x0 - 0.12 * span, x1 + 0.05 * span)
    ax.set_ylim(z0 - 0.1 * span, z1 + 0.05 * span)
    ax.grid(True, lw=0.3, color="#cccccc")
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Ancho / potencia y (m)", fontsize=8)
    ax.set_ylabel("Altura z (m)", fontsize=8)
    ax.set_title("Sección transversal: labor de perforación (azul), galería inferior (verde),\n"
                 "pivote (×) y carrusel; rojo = fuera de alcance",
                 fontsize=9, loc="left")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def charge_columns(res):
    """Columna tipo por zona: taco, carga y retacados intermedios (taladro representativo)."""
    zones = [z for z in res.zone_order if z in set(res.holes.zona)]
    fig, ax = plt.subplots(figsize=(10.8, 0.42 * len(zones) + 1.0), dpi=200)
    Lmax = float(res.holes.L.max())
    for k, z in enumerate(zones):
        g = res.holes[res.holes.zona == z]
        r = g.iloc[(g.L - g.L.median()).abs().argsort().iloc[0]]
        y = len(zones) - k
        ax.add_patch(plt.Rectangle((0, y - 0.3), r.L, 0.6, fc="#f4f4f4", ec="#444444", lw=0.6))
        for s0, s1 in (r.tacos or []):
            ax.add_patch(plt.Rectangle((s0, y - 0.3), s1 - s0, 0.6, fc="#b09b7c", ec="none"))
        for s0, s1, qq in r.cargas:
            ax.add_patch(plt.Rectangle((s0, y - 0.3), s1 - s0, 0.6, fc=_hex(res.zone_colors.get(z, "#777777")),
                                       ec="none", alpha=0.85))
        Lc = sum(b - a for a, b, _ in r.cargas)
        ax.text(-0.01 * Lmax, y, z, ha="right", va="center", fontsize=7)
        ax.text(r.L + 0.01 * Lmax, y, f"L = {r.L:.2f} m · taco = {r.Lt:.2f} m · carga = {Lc:.2f} m · {r.kg:.2f} kg",
                va="center", fontsize=6.5)
    ax.set_xlim(-0.35 * Lmax, 1.55 * Lmax)
    ax.set_ylim(0.3, len(zones) + 0.7)
    ax.set_yticks([])
    ax.set_xlabel("Distancia desde la boca del taladro (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Columna de carga típica por zona (marrón: taco / retacado)", fontsize=9, loc="left")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def energy_drawing(uu, vv, F, title, unit, thresholds=(), labels=("u (m)", "v (m)"), outline=None, marks=None,
                   vmax=None):
    fig, ax = plt.subplots(figsize=(10.8, 6.0), dpi=200)
    Z = F.reshape(len(vv), len(uu))
    vmax = vmax or float(np.nanpercentile(Z, 97))
    cs = ax.contourf(uu, vv, np.clip(Z, 0, vmax), levels=14, cmap="YlOrRd")
    fig.colorbar(cs, ax=ax, label=unit, shrink=0.85)
    for val, lab, col in thresholds:
        try:
            c = ax.contour(uu, vv, Z, levels=[val], colors=[col], linewidths=1.5)
            ax.clabel(c, fmt={val: f"{lab}"}, fontsize=6)
        except Exception:  # noqa: BLE001
            pass
    if outline is not None:
        a = np.asarray(outline)
        ax.plot(a[:, 0], a[:, 1], color="#222222", lw=1.4)
    if marks is not None and len(marks):
        m = np.asarray(marks)
        ax.plot(m[:, 0], m[:, 1], "o", ms=3, mfc="white", mec="#111111", mew=0.6)
    ax.set_aspect("equal")
    ax.set_xlabel(labels[0], fontsize=8)
    ax.set_ylabel(labels[1], fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9, loc="left")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Tabla de taladros
# ----------------------------------------------------------------------------
def hole_table_rows(res, seq=None):
    h = res.holes.copy()
    if seq is not None:
        cols = [c for c in ("id", "periodo", "n_detonador", "t_ms", "mecha_m") if c in seq.columns]
        h = h.merge(seq[cols], on="id", how="left")
    rows = []
    for _, r in h.iterrows():
        az = (math.degrees(math.atan2(r.ux, r.uy)) + 360) % 360
        inc = math.degrees(math.asin(max(-1.0, min(1.0, r.uz))))
        Lc = sum(b - a for a, b, _ in r.cargas)
        stem = sum(b - a for a, b in (r.tacos or []))
        row = [int(r.id), r.zona, f"{r.x:.2f}", f"{r.y:.2f}", f"{r.X0:.2f}", f"{r.Y0:.2f}", f"{r.Z0:.2f}",
               f"{az:.0f}", f"{inc:+.0f}", f"{r.L:.2f}", f"{stem:.2f}", f"{Lc:.2f}", str(r.explosivo),
               str(int(r.n_cart)) if r.n_cart else "—", f"{r.kg:.2f}",
               f"{r.B:.2f}" if r.B == r.B else "—", f"{r.S:.2f}" if r.S == r.S else "—"]
        if seq is not None:
            per = r.get("periodo", np.nan)
            row += [str(int(r.n_detonador)) if r.get("n_detonador", np.nan) == r.get("n_detonador", np.nan) else "—",
                    f"{r.t_ms:.0f}" if r.get("t_ms", np.nan) == r.get("t_ms", np.nan) else "—"]
            if "mecha_m" in h.columns:
                row += [f"{r.mecha_m:.2f}" if r.mecha_m == r.mecha_m else "—"]
            _ = per
        rows.append(row)
    head = ["N°", "Zona", "x", "y", "X₀", "Y₀", "Z₀", "Az. (°)", "Incl. (°)", "L (m)", "Taco (m)", "Carga (m)",
            "Explosivo", "Cart.", "kg", "B (m)", "S (m)"]
    if seq is not None:
        head += ["Det. N°", "t (ms)"] + (["Mecha (m)"] if "mecha_m" in h.columns else [])
    return head, rows


# ----------------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------------
def build_pdf(res, info: dict, params: list, seq=None, energy=None, system="") -> bytes:
    """info: proyecto, mina, labor, responsable, revisor, observaciones. params: [(parámetro, valor)].
    energy: dict(uu, vv, F, title, unit, thresholds, labels, outline, marks) o None."""
    _fonts()
    buf = io.BytesIO()
    W, H = landscape(A4)
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=16 * mm, bottomMargin=13 * mm, title=info.get("proyecto", "Diseño de voladura"),
                            author=info.get("responsable", ""))
    ss = getSampleStyleSheet()
    st_n = ParagraphStyle("n", parent=ss["Normal"], fontName=_FONT, fontSize=8, leading=10, alignment=TA_LEFT)
    st_h = ParagraphStyle("h", parent=ss["Heading2"], fontName=_FONT_B, fontSize=11.5, leading=14, spaceAfter=4)
    st_s = ParagraphStyle("s", parent=st_n, fontSize=7, leading=8.5, textColor=colors.HexColor("#444444"))
    titulo = f"{info.get('labor', res.labor)} — {res.metodo}"

    def on_page(canvas, d):
        canvas.saveState()
        canvas.setFont(_FONT_B, 9)
        canvas.drawString(12 * mm, H - 9 * mm, info.get("proyecto") or "Diseño de perforación y voladura")
        canvas.setFont(_FONT, 7.5)
        canvas.drawRightString(W - 12 * mm, H - 9 * mm, f"{info.get('mina', '')}  ·  {titulo[:110]}")
        canvas.setStrokeColor(colors.HexColor("#1f3b73"))
        canvas.setLineWidth(0.6)
        canvas.line(12 * mm, H - 11 * mm, W - 12 * mm, H - 11 * mm)
        canvas.setFont(_FONT, 7)
        canvas.drawString(12 * mm, 7 * mm, f"Generado el {date.today():%d/%m/%Y} · Fuentes: Carpio (2025), López Jimeno "
                                           f"(1987), Konya, material del curso de taladros largos")
        canvas.drawRightString(W - 12 * mm, 7 * mm, f"Página {d.page}")
        canvas.restoreState()

    story = []
    png, scale = design_drawing(res, f"Malla de perforación — {res.metodo}")
    story.append(Image(io.BytesIO(png), width=W - 24 * mm, height=(W - 24 * mm) * 6.6 / 10.8 * 0.93))
    tb = [["Proyecto", info.get("proyecto", ""), "Labor", titulo[:70], "Escala aprox.", f"1:{scale}"],
          ["Mina / unidad", info.get("mina", ""), "Elaborado por", info.get("responsable", ""), "Fecha",
           f"{date.today():%d/%m/%Y}"],
          ["Revisado por", info.get("revisor", ""), "Cotas y coordenadas", "metros; origen (0, 0) marcado", "Lámina", "1"]]
    t = Table(tb, colWidths=[24 * mm, 58 * mm, 34 * mm, 86 * mm, 22 * mm, 34 * mm])
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), _FONT, 7), ("FONT", (0, 0), (0, -1), _FONT_B, 7),
                           ("FONT", (2, 0), (2, -1), _FONT_B, 7), ("FONT", (4, 0), (4, -1), _FONT_B, 7),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#1f3b73")),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f8")),
                           ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#eef2f8")),
                           ("BACKGROUND", (4, 0), (4, -1), colors.HexColor("#eef2f8")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(t)
    story.append(PageBreak())
    hgt = (W - 24 * mm) * 6.6 / 10.8 * 0.97
    dpng = dims_drawing(res)
    if dpng:
        story.append(Image(io.BytesIO(dpng), width=W - 24 * mm, height=hgt))
        story.append(PageBreak())
    tpng = timing_drawing(res, seq, system)
    if tpng:
        story.append(Image(io.BytesIO(tpng), width=W - 24 * mm, height=hgt))
        story.append(PageBreak())
    spng = section_drawing(res)
    if spng:
        story.append(Image(io.BytesIO(spng), width=W - 24 * mm, height=(W - 24 * mm) * 6.4 / 10.8 * 0.95))
        story.append(PageBreak())

    # parámetros y resultados
    story.append(Paragraph("Parámetros de diseño y resultados", st_h))
    S = res.summary
    kp = [[k, f"{v:,.3f}" if isinstance(v, (int, float, np.floating)) and v == v else str(v)] for k, v in S.items()]
    pp = [[a, str(b)] for a, b in params]
    n = max(len(pp), len(kp))
    pp += [["", ""]] * (n - len(pp))
    kp += [["", ""]] * (n - len(kp))
    st_c = ParagraphStyle("c", parent=st_n, fontSize=7, leading=8.4)
    data = [["Parámetro de entrada", "Valor", "Indicador", "Valor"]] + \
           [[Paragraph(str(c), st_c) for c in (a + b)] for a, b in zip(pp, kp)]
    t = Table(data, colWidths=[52 * mm, 96 * mm, 72 * mm, 32 * mm], repeatRows=1)
    t.setStyle(_grid_style())
    story.append(t)
    zt = res.holes.groupby("zona", sort=False).agg(N=("id", "count"), L=("L", "mean"), kg=("kg", "sum"),
                                                   cart=("n_cart", "sum"))
    zt = zt.reindex([z for z in res.zone_order if z in zt.index])
    data = [["Zona", "N° taladros", "L media (m)", "Cartuchos", "Explosivo (kg)"]] + \
           [[z, int(r.N), f"{r.L:.2f}", int(r.cart), f"{r.kg:.2f}"] for z, r in zt.iterrows()] + \
           [["Total", int(zt.N.sum()), "", int(zt.cart.sum()), f"{zt.kg.sum():.2f}"]]
    t = Table(data, colWidths=[60 * mm, 28 * mm, 28 * mm, 28 * mm, 32 * mm], repeatRows=1)
    t.setStyle(_grid_style(total=True))
    story += [Spacer(1, 5 * mm), Paragraph("Distribución por zona", st_h), t]
    if res.warnings:
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph("Advertencias: " + " · ".join(dict.fromkeys(res.warnings)), st_s))
    if info.get("observaciones"):
        story.append(Paragraph("Observaciones: " + info["observaciones"], st_n))
    story.append(PageBreak())

    # columnas de carga y energía
    cpng = charge_columns(res)
    nz = len([z for z in res.zone_order if z in set(res.holes.zona)])
    story.append(Image(io.BytesIO(cpng), width=W - 24 * mm, height=(W - 24 * mm) * (0.42 * nz + 1.0) / 10.8))
    if energy is not None:
        epng = energy_drawing(energy["uu"], energy["vv"], energy["F"], energy["title"], energy["unit"],
                              energy.get("thresholds", ()), energy.get("labels", ("u", "v")),
                              energy.get("outline"), energy.get("marks"), energy.get("vmax"))
        story.append(KeepTogether([Spacer(1, 4 * mm),
                                   Image(io.BytesIO(epng), width=(W - 24 * mm) * 0.82,
                                         height=(W - 24 * mm) * 0.82 * 6.0 / 10.8)]))
    story.append(PageBreak())

    # tabla de taladros
    story.append(Paragraph("Cuadro de taladros: ubicación, orientación y carga", st_h))
    story.append(Paragraph("x, y: coordenadas en el plano del dibujo. X₀, Y₀, Z₀: boca del taladro en 3D. Az.: azimut "
                           "medido desde el eje Y hacia X; Incl.: inclinación respecto a la horizontal (+ arriba). "
                           "Taco: suma de retacados; Carga: longitud cargada.", st_s))
    head, rows = hole_table_rows(res, seq)
    if len(rows) > 1500:
        story.append(Paragraph(f"La malla tiene {len(rows)} taladros: se listan los primeros 1500 (el CSV de la pestaña "
                               f"Resumen contiene todos).", st_s))
        rows = rows[:1500]
    data = [head] + rows
    base_w = [9, 30, 12, 12, 12, 12, 12, 12, 13, 12, 13, 13, 50, 11, 13, 12, 12]
    extra = [13, 13, 15][:len(head) - len(base_w)]
    widths = base_w + extra
    k = (W - 24 * mm) / (sum(widths) * mm)
    t = LongTable(data, colWidths=[w * mm * k for w in widths], repeatRows=1)
    t.setStyle(_grid_style(fs=6.3))
    story.append(t)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return buf.getvalue()


def _grid_style(fs=7.2, total=False):
    cmds = [("FONT", (0, 0), (-1, -1), _FONT, fs), ("FONT", (0, 0), (-1, 0), _FONT_B, fs),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b73")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9aa7bd")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6fa")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]
    if total:
        cmds += [("FONT", (0, -1), (-1, -1), _FONT_B, fs), ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e2e8f2"))]
    return TableStyle(cmds)
