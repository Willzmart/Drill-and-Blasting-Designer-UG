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
    ax.set_title("Sección transversal: labor de perforación, pivote (×) y carrusel (rectángulo); rojo = fuera de alcance",
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
def build_pdf(res, info: dict, params: list, seq=None, energy=None) -> bytes:
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
