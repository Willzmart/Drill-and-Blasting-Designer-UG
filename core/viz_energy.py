"""Figuras de la distribución de energía (2D en un plano y 3D)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .viz import _box, _equal_scene, _ring, stope_traces

SCALE_KG = [[0, "#f7fbff"], [0.15, "#c6dbef"], [0.35, "#6baed6"], [0.55, "#fdd49e"], [0.75, "#fc8d59"],
            [0.9, "#d7301f"], [1, "#7f0000"]]


def energy_map_2d(res, uu, vv, F, frame, pos, title, unit, thresholds=(), vmax=None, log=False, height=640):
    i, j = frame["axes"]
    n = frame["normal"]
    Z = F.reshape(len(vv), len(uu))
    if vmax is None:
        vmax = float(np.nanpercentile(Z, 97)) if np.isfinite(Z).any() else 1.0
    Zp = np.clip(Z, 0, vmax)
    if log:
        Zp = np.log10(np.maximum(Zp, vmax * 1e-3))
    fig = go.Figure(go.Heatmap(x=uu, y=vv, z=Zp, colorscale=SCALE_KG, zsmooth="best",
                               colorbar=dict(title=unit + (" (log₁₀)" if log else ""), len=0.85),
                               hovertemplate=f"{frame['labels'][0]}=%{{x:.2f}}  {frame['labels'][1]}=%{{y:.2f}}"
                                             f"<br>%{{customdata:.3g}} {unit}<extra></extra>",
                               customdata=np.clip(Z, 0, None)))
    for val, lab, col in thresholds:
        lv = np.log10(val) if log else val
        fig.add_trace(go.Contour(x=uu, y=vv, z=Zp, contours=dict(start=lv, end=lv, size=1, coloring="none",
                                                                  showlabels=False),
                                 line=dict(color=col, width=2.5), showscale=False, name=f"{lab} = {val:.3g} {unit}",
                                 hoverinfo="skip", showlegend=True))
    v = res.view2d
    view_axes = (0, 2) if res.labor == "Avance" else ((0, 1) if (res.labor in ("Chimenea", "Pique") or
                                                                v.get("kind") == "plan") else (1, 2))
    if v.get("outline") and view_axes == (i, j):
        a = np.asarray(v["outline"])
        fig.add_trace(go.Scatter(x=a[:, 0], y=a[:, 1], mode="lines", line=dict(color="#222", width=2.5),
                                 name="Contorno", hoverinfo="skip"))
    lx, ly, px, py, txt = [], [], [], [], []
    for _, r in res.holes.iterrows():
        c = np.array([r.X0, r.Y0, r.Z0])
        u = np.array([r.ux, r.uy, r.uz])
        for s0, s1, _ in r.cargas:
            p0, p1 = c + u * s0, c + u * s1
            lx += [p0[i], p1[i], None]
            ly += [p0[j], p1[j], None]
        if abs(u[n]) > 1e-6:
            s = (pos - c[n]) / u[n]
            if 0 <= s <= r.L:
                p = c + u * s
                px.append(p[i])
                py.append(p[j])
                txt.append(f"N° {r.id} — {r.zona}<br>{r.kg:.2f} kg")
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color="rgba(20,20,20,0.55)", width=1.6),
                             name="Cargas (proyección)", hoverinfo="skip"))
    if px:
        fig.add_trace(go.Scatter(x=px, y=py, mode="markers", name="Taladro que corta el plano",
                                 marker=dict(size=7, color="white", line=dict(color="#111", width=1.5)),
                                 text=txt, hovertemplate="%{text}<extra></extra>"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, title=frame["labels"][1], zeroline=False)
    fig.update_xaxes(title=frame["labels"][0], zeroline=False)
    fig.update_layout(height=height, plot_bgcolor="white", margin=dict(l=10, r=10, t=50, b=10),
                      title=f"{title} — {frame['name']} = {pos:.2f} m", legend=dict(orientation="h", y=-0.2))
    return fig


def energy_3d(res, seg, iso=None, show_shell=True, height=720):
    """Cargas coloreadas por energía lineal (MJ/m) y, opcionalmente, isosuperficie de la concentración."""
    fig = go.Figure()
    sc = res.scene3d
    if show_shell:
        if sc.get("kind") == "tunnel":
            fig.add_trace(_ring(sc["outline"], "xz", 0.0, "Contorno del frente"))
            fig.add_trace(_ring(sc["outline"], "xz", sc["advance"], "Avance esperado", "dash", "#1f77b4"))
            for vc in sc.get("vein", []):
                fig.add_trace(_ring(vc, "xz", 0.0, "Veta en el frente", color="#d4a017"))
        elif sc.get("kind") == "vertical":
            fig.add_trace(_ring(sc["outline"], "xy", 0.0, "Contorno del frente"))
        elif "box" in sc:
            for tr in stope_traces(sc, False):
                fig.add_trace(tr)
    el = seg["q"] * seg["e"]
    vmin, vmax = float(el.min()), float(max(el.max(), el.min() + 1e-6))
    xs, ys, zs, cs = [], [], [], []
    for k in range(len(el)):
        p0 = seg["a"][k]
        p1 = p0 + seg["u"][k] * seg["L"][k]
        xs += [p0[0], p1[0], None]
        ys += [p0[1], p1[1], None]
        zs += [p0[2], p1[2], None]
        cs += [el[k], el[k], el[k]]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name="Energía lineal de la carga",
                               line=dict(color=cs, colorscale="Turbo", cmin=vmin, cmax=vmax, width=9,
                                         colorbar=dict(title="MJ/m", x=1.0, len=0.6)),
                               hoverinfo="skip"))
    h = res.holes
    fig.add_trace(go.Scatter3d(x=h.X0, y=h.Y0, z=h.Z0, mode="markers", marker=dict(size=2.5, color="#111"),
                               name="Boca de taladro",
                               text=[f"N° {r.id} — {r.zona}<br>{r.kg:.2f} kg" for _, r in h.iterrows()],
                               hovertemplate="%{text}<extra></extra>"))
    if iso is not None:
        X, Y, Z, F, thr, unit = iso
        fig.add_trace(go.Isosurface(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=F, isomin=thr,
                                    isomax=float(np.nanpercentile(F, 99.5)), surface_count=2, opacity=0.22,
                                    colorscale=[[0, "#fdae6b"], [1, "#a50f15"]], showscale=False,
                                    caps=dict(x_show=False, y_show=False, z_show=False),
                                    name=f"Zona ≥ {thr:.3g} {unit}", showlegend=True))
    A = np.vstack([seg["a"], seg["a"] + seg["u"] * seg["L"][:, None]])
    ext = A.max(axis=0) - A.min(axis=0)
    ext = np.where(ext > 1e-6, ext, 1e-6)
    eye = dict(x=-1.25, y=-2.1, z=0.75) if sc.get("kind") == "tunnel" else dict(x=1.6, y=-1.8, z=1.0)
    pts = [A]
    for tr in fig.data:
        if isinstance(tr, (go.Scatter3d, go.Mesh3d)) and tr.x is not None and len(tr.x):
            pts.append(np.c_[[v if v is not None else np.nan for v in tr.x],
                             [v if v is not None else np.nan for v in tr.y],
                             [v if v is not None else np.nan for v in tr.z]].astype(float))
    scene = _equal_scene(pts, eye)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=40, b=0), scene=scene,
                      title="Energía de las cargas en 3D", legend=dict(orientation="h", y=-0.02))
    return fig


def energy_bars(tab, zone_colors, height=360):
    fig = go.Figure()
    for z in tab.zona.unique():
        g = tab[tab.zona == z]
        fig.add_trace(go.Bar(x=g.id, y=g.E_MJ, name=z, marker_color=zone_colors.get(z, "#777"),
                             customdata=np.c_[g.kg, g.E_lineal_MJm],
                             hovertemplate="N° %{x}<br>%{y:.1f} MJ · %{customdata[0]:.2f} kg"
                                           "<br>%{customdata[1]:.2f} MJ/m<extra></extra>"))
    fig.update_layout(height=height, barmode="stack", plot_bgcolor="white", xaxis_title="N° de taladro",
                      yaxis_title="Energía (MJ)", margin=dict(l=10, r=10, t=30, b=10), title="Energía por taladro",
                      legend=dict(orientation="h", y=-0.25))
    return fig
