from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.colors

from ...core.physics import compute_energy_flow, simulate_energy_flow
from ...core_math import ollivier_ricci_edge


def make_energy_flow_figure_3d(
    G: nx.Graph,
    pos3d: dict,
    *,
    steps: int = 25,
    node_frames: Optional[List[Dict]] = None,
    edge_frames: Optional[List[Dict[Tuple, float]]] = None,
    flow_mode: str = "phys",
    damping: float = 1.0,
    sources: Optional[List] = None,
    phys_injection: float = 0.15,
    phys_leak: float = 0.02,
    phys_cap_mode: str = "strength",
    edge_bins: int = 7,
    hotspot_q: float = 0.92,
    hotspot_size_mult: float = 4.0,
    base_node_opacity: float = 0.25,
    rw_impulse: bool = True,
) -> go.Figure:
    if node_frames is None or edge_frames is None:
        node_frames, edge_frames = simulate_energy_flow(
            G,
            steps=steps,
            flow_mode=flow_mode,
            damping=damping,
            sources=sources,
            phys_injection=phys_injection,
            phys_leak=phys_leak,
            phys_cap_mode=phys_cap_mode,
            rw_impulse=rw_impulse,
        )

    nodes = list(G.nodes())
    if not nodes:
        return go.Figure()

    steps = min(int(steps), len(node_frames) - 1)

    Emax = 0.0
    for fr in node_frames[: steps + 1]:
        if fr:
            Emax = max(Emax, max(fr.values()))
    if Emax <= 0:
        Emax = 1.0

    all_edge_vals = []
    for fr in edge_frames[: steps + 1]:
        if fr:
            all_edge_vals.extend(list(fr.values()))
    if not all_edge_vals:
        all_edge_vals = [0.0]
    all_edge_vals = np.asarray(all_edge_vals, dtype=float)
    all_edge_vals = all_edge_vals[np.isfinite(all_edge_vals)]
    if all_edge_vals.size == 0:
        all_edge_vals = np.asarray([0.0], dtype=float)

    bin_edges = np.quantile(all_edge_vals, np.linspace(0.0, 1.0, int(edge_bins) + 1))
    bin_edges = np.unique(bin_edges)
    if bin_edges.size < 2:
        bin_edges = np.array([0.0, float(np.max(all_edge_vals) + 1e-9)])

    colors = plotly.colors.sample_colorscale(
        "Plasma",
        np.linspace(0.2, 1.0, max(2, bin_edges.size - 1)),
    )

    coords = np.array([pos3d.get(n, (0, 0, 0)) for n in nodes], dtype=float)
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

    base_node_sizes = np.full(len(nodes), 6.0, dtype=float)
    
    def _node_trace(frame_idx: int) -> go.Scatter3d:
        fr = node_frames[frame_idx]
        energies = np.array([float(fr.get(n, 0.0)) for n in nodes], dtype=float)
        q = float(np.quantile(energies, float(hotspot_q))) if energies.size else 0.0
        is_hot = energies >= q
        sizes = base_node_sizes.copy()
        sizes[is_hot] *= float(hotspot_size_mult)
        opac = np.full(len(nodes), float(base_node_opacity), dtype=float)
        opac[is_hot] = 1.0
        c = energies / float(Emax)
        return go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            marker=dict(size=sizes, color=c, colorscale="Viridis", opacity=opac),
            text=[str(n) for n in nodes],
            hoverinfo="text",
            name="nodes",
        )

    def _edges_traces(frame_idx: int) -> List[go.Scatter3d]:
        fr = edge_frames[frame_idx]
        buckets: List[List[Tuple[float, float, float, float, float, float]]] = [
            [] for _ in range(max(1, bin_edges.size - 1))
        ]
        for (u, v), val in fr.items():
            if u not in pos3d or v not in pos3d:
                continue
            x0, y0, z0 = pos3d[u]
            x1, y1, z1 = pos3d[v]
            b = int(np.searchsorted(bin_edges, float(val), side="right") - 1)
            b = max(0, min(b, len(buckets) - 1))
            buckets[b].append((x0, y0, z0, x1, y1, z1))

        traces = []
        for i, segs in enumerate(buckets):
            if not segs:
                continue
            ex = []
            ey = []
            ez = []
            for x0, y0, z0, x1, y1, z1 in segs:
                ex.extend([x0, x1, None])
                ey.extend([y0, y1, None])
                ez.extend([z0, z1, None])
            traces.append(
                go.Scatter3d(
                    x=ex,
                    y=ey,
                    z=ez,
                    mode="lines",
                    line=dict(color=colors[i], width=3),
                    hoverinfo="none",
                    name=f"bin_{i}",
                )
            )
        return traces

    data0 = [*_edges_traces(0), _node_trace(0)]

    frames = []
    for t in range(steps + 1):
        fr_traces = [*_edges_traces(t), _node_trace(t)]
        frames.append(go.Frame(data=fr_traces, name=str(t)))

    fig = go.Figure(data=data0, frames=frames)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="▶",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=80, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="⏸",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(method="animate", args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))], label=str(k))
                    for k in range(steps + 1)
                ],
                active=0,
            )
        ],
    )
    return fig


def make_3d_traces(
    G: nx.Graph,
    pos3d: Dict,
    edge_mode: str = "weight",
    edge_quantiles: int = 7,
    show_nodes: bool = True,
    show_labels: bool = False,
    node_size: int = 6,
    node_opacity: float = 0.85,
    edge_opacity: float = 0.55,
    edge_width_min: float = 1.0,
    edge_width_max: float = 6.0,
    node_color_mode: str = "degree",
    overlay_mode: str = "none",
    overlay_kappa_q: float = 0.15,
    overlay_energy_steps: int = 20,
    overlay_flow_mode: str = "rw",
    overlay_damping: float = 1.0,
) -> List[go.BaseTraceType]:
    nodes = list(G.nodes())
    if not nodes:
        return []

    xs = [pos3d.get(n, (0.0, 0.0, 0.0))[0] for n in nodes]
    ys = [pos3d.get(n, (0.0, 0.0, 0.0))[1] for n in nodes]
    zs = [pos3d.get(n, (0.0, 0.0, 0.0))[2] for n in nodes]

    if node_color_mode == "strength":
        cvals = np.array([G.degree(n, weight="weight") for n in nodes], dtype=float)
    else:
        cvals = np.array([G.degree(n) for n in nodes], dtype=float)

    traces: List[go.BaseTraceType] = []

    edges = []
    vals = []
    for u, v, d in G.edges(data=True):
        if u not in pos3d or v not in pos3d:
            continue
        edges.append((u, v))
        if edge_mode == "confidence":
            vals.append(float(d.get("confidence", 0.0)))
        elif edge_mode == "ricci":
            vals.append(float(ollivier_ricci_edge(G, u, v)))
        elif edge_mode == "energy":
            _, edge_flux = compute_energy_flow(G, steps=int(overlay_energy_steps), flow_mode=str(overlay_flow_mode), damping=float(overlay_damping))
            vals.append(float(edge_flux.get((u, v), edge_flux.get((v, u), 0.0))))
        else:
            vals.append(float(d.get("weight", 0.0)))

    if vals:
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            bins = np.array([0.0, 1.0])
        else:
            bins = np.quantile(v, np.linspace(0.0, 1.0, int(edge_quantiles) + 1))
            bins = np.unique(bins)
            if bins.size < 2:
                bins = np.array([float(v.min()), float(v.max() + 1e-9)])
    else:
        bins = np.array([0.0, 1.0])

    colors = plotly.colors.sample_colorscale("Plasma", np.linspace(0.2, 1.0, max(2, bins.size - 1)))

    buckets: List[List[int]] = [[] for _ in range(max(1, bins.size - 1))]
    for i, val in enumerate(vals):
        b = int(np.searchsorted(bins, float(val), side="right") - 1)
        b = max(0, min(b, len(buckets) - 1))
        buckets[b].append(i)

    for bi, idxs in enumerate(buckets):
        if not idxs:
            continue
        ex = []
        ey = []
        ez = []
        for i in idxs:
            u, v = edges[i]
            x0, y0, z0 = pos3d[u]
            x1, y1, z1 = pos3d[v]
            ex.extend([x0, x1, None])
            ey.extend([y0, y1, None])
            ez.extend([z0, z1, None])
        width = float(edge_width_min + (edge_width_max - edge_width_min) * (bi / max(1, len(buckets) - 1)))
        traces.append(
            go.Scatter3d(
                x=ex,
                y=ey,
                z=ez,
                mode="lines",
                line=dict(color=colors[bi], width=width),
                opacity=float(edge_opacity),
                hoverinfo="none",
                name=f"edges_{bi}",
            )
        )

    if overlay_mode and overlay_mode != "none":
        traces.extend(_build_edge_overlay_traces(G, pos3d, overlay_mode, overlay_kappa_q, overlay_energy_steps, overlay_flow_mode, overlay_damping))

    if show_nodes:
        traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers+text" if show_labels else "markers",
                marker=dict(
                    size=int(node_size),
                    color=cvals,
                    colorscale="Viridis",
                    opacity=float(node_opacity),
                ),
                text=[str(n) for n in nodes] if show_labels else None,
                hoverinfo="text",
                name="nodes",
            )
        )

    return traces


def _build_edge_overlay_traces(
    G: nx.Graph,
    pos3d: Dict,
    overlay_mode: str,
    overlay_kappa_q: float,
    overlay_energy_steps: int,
    overlay_flow_mode: str,
    overlay_damping: float,
) -> List[go.BaseTraceType]:
    if overlay_mode not in ("hot", "cold", "energy"):
        return []

    overlay_edges: Set[Tuple] = set()

    if overlay_mode == "energy":
        _, edge_flux = compute_energy_flow(
            G,
            steps=int(overlay_energy_steps),
            flow_mode=str(overlay_flow_mode),
            damping=float(overlay_damping),
        )
        vals = np.array(list(edge_flux.values()), dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return []
        thr = float(np.quantile(vals, 0.85))
        for (u, v), f in edge_flux.items():
            if float(f) >= thr:
                overlay_edges.add((u, v))
    else:
        kappas = []
        e_k = []
        for u, v in G.edges():
            k = float(ollivier_ricci_edge(G, u, v))
            if np.isfinite(k):
                kappas.append(k)
                e_k.append((u, v, k))
        if not kappas:
            return []
        kappas = np.array(kappas, dtype=float)
        q = float(overlay_kappa_q)
        if overlay_mode == "hot":
            thr = float(np.quantile(kappas, 1.0 - q))
            for u, v, k in e_k:
                if k >= thr:
                    overlay_edges.add((u, v))
        else:
            thr = float(np.quantile(kappas, q))
            for u, v, k in e_k:
                if k <= thr:
                    overlay_edges.add((u, v))

    if not overlay_edges:
        return []

    ex = []
    ey = []
    ez = []
    for u, v in overlay_edges:
        if u not in pos3d or v not in pos3d:
            continue
        x0, y0, z0 = pos3d[u]
        x1, y1, z1 = pos3d[v]
        ex.extend([x0, x1, None])
        ey.extend([y0, y1, None])
        ez.extend([z0, z1, None])

    return [
        go.Scatter3d(
            x=ex,
            y=ey,
            z=ez,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.9)", width=8),
            hoverinfo="none",
            name="overlay",
        )
    ]
