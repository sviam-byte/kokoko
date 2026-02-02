from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
    **_ignored: object,
) -> go.Figure:
    """Render an animated 3D energy flow figure.

    Extra keyword arguments are accepted via ``_ignored`` for compatibility with
    callers that pass through plotting options.
    """
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

    # Plotly иногда спотыкается об numpy-типы при JSON-сериализации
    # (особенно внутри frames). Поэтому приводим всё к простым python
    # спискам заранее.
    coords = np.array([pos3d.get(n, (0.0, 0.0, 0.0)) for n in nodes], dtype=float)
    xs = coords[:, 0].astype(float).tolist()
    ys = coords[:, 1].astype(float).tolist()
    zs = coords[:, 2].astype(float).tolist()

    base_node_sizes = np.full(len(nodes), 6.0, dtype=float)

    def _node_traces(frame_idx: int) -> List[go.Scatter3d]:
        """Build separate traces for hot/cold nodes to control opacity."""
        fr = node_frames[frame_idx]
        energies = np.array([float(fr.get(n, 0.0)) for n in nodes], dtype=float)
        q = float(np.quantile(energies, float(hotspot_q))) if energies.size else 0.0
        is_hot = energies >= q
        c = energies / float(Emax)
        cold_idx = np.where(~is_hot)[0]
        hot_idx = np.where(is_hot)[0]
        traces: List[go.Scatter3d] = []
        if cold_idx.size:
            traces.append(
                go.Scatter3d(
                    x=[xs[i] for i in cold_idx.tolist()],
                    y=[ys[i] for i in cold_idx.tolist()],
                    z=[zs[i] for i in cold_idx.tolist()],
                    mode="markers",
                    marker=dict(
                        size=base_node_sizes[cold_idx].astype(float).tolist(),
                        color=c[cold_idx].astype(float).tolist(),
                        colorscale="Viridis",
                        opacity=float(base_node_opacity),
                    ),
                    text=[str(nodes[i]) for i in cold_idx],
                    hoverinfo="text",
                    name="nodes",
                )
            )
        if hot_idx.size:
            traces.append(
                go.Scatter3d(
                    x=[xs[i] for i in hot_idx.tolist()],
                    y=[ys[i] for i in hot_idx.tolist()],
                    z=[zs[i] for i in hot_idx.tolist()],
                    mode="markers",
                    marker=dict(
                        size=(base_node_sizes[hot_idx] * float(hotspot_size_mult)).astype(float).tolist(),
                        color=c[hot_idx].astype(float).tolist(),
                        colorscale="Viridis",
                        opacity=1.0,
                    ),
                    text=[str(nodes[i]) for i in hot_idx],
                    hoverinfo="text",
                    name="nodes_hot",
                )
            )
        return traces

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
            buckets[b].append((float(x0), float(y0), float(z0), float(x1), float(y1), float(z1)))

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

    data0 = [*_edges_traces(0), *_node_traces(0)]

    frames = []
    for t in range(steps + 1):
        fr_traces = [*_edges_traces(t), *_node_traces(t)]
        frames.append(go.Frame(data=fr_traces, name=str(t)))

    fig = go.Figure(data=data0, frames=frames)
    # Скорость анимации (мс/кадр) можно передать через kwargs (например, anim_duration=...).
    anim_duration = int(_ignored.get("anim_duration", 80) or 80)

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
                                frame=dict(duration=anim_duration, redraw=True),
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
    *,
    show_scale: bool = False,
    edge_overlay: str = "weight",
    flow_mode: str = "rw",
    show_nodes: bool = True,
    show_labels: bool = False,
    node_size: int = 6,
    node_opacity: float = 0.85,
    edge_opacity: float = 0.55,
    edge_width_min: float = 1.0,
    edge_width_max: float = 6.0,
    edge_quantiles: int = 7,
) -> tuple[list[go.Scatter3d], go.Scatter3d | None]:
    """Build edge traces + a node trace for a 3D graph visualization.

    The function returns edge traces separately so callers can adjust node styling
    (size/labels) without rebuilding the edges. Set ``show_scale`` to include a
    colorbar for the selected ``edge_overlay`` metric.
    """
    nodes = list(G.nodes())
    if not nodes:
        return [], None

    xs = [pos3d.get(n, (0.0, 0.0, 0.0))[0] for n in nodes]
    ys = [pos3d.get(n, (0.0, 0.0, 0.0))[1] for n in nodes]
    zs = [pos3d.get(n, (0.0, 0.0, 0.0))[2] for n in nodes]

    # Color nodes by (unweighted) degree to keep scale stable across datasets.
    cvals = np.array([G.degree(n) for n in nodes], dtype=float)

    edge_traces: list[go.Scatter3d] = []

    edges = []
    vals = []
    edge_overlay = str(edge_overlay).lower()
    edge_flux: Dict[Tuple, float] | None = None
    if edge_overlay == "flux":
        # Precompute energy flow once to avoid per-edge work.
        _, edge_flux = compute_energy_flow(G, steps=20, flow_mode=str(flow_mode), damping=1.0)
    for u, v, d in G.edges(data=True):
        if u not in pos3d or v not in pos3d:
            continue
        edges.append((u, v))
        if edge_overlay == "confidence":
            vals.append(float(d.get("confidence", 0.0)))
        elif edge_overlay == "ricci":
            vals.append(float(ollivier_ricci_edge(G, u, v)))
        elif edge_overlay == "flux" and edge_flux is not None:
            vals.append(float(edge_flux.get((u, v), edge_flux.get((v, u), 0.0))))
        elif edge_overlay == "none":
            vals.append(0.0)
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
        edge_traces.append(
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

    if show_scale:
        vmin = float(bins.min())
        vmax = float(bins.max())
        edge_traces.append(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(
                    size=0.1,
                    color=[vmin, vmax],
                    colorscale="Plasma",
                    cmin=vmin,
                    cmax=vmax,
                    showscale=True,
                    colorbar=dict(title=edge_overlay),
                ),
                hoverinfo="none",
                name="edge_scale",
                showlegend=False,
            )
        )

    node_trace: go.Scatter3d | None = None
    if show_nodes:
        node_trace = go.Scatter3d(
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

    return edge_traces, node_trace
