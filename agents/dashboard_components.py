"""
Nexus-AI: Dashboard Components & Integration Adapters (Member 5)
================================================================

Provides visualizers, adapters, and data transformation helpers for Member 5's
Streamlit application. Connects Member 1 (Mapping), Member 2 (Risk Detection),
Member 3 (Disruption Simulation), and Member 4 (Financial Optimization).

Author: Nexus-AI Team (Member 5)
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Set the global Plotly default template to dark so ALL figures produced by
# this module — including their toolbars, hover labels, colour-pickers, and
# modebar pop-ups — are rendered on a dark background out of the box.
# Individual fig.update_layout() calls still override specific properties.
pio.templates.default = "plotly_dark"

from agents.mapping_agent import compute_hierarchical_layout
from agents.financial_optimization_agent import (
    DisruptionScenario,
    RecoveryOption,
    optimize_recovery_plan,
)
from agents.disruption_simulation_agent import get_financial_engine_payload


# Modern Enterprise Status & Role Colors for Resilience Graph
STATUS_COLORS: Dict[str, str] = {
    "failed": "#EF4444",                # High-Alert Red (Root shock failure)
    "confirmed_shortage": "#F43F5E",    # Vivid Rose / Crimson (Confirmed downstream shortage)
    "potentially_affected": "#F59E0B",  # Warning Amber / Orange (High-risk hub / potential shortage)
    "unaffected_raw": "#06B6D4",        # Cyan (Raw Material - Healthy)
    "unaffected_proc": "#38BDF8",       # Sky Blue (Processor - Healthy)
    "unaffected_supp": "#10B981",       # Emerald (Supplier - Healthy)
    "unaffected_fact": "#8B5CF6",       # Violet (Plant / Factory - Healthy)
    "unaffected_other": "#64748B",      # Slate Grey (Other Entity - Healthy)
}


def safe_format_currency(
    val: Any,
    default: str = "Not available",
    decimals: int = 2,
    prefix: str = "$",
) -> str:
    """
    Format a currency value safely. Returns `default` if `val` is None or non-numeric.
    Guards against TypeError: unsupported format string passed to NoneType.__format__.
    """
    if val is None:
        return default
    try:
        fval = float(val)
        return f"{prefix}{fval:,.{decimals}f}"
    except (ValueError, TypeError):
        return default


def safe_format_capacity(
    val: Any,
    default: str = "Not available",
    decimals: int = 1,
    unit: str = " units",
) -> str:
    """
    Format a capacity value safely. Returns `default` ('Not available' or 'Unknown')
    when capacity data is genuinely missing or None. Never fabricates zero.
    """
    if val is None:
        return default
    try:
        fval = float(val)
        return f"{fval:,.{decimals}f}{unit}"
    except (ValueError, TypeError):
        return default


def safe_format_number(
    val: Any,
    default: str = "Unknown",
    decimals: int = 1,
    suffix: str = "",
) -> str:
    """
    Format a generic numeric value safely, returning `default` if None or non-numeric.
    """
    if val is None:
        return default
    try:
        fval = float(val)
        return f"{fval:,.{decimals}f}{suffix}"
    except (ValueError, TypeError):
        return default


def safe_format_lead_time(
    val: Any,
    default: str = "Unknown",
    suffix: str = " days",
) -> str:
    """
    Format a lead time in days safely, returning `default` if None or non-numeric.
    """
    if val is None:
        return default
    try:
        fval = float(val)
        return f"{fval:.0f}{suffix}"
    except (ValueError, TypeError):
        return default


def classify_node_status(
    node: str,
    simulation_result: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """
    Determines the operational status and label for a node given simulation results.

    Returns:
        Tuple[str, str]: (status_key, human_label)
    """
    if not simulation_result:
        return "unaffected", "Operational"

    failed_entity = simulation_result.get("failed_entity", "")
    if node == failed_entity or f"'{node}'" in failed_entity:
        return "failed", "DISRUPTED (Root Failure)"

    confirmed = simulation_result.get("confirmed_shortage_entities")
    if confirmed is None:
        confirmed = [
            d["entity"]
            for d in simulation_result.get("entity_impact_details", [])
            if d.get("status") == "confirmed_shortage"
        ]

    if node in confirmed:
        return "confirmed_shortage", "Confirmed Shortage (Zero Inbound Paths)"

    potential = simulation_result.get("potential_shortage_entities")
    if potential is None:
        potential = [
            d["entity"]
            for d in simulation_result.get("entity_impact_details", [])
            if d.get("status") == "potentially_affected"
        ]

    if node in potential:
        return "potentially_affected", "Potentially Affected (Partial Path / Redundant)"

    return "unaffected", "Operational / Intact"


def create_resilient_network_figure(
    graph: nx.DiGraph,
    simulation_result: Optional[Dict[str, Any]] = None,
    selected_entity: Optional[str] = None,
    title: str = "Nexus-AI: Multi-Tier Supply Chain Resilience Map",
    animate: bool = True,
    show_replay: bool = False,
    show_propagation_signal: bool = False,
    **kwargs: Any,
) -> go.Figure:
    """
    Generates an interactive, state-aware Plotly visualization of the supply chain network.
    Features:
    - Subtle glowing halo auras and smooth pulsing animations on nodes.
    - Moving material particles flowing along operational connections in pipeline fashion.
    - Shockwave red pulsing visualization on disrupted failure roots and severed corridors.
    - Clear visual distinction for confirmed shortages and affected downstream suppliers.
    - Interactive Play/Pause/Replay telemetry controls and reduced-motion support.
    """
    # Safe fallback handling for kwargs
    if "animate" in kwargs:
        animate = bool(kwargs["animate"])
    if "show_replay" in kwargs:
        show_replay = bool(kwargs["show_replay"])
    if "show_propagation_signal" in kwargs:
        show_propagation_signal = bool(kwargs["show_propagation_signal"])

    fig = go.Figure()
    if graph.number_of_nodes() == 0:
        fig.update_layout(
            template="plotly_dark",
            title=dict(text=f"<b>{title}</b>", font=dict(color="#F8FAFC")),
            paper_bgcolor="#070B14",
            plot_bgcolor="#0A101D",
        )
        return fig

    # 1. Compute and scale layout coordinates for presentation spacing
    base_pos = compute_hierarchical_layout(graph)
    SCALE_X = 1.6
    SCALE_Y = 1.45
    pos: Dict[str, Tuple[float, float]] = {
        node: (coords[0] * SCALE_X, coords[1] * SCALE_Y)
        for node, coords in base_pos.items()
    }

    # 2. Identify broken corridors from simulation
    failed_edges: Set[Tuple[str, str]] = set()
    failed_entity = simulation_result.get("failed_entity", "") if simulation_result else ""

    if simulation_result:
        for u, v in graph.edges():
            if u == failed_entity or v == failed_entity:
                failed_edges.add((u, v))

        if " -> " in failed_entity:
            for u, v in graph.edges():
                if f"{u} -> {v}" in failed_entity:
                    failed_edges.add((u, v))

        # Include multi-hop cascade paths when in propagation telemetry mode
        if show_propagation_signal:
            paths = simulation_result.get("affected_paths") or simulation_result.get("disrupted_paths") or []
            for path in paths:
                for i in range(len(path) - 1):
                    failed_edges.add((path[i], path[i + 1]))

    # 3. Process Corridors: Line segments, directed annotations & midpoints
    annotations = []
    op_line_x: List[Optional[float]] = []
    op_line_y: List[Optional[float]] = []
    sev_line_x: List[Optional[float]] = []
    sev_line_y: List[Optional[float]] = []
    mid_x, mid_y, mid_text, mid_colors = [], [], [], []
    op_edges: List[Tuple[str, str]] = []
    sev_edges: List[Tuple[str, str]] = []

    for u, v, data in graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        material = data.get("material", "Commodity")
        capacity = data.get("capacity", "N/A")
        is_broken = (u, v) in failed_edges

        # Line paths
        if is_broken:
            sev_line_x.extend([x0, x1, None])
            sev_line_y.extend([y0, y1, None])
            sev_edges.append((u, v))
        else:
            op_line_x.extend([x0, x1, None])
            op_line_y.extend([y0, y1, None])
            op_edges.append((u, v))

        # Waypoint midpoints
        mx = (x0 + x1) / 2.0
        my = (y0 + y1) / 2.0
        mid_x.append(mx)
        mid_y.append(my)

        status_span = (
            "<span style='color:#EF4444;'><b>SEVERED ROUTE (SHOCK OUTAGE)</b></span>"
            if is_broken
            else "<span style='color:#10B981;'><b>OPERATIONAL CORRIDOR</b></span>"
        )
        cap_str = safe_format_capacity(capacity, default="Unknown", decimals=0, unit=" units")
        divider = "────────────────────────────"
        corridor_hover = (
            f"<b>Supply Corridor:</b> {u} &rarr; {v}<br>"
            f"<span style='color:#64748B;'>{divider}</span><br>"
            f"<b>Status:</b> {status_span}<br>"
            f"<b>Transported Material:</b> {material}<br>"
            f"<b>Capacity Throughput:</b> {cap_str}"
        )
        mid_text.append(corridor_hover)
        mid_colors.append("#EF4444" if is_broken else "#0284C7")

        # Directed Arrow calculation
        dx = x1 - x0
        dy = y1 - y0
        dist = math.hypot(dx, dy)
        if dist > 0:
            offset_start = min(0.35, dist * 0.20)
            offset_end = min(0.40, dist * 0.25)
            ax_pos = x0 + (dx / dist) * offset_start
            ay_pos = y0 + (dy / dist) * offset_start
            head_x = x1 - (dx / dist) * offset_end
            head_y = y1 - (dy / dist) * offset_end
        else:
            ax_pos, ay_pos, head_x, head_y = x0, y0, x1, y1

        annotations.append(
            dict(
                ax=ax_pos,
                ay=ay_pos,
                axref="x",
                ayref="y",
                x=head_x,
                y=head_y,
                xref="x",
                yref="y",
                showarrow=True,
                arrowhead=3 if is_broken else 2,
                arrowsize=1.5 if is_broken else 1.2,
                arrowwidth=3.0 if is_broken else 1.8,
                arrowcolor="#EF4444" if is_broken else "#38BDF8",
                opacity=0.95 if is_broken else 0.75,
            )
        )

    # Trace: Operational Corridors line trace
    if op_line_x:
        fig.add_trace(
            go.Scatter(
                x=op_line_x,
                y=op_line_y,
                mode="lines",
                line=dict(color="#0284C7", width=1.8),
                opacity=0.70,
                hoverinfo="none",
                name="Operational Corridor",
                showlegend=True,
            )
        )

    # Trace: Severed Corridors line trace
    if sev_line_x:
        fig.add_trace(
            go.Scatter(
                x=sev_line_x,
                y=sev_line_y,
                mode="lines",
                line=dict(color="#EF4444", width=3.0),
                opacity=0.95,
                hoverinfo="none",
                name="Severed Corridor",
                showlegend=True,
            )
        )

    # Trace: Route Telemetry Waypoint trace
    if mid_x:
        fig.add_trace(
            go.Scatter(
                x=mid_x,
                y=mid_y,
                mode="markers",
                marker=dict(
                    size=8,
                    color=mid_colors,
                    symbol="circle-dot",
                    line=dict(color="#FFFFFF", width=1.2),
                ),
                hoverinfo="text",
                hovertext=mid_text,
                name="Route Telemetry",
                showlegend=False,
            )
        )

    # Trace: Moving Material Particles along Operational Corridors
    particle_trace_idx: Optional[int] = None
    initial_px: List[float] = []
    initial_py: List[float] = []
    for u, v in op_edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        for p in (0.25, 0.75):
            initial_px.append(x0 + p * (x1 - x0))
            initial_py.append(y0 + p * (y1 - y0))

    if animate and op_edges:
        particle_trace_idx = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=initial_px,
                y=initial_py,
                mode="markers",
                marker=dict(
                    size=7,
                    color="#38BDF8",
                    symbol="circle",
                    line=dict(color="#FFFFFF", width=1.5),
                ),
                hoverinfo="none",
                name="Active Material Flow",
                showlegend=True,
            )
        )

    # Trace: Moving Disruption Shock Particles along Severed Corridors (Feature 3)
    disrupt_particle_trace_idx: Optional[int] = None
    initial_dx: List[float] = []
    initial_dy: List[float] = []
    if show_propagation_signal and animate and sev_edges:
        for u, v in sev_edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            for p in (0.30, 0.75):
                initial_dx.append(x0 + p * (x1 - x0))
                initial_dy.append(y0 + p * (y1 - y0))

        disrupt_particle_trace_idx = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=initial_dx,
                y=initial_dy,
                mode="markers",
                marker=dict(
                    size=9,
                    color="#F59E0B",
                    symbol="diamond",
                    line=dict(color="#FFFFFF", width=1.2),
                ),
                hoverinfo="none",
                name="Disruption Shock Signal",
                showlegend=True,
            )
        )

    # Trace: Origin Expanding Shockwave Ring (Feature 3)
    origin_shockwave_trace_idx: Optional[int] = None
    if show_propagation_signal and animate and failed_entity and failed_entity in pos:
        fx_orig, fy_orig = pos[failed_entity]
        origin_shockwave_trace_idx = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=[fx_orig],
                y=[fy_orig],
                mode="markers",
                marker=dict(
                    size=[34],
                    color="rgba(0,0,0,0)",
                    symbol="circle-open",
                    line=dict(color="#EF4444", width=3.0),
                ),
                hoverinfo="none",
                name="Origin Expanding Shockwave",
                showlegend=False,
            )
        )

    # Trace: Node Glowing Aura / Pulsing Halos (placed under foreground nodes)
    halo_trace_idx: Optional[int] = None
    halo_x: List[float] = []
    halo_y: List[float] = []
    halo_sizes: List[float] = []
    halo_colors: List[str] = []
    halo_line_colors: List[str] = []
    halo_line_widths: List[float] = []
    halo_node_keys: List[str] = []

    for node in graph.nodes():
        status, _ = classify_node_status(node, simulation_result)
        node_type = str(graph.nodes[node].get("node_type", "")).lower()
        hx, hy = pos[node]
        halo_x.append(hx)
        halo_y.append(hy)
        halo_node_keys.append(status)

        if status == "failed":
            halo_sizes.append(44.0)
            halo_colors.append("rgba(239, 68, 68, 0.40)")
            halo_line_colors.append("#EF4444")
            halo_line_widths.append(2.5)
        elif status == "confirmed_shortage":
            halo_sizes.append(36.0)
            halo_colors.append("rgba(244, 63, 94, 0.35)")
            halo_line_colors.append("#FB7185")
            halo_line_widths.append(2.0)
        elif status == "potentially_affected":
            halo_sizes.append(32.0)
            halo_colors.append("rgba(245, 158, 11, 0.30)")
            halo_line_colors.append("#FBBF24")
            halo_line_widths.append(1.8)
        else:
            halo_sizes.append(28.0)
            if "raw" in node_type:
                halo_colors.append("rgba(6, 182, 212, 0.22)")
                halo_line_colors.append("rgba(6, 182, 212, 0.6)")
            elif "proc" in node_type:
                halo_colors.append("rgba(56, 189, 248, 0.22)")
                halo_line_colors.append("rgba(56, 189, 248, 0.6)")
            else:
                halo_colors.append("rgba(16, 185, 129, 0.22)")
                halo_line_colors.append("rgba(16, 185, 129, 0.6)")
            halo_line_widths.append(1.2)

    if animate and halo_x:
        halo_trace_idx = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=halo_x,
                y=halo_y,
                mode="markers",
                marker=dict(
                    size=halo_sizes,
                    color=halo_colors,
                    symbol="circle",
                    line=dict(color=halo_line_colors, width=halo_line_widths),
                ),
                hoverinfo="none",
                name="Node Resonance Aura",
                showlegend=False,
            )
        )

    # 4. Group Nodes by State & Category for Foreground Nodes
    status_groups: Dict[str, Dict[str, Any]] = {
        "failed": {
            "nodes": [],
            "color": STATUS_COLORS["failed"],
            "symbol": "x",
            "name": "Disrupted Shock Root",
            "size": 30,
        },
        "confirmed_shortage": {
            "nodes": [],
            "color": STATUS_COLORS["confirmed_shortage"],
            "symbol": "diamond-tall",
            "name": "Confirmed Shortage",
            "size": 26,
        },
        "potentially_affected": {
            "nodes": [],
            "color": STATUS_COLORS["potentially_affected"],
            "symbol": "triangle-up",
            "name": "High-Risk / Affected",
            "size": 24,
        },
        "operational_raw": {
            "nodes": [],
            "color": STATUS_COLORS["unaffected_raw"],
            "symbol": "diamond",
            "name": "Raw Material (Operational)",
            "size": 24,
        },
        "operational_proc": {
            "nodes": [],
            "color": STATUS_COLORS["unaffected_proc"],
            "symbol": "square",
            "name": "Processor (Operational)",
            "size": 24,
        },
        "operational_supp": {
            "nodes": [],
            "color": STATUS_COLORS["unaffected_supp"],
            "symbol": "circle",
            "name": "Supplier (Operational)",
            "size": 24,
        },
        "operational_fact": {
            "nodes": [],
            "color": STATUS_COLORS["unaffected_fact"],
            "symbol": "hexagon",
            "name": "Factory / Plant (Operational)",
            "size": 26,
        },
        "operational_other": {
            "nodes": [],
            "color": STATUS_COLORS["unaffected_other"],
            "symbol": "circle",
            "name": "Other Entity (Operational)",
            "size": 22,
        },
    }

    for node, data in graph.nodes(data=True):
        status, _ = classify_node_status(node, simulation_result)
        node_type = str(data.get("node_type", "")).lower()

        if status == "failed":
            group_key = "failed"
        elif status == "confirmed_shortage":
            group_key = "confirmed_shortage"
        elif status == "potentially_affected":
            group_key = "potentially_affected"
        else:
            if "raw" in node_type or "mine" in node_type:
                group_key = "operational_raw"
            elif "proc" in node_type or "refin" in node_type:
                group_key = "operational_proc"
            elif "supp" in node_type:
                group_key = "operational_supp"
            elif "fact" in node_type or "plant" in node_type or "assembl" in node_type:
                group_key = "operational_fact"
            else:
                group_key = "operational_other"

        status_groups[group_key]["nodes"].append(node)

    # 5. Render Node Scatter Traces
    for group_key, group_data in status_groups.items():
        nodes = group_data["nodes"]
        if not nodes:
            continue

        nx_c = [pos[n][0] for n in nodes]
        ny_c = [pos[n][1] for n in nodes]
        hover_texts = []
        labels = []
        border_colors = []
        border_widths = []

        for n in nodes:
            status_key, status_label = classify_node_status(n, simulation_result)
            in_deg = graph.in_degree(n)
            out_deg = graph.out_degree(n)
            node_type = graph.nodes[n].get("node_type", "Unknown")
            alts = graph.nodes[n].get("alternatives", [])
            alts_str = ", ".join(alts) if alts else "None registered"

            is_selected = selected_entity and (selected_entity == n)
            selection_marker = " 🎯 [INSPECTED]" if is_selected else ""

            if is_selected:
                border_colors.append("#FBBF24")  # Gold target ring
                border_widths.append(3.5)
                labels.append(f"🎯 {n}")
            else:
                border_colors.append("#FFFFFF" if group_key in ("failed", "confirmed_shortage") else "#CBD5E1")
                border_widths.append(1.8)
                labels.append(n)

            status_color = group_data["color"]
            divider = "────────────────────────────"
            node_hover = (
                f"<b>{n}{selection_marker}</b><br>"
                f"<span style='color:#38BDF8;'>ROLE: {node_type.upper()}</span><br>"
                f"<span style='color:#64748B;'>{divider}</span><br>"
                f"<b>Operational Status:</b> <span style='color:{status_color};'><b>{status_label}</b></span><br>"
                f"<b>Inbound Feeds:</b> {in_deg} &nbsp;|&nbsp; <b>Outbound:</b> {out_deg}<br>"
                f"<b>Backup Availability:</b> {alts_str}"
            )
            hover_texts.append(node_hover)

        fig.add_trace(
            go.Scatter(
                x=nx_c,
                y=ny_c,
                mode="markers+text",
                marker=dict(
                    size=group_data["size"],
                    color=group_data["color"],
                    symbol=group_data["symbol"],
                    line=dict(
                        color=border_colors,
                        width=border_widths,
                    ),
                ),
                text=labels,
                textposition="bottom center",
                textfont=dict(size=11, color="#F8FAFC", family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
                hoverinfo="text",
                hovertext=hover_texts,
                name=group_data["name"],
                showlegend=True,
            )
        )

    # 6. Build Animation Frames if enabled
    if animate and (
        particle_trace_idx is not None
        or halo_trace_idx is not None
        or disrupt_particle_trace_idx is not None
        or origin_shockwave_trace_idx is not None
    ):
        NUM_FRAMES = 16
        frames = []
        anim_traces = []
        if particle_trace_idx is not None:
            anim_traces.append(particle_trace_idx)
        if halo_trace_idx is not None:
            anim_traces.append(halo_trace_idx)
        if disrupt_particle_trace_idx is not None:
            anim_traces.append(disrupt_particle_trace_idx)
        if origin_shockwave_trace_idx is not None:
            anim_traces.append(origin_shockwave_trace_idx)

        for f in range(NUM_FRAMES):
            frame_data = []

            # A. Update moving material particles along operational corridors
            if particle_trace_idx is not None:
                fx, fy = [], []
                for u, v in op_edges:
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    for p in range(2):
                        tau = ((f / NUM_FRAMES) + p * 0.5) % 1.0
                        t = 0.16 + tau * 0.68
                        fx.append(x0 + t * (x1 - x0))
                        fy.append(y0 + t * (y1 - y0))
                frame_data.append(go.Scatter(x=fx, y=fy))

            # B. Update pulsating node halos
            if halo_trace_idx is not None:
                fh_sizes = []
                pulse = math.sin(2 * math.pi * f / NUM_FRAMES)
                for key in halo_node_keys:
                    if key == "failed":
                        fh_sizes.append(44.0 + 10.0 * pulse)
                    elif key == "confirmed_shortage":
                        fh_sizes.append(36.0 + 6.0 * pulse)
                    elif key == "potentially_affected":
                        fh_sizes.append(32.0 + 4.0 * pulse)
                    else:
                        fh_sizes.append(28.0 + 2.5 * pulse)
                frame_data.append(
                    go.Scatter(
                        marker=dict(
                            size=fh_sizes,
                            color=halo_colors,
                            symbol="circle",
                            line=dict(color=halo_line_colors, width=halo_line_widths),
                        )
                    )
                )

            # C. Update disruption shock signal particles along severed corridors (Feature 3)
            if disrupt_particle_trace_idx is not None:
                dx, dy = [], []
                for u, v in sev_edges:
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    # Compute hop depth from failed origin for topological cascade
                    try:
                        hop_depth = (
                            nx.shortest_path_length(graph, failed_entity, u)
                            if (failed_entity in graph and nx.has_path(graph, failed_entity, u))
                            else 0
                        )
                    except Exception:
                        hop_depth = 0

                    for p in range(2):
                        tau = ((f / NUM_FRAMES) - (hop_depth * 0.35) + p * 0.5) % 1.0
                        t = 0.16 + tau * 0.68
                        dx.append(x0 + t * (x1 - x0))
                        dy.append(y0 + t * (y1 - y0))
                frame_data.append(go.Scatter(x=dx, y=dy))

            # D. Update origin expanding shockwave ring (Feature 3)
            if origin_shockwave_trace_idx is not None:
                tau = f / NUM_FRAMES
                shock_sz = 32.0 + 44.0 * tau
                alpha = max(0.08, 0.95 * (1.0 - tau))
                shock_col = f"rgba(239, 68, 68, {alpha:.2f})"
                shock_w = max(1.0, 3.5 * (1.0 - tau * 0.7))
                frame_data.append(
                    go.Scatter(
                        marker=dict(
                            size=[shock_sz],
                            color="rgba(0,0,0,0)",
                            symbol="circle-open",
                            line=dict(color=shock_col, width=shock_w),
                        )
                    )
                )

            frames.append(
                go.Frame(
                    data=frame_data,
                    traces=anim_traces,
                    name=f"frame_{f}",
                )
            )

        fig.frames = frames

    # 7. Animation Controls (Play/Pause/Replay Updatemenus)
    updatemenus = []
    if animate and fig.frames:
        buttons = [
            dict(
                label="▶ Play Flow",
                method="animate",
                args=[
                    None,
                    dict(
                        frame=dict(duration=80, redraw=False),
                        fromcurrent=False,
                        mode="immediate",
                        loop=True,
                        transition=dict(duration=0),
                    ),
                ],
            ),
            dict(
                label="⏸ Pause",
                method="animate",
                args=[
                    [None],
                    dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    ),
                ],
            ),
        ]

        if show_replay:
            buttons.append(
                dict(
                    label="🔁 Replay",
                    method="animate",
                    args=[
                        ["frame_0"],
                        dict(
                            frame=dict(duration=80, redraw=False),
                            fromcurrent=False,
                            mode="immediate",
                            loop=True,
                            transition=dict(duration=0),
                        ),
                    ],
                )
            )

        updatemenus = [
            dict(
                type="buttons",
                direction="right",
                x=0.98,
                y=1.06,
                xanchor="right",
                yanchor="bottom",
                pad=dict(r=6, t=4, b=4, l=6),
                bgcolor="#151F32",
                bordercolor="#334155",
                borderwidth=1,
                font=dict(color="#F8FAFC", size=11, family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
                active=-1,
                buttons=buttons,
            )
        ]

    # 8. Overall Layout Styling
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=16, color="#F8FAFC", family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
            x=0.02,
            y=0.97,
            xanchor="left",
            yanchor="top",
        ),
        updatemenus=updatemenus,
        annotations=annotations,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.16,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(17, 26, 46, 0.90)",
            bordercolor="#1E293B",
            borderwidth=1,
            font=dict(color="#CBD5E1", size=11),
            itemsizing="constant",
        ),
        hoverlabel=dict(
            bgcolor="#151F32",
            bordercolor="#334155",
            font=dict(color="#F8FAFC", size=12),
        ),
        hovermode="closest",
        margin=dict(b=70, l=40, r=40, t=75),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#0A101D",
        paper_bgcolor="#070B14",
        height=640,
        font=dict(color="#CBD5E1", family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
    )

    return fig


def render_risk_card(finding: Dict[str, Any]) -> str:
    """
    Renders an executive-grade HTML card for a single shared risk finding.
    Features severity pill, score meter bar, tier indicator, affected suppliers chips,
    business context on why it is a bottleneck, technical score calculations in expandable details,
    missing data disclosures, and governance recommendations.
    """
    node = finding.get("dependency_node", "Unknown Entity")
    score = float(finding.get("risk_score", 0.0) or 0.0)
    level = finding.get("risk_level", "Medium")
    is_shared = bool(finding.get("is_shared", False))
    tier = finding.get("dependency_tier", "Unknown Tier")
    affected_suppliers = finding.get("affected_suppliers", [])
    direct_suppliers = finding.get("direct_affected_suppliers", [])
    indirect_suppliers = finding.get("indirect_affected_suppliers", [])
    explanation = finding.get("explanation", "No explanation recorded.")
    missing_info = finding.get("missing_information", [])
    rec = finding.get("recommended_investigation", "Review vulnerability and qualify secondary suppliers.")

    if level == "Critical":
        badge_class = "badge-critical"
        border_color = "#F43F5E"
        score_color = "#FB7185"
    elif level == "High":
        badge_class = "badge-high"
        border_color = "#F59E0B"
        score_color = "#FBBF24"
    elif level == "Medium":
        badge_class = "badge-medium"
        border_color = "#EAB308"
        score_color = "#FDE047"
    else:
        badge_class = "badge-low"
        border_color = "#10B981"
        score_color = "#34D399"

    shared_badge = (
        '<span style="background:rgba(244,63,94,0.15); color:#FB7185; border:1px solid rgba(244,63,94,0.35); padding:3px 10px; border-radius:12px; font-size:0.72rem; font-weight:700; letter-spacing:0.04em;">⚠️ SHARED SINGLE POINT OF FAILURE</span>'
        if is_shared
        else '<span style="background:rgba(148,163,184,0.12); color:#CBD5E1; border:1px solid rgba(148,163,184,0.25); padding:3px 10px; border-radius:12px; font-size:0.72rem; font-weight:600;">STANDALONE COMPONENT</span>'
    )

    supp_chips = "".join([
        f'<span style="background:rgba(56,189,248,0.12); color:#38BDF8; border:1px solid rgba(56,189,248,0.3); padding:2px 8px; border-radius:10px; font-size:0.76rem; font-weight:600; margin-right:4px; display:inline-block; margin-bottom:3px;">{s}</span>'
        for s in affected_suppliers
    ]) if affected_suppliers else '<span style="color:#64748B; font-size:0.78rem;">None detected</span>'

    direct_chips = ", ".join(direct_suppliers) if direct_suppliers else "None"
    indirect_chips = ", ".join(indirect_suppliers) if indirect_suppliers else "None"

    if is_shared:
        why_bottleneck = (
            f"<b>Why this is a shared bottleneck:</b> Feeds <b>{len(affected_suppliers)} independent suppliers</b> simultaneously. "
            f"An operational outage at <b>{node}</b> interrupts parallel production lines with zero qualified backup alternatives."
        )
    else:
        why_bottleneck = (
            f"<b>Component Profile:</b> Dedicated single-path dependency feeding downstream manufacturing routes. "
            f"No cross-supplier contamination detected."
        )

    missing_html = ""
    if missing_info:
        missing_items = " &bull; ".join(missing_info)
        missing_html = (
            f'<div style="margin-top:8px; font-size:0.74rem; color:#FBBF24; background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.25); border-radius:6px; padding:6px 10px;">'
            f'<b>Data Gap Disclosure:</b> {missing_items} (penalized honestly; not assumed zero)'
            f'</div>'
        )

    # Format explanation into a clean vertical list with highlighted penalties and prominent final score
    calc_items = []
    clean_expl = re.sub(r'\.?\s*Final score:\s*[\d\.]+/100\.?', '', explanation, flags=re.IGNORECASE).strip()
    raw_lines = [item.strip() for item in clean_expl.split(";") if item.strip()]

    for line_item in raw_lines:
        line_clean = line_item.rstrip(".")
        # Highlight numerical penalties like (+35), (+15 uncertainty penalty), (+10), (+5)
        line_highlighted = re.sub(
            r'(\(\+[0-9]+(?:\s+[^)]+)?\))',
            r'<span style="background:rgba(245,158,11,0.15); color:#FBBF24; border:1px solid rgba(245,158,11,0.35); padding:2px 8px; border-radius:6px; font-weight:700; font-family:monospace; font-size:0.75rem; margin-left:6px;">\1</span>',
            line_clean
        )
        calc_items.append(
            f'<div style="padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.06); display:flex; justify-content:space-between; align-items:center; font-size:0.78rem; color:#CBD5E1;">'
            f'<span>&bull; {line_highlighted}</span>'
            f'</div>'
        )

    calc_list_html = "".join(calc_items) if calc_items else f'<div style="font-size:0.78rem; color:#CBD5E1;">{explanation}</div>'
    final_score_banner = (
        f'<div style="margin-top:10px; padding-top:8px; border-top:1px solid #1E293B; display:flex; justify-content:space-between; align-items:center;">'
        f'<span style="font-size:0.80rem; font-weight:700; color:#F8FAFC;">Final Calculated Vulnerability Score:</span>'
        f'<span style="font-size:1.15rem; font-weight:800; color:{score_color}; font-family:monospace;">{score:.1f} / 100</span>'
        f'</div>'
    )

    return f"""<div style="background:#151F32; border:1px solid #1E293B; border-top:3px solid {border_color}; border-radius:10px; padding:18px 20px; margin-bottom:14px; box-shadow:0 4px 16px rgba(0,0,0,0.35);">
<div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px; margin-bottom:12px;">
<div>
<div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
<span style="font-size:1.25rem; font-weight:800; color:#F8FAFC;">{node}</span>
{shared_badge}
</div>
<div style="font-size:0.78rem; color:#94A3B8; font-weight:600;">
SUPPLY CHAIN CORRIDOR: <span style="color:#38BDF8;">{tier}</span>
</div>
</div>
<div style="display:flex; align-items:center; gap:12px;">
<div style="text-align:right;">
<div style="font-size:0.70rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em;">Vulnerability Score</div>
<div style="font-size:1.25rem; font-weight:800; color:{score_color}; font-family:monospace;">{score:.1f} <span style="font-size:0.75rem; color:#64748B;">/ 100</span></div>
</div>
<span class="{badge_class}">{level.upper()} RISK</span>
</div>
</div>
<div style="background:rgba(255,255,255,0.02); border:1px solid #1E293B; border-radius:8px; padding:10px 14px; margin-bottom:12px; font-size:0.80rem; color:#CBD5E1; line-height:1.4;">
{why_bottleneck}
</div>
<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:14px; background:#0D1527; border:1px solid #1E293B; border-radius:8px; padding:12px 14px; margin-bottom:12px;">
<div>
<div style="font-size:0.70rem; color:#94A3B8; text-transform:uppercase; font-weight:700; margin-bottom:4px;">Affected Downstream Suppliers ({len(affected_suppliers)}):</div>
<div>{supp_chips}</div>
<div style="font-size:0.74rem; color:#64748B; margin-top:6px;">
<b>Direct Dependencies:</b> <span style="color:#CBD5E1;">{direct_chips}</span> &bull; <b>Multi-Tier Feeds:</b> <span style="color:#CBD5E1;">{indirect_chips}</span>
</div>
</div>
<div>
<div style="font-size:0.70rem; color:#94A3B8; text-transform:uppercase; font-weight:700; margin-bottom:4px;">Audited Resilience Directive:</div>
<div style="font-size:0.78rem; color:#CBD5E1; line-height:1.4;">{rec}</div>
{missing_html}
<details style="margin-top:8px; cursor:pointer;">
<summary style="font-size:0.74rem; color:#38BDF8; font-weight:700; outline:none;">🔬 View Technical Score Calculation & Penalty Details</summary>
<div style="margin-top:8px; background:#070B14; padding:10px 14px; border-radius:8px; border:1px solid #1E293B;">
{calc_list_html}
{final_score_banner}
</div>
</details>
</div>
</div>
</div>"""


def build_financial_scenario_from_simulation(
    simulation_result: Dict[str, Any],
    loss_per_unit: Optional[float] = 150.0,
    available_budget: Optional[float] = 50000.0,
    max_lead_time_days: Optional[float] = 14.0,
    scenario_id_override: Optional[str] = None,
    graph: Optional[nx.DiGraph] = None,
) -> DisruptionScenario:
    """
    Safe adapter converting Member 3's simulation payload into Member 4's DisruptionScenario.
    Respects anti-double-counting and extracts exact quantified throughput deficits.

    Safely handles:
    - Direct outgoing capacity loss (when available and positive).
    - Downstream capacity deficit (when direct loss is not available).
    - Terminal / sink entity disruptions (e.g. Supplier failures with 0 outgoing edges)
      by inspecting rated node capacity and inbound feeding edge capacity from the supply graph.
    - Missing or unrecorded capacity data: accurately records data gaps in metadata without
      fabricating values or crashing with NoneType comparison errors.
    - Optional financial constraints (loss_per_unit, available_budget, max_lead_time_days)
      with safe None-checking.
    """
    payload = get_financial_engine_payload(simulation_result)

    failed_entity = str(payload.get("failed_entity") or "Unknown Entity")
    raw_direct_loss = payload.get("direct_capacity_loss")
    raw_total_deficit = payload.get("total_downstream_capacity_deficit")

    # Safely parse numeric direct loss
    direct_loss: Optional[float] = None
    if raw_direct_loss is not None:
        try:
            val = float(raw_direct_loss)
            if not math.isnan(val) and not math.isinf(val):
                direct_loss = val
        except (ValueError, TypeError):
            direct_loss = None

    # Safely parse numeric downstream deficit
    total_deficit: Optional[float] = None
    if raw_total_deficit is not None:
        try:
            val = float(raw_total_deficit)
            if not math.isnan(val) and not math.isinf(val):
                total_deficit = val
        except (ValueError, TypeError):
            total_deficit = None

    # Determine primary affected quantity
    affected_quantity: float = 0.0
    capacity_source: str = "unrecorded"
    has_capacity_data: bool = False

    # 1. Prefer direct outgoing capacity loss if quantified and positive
    if direct_loss is not None and direct_loss > 0:
        affected_quantity = direct_loss
        capacity_source = "direct_outgoing_capacity"
        has_capacity_data = True

    # 2. Fall back to downstream capacity deficit if direct loss was not quantified
    elif total_deficit is not None and total_deficit > 0:
        affected_quantity = total_deficit
        capacity_source = "downstream_capacity_deficit"
        has_capacity_data = True

    # 3. For terminal/sink nodes, inspect the graph strictly for node rated capacity attribute.
    # Note: Inbound edge capacity must NEVER be treated as confirmed production volume.
    elif graph is not None and graph.has_node(failed_entity):
        node_data = graph.nodes.get(failed_entity, {})
        rated = node_data.get("capacity")
        if rated is not None:
            try:
                r_val = float(rated)
                if r_val > 0 and not math.isnan(r_val) and not math.isinf(r_val):
                    affected_quantity = r_val
                    capacity_source = "node_rated_capacity"
                    has_capacity_data = True
            except (ValueError, TypeError):
                pass

    # 4. If capacity is truly unrecorded / missing, do not fabricate numbers or borrow inbound flows.
    # Mark financial impact status as UNQUANTIFIED.
    if not has_capacity_data:
        affected_quantity = 0.0
        capacity_source = "missing_unrecorded"

    is_unquantified = not has_capacity_data

    # Check for recorded inbound feed links to distinguish inbound feed from confirmed node throughput
    inbound_feed_capacity: Optional[float] = None
    inbound_feed_details: List[str] = []
    if graph is not None and graph.has_node(failed_entity):
        inbound_caps = []
        for u, _, edata in graph.in_edges(failed_entity, data=True):
            cap = edata.get("capacity")
            if cap is not None:
                try:
                    c_val = float(cap)
                    inbound_caps.append(c_val)
                    mat = edata.get("material", "Material")
                    inbound_feed_details.append(f"{u} -> {failed_entity} ({mat}: {c_val:,.1f} units)")
                except (ValueError, TypeError):
                    pass
        if inbound_caps:
            inbound_feed_capacity = float(sum(inbound_caps))

    # Safely convert financial constraints (handling None safely)
    parsed_loss_per_unit = 150.0
    if loss_per_unit is not None:
        try:
            parsed_loss_per_unit = float(loss_per_unit)
        except (ValueError, TypeError):
            parsed_loss_per_unit = 150.0

    parsed_budget: Optional[float] = None
    if available_budget is not None:
        try:
            parsed_budget = float(available_budget)
        except (ValueError, TypeError):
            parsed_budget = None

    parsed_lead_time: Optional[float] = None
    if max_lead_time_days is not None:
        try:
            parsed_lead_time = float(max_lead_time_days)
        except (ValueError, TypeError):
            parsed_lead_time = None

    scen_id = scenario_id_override or f"SCEN-{failed_entity.replace(' ', '-').upper()}"

    missing_warnings = list(payload.get("missing_data_warnings", []))
    if not has_capacity_data:
        warn = f"Capacity throughput data is unrecorded for failed entity '{failed_entity}'. Financial impact is UNQUANTIFIED."
        if inbound_feed_capacity is not None:
            feed_nodes = [u for u, _ in graph.in_edges(failed_entity)]
            warn += (
                f" Note: Inbound feeding link capacity is recorded ({inbound_feed_capacity:,.1f} units from {', '.join(feed_nodes)}), "
                "but confirmed operational production throughput for this entity is unrecorded. "
                "Inbound feeding capacity is not assumed to equal node production throughput."
            )
        missing_warnings.append(warn)

    scenario = DisruptionScenario(
        scenario_id=scen_id,
        affected_quantity=affected_quantity,
        loss_per_unit=parsed_loss_per_unit,
        disrupted_node=failed_entity,
        available_budget=parsed_budget,
        max_lead_time_days=parsed_lead_time,
        description=f"Automated resilience scenario for {failed_entity} disruption.",
        additional_metadata={
            "impact_level": payload.get("impact_level"),
            "affected_materials": payload.get("affected_materials"),
            "confirmed_shortages": payload.get("confirmed_shortage_entities"),
            "potential_shortages": payload.get("potential_shortage_entities"),
            "has_capacity_data": has_capacity_data,
            "is_unquantified": is_unquantified,
            "financial_impact_status": "UNQUANTIFIED" if is_unquantified else "VERIFIED",
            "capacity_source": capacity_source,
            "inbound_recorded_feed_capacity": inbound_feed_capacity,
            "inbound_feed_details": inbound_feed_details,
            "missing_data_warnings": missing_warnings,
        },
    )

    return scenario


def generate_candidate_recovery_options(
    scenario: DisruptionScenario,
    graph: Optional[nx.DiGraph] = None,
) -> List[RecoveryOption]:
    """
    Generates realistic, domain-grounded candidate recovery options scaled
    to the specific disrupted quantity and scenario constraints.
    """
    qty = scenario.affected_quantity
    budget = scenario.available_budget if scenario.available_budget is not None else 50000.0
    has_capacity_data = scenario.additional_metadata.get("has_capacity_data", qty > 0)
    is_unquantified = scenario.additional_metadata.get("is_unquantified", not has_capacity_data or qty == 0.0)

    # If capacity is completely missing/unrecorded, candidate options must report
    # available_capacity as None and be clearly flagged as illustrative estimates so
    # Member 4 honestly flags missing capacity rather than declaring verified feasibility.
    if is_unquantified or (not has_capacity_data and qty == 0.0):
        opt1 = RecoveryOption(
            option_id="OPT-BUFFER-RELEASE",
            option_type="buffer_stock_release",
            description="[ILLUSTRATIVE ESTIMATE] Strategic reserve stock release (capacity unverified pending audit)",
            recovery_cost=round(budget * 0.36, 2),
            available_capacity=None,  # Missing capacity data
            lead_time_days=3.0,
            maximum_acceptable_lead_time_days=scenario.max_lead_time_days,
        )
        opt2 = RecoveryOption(
            option_id="OPT-EXPEDITE-AIR",
            option_type="expedited_freight",
            description="[ILLUSTRATIVE ESTIMATE] Air freight replacement shipment (capacity unverified pending audit)",
            recovery_cost=round(budget * 0.54, 2),
            available_capacity=None,  # Missing capacity data
            lead_time_days=6.0,
            maximum_acceptable_lead_time_days=scenario.max_lead_time_days,
        )
        return [opt1, opt2]

    unit_cost_est = (budget * 0.4) / max(qty, 1.0)

    # Option 1: Fast Reserve Inventory / Safety Buffer Release
    opt1 = RecoveryOption(
        option_id="OPT-BUFFER-RELEASE",
        option_type="buffer_stock_release",
        description="Release emergency strategic reserve stock from regional distribution warehouse",
        recovery_cost=round(qty * unit_cost_est * 0.9, 2),
        available_capacity=round(qty, 1),
        lead_time_days=3.0,
    )

    # Option 2: Expedited Air Freight from Qualified Secondary Supplier
    opt2 = RecoveryOption(
        option_id="OPT-EXPEDITE-AIR",
        option_type="expedited_freight",
        description="Air freight replacement shipment from secondary qualified manufacturing facility",
        recovery_cost=round(qty * unit_cost_est * 1.35, 2),
        available_capacity=round(qty * 1.1, 1),
        lead_time_days=6.0,
    )

    # Option 3: Surface Cargo / Standard Secondary Sourcing
    opt3 = RecoveryOption(
        option_id="OPT-SECONDARY-SOURCING",
        option_type="secondary_supplier",
        description="Procure full volume replenishment via standard logistics carrier",
        recovery_cost=round(qty * unit_cost_est * 0.7, 2),
        available_capacity=round(qty, 1),
        lead_time_days=18.0,  # Deliberately longer lead time to test constraint checking
    )

    # Option 4: Partial Expedite + Overtime Production Run
    opt4 = RecoveryOption(
        option_id="OPT-OVERTIME-PRODUCTION",
        option_type="overtime_shift",
        description="Contract weekend overtime shifts at secondary processing partner",
        recovery_cost=round(qty * unit_cost_est * 1.1, 2),
        available_capacity=round(qty * 0.8, 1),  # Capacity shortfall to test constraint checking
        lead_time_days=8.0,
    )

    return [opt1, opt2, opt3, opt4]

# -----------------------------------------------------------------------------
# Feature 2: Five-Agent Workflow Orchestration Panel Component
# -----------------------------------------------------------------------------

AGENT_THEMES = {
    "mapping": {
        "order": 1,
        "name": "Mapping Agent",
        "role": "Dependency Graph",
        "color": "#3B82F6",       # Blue
        "glow": "rgba(59, 130, 246, 0.45)",
        "accent_class": "agent-accent-blue",
        "proc_class": "agent-processing-blue",
        "icon": "🗺️",
    },
    "risk": {
        "order": 2,
        "name": "Risk Detection Agent",
        "role": "Bottlenecks & SPOFs",
        "color": "#06B6D4",       # Cyan
        "glow": "rgba(6, 182, 212, 0.45)",
        "accent_class": "agent-accent-cyan",
        "proc_class": "agent-processing-cyan",
        "icon": "🛡️",
    },
    "simulation": {
        "order": 3,
        "name": "Disruption Simulation Agent",
        "role": "Shock & Blast Radius",
        "color": "#F59E0B",       # Amber
        "glow": "rgba(245, 158, 11, 0.45)",
        "accent_class": "agent-accent-amber",
        "proc_class": "agent-processing-amber",
        "icon": "⚡",
    },
    "optimization": {
        "order": 4,
        "name": "Financial Optimization Agent",
        "role": "Cost & Lead Time LP",
        "color": "#10B981",       # Green
        "glow": "rgba(16, 185, 129, 0.45)",
        "accent_class": "agent-accent-green",
        "proc_class": "agent-processing-green",
        "icon": "💰",
    },
    "explanation": {
        "order": 5,
        "name": "Decision Explanation Agent",
        "role": "Strategy & Governance",
        "color": "#8B5CF6",       # Purple
        "glow": "rgba(139, 92, 246, 0.45)",
        "accent_class": "agent-accent-purple",
        "proc_class": "agent-processing-purple",
        "icon": "⚖️",
    },
}


def render_five_agent_workflow_panel(
    agents_state: List[Dict[str, Any]],
    total_pipeline_ms: Optional[float] = None,
    active_run_id: Optional[str] = None,
    is_live_executing: bool = False,
    current_agent_idx: Optional[int] = None,
) -> str:
    """
    Renders an enterprise Five-Agent Workflow Orchestration Panel showing real-time
    activation states (WAITING, PROCESSING, COMPLETED, ERROR) across the 5 specialized
    resilience agents with connecting pipeline flow and live metrics.

    NOTE: The returned HTML is stripped of leading indentation and internal blank lines
    to strictly prevent Streamlit's Markdown renderer from treating indented HTML blocks
    as literal <pre><code> code blocks.
    """
    cards_html = []
    completed_count = sum(1 for a in agents_state if a.get("state") == "COMPLETED")
    has_error = any(a.get("state") == "ERROR" for a in agents_state)

    for i, a in enumerate(agents_state):
        agent_id = a.get("id", "mapping")
        meta = AGENT_THEMES.get(agent_id, AGENT_THEMES["mapping"])
        state = a.get("state", "WAITING")
        duration = a.get("duration_ms", 0.0)
        summary = a.get("summary", "")
        error_msg = a.get("error")

        if state == "PROCESSING":
            card_class = f"agent-workflow-card agent-card-processing {meta['proc_class']}"
            badge_html = '<span class="agent-status-badge badge-state-processing"><span class="agent-spinner"></span> ACTIVE</span>'
            footer_html = (
                f'<div class="agent-progress-track"><div class="agent-progress-bar" style="background:{meta["color"]};"></div></div>'
                f'<div style="font-size:0.70rem; color:{meta["color"]}; font-weight:700; margin-top:4px;">Computing...</div>'
            )
        elif state == "COMPLETED":
            card_class = f"agent-workflow-card agent-card-completed {meta['accent_class']}"
            badge_html = '<span class="agent-status-badge badge-state-completed">✓ DONE</span>'
            summary_disp = summary or "Analysis Complete"
            footer_html = (
                f'<div style="font-size:0.74rem; font-weight:600; color:#F8FAFC; line-height:1.35; margin-bottom:4px; word-break:normal; white-space:normal;">{summary_disp}</div>'
                f'<div style="font-size:0.68rem; color:#64748B; font-family:monospace; margin-top:auto;">⚡ {duration:.1f}ms</div>'
            )
        elif state == "ERROR":
            card_class = "agent-workflow-card agent-card-error"
            badge_html = '<span class="agent-status-badge badge-state-error">✕ ERROR</span>'
            err_text = str(error_msg or "Failed")[:35]
            footer_html = f'<div style="font-size:0.70rem; color:#FCA5A5; font-weight:600; line-height:1.2;">{err_text}</div>'
        else:  # WAITING
            card_class = "agent-workflow-card agent-card-waiting"
            badge_html = '<span class="agent-status-badge badge-state-waiting">⏳ STANDBY</span>'
            footer_html = '<div style="font-size:0.70rem; color:#64748B;">Awaiting upstream trigger</div>'

        card = (
            f'<div class="{card_class}">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">'
            f'<div style="display:flex; align-items:center; gap:6px;">'
            f'<span style="font-size:1.10rem; line-height:1;">{meta["icon"]}</span>'
            f'<span class="agent-order-tag">0{meta["order"]}</span>'
            f'</div>'
            f'{badge_html}'
            f'</div>'
            f'<div style="flex-grow:1; display:flex; flex-direction:column; justify-content:center; margin:4px 0 6px 0;">'
            f'<div class="agent-card-name" style="color:{meta["color"]};">{meta["name"]}</div>'
            f'<div class="agent-card-role">{meta["role"]}</div>'
            f'</div>'
            f'<div style="margin-top:auto; padding-top:6px; border-top:1px solid rgba(255,255,255,0.06);">'
            f'{footer_html}'
            f'</div>'
            f'</div>'
        )
        cards_html.append(card)

    cards_joined = "".join(cards_html)

    # Status Banner Message
    if has_error:
        status_html = '<span style="color:#F87171; font-weight:700;">⚠️ Pipeline Interrupted: One or more agents encountered an error</span>'
    elif is_live_executing and current_agent_idx is not None and current_agent_idx < len(agents_state):
        curr_id = agents_state[current_agent_idx].get("id", "")
        curr_meta = AGENT_THEMES.get(curr_id, {})
        curr_name = agents_state[current_agent_idx].get("name") or curr_meta.get("name", "Agent")
        status_html = f'<span style="color:#38BDF8; font-weight:600;"><span class="telemetry-pulse"></span> Executing Agent {current_agent_idx + 1}/5: {curr_name}...</span>'
    elif completed_count == 5:
        total_time_str = f" &bull; Total Pipeline Runtime: {total_pipeline_ms:.1f}ms" if total_pipeline_ms else ""
        run_id_str = f" &bull; Run ID: <code>{active_run_id[:8]}</code>" if active_run_id else ""
        status_html = f'<span style="color:#34D399; font-weight:700;">✓ 5/5 Specialized Resilience Agents Synchronized{total_time_str}{run_id_str}</span>'
    else:
        status_html = f'<span style="color:#94A3B8; font-weight:600;">{completed_count}/5 Agents Completed</span>'

    panel_html = (
        f'<div class="agent-orchestration-panel">'
        f'<div class="agent-panel-header">'
        f'<div style="display:flex; align-items:center; gap:8px;">'
        f'<span class="telemetry-pill telemetry-standby" style="font-size:0.70rem; padding:2px 8px; font-weight:700;">'
        f'AGENT PIPELINE'
        f'</span>'
        f'<span style="font-size:0.86rem; font-weight:800; color:#F8FAFC; letter-spacing:-0.01em;">Autonomous Disruption Analysis & Orchestration</span>'
        f'</div>'
        f'<div style="font-size:0.75rem;">'
        f'{status_html}'
        f'</div>'
        f'</div>'
        f'<div class="workflow-pipeline-grid">'
        f'{cards_joined}'
        f'</div>'
        f'</div>'
    )
    # Strip any possible leading/trailing whitespace or newlines so Markdown never parses as a code block
    return "".join(line.strip() for line in panel_html.splitlines() if line.strip())


def create_disruption_propagation_figure(
    graph: nx.DiGraph,
    simulation_result: Optional[Dict[str, Any]] = None,
    selected_entity: Optional[str] = None,
    title: Optional[str] = None,
    animate: bool = True,
) -> go.Figure:
    """
    Renders an interactive animated Disruption Propagation canvas demonstrating
    how an operational shock originating at a selected entity flows through
    the multi-tier supply chain network to downstream partners.

    Feature 3 Highlights:
    - Origin Expanding Shockwave Ring around failed entity (prominent red radar pulse).
    - Severed Corridors with Amber / Crimson Disruption Shock Particles traveling downstream.
    - Operational Corridors with Blue Material Flow Particles.
    - Updatemenus with Play, Pause, and Replay controls.
    - Static mode fallback for reduced motion.
    """
    if title is None:
        failed_name = (
            simulation_result.get("failed_entity", selected_entity or "Target Facility")
            if simulation_result
            else (selected_entity or "Target Facility")
        )
        title = f"Live Disruption Propagation Flow: {failed_name}"

    return create_resilient_network_figure(
        graph=graph,
        simulation_result=simulation_result,
        selected_entity=selected_entity,
        title=title,
        animate=animate,
        show_replay=True,
        show_propagation_signal=True,
    )


def render_disruption_propagation_summary(
    simulation_result: Optional[Dict[str, Any]],
    graph: Optional[nx.DiGraph] = None,
) -> str:
    """
    Renders an executive disruption propagation summary component for Feature 3.
    Displays:
    - Disruption origin node and operational role
    - Disruption severity / impact level with status badge
    - Confirmed shortages vs. potentially affected entities
    - Total affected downstream entities
    - Severed connections count and direct capacity loss
    - Directional propagation cascade flow

    NOTE: The returned HTML is stripped of leading whitespace and internal blank lines
    to strictly prevent Streamlit's Markdown renderer from treating indented HTML blocks
    as literal <pre><code> code blocks.
    """
    if not simulation_result:
        return (
            '<div style="background:#151F32; border:1px solid #1E293B; border-radius:10px; padding:16px; margin-bottom:14px; text-align:center; color:#94A3B8;">'
            'No active disruption simulation loaded. Baseline network operational.'
            '</div>'
        )

    failed_entity = simulation_result.get("failed_entity", "Unknown")
    impact_level = str(simulation_result.get("impact_level", "Medium")).capitalize()
    disruption_type = simulation_result.get("disruption_type", "Operational Outage")
    affected_entities = simulation_result.get("affected_entities", [])
    affected_count = len(affected_entities)

    # Determine node role and icon
    node_role = "Entity"
    if graph and failed_entity in graph:
        node_role = str(graph.nodes[failed_entity].get("node_type", "Asset")).title()

    # Confirmed shortages vs potentially affected
    entity_details = simulation_result.get("entity_impact_details", [])
    confirmed_nodes = [
        e.get("entity", "") for e in entity_details
        if e.get("status") == "confirmed_shortage" or e.get("confirmed_shortage")
    ]
    potential_nodes = [
        e.get("entity", "") for e in entity_details
        if e.get("status") == "potentially_affected" or e.get("potentially_affected")
    ]
    confirmed_count = len(confirmed_nodes)
    potential_count = len(potential_nodes)

    # Severed connections count
    severed_edges = set()
    if graph:
        for u, v in graph.edges():
            if u == failed_entity or v == failed_entity:
                severed_edges.add((u, v))
        paths = simulation_result.get("affected_paths") or simulation_result.get("disrupted_paths") or []
        for p in paths:
            for i in range(len(p) - 1):
                severed_edges.add((p[i], p[i + 1]))
    severed_count = len(severed_edges)

    # Severity tier styling
    sev_class = {
        "Critical": "badge-critical",
        "High": "badge-high",
        "Medium": "badge-medium",
        "Low": "badge-low",
    }.get(impact_level, "badge-medium")

    sev_color = {
        "Critical": "#FB7185",
        "High": "#FBBF24",
        "Medium": "#FDE047",
        "Low": "#34D399",
    }.get(impact_level, "#FDE047")

    cap_impact = simulation_result.get("estimated_capacity_impact", {})
    direct_loss = cap_impact.get("estimated_direct_capacity_loss")
    loss_disp = safe_format_capacity(direct_loss, default="Unquantified", decimals=1, unit=" units")

    confirmed_disp = f"{confirmed_count} Confirmed" if confirmed_count > 0 else "0 Confirmed"
    potential_disp = f"{potential_count} Buffered" if potential_count > 0 else "0 Buffered"

    lines = [
        '<div class="propagation-impact-summary">',
        '  <div class="propagation-summary-header">',
        '    <div class="propagation-summary-title-group">',
        '      <div class="propagation-summary-kicker">LIVE BLAST RADIUS & SHOCK PROPAGATION TELEMETRY</div>',
        f'     <div class="propagation-summary-title">💥 Disruption Origin: <span style="color:#F8FAFC;">{failed_entity}</span> <span style="font-size:0.85rem; font-weight:600; color:#38BDF8; margin-left:8px;">({node_role})</span></div>',
        '    </div>',
        '    <div class="propagation-summary-badges">',
        f'     <span class="{sev_class}">{impact_level.upper()} IMPACT</span>',
        f'     <span class="badge-critical" style="background:rgba(239,68,68,0.2) !important;">{disruption_type.upper()}</span>',
        '    </div>',
        '  </div>',
        '  <div class="propagation-summary-metrics-grid">',
        '    <div class="propagation-summary-metric-card">',
        '      <div class="propagation-metric-label">Downstream Blast Radius</div>',
        f'     <div class="propagation-metric-value" style="color:#FB7185;">{affected_count} <span style="font-size:0.82rem; font-weight:500; color:#94A3B8;">Entities</span></div>',
        f'     <div class="propagation-metric-sub">{", ".join(affected_entities) if affected_entities else "Leaf node (no downstream)"}</div>',
        '    </div>',
        '    <div class="propagation-summary-metric-card">',
        '      <div class="propagation-metric-label">Confirmed vs Potential Shortages</div>',
        f'     <div class="propagation-metric-value" style="color:{sev_color};">{confirmed_disp}</div>',
        f'     <div class="propagation-metric-sub">{potential_disp} with operational fallback</div>',
        '    </div>',
        '    <div class="propagation-summary-metric-card">',
        '      <div class="propagation-metric-label">Severed Supply Corridors</div>',
        f'     <div class="propagation-metric-value" style="color:#F59E0B;">{severed_count} <span style="font-size:0.82rem; font-weight:500; color:#94A3B8;">Routes</span></div>',
        f'     <div class="propagation-metric-sub">Direct capacity offline: {loss_disp}</div>',
        '    </div>',
        '    <div class="propagation-summary-metric-card">',
        '      <div class="propagation-metric-label">Propagation Trajectory</div>',
        f'     <div class="propagation-metric-value" style="color:#38BDF8; font-size:1.0rem;">Directed Cascade</div>',
        f'     <div class="propagation-metric-sub">{failed_entity} &rarr; {" &rarr; ".join(affected_entities[:2]) if affected_entities else "Terminal Sink"}</div>',
        '    </div>',
        '  </div>',
        '</div>',
    ]
    return "".join(line.strip() for line in lines if line.strip())

