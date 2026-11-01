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
) -> go.Figure:
    """
    Generates an interactive, state-aware Plotly visualization of the supply chain network.
    Dynamically highlights disrupted nodes, confirmed shortages, potential shortages,
    and severed vs. operational transport routes.
    """
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

    # 3. Process Corridors: Line segments, directed annotations & midpoints
    annotations = []
    op_line_x: List[Optional[float]] = []
    op_line_y: List[Optional[float]] = []
    sev_line_x: List[Optional[float]] = []
    sev_line_y: List[Optional[float]] = []
    mid_x, mid_y, mid_text, mid_colors = [], [], [], []

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
        else:
            op_line_x.extend([x0, x1, None])
            op_line_y.extend([y0, y1, None])

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

    # Add Operational Corridors line trace
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

    # Add Severed Corridors line trace
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

    # Add Route Telemetry Waypoint trace
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

    # 4. Group Nodes by State & Category
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

    # 6. Overall Layout Styling
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=17, color="#F8FAFC", family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
            x=0.02,
            y=0.96,
        ),
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
        margin=dict(b=70, l=40, r=40, t=55),
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
<summary style="font-size:0.74rem; color:#38BDF8; font-weight:700;">🔬 View Technical Score Calculation & Penalty Details</summary>
<div style="margin-top:6px; font-size:0.75rem; color:#94A3B8; line-height:1.4; background:#070B14; padding:8px 10px; border-radius:6px; border:1px solid #1E293B;">
{explanation}
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
