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


# Status Colors for Resilience Graph
STATUS_COLORS: Dict[str, str] = {
    "failed": "#D32F2F",                # Strong Red
    "confirmed_shortage": "#C2185B",    # Crimson
    "potentially_affected": "#F57C00",  # Amber/Orange
    "unaffected_raw": "#FF8F00",        # Deep Amber
    "unaffected_proc": "#0288D1",       # Light Blue
    "unaffected_supp": "#2E7D32",       # Forest Green
    "unaffected_fact": "#6A1B9A",       # Royal Purple
    "unaffected_other": "#546E7A",      # Slate Grey
}


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
    pos = compute_hierarchical_layout(graph)
    fig = go.Figure()

    # Identify broken edges from simulation
    failed_edges: Set[Tuple[str, str]] = set()
    failed_entity = simulation_result.get("failed_entity", "") if simulation_result else ""

    if simulation_result:
        # If a node failed, all its outgoing and incoming edges are broken
        for u, v in graph.edges():
            if u == failed_entity or v == failed_entity:
                failed_edges.add((u, v))

        # Check explicit failed edge
        if " -> " in failed_entity:
            # Parse route or edge string e.g. "Route (Supplier A -> Factory 1)"
            for u, v in graph.edges():
                if f"{u} -> {v}" in failed_entity:
                    failed_edges.add((u, v))

    # Step 1: Draw Directed Edges & Route Annotations
    annotations = []
    mid_x, mid_y, mid_text, mid_colors = [], [], [], []

    for u, v, data in graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        material = data.get("material", "Commodity")
        capacity = data.get("capacity", "N/A")
        is_broken = (u, v) in failed_edges

        # Calculate midpoint
        mx = (x0 + x1) / 2.0
        my = (y0 + y1) / 2.0
        mid_x.append(mx)
        mid_y.append(my)

        status_str = "<b style='color:red;'>SEVERED ROUTE</b>" if is_broken else "<b style='color:green;'>OPERATIONAL</b>"
        cap_str = f"{capacity:,.0f} units" if isinstance(capacity, (int, float)) else str(capacity)
        mid_text.append(
            f"<b>Supply Route:</b> {u} &rarr; {v}<br>"
            f"<b>Status:</b> {status_str}<br>"
            f"<b>Material:</b> {material}<br>"
            f"<b>Capacity:</b> {cap_str}"
        )
        mid_colors.append("#D32F2F" if is_broken else "#455A64")

        # Arrow calculation
        dx = x1 - x0
        dy = y1 - y0
        dist = math.hypot(dx, dy)
        if dist > 0:
            offset_start = 0.20
            offset_end = 0.25
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
                arrowwidth=3 if is_broken else 1.8,
                arrowcolor="#D32F2F" if is_broken else "#90A4AE",
                opacity=0.9 if is_broken else 0.7,
            )
        )

    # Route hover trace
    fig.add_trace(
        go.Scatter(
            x=mid_x,
            y=mid_y,
            mode="markers",
            marker=dict(
                size=12,
                color=mid_colors,
                symbol="circle-dot",
                line=dict(color="#FFFFFF", width=1.5),
            ),
            hoverinfo="text",
            hovertext=mid_text,
            name="Supply Routes (Hover)",
            showlegend=True,
        )
    )

    # Step 2: Draw Nodes Grouped by State & Role
    # Define visual buckets
    status_groups: Dict[str, Dict[str, Any]] = {
        "failed": {"nodes": [], "color": STATUS_COLORS["failed"], "symbol": "x", "name": "Disrupted Root Failure", "size": 24},
        "confirmed_shortage": {"nodes": [], "color": STATUS_COLORS["confirmed_shortage"], "symbol": "diamond-tall", "name": "Confirmed Shortage", "size": 20},
        "potentially_affected": {"nodes": [], "color": STATUS_COLORS["potentially_affected"], "symbol": "triangle-up", "name": "Potentially Affected", "size": 18},
        "operational_raw": {"nodes": [], "color": STATUS_COLORS["unaffected_raw"], "symbol": "diamond", "name": "Raw Material (Intact)", "size": 16},
        "operational_proc": {"nodes": [], "color": STATUS_COLORS["unaffected_proc"], "symbol": "square", "name": "Processor (Intact)", "size": 16},
        "operational_supp": {"nodes": [], "color": STATUS_COLORS["unaffected_supp"], "symbol": "circle", "name": "Supplier (Intact)", "size": 16},
        "operational_fact": {"nodes": [], "color": STATUS_COLORS["unaffected_fact"], "symbol": "hexagon", "name": "Factory / Plant (Intact)", "size": 18},
        "operational_other": {"nodes": [], "color": STATUS_COLORS["unaffected_other"], "symbol": "circle", "name": "Other Entity (Intact)", "size": 15},
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

    for group_key, group_data in status_groups.items():
        nodes = group_data["nodes"]
        if not nodes:
            continue

        nx_c = [pos[n][0] for n in nodes]
        ny_c = [pos[n][1] for n in nodes]
        hover_texts = []
        labels = []

        for n in nodes:
            status_key, status_label = classify_node_status(n, simulation_result)
            in_deg = graph.in_degree(n)
            out_deg = graph.out_degree(n)
            node_type = graph.nodes[n].get("node_type", "Unknown")
            alts = graph.nodes[n].get("alternatives", [])
            alts_str = ", ".join(alts) if alts else "None registered"

            is_selected = selected_entity and (selected_entity == n)
            selection_marker = " 🎯 [SELECTED]" if is_selected else ""

            hover_html = (
                f"<b>Entity:</b> {n}{selection_marker}<br>"
                f"<b>Tier / Role:</b> {node_type}<br>"
                f"<b>Operational Status:</b> {status_label}<br>"
                f"<b>Inbound Feeds:</b> {in_deg}<br>"
                f"<b>Outbound Feeds:</b> {out_deg}<br>"
                f"<b>Known Alternatives:</b> {alts_str}"
            )
            hover_texts.append(hover_html)
            labels.append(n)

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
                        color="#FFD700" if (selected_entity in nodes) else "#FFFFFF",
                        width=2.5 if (selected_entity in nodes) else 1.5,
                    ),
                ),
                text=labels,
                textposition="bottom center",
                textfont=dict(size=11, color="#E2E8F0", family="Arial, sans-serif"),
                hoverinfo="text",
                hovertext=hover_texts,
                name=group_data["name"],
                showlegend=True,
            )
        )

    # Layout styling — dark theme
    # template="plotly_dark" makes the modebar, hover labels, colour pickers,
    # and axis defaults all dark-themed at the Plotly engine level.
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=18, color="#F8FAFC"),
            x=0.02,
            y=0.96,
        ),
        annotations=annotations,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(17, 24, 39, 0.85)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color="#CBD5E1"),
        ),
        hoverlabel=dict(
            bgcolor="#1E293B",
            bordercolor="#334155",
            font=dict(color="#F8FAFC", size=12),
        ),
        hovermode="closest",
        margin=dict(b=80, l=40, r=40, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#111827",
        height=620,
        font=dict(color="#CBD5E1", family="Arial, sans-serif"),
    )

    return fig


def build_financial_scenario_from_simulation(
    simulation_result: Dict[str, Any],
    loss_per_unit: float = 150.0,
    available_budget: float = 50000.0,
    max_lead_time_days: float = 14.0,
    scenario_id_override: Optional[str] = None,
) -> DisruptionScenario:
    """
    Safe adapter converting Member 3's simulation payload into Member 4's DisruptionScenario.
    Respects anti-double-counting and extracts exact quantified throughput deficits.
    """
    payload = get_financial_engine_payload(simulation_result)

    failed_entity = payload.get("failed_entity", "Unknown Entity")
    direct_loss = payload.get("direct_capacity_loss", 0.0)
    total_deficit = payload.get("total_downstream_capacity_deficit", 0.0)

    # Determine primary affected quantity: prefer direct capacity loss, fallback to downstream deficit
    if direct_loss > 0:
        affected_quantity = float(direct_loss)
    elif total_deficit > 0:
        affected_quantity = float(total_deficit)
    else:
        # Default minimum unit fallback if unrecorded
        affected_quantity = 100.0

    scen_id = scenario_id_override or f"SCEN-{failed_entity.replace(' ', '-').upper()}"

    scenario = DisruptionScenario(
        scenario_id=scen_id,
        affected_quantity=affected_quantity,
        loss_per_unit=float(loss_per_unit),
        disrupted_node=failed_entity,
        available_budget=float(available_budget),
        max_lead_time_days=float(max_lead_time_days),
        description=f"Automated resilience scenario for {failed_entity} disruption.",
        additional_metadata={
            "impact_level": payload.get("impact_level"),
            "affected_materials": payload.get("affected_materials"),
            "confirmed_shortages": payload.get("confirmed_shortage_entities"),
            "potential_shortages": payload.get("potential_shortage_entities"),
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
