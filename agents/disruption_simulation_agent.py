"""
Nexus-AI: Disruption Simulation Agent (Member 3)
=================================================

This module simulates the ripple effects of disruptions across multi-tier
supply chain networks.

Theme: Multi-Tier Supply Chain Resilience and Investment Optimization.

Supported Disruption Types:
- Supplier failure
- Processor failure
- Factory failure
- Transport route failure (edge or route segment disruption)
- Raw material failure / General entity failure

Graph Convention:
- Directed NetworkX graph (nx.DiGraph) where edges point from upstream
  entities to downstream entities:
      [Upstream Entity] ---> [Downstream Entity]
      (e.g., Raw Material P ---> Processor X ---> Supplier A ---> Factory 1)

Key Principles:
1. No Data Fabrication: Missing data is flagged as missing_information.
2. No Double-Counting: Capacities across consecutive tiers are not summed.
3. Realistic Assessment: Distinguishes between "potentially affected" entities
   and "confirmed shortages" based on alternate inbound paths.
4. Transparent Heuristics: Impact levels are computed via explicit, documented
   rules without claiming scientific validation.

Author: Nexus-AI Team (Member 3)
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import networkx as nx
import pandas as pd


# Canonical disruption types supported by the agent
SUPPORTED_DISRUPTION_TYPES = {
    "supplier_failure": "Supplier failure",
    "processor_failure": "Processor failure",
    "factory_failure": "Factory failure",
    "transport_route_failure": "Transport route failure",
    "raw_material_failure": "Raw material failure",
    "general_failure": "General entity failure",
}


def _infer_disruption_type(
    graph: nx.DiGraph,
    failed_entity: Union[str, Tuple[str, str]],
    user_type: Optional[str] = None,
) -> str:
    """
    Infers or normalizes the disruption type based on node attributes or input format.
    """
    if user_type:
        normalized = user_type.strip().lower()
        for key, label in SUPPORTED_DISRUPTION_TYPES.items():
            if normalized in (key, label.lower()):
                return label
        # Return user's provided label if non-empty
        return user_type

    # Edge failure check
    if isinstance(failed_entity, tuple) and len(failed_entity) == 2:
        return SUPPORTED_DISRUPTION_TYPES["transport_route_failure"]

    if failed_entity in graph.nodes:
        node_data = graph.nodes[failed_entity]
        node_type = str(node_data.get("node_type", "")).lower()
        if "supplier" in node_type or node_data.get("is_supplier"):
            return SUPPORTED_DISRUPTION_TYPES["supplier_failure"]
        if "processor" in node_type or "refiner" in node_type:
            return SUPPORTED_DISRUPTION_TYPES["processor_failure"]
        if "factory" in node_type or "plant" in node_type or "assembly" in node_type:
            return SUPPORTED_DISRUPTION_TYPES["factory_failure"]
        if "raw" in node_type or "mine" in node_type:
            return SUPPORTED_DISRUPTION_TYPES["raw_material_failure"]

    return SUPPORTED_DISRUPTION_TYPES["general_failure"]


def _find_disrupted_paths(
    graph: nx.DiGraph,
    start_nodes: List[str],
    cutoff_depth: int = 10,
) -> List[List[str]]:
    """
    Finds downstream paths originating from the disrupted nodes to reachable terminal nodes.
    Deduplicates and prevents infinite recursion if cycles exist.
    """
    paths: List[List[str]] = []
    visited_path_signatures: Set[str] = set()

    for start in start_nodes:
        # Find all reachable descendants
        descendants = nx.descendants(graph, start)
        if not descendants:
            # Terminal node with no downstream successors
            paths.append([start])
            continue

        # Target sinks or intermediate nodes
        sinks = [d for d in descendants if graph.out_degree(d) == 0]
        targets = sinks if sinks else list(descendants)

        for target in targets:
            try:
                for path in nx.all_simple_paths(graph, source=start, target=target, cutoff=cutoff_depth):
                    sig = "->".join(path)
                    if sig not in visited_path_signatures:
                        visited_path_signatures.add(sig)
                        paths.append(path)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    return paths


def _evaluate_entity_shortage(
    graph: nx.DiGraph,
    entity: str,
    failed_nodes: Set[str],
    failed_edges: Set[Tuple[str, str]],
) -> Dict[str, Any]:
    """
    Evaluates whether an entity is 'potentially affected' or faces a 'confirmed shortage',
    and verifies whether alternative supply paths provide sufficient capacity.

    Capacity & Demand Sufficiency Principle:
    - An alternative supply path is NOT treated as sufficient unless available
      capacity data verifies that remaining operational throughput meets requirements.
    - If remaining capacity is less than normal inbound flow, an explicit capacity
      deficit is flagged.
    - If capacity attributes are missing, sufficiency is flagged as unverified.
    """
    node_data = graph.nodes.get(entity, {})
    alternatives = node_data.get("alternatives") or []
    if isinstance(alternatives, str):
        alternatives = [alternatives]

    total_inbound = graph.in_degree(entity)

    # Build operational graph without failed nodes and edges
    operational_graph = graph.copy()
    operational_graph.remove_nodes_from(failed_nodes)
    operational_graph.remove_edges_from(failed_edges)

    # Identify original upstream sources (in-degree == 0 in full graph)
    original_sources = [n for n in graph.nodes if graph.in_degree(n) == 0]
    active_sources = [s for s in original_sources if s in operational_graph]

    # Check reachability from any operational source to this entity
    has_active_source_path = False
    active_sources_reaching = []
    if entity in operational_graph:
        for src in active_sources:
            try:
                if nx.has_path(operational_graph, src, entity):
                    has_active_source_path = True
                    active_sources_reaching.append(src)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

    # Evaluate operational inbound links and inbound capacities
    operational_inbound_count = 0
    normal_inbound_caps: List[float] = []
    operational_inbound_caps: List[float] = []
    severed_inbound_caps: List[float] = []
    has_missing_edge_capacity = False

    for u, v, d in graph.in_edges(entity, data=True):
        c = d.get("capacity")
        c_val: Optional[float] = None
        if c is not None:
            try:
                c_val = float(c)
                normal_inbound_caps.append(c_val)
            except (ValueError, TypeError):
                has_missing_edge_capacity = True
        else:
            has_missing_edge_capacity = True

        # An incoming link is truly operational if:
        # 1. The edge exists in operational_graph
        # 2. Its upstream node u has an active path to a surviving original source
        is_link_operational = False
        if operational_graph.has_edge(u, v):
            try:
                u_active = (u in active_sources) or any(
                    nx.has_path(operational_graph, src, u)
                    for src in active_sources
                    if src != u
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                u_active = False

            if u_active:
                is_link_operational = True

        if is_link_operational:
            operational_inbound_count += 1
            if c_val is not None:
                operational_inbound_caps.append(c_val)
        else:
            if c_val is not None:
                severed_inbound_caps.append(c_val)

    severed_inbound_count = total_inbound - operational_inbound_count

    total_normal_inbound_cap = sum(normal_inbound_caps) if normal_inbound_caps else None
    remaining_operational_cap = sum(operational_inbound_caps) if operational_inbound_caps else (0.0 if normal_inbound_caps else None)
    lost_inbound_cap = sum(severed_inbound_caps) if severed_inbound_caps else (0.0 if normal_inbound_caps else None)
    capacity_deficit = lost_inbound_cap if lost_inbound_cap is not None else None

    # Capacity sufficiency check for alternative supply paths
    is_alternative_sufficient: Optional[bool] = None
    alternative_sufficiency: str = "no_alternative"

    if total_inbound == 0:
        status = "potentially_affected"
        alternative_sufficiency = "root_node_no_inbound"
        reason = "Node has no inbound suppliers in this network model."
    elif not has_active_source_path:
        if len(alternatives) > 0:
            status = "potentially_affected"
            alternative_sufficiency = "external_alternatives_unverified"
            is_alternative_sufficient = None
            reason = (
                f"All multi-tier supply paths from upstream sources are severed, but "
                f"{len(alternatives)} external backup alternative(s) are documented ({', '.join(alternatives)}). "
                "Sufficiency cannot be confirmed without real-time external capacity and lead-time contracts."
            )
        else:
            status = "confirmed_shortage"
            alternative_sufficiency = "no_operational_inbound"
            is_alternative_sufficient = False
            reason = (
                f"All active supply paths from upstream material sources are severed, "
                f"and no external backup suppliers are documented for '{entity}'."
            )
    else:
        # Active source path exists through remaining branches
        status = "potentially_affected"
        if has_missing_edge_capacity or total_normal_inbound_cap is None:
            is_alternative_sufficient = None
            alternative_sufficiency = "unverified_missing_capacity_data"
            reason = (
                f"Direct incoming link(s) affected, but active supply path from source(s) "
                f"({', '.join(active_sources_reaching)}) remains operational. "
                "However, alternative path cannot be confirmed as sufficient because inbound capacity/demand data is incomplete."
            )
        elif remaining_operational_cap is not None and total_normal_inbound_cap is not None:
            if remaining_operational_cap >= total_normal_inbound_cap:
                is_alternative_sufficient = True
                alternative_sufficiency = "sufficient_capacity"
                reason = (
                    f"Direct incoming link(s) affected, but operational alternative path(s) "
                    f"provide sufficient inbound capacity ({remaining_operational_cap:.1f} >= {total_normal_inbound_cap:.1f} units) "
                    "to cover normal throughput."
                )
            else:
                is_alternative_sufficient = False
                alternative_sufficiency = "insufficient_capacity_deficit"
                reason = (
                    f"Direct incoming link(s) affected. Alternative supply path from source(s) "
                    f"({', '.join(active_sources_reaching)}) is active, but remaining operational inbound capacity "
                    f"({remaining_operational_cap:.1f} units) is insufficient to replace lost capacity "
                    f"({lost_inbound_cap:.1f} units), leaving an estimated capacity deficit of {capacity_deficit:.1f} units."
                )
        else:
            is_alternative_sufficient = None
            alternative_sufficiency = "unverified_capacity"
            reason = (
                f"Alternative supply path from source(s) ({', '.join(active_sources_reaching)}) remains operational via alternative branches."
            )

    return {
        "entity": entity,
        "status": status,
        "total_inbound_links": total_inbound,
        "severed_inbound_links": severed_inbound_count,
        "operational_inbound_links": operational_inbound_count,
        "backup_alternatives_count": len(alternatives),
        "has_active_upstream_source_path": has_active_source_path,
        "is_alternative_sufficient": is_alternative_sufficient,
        "alternative_sufficiency": alternative_sufficiency,
        "remaining_inbound_capacity": remaining_operational_cap,
        "lost_inbound_capacity": lost_inbound_cap,
        "capacity_deficit": capacity_deficit,
        "explanation": reason,
    }


def _calculate_capacity_impact(
    graph: nx.DiGraph,
    failed_entity: Union[str, Tuple[str, str]],
    is_edge_disruption: bool,
    affected_entities: List[str],
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Calculates estimated capacity impact while strictly avoiding multi-tier double counting.

    Rules for Capacity Accounting:
    1. If a node fails, we calculate the capacity of its DIRECT outgoing relationships.
       We DO NOT add downstream tier capacities (e.g., adding processor cap + supplier cap + factory cap)
       because downstream capacity processes the SAME material flow and summing them would double/triple-count.
    2. If an edge fails, we calculate the specific capacity associated with that edge.
    3. If capacity attributes are missing, we flag them clearly and return None for exact numbers.
    4. Label all metrics as estimates based on supplied data.
    """
    missing_info: List[str] = []
    outgoing_capacities: List[float] = []
    edges_without_capacity: List[str] = []

    node_rated_capacity: Optional[float] = None
    direct_outgoing_capacity_sum: Optional[float] = None
    capacity_unit = "units/period (as defined in input dataset)"

    if is_edge_disruption:
        u, v = failed_entity  # type: ignore
        edge_data = graph.get_edge_data(u, v, default={})
        cap = edge_data.get("capacity")
        if cap is not None:
            try:
                val = float(cap)
                outgoing_capacities.append(val)
                direct_outgoing_capacity_sum = val
            except (ValueError, TypeError):
                missing_info.append(f"Invalid capacity format '{cap}' on route ({u} -> {v})")
        else:
            missing_info.append(f"Missing capacity attribute on disrupted transport route ({u} -> {v})")
    else:
        # Node failure
        node_id = str(failed_entity)
        node_data = graph.nodes.get(node_id, {})
        rated = node_data.get("capacity")
        if rated is not None:
            try:
                node_rated_capacity = float(rated)
            except (ValueError, TypeError):
                missing_info.append(f"Invalid rated capacity format '{rated}' on node '{node_id}'")
        else:
            missing_info.append(f"Missing rated capacity attribute on failed node '{node_id}'")

        # Check outgoing edge capacities
        out_edges = list(graph.out_edges(node_id, data=True))
        if out_edges:
            for _, target, data in out_edges:
                edge_cap = data.get("capacity")
                if edge_cap is not None:
                    try:
                        outgoing_capacities.append(float(edge_cap))
                    except (ValueError, TypeError):
                        edges_without_capacity.append(target)
                else:
                    edges_without_capacity.append(target)

            if edges_without_capacity:
                missing_info.append(
                    f"Missing outgoing edge capacity data to: {', '.join(edges_without_capacity)}"
                )

            if outgoing_capacities:
                direct_outgoing_capacity_sum = sum(outgoing_capacities)

    has_complete_data = (len(missing_info) == 0) and (
        direct_outgoing_capacity_sum is not None or node_rated_capacity is not None
    )

    # Best single direct loss estimate
    estimated_direct_loss: Optional[float] = None
    if direct_outgoing_capacity_sum is not None:
        estimated_direct_loss = direct_outgoing_capacity_sum
    elif node_rated_capacity is not None:
        estimated_direct_loss = node_rated_capacity

    capacity_impact_record = {
        "estimated_direct_capacity_loss": estimated_direct_loss,
        "failed_node_rated_capacity": node_rated_capacity,
        "direct_outgoing_edge_capacity_sum": direct_outgoing_capacity_sum,
        "unit": capacity_unit,
        "has_complete_capacity_data": has_complete_data,
        "accounting_method": (
            "Direct tier-1 outgoing capacity only. Downstream multi-tier capacities are NOT summed "
            "to prevent double-counting identical material throughput across successive stages."
        ),
        "disclaimer": (
            "Heuristic simulated estimate based strictly on input graph attributes. "
            "Does not represent a confirmed real-world forecast."
        ),
    }

    return capacity_impact_record, missing_info


def _determine_impact_level(
    affected_entities_count: int,
    confirmed_shortages_count: int,
    is_shared_chokepoint: bool,
    has_capacity_loss: bool,
    direct_loss_val: Optional[float],
) -> Tuple[str, str]:
    """
    Transparent rule-based heuristic to assign an Impact Level.

    Heuristic Rules:
    - Critical: Confirmed shortages >= 2, or shared chokepoint affecting >= 3 entities,
      or very large direct capacity loss.
    - High: Confirmed shortages == 1, or shared chokepoint affecting 2 entities,
      or >= 3 potentially affected entities.
    - Medium: 1 or 2 potentially affected entities with operational alternatives.
    - Low: 0 downstream entities affected (terminal node) or single isolated link with alternatives.

    Note: These rules are explicit and heuristic; they are not scientifically certified predictions.
    """
    reasons: List[str] = []

    if confirmed_shortages_count >= 2 or (is_shared_chokepoint and affected_entities_count >= 3):
        level = "Critical"
        reasons.append(
            f"Severe multi-entity fallout: {confirmed_shortages_count} confirmed shortage(s) "
            f"across {affected_entities_count} affected downstream nodes"
        )
    elif confirmed_shortages_count == 1 or affected_entities_count >= 3 or is_shared_chokepoint:
        level = "High"
        reasons.append(
            f"Significant disruption: {confirmed_shortages_count} confirmed shortage(s), "
            f"{affected_entities_count} total affected entities"
        )
    elif affected_entities_count >= 1:
        level = "Medium"
        reasons.append(
            f"Moderate reach: {affected_entities_count} downstream entity(s) potentially affected "
            "with operational fallback paths"
        )
    else:
        level = "Low"
        reasons.append("Localized leaf impact: 0 downstream entities affected")

    if is_shared_chokepoint:
        reasons.append("Failed entity is a shared choke-point supplying multiple downstream entities")

    if direct_loss_val is not None and direct_loss_val > 0:
        reasons.append(f"Estimated direct capacity offline: {direct_loss_val:,.1f} units")

    return level, "; ".join(reasons)


def simulate_disruption(
    graph: nx.DiGraph,
    failed_node: Optional[Union[str, Tuple[str, str]]] = None,
    disruption_type: Optional[str] = None,
    failed_edge: Optional[Tuple[str, str]] = None,
    failed_route: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Simulates the multi-tier cascading disruption when an entity or transport route fails.

    Parameters
    ----------
    graph : nx.DiGraph
        Directed supply-chain graph where edges represent flow from upstream to downstream.
    failed_node : Optional[Union[str, Tuple[str, str]]], default=None
        Identifier of the failed supply chain entity (node name) or edge tuple (u, v).
    disruption_type : Optional[str], default=None
        Type of disruption (e.g. 'Supplier failure', 'Processor failure',
        'Factory failure', 'Transport route failure'). If None, automatically inferred.
    failed_edge : Optional[Tuple[str, str]], default=None
        Alternative explicit parameter to specify an edge disruption (u, v).
    failed_route : Optional[str], default=None
        Route identifier string matching an edge 'route' attribute.

    Returns
    -------
    Dict[str, Any]
        Structured simulation results containing:
        - failed_entity: Name or identifier of failed entity
        - disruption_type: Standardized disruption type
        - affected_entities: List of unique downstream entity names (no duplicates)
        - affected_paths: List of disrupted directed paths
        - affected_materials: List of unique material names impacted
        - estimated_capacity_impact: Dictionary detailing capacity loss and accounting method
        - impact_level: 'Low', 'Medium', 'High', or 'Critical'
        - explanation: Plain-English summary of causal disruption mechanisms
        - missing_information: List of data gaps preventing full estimation
        - assumptions: List of explicit modeling assumptions
        - recommended_next_actions: Actionable next steps for operations teams
        - entity_impact_details: Granular status per entity (potentially affected vs confirmed shortage)

    Raises
    ------
    TypeError
        If graph is not a networkx.DiGraph or is None.
    ValueError
        If inputs are missing, graph is empty, or failed entity is not found.
    """
    # 1. Input Validation
    if graph is None:
        raise TypeError("Graph cannot be None. Expected a networkx.DiGraph instance.")

    if not isinstance(graph, nx.DiGraph):
        raise TypeError(
            f"Invalid graph type: {type(graph).__name__}. Expected networkx.DiGraph."
        )

    if graph.number_of_nodes() == 0:
        raise ValueError("The provided supply chain graph is empty (0 nodes).")

    # Resolve target entity (node, edge, or route)
    target_edge: Optional[Tuple[str, str]] = None
    target_node: Optional[str] = None

    if failed_route is not None:
        # Search edge with route attribute
        found_edge = None
        for u, v, data in graph.edges(data=True):
            if data.get("route") == failed_route:
                found_edge = (u, v)
                break
        if not found_edge:
            raise ValueError(f"No edge found with route identifier '{failed_route}'.")
        target_edge = found_edge
    elif failed_edge is not None:
        if not isinstance(failed_edge, (tuple, list)) or len(failed_edge) != 2:
            raise TypeError("failed_edge must be a 2-tuple of (source_node, target_node).")
        target_edge = (str(failed_edge[0]), str(failed_edge[1]))
    elif isinstance(failed_node, (tuple, list)) and len(failed_node) == 2:
        target_edge = (str(failed_node[0]), str(failed_node[1]))
    elif failed_node is not None:
        target_node = str(failed_node)
    else:
        raise ValueError("Must specify either failed_node, failed_edge, or failed_route.")

    # Validate existence
    if target_edge is not None:
        u, v = target_edge
        if not graph.has_edge(u, v):
            raise ValueError(f"Disrupted edge ({u} -> {v}) does not exist in the graph.")
        failed_entity_repr = f"Route ({u} -> {v})"
        is_edge_disruption = True
    else:
        assert target_node is not None
        if target_node not in graph:
            raise ValueError(f"Entity '{target_node}' not found in the supply chain graph.")
        failed_entity_repr = target_node
        is_edge_disruption = False

    # 2. Determine Disruption Type
    effective_disruption_type = _infer_disruption_type(
        graph,
        failed_entity=target_edge if is_edge_disruption else target_node,  # type: ignore
        user_type=disruption_type,
    )

    # 3. Identify Affected Downstream Entities and Paths
    failed_nodes_set: Set[str] = set()
    failed_edges_set: Set[Tuple[str, str]] = set()

    if is_edge_disruption:
        assert target_edge is not None
        u, v = target_edge
        failed_edges_set.add((u, v))
        # Downstream entities starting from v
        downstream_entities_set: Set[str] = {v} | nx.descendants(graph, v)
        # Find paths starting with the broken edge
        raw_paths = _find_disrupted_paths(graph, [v])
        affected_paths = [[u] + p for p in raw_paths]
    else:
        assert target_node is not None
        failed_nodes_set.add(target_node)
        downstream_entities_set = nx.descendants(graph, target_node)
        affected_paths = _find_disrupted_paths(graph, [target_node])

    # De-duplicate affected entities in deterministic order
    affected_entities: List[str] = sorted(list(downstream_entities_set))

    # 4. Identify Affected Materials and Routes
    affected_materials_set: Set[str] = set()
    missing_material_warnings: List[str] = []

    # Check edges along disrupted paths
    for path in affected_paths:
        for i in range(len(path) - 1):
            edge_u, edge_v = path[i], path[i + 1]
            edge_data = graph.get_edge_data(edge_u, edge_v, default={})
            mat = edge_data.get("material") or edge_data.get("commodity")
            if mat:
                affected_materials_set.add(str(mat))
            else:
                missing_material_warnings.append(f"({edge_u} -> {edge_v})")

    # Also check node-level materials if present
    if not is_edge_disruption:
        node_mat = graph.nodes[target_node].get("material")
        if node_mat:
            affected_materials_set.add(str(node_mat))

    affected_materials: List[str] = sorted(list(affected_materials_set))

    # 5. Distinguish Potentially Affected vs Confirmed Shortages
    entity_evaluations: List[Dict[str, Any]] = []
    confirmed_shortages: List[str] = []
    potentially_affected: List[str] = []

    for ent in affected_entities:
        eval_res = _evaluate_entity_shortage(graph, ent, failed_nodes_set, failed_edges_set)
        entity_evaluations.append(eval_res)
        if eval_res["status"] == "confirmed_shortage":
            confirmed_shortages.append(ent)
        else:
            potentially_affected.append(ent)

    # 6. Calculate Capacity Impact (Without Multi-Tier Double Counting)
    capacity_impact, capacity_missing_info = _calculate_capacity_impact(
        graph=graph,
        failed_entity=target_edge if is_edge_disruption else target_node,  # type: ignore
        is_edge_disruption=is_edge_disruption,
        affected_entities=affected_entities,
    )

    # 7. Collect Missing Information
    all_missing_info: List[str] = []
    all_missing_info.extend(capacity_missing_info)

    if missing_material_warnings:
        # Deduplicate warnings
        unique_edges = sorted(list(set(missing_material_warnings)))
        all_missing_info.append(
            f"Missing material specifications on edge relationships: {', '.join(unique_edges)}"
        )

    # Check for missing lead times or inventory data
    for ent in affected_entities:
        ndata = graph.nodes.get(ent, {})
        if ndata.get("lead_time_days") is None and ndata.get("lead_time") is None:
            all_missing_info.append(f"Missing lead time data for affected node '{ent}'")
        if ndata.get("buffer_stock") is None and ndata.get("inventory") is None:
            all_missing_info.append(f"Missing safety inventory buffer data for affected node '{ent}'")

    # Deduplicate missing info
    seen_missing: Set[str] = set()
    clean_missing_info: List[str] = []
    for item in all_missing_info:
        if item not in seen_missing:
            seen_missing.add(item)
            clean_missing_info.append(item)

    # 8. Check Shared Choke-point Status
    # An entity is a shared choke-point if direct outgoing degree >= 2
    if is_edge_disruption:
        is_shared_chokepoint = False
    else:
        out_degree = graph.out_degree(target_node)
        is_shared_chokepoint = out_degree >= 2

    # 9. Compute Transparent Impact Level
    impact_level, level_reason = _determine_impact_level(
        affected_entities_count=len(affected_entities),
        confirmed_shortages_count=len(confirmed_shortages),
        is_shared_chokepoint=is_shared_chokepoint,
        has_capacity_loss=capacity_impact["estimated_direct_capacity_loss"] is not None,
        direct_loss_val=capacity_impact["estimated_direct_capacity_loss"],
    )

    # 10. Synthesize Plain-English Explanation
    explanation_parts: List[str] = []
    explanation_parts.append(
        f"Simulated failure of {failed_entity_repr} ({effective_disruption_type})."
    )

    if not affected_entities:
        explanation_parts.append(
            "This entity has no downstream dependencies in the current network model; no downstream nodes are impacted."
        )
    else:
        explanation_parts.append(
            f"A total of {len(affected_entities)} downstream entity(s) are in the blast path "
            f"across {len(affected_paths)} directed supply path(s)."
        )
        if confirmed_shortages:
            explanation_parts.append(
                f"Confirmed severe shortage risk detected for {len(confirmed_shortages)} node(s) "
                f"({', '.join(confirmed_shortages)}) due to complete lack of operational inbound suppliers."
            )
        if potentially_affected:
            explanation_parts.append(
                f"{len(potentially_affected)} node(s) ({', '.join(potentially_affected)}) are classified as "
                "potentially affected because alternative inbound paths or backup suppliers may soften the impact."
            )

    explanation_parts.append(f"Impact Level assessed as {impact_level} ({level_reason}).")

    if capacity_impact["estimated_direct_capacity_loss"] is not None:
        explanation_parts.append(
            f"Estimated direct capacity loss is {capacity_impact['estimated_direct_capacity_loss']:,.1f} {capacity_impact['unit']}."
        )
    else:
        explanation_parts.append(
            "Direct capacity loss could not be quantified due to missing capacity attributes."
        )

    explanation = " ".join(explanation_parts)

    # 11. Explicit Assumptions
    assumptions = [
        "Simulated disruption assumes 100% capacity curtailment of the failed entity or route.",
        "Downstream entities without specified buffer stock are assumed to have zero safety inventory.",
        "Static graph topology: dynamic supplier switching is not simulated unless alternative suppliers are listed.",
        "Capacity impact strictly measures direct outgoing flow to avoid double-counting multi-tier throughput.",
        "Impact scores and levels are rule-based heuristics and do not represent guaranteed real-world outcomes.",
    ]

    # 12. Recommended Next Actions
    recommended_next_actions: List[str] = []
    if confirmed_shortages:
        for cs in confirmed_shortages:
            recommended_next_actions.append(
                f"Emergency sourcing: Expedite qualification of backup suppliers for critical node '{cs}'"
            )
    if is_edge_disruption:
        assert target_edge is not None
        recommended_next_actions.append(
            f"Logistics re-routing: Activate secondary freight corridors bypassing route ({target_edge[0]} -> {target_edge[1]})"
        )
    if affected_materials:
        recommended_next_actions.append(
            f"Inventory audit: Review strategic reserves for affected materials: {', '.join(affected_materials)}"
        )
    if is_shared_chokepoint:
        recommended_next_actions.append(
            f"Mitigate single point of failure: Node '{target_node}' feeds multiple tiers; split allocation across secondary vendors"
        )
    if clean_missing_info:
        recommended_next_actions.append(
            f"Data enrichment: Collect missing operational parameters ({len(clean_missing_info)} gaps identified)"
        )
    if not recommended_next_actions:
        recommended_next_actions.append(
            "Maintain standard monitoring; no immediate downstream mitigation required."
        )

    # Return structured dictionary
    return {
        "failed_entity": failed_entity_repr,
        "disruption_type": effective_disruption_type,
        "affected_entities": affected_entities,
        "affected_paths": affected_paths,
        "affected_materials": affected_materials,
        "estimated_capacity_impact": capacity_impact,
        "impact_level": impact_level,
        "explanation": explanation,
        "missing_information": clean_missing_info,
        "assumptions": assumptions,
        "recommended_next_actions": recommended_next_actions,
        "entity_impact_details": entity_evaluations,
    }


class DisruptionSimulationAgent:
    """
    Object-oriented agent wrapper for the Disruption Simulation module.
    Provides convenience methods for running multiple scenarios and exporting
    results for Streamlit or Pandas tabular analysis.
    """

    def __init__(self, graph: Optional[nx.DiGraph] = None):
        self.graph = graph

    def load_graph(self, graph: nx.DiGraph) -> None:
        """Loads or updates the active supply chain graph."""
        if not isinstance(graph, nx.DiGraph):
            raise TypeError("Expected a networkx.DiGraph instance.")
        self.graph = graph

    def simulate(
        self,
        failed_node: Optional[Union[str, Tuple[str, str]]] = None,
        disruption_type: Optional[str] = None,
        failed_edge: Optional[Tuple[str, str]] = None,
        failed_route: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs the disruption simulation on the loaded graph."""
        if self.graph is None:
            raise ValueError("No graph loaded. Provide a graph or call load_graph() first.")
        return simulate_disruption(
            graph=self.graph,
            failed_node=failed_node,
            disruption_type=disruption_type,
            failed_edge=failed_edge,
            failed_route=failed_route,
        )

    def simulate_all_nodes(self) -> pd.DataFrame:
        """
        Simulates individual failure of every node in the graph and returns
        a consolidated comparison DataFrame for Streamlit display.
        """
        if self.graph is None:
            raise ValueError("No graph loaded.")

        records = []
        for node in self.graph.nodes:
            res = self.simulate(failed_node=node)
            records.append({
                "Entity": res["failed_entity"],
                "Disruption Type": res["disruption_type"],
                "Impact Level": res["impact_level"],
                "Affected Count": len(res["affected_entities"]),
                "Affected Entities": ", ".join(res["affected_entities"]) or "None",
                "Direct Capacity Loss": res["estimated_capacity_impact"]["estimated_direct_capacity_loss"],
                "Missing Data Count": len(res["missing_information"]),
            })

        return pd.DataFrame(records)

    def get_entity_impact_dataframe(self, simulation_result: Dict[str, Any]) -> pd.DataFrame:
        """
        Converts granular entity impact details from a simulation result into
        a formatted pandas DataFrame for Streamlit UI display (Member 5).
        """
        details = simulation_result.get("entity_impact_details", [])
        if not details:
            return pd.DataFrame(columns=[
                "Entity", "Status", "Operational Inbound", "Severed Inbound",
                "Capacity Deficit", "Alternative Sufficiency", "Explanation"
            ])

        records = []
        for d in details:
            records.append({
                "Entity": d.get("entity"),
                "Status": d.get("status"),
                "Operational Inbound": d.get("operational_inbound_links"),
                "Severed Inbound": d.get("severed_inbound_links"),
                "Capacity Deficit": d.get("capacity_deficit"),
                "Alternative Sufficiency": d.get("alternative_sufficiency"),
                "Explanation": d.get("explanation"),
            })
        return pd.DataFrame(records)

    def get_financial_payload(self, simulation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience method to format simulation output for Member 4's Financial Engine."""
        return get_financial_engine_payload(simulation_result)


def get_financial_engine_payload(simulation_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms disruption simulation output into the standardized input structure
    required by Member 4's Financial Loss & Investment Optimization Engine.

    Parameters
    ----------
    simulation_result : Dict[str, Any]
        Output dictionary from simulate_disruption().

    Returns
    -------
    Dict[str, Any]
        Standardized payload containing:
        - failed_entity: Name/ID of disrupted asset
        - disruption_type: Category of disruption
        - impact_level: Heuristic severity ('Low', 'Medium', 'High', 'Critical')
        - direct_capacity_loss: Direct rated/outgoing capacity offline
        - affected_entities: All downstream impacted entity names
        - confirmed_shortage_entities: Entities with 0 operational inbound source paths
        - potential_shortage_entities: Entities with operational alternative paths
        - capacity_deficit_by_entity: Mapping of downstream entity to capacity deficit
        - total_downstream_capacity_deficit: Sum of quantified downstream capacity deficits
        - affected_materials: Disrupted products/commodities
        - missing_data_warnings: Unresolved data gaps
        - assumptions: Modeling assumptions for loss calculation
    """
    details = simulation_result.get("entity_impact_details", [])
    confirmed = [d["entity"] for d in details if d.get("status") == "confirmed_shortage"]
    potential = [d["entity"] for d in details if d.get("status") == "potentially_affected"]

    deficits: Dict[str, Optional[float]] = {}
    total_deficit = 0.0
    for d in details:
        ent = d.get("entity", "")
        def_val = d.get("capacity_deficit")
        deficits[ent] = def_val
        if def_val is not None:
            total_deficit += def_val

    cap_impact = simulation_result.get("estimated_capacity_impact", {})

    return {
        "failed_entity": simulation_result.get("failed_entity"),
        "disruption_type": simulation_result.get("disruption_type"),
        "impact_level": simulation_result.get("impact_level"),
        "direct_capacity_loss": cap_impact.get("estimated_direct_capacity_loss"),
        "affected_entities": simulation_result.get("affected_entities", []),
        "confirmed_shortage_entities": confirmed,
        "potential_shortage_entities": potential,
        "capacity_deficit_by_entity": deficits,
        "total_downstream_capacity_deficit": total_deficit,
        "affected_materials": simulation_result.get("affected_materials", []),
        "missing_data_warnings": simulation_result.get("missing_information", []),
        "assumptions": simulation_result.get("assumptions", []),
    }


def create_synthetic_supply_chain() -> nx.DiGraph:
    r"""
    Creates a canonical synthetic multi-tier supply chain network for testing,
    demonstration, and integration with Member 1 (Mapping) & Member 2 (Risk Detection).

    Topology:
        Raw Material P (Mine)
              | [Silicon Ingot, cap=200, Route-North-1]
              v
        Processor X (Refiner, cap=200)
             / \
   [Wafer,  /   \  [Wafer,
   cap=100,/     \  cap=100,
   Route-1]       \ Route-2]
         v         v
    Supplier A    Supplier B
         |             |
   [ECU, |             | [Display,
  cap=100|             |  cap=100]
         v             v
      Factory 1     Factory 2
                       ^
                       | [Sub-assembly, cap=50]
                   Supplier C (Alternative source)
                       ^
                       |
                  Processor Y
                       ^
                       |
                 Raw Material Q

    Node Attributes:
    - node_type: 'raw_material', 'processor', 'supplier', 'factory'
    - capacity: Numeric rated capacity
    - lead_time_days: Integer lead time
    - alternatives: List of backup external alternatives (if known)

    Edge Attributes:
    - material: Name of physical product/material
    - capacity: Flow capacity on this transport corridor
    - route: Route identifier (e.g., 'Route-North-1')
    - lead_time_days: Transit lead time in days
    """
    G = nx.DiGraph()

    # Nodes
    G.add_node(
        "Raw Material P",
        node_type="raw_material",
        capacity=250.0,
        lead_time_days=60,
        alternatives=["Mine Alpha", "Mine Beta"],
    )
    G.add_node(
        "Raw Material Q",
        node_type="raw_material",
        capacity=150.0,
        lead_time_days=40,
        alternatives=[],
    )

    G.add_node(
        "Processor X",
        node_type="processor",
        capacity=200.0,
        lead_time_days=45,
        alternatives=[],
    )
    G.add_node(
        "Processor Y",
        node_type="processor",
        capacity=100.0,
        lead_time_days=20,
        alternatives=["Processor Z"],
    )

    G.add_node(
        "Supplier A",
        node_type="supplier",
        is_supplier=True,
        capacity=120.0,
        lead_time_days=25,
        alternatives=[],
    )
    G.add_node(
        "Supplier B",
        node_type="supplier",
        is_supplier=True,
        capacity=110.0,
        lead_time_days=30,
        alternatives=[],
    )
    G.add_node(
        "Supplier C",
        node_type="supplier",
        is_supplier=True,
        capacity=60.0,
        lead_time_days=15,
        alternatives=[],
    )

    G.add_node(
        "Factory 1",
        node_type="factory",
        capacity=100.0,
        lead_time_days=5,
        alternatives=[],
    )
    G.add_node(
        "Factory 2",
        node_type="factory",
        capacity=150.0,
        lead_time_days=7,
        alternatives=[],
    )

    # Directed Edges (Upstream -> Downstream)
    G.add_edge(
        "Raw Material P",
        "Processor X",
        material="Silicon Ingot",
        capacity=200.0,
        route="Route-RM-P",
        lead_time_days=14,
    )
    G.add_edge(
        "Raw Material Q",
        "Processor Y",
        material="Copper Cathode",
        capacity=100.0,
        route="Route-RM-Q",
        lead_time_days=10,
    )

    G.add_edge(
        "Processor X",
        "Supplier A",
        material="Semiconductor Wafers",
        capacity=100.0,
        route="Route-PX-SA",
        lead_time_days=7,
    )
    G.add_edge(
        "Processor X",
        "Supplier B",
        material="Semiconductor Wafers",
        capacity=100.0,
        route="Route-PX-SB",
        lead_time_days=8,
    )
    G.add_edge(
        "Processor Y",
        "Supplier C",
        material="Wiring Harness Components",
        capacity=60.0,
        route="Route-PY-SC",
        lead_time_days=5,
    )

    G.add_edge(
        "Supplier A",
        "Factory 1",
        material="Engine Control Units",
        capacity=100.0,
        route="Route-SA-F1",
        lead_time_days=3,
    )
    G.add_edge(
        "Supplier B",
        "Factory 2",
        material="Digital Cockpit Displays",
        capacity=100.0,
        route="Route-SB-F2",
        lead_time_days=4,
    )
    G.add_edge(
        "Supplier C",
        "Factory 2",
        material="Secondary Wire Harnesses",
        capacity=50.0,
        route="Route-SC-F2",
        lead_time_days=2,
    )

    return G
