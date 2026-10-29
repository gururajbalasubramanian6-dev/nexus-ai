"""
Nexus-AI: Shared Risk Detection Agent (Member 2)
=================================================

This module detects single points of failure and hidden shared dependencies
across multi-tier supply chains.

Graph Convention:
- Edges represent flow from upstream to downstream:
    Upstream Node  --->  Downstream Node
    (e.g., Processor X ---> Supplier A)

Author: Nexus-AI Team (Member 2)
"""

from typing import Any, Dict, List, Optional, Set, Union
import networkx as nx
import pandas as pd


def detect_shared_risks(
    graph: nx.DiGraph,
    target_suppliers: Optional[List[str]] = None,
    only_shared: bool = True,
    as_dataframe: bool = False,
) -> Union[List[Dict[str, Any]], pd.DataFrame]:
    """
    Identifies upstream dependencies that supply or support downstream suppliers,
    analyzes multi-tier reachability, flags missing data, and calculates an explainable risk score.

    Parameters
    ----------
    graph : nx.DiGraph
        A NetworkX directed graph where edges point from upstream entities
        to downstream entities (Upstream -> Downstream).
    target_suppliers : Optional[List[str]], default=None
        List of downstream supplier node identifiers to analyze.
        If None, target suppliers are automatically inferred:
        1. Nodes marked with attribute `node_type='supplier'` or `is_supplier=True`.
        2. If no such attributes exist, all sink nodes (out-degree == 0).
    only_shared : bool, default=True
        If True, only returns dependencies affecting 2 or more distinct suppliers.
        If False, returns findings for all upstream dependencies (including single-supplier dependencies).
    as_dataframe : bool, default=False
        If True, returns results formatted as a pandas DataFrame.
        If False, returns a list of dictionaries.

    Returns
    -------
    Union[List[Dict[str, Any]], pd.DataFrame]
        Structured risk analysis findings. Each record contains:
        - dependency_node: Name of upstream dependency
        - affected_suppliers: Unique list of downstream suppliers affected
        - affected_supplier_count: Number of unique suppliers affected
        - dependency_tier: Minimum hop distance to an affected supplier (e.g. 1 = Direct Tier-1, 2 = Tier-2)
        - is_shared: True if affected_supplier_count >= 2
        - direct_affected_suppliers: Direct downstream connections (hop = 1)
        - indirect_affected_suppliers: Multi-tier downstream connections (hop > 1)
        - risk_score: Explainable vulnerability score (0 to 100)
        - risk_level: Categorical level ("Low", "Medium", "High", "Critical")
        - explanation: Transparent textual breakdown of the score
        - missing_information: List of data gaps identified for this node
        - recommended_investigation: Recommended proactive action

    Raises
    ------
    TypeError
        If graph is None or not an instance of nx.DiGraph.
    ValueError
        If the graph is empty or contains no nodes.
    """
    # 1. Input Validation
    if graph is None:
        raise TypeError("Graph cannot be None. Expected a networkx.DiGraph instance.")

    if not isinstance(graph, nx.DiGraph):
        raise TypeError(
            f"Invalid graph type: {type(graph).__name__}. "
            "Graph must be an instance of networkx.DiGraph with directed edges (Upstream -> Downstream)."
        )

    if graph.number_of_nodes() == 0:
        if as_dataframe:
            return pd.DataFrame(columns=[
                "dependency_node", "affected_suppliers", "affected_supplier_count",
                "dependency_tier", "is_shared", "direct_affected_suppliers",
                "indirect_affected_suppliers", "risk_score", "risk_level",
                "explanation", "missing_information", "recommended_investigation"
            ])
        return []

    # 2. Identify Target Downstream Suppliers
    if target_suppliers is not None:
        if not isinstance(target_suppliers, (list, set, tuple)):
            raise TypeError("target_suppliers must be a list, set, or tuple of node names.")
        downstream_targets = set(target_suppliers)
        # Validate that target suppliers actually exist in the graph
        missing_targets = [t for t in downstream_targets if t not in graph]
        if missing_targets:
            raise ValueError(f"Specified target suppliers not found in graph: {missing_targets}")
    else:
        # Auto-detect downstream targets
        explicit_suppliers = {
            n for n, attr in graph.nodes(data=True)
            if attr.get("is_supplier") is True or str(attr.get("node_type", "")).lower() in ("supplier", "tier_1", "tier-1")
        }
        if explicit_suppliers:
            downstream_targets = explicit_suppliers
        else:
            # Fall back to leaf / sink nodes (nodes with no outgoing edges)
            downstream_targets = {n for n in graph.nodes() if graph.out_degree(n) == 0}

    findings: List[Dict[str, Any]] = []

    # 3. Analyze each node in the graph
    for candidate_node in graph.nodes():
        # A node does not count as its own downstream dependency
        # We find all downstream nodes reachable from candidate_node
        reachable_nodes = nx.descendants(graph, candidate_node)
        
        # Intersect with the target downstream suppliers to avoid counting intermediate nodes
        # Using a set ensures no supplier is double-counted even if multiple paths exist
        affected_suppliers_set = reachable_nodes.intersection(downstream_targets)

        # Exclude candidate_node itself if it is in downstream_targets
        affected_suppliers_set.discard(candidate_node)

        affected_count = len(affected_suppliers_set)

        # If filtering for shared dependencies, skip nodes affecting fewer than 2 suppliers
        if only_shared and affected_count < 2:
            continue
        
        # If not only_shared, only skip nodes that affect 0 suppliers
        if not only_shared and affected_count == 0:
            continue

        # Distinguish direct vs indirect dependencies
        direct_successors = set(graph.successors(candidate_node))
        direct_affected = sorted(list(affected_suppliers_set.intersection(direct_successors)))
        indirect_affected = sorted(list(affected_suppliers_set.difference(direct_successors)))

        # Determine dependency tier (minimum path length to any affected supplier)
        min_hops = min(nx.shortest_path_length(graph, candidate_node, s) for s in affected_suppliers_set)
        if min_hops == 1:
            tier_label = "Tier 1 (Direct)"
        elif min_hops == 2:
            tier_label = "Tier 2 (Indirect)"
        else:
            tier_label = f"Tier {min_hops} (Deep Indirect)"

        # Inspect node attributes for missing data & operational risk
        node_data = graph.nodes[candidate_node]
        missing_info, operational_meta = _audit_node_data(candidate_node, node_data)

        # Calculate explainable risk score
        risk_score, risk_level, explanation, recommendation = calculate_risk_score(
            node_name=candidate_node,
            affected_supplier_count=affected_count,
            min_hops=min_hops,
            operational_meta=operational_meta,
            missing_info=missing_info,
        )

        finding = {
            "dependency_node": candidate_node,
            "affected_suppliers": sorted(list(affected_suppliers_set)),
            "affected_supplier_count": affected_count,
            "dependency_tier": tier_label,
            "is_shared": affected_count >= 2,
            "direct_affected_suppliers": direct_affected,
            "indirect_affected_suppliers": indirect_affected,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "explanation": explanation,
            "missing_information": missing_info,
            "recommended_investigation": recommendation,
        }
        findings.append(finding)

    # Sort findings by risk score descending (highest risk first)
    findings.sort(key=lambda x: x["risk_score"], reverse=True)

    if as_dataframe:
        return pd.DataFrame(findings)

    return findings


def _audit_node_data(node_name: str, node_data: Dict[str, Any]) -> tuple[List[str], Dict[str, Any]]:
    """
    Audits a node's metadata dictionary for missing or unknown fields.
    Does NOT invent values.
    """
    missing: List[str] = []
    meta: Dict[str, Any] = {
        "alternatives_count": None,
        "capacity_utilization": None,
        "lead_time_days": None,
        "single_source": False,
    }

    # 1. Alternative Suppliers
    # Can be provided as list `alternatives`, `alternative_suppliers`, or integer count
    alts = node_data.get("alternatives") or node_data.get("alternative_suppliers")
    if alts is None:
        missing.append("Missing alternative-supplier information")
    elif isinstance(alts, (list, set, tuple)):
        meta["alternatives_count"] = len(alts)
        meta["single_source"] = len(alts) == 0
    elif isinstance(alts, (int, float)):
        meta["alternatives_count"] = int(alts)
        meta["single_source"] = int(alts) == 0
    else:
        # String like "unknown"
        if str(alts).strip().lower() in ("unknown", "none", "n/a", ""):
            missing.append("Alternative-supplier data is marked unknown")
        else:
            meta["alternatives_count"] = 1

    # 2. Capacity Utilization
    # Can be a float (e.g., 0.85 or 85.0)
    cap = node_data.get("capacity") or node_data.get("capacity_utilization")
    if cap is None:
        missing.append("Missing capacity utilization data")
    else:
        try:
            cap_val = float(cap)
            # Normalize percentage if passed as 0-100
            if cap_val > 1.0:
                cap_val = cap_val / 100.0
            meta["capacity_utilization"] = cap_val
        except (ValueError, TypeError):
            missing.append("Capacity utilization is non-numeric or invalid")

    # 3. Lead Time (days)
    lead = node_data.get("lead_time_days") or node_data.get("lead_time")
    if lead is None:
        missing.append("Missing lead-time data")
    else:
        try:
            meta["lead_time_days"] = float(lead)
        except (ValueError, TypeError):
            missing.append("Lead-time data is non-numeric or invalid")

    return missing, meta


def calculate_risk_score(
    node_name: str,
    affected_supplier_count: int,
    min_hops: int,
    operational_meta: Dict[str, Any],
    missing_info: List[str],
) -> tuple[float, str, str, str]:
    """
    Calculates an explainable rule-based risk score (0 to 100).

    IMPORTANT NOTE:
    This score is a transparent heuristic vulnerability index, NOT a scientifically
    validated probability of operational failure. It is designed to prioritize
    investigation and resilience measures.

    Scoring Logic:
    1. Base Impact (Breadth of blast radius):
       - 1 supplier: 15 pts
       - 2 suppliers: 35 pts
       - 3 suppliers: 50 pts
       - 4+ suppliers: 65 pts
    2. Alternative Sourcing (Fungibility):
       - Single source (0 alternatives): +20 pts
       - 1 alternative known: +5 pts
       - 2+ alternatives verified: -10 pts (reduces vulnerability)
       - Missing alternatives data: +15 pts (uncertainty penalty)
    3. Operational Bottlenecks:
       - High capacity (> 85%): +15 pts
       - Moderate capacity (70-85%): +5 pts
       - Long lead time (> 45 days): +10 pts
    4. Data Gap Penalty:
       - +10 pts per missing critical attribute (capacity, lead time)
    5. Tier Visibility Risk:
       - Multi-tier dependency (hops >= 2): +5 pts (harder to monitor)

    Returns
    -------
    (score, risk_level, explanation, recommended_investigation)
    """
    score = 0.0
    reasons: List[str] = []

    # 1. Blast Radius
    if affected_supplier_count == 1:
        score += 15.0
        reasons.append("Affects 1 supplier (+15)")
    elif affected_supplier_count == 2:
        score += 35.0
        reasons.append("Shared bottleneck affecting 2 suppliers (+35)")
    elif affected_supplier_count == 3:
        score += 50.0
        reasons.append("Shared critical bottleneck affecting 3 suppliers (+50)")
    else:
        score += 65.0
        reasons.append(f"Severe hub dependency affecting {affected_supplier_count} suppliers (+65)")

    # 2. Alternative Sources
    alts = operational_meta.get("alternatives_count")
    if alts is None:
        score += 15.0
        reasons.append("Alternative supplier availability is unknown (+15 uncertainty penalty)")
    elif alts == 0:
        score += 20.0
        reasons.append("Sole source dependency with no known alternatives (+20)")
    elif alts == 1:
        score += 5.0
        reasons.append("Only 1 known alternative (+5)")
    else:
        score -= 10.0
        reasons.append(f"{alts} alternatives available (-10 buffer)")

    # 3. Capacity
    cap = operational_meta.get("capacity_utilization")
    if cap is not None:
        if cap >= 0.85:
            score += 15.0
            reasons.append(f"High capacity utilization at {cap*100:.0f}% (+15 bottleneck risk)")
        elif cap >= 0.70:
            score += 5.0
            reasons.append(f"Moderate capacity utilization at {cap*100:.0f}% (+5)")
    else:
        score += 10.0
        reasons.append("Missing capacity data (+10 uncertainty penalty)")

    # 4. Lead Time
    lead = operational_meta.get("lead_time_days")
    if lead is not None:
        if lead >= 45:
            score += 10.0
            reasons.append(f"Long replacement lead time of {lead:.0f} days (+10)")
    else:
        score += 10.0
        reasons.append("Missing lead-time data (+10 uncertainty penalty)")

    # 5. Tier Visibility
    if min_hops >= 2:
        score += 5.0
        reasons.append(f"Multi-tier upstream dependency (Tier {min_hops}) with reduced visibility (+5)")

    # Clamp score to [0, 100]
    score = max(0.0, min(100.0, score))

    # Categorize Risk Level
    if score >= 85.0:
        risk_level = "Critical"
    elif score >= 65.0:
        risk_level = "High"
    elif score >= 40.0:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    explanation = "; ".join(reasons) + f". Final score: {score:.1f}/100."

    # Recommended action based on score and flags
    actions: List[str] = []
    if missing_info:
        actions.append(f"Audit missing data ({', '.join(missing_info)})")
    if operational_meta.get("single_source") or alts == 0:
        actions.append("Qualify secondary backup supplier immediately")
    if cap and cap >= 0.85:
        actions.append("Reserve safety buffer stock to alleviate capacity bottleneck")
    if affected_supplier_count >= 2:
        actions.append("Coordinate cross-supplier buffer inventory for shared dependency")

    if not actions:
        actions.append("Standard periodic supplier review")

    recommendation = ". ".join(actions) + "."

    return round(score, 1), risk_level, explanation, recommendation


def create_synthetic_supply_chain() -> nx.DiGraph:
    """
    Creates a sample multi-tier synthetic supply chain graph for development,
    demonstration, and integration testing with Member 1 & Member 5.

    Structure:
    - Raw Material P (Tier 2/3) -> Processor X
    - Raw Material Q (Tier 2/3) -> Processor Y
    - Processor X (Tier 2) -> Supplier A, Supplier B (Shared dependency!)
    - Processor Y (Tier 2) -> Supplier C (Single dependency)
    - Supplier A, B, C (Tier 1) -> Assembly Plant (Sink)
    """
    G = nx.DiGraph()

    # Raw Materials
    G.add_node("Raw Material P", node_type="raw_material", alternatives=["Mine Alpha", "Mine Beta"], capacity=0.70, lead_time_days=60)
    G.add_node("Raw Material Q", node_type="raw_material", alternatives=None, capacity=None, lead_time_days=None)

    # Processors
    G.add_node("Processor X", node_type="processor", alternatives=[], capacity=0.92, lead_time_days=45)
    G.add_node("Processor Y", node_type="processor", alternatives=["Processor Z"], capacity=0.60, lead_time_days=15)

    # Tier 1 Suppliers
    G.add_node("Supplier A", node_type="supplier", is_supplier=True, capacity=0.80, lead_time_days=20)
    G.add_node("Supplier B", node_type="supplier", is_supplier=True, capacity=0.85, lead_time_days=25)
    G.add_node("Supplier C", node_type="supplier", is_supplier=True, capacity=0.75, lead_time_days=10)

    # Edges (Upstream -> Downstream)
    G.add_edge("Raw Material P", "Processor X")
    G.add_edge("Raw Material Q", "Processor Y")
    G.add_edge("Processor X", "Supplier A")
    G.add_edge("Processor X", "Supplier B")
    G.add_edge("Processor Y", "Supplier C")

    return G
