"""
Unit and Integration Tests for Nexus-AI Shared Risk Detection Agent (Member 2)
==============================================================================

Covers Core Requirements:
- Test A: Shared dependency detection (Processor X -> Supplier A & Supplier B)
- Test B: Single supplier isolation (Processor Y -> Supplier C only)
- Test C: Multi-tier indirect shared dependency (Raw Material P -> Processor X -> A & B)
- Test D: Graph with no shared dependencies returns empty result without crashing
- Test E: Missing data detection (flags missing data without inventing values)
- Test F: Multi-path graphs do not double-count affected suppliers
- Test G: Input validation errors for invalid types / inputs
- Test H: DataFrame conversion and structured schema validation
"""

import pytest
import networkx as nx
import pandas as pd

from agents.risk_detection_agent import (
    detect_shared_risks,
    calculate_risk_score,
    create_synthetic_supply_chain,
)


# ============================================================================
# Test A: Basic Shared Dependency Detection
# ============================================================================
def test_a_processor_x_shared_dependency():
    """
    Test A:
    Processor X connects to Supplier A and Supplier B.
    Expected: Processor X is detected as a shared dependency.
    """
    G = nx.DiGraph()
    G.add_edge("Processor X", "Supplier A")
    G.add_edge("Processor X", "Supplier B")

    findings = detect_shared_risks(G)

    # Must find Processor X
    nodes_found = [f["dependency_node"] for f in findings]
    assert "Processor X" in nodes_found

    proc_x = next(f for f in findings if f["dependency_node"] == "Processor X")
    assert proc_x["affected_supplier_count"] == 2
    assert sorted(proc_x["affected_suppliers"]) == ["Supplier A", "Supplier B"]
    assert proc_x["is_shared"] is True
    assert proc_x["dependency_tier"] == "Tier 1 (Direct)"


# ============================================================================
# Test B: Single Supplier Isolation
# ============================================================================
def test_b_processor_y_not_reported_as_shared():
    """
    Test B:
    Processor Y connects only to Supplier C.
    Expected: It is not incorrectly reported as a shared dependency of multiple direct suppliers.
    """
    G = nx.DiGraph()
    G.add_edge("Processor X", "Supplier A")
    G.add_edge("Processor X", "Supplier B")
    G.add_edge("Processor Y", "Supplier C")

    findings = detect_shared_risks(G, only_shared=True)

    shared_nodes = [f["dependency_node"] for f in findings]
    assert "Processor Y" not in shared_nodes

    # When only_shared=False, Processor Y appears with affected_supplier_count == 1 and is_shared == False
    all_findings = detect_shared_risks(G, only_shared=False)
    proc_y = next((f for f in all_findings if f["dependency_node"] == "Processor Y"), None)
    assert proc_y is not None
    assert proc_y["affected_supplier_count"] == 1
    assert proc_y["is_shared"] is False


# ============================================================================
# Test C: Multi-Tier Dependency Analysis
# ============================================================================
def test_c_multi_tier_indirect_shared_dependency():
    """
    Test C:
    A multi-tier graph contains a common upstream raw material:
    Raw Material P -> Processor X -> Supplier A
                                  -> Supplier B
    Expected: The indirect shared dependency (Raw Material P) is detected
    as well as Processor X.
    """
    G = nx.DiGraph()
    G.add_edge("Raw Material P", "Processor X")
    G.add_edge("Processor X", "Supplier A")
    G.add_edge("Processor X", "Supplier B")

    findings = detect_shared_risks(G)
    nodes_found = {f["dependency_node"]: f for f in findings}

    # Both Raw Material P and Processor X must be flagged
    assert "Raw Material P" in nodes_found
    assert "Processor X" in nodes_found

    rm_p = nodes_found["Raw Material P"]
    assert rm_p["affected_supplier_count"] == 2
    assert sorted(rm_p["affected_suppliers"]) == ["Supplier A", "Supplier B"]
    assert rm_p["dependency_tier"] == "Tier 2 (Indirect)"
    assert len(rm_p["direct_affected_suppliers"]) == 0
    assert sorted(rm_p["indirect_affected_suppliers"]) == ["Supplier A", "Supplier B"]


# ============================================================================
# Test D: Graph With No Shared Dependencies
# ============================================================================
def test_d_no_shared_dependencies_returns_empty():
    """
    Test D:
    The graph has no shared dependencies (independent 1-to-1 supplier chains).
    Expected: The agent returns an empty result without crashing.
    """
    G = nx.DiGraph()
    G.add_edge("Processor 1", "Supplier 1")
    G.add_edge("Processor 2", "Supplier 2")
    G.add_edge("Processor 3", "Supplier 3")

    findings = detect_shared_risks(G, only_shared=True)
    assert isinstance(findings, list)
    assert len(findings) == 0

    # Also test as DataFrame
    df = detect_shared_risks(G, only_shared=True, as_dataframe=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ============================================================================
# Test E: Missing Data Handling
# ============================================================================
def test_e_missing_data_flagged_without_inventing_values():
    """
    Test E:
    Some capacity or alternative-supplier information is missing.
    Expected: The result flags the missing information rather than inventing values.
    """
    G = nx.DiGraph()
    # Processor Alpha has missing capacity and missing alternatives
    G.add_node("Processor Alpha", lead_time_days=30)
    # Processor Beta has full information
    G.add_node("Processor Beta", capacity=0.80, lead_time_days=25, alternatives=["Backup Beta"])

    G.add_edge("Processor Alpha", "Supplier 1")
    G.add_edge("Processor Alpha", "Supplier 2")
    G.add_edge("Processor Beta", "Supplier 1")
    G.add_edge("Processor Beta", "Supplier 2")

    findings = detect_shared_risks(G)
    alpha = next(f for f in findings if f["dependency_node"] == "Processor Alpha")
    beta = next(f for f in findings if f["dependency_node"] == "Processor Beta")

    # Alpha must flag missing data
    assert any("capacity" in msg.lower() for msg in alpha["missing_information"])
    assert any("alternative" in msg.lower() for msg in alpha["missing_information"])

    # Beta has full data, so no missing info
    assert len(beta["missing_information"]) == 0

    # Alpha's risk score should be higher due to uncertainty penalty
    assert alpha["risk_score"] > beta["risk_score"]


# ============================================================================
# Test F: Diamond Graph / Multiple Paths (No Double Counting)
# ============================================================================
def test_f_multiple_paths_do_not_double_count_supplier():
    """
    Test F:
    A graph contains multiple paths to the same supplier:
    Source S -> Intermediate M1 -> Supplier A
    Source S -> Intermediate M2 -> Supplier A
    Source S -> Supplier B
    Expected: Supplier A is counted only once, so total affected suppliers = 2 (A and B).
    """
    G = nx.DiGraph()
    G.add_edge("Source S", "Intermediate M1")
    G.add_edge("Source S", "Intermediate M2")
    G.add_edge("Intermediate M1", "Supplier A")
    G.add_edge("Intermediate M2", "Supplier A")
    G.add_edge("Source S", "Supplier B")

    findings = detect_shared_risks(G, target_suppliers=["Supplier A", "Supplier B"])
    source_s = next(f for f in findings if f["dependency_node"] == "Source S")

    assert source_s["affected_supplier_count"] == 2
    assert sorted(source_s["affected_suppliers"]) == ["Supplier A", "Supplier B"]


# ============================================================================
# Test G: Input Validation and Error Handling
# ============================================================================
def test_g_invalid_inputs_raise_clear_errors():
    """
    Test G:
    Invalid or unexpected input is provided.
    Expected: A clear validation error is returned.
    """
    # 1. None passed as graph
    with pytest.raises(TypeError, match="Graph cannot be None"):
        detect_shared_risks(None)

    # 2. Undirected graph passed
    with pytest.raises(TypeError, match="Invalid graph type"):
        detect_shared_risks(nx.Graph())

    # 3. String or non-graph object passed
    with pytest.raises(TypeError, match="Invalid graph type"):
        detect_shared_risks("not_a_graph")

    # 4. Target suppliers not in graph
    G = nx.DiGraph()
    G.add_edge("X", "Y")
    with pytest.raises(ValueError, match="Specified target suppliers not found"):
        detect_shared_risks(G, target_suppliers=["NonExistentSupplier"])

    # 5. Invalid target_suppliers type
    with pytest.raises(TypeError, match="target_suppliers must be a list"):
        detect_shared_risks(G, target_suppliers=123)


# ============================================================================
# Test H: Synthetic Supply Chain & Output Schema
# ============================================================================
def test_h_synthetic_supply_chain_and_schema():
    """
    Validates end-to-end execution on the synthetic supply chain
    and checks that all required output schema fields are present.
    """
    G = create_synthetic_supply_chain()
    df = detect_shared_risks(G, as_dataframe=True)

    expected_columns = [
        "dependency_node",
        "affected_suppliers",
        "affected_supplier_count",
        "dependency_tier",
        "is_shared",
        "direct_affected_suppliers",
        "indirect_affected_suppliers",
        "risk_score",
        "risk_level",
        "explanation",
        "missing_information",
        "recommended_investigation",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing expected column '{col}' in output"

    assert len(df) > 0
    # Top risks should be Processor X and Raw Material P
    top_nodes = list(df["dependency_node"])
    assert "Processor X" in top_nodes
    assert "Raw Material P" in top_nodes
