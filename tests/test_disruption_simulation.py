"""
Tests for Nexus-AI Disruption Simulation Agent (Member 3)
=========================================================

Verifies all required scenarios from Task 6:
- Test A: When Processor X fails, Supplier A and Supplier B are identified as potentially affected.
- Test B: An unrelated entity is not incorrectly marked as affected.
- Test C: A factory failure is handled correctly.
- Test D: A supplier failure is handled correctly.
- Test E: Missing capacity data is flagged without inventing values.
- Test F: Disconnected nodes are handled correctly.
- Test G: Invalid input produces a clear error.
- Test H: Multi-level downstream relationships are handled correctly.
- Test I: Repeated paths do not cause duplicate entities in the affected-entity list.
- Test J: The returned output follows the documented schema.
- Additional tests: Transport route failures, Class wrapper, Shortage distinction, Double-counting checks.
"""

import pytest
import networkx as nx
import pandas as pd
from agents.disruption_simulation_agent import (
    simulate_disruption,
    create_synthetic_supply_chain,
    DisruptionSimulationAgent,
)


@pytest.fixture
def synthetic_graph() -> nx.DiGraph:
    """Provides a fresh instance of the canonical synthetic supply chain."""
    return create_synthetic_supply_chain()


# ---------------------------------------------------------------------------
# Test A: When Processor X fails, Supplier A and Supplier B are identified
# ---------------------------------------------------------------------------
def test_processor_x_failure_affects_suppliers_a_and_b(synthetic_graph):
    """
    Test A: When Processor X fails, Supplier A and Supplier B must both be
    identified in the affected entities list.
    """
    result = simulate_disruption(synthetic_graph, failed_node="Processor X")
    affected = result["affected_entities"]

    assert "Supplier A" in affected, "Supplier A must be affected by Processor X failure"
    assert "Supplier B" in affected, "Supplier B must be affected by Processor X failure"
    assert result["disruption_type"] == "Processor failure"


# ---------------------------------------------------------------------------
# Test B: An unrelated entity is not incorrectly marked as affected
# ---------------------------------------------------------------------------
def test_unrelated_entity_not_marked_affected(synthetic_graph):
    """
    Test B: Unrelated entities (e.g. Raw Material Q, Processor Y, Supplier C)
    must NOT be marked as affected when Processor X fails.
    """
    result = simulate_disruption(synthetic_graph, failed_node="Processor X")
    affected = set(result["affected_entities"])

    assert "Raw Material Q" not in affected, "Upstream unrelated node should not be affected"
    assert "Processor Y" not in affected, "Parallel processor should not be affected"
    assert "Supplier C" not in affected, "Supplier C (supplied by Processor Y) should not be affected"
    assert "Raw Material P" not in affected, "Upstream parent of Processor X should not be downstream affected"


# ---------------------------------------------------------------------------
# Test C: A factory failure is handled correctly
# ---------------------------------------------------------------------------
def test_factory_failure_handled_correctly(synthetic_graph):
    """
    Test C: Factory 1 failure is a terminal/sink failure:
    - No downstream nodes affected
    - Handled cleanly without errors
    - Disruption type recognized as Factory failure
    - Impact level is Low
    """
    result = simulate_disruption(synthetic_graph, failed_node="Factory 1")

    assert result["disruption_type"] == "Factory failure"
    assert len(result["affected_entities"]) == 0
    assert result["impact_level"] == "Low"
    assert "no downstream dependencies" in result["explanation"].lower()


# ---------------------------------------------------------------------------
# Test D: A supplier failure is handled correctly
# ---------------------------------------------------------------------------
def test_supplier_failure_handled_correctly(synthetic_graph):
    """
    Test D: When Supplier A fails:
    - Disruption type is Supplier failure
    - Factory 1 is affected
    - Factory 2 is not affected (supplied by B and C)
    """
    result = simulate_disruption(synthetic_graph, failed_node="Supplier A")

    assert result["disruption_type"] == "Supplier failure"
    assert "Factory 1" in result["affected_entities"]
    assert "Factory 2" not in result["affected_entities"]
    assert "Engine Control Units" in result["affected_materials"]


# ---------------------------------------------------------------------------
# Test E: Missing capacity data is flagged without inventing values
# ---------------------------------------------------------------------------
def test_missing_capacity_data_flagged_without_inventing_values():
    """
    Test E: If a node or relationship lacks capacity:
    - Missing info warning must be explicitly populated
    - Capacity is NOT fabricated
    - has_complete_capacity_data is False
    """
    G = nx.DiGraph()
    G.add_node("Node A", node_type="supplier")  # No capacity
    G.add_node("Node B", node_type="factory")   # No capacity
    G.add_edge("Node A", "Node B")              # No capacity on edge

    result = simulate_disruption(G, failed_node="Node A")

    assert result["estimated_capacity_impact"]["has_complete_capacity_data"] is False
    assert result["estimated_capacity_impact"]["estimated_direct_capacity_loss"] is None
    assert len(result["missing_information"]) > 0
    assert any("capacity" in item.lower() for item in result["missing_information"])


# ---------------------------------------------------------------------------
# Test F: Disconnected nodes are handled correctly
# ---------------------------------------------------------------------------
def test_disconnected_nodes_handled_correctly():
    """
    Test F: An isolated node with no edges:
    - Simulates without crashes
    - Returns empty affected_entities list
    - Returns valid structured output
    """
    G = nx.DiGraph()
    G.add_node("Isolated Island", node_type="factory", capacity=50.0)

    result = simulate_disruption(G, failed_node="Isolated Island")

    assert result["failed_entity"] == "Isolated Island"
    assert result["affected_entities"] == []
    assert result["affected_paths"] == [["Isolated Island"]]
    assert result["impact_level"] == "Low"


# ---------------------------------------------------------------------------
# Test G: Invalid input produces a clear error
# ---------------------------------------------------------------------------
def test_invalid_input_produces_clear_error(synthetic_graph):
    """
    Test G: Invalid inputs must raise appropriate TypeError or ValueError with helpful messages.
    """
    # Graph is None
    with pytest.raises(TypeError, match="Graph cannot be None"):
        simulate_disruption(None, failed_node="Processor X")

    # Graph is wrong type
    with pytest.raises(TypeError, match="Expected networkx.DiGraph"):
        simulate_disruption("not_a_graph", failed_node="Processor X")

    # Empty graph
    with pytest.raises(ValueError, match="graph is empty"):
        simulate_disruption(nx.DiGraph(), failed_node="Processor X")

    # Non-existent node
    with pytest.raises(ValueError, match="not found in the supply chain graph"):
        simulate_disruption(synthetic_graph, failed_node="NonExistent_Node_999")

    # No target specified
    with pytest.raises(ValueError, match="Must specify either failed_node"):
        simulate_disruption(synthetic_graph)


# ---------------------------------------------------------------------------
# Test H: Multi-level downstream relationships are handled correctly
# ---------------------------------------------------------------------------
def test_multi_level_downstream_relationships(synthetic_graph):
    """
    Test H: Processor X is 2 hops away from Factory 1 and Factory 2.
    When Processor X fails, both Tier 1 suppliers (Supplier A, Supplier B)
    AND Tier 2 factories (Factory 1, Factory 2) must be in affected_entities.
    """
    result = simulate_disruption(synthetic_graph, failed_node="Processor X")
    affected = set(result["affected_entities"])

    expected = {"Supplier A", "Supplier B", "Factory 1", "Factory 2"}
    assert expected.issubset(affected), f"Expected {expected} to be in affected {affected}"


# ---------------------------------------------------------------------------
# Test I: Repeated paths do not cause duplicate entities
# ---------------------------------------------------------------------------
def test_repeated_paths_no_duplicate_entities():
    """
    Test I: Diamond topology with multiple redundant paths:
         A -> B -> D
         A -> C -> D
    When A fails, node D must appear EXACTLY ONCE in affected_entities.
    """
    diamond_G = nx.DiGraph()
    diamond_G.add_node("A", node_type="processor", capacity=100)
    diamond_G.add_node("B", node_type="supplier", capacity=50)
    diamond_G.add_node("C", node_type="supplier", capacity=50)
    diamond_G.add_node("D", node_type="factory", capacity=100)

    diamond_G.add_edge("A", "B", material="Part 1", capacity=50)
    diamond_G.add_edge("A", "C", material="Part 1", capacity=50)
    diamond_G.add_edge("B", "D", material="Subassembly 1", capacity=50)
    diamond_G.add_edge("C", "D", material="Subassembly 2", capacity=50)

    result = simulate_disruption(diamond_G, failed_node="A")
    affected = result["affected_entities"]

    assert len(affected) == len(set(affected)), "affected_entities must contain no duplicates"
    assert affected.count("D") == 1, "Entity D should appear exactly once"
    assert sorted(affected) == ["B", "C", "D"]


# ---------------------------------------------------------------------------
# Test J: The returned output follows the documented schema
# ---------------------------------------------------------------------------
def test_returned_output_schema(synthetic_graph):
    """
    Test J: Validates that all required top-level keys and nested structures
    are present and typed correctly according to Task 5.
    """
    result = simulate_disruption(synthetic_graph, failed_node="Processor X")

    required_keys = [
        "failed_entity",
        "disruption_type",
        "affected_entities",
        "affected_paths",
        "affected_materials",
        "estimated_capacity_impact",
        "impact_level",
        "explanation",
        "missing_information",
        "assumptions",
        "recommended_next_actions",
    ]

    for key in required_keys:
        assert key in result, f"Missing required output key: '{key}'"

    # Type checks
    assert isinstance(result["failed_entity"], str)
    assert isinstance(result["disruption_type"], str)
    assert isinstance(result["affected_entities"], list)
    assert isinstance(result["affected_paths"], list)
    assert isinstance(result["affected_materials"], list)
    assert isinstance(result["estimated_capacity_impact"], dict)
    assert isinstance(result["impact_level"], str)
    assert isinstance(result["explanation"], str)
    assert isinstance(result["missing_information"], list)
    assert isinstance(result["assumptions"], list)
    assert isinstance(result["recommended_next_actions"], list)

    # Validate impact level categories
    assert result["impact_level"] in ["Low", "Medium", "High", "Critical"]


# ---------------------------------------------------------------------------
# Additional Tests: Transport Route Failures, Shortages, Class Wrapper
# ---------------------------------------------------------------------------
def test_transport_route_failure_via_edge_tuple(synthetic_graph):
    """
    Tests transport route failure by passing an edge tuple.
    """
    result = simulate_disruption(
        synthetic_graph,
        failed_edge=("Processor X", "Supplier A"),
    )

    assert result["disruption_type"] == "Transport route failure"
    assert "Supplier A" in result["affected_entities"]
    assert "Factory 1" in result["affected_entities"]
    assert "Supplier B" not in result["affected_entities"]


def test_transport_route_failure_via_route_attribute(synthetic_graph):
    """
    Tests transport route failure by passing a route identifier string.
    """
    result = simulate_disruption(synthetic_graph, failed_route="Route-SA-F1")

    assert result["disruption_type"] == "Transport route failure"
    assert result["affected_entities"] == ["Factory 1"]
    assert "Engine Control Units" in result["affected_materials"]


def test_distinguish_potentially_affected_vs_confirmed_shortage(synthetic_graph):
    """
    When Processor X fails:
    - Factory 1 only gets supply from Supplier A (which gets from Processor X) -> confirmed shortage
    - Factory 2 gets supply from Supplier B (severed) AND Supplier C (operational via Processor Y)
      -> Factory 2 is potentially affected, not confirmed shortage!
    """
    result = simulate_disruption(synthetic_graph, failed_node="Processor X")
    details = {d["entity"]: d for d in result["entity_impact_details"]}

    assert details["Factory 1"]["status"] == "confirmed_shortage"
    assert details["Factory 2"]["status"] == "potentially_affected"
    assert details["Factory 2"]["operational_inbound_links"] > 0


def test_no_double_counting_capacity(synthetic_graph):
    """
    Verifies that multi-tier capacities are not summed together.
    Outgoing from Processor X: 100 to A, 100 to B -> total direct = 200.
    Multi-tier sum would be 200 + 100 + 100 + 100 + 100 = 600 (invalid double-counting).
    """
    result = simulate_disruption(synthetic_graph, failed_node="Processor X")
    cap_impact = result["estimated_capacity_impact"]

    assert cap_impact["estimated_direct_capacity_loss"] == 200.0
    assert "NOT summed" in cap_impact["accounting_method"]


def test_disruption_agent_class_wrapper(synthetic_graph):
    """
    Tests the DisruptionSimulationAgent OOP interface and simulate_all_nodes method.
    """
    agent = DisruptionSimulationAgent(synthetic_graph)
    df = agent.simulate_all_nodes()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == synthetic_graph.number_of_nodes()
    assert "Entity" in df.columns
    assert "Impact Level" in df.columns
# ---------------------------------------------------------------------------
# Test: Operational Inbound & Confirmed Shortage Propagation
# ---------------------------------------------------------------------------
from agents.mapping_agent import (
    load_and_validate_dependencies,
    build_supply_chain_graph,
    DEFAULT_DEPENDENCIES_CSV,
)


def test_processor_x_failure_operational_inbound_and_shortage(synthetic_graph):
    """
    Verifies that when Processor X fails:
    1. Direct children (Supplier A, Supplier B) have Operational Inbound = 0, Severed Inbound = 1,
       and Status = Confirmed Shortage.
    2. Factory 1 (downstream of Supplier A) has a physically existing link (Supplier A -> Factory 1),
       but because Supplier A is starved of upstream raw material, Factory 1 has:
       Operational Inbound = 0, Severed Inbound = 1, Status = Confirmed Shortage.
    3. Factory 2 (supplied by severed Supplier B and active Supplier C) has:
       Operational Inbound = 1, Severed Inbound = 1, Status = Potentially Affected.
    """
    res = simulate_disruption(synthetic_graph, failed_node="Processor X")
    details = {d["entity"]: d for d in res["entity_impact_details"]}

    # Supplier A and B: direct link from failed node is severed
    for supp in ["Supplier A", "Supplier B"]:
        assert details[supp]["operational_inbound_links"] == 0
        assert details[supp]["severed_inbound_links"] == 1
        assert details[supp]["status"] == "confirmed_shortage"

    # Factory 1: link from Supplier A exists physically, but is unusable due to upstream starvation
    assert details["Factory 1"]["operational_inbound_links"] == 0
    assert details["Factory 1"]["severed_inbound_links"] == 1
    assert details["Factory 1"]["status"] == "confirmed_shortage"

    # Factory 2: has 1 unusable link (Supplier B) and 1 usable operational link (Supplier C)
    assert details["Factory 2"]["operational_inbound_links"] == 1
    assert details["Factory 2"]["severed_inbound_links"] == 1
    assert details["Factory 2"]["status"] == "potentially_affected"


def test_raw_material_p_failure_operational_inbound_and_shortage():
    """
    Verifies that when Raw Material P fails on the standard/SAP HANA supply chain graph:
    1. Processor X shows Operational Inbound = 0, Severed Inbound = 1, Status = Confirmed Shortage.
    2. Downstream nodes Supplier A and Supplier B have physically existing links from Processor X,
       but because Processor X is starved of raw material, they show:
       Operational Inbound = 0, Severed Inbound = 1, Status = Confirmed Shortage.
    """
    df = load_and_validate_dependencies(DEFAULT_DEPENDENCIES_CSV)
    graph = build_supply_chain_graph(df)

    res = simulate_disruption(graph, failed_node="Raw Material P")
    details = {d["entity"]: d for d in res["entity_impact_details"]}

    assert details["Processor X"]["operational_inbound_links"] == 0
    assert details["Processor X"]["severed_inbound_links"] == 1
    assert details["Processor X"]["status"] == "confirmed_shortage"

    # Supplier A and Supplier B have physically existing edge from Processor X, but unusable
    for supp in ["Supplier A", "Supplier B"]:
        assert details[supp]["operational_inbound_links"] == 0
        assert details[supp]["severed_inbound_links"] == 1
        assert details[supp]["status"] == "confirmed_shortage"


def test_raw_material_q_failure_operational_inbound_and_shortage():
    """
    Verifies that when Raw Material Q fails on the standard/SAP HANA supply chain graph:
    1. Processor Y shows Operational Inbound = 0, Severed Inbound = 1, Status = Confirmed Shortage.
    2. Downstream node Supplier C has a physically existing link from Processor Y,
       but because Processor Y is starved of raw material, Supplier C shows:
       Operational Inbound = 0, Severed Inbound = 1, Status = Confirmed Shortage.
    """
    df = load_and_validate_dependencies(DEFAULT_DEPENDENCIES_CSV)
    graph = build_supply_chain_graph(df)

    res = simulate_disruption(graph, failed_node="Raw Material Q")
    details = {d["entity"]: d for d in res["entity_impact_details"]}

    assert details["Processor Y"]["operational_inbound_links"] == 0
    assert details["Processor Y"]["severed_inbound_links"] == 1
    assert details["Processor Y"]["status"] == "confirmed_shortage"

    # Supplier C has physically existing edge from Processor Y, but unusable
    assert details["Supplier C"]["operational_inbound_links"] == 0
    assert details["Supplier C"]["severed_inbound_links"] == 1
    assert details["Supplier C"]["status"] == "confirmed_shortage"
