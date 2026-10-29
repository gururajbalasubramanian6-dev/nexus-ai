"""
Integration Tests for Nexus-AI Disruption Simulation Agent (Member 3)
======================================================================

Validates multi-agent interface integration:
- Task 1: Capacity & demand sufficiency verification for alternative paths.
- Task 2 & 4: Interface compatibility with Member 1 (Mapping Agent).
- Task 4: Compatibility and payload delivery to Member 4 (Financial Engine).
- Task 4: Integration testing of Processor failure and Transport route failure.
- Task 3 & 4: Streamlit UI payload formatting for Member 5 (app.py).
"""

import pytest
import networkx as nx
import pandas as pd
from agents.disruption_simulation_agent import (
    simulate_disruption,
    create_synthetic_supply_chain,
    DisruptionSimulationAgent,
    get_financial_engine_payload,
)


@pytest.fixture
def canonical_graph() -> nx.DiGraph:
    """Provides the standard agreed synthetic dataset across all Nexus-AI agents."""
    return create_synthetic_supply_chain()


# ---------------------------------------------------------------------------
# Test Integration 1: Member 1 Mapping Agent Ingestion
# ---------------------------------------------------------------------------
def test_member1_mapping_agent_graph_ingestion(canonical_graph):
    """
    Verifies that the simulation agent ingests the standard directed graph
    produced by Member 1's Mapping Agent without requiring modifications.
    """
    # Verify graph conforms to agreed conventions
    assert isinstance(canonical_graph, nx.DiGraph)
    assert canonical_graph.number_of_nodes() >= 5
    assert canonical_graph.number_of_edges() >= 5

    # Run simulation directly on the graph
    result = simulate_disruption(canonical_graph, failed_node="Processor X")
    assert result["failed_entity"] == "Processor X"
    assert result["impact_level"] == "Critical"
    assert len(result["affected_entities"]) == 4


# ---------------------------------------------------------------------------
# Test Integration 2: Processor Failure Integration with Agreed Dataset
# ---------------------------------------------------------------------------
def test_processor_failure_integration(canonical_graph):
    """
    Tests simulation of Processor X failure on the agreed multi-tier dataset.
    Verifies affected paths, materials, direct capacity loss, and impact level.
    """
    result = simulate_disruption(canonical_graph, failed_node="Processor X")

    # Blast radius checks
    assert "Supplier A" in result["affected_entities"]
    assert "Supplier B" in result["affected_entities"]
    assert "Factory 1" in result["affected_entities"]
    assert "Factory 2" in result["affected_entities"]

    # Material checks
    materials = result["affected_materials"]
    assert "Semiconductor Wafers" in materials
    assert "Engine Control Units" in materials
    assert "Digital Cockpit Displays" in materials

    # Capacity loss check (Direct tier-1 capacity: 100 to A + 100 to B = 200)
    cap_impact = result["estimated_capacity_impact"]
    assert cap_impact["estimated_direct_capacity_loss"] == 200.0
    assert cap_impact["has_complete_capacity_data"] is True


# ---------------------------------------------------------------------------
# Test Integration 3: Transport Route Failure Integration with Agreed Dataset
# ---------------------------------------------------------------------------
def test_transport_route_failure_integration(canonical_graph):
    """
    Tests simulation of transport corridor failure using route identifier 'Route-PX-SA'
    (Processor X -> Supplier A corridor).
    """
    result = simulate_disruption(canonical_graph, failed_route="Route-PX-SA")

    assert result["disruption_type"] == "Transport route failure"
    assert "Route (Processor X -> Supplier A)" in result["failed_entity"]

    # Only Supplier A and Factory 1 should be in the blast radius
    assert "Supplier A" in result["affected_entities"]
    assert "Factory 1" in result["affected_entities"]

    # Supplier B and Factory 2 route is unaffected
    assert "Supplier B" not in result["affected_entities"]
    assert "Factory 2" not in result["affected_entities"]

    # Capacity loss on this specific corridor is 100 units
    assert result["estimated_capacity_impact"]["estimated_direct_capacity_loss"] == 100.0


# ---------------------------------------------------------------------------
# Test Integration 4: Alternative Supply Path Sufficiency & Deficit Check
# ---------------------------------------------------------------------------
def test_alternative_path_capacity_sufficiency(canonical_graph):
    """
    Verifies that alternative paths are NOT treated as sufficient unless capacity
    and demand support that conclusion (Task 1 requirement).

    In the agreed dataset:
    Factory 2 normally receives:
      - 100.0 units from Supplier B (severed by Processor X failure)
      - 50.0 units from Supplier C (remains operational via Processor Y)
    Remaining capacity (50.0) < Lost capacity (100.0).
    Therefore, the alternative path must NOT be marked sufficient!
    """
    result = simulate_disruption(canonical_graph, failed_node="Processor X")
    details = {d["entity"]: d for d in result["entity_impact_details"]}

    f2_detail = details["Factory 2"]
    assert f2_detail["status"] == "potentially_affected"
    assert f2_detail["is_alternative_sufficient"] is False
    assert f2_detail["alternative_sufficiency"] == "insufficient_capacity_deficit"
    assert f2_detail["remaining_inbound_capacity"] == 50.0
    assert f2_detail["lost_inbound_capacity"] == 100.0
    assert f2_detail["capacity_deficit"] == 100.0
    assert "insufficient to replace lost capacity" in f2_detail["explanation"]


# ---------------------------------------------------------------------------
# Test Integration 5: Member 4 Financial Engine Payload Delivery
# ---------------------------------------------------------------------------
def test_member4_financial_engine_payload(canonical_graph):
    """
    Validates that get_financial_engine_payload() creates the exact standardized
    structure required by Member 4's Financial Loss & Optimization Engine.
    """
    simulation_result = simulate_disruption(canonical_graph, failed_node="Processor X")
    financial_payload = get_financial_engine_payload(simulation_result)

    expected_financial_keys = [
        "failed_entity",
        "disruption_type",
        "impact_level",
        "direct_capacity_loss",
        "affected_entities",
        "confirmed_shortage_entities",
        "potential_shortage_entities",
        "capacity_deficit_by_entity",
        "total_downstream_capacity_deficit",
        "affected_materials",
        "missing_data_warnings",
        "assumptions",
    ]

    for key in expected_financial_keys:
        assert key in financial_payload, f"Financial payload missing key '{key}'"

    assert financial_payload["direct_capacity_loss"] == 200.0
    assert "Factory 1" in financial_payload["confirmed_shortage_entities"]
    assert "Factory 2" in financial_payload["potential_shortage_entities"]
    assert financial_payload["total_downstream_capacity_deficit"] > 0

    # Mock Member 4 financial computation:
    # Estimate financial cost = (direct capacity loss * unit price) + (downstream deficits * penalty)
    unit_price = 150.0  # $150 per wafer unit
    downtime_days = 30   # 30-day disruption
    estimated_daily_loss = financial_payload["direct_capacity_loss"] * unit_price
    total_loss_30d = estimated_daily_loss * downtime_days

    assert total_loss_30d == 200.0 * 150.0 * 30  # $900,000
    assert isinstance(total_loss_30d, float)


# ---------------------------------------------------------------------------
# Test Integration 6: Member 5 Streamlit UI DataFrames Delivery
# ---------------------------------------------------------------------------
def test_member5_streamlit_ui_dataframes(canonical_graph):
    """
    Validates that Member 5's Streamlit interface (app.py) can consume
    pre-formatted DataFrames for st.dataframe() and summary metrics.
    """
    agent = DisruptionSimulationAgent(canonical_graph)
    result = agent.simulate(failed_node="Processor X")

    # 1. Summary comparison table for all nodes
    all_nodes_df = agent.simulate_all_nodes()
    assert isinstance(all_nodes_df, pd.DataFrame)
    assert len(all_nodes_df) == canonical_graph.number_of_nodes()
    assert "Entity" in all_nodes_df.columns
    assert "Impact Level" in all_nodes_df.columns

    # 2. Granular entity breakdown table for the current scenario
    entity_df = agent.get_entity_impact_dataframe(result)
    assert isinstance(entity_df, pd.DataFrame)
    assert len(entity_df) == len(result["affected_entities"])
    assert "Status" in entity_df.columns
    assert "Capacity Deficit" in entity_df.columns
    assert "Alternative Sufficiency" in entity_df.columns
