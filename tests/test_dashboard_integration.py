"""
Integration Tests for Nexus-AI Streamlit Dashboard & Member 5 System
====================================================================

Tests all Phase 5 requirements:
1. Application imports successfully.
2. Dataset loading works (sample CSV and custom CSV).
3. Mapping agent integration works (graph construction & node classification).
4. Shared-risk analysis works (detects shared bottlenecks).
5. Disruption simulation works (node outages and blast radius).
6. Financial engine receives the expected input payload.
7. Decision explanation reflects actual outputs accurately.
8. Missing data is handled honestly without fabrication.
9. Invalid inputs show clear errors (DataValidationError, FileNotFoundError).
10. Changing the selected scenario changes the displayed results.
11. Plotly resilience network figure generates without error.
"""

from pathlib import Path
import tempfile
import pytest
import networkx as nx
import pandas as pd
import plotly.graph_objects as go

# Member 1
from agents.mapping_agent import (
    DataValidationError,
    build_supply_chain_graph,
    load_and_validate_dependencies,
    DEFAULT_DEPENDENCIES_CSV,
)

# Member 2
from agents.risk_detection_agent import detect_shared_risks

# Member 3
from agents.disruption_simulation_agent import (
    DisruptionSimulationAgent,
    create_synthetic_supply_chain,
    get_financial_engine_payload,
    simulate_disruption,
)

# Member 4
from agents.financial_optimization_agent import (
    DisruptionScenario,
    RecoveryOption,
    optimize_recovery_plan,
)

# Member 5
from agents.decision_explanation_agent import generate_decision_explanation
from agents.dashboard_components import (
    build_financial_scenario_from_simulation,
    create_resilient_network_figure,
    generate_candidate_recovery_options,
)


# =========================================================================
# Test 1: Application imports successfully
# =========================================================================
def test_01_application_imports_successfully():
    """Verify that all core modules, dashboard components, and entrypoints import without error."""
    import app
    import agents.mapping_agent
    import agents.risk_detection_agent
    import agents.disruption_simulation_agent
    import agents.financial_optimization_agent
    import agents.decision_explanation_agent
    import agents.dashboard_components

    assert hasattr(app, "run_streamlit_app")
    assert hasattr(app, "run_cli_fallback")


# =========================================================================
# Test 2: Dataset loading works
# =========================================================================
def test_02_dataset_loading_works():
    """Verify loading of default sample dependencies.csv."""
    assert DEFAULT_DEPENDENCIES_CSV.exists()
    df = load_and_validate_dependencies(DEFAULT_DEPENDENCIES_CSV)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert set(df.columns) == {"upstream_node", "downstream_node", "material", "capacity"}
    assert (df["capacity"] > 0).all()


# =========================================================================
# Test 3: Mapping agent integration works
# =========================================================================
def test_03_mapping_agent_integration_works():
    """Verify that the mapping agent builds a valid directed NetworkX graph with attributes."""
    df = load_and_validate_dependencies(DEFAULT_DEPENDENCIES_CSV)
    graph = build_supply_chain_graph(df)

    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == 7
    assert graph.number_of_edges() == 5

    # Check node classification
    assert graph.nodes["Raw Material P"]["node_type"] == "Raw Material"
    assert graph.nodes["Processor X"]["node_type"] == "Processor"
    assert graph.nodes["Supplier A"]["node_type"] == "Supplier"

    # Check edge attributes
    edge_data = graph["Processor X"]["Supplier A"]
    assert edge_data["material"] == "Battery cells"
    assert edge_data["capacity"] == 100.0


# =========================================================================
# Test 4: Shared-risk analysis works
# =========================================================================
def test_04_shared_risk_analysis_works():
    """Verify that Member 2's risk detection detects Processor X as a shared dependency."""
    df = load_and_validate_dependencies(DEFAULT_DEPENDENCIES_CSV)
    graph = build_supply_chain_graph(df)

    findings = detect_shared_risks(graph, only_shared=True)
    assert isinstance(findings, list)
    assert len(findings) >= 1

    dep_nodes = [f["dependency_node"] for f in findings]
    assert "Processor X" in dep_nodes

    px_finding = next(f for f in findings if f["dependency_node"] == "Processor X")
    assert px_finding["is_shared"] is True
    assert px_finding["affected_supplier_count"] == 2
    assert set(px_finding["affected_suppliers"]) == {"Supplier A", "Supplier B"}
    assert px_finding["risk_score"] > 0


# =========================================================================
# Test 5: Disruption simulation works
# =========================================================================
def test_05_disruption_simulation_works():
    """Verify that Member 3's simulation traces affected entities and capacities."""
    df = load_and_validate_dependencies(DEFAULT_DEPENDENCIES_CSV)
    graph = build_supply_chain_graph(df)

    sim = simulate_disruption(graph, failed_node="Processor X")
    assert sim["failed_entity"] == "Processor X"
    assert sim["impact_level"] == "Critical"
    assert set(sim["affected_entities"]) == {"Supplier A", "Supplier B"}
    assert "Battery cells" in sim["affected_materials"]
    assert sim["estimated_capacity_impact"]["estimated_direct_capacity_loss"] == 180.0


# =========================================================================
# Test 6: Financial engine receives the expected input
# =========================================================================
def test_06_financial_engine_receives_expected_input():
    """Verify that build_financial_scenario_from_simulation correctly maps payloads."""
    graph = create_synthetic_supply_chain()
    sim = simulate_disruption(graph, failed_node="Processor X")

    scenario = build_financial_scenario_from_simulation(
        simulation_result=sim,
        loss_per_unit=250.0,
        available_budget=60000.0,
        max_lead_time_days=10.0,
    )

    assert isinstance(scenario, DisruptionScenario)
    assert scenario.disrupted_node == "Processor X"
    assert scenario.affected_quantity == 200.0
    assert scenario.loss_per_unit == 250.0
    assert scenario.baseline_disruption_loss == 200.0 * 250.0  # 50,000.0
    assert scenario.available_budget == 60000.0
    assert scenario.max_lead_time_days == 10.0

    # Run optimizer
    options = generate_candidate_recovery_options(scenario, graph=graph)
    fin_result = optimize_recovery_plan(scenario, options)

    assert "recommended_option_ids" in fin_result
    assert "estimated_savings" in fin_result
    assert "feasible_options" in fin_result


# =========================================================================
# Test 7: Decision explanation reflects actual outputs
# =========================================================================
def test_07_decision_explanation_reflects_actual_outputs():
    """Verify that the decision explanation strictly uses actual calculated values."""
    graph = create_synthetic_supply_chain()
    sim = simulate_disruption(graph, failed_node="Processor X")
    scenario = build_financial_scenario_from_simulation(sim, loss_per_unit=150.0, available_budget=40000.0)
    options = generate_candidate_recovery_options(scenario, graph=graph)
    fin = optimize_recovery_plan(scenario, options)
    risks = detect_shared_risks(graph, only_shared=False)

    exp = generate_decision_explanation(
        graph=graph,
        disruption_result=sim,
        financial_result=fin,
        risk_findings=risks,
    )

    # Verify no hardcoded or fabricated numbers
    assert exp["what_failed"]["failed_entity"] == sim["failed_entity"]
    assert exp["affected_entities"]["total_affected_count"] == len(sim["affected_entities"])
    assert exp["financial_calculation"]["baseline_loss"] == fin["baseline_disruption_loss"]
    assert exp["financial_calculation"]["estimated_savings"] == fin["estimated_savings"]
    assert exp["selected_recovery_option"]["selected_ids"] == fin["recommended_option_ids"]


# =========================================================================
# Test 8: Missing data is handled honestly
# =========================================================================
def test_08_missing_data_handled_honestly():
    """Verify that missing capacity and lead-time attributes are disclosed, not fabricated."""
    df = load_and_validate_dependencies(DEFAULT_DEPENDENCIES_CSV)
    graph = build_supply_chain_graph(df)

    # In dependencies.csv, nodes don't have lead_time_days or alternatives registered
    sim = simulate_disruption(graph, failed_node="Processor X")
    scenario = build_financial_scenario_from_simulation(sim)
    options = generate_candidate_recovery_options(scenario, graph=graph)
    fin = optimize_recovery_plan(scenario, options)

    exp = generate_decision_explanation(graph=graph, disruption_result=sim, financial_result=fin)
    disc = exp["assumptions_and_missing_data"]

    # Verify data gaps are present and disclosed
    assert len(disc["missing_data"]) > 0
    assert any("missing" in m.lower() for m in disc["missing_data"])


# =========================================================================
# Test 9: Invalid inputs show clear errors
# =========================================================================
def test_09_invalid_inputs_show_clear_errors():
    """Verify that corrupt CSV data raises DataValidationError and FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_and_validate_dependencies("data/non_existent_path.csv")

    # Invalid header
    bad_header_csv = "node_a,node_b,mat,cap\nA,B,Metal,100\n"
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
        tmp.write(bad_header_csv)
        tmp_path = Path(tmp.name)

    try:
        with pytest.raises(DataValidationError) as exc_info:
            load_and_validate_dependencies(tmp_path)
        assert "missing required column" in str(exc_info.value).lower()
    finally:
        tmp_path.unlink(missing_ok=True)


# =========================================================================
# Test 10: Changing selected scenario changes displayed results
# =========================================================================
def test_10_changing_scenario_changes_results():
    """Verify that simulating Processor X produces different results from Processor Y."""
    graph = create_synthetic_supply_chain()

    res_px = simulate_disruption(graph, failed_node="Processor X")
    res_py = simulate_disruption(graph, failed_node="Processor Y")

    assert res_px["failed_entity"] != res_py["failed_entity"]
    assert res_px["affected_entities"] != res_py["affected_entities"]
    assert res_px["impact_level"] != res_py["impact_level"]
    assert res_px["affected_materials"] != res_py["affected_materials"]

    # Financial scenarios must also differ
    scen_px = build_financial_scenario_from_simulation(res_px, loss_per_unit=150.0)
    scen_py = build_financial_scenario_from_simulation(res_py, loss_per_unit=150.0)
    assert scen_px.affected_quantity != scen_py.affected_quantity
    assert scen_px.baseline_disruption_loss != scen_py.baseline_disruption_loss


# =========================================================================
# Test 11: Resilience network figure generation
# =========================================================================
def test_11_resilience_network_figure_generation():
    """Verify that create_resilient_network_figure creates a valid Plotly Figure."""
    graph = create_synthetic_supply_chain()
    sim = simulate_disruption(graph, failed_node="Processor X")

    fig = create_resilient_network_figure(
        graph=graph,
        simulation_result=sim,
        selected_entity="Processor X",
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # Contains scatter traces for routes and nodes
    assert len(fig.layout.annotations) > 0  # Contains directed edge arrows


# =========================================================================
# Test 12: SAP HANA Cloud live data source integration
# =========================================================================
def test_12_sap_hana_integration_in_dashboard():
    """Verify that SAP HANA supply chain graph integrates with dashboard components."""
    from unittest.mock import MagicMock, patch
    from agents.sap_adapter import SAPConnectionConfig, load_sap_supply_chain

    sample_nodes = [
        ("Raw Material P", "Raw Material"),
        ("Raw Material Q", "Raw Material"),
        ("Processor X", "Processor"),
        ("Processor Y", "Processor"),
        ("Supplier A", "Supplier"),
        ("Supplier B", "Supplier"),
        ("Supplier C", "Supplier"),
    ]
    sample_dependencies = [
        ("Raw Material P", "Processor X", "Raw material", 250.0),
        ("Raw Material Q", "Processor Y", "Raw material", 150.0),
        ("Processor X", "Supplier A", "Battery cells", 100.0),
        ("Processor X", "Supplier B", "Battery cells", 80.0),
        ("Processor Y", "Supplier C", "Battery cells", 120.0),
    ]

    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    def cursor_execute(query):
        if "NEXUS_DEPENDENCIES" in query:
            mock_cursor.description = [
                ("UPSTREAM_NODE",), ("DOWNSTREAM_NODE",), ("MATERIAL",), ("CAPACITY",)
            ]
            mock_cursor.fetchall.return_value = sample_dependencies
        elif "NEXUS_NODES" in query:
            mock_cursor.description = [("NODE_NAME",), ("NODE_TYPE",)]
            mock_cursor.fetchall.return_value = sample_nodes

    mock_cursor.execute.side_effect = cursor_execute
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_conn.__enter__.return_value = mock_conn

    config = SAPConnectionConfig(
        address="hana-cloud-host.hanacloud.ondemand.com",
        port=443,
        user="DBADMIN",
        password="TestPassword123!",
    )

    with patch("agents.sap_adapter.get_sap_connection", return_value=mock_conn):
        graph, dep_df, nodes_df = load_sap_supply_chain(config)

    assert graph.number_of_nodes() == 7
    assert graph.number_of_edges() == 5
    assert len(dep_df) == 5
    assert len(nodes_df) == 7

    # Run complete dashboard pipeline on SAP HANA graph
    sim = simulate_disruption(graph, failed_node="Processor X")
    fig = create_resilient_network_figure(graph, simulation_result=sim, selected_entity="Processor X")
    assert isinstance(fig, go.Figure)

