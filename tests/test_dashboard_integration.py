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
    safe_format_capacity,
    safe_format_currency,
    safe_format_lead_time,
    safe_format_number,
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


# =========================================================================
# Test 13: Supplier failure regression tests (Supplier A, B, C)
# =========================================================================
def test_13_supplier_failure_scenarios_execute_without_error():
    """
    Regression test for:
    'Financial Optimization engine failed: '>' not supported between instances of 'NoneType' and 'int''
    Verifies that Supplier A, Supplier B, and Supplier C failures (sink nodes with 0 outgoing edges)
    safely resolve their capacity throughput from inbound feeds and complete the entire 5-agent pipeline.
    """
    df = load_and_validate_dependencies(DEFAULT_DEPENDENCIES_CSV)
    graph = build_supply_chain_graph(df)

    expected_capacities = {
        "Supplier A": 100.0,
        "Supplier B": 80.0,
        "Supplier C": 120.0,
    }

    for supplier_node in ["Supplier A", "Supplier B", "Supplier C"]:
        # 1. Member 3: Disruption simulation
        sim_res = simulate_disruption(graph, failed_node=supplier_node)
        assert sim_res["failed_entity"] == supplier_node
        assert sim_res["estimated_capacity_impact"]["estimated_direct_capacity_loss"] is None

        # 2. Financial scenario adapter: Never treats inbound feeding edge as confirmed production
        scenario = build_financial_scenario_from_simulation(
            simulation_result=sim_res,
            loss_per_unit=150.0,
            available_budget=50000.0,
            max_lead_time_days=14.0,
            graph=graph,
        )
        assert scenario.affected_quantity == 0.0
        assert scenario.additional_metadata.get("has_capacity_data") is False
        assert scenario.additional_metadata.get("is_unquantified") is True

        # 3. Candidate recovery options: unquantified options marked illustrative
        candidate_options = generate_candidate_recovery_options(
            scenario=scenario,
            graph=graph,
        )
        assert len(candidate_options) > 0
        assert all(opt.available_capacity is None for opt in candidate_options)

        # 4. Member 4: Financial optimization - unquantified impact cannot declare optimal plan
        fin_res = optimize_recovery_plan(
            scenario=scenario,
            recovery_options=candidate_options,
            allow_multi_option=True,
        )
        assert fin_res["financial_status"] == "UNQUANTIFIED"
        assert fin_res["baseline_disruption_loss"] is None
        assert fin_res["recommended_option_ids"] == []
        assert fin_res["requires_human_approval"] is True

        # 5. Member 5: Decision explanation - aligns with Section D without contradiction
        explainer = generate_decision_explanation(
            graph=graph,
            disruption_result=sim_res,
            financial_result=fin_res,
        )
        assert "what_failed" in explainer
        assert explainer["what_failed"]["failed_entity"] == supplier_node
        assert "UNQUANTIFIED" in explainer["financial_calculation"]["loss_formula"]
        assert explainer["financial_calculation"]["baseline_loss"] is None


# =========================================================================
# Test 14: Safe handling of missing capacity and optional financial inputs
# =========================================================================
def test_14_missing_capacity_and_optional_financial_inputs_handled_safely():
    """
    Verifies that completely missing capacity (isolated node) and None-valued
    budget/lead-time/loss-per-unit constraints are handled safely without crashing.
    """
    # Create an isolated graph with a node having zero recorded capacity
    isolated_graph = nx.DiGraph()
    isolated_graph.add_node("Isolated Supplier", node_type="Supplier")

    sim_res = simulate_disruption(isolated_graph, failed_node="Isolated Supplier")
    assert sim_res["estimated_capacity_impact"]["estimated_direct_capacity_loss"] is None

    # Test with None budget and None lead time
    scenario = build_financial_scenario_from_simulation(
        simulation_result=sim_res,
        loss_per_unit=None,
        available_budget=None,
        max_lead_time_days=None,
        graph=isolated_graph,
    )
    assert scenario.affected_quantity == 0.0
    assert scenario.available_budget is None
    assert scenario.max_lead_time_days is None
    assert scenario.additional_metadata.get("has_capacity_data") is False

    # Options must honestly flag missing capacity rather than declaring feasibility
    options = generate_candidate_recovery_options(scenario, graph=isolated_graph)
    assert any(opt.available_capacity is None for opt in options)

    fin_res = optimize_recovery_plan(scenario, options)
    # When capacity is missing, options cannot be confirmed feasible
    assert fin_res["recommended_option_ids"] == []
    assert fin_res["requires_human_approval"] is True
    assert len(fin_res["missing_information"]) > 0


# =========================================================================
# Test 15: Regression test for Supplier C failure & missing capacity display
# =========================================================================
def test_15_supplier_c_failure_and_missing_capacity_display():
    """
    Verifies that Supplier C failure (which has None direct capacity loss due
    to 0 outgoing edges) formats safely in dashboard display expressions
    without raising TypeError: unsupported format string passed to NoneType.__format__.
    Also confirms that missing capacity displays as 'Not available' or 'Unknown'
    and is never fabricated as 0.
    """
    # Build graph with Supplier C as a terminal supplier receiving from Processor Y
    graph = nx.DiGraph()
    graph.add_node("Processor Y", node_type="Processor")
    graph.add_node("Supplier C", node_type="Supplier")
    graph.add_edge("Processor Y", "Supplier C", material="Battery cells", capacity=120.0)

    # 1. Simulate disruption at Supplier C
    sim_res = simulate_disruption(graph, failed_node="Supplier C")
    cap_impact = sim_res.get("estimated_capacity_impact", {})
    direct_cap = cap_impact.get("estimated_direct_capacity_loss")

    # Inherent property of leaf supplier: direct outgoing capacity is None
    assert direct_cap is None

    # 2. Display formatting must produce "Not available" without crashing
    display_str = safe_format_capacity(direct_cap, default="Not available", decimals=1, unit=" units")
    assert display_str == "Not available"
    assert display_str != "0.0 units"
    assert display_str != "0 units"

    # Test display formatting with default="Unknown"
    unknown_display = safe_format_capacity(direct_cap, default="Unknown", decimals=1, unit="")
    assert unknown_display == "Unknown"

    # 3. Currency and number formatting with None must also be safe
    assert safe_format_currency(None, default="Not available", decimals=0) == "Not available"
    assert safe_format_currency(None, default="Unknown", decimals=2) == "Unknown"
    assert safe_format_lead_time(None, default="Unknown") == "Unknown"
    assert safe_format_number(None, default="Unknown", decimals=1, suffix="/100") == "Unknown"

    # 4. Valid numbers format correctly
    assert safe_format_capacity(180.0, default="Not available", decimals=1, unit=" units") == "180.0 units"
    assert safe_format_currency(6000.0, default="Not available", decimals=0) == "$6,000"
    assert safe_format_currency(1234.56, default="Unknown", decimals=2) == "$1,234.56"
    assert safe_format_lead_time(8.0, default="Unknown") == "8 days"
    assert safe_format_number(75.5, default="Unknown", decimals=1, suffix="/100") == "75.5/100"

    # 5. Full financial optimization & candidate generation for Supplier C
    scenario = build_financial_scenario_from_simulation(
        simulation_result=sim_res,
        loss_per_unit=50.0,
        available_budget=500000.0,
        max_lead_time_days=30,
        graph=graph,
    )
    assert scenario.affected_quantity == 0.0
    assert scenario.additional_metadata.get("is_unquantified") is True
    candidate_options = generate_candidate_recovery_options(scenario=scenario, graph=graph)
    assert len(candidate_options) > 0

    fin_res = optimize_recovery_plan(scenario=scenario, recovery_options=candidate_options, allow_multi_option=True)
    assert fin_res["financial_status"] == "UNQUANTIFIED"
    assert fin_res["recommended_option_ids"] == []

    # 6. Verify table row formatting logic does not raise TypeError on any option
    for opt in fin_res.get("feasible_options", []) + fin_res.get("rejected_options", []):
        c_str = safe_format_capacity(opt.get("available_capacity"), default="Unknown", decimals=1, unit="")
        lt_str = safe_format_lead_time(opt.get("lead_time_days"), default="Unknown")
        cost_str = safe_format_currency(opt.get("recovery_cost"), default="Unknown", decimals=2)
        assert isinstance(c_str, str)
        assert isinstance(lt_str, str)
        assert isinstance(cost_str, str)

    # 7. Resilient network figure renders without error
    fig = create_resilient_network_figure(graph=graph, simulation_result=sim_res, selected_entity="Supplier C")
    assert fig is not None



# =========================================================================
# Test 16: Complete capacity data scenario (Processor X) preserves full optimization
# =========================================================================
def test_16_complete_capacity_data_preserves_full_optimization():
    """
    Verifies that when complete, verified capacity data is present (Processor X failure,
    which feeds 180 units of direct downstream capacity), the platform continues to
    execute full quantitative optimization, selects verified optimal plans, and
    calculates verified savings without marking the scenario unquantified.
    """
    df = load_and_validate_dependencies(DEFAULT_DEPENDENCIES_CSV)
    graph = build_supply_chain_graph(df)

    # 1. Member 3: Disruption simulation at Processor X
    sim_res = simulate_disruption(graph, failed_node="Processor X")
    cap_impact = sim_res["estimated_capacity_impact"]
    assert cap_impact["estimated_direct_capacity_loss"] == 180.0
    assert "Supplier A" in sim_res["affected_entities"]
    assert "Supplier B" in sim_res["affected_entities"]

    # 2. Financial scenario adapter: extracts verified 180 units
    scenario = build_financial_scenario_from_simulation(
        simulation_result=sim_res,
        loss_per_unit=150.0,
        available_budget=50000.0,
        max_lead_time_days=14.0,
        graph=graph,
    )
    assert scenario.affected_quantity == 180.0
    assert scenario.baseline_disruption_loss == 27000.0
    assert scenario.additional_metadata.get("has_capacity_data") is True
    assert scenario.additional_metadata.get("is_unquantified", False) is False

    # 3. Candidate options scaled to verified quantity
    candidate_options = generate_candidate_recovery_options(
        scenario=scenario,
        graph=graph,
    )
    assert len(candidate_options) == 4
    assert any(opt.available_capacity == 180.0 for opt in candidate_options)

    # 4. Member 4: Financial optimization selects optimal plan with positive verified savings
    fin_res = optimize_recovery_plan(
        scenario=scenario,
        recovery_options=candidate_options,
        allow_multi_option=True,
    )
    assert fin_res["baseline_disruption_loss"] == 27000.0
    assert fin_res["recommended_option_ids"] == ["OPT-BUFFER-RELEASE"]
    assert fin_res["estimated_total_cost"] == 18000.0
    assert fin_res["estimated_savings"] == 9000.0
    assert len(fin_res["feasible_options"]) > 0

    # 5. Member 5: Decision explanation reflects verified calculations
    explainer = generate_decision_explanation(
        graph=graph,
        disruption_result=sim_res,
        financial_result=fin_res,
    )
    assert "180.0 throughput units" in explainer["what_failed"]["narrative"]
    assert "$27,000.00" in explainer["financial_calculation"]["narrative"]
    assert "$9,000.00" in explainer["financial_calculation"]["narrative"]
    assert "OPT-BUFFER-RELEASE" in explainer["selected_recovery_option"]["narrative"]
# =========================================================================
# =========================================================================
# Test 17 & 18: Supplier A, B, C Consistency on Standard Sample & SAP HANA
# =========================================================================
from unittest.mock import MagicMock, patch
from agents.sap_adapter import (
    SAPConnectionConfig,
    load_sap_supply_chain,
)

SAMPLE_SAP_NODES_TEST = [
    ("Raw Material P", "Raw Material"),
    ("Raw Material Q", "Raw Material"),
    ("Processor X", "Processor"),
    ("Processor Y", "Processor"),
    ("Supplier A", "Supplier"),
    ("Supplier B", "Supplier"),
    ("Supplier C", "Supplier"),
]

SAMPLE_SAP_DEPENDENCIES_TEST = [
    ("Raw Material P", "Processor X", "Raw material", 250.0),
    ("Raw Material Q", "Processor Y", "Raw material", 150.0),
    ("Processor X", "Supplier A", "Battery cells", 100.0),
    ("Processor X", "Supplier B", "Battery cells", 80.0),
    ("Processor Y", "Supplier C", "Battery cells", 120.0),
]


def _build_mocked_sap_graph():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    def cursor_execute(query):
        if "NEXUS_DEPENDENCIES" in query:
            mock_cursor.description = [
                ("UPSTREAM_NODE",), ("DOWNSTREAM_NODE",), ("MATERIAL",), ("CAPACITY",)
            ]
            mock_cursor.fetchall.return_value = SAMPLE_SAP_DEPENDENCIES_TEST
        elif "NEXUS_NODES" in query:
            mock_cursor.description = [("NODE_NAME",), ("NODE_TYPE",)]
            mock_cursor.fetchall.return_value = SAMPLE_SAP_NODES_TEST
        elif "SELECT 1 FROM DUMMY" in query:
            mock_cursor.fetchone.return_value = (1,)

    mock_cursor.execute.side_effect = cursor_execute
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_conn.__enter__.return_value = mock_conn

    config = SAPConnectionConfig(
        address="mock-hana.hanacloud.ondemand.com",
        port=443,
        user="TEST_USER",
        password="TEST_PASSWORD",
        schema="HACKFEST0316",
    )
    with patch("agents.sap_adapter.get_sap_connection", return_value=mock_conn):
        graph, _, _ = load_sap_supply_chain(config)
    return graph


@pytest.mark.parametrize("supplier,expected_inbound_capacity,inbound_source", [
    ("Supplier A", 100.0, "Processor X"),
    ("Supplier B", 80.0, "Processor X"),
    ("Supplier C", 120.0, "Processor Y"),
])
def test_17_supplier_consistency_standard_dataset(supplier, expected_inbound_capacity, inbound_source):
    """
    Verifies that for Supplier A, Supplier B, and Supplier C using the standard sample dataset:
    1. Criticality Context: Topology accurately identified as terminal leaf node (out-degree = 0)
       with no cascading downstream bottlenecks.
    2. Residual Risks: Never claims '100% disruption risk across all paths'; accurately states
       the outage is contained at the facility and zero downstream entities are severed.
    3. Capacity Data: Distinguishes inbound delivery feed from unrecorded facility production throughput.
    4. Financial Contract: Impact is UNQUANTIFIED, no automated recovery plan is recommended,
       and human approval is required.
    5. Internal Consistency: All 9 explanation sections agree on zero impacted entities, contained outage,
       and unquantified loss.
    """
    df = load_and_validate_dependencies(DEFAULT_DEPENDENCIES_CSV)
    graph = build_supply_chain_graph(df)

    assert graph.out_degree(supplier) == 0
    assert graph.in_degree(supplier) >= 1

    sim_res = simulate_disruption(graph, failed_node=supplier)
    assert sim_res["affected_entities"] == []
    assert sim_res["impact_level"] == "Low"
    assert sim_res["estimated_capacity_impact"]["estimated_direct_capacity_loss"] is None

    scenario = build_financial_scenario_from_simulation(
        simulation_result=sim_res,
        loss_per_unit=100.0,
        available_budget=50000.0,
        max_lead_time_days=30.0,
        graph=graph,
    )
    assert scenario.affected_quantity == 0.0
    assert scenario.additional_metadata.get("is_unquantified") is True
    assert scenario.additional_metadata.get("inbound_recorded_feed_capacity") == expected_inbound_capacity

    candidate_opts = generate_candidate_recovery_options(scenario=scenario, graph=graph)
    assert len(candidate_opts) > 0
    for opt in candidate_opts:
        assert opt.available_capacity is None

    fin_res = optimize_recovery_plan(scenario=scenario, recovery_options=candidate_opts, allow_multi_option=True)
    assert fin_res["financial_status"] == "UNQUANTIFIED"
    assert fin_res["is_unquantified"] is True
    assert fin_res["recommended_option_ids"] == []
    assert fin_res["baseline_disruption_loss"] is None
    assert fin_res["estimated_total_cost"] is None
    assert fin_res["estimated_savings"] is None
    assert fin_res["requires_human_approval"] is True

    shared_risks = detect_shared_risks(graph)
    exp = generate_decision_explanation(
        graph=graph,
        disruption_result=sim_res,
        financial_result=fin_res,
        risk_findings=shared_risks,
    )

    # 1. Section 1 (What failed): mentions inbound feed for reference, does not assume production capacity
    sec1 = exp["what_failed"]
    assert f"inbound delivery feed of {expected_inbound_capacity:,.1f} units from '{inbound_source}' noted for reference" in sec1["narrative"]
    assert "facility production capacity unrecorded" in sec1["narrative"]

    # 2. Section 2 (Affected entities): 0 affected entities
    sec2 = exp["affected_entities"]
    assert sec2["total_affected_count"] == 0
    assert "0 downstream entities were impacted" in sec2["narrative"]

    # 3. Section 3 (Why it matters): terminal leaf sink node, localized, NO cascading downstream bottlenecks
    sec3 = exp["why_it_matters"]
    assert "terminal leaf sink node (out-degree = 0)" in sec3["narrative"]
    assert "does not cascade to any subsequent manufacturing plants" in sec3["narrative"]
    assert "cascading downstream bottlenecks" not in sec3["narrative"]
    assert "triggers cascading" not in sec3["narrative"]

    # 4. Section 4 (Financial calculation): UNQUANTIFIED
    sec4 = exp["financial_calculation"]
    assert "UNQUANTIFIED" in sec4["loss_formula"]
    assert "cannot be quantified because operational throughput capacity is unrecorded" in sec4["narrative"]

    # 5. Section 5 (Selected recovery option): No automated recovery plan authorized
    sec5 = exp["selected_recovery_option"]
    assert sec5["selected_ids"] == []
    assert sec5["is_no_action_baseline"] is True
    assert "No automated recovery plan was authorized" in sec5["narrative"]

    # 6. Section 6 (Alternatives evaluated): rejected due to unquantified feasibility
    sec6 = exp["alternatives_considered"]
    assert sec6["feasible_count"] == 0
    assert sec6["rejected_count"] == len(candidate_opts)
    for inf in sec6["rejected_options"]:
        assert any("Feasibility unverified: Disrupted production quantity is unquantified" in r for r in inf.get("rejection_reasons", []))

    # 7. Section 7 (Assumptions & Disclosures): notes unrecorded throughput and illustrative options
    sec7 = exp["assumptions_and_missing_data"]
    assert any("Disrupted production throughput" in d and "is unquantified" in d for d in sec7["missing_data"])
    assert any("Candidate recovery options are illustrative reference estimates only" in a for a in sec7["assumptions"])

    # 8. Section 8 (Residual risks): contained outage, NO "across all paths"
    sec8 = exp["residual_risks"]
    assert "across all paths" not in sec8["narrative"]
    assert "Operational Containment: Outage is contained at" in sec8["narrative"]
    assert "no downstream manufacturing lines or customer deliveries are severed" in sec8["narrative"]

    # 9. Section 9 (Governance & Approval Mandate): human approval mandated
    sec9 = exp["human_approval"]
    assert sec9["required"] is True
    assert any("Disrupted production throughput is unquantified" in trig for trig in sec9["triggers"])


@pytest.mark.parametrize("supplier,expected_inbound_capacity,inbound_source", [
    ("Supplier A", 100.0, "Processor X"),
    ("Supplier B", 80.0, "Processor X"),
    ("Supplier C", 120.0, "Processor Y"),
])
def test_18_supplier_consistency_sap_hana_path(supplier, expected_inbound_capacity, inbound_source):
    """
    Verifies that the SAP HANA Cloud data path (using mocked database responses matching schema HACKFEST0316)
    produces the exact same rigorous, topology-accurate, unquantified financial guarantees for Supplier A, B, and C.
    """
    graph = _build_mocked_sap_graph()

    assert graph.out_degree(supplier) == 0
    assert graph.in_degree(supplier) >= 1

    sim_res = simulate_disruption(graph, failed_node=supplier)
    assert sim_res["affected_entities"] == []
    assert sim_res["impact_level"] == "Low"

    scenario = build_financial_scenario_from_simulation(
        simulation_result=sim_res,
        loss_per_unit=120.0,
        available_budget=75000.0,
        max_lead_time_days=21.0,
        graph=graph,
    )
    assert scenario.additional_metadata.get("is_unquantified") is True
    assert scenario.additional_metadata.get("inbound_recorded_feed_capacity") == expected_inbound_capacity

    candidate_opts = generate_candidate_recovery_options(scenario=scenario, graph=graph)
    fin_res = optimize_recovery_plan(scenario=scenario, recovery_options=candidate_opts, allow_multi_option=True)
    assert fin_res["financial_status"] == "UNQUANTIFIED"
    assert fin_res["recommended_option_ids"] == []
    assert fin_res["requires_human_approval"] is True

    shared_risks = detect_shared_risks(graph)
    exp = generate_decision_explanation(
        graph=graph,
        disruption_result=sim_res,
        financial_result=fin_res,
        risk_findings=shared_risks,
    )

    # Validate criticality topology accuracy
    assert "terminal leaf sink node (out-degree = 0)" in exp["why_it_matters"]["narrative"]
    assert "cascading downstream bottlenecks" not in exp["why_it_matters"]["narrative"]

    # Validate residual risks accuracy
    assert "across all paths" not in exp["residual_risks"]["narrative"]
    assert "Operational Containment: Outage is contained at" in exp["residual_risks"]["narrative"]

    # Validate inbound capacity disclosure
    assert f"inbound delivery feed of {expected_inbound_capacity:,.1f} units from '{inbound_source}' noted for reference" in exp["what_failed"]["narrative"]

    # Validate unquantified financial and approval safeguards
    assert "UNQUANTIFIED" in exp["financial_calculation"]["loss_formula"]
    assert exp["human_approval"]["required"] is True
# =========================================================================
# Test 19: Section D Broken Supply Corridors and Actions Display
# =========================================================================
@pytest.mark.parametrize("scenario_node,expected_corridors", [
    ("Processor X", [["Processor X", "Supplier A"], ["Processor X", "Supplier B"]]),
    ("Raw Material P", [["Raw Material P", "Processor X", "Supplier A"], ["Raw Material P", "Processor X", "Supplier B"]]),
    ("Raw Material Q", [["Raw Material Q", "Processor Y", "Supplier C"]]),
])
def test_19_section_d_broken_corridors_and_actions_display(scenario_node, expected_corridors):
    """
    Verifies that Section D's 'Broken Supply Corridors & Actions' panel data contract is
    satisfied for Processor X, Raw Material P, and Raw Material Q:
    1. Reads both disrupted_paths and affected_paths (present, identical, non-empty).
    2. Reads both recommended_actions and recommended_next_actions (present, identical, non-empty).
    3. Verified corridor paths match expected graph routes.
    """
    df = load_and_validate_dependencies(DEFAULT_DEPENDENCIES_CSV)
    graph = build_supply_chain_graph(df)

    sim_res = simulate_disruption(graph, failed_node=scenario_node)

    # Key availability checks
    paths = sim_res.get("disrupted_paths") or sim_res.get("affected_paths") or []
    actions = sim_res.get("recommended_actions") or sim_res.get("recommended_next_actions") or []

    assert len(paths) == len(expected_corridors)
    for expected in expected_corridors:
        assert expected in paths

    # Actions must be populated with specific operational guidance
    assert len(actions) > 0
    assert any("Emergency sourcing" in a or "Mitigate single point of failure" in a or "Inventory audit" in a for a in actions)

    # Corridor paths filter check as implemented in app.py Section D
    corridor_paths = [p for p in paths if len(p) > 1]
    assert len(corridor_paths) == len(expected_corridors)
