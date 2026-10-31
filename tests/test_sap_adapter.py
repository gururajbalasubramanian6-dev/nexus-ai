"""
Tests for SAP HANA Cloud Database Adapter (Phase 2)
====================================================

Verifies:
1. Secure configuration loading and host:port parsing.
2. Protection against credential exposure (repr masking and error sanitization).
3. Record validation for NEXUS_NODES and NEXUS_DEPENDENCIES.
4. End-to-end graph construction using mocked SAP HANA Cloud data.
5. Flawless interoperability with all 5 Nexus-AI agents.
"""

from unittest.mock import MagicMock, patch
import networkx as nx
import pandas as pd
import pytest

from agents.sap_adapter import (
    SAPConfigurationError,
    SAPConnectionConfig,
    SAPConnectionError,
    SAPDataError,
    _sanitize_error_message,
    _sanitize_identifier,
    fetch_nexus_dependencies,
    fetch_nexus_nodes,
    get_sap_connection,
    is_sap_configured,
    load_sap_supply_chain,
    test_sap_connection as check_sap_connection,
    validate_sap_dependencies,
    validate_sap_nodes,
)
from agents.mapping_agent import create_interactive_graph
from agents.risk_detection_agent import detect_shared_risks
from agents.disruption_simulation_agent import simulate_disruption
from agents.financial_optimization_agent import DisruptionScenario, RecoveryOption, optimize_recovery_plan
from agents.decision_explanation_agent import DecisionExplanationAgent


# Standard sample records matching the user's populated SAP tables
SAMPLE_SAP_NODES = [
    ("Raw Material P", "Raw Material"),
    ("Raw Material Q", "Raw Material"),
    ("Processor X", "Processor"),
    ("Processor Y", "Processor"),
    ("Supplier A", "Supplier"),
    ("Supplier B", "Supplier"),
    ("Supplier C", "Supplier"),
]

SAMPLE_SAP_DEPENDENCIES = [
    ("Raw Material P", "Processor X", "Raw material", 250.0),
    ("Raw Material Q", "Processor Y", "Raw material", 150.0),
    ("Processor X", "Supplier A", "Battery cells", 100.0),
    ("Processor X", "Supplier B", "Battery cells", 80.0),
    ("Processor Y", "Supplier C", "Battery cells", 120.0),
]


class TestSAPConfiguration:
    """Tests configuration loading, host/port parsing, and secret protection."""

    def test_config_from_env_clean(self, monkeypatch):
        monkeypatch.setenv("HANA_ADDRESS", "my-instance.hanacloud.ondemand.com")
        monkeypatch.setenv("HANA_PORT", "443")
        monkeypatch.setenv("HANA_USER", "DBADMIN")
        monkeypatch.setenv("HANA_PASSWORD", "SuperSecret123!")
        monkeypatch.setenv("HANA_SCHEMA", "HACKFEST0316")

        config = SAPConnectionConfig.from_env()
        assert config.address == "my-instance.hanacloud.ondemand.com"
        assert config.port == 443
        assert config.user == "DBADMIN"
        assert config.password == "SuperSecret123!"
        assert config.schema == "HACKFEST0316"
        assert config.encrypt is True

    def test_config_parses_host_and_custom_port_from_endpoint(self, monkeypatch):
        """Verifies that custom ports like 30015 in host:port strings are correctly parsed."""
        monkeypatch.setenv("HANA_ADDRESS", "custom-instance.hanacloud.ondemand.com:30015")
        monkeypatch.delenv("HANA_PORT", raising=False)
        monkeypatch.setenv("HANA_USER", "TEST_USER")
        monkeypatch.setenv("HANA_PASSWORD", "SecretPass")

        config = SAPConnectionConfig.from_env()
        assert config.address == "custom-instance.hanacloud.ondemand.com"
        assert config.port == 30015

    def test_config_explicit_port_overrides_endpoint_port(self, monkeypatch):
        monkeypatch.setenv("HANA_ADDRESS", "custom-instance.hanacloud.ondemand.com:443")
        monkeypatch.setenv("HANA_PORT", "39015")
        monkeypatch.setenv("HANA_USER", "TEST_USER")
        monkeypatch.setenv("HANA_PASSWORD", "SecretPass")

        config = SAPConnectionConfig.from_env()
        assert config.address == "custom-instance.hanacloud.ondemand.com"
        assert config.port == 39015

    def test_config_masks_password_in_repr(self):
        config = SAPConnectionConfig(
            address="test-host",
            port=443,
            user="admin",
            password="UltraConfidentialPassword999",
        )
        rep = repr(config)
        assert "UltraConfidentialPassword999" not in rep
        assert "password='***'" in rep

    def test_missing_address_raises_config_error(self, monkeypatch):
        monkeypatch.delenv("HANA_ADDRESS", raising=False)
        monkeypatch.delenv("SAP_HANA_HOST", raising=False)
        monkeypatch.setenv("HANA_USER", "admin")
        monkeypatch.setenv("HANA_PASSWORD", "pass")

        with pytest.raises(SAPConfigurationError, match="Missing SAP HANA Cloud host"):
            SAPConnectionConfig.from_env()

    def test_sanitizes_error_message(self):
        config = SAPConnectionConfig("host", 443, "user", "MyPasswordXYZ")
        msg = "Error connecting with password='MyPasswordXYZ' on host"
        sanitized = _sanitize_error_message(msg, config)
        assert "MyPasswordXYZ" not in sanitized
        assert "***" in sanitized

    def test_sanitize_identifier(self):
        assert _sanitize_identifier("HACKFEST0316") == "HACKFEST0316"
        assert _sanitize_identifier("my_schema_1") == "my_schema_1"
        assert _sanitize_identifier("SCHEMA; DROP TABLE;") == "SCHEMADROPTABLE"
        with pytest.raises(ValueError):
            _sanitize_identifier(";;;")


class TestSAPDataValidation:
    """Tests validation of records received from SAP HANA."""

    def test_valid_sap_dependencies(self):
        raw_df = pd.DataFrame(
            SAMPLE_SAP_DEPENDENCIES,
            columns=["UPSTREAM_NODE", "DOWNSTREAM_NODE", "MATERIAL", "CAPACITY"],
        )
        validated = validate_sap_dependencies(raw_df)
        assert list(validated.columns) == ["upstream_node", "downstream_node", "material", "capacity"]
        assert len(validated) == 5
        assert validated["capacity"].dtype == float

    def test_empty_dependencies_rejected(self):
        empty_df = pd.DataFrame(columns=["UPSTREAM_NODE", "DOWNSTREAM_NODE", "MATERIAL", "CAPACITY"])
        with pytest.raises(SAPDataError, match="0 records"):
            validate_sap_dependencies(empty_df)

    def test_missing_column_rejected(self):
        df = pd.DataFrame({"UPSTREAM_NODE": ["A"], "DOWNSTREAM_NODE": ["B"]})
        with pytest.raises(SAPDataError, match="missing required column"):
            validate_sap_dependencies(df)

    def test_negative_capacity_rejected(self):
        df = pd.DataFrame({
            "UPSTREAM_NODE": ["A"],
            "DOWNSTREAM_NODE": ["B"],
            "MATERIAL": ["Steel"],
            "CAPACITY": [-50.0],
        })
        with pytest.raises(SAPDataError, match="Invalid capacity"):
            validate_sap_dependencies(df)

    def test_blank_node_names_rejected(self):
        df = pd.DataFrame({
            "UPSTREAM_NODE": [""],
            "DOWNSTREAM_NODE": ["B"],
            "MATERIAL": ["Steel"],
            "CAPACITY": [100.0],
        })
        with pytest.raises(SAPDataError, match="missing or blank"):
            validate_sap_dependencies(df)

    def test_duplicate_dependency_rejected(self):
        df = pd.DataFrame({
            "UPSTREAM_NODE": ["A", "A"],
            "DOWNSTREAM_NODE": ["B", "B"],
            "MATERIAL": ["Steel", "Steel"],
            "CAPACITY": [100.0, 150.0],
        })
        with pytest.raises(SAPDataError, match="Duplicate supply relationship"):
            validate_sap_dependencies(df)

    def test_valid_sap_nodes(self):
        raw_df = pd.DataFrame(SAMPLE_SAP_NODES, columns=["NODE_NAME", "NODE_TYPE"])
        validated = validate_sap_nodes(raw_df)
        assert list(validated.columns) == ["node_name", "node_type"]
        assert len(validated) == 7

    def test_blank_node_name_in_nodes_rejected(self):
        df = pd.DataFrame({"NODE_NAME": ["  "], "NODE_TYPE": ["Processor"]})
        with pytest.raises(SAPDataError, match="missing or blank"):
            validate_sap_nodes(df)


class TestSAPGraphConstructionAndAgentInterop:
    """Tests building the NetworkX graph from mocked SAP queries and feeding all 5 agents."""

    @pytest.fixture
    def mock_db_connection(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        def cursor_execute(query):
            if "NEXUS_DEPENDENCIES" in query:
                mock_cursor.description = [
                    ("UPSTREAM_NODE",), ("DOWNSTREAM_NODE",), ("MATERIAL",), ("CAPACITY",)
                ]
                mock_cursor.fetchall.return_value = SAMPLE_SAP_DEPENDENCIES
            elif "NEXUS_NODES" in query:
                mock_cursor.description = [("NODE_NAME",), ("NODE_TYPE",)]
                mock_cursor.fetchall.return_value = SAMPLE_SAP_NODES
            elif "SELECT 1 FROM DUMMY" in query:
                mock_cursor.fetchone.return_value = (1,)

        mock_cursor.execute.side_effect = cursor_execute
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        return mock_conn

    def test_load_sap_supply_chain_mocked(self, mock_db_connection):
        config = SAPConnectionConfig(
            address="test-hana.hanacloud.ondemand.com",
            port=443,
            user="TEST_USER",
            password="TEST_PASSWORD",
            schema="HACKFEST0316",
        )

        with patch("agents.sap_adapter.get_sap_connection", return_value=mock_db_connection):
            graph, dep_df, nodes_df = load_sap_supply_chain(config)

        # Verify graph topology
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() == 7
        assert graph.number_of_edges() == 5

        # Verify enriched node types from NEXUS_NODES
        assert graph.nodes["Raw Material P"]["node_type"] == "Raw Material"
        assert graph.nodes["Processor X"]["node_type"] == "Processor"
        assert graph.nodes["Supplier A"]["node_type"] == "Supplier"

        # Verify edge attributes
        edge_data = graph["Raw Material P"]["Processor X"]
        assert edge_data["material"] == "Raw material"
        assert edge_data["capacity"] == 250.0

    def test_all_five_agents_interoperate_with_sap_graph(self, mock_db_connection):
        """Verifies that Member 1, 2, 3, 4, 5 execute without modifications on the SAP graph."""
        config = SAPConnectionConfig(
            address="test-hana.hanacloud.ondemand.com",
            port=443,
            user="TEST_USER",
            password="TEST_PASSWORD",
        )

        with patch("agents.sap_adapter.get_sap_connection", return_value=mock_db_connection):
            graph, _, _ = load_sap_supply_chain(config)

        # 1. Member 1: Visualization figure creation
        fig = create_interactive_graph(graph)
        assert fig is not None

        # 2. Member 2: Shared Risk Detection
        risk_findings = detect_shared_risks(graph)
        assert isinstance(risk_findings, list)
        # Processor X supplies Supplier A and Supplier B, so it must be identified as shared
        shared_nodes = [r["dependency_node"] for r in risk_findings if r.get("is_shared")]
        assert "Processor X" in shared_nodes

        # 3. Member 3: Disruption Simulation
        disruption_result = simulate_disruption(graph, failed_node="Processor X")
        assert disruption_result["failed_entity"] == "Processor X"
        assert "Supplier A" in disruption_result["affected_entities"]
        assert "Supplier B" in disruption_result["affected_entities"]

        # 4. Member 4: Financial Optimization
        scenario = DisruptionScenario(
            scenario_id="SAP-SCENARIO-01",
            affected_quantity=180.0,
            loss_per_unit=500.0,
            disrupted_node="Processor X",
            available_budget=50000.0,
        )
        candidate_options = [
            RecoveryOption(
                option_id="OPT-1",
                option_type="alternate_supplier",
                description="Secondary Supplier Buffer",
                recovery_cost=10000.0,
                available_capacity=180.0,
                lead_time_days=3.0,
            )
        ]
        financial_result = optimize_recovery_plan(scenario, candidate_options)
        assert "recommended_option_ids" in financial_result
        assert financial_result["recommended_option_ids"] == ["OPT-1"]

        # 5. Member 5: Decision Explanation Agent
        explainer = DecisionExplanationAgent()
        explanation = explainer.explain(
            graph=graph,
            disruption_result=disruption_result,
            financial_result=financial_result,
            risk_findings=risk_findings,
        )
        assert "what_failed" in explanation
        assert "affected_entities" in explanation

    def test_connection_test_helper_success(self, mock_db_connection):
        config = SAPConnectionConfig("host", 443, "user", "pass")
        with patch("agents.sap_adapter.get_sap_connection", return_value=mock_db_connection):
            ok, msg = check_sap_connection(config)
            assert ok is True
            assert "Successfully connected" in msg

    def test_connection_test_helper_failure(self):
        config = SAPConnectionConfig("invalid-host", 443, "user", "pass")
        with patch("agents.sap_adapter.get_sap_connection", side_effect=SAPConnectionError("Host unreachable")):
            ok, msg = check_sap_connection(config)
            assert ok is False
            assert "Host unreachable" in msg
