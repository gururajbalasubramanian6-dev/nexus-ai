"""
Unit Tests for Supply Chain Mapping Agent (Member 1)
Nexus-AI Supply Chain Resilience Project

Tests verify:
- Sample CSV loads successfully.
- Directed graph contains expected nodes, edges, and preserved metadata.
- Processor X connects to Supplier A and Supplier B.
- Missing node names are rejected with descriptive errors.
- Invalid capacities (negative, zero, non-numeric, blank) are rejected.
- Duplicate supply relationships are rejected.
- Missing columns or files are rejected.
- Interactive visualization generates properly.
"""

from pathlib import Path
import tempfile
import unittest

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from agents.mapping_agent import (
    DataValidationError,
    build_supply_chain_graph,
    classify_node,
    create_interactive_graph,
    get_graph_summary,
    get_supply_chain_network,
    load_and_validate_dependencies,
)


class TestMappingAgent(unittest.TestCase):
    """Test suite for data validation, graph construction, and visualization."""

    def setUp(self):
        """Prepare paths and sample data for testing."""
        self.workspace_dir = Path(__file__).resolve().parent.parent
        self.sample_csv = self.workspace_dir / "data" / "dependencies.csv"

    # =========================================================================
    # Task 6 Requirement 1: The sample CSV loads successfully
    # =========================================================================
    def test_sample_csv_loads_successfully(self):
        """Test that data/dependencies.csv loads without error and is valid."""
        df = load_and_validate_dependencies(self.sample_csv)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5, "Sample CSV should contain exactly 5 rows.")
        expected_cols = ["upstream_node", "downstream_node", "material", "capacity"]
        self.assertListEqual(list(df.columns), expected_cols)

    # =========================================================================
    # Task 6 Requirement 2: The graph contains all expected nodes and edges
    # =========================================================================
    def test_graph_contains_expected_nodes_and_edges(self):
        """Test that the NetworkX DiGraph has the exact 7 nodes and 5 edges."""
        df = load_and_validate_dependencies(self.sample_csv)
        G = build_supply_chain_graph(df)

        expected_nodes = {
            "Processor X",
            "Processor Y",
            "Supplier A",
            "Supplier B",
            "Supplier C",
            "Raw Material P",
            "Raw Material Q",
        }
        self.assertEqual(set(G.nodes()), expected_nodes)
        self.assertEqual(G.number_of_nodes(), 7)
        self.assertEqual(G.number_of_edges(), 5)

        # Verify edge attributes preserved
        self.assertEqual(G["Processor X"]["Supplier A"]["material"], "Battery cells")
        self.assertEqual(G["Processor X"]["Supplier A"]["capacity"], 100.0)
        self.assertEqual(G["Raw Material P"]["Processor X"]["material"], "Raw material")
        self.assertEqual(G["Raw Material P"]["Processor X"]["capacity"], 250.0)

    # =========================================================================
    # Task 6 Requirement 3: Processor X connects to Supplier A and Supplier B
    # =========================================================================
    def test_processor_x_connections(self):
        """Verify Processor X connections (downstream to Supplier A/B, upstream from Raw Material P)."""
        df = load_and_validate_dependencies(self.sample_csv)
        G = build_supply_chain_graph(df)

        # Processor X should have outgoing edges to Supplier A and Supplier B
        downstream_nodes = list(G.successors("Processor X"))
        self.assertIn("Supplier A", downstream_nodes)
        self.assertIn("Supplier B", downstream_nodes)
        self.assertEqual(len(downstream_nodes), 2)

        # Raw Material P should connect upstream to Processor X
        upstream_nodes = list(G.predecessors("Processor X"))
        self.assertIn("Raw Material P", upstream_nodes)
        self.assertEqual(len(upstream_nodes), 1)

    # =========================================================================
    # Task 6 Requirement 4: Missing node names are rejected
    # =========================================================================
    def test_missing_upstream_node_rejected(self):
        """Verify that an empty or missing upstream_node raises DataValidationError."""
        csv_content = (
            "upstream_node,downstream_node,material,capacity\n"
            ",Supplier A,Battery cells,100\n"
        )
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write(csv_content)
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(DataValidationError) as ctx:
                load_and_validate_dependencies(tmp_path)
            self.assertIn("Missing or blank node names", str(ctx.exception))
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_missing_downstream_node_rejected(self):
        """Verify that an empty or missing downstream_node raises DataValidationError."""
        csv_content = (
            "upstream_node,downstream_node,material,capacity\n"
            "Processor X,   ,Battery cells,100\n"
        )
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write(csv_content)
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(DataValidationError) as ctx:
                load_and_validate_dependencies(tmp_path)
            self.assertIn("Missing or blank node names", str(ctx.exception))
        finally:
            tmp_path.unlink(missing_ok=True)

    # =========================================================================
    # Task 6 Requirement 5: Invalid capacities are rejected
    # =========================================================================
    def test_non_numeric_capacity_rejected(self):
        """Verify that non-numeric capacity raises DataValidationError."""
        csv_content = (
            "upstream_node,downstream_node,material,capacity\n"
            "Processor X,Supplier A,Battery cells,NOT_A_NUMBER\n"
        )
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write(csv_content)
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(DataValidationError) as ctx:
                load_and_validate_dependencies(tmp_path)
            self.assertIn("Invalid capacity values", str(ctx.exception))
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_negative_capacity_rejected(self):
        """Verify that negative capacity raises DataValidationError."""
        csv_content = (
            "upstream_node,downstream_node,material,capacity\n"
            "Processor X,Supplier A,Battery cells,-50\n"
        )
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write(csv_content)
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(DataValidationError) as ctx:
                load_and_validate_dependencies(tmp_path)
            self.assertIn("Invalid capacity values", str(ctx.exception))
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_zero_capacity_rejected(self):
        """Verify that zero capacity raises DataValidationError."""
        csv_content = (
            "upstream_node,downstream_node,material,capacity\n"
            "Processor X,Supplier A,Battery cells,0\n"
        )
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write(csv_content)
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(DataValidationError) as ctx:
                load_and_validate_dependencies(tmp_path)
            self.assertIn("Invalid capacity values", str(ctx.exception))
        finally:
            tmp_path.unlink(missing_ok=True)

    # =========================================================================
    # Task 6 Requirement 6: Duplicate relationships are handled appropriately
    # =========================================================================
    def test_duplicate_relationships_rejected(self):
        """Verify that duplicate (upstream_node, downstream_node) raises DataValidationError."""
        csv_content = (
            "upstream_node,downstream_node,material,capacity\n"
            "Processor X,Supplier A,Battery cells,100\n"
            "Processor X,Supplier A,Battery cells,80\n"
        )
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write(csv_content)
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(DataValidationError) as ctx:
                load_and_validate_dependencies(tmp_path)
            self.assertIn("Duplicate supply relationship", str(ctx.exception))
        finally:
            tmp_path.unlink(missing_ok=True)

    # =========================================================================
    # Additional Robustness Tests
    # =========================================================================
    def test_missing_required_column_rejected(self):
        """Verify that CSV missing a required header column raises DataValidationError."""
        csv_content = (
            "upstream_node,downstream_node,material\n"
            "Processor X,Supplier A,Battery cells\n"
        )
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write(csv_content)
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(DataValidationError) as ctx:
                load_and_validate_dependencies(tmp_path)
            self.assertIn("missing required column", str(ctx.exception).lower())
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_nonexistent_file_raises_error(self):
        """Verify that a non-existent file path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_and_validate_dependencies("data/non_existent_file.csv")

    def test_node_classification(self):
        """Verify node tier classification logic."""
        self.assertEqual(classify_node("Raw Material P", 0, 1), "Raw Material")
        self.assertEqual(classify_node("Processor X", 1, 2), "Processor")
        self.assertEqual(classify_node("Supplier A", 1, 0), "Supplier")

    def test_interactive_visualization_creation(self):
        """Verify Plotly figure generation and HTML export."""
        df = load_and_validate_dependencies(self.sample_csv)
        G = build_supply_chain_graph(df)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            tmp_html = Path(tmp.name)

        try:
            fig = create_interactive_graph(G, output_html=tmp_html)
            self.assertIsInstance(fig, go.Figure)
            self.assertTrue(tmp_html.exists())
            self.assertGreater(tmp_html.stat().st_size, 1000)  # Contains full HTML
        finally:
            tmp_html.unlink(missing_ok=True)

    def test_graph_summary(self):
        """Verify summary calculations."""
        df = load_and_validate_dependencies(self.sample_csv)
        G = build_supply_chain_graph(df)
        summary = get_graph_summary(G)

        self.assertEqual(summary["num_nodes"], 7)
        self.assertEqual(summary["num_edges"], 5)
        self.assertEqual(summary["total_capacity"], 700.0)


    def test_get_supply_chain_network(self):
        """Verify reusable get_supply_chain_network function for teammate integration."""
        # Test default call
        G = get_supply_chain_network()
        self.assertIsInstance(G, nx.DiGraph)
        self.assertEqual(G.number_of_nodes(), 7)
        self.assertEqual(G.number_of_edges(), 5)

        # Test explicit path call
        G_explicit = get_supply_chain_network(self.sample_csv)
        self.assertEqual(G_explicit.number_of_nodes(), 7)


if __name__ == "__main__":
    unittest.main()

