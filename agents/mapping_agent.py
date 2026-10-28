"""
Nexus-AI: Supply Chain Mapping Agent
Member 1: Supply Chain Mapping Component

This module is responsible for:
1. Validating and ingesting supply chain dependency data from CSV files.
2. Constructing a directed graph of supply chain relationships using NetworkX.
3. Creating an interactive visualization using Plotly with distinct tiers,
   directional flow indicators, and detailed hover metadata.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import math
import networkx as nx
import pandas as pd
import plotly.graph_objects as go


# Required CSV columns
REQUIRED_COLUMNS: List[str] = [
    "upstream_node",
    "downstream_node",
    "material",
    "capacity",
]

# Default data path for supply chain dependencies
DEFAULT_DEPENDENCIES_CSV: Path = (
    Path(__file__).resolve().parent.parent / "data" / "dependencies.csv"
)

# Color scheme for node tiers
TIER_COLORS: Dict[str, str] = {
    "Raw Material": "#E65100",  # Deep Amber/Orange
    "Processor": "#0288D1",     # Vibrant Blue
    "Supplier": "#2E7D32",      # Emerald Green
    "Other": "#546E7A",         # Slate Grey
}

# Node marker symbols
TIER_SYMBOLS: Dict[str, str] = {
    "Raw Material": "diamond",
    "Processor": "square",
    "Supplier": "circle",
    "Other": "circle",
}


class DataValidationError(ValueError):
    """Raised when the supply chain dependencies data fails validation."""
    pass


def is_blank_or_na(series: pd.Series) -> pd.Series:
    """Returns a boolean Series that is True if a value is NA/null, or empty/whitespace string."""
    is_na = series.isna()
    non_na = series.dropna()
    is_empty_str = non_na.astype(str).str.strip().isin(["", "nan", "None", "null", "NaN"])
    return is_na | is_empty_str.reindex(series.index, fill_value=False)


def load_and_validate_dependencies(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Reads and validates a supply chain dependencies CSV file.

    Validation rules:
    - File must exist and be readable.
    - Must contain required columns: upstream_node, downstream_node, material, capacity.
    - Node names cannot be missing, null, or empty whitespace.
    - Capacity must be a valid positive numeric value (> 0).
    - No duplicate relationships (upstream_node -> downstream_node).

    Parameters:
        file_path: Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned and validated DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        DataValidationError: If any validation rule is violated.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dependency data file not found at: {path.resolve()}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DataValidationError(f"Failed to read CSV file '{path.name}': {exc}") from exc

    # Check for empty file
    if df.empty:
        raise DataValidationError(f"The CSV file '{path.name}' is empty (no data rows found).")

    # Clean whitespace in column names
    df.columns = df.columns.str.strip()

    # Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"CSV file is missing required column(s): {missing_cols}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )

    # Check for missing or empty node names
    invalid_upstream = is_blank_or_na(df["upstream_node"])
    invalid_downstream = is_blank_or_na(df["downstream_node"])

    missing_node_rows = df[invalid_upstream | invalid_downstream].index.tolist()
    if missing_node_rows:
        row_numbers = [r + 2 for r in missing_node_rows]  # 1-based indexing + header
        raise DataValidationError(
            f"Missing or blank node names found at CSV row(s): {row_numbers}. "
            f"Both 'upstream_node' and 'downstream_node' must have valid non-empty names."
        )

    # Clean string columns: strip whitespace and convert to string
    for col in ["upstream_node", "downstream_node", "material"]:
        df[col] = df[col].astype(str).str.strip()

    # Check for invalid capacities
    numeric_capacities = pd.to_numeric(df["capacity"], errors="coerce")
    invalid_capacity_mask = (
        numeric_capacities.isna()
        | (numeric_capacities <= 0)
        | np_is_inf(numeric_capacities)
    )

    invalid_capacity_rows = df[invalid_capacity_mask].index.tolist()
    if invalid_capacity_rows:
        row_numbers = [r + 2 for r in invalid_capacity_rows]
        bad_values = df.loc[invalid_capacity_mask, "capacity"].tolist()
        raise DataValidationError(
            f"Invalid capacity values {bad_values} found at CSV row(s): {row_numbers}. "
            f"Capacity must be a positive number greater than zero."
        )

    # Overwrite capacity column with validated float values
    df["capacity"] = numeric_capacities.astype(float)

    # Check for duplicate relationships (upstream_node -> downstream_node)
    duplicate_mask = df.duplicated(subset=["upstream_node", "downstream_node"], keep=False)
    if duplicate_mask.any():
        dup_rows = df[duplicate_mask]
        dup_pairs = list(
            zip(dup_rows["upstream_node"], dup_rows["downstream_node"])
        )
        row_numbers = [r + 2 for r in dup_rows.index.tolist()]
        raise DataValidationError(
            f"Duplicate supply relationship(s) detected at CSV row(s) {row_numbers}: "
            f"{set(dup_pairs)}. Each relationship must be unique."
        )

    return df


def np_is_inf(series: pd.Series) -> pd.Series:
    """Helper to detect infinite values in a pandas series without direct numpy dependence."""
    return series.isin([float("inf"), float("-inf")])


def classify_node(name: str, in_degree: int, out_degree: int) -> str:
    """
    Classifies a supply chain node into a role/tier:
    'Raw Material', 'Processor', or 'Supplier'.

    Classification uses:
    1. Keyword heuristics from domain naming ('raw', 'processor', 'supplier').
    2. Network topology fallback:
       - in_degree == 0: Raw Material (source)
       - out_degree == 0: Supplier / Consumer (sink)
       - intermediate: Processor
    """
    lower = name.lower()
    if "raw" in lower or "material" in lower:
        return "Raw Material"
    if "processor" in lower:
        return "Processor"
    if "supplier" in lower:
        return "Supplier"

    # Topological fallback
    if in_degree == 0 and out_degree > 0:
        return "Raw Material"
    if out_degree == 0 and in_degree > 0:
        return "Supplier"
    return "Processor"


def build_supply_chain_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Constructs a NetworkX directed graph from a validated DataFrame.

    - Nodes represent supply chain entities (suppliers, processors, raw materials).
    - Directed edges point from upstream node to downstream node (upstream supplies downstream).
    - Material and capacity attributes are attached to each edge.

    Parameters:
        df: Validated DataFrame with dependencies.

    Returns:
        nx.DiGraph: Directed supply chain network graph.
    """
    G = nx.DiGraph()

    # Add edges with attributes
    for _, row in df.iterrows():
        u = str(row["upstream_node"])
        v = str(row["downstream_node"])
        material = str(row["material"])
        capacity = float(row["capacity"])

        G.add_edge(u, v, material=material, capacity=capacity)

    # Classify each node after all edges are populated
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        node_type = classify_node(node, in_deg, out_deg)
        G.nodes[node]["node_type"] = node_type
        G.nodes[node]["in_degree"] = in_deg
        G.nodes[node]["out_degree"] = out_deg

    return G


def get_supply_chain_network(file_path: Optional[Union[str, Path]] = None) -> nx.DiGraph:
    """
    Convenience function for integration with other modules (e.g. Member 2's Risk Detection Agent).
    Loads validated data from the given CSV file (or default data/dependencies.csv)
    and constructs the complete NetworkX directed graph in a single call.

    Parameters:
        file_path: Optional path to the CSV file. If None, defaults to data/dependencies.csv.

    Returns:
        nx.DiGraph: Validated, ready-to-use directed supply chain graph.
    """
    target_path = Path(file_path) if file_path is not None else DEFAULT_DEPENDENCIES_CSV
    df = load_and_validate_dependencies(target_path)
    return build_supply_chain_graph(df)


def compute_hierarchical_layout(G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
    """
    Computes a clean, left-to-right hierarchical layout for the directed graph.
    - Upstream sources (e.g. Raw Materials) appear on the left.
    - Processors appear in the center.
    - Downstream destinations (e.g. Suppliers) appear on the right.

    Falls back to spring layout if cycles exist.
    """
    if not nx.is_directed_acyclic_graph(G):
        return nx.spring_layout(G, seed=42)

    # Compute layer for each node (longest distance from any root node)
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    if not roots:
        roots = list(G.nodes)

    layers: Dict[str, int] = {}
    for root in roots:
        layers[root] = 0

    # Breadth-first / topological propagation of layers
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        if preds:
            layers[node] = max(layers.get(p, 0) + 1 for p in preds)
        elif node not in layers:
            layers[node] = 0

    # Group nodes by layer
    max_layer = max(layers.values()) if layers else 0
    layer_groups: Dict[int, List[str]] = {l: [] for l in range(max_layer + 1)}
    for node, layer in layers.items():
        layer_groups[layer].append(node)

    pos: Dict[str, Tuple[float, float]] = {}
    for layer, nodes in layer_groups.items():
        n_nodes = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            # Horizontal coordinate determined by layer
            x = float(layer) * 2.5
            # Vertical coordinate centered around 0
            y = (i - (n_nodes - 1) / 2.0) * 1.5
            pos[node] = (x, y)

    return pos


def create_interactive_graph(
    G: nx.DiGraph,
    title: str = "Nexus-AI: Supply Chain Dependency Map",
    output_html: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """
    Creates an interactive Plotly visualization for the supply chain graph.

    Features:
    - Tier-based color and shape coding (Raw Material, Processor, Supplier).
    - Directed visual arrows from upstream node to downstream node.
    - Interactive hover cards on nodes and supply routes.
    - Filterable legend to isolate tiers.
    - Standalone HTML export option.

    Parameters:
        G: NetworkX directed graph.
        title: Title for the chart.
        output_html: Optional file path to save standalone interactive HTML.

    Returns:
        go.Figure: Interactive Plotly figure.
    """
    pos = compute_hierarchical_layout(G)

    fig = go.Figure()

    # Step 1: Draw directed edges using annotations (arrows)
    annotations = []
    midpoint_x = []
    midpoint_y = []
    midpoint_text = []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        material = data.get("material", "N/A")
        capacity = data.get("capacity", "N/A")

        # Calculate midpoint for hover marker
        mx = (x0 + x1) / 2.0
        my = (y0 + y1) / 2.0
        midpoint_x.append(mx)
        midpoint_y.append(my)
        midpoint_text.append(
            f"<b>Supply Route:</b> {u} &rarr; {v}<br>"
            f"<b>Material:</b> {material}<br>"
            f"<b>Capacity:</b> {capacity:,.0f} units"
        )

        # Arrow shortening to prevent overlapping node marker
        dx = x1 - x0
        dy = y1 - y0
        dist = math.hypot(dx, dy)
        if dist > 0:
            offset_start = 0.18
            offset_end = 0.22
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
                arrowhead=2,
                arrowsize=1.4,
                arrowwidth=2,
                arrowcolor="#78909C",  # Blue-grey arrow
                opacity=0.85,
            )
        )

    # Edge midpoint trace for interactive hover details
    fig.add_trace(
        go.Scatter(
            x=midpoint_x,
            y=midpoint_y,
            mode="markers",
            marker=dict(
                size=12,
                color="#37474F",
                symbol="circle-dot",
                opacity=0.6,
                line=dict(color="#ECEFF1", width=1),
            ),
            hoverinfo="text",
            hovertext=midpoint_text,
            name="Supply Routes (Hover)",
            showlegend=True,
        )
    )

    # Step 2: Draw nodes grouped by tier for legend interactivity
    tier_nodes: Dict[str, List[str]] = {
        "Raw Material": [],
        "Processor": [],
        "Supplier": [],
        "Other": [],
    }

    for node, data in G.nodes(data=True):
        tier = data.get("node_type", "Other")
        tier_nodes.setdefault(tier, []).append(node)

    for tier, nodes in tier_nodes.items():
        if not nodes:
            continue

        nx_coords = [pos[n][0] for n in nodes]
        ny_coords = [pos[n][1] for n in nodes]
        hover_texts = []

        for n in nodes:
            in_deg = G.in_degree(n)
            out_deg = G.out_degree(n)
            in_edges = [f"{u} ({G[u][n]['material']})" for u in G.predecessors(n)]
            out_edges = [f"{v} ({G[n][v]['material']})" for v in G.successors(n)]

            hover_html = (
                f"<b>Node:</b> {n}<br>"
                f"<b>Tier / Role:</b> {tier}<br>"
                f"<b>Incoming Feeds ({in_deg}):</b> {', '.join(in_edges) if in_edges else 'None (Origin)'}<br>"
                f"<b>Outgoing Supplies ({out_deg}):</b> {', '.join(out_edges) if out_edges else 'None (Terminal)'}"
            )
            hover_texts.append(hover_html)

        fig.add_trace(
            go.Scatter(
                x=nx_coords,
                y=ny_coords,
                mode="markers+text",
                name=f"{tier} ({len(nodes)})",
                text=nodes,
                textposition="top center",
                textfont=dict(size=12, color="#212121", family="Arial, sans-serif"),
                hoverinfo="text",
                hovertext=hover_texts,
                marker=dict(
                    size=26,
                    color=TIER_COLORS.get(tier, TIER_COLORS["Other"]),
                    symbol=TIER_SYMBOLS.get(tier, "circle"),
                    line=dict(width=2, color="#FFFFFF"),
                    opacity=0.95,
                ),
            )
        )

    # Step 3: Layout configuration
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{title}</b><br>"
                "<span style='font-size:12px; color:#555;'>"
                "&rarr; Directed supply relationship: Upstream Node supplies Downstream Node. "
                "Hover on nodes or route dots for details.</span>"
            ),
            x=0.05,
            y=0.96,
        ),
        showlegend=True,
        legend=dict(
            title=dict(text="<b>Supply Chain Tiers</b>"),
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#CFD8DC",
            borderwidth=1,
        ),
        annotations=annotations,
        margin=dict(b=80, l=40, r=40, t=100),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="#FFFFFF",
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Arial, sans-serif",
            bordercolor="#90A4AE",
        ),
    )

    if output_html:
        out_p = Path(output_html)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_p))

    return fig


def get_graph_summary(G: nx.DiGraph) -> Dict[str, object]:
    """
    Computes human-readable summary statistics for the supply chain network.
    """
    total_capacity = sum(data.get("capacity", 0) for _, _, data in G.edges(data=True))

    tier_counts: Dict[str, int] = {}
    for _, data in G.nodes(data=True):
        tier = data.get("node_type", "Other")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "total_capacity": total_capacity,
        "tiers": tier_counts,
        "nodes": list(G.nodes()),
        "edges": [
            {
                "upstream": u,
                "downstream": v,
                "material": data.get("material"),
                "capacity": data.get("capacity"),
            }
            for u, v, data in G.edges(data=True)
        ],
    }


def print_graph_summary(G: nx.DiGraph) -> None:
    """Prints a friendly summary of the supply chain network to the console."""
    summary = get_graph_summary(G)
    print("\n" + "=" * 60)
    print(" NEXUS-AI: SUPPLY CHAIN GRAPH SUMMARY")
    print("=" * 60)
    print(f"Total Nodes (Facilities / Suppliers): {summary['num_nodes']}")
    print(f"Total Directed Supply Routes:         {summary['num_edges']}")
    print(f"Total Network Flow Capacity:          {summary['total_capacity']:,.0f} units")
    print("\nNodes by Supply Chain Tier:")
    for tier, count in summary["tiers"].items():
        print(f"  - {tier}: {count}")
    print("\nActive Supply Relationships:")
    for edge in summary["edges"]:
        print(
            f"  * {edge['upstream']} ---> {edge['downstream']} "
            f"[{edge['material']}, Cap: {edge['capacity']:,.0f}]"
        )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Self-test when executed directly
    csv_path = Path(__file__).parent.parent / "data" / "dependencies.csv"
    print(f"Loading and validating: {csv_path}...")
    validated_df = load_and_validate_dependencies(csv_path)
    print("Validation successful!")
    print(validated_df)

    graph = build_supply_chain_graph(validated_df)
    print_graph_summary(graph)

    output_file = Path(__file__).parent.parent / "supply_chain_graph.html"
    fig = create_interactive_graph(graph, output_html=output_file)
    print(f"Interactive graph saved to: {output_file.resolve()}")
