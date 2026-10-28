# Member 1 Handover Document: Supply Chain Mapping Agent
**Project:** Nexus-AI (Supply Chain Resilience Platform)  
**Author:** Member 1 (Supply Chain Mapping Component)  
**Status:** Tested & Integration-Ready  
**Python Environment:** Python 3.11 (`.venv`)

---

## 1. How the Agent Works

The Supply Chain Mapping Agent ingests raw supply-chain dependency data, verifies its integrity, constructs a directed network graph, and generates an interactive visualization.

### Pipeline Stages
1. **Validation & Ingestion (`load_and_validate_dependencies`)**:
   - Reads CSV using `pandas`.
   - Strips whitespace from column names and string values.
   - Enforces required columns: `upstream_node`, `downstream_node`, `material`, `capacity`.
   - Rejects blank, whitespace, or `NaN` node names.
   - Rejects non-numeric, zero, negative, or infinite capacities.
   - Detects and rejects duplicate directed relationships (`upstream_node -> downstream_node`).
2. **Graph Construction (`build_supply_chain_graph`)**:
   - Builds a `networkx.DiGraph` (directed graph).
   - Preserves `material` (str) and `capacity` (float) on every directed edge.
   - Categorizes each node into a supply tier (`Raw Material`, `Processor`, `Supplier`) stored as `G.nodes[n]['node_type']`.
   - Records `in_degree` and `out_degree` on each node.
3. **One-Call Integration (`get_supply_chain_network`)**:
   - Loads, validates, and builds the `networkx.DiGraph` in a single callable function.
4. **Interactive Visualization (`create_interactive_graph`)**:
   - Calculates a 3-tier hierarchical left-to-right layout (Raw Materials $\rightarrow$ Processors $\rightarrow$ Suppliers).
   - Draws directional supply route arrows pointing from upstream to downstream.
   - Adds interactive hover cards with route material, capacity, and node statistics.
   - Exports a standalone, portable HTML file (`supply_chain_graph.html`).

---

## 2. Required Data Files

The agent requires a CSV file at:
`data/dependencies.csv`

### Required Format:
```csv
upstream_node,downstream_node,material,capacity
Processor X,Supplier A,Battery cells,100
Processor X,Supplier B,Battery cells,80
Processor Y,Supplier C,Battery cells,120
Raw Material P,Processor X,Raw material,250
Raw Material Q,Processor Y,Raw material,150
```

### Column Specifications:
| Column | Type | Description |
| :--- | :--- | :--- |
| `upstream_node` | Non-empty String | Origin facility supplying material |
| `downstream_node` | Non-empty String | Destination facility receiving material |
| `material` | String | Description of item / commodity supplied |
| `capacity` | Positive Float / Int ($> 0$) | Maximum units supported by this route |

---

## 3. How to Run the Mapping Agent

Run the main runner using the project's virtual environment:

```powershell
.\.venv\Scripts\python.exe app.py
```

### Launch Options:
- **Default execution**: Validates data, prints network metrics to console, and creates `supply_chain_graph.html`.
- **Open in default browser**:
  ```powershell
  .\.venv\Scripts\python.exe app.py --open
  ```
- **Custom CSV input**:
  ```powershell
  .\.venv\Scripts\python.exe app.py --csv path/to/custom.csv
  ```

> **Note on Environment Restrictions**: In restricted Windows environments where background TCP socket binding is blocked (`WinError 10013`), `app.py` uses standalone HTML file generation (`fig.write_html()`) and opens it directly via standard browser file rendering (`file:///`). This requires no open network ports and works reliably.

---

## 4. How to Execute Unit Tests

The test suite covers 15 automated test cases:

```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_mapping_agent.py -v
```

### Covered Test Cases:
1. `test_sample_csv_loads_successfully`: Ingests and validates `data/dependencies.csv`.
2. `test_get_supply_chain_network`: Verifies one-call integration function.
3. `test_graph_contains_expected_nodes_and_edges`: Confirms exact 7 nodes and 5 edges.
4. `test_processor_x_connections`: Verifies connections to `Supplier A`, `Supplier B`, and `Raw Material P`.
5. `test_edge_attributes_preserved`: Validates material and capacity on edges.
6. `test_missing_upstream_node_rejected`: Rejects empty upstream node names.
7. `test_missing_downstream_node_rejected`: Rejects empty downstream node names.
8. `test_non_numeric_capacity_rejected`: Rejects string/garbage capacities.
9. `test_negative_capacity_rejected`: Rejects negative capacities.
10. `test_zero_capacity_rejected`: Rejects zero capacity.
11. `test_duplicate_relationships_rejected`: Rejects duplicate upstream-downstream pairs.
12. `test_missing_required_column_rejected`: Rejects CSVs missing required headers.
13. `test_nonexistent_file_raises_error`: Raises `FileNotFoundError` for missing files.
14. `test_node_classification`: Confirms accurate classification of tiers.
15. `test_interactive_visualization_creation`: Confirms HTML generation.

---

## 5. Reusable Interface for Teammates (Members 2–5)

To obtain the validated directed graph in any other module, import and call `get_supply_chain_network`:

### Exact Function Specification:
- **File Path**: `agents/mapping_agent.py`
- **Function Name**: `get_supply_chain_network`
- **Parameters**: `file_path: Optional[Union[str, Path]] = None`  
  *(If omitted, automatically defaults to `data/dependencies.csv`)*
- **Return Type**: `networkx.DiGraph`
- **Exceptions Raised**: `DataValidationError`, `FileNotFoundError`

### Example Usage:
```python
from pathlib import Path
from agents.mapping_agent import get_supply_chain_network

# Obtain the validated NetworkX directed graph
G = get_supply_chain_network()

# Access nodes and their attributes
for node, attrs in G.nodes(data=True):
    print(node, attrs["node_type"], attrs["in_degree"], attrs["out_degree"])

# Access edges and their attributes
for u, v, attrs in G.edges(data=True):
    print(f"{u} -> {v}: {attrs['material']} (Capacity: {attrs['capacity']})")
```

---

## 6. Integration Contract for Member 2 (Risk Detection Module)

Member 2 has not yet committed their module to the repository. Once Member 2 begins implementation, they can directly consume Member 1's graph without parsing CSV files or re-validating data:

### Available Graph Queries for Risk Analysis:
1. **Downstream Blast Radius (Impact of Node Disruption)**:
   ```python
   import networkx as nx
   # If 'Processor X' fails, which downstream nodes are impacted?
   impacted_nodes = nx.descendants(G, "Processor X")
   # Result: {'Supplier A', 'Supplier B'}
   ```
2. **Upstream Vulnerability (Root Cause Tracing)**:
   ```python
   # Where does 'Supplier A' get its components from?
   suppliers = list(G.predecessors("Supplier A"))
   all_upstream = nx.ancestors(G, "Supplier A")
   # Result: suppliers = ['Processor X'], all_upstream = {'Processor X', 'Raw Material P'}
   ```
3. **Single Point of Failure (SPOF) Detection**:
   ```python
   # Nodes with high out_degree or high betweenness centrality
   for node in G.nodes():
       if G.in_degree(node) > 0 and G.out_degree(node) > 1:
           print(f"High risk hub / bottleneck: {node}")
   ```
4. **Capacity Bottlenecks**:
   ```python
   # Total capacity fed into a node vs. output capacity
   in_capacity = sum(G[u][node]["capacity"] for u in G.predecessors(node))
   out_capacity = sum(G[node][v]["capacity"] for v in G.successors(node))
   ```

---

## 7. Known Limitations

1. **Cycle Handling**: The current supply chain visualization uses a 3-tier hierarchical layout optimized for Directed Acyclic Graphs (DAGs). If a cyclic dependency (A $\rightarrow$ B $\rightarrow$ A) is introduced, the visualizer automatically falls back to a spring layout (`nx.spring_layout`) to prevent layout failure.
2. **Single Link per Pair**: The validator strictly enforces unique directed relationships between each pair of nodes `(upstream_node, downstream_node)`. If a facility supplies multiple distinct materials to the same recipient, they currently must be grouped or modeled as distinct facilities.
3. **Local Web Server**: On systems where local port binding is restricted by security policies, opening the graph as a file (`supply_chain_graph.html`) is required rather than running a background HTTP socket server.

