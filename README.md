# Nexus-AI: Supply Chain Mapping Agent (Member 1)

Welcome to the **Supply Chain Mapping Agent** component of **Nexus-AI**, a supply-chain resilience platform.

---

## 📌 Project Overview

In supply chain management, understanding dependencies between raw material sources, processing plants, and suppliers is crucial. 

This component reads supply chain dependency data, validates it against common data issues, converts the relationships into a mathematical directed network graph, and generates an interactive web visualization.

### Key Concepts

- **Upstream vs. Downstream**: 
  - **Upstream Node**: The facility or source that *provides* or *supplies* inputs (e.g., `Raw Material P` supplies `Processor X`).
  - **Downstream Node**: The facility or recipient that *receives* inputs to produce further goods or distribute them (e.g., `Processor X` supplies `Supplier A`).
- **Capacity**: A hypothetical numerical measure of volume, throughput, or maximum units that can travel across that supply link.
- **Directed Graph (DiGraph)**: A mathematical network structure where links between entities have a specific direction (`Origin -> Destination`).

---

## 📂 Project Structure

```text
nexus-ai/
├── agents/
│   ├── __init__.py
│   └── mapping_agent.py      # Core agent: validation, graph construction, visualization
├── data/
│   └── dependencies.csv      # Sample supply chain dataset
├── tests/
│   ├── __init__.py
│   └── test_mapping_agent.py # Comprehensive unit test suite (14 tests)
├── app.py                    # Main application runner
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```

---

## 🚀 Quick Start Guide (Beginner Friendly)

### Step 1: Open PowerShell or Terminal

Navigate to the project directory:
```powershell
cd "c:\Users\Harshitha Shree\OneDrive\Desktop\nexus-ai"
```

### Step 2: Run the Application

Execute the Python script using the project's virtual environment:
```powershell
.\.venv\Scripts\python.exe app.py
```
*(If you have Python added to your system PATH, you can simply type `python app.py`)*

### Step 3: View the Interactive Map

After running `app.py`, an interactive file named `supply_chain_graph.html` is generated in your project folder.
Double-click `supply_chain_graph.html` to open it in your browser (Google Chrome, Edge, Firefox), or run:
```powershell
.\.venv\Scripts\python.exe app.py --open
```

### Step 4: Run the Automated Unit Tests

To run all 14 automated tests and verify all requirements:
```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_mapping_agent.py -v
```

---

## 🧪 Testing Results Summary

All 14 tests pass:
1. `test_sample_csv_loads_successfully`: Ingests and validates `data/dependencies.csv`.
2. `test_graph_contains_expected_nodes_and_edges`: Verifies all 7 nodes and 5 edges.
3. `test_processor_x_connections`: Verifies `Processor X` connects downstream to `Supplier A` and `Supplier B`, and receives from `Raw Material P`.
4. `test_edge_attributes_preserved`: Confirms material type and numerical capacity are retained on every edge.
5. `test_missing_upstream_node_rejected`: Confirms blank/missing upstream node names raise `DataValidationError`.
6. `test_missing_downstream_node_rejected`: Confirms blank/missing downstream node names raise `DataValidationError`.
7. `test_non_numeric_capacity_rejected`: Rejects string/garbage capacities.
8. `test_negative_capacity_rejected`: Rejects negative numbers.
9. `test_zero_capacity_rejected`: Rejects 0 capacity.
10. `test_duplicate_relationships_rejected`: Detects duplicate supply links.
11. `test_missing_required_column_rejected`: Rejects CSVs missing required headers.
12. `test_nonexistent_file_raises_error`: Raises `FileNotFoundError` when a file does not exist.
13. `test_node_classification`: Confirms accurate categorization into Raw Material, Processor, and Supplier.
14. `test_interactive_visualization_creation`: Confirms interactive HTML generation.

