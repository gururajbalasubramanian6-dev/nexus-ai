# Nexus-AI: Disruption Simulation Agent (Member 3)

## 1. What the Agent Does
The **Disruption Simulation Agent** is Member 3's core module in the **Nexus-AI** supply chain resilience system. It models "what-if" disruption scenarios across multi-tier supply networks.

When an entity or transport corridor experiences an outage, the agent:
1. **Traces the Blast Radius**: Identifies all affected downstream nodes across multiple tiers (direct Tier-1 suppliers down to assembly factories).
2. **Maps Disrupted Paths**: Enumerates all broken supply routes connecting the failure point to downstream destinations.
3. **Audits Affected Materials**: Catalogues the raw materials, sub-assemblies, and components whose delivery is jeopardized.
4. **Calculates Direct Capacity Loss**: Quantifies the immediate offline capacity without double-counting multi-tier throughput.
5. **Differentiates Shortage Severity**: Distinguishes between **confirmed shortages** (entities with zero operational inbound paths and no backups) and **potentially affected** nodes (entities protected by redundant suppliers or operational branches).
6. **Assigns Transparent Impact Levels**: Applies documented heuristic rules to categorize impact as `Low`, `Medium`, `High`, or `Critical`.
7. **Flags Missing Data**: Surfaces gaps in capacity, lead times, or inventory data without fabricating assumptions.
8. **Proposes Next Actions**: Recommends operational steps (e.g. emergency qualification, corridor re-routing, buffer audits).

---

## 2. Disruption Types Supported
The agent natively models 5 types of disruptions:
- **Processor failure**: Failure of an upstream processing plant, foundry, or refinery (e.g., `Processor X`).
- **Supplier failure**: Outage at a Tier-1 or component manufacturing supplier (e.g., `Supplier A`).
- **Factory failure**: Outage at a final assembly facility or manufacturing plant (e.g., `Factory 1`).
- **Transport route failure**: Interruption of a freight corridor or transport edge (e.g., edge tuple `("Supplier A", "Factory 1")` or route ID `"Route-SA-F1"`).
- **Raw material failure / General entity failure**: Disruption of a primary extraction source, mine, or general node.

Disruption types are automatically inferred from node/edge attributes if not explicitly passed by the caller.

---

## 3. Required Inputs & Graph Structure

### Graph Convention
The agent accepts a standard `networkx.DiGraph` where directed edges flow from upstream entities to downstream entities:
```
[Upstream Entity]  --->  [Downstream Entity]
(e.g., Raw Material P ---> Processor X ---> Supplier A ---> Factory 1)
```

### Node Attributes (Optional but Recommended)
- `node_type`: String (`"raw_material"`, `"processor"`, `"supplier"`, `"factory"`).
- `capacity`: Numeric rated throughput (e.g. `200.0`).
- `lead_time_days`: Replacement/replenishment time in days.
- `alternatives`: List of external qualified backup entities (e.g. `["Mine Alpha", "Mine Beta"]`).

### Edge Attributes (Optional but Recommended)
- `material`: Commodity or part name (e.g. `"Semiconductor Wafers"`).
- `capacity`: Maximum throughput on this transport edge (e.g. `100.0`).
- `route`: Logistics corridor code (e.g. `"Route-PX-SA"`).
- `lead_time_days`: Transit duration in days.

If an attribute is omitted, the agent does **not** invent a value; it records the gap in `missing_information`.

---

## 4. How to Call the Agent

### Functional Interface
```python
import networkx as nx
from agents.disruption_simulation_agent import simulate_disruption, create_synthetic_supply_chain

# 1. Load or create the graph
graph = create_synthetic_supply_chain()

# 2. Simulate node failure (e.g., Processor X)
result = simulate_disruption(graph, failed_node="Processor X")

# 3. Simulate transport route failure by edge tuple
route_result = simulate_disruption(graph, failed_edge=("Supplier A", "Factory 1"))

# 4. Simulate transport route failure by route ID
route_id_result = simulate_disruption(graph, failed_route="Route-SA-F1")
```

### Class-Based Interface (Ideal for Streamlit & UI Integration)
```python
from agents.disruption_simulation_agent import DisruptionSimulationAgent, create_synthetic_supply_chain

agent = DisruptionSimulationAgent(create_synthetic_supply_chain())

# Simulate a single node
result = agent.simulate(failed_node="Processor X")

# Generate batch comparison dataframe for Streamlit
df = agent.simulate_all_nodes()
# st.dataframe(df)
```

---

## 5. Output Schema & Meaning

The agent returns a structured Python dictionary with the following fields:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `failed_entity` | `str` | Name of the failed node or representation of the broken route. |
| `disruption_type` | `str` | Standardized category (e.g., `"Processor failure"`, `"Transport route failure"`). |
| `affected_entities` | `List[str]` | De-duplicated, sorted list of all reachable downstream entities. |
| `affected_paths` | `List[List[str]]` | Full paths from the failure point to downstream terminal nodes. |
| `affected_materials` | `List[str]` | Unique material names whose movement is disrupted. |
| `estimated_capacity_impact` | `Dict[str, Any]` | Detailed capacity accounting, avoiding multi-tier double counting. |
| `impact_level` | `str` | Categorical rating: `"Low"`, `"Medium"`, `"High"`, or `"Critical"`. |
| `explanation` | `str` | Plain-English summary explaining why entities and paths are affected. |
| `missing_information` | `List[str]` | Explicit list of data gaps that prevent complete estimation. |
| `assumptions` | `List[str]` | Transparent modeling assumptions applied during the simulation. |
| `recommended_next_actions` | `List[str]` | Prioritized mitigation steps for supply chain managers. |
| `entity_impact_details` | `List[Dict]` | Granular per-entity status: `"confirmed_shortage"` vs `"potentially_affected"`. |

---

## 6. Capacity Estimation & Anti-Double-Counting Methodology

### The Double-Counting Hazard
In a multi-tier supply chain:
```
Processor X (Cap: 200) ---> Supplier A (Cap: 100) ---> Factory 1 (Cap: 100)
```
If Processor X fails, summing capacity along the path would yield $200 + 100 + 100 = 400$ units. This is invalid because the 100 units at Factory 1 is the **same physical material** processed by Supplier A and Processor X.

### Nexus-AI Capacity Accounting Rule
1. **Direct Outgoing Sum**: When a node fails, the agent calculates the capacity of its **direct outgoing links** (or the node's rated capacity). Downstream tiers are strictly excluded from the sum.
2. **Missing Value Protection**: If edge or node capacity is missing, the estimate is set to `None`, `has_complete_capacity_data` is flagged `False`, and a warning is logged in `missing_information`.
3. **Simulated Heuristic Label**: All capacity loss figures are labeled as estimated approximations based solely on supplied attributes.

---

## 7. Shortage Distinction: Potentially Affected vs Confirmed Shortage

To prevent false alarms, the agent evaluates operational reachability in the subgraph $G' = G \setminus \{\text{disrupted entities}\}$:

- **Confirmed Shortage**:
  - All paths from upstream material sources (root nodes) to this entity are completely severed.
  - The entity has no documented external backup alternatives.
  - *Example*: Factory 1 when Processor X fails (sole supplier Supplier A is severed).
- **Potentially Affected**:
  - Direct upstream link was disrupted, but at least one operational path from an upstream raw material source remains active through alternative branches.
  - Or, documented backup suppliers exist outside the graph.
  - *Example*: Factory 2 when Processor X fails (Supplier B is severed, but Supplier C remains operational via Processor Y and Raw Material Q).

---

## 8. Transparent Impact Level Rules

Impact levels are calculated using transparent heuristic criteria:
- **`Critical`**:
  - $\ge 2$ confirmed shortages, OR
  - Shared choke-point outage affecting $\ge 3$ downstream entities, OR
  - Significant direct capacity loss.
- **`High`**:
  - Exactly 1 confirmed shortage, OR
  - Outage affecting $\ge 3$ downstream entities with operational redundancy, OR
  - Shared choke-point affecting 2 suppliers.
- **`Medium`**:
  - 1 or 2 downstream entities potentially affected with operational fallback paths intact.
- **`Low`**:
  - 0 downstream entities affected (terminal leaf nodes such as factories), or isolated leaf relationships.

*Note: These heuristic levels provide prioritization guidance for operators and hackathon judges; they do not claim scientific forecasting certitude.*

---

## 9. Running Automated Tests

Run the full pytest suite from the project directory:
```powershell
python -m pytest -v
```

### Verified Test Matrix
- **Test A**: Processor X failure identifies Supplier A and Supplier B.
- **Test B**: Unrelated entities (Processor Y, Supplier C, Raw Material Q) are not marked affected.
- **Test C**: Factory 1 failure handled cleanly as a terminal leaf outage.
- **Test D**: Supplier A failure affects Factory 1 and flags Engine Control Units.
- **Test E**: Missing capacity data flags warnings without inventing numbers.
- **Test F**: Disconnected nodes execute cleanly without exceptions.
- **Test G**: Invalid graph types, empty graphs, and unknown nodes raise clear errors.
- **Test H**: Multi-tier relationships (Tier 1 suppliers + Tier 2 factories) are captured.
- **Test I**: Diamond topologies and redundant paths do not produce duplicate entity entries.
- **Test J**: Complete output schema and data types strictly adhere to contract.
- **Test K & L**: Transport route failure via edge tuple and route ID attribute.
- **Test M**: Batch simulation dataframe export for Streamlit.
- **Test N**: Verification of confirmed shortage vs potentially affected distinction.
- **Test O**: Verification of multi-tier anti-double-counting policy.

---

## 10. Known Limitations & Modeling Assumptions

1. **Static Topology**: The simulation assumes network edges remain static unless alternative suppliers are explicitly recorded in node attributes.
2. **Buffer Inventory**: Downstream nodes without explicit safety stock attributes are assumed to have zero buffer inventory.
3. **100% Curtailment**: Disrupted nodes are modeled as experiencing 100% outage for the duration of the scenario.
4. **Heuristic Nature**: Outputs represent simulated structural resilience assessments, not guaranteed physical event forecasts.

---

## 11. Multi-Agent Integration Interfaces & Contracts

### A. Input Format from Member 1 (Mapping Agent)
Member 3 expects a `networkx.DiGraph` representing directed flow (`Upstream Node -> Downstream Node`).
- **Required Node Attributes**:
  - `node_type`: String (`"raw_material"`, `"processor"`, `"supplier"`, `"factory"`).
  - `capacity`: Float representing rated capacity.
  - `lead_time_days`: Integer replacement time.
  - `alternatives`: Optional list of strings (qualified backup entities).
- **Required Edge Attributes**:
  - `material`: String (e.g. `"Semiconductor Wafers"`).
  - `capacity`: Float throughput capacity.
  - `route`: Optional string identifier (e.g. `"Route-PX-SA"`).
  - `lead_time_days`: Transit duration in days.

### B. User Disruption Specification (via UI or CLI)
Users or the Streamlit UI can trigger disruptions using three distinct parameter modes:
1. **Node Failure**: `simulate_disruption(graph, failed_node="Processor X")`
2. **Edge Failure by Tuple**: `simulate_disruption(graph, failed_edge=("Supplier A", "Factory 1"))`
3. **Route Failure by ID**: `simulate_disruption(graph, failed_route="Route-PX-SA")`

### C. Output Payload for Member 4 (Financial Engine)
Member 4 accesses simulation results via `get_financial_engine_payload(result)`:
```python
from agents.disruption_simulation_agent import get_financial_engine_payload, simulate_disruption

result = simulate_disruption(graph, failed_node="Processor X")
financial_input = get_financial_engine_payload(result)
```

**Delivered Payload Fields**:
- `failed_entity`: Asset name / route.
- `disruption_type`: Disruption category.
- `impact_level`: Severity level (`"Critical"`, `"High"`, etc.).
- `direct_capacity_loss`: Quantified direct offline throughput.
- `affected_entities`: All downstream entities.
- `confirmed_shortage_entities`: Entities with 0 operational inbound paths from any source.
- `potential_shortage_entities`: Entities with surviving operational paths or backups.
- `capacity_deficit_by_entity`: Dictionary mapping `{entity: deficit_amount}`.
- `total_downstream_capacity_deficit`: Sum of all downstream capacity shortfalls.
- `affected_materials`: Impacted materials/components.
- `missing_data_warnings`: Incomplete data attributes to incorporate into risk discount margins.
- `assumptions`: Modeling parameters for auditability.

### D. Interface for Member 5 (Streamlit UI / `app.py`)
Member 5 can directly render simulation findings without custom UI parsing logic:
```python
from agents.disruption_simulation_agent import DisruptionSimulationAgent

agent = DisruptionSimulationAgent(graph)

# 1. Comparative summary table for st.dataframe()
overview_df = agent.simulate_all_nodes()
# st.dataframe(overview_df)

# 2. Granular entity vulnerability table for single scenario
res = agent.simulate(failed_node="Processor X")
detail_df = agent.get_entity_impact_dataframe(res)
# st.dataframe(detail_df)
```

