# Nexus-AI: Shared Risk Detection Agent (Member 2)

## 1. Overview & Purpose
In supply chain networks, companies frequently believe they have resilient multi-sourcing because they buy from different direct suppliers (e.g., Supplier A and Supplier B). However, deep in the supply tiers, both suppliers may depend on the **same single processor, sub-tier foundry, or raw material mine**. If that common upstream node fails, both suppliers halt simultaneously.

The **Shared Risk Detection Agent** analyzes the supply chain graph to:
1. **Identify Single Points of Failure & Shared Dependencies**: Uncover upstream components that feed multiple downstream suppliers.
2. **Conduct Multi-Tier Dependency Traversal**: Trace beyond Tier-1 to find indirect shared risks (Tier-2, Tier-3, etc.).
3. **Audit Missing Data**: Flag missing capacity figures, lead times, or alternative supplier options without hallucinating data.
4. **Compute Explainable Vulnerability Scores**: Provide a transparent 0–100 heuristic risk index with full plain-English justification.

---

## 2. Graph Direction & Core Conventions
In Nexus-AI, edges represent physical or material flow:
```
[Upstream Entity]  --->  [Downstream Entity]
(e.g., Raw Material P ---> Processor X ---> Supplier A)
```
- **Direct Dependency (Tier 1)**: An upstream node directly connected to a supplier (shortest path = 1 hop).
- **Indirect Dependency (Tier 2+)**: An upstream node connected to a supplier via intermediate nodes (shortest path > 1 hop).
- **Target Downstream Suppliers**: The focal entities being monitored (e.g. Tier-1 suppliers or sink nodes).

---

## 3. How Shared Dependencies are Detected
1. **Reachability Traversal**: For each upstream node $u$, the agent traverses the directed graph using `networkx.descendants(graph, u)` to find all reachable downstream nodes.
2. **De-duplication**: The reachable nodes are intersected with the target supplier set. Because sets only store unique elements, even if there are multiple redundant paths from an upstream mine to a supplier (e.g. diamond topologies), each supplier is counted **strictly once** (no double-counting).
3. **Shared Filter**: If an upstream node reaches 2 or more distinct downstream suppliers (`affected_supplier_count >= 2`), it is categorized as a **shared dependency**.

---

## 4. Risk Scoring Methodology

### Formula Breakdown (0 to 100 Scale)
The risk score is a heuristic vulnerability index designed to prioritize supply-chain resilience interventions:

$$\text{Risk Score} = \text{Blast Radius} + \text{Alternative Sourcing} + \text{Capacity Bottleneck} + \text{Lead Time Risk} + \text{Data Gap Penalty} + \text{Tier Visibility}$$

1. **Blast Radius (Sharedness & Breadth)**:
   - 1 affected supplier: `+15 pts`
   - 2 affected suppliers: `+35 pts`
   - 3 affected suppliers: `+50 pts`
   - 4+ affected suppliers: `+65 pts`
2. **Alternative Source Availability**:
   - Sole source (0 alternatives): `+20 pts`
   - 1 alternative known: `+5 pts`
   - 2+ verified alternatives: `-10 pts` (cushion / risk reduction)
   - Alternative data missing / unknown: `+15 pts` (uncertainty penalty)
3. **Operational Bottlenecks**:
   - Capacity utilization $\ge 85\%$: `+15 pts`
   - Capacity utilization $70\% - 85\%$: `+5 pts`
   - Replacement lead time $\ge 45\text{ days}$: `+10 pts`
4. **Data Completeness Penalty**:
   - Missing capacity data: `+10 pts`
   - Missing lead time data: `+10 pts`
5. **Tier Visibility Risk**:
   - Multi-tier indirect dependency (hop distance $\ge 2$): `+5 pts`

Score is bounded strictly between `0.0` and `100.0`.

### Risk Categories:
- **0.0 - 39.9**: `Low`
- **40.0 - 64.9**: `Medium`
- **65.0 - 84.9**: `High`
- **85.0 - 100.0**: `Critical`

> **Important Assumption & Limitation:**
> This score is **NOT** a scientific probability of physical failure. It is a prioritized vulnerability indicator measuring exposure, lack of buffers, and monitoring opacity.

---

## 5. Output Schema & Interface (For Members 1, 3, and 5)

### Python Function Signature:
```python
from agents.risk_detection_agent import detect_shared_risks

findings = detect_shared_risks(
    graph=supply_chain_graph,        # nx.DiGraph (Upstream -> Downstream)
    target_suppliers=None,            # Optional list of supplier names
    only_shared=True,                 # True: only dependencies affecting >=2 suppliers
    as_dataframe=False                # Set True to get a pandas.DataFrame
)
```

### Result Schema (Dictionary / DataFrame columns):
| Field | Type | Description |
| :--- | :--- | :--- |
| `dependency_node` | `str` | Name / ID of the upstream dependency |
| `affected_suppliers` | `List[str]` | Sorted list of unique downstream supplier names |
| `affected_supplier_count` | `int` | Count of unique downstream suppliers affected |
| `dependency_tier` | `str` | e.g. "Tier 1 (Direct)", "Tier 2 (Indirect)" |
| `is_shared` | `bool` | `True` if `affected_supplier_count >= 2` |
| `direct_affected_suppliers` | `List[str]` | Suppliers directly connected (1 hop) |
| `indirect_affected_suppliers` | `List[str]` | Suppliers reached via multi-tier paths (> 1 hop) |
| `risk_score` | `float` | Vulnerability index from 0.0 to 100.0 |
| `risk_level` | `str` | "Low", "Medium", "High", or "Critical" |
| `explanation` | `str` | Transparent audit trail of score components |
| `missing_information` | `List[str]` | Data gaps identified (capacity, lead time, alts) |
| `recommended_investigation`| `str` | Actionable mitigation advice |

---

## 6. How to Run

### Run the Demo
```bash
python run_demo.py
```

### Run Automated Tests
```bash
python -m pytest -v
```

---

## 7. Known Limitations
1. **Static Analysis**: The agent analyzes structural graph paths and point-in-time attributes. It does not perform dynamic continuous time-series flow simulation (which is handled by Member 3's Disruption Simulation Agent).
2. **Attribute Availability**: If upstream suppliers do not disclose sub-tier relationships or capacity, the agent flags uncertainty penalties but cannot magically guess missing suppliers.
3. **Capacity Thresholds**: Assumes linear risk thresholds for capacity and lead time until Member 4/5 integrates dynamic economic cost models.
