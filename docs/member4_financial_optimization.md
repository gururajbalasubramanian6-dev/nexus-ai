# Member 4 Handover Document: Financial Impact & Recovery Optimization Engine
**Project:** Nexus-AI (Multi-Tier Supply Chain Resilience and Investment Optimization)  
**Author:** Member 4 (Financial Impact & Recovery Optimization Component)  
**Module:** [`agents/financial_optimization_agent.py`](file:///c:/Users/Harshitha%20Shree/OneDrive/Desktop/nexusAI-M4/agents/financial_optimization_agent.py)  
**Test Suite:** [`tests/test_financial_optimization.py`](file:///c:/Users/Harshitha%20Shree/OneDrive/Desktop/nexusAI-M4/tests/test_financial_optimization.py)  
**Status:** Tested & Integration-Ready (15/15 tests passing)  
**Python Environment:** Python 3.11 (`.venv`)

---

## 1. Purpose of the Component

When an upstream disruption occurs (such as a factory shutdown, supplier bottleneck, or material shortage identified by Member 1, 2, or 3), supply chain managers must determine:
1. **What is the baseline financial loss** if no action is taken?
2. **Which recovery options are feasible** under real-world constraints (recovery budget, available supplier capacities, and acceptable delivery lead times)?
3. **Which recovery plan minimizes total financial loss** while maximizing net savings?
4. **When should human operational approval be required** before committing capital?

The **Financial Impact & Recovery Optimization Engine** provides deterministic, transparent, and auditable calculations to solve these questions without relying on opaque black-box heuristics or paid external services.

---

## 2. Mathematical Formulas & Rules

All calculations are deterministic, transparent, and avoid double counting:

### A. Gross Disruption Loss (Baseline Loss)
$$\text{Gross Disruption Loss} = \text{Affected Quantity} \times \text{Loss Per Unit}$$
- **Affected Quantity**: Total disrupted units (pieces, kilograms, batches). Must be $\ge 0$.
- **Loss Per Unit**: Financial impact incurred per unfulfilled unit (revenue loss, contractual delay penalties). Must be $\ge 0$.
- **Baseline Scenario**: Represents the cost of taking **no recovery action**.

### B. Recovery Plan Total Cost
$$\text{Recovery Plan Total Cost} = \text{Recovery Expenditure} + \text{Residual Disruption Loss}$$
- **Recovery Expenditure**: Direct payments for expediting, alternative sourcing, overtime, or buffer stock release.
- **Residual Disruption Loss**: Unmitigated disruption loss remaining after recovery:
  $$\text{Residual Disruption Loss} = \max(0, \text{Required Quantity} - \text{Recovered Capacity}) \times \text{Loss Per Unit}$$
- *Anti-Double-Counting Rule*: Baseline gross disruption loss is **not** added to the recovery plan total cost. Only unrecovered quantity incurs residual loss.

### C. Estimated Savings
$$\text{Estimated Net Savings} = \text{Baseline Disruption Loss} - \text{Recovery Plan Total Cost}$$
- **Positive Savings ($> 0$)**: The recovery plan preserves capital and is economically advantageous.
- **Break-Even ($= 0$)**: The recovery plan costs the exact amount it saves.
- **Negative Savings ($< 0$)**: The recovery plan costs more than the loss it prevents. In such cases, taking no action is financially preferred, and human approval is triggered.

---

## 3. Optimization Objective & Constraints

### Optimization Objective
$$\min \left( \text{Total Estimated Cost} = \text{Recovery Expenditure} + \text{Residual Disruption Loss} \right)$$

### Operational Constraints
Candidate recovery options must pass four strict feasibility checks:

| Constraint | Rule | Infeasibility Handling |
| :--- | :--- | :--- |
| **Budget Constraint** | $\text{Recovery Cost} \le \text{Available Budget}$ | Rejected with detailed overage explanation. |
| **Capacity Constraint** | $\text{Available Capacity} \ge \text{Required Quantity}$ | Rejected with exact shortfall volume. |
| **Lead-Time Constraint** | $\text{Lead Time (Days)} \le \text{Max Acceptable Lead Time}$ | Rejected with days exceeded. |
| **Data Completeness** | Capacity, lead time, and costs must be present | Missing data is **never** silently assumed as 0; rejected with missing-field alerts. |

If no single option covers 100% of the disrupted quantity and `allow_multi_option=True` is enabled, the engine evaluates combinations of non-competing options that together fulfill the demanded quantity within budget and lead time.

---

## 4. Input Schema

The module accepts either typed Python dataclasses ([`DisruptionScenario`](file:///c:/Users/Harshitha%20Shree/OneDrive/Desktop/nexusAI-M4/agents/financial_optimization_agent.py#L32), [`RecoveryOption`](file:///c:/Users/Harshitha%20Shree/OneDrive/Desktop/nexusAI-M4/agents/financial_optimization_agent.py#L125)) or standard dictionaries.

### Disruption Scenario (`DisruptionScenario`)
| Field Name | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `scenario_id` | `str` | Yes | Unique disruption event code (e.g. `"SCEN-001"`). |
| `affected_quantity` | `float` | Yes | Disrupted quantity (must be $\ge 0$). |
| `loss_per_unit` | `float` | Yes | Financial loss per unfulfilled unit (must be $\ge 0$). |
| `disrupted_node` | `str` | No | Impacted facility/supplier name (e.g. `"Processor X"`). |
| `available_budget` | `float` | No | Maximum capital allocated for recovery. |
| `max_lead_time_days` | `float` | No | Maximum tolerable delivery delay in days. |
| `description` | `str` | No | Contextual summary of the disruption. |
| `currency` | `str` | No | Currency code (default: `"USD"`). |

### Recovery Option (`RecoveryOption`)
| Field Name | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `option_id` | `str` | Yes | Unique option identifier (e.g. `"OPT-EXPEDITE-01"`). |
| `option_type` | `str` | Yes | Strategy tag (e.g. `"expedite"`, `"alternate_supplier"`). |
| `description` | `str` | Yes | Description of proposed recovery action. |
| `recovery_cost` | `float` | Yes | Cost to implement this option. |
| `available_capacity` | `float` | Yes | Units this option can recover. |
| `lead_time_days` | `float` | Yes | Days required to execute. |
| `supplier_name` | `str` | No | Alternative vendor or provider name. |
| `maximum_acceptable_lead_time_days` | `float` | No | Option-specific deadline ceiling. |
| `estimated_residual_loss` | `float` | No | Pre-computed residual loss override. |

---

## 5. Output Schema (Ready for Member 5 Streamlit Dashboard)

The primary function [`optimize_recovery_plan`](file:///c:/Users/Harshitha%20Shree/OneDrive/Desktop/nexusAI-M4/agents/financial_optimization_agent.py#L359) returns a clean, structured Python dictionary:

```python
{
    "scenario_id": "SCEN-TEST-001",
    "disrupted_node": "Processor X",
    "affected_quantity": 100.0,
    "loss_per_unit": 250.0,
    "baseline_disruption_loss": 25000.0,
    "available_budget": 30000.0,
    "max_lead_time_days": 14.0,
    "recommended_option_ids": ["OPT-B"],
    "recovery_cost": 9500.0,
    "residual_disruption_loss": 0.0,
    "estimated_total_cost": 9500.0,
    "estimated_savings": 15500.0,
    "feasible_options": [ ... ],
    "rejected_options": [ ... ],
    "assumptions": [ ... ],
    "missing_information": [ ... ],
    "recommendation_explanation": "Recommended recovery option: 'OPT-B'...",
    "requires_human_approval": False
}
```

---

## 6. How Member 5 Can Call the Module (Integration Guide)

Member 5 can integrate the optimization engine into the Streamlit UI with just 4 lines of code:

```python
from agents.financial_optimization_agent import (
    DisruptionScenario,
    RecoveryOption,
    optimize_recovery_plan,
)

# 1. Define the scenario (can be bound to Streamlit input widgets)
scenario = DisruptionScenario(
    scenario_id="SCEN-001",
    affected_quantity=100.0,
    loss_per_unit=250.0,
    disrupted_node="Processor X",
    available_budget=20000.0,
    max_lead_time_days=10.0,
)

# 2. Define candidate recovery options
options = [
    RecoveryOption(
        option_id="OPT-EXPEDITE",
        option_type="expedite_air",
        description="Air freight replacement from secondary supplier",
        recovery_cost=8500.0,
        available_capacity=100.0,
        lead_time_days=5.0,
    ),
    RecoveryOption(
        option_id="OPT-BUFFER",
        option_type="buffer_release",
        description="Release local reserve stock",
        recovery_cost=12000.0,
        available_capacity=100.0,
        lead_time_days=2.0,
    ),
]

# 3. Run deterministic optimization
result = optimize_recovery_plan(scenario, options)

# 4. Display directly in Streamlit
import streamlit as st
st.metric("Baseline Loss", f"${result['baseline_disruption_loss']:,.2f}")
st.metric("Recommended Plan Cost", f"${result['estimated_total_cost']:,.2f}")
st.metric("Estimated Savings", f"${result['estimated_savings']:,.2f}")
st.info(result["recommendation_explanation"])

if result["requires_human_approval"]:
    st.warning("⚠️ Human approval required before executing this recommendation.")
```

---

## 7. How to Run the Automated Unit Tests

From the project root directory, run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_financial_optimization.py -v
```

Or to run the full team test suite (Member 1 + Member 4):
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests -v
```

### Verified Test Results (15/15 Passed):
- `test_01_calculate_gross_disruption_loss`: Validates $Q \times L$.
- `test_02_calculate_recovery_plan_total_cost`: Validates cost + residual.
- `test_03_calculate_estimated_savings`: Validates baseline minus total cost.
- `test_04_recovery_option_within_budget`: Validates feasibility under budget.
- `test_05_recovery_option_exceeding_budget`: Rejects option over budget.
- `test_06_recovery_option_exceeding_available_capacity`: Rejects option with capacity shortfall.
- `test_07_recovery_option_violating_lead_time_limit`: Rejects option exceeding max lead time.
- `test_08_selection_of_lowest_total_cost_feasible_option`: Selects minimal total cost option.
- `test_09_no_feasible_option_available`: Handles zero-feasible scenarios gracefully.
- `test_10_missing_financial_data_reported`: Rejects missing data and logs warnings.
- `test_11_invalid_negative_quantities_or_costs`: Enforces `FinancialValidationError` on negatives.
- `test_12_no_double_counting_of_financial_losses`: Prevents double counting baseline loss.
- `test_13_no_action_baseline_comparison`: Verifies no-action baseline metrics.
- `test_14_correct_structured_output_schema`: Validates all required dictionary keys and types.
- `test_15_multi_option_combination`: Tests combination optimization across multiple suppliers.

---

## 8. Assumptions & Known Limitations

1. **Independent Disruption Events**: Disruption scenarios are evaluated on a per-event basis. Cascading second-order impacts across long time horizons can be modeled by passing aggregated quantities from Member 2/3.
2. **Deterministic Inputs**: Cost and lead time parameters are assumed deterministic point estimates. If Member 3 provides stochastic distributions or confidence intervals, the expected mean should be passed.
3. **Currency Homogeneity**: All financial amounts in a given scenario are assumed to be in the same currency unit (default: USD).
4. **No Hidden Costs**: Avoids speculative assumptions; only explicit recovery costs and residual shortage losses are computed.

