"""
Unit Tests for Nexus-AI Decision Explanation Agent (Member 5)
=============================================================

Validates the deterministic 9-point decision explanation component:
1. Root cause and failed entity narrative.
2. Affected downstream entities, confirmed shortages, and materials.
3. Vulnerability context and shared-risk cross-referencing.
4. Financial impact formulas and calculations.
5. Selected recovery plan justification.
6. Evaluation of feasible and rejected alternatives with constraint explanations.
7. Complete honest disclosure of assumptions and missing operational data.
8. Remaining residual risks and budget depletion metrics.
9. Governance and human approval triggers.
"""

import pytest
import networkx as nx

from agents.decision_explanation_agent import (
    DecisionExplanationAgent,
    generate_decision_explanation,
)
from agents.disruption_simulation_agent import (
    create_synthetic_supply_chain as create_sim_chain,
    simulate_disruption,
)
from agents.risk_detection_agent import detect_shared_risks
from agents.financial_optimization_agent import (
    DisruptionScenario,
    RecoveryOption,
    optimize_recovery_plan,
)
from agents.dashboard_components import (
    build_financial_scenario_from_simulation,
    generate_candidate_recovery_options,
)


@pytest.fixture
def base_supply_chain():
    return create_sim_chain()


@pytest.fixture
def executed_pipeline(base_supply_chain):
    """Executes the complete upstream multi-agent pipeline for Processor X."""
    # Member 2: Shared risks
    risk_findings = detect_shared_risks(base_supply_chain, only_shared=False)

    # Member 3: Disruption simulation
    sim_result = simulate_disruption(base_supply_chain, failed_node="Processor X")

    # Member 4: Financial optimization
    scenario = build_financial_scenario_from_simulation(
        simulation_result=sim_result,
        loss_per_unit=150.0,
        available_budget=50000.0,
        max_lead_time_days=14.0,
    )
    candidate_options = generate_candidate_recovery_options(scenario, graph=base_supply_chain)
    fin_result = optimize_recovery_plan(scenario, candidate_options, allow_multi_option=True)

    return {
        "graph": base_supply_chain,
        "risk_findings": risk_findings,
        "sim_result": sim_result,
        "fin_result": fin_result,
    }


def test_01_what_failed_explanation(executed_pipeline):
    agent = DecisionExplanationAgent()
    exp = agent.explain(
        graph=executed_pipeline["graph"],
        disruption_result=executed_pipeline["sim_result"],
        financial_result=executed_pipeline["fin_result"],
        risk_findings=executed_pipeline["risk_findings"],
    )

    what_failed = exp["what_failed"]
    assert what_failed["failed_entity"] == "Processor X"
    assert what_failed["disruption_type"] == "Processor failure"
    assert what_failed["direct_capacity_loss"] == 200.0
    assert "Processor X" in what_failed["narrative"]
    assert "200.0 throughput units" in what_failed["narrative"]


def test_02_affected_entities_explanation(executed_pipeline):
    agent = DecisionExplanationAgent()
    exp = agent.explain(
        graph=executed_pipeline["graph"],
        disruption_result=executed_pipeline["sim_result"],
        financial_result=executed_pipeline["fin_result"],
        risk_findings=executed_pipeline["risk_findings"],
    )

    affected = exp["affected_entities"]
    assert affected["total_affected_count"] == 4
    assert "Supplier A" in affected["all_affected"]
    assert "Supplier B" in affected["all_affected"]
    assert "Factory 1" in affected["confirmed_shortages"]
    assert "Factory 2" in affected["potentially_affected"]
    assert "Semiconductor Wafers" in affected["affected_materials"]
    assert "Confirmed Shortages" in affected["narrative"]
    assert "Potentially Affected" in affected["narrative"]


def test_03_why_it_matters_shared_risk_context(executed_pipeline):
    agent = DecisionExplanationAgent()
    exp = agent.explain(
        graph=executed_pipeline["graph"],
        disruption_result=executed_pipeline["sim_result"],
        financial_result=executed_pipeline["fin_result"],
        risk_findings=executed_pipeline["risk_findings"],
    )

    why = exp["why_it_matters"]
    assert why["impact_level"] == "Critical"
    assert why["is_shared_dependency"] is True
    assert why["shared_risk_score"] is not None
    assert "SHARED SINGLE POINT OF FAILURE" in why["narrative"]


def test_04_financial_calculation_details(executed_pipeline):
    agent = DecisionExplanationAgent()
    exp = agent.explain(
        graph=executed_pipeline["graph"],
        disruption_result=executed_pipeline["sim_result"],
        financial_result=executed_pipeline["fin_result"],
        risk_findings=executed_pipeline["risk_findings"],
    )

    fin_calc = exp["financial_calculation"]
    assert fin_calc["baseline_loss"] == 200.0 * 150.0  # $30,000.00
    assert fin_calc["affected_quantity"] == 200.0
    assert fin_calc["loss_per_unit"] == 150.0
    assert fin_calc["estimated_savings"] > 0
    assert "Gross Disruption Loss" in fin_calc["loss_formula"]
    assert "Recovery Plan Total Cost" in fin_calc["cost_formula"]


def test_05_selected_recovery_option(executed_pipeline):
    agent = DecisionExplanationAgent()
    exp = agent.explain(
        graph=executed_pipeline["graph"],
        disruption_result=executed_pipeline["sim_result"],
        financial_result=executed_pipeline["fin_result"],
        risk_findings=executed_pipeline["risk_findings"],
    )

    selected = exp["selected_recovery_option"]
    assert selected["is_no_action_baseline"] is False
    assert len(selected["selected_ids"]) > 0
    assert "Selected Plan:" in selected["narrative"]


def test_06_alternatives_considered_breakdown(executed_pipeline):
    agent = DecisionExplanationAgent()
    exp = agent.explain(
        graph=executed_pipeline["graph"],
        disruption_result=executed_pipeline["sim_result"],
        financial_result=executed_pipeline["fin_result"],
        risk_findings=executed_pipeline["risk_findings"],
    )

    alts = exp["alternatives_considered"]
    assert alts["total_considered"] >= 4
    assert alts["rejected_count"] >= 1
    assert "Infeasible Alternatives Rejected" in alts["narrative"]


def test_07_assumptions_and_honest_missing_data(executed_pipeline):
    agent = DecisionExplanationAgent()
    exp = agent.explain(
        graph=executed_pipeline["graph"],
        disruption_result=executed_pipeline["sim_result"],
        financial_result=executed_pipeline["fin_result"],
        risk_findings=executed_pipeline["risk_findings"],
    )

    disc = exp["assumptions_and_missing_data"]
    assert len(disc["assumptions"]) > 0
    assert "NOT hallucinated" in disc["narrative"]


def test_08_residual_risks(executed_pipeline):
    agent = DecisionExplanationAgent()
    exp = agent.explain(
        graph=executed_pipeline["graph"],
        disruption_result=executed_pipeline["sim_result"],
        financial_result=executed_pipeline["fin_result"],
        risk_findings=executed_pipeline["risk_findings"],
    )

    res = exp["residual_risks"]
    assert len(res["residual_risks_list"]) > 0
    assert "Financial Exposure:" in res["narrative"]
    assert "Capital Depletion:" in res["narrative"]


def test_09_human_approval_trigger_conditions(base_supply_chain):
    agent = DecisionExplanationAgent()

    # Case A: High budget utilization scenario (> 85%)
    sim_res = simulate_disruption(base_supply_chain, failed_node="Processor X")
    scen = DisruptionScenario(
        scenario_id="SCEN-HIGH-UTIL",
        affected_quantity=100.0,
        loss_per_unit=200.0,
        available_budget=10000.0,  # Small budget
        max_lead_time_days=14.0,
    )
    opts = [
        RecoveryOption(
            option_id="OPT-EXPENSIVE",
            option_type="air",
            description="Air shipment",
            recovery_cost=9000.0,  # 90% of budget!
            available_capacity=100.0,
            lead_time_days=5.0,
        )
    ]
    fin_res = optimize_recovery_plan(scen, opts)
    exp = agent.explain(
        graph=base_supply_chain,
        disruption_result=sim_res,
        financial_result=fin_res,
    )

    assert exp["human_approval"]["required"] is True
    assert "High emergency budget utilization" in "; ".join(exp["human_approval"]["triggers"])
    assert "[ALERT: HUMAN APPROVAL REQUIRED]" in exp["human_approval"]["narrative"]


def test_10_markdown_report_formatting(executed_pipeline):
    exp = generate_decision_explanation(
        graph=executed_pipeline["graph"],
        disruption_result=executed_pipeline["sim_result"],
        financial_result=executed_pipeline["fin_result"],
        risk_findings=executed_pipeline["risk_findings"],
    )

    md = exp["markdown_report"]
    assert isinstance(md, str)
    assert "# Nexus-AI: Multi-Tier Supply Chain Decision Explanation" in md
    assert "### 1. Root Cause & Disrupted Entity" in md
    assert "### 2. Downstream Blast Radius & Impacted Entities" in md
    assert "### 3. Criticality & Vulnerability Context" in md
    assert "### 4. Financial Impact Computation" in md
    assert "### 5. Recommended Recovery Decision" in md
    assert "### 6. Alternatives Evaluated" in md
    assert "### 7. Governance, Assumptions & Data Disclosures" in md
    assert "### 8. Post-Recovery Residual Risks" in md
    assert "### 9. Governance & Approval Protocol" in md

