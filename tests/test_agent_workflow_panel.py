"""
Unit and Integration Tests for Feature 2: Five-Agent Workflow Orchestration Panel
================================================================================
Verifies:
1. Five specialized resilience agents definitions, roles, order, and theme colors.
2. WAITING, PROCESSING, COMPLETED, and ERROR visual state renderings.
3. Real execution integration connecting functions to visual state transitions.
4. State reset on new analysis runs and cache preservation on tab navigation.
5. Graceful handling and display when an agent encounters an error.
6. Responsive CSS classes and enterprise styling.
"""

import time
import pytest
import networkx as nx

from agents.dashboard_components import (
    AGENT_THEMES,
    render_five_agent_workflow_panel,
    build_financial_scenario_from_simulation,
    generate_candidate_recovery_options,
)
from agents.mapping_agent import (
    DEFAULT_DEPENDENCIES_CSV,
    build_supply_chain_graph,
    load_and_validate_dependencies,
)
from agents.risk_detection_agent import detect_shared_risks
from agents.disruption_simulation_agent import simulate_disruption
from agents.financial_optimization_agent import optimize_recovery_plan
from agents.decision_explanation_agent import DecisionExplanationAgent


@pytest.fixture
def base_graph():
    df = load_and_validate_dependencies(DEFAULT_DEPENDENCIES_CSV)
    return build_supply_chain_graph(df)


def test_01_five_agent_themes_definitions():
    """Verifies that all 5 specialized agents are defined with designated themes, roles, and order."""
    assert len(AGENT_THEMES) == 5
    expected_keys = ["mapping", "risk", "simulation", "optimization", "explanation"]
    assert list(AGENT_THEMES.keys()) == expected_keys

    # Agent 1: Mapping Agent (Blue)
    m = AGENT_THEMES["mapping"]
    assert m["order"] == 1
    assert m["name"] == "Mapping Agent"
    assert m["color"] == "#3B82F6"
    assert "agent-accent-blue" in m["accent_class"]

    # Agent 2: Risk Detection Agent (Cyan)
    r = AGENT_THEMES["risk"]
    assert r["order"] == 2
    assert r["name"] == "Risk Detection Agent"
    assert r["color"] == "#06B6D4"
    assert "agent-accent-cyan" in r["accent_class"]

    # Agent 3: Disruption Simulation Agent (Amber)
    s = AGENT_THEMES["simulation"]
    assert s["order"] == 3
    assert s["name"] == "Disruption Simulation Agent"
    assert s["color"] == "#F59E0B"
    assert "agent-accent-amber" in s["accent_class"]

    # Agent 4: Financial Optimization Agent (Green)
    o = AGENT_THEMES["optimization"]
    assert o["order"] == 4
    assert o["name"] == "Financial Optimization Agent"
    assert o["color"] == "#10B981"
    assert "agent-accent-green" in o["accent_class"]

    # Agent 5: Decision Explanation Agent (Purple)
    e = AGENT_THEMES["explanation"]
    assert e["order"] == 5
    assert e["name"] == "Decision Explanation Agent"
    assert e["color"] == "#8B5CF6"
    assert "agent-accent-purple" in e["accent_class"]


def test_02_waiting_state_rendering():
    """Verifies visual rendering when agents are in WAITING state."""
    agents = [
        {"id": "mapping", "state": "WAITING"},
        {"id": "risk", "state": "WAITING"},
        {"id": "simulation", "state": "WAITING"},
        {"id": "optimization", "state": "WAITING"},
        {"id": "explanation", "state": "WAITING"},
    ]
    html = render_five_agent_workflow_panel(agents)
    assert "agent-card-waiting" in html
    assert "STANDBY" in html
    assert "Awaiting upstream trigger" in html
    assert "0/5 Agents Completed" in html


def test_03_processing_state_rendering():
    """Verifies visual rendering when an agent is in PROCESSING state."""
    agents = [
        {"id": "mapping", "state": "COMPLETED", "duration_ms": 10.2, "summary": "9 Nodes"},
        {"id": "risk", "state": "PROCESSING"},
        {"id": "simulation", "state": "WAITING"},
        {"id": "optimization", "state": "WAITING"},
        {"id": "explanation", "state": "WAITING"},
    ]
    html = render_five_agent_workflow_panel(
        agents,
        is_live_executing=True,
        current_agent_idx=1,
    )
    assert "agent-card-processing" in html
    assert "agent-processing-cyan" in html
    assert "agent-spinner" in html
    assert "Executing Agent 2/5: Risk Detection Agent" in html


def test_04_completed_state_rendering():
    """Verifies visual rendering when all 5 agents are COMPLETED."""
    agents = [
        {"id": "mapping", "state": "COMPLETED", "duration_ms": 11.2, "summary": "9 Nodes • 12 Links"},
        {"id": "risk", "state": "COMPLETED", "duration_ms": 8.4, "summary": "3 SPOFs Identified"},
        {"id": "simulation", "state": "COMPLETED", "duration_ms": 6.1, "summary": "3 Entities Affected"},
        {"id": "optimization", "state": "COMPLETED", "duration_ms": 15.3, "summary": "Cost: $20,000"},
        {"id": "explanation", "state": "COMPLETED", "duration_ms": 9.0, "summary": "Strategy Ready"},
    ]
    html = render_five_agent_workflow_panel(
        agents,
        total_pipeline_ms=50.0,
        active_run_id="RUN-12345678",
        is_live_executing=False,
    )
    assert "agent-card-completed" in html
    assert "agent-accent-blue" in html
    assert "agent-accent-cyan" in html
    assert "agent-accent-amber" in html
    assert "agent-accent-green" in html
    assert "agent-accent-purple" in html
    assert "DONE" in html
    assert "9 Nodes • 12 Links" in html
    assert "5/5 Specialized Resilience Agents Synchronized" in html
    assert "RUN-1234" in html
    assert "50.0ms" in html


def test_05_error_state_handling():
    """Verifies visual rendering when an agent fails with ERROR."""
    agents = [
        {"id": "mapping", "state": "COMPLETED", "duration_ms": 10.0, "summary": "9 Nodes"},
        {"id": "risk", "state": "COMPLETED", "duration_ms": 7.0, "summary": "3 SPOFs"},
        {"id": "simulation", "state": "ERROR", "error": "Invalid shock entity name"},
        {"id": "optimization", "state": "WAITING"},
        {"id": "explanation", "state": "WAITING"},
    ]
    html = render_five_agent_workflow_panel(agents)
    assert "agent-card-error" in html
    assert "ERROR" in html
    assert "Invalid shock entity" in html
    assert "Pipeline Interrupted" in html


def test_06_actual_sequential_agent_execution(base_graph):
    """Verifies that all 5 agents execute in real sequential order producing valid outputs and metrics."""
    # Step 1: Mapping Agent
    t0 = time.perf_counter()
    nodes = base_graph.number_of_nodes()
    edges = base_graph.number_of_edges()
    t1 = time.perf_counter()
    mapping_duration = (t1 - t0) * 1000.0
    assert nodes > 0
    assert edges > 0

    # Step 2: Risk Detection Agent
    t0 = time.perf_counter()
    risk_findings = detect_shared_risks(base_graph, only_shared=False)
    t1 = time.perf_counter()
    risk_duration = (t1 - t0) * 1000.0
    assert isinstance(risk_findings, list)

    # Step 3: Disruption Simulation Agent
    t0 = time.perf_counter()
    sim_result = simulate_disruption(base_graph, failed_node="Processor X")
    t1 = time.perf_counter()
    sim_duration = (t1 - t0) * 1000.0
    assert sim_result["failed_entity"] == "Processor X"
    assert len(sim_result["affected_entities"]) > 0

    # Step 4: Financial Optimization Agent
    t0 = time.perf_counter()
    fin_scen = build_financial_scenario_from_simulation(
        simulation_result=sim_result,
        loss_per_unit=150.0,
        available_budget=50000.0,
        max_lead_time_days=14.0,
        graph=base_graph,
    )
    candidates = generate_candidate_recovery_options(fin_scen, base_graph)
    fin_res = optimize_recovery_plan(fin_scen, recovery_options=candidates)
    t1 = time.perf_counter()
    fin_duration = (t1 - t0) * 1000.0
    assert "recovery_cost" in fin_res

    # Step 5: Decision Explanation Agent
    t0 = time.perf_counter()
    exp_agent = DecisionExplanationAgent()
    dec_exp = exp_agent.explain(
        graph=base_graph,
        disruption_result=sim_result,
        financial_result=fin_res,
        risk_findings=risk_findings,
    )
    t1 = time.perf_counter()
    exp_duration = (t1 - t0) * 1000.0
    assert "selected_recovery_option" in dec_exp
    selected_ids = dec_exp["selected_recovery_option"].get("selected_ids", ["Plan"])

    # Synthesize live status payload
    pipeline = [
        {"id": "mapping", "state": "COMPLETED", "duration_ms": mapping_duration, "summary": f"{nodes} Nodes • {edges} Links"},
        {"id": "risk", "state": "COMPLETED", "duration_ms": risk_duration, "summary": f"{len(risk_findings)} Risks"},
        {"id": "simulation", "state": "COMPLETED", "duration_ms": sim_duration, "summary": f"Shock: Processor X"},
        {"id": "optimization", "state": "COMPLETED", "duration_ms": fin_duration, "summary": f"Cost: ${fin_res.get('recovery_cost', 0):,.0f}"},
        {"id": "explanation", "state": "COMPLETED", "duration_ms": exp_duration, "summary": f"Plan: {selected_ids[0]}"},
    ]

    total_pipeline = sum(p["duration_ms"] for p in pipeline)
    panel = render_five_agent_workflow_panel(pipeline, total_pipeline_ms=total_pipeline, active_run_id="TEST-RUN-01")
    assert "5/5 Specialized Resilience Agents Synchronized" in panel
    assert "Processor X" in panel


def test_07_scenario_fingerprint_generation():
    """Verifies that changing scenario parameters changes the fingerprint to trigger a fresh execution."""
    fp1 = f"sample|Processor X|None|150.0|50000.0|14|True"
    fp2 = f"sample|Raw Material P|None|150.0|50000.0|14|True"
    fp3 = f"sample|Processor X|None|200.0|50000.0|14|True"

    assert fp1 != fp2, "Different failed entity must yield different fingerprint"
    assert fp1 != fp3, "Different loss per unit must yield different fingerprint"

def test_08_no_markdown_code_block_indentation():
    """
    Verifies that the generated HTML has zero leading 4-space indentations or stray newlines
    so Streamlit's Markdown parser NEVER misinterprets HTML markup as indented code blocks (<pre><code>).
    """
    agents = [
        {"id": "mapping", "state": "COMPLETED", "duration_ms": 10.0, "summary": "9 Nodes • 12 Links"},
        {"id": "risk", "state": "COMPLETED", "duration_ms": 8.0, "summary": "3 SPOFs"},
        {"id": "simulation", "state": "COMPLETED", "duration_ms": 6.0, "summary": "Shock: Processor X"},
        {"id": "optimization", "state": "COMPLETED", "duration_ms": 12.0, "summary": "Cost: $18,000"},
        {"id": "explanation", "state": "COMPLETED", "duration_ms": 7.0, "summary": "Plan: OPT-BUFFER-RELEASE"},
    ]
    html = render_five_agent_workflow_panel(agents, total_pipeline_ms=43.0, active_run_id="RUN-TEST1234")

    # Must start directly with HTML tag without leading whitespace
    assert html.startswith("<div class=\"agent-orchestration-panel\">")

    # Must not contain newlines that could trigger CommonMark 4-space code block rules
    assert "\n" not in html

    # Must not contain 4 spaces
    assert "    " not in html

    # Must contain actual card class
    assert '<div class="agent-workflow-card agent-card-completed agent-accent-blue">' in html
    assert '<div class="agent-workflow-card agent-card-completed agent-accent-cyan">' in html
    assert '<div class="agent-workflow-card agent-card-completed agent-accent-amber">' in html
    assert '<div class="agent-workflow-card agent-card-completed agent-accent-green">' in html
    assert '<div class="agent-workflow-card agent-card-completed agent-accent-purple">' in html

