"""
Unit and Integration Tests for Feature 3: Disruption Propagation Animation
==========================================================================
Tests:
1. create_disruption_propagation_figure() figure structure, frames, and updatemenus.
2. Play, Pause, and Replay controls (3 buttons, fromcurrent=False).
3. Origin Expanding Shockwave Ring (centered on failed entity, expanding across frames).
4. Disruption Shock Signal particles along severed corridors.
5. Scenario 1: Processor X (origin Processor X, downstream Supplier A & B).
6. Scenario 2: Raw Material P (multi-tier cascade RM P -> Processor X -> Supplier A & B).
7. Scenario 3: Raw Material Q (cascade RM Q -> Processor Y -> Supplier C).
8. render_disruption_propagation_summary() output, metrics, and zero Markdown raw HTML bugs.
9. Static mode / reduced-motion fallback (animate=False).
10. Backward compatibility for create_resilient_network_figure().
"""

import math
import networkx as nx
import plotly.graph_objects as go
import pytest

from agents.dashboard_components import (
    create_disruption_propagation_figure,
    create_resilient_network_figure,
    render_disruption_propagation_summary,
    classify_node_status,
)
from agents.mapping_agent import (
    DEFAULT_DEPENDENCIES_CSV,
    build_supply_chain_graph,
    load_and_validate_dependencies,
)
from agents.disruption_simulation_agent import simulate_disruption


@pytest.fixture
def base_graph():
    df = load_and_validate_dependencies(DEFAULT_DEPENDENCIES_CSV)
    return build_supply_chain_graph(df)


def test_01_propagation_figure_creation_and_controls(base_graph):
    """Verifies that create_disruption_propagation_figure builds 16 frames and 3 buttons (Play, Pause, Replay)."""
    sim = simulate_disruption(base_graph, failed_node="Processor X", actor_role="ADMIN")
    fig = create_disruption_propagation_figure(
        graph=base_graph,
        simulation_result=sim,
        selected_entity="Processor X",
        animate=True,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert len(fig.layout.annotations) > 0

    # Frames check
    assert hasattr(fig, "frames")
    assert len(fig.frames) == 16, "Expected 16 animation frames for smooth propagation cycle"

    # Updatemenus check (Play, Pause, Replay)
    assert len(fig.layout.updatemenus) == 1
    buttons = fig.layout.updatemenus[0].buttons
    assert len(buttons) == 3, "Expected Play, Pause, and Replay buttons"
    assert "Play" in buttons[0].label
    assert "Pause" in buttons[1].label
    assert "Replay" in buttons[2].label

    # Check button execution parameters
    assert buttons[0].args[1]["fromcurrent"] is False
    assert buttons[2].args[1]["fromcurrent"] is False
    assert list(buttons[2].args[0]) == ["frame_0"], "Replay button must jump directly to frame_0"


def test_02_origin_shockwave_ring_and_expansion(base_graph):
    """Verifies that the origin expanding shockwave ring exists and expands across frame cycles."""
    sim = simulate_disruption(base_graph, failed_node="Processor X", actor_role="ADMIN")
    fig = create_disruption_propagation_figure(
        graph=base_graph,
        simulation_result=sim,
        selected_entity="Processor X",
        animate=True,
    )

    trace_names = [t.name for t in fig.data if t.name]
    assert "Origin Expanding Shockwave" in trace_names, "Origin shockwave ring trace must be added"

    # Inspect shockwave ring expansion across frames
    f0 = fig.frames[0]
    f8 = fig.frames[8]
    f15 = fig.frames[15]

    # Trace index for shockwave ring is the last frame trace
    shock_sz_0 = f0.data[-1].marker.size[0]
    shock_sz_8 = f8.data[-1].marker.size[0]
    shock_sz_15 = f15.data[-1].marker.size[0]

    assert shock_sz_8 > shock_sz_0, "Shockwave ring must expand outward from origin"
    assert shock_sz_15 > shock_sz_8, "Shockwave ring must reach peak radius at end of cycle"


def test_03_disruption_shock_particles_along_severed_corridors(base_graph):
    """Verifies that amber/crimson shock particles travel along severed supply corridors."""
    sim = simulate_disruption(base_graph, failed_node="Processor X", actor_role="ADMIN")
    fig = create_disruption_propagation_figure(
        graph=base_graph,
        simulation_result=sim,
        selected_entity="Processor X",
        animate=True,
    )

    trace_names = [t.name for t in fig.data if t.name]
    assert "Disruption Shock Signal" in trace_names
    assert "Active Material Flow" in trace_names

    # Check particle movement across frames
    f0 = fig.frames[0]
    f1 = fig.frames[1]
    # Disruption particles trace is at index 2 in frame data
    p0_x = f0.data[2].x
    p1_x = f1.data[2].x
    assert len(p0_x) > 0
    assert p0_x != p1_x, "Shock particles must advance position in each frame"


def test_04_processor_x_propagation_cascade(base_graph):
    """
    Scenario 1: Processor X
    Origin: Processor X
    Downstream affected suppliers: Supplier A and Supplier B.
    Unaffected entities: Raw Material Q, Processor Y, Supplier C.
    """
    sim = simulate_disruption(base_graph, failed_node="Processor X", actor_role="ADMIN")
    fig = create_disruption_propagation_figure(
        graph=base_graph,
        simulation_result=sim,
        selected_entity="Processor X",
        animate=True,
    )

    assert sim["failed_entity"] == "Processor X"
    assert set(sim["affected_entities"]) == {"Supplier A", "Supplier B"}

    # Severed corridors must include both Processor X -> Supplier A and Processor X -> Supplier B
    paths = sim.get("affected_paths", [])
    expected_edges = {("Processor X", "Supplier A"), ("Processor X", "Supplier B")}
    path_edges = set()
    for p in paths:
        for i in range(len(p) - 1):
            path_edges.add((p[i], p[i + 1]))
    assert expected_edges.issubset(path_edges)

    # Check node status classification
    status_x, _ = classify_node_status("Processor X", sim)
    status_a, _ = classify_node_status("Supplier A", sim)
    status_b, _ = classify_node_status("Supplier B", sim)
    status_y, _ = classify_node_status("Processor Y", sim)
    status_c, _ = classify_node_status("Supplier C", sim)

    assert status_x == "failed"
    assert status_a == "confirmed_shortage"
    assert status_b == "confirmed_shortage"
    assert status_y == "unaffected"
    assert status_c == "unaffected"


def test_05_raw_material_p_cascade_multi_hop(base_graph):
    """
    Scenario 2: Raw Material P
    Origin: Raw Material P
    Cascade travels through Processor X to Supplier A and Supplier B.
    Unaffected entities: Raw Material Q, Processor Y, Supplier C.
    """
    sim = simulate_disruption(base_graph, failed_node="Raw Material P", actor_role="ADMIN")
    fig = create_disruption_propagation_figure(
        graph=base_graph,
        simulation_result=sim,
        selected_entity="Raw Material P",
        animate=True,
    )

    assert sim["failed_entity"] == "Raw Material P"
    assert set(sim["affected_entities"]) == {"Processor X", "Supplier A", "Supplier B"}

    # Cascade paths must travel through Processor X
    paths = sim.get("affected_paths", [])
    assert any("Processor X" in p and "Supplier A" in p for p in paths)
    assert any("Processor X" in p and "Supplier B" in p for p in paths)

    # Status checks
    status_p, _ = classify_node_status("Raw Material P", sim)
    status_x, _ = classify_node_status("Processor X", sim)
    status_a, _ = classify_node_status("Supplier A", sim)
    status_y, _ = classify_node_status("Processor Y", sim)

    assert status_p == "failed"
    assert status_x == "confirmed_shortage"
    assert status_a == "confirmed_shortage"
    assert status_y == "unaffected"


def test_06_raw_material_q_cascade(base_graph):
    """
    Scenario 3: Raw Material Q
    Origin: Raw Material Q
    Cascade travels through Processor Y to Supplier C.
    Unaffected entities: Raw Material P, Processor X, Supplier A, Supplier B.
    """
    sim = simulate_disruption(base_graph, failed_node="Raw Material Q", actor_role="ADMIN")
    fig = create_disruption_propagation_figure(
        graph=base_graph,
        simulation_result=sim,
        selected_entity="Raw Material Q",
        animate=True,
    )

    assert sim["failed_entity"] == "Raw Material Q"
    assert set(sim["affected_entities"]) == {"Processor Y", "Supplier C"}

    # Cascade path travels through Processor Y to Supplier C
    paths = sim.get("affected_paths", [])
    assert ["Raw Material Q", "Processor Y", "Supplier C"] in paths

    status_q, _ = classify_node_status("Raw Material Q", sim)
    status_y, _ = classify_node_status("Processor Y", sim)
    status_c, _ = classify_node_status("Supplier C", sim)
    status_x, _ = classify_node_status("Processor X", sim)

    assert status_q == "failed"
    assert status_y == "confirmed_shortage"
    assert status_c == "confirmed_shortage"
    assert status_x == "unaffected"


def test_07_propagation_impact_summary_html_and_metrics(base_graph):
    """Verifies that render_disruption_propagation_summary() produces valid, unindented HTML with accurate metrics."""
    sim = simulate_disruption(base_graph, failed_node="Processor X", actor_role="ADMIN")
    summary_html = render_disruption_propagation_summary(sim, base_graph)

    assert isinstance(summary_html, str)
    assert "Processor X" in summary_html
    assert "CRITICAL IMPACT" in summary_html
    assert "Downstream Blast Radius" in summary_html
    assert ">2 <" in summary_html  # 2 Entities impacted
    assert "Supplier A, Supplier B" in summary_html  # Detailed entity list
    assert "Confirmed vs Potential Shortages" in summary_html
    assert "Severed Supply Corridors" in summary_html

    # Ensure no raw Markdown indented code block bug exists
    lines = summary_html.splitlines()
    for idx, line in enumerate(lines):
        assert not line.startswith("    "), f"Line {idx} has leading 4-space indentation which triggers raw HTML bug"
        assert not line.startswith("\t"), f"Line {idx} has leading tab indentation"

    # Test fallback when simulation_result is None
    empty_html = render_disruption_propagation_summary(None, base_graph)
    assert "No active disruption simulation" in empty_html


def test_08_static_mode_reduced_motion(base_graph):
    """Verifies that animate=False returns a clean static figure with zero frames and no updatemenus."""
    sim = simulate_disruption(base_graph, failed_node="Processor X", actor_role="ADMIN")
    fig = create_disruption_propagation_figure(
        graph=base_graph,
        simulation_result=sim,
        selected_entity="Processor X",
        animate=False,
    )

    assert len(fig.frames) == 0, "Static mode must have 0 animation frames"
    assert len(fig.layout.updatemenus) == 0, "Static mode must have 0 updatemenus"
    assert len(fig.data) > 0, "Static mode must preserve all routes and node scatter traces"


def test_09_backward_compatibility_create_resilient_network_figure(base_graph):
    """Verifies that create_resilient_network_figure() defaults preserve backward compatibility with existing tests."""
    sim = simulate_disruption(base_graph, failed_node="Processor X", actor_role="ADMIN")
    # Default call without show_replay or show_propagation_signal
    fig = create_resilient_network_figure(
        graph=base_graph,
        simulation_result=sim,
        selected_entity="Processor X",
        animate=True,
    )

    # Must have exactly 2 buttons by default (Play, Pause) for test_animated_map.py compliance
    assert len(fig.layout.updatemenus[0].buttons) == 2
    # Must have exactly 2 frame data items by default for test_animated_map.py compliance
    assert len(fig.frames[0].data) == 2


def test_10_propagation_trajectory_parallel_branches_vs_sequential_chain(base_graph):
    """
    Regression test for Issue 2:
    Verifies that Propagation Trajectory in render_disruption_propagation_summary()
    represents Processor X -> Supplier A and Processor X -> Supplier B as parallel branches
    and NEVER as a sequential chain 'Processor X -> Supplier A -> Supplier B'.
    """
    # 1. Processor X: two independent downstream branches
    sim_x = simulate_disruption(base_graph, failed_node="Processor X", actor_role="ADMIN")
    html_x = render_disruption_propagation_summary(sim_x, base_graph)

    assert "Parallel Branches" in html_x
    assert "Processor X &rarr; Supplier A" in html_x
    assert "Processor X &rarr; Supplier B" in html_x
    assert "Supplier A &rarr; Supplier B" not in html_x
    assert "Processor X &rarr; Supplier A &rarr; Supplier B" not in html_x

    # 2. Raw Material P: two multi-hop parallel branches through Processor X
    sim_p = simulate_disruption(base_graph, failed_node="Raw Material P", actor_role="ADMIN")
    html_p = render_disruption_propagation_summary(sim_p, base_graph)

    assert "Parallel Branches" in html_p
    assert "Raw Material P &rarr; Processor X &rarr; Supplier A" in html_p
    assert "Raw Material P &rarr; Processor X &rarr; Supplier B" in html_p
    assert "Supplier A &rarr; Supplier B" not in html_p

    # 3. Raw Material Q: single linear cascade
    sim_q = simulate_disruption(base_graph, failed_node="Raw Material Q", actor_role="ADMIN")
    html_q = render_disruption_propagation_summary(sim_q, base_graph)

    assert "Directed Cascade" in html_q
    assert "Raw Material Q &rarr; Processor Y &rarr; Supplier C" in html_q

    # 4. Terminal sink node (Supplier A): localized outage
    sim_a = simulate_disruption(base_graph, failed_node="Supplier A", actor_role="ADMIN")
    html_a = render_disruption_propagation_summary(sim_a, base_graph)

    assert "Localized Outage" in html_a
    assert "Supplier A (Terminal Sink)" in html_a


def test_11_disruption_simulator_ui_downstream_status_and_trajectory_consistency(tmp_path, monkeypatch):
    """
    Regression test verifying that the Streamlit Disruption Simulator view (Section 4)
    renders Supplier A and Supplier B with CONFIRMED SHORTAGE, Severed Inbound Paths: 1,
    Operational Status vs Shortage Status distinction, Backup Alternatives: Unknown,
    and parallel propagation trajectory branches.
    """
    import time
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from agents.auth_manager import AuthManager, UserRole

    temp_auth_db = tmp_path / "test_nexus_auth.db"
    temp_audit_db = tmp_path / "test_nexus_audit.db"
    monkeypatch.setenv("NEXUS_AUTH_DB_PATH", str(temp_auth_db))
    monkeypatch.setenv("NEXUS_AUDIT_DB_PATH", str(temp_audit_db))

    root_dir = Path(__file__).resolve().parent.parent
    app_path = str(root_dir / "app.py")
    mgr = AuthManager(str(temp_auth_db))
    ok_adm, _ = mgr.init_first_admin("admin_test", "admin@nexus.test", "AdminPass123!")
    assert ok_adm
    ok_view, _, viewer_u = mgr.create_user(
        "admin_test", "viewer_test", "viewer@nexus.test", "ViewerPass123!", UserRole.VIEWER
    )
    assert ok_view and viewer_u is not None

    at = AppTest.from_file(app_path, default_timeout=30)
    at.session_state["nexus_auth_authenticated"] = True
    at.session_state["nexus_auth_user"] = viewer_u
    at.session_state["nexus_auth_last_activity"] = time.time()
    at.run()

    # Navigate to Section 4: Disruption Simulator
    at.sidebar.radio[0].set_value("4. ⚡ Disruption Simulator").run()

    all_markdown = "\n".join(m.value for m in at.markdown)

    # Issue 1 UI assertions
    assert "CONFIRMED SHORTAGE" in all_markdown
    assert "Operational (Inbound Supply Starved)" in all_markdown
    assert "Confirmed Shortage" in all_markdown
    assert '<b>Severed Inbound Paths:</b> <span style="color:#F8FAFC;">1</span>' in all_markdown
    assert '<b>Operational Inbound Paths:</b> <span style="color:#F8FAFC;">0</span>' in all_markdown
    assert '<b>Backup Alternatives:</b> <span style="color:#CBD5E1;">Unknown</span>' in all_markdown

    # Issue 2 UI assertions
    assert "Parallel Branches" in all_markdown
    assert "Processor X &rarr; Supplier A" in all_markdown
    assert "Processor X &rarr; Supplier B" in all_markdown
    assert "Supplier A &rarr; Supplier B" not in all_markdown



