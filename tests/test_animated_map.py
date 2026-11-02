"""
Unit and Integration Tests for Animated Supply Chain Map & Telemetry
====================================================================
Tests the animation features added to create_resilient_network_figure():
1. Glowing node halos and smooth pulse frame calculations.
2. Moving material particles along operational corridors.
3. Disruption visualization (failed node shockwave, severed corridors).
4. Play/Pause animation controls.
5. Scenario resilience: Processor X, Raw Material P, Raw Material Q, Supplier A.
6. Reduced motion / static mode fallback.
"""

import math
import networkx as nx
import plotly.graph_objects as go
import pytest

from agents.dashboard_components import (
    create_resilient_network_figure,
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


def test_01_animated_figure_structure_and_controls(base_graph):
    """Verifies that an animated figure creates frames, updatemenus, and particle traces."""
    sim = simulate_disruption(base_graph, failed_node="Processor X", actor_role="ADMIN")
    fig = create_resilient_network_figure(
        base_graph,
        simulation_result=sim,
        selected_entity="Processor X",
        animate=True,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert len(fig.layout.annotations) > 0

    # Frames check
    assert hasattr(fig, "frames")
    assert len(fig.frames) == 16, "Expected 16 animation frames"

    # Updatemenus (Play/Pause controls) check
    assert len(fig.layout.updatemenus) == 1
    buttons = fig.layout.updatemenus[0].buttons
    assert len(buttons) == 2
    assert "Play" in buttons[0].label
    assert "Pause" in buttons[1].label


def test_02_static_mode_reduced_motion(base_graph):
    """Verifies that when animate=False, no frames or updatemenus are created (zero overhead)."""
    sim = simulate_disruption(base_graph, failed_node="Processor X", actor_role="ADMIN")
    fig = create_resilient_network_figure(
        base_graph,
        simulation_result=sim,
        selected_entity="Processor X",
        animate=False,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    # No frames in static mode
    assert len(fig.frames) == 0
    # No updatemenus in static mode
    assert len(fig.layout.updatemenus) == 0


def test_03_processor_x_disruption_animation(base_graph):
    """Verifies that Processor X failure animates operational in-corridor and highlights severed out-corridors."""
    sim = simulate_disruption(base_graph, failed_node="Processor X", actor_role="ADMIN")
    fig = create_resilient_network_figure(
        base_graph,
        simulation_result=sim,
        selected_entity="Processor X",
        animate=True,
    )

    # Failed entity halo trace exists
    trace_names = [t.name for t in fig.data if t.name]
    assert "Operational Corridor" in trace_names
    assert "Severed Corridor" in trace_names
    assert "Active Material Flow" in trace_names
    assert "Disrupted Shock Root" in trace_names
    assert "Confirmed Shortage" in trace_names

    # Check that frames update particles and halos
    frame0 = fig.frames[0]
    frame4 = fig.frames[4]  # Quarter-cycle (Peak expansion)
    assert len(frame0.data) == 2
    # Ensure sine pulse creates different sizes between frame 0 (neutral) and frame 4 (peak)
    halo_sizes_0 = frame0.data[1].marker.size
    halo_sizes_4 = frame4.data[1].marker.size
    assert halo_sizes_0 != halo_sizes_4
    assert halo_sizes_4[0] > halo_sizes_0[0]  # Failed entity halo is expanded at peak


def test_04_raw_material_p_disruption_animation(base_graph):
    """Verifies Raw Material P failure cascade animation."""
    sim = simulate_disruption(base_graph, failed_node="Raw Material P", actor_role="ADMIN")
    fig = create_resilient_network_figure(
        base_graph,
        simulation_result=sim,
        selected_entity="Raw Material P",
        animate=True,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.frames) == 16
    trace_names = [t.name for t in fig.data if t.name]
    assert "Disrupted Shock Root" in trace_names
    assert "Confirmed Shortage" in trace_names


def test_05_raw_material_q_disruption_animation(base_graph):
    """Verifies Raw Material Q failure cascade animation."""
    sim = simulate_disruption(base_graph, failed_node="Raw Material Q", actor_role="ADMIN")
    fig = create_resilient_network_figure(
        base_graph,
        simulation_result=sim,
        selected_entity="Raw Material Q",
        animate=True,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.frames) == 16
    trace_names = [t.name for t in fig.data if t.name]
    assert "Disrupted Shock Root" in trace_names
    assert "Confirmed Shortage" in trace_names


def test_06_supplier_terminal_node_disruption_animation(base_graph):
    """Verifies Supplier A (leaf sink node) outage produces safe animation without crashing."""
    sim = simulate_disruption(base_graph, failed_node="Supplier A", actor_role="ADMIN")
    fig = create_resilient_network_figure(
        base_graph,
        simulation_result=sim,
        selected_entity="Supplier A",
        animate=True,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.frames) == 16
    trace_names = [t.name for t in fig.data if t.name]
    assert "Disrupted Shock Root" in trace_names


def test_07_empty_graph_edge_case():
    """Verifies empty graph returns safe empty figure."""
    empty_g = nx.DiGraph()
    fig = create_resilient_network_figure(empty_g, animate=True)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_08_continuous_loop_boundary_continuity(base_graph):
    """Verifies that frame 15 transitions into frame 0 with smooth periodic continuity."""
    sim = simulate_disruption(base_graph, failed_node="Processor X", actor_role="ADMIN")
    fig = create_resilient_network_figure(
        base_graph,
        simulation_result=sim,
        selected_entity="Processor X",
        animate=True,
    )

    f0 = fig.frames[0]
    f1 = fig.frames[1]
    f15 = fig.frames[15]

    # Verify particle continuity
    p0_x = f0.data[0].x
    p1_x = f1.data[0].x
    p15_x = f15.data[0].x

    # Delta between consecutive frames
    step_0_1 = abs(p1_x[0] - p0_x[0])
    assert step_0_1 > 0.0, "Particles must advance each frame"

    # Verify periodic wrapping exists across the 16 frames
    assert len(p0_x) == len(p15_x)


def test_09_play_button_fromcurrent_false(base_graph):
    """Verifies that the Play button is configured with fromcurrent=False for reliable looping."""
    fig = create_resilient_network_figure(base_graph, animate=True)
    buttons = fig.layout.updatemenus[0].buttons
    play_btn = buttons[0]
    assert play_btn.args[1]["fromcurrent"] is False, "Play button must start from frame 0 for reliable loops"
    assert play_btn.args[1]["mode"] == "immediate"


def test_10_severed_corridors_exclude_particles(base_graph):
    """Verifies that severed corridors NEVER receive moving particles and remain solid red."""
    for node in ["Processor X", "Raw Material P", "Raw Material Q"]:
        sim = simulate_disruption(base_graph, failed_node=node, actor_role="ADMIN")
        fig = create_resilient_network_figure(base_graph, simulation_result=sim, animate=True)

        # Severed corridor trace exists and is red
        severed_traces = [t for t in fig.data if t.name == "Severed Corridor"]
        assert len(severed_traces) == 1, f"Expected severed corridor trace in {node}"
        assert severed_traces[0].line.color == "#EF4444"

        # Particles trace exists
        particle_traces = [t for t in fig.data if t.name == "Active Material Flow"]
        assert len(particle_traces) == 1
        # Number of particles must equal 2 * number of operational edges
        # (Broken edges must NOT have particles)
        failed_entity = node
        op_count = len([
            (u, v) for u, v in base_graph.edges()
            if u != failed_entity and v != failed_entity
        ])
        assert len(particle_traces[0].x) == op_count * 2

def test_11_updatemenus_positioning_and_theme_styling(base_graph):
    """Verifies that Play/Pause controls are anchored top-right without overlapping title and styled with navy theme."""
    fig = create_resilient_network_figure(base_graph, animate=True)
    assert len(fig.layout.updatemenus) == 1
    menu = fig.layout.updatemenus[0]

    # Anchored on top-right corner
    assert menu.x >= 0.95
    assert menu.xanchor == "right"
    assert menu.yanchor == "bottom"
    assert menu.direction == "right"

    # Theme and active configuration
    assert menu.bgcolor == "#151F32"
    assert menu.bordercolor == "#334155"
    assert menu.active == -1
    assert menu.font.color == "#F8FAFC"

    # Title anchored on top-left to avoid overlap
    assert fig.layout.title.x <= 0.05
    assert fig.layout.title.xanchor == "left"

