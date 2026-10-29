"""
Nexus-AI: Multi-Tier Supply Chain Resilience Platform
Member 5: Streamlit Dashboard, Multi-Agent Integration & Decision Explanation
==============================================================================

Integrates:
- Member 1: Supply Chain Mapping Agent (Graph construction & validation)
- Member 2: Shared Risk Detection Agent (SPOF & shared vulnerability detection)
- Member 3: Disruption Simulation Agent (Blast radius & shortage modeling)
- Member 4: Financial Impact and Recovery Optimization Engine (Deterministic optimization)
- Member 5: Executive Dashboard & 9-Point Decision Explanation Component
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure workspace is on sys.path and configure UTF-8 output on Windows
WORKSPACE_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import networkx as nx
import pandas as pd

# Member 1 Imports
from agents.mapping_agent import (
    DataValidationError,
    build_supply_chain_graph,
    load_and_validate_dependencies,
    DEFAULT_DEPENDENCIES_CSV,
)

# Member 2 Imports
from agents.risk_detection_agent import (
    detect_shared_risks,
    create_synthetic_supply_chain as create_m2_synthetic_chain,
)

# Member 3 Imports
from agents.disruption_simulation_agent import (
    DisruptionSimulationAgent,
    SUPPORTED_DISRUPTION_TYPES,
    create_synthetic_supply_chain as create_canonical_supply_chain,
    simulate_disruption,
)

# Member 4 Imports
from agents.financial_optimization_agent import (
    DisruptionScenario,
    RecoveryOption,
    optimize_recovery_plan,
)

# Member 5 Imports
from agents.decision_explanation_agent import (
    DecisionExplanationAgent,
    generate_decision_explanation,
)
from agents.dashboard_components import (
    build_financial_scenario_from_simulation,
    create_resilient_network_figure,
    generate_candidate_recovery_options,
)


def run_streamlit_app() -> None:
    """Renders the comprehensive Streamlit interactive dashboard."""
    import streamlit as st

    # 1. Page Configuration
    st.set_page_config(
        page_title="Nexus-AI | Supply Chain Resilience",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for dark-mode styling.
    # NOTE: Streamlit renders inside Shadow-DOM-like React containers, so plain
    # "body {}" selectors have no effect.  We must target Streamlit's own
    # data-testid attributes and class names to override colours reliably.
    st.markdown(
        """
        <style>
        /* ── Global app background ───────────────────────────────────── */
        [data-testid="stAppViewContainer"],
        [data-testid="stApp"],
        .main .block-container {
            background-color: #0B1220 !important;
        }

        /* ── Main content area wrapper ──────────────────────────────── */
        section.main > div,
        .block-container {
            background-color: #0B1220 !important;
            color: #F8FAFC !important;
        }

        /* ── Sidebar ────────────────────────────────────────────────── */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
            background-color: #111827 !important;
            border-right: 1px solid #1E293B;
        }
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stNumberInput label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #CBD5E1 !important;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #F8FAFC !important;
        }

        /* ── All generic text in the main panel ─────────────────────── */
        .stMarkdown p,
        .stMarkdown li,
        .stMarkdown span,
        .stMarkdown label,
        p, span, li, label {
            color: #F8FAFC !important;
        }
        h1, h2, h3, h4, h5, h6,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #F8FAFC !important;
        }

        /* ── st.metric cards ────────────────────────────────────────── */
        [data-testid="stMetric"] {
            background: #1E293B !important;
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
            padding: 14px 18px !important;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] p {
            color: #94A3B8 !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] div {
            color: #F8FAFC !important;
            font-size: 1.55rem !important;
            font-weight: 700 !important;
        }
        [data-testid="stMetricDelta"] svg { display: none; }
        [data-testid="stMetricDelta"] {
            color: #2DD4BF !important;
            font-weight: 600 !important;
        }

        /* ── Tabs ───────────────────────────────────────────────────── */
        [data-testid="stTabs"] [role="tablist"] {
            background-color: #1E293B !important;
            border-radius: 8px 8px 0 0;
            border-bottom: 1px solid #334155;
        }
        [data-testid="stTabs"] [role="tab"] {
            color: #94A3B8 !important;
            font-weight: 500;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            color: #2DD4BF !important;
            border-bottom: 2px solid #2DD4BF !important;
            font-weight: 700;
        }
        [data-testid="stTabs"] [role="tabpanel"] {
            background-color: #131C2E !important;
            border: 1px solid #1E293B;
            border-top: none;
            border-radius: 0 0 8px 8px;
            padding: 16px;
        }

        /* ── Dataframes / tables ────────────────────────────────────── */
        /* Style the dataframe toolbar (download, search, fullscreen icons) */
        [data-testid="stDataFrame"] [data-testid="stElementToolbar"],
        [data-testid="stElementToolbar"] {
            background-color: #1E293B !important;
            border: 1px solid #334155 !important;
            border-radius: 6px !important;
        }
        [data-testid="stElementToolbar"] button,
        [data-testid="stElementToolbar"] button svg {
            color: #94A3B8 !important;
            fill: #94A3B8 !important;
        }
        [data-testid="stElementToolbar"] button:hover svg {
            fill: #2DD4BF !important;
        }

        /* ── Selectbox, radio, number inputs ────────────────────────── */
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input {
            background-color: #1E293B !important;
            color: #F8FAFC !important;
            border: 1px solid #334155 !important;
            border-radius: 6px !important;
        }

        /* ── Horizontal rule ────────────────────────────────────────── */
        hr { border-color: #1E293B !important; }

        /* ── Success / warning / error / info banners ───────────────── */
        [data-testid="stAlert"] {
            border-radius: 8px !important;
        }

        /* ── Caption / secondary text ───────────────────────────────── */
        .stCaption, [data-testid="stCaptionContainer"] p {
            color: #64748B !important;
        }

        /* ── Code blocks ────────────────────────────────────────────── */
        [data-testid="stCode"] pre,
        code, pre {
            background-color: #0F172A !important;
            color: #7DD3FC !important;
            border: 1px solid #1E293B !important;
            border-radius: 6px !important;
        }

        /* ── Download button ────────────────────────────────────────── */
        [data-testid="stDownloadButton"] button {
            background-color: #2DD4BF !important;
            color: #0B1220 !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
        }
        [data-testid="stDownloadButton"] button:hover {
            background-color: #14B8A6 !important;
        }

        /* ── Custom HTML elements ───────────────────────────────────── */
        .main-header {
            font-size: 2.1rem;
            font-weight: 800;
            color: #F8FAFC !important;
            letter-spacing: -0.02em;
            margin-bottom: 0.15rem;
        }
        .main-subtitle {
            font-size: 0.95rem;
            color: #2DD4BF;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }
        .sub-header {
            font-size: 1rem;
            color: #94A3B8 !important;
            margin-bottom: 1.2rem;
        }

        /* ── Risk badges ────────────────────────────────────────────── */
        .badge-critical {
            background-color: #DC2626;
            color: #fff;
            padding: 3px 10px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 0.82rem;
        }
        .badge-high {
            background-color: #D97706;
            color: #fff;
            padding: 3px 10px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 0.82rem;
        }
        .badge-medium {
            background-color: #CA8A04;
            color: #fff;
            padding: 3px 10px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 0.82rem;
        }
        .badge-low {
            background-color: #059669;
            color: #fff;
            padding: 3px 10px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 0.82rem;
        }

        /* ── Section dividers ───────────────────────────────────────── */
        .section-divider {
            border: none;
            border-top: 1px solid #1E293B;
            margin: 1.5rem 0;
        }

        /* ── KPI row: uniform column height & gap ───────────────────── */
        [data-testid="column"] {
            padding-left: 6px !important;
            padding-right: 6px !important;
        }
        /* Make every st.metric fill column height */
        [data-testid="stMetric"] {
            height: 100%;
            min-height: 90px;
        }
        /* Reduce oversized metric font on narrow cols */
        @media (max-width: 1200px) {
            [data-testid="stMetricValue"] div { font-size: 1.15rem !important; }
            [data-testid="stMetricLabel"] p  { font-size: 0.72rem !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 2. Sidebar: Configuration & Controls
    st.sidebar.image("https://img.icons8.com/fluency/96/network.png", width=64)
    st.sidebar.title("Nexus-AI Controls")
    st.sidebar.caption("Supply Chain Resilience & Investment Optimization")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 1. Dataset Selection")

    data_choice = st.sidebar.radio(
        "Choose Supply Chain Data Source:",
        [
            "Standard Sample (data/dependencies.csv)",
            "Canonical Multi-Tier Network (with Factories & Routes)",
            "Upload Custom Dependencies CSV",
        ],
        index=0,
    )

    graph: nx.DiGraph
    data_source_name: str = ""

    if data_choice == "Standard Sample (data/dependencies.csv)":
        data_source_name = "Sample Dependencies CSV "
        try:
            df = load_and_validate_dependencies(DEFAULT_DEPENDENCIES_CSV)
            graph = build_supply_chain_graph(df)
            st.sidebar.success(f"Loaded {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        except Exception as exc:
            st.sidebar.error(f"Error loading dependencies.csv: {exc}")
            st.error(f"Data loading error: {exc}")
            return

    elif data_choice == "Canonical Multi-Tier Network (with Factories & Routes)":
        data_source_name = "Canonical Multi-Tier Synthetic Network (Members 2 & 3)"
        graph = create_canonical_supply_chain()
        st.sidebar.success(f"Loaded {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    else:
        data_source_name = "User Uploaded CSV"
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV (must contain upstream_node, downstream_node, material, capacity)",
            type=["csv"],
        )
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = Path(tmp.name)
            try:
                df = load_and_validate_dependencies(tmp_path)
                graph = build_supply_chain_graph(df)
                st.sidebar.success(f"Uploaded: {len(df)} edges, {graph.number_of_nodes()} nodes")
            except DataValidationError as dve:
                st.sidebar.error(f"Validation failed: {dve}")
                st.error(f"Uploaded CSV is invalid: {dve}")
                return
            except Exception as exc:
                st.sidebar.error(f"Upload error: {exc}")
                return
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            st.sidebar.info("Awaiting CSV upload. Falling back to default dataset.")
            df = load_and_validate_dependencies(DEFAULT_DEPENDENCIES_CSV)
            graph = build_supply_chain_graph(df)

    # 3. Sidebar: Disruption Simulator Configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ 2. Disruption Scenario")

    # Node selection
    available_nodes = sorted(list(graph.nodes()))
    default_node_idx = available_nodes.index("Processor X") if "Processor X" in available_nodes else 0
    selected_failed_node = st.sidebar.selectbox(
        "Select Failed Entity:",
        options=available_nodes,
        index=default_node_idx,
    )

    # Disruption type override option
    disruption_type_options = ["Auto-infer from Entity Type"] + list(SUPPORTED_DISRUPTION_TYPES.values())
    selected_disruption_type = st.sidebar.selectbox(
        "Disruption Category:",
        options=disruption_type_options,
        index=0,
    )
    passed_disruption_type = (
        None if selected_disruption_type == "Auto-infer from Entity Type" else selected_disruption_type
    )

    # 4. Sidebar: Financial Optimization Parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 3. Financial Parameters")

    unit_loss_input = st.sidebar.number_input(
        "Loss Per Disrupted Unit ($):",
        min_value=1.0,
        max_value=10000.0,
        value=150.0,
        step=10.0,
    )

    budget_input = st.sidebar.number_input(
        "Emergency Recovery Budget ($):",
        min_value=1000.0,
        max_value=1000000.0,
        value=50000.0,
        step=5000.0,
    )

    max_lead_time_input = st.sidebar.slider(
        "Max Acceptable Lead Time (Days):",
        min_value=1,
        max_value=60,
        value=14,
    )

    allow_multi_option = st.sidebar.checkbox(
        "Enable Multi-Supplier Combination Optimization",
        value=True,
    )

    # =========================================================================
    # Run Agent Pipelines
    # =========================================================================

    # 1. Member 2: Shared Risk Detection
    try:
        risk_findings: List[Dict[str, Any]] = detect_shared_risks(graph, only_shared=False)
        shared_only_findings = [r for r in risk_findings if r.get("is_shared")]
    except Exception as exc:
        risk_findings = []
        shared_only_findings = []
        st.sidebar.warning(f"Risk Detection Notice: {exc}")

    # 2. Member 3: Disruption Simulation
    try:
        sim_result = simulate_disruption(
            graph,
            failed_node=selected_failed_node,
            disruption_type=passed_disruption_type,
        )
    except Exception as exc:
        st.error(f"Simulation failed for entity '{selected_failed_node}': {exc}")
        return

    # 3. Member 4: Financial Optimization
    try:
        financial_scenario = build_financial_scenario_from_simulation(
            simulation_result=sim_result,
            loss_per_unit=unit_loss_input,
            available_budget=budget_input,
            max_lead_time_days=max_lead_time_input,
        )
        candidate_options = generate_candidate_recovery_options(
            scenario=financial_scenario,
            graph=graph,
        )
        financial_result = optimize_recovery_plan(
            scenario=financial_scenario,
            recovery_options=candidate_options,
            allow_multi_option=allow_multi_option,
        )
    except Exception as exc:
        st.error(f"Financial Optimization engine failed: {exc}")
        return

    # 4. Member 5: Decision Explanation
    explanation_agent = DecisionExplanationAgent()
    decision_exp = explanation_agent.explain(
        graph=graph,
        disruption_result=sim_result,
        financial_result=financial_result,
        risk_findings=risk_findings,
    )

    # =========================================================================
    # Header & Overview
    # =========================================================================
    st.markdown(
        '<div class="main-subtitle">Supply Chain Resilience Control Tower</div>'
        '<div class="main-header">Nexus-AI</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sub-header">Active Dataset: <b>{data_source_name}</b> &nbsp;|&nbsp; '
        f'Disruption Target: <span class="badge-critical">{selected_failed_node}</span></div>',
        unsafe_allow_html=True,
    )

    # =========================================================================
    # SECTION A: EXECUTIVE DASHBOARD
    # =========================================================================
    st.markdown("### 📊 Executive Overview")
    st.caption("Live metrics derived from the active dataset and all agent pipeline outputs.")

    total_entities = graph.number_of_nodes()
    total_suppliers = len([n for n, d in graph.nodes(data=True) if "supp" in str(d.get("node_type", "")).lower() or d.get("is_supplier")])
    total_factories = len([n for n, d in graph.nodes(data=True) if "fact" in str(d.get("node_type", "")).lower() or "plant" in str(d.get("node_type", "")).lower()])
    shared_deps_count = len(shared_only_findings)
    affected_count = len(sim_result.get("affected_entities", []))
    impact_level = sim_result.get("impact_level", "Unknown")
    gross_loss = financial_result.get("baseline_disruption_loss", 0.0)
    rec_cost = financial_result.get("recovery_cost", 0.0)
    est_savings = financial_result.get("estimated_savings", 0.0)
    residual_loss = financial_result.get("residual_disruption_loss", 0.0)

    level_class = {
        "Critical": "badge-critical",
        "High": "badge-high",
        "Medium": "badge-medium",
        "Low": "badge-low",
    }.get(impact_level, "badge-medium")
    approval_req = decision_exp["requires_human_approval"]

    # ── Row 1: Network & Disruption Metrics ──────────────────────────────────
    # Use 4 st.metric columns + 1 custom badge card in a 5-column row.
    # The badge card uses inline HTML to match st.metric height visually.
    kpi_row1 = st.columns([1, 1, 1, 1, 1])
    with kpi_row1[0]:
        st.metric("Total Entities", f"{total_entities}")
    with kpi_row1[1]:
        st.metric("Suppliers", f"{total_suppliers}")
    with kpi_row1[2]:
        st.metric("Plants / Factories", f"{total_factories}")
    with kpi_row1[3]:
        st.metric("Shared Risk Hubs", f"{shared_deps_count}")
    with kpi_row1[4]:
        st.metric("Disrupted Entities", f"{affected_count}")

    # ── Row 2: Financial & Governance Metrics ────────────────────────────────
    kpi_row2 = st.columns([1, 1, 1, 1, 1])
    with kpi_row2[0]:
        st.metric("Baseline Gross Loss", f"${gross_loss:,.0f}")
    with kpi_row2[1]:
        st.metric("Plan Expenditure", f"${rec_cost:,.0f}")
    with kpi_row2[2]:
        st.metric("Net Savings", f"${est_savings:,.0f}")
    with kpi_row2[3]:
        st.metric("Residual Risk", f"${residual_loss:,.0f}")
    with kpi_row2[4]:
        # Severity + Governance in a single styled card (matches metric card height)
        sev_label = impact_level.upper()
        gov_label = "ACTION REQUIRED" if approval_req else "AUTO-APPROVED"
        gov_class = "badge-critical" if approval_req else "badge-low"
        st.markdown(
            f"""
            <div style="background:#1E293B;border:1px solid #334155;border-radius:10px;
                        padding:14px 18px;height:100%;box-sizing:border-box;">
              <div style="color:#94A3B8;font-size:0.76rem;font-weight:600;
                          text-transform:uppercase;letter-spacing:0.05em;
                          margin-bottom:6px;">Severity / Governance</div>
              <span class="{level_class}">{sev_label}</span>
              &nbsp;
              <span class="{gov_class}">{gov_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # =========================================================================
    # SECTION B: SUPPLY CHAIN NETWORK
    # =========================================================================
    st.markdown("### 🗺️ Section B: Supply Chain Network Topology")
    st.caption("Interactive directed network graph: upstream sources feed downstream processors and suppliers.")

    col_map, col_inspect = st.columns([3, 1])

    with col_inspect:
        st.markdown("#### 🔍 Inspect Entity")
        inspect_node = st.selectbox(
            "Select entity to inspect:",
            options=available_nodes,
            index=default_node_idx,
            key="inspect_select",
        )
        if inspect_node in graph:
            node_data = graph.nodes[inspect_node]
            st.info(f"**Role/Tier:** {node_data.get('node_type', 'Not specified')}")
            in_deg = graph.in_degree(inspect_node)
            out_deg = graph.out_degree(inspect_node)
            st.write(f"- **Inbound Connections:** {in_deg}")
            st.write(f"- **Outbound Feeds:** {out_deg}")
            preds = list(graph.predecessors(inspect_node))
            if preds:
                st.write(f"- **Upstream Providers:** {', '.join(preds)}")
            succs = list(graph.successors(inspect_node))
            if succs:
                st.write(f"- **Downstream Customers:** {', '.join(succs)}")
            alts = node_data.get("alternatives", [])
            if alts:
                st.write(f"- **Backup Alternatives:** {', '.join(alts)}")
            else:
                st.caption("No registered backup alternatives")

    with col_map:
        fig = create_resilient_network_figure(
            graph=graph,
            simulation_result=sim_result,
            selected_entity=inspect_node,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # =========================================================================
    # SECTION C: SHARED RISK ANALYSIS
    # =========================================================================
    st.markdown("### 🛡️ Section C: Shared Risk Detection ")
    st.caption("Audits upstream dependencies to identify multi-tier single points of failure feeding multiple suppliers.")

    tab_shared, tab_all_risks = st.tabs(["Shared Dependencies Only (SPOF)", "All Dependency Findings"])

    with tab_shared:
        if shared_only_findings:
            shared_df_rows = []
            for r in shared_only_findings:
                shared_df_rows.append({
                    "Dependency Node": r.get("dependency_node"),
                    "Affected Suppliers": ", ".join(r.get("affected_suppliers", [])),
                    "Supplier Count": r.get("affected_supplier_count"),
                    "Dependency Tier": r.get("dependency_tier"),
                    "Risk Score": f"{r.get('risk_score', 0):.1f}/100",
                    "Risk Level": r.get("risk_level"),
                    "Explanation": r.get("explanation"),
                    "Recommended Investigation": r.get("recommended_investigation"),
                })
            st.dataframe(pd.DataFrame(shared_df_rows), use_container_width=True)
        else:
            st.info("No shared dependencies affecting ≥2 suppliers found in the active dataset.")

    with tab_all_risks:
        if risk_findings:
            all_df_rows = []
            for r in risk_findings:
                all_df_rows.append({
                    "Dependency Node": r.get("dependency_node"),
                    "Is Shared?": "Yes" if r.get("is_shared") else "No",
                    "Suppliers": ", ".join(r.get("affected_suppliers", [])),
                    "Risk Score": f"{r.get('risk_score', 0):.1f}/100",
                    "Risk Level": r.get("risk_level"),
                    "Missing Info": ", ".join(r.get("missing_information", [])) or "None",
                    "Recommended Action": r.get("recommended_investigation"),
                })
            st.dataframe(pd.DataFrame(all_df_rows), use_container_width=True)
        else:
            st.info("No risk records generated.")

    st.markdown("---")

    # =========================================================================
    # SECTION D: DISRUPTION SIMULATOR
    # =========================================================================
    st.markdown("### ⚡ Section D: Disruption Simulator ")
    st.caption("Simulates outage ripple effects, tracing broken paths, affected materials, and capacity shortfalls.")

    d_col1, d_col2 = st.columns([1, 1])

    with d_col1:
        st.markdown("#### Scenario Impact Summary")
        cap_impact = sim_result.get("estimated_capacity_impact", {})
        st.write(f"- **Failed Entity:** `{sim_result.get('failed_entity')}`")
        st.write(f"- **Disruption Type:** {sim_result.get('disruption_type')}")
        st.write(f"- **Impact Classification:** **{sim_result.get('impact_level')}**")
        st.write(f"- **Direct Capacity Offline:** **{cap_impact.get('estimated_direct_capacity_loss', 0):,.1f} units**")
        st.caption("Note: Anti-double-counting policy strictly prevents adding multi-tier intermediate throughput.")

        materials = sim_result.get("affected_materials", [])
        if materials:
            st.write(f"- **Jeopardized Commodities:** {', '.join(materials)}")

    with d_col2:
        st.markdown("#### Broken Supply Corridors & Actions")
        paths = sim_result.get("disrupted_paths", [])
        if paths:
            path_strings = [" &rarr; ".join(p) for p in paths[:5]]
            st.write("**Disrupted Paths (Top 5):**")
            for ps in path_strings:
                st.markdown(f"- {ps}", unsafe_allow_html=True)
            if len(paths) > 5:
                st.caption(f"... plus {len(paths) - 5} additional severed sub-paths.")

        actions = sim_result.get("recommended_actions", [])
        if actions:
            st.write("**Recommended Next Actions:**")
            for act in actions:
                st.markdown(f"- {act}")

    # Entity Breakdown Table (Confirmed Shortages vs Potential)
    st.markdown("#### Entity Shortage Severity Breakdown")
    entity_agent = DisruptionSimulationAgent(graph)
    try:
        detail_df = entity_agent.get_entity_impact_dataframe(sim_result)
        st.dataframe(detail_df, use_container_width=True)
    except Exception:
        # Fallback manual table
        entity_details = sim_result.get("entity_impact_details", [])
        if entity_details:
            st.dataframe(pd.DataFrame(entity_details), use_container_width=True)
        else:
            st.info("No downstream entities were impacted by this scenario.")

    st.markdown("---")

    # =========================================================================
    # SECTION E: FINANCIAL IMPACT AND RECOVERY
    # =========================================================================
    st.markdown("### 💵 Section E: Financial Impact & Recovery Optimization ")
    st.caption("Mathematically selects the lowest-cost feasible recovery plan subject to budget, capacity, and lead-time constraints.")

    # Recommendation Banner — use plain st.markdown inside a styled div so
    # Markdown bold/backtick syntax renders correctly (st.success renders
    # plain text only and shows literal ** characters).
    rec_ids = financial_result.get("recommended_option_ids", [])
    if rec_ids:
        total_cost = financial_result.get("estimated_total_cost", 0.0)
        net_savings = financial_result.get("estimated_savings", 0.0)
        plan_ids = ", ".join(f"`{rid}`" for rid in rec_ids)
        st.markdown(
            f"""
            <div style="background:#064E3B;border:1px solid #059669;border-radius:8px;
                        padding:14px 18px;margin-bottom:12px;">
              <span style="font-weight:700;color:#6EE7B7;">OPTIMAL RECOVERY PLAN: </span>
              <span style="color:#F0FDF4;">{', '.join(rec_ids)}</span>
              &nbsp;&mdash;&nbsp;
              <span style="color:#A7F3D0;">Total Plan Cost:</span>
              <span style="color:#F0FDF4;font-weight:700;"> ${total_cost:,.2f}</span>
              &nbsp;&nbsp;
              <span style="color:#A7F3D0;">Net Capital Preserved:</span>
              <span style="color:#6EE7B7;font-weight:700;"> ${net_savings:,.2f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            "NO-ACTION BASELINE INVOKED: No evaluated recovery option passed all budget, capacity, and lead-time constraints."
        )

    if financial_result.get("requires_human_approval"):
        st.error(
            "🚨 **EXECUTIVE APPROVAL REQUIRED:** Proposed expenditures exceed safety governance thresholds "
            "(e.g., negative net savings, high budget utilization, or baseline fallback)."
        )

    # Feasibility Comparison Table
    feasible_opts = financial_result.get("feasible_options", [])
    rejected_opts = financial_result.get("rejected_options", [])

    all_opts_rows = []
    for f in feasible_opts:
        is_rec = f.get("option_id") in rec_ids
        all_opts_rows.append({
            "Option ID": f.get("option_id") + (" ⭐ [SELECTED]" if is_rec else ""),
            "Status": "FEASIBLE",
            "Recovery Cost": f"${f.get('recovery_cost', 0):,.2f}",
            "Available Capacity": f"{f.get('available_capacity', 0):,.1f}",
            "Lead Time": f"{f.get('lead_time_days', 0):.0f} days",
            "Total Plan Cost": f"${f.get('estimated_total_cost', 0):,.2f}",
            "Net Savings": f"${f.get('estimated_savings', 0):,.2f}",
            "Constraint Audit": "Satisfies all constraints",
        })

    for r in rejected_opts:
        all_opts_rows.append({
            "Option ID": r.get("option_id", ""),
            "Status": "REJECTED",
            "Recovery Cost": f"${r.get('recovery_cost', 0):,.2f}",
            "Available Capacity": f"{r.get('available_capacity', 0):,.1f}",
            "Lead Time": f"{r.get('lead_time_days', 0):.0f} days",
            "Total Plan Cost": "N/A (Infeasible)",
            "Net Savings": "N/A (Infeasible)",
            "Constraint Audit": "; ".join(r.get("rejection_reasons", [])),
        })

    st.markdown("#### Candidate Recovery Options — Feasibility Matrix")
    if all_opts_rows:
        opts_df = pd.DataFrame(all_opts_rows)
        st.dataframe(
            opts_df,
            use_container_width=True,
            column_config={
                "Option ID":          st.column_config.TextColumn("Option ID",        width="medium"),
                "Status":             st.column_config.TextColumn("Status",           width="small"),
                "Recovery Cost":      st.column_config.TextColumn("Recovery Cost",    width="small"),
                "Available Capacity": st.column_config.TextColumn("Capacity (units)", width="small"),
                "Lead Time":          st.column_config.TextColumn("Lead Time",        width="small"),
                "Total Plan Cost":    st.column_config.TextColumn("Total Cost",       width="small"),
                "Net Savings":        st.column_config.TextColumn("Net Savings",      width="small"),
                "Constraint Audit":   st.column_config.TextColumn("Constraint Audit", width="large"),
            },
            hide_index=True,
        )
    else:
        st.info("No recovery options were evaluated.")

    st.markdown("---")

    # =========================================================================
    # SECTION F: DECISION EXPLANATION
    # =========================================================================
    st.markdown("### 📝 Section F: Auditable 9-Point Decision Explanation ")
    st.caption("Synthesizes multi-agent intelligence into a fully deterministic, explainable report for leadership.")

    tabs = st.tabs([
        "1. Root Failure",
        "2. Impacted Entities",
        "3. Criticality Context",
        "4. Financial Formula",
        "5. Chosen Plan",
        "6. Alternatives Evaluated",
        "7. Governance & Gaps",
        "8. Residual Risks",
        "9. Approval Status",
    ])

    with tabs[0]:
        st.markdown(f"**What Failed:**\n\n{decision_exp['what_failed']['narrative']}")
    with tabs[1]:
        st.markdown(f"**Which Entities Were Affected:**\n\n{decision_exp['affected_entities']['narrative']}")
    with tabs[2]:
        st.markdown(f"**Why the Disruption Matters:**\n\n{decision_exp['why_it_matters']['narrative']}")
    with tabs[3]:
        st.markdown(f"**Financial Computation:**\n\n{decision_exp['financial_calculation']['narrative']}")
        st.code(decision_exp['financial_calculation']['loss_formula'], language="text")
        st.code(decision_exp['financial_calculation']['cost_formula'], language="text")
    with tabs[4]:
        st.markdown(f"**Selected Recovery Option:**\n\n{decision_exp['selected_recovery_option']['narrative']}")
    with tabs[5]:
        st.markdown(f"**Alternatives Considered:**\n\n{decision_exp['alternatives_considered']['narrative']}")
    with tabs[6]:
        st.markdown(f"**Assumptions and Missing Data Disclosures:**\n\n{decision_exp['assumptions_and_missing_data']['narrative']}")
    with tabs[7]:
        st.markdown(f"**Residual Risks Remaining:**\n\n{decision_exp['residual_risks']['narrative']}")
    with tabs[8]:
        st.markdown(f"**Human Approval Protocol:**\n\n{decision_exp['human_approval']['narrative']}")

    # Download Button for Full Markdown Report
    st.download_button(
        label="📥 Download Full Decision Explanation Report (.md)",
        data=decision_exp["markdown_report"],
        file_name=f"nexus_ai_decision_report_{selected_failed_node.replace(' ', '_')}.md",
        mime="text/markdown",
    )


def run_cli_fallback(args: argparse.Namespace) -> None:
    """CLI fallback runner when app.py is invoked directly as a Python script."""
    print("=" * 70)
    print("        NEXUS-AI: MULTI-TIER SUPPLY CHAIN RESILIENCE PLATFORM")
    print("               Member 5 Integration & Decision Report")
    print("=" * 70)

    # 1. Load Data
    csv_path = args.csv if args.csv else DEFAULT_DEPENDENCIES_CSV
    print(f"\n[Step 1/5] Ingesting & validating data from: {csv_path}")
    try:
        df = load_and_validate_dependencies(csv_path)
        graph = build_supply_chain_graph(df)
        print(f"           SUCCESS: Loaded {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
    except Exception as exc:
        print(f"[ERROR] Failed to load dataset: {exc}")
        sys.exit(1)

    # 2. Member 2: Shared Risk
    print("\n[Step 2/5] Running Member 2 Shared Risk Detection...")
    shared_risks = detect_shared_risks(graph, only_shared=True)
    print(f"           Identified {len(shared_risks)} shared single points of failure.")

    # 3. Member 3: Disruption Simulation
    failed_target = args.fail_node or ("Processor X" if "Processor X" in graph else list(graph.nodes())[0])
    print(f"\n[Step 3/5] Simulating disruption at target: '{failed_target}'...")
    sim_res = simulate_disruption(graph, failed_node=failed_target)
    print(f"           Impact Level: {sim_res.get('impact_level')}")
    print(f"           Affected Entities ({len(sim_res.get('affected_entities', []))}): {sim_res.get('affected_entities', [])}")

    # 4. Member 4: Financial Optimization
    print("\n[Step 4/5] Executing Member 4 Financial Impact & Recovery Optimization...")
    scenario = build_financial_scenario_from_simulation(
        simulation_result=sim_res,
        loss_per_unit=args.loss_per_unit,
        available_budget=args.budget,
        max_lead_time_days=args.max_lead_time,
    )
    candidate_opts = generate_candidate_recovery_options(scenario, graph=graph)
    fin_res = optimize_recovery_plan(scenario, candidate_opts, allow_multi_option=True)
    print(f"           Baseline Disruption Loss: ${fin_res.get('baseline_disruption_loss', 0):,.2f}")
    print(f"           Selected Recovery Plan:   {fin_res.get('recommended_option_ids')}")
    print(f"           Total Recovery Cost:      ${fin_res.get('estimated_total_cost', 0):,.2f}")
    print(f"           Net Estimated Savings:    ${fin_res.get('estimated_savings', 0):,.2f}")

    # 5. Member 5: Decision Explanation
    print("\n[Step 5/5] Synthesizing Member 5 Deterministic Decision Explanation...")
    exp = generate_decision_explanation(
        graph=graph,
        disruption_result=sim_res,
        financial_result=fin_res,
        risk_findings=shared_risks,
    )
    print("\n" + "=" * 70)
    print(exp["markdown_report"])
    print("=" * 70)
    print("\n[Tip] To launch the full interactive web dashboard, run:")
    print("      streamlit run app.py\n")


def parse_cli_args() -> argparse.Namespace:
    """Parses optional command-line arguments."""
    parser = argparse.ArgumentParser(description="Nexus-AI Supply Chain Resilience Application")
    parser.add_argument("--csv", type=Path, default=None, help="Path to dependencies CSV file")
    parser.add_argument("--fail-node", type=str, default=None, help="Target entity to disrupt")
    parser.add_argument("--loss-per-unit", type=float, default=150.0, help="Loss per unit ($)")
    parser.add_argument("--budget", type=float, default=50000.0, help="Available recovery budget ($)")
    parser.add_argument("--max-lead-time", type=float, default=14.0, help="Max acceptable lead time (days)")
    parser.add_argument("--cli", action="store_true", help="Force command-line interface execution")
    return parser.parse_args()


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    # Check if executed inside Streamlit runtime
    # When streamlit runs the script, 'streamlit.runtime' or streamlit context is active
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        is_streamlit = get_script_run_ctx() is not None
    except Exception:
        is_streamlit = False

    if is_streamlit:
        run_streamlit_app()
    else:
        # Check CLI arguments
        args = parse_cli_args()
        if args.cli or args.fail_node or args.csv:
            run_cli_fallback(args)
        else:
            # If run directly as 'python app.py' with no args, prompt user or run CLI demo
            print("Launching Nexus-AI in CLI demonstration mode.")
            print("(To open the interactive web UI, run: streamlit run app.py)\n")
            run_cli_fallback(args)
