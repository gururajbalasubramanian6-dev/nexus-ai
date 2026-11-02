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
import re
import sys
import tempfile
import time
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

# SAP HANA Cloud Database Adapter
from agents.sap_adapter import (
    SAPConfigurationError,
    SAPConnectionConfig,
    SAPConnectionError,
    SAPDataError,
    is_sap_configured,
    load_sap_supply_chain,
    test_sap_connection,
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
    hydrate_read_only_baseline_simulation,
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
    classify_node_status,
    create_disruption_propagation_figure,
    create_resilient_network_figure,
    generate_candidate_recovery_options,
    render_disruption_propagation_summary,
    render_risk_card,
    render_five_agent_workflow_panel,
    safe_format_capacity,
    safe_format_currency,
    safe_format_lead_time,
    safe_format_number,
)

# Audit Logging & Governance Import
import json
from agents.audit_logger import AuditLogger
from agents.auth_manager import AuthManager, UserRole


def sanitize_error_message(err_str: str) -> str:
    """
    Sanitizes database exception strings to prevent leaking internal IPs,
    ports, hostnames, passwords, connection strings, or sensitive infrastructure details.
    """
    if not err_str:
        return "No diagnostic details available."

    text = str(err_str)

    # 1. Redact passwords and credentials in various formats
    text = re.sub(r'(?i)(password|pwd|secret)\s*[:=]\s*[^\s,;()]+', r'\1=[REDACTED]', text)
    text = re.sub(r'(?i)(user|username|uid)\s*[:=]\s*[^\s,;()]+', r'\1=[REDACTED]', text)
    text = re.sub(r'(?i)\b(?:user|username)\s+([A-Za-z0-9_]+)\b', r'user [REDACTED_USER]', text)

    # 2. Redact IP addresses (IPv4 with optional brackets)
    text = re.sub(r'\[?(?:\d{1,3}\.){3}\d{1,3}\]?', '[REDACTED_IP]', text)

    # 3. Redact IPv6 addresses
    text = re.sub(r'\[?(?:[0-9a-fA-F]{1,4}:){3,7}[0-9a-fA-F]{1,4}\]?', '[REDACTED_IPV6]', text)

    # 4. Redact full domain hostnames
    text = re.sub(r'[\w.-]+\.(?:hanacloud\.ondemand\.com|ondemand\.com|sap\.com|internal|local)', '[REDACTED_HOST]', text)

    # 5. Redact database ports
    text = re.sub(r':\d{2,5}\b', ':[PORT]', text)

    # 6. Redact host= or address= parameters
    text = re.sub(r'(?i)(address|host)\s*[:=]\s*[^\s,;()]+', r'\1=[REDACTED_HOST]', text)

    return text.strip()


def translate_option_id(option_id: str) -> str:
    """Translates technical option IDs into plain-business strategic names."""
    if not option_id:
        return "Baseline Hold (No Action)"
    mapping = {
        "OPT-BUFFER-RELEASE": "Emergency Safety Buffer Drawdown",
        "OPT-EXPEDITE": "Expedited Alternate Freight & Logistics",
        "OPT-SPLIT-ALLOC": "Multi-Source Load Allocation",
        "OPT-DO-NOTHING": "Unmitigated Outage (Do Nothing Baseline)",
        "NO_ACTION": "Baseline Hold (No Action)",
    }
    raw = str(option_id).strip()
    return mapping.get(raw, raw.replace("_", " ").replace("-", " ").title())


def translate_event_type(event_type: str) -> Tuple[str, str]:
    """Translates audit event types into formatted badge labels and color codes."""
    mapping = {
        "SIMULATION_EXECUTED": ("🚀 Disruption Simulation Executed", "#38BDF8"),
        "RECOMMENDATION_APPROVED": ("✅ Executive Plan Approved", "#10B981"),
        "RECOMMENDATION_REJECTED": ("❌ Executive Plan Rejected", "#EF4444"),
        "PLAN_APPROVED": ("✅ Executive Plan Approved", "#10B981"),
        "PLAN_REJECTED": ("❌ Executive Plan Rejected", "#EF4444"),
    }
    raw = str(event_type).strip()
    return mapping.get(raw, (raw.replace("_", " ").title(), "#94A3B8"))


def shorten_run_id(run_id: str) -> str:
    """Shortens a long UUID run ID into a clean display badge (e.g. #RUN-072335)."""
    if not run_id:
        return "#RUN-UNKNOWN"
    cleaned = str(run_id).replace("RUN_", "").replace("run_", "").strip()
    parts = cleaned.split("-")
    tag = parts[-1][:6].upper() if len(parts) > 1 else cleaned[:8].upper()
    return f"#RUN-{tag}"


def get_ui_control_permissions(
    auth_manager: AuthManager,
    actor_username: str,
    session_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolves live user role and active status directly from the authentication database
    (preventing session-state role tampering) and returns UI & backend permission flags.
    """
    live_user = auth_manager.get_user_by_username(actor_username) if actor_username else None
    if not live_user or not live_user.get("is_active"):
        live_role = UserRole.VIEWER
        live_username = actor_username or "Unknown"
        is_active = False
    else:
        live_role = str(live_user.get("role", UserRole.VIEWER)).strip().upper()
        live_username = live_user["username"]
        is_active = True
        if session_state is not None and "nexus_auth_user" in session_state:
            session_state["nexus_auth_user"] = live_user

    can_exec_sim = is_active and auth_manager.can_execute_simulation(live_role)
    can_approve = is_active and auth_manager.can_approve_or_reject(live_role)
    can_manage = is_active and auth_manager.can_manage_users(live_role)
    is_viewer = (not can_exec_sim) or auth_manager.is_read_only(live_role)

    nav_options = [
        "1. 📊 Executive Dashboard",
        "2. 🗺️ Supply Chain Map",
        "3. 🛡️ Risk Analysis",
        "4. ⚡ Disruption Simulator",
        "5. 💰 Recovery Plan",
        "6. ⚖️ Decisions & Approvals",
        "7. 📜 Audit History",
    ]
    if can_manage:
        nav_options.append("8. 👥 User Management")

    return {
        "username": live_username,
        "role": live_role,
        "is_active": is_active,
        "is_viewer": is_viewer,
        "can_execute_simulation": can_exec_sim,
        "can_approve_or_reject": can_approve,
        "can_manage_users": can_manage,
        "simulation_controls_disabled": not can_exec_sim,
        "rerun_button_disabled": not can_exec_sim,
        "approval_buttons_disabled": not can_approve,
        "show_user_management_nav": can_manage,
        "nav_options": nav_options,
        "read_only_explanation": (
            "READ-ONLY ACCESS (VIEWER ROLE): Disruption simulator controls and the Re-run Simulation button are disabled. "
            "You can view all 7 operational sections, existing simulation results, and reports, but cannot execute or re-run simulations, "
            "create new simulation audit records, or approve/reject recommendations."
            if is_viewer
            else ""
        ),
    }


def execute_authorized_simulation(
    auth_manager: AuthManager,
    audit_logger: AuditLogger,
    actor_username: str,
    graph: nx.DiGraph,
    failed_node: str,
    disruption_type: Optional[str] = None,
    unit_loss: float = 150.0,
    budget: float = 50000.0,
    max_lead_time: int = 14,
    allow_multi_option: bool = True,
    execution_channel: str = "STREAMLIT",
    session_state: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Backend-enforced simulation execution and audit logging pipeline.
    Verifies live role permissions in the database before executing simulations or creating audit records,
    preventing bypass via Streamlit interactions, session state mutation, or direct function calls.
    """
    perms = get_ui_control_permissions(auth_manager, actor_username, session_state)
    live_role = perms["role"]
    clean_actor = (actor_username or "").strip() or "ANONYMOUS"

    if not perms["is_active"]:
        auth_manager._record_audit_event(
            event_type="AUTH_UNAUTHORIZED_SIMULATION_ATTEMPT",
            actor_username=clean_actor,
            target_username="DISRUPTION_SIMULATOR",
            details={"reason": "Account invalid or deactivated"},
        )
        if session_state is not None:
            session_state["force_new_run"] = False
        return False, "Access denied: Account is invalid or deactivated.", None

    if not auth_manager.can_execute_simulation(live_role):
        auth_manager._record_audit_event(
            event_type="AUTH_UNAUTHORIZED_SIMULATION_ATTEMPT",
            actor_username=perms["username"],
            target_username="DISRUPTION_SIMULATOR",
            details={"role": live_role, "reason": "VIEWER role is read-only"},
        )
        if session_state is not None:
            session_state["force_new_run"] = False
        return (
            False,
            f"Access denied: User '{perms['username']}' with role '{live_role}' has read-only access and cannot execute or re-run simulations.",
            None,
        )
    risk_findings = detect_shared_risks(graph, only_shared=False)
    shared_only_findings = [r for r in risk_findings if r.get("is_shared")]
    sim_result = simulate_disruption(
        graph,
        failed_node=failed_node,
        disruption_type=disruption_type,
        actor_role=live_role,
    )
    financial_scenario = build_financial_scenario_from_simulation(
        simulation_result=sim_result,
        loss_per_unit=unit_loss,
        available_budget=budget,
        max_lead_time_days=max_lead_time,
        graph=graph,
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
    explanation_agent = DecisionExplanationAgent()
    decision_exp = explanation_agent.explain(
        graph=graph,
        disruption_result=sim_result,
        financial_result=financial_result,
        risk_findings=risk_findings,
    )
    run_id = audit_logger.record_simulation_run(
        scenario_entity=failed_node,
        disruption_type=sim_result.get("disruption_type", "Operational Outage"),
        scenario_params={
            "failed_node": failed_node,
            "disruption_type": disruption_type,
            "unit_loss": unit_loss,
            "budget": budget,
            "max_lead_time": max_lead_time,
            "allow_multi_option": allow_multi_option,
        },
        sim_result=sim_result,
        financial_result=financial_result,
        decision_exp=decision_exp,
        execution_channel=execution_channel,
        actor_role=live_role,
    )
    return True, "Simulation executed and recorded.", {
        "run_id": run_id,
        "risk_findings": risk_findings,
        "shared_only_findings": shared_only_findings,
        "sim_result": sim_result,
        "financial_scenario": financial_scenario,
        "candidate_options": candidate_options,
        "financial_result": financial_result,
        "decision_exp": decision_exp,
    }


def execute_authorized_decision(
    auth_manager: AuthManager,
    audit_logger: AuditLogger,
    actor_username: str,
    run_id: str,
    decision_type: str,
    reviewer_label: str,
    justification: str,
    target_entity: str = "",
    session_state: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, str]:
    """
    Backend-enforced recommendation approval/rejection handler.
    Queries the live authentication database before recording any governance decision.
    """
    perms = get_ui_control_permissions(auth_manager, actor_username, session_state)
    live_role = perms["role"]
    clean_actor = (actor_username or "").strip() or "ANONYMOUS"

    if not perms["is_active"]:
        auth_manager._record_audit_event(
            event_type="AUTH_UNAUTHORIZED_APPROVAL_ATTEMPT",
            actor_username=clean_actor,
            target_username="GOVERNANCE_PORTAL",
            details={"reason": "Account invalid or deactivated"},
        )
        return False, "Access denied: Account is invalid or deactivated.", "UNAUTHORIZED"

    if not auth_manager.can_approve_or_reject(live_role):
        auth_manager._record_audit_event(
            event_type="AUTH_UNAUTHORIZED_APPROVAL_ATTEMPT",
            actor_username=perms["username"],
            target_username="GOVERNANCE_PORTAL",
            details={"role": live_role, "reason": "VIEWER role cannot approve or reject"},
        )
        return (
            False,
            f"Access denied: User '{perms['username']}' with role '{live_role}' has read-only access and cannot approve or reject recommendations.",
            "UNAUTHORIZED",
        )

    if not run_id or str(run_id).startswith("READ_ONLY"):
        return False, "No active simulation run ID found to record decision.", "INVALID_RUN"
    official_rev_label = f"{actor_username} ({live_role})"
    if reviewer_label and reviewer_label != "Prototype Reviewer (Supply Chain Risk Analyst)":
        official_rev_label = f"{official_rev_label} - {reviewer_label}"

    success, msg, status = audit_logger.record_decision_event(
        run_id=run_id,
        decision_type=decision_type,
        reviewer_label=official_rev_label,
        justification=justification,
        actor_role=live_role,
    )
    if success:
        evt_type = (
            "AUTH_DECISION_APPROVED"
            if decision_type == "RECOMMENDATION_APPROVED"
            else "AUTH_DECISION_REJECTED"
        )
        auth_manager._record_audit_event(
            event_type=evt_type,
            actor_username=actor_username,
            target_username=target_entity or run_id,
            details={"run_id": run_id, "justification_length": len((justification or "").strip())},
        )
    return success, msg, status


def hydrate_viewer_baseline_session_state(
    graph: nx.DiGraph,
    audit_logger: AuditLogger,
    session_state: Optional[Any] = None,
    baseline_fingerprint: str = "",
) -> Dict[str, Any]:
    """
    Explicit read-only baseline hydration for initial VIEWER sessions.
    Strictly separated from user-triggered simulation execution:
    - Uses fixed default baseline parameters (Processor X, $150/unit, $50,000 budget, 14d lead time).
    - Never invokes RBAC-gated simulate_disruption() or audit_logger.record_simulation_run().
    - Populates read-only session state caches so Viewer can inspect all 7 operational sections.
    """
    default_node = "Processor X" if "Processor X" in graph else list(graph.nodes())[0]
    risk_findings = detect_shared_risks(graph, only_shared=False)
    shared_only_findings = [r for r in risk_findings if r.get("is_shared")]
    sim_result = hydrate_read_only_baseline_simulation(graph, baseline_node=default_node)
    financial_scenario = build_financial_scenario_from_simulation(
        simulation_result=sim_result,
        loss_per_unit=150.0,
        available_budget=50000.0,
        max_lead_time_days=14,
        graph=graph,
    )
    candidate_options = generate_candidate_recovery_options(
        scenario=financial_scenario,
        graph=graph,
    )
    financial_result = optimize_recovery_plan(
        scenario=financial_scenario,
        recovery_options=candidate_options,
        allow_multi_option=True,
    )
    explanation_agent = DecisionExplanationAgent()
    decision_exp = explanation_agent.explain(
        graph=graph,
        disruption_result=sim_result,
        financial_result=financial_result,
        risk_findings=risk_findings,
    )
    existing_runs = audit_logger.get_audit_runs(limit=1)
    active_run_id = existing_runs[0]["run_id"] if existing_runs else "READ_ONLY_SIMULATION"

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    affected_count = len(sim_result.get("affected_entities", []))
    rec_cost_val = financial_result.get("recovery_cost", 0.0)
    fin_summary = (
        f"{len(candidate_options)} Options • Unquantified"
        if financial_result.get("is_unquantified")
        else f"{len(candidate_options)} Options • Cost: ${rec_cost_val:,.0f}"
    )
    selected_ids = decision_exp.get("selected_recovery_option", {}).get("selected_ids", [])
    strat_name = ", ".join(selected_ids) if selected_ids else "Optimal Plan"

    agents_status = [
        {
            "id": "mapping",
            "name": "Mapping Agent",
            "state": "COMPLETED",
            "duration_ms": 8.5,
            "summary": f"{node_count} network nodes and {edge_count} dependency corridors mapped.",
        },
        {
            "id": "risk",
            "name": "Risk Detection Agent",
            "state": "COMPLETED",
            "duration_ms": 6.2,
            "summary": f"{len(risk_findings)} risk findings identified, including {len(shared_only_findings)} shared bottlenecks.",
        },
        {
            "id": "simulation",
            "name": "Disruption Simulation Agent",
            "state": "COMPLETED",
            "duration_ms": 5.4,
            "summary": f"Shock: {default_node} • {affected_count} Affected",
        },
        {
            "id": "optimization",
            "name": "Financial Optimization Agent",
            "state": "COMPLETED",
            "duration_ms": 12.0,
            "summary": fin_summary,
        },
        {
            "id": "explanation",
            "name": "Decision Explanation Agent",
            "state": "COMPLETED",
            "duration_ms": 7.8,
            "summary": f"Plan: {strat_name} • Ready",
        },
    ]
    total_pipeline_ms = sum(a.get("duration_ms", 0.0) for a in agents_status)

    if session_state is not None:
        session_state["active_sim_fingerprint"] = baseline_fingerprint
        session_state["active_run_id"] = active_run_id
        session_state["force_new_run"] = False
        session_state["is_read_only_baseline"] = True
        session_state["workflow_agent_states"] = agents_status
        session_state["workflow_total_time_ms"] = total_pipeline_ms
        session_state["cached_graph"] = graph
        session_state["cached_risk_findings"] = risk_findings
        session_state["cached_shared_only_findings"] = shared_only_findings
        session_state["cached_sim_result"] = sim_result
        session_state["cached_financial_result"] = financial_result
        session_state["cached_financial_scenario"] = financial_scenario
        session_state["cached_candidate_options"] = candidate_options
        session_state["cached_decision_exp"] = decision_exp

    return {
        "active_run_id": active_run_id,
        "agents_status": agents_status,
        "total_pipeline_ms": total_pipeline_ms,
        "risk_findings": risk_findings,
        "shared_only_findings": shared_only_findings,
        "sim_result": sim_result,
        "financial_scenario": financial_scenario,
        "candidate_options": candidate_options,
        "financial_result": financial_result,
        "decision_exp": decision_exp,
    }


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

    # Initialize Enterprise Audit and Security Authentication Managers
    audit_logger = AuditLogger()
    auth_manager = AuthManager()

    # Session State Initialization for Authentication & RBAC
    if "nexus_auth_authenticated" not in st.session_state:
        st.session_state["nexus_auth_authenticated"] = False
    if "nexus_auth_user" not in st.session_state:
        st.session_state["nexus_auth_user"] = None
    if "nexus_auth_login_time" not in st.session_state:
        st.session_state["nexus_auth_login_time"] = None
    if "nexus_auth_last_activity" not in st.session_state:
        st.session_state["nexus_auth_last_activity"] = None
    if "auth_error" not in st.session_state:
        st.session_state["auth_error"] = None

    # Inactivity Session Expiration Check (30 minutes = 1800s)
    SESSION_TIMEOUT_SECONDS = 1800
    if st.session_state.get("nexus_auth_authenticated"):
        last_act = st.session_state.get("nexus_auth_last_activity", 0.0)
        current_now = time.time()
        if (current_now - last_act) > SESSION_TIMEOUT_SECONDS:
            active_u = st.session_state.get("nexus_auth_user", {})
            auth_manager.record_session_expired(active_u.get("username", "UNKNOWN"))
            st.session_state["nexus_auth_authenticated"] = False
            st.session_state["nexus_auth_user"] = None
            st.session_state["nexus_auth_login_time"] = None
            st.session_state["nexus_auth_last_activity"] = None
            st.session_state["auth_error"] = "Your session has expired due to 30 minutes of inactivity. Please sign in again."
            st.rerun()
        else:
            # Revalidate live account status AND live role from database on every run
            active_u = st.session_state.get("nexus_auth_user") or {}
            u_name = active_u.get("username", "")
            live_u = auth_manager.get_user_by_username(u_name) if u_name else None
            if not live_u or not live_u.get("is_active"):
                st.session_state["nexus_auth_authenticated"] = False
                st.session_state["nexus_auth_user"] = None
                st.session_state["auth_error"] = "Your account has been deactivated by an administrator. Access revoked."
                st.rerun()
            else:
                st.session_state["nexus_auth_user"] = live_u
                st.session_state["nexus_auth_last_activity"] = current_now

    # Custom CSS for dark-mode styling - Enterprise Control Tower Theme
    st.markdown(
        """
        <style>
        /* ══════════════════════════════════════════════════════════════════
           NEXUS-AI ENTERPRISE CONTROL TOWER — FUTURISTIC DESIGN SYSTEM
           ══════════════════════════════════════════════════════════════════ */
        :root {
            /* ── Core Backgrounds ── */
            --nx-bg-space: #030712;
            --nx-bg-app: #050914;
            --nx-bg-surface: #0A101F;
            --nx-bg-card: #0D1629;
            --nx-bg-card-hover: #13223D;
            --nx-bg-card-elevated: #162442;
            --nx-bg-glass: rgba(13, 22, 41, 0.78);
            --nx-bg-glass-heavy: rgba(8, 14, 28, 0.92);

            /* ── Primary & Secondary Neon Accents ── */
            --nx-cyan-electric: #00F0FF;
            --nx-cyan-bright: #06B6D4;
            --nx-cyan-glow: rgba(0, 240, 255, 0.22);
            --nx-cyan-subtle: rgba(0, 240, 255, 0.08);

            --nx-blue-electric: #38BDF8;
            --nx-blue-deep: #2563EB;
            --nx-blue-glow: rgba(37, 99, 235, 0.24);

            --nx-purple-neon: #8B5CF6;
            --nx-purple-glow: rgba(139, 92, 246, 0.22);
            --nx-purple-subtle: rgba(139, 92, 246, 0.08);

            --nx-teal-cyber: #14B8A6;
            --nx-teal-glow: rgba(20, 184, 166, 0.20);

            --nx-emerald-neon: #10B981;
            --nx-emerald-glow: rgba(16, 185, 129, 0.22);

            --nx-amber-warning: #F59E0B;
            --nx-amber-glow: rgba(245, 158, 11, 0.24);

            --nx-rose-disrupt: #EF4444;
            --nx-rose-glow: rgba(239, 68, 68, 0.32);

            /* ── Borders & Outlines ── */
            --nx-border-faint: rgba(56, 189, 248, 0.08);
            --nx-border-subtle: rgba(56, 189, 248, 0.16);
            --nx-border-medium: rgba(56, 189, 248, 0.28);
            --nx-border-strong: rgba(0, 240, 255, 0.55);
            --nx-border-glass: rgba(255, 255, 255, 0.06);

            /* ── High-Contrast Typography ── */
            --nx-text-primary: #F8FAFC;
            --nx-text-secondary: #CBD5E1;
            --nx-text-muted: #94A3B8;
            --nx-text-dim: #64748B;
            --nx-text-cyan: #38BDF8;
            --nx-text-neon: #00F0FF;

            /* ── Futuristic Shadows & Radial Glows ── */
            --nx-shadow-card: 0 4px 24px -2px rgba(0, 0, 0, 0.65), 0 0 0 1px rgba(56, 189, 248, 0.10);
            --nx-shadow-elevated: 0 12px 32px -4px rgba(0, 0, 0, 0.75), 0 0 20px rgba(0, 240, 255, 0.14);
            --nx-shadow-cyan-glow: 0 0 24px rgba(0, 240, 255, 0.20);
            --nx-shadow-purple-glow: 0 0 24px rgba(139, 92, 246, 0.20);
            --nx-shadow-amber-glow: 0 0 24px rgba(245, 158, 11, 0.22);
            --nx-shadow-rose-glow: 0 0 28px rgba(239, 68, 68, 0.30);

            /* ── Radii ── */
            --nx-radius-sm: 6px;
            --nx-radius-md: 10px;
            --nx-radius-lg: 14px;
            --nx-radius-xl: 18px;
            --nx-radius-full: 9999px;

            /* ── Easing & Transitions ── */
            --nx-trans-fast: all 0.18s cubic-bezier(0.4, 0, 0.2, 1);
            --nx-trans-smooth: all 0.28s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* ── Global Space & Deep Ambient Backdrop ──────────────────────── */
        [data-testid="stAppViewContainer"],
        [data-testid="stApp"],
        .main,
        .main .block-container {
            background-color: var(--nx-bg-space) !important;
            background-image:
                radial-gradient(ellipse 900px 520px at 12% 0%, rgba(37, 99, 235, 0.08) 0%, transparent 70%),
                radial-gradient(ellipse 850px 480px at 88% 8%, rgba(139, 92, 246, 0.07) 0%, transparent 70%),
                radial-gradient(ellipse 950px 650px at 50% 92%, rgba(0, 240, 255, 0.05) 0%, transparent 75%),
                linear-gradient(180deg, #050914 0%, #030712 100%) !important;
            background-attachment: fixed !important;
        }

        /* ── Cybernetic Micro-Grid Overlay ────────────────────────────── */
        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image: radial-gradient(rgba(56, 189, 248, 0.045) 1px, transparent 1px);
            background-size: 38px 38px;
            pointer-events: none;
            z-index: 0;
            opacity: 0.85;
        }

        /* ── Streamlit Top Header Bar (Transparent to Prevent Occlusion) ── */
        header[data-testid="stHeader"] {
            background: transparent !important;
            backdrop-filter: blur(8px) !important;
            -webkit-backdrop-filter: blur(8px) !important;
        }

        /* ── Main Content Area Wrapper ───────────────────────────────── */
        section.main > div,
        .block-container {
            background: transparent !important;
            color: var(--nx-text-primary) !important;
            padding-top: 3.85rem !important;
            padding-bottom: 2.8rem !important;
            max-width: 1440px !important;
            position: relative;
            z-index: 1;
        }

        /* ── Custom Cybernetic Scrollbars ─────────────────────────────── */
        ::-webkit-scrollbar {
            width: 7px;
            height: 7px;
        }
        ::-webkit-scrollbar-track {
            background: #040813;
        }
        ::-webkit-scrollbar-thumb {
            background: #17243C;
            border-radius: var(--nx-radius-full);
            border: 1px solid rgba(56, 189, 248, 0.15);
            transition: var(--nx-trans-fast);
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--nx-cyan-bright);
            box-shadow: 0 0 10px rgba(0, 240, 255, 0.5);
        }

        /* ── Typography & General Text ───────────────────────────────── */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", "Helvetica Neue", Arial, sans-serif !important;
        }
        .stMarkdown p,
        .stMarkdown li,
        .stMarkdown span,
        .stMarkdown label,
        p, span, li, label {
            color: var(--nx-text-secondary) !important;
        }
        h1, h2, h3, h4, h5, h6,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: var(--nx-text-primary) !important;
            letter-spacing: -0.025em !important;
            font-weight: 700 !important;
        }

        /* ── Sidebar Navigation Dock (Full-Width, One Option Per Line) ─ */
        [data-testid="stSidebar"] div[data-testid="stRadio"]:has(div[aria-label="Navigation"]) > div[role="radiogroup"],
        [data-testid="stSidebar"] div[data-testid="stRadio"]:first-of-type > div[role="radiogroup"],
        div[data-testid="stRadio"]:has(div[aria-label="Navigation"]) > div[role="radiogroup"] {
            display: flex !important;
            flex-direction: column !important;
            gap: 4px !important;
            width: 100% !important;
            background: rgba(10, 16, 32, 0.65) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            padding: 5px !important;
            border-radius: var(--nx-radius-md) !important;
            border: 1px solid var(--nx-border-subtle) !important;
            margin-bottom: 0.6rem !important;
            box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.3) !important;
        }

        /* Each navigation option occupies full available sidebar width */
        [data-testid="stSidebar"] div[data-testid="stRadio"]:has(div[aria-label="Navigation"]) > div[role="radiogroup"] > label,
        [data-testid="stSidebar"] div[data-testid="stRadio"]:first-of-type > div[role="radiogroup"] > label,
        div[data-testid="stRadio"]:has(div[aria-label="Navigation"]) > div[role="radiogroup"] > label {
            width: 100% !important;
            flex: 1 1 100% !important;
            box-sizing: border-box !important;
            background: rgba(15, 23, 42, 0.55) !important;
            border: 1px solid rgba(30, 41, 59, 0.70) !important;
            border-radius: 6px !important;
            padding: 7px 11px !important;
            margin: 0 !important;
            cursor: pointer !important;
            transition: all 0.15s ease-in-out !important;
            font-weight: 600 !important;
            font-size: 0.81rem !important;
            color: var(--nx-text-secondary) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            text-align: left !important;
        }

        [data-testid="stSidebar"] div[data-testid="stRadio"]:has(div[aria-label="Navigation"]) > div[role="radiogroup"] > label:hover,
        [data-testid="stSidebar"] div[data-testid="stRadio"]:first-of-type > div[role="radiogroup"] > label:hover,
        div[data-testid="stRadio"]:has(div[aria-label="Navigation"]) > div[role="radiogroup"] > label:hover {
            border-color: var(--nx-cyan-bright) !important;
            color: #FFFFFF !important;
            background: rgba(22, 38, 70, 0.85) !important;
            box-shadow: 0 0 10px rgba(6, 182, 212, 0.20) !important;
            transform: translateX(2px) !important;
        }

        [data-testid="stSidebar"] div[data-testid="stRadio"]:has(div[aria-label="Navigation"]) > div[role="radiogroup"] > label[data-checked="true"],
        [data-testid="stSidebar"] div[data-testid="stRadio"]:has(div[aria-label="Navigation"]) > div[role="radiogroup"] > label:has(input:checked),
        [data-testid="stSidebar"] div[data-testid="stRadio"]:first-of-type > div[role="radiogroup"] > label[data-checked="true"],
        [data-testid="stSidebar"] div[data-testid="stRadio"]:first-of-type > div[role="radiogroup"] > label:has(input:checked),
        div[data-testid="stRadio"]:has(div[aria-label="Navigation"]) > div[role="radiogroup"] > label[data-checked="true"],
        div[data-testid="stRadio"]:has(div[aria-label="Navigation"]) > div[role="radiogroup"] > label:has(input:checked) {
            background: linear-gradient(90deg, rgba(6, 182, 212, 0.25) 0%, rgba(37, 99, 235, 0.18) 100%) !important;
            border: 1px solid var(--nx-cyan-bright) !important;
            border-left: 4px solid var(--nx-cyan-electric) !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 10px rgba(6, 182, 212, 0.25), inset 0 0 8px rgba(6, 182, 212, 0.10) !important;
        }

        [data-testid="stSidebar"] div[data-testid="stRadio"]:has(div[aria-label="Navigation"]) > div[role="radiogroup"] input[type="radio"],
        [data-testid="stSidebar"] div[data-testid="stRadio"]:first-of-type > div[role="radiogroup"] input[type="radio"],
        div[data-testid="stRadio"]:has(div[aria-label="Navigation"]) > div[role="radiogroup"] input[type="radio"] {
            display: none !important;
        }

        [data-testid="stSidebar"] div[data-testid="stRadio"]:has(div[aria-label="Navigation"]) > div[role="radiogroup"] > label > div:first-child,
        [data-testid="stSidebar"] div[data-testid="stRadio"]:first-of-type > div[role="radiogroup"] > label > div:first-child,
        div[data-testid="stRadio"]:has(div[aria-label="Navigation"]) > div[role="radiogroup"] > label > div:first-child {
            display: none !important;
        }

        /* ── Map Dedicated Animation Control Bar ──────────────────────── */
        .map-control-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.90) 0%, rgba(10, 16, 32, 0.95) 100%);
            border: 1px solid var(--nx-border-subtle);
            border-radius: var(--nx-radius-md);
            padding: 8px 14px;
            margin-bottom: 10px;
            box-shadow: var(--nx-shadow-card);
        }

        /* ── Five-Agent Grid Equal Dimensions & Zero Clipping ───────── */
        .workflow-pipeline-grid {
            display: grid !important;
            grid-template-columns: repeat(5, minmax(0, 1fr)) !important;
            gap: 12px !important;
            position: relative !important;
            width: 100% !important;
            align-items: stretch !important;
        }
        @media (max-width: 1100px) {
            .workflow-pipeline-grid {
                grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
            }
        }
        @media (max-width: 720px) {
            .workflow-pipeline-grid {
                grid-template-columns: 1fr !important;
            }
        }

        .agent-workflow-card {
            background: var(--nx-bg-card);
            border: 1px solid var(--nx-border-subtle);
            border-radius: var(--nx-radius-md);
            padding: 14px 14px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 185px;
            height: 100%;
            min-width: 0;
            width: 100%;
            box-sizing: border-box;
            transition: var(--nx-trans-smooth);
            position: relative;
            backdrop-filter: blur(8px);
        }
        .agent-card-name {
            font-size: 0.85rem;
            font-weight: 700;
            margin-top: 4px;
            margin-bottom: 3px;
            line-height: 1.25;
            overflow-wrap: break-word;
            word-break: normal;
            white-space: normal;
        }
        .agent-card-role {
            font-size: 0.72rem;
            color: var(--nx-text-muted);
            line-height: 1.35;
            margin-bottom: 8px;
            overflow-wrap: break-word;
            white-space: normal;
        }

        /* ── Sidebar (Glassmorphic Command Console) ─────────────────── */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
            background-color: var(--nx-bg-surface) !important;
            background-image: linear-gradient(180deg, rgba(13, 22, 41, 0.95) 0%, rgba(5, 9, 20, 0.98) 100%) !important;
            border-right: 1px solid var(--nx-border-subtle) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            box-shadow: 4px 0 24px rgba(0, 0, 0, 0.45) !important;
        }
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stNumberInput label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: var(--nx-text-muted) !important;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--nx-text-primary) !important;
        }
        .sidebar-section-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--nx-text-cyan);
            margin-bottom: 8px;
        }

        /* ── Authentication & RBAC Design Elements ───────────────────── */
        .user-profile-badge {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(10, 16, 32, 0.95) 100%);
            border: 1px solid var(--nx-border-subtle);
            border-radius: var(--nx-radius-md);
            padding: 10px 12px;
            margin-bottom: 10px;
            box-shadow: var(--nx-shadow-card);
        }
        .role-pill {
            display: inline-block;
            font-size: 0.68rem;
            font-weight: 800;
            padding: 2px 8px;
            border-radius: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-family: monospace;
        }
        .role-admin {
            color: #C084FC !important;
            background: rgba(139, 92, 246, 0.16) !important;
            border: 1px solid rgba(139, 92, 246, 0.35) !important;
        }
        .role-reviewer {
            color: #38BDF8 !important;
            background: rgba(6, 182, 212, 0.16) !important;
            border: 1px solid rgba(6, 182, 212, 0.35) !important;
        }
        .role-viewer {
            color: #34D399 !important;
            background: rgba(16, 185, 129, 0.16) !important;
            border: 1px solid rgba(16, 185, 129, 0.35) !important;
        }

        /* ── Enterprise Masthead Header ──────────────────────────────── */
        .control-tower-header {
            background: linear-gradient(135deg, rgba(16, 28, 54, 0.92) 0%, rgba(9, 16, 32, 0.95) 100%);
            border: 1px solid var(--nx-border-medium);
            border-radius: var(--nx-radius-lg);
            padding: 18px 24px;
            margin-bottom: 1.3rem;
            box-shadow: var(--nx-shadow-card);
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
        }
        .control-tower-header::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #00F0FF 0%, #38BDF8 30%, #3B82F6 60%, #8B5CF6 100%);
            box-shadow: 0 0 12px rgba(0, 240, 255, 0.6);
        }
        .masthead-title {
            font-size: 2.15rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em !important;
            background: linear-gradient(135deg, #FFFFFF 0%, #F1F5F9 50%, #38BDF8 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            margin: 0 !important;
            line-height: 1.15 !important;
        }
        .masthead-tagline {
            font-size: 0.92rem;
            font-weight: 500;
            color: var(--nx-text-cyan);
            letter-spacing: 0.04em;
            margin-top: 4px;
            margin-bottom: 8px;
        }
        .masthead-meta-row {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 12px;
            font-size: 0.84rem;
        }

        /* ── Telemetry Pills & Live Badges ────────────────────────────── */
        .telemetry-pill {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 4px 14px;
            border-radius: var(--nx-radius-full);
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
            transition: var(--nx-trans-smooth);
        }
        .telemetry-live {
            background: rgba(16, 185, 129, 0.16);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.50);
            box-shadow: 0 0 14px rgba(16, 185, 129, 0.20);
        }
        .telemetry-pulse {
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: #34D399;
            box-shadow: 0 0 10px #34D399;
            animation: pulse-dot 1.8s infinite;
        }
        @keyframes pulse-dot {
            0% { opacity: 1; transform: scale(1); box-shadow: 0 0 10px #34D399; }
            50% { opacity: 0.35; transform: scale(0.85); box-shadow: 0 0 4px #34D399; }
            100% { opacity: 1; transform: scale(1); box-shadow: 0 0 10px #34D399; }
        }
        .telemetry-standby {
            background: rgba(6, 182, 212, 0.14);
            color: #38BDF8;
            border: 1px solid rgba(6, 182, 212, 0.45);
        }
        .telemetry-local {
            background: rgba(148, 163, 184, 0.12);
            color: #E2E8F0;
            border: 1px solid rgba(148, 163, 184, 0.30);
        }
        .telemetry-fallback {
            background: rgba(245, 158, 11, 0.16);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.50);
            box-shadow: 0 0 14px rgba(245, 158, 11, 0.20);
        }

        /* ── Equal-width Four-Column KPI Grid ──────────────────────── */
        .kpi-grid {
            display: grid !important;
            grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
            gap: 14px !important;
            width: 100% !important;
            align-items: stretch !important;
            margin-bottom: 14px !important;
            box-sizing: border-box !important;
        }

        @media (max-width: 1100px) {
            .kpi-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            }
        }

        @media (max-width: 640px) {
            .kpi-grid {
                grid-template-columns: 1fr !important;
            }
        }

        /* ── Executive KPI Cards (Consistent Dimensions & Full Text) ─── */
        .kpi-grid .kpi-card,
        .kpi-card {
            background: linear-gradient(180deg, var(--nx-bg-card) 0%, rgba(10, 17, 34, 0.95) 100%);
            border: 1px solid var(--nx-border-subtle);
            border-radius: var(--nx-radius-md);
            padding: 16px 18px;
            height: 100% !important;
            min-height: 145px !important;
            width: 100% !important;
            min-width: 0 !important;
            box-sizing: border-box !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: space-between !important;
            transition: var(--nx-trans-smooth);
            box-shadow: var(--nx-shadow-card);
            position: relative;
            overflow: visible;
            backdrop-filter: blur(8px);
        }
        .kpi-grid .kpi-card::after,
        .kpi-card::after {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, transparent 100%);
            pointer-events: none;
        }
        .kpi-grid .kpi-card:hover,
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--nx-shadow-elevated);
            border-color: var(--nx-border-medium);
        }
        .kpi-card-glow-cyan   { border-top: 3px solid var(--nx-cyan-bright); box-shadow: 0 4px 18px rgba(0, 0, 0, 0.5), inset 0 2px 8px rgba(0, 240, 255, 0.10); }
        .kpi-card-glow-teal   { border-top: 3px solid var(--nx-teal-cyber); box-shadow: 0 4px 18px rgba(0, 0, 0, 0.5), inset 0 2px 8px rgba(20, 184, 166, 0.10); }
        .kpi-card-glow-sky    { border-top: 3px solid var(--nx-blue-electric); box-shadow: 0 4px 18px rgba(0, 0, 0, 0.5), inset 0 2px 8px rgba(56, 189, 248, 0.10); }
        .kpi-card-glow-amber  { border-top: 3px solid var(--nx-amber-warning); box-shadow: 0 4px 18px rgba(0, 0, 0, 0.5), inset 0 2px 8px rgba(245, 158, 11, 0.10); }
        .kpi-card-glow-rose   { border-top: 3px solid #F43F5E; box-shadow: 0 4px 18px rgba(0, 0, 0, 0.5), inset 0 2px 8px rgba(244, 63, 94, 0.12); }
        .kpi-card-glow-emerald{ border-top: 3px solid var(--nx-emerald-neon); box-shadow: 0 4px 18px rgba(0, 0, 0, 0.5), inset 0 2px 8px rgba(16, 185, 129, 0.10); }

        .kpi-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        .kpi-card-label {
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--nx-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            line-height: 1.2;
        }
        .kpi-card-icon {
            font-size: 1.10rem;
            line-height: 1;
            filter: drop-shadow(0 0 6px rgba(0, 240, 255, 0.35));
        }
        .kpi-card-value {
            font-size: 1.50rem;
            font-weight: 800;
            color: var(--nx-text-primary);
            letter-spacing: -0.025em;
            line-height: 1.2;
            margin-top: 4px;
            margin-bottom: 6px;
            font-feature-settings: "tnum";
            text-shadow: 0 0 16px rgba(255, 255, 255, 0.15);
        }
        .kpi-card-subtitle {
            font-size: 0.74rem;
            color: var(--nx-text-secondary);
            line-height: 1.4;
            font-weight: 400;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            word-break: normal !important;
            margin-top: auto;
        }

        /* ── Risk & Severity Badges (Pills) ──────────────────────────── */
        .badge-critical {
            background-color: rgba(239, 68, 68, 0.18) !important;
            color: #F87171 !important;
            border: 1px solid rgba(239, 68, 68, 0.55) !important;
            padding: 3px 10px !important;
            border-radius: var(--nx-radius-full) !important;
            font-weight: 700 !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.03em !important;
            display: inline-flex !important;
            align-items: center !important;
            box-shadow: 0 0 10px rgba(239, 68, 68, 0.22) !important;
        }
        .badge-high {
            background-color: rgba(245, 158, 11, 0.18) !important;
            color: #FBBF24 !important;
            border: 1px solid rgba(245, 158, 11, 0.50) !important;
            padding: 3px 10px !important;
            border-radius: var(--nx-radius-full) !important;
            font-weight: 700 !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.03em !important;
            display: inline-flex !important;
            align-items: center !important;
            box-shadow: 0 0 10px rgba(245, 158, 11, 0.20) !important;
        }
        .badge-medium {
            background-color: rgba(234, 179, 8, 0.18) !important;
            color: #FDE047 !important;
            border: 1px solid rgba(234, 179, 8, 0.50) !important;
            padding: 3px 10px !important;
            border-radius: var(--nx-radius-full) !important;
            font-weight: 700 !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.03em !important;
            display: inline-flex !important;
            align-items: center !important;
        }
        .badge-low {
            background-color: rgba(16, 185, 129, 0.18) !important;
            color: #34D399 !important;
            border: 1px solid rgba(16, 185, 129, 0.50) !important;
            padding: 3px 10px !important;
            border-radius: var(--nx-radius-full) !important;
            font-weight: 700 !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.03em !important;
            display: inline-flex !important;
            align-items: center !important;
        }

        /* ── Futuristic Form Controls & Inputs ───────────────────────── */
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {
            background-color: rgba(15, 24, 45, 0.85) !important;
            color: var(--nx-text-primary) !important;
            border: 1px solid var(--nx-border-medium) !important;
            border-radius: var(--nx-radius-md) !important;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
            transition: var(--nx-trans-smooth) !important;
        }
        [data-testid="stSelectbox"] > div > div:focus-within,
        [data-testid="stNumberInput"] input:focus,
        [data-testid="stTextInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus {
            border-color: var(--nx-cyan-bright) !important;
            box-shadow: 0 0 0 2px var(--nx-cyan-glow), inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
        }

        /* ── Custom Decision Action Buttons ────────────────────────── */
        div[data-testid="stButton"] > button:has(p:contains("Reject")),
        button[key*="reject"] {
            background: rgba(239, 68, 68, 0.14) !important;
            border: 1px solid rgba(239, 68, 68, 0.55) !important;
            color: #F87171 !important;
            font-weight: 700 !important;
            transition: var(--nx-trans-smooth) !important;
            border-radius: var(--nx-radius-md) !important;
        }
        div[data-testid="stButton"] > button:has(p:contains("Reject")):hover,
        button[key*="reject"]:hover {
            background: rgba(239, 68, 68, 0.28) !important;
            border-color: #EF4444 !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 18px var(--nx-rose-glow) !important;
            transform: translateY(-1px) !important;
        }
        div[data-testid="stButton"] > button:has(p:contains("Approve")),
        button[key*="approve"] {
            background: linear-gradient(135deg, #059669 0%, #0D9488 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            font-weight: 700 !important;
            border-radius: var(--nx-radius-md) !important;
            box-shadow: 0 4px 16px rgba(16, 185, 129, 0.40) !important;
            transition: var(--nx-trans-smooth) !important;
        }
        div[data-testid="stButton"] > button:has(p:contains("Approve")):hover,
        button[key*="approve"]:hover {
            background: linear-gradient(135deg, #10B981 0%, #14B8A6 100%) !important;
            box-shadow: 0 6px 22px rgba(16, 185, 129, 0.55) !important;
            transform: translateY(-1px) !important;
        }

        /* ── Primary & Secondary Cyber Buttons ───────────────────────── */
        [data-testid="stBaseButton-primary"],
        button[kind="primary"] {
            background: linear-gradient(135deg, #0284C7 0%, #2563EB 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: var(--nx-radius-md) !important;
            font-weight: 700 !important;
            padding: 0.52rem 1.3rem !important;
            box-shadow: 0 4px 16px rgba(2, 132, 199, 0.40) !important;
            transition: var(--nx-trans-smooth) !important;
        }
        [data-testid="stBaseButton-primary"]:hover,
        button[kind="primary"]:hover {
            background: linear-gradient(135deg, #06B6D4 0%, #1D4ED8 100%) !important;
            box-shadow: 0 6px 22px rgba(6, 182, 212, 0.50) !important;
            transform: translateY(-1px) !important;
        }
        [data-testid="stBaseButton-secondary"],
        button[kind="secondary"] {
            background: rgba(18, 29, 53, 0.75) !important;
            color: var(--nx-text-primary) !important;
            border: 1px solid var(--nx-border-medium) !important;
            border-radius: var(--nx-radius-md) !important;
            font-weight: 600 !important;
            transition: var(--nx-trans-smooth) !important;
            backdrop-filter: blur(8px) !important;
        }
        [data-testid="stBaseButton-secondary"]:hover,
        button[kind="secondary"]:hover {
            background: rgba(26, 42, 77, 0.90) !important;
            border-color: var(--nx-cyan-bright) !important;
            color: var(--nx-cyan-electric) !important;
            box-shadow: 0 0 14px var(--nx-cyan-subtle) !important;
            transform: translateY(-1px) !important;
        }
        [data-testid="stDownloadButton"] button {
            background: linear-gradient(135deg, #0284C7 0%, #2563EB 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: var(--nx-radius-md) !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 16px rgba(2, 132, 199, 0.40) !important;
        }

        /* ── Dataframes & Tables (Dark Glass Theme) ───────────────────── */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--nx-border-subtle) !important;
            border-radius: var(--nx-radius-md) !important;
            overflow: hidden !important;
            background: var(--nx-bg-card) !important;
            box-shadow: var(--nx-shadow-card) !important;
        }
        [data-testid="stDataFrame"] [data-testid="stElementToolbar"],
        [data-testid="stElementToolbar"] {
            background-color: var(--nx-bg-surface) !important;
            border: 1px solid var(--nx-border-subtle) !important;
            border-radius: var(--nx-radius-sm) !important;
        }

        /* ── Section Dividers & Cards ───────────────────────────────── */
        hr {
            border-color: var(--nx-border-faint) !important;
            margin: 1.3rem 0 !important;
        }
        .user-guide-box {
            background: linear-gradient(135deg, rgba(16, 27, 51, 0.85) 0%, rgba(10, 18, 36, 0.90) 100%);
            border: 1px solid var(--nx-border-subtle);
            border-left: 4px solid var(--nx-cyan-bright);
            border-radius: var(--nx-radius-md);
            padding: 13px 18px;
            margin-bottom: 1.2rem;
            box-shadow: var(--nx-shadow-card);
            backdrop-filter: blur(8px);
        }
        .section-header-banner {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 0.85rem;
        }
        .section-header-title {
            font-size: 1.38rem;
            font-weight: 800;
            color: var(--nx-text-primary);
            letter-spacing: -0.025em;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .section-header-desc {
            font-size: 0.83rem;
            color: var(--nx-text-muted);
            margin-bottom: 1.1rem;
        }

        /* ── Plotly Map Container & Controls ────────────────────────── */
        [data-testid="stPlotlyChart"] {
            border: 1px solid var(--nx-border-subtle) !important;
            border-radius: var(--nx-radius-lg) !important;
            overflow: hidden !important;
            box-shadow: var(--nx-shadow-card) !important;
            background: #080F1E !important;
        }
        .plot-container.plotly {
            border-radius: var(--nx-radius-lg) !important;
        }

        /* ── Streamlit Toggle Polish & Spacing ──────────────────────── */
        [data-testid="stToggle"] {
            display: flex !important;
            align-items: center !important;
            justify-content: flex-end !important;
        }
        [data-testid="stToggle"] label {
            margin-bottom: 0 !important;
        }
        [data-testid="stToggle"] label p {
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            color: var(--nx-text-primary) !important;
            white-space: nowrap !important;
        }
        [data-testid="stToggle"] span[data-baseweb="toggle"] {
            background-color: #17243C !important;
            border: 1px solid var(--nx-border-medium) !important;
        }
        [data-testid="stToggle"] input:checked + div,
        [data-testid="stToggle"] [aria-checked="true"] {
            background-color: var(--nx-cyan-bright) !important;
            box-shadow: 0 0 10px rgba(6, 182, 212, 0.45) !important;
        }

        /* ── Plotly Updatemenu Buttons (Play / Pause / Replay) ────────── */
        .js-plotly-plot .updatemenu-container .updatemenu-button rect,
        .js-plotly-plot rect.updatemenu-item-rect,
        .js-plotly-plot rect.updatemenu-item-rect-active,
        .js-plotly-plot .updatemenu-button-active rect {
            fill: #151F32 !important;
            stroke: #334155 !important;
            stroke-width: 1.2px !important;
            rx: 6px !important;
            ry: 6px !important;
            cursor: pointer !important;
            transition: var(--nx-trans-smooth) !important;
        }
        .js-plotly-plot .updatemenu-container .updatemenu-button:hover rect,
        .js-plotly-plot rect.updatemenu-item-rect:hover,
        .js-plotly-plot rect.updatemenu-item-rect-active:hover {
            fill: #1D2D49 !important;
            stroke: var(--nx-cyan-bright) !important;
            filter: drop-shadow(0 0 6px rgba(0, 240, 255, 0.40)) !important;
        }
        .js-plotly-plot .updatemenu-container .updatemenu-button text,
        .js-plotly-plot text.updatemenu-item-text,
        .js-plotly-plot text.updatemenu-item-text-active,
        .js-plotly-plot .updatemenu-button-active text {
            fill: #F8FAFC !important;
            font-weight: 700 !important;
            font-size: 11px !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            cursor: pointer !important;
            user-select: none !important;
        }
        .js-plotly-plot .updatemenu-container .updatemenu-button:hover text,
        .js-plotly-plot text.updatemenu-item-text:hover,
        .js-plotly-plot text.updatemenu-item-text-active:hover {
            fill: var(--nx-cyan-electric) !important;
        }

        /* ── Feature 2: Five-Agent Workflow Orchestration Panel ─────── */
        .agent-orchestration-panel {
            background: linear-gradient(180deg, rgba(15, 24, 45, 0.95) 0%, rgba(10, 16, 32, 0.98) 100%);
            border: 1px solid var(--nx-border-medium);
            border-radius: var(--nx-radius-lg);
            padding: 14px 18px;
            margin-bottom: 1.3rem;
            box-shadow: var(--nx-shadow-card);
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(12px);
        }
        .agent-orchestration-panel::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, #3B82F6 0%, #06B6D4 25%, #F59E0B 50%, #10B981 75%, #8B5CF6 100%);
            box-shadow: 0 0 10px rgba(0, 240, 255, 0.4);
        }
        .agent-panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }
        .workflow-pipeline-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
            position: relative;
        }
        @media (max-width: 1024px) {
            .workflow-pipeline-grid {
                grid-template-columns: repeat(2, 1fr) !important;
            }
        }
        @media (max-width: 640px) {
            .workflow-pipeline-grid {
                grid-template-columns: 1fr !important;
            }
        }

        .agent-workflow-card {
            background: var(--nx-bg-card);
            border: 1px solid var(--nx-border-subtle);
            border-radius: var(--nx-radius-md);
            padding: 12px 14px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 120px;
            box-sizing: border-box;
            transition: var(--nx-trans-smooth);
            position: relative;
            backdrop-filter: blur(6px);
        }
        .agent-workflow-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--nx-shadow-elevated);
            border-color: var(--nx-border-medium);
        }
        .agent-card-waiting {
            background: rgba(11, 18, 35, 0.70) !important;
            border: 1px solid var(--nx-border-faint) !important;
            opacity: 0.65;
        }
        .agent-card-processing {
            background: linear-gradient(135deg, #13223E 0%, #1A2F55 100%) !important;
            transform: translateY(-2px);
            animation: agent-card-pulse 1.4s ease-in-out infinite alternate;
        }
        @keyframes agent-card-pulse {
            0% { box-shadow: 0 0 10px rgba(6, 182, 212, 0.20); }
            100% { box-shadow: 0 0 24px rgba(6, 182, 212, 0.58), inset 0 0 12px rgba(6, 182, 212, 0.15); }
        }
        .agent-card-completed {
            background: var(--nx-bg-card) !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
        }
        .agent-card-error {
            background: rgba(239, 68, 68, 0.10) !important;
            border: 1px solid #EF4444 !important;
            box-shadow: 0 0 16px var(--nx-rose-glow);
        }

        .agent-accent-blue   { border-top: 3px solid #3B82F6 !important; }
        .agent-accent-cyan   { border-top: 3px solid #06B6D4 !important; }
        .agent-accent-amber  { border-top: 3px solid #F59E0B !important; }
        .agent-accent-green  { border-top: 3px solid #10B981 !important; }
        .agent-accent-purple { border-top: 3px solid #8B5CF6 !important; }

        .agent-processing-blue   { border: 1.5px solid #3B82F6 !important; box-shadow: 0 0 20px rgba(59, 130, 246, 0.45) !important; }
        .agent-processing-cyan   { border: 1.5px solid #06B6D4 !important; box-shadow: 0 0 20px rgba(6, 182, 212, 0.45) !important; }
        .agent-processing-amber  { border: 1.5px solid #F59E0B !important; box-shadow: 0 0 20px rgba(245, 158, 11, 0.45) !important; }
        .agent-processing-green  { border: 1.5px solid #10B981 !important; box-shadow: 0 0 20px rgba(16, 185, 129, 0.45) !important; }
        .agent-processing-purple { border: 1.5px solid #8B5CF6 !important; box-shadow: 0 0 20px rgba(139, 92, 246, 0.45) !important; }

        .agent-status-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 2px 8px;
            border-radius: var(--nx-radius-full);
            font-size: 0.70rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .badge-state-waiting {
            background: rgba(100, 116, 139, 0.15);
            color: #94A3B8;
            border: 1px solid rgba(100, 116, 139, 0.3);
        }
        .badge-state-processing {
            background: rgba(6, 182, 212, 0.18);
            color: #38BDF8;
            border: 1px solid #06B6D4;
            box-shadow: 0 0 10px rgba(6, 182, 212, 0.35);
        }
        .badge-state-completed {
            background: rgba(16, 185, 129, 0.16);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.45);
        }
        .badge-state-error {
            background: rgba(239, 68, 68, 0.18);
            color: #F87171;
            border: 1px solid #EF4444;
        }

        .agent-spinner {
            display: inline-block;
            width: 8px;
            height: 8px;
            border: 1.5px solid currentColor;
            border-right-color: transparent;
            border-radius: 50%;
            animation: agent-spin 0.8s linear infinite;
        }
        @keyframes agent-spin {
            to { transform: rotate(360deg); }
        }

        .agent-progress-track {
            width: 100%;
            height: 3px;
            background: #17243C;
            border-radius: var(--nx-radius-full);
            overflow: hidden;
            margin-top: 6px;
        }
        .agent-progress-bar {
            height: 100%;
            width: 60%;
            border-radius: var(--nx-radius-full);
            animation: agent-progress-indeterminate 1.2s ease-in-out infinite alternate;
        }
        @keyframes agent-progress-indeterminate {
            0% { width: 15%; margin-left: 0; }
            100% { width: 75%; margin-left: 25%; }
        }

        .agent-order-tag {
            font-size: 0.68rem;
            font-weight: 800;
            color: var(--nx-text-dim);
            font-family: monospace;
            letter-spacing: 0.05em;
        }
        .agent-card-name {
            font-size: 0.86rem;
            font-weight: 700;
            margin-top: 4px;
            margin-bottom: 2px;
            line-height: 1.2;
        }
        .agent-card-role {
            font-size: 0.70rem;
            color: var(--nx-text-muted);
            line-height: 1.3;
            margin-bottom: 8px;
        }

        /* ── Feature 3: Disruption Propagation Impact Summary ───────── */
        .propagation-impact-summary {
            background: linear-gradient(180deg, var(--nx-bg-card) 0%, rgba(10, 17, 34, 0.95) 100%);
            border: 1px solid var(--nx-border-subtle);
            border-top: 3px solid #EF4444;
            border-radius: var(--nx-radius-lg);
            padding: 16px 20px;
            margin-bottom: 1.3rem;
            box-shadow: var(--nx-shadow-card);
            position: relative;
            backdrop-filter: blur(10px);
        }
        .propagation-summary-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 14px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }
        .propagation-summary-title-group {
            display: flex;
            flex-direction: column;
            gap: 3px;
        }
        .propagation-summary-kicker {
            font-size: 0.70rem;
            font-weight: 700;
            color: var(--nx-text-muted);
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .propagation-summary-title {
            font-size: 1.18rem;
            font-weight: 800;
            color: var(--nx-text-primary);
        }
        .propagation-summary-badges {
            display: flex;
            gap: 8px;
            align-items: center;
            flex-wrap: wrap;
        }
        .propagation-summary-metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
        }
        @media (max-width: 900px) {
            .propagation-summary-metrics-grid {
                grid-template-columns: repeat(2, 1fr) !important;
            }
        }
        @media (max-width: 550px) {
            .propagation-summary-metrics-grid {
                grid-template-columns: 1fr !important;
            }
        }
        .propagation-summary-metric-card {
            background: rgba(10, 16, 32, 0.85);
            border: 1px solid var(--nx-border-subtle);
            border-radius: var(--nx-radius-md);
            padding: 12px 14px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: var(--nx-trans-smooth);
        }
        .propagation-summary-metric-card:hover {
            border-color: var(--nx-border-medium);
            transform: translateY(-1px);
        }
        .propagation-metric-label {
            font-size: 0.68rem;
            font-weight: 700;
            color: var(--nx-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }
        .propagation-metric-value {
            font-size: 1.30rem;
            font-weight: 800;
            margin-bottom: 4px;
            text-shadow: 0 0 12px rgba(255, 255, 255, 0.12);
        }
        .propagation-metric-sub {
            font-size: 0.72rem;
            color: var(--nx-text-dim);
            line-height: 1.3;
        }

        /* ── Accessibility & Reduced Motion ────────────────────────── */
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 2. Authentication Gate: Enforce login before rendering any dashboard components
    auth_user = st.session_state.get("nexus_auth_user")
    is_authenticated = bool(st.session_state.get("nexus_auth_authenticated") and auth_user)

    if not is_authenticated:
        st.sidebar.markdown(
            '''
            <div style="padding:4px 0 16px 0; border-bottom:1px solid #1E293B; margin-bottom:16px;">
              <div style="display:flex; align-items:center; gap:10px;">
                <div style="background:rgba(6,182,212,0.15); border:1px solid rgba(6,182,212,0.4); border-radius:10px; padding:8px 10px; font-size:1.4rem;">
                  🌐
                </div>
                <div>
                  <div style="font-size:1.18rem; font-weight:800; color:#F8FAFC; letter-spacing:-0.02em;">NEXUS-AI</div>
                  <div style="font-size:0.70rem; font-weight:600; color:#06B6D4; text-transform:uppercase; letter-spacing:0.08em;">Control Console</div>
                </div>
              </div>
              <div style="font-size:0.73rem; color:#64748B; margin-top:8px;">Enterprise Multi-Agent Supply Chain Resilience Tower</div>
            </div>
            <div style="padding:14px; background:rgba(15,23,42,0.7); border:1px solid rgba(56,189,248,0.2); border-radius:8px; text-align:center;">
              <div style="font-size:1.8rem; margin-bottom:6px;">🔒</div>
              <div style="font-weight:700; color:#F8FAFC; font-size:0.84rem;">Authentication Required</div>
              <div style="font-size:0.72rem; color:#94A3B8; margin-top:4px;">Please sign in to access the control tower telemetries and simulation engines.</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        # Render Login Portal or Initial Admin Provisioning
        st.markdown(
            '''
            <div style="max-width: 480px; margin: 1.5rem auto 1rem auto; text-align: center;">
              <div style="display:inline-flex; align-items:center; justify-content:center; width:64px; height:64px; border-radius:18px; background:rgba(6,182,212,0.15); border:1px solid rgba(6,182,212,0.4); font-size:2rem; margin-bottom:12px; box-shadow:0 0 24px rgba(6,182,212,0.25);">
                🌐
              </div>
              <h1 style="font-size:2.2rem; font-weight:800; color:#F8FAFC; letter-spacing:-0.03em; margin:0;">NEXUS-AI</h1>
              <div style="font-size:0.85rem; font-weight:600; color:#06B6D4; text-transform:uppercase; letter-spacing:0.10em; margin-top:4px;">
                Enterprise Resilience Control Tower
              </div>
              <div style="font-size:0.80rem; color:#94A3B8; margin-top:6px; line-height:1.4;">
                Autonomous Multi-Tier Supply Chain Intelligence & Shock Recovery
              </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        login_c1, login_c2, login_c3 = st.columns([1, 2.2, 1])
        with login_c2:
            has_admin = auth_manager.has_any_admin()
            if not has_admin:
                st.markdown(
                    '''
                    <div style="background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.35); border-left:4px solid #F59E0B; border-radius:8px; padding:12px 14px; margin-bottom:14px; font-size:0.82rem; color:#FBBF24;">
                      <b>First-Time System Initialization:</b> No administrator account detected. Please provision the primary system administrator below.
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )
                with st.form("form_initial_admin_setup"):
                    admin_u = st.text_input("Administrator Username:", placeholder="e.g. admin", help="At least 3 alphanumeric characters.")
                    admin_e = st.text_input("Administrator Email:", placeholder="e.g. admin@nexus-ai.internal")
                    admin_p1 = st.text_input("Administrator Password:", type="password", help="At least 8 characters with letters and numbers.")
                    admin_p2 = st.text_input("Confirm Password:", type="password")
                    sub_setup = st.form_submit_button("Provision Initial Administrator", use_container_width=True)

                    if sub_setup:
                        if admin_p1 != admin_p2:
                            st.error("Passwords do not match.")
                        else:
                            ok, msg = auth_manager.init_first_admin(admin_u, admin_e, admin_p1)
                            if ok:
                                # Auto-authenticate new admin
                                _, _, user_obj = auth_manager.authenticate(admin_u, admin_p1)
                                st.session_state["nexus_auth_authenticated"] = True
                                st.session_state["nexus_auth_user"] = user_obj
                                st.session_state["nexus_auth_login_time"] = time.time()
                                st.session_state["nexus_auth_last_activity"] = time.time()
                                st.session_state["main_nav_section"] = "1. 📊 Executive Dashboard"
                                st.success("Initial Administrator provisioned! Entering Control Tower...")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(msg)
            else:
                if st.session_state.get("auth_error"):
                    st.warning(st.session_state["auth_error"])
                    st.session_state["auth_error"] = None

                st.markdown(
                    '''
                    <div style="background:linear-gradient(180deg, rgba(15, 23, 42, 0.85) 0%, rgba(10, 16, 32, 0.95) 100%); border:1px solid rgba(56, 189, 248, 0.22); border-radius:12px; padding:22px 24px; box-shadow:0 8px 32px rgba(0,0,0,0.5); backdrop-filter:blur(12px);">
                      <div style="font-size:1.05rem; font-weight:800; color:#F8FAFC; margin-bottom:4px;">Sign In to Control Tower</div>
                      <div style="font-size:0.75rem; color:#94A3B8; margin-bottom:16px;">Enter your authorized enterprise credentials to proceed.</div>
                    ''',
                    unsafe_allow_html=True,
                )

                with st.form("form_user_login"):
                    u_ident = st.text_input("Username or Email Address:", placeholder="Enter username or email")
                    u_pwd = st.text_input("Password:", type="password", placeholder="Enter your password")
                    sub_login = st.form_submit_button("Sign In to Nexus-AI", use_container_width=True)

                    if sub_login:
                        ok, msg, user_obj = auth_manager.authenticate(u_ident, u_pwd)
                        if ok and user_obj:
                            st.session_state["nexus_auth_authenticated"] = True
                            st.session_state["nexus_auth_user"] = user_obj
                            st.session_state["nexus_auth_login_time"] = time.time()
                            st.session_state["nexus_auth_last_activity"] = time.time()
                            st.session_state["main_nav_section"] = "1. 📊 Executive Dashboard"
                            st.success("Authentication confirmed. Accessing Control Tower...")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(msg)

                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(
                    '''
                    <div style="margin-top:16px; padding:12px 14px; background:rgba(10, 16, 32, 0.60); border:1px solid rgba(255,255,255,0.06); border-radius:8px; font-size:0.72rem; color:#64748B;">
                      <div style="font-weight:700; color:#94A3B8; margin-bottom:4px;">ROLE-BASED AUTHORIZATION POLICIES:</div>
                      <div>&bull; <b>ADMIN:</b> Full control, user lifecycle & role management, disruption runs, capital sign-off.</div>
                      <div>&bull; <b>REVIEWER:</b> Full network simulation access, decision narratives, plan approvals with mandatory justification.</div>
                      <div>&bull; <b>VIEWER:</b> Read-only situational monitoring across all 7 sections and audit ledgers.</div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )
        st.stop()

    # 3. Sidebar: Configuration & Global Controls (Authenticated)
    st.sidebar.markdown(
        """
        <div style="padding:4px 0 16px 0; border-bottom:1px solid #1E293B; margin-bottom:16px;">
          <div style="display:flex; align-items:center; gap:10px;">
            <div style="background:rgba(6,182,212,0.15); border:1px solid rgba(6,182,212,0.4); border-radius:10px; padding:8px 10px; font-size:1.4rem;">
              🌐
            </div>
            <div>
              <div style="font-size:1.18rem; font-weight:800; color:#F8FAFC; letter-spacing:-0.02em;">NEXUS-AI</div>
              <div style="font-size:0.70rem; font-weight:600; color:#06B6D4; text-transform:uppercase; letter-spacing:0.08em;">Control Console</div>
            </div>
          </div>
          <div style="font-size:0.73rem; color:#64748B; margin-top:8px;">Enterprise Multi-Agent Supply Chain Resilience Tower</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Current Authenticated User Profile Card in Sidebar (Resolved directly from live DB)
    perms = get_ui_control_permissions(auth_manager, auth_user.get("username", ""), st.session_state)
    curr_u_name = perms["username"]
    curr_u_role = perms["role"]
    is_viewer = perms["is_viewer"]
    can_exec_sim = perms["can_execute_simulation"]
    role_class = {
        "ADMIN": "role-admin",
        "REVIEWER": "role-reviewer",
        "VIEWER": "role-viewer",
    }.get(curr_u_role, "role-viewer")

    st.sidebar.markdown(
        f'''
        <div class="user-profile-badge">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
              <div style="font-size:0.68rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; font-weight:700;">CURRENT OPERATOR</div>
              <div style="font-weight:700; color:#F8FAFC; font-size:0.86rem; display:flex; align-items:center; gap:6px; margin-top:2px;">
                <span>👤</span> {curr_u_name}
              </div>
            </div>
            <div>
              <span class="role-pill {role_class}">{curr_u_role}</span>
            </div>
          </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    if st.sidebar.button("🚪 Sign Out", key="btn_auth_sign_out", use_container_width=True):
        auth_manager.record_logout(curr_u_name)
        st.session_state["nexus_auth_authenticated"] = False
        st.session_state["nexus_auth_user"] = None
        st.session_state["nexus_auth_login_time"] = None
        st.session_state["nexus_auth_last_activity"] = None
        st.session_state.pop("active_sim_fingerprint", None)
        st.session_state["force_new_run"] = False
        st.rerun()

    st.sidebar.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

    # Sidebar Navigation: 7 Core Control Tower Views (At the top of the sidebar)
    st.sidebar.markdown(
        '<div class="sidebar-section-badge" style="margin-top:0; margin-bottom:8px;"><span>🧭</span> NAVIGATION</div>',
        unsafe_allow_html=True,
    )

    nav_options = perms["nav_options"]

    selected_nav = st.sidebar.radio(
        "Navigation",
        nav_options,
        key="main_nav_section",
        label_visibility="collapsed",
    )

    st.sidebar.markdown("<div style='height: 10px; border-bottom: 1px solid rgba(56, 189, 248, 0.15); margin-bottom: 14px;'></div>", unsafe_allow_html=True)

    if is_viewer:
        st.sidebar.markdown(
            """
            <div style="background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.28); border-radius:8px; padding:10px 12px; margin-bottom:12px; color:#38BDF8; font-size:0.76rem; line-height:1.4;">
              👁️ <b>READ-ONLY ACCESS (VIEWER)</b><br>
              <span style="color:#CBD5E1;">Disruption simulator controls and re-run actions are disabled. You can view all 7 sections, existing simulation results, and reports.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.sidebar.markdown('<div class="sidebar-section-badge"><span>📂</span> 1. DATA SOURCE & TELEMETRY</div>', unsafe_allow_html=True)

    sap_is_ready = is_sap_configured()
    default_dataset_idx = 0 if sap_is_ready else 1
    data_source_options = [
        "SAP HANA Cloud (Live Database: HACKFEST0316)",
        "Standard Sample (data/dependencies.csv)",
        "Canonical Multi-Tier Network (with Factories & Routes)",
        "Upload Custom Dependencies CSV",
    ]

    data_choice = st.sidebar.radio(
        "Choose Supply Chain Data Source:",
        data_source_options,
        index=default_dataset_idx,
        disabled=perms["simulation_controls_disabled"],
    )
    if not can_exec_sim:
        data_choice = data_source_options[default_dataset_idx]

    graph: nx.DiGraph
    data_source_name: str = ""
    is_sap_active: bool = False
    is_sap_fallback: bool = False
    sap_diagnostics: str = ""

    if data_choice == "SAP HANA Cloud (Live Database: HACKFEST0316)":
        try:
            if "sap_cached_data" in st.session_state and not st.session_state.get("force_new_run", False):
                graph, dep_df, nodes_df = st.session_state["sap_cached_data"]
            else:
                with st.spinner("Connecting to SAP HANA Cloud & querying NEXUS tables..."):
                    graph, dep_df, nodes_df = load_sap_supply_chain()
                st.session_state["sap_cached_data"] = (graph, dep_df, nodes_df)
            is_sap_active = True
            is_sap_fallback = False
            data_source_name = "SAP HANA Cloud (Live Database: HACKFEST0316)"
            st.sidebar.success(
                f"SAP HANA Connected: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
            )
            with st.sidebar.expander("SAP Database Master", expanded=False):
                st.caption("Schema: HACKFEST0316")
                st.write(f"**Nodes ({len(nodes_df)}):**")
                st.dataframe(nodes_df, hide_index=True)
                st.write(f"**Dependencies ({len(dep_df)}):**")
                st.dataframe(dep_df, hide_index=True)
        except SAPConfigurationError as cfg_err:
            is_sap_active = False
            is_sap_fallback = True
            sap_diagnostics = sanitize_error_message(str(cfg_err))
            st.sidebar.warning("SAP HANA connection unavailable. The application is currently using the sample dataset.")
            with st.sidebar.expander("View Technical Diagnostics", expanded=False):
                st.caption("Configuration Diagnostics:")
                st.code(sap_diagnostics, language="text")

            data_source_name = "Sample Dependencies CSV (Fallback Active)"
            df = load_and_validate_dependencies(DEFAULT_DEPENDENCIES_CSV)
            graph = build_supply_chain_graph(df)
        except Exception as exc:
            is_sap_active = False
            is_sap_fallback = True
            sap_diagnostics = sanitize_error_message(str(exc))
            st.sidebar.warning("SAP HANA connection unavailable. The application is currently using the sample dataset.")
            with st.sidebar.expander("View Technical Diagnostics", expanded=False):
                st.caption("Connection Diagnostics:")
                st.code(sap_diagnostics, language="text")

            data_source_name = "Sample Dependencies CSV (Fallback Active)"
            df = load_and_validate_dependencies(DEFAULT_DEPENDENCIES_CSV)
            graph = build_supply_chain_graph(df)

    elif data_choice == "Standard Sample (data/dependencies.csv)":
        data_source_name = "Standard Sample (data/dependencies.csv)"
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
            disabled=perms["simulation_controls_disabled"],
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
    st.sidebar.markdown('<div class="sidebar-section-badge"><span>⚡</span> 2. DISRUPTION SCENARIO</div>', unsafe_allow_html=True)

    available_nodes = sorted(list(graph.nodes()))
    default_node_idx = available_nodes.index("Processor X") if "Processor X" in available_nodes else 0
    selected_failed_node = st.sidebar.selectbox(
        "Select Failed Entity:",
        options=available_nodes,
        index=default_node_idx,
        disabled=is_viewer,
    )

    disruption_type_options = ["Auto-infer from Entity Type"] + list(SUPPORTED_DISRUPTION_TYPES.values())
    selected_disruption_type = st.sidebar.selectbox(
        "Disruption Category:",
        options=disruption_type_options,
        index=0,
        disabled=is_viewer,
    )
    passed_disruption_type = (
        None if selected_disruption_type == "Auto-infer from Entity Type" else selected_disruption_type
    )

    # 4. Sidebar: Financial Optimization Parameters
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-section-badge"><span>💰</span> 3. FINANCIAL CONSTRAINTS</div>', unsafe_allow_html=True)

    unit_loss_input = st.sidebar.number_input(
        "Loss Per Disrupted Unit ($):",
        min_value=1.0,
        max_value=10000.0,
        value=150.0,
        step=10.0,
        disabled=is_viewer,
    )

    budget_input = st.sidebar.number_input(
        "Emergency Recovery Budget ($):",
        min_value=1000.0,
        max_value=1000000.0,
        value=50000.0,
        step=5000.0,
        disabled=is_viewer,
    )

    max_lead_time_input = st.sidebar.slider(
        "Max Acceptable Lead Time (Days):",
        min_value=1,
        max_value=60,
        value=14,
        disabled=is_viewer,
    )

    allow_multi_option = st.sidebar.checkbox(
        "Enable Multi-Supplier Combination Optimization",
        value=True,
        disabled=is_viewer,
    )

    # Backend enforcement: VIEWER role cannot alter simulation scenario or financial parameters
    if not can_exec_sim:
        selected_failed_node = available_nodes[default_node_idx]
        passed_disruption_type = None
        unit_loss_input = 150.0
        budget_input = 50000.0
        max_lead_time_input = 14
        allow_multi_option = True
        st.session_state["force_new_run"] = False

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-section-badge"><span>📜</span> 4. GOVERNANCE & AUDIT</div>', unsafe_allow_html=True)
    if st.sidebar.button(
        "🔁 Re-run Simulation (New Audit Record)",
        key="btn_rerun_simulation",
        disabled=perms["rerun_button_disabled"],
        help="Disabled for read-only VIEWER accounts." if is_viewer else "Starts an intentional new simulation run with identical parameters, minting a distinct Run ID and audit record."
    ):
        live_sim_perms = get_ui_control_permissions(auth_manager, curr_u_name, st.session_state)
        if not live_sim_perms["is_active"] or not auth_manager.can_execute_simulation(live_sim_perms["role"]):
            auth_manager._record_audit_event(
                event_type="AUTH_UNAUTHORIZED_SIMULATION_ATTEMPT",
                actor_username=curr_u_name or "ANONYMOUS",
                target_username="DISRUPTION_SIMULATOR",
                details={"role": live_sim_perms["role"], "reason": "VIEWER role is read-only"},
            )
            st.sidebar.error(
                f"Access denied: User '{curr_u_name}' with role '{live_sim_perms['role']}' has read-only access and cannot execute or re-run simulations."
            )
            st.session_state["force_new_run"] = False
        else:
            st.session_state["force_new_run"] = True
            st.session_state["audit_justification_field"] = ""

    if is_viewer:
        st.sidebar.caption("🔒 Re-run Simulation is disabled for VIEWER accounts (read-only access).")

    # =========================================================================
    # Five-Agent Pipeline Orchestration & Enterprise Masthead
    # =========================================================================

    sim_fingerprint = (
        f"{data_choice}|{selected_failed_node}|{passed_disruption_type}|"
        f"{unit_loss_input:.1f}|{budget_input:.1f}|{max_lead_time_input}|{allow_multi_option}"
    )
    force_new = st.session_state.get("force_new_run", False)

    # Backend enforcement: Explicitly separate read-only VIEWER baseline hydration from authorized simulation execution
    if not can_exec_sim:
        if force_new:
            auth_manager._record_audit_event(
                event_type="AUTH_UNAUTHORIZED_SIMULATION_ATTEMPT",
                actor_username=curr_u_name or "ANONYMOUS",
                target_username="DISRUPTION_SIMULATOR",
                details={"role": curr_u_role, "reason": "VIEWER role is read-only"},
            )
            st.sidebar.error(
                f"Access denied: User '{curr_u_name}' with role '{curr_u_role}' has read-only access and cannot execute or re-run simulations."
            )
            st.session_state["force_new_run"] = False
            force_new = False
        # Viewer never executes a simulation; only hydrates read-only baseline if session cache is empty
        should_execute_simulation = False
        should_hydrate_viewer_baseline = ("cached_sim_result" not in st.session_state)
    else:
        # If switching from a read-only viewer session to an authorized Reviewer/Admin session, allow fresh run
        was_read_only_run = (
            str(st.session_state.get("active_run_id", "")).startswith("READ_ONLY")
            or bool(st.session_state.get("is_read_only_baseline", False))
        )
        should_execute_simulation = (
            force_new
            or was_read_only_run
            or (st.session_state.get("active_sim_fingerprint") != sim_fingerprint)
        )
        should_hydrate_viewer_baseline = False

    # 1. Masthead Header & Enterprise Telemetry
    if is_sap_active:
        sap_badge_html = (
            '<div class="telemetry-pill telemetry-live">'
            '<span class="telemetry-pulse"></span>'
            '<span>SAP HANA CLOUD &bull; LIVE CONNECTION</span>'
            '</div>'
        )
    elif is_sap_fallback:
        sap_badge_html = (
            '<div class="telemetry-pill telemetry-fallback">'
            '<span>⚠️ SAP CONNECTION UNAVAILABLE &bull; SAMPLE DATA ACTIVE</span>'
            '</div>'
        )
    else:
        sap_badge_html = (
            '<div class="telemetry-pill telemetry-local">'
            '<span>● LOCAL DATA SOURCE ACTIVE</span>'
            '</div>'
        )

    def render_control_masthead(corridor_type_label: str) -> str:
        return f"""<div class="control-tower-header">
<div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
<div>
<div style="display:inline-flex; align-items:center; gap:8px; margin-bottom:4px;">
<span style="font-size:0.70rem; font-weight:700; color:#06B6D4; text-transform:uppercase; letter-spacing:0.12em; background:rgba(6,182,212,0.12); padding:3px 10px; border-radius:12px; border:1px solid rgba(6,182,212,0.25);">
ENTERPRISE RESILIENCE CONTROL TOWER
</span>
</div>
<h1 class="masthead-title">Nexus-AI</h1>
<div class="masthead-tagline">Autonomous Multi-Tier Supply Chain Intelligence & Shock Recovery</div>
<div style="font-size:0.78rem; color:#94A3B8; margin-top:2px; max-width:640px; line-height:1.4;">
Continuous dependency mapping, automated single-point-of-failure detection, and deterministic disruption stress-testing.
</div>
</div>
<div style="display:flex; flex-direction:column; align-items:flex-end; gap:8px;">
{sap_badge_html}
<div style="font-size:0.72rem; color:#64748B; font-family:monospace; letter-spacing:0.04em;">
FIVE RESILIENCE AGENTS SYNCHRONIZED
</div>
</div>
</div>
<div class="masthead-meta-row" style="margin-top:12px; padding-top:10px; border-top:1px solid rgba(255,255,255,0.06);">
<span style="color:#94A3B8; font-size:0.80rem;">Active Source:</span>
<span style="color:#F8FAFC; font-weight:600; font-size:0.82rem;">{data_source_name}</span>
<span style="color:#475569;">&bull;</span>
<span style="color:#94A3B8; font-size:0.80rem;">Simulated Shock Target:</span>
<span class="badge-critical">{selected_failed_node}</span>
<span style="color:#475569;">&bull;</span>
<span style="color:#94A3B8; font-size:0.80rem;">Corridor Type:</span>
<span style="color:#38BDF8; font-weight:600; font-size:0.82rem;">{corridor_type_label}</span>
<span style="color:#475569;">&bull;</span>
<span style="color:#94A3B8; font-size:0.80rem;">Monitored Assets:</span>
<span style="color:#F8FAFC; font-weight:600; font-size:0.82rem;">{graph.number_of_nodes()} Nodes ({graph.number_of_edges()} Corridors)</span>
</div>
</div>"""

    masthead_placeholder = st.empty()
    workflow_placeholder = st.empty()

    initial_corridor_label = passed_disruption_type or "Operational Outage"
    masthead_placeholder.markdown(render_control_masthead(initial_corridor_label), unsafe_allow_html=True)

    # 2. Sequential Agent Execution (Authorized Live Execution vs Read-Only Baseline Hydration vs Cached Recovery)
    if should_execute_simulation:
        live_sim_perms = get_ui_control_permissions(auth_manager, curr_u_name, st.session_state)
        if not live_sim_perms["is_active"] or not auth_manager.can_execute_simulation(live_sim_perms["role"]):
            auth_manager._record_audit_event(
                event_type="AUTH_UNAUTHORIZED_SIMULATION_ATTEMPT",
                actor_username=curr_u_name or "ANONYMOUS",
                target_username="DISRUPTION_SIMULATOR",
                details={"role": live_sim_perms["role"], "reason": "VIEWER role is read-only"},
            )
            st.error(
                f"Access denied: User '{curr_u_name}' with role '{live_sim_perms['role']}' has read-only access and cannot execute or re-run simulations."
            )
            hydrated = hydrate_viewer_baseline_session_state(
                graph=graph,
                audit_logger=audit_logger,
                session_state=st.session_state,
                baseline_fingerprint=sim_fingerprint,
            )
            agents_status = hydrated["agents_status"]
            total_pipeline_ms = hydrated["total_pipeline_ms"]
            active_run_id = hydrated["active_run_id"]
            risk_findings = hydrated["risk_findings"]
            shared_only_findings = hydrated["shared_only_findings"]
            sim_result = hydrated["sim_result"]
            financial_scenario = hydrated["financial_scenario"]
            candidate_options = hydrated["candidate_options"]
            financial_result = hydrated["financial_result"]
            decision_exp = hydrated["decision_exp"]
        else:
            agents_status = [
                {"id": "mapping", "name": "Mapping Agent", "state": "WAITING", "duration_ms": 0.0, "summary": ""},
                {"id": "risk", "name": "Risk Detection Agent", "state": "WAITING", "duration_ms": 0.0, "summary": ""},
                {"id": "simulation", "name": "Disruption Simulation Agent", "state": "WAITING", "duration_ms": 0.0, "summary": ""},
                {"id": "optimization", "name": "Financial Optimization Agent", "state": "WAITING", "duration_ms": 0.0, "summary": ""},
                {"id": "explanation", "name": "Decision Explanation Agent", "state": "WAITING", "duration_ms": 0.0, "summary": ""},
            ]

            # ── Step 1: Mapping Agent (Blue) ──
            agents_status[0]["state"] = "PROCESSING"
            workflow_placeholder.markdown(
                render_five_agent_workflow_panel(agents_status, is_live_executing=True, current_agent_idx=0),
                unsafe_allow_html=True,
            )
            t0 = time.perf_counter()
            node_count = graph.number_of_nodes()
            edge_count = graph.number_of_edges()
            t1 = time.perf_counter()
            mapping_ms = max((t1 - t0) * 1000.0, 8.5)
            agents_status[0]["state"] = "COMPLETED"
            agents_status[0]["duration_ms"] = mapping_ms
            agents_status[0]["summary"] = f"{node_count} network nodes and {edge_count} dependency corridors mapped."
            workflow_placeholder.markdown(
                render_five_agent_workflow_panel(agents_status, is_live_executing=True, current_agent_idx=0),
                unsafe_allow_html=True,
            )
            time.sleep(0.12)

            # ── Step 2: Risk Detection Agent (Cyan) ──
            agents_status[1]["state"] = "PROCESSING"
            workflow_placeholder.markdown(
                render_five_agent_workflow_panel(agents_status, is_live_executing=True, current_agent_idx=1),
                unsafe_allow_html=True,
            )
            t0 = time.perf_counter()
            try:
                risk_findings = detect_shared_risks(graph, only_shared=False)
                shared_only_findings = [r for r in risk_findings if r.get("is_shared")]
                t1 = time.perf_counter()
                risk_ms = max((t1 - t0) * 1000.0, 6.2)
                agents_status[1]["state"] = "COMPLETED"
                agents_status[1]["duration_ms"] = risk_ms
                agents_status[1]["summary"] = f"{len(risk_findings)} risk findings identified, including {len(shared_only_findings)} shared bottlenecks."
            except Exception as exc:
                agents_status[1]["state"] = "ERROR"
                agents_status[1]["error"] = str(exc)
                risk_findings, shared_only_findings = [], []
                workflow_placeholder.markdown(render_five_agent_workflow_panel(agents_status), unsafe_allow_html=True)
                st.error(f"Risk Detection Agent failed: {exc}")
                return
            workflow_placeholder.markdown(
                render_five_agent_workflow_panel(agents_status, is_live_executing=True, current_agent_idx=1),
                unsafe_allow_html=True,
            )
            time.sleep(0.12)

            # ── Step 3: Disruption Simulation Agent (Amber) ──
            agents_status[2]["state"] = "PROCESSING"
            workflow_placeholder.markdown(
                render_five_agent_workflow_panel(agents_status, is_live_executing=True, current_agent_idx=2),
                unsafe_allow_html=True,
            )
            t0 = time.perf_counter()
            try:
                sim_result = simulate_disruption(
                    graph,
                    failed_node=selected_failed_node,
                    disruption_type=passed_disruption_type,
                    actor_role=live_sim_perms["role"],
                )
                t1 = time.perf_counter()
                sim_ms = max((t1 - t0) * 1000.0, 5.4)
                agents_status[2]["state"] = "COMPLETED"
                agents_status[2]["duration_ms"] = sim_ms
                affected_count = len(sim_result.get("affected_entities", []))
                agents_status[2]["summary"] = f"Shock: {selected_failed_node} • {affected_count} Affected"
                actual_corridor_label = sim_result.get("disruption_type", "Operational Outage")
                masthead_placeholder.markdown(render_control_masthead(actual_corridor_label), unsafe_allow_html=True)
            except Exception as exc:
                agents_status[2]["state"] = "ERROR"
                agents_status[2]["error"] = str(exc)
                workflow_placeholder.markdown(render_five_agent_workflow_panel(agents_status), unsafe_allow_html=True)
                st.error(f"Disruption Simulation Agent failed: {exc}")
                return
            workflow_placeholder.markdown(
                render_five_agent_workflow_panel(agents_status, is_live_executing=True, current_agent_idx=2),
                unsafe_allow_html=True,
            )
            time.sleep(0.12)

            # ── Step 4: Financial Optimization Agent (Green) ──
            agents_status[3]["state"] = "PROCESSING"
            workflow_placeholder.markdown(
                render_five_agent_workflow_panel(agents_status, is_live_executing=True, current_agent_idx=3),
                unsafe_allow_html=True,
            )
            t0 = time.perf_counter()
            try:
                financial_scenario = build_financial_scenario_from_simulation(
                    simulation_result=sim_result,
                    loss_per_unit=unit_loss_input,
                    available_budget=budget_input,
                    max_lead_time_days=max_lead_time_input,
                    graph=graph,
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
                t1 = time.perf_counter()
                fin_ms = max((t1 - t0) * 1000.0, 12.0)
                agents_status[3]["state"] = "COMPLETED"
                agents_status[3]["duration_ms"] = fin_ms
                rec_cost_val = financial_result.get("recovery_cost", 0.0)
                if financial_result.get("is_unquantified"):
                    fin_summary = f"{len(candidate_options)} Options • Unquantified"
                else:
                    fin_summary = f"{len(candidate_options)} Options • Cost: ${rec_cost_val:,.0f}"
                agents_status[3]["summary"] = fin_summary
            except Exception as exc:
                agents_status[3]["state"] = "ERROR"
                agents_status[3]["error"] = str(exc)
                workflow_placeholder.markdown(render_five_agent_workflow_panel(agents_status), unsafe_allow_html=True)
                st.error(f"Financial Optimization Agent failed: {exc}")
                return
            workflow_placeholder.markdown(
                render_five_agent_workflow_panel(agents_status, is_live_executing=True, current_agent_idx=3),
                unsafe_allow_html=True,
            )
            time.sleep(0.12)

            # ── Step 5: Decision Explanation Agent (Purple) ──
            agents_status[4]["state"] = "PROCESSING"
            workflow_placeholder.markdown(
                render_five_agent_workflow_panel(agents_status, is_live_executing=True, current_agent_idx=4),
                unsafe_allow_html=True,
            )
            t0 = time.perf_counter()
            try:
                explanation_agent = DecisionExplanationAgent()
                decision_exp = explanation_agent.explain(
                    graph=graph,
                    disruption_result=sim_result,
                    financial_result=financial_result,
                    risk_findings=risk_findings,
                )
                t1 = time.perf_counter()
                exp_ms = max((t1 - t0) * 1000.0, 7.8)
                agents_status[4]["state"] = "COMPLETED"
                agents_status[4]["duration_ms"] = exp_ms
                selected_ids = decision_exp.get("selected_recovery_option", {}).get("selected_ids", [])
                strat_name = ", ".join(selected_ids) if selected_ids else "Optimal Plan"
                agents_status[4]["summary"] = f"Plan: {strat_name} • Ready"
            except Exception as exc:
                agents_status[4]["state"] = "ERROR"
                agents_status[4]["error"] = str(exc)
                workflow_placeholder.markdown(render_five_agent_workflow_panel(agents_status), unsafe_allow_html=True)
                st.error(f"Decision Explanation Agent failed: {exc}")
                return
            workflow_placeholder.markdown(
                render_five_agent_workflow_panel(agents_status, is_live_executing=True, current_agent_idx=4),
                unsafe_allow_html=True,
            )
            time.sleep(0.08)

            total_pipeline_ms = sum(a.get("duration_ms", 0.0) for a in agents_status)

            # Audit Logging (Write operations strictly restricted to ADMIN and REVIEWER)
            audit_logger = AuditLogger()
            if force_new:
                st.session_state["audit_justification_field"] = ""

            active_run_id = audit_logger.record_simulation_run(
                scenario_entity=selected_failed_node,
                disruption_type=sim_result.get("disruption_type", "Operational Outage"),
                scenario_params={
                    "failed_node": selected_failed_node,
                    "disruption_type": passed_disruption_type,
                    "unit_loss": unit_loss_input,
                    "budget": budget_input,
                    "max_lead_time": max_lead_time_input,
                    "allow_multi_option": allow_multi_option,
                },
                sim_result=sim_result,
                financial_result=financial_result,
                decision_exp=decision_exp,
                execution_channel="STREAMLIT",
                actor_role=live_sim_perms["role"],
            )
            st.session_state["active_sim_fingerprint"] = sim_fingerprint
            st.session_state["active_run_id"] = active_run_id
            st.session_state["force_new_run"] = False
            st.session_state["is_read_only_baseline"] = False

            # Persist session state caches
            st.session_state["workflow_agent_states"] = agents_status
            st.session_state["workflow_total_time_ms"] = total_pipeline_ms
            st.session_state["cached_graph"] = graph
            st.session_state["cached_risk_findings"] = risk_findings
            st.session_state["cached_shared_only_findings"] = shared_only_findings
            st.session_state["cached_sim_result"] = sim_result
            st.session_state["cached_financial_result"] = financial_result
            st.session_state["cached_financial_scenario"] = financial_scenario
            st.session_state["cached_candidate_options"] = candidate_options
            st.session_state["cached_decision_exp"] = decision_exp

        workflow_placeholder.empty()

    elif should_hydrate_viewer_baseline:
        hydrated = hydrate_viewer_baseline_session_state(
            graph=graph,
            audit_logger=audit_logger,
            session_state=st.session_state,
            baseline_fingerprint=sim_fingerprint,
        )
        agents_status = hydrated["agents_status"]
        total_pipeline_ms = hydrated["total_pipeline_ms"]
        active_run_id = hydrated["active_run_id"]
        risk_findings = hydrated["risk_findings"]
        shared_only_findings = hydrated["shared_only_findings"]
        sim_result = hydrated["sim_result"]
        financial_scenario = hydrated["financial_scenario"]
        candidate_options = hydrated["candidate_options"]
        financial_result = hydrated["financial_result"]
        decision_exp = hydrated["decision_exp"]
        actual_corridor_label = sim_result.get("disruption_type", "Operational Outage")
        masthead_placeholder.markdown(render_control_masthead(actual_corridor_label), unsafe_allow_html=True)

    else:
        # Load from session cache for instantaneous tab navigation (0ms delay)
        agents_status = st.session_state.get("workflow_agent_states", [])
        total_pipeline_ms = st.session_state.get("workflow_total_time_ms", 0.0)
        active_run_id = st.session_state.get("active_run_id", "ACTIVE-RUN")
        graph = st.session_state["cached_graph"]
        risk_findings = st.session_state["cached_risk_findings"]
        shared_only_findings = st.session_state["cached_shared_only_findings"]
        sim_result = st.session_state["cached_sim_result"]
        financial_result = st.session_state["cached_financial_result"]
        financial_scenario = st.session_state.get("cached_financial_scenario")
        candidate_options = st.session_state.get("cached_candidate_options", [])
        decision_exp = st.session_state["cached_decision_exp"]

        actual_corridor_label = sim_result.get("disruption_type", "Operational Outage")
        masthead_placeholder.markdown(render_control_masthead(actual_corridor_label), unsafe_allow_html=True)
        workflow_placeholder.empty()

    # Compute key executive figures safely
    total_entities = graph.number_of_nodes()
    total_suppliers = len([n for n, d in graph.nodes(data=True) if "supp" in str(d.get("node_type", "")).lower() or d.get("is_supplier")])
    total_factories = len([n for n, d in graph.nodes(data=True) if "fact" in str(d.get("node_type", "")).lower() or "plant" in str(d.get("node_type", "")).lower() or "proc" in str(d.get("node_type", "")).lower()])
    shared_deps_count = len(shared_only_findings)
    affected_count = len(sim_result.get("affected_entities", []))
    impact_level = sim_result.get("impact_level", "Unknown")
    is_unquantified = bool(financial_result.get("is_unquantified", False) or financial_result.get("baseline_disruption_loss") is None)
    gross_loss = financial_result.get("baseline_disruption_loss")
    rec_cost = financial_result.get("recovery_cost")
    est_savings = financial_result.get("estimated_savings")
    residual_loss = financial_result.get("residual_disruption_loss")

    gross_loss_disp = "UNQUANTIFIED" if is_unquantified else safe_format_currency(gross_loss, default="Not available", decimals=0)
    rec_cost_disp = safe_format_currency(rec_cost, default="$0", decimals=0)
    est_savings_disp = "UNQUANTIFIED" if is_unquantified else safe_format_currency(est_savings, default="Not available", decimals=0)
    residual_loss_disp = "UNQUANTIFIED" if is_unquantified else safe_format_currency(residual_loss, default="Not available", decimals=0)

    # Helper function for rendering KPI cards
    def render_kpi_card(
        label: str,
        value: str,
        icon: str,
        glow_class: str,
        subtitle: str = "",
        value_color: str = "#F8FAFC",
    ) -> str:
        return (
            f'<div class="kpi-card {glow_class}">'
            f'<div class="kpi-card-header">'
            f'<span class="kpi-card-label">{label}</span>'
            f'<span class="kpi-card-icon">{icon}</span>'
            f'</div>'
            f'<div class="kpi-card-value" style="color:{value_color};">{value}</div>'
            f'<div class="kpi-card-subtitle">{subtitle}</div>'
            f'</div>'
        )

    if is_sap_fallback:
        st.warning(
            "⚠️ **SAP HANA connection unavailable. The application is currently using the sample dataset.**\n\n"
            "Active network topology and component parameters have defaulted to the verified local benchmark (`data/dependencies.csv`). "
            "All disruption shocks, network blast radius calculations, and financial optimizations reflect sample data."
        )

    # =========================================================================
    # SECTION 1: EXECUTIVE DASHBOARD
    # =========================================================================
    if "1. 📊" in selected_nav:
        # Step 1: Page heading and short explanation
        st.markdown(
            """
            <div class="section-header-banner">
              <div>
                <div class="section-header-title">📊 Executive Dashboard</div>
                <div class="section-header-desc">
                  Real-time operational situational awareness, network baseline health, and active disruption posture across all monitored supply tiers.
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Step 2: Important KPI cards (Equal-width 4-column CSS grid on desktop with complete descriptions)
        exec_kpi_html = (
            '<div class="kpi-grid">'
            + render_kpi_card(
                "Monitored Assets",
                f"{total_entities}",
                "🌐",
                "kpi-card-glow-cyan",
                "Number of active nodes monitored across the supply network.",
                "#38BDF8",
            )
            + render_kpi_card(
                "Critical Bottlenecks",
                f"{shared_deps_count}",
                "🛡️",
                "kpi-card-glow-amber",
                "Shared single points of failure feeding multiple downstream partners.",
                "#FBBF24",
            )
            + render_kpi_card(
                "Inaction Gross Loss",
                gross_loss_disp,
                "💥",
                "kpi-card-glow-rose",
                "Projected financial loss if the disruption remains unmitigated.",
                "#FB7185",
            )
            + render_kpi_card(
                "Net Preserved Capital",
                est_savings_disp,
                "💰",
                "kpi-card-glow-emerald",
                "Capital preserved through AI-recommended recovery actions.",
                "#34D399",
            )
            + '</div>'
        )
        st.markdown(exec_kpi_html, unsafe_allow_html=True)
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Step 3: Active alerts and disruption summary
        sev_label = impact_level.upper()
        approval_req = decision_exp["requires_human_approval"]
        gov_label = "MANDATORY HUMAN REVIEW" if approval_req else "AUTONOMOUS BOUNDS"
        gov_class = "badge-critical" if approval_req else "badge-low"
        sev_class = {
            "Critical": "badge-critical",
            "High": "badge-high",
            "Medium": "badge-medium",
            "Low": "badge-low",
        }.get(impact_level, "badge-medium")

        rec_strategy_str = ', '.join([translate_option_id(o) for o in financial_result.get('recommended_option_ids', [])]) or 'Baseline Hold (No Action)'

        st.markdown(
            f"""
            <div style="background:#151F32; border:1px solid #1E293B; border-radius:10px; padding:18px 20px; box-shadow:0 4px 16px rgba(0,0,0,0.35); margin-bottom:14px;">
              <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px; margin-bottom:12px;">
                <div>
                  <div style="font-size:0.72rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; font-weight:700;">Active Disruption Simulation Loaded</div>
                  <div style="font-size:1.25rem; font-weight:800; color:#F8FAFC; margin-top:2px;">Target Entity: {selected_failed_node}</div>
                </div>
                <div style="display:flex; gap:8px; align-items:center;">
                  <span class="{sev_class}">{sev_label} SEVERITY</span>
                  <span class="{gov_class}">{gov_label}</span>
                </div>
              </div>
              <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px; background:#0D1527; padding:12px 14px; border-radius:8px; border:1px solid #1E293B;">
                <div>
                  <div style="font-size:0.70rem; color:#94A3B8; text-transform:uppercase;">Disruption Blast Radius</div>
                  <div style="font-size:1.15rem; font-weight:800; color:#FB7185;">{affected_count} Entities Impacted</div>
                </div>
                <div>
                  <div style="font-size:0.70rem; color:#94A3B8; text-transform:uppercase;">Estimated Inaction Loss</div>
                  <div style="font-size:1.15rem; font-weight:800; color:#F8FAFC;">{gross_loss_disp}</div>
                </div>
                <div>
                  <div style="font-size:0.70rem; color:#94A3B8; text-transform:uppercase;">Recommended Recovery</div>
                  <div style="font-size:1.15rem; font-weight:800; color:#38BDF8;">{rec_strategy_str}</div>
                </div>
                <div>
                  <div style="font-size:0.70rem; color:#94A3B8; text-transform:uppercase;">Net Capital Preserved</div>
                  <div style="font-size:1.15rem; font-weight:800; color:#34D399;">{est_savings_disp}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Step 4: Five-agent execution pipeline
        st.markdown(
            render_five_agent_workflow_panel(
                agents_status,
                total_pipeline_ms=total_pipeline_ms,
                active_run_id=active_run_id,
                is_live_executing=False,
            ),
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Step 5: Additional operational details (expandable deep-dive) (expandable deep-dive)
        with st.expander("📋 Additional Operational & Network Topology Details", expanded=False):
            st.markdown(
                f"""
                <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin-bottom:12px;">
                  <div style="background:#0D1527; padding:10px 14px; border-radius:8px; border:1px solid #1E293B;">
                    <div style="font-size:0.70rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Direct Suppliers</div>
                    <div style="font-size:1.2rem; font-weight:800; color:#2DD4BF;">{total_suppliers}</div>
                    <div style="font-size:0.68rem; color:#64748B;">Tier-1 & Tier-2 partner sources</div>
                  </div>
                  <div style="background:#0D1527; padding:10px 14px; border-radius:8px; border:1px solid #1E293B;">
                    <div style="font-size:0.70rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Processing Plants</div>
                    <div style="font-size:1.2rem; font-weight:800; color:#60A5FA;">{total_factories}</div>
                    <div style="font-size:0.68rem; color:#64748B;">Intermediate transformation hubs</div>
                  </div>
                  <div style="background:#0D1527; padding:10px 14px; border-radius:8px; border:1px solid #1E293B;">
                    <div style="font-size:0.70rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Directed Corridors</div>
                    <div style="font-size:1.2rem; font-weight:800; color:#FBBF24;">{graph.number_of_edges()}</div>
                    <div style="font-size:0.68rem; color:#64748B;">Material dependency flows</div>
                  </div>
                  <div style="background:#0D1527; padding:10px 14px; border-radius:8px; border:1px solid #1E293B;">
                    <div style="font-size:0.70rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Data Provenance</div>
                    <div style="font-size:0.95rem; font-weight:700; color:#F8FAFC; margin-top:2px;">{data_source_name.split()[0]}</div>
                    <div style="font-size:0.68rem; color:#64748B;">Active verification tier</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Active Financial Constraints: Unit Loss: $" + f"{unit_loss_input:,.0f}" + " | Emergency Budget: $" + f"{budget_input:,.0f}" + f" | Max Lead Time: {max_lead_time_input} days")

    # =========================================================================
    # SECTION 2: SUPPLY CHAIN MAP
    # =========================================================================
    elif "2. 🗺️" in selected_nav:
        # ROW 1: Chart Heading on the left
        st.markdown(
            """
            <div class="section-header-banner">
              <div>
                <div class="section-header-title">🗺️ Nexus-AI: Multi-Tier Supply Chain Resilience Map</div>
                <div class="section-header-desc">
                  Interactive directed dependency topology with live material flow and disruption telemetries.
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_map, col_inspect = st.columns([10, 5], gap="medium")

        with col_inspect:
            st.markdown('<div class="sidebar-section-badge"><span>🔍</span> ASSET INSPECTOR</div>', unsafe_allow_html=True)
            inspect_node = st.selectbox(
                "Select entity to inspect:",
                options=available_nodes,
                index=default_node_idx,
                key="inspect_select",
            )
            if inspect_node in graph:
                node_data = graph.nodes[inspect_node]
                node_type = str(node_data.get("node_type", "General Entity"))
                in_deg = graph.in_degree(inspect_node)
                out_deg = graph.out_degree(inspect_node)
                preds = list(graph.predecessors(inspect_node))
                succs = list(graph.successors(inspect_node))
                alts = node_data.get("alternatives", [])
                status_key, status_label = classify_node_status(inspect_node, sim_result)

                status_badge_class = {
                    "failed": "badge-critical",
                    "confirmed_shortage": "badge-critical",
                    "potentially_affected": "badge-high",
                }.get(status_key, "badge-low")

                tier_color = "#06B6D4" if "raw" in node_type.lower() else ("#38BDF8" if "proc" in node_type.lower() else "#10B981")

                preds_html = "".join([f'<span style="background:rgba(56,189,248,0.12); color:#38BDF8; border:1px solid rgba(56,189,248,0.25); padding:2px 8px; border-radius:10px; font-size:0.75rem; margin-right:4px; display:inline-block; margin-bottom:4px;">{p}</span>' for p in preds]) if preds else '<span style="color:#64748B; font-size:0.8rem;">Root Inflow (None)</span>'
                succs_html = "".join([f'<span style="background:rgba(16,185,129,0.12); color:#34D399; border:1px solid rgba(16,185,129,0.25); padding:2px 8px; border-radius:10px; font-size:0.75rem; margin-right:4px; display:inline-block; margin-bottom:4px;">{s}</span>' for s in succs]) if succs else '<span style="color:#64748B; font-size:0.8rem;">Terminal Node (None)</span>'
                alts_html = "".join([f'<span style="background:rgba(245,158,11,0.12); color:#FBBF24; border:1px solid rgba(245,158,11,0.25); padding:2px 8px; border-radius:10px; font-size:0.75rem; margin-right:4px; display:inline-block; margin-bottom:4px;">{a}</span>' for a in alts]) if alts else '<span style="color:#F43F5E; font-size:0.76rem; font-weight:600;">⚠️ Zero backup alternatives (SPOF)</span>'

                st.markdown(
                    f"""
                    <div style="background:#151F32; border:1px solid #1E293B; border-radius:10px; padding:16px; margin-top:10px; box-shadow:0 4px 16px rgba(0,0,0,0.35);">
                      <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
                        <div>
                          <div style="font-size:0.70rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; font-weight:700;">Selected Asset</div>
                          <div style="font-size:1.15rem; font-weight:800; color:#F8FAFC;">{inspect_node}</div>
                        </div>
                        <span class="{status_badge_class}">{status_key.upper()}</span>
                      </div>
                      <div style="display:inline-block; font-size:0.74rem; font-weight:700; color:{tier_color}; background:rgba(255,255,255,0.05); padding:2px 8px; border-radius:6px; margin-bottom:12px;">
                        ROLE: {node_type.upper()}
                      </div>
                      <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:12px;">
                        <div style="background:#0D1527; padding:8px 10px; border-radius:6px; border:1px solid #1E293B;">
                          <div style="font-size:0.68rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Inbound Feeds</div>
                          <div style="font-size:1.15rem; font-weight:800; color:#38BDF8;">{in_deg}</div>
                        </div>
                        <div style="background:#0D1527; padding:8px 10px; border-radius:6px; border:1px solid #1E293B;">
                          <div style="font-size:0.68rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Outbound Feeds</div>
                          <div style="font-size:1.15rem; font-weight:800; color:#2DD4BF;">{out_deg}</div>
                        </div>
                      </div>
                      <div style="margin-bottom:8px;">
                        <div style="font-size:0.70rem; color:#94A3B8; font-weight:700; text-transform:uppercase; margin-bottom:4px;">Upstream Dependencies:</div>
                        <div>{preds_html}</div>
                      </div>
                      <div style="margin-bottom:8px;">
                        <div style="font-size:0.70rem; color:#94A3B8; font-weight:700; text-transform:uppercase; margin-bottom:4px;">Downstream Suppliers:</div>
                        <div>{succs_html}</div>
                      </div>
                      <div>
                        <div style="font-size:0.70rem; color:#94A3B8; font-weight:700; text-transform:uppercase; margin-bottom:4px;">Backup Availability:</div>
                        <div>{alts_html}</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_map:
            # ROW 2: Play Flow, Pause, and Telemetry Controls properly aligned beneath heading
            st.markdown(
                """
                <div class="map-control-bar">
                  <div style="display:flex; align-items:center; gap:10px;">
                    <span class="telemetry-pill telemetry-live" style="font-size:0.75rem; padding:3px 10px;">
                      <span class="telemetry-pulse"></span> LIVE MATERIAL FLOW
                    </span>
                    <span style="font-size:0.80rem; color:#94A3B8;">Continuous material flow & node resonance active</span>
                  </div>
                  <div style="font-size:0.74rem; color:#64748B; font-family:monospace;">
                    CYCLE: 80ms &bull; 16 FRAMES
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            ctrl_c1, ctrl_c2 = st.columns([6, 6], vertical_alignment="center")
            with ctrl_c1:
                st.caption("Play/Pause controls are anchored above the canvas. Moving particles indicate operational routes.")
            with ctrl_c2:
                enable_animations = st.toggle(
                    "Live Material Flow & Pulse Telemetry",
                    value=True,
                    help="Toggles smooth material flow particles along operational corridors and pulsating shockwave auras on disrupted/affected entities.",
                    key="toggle_map_animations",
                )

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

            # ROW 3: The actual network graph (canvas title set to empty string to strictly prevent overlapping heading)
            fig = create_resilient_network_figure(
                graph=graph,
                simulation_result=sim_result,
                selected_entity=inspect_node,
                animate=enable_animations,
                title="",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": "hover", "displaylogo": False})

            if enable_animations:
                import streamlit.components.v1 as _st_components
                continuous_loop_js = """
                <script>
                (function() {
                    var isLooping = true;
                    var retryCount = 0;
                    var maxRetries = 50;

                    function getParentDoc() {
                        try { return window.parent.document; } catch(e) { return null; }
                    }

                    function isReducedMotion() {
                        try {
                            return window.parent.matchMedia('(prefers-reduced-motion: reduce)').matches;
                        } catch(e) { return false; }
                    }

                    if (isReducedMotion()) { return; }

                    function setupContinuousTelemetry() {
                        var pDoc = getParentDoc();
                        if (!pDoc) return;
                        var gd = pDoc.querySelector('.js-plotly-plot');
                        var Plotly = window.parent.Plotly || (window.parent.window && window.parent.window.Plotly);

                        if (!gd || !Plotly || !gd._fullLayout) {
                            if (retryCount++ < maxRetries) {
                                setTimeout(setupContinuousTelemetry, 100);
                            }
                            return;
                        }

                        // Bind directly to Plotly updatemenu Play / Pause / Replay buttons
                        var buttons = gd.querySelectorAll('.updatemenu-button');
                        buttons.forEach(function(btn) {
                            if (btn.dataset.nexusLoopBound) return;
                            btn.dataset.nexusLoopBound = 'true';

                            btn.addEventListener('click', function() {
                                var txt = (btn.textContent || '').toLowerCase();
                                if (txt.indexOf('pause') !== -1) {
                                    isLooping = false;
                                    try { Plotly.animate(gd, [null], {mode: 'immediate'}); } catch(e){}
                                } else if (txt.indexOf('replay') !== -1) {
                                    isLooping = true;
                                    try {
                                        Plotly.animate(gd, ['frame_0'], {fromcurrent: false, mode: 'immediate'}).then(function() {
                                            runContinuousLoop();
                                        });
                                    } catch(e) {
                                        runContinuousLoop();
                                    }
                                } else if (txt.indexOf('play') !== -1) {
                                    isLooping = true;
                                    runContinuousLoop();
                                }
                            });
                        });

                        function runContinuousLoop() {
                            if (!isLooping) return;
                            if (!pDoc.contains(gd)) return; // Stop gracefully if unmounted

                            Plotly.animate(gd, null, {
                                fromcurrent: false,
                                mode: 'immediate',
                                frame: { duration: 80, redraw: false },
                                transition: { duration: 0 }
                            }).then(function() {
                                if (isLooping && pDoc.contains(gd)) {
                                    // Seamlessly restart next cycle immediately
                                    requestAnimationFrame(runContinuousLoop);
                                }
                            }).catch(function() {
                                // Paused or interrupted
                            });
                        }

                        // Auto-start continuous animation loop immediately on load
                        runContinuousLoop();
                    }

                    if (document.readyState === 'loading') {
                        document.addEventListener('DOMContentLoaded', setupContinuousTelemetry);
                    } else {
                        setupContinuousTelemetry();
                    }
                })();
                </script>
                """
                _st_components.html(continuous_loop_js, height=0)

    # =========================================================================
    # SECTION 3: RISK ANALYSIS
    # =========================================================================
    elif "3. 🛡️" in selected_nav:
        st.markdown(
            """
            <div class="section-header-banner">
              <div>
                <div class="section-header-title">🛡️ Upstream Risk & Bottleneck Analysis</div>
                <div class="section-header-desc">Pre-disruption vulnerability audit identifying multi-tier single points of failure (SPOF) and shared choke-points feeding multiple suppliers.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        total_audited = len(risk_findings)
        shared_count = len(shared_only_findings)
        max_risk = max([float(r.get("risk_score", 0.0) or 0.0) for r in risk_findings], default=0.0)
        avg_risk = sum([float(r.get("risk_score", 0.0) or 0.0) for r in risk_findings]) / max(total_audited, 1)

        risk_kpi_html = (
            '<div class="kpi-grid">'
            + render_kpi_card(
                "Audited Assets",
                f"{total_audited}",
                "🔬",
                "kpi-card-glow-sky",
                "Upstream supply chain dependencies evaluated across all network tiers.",
                "#38BDF8",
            )
            + render_kpi_card(
                "Shared Bottlenecks",
                f"{shared_count}",
                "⚠️",
                "kpi-card-glow-rose" if shared_count > 0 else "kpi-card-glow-teal",
                "Shared choke-points feeding multiple independent downstream suppliers.",
                "#FB7185" if shared_count > 0 else "#2DD4BF",
            )
            + render_kpi_card(
                "Peak Risk Score",
                f"{max_risk:.1f}/100",
                "⚡",
                "kpi-card-glow-amber",
                "Highest single vulnerability score detected across all dependency components.",
                "#FBBF24",
            )
            + render_kpi_card(
                "Average Risk",
                f"{avg_risk:.1f}/100",
                "📊",
                "kpi-card-glow-teal",
                "Composite vulnerability benchmark representing overall network risk exposure.",
                "#2DD4BF",
            )
            + '</div>'
        )
        st.markdown(risk_kpi_html, unsafe_allow_html=True)
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        tab_shared, tab_all_risks = st.tabs(["Shared Dependencies Only (SPOF)", "All Dependency Findings"])

        with tab_shared:
            if shared_only_findings:
                for r in shared_only_findings:
                    st.markdown(render_risk_card(r), unsafe_allow_html=True)

                with st.expander("📋 View Raw SPOF Audit Table (Data Ledger)", expanded=False):
                    shared_df_rows = []
                    for r in shared_only_findings:
                        score_str = safe_format_number(r.get("risk_score"), default="Unknown", decimals=1, suffix="/100")
                        shared_df_rows.append({
                            "Dependency Node": r.get("dependency_node"),
                            "Affected Suppliers": ", ".join(r.get("affected_suppliers", [])),
                            "Supplier Count": r.get("affected_supplier_count"),
                            "Dependency Tier": r.get("dependency_tier"),
                            "Risk Score": score_str,
                            "Risk Level": r.get("risk_level"),
                            "Explanation": r.get("explanation"),
                            "Recommended Investigation": r.get("recommended_investigation"),
                        })
                    st.dataframe(pd.DataFrame(shared_df_rows), use_container_width=True)
            else:
                st.info("No shared dependencies affecting ≥2 suppliers found in the active dataset.")

        with tab_all_risks:
            if risk_findings:
                for r in risk_findings:
                    st.markdown(render_risk_card(r), unsafe_allow_html=True)

                with st.expander("📋 View Full Dependency Audit Table (Data Ledger)", expanded=False):
                    all_df_rows = []
                    for r in risk_findings:
                        score_str = safe_format_number(r.get("risk_score"), default="Unknown", decimals=1, suffix="/100")
                        all_df_rows.append({
                            "Dependency Node": r.get("dependency_node"),
                            "Is Shared?": "Yes" if r.get("is_shared") else "No",
                            "Suppliers": ", ".join(r.get("affected_suppliers", [])),
                            "Risk Score": score_str,
                            "Risk Level": r.get("risk_level"),
                            "Missing Info": ", ".join(r.get("missing_information", [])) or "None",
                            "Recommended Action": r.get("recommended_investigation"),
                        })
                    st.dataframe(pd.DataFrame(all_df_rows), use_container_width=True)
            else:
                st.info("No risk records generated.")

    # =========================================================================
    # SECTION 4: DISRUPTION SIMULATOR
    # =========================================================================
    elif "4. ⚡" in selected_nav:
        st.markdown(
            """
            <div class="section-header-banner">
              <div>
                <div class="section-header-title">⚡ Disruption Simulator & Blast Radius</div>
                <div class="section-header-desc">Simulate operational shocks to stress-test resilience, tracing severed paths, affected commodities, and downstream outages.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if is_viewer:
            st.markdown(
                """
                <div style="background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.28); border-radius:8px; padding:12px 16px; margin-bottom:14px; color:#38BDF8; font-size:0.82rem; line-height:1.45;">
                  👁️ <b>READ-ONLY ACCESS (VIEWER ROLE):</b> Disruption Simulator configuration controls and the <b>Re-run Simulation</b> button are disabled for your account. You can view the simulation results, blast-radius propagation, and downstream shortage reports below, but you cannot execute or re-run simulations or create new simulation audit records.
                </div>
                """,
                unsafe_allow_html=True,
            )

        cap_impact = sim_result.get("estimated_capacity_impact", {})
        direct_cap_val = cap_impact.get("estimated_direct_capacity_loss")
        direct_cap_str = safe_format_capacity(direct_cap_val, default="Not available", decimals=1, unit=" units")
        materials = sim_result.get("affected_materials", [])
        paths = sim_result.get("disrupted_paths") or sim_result.get("affected_paths") or []
        corridor_paths = [p for p in paths if len(p) > 1]
        actions = sim_result.get("recommended_actions") or sim_result.get("recommended_next_actions") or []
        entity_details = sim_result.get("entity_impact_details", [])

        # Determine dynamic severity styling based on topological impact level
        sev_color_map = {
            "Critical": "#FB7185",  # Rose/Red
            "High": "#FBBF24",      # Amber
            "Medium": "#38BDF8",    # Cyan/Sky
            "Low": "#34D399",       # Emerald/Green
        }
        sev_glow_map = {
            "Critical": "kpi-card-glow-rose",
            "High": "kpi-card-glow-amber",
            "Medium": "kpi-card-glow-sky",
            "Low": "kpi-card-glow-emerald",
        }
        sev_color = sev_color_map.get(impact_level, "#FBBF24")
        sev_glow = sev_glow_map.get(impact_level, "kpi-card-glow-amber")

        # 4 Outage KPI Cards (Equal-width 4-column CSS grid on desktop with complete descriptions)
        disrupt_kpi_html = (
            '<div class="kpi-grid">'
            + render_kpi_card(
                "Failed Facility",
                f"{selected_failed_node}",
                "💥",
                "kpi-card-glow-rose",
                f"Primary shock origin entity: {sim_result.get('disruption_type', 'Operational Outage')}.",
                "#FB7185",
            )
            + render_kpi_card(
                "Severity Tier",
                f"{impact_level.upper()}",
                "⚡",
                sev_glow,
                "Topological blast radius severity classification across dependency routes.",
                sev_color,
            )
            + render_kpi_card(
                "Direct Capacity Severed",
                direct_cap_str,
                "📉",
                "kpi-card-glow-cyan",
                "Immediate rated production throughput unavailable due to primary shock.",
                "#38BDF8",
            )
            + render_kpi_card(
                "Downstream Entities",
                f"{affected_count} Entities",
                "🏢",
                "kpi-card-glow-teal",
                "Total downstream production and distribution facilities facing disruptions.",
                "#2DD4BF",
            )
            + '</div>'
        )
        st.markdown(disrupt_kpi_html, unsafe_allow_html=True)
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Feature 3: Disruption Propagation Impact Summary Card
        st.markdown(render_disruption_propagation_summary(sim_result, graph), unsafe_allow_html=True)

        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

        # Feature 3: Live Disruption Propagation Animation Canvas
        prop_c1, prop_c2 = st.columns([7, 5], vertical_alignment="center")
        with prop_c1:
            st.markdown(
                """
                <div style="display:flex; align-items:center; gap:8px; padding:2px 0;">
                  <span class="telemetry-pill telemetry-fallback" style="font-size:0.75rem; padding:3px 10px;">
                    <span class="telemetry-pulse"></span> SHOCK PROPAGATION FLOW
                  </span>
                  <span style="font-size:0.80rem; color:#94A3B8;">Multi-hop topological shockwave cascade</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with prop_c2:
            enable_prop_animations = st.toggle(
                "Live Shock Propagation Telemetry",
                value=True,
                help="Animates amber/crimson shock signals along severed corridors and expanding origin radar wave.",
                key="toggle_propagation_animations",
            )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        fig_prop = create_disruption_propagation_figure(
            graph=graph,
            simulation_result=sim_result,
            selected_entity=selected_failed_node,
            title=f"⚡ Live Propagation Flow: {selected_failed_node} Shockwave",
            animate=enable_prop_animations,
        )
        st.plotly_chart(fig_prop, use_container_width=True, config={"displayModeBar": "hover", "displaylogo": False})

        if enable_prop_animations:
            import streamlit.components.v1 as _st_components
            prop_loop_js = """
            <script>
            (function() {
                var isLooping = true;
                var retryCount = 0;
                var maxRetries = 50;

                function getParentDoc() {
                    try { return window.parent.document; } catch(e) { return null; }
                }

                function isReducedMotion() {
                    try {
                        return window.parent.matchMedia('(prefers-reduced-motion: reduce)').matches;
                    } catch(e) { return false; }
                }

                if (isReducedMotion()) { return; }

                function setupPropagationTelemetry() {
                    var pDoc = getParentDoc();
                    if (!pDoc) return;
                    var plots = pDoc.querySelectorAll('.js-plotly-plot');
                    var gd = plots.length > 0 ? plots[plots.length - 1] : null;
                    var Plotly = window.parent.Plotly || (window.parent.window && window.parent.window.Plotly);

                    if (!gd || !Plotly || !gd._fullLayout) {
                        if (retryCount++ < maxRetries) {
                            setTimeout(setupPropagationTelemetry, 100);
                        }
                        return;
                    }

                    var buttons = gd.querySelectorAll('.updatemenu-button');
                    buttons.forEach(function(btn) {
                        if (btn.dataset.nexusPropBound) return;
                        btn.dataset.nexusPropBound = 'true';

                        btn.addEventListener('click', function() {
                            var txt = (btn.textContent || '').toLowerCase();
                            if (txt.indexOf('pause') !== -1) {
                                isLooping = false;
                                try { Plotly.animate(gd, [null], {mode: 'immediate'}); } catch(e){}
                            } else if (txt.indexOf('replay') !== -1) {
                                isLooping = true;
                                try {
                                    Plotly.animate(gd, ['frame_0'], {fromcurrent: false, mode: 'immediate'}).then(function() {
                                        runPropagationLoop();
                                    });
                                } catch(e) {
                                    runPropagationLoop();
                                }
                            } else if (txt.indexOf('play') !== -1) {
                                isLooping = true;
                                runPropagationLoop();
                            }
                        });
                    });

                    function runPropagationLoop() {
                        if (!isLooping) return;
                        if (!pDoc.contains(gd)) return;

                        Plotly.animate(gd, null, {
                            fromcurrent: false,
                            mode: 'immediate',
                            frame: { duration: 80, redraw: false },
                            transition: { duration: 0 }
                        }).then(function() {
                            if (isLooping && pDoc.contains(gd)) {
                                requestAnimationFrame(runPropagationLoop);
                            }
                        }).catch(function() {});
                    }

                    runPropagationLoop();
                }

                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', setupPropagationTelemetry);
                } else {
                    setupPropagationTelemetry();
                }
            })();
            </script>
            """
            _st_components.html(prop_loop_js, height=0)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        d_col1, d_col2 = st.columns([1, 1])
        with d_col1:
            st.markdown('<div class="sidebar-section-badge"><span>🔗</span> SEVERED SUPPLY PATHS & COMMODITIES</div>', unsafe_allow_html=True)
            if materials:
                mat_chips = "".join([f'<span style="background:rgba(56,189,248,0.12); color:#38BDF8; border:1px solid rgba(56,189,248,0.3); padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; margin-right:6px; display:inline-block; margin-bottom:6px;">{m}</span>' for m in materials])
                st.markdown(f'<div style="margin-bottom:12px;"><div style="font-size:0.72rem; color:#94A3B8; font-weight:700; text-transform:uppercase; margin-bottom:6px;">Jeopardized Materials:</div><div>{mat_chips}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-size:0.80rem; color:#64748B; margin-bottom:12px;">No specific commodities recorded on severed corridors.</div>', unsafe_allow_html=True)

            if corridor_paths:
                st.markdown('<div style="font-size:0.72rem; color:#94A3B8; font-weight:700; text-transform:uppercase; margin-bottom:6px;">Severed Dependency Corridors:</div>', unsafe_allow_html=True)
                for cp in corridor_paths[:5]:
                    route_str = " &nbsp;&rarr;&nbsp; ".join([f"<b>{node}</b>" for node in cp])
                    st.markdown(f'<div style="background:#151F32; border:1px solid #1E293B; border-left:3px solid #F43F5E; border-radius:6px; padding:6px 12px; margin-bottom:6px; font-size:0.80rem; color:#CBD5E1;">{route_str}</div>', unsafe_allow_html=True)
                if len(corridor_paths) > 5:
                    st.caption(f"... plus {len(corridor_paths) - 5} additional severed corridors.")
            else:
                st.info("No downstream supply paths severed (disruption is localized to terminal node).")

        with d_col2:
            st.markdown('<div class="sidebar-section-badge"><span>📋</span> RECOMMENDED RESILIENCE DIRECTIVES</div>', unsafe_allow_html=True)
            if actions:
                for idx, act in enumerate(actions, start=1):
                    st.markdown(
                        f"""<div style="background:#151F32; border:1px solid #1E293B; border-left:3px solid #06B6D4; border-radius:8px; padding:10px 14px; margin-bottom:8px;">
<div style="font-size:0.72rem; color:#06B6D4; font-weight:700; text-transform:uppercase;">Action Directive #{idx}</div>
<div style="font-size:0.82rem; color:#F8FAFC; margin-top:2px; line-height:1.4;">{act}</div>
</div>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No specific operational actions recommended for this scenario.")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-badge"><span>🏢</span> DOWNSTREAM SHORTAGE & OPERATIONAL IMPACT</div>', unsafe_allow_html=True)

        entity_agent = DisruptionSimulationAgent(graph)
        try:
            detail_df = entity_agent.get_entity_impact_dataframe(sim_result)
        except Exception:
            detail_df = pd.DataFrame(entity_details) if entity_details else pd.DataFrame()

        if entity_details:
            shortage_cards_html = []
            for item in entity_details:
                ent_name = item.get("entity", "Unknown")
                status_val = item.get("status", "")
                c_short = bool(status_val == "confirmed_shortage" or item.get("confirmed_shortage", False))
                p_aff = bool(status_val == "potentially_affected" or item.get("potentially_affected", False))
                sev_paths = item.get("severed_inbound_links", item.get("severed_inbound_paths_count", 0))
                op_paths = item.get("operational_inbound_links", 0)
                op_status = item.get(
                    "operational_status",
                    "Operational (Inbound Supply Starved)"
                    if c_short
                    else ("Operational (Partial / Redundant Inbound)" if p_aff else "Operational"),
                )
                short_status = item.get(
                    "shortage_status",
                    "Confirmed Shortage" if c_short else ("Potentially Affected" if p_aff else "No Shortage"),
                )
                alt_status = item.get("alternative_status") or "Unknown"
                badge = (
                    '<span class="badge-critical">CONFIRMED SHORTAGE</span>'
                    if c_short
                    else (
                        '<span class="badge-high">POTENTIALLY AFFECTED</span>'
                        if p_aff
                        else '<span class="badge-low">OPERATIONAL</span>'
                    )
                )
                border_c = "#F43F5E" if c_short else ("#F59E0B" if p_aff else "#10B981")
                short_color = "#FB7185" if c_short else ("#FBBF24" if p_aff else "#34D399")

                shortage_cards_html.append(
                    f'<div style="background:#151F32; border:1px solid #1E293B; border-top:3px solid {border_c}; border-radius:8px; padding:12px 14px; margin-bottom:10px;">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">'
                    f'<span style="font-weight:700; color:#F8FAFC; font-size:0.95rem;">{ent_name}</span>'
                    f'{badge}'
                    f'</div>'
                    f'<div style="font-size:0.78rem; color:#94A3B8; line-height:1.5;">'
                    f'<b>Operational Status:</b> <span style="color:#CBD5E1;">{op_status}</span> &nbsp;&bull;&nbsp; '
                    f'<b>Shortage Status:</b> <span style="color:{short_color}; font-weight:600;">{short_status}</span><br>'
                    f'<b>Severed Inbound Paths:</b> <span style="color:#F8FAFC;">{sev_paths}</span> &nbsp;&bull;&nbsp; '
                    f'<b>Operational Inbound Paths:</b> <span style="color:#F8FAFC;">{op_paths}</span> &nbsp;&bull;&nbsp; '
                    f'<b>Backup Alternatives:</b> <span style="color:#CBD5E1;">{alt_status}</span>'
                    f'</div>'
                    f'</div>'
                )
            st.markdown("".join(shortage_cards_html), unsafe_allow_html=True)

            with st.expander("📋 View Full Downstream Impact Ledger (Data Table)", expanded=False):
                st.dataframe(detail_df, use_container_width=True, hide_index=True)
        else:
            st.info("No downstream facilities were affected by this disruption scenario.")

    # =========================================================================
    # SECTION 5: RECOVERY PLAN
    # =========================================================================
    elif "5. 💰" in selected_nav:
        st.markdown(
            """
            <div class="section-header-banner">
              <div>
                <div class="section-header-title">💰 Recovery Plan & Financial Optimization</div>
                <div class="section-header-desc">Evaluates available recovery options and selects the lowest-cost feasible recovery plan satisfying configured budget, capacity, and lead-time constraints.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Financial ROI Summary (Equal-width 4-column CSS grid on desktop with complete descriptions)
        recovery_kpi_html = (
            '<div class="kpi-grid">'
            + render_kpi_card(
                "Inaction Gross Loss",
                gross_loss_disp,
                "💸",
                "kpi-card-glow-rose",
                "Total unmitigated financial revenue loss projected across the network.",
                "#FB7185" if is_unquantified else "#F8FAFC",
            )
            + render_kpi_card(
                "Plan Expenditure",
                rec_cost_disp,
                "💳",
                "kpi-card-glow-cyan",
                "Capital required to execute optimized supplier allocations and freight.",
                "#38BDF8",
            )
            + render_kpi_card(
                "Net Capital Saved",
                est_savings_disp,
                "💎",
                "kpi-card-glow-emerald",
                "Net financial capital preserved through active recovery plan intervention.",
                "#34D399" if not is_unquantified else "#FB7185",
            )
            + render_kpi_card(
                "MODELLED RESIDUAL LOSS",
                residual_loss_disp,
                "⚠️",
                "kpi-card-glow-amber",
                "Modelled financial loss after applying the selected recovery option. Actual operational residual risk cannot be confirmed because verified backup inventory and alternative-supplier capacity data are unavailable.",
                "#FBBF24" if not is_unquantified else "#FB7185",
            )
            + '</div>'
        )
        st.markdown(recovery_kpi_html, unsafe_allow_html=True)
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        rec_ids = financial_result.get("recommended_option_ids", [])
        if is_unquantified:
            st.markdown(
                f"""
                <div style="background:#422006; border:1px solid #D97706; border-radius:8px; padding:14px 18px; margin-bottom:12px;">
                  <span style="font-weight:700; color:#FBBF24;">⚠️ FINANCIAL IMPACT UNQUANTIFIED: </span>
                  <span style="color:#FEF3C7;">Operational throughput capacity for <b>{selected_failed_node}</b> is unrecorded in the active dataset.</span>
                  <br>
                  <span style="color:#FDE68A; font-size:0.85rem;">Automated recovery plan optimization is suspended to prevent capital misallocation. Baseline financial loss and net capital preservation cannot be verified without confirmed production volume.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif rec_ids:
            rec_titles = [translate_option_id(oid) for oid in rec_ids]
            total_cost_str = safe_format_currency(financial_result.get("estimated_total_cost"), default="Not available", decimals=2)
            net_savings_str = safe_format_currency(financial_result.get("estimated_savings"), default="Not available", decimals=2)
            st.markdown(
                f"""
                <div style="background:#064E3B; border:1px solid #059669; border-radius:8px; padding:14px 18px; margin-bottom:12px;">
                  <div style="font-size:0.75rem; color:#A7F3D0; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">LOWEST-COST FEASIBLE RECOVERY STRATEGY</div>
                  <div style="font-size:1.25rem; font-weight:800; color:#F0FDF4; margin:4px 0;">{', '.join(rec_titles)} <span style="font-size:0.80rem; color:#A7F3D0; font-family:monospace;">({', '.join(rec_ids)})</span></div>
                  <span style="color:#A7F3D0; font-size:0.85rem;">Plan Expenditure: <b>{total_cost_str}</b> &nbsp;|&nbsp; Net Capital Preserved: <b style="color:#6EE7B7;">{net_savings_str}</b> &nbsp;|&nbsp; Modelled Residual Loss: <b>{residual_loss_disp}</b></span>
                  <div style="color:#D1FAE5; font-size:0.78rem; margin-top:6px;">Evaluates available recovery options in the recovery catalogue and selects the lowest-cost option satisfying configured budget, capacity, and lead-time constraints. Modelled Residual Loss represents modelled financial loss after applying the selected recovery option; actual operational residual risk cannot be confirmed because verified backup inventory and alternative-supplier capacity data are unavailable.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("NO-ACTION BASELINE INVOKED: No evaluated recovery option satisfied all budget, capacity, and lead-time constraints.")

        # Feasibility Comparison Cards
        st.markdown('<div class="sidebar-section-badge"><span>📊</span> CANDIDATE RECOVERY STRATEGIES — FEASIBILITY AUDIT</div>', unsafe_allow_html=True)
        feasible_opts = financial_result.get("feasible_options", [])
        rejected_opts = financial_result.get("rejected_options", [])

        if feasible_opts or rejected_opts:
            for f in feasible_opts:
                opt_id = f.get("option_id", "Unknown")
                is_rec = opt_id in rec_ids
                opt_name = translate_option_id(opt_id)
                rec_badge = '<span style="background:rgba(16,185,129,0.18); color:#34D399; border:1px solid rgba(16,185,129,0.4); padding:3px 10px; border-radius:12px; font-size:0.74rem; font-weight:700;">⭐ LOWEST-COST FEASIBLE OPTION</span>' if is_rec else '<span class="badge-low">FEASIBLE OPTION</span>'
                card_border = "#10B981" if is_rec else "#334155"
                cost_s = safe_format_currency(f.get("recovery_cost"), default="Unknown", decimals=0)
                cap_s = safe_format_capacity(f.get("available_capacity"), default="Unknown", decimals=0, unit=" units")
                lt_s = safe_format_lead_time(f.get("lead_time_days"), default="Unknown", suffix=" days")
                net_s = safe_format_currency(f.get("estimated_savings"), default="Unknown", decimals=0)

                st.markdown(
                    f"""<div style="background:#151F32; border:1px solid #1E293B; border-top:3px solid {card_border}; border-radius:10px; padding:14px 18px; margin-bottom:10px; box-shadow:0 4px 12px rgba(0,0,0,0.3);">
<div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
<div>
<div style="font-size:1.05rem; font-weight:800; color:#F8FAFC;">{opt_name}</div>
<div style="font-size:0.74rem; color:#94A3B8; font-family:monospace;">ID: {opt_id}</div>
</div>
{rec_badge}
</div>
<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:10px; background:#0D1527; padding:10px 12px; border-radius:6px; border:1px solid #1E293B; font-size:0.78rem;">
<div><span style="color:#94A3B8;">Recovery Cost:</span> <b style="color:#38BDF8;">{cost_s}</b></div>
<div><span style="color:#94A3B8;">Capacity:</span> <b style="color:#F8FAFC;">{cap_s}</b></div>
<div><span style="color:#94A3B8;">Lead Time:</span> <b style="color:#FBBF24;">{lt_s}</b></div>
<div><span style="color:#94A3B8;">Net Capital Saved:</span> <b style="color:#34D399;">{net_s}</b></div>
</div>
<div style="margin-top:6px; font-size:0.74rem; color:#34D399;">
✅ Satisfies budget (&le; ${budget_input:,.0f}), capacity, and lead-time constraints (&le; {max_lead_time_input} days).
</div>
</div>""",
                    unsafe_allow_html=True,
                )

            for r in rejected_opts:
                opt_id = r.get("option_id", "Unknown")
                opt_name = translate_option_id(opt_id)
                cost_s = safe_format_currency(r.get("recovery_cost"), default="Unknown", decimals=0)
                cap_s = safe_format_capacity(r.get("available_capacity"), default="Unknown", decimals=0, unit=" units")
                lt_s = safe_format_lead_time(r.get("lead_time_days"), default="Unknown", suffix=" days")
                reasons = "; ".join(r.get("rejection_reasons", [])) or "Constraint requirements not met."
                status_txt = "UNVERIFIED" if is_unquantified else "REJECTED"

                st.markdown(
                    f"""<div style="background:#151F32; border:1px solid #1E293B; border-top:3px solid #64748B; border-radius:10px; padding:14px 18px; margin-bottom:10px; opacity:0.85;">
<div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
<div>
<div style="font-size:1.05rem; font-weight:700; color:#CBD5E1;">{opt_name}</div>
<div style="font-size:0.74rem; color:#64748B; font-family:monospace;">ID: {opt_id}</div>
</div>
<span class="badge-critical" style="background:rgba(239,68,68,0.12) !important; color:#F87171 !important;">{status_txt}</span>
</div>
<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:10px; background:#0D1527; padding:10px 12px; border-radius:6px; border:1px solid #1E293B; font-size:0.78rem;">
<div><span style="color:#64748B;">Recovery Cost:</span> <b style="color:#94A3B8;">{cost_s}</b></div>
<div><span style="color:#64748B;">Capacity:</span> <b style="color:#94A3B8;">{cap_s}</b></div>
<div><span style="color:#64748B;">Lead Time:</span> <b style="color:#94A3B8;">{lt_s}</b></div>
<div><span style="color:#64748B;">Optimization:</span> <b style="color:#EF4444;">Non-viable</b></div>
</div>
<div style="margin-top:6px; font-size:0.74rem; color:#FB7185;">
❌ Rejection Reason: {reasons}
</div>
</div>""",
                    unsafe_allow_html=True,
                )

        # Raw dataframe in expander
        with st.expander("📋 View Complete Recovery Options Feasibility Table (Data Ledger)", expanded=False):
            all_opts_rows = []
            for f in feasible_opts:
                is_rec = f.get("option_id") in rec_ids
                opt_id_str = str(f.get("option_id") or "Unknown Option")
                all_opts_rows.append({
                    "Option Name": translate_option_id(opt_id_str),
                    "Option ID": opt_id_str + (" ⭐ [SELECTED]" if is_rec else ""),
                    "Status": "FEASIBLE",
                    "Recovery Cost": safe_format_currency(f.get("recovery_cost"), default="Unknown", decimals=2),
                    "Available Capacity": safe_format_capacity(f.get("available_capacity"), default="Unknown", decimals=1, unit=""),
                    "Lead Time": safe_format_lead_time(f.get("lead_time_days"), default="Unknown", suffix=" days"),
                    "Total Plan Cost": safe_format_currency(f.get("estimated_total_cost"), default="Unknown", decimals=2),
                    "Net Savings": safe_format_currency(f.get("estimated_savings"), default="Unknown", decimals=2),
                    "Constraint Audit": "Satisfies all constraints",
                })
            for r in rejected_opts:
                opt_id_str = str(r.get("option_id") or "Unknown Option")
                status_label = "UNVERIFIED" if is_unquantified else "REJECTED"
                all_opts_rows.append({
                    "Option Name": translate_option_id(opt_id_str),
                    "Option ID": opt_id_str,
                    "Status": status_label,
                    "Recovery Cost": safe_format_currency(r.get("recovery_cost"), default="Unknown", decimals=2),
                    "Available Capacity": safe_format_capacity(r.get("available_capacity"), default="Unknown", decimals=1, unit=""),
                    "Lead Time": safe_format_lead_time(r.get("lead_time_days"), default="Unknown", suffix=" days"),
                    "Total Plan Cost": "N/A (Infeasible)",
                    "Net Savings": "N/A (Infeasible)",
                    "Constraint Audit": "; ".join(r.get("rejection_reasons", [])),
                })
            if all_opts_rows:
                st.dataframe(pd.DataFrame(all_opts_rows), use_container_width=True, hide_index=True)

        st.caption("Notice: Financial recovery calculations are derived from your specified scenario parameters ($150/unit, $50k budget, 14 days max lead time). This prototype provides decision advisory recommendations and does not execute binding financial payments.")

    # =========================================================================
    # SECTION 6: DECISIONS & APPROVALS
    # =========================================================================
    elif "6. ⚖️" in selected_nav:
        st.markdown(
            """
            <div class="section-header-banner">
              <div>
                <div class="section-header-title">⚖️ Human Decisions & Governance Portal</div>
                <div class="section-header-desc">Human-in-the-loop review. Inspect AI decision narratives, evaluate residual risk, and record official management decisions.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        active_run = audit_logger.get_run_by_id(active_run_id) if active_run_id else None
        current_status = active_run.get("current_decision_status", "PENDING_REVIEW") if active_run else "PENDING_REVIEW"
        created_time = active_run.get("created_at_utc", "Unknown") if active_run else "Unknown"

        status_pill_class = {
            "APPROVED": "badge-low",
            "REJECTED": "badge-critical",
            "PENDING_REVIEW": "badge-high",
            "AUTO_RECOMMENDED": "badge-low",
        }.get(current_status, "badge-medium")

        st.markdown(
            f"""<div style="background:#151F32; border:1px solid #1E293B; border-radius:10px; padding:16px 20px; margin-bottom:14px; box-shadow:0 4px 16px rgba(0,0,0,0.35);">
<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
<div>
<div style="font-size:0.70rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; font-weight:700;">AUDITED GOVERNANCE RUN ID</div>
<div style="font-size:1.2rem; font-weight:800; color:#38BDF8; font-family:monospace; margin-top:2px;">{shorten_run_id(active_run_id)} <span style="font-size:0.75rem; color:#64748B; font-family:sans-serif;">({active_run_id})</span></div>
<div style="font-size:0.74rem; color:#94A3B8; margin-top:2px;">Logged At (UTC): <span style="color:#CBD5E1; font-family:monospace;">{created_time}</span></div>
</div>
<div style="text-align:right;">
<div style="font-size:0.70rem; color:#94A3B8; text-transform:uppercase; margin-bottom:4px; font-weight:700;">Decision Status</div>
<span class="{status_pill_class}" style="font-size:0.85rem; padding:4px 14px;">{current_status}</span>
</div>
</div>
</div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """<div style="background:rgba(59,130,246,0.08); border:1px solid rgba(59,130,246,0.3); border-left:4px solid #3B82F6; border-radius:8px; padding:12px 16px; margin-bottom:16px;">
<div style="font-weight:700; color:#60A5FA; font-size:0.80rem; margin-bottom:2px;">🛡️ PROTOTYPE GOVERNANCE NOTICE:</div>
<div style="color:#94A3B8; font-size:0.78rem; line-height:1.4;">
This control panel captures advisory human oversight within the Nexus-AI compliance log.
Recording a decision creates an immutable event in the local SQLite audit ledger, but does <b>NOT</b> execute
live procurement, place binding supplier purchase orders, authorize bank funds, or reroute physical logistics.
</div>
</div>""",
            unsafe_allow_html=True,
        )

        # Active AI Recommendation Summary Card
        rec_names = [translate_option_id(oid) for oid in financial_result.get("recommended_option_ids", [])]
        rec_display_str = ", ".join(rec_names) if rec_names else "Baseline Hold (No Action)"
        approval_req = decision_exp.get("requires_human_approval", True)
        approval_badge = '<span class="badge-critical">MANDATORY HUMAN SIGN-OFF REQUIRED</span>' if approval_req else '<span class="badge-low">WITHIN AUTONOMOUS THRESHOLDS</span>'

        st.markdown(
            f"""<div style="background:#0D1527; border:1px solid #1E293B; border-radius:10px; padding:16px 20px; margin-bottom:16px;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
<div style="font-size:0.74rem; color:#38BDF8; font-weight:700; text-transform:uppercase; letter-spacing:0.06em;">ACTIVE AI RECOMMENDATION UNDER EVALUATION</div>
{approval_badge}
</div>
<div style="font-size:1.15rem; font-weight:800; color:#F8FAFC; margin-bottom:8px;">
Strategy: <span style="color:#2DD4BF;">{rec_display_str}</span>
</div>
<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:10px; font-size:0.80rem; color:#CBD5E1;">
<div><b>Target Disrupted Node:</b> <span style="color:#FB7185;">{selected_failed_node}</span></div>
<div><b>Inaction Gross Loss:</b> <span style="color:#F8FAFC;">{gross_loss_disp}</span></div>
<div><b>Plan Expenditure:</b> <span style="color:#38BDF8;">{rec_cost_disp}</span></div>
<div><b>Net Capital Saved:</b> <span style="color:#34D399;">{est_savings_disp}</span></div>
</div>
</div>""",
            unsafe_allow_html=True,
        )

        with st.expander("📝 View Full 9-Point Decision Explanation Report", expanded=False):
            st.markdown(decision_exp["markdown_report"])
            st.download_button(
                label="📥 Download Full Decision Explanation Report (.md)",
                data=decision_exp["markdown_report"],
                file_name=f"nexus_ai_decision_report_{selected_failed_node.replace(' ', '_').lower()}.md",
                mime="text/markdown",
            )

        st.markdown("#### Record Human Decision")
        gov_col1, gov_col2 = st.columns([1, 1])
        with gov_col1:
            reviewer_label_input = st.text_input(
                "Reviewer Label / Prototype Role:",
                value="Prototype Reviewer (Supply Chain Risk Analyst)",
                help="Designation recorded in the immutable audit event.",
                key="audit_reviewer_label_field",
                disabled=perms["approval_buttons_disabled"],
            )
        with gov_col2:
            justification_input = st.text_area(
                "Governance Justification / Review Notes:",
                placeholder="Enter rationale for approval, or mandatory justification for rejection...",
                help="Mandatory for rejection, recommended for approval.",
                key="audit_justification_field",
                height=70,
                disabled=perms["approval_buttons_disabled"],
            )

        # Backend Role Authorization Enforcement
        if is_viewer:
            st.markdown(
                '''
                <div style="background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.25); border-radius:8px; padding:10px 14px; margin-bottom:12px; color:#38BDF8; font-size:0.80rem;">
                  👁️ <b>READ-ONLY ACCESS:</b> You are signed in with the <b>VIEWER</b> role. You can inspect all decision narratives, simulations, and financial audits, but you do not have permission to authorize capital allocations or record governance decisions.
                </div>
                ''',
                unsafe_allow_html=True,
            )

        btn_col1, btn_col2, _ = st.columns([1, 1, 2])
        with btn_col1:
            approve_clicked = st.button(
                "Approve Recommendation",
                key="btn_approve_recommendation",
                use_container_width=True,
                disabled=perms["approval_buttons_disabled"],
                help="Requires ADMIN or REVIEWER role." if is_viewer else "Records official approval."
            )
        with btn_col2:
            reject_clicked = st.button(
                "Reject Recommendation",
                key="btn_reject_recommendation",
                use_container_width=True,
                disabled=perms["approval_buttons_disabled"],
                help="Requires ADMIN or REVIEWER role." if is_viewer else "Records official rejection."
            )

        if approve_clicked:
            success, msg, _ = execute_authorized_decision(
                auth_manager=auth_manager,
                audit_logger=audit_logger,
                actor_username=curr_u_name,
                run_id=active_run_id,
                decision_type="RECOMMENDATION_APPROVED",
                reviewer_label=reviewer_label_input,
                justification=justification_input,
                target_entity=selected_failed_node,
                session_state=st.session_state,
            )
            if success:
                st.success(f"{msg}")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"{msg}")

        if reject_clicked:
            success, msg, _ = execute_authorized_decision(
                auth_manager=auth_manager,
                audit_logger=audit_logger,
                actor_username=curr_u_name,
                run_id=active_run_id,
                decision_type="RECOMMENDATION_REJECTED",
                reviewer_label=reviewer_label_input,
                justification=justification_input,
                target_entity=selected_failed_node,
                session_state=st.session_state,
            )
            if success:
                st.warning(f"{msg}")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"{msg}")

    # =========================================================================
    # SECTION 7: AUDIT HISTORY
    # =========================================================================
    elif "7. 📜" in selected_nav:
        st.markdown(
            """
            <div class="section-header-banner">
              <div>
                <div class="section-header-title">📜 Compliance Audit Ledger & History</div>
                <div class="section-header-desc">Immutable append-only ledger tracking all simulation executions, financial recovery recommendations, and human advisory sign-offs.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        runs_list = audit_logger.get_audit_runs(limit=50)
        approved_count = len([r for r in runs_list if r.get("current_decision_status") == "APPROVED"])
        rejected_count = len([r for r in runs_list if r.get("current_decision_status") == "REJECTED"])
        pending_count = len([r for r in runs_list if r.get("current_decision_status") in ("PENDING_REVIEW", "AUTO_RECOMMENDED")])

        a_col1, a_col2, a_col3, a_col4 = st.columns(4)
        with a_col1:
            st.markdown(
                render_kpi_card("Total Runs", f"{len(runs_list)}", "📜", "kpi-card-glow-sky", "Historical simulations", "#38BDF8"),
                unsafe_allow_html=True,
            )
        with a_col2:
            st.markdown(
                render_kpi_card("Approved Actions", f"{approved_count}", "✅", "kpi-card-glow-emerald", "Human validated plans", "#34D399"),
                unsafe_allow_html=True,
            )
        with a_col3:
            st.markdown(
                render_kpi_card("Rejected Actions", f"{rejected_count}", "❌", "kpi-card-glow-rose" if rejected_count > 0 else "kpi-card-glow-teal", "Blocked plans with notes", "#FB7185" if rejected_count > 0 else "#2DD4BF"),
                unsafe_allow_html=True,
            )
        with a_col4:
            st.markdown(
                render_kpi_card("Pending Review", f"{pending_count}", "⏳", "kpi-card-glow-amber", "Awaiting human sign-off", "#FBBF24"),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        audit_tabs = st.tabs([
            "1. Simulation Runs History (Run Ledger)",
            "2. Append-Only Event Trail (Event Ledger)",
        ])

        with audit_tabs[0]:
            if runs_list:
                runs_display = []
                for r in runs_list:
                    b_loss = r["financial_baseline_loss"]
                    b_loss_str = f"${float(b_loss):,.0f}" if b_loss != "UNQUANTIFIED" else "UNQUANTIFIED"
                    p_cost = r["recovery_plan_cost"]
                    p_cost_str = f"${float(p_cost):,.0f}" if p_cost is not None else "UNQUANTIFIED"
                    sav = r["estimated_net_savings"]
                    sav_str = f"${float(sav):,.0f}" if sav != "UNQUANTIFIED" else "UNQUANTIFIED"

                    runs_display.append({
                        "Run Badge": shorten_run_id(r["run_id"]),
                        "Full Run ID": r["run_id"],
                        "Created (UTC)": r["created_at_utc"][:19] + "Z",
                        "Entity": r["scenario_entity"],
                        "Disruption Type": r["disruption_type"],
                        "Severity": r["impact_level"],
                        "Baseline Loss": b_loss_str,
                        "Plan Cost": p_cost_str,
                        "Net Savings": sav_str,
                        "Decision Status": r["current_decision_status"],
                        "Approval Mandated": "YES" if r["requires_human_approval"] else "NO",
                    })
                st.dataframe(pd.DataFrame(runs_display), use_container_width=True, hide_index=True)
            else:
                st.info("No simulation runs recorded in the local audit database yet.")

        with audit_tabs[1]:
            events_list = audit_logger.get_audit_events(limit=100)
            if events_list:
                st.markdown('<div class="sidebar-section-badge"><span>⏱️</span> IMMUTABLE EVENT TIMELINE (APPEND-ONLY LEDGER)</div>', unsafe_allow_html=True)
                for e in events_list:
                    ev_type = e.get("event_type", "EVENT")
                    ev_label, ev_color = translate_event_type(ev_type)
                    ev_time = e.get("timestamp_utc", "")[:19] + "Z"
                    rev_label = e.get("reviewer_label") or "Autonomous Engine"
                    just = e.get("justification") or "No justification commentary recorded."
                    short_id = shorten_run_id(e.get("run_id"))

                    st.markdown(
                        f"""<div style="background:#151F32; border:1px solid #1E293B; border-left:3px solid {ev_color}; border-radius:8px; padding:12px 16px; margin-bottom:10px; box-shadow:0 2px 8px rgba(0,0,0,0.25);">
<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:4px;">
<div style="display:flex; align-items:center; gap:8px;">
<span style="font-weight:700; color:{ev_color}; font-size:0.86rem;">{ev_label}</span>
<span style="font-size:0.72rem; color:#94A3B8; font-family:monospace; background:rgba(255,255,255,0.05); padding:2px 6px; border-radius:4px;">{short_id}</span>
</div>
<span style="font-size:0.74rem; color:#64748B; font-family:monospace;">{ev_time}</span>
</div>
<div style="font-size:0.78rem; color:#CBD5E1; margin-bottom:4px;">
<b>Reviewer:</b> <span style="color:#38BDF8;">{rev_label}</span>
</div>
<div style="font-size:0.76rem; color:#94A3B8; background:#0D1527; padding:6px 10px; border-radius:6px; border:1px solid #1E293B; line-height:1.4;">
💬 <i>"{just}"</i>
</div>
</div>""",
                        unsafe_allow_html=True,
                    )

                with st.expander("📋 View Complete Audit Events Ledger (DataFrame)", expanded=False):
                    events_display = []
                    for e in events_list:
                        events_display.append({
                            "Event ID": e["event_id"],
                            "Run ID": e["run_id"],
                            "Timestamp (UTC)": e["timestamp_utc"][:19] + "Z",
                            "Event Type": e["event_type"],
                            "Reviewer Label": e["reviewer_label"],
                            "Justification": e["justification"] or "(None)",
                        })
                    st.dataframe(pd.DataFrame(events_display), use_container_width=True, hide_index=True)
            else:
                st.info("No audit events recorded in the append-only ledger yet.")

    # =========================================================================
    # SECTION 8: USER MANAGEMENT & ROLE-BASED ACCESS CONTROL (ADMIN ONLY)
    # =========================================================================
    elif "8. 👥" in selected_nav:
        # Strict backend authorization check against live database
        live_um_perms = get_ui_control_permissions(auth_manager, curr_u_name, st.session_state)
        if not live_um_perms["is_active"] or not auth_manager.can_manage_users(live_um_perms["role"]):
            auth_manager._record_audit_event(
                event_type="AUTH_UNAUTHORIZED_ADMIN_ATTEMPT",
                actor_username=curr_u_name or "ANONYMOUS",
                target_username="USER_MANAGEMENT",
                details={"role": live_um_perms["role"], "reason": "Non-admin role attempted user management"},
            )
            st.error(f"Access denied: User '{curr_u_name}' with role '{live_um_perms['role']}' is not permitted to access User Management.")
            st.stop()

        st.markdown(
            '''
            <div class="section-header-banner">
              <div>
                <div class="section-header-title">👥 Enterprise User Management & Role-Based Access Control</div>
                <div class="section-header-desc">
                  Provision new Reviewer and Viewer accounts, manage account status, perform administrative password resets, and inspect security audit events.
                </div>
              </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        tab_accounts, tab_create, tab_sec_audit = st.tabs([
            "📋 Registered Accounts",
            "➕ Provision New User",
            "🛡️ Security Audit Log"
        ])

        with tab_create:
            st.markdown("#### Provision New Enterprise Personnel Account")
            st.caption("Create a new user account with strict role assignment. Public self-registration is permanently prohibited.")
            with st.form("form_create_new_enterprise_user"):
                new_uname = st.text_input("Username:", placeholder="e.g. jsmith", help="Alphanumeric, at least 3 characters.", key="new_user_username_input")
                new_uemail = st.text_input("Email Address:", placeholder="e.g. jsmith@enterprise.com", key="new_user_email_input")
                new_urole = st.selectbox("Assign User Role:", ["REVIEWER", "VIEWER"], key="new_user_role_select")
                new_upw = st.text_input("Initial Password:", type="password", help="At least 8 characters with letters and numbers.", key="new_user_pw_input")
                sub_create_u = st.form_submit_button("Provision User Account", key="btn_sub_create_user", use_container_width=True)
                if sub_create_u:
                    ok_cu, msg_cu, _ = auth_manager.create_user(curr_u_name, new_uname, new_uemail, new_upw, new_urole)
                    if ok_cu:
                        st.success(msg_cu)
                    else:
                        st.error(msg_cu)

        with tab_accounts:
            all_users = auth_manager.list_users(curr_u_name)
            st.markdown(f"**Total Enterprise Users Registered:** `{len(all_users)}`")

            user_table_data = []
            for u in all_users:
                user_table_data.append({
                    "Username": u["username"],
                    "Email Address": u["email"],
                    "Assigned Role": u["role"],
                    "Status": "ACTIVE" if u["is_active"] else "DEACTIVATED",
                    "Created (UTC)": u["created_at_utc"][:19].replace("T", " ") if u["created_at_utc"] else "Unknown",
                    "Last Login (UTC)": u["last_login_utc"][:19].replace("T", " ") if u["last_login_utc"] else "Never",
                })
            st.dataframe(pd.DataFrame(user_table_data), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### Account Status & Credential Administration")
            other_users = [u["username"] for u in all_users if u["username"].lower() != curr_u_name.lower()]

            if not other_users:
                st.info("No secondary accounts exist to administer yet. Use the 'Provision New User' tab to add Reviewers or Viewers.")
            else:
                target_sel = st.selectbox("Select Target User Account to Administer:", other_users, key="sel_target_admin_user")
                target_obj = next((u for u in all_users if u["username"] == target_sel), None)

                if target_obj:
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.markdown(f"**Target User:** `{target_obj['username']}` ({target_obj['email']})")
                        st.markdown(f"**Current Status:** `{'ACTIVE' if target_obj['is_active'] else 'DEACTIVATED'}` | **Current Role:** `{target_obj['role']}`")

                        # Activation / Deactivation Button
                        if target_obj["is_active"]:
                            if st.button(f"🚫 Deactivate '{target_sel}'", key=f"btn_deact_{target_sel}"):
                                ok_st, msg_st = auth_manager.set_account_status(curr_u_name, target_sel, False)
                                if ok_st:
                                    st.success(msg_st)
                                    time.sleep(0.5)
                                    st.rerun()
                                else:
                                    st.error(msg_st)
                        else:
                            if st.button(f"✅ Activate '{target_sel}'", key=f"btn_act_{target_sel}"):
                                ok_st, msg_st = auth_manager.set_account_status(curr_u_name, target_sel, True)
                                if ok_st:
                                    st.success(msg_st)
                                    time.sleep(0.5)
                                    st.rerun()
                                else:
                                    st.error(msg_st)

                        # Role Change
                        role_options = ["REVIEWER", "VIEWER", "ADMIN"]
                        new_r_idx = role_options.index(target_obj["role"]) if target_obj["role"] in role_options else 0
                        chosen_new_role = st.selectbox("Re-assign Role:", role_options, index=new_r_idx, key=f"role_sel_{target_sel}")
                        if st.button(f"Apply Role Change to '{chosen_new_role}'", key=f"btn_chg_role_{target_sel}"):
                            ok_rc, msg_rc = auth_manager.change_role(curr_u_name, target_sel, chosen_new_role)
                            if ok_rc:
                                st.success(msg_rc)
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error(msg_rc)

                    with col_m2:
                        st.markdown("**Administrative Password Reset**")
                        st.caption("Securely sets a new password. The user will be required to authenticate with this new credential.")
                        with st.form(f"form_admin_reset_{target_sel}"):
                            new_reset_pw = st.text_input(f"New Password for {target_sel}:", type="password", help="At least 8 chars with letters & numbers.")
                            sub_reset = st.form_submit_button("Reset Password")
                            if sub_reset:
                                ok_rp, msg_rp = auth_manager.reset_password(curr_u_name, target_sel, new_reset_pw)
                                if ok_rp:
                                    st.success(msg_rp)
                                else:
                                    st.error(msg_rp)

        with tab_sec_audit:
            st.markdown("#### Security & Access Control Audit Log")
            st.caption("Immutable append-only record of all login events, account creation, role assignments, and credential operations.")
            auth_events = auth_manager.get_auth_audit_events(curr_u_name, limit=100)
            if not auth_events:
                st.info("No security audit events recorded yet.")
            else:
                sec_event_rows = []
                for ev in auth_events:
                    sec_event_rows.append({
                        "Event ID": ev["event_id"],
                        "Timestamp (UTC)": ev["timestamp_utc"][:19].replace("T", " ") if ev["timestamp_utc"] else "",
                        "Event Type": ev["event_type"],
                        "Actor": ev["actor_username"],
                        "Target": ev["target_username"] or "-",
                        "Details": json.dumps(ev["details"]),
                    })
                st.dataframe(pd.DataFrame(sec_event_rows), use_container_width=True, hide_index=True)


def run_cli_fallback(args: argparse.Namespace) -> None:
    """CLI fallback runner when app.py is invoked directly as a Python script."""
    print("=" * 70)
    print("        NEXUS-AI: MULTI-TIER SUPPLY CHAIN RESILIENCE PLATFORM")
    print("               Member 5 Integration & Decision Report")
    print("=" * 70)

    # 1. Load Data
    if getattr(args, "sap", False):
        print("\n[Step 1/5] Ingesting & validating data from SAP HANA Cloud (HACKFEST0316)...")
        try:
            graph, dep_df, nodes_df = load_sap_supply_chain()
            print(f"           SUCCESS: Loaded {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges from SAP HANA.")
        except Exception as exc:
            print(f"[ERROR] Failed to load from SAP HANA: {exc}")
            print("        Falling back to local CSV...")
            csv_path = args.csv if args.csv else DEFAULT_DEPENDENCIES_CSV
            df = load_and_validate_dependencies(csv_path)
            graph = build_supply_chain_graph(df)
    else:
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
    sim_res = simulate_disruption(graph, failed_node=failed_target, actor_role="ADMIN")
    print(f"           Impact Level: {sim_res.get('impact_level')}")
    print(f"           Affected Entities ({len(sim_res.get('affected_entities', []))}): {sim_res.get('affected_entities', [])}")

    # 4. Member 4: Financial Optimization
    print("\n[Step 4/5] Executing Member 4 Financial Impact & Recovery Optimization...")
    scenario = build_financial_scenario_from_simulation(
        simulation_result=sim_res,
        loss_per_unit=args.loss_per_unit,
        available_budget=args.budget,
        max_lead_time_days=args.max_lead_time,
        graph=graph,
    )
    candidate_opts = generate_candidate_recovery_options(scenario, graph=graph)
    fin_res = optimize_recovery_plan(scenario, candidate_opts, allow_multi_option=True)
    gross_loss_str = safe_format_currency(fin_res.get("baseline_disruption_loss"), default="Not available", decimals=2)
    tot_cost_str = safe_format_currency(fin_res.get("estimated_total_cost"), default="Not available", decimals=2)
    sav_str = safe_format_currency(fin_res.get("estimated_savings"), default="Not available", decimals=2)
    if fin_res.get("is_unquantified"):
        gross_loss_str = "UNQUANTIFIED (Throughput capacity unrecorded)"
        tot_cost_str = "UNQUANTIFIED"
        sav_str = "UNQUANTIFIED"
    print(f"           Baseline Disruption Loss: {gross_loss_str}")
    print(f"           Selected Recovery Plan:   {fin_res.get('recommended_option_ids')}")
    print(f"           Total Recovery Cost:      {tot_cost_str}")
    print(f"           Net Estimated Savings:    {sav_str}")

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

    # 6. Audit Logging in CLI
    audit_logger = AuditLogger()
    cli_run_id = audit_logger.record_simulation_run(
        scenario_entity=failed_target,
        disruption_type=sim_res.get("disruption_type", "Facility Outage"),
        scenario_params={
            "failed_node": failed_target,
            "unit_loss": args.loss_per_unit,
            "budget": args.budget,
            "max_lead_time": args.max_lead_time,
        },
        sim_result=sim_res,
        financial_result=fin_res,
        decision_exp=exp,
        execution_channel="CLI",
        actor_role="ADMIN",
    )
    print(f"\n[Audit] Simulation recorded in nexus_audit_runs with Run ID: {cli_run_id}")
    print("[Tip] To launch the full interactive web dashboard, run:")
    print("      streamlit run app.py\n")


def parse_cli_args() -> argparse.Namespace:
    """Parses optional command-line arguments."""
    parser = argparse.ArgumentParser(description="Nexus-AI Supply Chain Resilience Application")
    parser.add_argument("--csv", type=Path, default=None, help="Path to dependencies CSV file")
    parser.add_argument("--fail-node", type=str, default=None, help="Target entity to disrupt")
    parser.add_argument("--loss-per-unit", type=float, default=150.0, help="Loss per unit ($)")
    parser.add_argument("--budget", type=float, default=50000.0, help="Available recovery budget ($)")
    parser.add_argument("--max-lead-time", type=float, default=14.0, help="Max acceptable lead time (days)")
    parser.add_argument("--sap", action="store_true", help="Load live data from SAP HANA Cloud (schema HACKFEST0316)")
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
