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
    create_resilient_network_figure,
    generate_candidate_recovery_options,
    render_risk_card,
    safe_format_capacity,
    safe_format_currency,
    safe_format_lead_time,
    safe_format_number,
)

# Audit Logging & Governance Import
from agents.audit_logger import AuditLogger


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

    # Custom CSS for dark-mode styling - Enterprise Control Tower Theme
    st.markdown(
        """
        <style>
        /* ── Global app background ───────────────────────────────────── */
        [data-testid="stAppViewContainer"],
        [data-testid="stApp"],
        .main,
        .main .block-container {
            background-color: #070B14 !important;
        }

        /* ── Main content area wrapper ──────────────────────────────── */
        section.main > div,
        .block-container {
            background-color: #070B14 !important;
            color: #F8FAFC !important;
            padding-top: 1.2rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1440px !important;
        }

        /* ── Scrollbar Styling ──────────────────────────────────────── */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #070B14;
        }
        ::-webkit-scrollbar-thumb {
            background: #1E293B;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #06B6D4;
        }

        /* ── Typography & General Text ───────────────────────────────── */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        }
        .stMarkdown p,
        .stMarkdown li,
        .stMarkdown span,
        .stMarkdown label,
        p, span, li, label {
            color: #CBD5E1 !important;
        }
        h1, h2, h3, h4, h5, h6,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #F8FAFC !important;
            letter-spacing: -0.02em !important;
        }

        /* ── Top Navigation Segmented Control ───────────────────────── */
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: wrap !important;
            gap: 8px !important;
            background-color: #0D1527 !important;
            padding: 6px 8px !important;
            border-radius: 12px !important;
            border: 1px solid #1E293B !important;
            margin-bottom: 1.2rem !important;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] > label {
            background-color: #151F32 !important;
            border: 1px solid #1E293B !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            margin: 0 !important;
            cursor: pointer !important;
            transition: all 0.15s ease !important;
            font-weight: 600 !important;
            font-size: 0.84rem !important;
            color: #CBD5E1 !important;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
            border-color: #06B6D4 !important;
            color: #F8FAFC !important;
            background-color: #1E293B !important;
            transform: translateY(-1px) !important;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
        div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
            background: linear-gradient(135deg, rgba(6,182,212,0.25) 0%, rgba(20,184,166,0.15) 100%) !important;
            border: 1px solid #06B6D4 !important;
            color: #38BDF8 !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 10px rgba(6, 182, 212, 0.25) !important;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] input[type="radio"] {
            display: none !important;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
            display: none !important;
        }

        /* ── Sidebar ────────────────────────────────────────────────── */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
            background-color: #0D1527 !important;
            border-right: 1px solid #1E293B !important;
        }
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stNumberInput label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #94A3B8 !important;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #F8FAFC !important;
        }
        .sidebar-section-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #38BDF8;
            margin-bottom: 8px;
        }

        /* ── Enterprise Masthead Header ──────────────────────────────── */
        .control-tower-header {
            background: linear-gradient(135deg, #101B30 0%, #0B1323 100%);
            border: 1px solid #1E293B;
            border-radius: 12px;
            padding: 18px 24px;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
            position: relative;
            overflow: hidden;
        }
        .control-tower-header::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #06B6D4 0%, #14B8A6 50%, #38BDF8 100%);
        }
        .masthead-title {
            font-size: 2.1rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em !important;
            color: #FFFFFF !important;
            margin: 0 !important;
            line-height: 1.15 !important;
        }
        .masthead-tagline {
            font-size: 0.92rem;
            font-weight: 500;
            color: #38BDF8;
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
        .telemetry-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
        }
        .telemetry-live {
            background: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.45);
        }
        .telemetry-pulse {
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: #34D399;
            box-shadow: 0 0 8px #34D399;
            animation: pulse-dot 2s infinite;
        }
        @keyframes pulse-dot {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.35; transform: scale(0.85); }
            100% { opacity: 1; transform: scale(1); }
        }
        .telemetry-standby {
            background: rgba(6, 182, 212, 0.12);
            color: #38BDF8;
            border: 1px solid rgba(6, 182, 212, 0.4);
        }
        .telemetry-local {
            background: rgba(148, 163, 184, 0.12);
            color: #CBD5E1;
            border: 1px solid rgba(148, 163, 184, 0.3);
        }
        .telemetry-fallback {
            background: rgba(245, 158, 11, 0.15);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.45);
        }

        /* ── KPI Cards ──────────────────────────────────────────────── */
        .kpi-card {
            background: #151F32;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 14px 16px;
            height: 100%;
            min-height: 100px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            position: relative;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.45);
            border-color: #334155;
        }
        .kpi-card-glow-cyan   { border-top: 3px solid #06B6D4; }
        .kpi-card-glow-teal   { border-top: 3px solid #14B8A6; }
        .kpi-card-glow-sky    { border-top: 3px solid #38BDF8; }
        .kpi-card-glow-amber  { border-top: 3px solid #F59E0B; }
        .kpi-card-glow-rose   { border-top: 3px solid #F43F5E; }
        .kpi-card-glow-emerald{ border-top: 3px solid #10B981; }

        .kpi-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        .kpi-card-label {
            font-size: 0.72rem;
            font-weight: 700;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .kpi-card-icon {
            font-size: 1.05rem;
            line-height: 1;
        }
        .kpi-card-value {
            font-size: 1.45rem;
            font-weight: 800;
            color: #F8FAFC;
            letter-spacing: -0.02em;
            line-height: 1.15;
            font-feature-settings: "tnum";
        }
        .kpi-card-subtitle {
            font-size: 0.70rem;
            color: #64748B;
            margin-top: 4px;
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* ── Modern Risk Badges (Pills) ─────────────────────────────── */
        .badge-critical {
            background-color: rgba(244, 63, 94, 0.18) !important;
            color: #FB7185 !important;
            border: 1px solid rgba(244, 63, 94, 0.45) !important;
            padding: 3px 10px !important;
            border-radius: 16px !important;
            font-weight: 700 !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.03em !important;
            display: inline-flex !important;
            align-items: center !important;
        }
        .badge-high {
            background-color: rgba(245, 158, 11, 0.18) !important;
            color: #FBBF24 !important;
            border: 1px solid rgba(245, 158, 11, 0.45) !important;
            padding: 3px 10px !important;
            border-radius: 16px !important;
            font-weight: 700 !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.03em !important;
            display: inline-flex !important;
            align-items: center !important;
        }
        .badge-medium {
            background-color: rgba(234, 179, 8, 0.18) !important;
            color: #FDE047 !important;
            border: 1px solid rgba(234, 179, 8, 0.45) !important;
            padding: 3px 10px !important;
            border-radius: 16px !important;
            font-weight: 700 !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.03em !important;
            display: inline-flex !important;
            align-items: center !important;
        }
        .badge-low {
            background-color: rgba(16, 185, 129, 0.18) !important;
            color: #34D399 !important;
            border: 1px solid rgba(16, 185, 129, 0.45) !important;
            padding: 3px 10px !important;
            border-radius: 16px !important;
            font-weight: 700 !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.03em !important;
            display: inline-flex !important;
            align-items: center !important;
        }

        /* ── Form Controls & Inputs ─────────────────────────────────── */
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {
            background-color: #151F32 !important;
            color: #F8FAFC !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2) !important;
        }
        [data-testid="stSelectbox"] > div > div:focus-within,
        [data-testid="stNumberInput"] input:focus,
        [data-testid="stTextInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus {
            border-color: #06B6D4 !important;
            box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.25) !important;
        }

        /* ── Custom Decision Action Buttons ────────────────────────── */
        div[data-testid="stButton"] > button:has(p:contains("Reject")),
        button[key*="reject"] {
            background: rgba(239, 68, 68, 0.12) !important;
            border: 1px solid rgba(239, 68, 68, 0.5) !important;
            color: #F87171 !important;
            font-weight: 700 !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stButton"] > button:has(p:contains("Reject")):hover,
        button[key*="reject"]:hover {
            background: rgba(239, 68, 68, 0.25) !important;
            border-color: #EF4444 !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4) !important;
            transform: translateY(-1px) !important;
        }
        div[data-testid="stButton"] > button:has(p:contains("Approve")),
        button[key*="approve"] {
            background: linear-gradient(135deg, #059669 0%, #0D9488 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stButton"] > button:has(p:contains("Approve")):hover,
        button[key*="approve"]:hover {
            background: linear-gradient(135deg, #10B981 0%, #14B8A6 100%) !important;
            box-shadow: 0 6px 18px rgba(16, 185, 129, 0.5) !important;
            transform: translateY(-1px) !important;
        }

        /* ── Buttons ────────────────────────────────────────────────── */
        [data-testid="stBaseButton-primary"],
        button[kind="primary"] {
            background: linear-gradient(135deg, #06B6D4 0%, #0D9488 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            padding: 0.5rem 1.25rem !important;
            box-shadow: 0 4px 14px rgba(6, 182, 212, 0.35) !important;
            transition: all 0.2s ease !important;
        }
        [data-testid="stBaseButton-primary"]:hover,
        button[kind="primary"]:hover {
            background: linear-gradient(135deg, #0891B2 0%, #0F766E 100%) !important;
            box-shadow: 0 6px 18px rgba(6, 182, 212, 0.5) !important;
            transform: translateY(-1px) !important;
        }
        [data-testid="stBaseButton-secondary"],
        button[kind="secondary"] {
            background-color: #1E293B !important;
            color: #E2E8F0 !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        [data-testid="stBaseButton-secondary"]:hover,
        button[kind="secondary"]:hover {
            background-color: #273549 !important;
            border-color: #06B6D4 !important;
            color: #38BDF8 !important;
        }
        [data-testid="stDownloadButton"] button {
            background: linear-gradient(135deg, #06B6D4 0%, #0D9488 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 14px rgba(6, 182, 212, 0.35) !important;
        }

        /* ── Dataframes & Tables ────────────────────────────────────── */
        [data-testid="stDataFrame"] {
            border: 1px solid #1E293B !important;
            border-radius: 8px !important;
            overflow: hidden !important;
        }
        [data-testid="stDataFrame"] [data-testid="stElementToolbar"],
        [data-testid="stElementToolbar"] {
            background-color: #151F32 !important;
            border: 1px solid #1E293B !important;
            border-radius: 6px !important;
        }

        /* ── Section Dividers & Cards ───────────────────────────────── */
        hr {
            border-color: #1E293B !important;
            margin: 1.2rem 0 !important;
        }
        .user-guide-box {
            background: linear-gradient(135deg, #101B30 0%, #0D1629 100%);
            border: 1px solid #1E293B;
            border-left: 4px solid #06B6D4;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 1.2rem;
        }
        .section-header-banner {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 0.8rem;
        }
        .section-header-title {
            font-size: 1.35rem;
            font-weight: 800;
            color: #F8FAFC;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .section-header-desc {
            font-size: 0.82rem;
            color: #94A3B8;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 2. Sidebar: Configuration & Global Controls
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

    st.sidebar.markdown('<div class="sidebar-section-badge"><span>📂</span> 1. DATA SOURCE & TELEMETRY</div>', unsafe_allow_html=True)

    sap_is_ready = is_sap_configured()
    default_dataset_idx = 0 if sap_is_ready else 1

    data_choice = st.sidebar.radio(
        "Choose Supply Chain Data Source:",
        [
            "SAP HANA Cloud (Live Database: HACKFEST0316)",
            "Standard Sample (data/dependencies.csv)",
            "Canonical Multi-Tier Network (with Factories & Routes)",
            "Upload Custom Dependencies CSV",
        ],
        index=default_dataset_idx,
    )

    graph: nx.DiGraph
    data_source_name: str = ""
    is_sap_active: bool = False
    is_sap_fallback: bool = False
    sap_diagnostics: str = ""

    if data_choice == "SAP HANA Cloud (Live Database: HACKFEST0316)":
        try:
            with st.spinner("Connecting to SAP HANA Cloud & querying NEXUS tables..."):
                graph, dep_df, nodes_df = load_sap_supply_chain()
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
    )

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
    st.sidebar.markdown('<div class="sidebar-section-badge"><span>💰</span> 3. FINANCIAL CONSTRAINTS</div>', unsafe_allow_html=True)

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

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-section-badge"><span>📜</span> 4. GOVERNANCE & AUDIT</div>', unsafe_allow_html=True)
    if st.sidebar.button("🔁 Re-run Simulation (New Audit Record)", help="Starts an intentional new simulation run with identical parameters, minting a distinct Run ID and audit record."):
        st.session_state["force_new_run"] = True
        st.session_state["audit_justification_field"] = ""

    # =========================================================================
    # Run Agent Pipelines Deterministically
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

    # 5. Audit Logging & State Deduplication
    audit_logger = AuditLogger()
    sim_fingerprint = f"{selected_failed_node}|{passed_disruption_type}|{unit_loss_input:.1f}|{budget_input:.1f}|{max_lead_time_input}|{allow_multi_option}"
    force_new = st.session_state.get("force_new_run", False)

    if force_new or st.session_state.get("active_sim_fingerprint") != sim_fingerprint:
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
        )
        st.session_state["active_sim_fingerprint"] = sim_fingerprint
        st.session_state["active_run_id"] = active_run_id
        st.session_state["force_new_run"] = False
    else:
        active_run_id = st.session_state.get("active_run_id")
        if not active_run_id:
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
            )
            st.session_state["active_sim_fingerprint"] = sim_fingerprint
            st.session_state["active_run_id"] = active_run_id

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

    # =========================================================================
    # Header & Enterprise Telemetry
    # =========================================================================
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

    st.markdown(
        f"""<div class="control-tower-header">
<div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
<div>
<div style="display:inline-flex; align-items:center; gap:8px; margin-bottom:4px;">
<span style="font-size:0.70rem; font-weight:700; color:#06B6D4; text-transform:uppercase; letter-spacing:0.12em; background:rgba(6,182,212,0.12); padding:3px 10px; border-radius:12px; border:1px solid rgba(6,182,212,0.25);">
ENTERPRISE AGENTIC CONTROL TOWER
</span>
</div>
<h1 class="masthead-title">Nexus-AI</h1>
<div class="masthead-tagline">Intelligence for Resilient Supply Chains</div>
</div>
<div style="display:flex; flex-direction:column; align-items:flex-end; gap:8px;">
{sap_badge_html}
<div style="font-size:0.74rem; color:#64748B; font-family:monospace; letter-spacing:0.04em;">
FIVE SPECIALIZED RESILIENCE AGENTS SYNCHRONIZED
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
<span style="color:#38BDF8; font-weight:600; font-size:0.82rem;">{sim_result.get("disruption_type", "Operational Outage")}</span>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    if is_sap_fallback:
        st.warning(
            "⚠️ **SAP HANA connection unavailable. The application is currently using the sample dataset.**\n\n"
            "Active network topology and component parameters have defaulted to the verified local benchmark (`data/dependencies.csv`). "
            "All disruption shocks, network blast radius calculations, and financial optimizations reflect sample data."
        )

    # =========================================================================
    # Navigation: 7 Clean Functional Views with Conditional Rendering
    # =========================================================================
    nav_options = [
        "1. 📊 Executive Dashboard",
        "2. 🗺️ Supply Chain Map",
        "3. 🛡️ Risk Analysis",
        "4. ⚡ Disruption Simulator",
        "5. 💰 Recovery Plan",
        "6. ⚖️ Decisions & Approvals",
        "7. 📜 Audit History",
    ]

    selected_nav = st.radio(
        "Navigation",
        nav_options,
        horizontal=True,
        key="main_nav_section",
        label_visibility="collapsed",
    )

    # =========================================================================
    # SECTION 1: EXECUTIVE DASHBOARD
    # =========================================================================
    if "1. 📊" in selected_nav:
        st.markdown(
            """
            <div class="section-header-banner">
              <div>
                <div class="section-header-title">📊 Executive Dashboard</div>
                <div class="section-header-desc">Real-time situational awareness, network baseline health, and active disruption status.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:12px; margin-bottom:1.2rem;">
<div style="background:#151F32; border:1px solid #1E293B; border-top:3px solid #06B6D4; border-radius:8px; padding:14px; box-shadow:0 4px 12px rgba(0,0,0,0.25);">
<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
<span style="font-size:1.1rem;">🗺️</span>
<span style="font-size:0.85rem; font-weight:700; color:#F8FAFC;">1. Map Multi-Tier Network</span>
</div>
<div style="font-size:0.75rem; color:#94A3B8; line-height:1.4;">
Trace deep-tier dependencies from raw materials through processors to downstream suppliers.
</div>
</div>
<div style="background:#151F32; border:1px solid #1E293B; border-top:3px solid #14B8A6; border-radius:8px; padding:14px; box-shadow:0 4px 12px rgba(0,0,0,0.25);">
<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
<span style="font-size:1.1rem;">🛡️</span>
<span style="font-size:0.85rem; font-weight:700; color:#F8FAFC;">2. Detect Hidden Risks</span>
</div>
<div style="font-size:0.75rem; color:#94A3B8; line-height:1.4;">
Detect single points of failure (SPOF) and shared choke-points feeding multiple parallel suppliers.
</div>
</div>
<div style="background:#151F32; border:1px solid #1E293B; border-top:3px solid #F59E0B; border-radius:8px; padding:14px; box-shadow:0 4px 12px rgba(0,0,0,0.25);">
<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
<span style="font-size:1.1rem;">⚡</span>
<span style="font-size:0.85rem; font-weight:700; color:#F8FAFC;">3. Simulate Shocks</span>
</div>
<div style="font-size:0.75rem; color:#94A3B8; line-height:1.4;">
Stress-test resilience by injecting shocks to calculate deterministic blast radius and shortage impact.
</div>
</div>
<div style="background:#151F32; border:1px solid #1E293B; border-top:3px solid #10B981; border-radius:8px; padding:14px; box-shadow:0 4px 12px rgba(0,0,0,0.25);">
<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
<span style="font-size:1.1rem;">⚖️</span>
<span style="font-size:0.85rem; font-weight:700; color:#F8FAFC;">4. Human Governance</span>
</div>
<div style="font-size:0.75rem; color:#94A3B8; line-height:1.4;">
Review mathematically optimized recovery options and record human sign-off into an immutable audit trail.
</div>
</div>
</div>""",
            unsafe_allow_html=True,
        )

        # Baseline Operational Network Telemetry
        kpi_row = st.columns(4)
        with kpi_row[0]:
            st.markdown(
                render_kpi_card("Total Entities", f"{total_entities}", "🌐", "kpi-card-glow-cyan", "Active network nodes", "#38BDF8"),
                unsafe_allow_html=True,
            )
        with kpi_row[1]:
            st.markdown(
                render_kpi_card("Suppliers", f"{total_suppliers}", "🏭", "kpi-card-glow-teal", "Direct tier-1/tier-2 sources", "#2DD4BF"),
                unsafe_allow_html=True,
            )
        with kpi_row[2]:
            st.markdown(
                render_kpi_card("Processing Plants", f"{total_factories}", "🏢", "kpi-card-glow-sky", "Manufacturing facilities", "#60A5FA"),
                unsafe_allow_html=True,
            )
        with kpi_row[3]:
            st.markdown(
                render_kpi_card("Supply Corridors", f"{graph.number_of_edges()}", "🔗", "kpi-card-glow-amber", "Directed material dependencies", "#FBBF24"),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Active Stress Simulation Quick Card
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

        st.markdown(
            f"""
            <div style="background:#151F32; border:1px solid #1E293B; border-radius:10px; padding:18px 20px; box-shadow:0 4px 16px rgba(0,0,0,0.35);">
              <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px; margin-bottom:12px;">
                <div>
                  <div style="font-size:0.72rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; font-weight:700;">Active Disruption Simulation Loaded</div>
                  <div style="font-size:1.25rem; font-weight:800; color:#F8FAFC; margin-top:2px;">Target: {selected_failed_node}</div>
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
                  <div style="font-size:1.15rem; font-weight:800; color:#38BDF8;">{', '.join([translate_option_id(o) for o in financial_result.get('recommended_option_ids', [])]) or 'Baseline Hold'}</div>
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

    # =========================================================================
    # SECTION 2: SUPPLY CHAIN MAP
    # =========================================================================
    elif "2. 🗺️" in selected_nav:
        st.markdown(
            """
            <div class="section-header-banner">
              <div>
                <div class="section-header-title">🗺️ Supply Chain Network Topology</div>
                <div class="section-header-desc">Interactive directed multi-tier map: upstream raw materials feed intermediate processors, which supply downstream partners.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_map, col_inspect = st.columns([3, 1])

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
            fig = create_resilient_network_figure(
                graph=graph,
                simulation_result=sim_result,
                selected_entity=inspect_node,
            )
            st.plotly_chart(fig, use_container_width=True)

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

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.markdown(
                render_kpi_card("Audited Assets", f"{total_audited}", "🔬", "kpi-card-glow-sky", "Upstream nodes evaluated", "#38BDF8"),
                unsafe_allow_html=True,
            )
        with col_r2:
            st.markdown(
                render_kpi_card("Shared Bottlenecks", f"{shared_count}", "⚠️", "kpi-card-glow-rose" if shared_count > 0 else "kpi-card-glow-teal", "Affecting ≥2 suppliers", "#FB7185" if shared_count > 0 else "#2DD4BF"),
                unsafe_allow_html=True,
            )
        with col_r3:
            st.markdown(
                render_kpi_card("Peak Risk Score", f"{max_risk:.1f}/100", "⚡", "kpi-card-glow-amber", "Worst-case vulnerability", "#FBBF24"),
                unsafe_allow_html=True,
            )
        with col_r4:
            st.markdown(
                render_kpi_card("Average Risk", f"{avg_risk:.1f}/100", "📊", "kpi-card-glow-teal", "Network exposure index", "#2DD4BF"),
                unsafe_allow_html=True,
            )

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

        cap_impact = sim_result.get("estimated_capacity_impact", {})
        direct_cap_val = cap_impact.get("estimated_direct_capacity_loss")
        direct_cap_str = safe_format_capacity(direct_cap_val, default="Not available", decimals=1, unit=" units")
        materials = sim_result.get("affected_materials", [])
        paths = sim_result.get("disrupted_paths") or sim_result.get("affected_paths") or []
        corridor_paths = [p for p in paths if len(p) > 1]
        actions = sim_result.get("recommended_actions") or sim_result.get("recommended_next_actions") or []
        entity_details = sim_result.get("entity_impact_details", [])

        # 4 Outage KPI Cards
        o_col1, o_col2, o_col3, o_col4 = st.columns(4)
        with o_col1:
            st.markdown(
                render_kpi_card("Failed Facility", f"{selected_failed_node}", "💥", "kpi-card-glow-rose", sim_result.get("disruption_type", "Operational Outage"), "#FB7185"),
                unsafe_allow_html=True,
            )
        with o_col2:
            sev_color = {
                "Critical": "#FB7185",
                "High": "#FBBF24",
                "Medium": "#FDE047",
                "Low": "#34D399",
            }.get(impact_level, "#FDE047")
            st.markdown(
                render_kpi_card("Severity Tier", f"{impact_level.upper()}", "⚡", "kpi-card-glow-amber", "Blast radius classification", sev_color),
                unsafe_allow_html=True,
            )
        with o_col3:
            st.markdown(
                render_kpi_card("Direct Capacity Severed", direct_cap_str, "📉", "kpi-card-glow-cyan", "Immediate throughput lost", "#38BDF8"),
                unsafe_allow_html=True,
            )
        with o_col4:
            st.markdown(
                render_kpi_card("Downstream Entities", f"{affected_count} Entities", "🏢", "kpi-card-glow-teal", "Total facilities affected", "#2DD4BF"),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Visual Propagation Cascade
        mat_count = len(materials)
        cascade_html = (
            f'<div style="background:#0D1527; border:1px solid #1E293B; border-radius:10px; padding:14px 18px; margin-bottom:16px;">'
            f'<div style="font-size:0.72rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; font-weight:700; margin-bottom:8px;">'
            f'DISRUPTION SHOCK PROPAGATION CASCADE'
            f'</div>'
            f'<div style="display:flex; align-items:center; flex-wrap:wrap; gap:10px; font-size:0.84rem;">'
            f'<span style="background:rgba(244,63,94,0.18); color:#FB7185; border:1px solid rgba(244,63,94,0.4); padding:4px 12px; border-radius:16px; font-weight:700;">💥 ROOT OUTAGE: {selected_failed_node}</span>'
            f'<span style="color:#64748B; font-weight:700;">&rarr;</span>'
            f'<span style="background:rgba(245,158,11,0.18); color:#FBBF24; border:1px solid rgba(245,158,11,0.4); padding:4px 12px; border-radius:16px; font-weight:700;">⚡ BLAST RADIUS: {affected_count} Downstream Nodes</span>'
            f'<span style="color:#64748B; font-weight:700;">&rarr;</span>'
            f'<span style="background:rgba(56,189,248,0.18); color:#38BDF8; border:1px solid rgba(56,189,248,0.4); padding:4px 12px; border-radius:16px; font-weight:700;">📦 COMMODITIES: {mat_count} Jeopardized</span>'
            f'</div>'
            f'</div>'
        )
        st.markdown(cascade_html, unsafe_allow_html=True)

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
                c_short = item.get("confirmed_shortage", False)
                p_aff = item.get("potentially_affected", False)
                sev_paths = item.get("severed_inbound_paths_count", 0)
                alt_status = item.get("alternative_status", "Unknown")
                badge = '<span class="badge-critical">CONFIRMED SHORTAGE</span>' if c_short else ('<span class="badge-high">POTENTIALLY AFFECTED</span>' if p_aff else '<span class="badge-low">OPERATIONAL</span>')
                border_c = "#F43F5E" if c_short else ("#F59E0B" if p_aff else "#10B981")

                shortage_cards_html.append(
                    f'<div style="background:#151F32; border:1px solid #1E293B; border-top:3px solid {border_c}; border-radius:8px; padding:12px 14px; margin-bottom:10px;">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">'
                    f'<span style="font-weight:700; color:#F8FAFC; font-size:0.95rem;">{ent_name}</span>'
                    f'{badge}'
                    f'</div>'
                    f'<div style="font-size:0.78rem; color:#94A3B8;">'
                    f'<b>Severed Inbound Paths:</b> <span style="color:#F8FAFC;">{sev_paths}</span> &nbsp;&bull;&nbsp; '
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
                <div class="section-header-desc">Mathematically selects the lowest-cost recovery plan satisfying budget, capacity, and lead-time constraints.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Financial ROI Summary
        f_row = st.columns(4)
        with f_row[0]:
            st.markdown(
                render_kpi_card("Inaction Gross Loss", gross_loss_disp, "💸", "kpi-card-glow-rose", "Unmitigated shock impact", "#FB7185" if is_unquantified else "#F8FAFC"),
                unsafe_allow_html=True,
            )
        with f_row[1]:
            st.markdown(
                render_kpi_card("Plan Expenditure", rec_cost_disp, "💳", "kpi-card-glow-cyan", "Required capital allocation", "#38BDF8"),
                unsafe_allow_html=True,
            )
        with f_row[2]:
            st.markdown(
                render_kpi_card("Net Capital Saved", est_savings_disp, "💎", "kpi-card-glow-emerald", "Capital preserved by intervention", "#34D399" if not is_unquantified else "#FB7185"),
                unsafe_allow_html=True,
            )
        with f_row[3]:
            st.markdown(
                render_kpi_card("Residual Risk", residual_loss_disp, "⚠️", "kpi-card-glow-amber", "Remaining unmitigated exposure", "#FBBF24" if not is_unquantified else "#FB7185"),
                unsafe_allow_html=True,
            )

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
                  <div style="font-size:0.75rem; color:#A7F3D0; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">Mathematically Optimal Recovery Strategy</div>
                  <div style="font-size:1.25rem; font-weight:800; color:#F0FDF4; margin:4px 0;">{', '.join(rec_titles)} <span style="font-size:0.80rem; color:#A7F3D0; font-family:monospace;">({', '.join(rec_ids)})</span></div>
                  <span style="color:#A7F3D0; font-size:0.85rem;">Plan Expenditure: <b>{total_cost_str}</b> &nbsp;|&nbsp; Net Capital Preserved: <b style="color:#6EE7B7;">{net_savings_str}</b></span>
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
                rec_badge = '<span style="background:rgba(16,185,129,0.18); color:#34D399; border:1px solid rgba(16,185,129,0.4); padding:3px 10px; border-radius:12px; font-size:0.74rem; font-weight:700;">⭐ MATHEMATICALLY RECOMMENDED</span>' if is_rec else '<span class="badge-low">FEASIBLE OPTION</span>'
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
            )
        with gov_col2:
            justification_input = st.text_area(
                "Governance Justification / Review Notes:",
                placeholder="Enter rationale for approval, or mandatory justification for rejection...",
                help="Mandatory for rejection, recommended for approval.",
                key="audit_justification_field",
                height=70,
            )

        btn_col1, btn_col2, _ = st.columns([1, 1, 2])
        with btn_col1:
            approve_clicked = st.button("Approve Recommendation", key="btn_approve_recommendation", use_container_width=True)
        with btn_col2:
            reject_clicked = st.button("Reject Recommendation", key="btn_reject_recommendation", use_container_width=True)

        if approve_clicked:
            if not active_run_id:
                st.error("No active simulation run ID found to approve.")
            else:
                success, msg, _ = audit_logger.record_decision_event(
                    run_id=active_run_id,
                    decision_type="RECOMMENDATION_APPROVED",
                    reviewer_label=reviewer_label_input,
                    justification=justification_input,
                )
                if success:
                    st.success(f"{msg}")
                    st.rerun()
                else:
                    st.error(f"{msg}")

        if reject_clicked:
            if not active_run_id:
                st.error("No active simulation run ID found to reject.")
            elif not justification_input or not justification_input.strip():
                st.error("Rejection blocked: Specific justification is required to record a rejection recommendation.")
            else:
                success, msg, _ = audit_logger.record_decision_event(
                    run_id=active_run_id,
                    decision_type="RECOMMENDATION_REJECTED",
                    reviewer_label=reviewer_label_input,
                    justification=justification_input,
                )
                if success:
                    st.warning(f"{msg}")
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
