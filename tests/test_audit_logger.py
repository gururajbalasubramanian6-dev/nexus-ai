"""
Unit and integration tests for Nexus-AI SQLite Audit Logging and Governance System.

Verifies:
  1. Two-table relational schema and index creation.
  2. Transaction safety (run + initial event atomic commit / rollback).
  3. Strict UTC ISO 8601 timestamp compliance.
  4. Handling of unquantified financial and capacity scenarios.
  5. Approval safety guards (prevent approving unquantified scenarios).
  6. Justification enforcement for rejections.
  7. Append-only event preservation (events never mutated or deleted).
  8. Accidental duplicate approval/rejection prevention.
  9. Streamlit deduplication fingerprint logic vs intentional repeated runs.
 10. Chronological descending ordering and database isolation.
"""

from __future__ import annotations

import json
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pytest

from agents.audit_logger import AuditLogger, get_utc_iso_timestamp


ISO_8601_REGEX = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z$"
)


@pytest.fixture
def memory_logger() -> AuditLogger:
    """Fixture providing an isolated in-memory AuditLogger instance."""
    return AuditLogger(db_path=":memory:")


@pytest.fixture
def sample_sim_result() -> Dict[str, Any]:
    return {
        "failed_entity": "Processor X",
        "disruption_type": "Facility Outage",
        "impact_level": "Critical",
        "affected_entities": ["Supplier A", "Supplier B"],
        "disrupted_paths": [["Processor X", "Supplier A"], ["Processor X", "Supplier B"]],
        "estimated_capacity_impact": {
            "estimated_direct_capacity_loss": 100.0,
        },
    }


@pytest.fixture
def sample_financial_result() -> Dict[str, Any]:
    return {
        "baseline_disruption_loss": 15000.0,
        "recovery_cost": 5000.0,
        "estimated_total_cost": 5000.0,
        "estimated_savings": 10000.0,
        "recommended_option_ids": ["REC_ALT_1"],
        "requires_human_approval": True,
        "is_unquantified": False,
    }


# =============================================================================
# 1. Schema & Table Architecture Tests
# =============================================================================

def test_01_two_table_schema_initialization(memory_logger: AuditLogger) -> None:
    """Verifies that nexus_audit_runs and nexus_audit_events tables and indices exist."""
    conn = memory_logger._get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('nexus_audit_runs', 'nexus_audit_events');"
    )
    tables = {row["name"] for row in cursor.fetchall()}
    assert "nexus_audit_runs" in tables
    assert "nexus_audit_events" in tables

    # Verify column presence in nexus_audit_runs
    cursor.execute("PRAGMA table_info(nexus_audit_runs);")
    cols_runs = {row["name"] for row in cursor.fetchall()}
    expected_run_cols = {
        "run_id", "created_at_utc", "scenario_entity", "disruption_type",
        "scenario_params_json", "impact_level", "affected_nodes_json",
        "severed_corridors_json", "direct_capacity_loss", "financial_baseline_loss",
        "recovery_plan_cost", "estimated_net_savings", "recommended_options_json",
        "requires_human_approval", "current_decision_status", "updated_at_utc",
        "execution_channel"
    }
    assert expected_run_cols.issubset(cols_runs)

    # Verify column presence in nexus_audit_events
    cursor.execute("PRAGMA table_info(nexus_audit_events);")
    cols_events = {row["name"] for row in cursor.fetchall()}
    expected_event_cols = {
        "event_id", "run_id", "timestamp_utc", "event_type",
        "reviewer_label", "justification", "event_details_json"
    }
    assert expected_event_cols.issubset(cols_events)


# =============================================================================
# 2. Transaction Safety & Simulation Recording Tests
# =============================================================================

def test_02_record_simulation_run_atomic_transaction(
    memory_logger: AuditLogger,
    sample_sim_result: Dict[str, Any],
    sample_financial_result: Dict[str, Any],
) -> None:
    """Verifies that simulation insertion creates 1 run record and 1 SIMULATION_EXECUTED event atomically."""
    params = {"unit_loss": 150.0, "budget": 50000.0, "max_lead_time": 14}
    run_id = memory_logger.record_simulation_run(
        scenario_entity="Processor X",
        disruption_type="Facility Outage",
        scenario_params=params,
        sim_result=sample_sim_result,
        financial_result=sample_financial_result,
        decision_exp={"requires_human_approval": True},
        execution_channel="STREAMLIT",
        actor_role="ADMIN",
    )

    assert run_id.startswith("RUN-")

    # Verify run record
    run = memory_logger.get_run_by_id(run_id)
    assert run is not None
    assert run["scenario_entity"] == "Processor X"
    assert run["impact_level"] == "Critical"
    assert run["current_decision_status"] == "PENDING_REVIEW"
    assert run["requires_human_approval"] == 1
    assert float(run["financial_baseline_loss"]) == 15000.0
    assert run["recovery_plan_cost"] == 5000.0
    assert float(run["estimated_net_savings"]) == 10000.0

    # Verify initial SIMULATION_EXECUTED event
    events = memory_logger.get_audit_events(run_id=run_id)
    assert len(events) == 1
    evt = events[0]
    assert evt["event_type"] == "SIMULATION_EXECUTED"
    assert evt["reviewer_label"] == "System (Automated Pipeline)"
    assert evt["run_id"] == run_id

    details = json.loads(evt["event_details_json"])
    assert details["event_action"] == "SIMULATION_EXECUTED"
    assert "prototype_notice" in details


def test_03_timestamp_strict_iso8601_format(
    memory_logger: AuditLogger,
    sample_sim_result: Dict[str, Any],
    sample_financial_result: Dict[str, Any],
) -> None:
    """Validates that timestamps stored in runs and events follow UTC ISO 8601 with Z suffix."""
    run_id = memory_logger.record_simulation_run(
        scenario_entity="Processor X",
        disruption_type="Facility Outage",
        scenario_params={},
        sim_result=sample_sim_result,
        financial_result=sample_financial_result,
        actor_role="ADMIN",
    )

    run = memory_logger.get_run_by_id(run_id)
    assert run is not None
    assert ISO_8601_REGEX.match(run["created_at_utc"])
    assert ISO_8601_REGEX.match(run["updated_at_utc"])

    events = memory_logger.get_audit_events(run_id=run_id)
    assert len(events) >= 1
    assert ISO_8601_REGEX.match(events[0]["timestamp_utc"])


# =============================================================================
# 3. Unquantified Scenario Handling
# =============================================================================

def test_04_record_unquantified_scenario_gracefully(
    memory_logger: AuditLogger,
) -> None:
    """Verifies that scenarios with missing capacity and unquantified loss store cleanly."""
    unquantified_sim = {
        "failed_entity": "Supplier A",
        "disruption_type": "Production Line Failure",
        "impact_level": "Medium",
        "affected_entities": [],
        "disrupted_paths": [["Supplier A"]],
        "estimated_capacity_impact": {
            "estimated_direct_capacity_loss": None,
        },
    }
    unquantified_fin = {
        "is_unquantified": True,
        "baseline_disruption_loss": None,
        "recovery_cost": 0.0,
        "estimated_total_cost": None,
        "estimated_savings": None,
        "recommended_option_ids": [],
        "requires_human_approval": True,
    }

    run_id = memory_logger.record_simulation_run(
        scenario_entity="Supplier A",
        disruption_type="Production Line Failure",
        scenario_params={"unit_loss": 150.0},
        sim_result=unquantified_sim,
        financial_result=unquantified_fin,
        actor_role="ADMIN",
    )

    run = memory_logger.get_run_by_id(run_id)
    assert run is not None
    assert run["direct_capacity_loss"] == "UNQUANTIFIED"
    assert run["financial_baseline_loss"] == "UNQUANTIFIED"
    assert run["estimated_net_savings"] == "UNQUANTIFIED"
    assert run["recovery_plan_cost"] is None
    assert run["current_decision_status"] == "PENDING_REVIEW"


# =============================================================================
# 4. Approval Safety Guards
# =============================================================================

def test_05_unquantified_scenarios_cannot_be_approved(
    memory_logger: AuditLogger,
) -> None:
    """Approval safety guard: Unquantified scenarios MUST NOT be approved."""
    unquantified_sim = {
        "failed_entity": "Supplier A",
        "disruption_type": "Production Line Failure",
        "impact_level": "Medium",
        "affected_entities": [],
        "disrupted_paths": [],
        "estimated_capacity_impact": {"estimated_direct_capacity_loss": None},
    }
    unquantified_fin = {
        "is_unquantified": True,
        "baseline_disruption_loss": None,
        "recommended_option_ids": [],
    }

    run_id = memory_logger.record_simulation_run(
        scenario_entity="Supplier A",
        disruption_type="Production Line Failure",
        scenario_params={},
        sim_result=unquantified_sim,
        financial_result=unquantified_fin,
        actor_role="ADMIN",
    )

    # Attempt to approve
    success, msg, status = memory_logger.record_decision_event(
        run_id=run_id,
        decision_type="RECOMMENDATION_APPROVED",
        reviewer_label="Test Reviewer",
        justification="Attempting to approve unquantified plan",
        actor_role="ADMIN",
    )

    assert not success
    assert "Approval prohibited" in msg
    assert "UNQUANTIFIED" in msg
    assert status == "PENDING_REVIEW"

    # Verify run remains PENDING_REVIEW
    run = memory_logger.get_run_by_id(run_id)
    assert run["current_decision_status"] == "PENDING_REVIEW"

    # Verify NO approval event was appended
    events = memory_logger.get_audit_events(run_id=run_id)
    assert len(events) == 1
    assert events[0]["event_type"] == "SIMULATION_EXECUTED"


def test_06_rejection_requires_justification(
    memory_logger: AuditLogger,
    sample_sim_result: Dict[str, Any],
    sample_financial_result: Dict[str, Any],
) -> None:
    """Verifies that rejecting a recommendation requires non-empty justification."""
    run_id = memory_logger.record_simulation_run(
        scenario_entity="Processor X",
        disruption_type="Facility Outage",
        scenario_params={},
        sim_result=sample_sim_result,
        financial_result=sample_financial_result,
        actor_role="ADMIN",
    )

    # Attempt rejection with empty justification
    success, msg, status = memory_logger.record_decision_event(
        run_id=run_id,
        decision_type="RECOMMENDATION_REJECTED",
        reviewer_label="Reviewer",
        justification="",
        actor_role="ADMIN",
    )
    assert not success
    assert "justification is required" in msg

    # Provide valid justification
    success2, msg2, status2 = memory_logger.record_decision_event(
        run_id=run_id,
        decision_type="RECOMMENDATION_REJECTED",
        reviewer_label="Reviewer",
        justification="Rejected due to operational maintenance window conflict.",
        actor_role="ADMIN",
    )
    assert success2
    assert status2 == "REJECTED"

    run = memory_logger.get_run_by_id(run_id)
    assert run["current_decision_status"] == "REJECTED"


def test_07_prevent_accidental_duplicate_decisions(
    memory_logger: AuditLogger,
    sample_sim_result: Dict[str, Any],
    sample_financial_result: Dict[str, Any],
) -> None:
    """Verifies that approving or rejecting twice returns a duplicate prevented error."""
    run_id = memory_logger.record_simulation_run(
        scenario_entity="Processor X",
        disruption_type="Facility Outage",
        scenario_params={},
        sim_result=sample_sim_result,
        financial_result=sample_financial_result,
        actor_role="ADMIN",
    )

    # First approval succeeds
    ok1, _, status1 = memory_logger.record_decision_event(
        run_id=run_id,
        decision_type="RECOMMENDATION_APPROVED",
        reviewer_label="Supply Chain Lead",
        justification="Plan satisfies all constraints.",
        actor_role="ADMIN",
    )
    assert ok1
    assert status1 == "APPROVED"

    # Second approval fails duplicate check
    ok2, msg2, status2 = memory_logger.record_decision_event(
        run_id=run_id,
        decision_type="RECOMMENDATION_APPROVED",
        reviewer_label="Supply Chain Lead",
        justification="Accidental double click",
        actor_role="ADMIN",
    )
    assert not ok2
    assert "duplicate prevented" in msg2.lower()
    assert status2 == "APPROVED"


# =============================================================================
# 5. Append-Only Event History Tests
# =============================================================================

def test_08_events_are_strictly_append_only(
    memory_logger: AuditLogger,
    sample_sim_result: Dict[str, Any],
    sample_financial_result: Dict[str, Any],
) -> None:
    """
    Verifies that when decisions occur, new rows are appended to nexus_audit_events
    and existing events are never overwritten or deleted.
    """
    run_id = memory_logger.record_simulation_run(
        scenario_entity="Processor X",
        disruption_type="Facility Outage",
        scenario_params={},
        sim_result=sample_sim_result,
        financial_result=sample_financial_result,
        actor_role="ADMIN",
    )

    # Initially 1 event
    events_step1 = memory_logger.get_audit_events(run_id=run_id)
    assert len(events_step1) == 1
    evt1_id = events_step1[0]["event_id"]
    evt1_type = events_step1[0]["event_type"]

    # Now approve
    memory_logger.record_decision_event(
        run_id=run_id,
        decision_type="RECOMMENDATION_APPROVED",
        reviewer_label="Risk Director",
        justification="Approved for demonstration.",
        actor_role="ADMIN",
    )

    # Now should have 2 events
    events_step2 = memory_logger.get_audit_events(run_id=run_id)
    assert len(events_step2) == 2

    # Events ordered descending: newest is index 0
    assert events_step2[0]["event_type"] == "RECOMMENDATION_APPROVED"
    assert events_step2[0]["reviewer_label"] == "Risk Director"

    # Original event is preserved at index 1
    assert events_step2[1]["event_id"] == evt1_id
    assert events_step2[1]["event_type"] == evt1_type


# =============================================================================
# 6. Streamlit Deduplication & Intentional Rerun Tests
# =============================================================================

def test_09_deduplication_fingerprint_vs_intentional_rerun(
    memory_logger: AuditLogger,
    sample_sim_result: Dict[str, Any],
    sample_financial_result: Dict[str, Any],
) -> None:
    """
    Simulates the Streamlit rerun deduplication logic:
      - Repeated execution with identical parameters reuses active run_id.
      - User clicking 'Re-run Simulation' creates a distinct new run_id.
    """
    session_state: Dict[str, Any] = {}

    def simulate_app_cycle(params: Dict[str, Any], force_new: bool = False) -> str:
        fingerprint = f"{params['entity']}|{params['type']}|{params['budget']:.1f}"

        if not force_new and session_state.get("active_fingerprint") == fingerprint:
            # Ordinary rerun (e.g. user browsing tabs): reuse existing run_id
            return session_state["active_run_id"]

        # Parameter changed or intentional re-run:
        new_run_id = memory_logger.record_simulation_run(
            scenario_entity=params["entity"],
            disruption_type=params["type"],
            scenario_params=params,
            sim_result=sample_sim_result,
            financial_result=sample_financial_result,
            actor_role="ADMIN",
        )
        session_state["active_fingerprint"] = fingerprint
        session_state["active_run_id"] = new_run_id
        return new_run_id

    base_params = {"entity": "Processor X", "type": "Facility Outage", "budget": 50000.0}

    # Cycle 1: First simulation
    run_1 = simulate_app_cycle(base_params)

    # Cycle 2: User switches tab (ordinary Streamlit rerun) -> Must NOT create new run
    run_2 = simulate_app_cycle(base_params)
    assert run_1 == run_2

    # Verify only 1 run exists in SQLite
    runs_in_db = memory_logger.get_audit_runs()
    assert len(runs_in_db) == 1

    # Cycle 3: User intentionally clicks 'Re-run Simulation' with same parameters
    run_3 = simulate_app_cycle(base_params, force_new=True)
    assert run_3 != run_1

    # Verify now 2 distinct runs exist in SQLite
    runs_in_db_after = memory_logger.get_audit_runs()
    assert len(runs_in_db_after) == 2


# =============================================================================
# 7. Database Isolation Test
# =============================================================================

def test_10_database_isolation_does_not_affect_production_file(
    sample_sim_result: Dict[str, Any],
    sample_financial_result: Dict[str, Any],
) -> None:
    """Verifies that tests using temporary databases or in-memory DB never touch nexus_audit.db."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_db_path = str(Path(tmp_dir) / "test_isolated.db")
        temp_logger = AuditLogger(db_path=temp_db_path)

        run_id = temp_logger.record_simulation_run(
            scenario_entity="Processor X",
            disruption_type="Facility Outage",
            scenario_params={},
            sim_result=sample_sim_result,
            financial_result=sample_financial_result,
            actor_role="ADMIN",
        )

        assert Path(temp_db_path).exists()
        assert temp_logger.get_run_by_id(run_id) is not None


def test_11_transaction_safety_rollback_on_event_failure(
    sample_sim_result: Dict[str, Any],
    sample_financial_result: Dict[str, Any],
) -> None:
    """
    Transaction Safety: If the event insertion fails, the entire transaction rolls back,
    ensuring no orphan row is committed to nexus_audit_runs.
    """
    logger = AuditLogger(db_path=":memory:")
    conn = logger._get_connection()

    # Install a trigger that aborts insertion into nexus_audit_events
    with conn:
        conn.execute(
            """
            CREATE TRIGGER abort_event_insert
            BEFORE INSERT ON nexus_audit_events
            BEGIN
                SELECT RAISE(ABORT, 'Simulated database event failure');
            END;
            """
        )

    # Calling record_simulation_run should fail with IntegrityError
    with pytest.raises(sqlite3.IntegrityError, match="Simulated database event failure"):
        logger.record_simulation_run(
            scenario_entity="Processor X",
            disruption_type="Facility Outage",
            scenario_params={},
            sim_result=sample_sim_result,
            financial_result=sample_financial_result,
            actor_role="ADMIN",
        )

    # Verify that nexus_audit_runs is completely empty due to rollback
    runs = logger.get_audit_runs()
    assert len(runs) == 0

    events = logger.get_audit_events()
    assert len(events) == 0


def test_12_decision_finality_locks_run_against_conflicting_decisions(
    memory_logger: AuditLogger,
    sample_sim_result: Dict[str, Any],
    sample_financial_result: Dict[str, Any],
) -> None:
    """
    Decision Finality: Once a run is decided, both identical and opposite decisions are blocked,
    and no further events are appended to nexus_audit_events.
    """
    # 1. Test Approved Run cannot be Rejected
    run_1 = memory_logger.record_simulation_run(
        scenario_entity="Processor X",
        disruption_type="Facility Outage",
        scenario_params={},
        sim_result=sample_sim_result,
        financial_result=sample_financial_result,
        actor_role="ADMIN",
    )
    ok_app, _, _ = memory_logger.record_decision_event(
        run_id=run_1,
        decision_type="RECOMMENDATION_APPROVED",
        reviewer_label="Lead",
        justification="Initial approval",
        actor_role="ADMIN",
    )
    assert ok_app
    assert len(memory_logger.get_audit_events(run_id=run_1)) == 2

    # Attempt to reject the already-approved run
    ok_opp, msg_opp, status_opp = memory_logger.record_decision_event(
        run_id=run_1,
        decision_type="RECOMMENDATION_REJECTED",
        reviewer_label="Different Reviewer",
        justification="Trying to override approval with rejection",
        actor_role="ADMIN",
    )
    assert not ok_opp
    assert "Decision finality enforced" in msg_opp
    assert status_opp == "APPROVED"
    assert len(memory_logger.get_audit_events(run_id=run_1)) == 2

    # 2. Test Rejected Run cannot be Approved
    run_2 = memory_logger.record_simulation_run(
        scenario_entity="Processor X",
        disruption_type="Facility Outage",
        scenario_params={},
        sim_result=sample_sim_result,
        financial_result=sample_financial_result,
        actor_role="ADMIN",
    )
    ok_rej, _, _ = memory_logger.record_decision_event(
        run_id=run_2,
        decision_type="RECOMMENDATION_REJECTED",
        reviewer_label="Lead",
        justification="Initial rejection due to maintenance",
        actor_role="ADMIN",
    )
    assert ok_rej
    assert len(memory_logger.get_audit_events(run_id=run_2)) == 2

    # Attempt to approve the already-rejected run
    ok_opp2, msg_opp2, status_opp2 = memory_logger.record_decision_event(
        run_id=run_2,
        decision_type="RECOMMENDATION_APPROVED",
        reviewer_label="Different Reviewer",
        justification="Trying to override rejection with approval",
        actor_role="ADMIN",
    )
    assert not ok_opp2
    assert "Decision finality enforced" in msg_opp2
    assert status_opp2 == "REJECTED"
    assert len(memory_logger.get_audit_events(run_id=run_2)) == 2


def test_13_mandatory_approval_requires_justification(
    memory_logger: AuditLogger,
    sample_sim_result: Dict[str, Any],
    sample_financial_result: Dict[str, Any],
) -> None:
    """
    Approval Safety: When requires_human_approval is flagged (1), approval cannot be granted
    with an empty justification; the run must remain in PENDING_REVIEW until documented.
    """
    run_id = memory_logger.record_simulation_run(
        scenario_entity="Processor X",
        disruption_type="Facility Outage",
        scenario_params={},
        sim_result=sample_sim_result,
        financial_result=sample_financial_result,
        decision_exp={"requires_human_approval": True},
        actor_role="ADMIN",
    )

    run = memory_logger.get_run_by_id(run_id)
    assert run["requires_human_approval"] == 1
    assert run["current_decision_status"] == "PENDING_REVIEW"

    # Attempt approval with blank justification
    ok_blank, msg_blank, status_blank = memory_logger.record_decision_event(
        run_id=run_id,
        decision_type="RECOMMENDATION_APPROVED",
        reviewer_label="Reviewer",
        justification="",
        actor_role="ADMIN",
    )
    assert not ok_blank
    assert "Approval blocked: Mandatory human approval is required" in msg_blank
    assert status_blank == "PENDING_REVIEW"

    # Run remains PENDING_REVIEW in database
    run_after = memory_logger.get_run_by_id(run_id)
    assert run_after["current_decision_status"] == "PENDING_REVIEW"
    assert len(memory_logger.get_audit_events(run_id=run_id)) == 1

    # Now approve with documented justification
    ok_doc, msg_doc, status_doc = memory_logger.record_decision_event(
        run_id=run_id,
        decision_type="RECOMMENDATION_APPROVED",
        reviewer_label="Supply Chain Lead",
        justification="Verified capacity with Tier-2 plant manager; authorized allocation.",
        actor_role="ADMIN",
    )
    assert ok_doc
    assert status_doc == "APPROVED"
    assert len(memory_logger.get_audit_events(run_id=run_id)) == 2


def test_14_database_triggers_prevent_update_and_delete_on_events(
    memory_logger: AuditLogger,
    sample_sim_result: Dict[str, Any],
    sample_financial_result: Dict[str, Any],
) -> None:
    """
    Database-Level Append-Only Protection:
    Triggers must abort any UPDATE or DELETE statement executed against nexus_audit_events.
    """
    run_id = memory_logger.record_simulation_run(
        scenario_entity="Processor X",
        disruption_type="Facility Outage",
        scenario_params={},
        sim_result=sample_sim_result,
        financial_result=sample_financial_result,
        actor_role="ADMIN",
    )

    conn = memory_logger._get_connection()
    events = memory_logger.get_audit_events(run_id=run_id)
    assert len(events) == 1
    event_id = events[0]["event_id"]

    # 1. Attempt UPDATE on nexus_audit_events
    with pytest.raises(sqlite3.IntegrityError, match="Append-only violation: UPDATE operations are strictly prohibited"):
        with conn:
            conn.execute(
                "UPDATE nexus_audit_events SET reviewer_label = 'Malicious Override' WHERE event_id = ?;",
                (event_id,),
            )

    # 2. Attempt DELETE on nexus_audit_events
    with pytest.raises(sqlite3.IntegrityError, match="Append-only violation: DELETE operations are strictly prohibited"):
        with conn:
            conn.execute(
                "DELETE FROM nexus_audit_events WHERE event_id = ?;",
                (event_id,),
            )

    # Verify event remains completely unaltered
    fresh_events = memory_logger.get_audit_events(run_id=run_id)
    assert len(fresh_events) == 1
    assert fresh_events[0]["reviewer_label"] == "System (Automated Pipeline)"


def test_15_intentional_rerun_clears_justification_and_preserves_reviewer(
    memory_logger: AuditLogger,
    sample_sim_result: Dict[str, Any],
    sample_financial_result: Dict[str, Any],
) -> None:
    """
    UI State Management:
      - Ordinary reruns (tab change, metric inspection) preserve both justification and reviewer label.
      - Intentional new simulation run (Re-run Simulation button) clears previous justification to ''
        while preserving the reviewer label.
      - Historical audit records remain completely intact.
    """
    session_state: Dict[str, Any] = {
        "audit_reviewer_label_field": "Dr. Smith (Lead Risk Officer)",
        "audit_justification_field": "Previous justification notes from earlier run.",
        "active_sim_fingerprint": None,
        "active_run_id": None,
        "force_new_run": False,
    }

    def simulate_dashboard_cycle(params: Dict[str, Any], click_rerun_btn: bool = False) -> str:
        fingerprint = f"{params['entity']}|{params['type']}|{params['budget']:.1f}"

        # If user clicked "Re-run Simulation" button in sidebar:
        if click_rerun_btn:
            session_state["force_new_run"] = True
            session_state["audit_justification_field"] = ""

        force_new = session_state.get("force_new_run", False)

        if force_new or session_state.get("active_sim_fingerprint") != fingerprint:
            if force_new:
                session_state["audit_justification_field"] = ""
            new_run_id = memory_logger.record_simulation_run(
                scenario_entity=params["entity"],
                disruption_type=params["type"],
                scenario_params=params,
                sim_result=sample_sim_result,
                financial_result=sample_financial_result,
                actor_role="ADMIN",
            )
            session_state["active_sim_fingerprint"] = fingerprint
            session_state["active_run_id"] = new_run_id
            session_state["force_new_run"] = False
            return new_run_id
        else:
            return session_state["active_run_id"]

    params = {"entity": "Processor X", "type": "Facility Outage", "budget": 50000.0}

    # 1. First Run
    run_1 = simulate_dashboard_cycle(params)
    assert run_1.startswith("RUN-")
    # Simulate user typing a justification
    session_state["audit_justification_field"] = "Approved for Q4 production commitments."

    # 2. Normal Streamlit Rerun (e.g. user browsing tabs in Section F or Section G)
    run_2 = simulate_dashboard_cycle(params)
    assert run_2 == run_1
    assert session_state["audit_justification_field"] == "Approved for Q4 production commitments."
    assert session_state["audit_reviewer_label_field"] == "Dr. Smith (Lead Risk Officer)"

    # 3. Intentional Re-run Simulation button clicked
    run_3 = simulate_dashboard_cycle(params, click_rerun_btn=True)
    assert run_3 != run_1
    # Justification must be cleared
    assert session_state["audit_justification_field"] == ""
    # Reviewer label must be preserved
    assert session_state["audit_reviewer_label_field"] == "Dr. Smith (Lead Risk Officer)"

    # 4. Verify historical runs in SQLite remain completely intact
    runs = memory_logger.get_audit_runs()
    assert len(runs) == 2
    events = memory_logger.get_audit_events()
    assert len(events) == 2
