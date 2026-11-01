"""
Audit Logging and Governance Module for Nexus-AI.

Provides persistent, temporal audit tracking using a local SQLite database with
a two-table relational architecture:
  1. nexus_audit_runs: Exactly one record per simulation run.
  2. nexus_audit_events: Append-only immutable ledger of all automated and human decisions.

Design Principles:
  - Zero external dependencies (uses standard-library sqlite3, uuid, datetime, json).
  - Transaction safety: Run insertion and initial SIMULATION_EXECUTED event are committed together.
  - Append-only event history: Existing events are never updated or deleted.
  - Approval safety: Prevents approval of unquantified scenarios or runs missing throughput data.
  - Deduplication: Works with app-level session state fingerprints to avoid rerun spam.
  - Disclaimed prototype scope: Explicitly tracks advisory human decisions without real-world execution claims.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def get_utc_iso_timestamp() -> str:
    """Returns current UTC timestamp in strict ISO 8601 format with Z suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class AuditRunRecord:
    run_id: str
    created_at_utc: str
    scenario_entity: str
    disruption_type: str
    scenario_params_json: str
    impact_level: str
    affected_nodes_json: str
    severed_corridors_json: str
    direct_capacity_loss: str
    financial_baseline_loss: str
    recovery_plan_cost: Optional[float]
    estimated_net_savings: str
    recommended_options_json: str
    requires_human_approval: int
    current_decision_status: str
    updated_at_utc: str
    execution_channel: str = "STREAMLIT"


@dataclass
class AuditEventRecord:
    event_id: str
    run_id: str
    timestamp_utc: str
    event_type: str
    reviewer_label: str
    justification: str
    event_details_json: str


class AuditLogger:
    """
    Manages local SQLite audit logging for Nexus-AI.
    Thread-safe and compatible with Streamlit and CLI environments.
    """

    def __init__(self, db_path: str = "nexus_audit.db") -> None:
        self.db_path = db_path
        self._is_memory = (db_path == ":memory:")
        self._shared_mem_conn: Optional[sqlite3.Connection] = None

        if self._is_memory:
            # Persistent connection for in-memory testing
            self._shared_mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_mem_conn.row_factory = sqlite3.Row
            self._init_schema(self._shared_mem_conn)
        else:
            # Ensure directory exists for file path
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = self._get_connection()
            try:
                self._init_schema(conn)
            finally:
                conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        if self._is_memory and self._shared_mem_conn is not None:
            return self._shared_mem_conn
        conn = sqlite3.connect(self.db_path, timeout=20.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        """Initializes the two relational audit tables and supporting indices."""
        with conn:
            cursor = conn.cursor()
            # 1. nexus_audit_runs: One record per simulation run
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS nexus_audit_runs (
                    run_id                  TEXT PRIMARY KEY,
                    created_at_utc          TEXT NOT NULL,
                    scenario_entity         TEXT NOT NULL,
                    disruption_type         TEXT NOT NULL,
                    scenario_params_json    TEXT NOT NULL,
                    impact_level            TEXT NOT NULL,
                    affected_nodes_json     TEXT NOT NULL,
                    severed_corridors_json  TEXT NOT NULL,
                    direct_capacity_loss    TEXT NOT NULL,
                    financial_baseline_loss TEXT NOT NULL,
                    recovery_plan_cost      REAL,
                    estimated_net_savings   TEXT NOT NULL,
                    recommended_options_json TEXT NOT NULL,
                    requires_human_approval INTEGER NOT NULL,
                    current_decision_status TEXT NOT NULL DEFAULT 'PENDING_REVIEW',
                    updated_at_utc          TEXT NOT NULL,
                    execution_channel       TEXT NOT NULL DEFAULT 'STREAMLIT'
                );
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_created ON nexus_audit_runs (created_at_utc DESC);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_entity ON nexus_audit_runs (scenario_entity);"
            )

            # 2. nexus_audit_events: Append-only immutable event history
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS nexus_audit_events (
                    event_id            TEXT PRIMARY KEY,
                    run_id              TEXT NOT NULL,
                    timestamp_utc       TEXT NOT NULL,
                    event_type          TEXT NOT NULL,
                    reviewer_label      TEXT NOT NULL,
                    justification       TEXT DEFAULT '',
                    event_details_json  TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES nexus_audit_runs(run_id)
                );
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_run_id ON nexus_audit_events (run_id);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON nexus_audit_events (timestamp_utc DESC);"
            )

            # 3. Database-level append-only protection on nexus_audit_events
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS prevent_audit_events_update
                BEFORE UPDATE ON nexus_audit_events
                BEGIN
                    SELECT RAISE(ABORT, 'Append-only violation: UPDATE operations are strictly prohibited on nexus_audit_events.');
                END;
                """
            )
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS prevent_audit_events_delete
                BEFORE DELETE ON nexus_audit_events
                BEGIN
                    SELECT RAISE(ABORT, 'Append-only violation: DELETE operations are strictly prohibited on nexus_audit_events.');
                END;
                """
            )

    def record_simulation_run(
        self,
        scenario_entity: str,
        disruption_type: str,
        scenario_params: Dict[str, Any],
        sim_result: Dict[str, Any],
        financial_result: Dict[str, Any],
        decision_exp: Optional[Dict[str, Any]] = None,
        execution_channel: str = "STREAMLIT",
    ) -> str:
        """
        Atomically records a new simulation run in nexus_audit_runs and appends the
        initial SIMULATION_EXECUTED event to nexus_audit_events within one transaction.
        Rolls back both operations if an error occurs.
        """
        now_utc = get_utc_iso_timestamp()
        run_suffix = uuid.uuid4().hex[:6].upper()
        timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id = f"RUN-{timestamp_slug}-{run_suffix}"
        event_id = f"EVT-{timestamp_slug}-{run_suffix}"

        # Detect unquantified financial conditions
        is_unquantified = bool(
            financial_result.get("is_unquantified", False)
            or financial_result.get("baseline_disruption_loss") is None
            or str(financial_result.get("baseline_disruption_loss", "")).upper() == "UNQUANTIFIED"
        )

        # Detect direct capacity offline
        cap_impact = sim_result.get("estimated_capacity_impact", {})
        direct_cap = cap_impact.get("estimated_direct_capacity_loss")
        direct_cap_str = (
            "UNQUANTIFIED"
            if direct_cap is None
            else f"{float(direct_cap):.2f}"
        )

        # Financial values
        if is_unquantified:
            baseline_loss_str = "UNQUANTIFIED"
            plan_cost = None
            savings_str = "UNQUANTIFIED"
        else:
            base_loss = financial_result.get("baseline_disruption_loss")
            baseline_loss_str = f"{float(base_loss):.2f}" if base_loss is not None else "UNQUANTIFIED"
            tot_cost = financial_result.get("estimated_total_cost")
            if tot_cost is None:
                tot_cost = financial_result.get("recovery_cost")
            plan_cost = float(tot_cost) if tot_cost is not None else 0.0
            sav = financial_result.get("estimated_savings")
            savings_str = f"{float(sav):.2f}" if sav is not None else "UNQUANTIFIED"

        # Requires approval determination
        req_approval = False
        if is_unquantified:
            req_approval = True
        elif decision_exp and decision_exp.get("requires_human_approval"):
            req_approval = True
        elif financial_result.get("requires_human_approval"):
            req_approval = True

        # Initial status:
        # If unquantified or approval required, status is PENDING_REVIEW.
        # Otherwise, AUTO_RECOMMENDED.
        initial_status = "PENDING_REVIEW" if req_approval else "AUTO_RECOMMENDED"

        # Corridors and entities
        affected_entities = sim_result.get("affected_entities", [])
        raw_paths = sim_result.get("disrupted_paths") or sim_result.get("affected_paths") or []
        corridors = [p for p in raw_paths if isinstance(p, (list, tuple))]
        rec_options = financial_result.get("recommended_option_ids", [])

        # Details for initial SIMULATION_EXECUTED event
        initial_event_details = {
            "event_action": "SIMULATION_EXECUTED",
            "execution_channel": execution_channel,
            "impact_level": sim_result.get("impact_level", "Unknown"),
            "direct_capacity_loss": direct_cap_str,
            "financial_baseline_loss": baseline_loss_str,
            "is_unquantified": is_unquantified,
            "requires_human_approval": req_approval,
            "recommended_options": rec_options,
            "affected_count": len(affected_entities),
            "corridors_count": len(corridors),
            "prototype_notice": (
                "Automated multi-agent resilience simulation executed. All data retrieved "
                "under read-only SAP HANA protocol. No procurement or physical operations dispatched."
            ),
        }

        conn = self._get_connection()
        try:
            # Transaction block: commits on exit, rolls back on exception
            with conn:
                cursor = conn.cursor()
                # 1. Insert run record
                cursor.execute(
                    """
                    INSERT INTO nexus_audit_runs (
                        run_id, created_at_utc, scenario_entity, disruption_type,
                        scenario_params_json, impact_level, affected_nodes_json,
                        severed_corridors_json, direct_capacity_loss,
                        financial_baseline_loss, recovery_plan_cost,
                        estimated_net_savings, recommended_options_json,
                        requires_human_approval, current_decision_status,
                        updated_at_utc, execution_channel
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        run_id,
                        now_utc,
                        scenario_entity,
                        disruption_type,
                        json.dumps(scenario_params),
                        sim_result.get("impact_level", "Medium"),
                        json.dumps(affected_entities),
                        json.dumps(corridors),
                        direct_cap_str,
                        baseline_loss_str,
                        plan_cost,
                        savings_str,
                        json.dumps(rec_options),
                        1 if req_approval else 0,
                        initial_status,
                        now_utc,
                        execution_channel,
                    ),
                )

                # 2. Append initial event
                cursor.execute(
                    """
                    INSERT INTO nexus_audit_events (
                        event_id, run_id, timestamp_utc, event_type,
                        reviewer_label, justification, event_details_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        event_id,
                        run_id,
                        now_utc,
                        "SIMULATION_EXECUTED",
                        "System (Automated Pipeline)",
                        "Automated multi-agent resilience simulation completed.",
                        json.dumps(initial_event_details),
                    ),
                )
        finally:
            if not self._is_memory:
                conn.close()

        return run_id

    def record_decision_event(
        self,
        run_id: str,
        decision_type: str,
        reviewer_label: str,
        justification: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, str]:
        """
        Records a human governance decision (approval or rejection) in an append-only manner.

        Safeguards:
          - Prevents approval of unquantified scenarios.
          - Rejects duplicate approval/rejection requests for already decided runs.
          - Mandates justification for rejections.
          - Never updates or deletes existing records in nexus_audit_events.
          - Only updates current_decision_status in nexus_audit_runs.
          - Explicitly attaches prototype advisory disclaimer.

        Returns:
          (success: bool, message: str, current_status: str)
        """
        valid_types = {"RECOMMENDATION_APPROVED", "RECOMMENDATION_REJECTED"}
        if decision_type not in valid_types:
            return False, f"Invalid decision type: '{decision_type}'. Must be one of {valid_types}.", "UNKNOWN"

        clean_label = (reviewer_label or "").strip()
        if not clean_label:
            clean_label = "Prototype Reviewer (Supply Chain Risk Analyst)"

        clean_justification = (justification or "").strip()

        # Mandate justification for rejection
        if decision_type == "RECOMMENDATION_REJECTED" and not clean_justification:
            return False, "Rejection blocked: Specific justification is required to record a rejection recommendation.", "PENDING_REVIEW"

        now_utc = get_utc_iso_timestamp()
        event_id = f"EVT-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"

        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT run_id, current_decision_status, financial_baseline_loss,
                           recommended_options_json, requires_human_approval, scenario_entity
                    FROM nexus_audit_runs
                    WHERE run_id = ?;
                    """,
                    (run_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return False, f"Simulation run ID '{run_id}' not found in audit store.", "NOT_FOUND"

                current_status = row["current_decision_status"]
                base_loss_str = row["financial_baseline_loss"]
                scenario_entity = row["scenario_entity"]
                req_approval = int(row["requires_human_approval"] or 0)
                rec_opts_json = row["recommended_options_json"]
                try:
                    rec_opts = json.loads(rec_opts_json)
                except Exception:
                    rec_opts = []

                # DECISION FINALITY: Once APPROVED or REJECTED, prevent any further decision (same or opposite)
                if current_status in ("APPROVED", "REJECTED"):
                    return (
                        False,
                        f"Decision finality enforced: Run '{run_id}' already has a final decision ('{current_status}'). "
                        "Accidental duplicate prevented: Further approval or rejection decisions are prohibited.",
                        current_status,
                    )

                # APPROVAL SAFETY GUARD: Unquantified scenarios or missing throughput data cannot be approved
                if decision_type == "RECOMMENDATION_APPROVED":
                    if base_loss_str == "UNQUANTIFIED" or not rec_opts:
                        reason = (
                            f"Approval prohibited: Disrupted throughput for '{scenario_entity}' is UNQUANTIFIED "
                            "in SAP HANA, or no viable recovery options exist. Capital allocation cannot proceed "
                            "without confirmed operational production volume. Run remains in PENDING_REVIEW."
                        )
                        return False, reason, current_status

                    # Mandatory Human Approval Rule: If mandatory approval is flagged, justification is strictly required
                    if req_approval == 1 and not clean_justification:
                        reason = (
                            f"Approval blocked: Mandatory human approval is required for '{scenario_entity}' prior to "
                            "capital allocation. A documented justification must be provided to record approval. "
                            "Run remains in PENDING_REVIEW."
                        )
                        return False, reason, current_status

                new_status = "APPROVED" if decision_type == "RECOMMENDATION_APPROVED" else "REJECTED"

                event_details = {
                    "decision": new_status,
                    "previous_status": current_status,
                    "target_entity": scenario_entity,
                    "prototype_notice": (
                        "ADVISORY GOVERNANCE RECORD: This decision is recorded for compliance audit and "
                        "hackathon demonstration. It does NOT authorize financial disbursements, release "
                        "purchase orders, or execute logistics rerouting in live enterprise systems."
                    ),
                    "context_metadata": metadata or {},
                }

                # 1. Update ONLY the run state in nexus_audit_runs
                cursor.execute(
                    """
                    UPDATE nexus_audit_runs
                    SET current_decision_status = ?, updated_at_utc = ?
                    WHERE run_id = ?;
                    """,
                    (new_status, now_utc, run_id),
                )

                # 2. Append new event to nexus_audit_events (NEVER updates or deletes previous events)
                cursor.execute(
                    """
                    INSERT INTO nexus_audit_events (
                        event_id, run_id, timestamp_utc, event_type,
                        reviewer_label, justification, event_details_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        event_id,
                        run_id,
                        now_utc,
                        decision_type,
                        clean_label,
                        clean_justification,
                        json.dumps(event_details),
                    ),
                )

            return True, f"Decision '{new_status}' successfully recorded in audit log.", new_status
        finally:
            if not self._is_memory:
                conn.close()

    def get_audit_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves historical simulation runs ordered descending by creation timestamp."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT run_id, created_at_utc, scenario_entity, disruption_type,
                       scenario_params_json, impact_level, affected_nodes_json,
                       severed_corridors_json, direct_capacity_loss,
                       financial_baseline_loss, recovery_plan_cost,
                       estimated_net_savings, recommended_options_json,
                       requires_human_approval, current_decision_status,
                       updated_at_utc, execution_channel
                FROM nexus_audit_runs
                ORDER BY created_at_utc DESC
                LIMIT ?;
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            if not self._is_memory:
                conn.close()

    def get_audit_events(self, run_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieves immutable historical events ordered descending by event timestamp.
        Optionally filters by run_id.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if run_id:
                cursor.execute(
                    """
                    SELECT event_id, run_id, timestamp_utc, event_type,
                           reviewer_label, justification, event_details_json
                    FROM nexus_audit_events
                    WHERE run_id = ?
                    ORDER BY timestamp_utc DESC
                    LIMIT ?;
                    """,
                    (run_id, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT event_id, run_id, timestamp_utc, event_type,
                           reviewer_label, justification, event_details_json
                    FROM nexus_audit_events
                    ORDER BY timestamp_utc DESC
                    LIMIT ?;
                    """,
                    (limit,),
                )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            if not self._is_memory:
                conn.close()

    def get_run_by_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Fetches full state of a single simulation run by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT run_id, created_at_utc, scenario_entity, disruption_type,
                       scenario_params_json, impact_level, affected_nodes_json,
                       severed_corridors_json, direct_capacity_loss,
                       financial_baseline_loss, recovery_plan_cost,
                       estimated_net_savings, recommended_options_json,
                       requires_human_approval, current_decision_status,
                       updated_at_utc, execution_channel
                FROM nexus_audit_runs
                WHERE run_id = ?;
                """,
                (run_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            if not self._is_memory:
                conn.close()
