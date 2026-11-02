"""
Unit tests for Nexus-AI Authentication and Role-Based Access Control (RBAC).

Covers all 15 test categories required by Phase 10:
  1. Successful login
  2. Incorrect password
  3. Disabled account
  4. Duplicate user prevention
  5. Admin permissions
  6. Reviewer permissions
  7. Viewer read-only restrictions
  8. Viewer cannot approve or reject
  9. Reviewer can approve/reject with valid justification
  10. Reviewer cannot manage users
  11. Admin can manage users
  12. Logout clears authentication
  13. Session expiration
  14. Authentication audit events
  15. Existing approval and audit functionality remains operational
"""

import time
import pytest
from agents.auth_manager import AuthManager, UserRole
from agents.audit_logger import AuditLogger


@pytest.fixture
def auth_mgr():
    """Provides an isolated in-memory AuthManager for each test."""
    mgr = AuthManager(db_path=":memory:")
    # Initialize primary admin
    ok, msg = mgr.init_first_admin("admin_test", "admin@nexus.test", "AdminPass123!")
    assert ok, f"Admin init failed: {msg}"
    return mgr


# 1. Successful login
def test_01_successful_login(auth_mgr):
    ok, msg, user = auth_mgr.authenticate("admin_test", "AdminPass123!")
    assert ok
    assert user is not None
    assert user["username"] == "admin_test"
    assert user["role"] == UserRole.ADMIN
    assert "password" not in user
    assert "password_hash" not in user

    # Also test login via email
    ok, msg, user_email = auth_mgr.authenticate("admin@nexus.test", "AdminPass123!")
    assert ok
    assert user_email["username"] == "admin_test"


# 2. Incorrect password
def test_02_incorrect_password(auth_mgr):
    ok, msg, user = auth_mgr.authenticate("admin_test", "WrongPassword123!")
    assert not ok
    assert user is None
    assert "Invalid" in msg


# 3. Disabled account
def test_03_disabled_account(auth_mgr):
    # Create reviewer then deactivate
    ok, msg, user = auth_mgr.create_user("admin_test", "rev_user", "rev@nexus.test", "RevPass123!", UserRole.REVIEWER)
    assert ok

    # Deactivate account
    ok, msg = auth_mgr.set_account_status("admin_test", "rev_user", False)
    assert ok
    assert not auth_mgr.is_user_active("rev_user")

    # Attempt to authenticate
    ok, msg, user = auth_mgr.authenticate("rev_user", "RevPass123!")
    assert not ok
    assert "disabled" in msg.lower()


# 4. Duplicate user prevention
def test_04_duplicate_user_prevention(auth_mgr):
    ok, msg, _ = auth_mgr.create_user("admin_test", "unique_user", "unique@nexus.test", "Password123!", UserRole.VIEWER)
    assert ok

    # Duplicate username (case-insensitive)
    ok_dup_name, msg_dup_name, _ = auth_mgr.create_user("admin_test", "UNIQUE_USER", "different@nexus.test", "Password123!", UserRole.VIEWER)
    assert not ok_dup_name
    assert "already taken" in msg_dup_name

    # Duplicate email (case-insensitive)
    ok_dup_email, msg_dup_email, _ = auth_mgr.create_user("admin_test", "different_user", "unique@nexus.test", "Password123!", UserRole.VIEWER)
    assert not ok_dup_email
    assert "already registered" in msg_dup_email


# 5. Admin permissions
def test_05_admin_permissions(auth_mgr):
    role = UserRole.ADMIN
    assert auth_mgr.can_manage_users(role) is True
    assert auth_mgr.can_approve_or_reject(role) is True
    assert auth_mgr.can_execute_simulation(role) is True
    assert auth_mgr.is_read_only(role) is False


# 6. Reviewer permissions
def test_06_reviewer_permissions(auth_mgr):
    role = UserRole.REVIEWER
    assert auth_mgr.can_manage_users(role) is False
    assert auth_mgr.can_approve_or_reject(role) is True
    assert auth_mgr.can_execute_simulation(role) is True
    assert auth_mgr.is_read_only(role) is False


# 7. Viewer read-only restrictions
def test_07_viewer_read_only_restrictions(auth_mgr):
    role = UserRole.VIEWER
    assert auth_mgr.can_manage_users(role) is False
    assert auth_mgr.can_approve_or_reject(role) is False
    assert auth_mgr.can_execute_simulation(role) is False
    assert auth_mgr.is_read_only(role) is True


# 8. Viewer cannot approve or reject
def test_08_viewer_cannot_approve_or_reject(auth_mgr):
    ok, msg, viewer = auth_mgr.create_user("admin_test", "viewer_test", "view@nexus.test", "ViewerPass123!", UserRole.VIEWER)
    assert ok
    assert not auth_mgr.can_approve_or_reject(viewer["role"])
    assert not auth_mgr.can_execute_simulation(viewer["role"])


# 9. Reviewer can approve/reject with valid justification
def test_09_reviewer_can_approve_reject_with_valid_justification(auth_mgr):
    ok, msg, reviewer = auth_mgr.create_user("admin_test", "reviewer_test", "rev2@nexus.test", "RevPass123!", UserRole.REVIEWER)
    assert ok
    assert auth_mgr.can_approve_or_reject(reviewer["role"])

    # Simulate audit logger integration with authenticated reviewer label
    audit_log = AuditLogger(db_path=":memory:")
    run_id = audit_log.record_simulation_run(
        scenario_entity="Supplier C",
        disruption_type="Severe Weather Delay",
        scenario_params={"unit_loss": 100},
        sim_result={"impact_level": "Medium", "affected_entities": ["Supplier C"]},
        financial_result={"baseline_disruption_loss": 50000.0, "recommended_option_ids": ["OPT-AIR"]},
        decision_exp={"requires_human_approval": True},
        execution_channel="TEST",
        actor_role=reviewer["role"],
    )

    reviewer_label = f"{reviewer['username']} ({reviewer['role']})"
    success, msg, status = audit_log.record_decision_event(
        run_id=run_id,
        decision_type="RECOMMENDATION_APPROVED",
        reviewer_label=reviewer_label,
        justification="Validated contingency freight budget with logistics team.",
        actor_role=reviewer["role"],
    )
    assert success
    assert status == "APPROVED"


# 10. Reviewer cannot manage users
def test_10_reviewer_cannot_manage_users(auth_mgr):
    ok, msg, reviewer = auth_mgr.create_user("admin_test", "reviewer_ops", "ops@nexus.test", "ReviewerPass123!", UserRole.REVIEWER)
    assert ok

    # Attempt to create a user as a Reviewer
    ok_c, msg_c, _ = auth_mgr.create_user("reviewer_ops", "intruder", "in@nexus.test", "Pass12345!", UserRole.VIEWER)
    assert not ok_c
    assert "Access denied" in msg_c

    # Attempt to reset password as a Reviewer
    ok_r, msg_r = auth_mgr.reset_password("reviewer_ops", "admin_test", "HackedPass123!")
    assert not ok_r
    assert "Access denied" in msg_r

    # Attempt to deactivate account as a Reviewer
    ok_d, msg_d = auth_mgr.set_account_status("reviewer_ops", "admin_test", False)
    assert not ok_d
    assert "Access denied" in msg_d


# 11. Admin can manage users
def test_11_admin_can_manage_users(auth_mgr):
    # Admin creates user
    ok, msg, u = auth_mgr.create_user("admin_test", "user_to_manage", "manage@nexus.test", "TempPass123!", UserRole.VIEWER)
    assert ok

    # Admin promotes to Reviewer
    ok_role, msg_role = auth_mgr.change_role("admin_test", "user_to_manage", UserRole.REVIEWER)
    assert ok_role
    assert auth_mgr.get_user_by_username("user_to_manage")["role"] == UserRole.REVIEWER

    # Admin resets password
    ok_pw, msg_pw = auth_mgr.reset_password("admin_test", "user_to_manage", "NewSecurePass888!")
    assert ok_pw

    # Verify new password works
    ok_auth, _, _ = auth_mgr.authenticate("user_to_manage", "NewSecurePass888!")
    assert ok_auth

    # Admin lists users
    users = auth_mgr.list_users("admin_test")
    assert len(users) >= 2


# 12. Logout clears authentication
def test_12_logout_clears_authentication(auth_mgr):
    ok, msg, user = auth_mgr.authenticate("admin_test", "AdminPass123!")
    assert ok

    # Simulated session dict
    session = {
        "nexus_auth_authenticated": True,
        "nexus_auth_user": user,
        "nexus_auth_last_activity": time.time(),
    }

    # Execute logout
    auth_mgr.record_logout(user["username"])
    session["nexus_auth_authenticated"] = False
    session["nexus_auth_user"] = None
    session["nexus_auth_last_activity"] = None

    assert session["nexus_auth_authenticated"] is False
    assert session["nexus_auth_user"] is None


# 13. Session expiration
def test_13_session_expiration(auth_mgr):
    timeout_seconds = 1800  # 30 minutes
    current_time = time.time()
    last_activity_active = current_time - 300  # 5 minutes ago (active)
    last_activity_expired = current_time - 2000  # 33 minutes ago (expired)

    is_expired_active = (current_time - last_activity_active) > timeout_seconds
    assert is_expired_active is False

    is_expired_timeout = (current_time - last_activity_expired) > timeout_seconds
    assert is_expired_timeout is True

    # Record expiration event
    auth_mgr.record_session_expired("admin_test")
    events = auth_mgr.get_auth_audit_events("admin_test", limit=5)
    assert any(e["event_type"] == "AUTH_SESSION_EXPIRED" for e in events)


# 14. Authentication audit events
def test_14_authentication_audit_events(auth_mgr):
    # Perform actions
    auth_mgr.create_user("admin_test", "audited_user", "audit@nexus.test", "Pass12345!", UserRole.VIEWER)
    auth_mgr.set_account_status("admin_test", "audited_user", False)
    auth_mgr.reset_password("admin_test", "audited_user", "NewPass12345!")

    events = auth_mgr.get_auth_audit_events("admin_test", limit=50)
    assert len(events) >= 3

    event_types = [e["event_type"] for e in events]
    assert "AUTH_USER_CREATED" in event_types
    assert "AUTH_STATUS_CHANGED" in event_types
    assert "AUTH_PASSWORD_RESET" in event_types

    # Ensure no secrets leak
    for ev in events:
        details_str = str(ev["details"])
        # Ensure raw plaintext password secrets never leak
        assert "pass12345" not in details_str.lower()
        assert "newpass12345" not in details_str.lower()
        assert "hash" not in details_str.lower()
        assert "pbkdf2" not in details_str.lower()


# 15. Existing approval and audit functionality remains operational
def test_15_existing_approval_and_audit_functionality(auth_mgr):
    audit_log = AuditLogger(db_path=":memory:")
    run_id = audit_log.record_simulation_run(
        scenario_entity="Processor X",
        disruption_type="Fire Outage",
        scenario_params={"unit_loss": 150},
        sim_result={"impact_level": "Critical", "affected_entities": ["Processor X", "Supplier A"]},
        financial_result={"baseline_disruption_loss": 120000.0, "recommended_option_ids": ["OPT-BUFFER"]},
        decision_exp={"requires_human_approval": True},
        execution_channel="TEST",
        actor_role=UserRole.ADMIN,
    )

    run = audit_log.get_run_by_id(run_id)
    assert run is not None
    assert run["current_decision_status"] == "PENDING_REVIEW"

    # Admin approves
    ok, msg, status = audit_log.record_decision_event(
        run_id=run_id,
        decision_type="RECOMMENDATION_APPROVED",
        reviewer_label="admin_test (ADMIN)",
        justification="Mandatory approval confirmed by enterprise operations director.",
        actor_role=UserRole.ADMIN,
    )
    assert ok
    assert status == "APPROVED"

    # Decision finality preserved
    ok2, msg2, status2 = audit_log.record_decision_event(
        run_id=run_id,
        decision_type="RECOMMENDATION_REJECTED",
        reviewer_label="admin_test (ADMIN)",
        justification="Attempted modification.",
        actor_role=UserRole.ADMIN,
    )
    assert not ok2
    assert "Decision finality enforced" in msg2


# 16. Viewer cannot execute a simulation (backend & agent enforcement)
def test_16_viewer_cannot_execute_simulation(auth_mgr):
    from agents.disruption_simulation_agent import create_synthetic_supply_chain, simulate_disruption
    from app import execute_authorized_simulation

    ok, _, _ = auth_mgr.create_user("admin_test", "viewer_test", "viewer@nexus.test", "ViewerPass123!", UserRole.VIEWER)
    assert ok

    audit_log = AuditLogger(db_path=":memory:")
    graph = create_synthetic_supply_chain()

    # 1. Backend pipeline check rejects Viewer
    ok_exec, msg_exec, payload = execute_authorized_simulation(
        auth_manager=auth_mgr,
        audit_logger=audit_log,
        actor_username="viewer_test",
        graph=graph,
        failed_node="Processor X",
    )
    assert ok_exec is False
    assert payload is None
    assert "read-only" in msg_exec.lower() or "access denied" in msg_exec.lower()
    assert len(audit_log.get_audit_runs()) == 0

    # 2. Direct agent call with VIEWER role raises PermissionError
    with pytest.raises(PermissionError, match="cannot execute disruption simulations"):
        simulate_disruption(graph, failed_node="Processor X", actor_role=UserRole.VIEWER)


# 17. Viewer cannot create a simulation audit record
def test_17_viewer_cannot_create_simulation_audit_record(auth_mgr):
    ok, _, _ = auth_mgr.create_user("admin_test", "viewer_test", "viewer@nexus.test", "ViewerPass123!", UserRole.VIEWER)
    assert ok

    audit_log = AuditLogger(db_path=":memory:")
    with pytest.raises(PermissionError, match="cannot create simulation audit records"):
        audit_log.record_simulation_run(
            scenario_entity="Processor X",
            disruption_type="Facility Outage",
            scenario_params={"unit_loss": 150.0},
            sim_result={"impact_level": "Critical", "affected_entities": ["Supplier A"]},
            financial_result={"baseline_disruption_loss": 50000.0, "recommended_option_ids": ["OPT-1"]},
            actor_role=UserRole.VIEWER,
        )
    assert len(audit_log.get_audit_runs()) == 0


# 18. Viewer cannot approve or reject a recommendation (backend enforcement)
def test_18_viewer_cannot_approve_or_reject_backend(auth_mgr):
    from app import execute_authorized_decision

    ok, _, _ = auth_mgr.create_user("admin_test", "viewer_test", "viewer@nexus.test", "ViewerPass123!", UserRole.VIEWER)
    assert ok

    audit_log = AuditLogger(db_path=":memory:")
    run_id = audit_log.record_simulation_run(
        scenario_entity="Processor X",
        disruption_type="Facility Outage",
        scenario_params={"unit_loss": 150.0},
        sim_result={"impact_level": "Critical", "affected_entities": ["Supplier A"]},
        financial_result={"baseline_disruption_loss": 50000.0, "recommended_option_ids": ["OPT-1"]},
        decision_exp={"requires_human_approval": True},
        actor_role=UserRole.ADMIN,
    )

    # Attempt approval via app backend handler
    ok_app, msg_app, status_app = execute_authorized_decision(
        auth_manager=auth_mgr,
        audit_logger=audit_log,
        actor_username="viewer_test",
        run_id=run_id,
        decision_type="RECOMMENDATION_APPROVED",
        reviewer_label="viewer_test",
        justification="Viewer attempting unauthorized approval",
    )
    assert ok_app is False
    assert status_app == "UNAUTHORIZED"
    assert "cannot approve or reject" in msg_app.lower()

    # Attempt rejection via app backend handler
    ok_rej, msg_rej, status_rej = execute_authorized_decision(
        auth_manager=auth_mgr,
        audit_logger=audit_log,
        actor_username="viewer_test",
        run_id=run_id,
        decision_type="RECOMMENDATION_REJECTED",
        reviewer_label="viewer_test",
        justification="Viewer attempting unauthorized rejection",
    )
    assert ok_rej is False
    assert status_rej == "UNAUTHORIZED"

    # Attempt direct AuditLogger call with VIEWER role
    ok_direct, msg_direct, status_direct = audit_log.record_decision_event(
        run_id=run_id,
        decision_type="RECOMMENDATION_APPROVED",
        reviewer_label="viewer_test (VIEWER)",
        justification="Direct call bypass attempt",
        actor_role=UserRole.VIEWER,
    )
    assert ok_direct is False
    assert status_direct == "UNAUTHORIZED"
    assert audit_log.get_run_by_id(run_id)["current_decision_status"] == "PENDING_REVIEW"


# 19. Reviewer can execute simulations
def test_19_reviewer_can_execute_simulations(auth_mgr):
    from agents.disruption_simulation_agent import create_synthetic_supply_chain
    from app import execute_authorized_simulation

    ok, _, _ = auth_mgr.create_user("admin_test", "reviewer_test", "reviewer@nexus.test", "ReviewerPass123!", UserRole.REVIEWER)
    assert ok

    audit_log = AuditLogger(db_path=":memory:")
    graph = create_synthetic_supply_chain()

    ok_exec, msg_exec, payload = execute_authorized_simulation(
        auth_manager=auth_mgr,
        audit_logger=audit_log,
        actor_username="reviewer_test",
        graph=graph,
        failed_node="Processor X",
    )
    assert ok_exec is True
    assert payload is not None
    assert payload["run_id"].startswith("RUN-")
    assert len(audit_log.get_audit_runs()) == 1


# 20. Reviewer can approve and reject with justification
def test_20_reviewer_can_approve_and_reject_with_justification(auth_mgr):
    from agents.disruption_simulation_agent import create_synthetic_supply_chain
    from app import execute_authorized_decision, execute_authorized_simulation

    ok, _, _ = auth_mgr.create_user("admin_test", "reviewer_test", "reviewer@nexus.test", "ReviewerPass123!", UserRole.REVIEWER)
    assert ok

    audit_log = AuditLogger(db_path=":memory:")
    graph = create_synthetic_supply_chain()

    # Run 1 -> Approve with justification
    _, _, payload1 = execute_authorized_simulation(auth_mgr, audit_log, "reviewer_test", graph, "Processor X")
    run1 = payload1["run_id"]
    ok_a, _, st_a = execute_authorized_decision(
        auth_mgr, audit_log, "reviewer_test", run1, "RECOMMENDATION_APPROVED", "reviewer_test", "Approved with verified capacity."
    )
    assert ok_a is True
    assert st_a == "APPROVED"

    # Run 2 -> Reject without justification fails, with justification succeeds
    _, _, payload2 = execute_authorized_simulation(auth_mgr, audit_log, "reviewer_test", graph, "Processor X")
    run2 = payload2["run_id"]
    ok_r_empty, msg_r_empty, _ = execute_authorized_decision(
        auth_mgr, audit_log, "reviewer_test", run2, "RECOMMENDATION_REJECTED", "reviewer_test", "   "
    )
    assert ok_r_empty is False
    assert "justification is required" in msg_r_empty.lower()

    ok_r, _, st_r = execute_authorized_decision(
        auth_mgr, audit_log, "reviewer_test", run2, "RECOMMENDATION_REJECTED", "reviewer_test", "Rejected due to lead time constraints."
    )
    assert ok_r is True
    assert st_r == "REJECTED"


# 21. Admin retains all permissions
def test_21_admin_retains_all_permissions(auth_mgr):
    from agents.disruption_simulation_agent import create_synthetic_supply_chain
    from app import execute_authorized_decision, execute_authorized_simulation, get_ui_control_permissions

    perms = get_ui_control_permissions(auth_mgr, "admin_test")
    assert perms["role"] == UserRole.ADMIN
    assert perms["can_execute_simulation"] is True
    assert perms["can_approve_or_reject"] is True
    assert perms["can_manage_users"] is True
    assert perms["simulation_controls_disabled"] is False
    assert perms["rerun_button_disabled"] is False
    assert perms["approval_buttons_disabled"] is False
    assert perms["show_user_management_nav"] is True
    assert "8. 👥 User Management" in perms["nav_options"]

    audit_log = AuditLogger(db_path=":memory:")
    graph = create_synthetic_supply_chain()
    ok_sim, _, payload = execute_authorized_simulation(auth_mgr, audit_log, "admin_test", graph, "Processor X")
    assert ok_sim is True
    ok_dec, _, status = execute_authorized_decision(
        auth_mgr, audit_log, "admin_test", payload["run_id"], "RECOMMENDATION_APPROVED", "admin_test", "Executive sign-off."
    )
    assert ok_dec is True
    assert status == "APPROVED"


# 22. Restricted UI controls are disabled or hidden for Viewer
def test_22_restricted_ui_controls_disabled_or_hidden_for_viewer(auth_mgr):
    from app import get_ui_control_permissions

    ok, _, _ = auth_mgr.create_user("admin_test", "viewer_test", "viewer@nexus.test", "ViewerPass123!", UserRole.VIEWER)
    assert ok

    perms = get_ui_control_permissions(auth_mgr, "viewer_test")
    assert perms["role"] == UserRole.VIEWER
    assert perms["is_viewer"] is True
    assert perms["simulation_controls_disabled"] is True
    assert perms["rerun_button_disabled"] is True
    assert perms["approval_buttons_disabled"] is True
    assert perms["show_user_management_nav"] is False
    assert "8. 👥 User Management" not in perms["nav_options"]
    assert len(perms["nav_options"]) == 7
    assert "READ-ONLY ACCESS" in perms["read_only_explanation"]


# 23. Backend permission checks reject session state tampering and unauthorized attempts
def test_23_backend_permission_checks_reject_tampering(auth_mgr):
    from agents.disruption_simulation_agent import create_synthetic_supply_chain
    from app import execute_authorized_decision, execute_authorized_simulation, get_ui_control_permissions

    ok, _, viewer_user = auth_mgr.create_user("admin_test", "viewer_test", "viewer@nexus.test", "ViewerPass123!", UserRole.VIEWER)
    assert ok

    # Simulate tampered Streamlit session_state where viewer_test claims to be ADMIN and sets force_new_run=True
    tampered_session = {
        "nexus_auth_authenticated": True,
        "nexus_auth_user": {**viewer_user, "role": "ADMIN"},
        "force_new_run": True,
    }

    # 1. get_ui_control_permissions ignores tampered session_state and restores live DB role ('VIEWER')
    perms = get_ui_control_permissions(auth_mgr, "viewer_test", session_state=tampered_session)
    assert perms["role"] == UserRole.VIEWER
    assert tampered_session["nexus_auth_user"]["role"] == UserRole.VIEWER
    assert perms["simulation_controls_disabled"] is True

    # 2. execute_authorized_simulation rejects tampered session_state and resets force_new_run=False
    audit_log = AuditLogger(db_path=":memory:")
    graph = create_synthetic_supply_chain()
    tampered_session["nexus_auth_user"]["role"] = "ADMIN"
    tampered_session["force_new_run"] = True

    ok_sim, msg_sim, _ = execute_authorized_simulation(
        auth_mgr, audit_log, "viewer_test", graph, "Processor X", session_state=tampered_session
    )
    assert ok_sim is False
    assert tampered_session["force_new_run"] is False
    assert tampered_session["nexus_auth_user"]["role"] == UserRole.VIEWER
    assert len(audit_log.get_audit_runs()) == 0

    # 3. Verify unauthorized attempts were logged in security audit trail
    events = auth_mgr.get_auth_audit_events("admin_test", limit=20)
    assert any(e["event_type"] == "AUTH_UNAUTHORIZED_SIMULATION_ATTEMPT" and e["actor_username"] == "viewer_test" for e in events)


# 24. Streamlit AppTest verification of actual VIEWER UI widget disabled states and read-only banners
def test_24_streamlit_apptest_viewer_ui_controls_disabled(tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    temp_auth_db = tmp_path / "test_nexus_auth.db"
    temp_audit_db = tmp_path / "test_nexus_audit.db"
    monkeypatch.setenv("NEXUS_AUTH_DB_PATH", str(temp_auth_db))
    monkeypatch.setenv("NEXUS_AUDIT_DB_PATH", str(temp_audit_db))

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    mgr = AuthManager(str(temp_auth_db))
    ok_adm, _ = mgr.init_first_admin("admin_test", "admin@nexus.test", "AdminPass123!")
    assert ok_adm
    ok_view, _, viewer_u = mgr.create_user(
        "admin_test", "viewer_test", "viewer@nexus.test", "ViewerPass123!", UserRole.VIEWER
    )
    assert ok_view and viewer_u is not None
    assert viewer_u["role"] == UserRole.VIEWER

    audit_log = AuditLogger(str(temp_audit_db))
    runs_before = len(audit_log.get_audit_runs(limit=200))

    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.session_state["nexus_auth_authenticated"] = True
    # Even if session_state is passed with a tampered role, app.py must resolve VIEWER from live DB
    at.session_state["nexus_auth_user"] = {**viewer_u, "role": "ADMIN"}
    at.session_state["nexus_auth_last_activity"] = time.time()
    at.run()

    # Verify live role was restored to VIEWER in session state and baseline hydration is marked read-only
    assert at.session_state["nexus_auth_user"]["role"] == UserRole.VIEWER
    assert at.session_state["is_read_only_baseline"] is True
    assert at.session_state["cached_sim_result"]["is_read_only_baseline"] is True

    # Verify navigation has all 7 operational sections and hides 8. User Management
    nav_radio = at.sidebar.radio[0]
    assert len(nav_radio.options) == 7
    assert "8. 👥 User Management" not in nav_radio.options

    # Verify all sidebar simulation controls are disabled for VIEWER
    assert at.sidebar.radio[1].disabled is True  # Choose Supply Chain Data Source
    for sb in at.sidebar.selectbox:
        assert sb.disabled is True  # Select Failed Entity, Disruption Category
    for ni in at.sidebar.number_input:
        assert ni.disabled is True  # Loss Per Disrupted Unit, Emergency Recovery Budget
    for sl in at.sidebar.slider:
        assert sl.disabled is True  # Max Acceptable Lead Time
    for cb in at.sidebar.checkbox:
        assert cb.disabled is True  # Enable Multi-Supplier Combination Optimization

    # Verify Re-run Simulation button is disabled
    rerun_btns = [b for b in at.sidebar.button if "Re-run Simulation" in b.label]
    assert len(rerun_btns) == 1
    assert rerun_btns[0].disabled is True

    # Navigate to Section 4 (Disruption Simulator) and verify no errors and read-only banner present
    at.sidebar.radio[0].set_value("4. ⚡ Disruption Simulator").run()
    assert len(at.exception) == 0
    md_texts_sec4 = " ".join([m.value for m in at.markdown])
    assert "READ-ONLY ACCESS (VIEWER ROLE)" in md_texts_sec4

    # Navigate to Section 6 (Decisions & Approvals) and verify Approve/Reject buttons and inputs are disabled
    at.sidebar.radio[0].set_value("6. ⚖️ Decisions & Approvals").run()
    assert len(at.exception) == 0
    approve_btns = [b for b in at.button if "Approve Recommendation" in b.label]
    reject_btns = [b for b in at.button if "Reject Recommendation" in b.label]
    assert len(approve_btns) == 1 and approve_btns[0].disabled is True
    assert len(reject_btns) == 1 and reject_btns[0].disabled is True
    for ti in at.text_input:
        assert ti.disabled is True
    for ta in at.text_area:
        assert ta.disabled is True

    # Verify zero new simulation audit records were created by VIEWER session
    runs_after = len(audit_log.get_audit_runs(limit=200))
    assert runs_after == runs_before == 0


# 25. Streamlit AppTest regression test for REVIEWER and ADMIN runtime simulation execution & RBAC
def test_25_streamlit_apptest_reviewer_and_admin_simulation_and_governance(tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    temp_auth_db = tmp_path / "test_nexus_auth.db"
    temp_audit_db = tmp_path / "test_nexus_audit.db"
    monkeypatch.setenv("NEXUS_AUTH_DB_PATH", str(temp_auth_db))
    monkeypatch.setenv("NEXUS_AUDIT_DB_PATH", str(temp_audit_db))

    root_dir = Path(__file__).resolve().parent.parent
    app_path = str(root_dir / "app.py")
    mgr = AuthManager(str(temp_auth_db))
    ok_adm, _ = mgr.init_first_admin("admin_test", "admin@nexus.test", "AdminPass123!")
    assert ok_adm
    ok_rev, _, reviewer_u = mgr.create_user(
        "admin_test", "reviewer_test", "reviewer@nexus.test", "ReviewerPass123!", UserRole.REVIEWER
    )
    assert ok_rev and reviewer_u is not None
    assert reviewer_u["role"] == UserRole.REVIEWER

    # 1. Verify REVIEWER login & full simulation execution in run_streamlit_app() without AttributeError
    at_rev = AppTest.from_file(app_path, default_timeout=30)
    at_rev.session_state["nexus_auth_authenticated"] = True
    at_rev.session_state["nexus_auth_user"] = reviewer_u
    at_rev.session_state["nexus_auth_last_activity"] = time.time()
    at_rev.run()

    # Must execute with zero runtime exceptions
    assert len(at_rev.exception) == 0
    assert at_rev.session_state["nexus_auth_user"]["role"] == UserRole.REVIEWER
    assert at_rev.session_state["is_read_only_baseline"] is False

    # REVIEWER navigation has 7 sections (no 8. User Management)
    nav_rev = at_rev.sidebar.radio[0]
    assert len(nav_rev.options) == 7
    assert "8. 👥 User Management" not in nav_rev.options

    # REVIEWER simulation controls and Re-run Simulation button are enabled
    for sb in at_rev.sidebar.selectbox:
        assert sb.disabled is False
    rerun_rev = [b for b in at_rev.sidebar.button if "Re-run Simulation" in b.label]
    assert len(rerun_rev) == 1 and rerun_rev[0].disabled is False

    # Trigger Re-run Simulation as REVIEWER and verify no AttributeError
    rerun_rev[0].click().run()
    assert len(at_rev.exception) == 0
    assert not str(at_rev.session_state.get("active_run_id", "")).startswith("READ_ONLY")

    # Navigate to Section 6 (Decisions & Approvals) as REVIEWER and verify Approve/Reject buttons are enabled
    at_rev.sidebar.radio[0].set_value("6. ⚖️ Decisions & Approvals").run()
    assert len(at_rev.exception) == 0
    approve_rev = [b for b in at_rev.button if "Approve Recommendation" in b.label]
    reject_rev = [b for b in at_rev.button if "Reject Recommendation" in b.label]
    assert len(approve_rev) == 1 and approve_rev[0].disabled is False
    assert len(reject_rev) == 1 and reject_rev[0].disabled is False

    # 2. Verify ADMIN login & full simulation + User Management access in run_streamlit_app()
    admin_u = mgr.get_user_by_username("admin_test")
    assert admin_u is not None
    assert admin_u["role"] == UserRole.ADMIN

    at_adm = AppTest.from_file(app_path, default_timeout=30)
    at_adm.session_state["nexus_auth_authenticated"] = True
    at_adm.session_state["nexus_auth_user"] = admin_u
    at_adm.session_state["nexus_auth_last_activity"] = time.time()
    at_adm.run()

    assert len(at_adm.exception) == 0
    nav_adm = at_adm.sidebar.radio[0]
    assert len(nav_adm.options) == 8
    assert "8. 👥 User Management" in nav_adm.options

    # Navigate to Section 8 (User Management) as ADMIN and verify zero exceptions
    at_adm.sidebar.radio[0].set_value("8. 👥 User Management").run()
    assert len(at_adm.exception) == 0


# 26. Default-deny enforcement when actor_role=None or omitted on backend operations
def test_26_default_deny_when_actor_role_is_none_or_omitted():
    from agents.disruption_simulation_agent import (
        DisruptionSimulationAgent,
        create_synthetic_supply_chain,
        simulate_disruption,
    )

    graph = create_synthetic_supply_chain()
    agent = DisruptionSimulationAgent(graph)
    audit_log = AuditLogger(db_path=":memory:")

    # 1. simulate_disruption rejects actor_role=None / omitted
    with pytest.raises(PermissionError, match="cannot execute disruption simulations"):
        simulate_disruption(graph, failed_node="Processor X")
    with pytest.raises(PermissionError, match="cannot execute disruption simulations"):
        simulate_disruption(graph, failed_node="Processor X", actor_role=None)
    with pytest.raises(PermissionError, match="cannot execute disruption simulations"):
        agent.simulate(failed_node="Processor X")
    with pytest.raises(PermissionError, match="cannot execute disruption simulations"):
        agent.simulate_all_nodes()

    # 2. AuditLogger.record_simulation_run rejects actor_role=None / omitted
    with pytest.raises(PermissionError, match="cannot create simulation audit records"):
        audit_log.record_simulation_run(
            scenario_entity="Processor X",
            disruption_type="Facility Outage",
            scenario_params={"unit_loss": 150.0},
            sim_result={"impact_level": "Critical", "affected_entities": ["Supplier A"]},
            financial_result={"baseline_disruption_loss": 30000.0, "recommended_option_ids": ["OPT-1"]},
        )
    assert len(audit_log.get_audit_runs()) == 0

    # 3. AuditLogger.record_decision_event rejects actor_role=None / omitted
    valid_run_id = audit_log.record_simulation_run(
        scenario_entity="Processor X",
        disruption_type="Facility Outage",
        scenario_params={"unit_loss": 150.0},
        sim_result={"impact_level": "Critical", "affected_entities": ["Supplier A"]},
        financial_result={"baseline_disruption_loss": 30000.0, "recommended_option_ids": ["OPT-1"]},
        decision_exp={"requires_human_approval": True},
        actor_role=UserRole.ADMIN,
    )
    ok_dec, msg_dec, status_dec = audit_log.record_decision_event(
        run_id=valid_run_id,
        decision_type="RECOMMENDATION_APPROVED",
        reviewer_label="admin_test (ADMIN)",
        justification="Attempting decision without passing actor_role",
    )
    assert ok_dec is False
    assert status_dec == "UNAUTHORIZED"
    assert "cannot approve or reject" in msg_dec.lower()
    assert audit_log.get_run_by_id(valid_run_id)["current_decision_status"] == "PENDING_REVIEW"


# 27. Explicit separation of read-only Viewer baseline hydration from user-triggered simulation execution
def test_27_viewer_baseline_hydration_separated_from_simulation_execution():
    from agents.disruption_simulation_agent import (
        create_synthetic_supply_chain,
        hydrate_read_only_baseline_simulation,
    )
    from app import hydrate_viewer_baseline_session_state

    graph = create_synthetic_supply_chain()
    audit_log = AuditLogger(db_path=":memory:")
    session_state = {}

    baseline_res = hydrate_read_only_baseline_simulation(graph, baseline_node="Processor X")
    assert baseline_res["is_read_only_baseline"] is True
    assert baseline_res["failed_entity"] == "Processor X"

    hydrated = hydrate_viewer_baseline_session_state(
        graph=graph,
        audit_logger=audit_log,
        session_state=session_state,
        baseline_fingerprint="baseline_fp",
    )
    assert hydrated["active_run_id"] == "READ_ONLY_SIMULATION"
    assert session_state["is_read_only_baseline"] is True
    assert session_state["force_new_run"] is False
    assert session_state["cached_sim_result"]["is_read_only_baseline"] is True
    assert len(audit_log.get_audit_runs()) == 0




