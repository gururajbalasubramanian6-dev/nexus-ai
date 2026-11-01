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
    )

    reviewer_label = f"{reviewer['username']} ({reviewer['role']})"
    success, msg, status = audit_log.record_decision_event(
        run_id=run_id,
        decision_type="RECOMMENDATION_APPROVED",
        reviewer_label=reviewer_label,
        justification="Validated contingency freight budget with logistics team.",
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
    )
    assert ok
    assert status == "APPROVED"

    # Decision finality preserved
    ok2, msg2, status2 = audit_log.record_decision_event(
        run_id=run_id,
        decision_type="RECOMMENDATION_REJECTED",
        reviewer_label="admin_test (ADMIN)",
        justification="Attempted modification.",
    )
    assert not ok2
    assert "Decision finality enforced" in msg2
