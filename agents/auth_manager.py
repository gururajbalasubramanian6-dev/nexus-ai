"""
Nexus-AI: Enterprise Authentication & Role-Based Access Control (RBAC)
======================================================================

Provides secure user authentication, role-based authorization, session management,
and append-only security auditing for Nexus-AI.

Roles:
  - ADMIN: Full control over 7 dashboard sections, disruption simulations,
           recommendation approval/rejection, user lifecycle, and role assignments.
  - REVIEWER: Access all 7 dashboard sections, run simulations, review recommendations,
              approve/reject with mandatory justification. Cannot manage users.
  - VIEWER: Read-only access across all 7 dashboard sections. Cannot approve/reject
            or manage user accounts.

Security Safeguards:
  - PBKDF2-HMAC-SHA256 password hashing with 100,000 iterations and unique 16-byte cryptographically secure salts.
  - Constant-time password verification using hmac.compare_digest to prevent timing attacks.
  - Parameterized SQLite queries to prevent SQL injection.
  - Separate security database (data/nexus_auth.db) isolated from the operational audit ledger.
  - Brute-force lockout protection (locks account for 15 minutes after 5 consecutive failed attempts).
  - Explicit server-side authorization enforcement on all privileged operations.
  - Append-only security audit log recording all authentication and governance events without exposing secrets.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union


class UserRole:
    ADMIN = "ADMIN"
    REVIEWER = "REVIEWER"
    VIEWER = "VIEWER"

    ALL_ROLES = (ADMIN, REVIEWER, VIEWER)
    PRIVILEGED_ROLES = (ADMIN, REVIEWER)


def get_utc_iso_timestamp() -> str:
    """Returns current UTC timestamp in standardized ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class AuthManager:
    """
    Manages user authentication, role authorization, and security event auditing.
    Persists data in SQLite with strict parameterized queries.
    """

    def __init__(self, db_path: str = "data/nexus_auth.db") -> None:
        self.db_path = db_path
        self._is_memory = (db_path == ":memory:")
        self._memory_conn: Optional[sqlite3.Connection] = None

        if not self._is_memory:
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            conn = self._get_connection()
            try:
                self._init_schema(conn)
            finally:
                conn.close()
        else:
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
            self._init_schema(self._memory_conn)

        # Attempt auto-provisioning from environment variables if database has 0 admins
        self._check_env_provisioning()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection configured with Row factory and foreign keys."""
        if self._is_memory:
            assert self._memory_conn is not None
            return self._memory_conn
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        """Creates the authentication tables and indexes if they do not exist."""
        with conn:
            cursor = conn.cursor()
            # 1. Users Table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS nexus_users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                    email TEXT UNIQUE NOT NULL COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('ADMIN', 'REVIEWER', 'VIEWER')),
                    is_active INTEGER NOT NULL DEFAULT 1 CHECK(is_active IN (0, 1)),
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    last_login_utc TEXT,
                    failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                    locked_until_utc TEXT
                );
                """
            )

            # 2. Security Audit Events Table (Append-only)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS nexus_auth_audit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp_utc TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor_username TEXT NOT NULL,
                    target_username TEXT,
                    details_json TEXT NOT NULL
                );
                """
            )

            # Indexes for fast lookup
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON nexus_users (username);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON nexus_users (email);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_auth_events_time ON nexus_auth_audit_events (timestamp_utc);")

    # =========================================================================
    # Cryptographic Password Hashing & Verification
    # =========================================================================

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hashes password using PBKDF2-HMAC-SHA256 with 100,000 iterations and a unique 16-byte salt.
        Format: pbkdf2:sha256:100000$<salt_hex>$<hash_hex>
        """
        if not password or not isinstance(password, str):
            raise ValueError("Password must be a non-empty string.")
        salt = secrets.token_bytes(16)
        iterations = 100_000
        key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return f"pbkdf2:sha256:{iterations}${salt.hex()}${key.hex()}"

    @staticmethod
    def verify_password(password: str, stored_hash: str) -> bool:
        """
        Verifies candidate password against stored hash using constant-time comparison.
        Prevents timing analysis attacks.
        """
        if not password or not stored_hash or not isinstance(stored_hash, str):
            return False

        try:
            parts = stored_hash.split("$")
            if len(parts) != 3:
                return False
            algo_spec, salt_hex, expected_hex = parts
            algo_parts = algo_spec.split(":")
            if len(algo_parts) != 3 or algo_parts[0] != "pbkdf2" or algo_parts[1] != "sha256":
                return False

            iterations = int(algo_parts[2])
            salt = bytes.fromhex(salt_hex)
            expected_key = bytes.fromhex(expected_hex)

            candidate_key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
            return hmac.compare_digest(candidate_key, expected_key)
        except Exception:
            return False

    COMMON_PROHIBITED_PASSWORDS = {
        "password", "password1", "password123", "admin123", "admin1234", "administrator",
        "12345678", "123456789", "1234567890", "qwerty123", "welcome1", "welcome123",
        "nexus123", "changeme", "changeme1", "default123", "letmein123",
        "nexusadmin2026!", "reviewer2026!", "viewer2026!", "adminpassword123!"
    }

    @classmethod
    def validate_password_strength(cls, password: str, username: Optional[str] = None) -> Tuple[bool, str]:
        """Validates that password meets enterprise security requirements."""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long."
        if not re.search(r"[A-Za-z]", password):
            return False, "Password must contain at least one letter."
        if not re.search(r"[0-9]", password):
            return False, "Password must contain at least one numeric digit."

        normalized = password.strip().lower()
        if normalized in cls.COMMON_PROHIBITED_PASSWORDS:
            return False, "Password is too common or a known default. Please choose a more complex credential."

        if username and username.strip().lower() in normalized:
            return False, "Password must not contain the account username."

        return True, "Password meets strength criteria."

    # =========================================================================
    # First-Time Administrator Setup
    # =========================================================================

    def has_any_admin(self) -> bool:
        """Returns True if at least one active ADMIN account exists."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) as count FROM nexus_users WHERE role = ? AND is_active = 1;",
                (UserRole.ADMIN,),
            )
            row = cursor.fetchone()
            return bool(row and row["count"] > 0)
        finally:
            if not self._is_memory:
                conn.close()

    def has_any_users(self) -> bool:
        """Returns True if any user exists in the database."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM nexus_users;")
            row = cursor.fetchone()
            return bool(row and row["count"] > 0)
        finally:
            if not self._is_memory:
                conn.close()

    def _check_env_provisioning(self) -> None:
        """Provisions first admin from environment variables if database has 0 admins."""
        env_user = os.environ.get("NEXUS_ADMIN_USERNAME", "").strip()
        env_email = os.environ.get("NEXUS_ADMIN_EMAIL", "").strip()
        env_pass = os.environ.get("NEXUS_ADMIN_PASSWORD", "").strip()

        if env_user and env_email and env_pass:
            if not self.has_any_admin():
                success, msg = self.init_first_admin(env_user, env_email, env_pass)
                if success:
                    # Clear password from memory dictionary representation
                    pass

    def init_first_admin(self, username: str, email: str, password: str) -> Tuple[bool, str]:
        """
        Provisions the initial administrator account.
        Only allowed when ZERO admin users exist in the system.
        """
        clean_user = (username or "").strip()
        clean_email = (email or "").strip().lower()

        if self.has_any_admin():
            return False, "Initial setup blocked: An active Administrator account already exists."

        if not clean_user or len(clean_user) < 3:
            return False, "Username must be at least 3 characters."
        if not re.match(r"^[\w\.\-]+$", clean_user):
            return False, "Username may only contain letters, numbers, periods, and hyphens."

        if not clean_email or "@" not in clean_email or "." not in clean_email.split("@")[-1]:
            return False, "A valid email address is required."

        is_valid_pw, pw_msg = self.validate_password_strength(password, username)
        if not is_valid_pw:
            return False, pw_msg

        user_id = f"USR-{uuid.uuid4().hex[:8].upper()}"
        pw_hash = self.hash_password(password)
        now_utc = get_utc_iso_timestamp()

        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO nexus_users (
                        user_id, username, email, password_hash, role,
                        is_active, created_at_utc, updated_at_utc,
                        failed_login_attempts
                    ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, 0);
                    """,
                    (user_id, clean_user, clean_email, pw_hash, UserRole.ADMIN, now_utc, now_utc),
                )
                self._record_audit_event_cursor(
                    cursor,
                    event_type="AUTH_FIRST_ADMIN_PROVISIONED",
                    actor_username=clean_user,
                    target_username=clean_user,
                    details={"email": clean_email, "role": UserRole.ADMIN},
                )
            return True, f"Initial administrator '{clean_user}' provisioned successfully."
        except sqlite3.IntegrityError as e:
            return False, f"Account creation failed: User already exists ({e})."
        finally:
            if not self._is_memory:
                conn.close()

    # =========================================================================
    # User Authentication
    # =========================================================================

    def authenticate(self, username_or_email: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Authenticates user with username or email and password.
        Handles failed attempts and temporary account lockout.

        Returns:
          (success: bool, message: str, user_dict: Optional[Dict])
        """
        identifier = (username_or_email or "").strip()
        if not identifier or not password:
            return False, "Username and password are required.", None

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT user_id, username, email, password_hash, role, is_active,
                       failed_login_attempts, locked_until_utc, created_at_utc, last_login_utc
                FROM nexus_users
                WHERE username = ? COLLATE NOCASE OR email = ? COLLATE NOCASE;
                """,
                (identifier, identifier),
            )
            row = cursor.fetchone()

            if not row:
                # Run dummy verification to prevent timing side-channel attacks
                self.verify_password("dummy_password", "pbkdf2:sha256:100000$0000000000000000$0000000000000000")
                self._record_audit_event(
                    event_type="AUTH_LOGIN_FAILED",
                    actor_username=identifier,
                    target_username=identifier,
                    details={"reason": "User not found"},
                )
                return False, "Invalid username, email, or password.", None

            username = row["username"]
            stored_hash = row["password_hash"]
            is_active = bool(row["is_active"])
            failed_attempts = int(row["failed_login_attempts"] or 0)
            locked_until_str = row["locked_until_utc"]

            now_iso = get_utc_iso_timestamp()

            # Check if account is temporarily locked
            if locked_until_str:
                try:
                    locked_time = datetime.fromisoformat(locked_until_str)
                    if datetime.now(timezone.utc) < locked_time:
                        remaining_min = max(1, int((locked_time - datetime.now(timezone.utc)).total_seconds() / 60))
                        self._record_audit_event(
                            event_type="AUTH_LOGIN_BLOCKED_LOCKED",
                            actor_username=username,
                            target_username=username,
                            details={"remaining_minutes": remaining_min},
                        )
                        return (
                            False,
                            f"Account is temporarily locked due to multiple failed login attempts. Try again in {remaining_min} minutes.",
                            None,
                        )
                except Exception:
                    pass

            # Check if account is disabled
            if not is_active:
                self._record_audit_event(
                    event_type="AUTH_LOGIN_FAILED_DEACTIVATED",
                    actor_username=username,
                    target_username=username,
                    details={"reason": "Account is disabled"},
                )
                return False, "Account is disabled. Please contact your system administrator.", None

            # Verify password hash
            pw_matches = self.verify_password(password, stored_hash)

            if not pw_matches:
                new_failed = failed_attempts + 1
                locked_until = None
                lock_msg = ""
                # Lock for 15 minutes after 5 consecutive failures
                if new_failed >= 5:
                    lock_dt = datetime.now(timezone.utc)
                    locked_until = datetime.fromtimestamp(lock_dt.timestamp() + 900, timezone.utc).isoformat()
                    lock_msg = " Account locked for 15 minutes."

                with conn:
                    conn.cursor().execute(
                        """
                        UPDATE nexus_users
                        SET failed_login_attempts = ?, locked_until_utc = ?
                        WHERE username = ?;
                        """,
                        (new_failed, locked_until, username),
                    )
                    self._record_audit_event_cursor(
                        conn.cursor(),
                        event_type="AUTH_LOGIN_FAILED",
                        actor_username=username,
                        target_username=username,
                        details={"failed_attempts": new_failed, "locked": bool(locked_until)},
                    )

                return False, f"Invalid username, email, or password.{lock_msg}", None

            # Successful login: reset failed attempts and update last_login
            with conn:
                conn.cursor().execute(
                    """
                    UPDATE nexus_users
                    SET failed_login_attempts = 0, locked_until_utc = NULL, last_login_utc = ?
                    WHERE username = ?;
                    """,
                    (now_iso, username),
                )
                self._record_audit_event_cursor(
                    conn.cursor(),
                    event_type="AUTH_LOGIN_SUCCESS",
                    actor_username=username,
                    target_username=username,
                    details={"role": row["role"]},
                )

            user_dict = {
                "user_id": row["user_id"],
                "username": row["username"],
                "email": row["email"],
                "role": row["role"],
                "is_active": True,
                "created_at_utc": row["created_at_utc"],
                "last_login_utc": now_iso,
            }
            return True, "Login successful.", user_dict
        finally:
            if not self._is_memory:
                conn.close()

    # =========================================================================
    # Role-Based Authorization Checks
    # =========================================================================

    @staticmethod
    def can_approve_or_reject(role: str) -> bool:
        """Returns True if role is permitted to approve or reject recovery recommendations."""
        return role in (UserRole.ADMIN, UserRole.REVIEWER)

    @staticmethod
    def can_execute_simulation(role: str) -> bool:
        """Returns True if role is permitted to execute disruption simulations / write operations."""
        return role in (UserRole.ADMIN, UserRole.REVIEWER)

    @staticmethod
    def can_manage_users(role: str) -> bool:
        """Returns True if role is permitted to manage user accounts and access credentials."""
        return role == UserRole.ADMIN

    @staticmethod
    def is_read_only(role: str) -> bool:
        """Returns True if role has read-only restrictions."""
        return role == UserRole.VIEWER

    def is_user_active(self, username: str) -> bool:
        """Checks live database status to ensure user account has not been deactivated."""
        if not username:
            return False
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT is_active FROM nexus_users WHERE username = ? COLLATE NOCASE;",
                (username.strip(),),
            )
            row = cursor.fetchone()
            return bool(row and row["is_active"] == 1)
        finally:
            if not self._is_memory:
                conn.close()

    # =========================================================================
    # Admin User Management
    # =========================================================================

    def create_user(
        self,
        actor_username: str,
        username: str,
        email: str,
        password: str,
        role: str,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Creates a new user account. Restricted to ADMIN role.
        """
        actor = self.get_user_by_username(actor_username)
        if not actor or actor["role"] != UserRole.ADMIN:
            return False, "Access denied: Only an Administrator can create new users.", None

        clean_user = (username or "").strip()
        clean_email = (email or "").strip().lower()
        clean_role = (role or "").strip().upper()

        if not clean_user or len(clean_user) < 3:
            return False, "Username must be at least 3 characters.", None
        if not re.match(r"^[\w\.\-]+$", clean_user):
            return False, "Username may only contain letters, numbers, periods, and hyphens.", None

        if not clean_email or "@" not in clean_email or "." not in clean_email.split("@")[-1]:
            return False, "A valid email address is required.", None

        if clean_role not in UserRole.ALL_ROLES:
            return False, f"Invalid role '{clean_role}'. Must be one of {UserRole.ALL_ROLES}.", None

        is_valid_pw, pw_msg = self.validate_password_strength(password, username)
        if not is_valid_pw:
            return False, pw_msg, None

        # Check for duplicates before insert
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT username, email FROM nexus_users WHERE username = ? COLLATE NOCASE OR email = ? COLLATE NOCASE;",
                (clean_user, clean_email),
            )
            existing = cursor.fetchone()
            if existing:
                if existing["username"].lower() == clean_user.lower():
                    return False, f"Username '{clean_user}' is already taken.", None
                if existing["email"].lower() == clean_email.lower():
                    return False, f"Email address '{clean_email}' is already registered.", None

            user_id = f"USR-{uuid.uuid4().hex[:8].upper()}"
            pw_hash = self.hash_password(password)
            now_utc = get_utc_iso_timestamp()

            with conn:
                cursor.execute(
                    """
                    INSERT INTO nexus_users (
                        user_id, username, email, password_hash, role,
                        is_active, created_at_utc, updated_at_utc,
                        failed_login_attempts
                    ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, 0);
                    """,
                    (user_id, clean_user, clean_email, pw_hash, clean_role, now_utc, now_utc),
                )
                self._record_audit_event_cursor(
                    cursor,
                    event_type="AUTH_USER_CREATED",
                    actor_username=actor_username,
                    target_username=clean_user,
                    details={"email": clean_email, "role": clean_role},
                )

            new_user = {
                "user_id": user_id,
                "username": clean_user,
                "email": clean_email,
                "role": clean_role,
                "is_active": True,
                "created_at_utc": now_utc,
                "last_login_utc": None,
            }
            return True, f"User '{clean_user}' created successfully with role '{clean_role}'.", new_user
        except sqlite3.IntegrityError as exc:
            return False, f"Failed to create user: {exc}", None
        finally:
            if not self._is_memory:
                conn.close()

    def set_account_status(self, actor_username: str, target_username: str, is_active: bool) -> Tuple[bool, str]:
        """
        Activates or deactivates a user account. Restricted to ADMIN role.
        Prevents deactivating the last active administrator.
        """
        actor = self.get_user_by_username(actor_username)
        if not actor or actor["role"] != UserRole.ADMIN:
            return False, "Access denied: Only an Administrator can change account status."

        target = self.get_user_by_username(target_username)
        if not target:
            return False, f"User '{target_username}' not found."

        # Prevent deactivating the last active administrator
        if not is_active and target["role"] == UserRole.ADMIN:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) as count FROM nexus_users WHERE role = ? AND is_active = 1;",
                    (UserRole.ADMIN,),
                )
                row = cursor.fetchone()
                if row and row["count"] <= 1:
                    return False, "Cannot deactivate the only active Administrator account."
            finally:
                if not self._is_memory:
                    conn.close()

        now_utc = get_utc_iso_timestamp()
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE nexus_users
                    SET is_active = ?, updated_at_utc = ?
                    WHERE username = ? COLLATE NOCASE;
                    """,
                    (1 if is_active else 0, now_utc, target_username),
                )
                self._record_audit_event_cursor(
                    cursor,
                    event_type="AUTH_STATUS_CHANGED",
                    actor_username=actor_username,
                    target_username=target_username,
                    details={"new_status": "ACTIVE" if is_active else "DEACTIVATED"},
                )
            status_text = "activated" if is_active else "deactivated"
            return True, f"Account '{target_username}' successfully {status_text}."
        finally:
            if not self._is_memory:
                conn.close()

    def reset_password(self, actor_username: str, target_username: str, new_password: str) -> Tuple[bool, str]:
        """
        Resets user password. Restricted to ADMIN role.
        """
        actor = self.get_user_by_username(actor_username)
        if not actor or actor["role"] != UserRole.ADMIN:
            return False, "Access denied: Only an Administrator can reset user passwords."

        target = self.get_user_by_username(target_username)
        if not target:
            return False, f"User '{target_username}' not found."

        is_valid_pw, pw_msg = self.validate_password_strength(new_password, target_username)
        if not is_valid_pw:
            return False, pw_msg

        pw_hash = self.hash_password(new_password)
        now_utc = get_utc_iso_timestamp()

        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE nexus_users
                    SET password_hash = ?, failed_login_attempts = 0, locked_until_utc = NULL, updated_at_utc = ?
                    WHERE username = ? COLLATE NOCASE;
                    """,
                    (pw_hash, now_utc, target_username),
                )
                self._record_audit_event_cursor(
                    cursor,
                    event_type="AUTH_PASSWORD_RESET",
                    actor_username=actor_username,
                    target_username=target_username,
                    details={"action": "Administrative password reset"},
                )
            return True, f"Password for '{target_username}' has been successfully reset."
        finally:
            if not self._is_memory:
                conn.close()

    def change_role(self, actor_username: str, target_username: str, new_role: str) -> Tuple[bool, str]:
        """
        Changes a user's role. Restricted to ADMIN role.
        Prevents demoting the last active administrator.
        """
        actor = self.get_user_by_username(actor_username)
        if not actor or actor["role"] != UserRole.ADMIN:
            return False, "Access denied: Only an Administrator can assign roles."

        clean_role = (new_role or "").strip().upper()
        if clean_role not in UserRole.ALL_ROLES:
            return False, f"Invalid role '{clean_role}'. Must be one of {UserRole.ALL_ROLES}."

        target = self.get_user_by_username(target_username)
        if not target:
            return False, f"User '{target_username}' not found."

        old_role = target["role"]
        if old_role == clean_role:
            return True, f"User '{target_username}' already has role '{clean_role}'."

        # Prevent demoting the last admin
        if old_role == UserRole.ADMIN and clean_role != UserRole.ADMIN:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) as count FROM nexus_users WHERE role = ? AND is_active = 1;",
                    (UserRole.ADMIN,),
                )
                row = cursor.fetchone()
                if row and row["count"] <= 1:
                    return False, "Cannot demote the only active Administrator account."
            finally:
                if not self._is_memory:
                    conn.close()

        now_utc = get_utc_iso_timestamp()
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE nexus_users
                    SET role = ?, updated_at_utc = ?
                    WHERE username = ? COLLATE NOCASE;
                    """,
                    (clean_role, now_utc, target_username),
                )
                self._record_audit_event_cursor(
                    cursor,
                    event_type="AUTH_ROLE_CHANGED",
                    actor_username=actor_username,
                    target_username=target_username,
                    details={"old_role": old_role, "new_role": clean_role},
                )
            return True, f"Role for '{target_username}' changed from {old_role} to {clean_role}."
        finally:
            if not self._is_memory:
                conn.close()

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Retrieves user profile dictionary by username without exposing password hash."""
        if not username:
            return None
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT user_id, username, email, role, is_active,
                       created_at_utc, last_login_utc
                FROM nexus_users
                WHERE username = ? COLLATE NOCASE;
                """,
                (username.strip(),),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "user_id": row["user_id"],
                "username": row["username"],
                "email": row["email"],
                "role": row["role"],
                "is_active": bool(row["is_active"]),
                "created_at_utc": row["created_at_utc"],
                "last_login_utc": row["last_login_utc"],
            }
        finally:
            if not self._is_memory:
                conn.close()

    def list_users(self, actor_username: str) -> List[Dict[str, Any]]:
        """
        Lists all users without exposing password hashes. Restricted to ADMIN role.
        """
        actor = self.get_user_by_username(actor_username)
        if not actor or actor["role"] != UserRole.ADMIN:
            return []

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT user_id, username, email, role, is_active,
                       created_at_utc, last_login_utc
                FROM nexus_users
                ORDER BY created_at_utc ASC;
                """
            )
            rows = cursor.fetchall()
            return [
                {
                    "user_id": r["user_id"],
                    "username": r["username"],
                    "email": r["email"],
                    "role": r["role"],
                    "is_active": bool(r["is_active"]),
                    "created_at_utc": r["created_at_utc"],
                    "last_login_utc": r["last_login_utc"],
                }
                for r in rows
            ]
        finally:
            if not self._is_memory:
                conn.close()

    # =========================================================================
    # Security Audit Event Logging (Append-Only)
    # =========================================================================

    def record_logout(self, username: str) -> None:
        """Records user logout event."""
        if username:
            self._record_audit_event(
                event_type="AUTH_LOGOUT",
                actor_username=username,
                target_username=username,
                details={"action": "User initiated logout"},
            )

    def record_session_expired(self, username: str) -> None:
        """Records session expiration due to inactivity."""
        if username:
            self._record_audit_event(
                event_type="AUTH_SESSION_EXPIRED",
                actor_username=username,
                target_username=username,
                details={"action": "Session expired due to inactivity"},
            )

    def _record_audit_event(
        self,
        event_type: str,
        actor_username: str,
        target_username: Optional[str],
        details: Dict[str, Any],
    ) -> None:
        """Appends a security audit event record."""
        conn = self._get_connection()
        try:
            with conn:
                self._record_audit_event_cursor(conn.cursor(), event_type, actor_username, target_username, details)
        finally:
            if not self._is_memory:
                conn.close()

    def _record_audit_event_cursor(
        self,
        cursor: sqlite3.Cursor,
        event_type: str,
        actor_username: str,
        target_username: Optional[str],
        details: Dict[str, Any],
    ) -> None:
        """Helper to insert security audit event within existing transaction cursor."""
        event_id = f"AUTH-EVT-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        now_utc = get_utc_iso_timestamp()

        # Sanitize details to guarantee no secrets or hashes leak into audit log
        clean_details = {k: v for k, v in details.items() if "password" not in k.lower() and "hash" not in k.lower() and "token" not in k.lower()}

        cursor.execute(
            """
            INSERT INTO nexus_auth_audit_events (
                event_id, timestamp_utc, event_type,
                actor_username, target_username, details_json
            ) VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                event_id,
                now_utc,
                event_type,
                actor_username or "ANONYMOUS",
                target_username or actor_username or "SYSTEM",
                json.dumps(clean_details),
            ),
        )

    def get_auth_audit_events(self, actor_username: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Returns recent authentication audit events. Restricted to ADMIN role.
        """
        actor = self.get_user_by_username(actor_username)
        if not actor or actor["role"] != UserRole.ADMIN:
            return []

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT event_id, timestamp_utc, event_type,
                       actor_username, target_username, details_json
                FROM nexus_auth_audit_events
                ORDER BY timestamp_utc DESC
                LIMIT ?;
                """,
                (max(1, limit),),
            )
            rows = cursor.fetchall()
            events = []
            for r in rows:
                try:
                    dt = json.loads(r["details_json"])
                except Exception:
                    dt = {}
                events.append({
                    "event_id": r["event_id"],
                    "timestamp_utc": r["timestamp_utc"],
                    "event_type": r["event_type"],
                    "actor_username": r["actor_username"],
                    "target_username": r["target_username"],
                    "details": dt,
                })
            return events
        finally:
            if not self._is_memory:
                conn.close()


def main() -> None:
    """CLI utility for managing Nexus-AI authentication."""
    import argparse
    import getpass

    parser = argparse.ArgumentParser(description="Nexus-AI Security & User Authentication CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # init-admin
    init_p = subparsers.add_parser("init-admin", help="Provision the initial administrator account")
    init_p.add_argument("--username", help="Administrator username")
    init_p.add_argument("--email", help="Administrator email")
    init_p.add_argument("--password", help="Administrator password (prompted if omitted)")
    init_p.add_argument("--db", default="data/nexus_auth.db", help="Path to auth database")

    # list-users
    list_p = subparsers.add_parser("list-users", help="List all registered users (Admin only)")
    list_p.add_argument("--admin", required=True, help="Admin username")
    list_p.add_argument("--db", default="data/nexus_auth.db", help="Path to auth database")

    # create-user
    create_p = subparsers.add_parser("create-user", help="Create a new user (Admin only)")
    create_p.add_argument("--admin", required=True, help="Admin username")
    create_p.add_argument("--username", required=True, help="New username")
    create_p.add_argument("--email", required=True, help="New user email")
    create_p.add_argument("--role", required=True, choices=["ADMIN", "REVIEWER", "VIEWER"], help="User role")
    create_p.add_argument("--password", help="New user password (prompted if omitted)")
    create_p.add_argument("--db", default="data/nexus_auth.db", help="Path to auth database")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    mgr = AuthManager(db_path=args.db)

    if args.command == "init-admin":
        user = args.username or input("Enter initial admin username: ").strip()
        email = args.email or input("Enter initial admin email: ").strip()
        pw = args.password or getpass.getpass("Enter initial admin password: ")
        success, msg = mgr.init_first_admin(user, email, pw)
        print(("SUCCESS: " if success else "ERROR: ") + msg)

    elif args.command == "list-users":
        users = mgr.list_users(actor_username=args.admin)
        if not users:
            print("No users found or actor is not authorized.")
        else:
            print(f"{'Username':<18} {'Email':<30} {'Role':<12} {'Status':<10} {'Created (UTC)':<24}")
            print("-" * 96)
            for u in users:
                status = "ACTIVE" if u["is_active"] else "DISABLED"
                print(f"{u['username']:<18} {u['email']:<30} {u['role']:<12} {status:<10} {u['created_at_utc']:<24}")

    elif args.command == "create-user":
        pw = args.password or getpass.getpass(f"Enter password for '{args.username}': ")
        success, msg, _ = mgr.create_user(args.admin, args.username, args.email, pw, args.role)
        print(("SUCCESS: " if success else "ERROR: ") + msg)


if __name__ == "__main__":
    main()

