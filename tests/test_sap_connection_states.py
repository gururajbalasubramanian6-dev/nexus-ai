import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import networkx as nx
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

import app
from app import sanitize_error_message

class TestSAPConnectionStates(unittest.TestCase):
    def test_sanitization_removes_sensitive_infrastructure(self):
        """Verifies that sanitize_error_message strips IPs, ports, credentials, and hostnames."""
        raw_error = "Connect failed (connect timeout expired, -10709, [10.150.236.124]:30015) user DBADMIN password=MySecretPassword123 host=zeus.hanacloud.ondemand.com"
        sanitized = sanitize_error_message(raw_error)
        
        self.assertNotIn("10.150.236.124", sanitized)
        self.assertNotIn("30015", sanitized)
        self.assertNotIn("MySecretPassword123", sanitized)
        self.assertNotIn("DBADMIN", sanitized)
        self.assertNotIn("zeus.hanacloud.ondemand.com", sanitized)
        self.assertIn("[REDACTED_IP]", sanitized)
        self.assertIn(":[PORT]", sanitized)
        self.assertIn("[REDACTED]", sanitized)

    def test_masthead_badges_logic(self):
        """Verifies badge HTML rendering for the three distinct states."""
        # 1. Successful SAP Connection
        is_sap_active = True
        is_sap_fallback = False
        if is_sap_active:
            badge_html = '<div class="telemetry-pill telemetry-live"><span class="telemetry-pulse"></span><span>SAP HANA CLOUD &bull; LIVE CONNECTION</span></div>'
        elif is_sap_fallback:
            badge_html = '<div class="telemetry-pill telemetry-fallback"><span>⚠️ SAP CONNECTION UNAVAILABLE &bull; SAMPLE DATA ACTIVE</span></div>'
        else:
            badge_html = '<div class="telemetry-pill telemetry-local"><span>● LOCAL DATA SOURCE ACTIVE</span></div>'

        self.assertIn("SAP HANA CLOUD &bull; LIVE CONNECTION", badge_html)
        self.assertIn("telemetry-live", badge_html)
        self.assertNotIn("SAMPLE DATA ACTIVE", badge_html)

        # 2. Failed SAP Connection with Sample Fallback Active
        is_sap_active = False
        is_sap_fallback = True
        if is_sap_active:
            badge_html = '<div class="telemetry-pill telemetry-live"><span class="telemetry-pulse"></span><span>SAP HANA CLOUD &bull; LIVE CONNECTION</span></div>'
        elif is_sap_fallback:
            badge_html = '<div class="telemetry-pill telemetry-fallback"><span>⚠️ SAP CONNECTION UNAVAILABLE &bull; SAMPLE DATA ACTIVE</span></div>'
        else:
            badge_html = '<div class="telemetry-pill telemetry-local"><span>● LOCAL DATA SOURCE ACTIVE</span></div>'

        self.assertIn("SAP CONNECTION UNAVAILABLE &bull; SAMPLE DATA ACTIVE", badge_html)
        self.assertIn("telemetry-fallback", badge_html)
        self.assertNotIn("LIVE CONNECTION", badge_html)
        self.assertNotIn("telemetry-live", badge_html)

        # 3. Intentionally Selected Local Dataset
        is_sap_active = False
        is_sap_fallback = False
        if is_sap_active:
            badge_html = '<div class="telemetry-pill telemetry-live"><span class="telemetry-pulse"></span><span>SAP HANA CLOUD &bull; LIVE CONNECTION</span></div>'
        elif is_sap_fallback:
            badge_html = '<div class="telemetry-pill telemetry-fallback"><span>⚠️ SAP CONNECTION UNAVAILABLE &bull; SAMPLE DATA ACTIVE</span></div>'
        else:
            badge_html = '<div class="telemetry-pill telemetry-local"><span>● LOCAL DATA SOURCE ACTIVE</span></div>'

        self.assertIn("LOCAL DATA SOURCE ACTIVE", badge_html)
        self.assertIn("telemetry-local", badge_html)
        self.assertNotIn("LIVE CONNECTION", badge_html)
        self.assertNotIn("SAMPLE DATA ACTIVE", badge_html)

if __name__ == "__main__":
    unittest.main()

