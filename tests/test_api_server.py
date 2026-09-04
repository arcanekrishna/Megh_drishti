"""
API & Server Unit Tests (Zero-Socket / In-Process)
-------------------------------------------------
Validates that all REST endpoints return valid JSON and expected schemas.
"""

import os
import sys
import unittest
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from api.server import (
    get_system_status,
    get_nowcast_data,
    get_active_alerts,
    get_xai_attribution,
    trigger_simulation,
    state
)

class TestAPIServerEndpoints(unittest.TestCase):
    def test_status_endpoint(self):
        status = get_system_status()
        self.assertEqual(status["status"], "OPERATIONAL")
        self.assertIn("spatial_resolution", status)
        self.assertIn("prediction_lead_time", status)
        self.assertGreater(status["active_alert_count"], 0)

    def test_nowcast_endpoint(self):
        nowcast = get_nowcast_data(step=1)
        self.assertEqual(nowcast["lead_time_hours"], 1.0)
        self.assertIn("probabilities", nowcast)
        self.assertIn("cloudburst", nowcast["probabilities"])
        self.assertIn("severe_thunderstorm", nowcast["probabilities"])
        self.assertIn("flash_flood", nowcast["probabilities"])
        self.assertIn("hotspots", nowcast)

    def test_alerts_endpoint(self):
        alerts_resp = get_active_alerts()
        self.assertIn("alerts", alerts_resp)
        self.assertGreater(alerts_resp["count"], 0)
        
        first = alerts_resp["alerts"][0]
        self.assertIn("severity", first)
        self.assertIn("hazard_type", first)
        self.assertIn("actionable_sop", first)
        self.assertIn("coordinates", first)

    def test_xai_endpoint(self):
        xai_resp = get_xai_attribution(lat=31.7, lon=76.93, step=1)
        self.assertIn("trigger_breakdown_percent", xai_resp)
        self.assertIn("meteorological_rationale", xai_resp)
        self.assertIn("dominant_trigger", xai_resp)

    def test_simulation_trigger(self):
        res = trigger_simulation("uttarakhand_kedarnath")
        self.assertEqual(res["scenario"], "uttarakhand_kedarnath")
        self.assertEqual(state.current_scenario, "uttarakhand_kedarnath")


if __name__ == "__main__":
    unittest.main()
