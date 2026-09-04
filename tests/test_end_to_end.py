"""
End-to-End System Integration & Verification Tests (SIH 2026)
-------------------------------------------------------------
Validates:
1. Multi-Task Deep Learning model predictions and tensor shapes
2. Explainable AI (XAI) attribution calculation and normalization
3. Automated Alert Manager threshold detection and SOP generation
4. Meteorological benchmark metrics (POD, FAR, CSI)
5. API Server state and simulation scenarios
"""

import os
import sys
import unittest
import numpy as np

# Ensure workspace root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from models.spatiotemporal_mtl import WeatherNowcastingInferenceEngine
from models.train_or_evaluate import compute_contingency_metrics
from xai.trigger_explainer import WeatherXAIExplainer
from api.alert_manager import AutomatedAlertEngine
from api.server import WeatherEngineState


class TestEarlyWarningSystem(unittest.TestCase):
    def setUp(self):
        self.in_steps = 6
        self.out_steps = 6
        self.h = 40
        self.w = 40
        self.channels = 10
        self.inference_engine = WeatherNowcastingInferenceEngine(in_steps=self.in_steps, out_steps=self.out_steps)
        self.xai_explainer = WeatherXAIExplainer()
        self.alert_engine = AutomatedAlertEngine()

    def test_01_multitask_inference_shapes(self):
        """Verify model returns correct shapes for all 3 tasks across lead times."""
        dummy_tensor = np.random.uniform(0.1, 0.9, size=(1, self.in_steps, self.channels, self.h, self.w)).astype(np.float32)
        results = self.inference_engine.predict(dummy_tensor)

        self.assertIn("thunderstorm", results)
        self.assertIn("cloudburst", results)
        self.assertIn("flash_flood", results)
        self.assertIn("lead_times_hours", results)
        self.assertIn("max_risk_level", results)

        self.assertEqual(results["thunderstorm"].shape, (self.out_steps, self.h, self.w))
        self.assertEqual(results["cloudburst"].shape, (self.out_steps, self.h, self.w))
        self.assertEqual(results["flash_flood"].shape, (self.out_steps, self.h, self.w))
        self.assertEqual(len(results["lead_times_hours"]), self.out_steps)

        # Probabilities should be strictly in [0, 1]
        self.assertTrue(np.all(results["thunderstorm"] >= 0.0) and np.all(results["thunderstorm"] <= 1.0))
        self.assertTrue(np.all(results["cloudburst"] >= 0.0) and np.all(results["cloudburst"] <= 1.0))
        self.assertTrue(np.all(results["flash_flood"] >= 0.0) and np.all(results["flash_flood"] <= 1.0))

    def test_02_xai_attribution_percentages(self):
        """Verify XAI attribution breaks down into valid sum-to-100 percentages."""
        dummy_inputs = np.random.uniform(0.1, 0.9, size=(self.in_steps, self.channels, self.h, self.w)).astype(np.float32)
        dummy_preds = {
            "thunderstorm": np.random.uniform(0.1, 0.9, size=(self.out_steps, self.h, self.w)),
            "cloudburst": np.random.uniform(0.1, 0.9, size=(self.out_steps, self.h, self.w)),
            "flash_flood": np.random.uniform(0.1, 0.9, size=(self.out_steps, self.h, self.w))
        }

        explanation = self.xai_explainer.explain_cell(
            lat=31.7, lon=76.93,
            inputs=dummy_inputs,
            predictions=dummy_preds,
            lead_time_step=1
        )

        self.assertIn("trigger_breakdown_percent", explanation)
        self.assertIn("dominant_trigger", explanation)
        self.assertIn("meteorological_rationale", explanation)

        breakdown = explanation["trigger_breakdown_percent"]
        total_pct = sum(breakdown.values())
        # Should sum to approximately 100% (+/- rounding tolerance)
        self.assertAlmostEqual(total_pct, 100.0, delta=1.5)

    def test_03_alert_generation_and_sops(self):
        """Verify prioritized alerts and SOP checklists are generated."""
        lats = np.linspace(28.0, 32.5, self.h)
        lons = np.linspace(75.5, 80.5, self.w)
        
        # High risk synthetic prediction (triggering RED alert)
        preds = {
            "thunderstorm": np.full((self.out_steps, self.h, self.w), 0.75, dtype=np.float32),
            "cloudburst": np.full((self.out_steps, self.h, self.w), 0.85, dtype=np.float32),
            "flash_flood": np.full((self.out_steps, self.h, self.w), 0.80, dtype=np.float32)
        }

        alerts = self.alert_engine.generate_alerts(preds, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], lats, lons)
        self.assertGreater(len(alerts), 0)

        top_alert = alerts[0]
        self.assertEqual(top_alert["severity"], "RED")
        self.assertEqual(top_alert["hazard_type"], "CLOUDBURST")
        self.assertIn("actionable_sop", top_alert)
        self.assertGreater(len(top_alert["actionable_sop"]), 0)

    def test_04_contingency_metrics(self):
        """Verify verification metrics (POD, FAR, CSI) calculate correctly."""
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([0.9, 0.8, 0.2, 0.7, 0.1])
        # Hits: 2 (indices 0, 1), Misses: 1 (index 2), False Alarms: 1 (index 3), Correct Neg: 1 (index 4)
        metrics = compute_contingency_metrics(y_true, y_pred, threshold=0.5)

        self.assertEqual(metrics["hits"], 2)
        self.assertEqual(metrics["misses"], 1)
        self.assertEqual(metrics["false_alarms"], 1)
        self.assertAlmostEqual(metrics["pod"], 2/3, places=2)
        self.assertAlmostEqual(metrics["far"], 1/3, places=2)
        self.assertAlmostEqual(metrics["csi"], 2/4, places=2)

    def test_05_simulation_scenarios(self):
        """Verify simulation scenarios switch correctly."""
        state = WeatherEngineState()
        self.assertEqual(state.current_scenario, "himachal_cloudburst_2023")
        self.assertGreater(len(state.active_alerts), 0)

        # Switch to Kedarnath
        state.run_simulation_scenario("uttarakhand_kedarnath")
        self.assertEqual(state.current_scenario, "uttarakhand_kedarnath")
        self.assertIsNotNone(state.active_predictions)


if __name__ == "__main__":
    unittest.main()
