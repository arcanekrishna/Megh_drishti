"""
Automated Weather Alerting & Early Warning Engine (SIH 2026)
------------------------------------------------------------
Monitors multi-task hazard probability grids and generates prioritized early warnings
with 2-6 hour actionable lead times for Disaster Management Authorities (NDMA/SDMA/DDMA).
"""

import uuid
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Any, Optional

# Representative vulnerable regions in Western/Central Himalayas & Convective Zones
CRITICAL_VULNERABLE_ZONES = [
    {"name": "Mandi - Beas Valley, Himachal Pradesh", "lat": 31.70, "lon": 76.93, "basin": "Beas River Basin"},
    {"name": "Kullu - Parvati Catchment, Himachal Pradesh", "lat": 31.95, "lon": 77.10, "basin": "Upper Beas Basin"},
    {"name": "Shimla - Rampur Corridor, Himachal Pradesh", "lat": 31.10, "lon": 77.17, "basin": "Sutlej River Basin"},
    {"name": "Kedarnath - Mandakini Valley, Uttarakhand", "lat": 30.73, "lon": 79.06, "basin": "Mandakini River Catchment"},
    {"name": "Chamoli - Alaknanda Valley, Uttarakhand", "lat": 30.41, "lon": 79.33, "basin": "Alaknanda River Catchment"},
    {"name": "Uttarkashi - Bhagirathi Gorge, Uttarakhand", "lat": 30.72, "lon": 78.44, "basin": "Bhagirathi River Basin"},
    {"name": "Dehradun - Doon Valley, Uttarakhand", "lat": 30.31, "lon": 78.03, "basin": "Song River Catchment"},
    {"name": "Dharamsala - Kangra Valley, Himachal Pradesh", "lat": 32.21, "lon": 76.32, "basin": "Gaj Khad Catchment"}
]

class AutomatedAlertEngine:
    def __init__(
        self,
        red_threshold: float = 0.70,
        orange_threshold: float = 0.45,
        yellow_threshold: float = 0.25
    ):
        self.red_threshold = red_threshold
        self.orange_threshold = orange_threshold
        self.yellow_threshold = yellow_threshold

    def generate_alerts(
        self,
        predictions: Dict[str, np.ndarray], # {'thunderstorm': (T,H,W), 'cloudburst': ..., 'flash_flood': ...}
        lead_times_hours: List[float],
        lats: np.ndarray,
        lons: np.ndarray,
        reference_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Scans all nowcast lead-time frames (T+0.5h to T+6h) and generates categorized alerts.
        """
        if reference_time is None:
            reference_time = datetime.now()

        alerts = []
        n_steps, h, w = predictions["cloudburst"].shape

        # Iterate over vulnerable zones and spatial hotspots
        for zone in CRITICAL_VULNERABLE_ZONES:
            # Match zone to nearest grid indices
            iy = int(np.argmin(np.abs(lats - zone["lat"])))
            ix = int(np.argmin(np.abs(lons - zone["lon"])))

            # Check peak threat across all future lead times for this zone
            max_cb_prob = 0.0
            max_ts_prob = 0.0
            max_ff_prob = 0.0
            critical_step = 0

            for t in range(n_steps):
                p_cb = float(predictions["cloudburst"][t, iy, ix])
                p_ts = float(predictions["thunderstorm"][t, iy, ix])
                p_ff = float(predictions["flash_flood"][t, iy, ix])

                if p_cb > max_cb_prob:
                    max_cb_prob = p_cb
                    critical_step = t
                if p_ts > max_ts_prob:
                    max_ts_prob = p_ts
                if p_ff > max_ff_prob:
                    max_ff_prob = p_ff

            overall_max = max(max_cb_prob, max_ts_prob, max_ff_prob)

            if overall_max >= self.yellow_threshold:
                # Determine Severity Category
                if overall_max >= self.red_threshold:
                    severity = "RED"
                    category_title = "CRITICAL EMERGENCY WARNING"
                elif overall_max >= self.orange_threshold:
                    severity = "ORANGE"
                    category_title = "SEVERE WEATHER ALERT"
                else:
                    severity = "YELLOW"
                    category_title = "WEATHER WATCH ADVISORY"

                # Identify primary threatening event
                if max_cb_prob >= max_ts_prob and max_cb_prob >= max_ff_prob:
                    primary_event = "CLOUDBURST"
                    event_description = "High-velocity convective rainburst with extreme rainfall rate (>100mm/hr) potential."
                elif max_ff_prob >= max_ts_prob:
                    primary_event = "FLASH_FLOOD"
                    event_description = "Severe hydrological surge and valley drainage inundation fueled by upstream cloudburst."
                else:
                    primary_event = "SEVERE_THUNDERSTORM"
                    event_description = "Violent convective storm with intense vertical updrafts, lightning, and high wind gusts."

                lead_time_h = lead_times_hours[critical_step] if critical_step < len(lead_times_hours) else 2.0
                impact_eta = reference_time + timedelta(hours=lead_time_h)

                # Prescribe Actionable Standard Operating Procedures (SOPs)
                sop_actions = self._prescribe_sop(severity, primary_event, zone["name"])

                alert = {
                    "alert_id": f"ALRT-{uuid.uuid4().hex[:8].upper()}",
                    "severity": severity,
                    "category_title": category_title,
                    "hazard_type": primary_event,
                    "target_region": zone["name"],
                    "drainage_basin": zone["basin"],
                    "coordinates": {"lat": zone["lat"], "lon": zone["lon"]},
                    "lead_time_hours": lead_time_h,
                    "estimated_time_to_impact": impact_eta.strftime("%Y-%m-%d %H:%M:%S IST"),
                    "peak_probability": round(overall_max * 100, 1),
                    "hazard_probabilities": {
                        "cloudburst": round(max_cb_prob * 100, 1),
                        "severe_thunderstorm": round(max_ts_prob * 100, 1),
                        "flash_flood": round(max_ff_prob * 100, 1)
                    },
                    "description": event_description,
                    "actionable_sop": sop_actions,
                    "issued_at": reference_time.strftime("%Y-%m-%d %H:%M:%S IST")
                }
                alerts.append(alert)

        # Sort alerts by severity (RED > ORANGE > YELLOW) then highest probability
        severity_order = {"RED": 0, "ORANGE": 1, "YELLOW": 2}
        alerts.sort(key=lambda a: (severity_order.get(a["severity"], 3), -a["peak_probability"]))
        return alerts

    def _prescribe_sop(self, severity: str, event_type: str, region: str) -> List[str]:
        """
        Generates standard operating protocol checklists for District Disaster Management Authorities.
        """
        if severity == "RED":
            return [
                "IMMEDIATE EVACUATION: Clear riverside settlements, camp sites, and low-lying riverbeds within 45 minutes.",
                "NDRF / SDRF ALERT: Deploy quick response disaster teams to vulnerable bridges and choke points.",
                "TRAFFIC RESTRICTION: Close high-altitude highway corridors and halt pilgrimage vehicular movement.",
                "DAM & BARRAGE OPERATIONS: Issue pre-release alerts to downstream hydel project authorities."
            ]
        elif severity == "ORANGE":
            return [
                "CIVIC ALERT: Restrict public movement near mountain streams and flood-prone drainage nullahs.",
                "EARTHMOVING GEAR: Position JCBs and clearance machinery at landslide-prone highway bottlenecks.",
                "RELIEF CAMPS: Place emergency shelter centers and medical first-aid teams on active standby."
            ]
        else:
            return [
                "MONITORING: Track high-resolution satellite IWV updates at 15-minute intervals.",
                "COMMUNITY ADVISORY: Send advisory SMS to local Gram Panchayats and tourist hubs."
            ]


if __name__ == "__main__":
    engine = AutomatedAlertEngine()
    lats = np.linspace(28.0, 32.0, 50)
    lons = np.linspace(76.0, 80.0, 50)
    sample_preds = {
        "thunderstorm": np.random.uniform(0.1, 0.9, size=(6, 50, 50)),
        "cloudburst": np.random.uniform(0.1, 0.95, size=(6, 50, 50)),
        "flash_flood": np.random.uniform(0.1, 0.9, size=(6, 50, 50))
    }
    alerts = engine.generate_alerts(sample_preds, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], lats, lons)
    print(f"Generated {len(alerts)} alerts!")
    if alerts:
        print("Top Alert:", alerts[0])
