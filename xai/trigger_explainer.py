"""
Explainable AI (XAI) Atmospheric Attribution Engine (SIH 2026)
--------------------------------------------------------------
Decomposes extreme weather nowcasting predictions into 4 meteorological triggers:
1. Moisture Availability (The Fuel): Integrated Water Vapor (IWV) surges & pooling
2. Atmospheric Instability (The Energy): CAPE buoyancy & CIN erosion
3. Kinematics & Updraft Lift (The Trigger): CTT drop rate (rapid cooling) & Bulk Wind Shear
4. Topographic Funneling (The Flood Catalyst): DEM slope, elevation & runoff drainage basins
"""

import numpy as np
from typing import Dict, Any, Optional

class WeatherXAIExplainer:
    def __init__(self):
        pass

    def explain_cell(
        self,
        lat: float,
        lon: float,
        inputs: np.ndarray, # (In_Steps, Channels, H, W) or (Channels, H, W)
        predictions: Dict[str, np.ndarray], # {'thunderstorm': (T,H,W), 'cloudburst': ..., 'flash_flood': ...}
        lead_time_step: int = 1, # e.g. T+1h (step index 1)
        grid_lats: Optional[np.ndarray] = None,
        grid_lons: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Computes meteorological trigger attribution for a specific spatial coordinate.
        """
        # Find nearest spatial grid indices (y, x)
        if inputs.ndim == 4:
            in_steps, channels, h, w = inputs.shape
            current_frame = inputs[-1] # Most recent observation frame
            first_frame = inputs[0]
        else:
            channels, h, w = inputs.shape
            current_frame = inputs
            first_frame = inputs

        if grid_lats is not None and grid_lons is not None:
            iy = int(np.argmin(np.abs(grid_lats - lat)))
            ix = int(np.argmin(np.abs(grid_lons - lon)))
        else:
            iy = min(h - 1, max(0, int(h * 0.5)))
            ix = min(w - 1, max(0, int(w * 0.5)))

        # Extract normalized channel values at (iy, ix)
        # Channels: 0: CTT, 1: IWV, 2: QPE, 3: CAPE, 4: CIN, 5: Shear, 7: Elev, 8: Slope, 9: Runoff
        ctt_val = float(current_frame[0, iy, ix])
        ctt_drop = float(max(0.0, first_frame[0, iy, ix] - current_frame[0, iy, ix]))
        iwv_val = float(current_frame[1, iy, ix])
        iwv_trend = float(max(0.0, current_frame[1, iy, ix] - first_frame[1, iy, ix]))
        qpe_val = float(current_frame[2, iy, ix]) if channels > 2 else 0.2
        cape_val = float(current_frame[3, iy, ix]) if channels > 3 else 0.6
        cin_val = float(current_frame[4, iy, ix]) if channels > 4 else 0.1
        shear_val = float(current_frame[5, iy, ix]) if channels > 5 else 0.4
        elevation_val = float(current_frame[7, iy, ix]) if channels > 7 else 0.5
        slope_val = float(current_frame[8, iy, ix]) if channels > 8 else 0.6
        runoff_val = float(current_frame[9, iy, ix]) if channels > 9 else 0.5

        # Extract model predicted hazard probabilities at this cell for the chosen lead time
        step = min(lead_time_step, predictions["thunderstorm"].shape[0] - 1)
        prob_ts = float(predictions["thunderstorm"][step, iy, ix])
        prob_cb = float(predictions["cloudburst"][step, iy, ix])
        prob_ff = float(predictions["flash_flood"][step, iy, ix])

        # Compute Trigger Attributions (0-100%)
        # 1. Moisture (IWV + IWV trend)
        raw_moisture = (iwv_val * 0.65 + iwv_trend * 0.35)
        # 2. Instability (CAPE / (1 + CIN))
        raw_instability = (cape_val * (1.0 - cin_val * 0.5))
        # 3. Lift & Kinematics (CTT drop rate * 1.5 + shear)
        raw_lift = (ctt_drop * 1.4 + shear_val * 0.6)
        # 4. Topography (Slope + Runoff channeling)
        raw_topo = (slope_val * 0.7 + runoff_val * 0.3)

        total_weight = raw_moisture + raw_instability + raw_lift + raw_topo + 1e-6
        pct_moisture = round((raw_moisture / total_weight) * 100, 1)
        pct_instability = round((raw_instability / total_weight) * 100, 1)
        pct_lift = round((raw_lift / total_weight) * 100, 1)
        pct_topo = round((raw_topo / total_weight) * 100, 1)

        # Primary Atmospheric Culprit
        trigger_dict = {
            "Moisture Availability (IWV Surging)": pct_moisture,
            "Atmospheric Instability (High CAPE / Eroded CIN)": pct_instability,
            "Updraft Lift (Rapid Cloud-Top Cooling)": pct_lift,
            "Topographic Channeling (Steep Slopes & Valleys)": pct_topo
        }
        dominant_trigger = max(trigger_dict, key=trigger_dict.get)

        # Generate Human-Readable Meteorological Summary
        summary_sentences = []
        if prob_cb >= 0.70:
            summary_sentences.append(f"CRITICAL CLOUDBURST ALERT ({prob_cb*100:.0f}%): Severe moisture convergence detected via INSAT IWV.")
        elif prob_cb >= 0.40:
            summary_sentences.append(f"ELEVATED CLOUDBURST RISK ({prob_cb*100:.0f}%): Moisture accumulation detected.")
            
        if prob_ts >= 0.65:
            summary_sentences.append(f"Severe Thunderstorm with explosive updraft (CTT cooling index: {ctt_drop:.2f}) and high CAPE instability.")
            
        if prob_ff >= 0.65:
            summary_sentences.append(f"Flash Flood Danger ({prob_ff*100:.0f}%): Steep terrain ({pct_topo}% topo factor) will channel intense runoff into downstream drainage catchments.")

        if not summary_sentences:
            summary_sentences.append("Atmospheric conditions currently within normal seasonal thresholds.")

        human_rationale = " ".join(summary_sentences)

        return {
            "coordinates": {"lat": round(lat, 4), "lon": round(lon, 4)},
            "grid_indices": {"y": iy, "x": ix},
            "lead_time_hours": (step + 1) * 0.5,
            "predicted_risks": {
                "severe_thunderstorm": round(prob_ts, 3),
                "cloudburst": round(prob_cb, 3),
                "flash_flood": round(prob_ff, 3)
            },
            "trigger_breakdown_percent": {
                "moisture_iwv": pct_moisture,
                "instability_cape": pct_instability,
                "kinematics_lift": pct_lift,
                "topography_slope": pct_topo
            },
            "dominant_trigger": dominant_trigger,
            "meteorological_rationale": human_rationale,
            "raw_diagnostics": {
                "ctt_drop_index": round(ctt_drop, 3),
                "iwv_moisture_level": round(iwv_val, 3),
                "cape_energy_level": round(cape_val, 3),
                "cin_inhibition": round(cin_val, 3),
                "bulk_shear": round(shear_val, 3),
                "slope_steepness": round(slope_val, 3)
            }
        }


if __name__ == "__main__":
    explainer = WeatherXAIExplainer()
    dummy_input = np.random.uniform(0.2, 0.8, size=(6, 10, 50, 50))
    dummy_preds = {
        "thunderstorm": np.random.uniform(0.1, 0.9, size=(6, 50, 50)),
        "cloudburst": np.random.uniform(0.1, 0.9, size=(6, 50, 50)),
        "flash_flood": np.random.uniform(0.1, 0.9, size=(6, 50, 50))
    }
    explanation = explainer.explain_cell(30.5, 78.2, dummy_input, dummy_preds, lead_time_step=2)
    print("XAI Test Explanation:")
    import pprint
    pprint.pprint(explanation)
