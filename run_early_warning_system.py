#!/usr/bin/env python3
"""
MEGH-DRISHTI: AI Weather Nowcasting & Early Warning Master Runner (SIH 2026)
---------------------------------------------------------------------------
Launches the full prototype:
1. Multi-Task Deep Learning Nowcasting Engine (Cloudburst, Thunderstorm, Flash Flood)
2. Explainable AI (XAI) Atmospheric Trigger Attribution Engine
3. Automated Alerting & SOP Dispatcher
4. Interactive GIS Command-Center Dashboard on http://localhost:8000
"""

import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from api.server import state, app, FASTAPI_AVAILABLE

class StandaloneEarlyWarningHandler(SimpleHTTPRequestHandler):
    """
    Standard-library HTTP server fallback supporting REST API endpoints and dashboard UI.
    Guarantees zero-dependency out-of-the-box execution anywhere.
    """
    def __init__(self, *args, **kwargs):
        dashboard_dir = os.path.join(BASE_DIR, "dashboard")
        super().__init__(*args, directory=dashboard_dir, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/api/status":
            self.send_json({
                "status": "OPERATIONAL",
                "engine": "Multi-Task Spatiotemporal Transformer / Residual ConvNet",
                "spatial_resolution": "0.04° (~4km)",
                "prediction_lead_time": "2 to 6 Hours (30-min steps)",
                "active_scenario": state.current_scenario,
                "last_nowcast_time": state.last_update_time.strftime("%Y-%m-%d %H:%M:%S IST"),
                "monitored_vulnerable_zones": 8,
                "active_alert_count": len(state.active_alerts)
            })
        elif path == "/api/nowcast":
            step = int(query.get("step", [1])[0])
            step = max(0, min(step, state.out_steps - 1))
            lead_time_h = state.lead_times_hours[step]
            
            # Subsample grid for fast rendering
            step_down = 2
            sub_lats = state.lats[::step_down]
            sub_lons = state.lons[::step_down]
            
            ts_grid = state.active_predictions["thunderstorm"][step, ::step_down, ::step_down]
            cb_grid = state.active_predictions["cloudburst"][step, ::step_down, ::step_down]
            ff_grid = state.active_predictions["flash_flood"][step, ::step_down, ::step_down]

            hotspots = []
            for iy in range(0, len(sub_lats), 3):
                for ix in range(0, len(sub_lons), 3):
                    p_cb = float(cb_grid[iy, ix])
                    p_ts = float(ts_grid[iy, ix])
                    p_ff = float(ff_grid[iy, ix])
                    max_p = max(p_cb, p_ts, p_ff)
                    if max_p >= 0.40:
                        hotspots.append({
                            "lat": round(float(sub_lats[iy]), 3),
                            "lon": round(float(sub_lons[ix]), 3),
                            "cloudburst_risk": round(p_cb, 3),
                            "thunderstorm_risk": round(p_ts, 3),
                            "flash_flood_risk": round(p_ff, 3),
                            "threat_level": "RED" if max_p >= 0.70 else "ORANGE"
                        })

            self.send_json({
                "lead_time_hours": lead_time_h,
                "lead_time_label": f"+{lead_time_h} Hours Nowcast",
                "timestamp": state.last_update_time.strftime("%Y-%m-%d %H:%M:%S IST"),
                "grid_meta": {
                    "shape": [len(sub_lats), len(sub_lons)],
                    "lats": [round(float(x), 3) for x in sub_lats],
                    "lons": [round(float(x), 3) for x in sub_lons]
                },
                "probabilities": {
                    "cloudburst": cb_grid.round(3).tolist(),
                    "thunderstorm": ts_grid.round(3).tolist(),
                    "flash_flood": ff_grid.round(3).tolist()
                },
                "hotspot_count": len(hotspots),
                "hotspots": hotspots[:30]
            })
        elif path == "/api/alerts":
            self.send_json({
                "count": len(state.active_alerts),
                "scenario": state.current_scenario,
                "alerts": state.active_alerts
            })
        elif path == "/api/xai/point":
            lat = float(query.get("lat", [31.7])[0])
            lon = float(query.get("lon", [76.93])[0])
            step = int(query.get("step", [1])[0])
            explanation = state.xai_explainer.explain_cell(
                lat=lat, lon=lon,
                inputs=state.active_tensor,
                predictions=state.active_predictions,
                lead_time_step=step,
                grid_lats=state.lats,
                grid_lons=state.lons
            )
            self.send_json(explanation)
        else:
            # Serve static files (HTML, CSS, JS)
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/api/simulate":
            scenario = query.get("scenario", ["himachal_cloudburst_2023"])[0]
            if scenario == "live_data":
                state.run_live_inference()
            else:
                state.run_simulation_scenario(scenario)
            self.send_json({
                "message": f"Activated scenario '{scenario}'",
                "scenario": scenario,
                "alerts_count": len(state.active_alerts)
            })
        else:
            self.send_error(404, "Endpoint not found")

    def send_json(self, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def main():
    port = 8000
    print("=" * 75)
    print("⛈️  MEGH-DRISHTI: AI-Driven Hyper-Local Weather Nowcasting Engine (SIH 2026)")
    print("=" * 75)
    print(f"[+] Initialized Multi-Task Model (Thunderstorm, Cloudburst, Flash Flood)")
    print(f"[+] Initialized Explainable AI (XAI) Trigger Attribution Module")
    print(f"[+] Initialized Automated Alerting & SOP Dispatcher")
    print(f"[+] Starting Live Command-Center Web Dashboard on: http://localhost:{port}")
    print("=" * 75)
    
    if FASTAPI_AVAILABLE:
        import uvicorn
        uvicorn.run("api.server:app", host="0.0.0.0", port=port, log_level="info")
    else:
        server = HTTPServer(("0.0.0.0", port), StandaloneEarlyWarningHandler)
        print(f"[✓] Zero-dependency HTTP server running at http://localhost:{port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == "__main__":
    main()
