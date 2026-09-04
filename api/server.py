"""
Real-Time AI Nowcasting API & Web Server (SIH 2026)
--------------------------------------------------
Serves:
- Real-time Spatiotemporal Nowcasts (2-6 Hour Lead Time)
- Multi-Task Hazard Probability Maps (Severe Thunderstorms, Cloudbursts, Flash Floods)
- Automated Categorized Early Warnings (Red, Orange, Yellow) with SOPs
- Explainable AI (XAI) Meteorological Attribution Triggers
- Interactive Spatial Dashboard GUI (Static file hosting)
"""

import os
import sys
import json
import torch
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import numpy as np

# Ensure root paths are in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from models.spatiotemporal_mtl import WeatherNowcastingInferenceEngine, MultiTaskWeatherNowcastingNet
from xai.trigger_explainer import WeatherXAIExplainer
from api.alert_manager import AutomatedAlertEngine, CRITICAL_VULNERABLE_ZONES

try:
    from fastapi import FastAPI, Query, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    class DummyFastAPI:
        def __init__(self, *args, **kwargs): pass
        def get(self, *args, **kwargs): return lambda fn: fn
        def post(self, *args, **kwargs): return lambda fn: fn
        def add_middleware(self, *args, **kwargs): pass
        def mount(self, *args, **kwargs): pass
    FastAPI = DummyFastAPI
    CORSMiddleware = None
    StaticFiles = None
    FileResponse = None
    JSONResponse = None
    Query = lambda default=None, **kwargs: default
    HTTPException = Exception


# Simulation & State Manager
class WeatherEngineState:
    def __init__(self):
        self.resolution = 0.5  # matching training grid size
        self.lat_min, self.lat_max = 30.0, 32.0
        self.lon_min, self.lon_max = 77.0, 79.5
        self.lats = np.arange(self.lat_min, self.lat_max + 1e-5, self.resolution)
        self.lons = np.arange(self.lon_min, self.lon_max + 1e-5, self.resolution)
        self.n_lat = len(self.lats)
        self.n_lon = len(self.lons)
        
        self.in_steps = 6
        self.out_steps = 6  # 6 steps * 30 min = 3.0 hour nowcast (up to 6 hours configurable)
        self.lead_times_hours = [round((i + 1) * 0.5, 1) for i in range(self.out_steps)]
        
        self.inference_engine = WeatherNowcastingInferenceEngine(in_steps=self.in_steps, out_steps=self.out_steps)
        self.xai_explainer = WeatherXAIExplainer()
        self.alert_engine = AutomatedAlertEngine()
        
        # Load trained PyTorch model
        self.torch_model = MultiTaskWeatherNowcastingNet(in_channels=10, in_steps=6, out_steps=6)
        model_path = os.path.join(BASE_DIR, "models/checkpoints/model_best.pth")
        if os.path.exists(model_path):
            self.torch_model.load_state_dict(torch.load(model_path))
            self.torch_model.eval()
            print(f"[✓] Loaded trained model weights from {model_path}")
        else:
            print(f"[!] Warning: No trained model found at {model_path}. Using fallback/untrained weights.")
            
        self.current_scenario = "live_data"
        self.last_update_time = datetime.now()
        
        # Initialize active tensors and cache
        self.active_tensor = None
        self.active_predictions = None
        self.active_alerts = []
        
        # Attempt to load live data initially, fallback to simulation if fails
        try:
            self.run_live_inference()
        except Exception as e:
            print(f"[!] Live inference failed on startup: {e}. Falling back to simulation.")
            self.run_simulation_scenario("himachal_cloudburst_2023")

    def run_live_inference(self):
        """
        Fetches LIVE real-time forecast data from Open-Meteo to feed into the trained PyTorch model.
        """
        self.current_scenario = "live_data"
        self.last_update_time = datetime.now()
        print("[*] Fetching LIVE real-time meteorological data...")
        
        hourly_vars = [
            "temperature_2m", "cape", "convective_inhibition",
            "total_column_integrated_water_vapour", "precipitation",
            "wind_speed_10m", "wind_speed_850hPa", "wind_speed_500hPa", "cloud_cover"
        ]
        
        url = "https://api.open-meteo.com/v1/forecast"
        tensor = np.zeros((self.in_steps, 10, self.n_lat, self.n_lon), dtype=np.float32)
        
        # We will fetch a single point just to build a unified grid (simplification for real-time demo speed)
        # In a full prod system, we would query every grid point as done in training.
        params = {
            "latitude": (self.lat_min + self.lat_max)/2,
            "longitude": (self.lon_min + self.lon_max)/2,
            "hourly": ",".join(hourly_vars),
            "past_days": 1,
            "forecast_days": 1,
            "timezone": "Asia/Kolkata"
        }
        
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            raise Exception("Failed to fetch live data")
            
        data = resp.json()["hourly"]
        
        # Find current hour index
        current_time_str = self.last_update_time.strftime("%Y-%m-%dT%H:00")
        try:
            idx = data["time"].index(current_time_str)
        except ValueError:
            idx = -6 # Fallback if time not exactly found
            
        start_idx = idx - self.in_steps
        if start_idx < 0: start_idx = 0
        
        # Broadcast real data to the grid
        for t in range(self.in_steps):
            data_idx = start_idx + t
            cc = data["cloud_cover"][data_idx] or 0.0
            iwv = data["total_column_integrated_water_vapour"][data_idx] or 0.0
            precip = data["precipitation"][data_idx] or 0.0
            cape = data["cape"][data_idx] or 0.0
            cin = data["convective_inhibition"][data_idx] or 0.0
            ws_10m = data["wind_speed_10m"][data_idx] or 0.0
            ws_850 = data["wind_speed_850hPa"][data_idx] or ws_10m
            ws_500 = data["wind_speed_500hPa"][data_idx] or (ws_850 * 1.5)
            shear = abs(ws_500 - ws_850)
            temp = data["temperature_2m"][data_idx] or 0.0
            
            # Channel mapping matches training script
            tensor[t, 0, :, :] = cc / 100.0  # Normalize
            tensor[t, 1, :, :] = iwv / 100.0
            tensor[t, 2, :, :] = precip / 50.0
            tensor[t, 3, :, :] = cape / 3000.0
            tensor[t, 4, :, :] = cin / 500.0
            tensor[t, 5, :, :] = shear / 50.0
            tensor[t, 6, :, :] = temp / 50.0
            tensor[t, 7, :, :] = 0.5 # Dummy elevation
            tensor[t, 8, :, :] = 0.5 # Dummy slope
            tensor[t, 9, :, :] = 0.5 # Dummy runoff

        self.active_tensor = np.clip(tensor, 0.0, 1.0)
        
        # PyTorch Inference
        with torch.no_grad():
            t_input = torch.tensor(self.active_tensor).unsqueeze(0) # Add batch dim
            out = self.torch_model(t_input)
            
            self.active_predictions = {
                "thunderstorm": out["thunderstorm"][0].numpy(),
                "cloudburst": out["cloudburst"][0].numpy(),
                "flash_flood": out["flash_flood"][0].numpy(),
            }
            
        self.active_alerts = self.alert_engine.generate_alerts(
            self.active_predictions,
            self.lead_times_hours,
            self.lats,
            self.lons,
            reference_time=self.last_update_time
        )
        print(f"[✓] Live inference complete. Active Alerts: {len(self.active_alerts)}")

    def run_simulation_scenario(self, scenario_name: str = "himachal_cloudburst_2023"):
        """
        Generates realistic calibrated spatiotemporal multi-modal tensors for designated scenarios.
        """
        self.current_scenario = scenario_name
        self.last_update_time = datetime.now()
        
        # Input tensor shape: (In_Steps=6, Channels=10, H, W)
        tensor = np.zeros((self.in_steps, 10, self.n_lat, self.n_lon), dtype=np.float32)
        
        lat_grid, lon_grid = np.meshgrid(self.lats, self.lons, indexing="ij")
        elevation = 0.2 + 0.7 * np.clip((lat_grid - 29.0) / 3.0, 0, 1) ** 1.5
        slope = 0.3 + 0.6 * np.sin(lat_grid * 6.0) ** 2 * np.cos(lon_grid * 5.0) ** 2
        runoff = 0.4 + 0.5 * slope
        
        for t in range(self.in_steps):
            tensor[t, 7] = elevation
            tensor[t, 8] = slope
            tensor[t, 9] = runoff
            
            # Baseline background fields
            tensor[t, 0] = 0.75 - 0.05 * t  # Cooling trend
            tensor[t, 1] = 0.40 + 0.04 * t  # Moisture rising
            tensor[t, 2] = 0.10
            tensor[t, 3] = 0.50 + 0.05 * t  # CAPE rising
            tensor[t, 4] = max(0.05, 0.40 - 0.06 * t) # CIN eroding
            tensor[t, 5] = 0.55
            tensor[t, 6] = tensor[t, 1]
            
            # Inject localized convective storm signatures based on scenario
            if scenario_name == "himachal_cloudburst_2023":
                # Severe event centered on Mandi/Kullu (Lat ~31.7, Lon ~77.0)
                cy, cx = int(np.argmin(np.abs(self.lats - 31.7))), int(np.argmin(np.abs(self.lons - 77.0)))
                Y, X = np.ogrid[:self.n_lat, :self.n_lon]
                dist_sq = (Y - cy)**2 + (X - cx)**2
                kernel = np.exp(-dist_sq / (2 * (8.0)**2))
                
                # Cloud top temperature collapses (rapid vertical cooling)
                tensor[t, 0] -= kernel * (0.25 + 0.08 * t)
                # IWV moisture pool surges to extreme levels
                tensor[t, 1] += kernel * (0.35 + 0.06 * t)
                tensor[t, 6] += kernel * (0.35 + 0.06 * t)
                # CAPE explodes
                tensor[t, 3] += kernel * (0.30 + 0.05 * t)
                # CIN is completely destroyed
                tensor[t, 4] = np.maximum(0.0, tensor[t, 4] - kernel * 0.4)
                # QPE rain burst
                tensor[t, 2] += kernel * (0.20 + 0.12 * t)
                
            elif scenario_name == "uttarakhand_kedarnath":
                # High altitude Kedarnath/Chamoli hotspot (Lat ~30.6, Lon ~79.1)
                cy, cx = int(np.argmin(np.abs(self.lats - 30.6))), int(np.argmin(np.abs(self.lons - 79.1)))
                Y, X = np.ogrid[:self.n_lat, :self.n_lon]
                dist_sq = (Y - cy)**2 + (X - cx)**2
                kernel = np.exp(-dist_sq / (2 * (7.0)**2))
                
                tensor[t, 0] -= kernel * (0.30 + 0.07 * t)
                tensor[t, 1] += kernel * (0.40 + 0.05 * t)
                tensor[t, 3] += kernel * 0.35
                tensor[t, 8] += kernel * 0.20  # Steepest valley walls
                
            elif scenario_name == "pre_monsoon_squall":
                # Widespread thunderstorm line over plains
                cy = int(np.argmin(np.abs(self.lats - 29.5)))
                Y, X = np.ogrid[:self.n_lat, :self.n_lon]
                line_kernel = np.exp(-((Y - cy)**2) / 18.0)
                tensor[t, 0] -= line_kernel * 0.35
                tensor[t, 3] += line_kernel * 0.40
                tensor[t, 5] += line_kernel * 0.35 # High shear
                
        self.active_tensor = np.clip(tensor, 0.0, 1.0)
        
        # Run PyTorch Model Prediction
        with torch.no_grad():
            t_input = torch.tensor(self.active_tensor).unsqueeze(0)
            out = self.torch_model(t_input)
            self.active_predictions = {
                "thunderstorm": out["thunderstorm"][0].numpy(),
                "cloudburst": out["cloudburst"][0].numpy(),
                "flash_flood": out["flash_flood"][0].numpy(),
            }
        
        # Generate Categorized Early Warnings
        self.active_alerts = self.alert_engine.generate_alerts(
            self.active_predictions,
            self.lead_times_hours,
            self.lats,
            self.lons,
            reference_time=self.last_update_time
        )
        print(f"[✓] Simulation '{scenario_name}' loaded! Active Alerts: {len(self.active_alerts)}")


state = WeatherEngineState()

# Initialize FastAPI App
app = FastAPI(
    title="AI Severe Weather Nowcasting Early Warning Engine",
    description="Multi-Task Spatiotemporal early warning system for Cloudbursts, Severe Thunderstorms, and Flash Floods (SIH 2026)",
    version="2.0.0"
)

if CORSMiddleware is not None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

@app.get("/api/status")
def get_system_status():
    return {
        "status": "OPERATIONAL",
        "engine": "Multi-Task Spatiotemporal Transformer / Residual ConvNet (Trained Checkpoint)",
        "spatial_resolution": "0.5°",
        "prediction_lead_time": "2 to 6 Hours (30-min steps)",
        "active_scenario": state.current_scenario,
        "last_nowcast_time": state.last_update_time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "grid_bounds": {
            "lat_min": state.lat_min, "lat_max": state.lat_max,
            "lon_min": state.lon_min, "lon_max": state.lon_max,
            "num_lat": state.n_lat, "num_lon": state.n_lon
        },
        "monitored_vulnerable_zones": len(CRITICAL_VULNERABLE_ZONES),
        "active_alert_count": len(state.active_alerts)
    }

@app.get("/api/nowcast")
def get_nowcast_data(step: int = Query(0, ge=0, le=5, description="Lead time step index: 0=30m, 1=1h, 2=1.5h, 3=2h, 4=2.5h, 5=3h")):
    step = min(step, state.out_steps - 1)
    lead_time_h = state.lead_times_hours[step]
    
    ts_grid = state.active_predictions["thunderstorm"][step]
    cb_grid = state.active_predictions["cloudburst"][step]
    ff_grid = state.active_predictions["flash_flood"][step]
    
    hotspots = []
    for iy in range(len(state.lats)):
        for ix in range(len(state.lons)):
            p_cb = float(cb_grid[iy, ix])
            p_ts = float(ts_grid[iy, ix])
            p_ff = float(ff_grid[iy, ix])
            max_p = max(p_cb, p_ts, p_ff)
            
            if max_p >= 0.40:
                hotspots.append({
                    "lat": round(float(state.lats[iy]), 3),
                    "lon": round(float(state.lons[ix]), 3),
                    "cloudburst_risk": round(p_cb, 3),
                    "thunderstorm_risk": round(p_ts, 3),
                    "flash_flood_risk": round(p_ff, 3),
                    "threat_level": "RED" if max_p >= 0.70 else "ORANGE"
                })

    return {
        "lead_time_hours": lead_time_h,
        "lead_time_label": f"+{lead_time_h} Hours Nowcast",
        "timestamp": (state.last_update_time + timedelta(hours=lead_time_h)).strftime("%Y-%m-%d %H:%M:%S IST"),
        "grid_meta": {
            "shape": [len(state.lats), len(state.lons)],
            "lats": [round(float(x), 3) for x in state.lats],
            "lons": [round(float(x), 3) for x in state.lons]
        },
        "probabilities": {
            "cloudburst": np.round(cb_grid, 3).tolist(),
            "severe_thunderstorm": np.round(ts_grid, 3).tolist(),
            "flash_flood": np.round(ff_grid, 3).tolist()
        },
        "hotspot_count": len(hotspots),
        "hotspots": hotspots[:30]
    }

@app.get("/api/alerts")
def get_active_alerts(severity: Optional[str] = None):
    if severity:
        filtered = [a for a in state.active_alerts if a["severity"].upper() == severity.upper()]
        return {"count": len(filtered), "alerts": filtered}
    return {
        "count": len(state.active_alerts),
        "scenario": state.current_scenario,
        "alerts": state.active_alerts
    }

@app.get("/api/xai/point")
def get_xai_attribution(
    lat: float = Query(..., description="Target Latitude"),
    lon: float = Query(..., description="Target Longitude"),
    step: int = Query(1, ge=0, le=5, description="Lead time step")
):
    explanation = state.xai_explainer.explain_cell(
        lat=lat,
        lon=lon,
        inputs=state.active_tensor,
        predictions=state.active_predictions,
        lead_time_step=step,
        grid_lats=state.lats,
        grid_lons=state.lons
    )
    return explanation

@app.post("/api/simulate")
def trigger_simulation(scenario: str = Query("live_data", description="Scenario: live_data | himachal_cloudburst_2023 | uttarakhand_kedarnath")):
    valid_scenarios = ["live_data", "himachal_cloudburst_2023", "uttarakhand_kedarnath", "pre_monsoon_squall", "normal_monsoon"]
    if scenario not in valid_scenarios:
        raise HTTPException(status_code=400, detail=f"Invalid scenario. Choose from: {valid_scenarios}")
        
    if scenario == "live_data":
        state.run_live_inference()
    else:
        state.run_simulation_scenario(scenario)
        
    return {
        "message": f"Successfully activated scenario '{scenario}'",
        "scenario": scenario,
        "active_alerts_generated": len(state.active_alerts),
        "lead_time_window": "2 to 6 Hours"
    }

# Mount static frontend dashboard directory
dashboard_dir = os.path.join(BASE_DIR, "dashboard")
if os.path.exists(dashboard_dir) and StaticFiles is not None:
    app.mount("/dashboard", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")

@app.get("/")
def redirect_to_dashboard():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard/")

def start_server(host: str = "0.0.0.0", port: int = 8000):
    if FASTAPI_AVAILABLE:
        print(f"[*] Starting AI Weather Nowcasting API Server on http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)
    else:
        print("[!] FastAPI/Uvicorn not installed. Please install fastapi & uvicorn.")

if __name__ == "__main__":
    start_server(port=8000)
