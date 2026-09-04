import os
import sys
import numpy as np

# Ensure data_loaders package can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), "data_loaders"))

from fetch_insat_satellite import generate_insat_timeseries, compute_ctt_drop_rate
from fetch_thermodynamics import fetch_grid_thermodynamics
from fetch_dem_topography import generate_or_fetch_dem
from fusion_pipeline import MultimodalWeatherFusionEngine

def run_pipeline(num_timesteps: int = 120, grid_resolution: float = 0.04):
    """
    Runs end-to-end multi-modal data fusion with 3-digit spatiotemporal dimensions:
    - num_timesteps: e.g. 120 frames (60 hours at 30-min intervals)
    - grid_resolution: 0.04° (~4km grid over India, yielding 100x100 spatial grids)
    """
    print("=" * 75)
    print("⚡ AI-Driven Hyper-Local Weather Nowcasting: Spatiotemporal Pipeline")
    print("=" * 75)
    
    # Target convective region: Uttarakhand / Himachal / North Convective Zone
    # Lat: 28.0N to 32.0N (100 grid points at 0.04° step)
    # Lon: 76.0E to 80.0E (100 grid points at 0.04° step)
    lat_min, lat_max = 28.0, 32.0
    lon_min, lon_max = 76.0, 80.0
    
    print(f"\n[Step 1/4] Ingesting INSAT-3D/3DR Satellite Multi-Spectral Time Series ({num_timesteps} Steps)...")
    sat_series = generate_insat_timeseries(
        start_time="2023-07-08T00:00:00",
        num_timesteps=num_timesteps,
        freq_minutes=30,
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max,
        resolution=grid_resolution
    )
    print(f" -> Satellite Dimensions: {dict(sat_series.dims)}")
    
    print(f"\n[Step 2/4] Ingesting Atmospheric Instability & Thermodynamics (CAPE, CIN, IWV, Shear)...")
    thermo_ds = fetch_grid_thermodynamics(
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max,
        start_date="2023-07-08", end_date="2023-07-13",
        grid_step=grid_resolution
    )
    print(f" -> Thermodynamics Dimensions: {dict(thermo_ds.dims)}")
    
    print(f"\n[Step 3/4] Ingesting Topographic DEM & Runoff Channeling Baselines...")
    dem_ds = generate_or_fetch_dem(
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max,
        resolution=grid_resolution
    )
    print(f" -> Topography Dimensions: {dict(dem_ds.dims)}")
    
    print(f"\n[Step 4/4] Multi-Modal Spatiotemporal Fusion & Tensor Construction...")
    engine = MultimodalWeatherFusionEngine(target_resolution=grid_resolution)
    fused_ds = engine.align_and_fuse(sat_series, thermo_ds, dem_ds)
    
    print(f"\n📌 Fused Multi-Modal Spatiotemporal Dataset Output:")
    print(f"   -> Dimension Sizes: {dict(fused_ds.dims)}")
    
    tensor, feat_names = engine.extract_model_tensors(fused_ds)
    print(f"   -> Full Tensor Shape: {tensor.shape}  [Time={tensor.shape[0]}, Channels={tensor.shape[1]}, Lat={tensor.shape[2]}, Lon={tensor.shape[3]}]")
    
    print("\n[Step 5/5] Generating Supervised Sliding Windows for 2-6 Hour Nowcasting...")
    # in_steps = 6 frames (past 3 hrs precursors), out_steps = 6 frames (future 3 hrs lead time)
    X, Y = engine.create_spatiotemporal_sliding_windows(tensor, in_steps=6, out_steps=6)
    
    print("\n" + "=" * 75)
    print("🎉 SUCCESS: Multi-modal Data Matrix Ready for AI Model Training & Inference!")
    print(f"   Input X Shape:  {X.shape}  -> ({X.shape[0]} samples, {X.shape[1]} past frames, {X.shape[2]} channels, {X.shape[3]}x{X.shape[4]} grid)")
    print(f"   Target Y Shape: {Y.shape}  -> ({Y.shape[0]} samples, {Y.shape[1]} lead-time frames, {Y.shape[2]} channels, {Y.shape[3]}x{Y.shape[4]} grid)")
    print(f"   Feature Channels:")
    for idx, name in enumerate(feat_names):
        print(f"     [{idx}] {name}")
    print("=" * 75)
    
    return fused_ds, tensor, X, Y

if __name__ == "__main__":
    # Generate 120 timesteps over a 100x100 spatial grid
    run_pipeline(num_timesteps=120, grid_resolution=0.04)

