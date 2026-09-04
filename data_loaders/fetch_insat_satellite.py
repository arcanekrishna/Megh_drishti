"""
INSAT-3D / 3DR Satellite Data Handler & Processor
-------------------------------------------------
Handles ISRO MOSDAC (Meteorological and Oceanographic Satellite Data Archival Centre) products:
1. Thermal Infrared (TIR1 / TIR2) -> Computes Cloud Top Temperature (CTT) & CTT Drop Rate (Updraft speed).
2. Water Vapor (WV) Channel -> Tracks Integrated Water Vapor (IWV) accumulation.
3. Hydro-Estimator / QPE -> High-resolution satellite rainfall rate estimates.

Supports reading HDF5 (.h5) INSAT data and generating calibrated test grids.
"""

import os
try:
    import h5py
except ImportError:
    h5py = None
import numpy as np
import xarray as xr
from datetime import datetime

def read_insat_h5(h5_file_path: str) -> xr.Dataset:
    """
    Parses a native ISRO INSAT-3D / INSAT-3DR HDF5 file and extracts key nowcasting variables.
    """
    if h5py is None:
        raise ImportError("h5py is required to read native .h5 files. Please install with `pip install h5py`.")
    if not os.path.exists(h5_file_path):
        raise FileNotFoundError(f"File not found: {h5_file_path}")

    with h5py.File(h5_file_path, "r") as f:
        # MOSDAC HDF5 typical structures
        print(f"[*] Keys in HDF5 file: {list(f.keys())}")
        
        # Extract metadata
        attrs = dict(f.attrs)
        
        # Extract coordinates (Latitude, Longitude) if present
        # In MOSDAC L1B/L2, coordinates are either stored as lookup tables or calculated via projection
        lat = f["Latitude"][:] if "Latitude" in f else None
        lon = f["Longitude"][:] if "Longitude" in f else None
        
        extracted_vars = {}
        
        # Check for TIR1 / Cloud Top Temperature (Brightness Temperature in Kelvin)
        for key in ["IMG_TIR1", "TIR1", "CTT", "Cloud_Top_Temp"]:
            if key in f:
                raw_val = f[key][:]
                # Convert to Celsius or physical Kelvin scale
                extracted_vars["cloud_top_temperature"] = raw_val
                break
                
        # Check for Water Vapor Channel
        for key in ["IMG_WV", "WV", "TPW", "Precipitable_Water"]:
            if key in f:
                extracted_vars["water_vapor_intensity"] = f[key][:]
                break
                
        # Check for Quantitative Precipitation Estimate / HEM
        for key in ["HEM", "QPE", "Rainfall_Rate"]:
            if key in f:
                extracted_vars["rainfall_rate_estimate"] = f[key][:]
                break

    print(f"[✓] Extracted {len(extracted_vars)} variables from {os.path.basename(h5_file_path)}")
    return extracted_vars

def generate_insat_timeseries(
    start_time: str = "2023-07-08T00:00:00",
    num_timesteps: int = 120, # e.g. 120 timesteps of 30-min data (60 hours)
    freq_minutes: int = 30,
    lat_min: float = 28.0, lat_max: float = 32.0,
    lon_min: float = 76.0, lon_max: float = 80.0,
    resolution: float = 0.04, # ~4km INSAT resolution
    output_dir: str = "data/raw/insat"
) -> xr.Dataset:
    """
    Generates a full multi-timestep INSAT-3D/3DR Spatio-Temporal sequence.
    Produces 3-digit time series (e.g., 120+ sequential frames) capturing:
    - Diurnal temperature cycles
    - Storm cell initiation & rapid Cloud Top Cooling (CTT drop)
    - Integrated Water Vapor (IWV) pooling and surge
    - High-intensity convective precipitation
    """
    os.makedirs(output_dir, exist_ok=True)
    lats = np.arange(lat_min, lat_max, resolution)
    lons = np.arange(lon_min, lon_max, resolution)
    n_lat, n_lon = len(lats), len(lons)
    
    # Generate continuous timestamp array
    start_dt = np.datetime64(start_time)
    time_deltas = np.arange(num_timesteps) * np.timedelta64(freq_minutes, 'm')
    times = start_dt + time_deltas
    
    print(f"[*] Generating {num_timesteps} sequential INSAT satellite frames ({times[0]} to {times[-1]})...")
    
    # Base 3D arrays: (Time, Lat, Lon)
    ctt_series = np.zeros((num_timesteps, n_lat, n_lon), dtype=np.float32)
    iwv_series = np.zeros((num_timesteps, n_lat, n_lon), dtype=np.float32)
    qpe_series = np.zeros((num_timesteps, n_lat, n_lon), dtype=np.float32)
    
    # Simulate realistic meteorological dynamics across time
    Y, X = np.ogrid[:n_lat, :n_lon]
    center_y, center_x = n_lat // 2, n_lon // 2
    
    for t in range(num_timesteps):
        # Diurnal heating cycle (warmer afternoon, cooler night)
        diurnal_factor = np.sin(2 * np.pi * t / 48) # 24hr cycle in 30min steps
        
        # Base background values
        ctt = 278.0 + 6.0 * diurnal_factor + np.random.normal(0, 1.5, size=(n_lat, n_lon))
        iwv = 40.0 + 8.0 * np.cos(2 * np.pi * t / 48) + np.random.normal(0, 2.0, size=(n_lat, n_lon))
        qpe = np.maximum(0.0, np.random.normal(0.2, 0.5, size=(n_lat, n_lon)))
        
        # Simulate developing storm lifecycle (Precursor -> Explosion -> Decay)
        # Storm builds around step 30 to 70
        if 25 <= t <= 75:
            # Storm intensity curve (Gaussian wave over time)
            storm_intensity = np.exp(-((t - 45) ** 2) / (2 * (12 ** 2)))
            # Moving storm cell (steering eastward/northward with wind)
            drift_y = int((t - 45) * 0.4)
            drift_x = int((t - 45) * 0.6)
            sy = np.clip(center_y + drift_y, 0, n_lat - 1)
            sx = np.clip(center_x + drift_x, 0, n_lon - 1)
            
            dist_sq = (X - sx)**2 + (Y - sy)**2
            storm_spatial = np.exp(-dist_sq / (2 * (n_lat / 7.0)**2))
            
            # 1. Cloud Top Temperature drops rapidly (deep convective anvil)
            ctt -= storm_spatial * (65.0 * storm_intensity)
            # 2. IWV surges (moisture pooling)
            iwv += storm_spatial * (45.0 * storm_intensity)
            # 3. Heavy cloudburst rainfall (> 80 mm/hr peak)
            qpe += storm_spatial * (85.0 * storm_intensity)
            
        ctt_series[t] = ctt
        iwv_series[t] = iwv
        qpe_series[t] = qpe
        
    ds = xr.Dataset(
        data_vars={
            "cloud_top_temperature_kelvin": (["time", "latitude", "longitude"], ctt_series),
            "integrated_water_vapor_index": (["time", "latitude", "longitude"], iwv_series),
            "satellite_qpe_rainfall_rate": (["time", "latitude", "longitude"], qpe_series),
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
        attrs={
            "satellite": "INSAT-3DR Multi-Spectral Sequence",
            "temporal_resolution": f"{freq_minutes} minutes",
            "spatial_resolution": f"{resolution} degrees (~4km)",
            "num_timesteps": num_timesteps
        }
    )
    
    out_file = os.path.join(output_dir, f"insat_3dr_series_{num_timesteps}_timesteps.nc")
    ds.to_netcdf(out_file)
    print(f"[✓] Saved {num_timesteps}-step Satellite Time Series: {out_file}")
    print(f"    Dimensions: {dict(ds.sizes)}")
    return ds

def create_sample_insat_granule(
    timestamp: str = "2023-07-09T14:00:00",
    lat_min: float = 28.0, lat_max: float = 32.0,
    lon_min: float = 76.0, lon_max: float = 80.0,
    resolution: float = 0.04,
    simulated_storm: bool = True,
    output_dir: str = "data/raw/insat"
) -> xr.Dataset:
    """
    Single snapshot generator for backward compatibility.
    """
    return generate_insat_timeseries(
        start_time=timestamp,
        num_timesteps=1,
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max,
        resolution=resolution,
        output_dir=output_dir
    )

def compute_ctt_drop_rate(granule_t0: xr.Dataset, granule_t1: xr.Dataset, dt_minutes: float = 30.0) -> xr.DataArray:
    """
    Computes Cloud Top Temperature (CTT) drop rate in °C / hour.
    A rapid drop (> 20°C/hr or > 10°C/30min) signifies violent vertical updraft and thunderstorm intensification.
    """
    ctt0 = granule_t0["cloud_top_temperature_kelvin"].squeeze()
    ctt1 = granule_t1["cloud_top_temperature_kelvin"].squeeze()
    
    # Drop rate: (T_prev - T_curr) / (dt_hours)
    dt_hours = dt_minutes / 60.0
    ctt_drop_rate = (ctt0 - ctt1) / dt_hours
    return ctt_drop_rate

if __name__ == "__main__":
    # Test generating 120 timesteps (3-digit time series)
    generate_insat_timeseries(num_timesteps=120)

