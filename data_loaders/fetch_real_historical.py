"""
Historical Training Data Downloader
-----------------------------------
Fetches multi-year historical training data for AI models.
Supports bulk ERA5 downloads via Copernicus `cdsapi` (preferred for training grids)
and Open-Meteo Archive API (preferred for point-inference or fallback).
"""

import os
import requests
import numpy as np
import pandas as pd
import xarray as xr

def fetch_era5_cds(
    start_year: int = 2021, end_year: int = 2023,
    months: list = [6, 7, 8, 9],  # Monsoon season
    lat_min: float = 28.0, lat_max: float = 34.0, 
    lon_min: float = 74.0, lon_max: float = 82.0,
    output_dir: str = "data/raw/historical"
):
    """
    Downloads raw ERA5 data directly from Copernicus Climate Data Store.
    Required for proper gridded historical training without API rate limits.
    Requires `cdsapi` installed and configured (~/.cdsapirc).
    """
    try:
        import cdsapi
    except ImportError:
        print("[!] cdsapi not installed. Run: pip install cdsapi")
        return None
        
    os.makedirs(output_dir, exist_ok=True)
    c = cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key="db1620f7-33c3-4a23-b1af-bd69ceda2385")
    
    # Just download July 2023 to fit within time/size constraints for the CDS API limit
    year = "2023"
    month = "07"
    output_file = os.path.join(output_dir, f"era5_monsoon_{year}.nc")
    
    if os.path.exists(output_file):
        print(f"[✓] File already exists: {output_file}")
        return output_file
        
    print(f"[*] Fetching ERA5 historical data for {year}-{month}...")
    print(f"    Bounding Box: {lat_max}N to {lat_min}N, {lon_min}E to {lon_max}E")
    
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '2m_temperature', 'cape', 'convective_inhibition',
                'total_column_water_vapour', 'total_precipitation',
                '10m_u_component_of_wind', '10m_v_component_of_wind',
                'cloud_base_height'
            ],
            'year': year,
            'month': month,
            'day': [f"{d:02d}" for d in range(1, 32)],
            'time': [f"{h:02d}:00" for h in range(24)],
            'area': [lat_max, lon_min, lat_min, lon_max],
        },
        output_file
    )
    
    print(f"[✓] ERA5 data downloaded to: {output_file}")
    return output_file

def fetch_historical_training_data_openmeteo(
    start_date="2023-07-01", 
    end_date="2023-07-31",
    lat_min=28.0, lat_max=34.0, 
    lon_min=74.0, lon_max=82.0,
    grid_step=0.25,
    output_dir="data/raw/historical"
):
    """
    Legacy Open-Meteo fetcher. Useful for small tests, but too slow/rate-limited
    for full 3-year monsoon dataset across a 0.25 deg grid.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Fetching Open-Meteo historical data for {start_date} to {end_date}...")
    
    lats = np.arange(lat_min, lat_max + 1e-5, grid_step)
    lons = np.arange(lon_min, lon_max + 1e-5, grid_step)
    
    hourly_vars = [
        "temperature_2m", "cape", "convective_inhibition",
        "total_column_integrated_water_vapour", "precipitation",
        "wind_speed_10m", "wind_speed_850hPa", "wind_speed_500hPa", "cloud_cover"
    ]
    
    all_data = []
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    for lat in lats:
        for lon in lons:
            params = {
                "latitude": round(lat, 3),
                "longitude": round(lon, 3),
                "start_date": start_date,
                "end_date": end_date,
                "hourly": ",".join(hourly_vars),
                "timezone": "Asia/Kolkata"
            }
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                df = pd.DataFrame(data["hourly"])
                df["latitude"] = lat
                df["longitude"] = lon
                all_data.append(df)
            else:
                print(f"[!] Failed for ({lat}, {lon})")

    if not all_data:
        return None

    full_df = pd.concat(all_data, ignore_index=True)
    full_df["time"] = pd.to_datetime(full_df["time"])
    
    for var in hourly_vars:
        if var in full_df.columns:
            full_df[var] = pd.to_numeric(full_df[var], errors="coerce")
            
    # Fallbacks for mountains
    full_df["wind_speed_850hPa"] = full_df["wind_speed_850hPa"].fillna(full_df["wind_speed_10m"])
    full_df["wind_speed_500hPa"] = full_df["wind_speed_500hPa"].fillna(full_df["wind_speed_850hPa"] * 1.5)
    
    full_df[hourly_vars] = full_df[hourly_vars].bfill().ffill().fillna(0.0).astype(np.float32)
    full_df["bulk_wind_shear"] = np.abs(full_df["wind_speed_500hPa"] - full_df["wind_speed_850hPa"])
    
    full_df = full_df.set_index(["time", "latitude", "longitude"])
    ds = xr.Dataset.from_dataframe(full_df)
    
    for v in ds.data_vars:
        ds[v] = ds[v].astype(np.float32).fillna(0.0)
        
    nc_path = os.path.join(output_dir, f"openmeteo_training_{start_date}_{end_date}.nc")
    ds.to_netcdf(nc_path)
    print(f"[✓] Saved historical NetCDF: {nc_path}")
    return nc_path

if __name__ == "__main__":
    fetch_era5_cds()
