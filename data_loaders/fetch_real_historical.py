import os
import requests
import numpy as np
import pandas as pd
import xarray as xr

def fetch_historical_training_data(
    start_date="2023-07-01", 
    end_date="2023-07-31",
    lat_min=30.0, lat_max=32.0, 
    lon_min=77.0, lon_max=79.5,
    grid_step=0.5,
    output_dir="data/raw/historical"
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Fetching ERA5 historical data for {start_date} to {end_date}...")
    
    lats = np.arange(lat_min, lat_max + 1e-5, grid_step)
    lons = np.arange(lon_min, lon_max + 1e-5, grid_step)
    
    hourly_vars = [
        "temperature_2m",
        "cape",
        "convective_inhibition",
        "total_column_integrated_water_vapour",
        "precipitation",
        "wind_speed_10m",
        "wind_speed_850hPa",
        "wind_speed_500hPa",
        "cloud_cover"
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

    full_df = pd.concat(all_data, ignore_index=True)
    full_df["time"] = pd.to_datetime(full_df["time"])
    
    for var in hourly_vars:
        if var in full_df.columns:
            full_df[var] = pd.to_numeric(full_df[var], errors="coerce")
            
    # Fallbacks for mountains
    full_df["wind_speed_850hPa"] = full_df["wind_speed_850hPa"].fillna(full_df["wind_speed_10m"])
    full_df["wind_speed_500hPa"] = full_df["wind_speed_500hPa"].fillna(full_df["wind_speed_850hPa"] * 1.5)
    
    # Fill remaining NaNs
    full_df[hourly_vars] = full_df[hourly_vars].bfill().ffill().fillna(0.0).astype(np.float32)
    
    # Compute derived vars
    full_df["bulk_wind_shear"] = np.abs(full_df["wind_speed_500hPa"] - full_df["wind_speed_850hPa"])
    
    full_df = full_df.set_index(["time", "latitude", "longitude"])
    ds = xr.Dataset.from_dataframe(full_df)
    
    for v in ds.data_vars:
        ds[v] = ds[v].astype(np.float32).fillna(0.0)
        
    nc_path = os.path.join(output_dir, f"era5_training_{start_date}_{end_date}.nc")
    ds.to_netcdf(nc_path)
    print(f"[✓] Saved historical NetCDF: {nc_path}")
    print(ds)
    return nc_path

if __name__ == "__main__":
    fetch_historical_training_data()
