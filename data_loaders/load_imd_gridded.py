"""
IMD Gridded Data Loader (Historical Baseline)
--------------------------------------------
Fetches official India Meteorological Department (IMD) gridded binary data:
- Rainfall (0.25 x 0.25 degree resolution)
- Maximum & Minimum Temperature (0.5 x 0.5 degree resolution)
Converts data into xarray Dataset and NetCDF format for AI model ingestion.
"""

import os
import sys
import numpy as np
import xarray as xr

def fetch_imd_rainfall(start_year: int, end_year: int, output_dir: str = "data/raw/imd_rain"):
    """
    Downloads and converts IMD gridded rainfall data.
    """
    try:
        import imdlib as imd
    except ImportError:
        print("[!] imdlib is not installed. Run: pip install imdlib")
        return None

    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Downloading IMD Rainfall data for {start_year} to {end_year}...")
    
    # Download daily gridded rainfall (0.25° x 0.25°)
    rain_data = imd.get_data('rain', start_year, end_year, fn_format='yearwise', file_dir=output_dir)
    
    # Convert to xarray Dataset
    ds = rain_data.get_xarray()
    
    # Clean and save to NetCDF
    nc_path = os.path.join(output_dir, f"imd_rainfall_{start_year}_{end_year}.nc")
    ds.to_netcdf(nc_path)
    print(f"[✓] IMD Rainfall data saved to: {nc_path}")
    print(f"    Grid dimensions: {ds.dims}, Variables: {list(ds.data_vars.keys())}")
    return ds

def fetch_imd_temperature(start_year: int, end_year: int, var_type: str = "tmax", output_dir: str = "data/raw/imd_temp"):
    """
    Downloads and converts IMD gridded temperature data (tmax or tmin).
    """
    try:
        import imdlib as imd
    except ImportError:
        print("[!] imdlib is not installed. Run: pip install imdlib")
        return None

    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Downloading IMD {var_type.upper()} data for {start_year} to {end_year}...")
    
    temp_data = imd.get_data(var_type, start_year, end_year, fn_format='yearwise', file_dir=output_dir)
    ds = temp_data.get_xarray()
    
    nc_path = os.path.join(output_dir, f"imd_{var_type}_{start_year}_{end_year}.nc")
    ds.to_netcdf(nc_path)
    print(f"[✓] IMD {var_type.upper()} data saved to: {nc_path}")
    return ds

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download IMD Gridded Data")
    parser.add_argument("--start_year", type=int, default=2023, help="Start year")
    parser.add_argument("--end_year", type=int, default=2023, help="End year")
    parser.add_argument("--var", type=str, choices=["rain", "tmax", "tmin", "all"], default="rain")
    args = parser.parse_args()

    if args.var in ["rain", "all"]:
        fetch_imd_rainfall(args.start_year, args.end_year)
    if args.var in ["tmax", "all"]:
        fetch_imd_temperature(args.start_year, args.end_year, "tmax")
    if args.var in ["tmin", "all"]:
        fetch_imd_temperature(args.start_year, args.end_year, "tmin")
