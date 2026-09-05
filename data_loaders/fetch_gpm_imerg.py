"""
GPM IMERG Precipitation Downloader (Label Source)
-------------------------------------------------
Fetches precipitation data from NASA GES DISC for ground-truth cloudburst labels.
Requires NASA Earthdata account credentials.
"""

import os
import sys

def fetch_gpm_imerg_data(
    start_date: str = "2023-07-01",
    end_date: str = "2023-07-31",
    lat_min: float = 28.0, lat_max: float = 34.0,
    lon_min: float = 74.0, lon_max: float = 82.0,
    output_dir: str = "data/raw/gpm_imerg"
):
    """
    Placeholder for NASA GES DISC GPM IMERG download.
    In production, this would use Earthdata login to fetch NetCDF files.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] GPM IMERG 30-min data fetch requested for {start_date} to {end_date}")
    print("[!] NASA Earthdata authentication required for real download.")
    print("[!] Please set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables.")
    # Actual download logic would go here using pydap or directly downloading URLs
    # generated via GES DISC subsetter.
    return None

if __name__ == "__main__":
    fetch_gpm_imerg_data()
