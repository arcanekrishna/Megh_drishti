"""
Ground Truth Label Generator (SIH 2026)
---------------------------------------
Generates scientifically valid binary ground-truth labels for extreme weather events
based on established IMD and literature thresholds.
"""

import numpy as np

def generate_cloudburst_labels(precip_mmhr: np.ndarray, threshold: float = 50.0) -> np.ndarray:
    """
    Cloudburst label: Rainfall rate exceeding threshold.
    IMD standard is 100mm/hr, but satellite estimates (IMERG or ERA5) often underestimate over complex terrain.
    50mm/hr is an established proxy threshold for satellite/reanalysis data.
    Returns binary grid (1 for cloudburst, 0 otherwise).
    """
    return (precip_mmhr >= threshold).astype(np.float32)

def generate_thunderstorm_labels(cape_jkg: np.ndarray, wind_gust_ms: np.ndarray, cape_thresh: float = 2000.0, wind_thresh: float = 17.0) -> np.ndarray:
    """
    Severe Thunderstorm label: High convective instability (CAPE > 2000) AND 
    severe wind gusts (> 17 m/s, matching IMD severe threshold of 62 km/h).
    Returns binary grid (1 for severe thunderstorm, 0 otherwise).
    """
    return ((cape_jkg >= cape_thresh) & (wind_gust_ms >= wind_thresh)).astype(np.float32)

def generate_flash_flood_labels(precip_6h_accum: np.ndarray, slope_deg: np.ndarray, precip_thresh: float = 100.0, slope_thresh: float = 15.0) -> np.ndarray:
    """
    Flash Flood label: High accumulated rainfall over steep terrain.
    Proxy for flood runoff accumulation based on slope and prolonged intense rainfall.
    Returns binary grid (1 for flash flood risk, 0 otherwise).
    """
    return ((precip_6h_accum >= precip_thresh) & (slope_deg >= slope_thresh)).astype(np.float32)

if __name__ == "__main__":
    print("Label Generator ready.")
