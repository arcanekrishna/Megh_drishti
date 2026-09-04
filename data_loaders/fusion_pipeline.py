"""
Multi-Modal Spatiotemporal Data Fusion & Tensor Builder
-------------------------------------------------------
Aligns heterogeneous data sources into a unified Spatiotemporal Grid Tensor:
1. Dynamic Satellite Grid (INSAT-3D/3DR: CTT, IWV, QPE) - High Temporal Res (15-30 min)
2. Dynamic Atmospheric Grid (IMDAA/ERA5: CAPE, CIN, Wind Shear) - Hourly Res
3. Static Topography Grid (DEM: Elevation, Slope, Drainage) - Static Spatial Res

Outputs clean PyTorch/NumPy ready multi-channel tensors for AI Nowcasting.
"""

import os
import sys
import numpy as np
import xarray as xr
from typing import Tuple, Dict

# Ensure current module directory is in sys.path for direct script execution
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

class MultimodalWeatherFusionEngine:
    def __init__(self, target_resolution: float = 0.04):
        """
        target_resolution: Grid step in degrees (~0.04° is ~4km resolution)
        """
        self.resolution = target_resolution
        
    def align_and_fuse(
        self,
        satellite_ds: xr.Dataset,
        thermo_ds: xr.Dataset,
        dem_ds: xr.Dataset
    ) -> xr.Dataset:
        """
        Interpolates and harmonizes all multi-source datasets onto a unified spatial grid.
        """
        print("[*] Performing Spatio-temporal Grid Fusion...")
        
        # 1. Define Unified Reference Coordinate Grid from Satellite dataset
        ref_lat = satellite_ds.coords["latitude"].values
        ref_lon = satellite_ds.coords["longitude"].values
        ref_time = satellite_ds.coords["time"].values
        
        # 2. Resample / Interpolate Thermodynamics dataset onto reference grid
        thermo_interp = thermo_ds.interp(
            latitude=ref_lat,
            longitude=ref_lon,
            method="linear"
        )
        # Match time to nearest satellite observation
        thermo_interp = thermo_interp.reindex(time=ref_time, method="nearest")
        # Defensively fill any interpolation boundary NaNs cleanly without bottleneck dependency
        thermo_interp = thermo_interp.fillna(0.0)
        
        # 3. Resample / Interpolate DEM dataset onto reference grid
        dem_interp = dem_ds.interp(
            latitude=ref_lat,
            longitude=ref_lon,
            method="linear"
        )
        dem_interp = dem_interp.fillna(0.0)
        
        # 4. Merge all into unified xarray Dataset
        fused_ds = xr.Dataset()
        
        # Add Satellite Features
        fused_ds["sat_cloud_top_temp"] = satellite_ds["cloud_top_temperature_kelvin"]
        fused_ds["sat_water_vapor_iwv"] = satellite_ds["integrated_water_vapor_index"]
        fused_ds["sat_qpe_rainfall_rate"] = satellite_ds["satellite_qpe_rainfall_rate"]
        
        # Add Thermodynamic Features
        if "cape" in thermo_interp:
            fused_ds["thermo_cape"] = thermo_interp["cape"]
        if "convective_inhibition" in thermo_interp:
            fused_ds["thermo_cin"] = thermo_interp["convective_inhibition"]
        if "bulk_wind_shear_850_500" in thermo_interp:
            fused_ds["thermo_wind_shear"] = thermo_interp["bulk_wind_shear_850_500"]
        if "total_column_integrated_water_vapour" in thermo_interp:
            fused_ds["thermo_total_iwv"] = thermo_interp["total_column_integrated_water_vapour"]
            
        # Add Static DEM Features (Broadcast across time dimension)
        fused_ds["dem_elevation"] = dem_interp["elevation_meters"].expand_dims(time=ref_time)
        fused_ds["dem_slope"] = dem_interp["slope_degrees"].expand_dims(time=ref_time)
        fused_ds["dem_runoff_potential"] = dem_interp["runoff_channeling_index"].expand_dims(time=ref_time)
        
        print(f"[✓] Fusion Complete! Unified Feature Grid: {dict(fused_ds.sizes)}")
        print(f"    Total Feature Channels: {len(fused_ds.data_vars)}")
        return fused_ds

    def extract_model_tensors(self, fused_ds: xr.Dataset) -> Tuple[np.ndarray, list]:
        """
        Converts xarray Dataset into a standardized Normalized ML Tensor:
        Shape: (Batch/Time, Channels, Height/Lat, Width/Lon)
        """
        feature_names = list(fused_ds.data_vars.keys())
        arrays = []
        
        for var in feature_names:
            arr = fused_ds[var].values # Shape: (Time, Lat, Lon)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=0)
            # Handle NaNs / Missing values
            arr = np.nan_to_num(arr, nan=0.0)
            
            # Channel-wise Min-Max Normalization (0 to 1) for Deep Learning stability
            min_val, max_val = np.min(arr), np.max(arr)
            if max_val > min_val:
                arr_norm = (arr - min_val) / (max_val - min_val)
            else:
                arr_norm = np.zeros_like(arr)
                
            arrays.append(arr_norm)
            
        # Stack along Channel axis: Shape (Time, Channels, Lat, Lon)
        tensor = np.stack(arrays, axis=1)
        print(f"[✓] Built ML Tensor: Shape = {tensor.shape} (Time/Frames, Channels, Lat, Lon)")
        return tensor, feature_names

    def create_spatiotemporal_sliding_windows(
        self,
        tensor: np.ndarray,
        in_steps: int = 6,   # Past 3 hours (6 x 30min frames)
        out_steps: int = 6   # Future 3 to 6 hours lead-time (6 x 30min frames)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms a continuous multi-timestep tensor into supervised (Input_Sequence, Target_Sequence) pairs.
        
        Args:
            tensor: Shape (Total_Timesteps, Channels, Lat, Lon) e.g. (120, 9, 100, 100)
            in_steps: Number of past precursor frames fed into AI model (default 6 frames = 3 hrs)
            out_steps: Number of future nowcasting frames predicted (default 6 frames = 3 hrs lead time)
            
        Returns:
            X (Inputs):  Shape (Num_Samples, In_Steps, Channels, Lat, Lon)
            Y (Targets): Shape (Num_Samples, Out_Steps, Target_Channels, Lat, Lon)
        """
        total_time, n_channels, n_lat, n_lon = tensor.shape
        window_size = in_steps + out_steps
        
        if total_time < window_size:
            raise ValueError(f"Total time ({total_time}) must be >= in_steps + out_steps ({window_size})")
            
        num_samples = total_time - window_size + 1
        X = np.zeros((num_samples, in_steps, n_channels, n_lat, n_lon), dtype=np.float32)
        Y = np.zeros((num_samples, out_steps, n_channels, n_lat, n_lon), dtype=np.float32)
        
        for i in range(num_samples):
            X[i] = tensor[i : i + in_steps]
            Y[i] = tensor[i + in_steps : i + window_size]
            
        print(f"[✓] Created Supervised Spatio-Temporal Dataset:")
        print(f"    Total Samples: {num_samples} (3-digit dataset size)")
        print(f"    Input X (Precursors):  Shape {X.shape} -> (Batch, Past_Steps, Channels, H, W)")
        print(f"    Target Y (Nowcasting): Shape {Y.shape} -> (Batch, Future_Lead_Steps, Channels, H, W)")
        return X, Y


if __name__ == "__main__":
    from fetch_insat_satellite import create_sample_insat_granule
    from fetch_thermodynamics import fetch_grid_thermodynamics
    from fetch_dem_topography import generate_or_fetch_dem
    
    # 1. Load components
    sat_ds = create_sample_insat_granule("2023-07-09T14:30:00")
    thermo_ds = fetch_grid_thermodynamics(lat_min=28.0, lat_max=32.0, lon_min=76.0, lon_max=80.0)
    dem_ds = generate_or_fetch_dem(lat_min=28.0, lat_max=32.0, lon_min=76.0, lon_max=80.0)
    
    # 2. Fuse
    engine = MultimodalWeatherFusionEngine()
    fused_ds = engine.align_and_fuse(sat_ds, thermo_ds, dem_ds)
    
    # 3. Create Model Tensor
    tensor, feat_names = engine.extract_model_tensors(fused_ds)
    print("\nFeature Channels in Tensor:")
    for i, name in enumerate(feat_names):
        print(f"  Channel {i}: {name}")
