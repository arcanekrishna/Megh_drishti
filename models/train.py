import os
import sys
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Fix imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.spatiotemporal_mtl import MultiTaskWeatherNowcastingNet
from data_loaders.fusion_pipeline import PhysicsNormalizer
from data_loaders.label_generator import (
    generate_cloudburst_labels,
    generate_thunderstorm_labels,
    generate_flash_flood_labels
)

def create_training_dataset(nc_path, in_steps=6, out_steps=6):
    ds = xr.open_dataset(nc_path)
    
    time_len = ds.sizes.get('time', ds.sizes.get('valid_time'))
    lat_len = ds.sizes['latitude']
    lon_len = ds.sizes['longitude']
    
    # Extract physical tensors before normalization for accurate labeling
    physical_tensor = np.zeros((time_len, 10, lat_len, lon_len), dtype=np.float32)
    # CTT / Cloud base height proxy
    physical_tensor[:, 0] = ds.get('cloud_cover', ds.get('cbh', ds.get('cloud_base_height', xr.zeros_like(ds['latitude'])))).values
    # IWV
    physical_tensor[:, 1] = ds.get('tcwv', ds.get('total_column_integrated_water_vapour', xr.zeros_like(ds['latitude']))).values
    # Precipitation
    physical_tensor[:, 2] = ds.get('tp', ds.get('precipitation', xr.zeros_like(ds['latitude']))).values
    # CAPE
    physical_tensor[:, 3] = ds.get('cape', xr.zeros_like(ds['latitude'])).values
    # CIN
    physical_tensor[:, 4] = ds.get('cin', ds.get('convective_inhibition', xr.zeros_like(ds['latitude']))).values
    
    # Wind shear proxy if not already computed
    if 'bulk_wind_shear' in ds:
        physical_tensor[:, 5] = ds['bulk_wind_shear'].values
    else:
        try:
            u_wind = ds.get('u10', ds.get('10m_u_component_of_wind')).values
            v_wind = ds.get('v10', ds.get('10m_v_component_of_wind')).values
            physical_tensor[:, 5] = np.abs(u_wind - v_wind)
        except Exception as e:
            pass
            
    physical_tensor[:, 6] = ds.get('t2m', ds.get('temperature_2m', xr.zeros_like(ds['latitude']))).values
    
    # Fill any NaNs from clear-sky variables (like cloud base height)
    physical_tensor = np.nan_to_num(physical_tensor, nan=0.0)
    
    # Normalization (Physics-based)
    normalizer = PhysicsNormalizer()
    tensor_norm = np.zeros_like(physical_tensor)
    var_names = ["cloud_top_temp", "water_vapor_iwv", "precipitation", "cape", "cin", "wind_shear", "temp", "elevation", "slope", "runoff"]
    
    for c, var_name in enumerate(var_names):
        tensor_norm[:, c] = normalizer.normalize(physical_tensor[:, c], var_name)

    num_samples = time_len - in_steps - out_steps + 1
    X = np.zeros((num_samples, in_steps, 10, lat_len, lon_len), dtype=np.float32)
    
    # Targets
    Y_ts = np.zeros((num_samples, out_steps, lat_len, lon_len), dtype=np.float32)
    Y_cb = np.zeros((num_samples, out_steps, lat_len, lon_len), dtype=np.float32)
    Y_ff = np.zeros((num_samples, out_steps, lat_len, lon_len), dtype=np.float32)

    for i in range(num_samples):
        # Input features are NORMALIZED
        X[i] = tensor_norm[i : i + in_steps]
        
        # Future 6 hours window (PHYSICAL VALUES)
        future_physical = physical_tensor[i + in_steps : i + in_steps + out_steps]
        
        precip = future_physical[:, 2, :, :]
        cape = future_physical[:, 3, :, :]
        wind_shear = future_physical[:, 5, :, :] # Using shear/wind proxy
        
        # Terrain
        slope = physical_tensor[0, 8, :, :] # Static over time
        
        # Generate Ground Truth Labels based on established thresholds
        Y_ts[i] = generate_thunderstorm_labels(cape, wind_shear)
        Y_cb[i] = generate_cloudburst_labels(precip)
        
        # For flash flood, use accumulated precip over the out_steps window (approx 6h)
        accum_precip = np.sum(precip, axis=0)
        ff_label = generate_flash_flood_labels(accum_precip, slope)
        # Broadcast across time steps
        Y_ff[i] = np.repeat(ff_label[np.newaxis, :, :], out_steps, axis=0)

    return X, Y_ts, Y_cb, Y_ff

def weighted_bce_loss(pred, target, pos_weight):
    """Custom BCE loss that supports pos_weight for probability outputs."""
    pred = torch.clamp(pred, 1e-7, 1.0 - 1e-7)
    loss = - (pos_weight * target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
    return loss.mean()

def train_model():
    print("=" * 60)
    print("🚀 Training Multi-Task Nowcasting Engine (SIH 2026 V2)")
    print("=" * 60)
    
    # Try multiple possible dataset names, preferring the full multi-year one
    nc_paths = [
        "data/raw/historical/era5_monsoon_2023.nc",
        "data/raw/historical/era5_training_2023-07-01_2023-07-31.nc",
        "data/raw/historical/openmeteo_training_2023-07-01_2023-07-31.nc"
    ]
    
    nc_path = None
    for p in nc_paths:
        if os.path.exists(p):
            nc_path = p
            break
            
    if nc_path is None:
        print(f"[!] No training dataset found.")
        return
        
    X, Y_ts, Y_cb, Y_ff = create_training_dataset(nc_path)
    
    # Temporal split: 80% Train, 20% Val (DO NOT SHUFFLE ACROSS TIME)
    split_idx = int(X.shape[0] * 0.8)
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_ts_train, Y_ts_val = Y_ts[:split_idx], Y_ts[split_idx:]
    Y_cb_train, Y_cb_val = Y_cb[:split_idx], Y_cb[split_idx:]
    Y_ff_train, Y_ff_val = Y_ff[:split_idx], Y_ff[split_idx:]

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    
    train_dataset = TensorDataset(
        torch.tensor(X_train), torch.tensor(Y_ts_train), 
        torch.tensor(Y_cb_train), torch.tensor(Y_ff_train)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val), torch.tensor(Y_ts_val), 
        torch.tensor(Y_cb_val), torch.tensor(Y_ff_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    model = MultiTaskWeatherNowcastingNet(in_channels=10, in_steps=6, out_steps=6, hidden_dim=64)
    
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # Class weights for rare events
    w_ts, w_cb, w_ff = 5.0, 10.0, 10.0
    
    epochs = 10
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    os.makedirs("models/checkpoints", exist_ok=True)
    save_path = "models/checkpoints/model_best.pth"
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_ts, batch_cb, batch_ff in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_X)
            
            loss_ts = weighted_bce_loss(outputs["thunderstorm"], batch_ts, w_ts)
            loss_cb = weighted_bce_loss(outputs["cloudburst"], batch_cb, w_cb)
            loss_ff = weighted_bce_loss(outputs["flash_flood"], batch_ff, w_ff)
            
            loss = loss_ts + 2.0 * loss_cb + 1.5 * loss_ff
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_ts, batch_cb, batch_ff in val_loader:
                outputs = model(batch_X)
                l_ts = weighted_bce_loss(outputs["thunderstorm"], batch_ts, w_ts)
                l_cb = weighted_bce_loss(outputs["cloudburst"], batch_cb, w_cb)
                l_ff = weighted_bce_loss(outputs["flash_flood"], batch_ff, w_ff)
                
                loss = l_ts + 2.0 * l_cb + 1.5 * l_ff
                val_loss += loss.item()
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"\n[✓] Training complete. Best weights saved to {save_path}")

if __name__ == "__main__":
    train_model()
