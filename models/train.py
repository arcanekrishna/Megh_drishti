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

def create_training_dataset(nc_path, in_steps=6, out_steps=6):
    ds = xr.open_dataset(nc_path)
    
    # 10 Channels matching the engine:
    # 0: ctt (using cloud_cover as proxy)
    # 1: iwv
    # 2: qpe (precipitation)
    # 3: cape
    # 4: cin
    # 5: wind_shear
    # 6: temp
    # 7: dem_elevation (dummy for now)
    # 8: dem_slope (dummy for now)
    # 9: dem_runoff (dummy for now)
    
    time_len = ds.sizes['time']
    lat_len = ds.sizes['latitude']
    lon_len = ds.sizes['longitude']
    
    tensor = np.zeros((time_len, 10, lat_len, lon_len), dtype=np.float32)
    tensor[:, 0] = ds['cloud_cover'].values
    tensor[:, 1] = ds['total_column_integrated_water_vapour'].values
    tensor[:, 2] = ds['precipitation'].values
    tensor[:, 3] = ds['cape'].values
    tensor[:, 4] = ds['convective_inhibition'].values
    tensor[:, 5] = ds['bulk_wind_shear'].values
    tensor[:, 6] = ds['temperature_2m'].values
    
    # Normalization (Min-Max)
    for c in range(10):
        c_max = np.max(tensor[:, c])
        c_min = np.min(tensor[:, c])
        if c_max > c_min:
            tensor[:, c] = (tensor[:, c] - c_min) / (c_max - c_min)

    num_samples = time_len - in_steps - out_steps + 1
    X = np.zeros((num_samples, in_steps, 10, lat_len, lon_len), dtype=np.float32)
    
    # Targets
    Y_ts = np.zeros((num_samples, out_steps, lat_len, lon_len), dtype=np.float32)
    Y_cb = np.zeros((num_samples, out_steps, lat_len, lon_len), dtype=np.float32)
    Y_ff = np.zeros((num_samples, out_steps, lat_len, lon_len), dtype=np.float32)

    for i in range(num_samples):
        X[i] = tensor[i : i + in_steps]
        
        # Future 6 hours window
        future_window = tensor[i + in_steps : i + in_steps + out_steps]
        
        # Derive Ground Truth Labels from future precipitation and CAPE
        # Thresholds are arbitrary for demo training purposes
        precip = future_window[:, 2, :, :]
        cape = future_window[:, 3, :, :]
        
        # Thunderstorm: High CAPE + Some Rain
        Y_ts[i] = ((cape > 0.6) & (precip > 0.1)).astype(np.float32)
        # Cloudburst: Extreme Rain
        Y_cb[i] = (precip > 0.8).astype(np.float32)
        # Flash Flood: Prolonged/Extreme Rain
        Y_ff[i] = (precip > 0.6).astype(np.float32)

    return X, Y_ts, Y_cb, Y_ff

def train_model():
    print("=" * 60)
    print("🚀 Training Multi-Task Nowcasting Engine on Real ERA5 Data")
    print("=" * 60)
    
    nc_path = "data/raw/historical/era5_training_2023-07-01_2023-07-31.nc"
    if not os.path.exists(nc_path):
        print(f"Dataset not found at {nc_path}")
        return
        
    X, Y_ts, Y_cb, Y_ff = create_training_dataset(nc_path)
    print(f"Training samples generated: {X.shape[0]}")
    
    # To PyTorch Tensors
    dataset = TensorDataset(
        torch.tensor(X), 
        torch.tensor(Y_ts), 
        torch.tensor(Y_cb), 
        torch.tensor(Y_ff)
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = MultiTaskWeatherNowcastingNet(in_channels=10, in_steps=6, out_steps=6)
    
    # Focal Loss approximation using BCE with pos_weight to handle class imbalance (rare events)
    criterion_ts = nn.BCELoss()
    criterion_cb = nn.BCELoss()
    criterion_ff = nn.BCELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_ts, batch_cb, batch_ff in dataloader:
            optimizer.zero_grad()
            
            outputs = model(batch_X)
            
            loss_ts = criterion_ts(outputs["thunderstorm"], batch_ts)
            loss_cb = criterion_cb(outputs["cloudburst"], batch_cb)
            loss_ff = criterion_ff(outputs["flash_flood"], batch_ff)
            
            # Weighted multi-task loss
            loss = loss_ts + 2.0 * loss_cb + 1.5 * loss_ff
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Total Loss: {epoch_loss/len(dataloader):.4f}")

    os.makedirs("models/checkpoints", exist_ok=True)
    save_path = "models/checkpoints/model_best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n[✓] Training complete. Weights saved to {save_path}")

if __name__ == "__main__":
    train_model()
