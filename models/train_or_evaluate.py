"""
AI Nowcasting Evaluation & Benchmark Suite (SIH 2026 V2)
--------------------------------------------------------
Computes quantitative meteorological metrics on real held-out test data.
- POD (Probability of Detection) = Hits / (Hits + Misses)
- FAR (False Alarm Ratio) = False Alarms / (Hits + False Alarms)
- CSI (Critical Success Index) = Hits / (Hits + Misses + False Alarms)
"""

import os
import sys
import time
import numpy as np
import torch
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.spatiotemporal_mtl import WeatherNowcastingInferenceEngine, MultiTaskWeatherNowcastingNet
from models.train import create_training_dataset

def compute_contingency_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes standard WMO/IMD operational verification metrics."""
    pred_binary = (y_pred >= threshold).astype(int)
    true_binary = (y_true >= threshold).astype(int)
    
    hits = int(np.sum((pred_binary == 1) & (true_binary == 1)))
    misses = int(np.sum((pred_binary == 0) & (true_binary == 1)))
    false_alarms = int(np.sum((pred_binary == 1) & (true_binary == 0)))
    correct_negatives = int(np.sum((pred_binary == 0) & (true_binary == 0)))
    
    pod = hits / (hits + misses + 1e-6)
    far = false_alarms / (hits + false_alarms + 1e-6)
    csi = hits / (hits + misses + false_alarms + 1e-6)
    
    return {
        "hits": hits, "misses": misses, "false_alarms": false_alarms,
        "pod": round(float(pod), 4),
        "far": round(float(far), 4),
        "csi": round(float(csi), 4)
    }

def evaluate_on_test_set(model_path: str = "models/checkpoints/model_best.pth", test_nc_path: str = None):
    print("=" * 70)
    print("🏆 SIH 2026: AI Weather Nowcasting Evaluation on Test Set")
    print("=" * 70)
    
    if test_nc_path is None or not os.path.exists(test_nc_path):
        print(f"[!] Warning: Test dataset '{test_nc_path}' not found. Cannot perform real evaluation.")
        print("    Running mock benchmark for architecture validation only (NOT OPERATIONAL PERFORMANCE).")
        return mock_benchmark()

    print(f"[*] Loading test data from: {test_nc_path}")
    X_test, Y_ts, Y_cb, Y_ff = create_training_dataset(test_nc_path)
    
    model = MultiTaskWeatherNowcastingNet(in_channels=10, in_steps=6, out_steps=6, hidden_dim=64)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"[✓] Loaded trained weights from {model_path}")
    else:
        print(f"[!] Warning: Trained weights not found at {model_path}. Evaluating untrained model.")
        
    model.eval()
    
    start_time = time.perf_counter()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        outputs = model(X_tensor)
        
        pred_ts = outputs["thunderstorm"].numpy()
        pred_cb = outputs["cloudburst"].numpy()
        pred_ff = outputs["flash_flood"].numpy()
        
    latency = time.perf_counter() - start_time
    ms_per_sample = (latency / len(X_test)) * 1000.0
    
    print(f"\n[+] AI Inference Latency: {ms_per_sample:.2f} ms per sample")
    
    # Global metrics
    print("\n[+] Global Verification Metrics:")
    ts_metrics = compute_contingency_metrics(Y_ts, pred_ts)
    cb_metrics = compute_contingency_metrics(Y_cb, pred_cb)
    ff_metrics = compute_contingency_metrics(Y_ff, pred_ff)
    
    print(f"    • Thunderstorm -> POD: {ts_metrics['pod']:.2f} | FAR: {ts_metrics['far']:.2f} | CSI: {ts_metrics['csi']:.2f}")
    print(f"    • Cloudburst   -> POD: {cb_metrics['pod']:.2f} | FAR: {cb_metrics['far']:.2f} | CSI: {cb_metrics['csi']:.2f}")
    print(f"    • Flash Flood  -> POD: {ff_metrics['pod']:.2f} | FAR: {ff_metrics['far']:.2f} | CSI: {ff_metrics['csi']:.2f}")

    # Per-lead-time evaluation
    print("\n[+] Per-Lead-Time Performance (POD / FAR / CSI):")
    for step in range(6):
        lead_time = step + 1
        ts_step = compute_contingency_metrics(Y_ts[:, step], pred_ts[:, step])
        cb_step = compute_contingency_metrics(Y_cb[:, step], pred_cb[:, step])
        print(f"    + {lead_time}h Lead | TS: {ts_step['pod']:.2f}/{ts_step['far']:.2f}/{ts_step['csi']:.2f} | CB: {cb_step['pod']:.2f}/{cb_step['far']:.2f}/{cb_step['csi']:.2f}")

def mock_benchmark():
    # Placeholder when real data isn't available
    engine = WeatherNowcastingInferenceEngine(in_steps=6, out_steps=6)
    np.random.seed(42)
    test_inputs = np.random.uniform(0.1, 0.9, size=(50, 6, 10, 50, 50)).astype(np.float32)
    
    start_time = time.perf_counter()
    for i in range(50):
        engine.predict(test_inputs[i:i+1])
    ai_latency = ((time.perf_counter() - start_time) / 50) * 1000.0
    
    print(f"\n[+] AI Inference Latency: {ai_latency:.2f} ms per frame")
    print("[!] Performance metrics require real test dataset. Use evaluate_on_test_set(test_nc_path).")

if __name__ == "__main__":
    test_path = "data/raw/historical/era5_monsoon_2023.nc" 
    evaluate_on_test_set(test_nc_path=test_path)
