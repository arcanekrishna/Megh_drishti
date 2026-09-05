#!/bin/bash
source .venv/bin/activate
echo "======================================"
echo "1. FETCHING COPERNICUS ERA5 DATA"
echo "======================================"
python data_loaders/fetch_real_historical.py

echo ""
echo "======================================"
echo "2. TRAINING V2 MODEL"
echo "======================================"
python models/train.py

echo ""
echo "======================================"
echo "3. GENERATING BENCHMARK EVALUATION"
echo "======================================"
python models/train_or_evaluate.py

