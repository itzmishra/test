# Model Files Location Report

## Currently Found Model Files

✅ **Found in:** `D:\project work\test\`
1. `engine_rf_model.pkl` - Single-stage Random Forest model
2. `engine_scaler.pkl` - Feature scaler for the model

## Required Model Files for Two-Stage Pipeline

❌ **Missing from:** `D:\project work\test\`
1. `vehicle_model.pkl` - Vehicle identification model (Stage 1)
2. `vehicle_scaler.pkl` - Scaler for vehicle model
3. `fault_model.pkl` - Fault classification model (Stage 2)
4. `fault_scaler.pkl` - Scaler for fault model

## Solution

To generate the required two-stage models, run:

```bash
cd "D:\project work\test"
python ml_pipeline.py MASTER.csv
```

This will train and save:
- `vehicle_model.pkl`
- `vehicle_scaler.pkl`
- `fault_model.pkl`
- `fault_scaler.pkl`

## Alternative: Use Single-Stage Model

If you prefer to use the existing `engine_rf_model.pkl`, you would need to:
1. Modify the app to use `EngineFaultDetector` instead of `TwoStageMLPipeline`
2. Or create a wrapper/adapter to use the existing model

## Search Paths

The app searches for models in these locations (in order):
1. `D:\project work\test\` (script directory)
2. Current working directory
3. Parent directory of script
