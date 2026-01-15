# Training Models for Engine Fault Detection

## Quick Start

To train the models required by the Streamlit app, run:

```bash
python ml_pipeline.py MASTER.csv
```

This will:
1. Load data from `MASTER.csv`
2. Extract features from audio files
3. Train two-stage ML pipeline:
   - **Stage 1**: Vehicle identification (Ford EcoSport, Ford Figo, Other/Unknown)
   - **Stage 2**: Fault classification (Healthy, Misfire, Air Intake Irregularity)
4. Save models as:
   - `vehicle_model.pkl`
   - `vehicle_scaler.pkl`
   - `fault_model.pkl`
   - `fault_scaler.pkl`

## Prerequisites

- Python 3.7+
- Required packages (install via `pip install -r requirements.txt`)
- `MASTER.csv` file with training data

## Verification

After training, verify models exist:
```bash
ls -la *.pkl
```

You should see all four model files listed above.
