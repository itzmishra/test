# How to Run Streamlit App

## Method 1: Using Virtual Environment (Recommended)

```powershell
cd "D:\project work\test"
.\.venv\Scripts\streamlit.exe run streamlit_app.py
```

## Method 2: Using Python Module

```powershell
cd "D:\project work\test"
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## Method 3: If Streamlit is in PATH

```powershell
cd "D:\project work\test"
streamlit run streamlit_app.py
```

## What to Expect

After running the command, you should see:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

## Troubleshooting

### If localhost doesn't open automatically:
1. **Copy the URL** from the terminal (usually `http://localhost:8501`)
2. **Paste it in your browser** (Chrome, Firefox, Edge, etc.)
3. Press Enter

### If port 8501 is already in use:
```powershell
# Use a different port
streamlit run streamlit_app.py --server.port 8502
```

### If you see "ModuleNotFoundError":
```powershell
# Install streamlit in your virtual environment
.\.venv\Scripts\python.exe -m pip install streamlit
```

### To stop the app:
- Press `Ctrl+C` in the terminal where Streamlit is running

## Quick Start

1. Open PowerShell in the project directory
2. Activate virtual environment: `.\.venv\Scripts\Activate.ps1`
3. Run: `streamlit run streamlit_app.py`
4. Open browser to: `http://localhost:8501`

## Features Available

✅ **Model loaded**: Random Forest (95.19% accuracy)
✅ **Per-class accuracy**:
   - Air_Leak: 95.83%
   - Healthy: 91.67%
   - MAP: 100.00%
   - Misfire: 91.67%
✅ **Upload audio files** (.wav or .mp3, max 1 MB)
✅ **Automatic fault detection**
✅ **Visualizations**: Spectrogram, MFCC, Bispectrum, Envelope
