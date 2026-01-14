"""
Sound-Based Engine Fault Detection System
==========================================
A Streamlit application for classifying engine health based on audio analysis.
Uses machine learning to detect engine faults from sound recordings.
"""

import streamlit as st
import wave
import os
import tempfile
import sys
from pathlib import Path

# Add directories to path to import backend module
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent.absolute()
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(parent_dir))

# Import backend module
try:
    # Try importing from streamlit directory first
    import importlib.util
    backend_path = current_dir / "engine_ml_backend.py"
    if backend_path.exists():
        spec = importlib.util.spec_from_file_location("engine_ml_backend", backend_path)
        engine_ml_backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine_ml_backend)
        EngineFaultDetector = engine_ml_backend.EngineFaultDetector
        predict_engine_health = engine_ml_backend.predict_engine_health
    else:
        raise ImportError("engine_ml_backend.py not found in streamlit directory")
except Exception as e:
    st.error(f"❌ Error: Could not import engine_ml_backend module: {str(e)}")
    st.info("💡 Please ensure engine_ml_backend.py is in the streamlit directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Engine Fault Detection System",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .healthy-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .unhealthy-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;cd "D:\project work\test\streamlit"

    }
    .confidence-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin-top: 1rem;
    }
    .stProgress > div > div > div {
        background-color: #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    try:
        # Try loading model from different possible locations
        model_paths = [
            "engine_rf_model.pkl",
            "../engine_rf_model.pkl",
            "../../engine_rf_model.pkl"
        ]
        scaler_paths = [
            "engine_scaler.pkl",
            "../engine_scaler.pkl",
            "../../engine_scaler.pkl"
        ]
        
        detector = None
        for mp, sp in zip(model_paths, scaler_paths):
            if os.path.exists(mp) and os.path.exists(sp):
                detector = EngineFaultDetector(mp, sp)
                break
        
        if detector is None:
            # Try default paths
            detector = EngineFaultDetector()
        
        st.session_state.detector = detector
        st.session_state.model_loaded = True
    except Exception as e:
        st.session_state.model_loaded = False
        st.session_state.model_error = str(e)

# Header
st.markdown('<h1 class="main-header">🔧 Engine Fault Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload an engine sound recording (.wav) to analyze engine health using AI-powered audio analysis</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **How it works:**
    1. Upload a .wav audio file of engine sound
    2. The system extracts advanced audio features
    3. ML model analyzes the features
    4. Get instant health diagnosis
    
    **Supported formats:**
    - WAV files only
    - Recommended: 48kHz sample rate
    
    **Detection capabilities:**
    - ✅ Healthy engine
    - ⚠️ Unhealthy engine (faults detected)
    """)
    
    st.markdown("---")
    st.header("🔬 Technical Details")
    st.markdown("""
    **Features extracted:**
    - MFCC coefficients (13)
    - Spectral features
    - Wavelet transforms
    - Bispectrum analysis
    - Amplitude envelope
    
    **Model:** Random Forest Classifier
    """)
    
    if st.session_state.get('model_loaded', False):
        st.success("✅ Model loaded successfully")
    else:
        st.error(f"❌ Model loading failed: {st.session_state.get('model_error', 'Unknown error')}")

# Main content area
if not st.session_state.get('model_loaded', False):
    st.error("⚠️ **Model not loaded. Please ensure `engine_rf_model.pkl` and `engine_scaler.pkl` are available.**")
    st.info("💡 Place the model files in the project root directory or update the paths in the code.")
    st.stop()

# File uploader
st.markdown("---")
st.subheader("📁 Upload Engine Sound Recording")

uploaded_file = st.file_uploader(
    "Choose a .wav file",
    type=["wav"],
    help="Upload a WAV audio file containing engine sound for analysis"
)

if uploaded_file is not None:
    # Display file info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📄 File Name", uploaded_file.name)
    
    file_size_mb = uploaded_file.size / (1024 * 1024)
    with col2:
        st.metric("💾 File Size", f"{file_size_mb:.2f} MB")
    
    with col3:
        st.metric("📊 File Type", uploaded_file.type)
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name
    
    try:
        # Read audio properties
        with wave.open(tmp_path, 'rb') as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            duration = n_frames / frame_rate
        
        # Display audio properties
        st.markdown("### 🎵 Audio Properties")
        prop_col1, prop_col2, prop_col3, prop_col4 = st.columns(4)
        
        with prop_col1:
            st.metric("🔊 Channels", channels)
        with prop_col2:
            st.metric("📈 Sample Rate", f"{frame_rate:,} Hz")
        with prop_col3:
            st.metric("⏱️ Duration", f"{duration:.2f} sec")
        with prop_col4:
            st.metric("📏 Sample Width", f"{sample_width} bytes")
        
        # Audio player
        st.markdown("### 🎧 Audio Playback")
        st.audio(uploaded_file, format='audio/wav')
        
        # Analysis button
        st.markdown("---")
        analyze_button = st.button("🔍 Analyze Engine Health", type="primary", use_container_width=True)
        
        if analyze_button:
            with st.spinner("🔄 Processing audio and extracting features..."):
                try:
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("📊 Loading audio file...")
                    progress_bar.progress(20)
                    
                    status_text.text("🧹 Applying noise reduction...")
                    progress_bar.progress(40)
                    
                    status_text.text("🔬 Extracting features (MFCC, Spectral, Wavelet, Bispectrum)...")
                    progress_bar.progress(60)
                    
                    status_text.text("🤖 Running ML model prediction...")
                    progress_bar.progress(80)
                    
                    # Make prediction
                    result = st.session_state.detector.predict(tmp_path)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")
                    
                    # Prediction result with styling
                    prediction = result['prediction']
                    confidence = result['confidence']
                    probabilities = result['probabilities']
                    
                    if prediction == "Healthy":
                        st.markdown(f"""
                        <div class="prediction-box healthy-box">
                            <h2 style="font-size: 2.5rem; margin: 0;">✅ HEALTHY ENGINE</h2>
                            <p style="font-size: 1.2rem; margin-top: 1rem;">Your engine appears to be operating normally.</p>
                            <div class="confidence-badge" style="background-color: rgba(255,255,255,0.3);">
                                Confidence: {confidence*100:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box unhealthy-box">
                            <h2 style="font-size: 2.5rem; margin: 0;">⚠️ UNHEALTHY ENGINE</h2>
                            <p style="font-size: 1.2rem; margin-top: 1rem;">Potential engine fault detected. Please consult a mechanic.</p>
                            <div class="confidence-badge" style="background-color: rgba(255,255,255,0.3);">
                                Confidence: {confidence*100:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Detailed probabilities
                    st.markdown("### 📈 Detailed Probabilities")
                    prob_col1, prob_col2 = st.columns(2)
                    
                    for label, prob in probabilities.items():
                        with prob_col1 if label == "Healthy" else prob_col2:
                            st.metric(
                                f"{'✅' if label == 'Healthy' else '⚠️'} {label}",
                                f"{prob*100:.2f}%"
                            )
                            st.progress(prob)
                    
                    # Recommendations
                    st.markdown("### 💡 Recommendations")
                    if prediction == "Healthy":
                        st.success("""
                        ✅ **Engine Status: Normal**
                        - Continue regular maintenance
                        - Monitor engine performance
                        - Schedule routine checkups as recommended
                        """)
                    else:
                        st.warning("""
                        ⚠️ **Engine Status: Fault Detected**
                        - **Immediate Action Recommended**
                        - Consult a qualified mechanic
                        - Avoid prolonged operation if possible
                        - Check for visible signs of damage
                        - Review maintenance history
                        """)
                    
                    # Technical details expander
                    with st.expander("🔬 View Technical Details"):
                        st.json({
                            "Prediction": prediction,
                            "Confidence": f"{confidence*100:.2f}%",
                            "Class Probabilities": probabilities,
                            "Audio Properties": {
                                "Sample Rate": f"{frame_rate} Hz",
                                "Duration": f"{duration:.2f} seconds",
                                "Channels": channels
                            }
                        })
                
                except Exception as e:
                    st.error(f"❌ **Error during analysis:** {str(e)}")
                    st.info("💡 Please ensure the audio file is valid and try again.")
                    st.exception(e)
        
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    except Exception as e:
        st.error(f"⚠️ **Error processing audio file:** {str(e)}")
        st.warning("Please ensure the uploaded file is a valid .wav audio file.")
        try:
            os.unlink(tmp_path)
        except:
            pass

else:
    # Welcome message when no file is uploaded
    st.info("👆 **Please upload a .wav audio file above to begin analysis.**")
    
    # Example use case
    st.markdown("---")
    st.markdown("### 📝 How to Use")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Step 1: Record Engine Sound**
        - Use a smartphone or recording device
        - Record in a quiet environment
        - Capture 5-30 seconds of engine sound
        - Save as .wav format
        
        **Step 2: Upload File**
        - Click "Browse files" above
        - Select your .wav file
        - Wait for file validation
        """)
    
    with col2:
        st.markdown("""
        **Step 3: Analyze**
        - Click "Analyze Engine Health" button
        - Wait for processing (10-30 seconds)
        - Review results and recommendations
        
        **Step 4: Take Action**
        - Follow recommendations
        - Consult mechanic if unhealthy
        - Keep records for tracking
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 2rem;'>"
    "🔧 Engine Fault Detection System | Powered by Machine Learning"
    "</div>",
    unsafe_allow_html=True
)
