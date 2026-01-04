"""
Enhanced Streamlit Web Application for Engine Fault Detection
==============================================================
Features:
- MP3 and WAV file upload support (max 1 MB)
- Two-stage ML pipeline (Vehicle + Fault classification)
- Comprehensive visualizations (MFCC, Spectrogram, Bispectrum, Envelope)
- Confidence scores and warnings
- Production-ready error handling
"""

import streamlit as st
import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

try:
    from feature_extraction import OptimizedFeatureExtractor
    from ml_pipeline import TwoStageMLPipeline
    from visualizations import create_all_visualizations, figure_to_base64
    import librosa
except ImportError as e:
    st.error(f"❌ Import Error: {str(e)}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Engine Fault Detection System",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .fault-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .warning-box {
        background: linear-gradient(135deg, #ffa726 0%, #fb8c00 100%);
        color: white;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    try:
        pipeline = TwoStageMLPipeline()
        pipeline.load_models()
        st.session_state.pipeline = pipeline
        st.session_state.models_loaded = True
    except Exception as e:
        st.session_state.models_loaded = False
        st.session_state.model_error = str(e)

# Header
st.markdown('<h1 class="main-header">🔧 Engine Fault Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced ML-based engine health diagnosis using audio signal analysis</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **How it works:**
    1. Upload an audio file (.wav or .mp3, max 1 MB)
    2. System extracts audio features using signal processing
    3. Two-stage ML pipeline analyzes the signal:
       - **Stage 1:** Vehicle identification
       - **Stage 2:** Fault detection (vehicle-agnostic)
    4. Get detailed diagnosis with visualizations
    
    **Supported formats:**
    - WAV files
    - MP3 files
    - Maximum size: 1 MB
    - Recommended: 48kHz sample rate
    
    **Detection capabilities:**
    - ✅ Healthy engine
    - ⚠️ Misfire detection
    - ⚠️ Air intake irregularity
    - 🚗 Vehicle type identification
    """)
    
    st.markdown("---")
    st.header("🔬 Technical Details")
    st.markdown("""
    **Features extracted:**
    - MFCC coefficients (13)
    - Spectral features
    - Wavelet transforms (DWT)
    - Bispectrum analysis
    - Amplitude envelope
    
    **ML Models:**
    - Stage 1: Random Forest (Vehicle)
    - Stage 2: Random Forest (Fault)
    
    **Performance:**
    - Time Complexity: O(n)
    - Space Complexity: O(n)
    """)
    
    if st.session_state.get('models_loaded', False):
        st.success("✅ Models loaded successfully")
    else:
        st.error(f"❌ Model loading failed: {st.session_state.get('model_error', 'Unknown error')}")
        st.info("💡 Ensure model files are in the project directory:\n"
                "- vehicle_model.pkl\n"
                "- vehicle_scaler.pkl\n"
                "- fault_model.pkl\n"
                "- fault_scaler.pkl")

# Main content
if not st.session_state.get('models_loaded', False):
    st.error("⚠️ **Models not loaded. Please ensure model files are available.**")
    st.info("💡 Train models using: `python ml_pipeline.py MASTER.csv`")
    st.stop()

# File uploader
st.markdown("---")
st.subheader("📁 Upload Engine Sound Recording")

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3"],
    help="Upload a WAV or MP3 audio file containing engine sound for analysis (max 1 MB)"
)

if uploaded_file is not None:
    # File validation
    file_size_mb = uploaded_file.size / (1024 * 1024)
    
    if file_size_mb > 1.0:
        st.error(f"❌ File size ({file_size_mb:.2f} MB) exceeds maximum allowed size (1 MB)")
        st.stop()
    
    # Display file info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 File Name", uploaded_file.name)
    with col2:
        st.metric("💾 File Size", f"{file_size_mb:.2f} MB")
    with col3:
        st.metric("📊 File Type", uploaded_file.type)
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name
    
    try:
        # Load audio to get properties
        y, sr = librosa.load(tmp_path, sr=None, mono=True)
        duration = len(y) / sr if sr > 0 else 0
        
        # Display audio properties
        st.markdown("### 🎵 Audio Properties")
        prop_col1, prop_col2, prop_col3, prop_col4 = st.columns(4)
        with prop_col1:
            st.metric("🔊 Sample Rate", f"{sr:,} Hz")
        with prop_col2:
            st.metric("⏱️ Duration", f"{duration:.2f} sec")
        with prop_col3:
            st.metric("📏 Samples", f"{len(y):,}")
        with prop_col4:
            st.metric("📊 Channels", "Mono" if y.ndim == 1 else f"{y.shape[1]}")
        
        # Audio player
        st.markdown("### 🎧 Audio Playback")
        st.audio(uploaded_file, format=uploaded_file.type)
        
        # Analysis button
        st.markdown("---")
        analyze_button = st.button("🔍 Analyze Engine Health", type="primary", use_container_width=True)
        
        if analyze_button:
            with st.spinner("🔄 Processing audio and extracting features..."):
                try:
                    # Progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("📊 Loading and preprocessing audio...")
                    progress_bar.progress(20)
                    
                    status_text.text("🧹 Applying noise reduction (wavelet denoising)...")
                    progress_bar.progress(40)
                    
                    status_text.text("🔬 Extracting features (MFCC, Spectral, Wavelet, Bispectrum)...")
                    progress_bar.progress(60)
                    
                    # Make prediction
                    pipeline = st.session_state.pipeline
                    result = pipeline.predict(tmp_path, sr=sr)
                    
                    status_text.text("📈 Generating visualizations...")
                    progress_bar.progress(80)
                    
                    # Generate visualizations
                    feature_data = result['feature_data']
                    visualizations = create_all_visualizations(feature_data, sr=int(sr))
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    # Clear progress
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")
                    
                    # Vehicle prediction
                    vehicle_result = result['vehicle']
                    fault_result = result['fault']
                    
                    # Vehicle identification
                    st.markdown("### 🚗 Vehicle Identification")
                    vehicle_col1, vehicle_col2 = st.columns([2, 1])
                    
                    with vehicle_col1:
                        vehicle_icon = "🚗" if vehicle_result['prediction'] != "Other/Unknown" else "❓"
                        st.markdown(f"**{vehicle_icon} Detected Vehicle:** {vehicle_result['prediction']}")
                        st.progress(vehicle_result['confidence'])
                        st.caption(f"Confidence: {vehicle_result['confidence']*100:.1f}%")
                    
                    with vehicle_col2:
                        st.metric("Confidence", f"{vehicle_result['confidence']*100:.1f}%")
                    
                    if vehicle_result.get('warning'):
                        st.warning(vehicle_result['warning'])
                    
                    # Fault classification
                    st.markdown("### ⚙️ Engine Fault Classification")
                    
                    fault_prediction = fault_result['prediction']
                    fault_confidence = fault_result['confidence']
                    
                    if fault_prediction == "Healthy":
                        st.markdown(f"""
                        <div class="prediction-box healthy-box">
                            <h2 style="font-size: 2.5rem; margin: 0;">✅ HEALTHY ENGINE</h2>
                            <p style="font-size: 1.2rem; margin-top: 1rem;">Your engine appears to be operating normally.</p>
                            <div class="confidence-badge" style="background-color: rgba(255,255,255,0.3);">
                                Confidence: {fault_confidence*100:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box fault-box">
                            <h2 style="font-size: 2.5rem; margin: 0;">⚠️ FAULT DETECTED: {fault_prediction.upper()}</h2>
                            <p style="font-size: 1.2rem; margin-top: 1rem;">Potential engine issue detected. Please consult a mechanic.</p>
                            <div class="confidence-badge" style="background-color: rgba(255,255,255,0.3);">
                                Confidence: {fault_confidence*100:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if fault_result.get('warning'):
                        st.warning(fault_result['warning'])
                    
                    # Detailed probabilities
                    st.markdown("### 📈 Detailed Probabilities")
                    
                    # Vehicle probabilities
                    st.markdown("**Vehicle Probabilities:**")
                    vehicle_prob_cols = st.columns(len(vehicle_result['probabilities']))
                    for i, (label, prob) in enumerate(vehicle_result['probabilities'].items()):
                        with vehicle_prob_cols[i]:
                            st.metric(label, f"{prob*100:.2f}%")
                            st.progress(prob)
                    
                    # Fault probabilities
                    st.markdown("**Fault Probabilities:**")
                    fault_prob_cols = st.columns(len(fault_result['probabilities']))
                    for i, (label, prob) in enumerate(fault_result['probabilities'].items()):
                        with fault_prob_cols[i]:
                            icon = "✅" if label == "Healthy" else "⚠️"
                            st.metric(f"{icon} {label}", f"{prob*100:.2f}%")
                            st.progress(prob)
                    
                    # Visualizations
                    st.markdown("---")
                    st.markdown("## 📊 Signal Analysis Visualizations")
                    
                    # MFCC
                    if 'mfcc' in visualizations:
                        st.markdown("### 🎵 MFCC (Mel-Frequency Cepstral Coefficients)")
                        st.pyplot(visualizations['mfcc'])
                    
                    # Spectrogram
                    if 'spectrogram' in visualizations:
                        st.markdown("### 📈 Spectrogram")
                        st.pyplot(visualizations['spectrogram'])
                    
                    # Bispectrum
                    if 'bispectrum' in visualizations:
                        st.markdown("### 🔄 Bispectrum Analysis")
                        st.markdown("Bispectrum reveals non-linear frequency interactions in the signal.")
                        st.pyplot(visualizations['bispectrum'])
                    
                    # Amplitude Envelope
                    if 'envelope' in visualizations:
                        st.markdown("### 📊 Amplitude Envelope")
                        st.markdown("RMS and Hilbert envelopes show signal amplitude variations over time.")
                        st.pyplot(visualizations['envelope'])
                    
                    # Recommendations
                    st.markdown("---")
                    st.markdown("### 💡 Recommendations")
                    
                    if fault_prediction == "Healthy":
                        st.success("""
                        ✅ **Engine Status: Normal**
                        - Continue regular maintenance schedule
                        - Monitor engine performance periodically
                        - Schedule routine checkups as recommended by manufacturer
                        - Keep engine clean and well-maintained
                        """)
                    else:
                        st.error(f"""
                        ⚠️ **Engine Status: {fault_prediction} Detected**
                        - **Immediate Action Recommended**
                        - Consult a qualified mechanic for inspection
                        - Avoid prolonged operation if possible
                        - Check for visible signs of damage or unusual behavior
                        - Review maintenance history and service records
                        - Consider diagnostic scan tools for detailed analysis
                        """)
                    
                    # Technical details expander
                    with st.expander("🔬 View Technical Details"):
                        st.json({
                            "Vehicle Prediction": vehicle_result['prediction'],
                            "Vehicle Confidence": f"{vehicle_result['confidence']*100:.2f}%",
                            "Fault Prediction": fault_result['prediction'],
                            "Fault Confidence": f"{fault_confidence*100:.2f}%",
                            "Vehicle Probabilities": vehicle_result['probabilities'],
                            "Fault Probabilities": fault_result['probabilities'],
                            "Audio Properties": {
                                "Sample Rate": f"{sr} Hz",
                                "Duration": f"{duration:.2f} seconds",
                                "Samples": len(y)
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
        st.warning("Please ensure the uploaded file is a valid audio file.")
        try:
            os.unlink(tmp_path)
        except:
            pass

else:
    # Welcome message
    st.info("👆 **Please upload an audio file (.wav or .mp3, max 1 MB) above to begin analysis.**")
    
    st.markdown("---")
    st.markdown("### 📝 How to Use")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Step 1: Record Engine Sound**
        - Use a smartphone or recording device
        - Record in a quiet environment
        - Capture 5-30 seconds of engine sound
        - Save as .wav or .mp3 format
        
        **Step 2: Upload File**
        - Click "Browse files" above
        - Select your audio file
        - Ensure file size is under 1 MB
        """)
    
    with col2:
        st.markdown("""
        **Step 3: Analyze**
        - Click "Analyze Engine Health" button
        - Wait for processing (typically 5-15 seconds)
        - Review results and visualizations
        
        **Step 4: Take Action**
        - Follow recommendations provided
        - Consult mechanic if faults detected
        - Keep records for tracking over time
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 2rem;'>"
    "🔧 Engine Fault Detection System | Powered by Machine Learning & Signal Processing"
    "</div>",
    unsafe_allow_html=True
)

