import streamlit as st
import wave
import os
from datetime import datetime

st.set_page_config(
    page_title="Engine Fault Detection System",
    page_icon="🔧",
    layout="centered"
)

st.title("🔧 Engine Fault Detection System")

st.markdown("---")

st.subheader("Select Engine Sound Condition")

condition = st.radio(
    "Choose the condition of the engine:",
    options=["Healthy", "Unhealthy"],
    help="Select whether the engine sound is healthy or unhealthy"
)

fault_type = None
if condition == "Unhealthy":
    fault_type = st.selectbox(
        "Select Fault Type:",
        options=["Misfire", "Air Abnormalities", "Other Faults"],
        help="Choose the specific type of fault"
    )

st.markdown("---")

st.subheader("Upload Audio File")

uploaded_file = st.file_uploader(
    "Upload a .wav audio file",
    type=["wav"],
    help="Only .wav format is supported"
)

if uploaded_file is not None:
    st.success(f"✅ File uploaded successfully: **{uploaded_file.name}**")

    st.markdown("### File Details")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("File Name", uploaded_file.name)
        file_size_kb = uploaded_file.size / 1024
        st.metric("File Size", f"{file_size_kb:.2f} KB")

    with col2:
        st.metric("File Type", uploaded_file.type)
        st.metric("Condition", condition)
        if fault_type:
            st.metric("Fault Type", fault_type)

    try:
        temp_file_path = f"/tmp/{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with wave.open(temp_file_path, 'rb') as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            duration = n_frames / frame_rate

            st.markdown("### Audio Properties")

            prop_col1, prop_col2, prop_col3 = st.columns(3)

            with prop_col1:
                st.metric("Channels", channels)
                st.metric("Sample Rate", f"{frame_rate} Hz")

            with prop_col2:
                st.metric("Sample Width", f"{sample_width} bytes")
                st.metric("Duration", f"{duration:.2f} sec")

            with prop_col3:
                st.metric("Total Frames", f"{n_frames:,}")

        os.remove(temp_file_path)

        st.audio(uploaded_file, format='audio/wav')

        st.info("ℹ️ Audio file processed successfully. All details extracted.")

    except Exception as e:
        st.error(f"⚠️ Error processing audio file: {str(e)}")
        st.warning("Please ensure the uploaded file is a valid .wav audio file.")
else:
    st.info("📁 Please upload a .wav audio file to view details.")

st.markdown("---")

with st.expander("ℹ️ About This Application"):
    st.write("""
    **Engine Fault Detection System**

    This application allows you to:
    - Select the engine sound condition (Healthy or Unhealthy)
    - Specify the type of fault if unhealthy
    - Upload .wav audio files for analysis
    - View detailed information about the uploaded audio file

    **Supported Fault Types:**
    - Misfire
    - Air Abnormalities
    - Other Faults

    **Note:** This is a frontend-only demonstration. No ML inference or backend processing is performed.
    """)
