import streamlit as st

st.title("Hello, Fault Detection System!")

if st.button("Click to see a greeting"):
    
    st.write("Welcome to the Fault Detection System Interface!")
st.subheader("This is a simple Streamlit app for fault detection.")
st.success("Fault detection system is up and running!")

add_sound=st.checkbox("Show more info")

if add_sound:
    st.text("Here you can monitor and analyze fault detection data.")
    st.write("You can add more functionalities here to analyze and visualize fault detection data.")

sound_type = st.radio("Select the type of sound:", ("Healthy", "Unhealthy"))

st.write(f"You selected: {sound_type}")