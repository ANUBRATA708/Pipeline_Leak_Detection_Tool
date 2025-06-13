import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, welch
from tensorflow.keras.models import load_model
from leak_detector import apply_filter, extract_features

# Page setup
st.set_page_config(page_title="🚰 Water Leak Detection App", layout="wide")

# Custom CSS styles
st.markdown("""
    <style>
    body {
        background-color: #f0f4f7;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        color: #2a7f62;
    }
    .stButton>button {
        background-color: #2a7f62;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1.5rem;
    }
    .stButton>button:hover {
        background-color: #145c4b;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚰 AI-Powered Water Leak Detection")
st.markdown("Detect pipeline leaks using deep learning and signal processing. Upload your vibration signal to begin.")

# Load model
fs = 1000
model = load_model("models/leak_model.h5")

uploaded = st.file_uploader("📤 Upload Signal File (.csv)", type=["csv"])

def plot_signal_analysis(raw_signal, filtered_signal, fs):
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={'hspace': 0.5})
    t = np.linspace(0, 1, len(raw_signal))

    axs[0].plot(t, raw_signal)
    axs[0].set_title("🔹 Raw Signal")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)

    axs[1].plot(t, filtered_signal, color='orange')
    axs[1].set_title("🔸 Bandpass Filtered Signal (20–300 Hz)")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(True)

    freqs, psd = welch(filtered_signal, fs=fs)
    axs[2].semilogy(freqs, psd, color='green')
    axs[2].set_title("🔻 Frequency Spectrum")
    axs[2].set_xlabel("Frequency [Hz]")
    axs[2].set_ylabel("Power Spectral Density")
    axs[2].grid(True)

    plt.tight_layout()  # ensures no overlap
    st.pyplot(fig)


if uploaded:
    st.success("✅ File uploaded successfully!")
    signal = pd.read_csv(uploaded, header=None).values.flatten()
    filtered = apply_filter(signal, fs=fs)
    features = extract_features(filtered, fs)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0][0]

    st.subheader("🔍 Leak Detection Result")
    st.write(f"**Leak Probability:** `{round(prediction, 2)}`")

    if prediction > 0.5:
        st.error("🚨 Leak Detected! Please inspect pipeline urgently.")
    else:
        st.success("✅ No Leak Detected. System appears normal.")

    st.markdown("---")
    st.subheader("📊 Signal Analysis Dashboard")
    plot_signal_analysis(signal, filtered, fs)
else:
    st.info("👈 Upload a `.csv` signal file to get started.")
