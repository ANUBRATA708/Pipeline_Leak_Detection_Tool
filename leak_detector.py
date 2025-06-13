import numpy as np
from scipy.signal import butter, lfilter, welch

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = fs * 0.5
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, lowcut=20, highcut=300, fs=1000):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data)

def extract_features(signal, fs):
    rms = np.sqrt(np.mean(signal**2))
    kurtosis = np.mean((signal - np.mean(signal))**4) / (np.std(signal)**4)
    freqs, psd = welch(signal, fs=fs)
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
    return [rms, kurtosis, spectral_centroid]

def generate_dataset(num_samples=1000, fs=1000, leak_prob=0.5):
    X, y = [], []
    for _ in range(num_samples):
        t = np.linspace(0, 1, fs)
        base = np.sin(2 * np.pi * 50 * t)
        if np.random.rand() < leak_prob:
            sig = base + 0.5 * np.sin(2 * np.pi * 120 * t) + 0.3 * np.random.randn(len(t))
            label = 1
        else:
            sig = base + 0.1 * np.random.randn(len(t))
            label = 0
        sig = apply_filter(sig, fs=fs)
        features = extract_features(sig, fs)
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = generate_dataset()
    np.save("data/features.npy", X)
    np.save("data/labels.npy", y)