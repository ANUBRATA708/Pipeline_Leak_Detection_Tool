<h1 align="center">🚰 Water Pipeline Leak Detection</h1>
<p align="center">
  An AI-powered tool that detects water pipeline leaks by analyzing vibration signals using signal processing techniques and a deep learning model.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-Keras-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Signal%20Processing-FFT-yellow?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge"/>
</p>

---

## 🎯 Features

- 🔍 Upload vibration signals in `.csv` format
- 🎛️ Apply bandpass filter (20–300 Hz)
- 📈 View raw, filtered, and FFT signal graphs
- 🧠 Predict leaks using deep learning classifier (binary)
- 🌈 Streamlit-based colorful and interactive UI
- 💾 Load pre-trained model and test with sample data

---

## 📦 Project Structure


waterleak_detection/

├── app.py                     # Streamlit app with dashboard

├── launch_app.bat             # Direct launcher

├── leak_detector.py           # Signal filtering and feature extraction

├── models/

│   └── leak_model.h5          # Pre-trained DL model

├── data/

│   ├── sample_leak_signal.csv     # Signal data (leak)

│   ├── sample_no_leak_signal.csv # Signal data (no leak)

│   ├── features.npy               # Optional extracted features

│   └── labels.npy                # Ground truth labels

├── requirements.txt           # Python dependencies

└── README.md                  # Project documentation

   

## 🖼 Sample Output
![image](https://github.com/user-attachments/assets/f9f98b1e-b69b-43e8-8354-80eec59150e2)
![image](https://github.com/user-attachments/assets/19035a99-0703-4bed-839d-c8b3800f41a8)
![image](https://github.com/user-attachments/assets/4f65222e-3394-4969-825f-95a05eca2b98)

## 🧠 Built With

Python

Streamlit

NumPy / Pandas

SciPy

Keras / TensorFlow



## 🤝 Contributors

👨‍💻 ANUBRATA MAJUMDAR 


📃 License
This project is licensed under the MIT License.




