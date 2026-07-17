# Real-Time Audio Analysis Toool

A real-time digital signal processing (DSP) application designed to capture, analyze, and visualize live audio input with low latency.

## Features
*   **Live Stream Capture:** Low-latency audio ingestion from primary input devices.
*   **Fast Fourier Transform (FFT):** Real-time frequency domain conversion for spectrum analysis and musical note calculation.
*   **Visualizations:** Dynamic composit visuals based off of computed musical notes in the input.

## Tech Stack & Libraries
*   **Core Logic:** Python
*   **Audio Handling:** Pyaudio/numpy
*   **Visualization/GUI:** Pygame

## 📋 Prerequisites
Before running this project, ensure you have the following installed on your machine:
*   Python 3.13.3 or higher 
*   A working microphone or system audio input device.

## ⚙️ Installation & Setup

1. Clone the repository using your configured SSH key:
   ```bash
   git clone git@github-work:devonVillareal/RealTimeAudioAnalysis.git
   cd RealTimeAudioAnalysis
   ```

2. Install the required dependencies:
   ```bash
   # Add your installation command here (e.g., pip install -r requirements.txt)
   ```

## 🏃 Running the Application
To launch the real-time analyzer, execute the main script:
```bash
# Add your execution command here (e.g., python main.py)
```

## 🛠️ Architecture & How It Works
1.  **Audio Buffer:** Captures raw pulse-code modulation (PCM) data in small chunk sizes to minimize delay.
2.  **Windowing Function:** Applies a Hanning/Hamming window to prevent spectral leakage.
3.  **FFT Processing:** Computes the magnitude spectrum to extract frequency bins.
4.  **Render Thread:** Decouples audio processing from the UI thread to prevent visual stuttering.
