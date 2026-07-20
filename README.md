# Real-Time Audio Analysis Toool

A real-time digital signal processing (DSP) application that takes live audio input and trasforms it into particle based animations. The project analyzes bass, mid and treble frequencies in real time, detects musical notes, and uses this information to generates parameterized geometric shapes that react fluidly to the audio being played.

![Demo](Demo.gif)

## Features
*   **Live Stream Capture:** Low-latency audio ingestion from primary input devices.
*   **Fast Fourier Transform (FFT):** Real-time frequency domain conversion for spectrum analysis and musical note calculation.
*   **Visualizations:** Dynamic composit visuals based off of computed musical notes in the input.

## Tech Stack & Libraries
*   **Core Logic:** Python
*   **Audio Handling:** Pyaudio/numpy
*   **Visualization/GUI:** Pygame

## Prerequisites
Before running this project, ensure you have the following installed on your machine:
*   Python 3.8 or higher 
*   A working microphone or system audio input device.

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone git@github-work:devonVillareal/RealTimeAudioAnalysis.git
   cd RealTimeAudioAnalysis
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application
To launch the real-time analyzer, execute the main script:
```bash
python3 visualizer.py
```

## Architecture & How It Works
1.  **Audio Buffer:** Captures raw audio data in small chunk sizes to minimize delay.
2.  **FFT Processing:** Computes the magnitude spectrum to extract frequency bins.
3.  **Render Thread:** Decouples audio processing from the UI thread to prevent visual stuttering.
