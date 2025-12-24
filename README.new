# Sound Analytics & Edge Computing on Raspberry Pi

**Objective:** By the end of this session, students will understand how to set up a microphone with the Raspberry Pi, capture audio, and conduct basic sound analytics with an emphasis on edge efficiency and feature extraction.

---

## 1. Prerequisites

1. Raspberry Pi with Raspbian OS installed.
2. USB microphone compatible with Raspberry Pi (we will be using the microphone built into the webcam).
3. Internet connectivity (Wi-Fi or Ethernet).
4. (Optional) A Speaker/USB headset to hear playback.

**Introduction**
Developing an embedded-based audio listening system similar to the human hearing to make sense of sounds is one of the growing areas of research. There are various application to the same not limited to 1) voice enabled services (e.g. Alexa Dot), 2) healthcare applications (e.g. lung sound analysis /  digital stethoscope) and 3) audio chatbots (e.g. IVR services). In this lab, we will be working on ways to capture audio data, analyze it and visualize the same on Raspberry Pi. A basic approach for the same is as shown below:
![image](https://github.com/drfuzzi/INF2009_SoundAnalytics/assets/52023898/bb9a2d8a-b4ae-4207-8272-21162987c821)

---

## 2. Advanced Requirement Setup

To ensure the lab is reproducible and avoids common environment errors, we use a dedicated virtual environment and a requirements file.

### Step A: Install System Dependencies

Install the low-level audio I/O library and the FLAC encoder needed for cloud communication:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y portaudio19-dev flac python3-venv

```

### Step B: Create and Activate Virtual Environment

```bash
python3 -m venv audio_env
source audio_env/bin/activate

```

### Step C: Install Python Packages via Requirements

Create a file named `requirements.txt` with the following content:

```text
pyaudio
sounddevice
scipy
matplotlib
librosa
SpeechRecognition
pocketsphinx

```

Then run:

```bash
pip install -r requirements.txt

```

---

## 3. Hardware Testing

Physically connect your microphone and test the recording capability directly from the terminal:

```bash
arecord --duration=10 test.wav
aplay test.wav

```

*Note: Do not delete `test.wav`, as it is required for feature extraction.*

---

## 4. Edge Computing Module: Performance Benchmarking

Edge computing requires balancing audio quality with CPU load. This exercise compares the "cost" of different sample rates.

**Exercise: Latency Test**

```python
import sounddevice as sd
import time

def benchmark_sample_rate(rate, duration=3):
    print(f"Testing {rate}Hz...")
    start = time.perf_counter()
    recording = sd.rec(int(duration * rate), samplerate=rate, channels=1)
    sd.wait() 
    end = time.perf_counter()
    print(f"Process took: {end-start:.4f}s")

benchmark_sample_rate(44100) # CD Quality
benchmark_sample_rate(16000) # Speech Quality (Standard for Edge AI)

```

---

## 5. Analytics Module: Feature Extraction

Raw audio is too large for real-time edge analysis. We extract mathematical "fingerprints" to understand sound.

### Audio Feature Comparison Table

| Feature | What it represents | Edge Use Case |
| --- | --- | --- |
| **Spectrogram** | Energy across different frequencies over time. | Detecting specific events (e.g., glass breaking). |
| **Chromagram** | Twelve different pitch classes (musical notes). | Analyzing music or harmonic sounds. |
| **MFCC** | The "shape" of the vocal tract or sound source. | Speech Recognition and Voice Assistants. |

**Exercise: Plotting MFCCs**

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load('test.wav')

plt.figure(figsize=(10, 6))
# Plot Spectrogram
plt.subplot(2, 1, 1)
S = librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel')
plt.title('Mel Spectrogram')

# Plot MFCCs
plt.subplot(2, 1, 2)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
librosa.display.specshow(mfccs, x_axis='time')
plt.title('MFCC Fingerprint')
plt.tight_layout()
plt.show()

```

---

## 6. Advanced Module: Hybrid Edge-to-Cloud Recognition

This demonstrates a hybrid architecture: the Pi handles volume detection locally (Edge Trigger) and only uses the Cloud when a threshold is met to save bandwidth.

```python
import speech_recognition as sr
import numpy as np

r = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    print("Listening... (Edge Threshold: RMS 500)")
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)
    
    # Calculate volume locally
    data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
    rms = np.sqrt(np.mean(data**2))
    
    if rms > 500:
        print("Threshold met! Sending to Google Cloud...")
        print("Recognized: " + r.recognize_google(audio))
    else:
        print(f"Sound too quiet ({rms:.0f} RMS). Ignored at the Edge.")

```

---

**Would you like me to generate the `requirements.txt` file content as a standalone code block for your GitHub repository?**
