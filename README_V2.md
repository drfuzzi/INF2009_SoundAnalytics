# Sound Analytics & Edge Computing on Raspberry Pi

**Objective:** By the end of this session, students will understand how to set up a microphone with the Raspberry Pi, capture audio, and conduct basic sound analytics with an emphasis on edge efficiency and feature extraction.

---

## 1. Prerequisites

1. Raspberry Pi with Raspbian OS installed.
2. USB microphone compatible with Raspberry Pi (we will be using the microphone built into the webcam).
3. Internet connectivity (Wi-Fi or Ethernet).
4. (Optional) A Speaker/USB headset to hear playback.

**Introduction**

Developing an embedded audio listening system that emulates human hearing to interpret and process sounds is an emerging and impactful area of research. Such systems have diverse applications, including voice-enabled services like smart assistants (e.g., Amazon Alexa), healthcare solutions such as lung sound analysis and digital stethoscopes, and audio-based conversational interfaces like IVR systems and chatbots. These technologies aim to make sense of complex audio environments and deliver meaningful insights for real-world use cases.

In this lab, we will focus on building a system using Raspberry Pi to capture audio data, analyze it, and visualize the results. The workflow involves three key steps:
- acquiring audio signals through sensors or microphones
- applying signal processing techniques to extract relevant features
- presenting the processed data in an intuitive visual format.

This hands-on approach will provide practical experience in embedded audio processing and demonstrate how such systems can be applied across multiple domains.

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

> **Note to Students:** In the commands below, `audio_env` is simply a **label** for your virtual environment. You can choose any name you like (e.g., `lab1_env` or `my_sound_project`), but you must use that same name consistently when activating it.

1. **Create the environment** (using `audio_env` as the label):
```bash
python3 -m venv audio_env

```


2. **Activate it** so your terminal knows to use this isolated space:
```bash
source audio_env/bin/activate

```


*Once activated, you should see `(audio_env)` appear at the start of your command prompt.*


### Step C: Install Python Packages via Requirements

With the environment active, install the required analytics libraries by creating a file named `requirements.txt` with the following content:

```text
pyaudio
sounddevice
scipy
matplotlib
librosa
SpeechRecognition
swig
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

Edge computing requires balancing audio quality with CPU load. This exercise compares the "cost" of different sample rates by measuring the execution time and data overhead.

**Exercise: Latency Test**

```python
import sounddevice as sd
import time

def benchmark_sample_rate(rate, duration=3):
    print(f"Testing {rate}Hz...")
    start = time.perf_counter()
    # Capture audio at the specified rate
    recording = sd.rec(int(duration * rate), samplerate=rate, channels=1)
    sd.wait() # Wait for recording to finish
    end = time.perf_counter()
    print(f"Process took: {end-start:.4f}s")

# Test high-fidelity vs. edge-standard rates
benchmark_sample_rate(44100) # CD Quality
benchmark_sample_rate(16000) # Speech Quality (Standard for Edge AI)

```
> *Observe the 'Process took' time for both tests. Even though both recorded for exactly 3 seconds, you may notice a slight increase in time for the 44100Hz test. Considering that 44.1kHz generates ~132,000 samples while 16kHz generates only ~48,000 samples, why is it often better to use a lower sample rate for real-time analytics on an edge device like the Raspberry Pi? Using a lower sample rate like 16kHz naturally results in a loss of audio high-frequency detail (accuracy). When is it acceptable to accept this "loss of quality," and when would it be a failure for your edge system?*
---

## 5. Sound Analytics: Filter (Clean) $\rightarrow$ Transform (Visualize) $\rightarrow$ Extract (Analyze).

| Feature | Mathematical Basis | Best For |
| --- | --- | --- |
| **Bandpass Filter** | Signal Attenuation | Removing background noise/static. |
| **Mel Spectrogram** | Log-scale Frequency | General event detection (glass break, alarm). |
| **Chromagram** | 12 Pitch Classes | Music analysis, identifying specific tones. |
| **MFCC** | Cosine Transform of Log Spectrum | Speech recognition and identifying "who" or "what" made the sound. |


### A. Pre-Processing: The Bandpass Filter

Before extracting features, we remove unwanted noise. A **Bandpass Filter** allows frequencies within a specific range to pass through while blocking others. This is essential for isolating a specific sound, like a "tap" or a human voice, from background hum. 

```python
import numpy as np
from scipy.signal import butter, lfilter

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

# Example: Keeping only frequencies between 500Hz and 4000Hz (Speech range)
# filtered_audio = butter_bandpass_filter(y, 500, 4000, sr)

```

### B. Feature Extraction & Fingerprinting (Visualization)

Raw audio is too large for real-time edge analysis. We extract mathematical "fingerprints" to understand sound. `librosa` is used to convert the time-series data into mathematical representations. 

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the audio
y, sr = librosa.load('test.wav')

plt.figure(figsize=(12, 10))

# --- Plot 1: Mel Spectrogram (Human Perception) ---
plt.subplot(3, 1, 1)
S = librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel')
plt.title('Mel Spectrogram (Energy across Perceptual Frequencies)')
plt.colorbar(format='%+2.0f dB')

# --- Plot 2: Chromagram (Musical/Pitch Classes) ---
plt.subplot(3, 1, 2)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.title('Chromagram (12 Pitch Classes / Notes)')
plt.colorbar()

# --- Plot 3: MFCCs (The Sound "Fingerprint") ---
plt.subplot(3, 1, 3)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
librosa.display.specshow(mfccs, x_axis='time')
plt.title('MFCC Fingerprint (Vocal Tract Shape)')
plt.colorbar()

plt.tight_layout()
plt.show()

```

The spectrogram is a visual representation of the spectrum of frequencies of a signal as it varies with time.
 ![image](https://github.com/drfuzzi/INF2009_SoundAnalytics/assets/52023898/0ff75402-20c6-492f-9ee8-1f20c954c0a3)
 
In Western music, the term chroma feature or chromagram closely relates to twelve different pitch classes. Chroma-based features, which are also referred to as "pitch class profiles", are a powerful tool for analyzing music whose pitches can be meaningfully categorized (often into twelve categories) and whose tuning approximates to the equal-tempered scale.
![image](https://github.com/drfuzzi/INF2009_SoundAnalytics/assets/52023898/7be397f9-9b7e-4c98-a1ea-38dab4b2caba)

A Mel Spectrogram makes two important changes relative to a regular Spectrogram that plots Frequency vs Time. It uses the Mel Scale instead of Frequency on the y-axis. It uses the Decibel Scale instead of Amplitude to indicate colors. The Mel Scale is a perceptual scale of pitches judged by listeners to be equal in distance from one another.
![image](https://github.com/drfuzzi/INF2009_SoundAnalytics/assets/52023898/4663a522-8c0e-416f-95e3-eefe42a3696b)

This code is designed to take a raw audio file and decompose it into three distinct "mathematical lenses." Each lens reveals a different characteristic of the sound: its **energy**, its **pitch**, and its **texture**.

Here is the breakdown of the three key features:

---

### 1. Mel Spectrogram (The "Energy Map")

The code `librosa.feature.melspectrogram` calculates how much energy (loudness) exists at different frequencies over time.

* **The "Mel" Difference:** Standard spectrograms use a linear scale (Hertz). However, humans are much better at detecting changes in low frequencies than high ones. The **Mel Scale** warps the frequency axis to mimic human hearing.
* **The Math:** `librosa.power_to_db` converts the raw power into **Decibels**. This is essential because human hearing is logarithmic; without this conversion, quiet background noises would be invisible on the graph.
* **Use Case:** Identifying *when* a sound happened and how much "punch" or energy it had.

---

### 2. Chromagram (The "Musical/Pitch Map")

The code `librosa.feature.chroma_stft` creates a "Chroma" representation. This is the most "musical" of the three features.

* **How it works:** It maps the entire frequency spectrum into **12 bins**, corresponding to the 12 semitones of the Western musical scale (C, C#, D, D#, etc.).
* **Octave Agnostic:** It ignores how high or low a note is (the octave) and only focuses on the **pitch class**. For example, a "High C" and a "Low C" will both show up in the same bin.
* **Use Case:** Identifying chords, melodies, or harmonic content. If you are analyzing a machine and it starts "whining" at a specific musical note, the Chromagram will catch it.

---

### 3. MFCC: Mel Frequency Cepstral Coefficients (The "Fingerprint")

The code `librosa.feature.mfcc` generates the most compressed and powerful feature for machine learning.

* **The "Shape" of Sound:** If a Spectrogram shows "what" frequencies are present, the MFCC shows the **overall envelope** or "shape" of the sound. In humans, this represents the physical shape of the vocal tract (throat and mouth) while speaking.
* **Dimensionality Reduction:** Instead of thousands of frequency points, we use `n_mfcc=13` to condense the sound into just 13 coefficients. This makes it "lightweight" enough for edge devices (like a smart speaker) to process in real-time.
* **Use Case:** **Speech Recognition** and **Voice Biometrics**. It is the "fingerprint" that allows a computer to distinguish between a "Ba" sound and a "Pa" sound, or between User A and User B.

---

### Summary of Differences

| Feature | Focus | Human Equivalent |
| --- | --- | --- |
| **Mel Spectrogram** | Volume & Frequency | "How loud was that high-pitched beep?" |
| **Chromagram** | Harmonic/Musical Note | "What note is that whistle humming?" |
| **MFCC** | Texture & Timbre | "Is that a human voice or a guitar string?" |

---

## 6. Advanced Module: Hybrid Edge-to-Cloud Recognition

Tier 1 (Local): Use the RMS threshold (volume check) to filter out silence.

Tier 2 (Local Edge AI): Use [CMUSphinx](https://cmusphinx.github.io/wiki/) to recognize a specific "Wake Word" (e.g., "hello"). This runs entirely offline.

Tier 3 (Cloud AI): If the wake word is detected, only then do we send the audio to [Google Speech Recognition](https://github.com/Uberi/speech_recognition/tree/master/third-party/Source%20code%20for%20Google%20API%20Client%20Library%20for%20Python%20and%20its%20dependencies) for high-accuracy transcription.

```python
import speech_recognition as sr
import numpy as np

r = sr.Recognizer()
mic = sr.Microphone()

# Note: You must have 'pocketsphinx' installed: pip install pocketsphinx
WAKE_WORD = "hello"

with mic as source:
    print(f"Listening... (Threshold: RMS 500 | Wake Word: '{WAKE_WORD}')")
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)
    
    # --- TIER 1: Local Volume Check ---
    data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
    rms = np.sqrt(np.mean(data**2))
    
    if rms > 500:
        print(f"Sound detected ({rms:.0f} RMS). Checking for Wake Word via CMUSphinx...")
        
        try:
            # --- TIER 2: Local Wake Word Recognition (Offline) ---
            # Sphinx is fast and private but less accurate for long sentences.
            local_recognition = r.recognize_sphinx(audio).lower()
            
            if WAKE_WORD in local_recognition:
                print(f"Wake word '{WAKE_WORD}' detected! Sending to Google Cloud...")
                
                # --- TIER 3: Cloud Recognition (Online) ---
                # Only use Google for the heavy lifting.
                text = r.recognize_google(audio)
                print("Google Cloud Recognized: " + text)
            else:
                print(f"Local Sphinx heard '{local_recognition}', not the wake word. Ignoring.")
                
        except sr.UnknownValueError:
            print("Sphinx could not understand the audio.")
        except sr.RequestError as e:
            print(f"Sphinx error; {e}")
            
    else:
        print(f"Sound too quiet ({rms:.0f} RMS). Ignored at the Edge.")

```

---

### Test Scenario 1: The "Silence/Ambient" Test

**Objective:** Observe the first line of defense (Tier 1: Energy Threshold).

* **Student Action:** Run the script and sit in total silence for 5 seconds.
* **Expected Console Output:**
> `Listening... (Threshold: RMS 500 | Wake Word: 'hello')`
> `Sound too quiet (112 RMS). Ignored at the Edge.`


* **Observation:** The code didn't even try to "read" the audio. It calculated a single number (RMS) and stopped because it was below 500.

---

### Test Scenario 2: The "Sharp Noise" Test (Non-Speech)

**Objective:** Observe Tier 2 rejecting loud sounds that aren't human language.

* **Student Action:** Clap your hands loudly or tap the table near the microphone.
* **Expected Console Output:**
> `Sound detected (2150 RMS). Checking for Wake Word via CMUSphinx...`
> `Sphinx could not understand the audio.`


* **Observation:** The sound was loud enough to pass Tier 1, but the "Local Brain" (Sphinx) couldn't find any phonetic patterns that look like words, so it crashed out.

---

### Test Scenario 3: The "Wrong Intent" Test

**Objective:** Observe the Wake Word filter (Tier 2 preventing Tier 3).

* **Student Action:** Say clearly, *"Good morning, how is the weather?"*
* **Expected Console Output:**
> `Sound detected (1800 RMS). Checking for Wake Word via CMUSphinx...`
> `Local Sphinx heard 'good morning how is the weather', not the wake word. Ignoring.`


* **Observation:** The device "understood" you locally, but because you didn't say **"hello"**, it protected your privacy and saved your API quota by **not** sending the data to Google Cloud.

---

### Test Scenario 4: The "Full Pipeline" Test

**Objective:** Observe the successful handoff from Edge to Cloud.

* **Student Action:** Say, *"Hello, tell me a joke."*
* **Expected Console Output:**
> `Sound detected (2450 RMS). Checking for Wake Word via CMUSphinx...`
> `Wake word 'hello' detected! Sending to Google Cloud...`
> `Google Cloud Recognized: hello tell me a joke`


* **Observation:** This is the only scenario where the data traveled to the internet. The high accuracy of the final sentence is thanks to the Cloud's massive neural networks.

---

**Why this matters for Edge Computing**

In a real-world scenario (like a smart mirror or a robot), the device doesn't have the battery life to stream audio to the cloud 24/7. This code mimics a "Low Power Mode" where the cloud is only engaged when the local processor is "sure" there is something worth transcribing.

---

## Exercise: Latency vs. Accuracy Investigation

In this task, we compare **CMUSphinx** (Local/Edge) and **Google Cloud** (Remote/Cloud). While the Cloud is more "intelligent," it requires sending data over the network, which introduces a delay.

### 1. Modified Code for Performance Profiling

Ask students to use this version of the script, which uses `time.time()` to measure exactly how long each engine takes to respond.

```python
import time
import speech_recognition as sr

r = sr.Recognizer()
# ... (inside your microphone context manager)

# --- Timing CMUSphinx (Edge) ---
start_edge = time.time()
try:
    edge_text = r.recognize_sphinx(audio)
    edge_latency = time.time() - start_edge
    print(f"Edge Result: '{edge_text}' | Latency: {edge_latency:.2f}s")
except:
    print("Edge failed.")

# --- Timing Google Cloud (Cloud) ---
start_cloud = time.time()
try:
    cloud_text = r.recognize_google(audio)
    cloud_latency = time.time() - start_cloud
    print(f"Cloud Result: '{cloud_text}' | Latency: {cloud_latency:.2f}s")
except:
    print("Cloud failed.")

```

---

### 2. Student Exploration Task

Perform the following tests and note down your findings:

* **Test A (Simple):** Say a single, clear word (e.g., "Apple").
* **Test B (Complex):** Say a long, complex sentence (e.g., "The thermodynamic properties of the engine were fluctuating.")
* **Test C (Noisy):** Say "Hello" while playing music or background noise.

#### Observations Table

| Test Case | Edge Latency (s) | Cloud Latency (s) | Which was more accurate? |
| --- | --- | --- | --- |
| **Simple Word** |  |  |  |
| **Complex Sentence** |  |  |  |
| **Noisy Environment** |  |  |  |

---

### 3. Discussion Questions

1. **The Bottleneck:** Why is the Cloud Latency usually much higher than Edge Latency? (Hint: Think about `upload speed` and `server processing`).
2. **The Brain Power:** Why did CMUSphinx struggle with the "Complex Sentence" compared to Google?
3. **Real-world Application:** If you were building a **Voice-Controlled Car**, which engine would you use for "STOP NOW!" and why?

---

