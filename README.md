**Sound Analytics with Raspberry Pi 4 using Microphone**

**Objective:** By the end of this session, participants will understand how to set up a microphone with the Raspberry Pi 4, capture audio, and conduct basic sound analytics.

---

**Prerequisites:**
1. Raspberry Pi 4 with Raspbian OS installed.
2. MicroSD card (16GB or more recommended).
3. USB keyboard, mouse, and monitor. You can also use Real VNC to access the Rasperry Pi (Need to enable VNC configuration in your Pi)
4. USB microphone compatible with Raspberry Pi.
5. Internet connectivity (Wi-Fi or Ethernet).
6. Basic knowledge of Python and Linux commands.

---

**1. Introduction (10 minutes)**
- Overview of sound analytics and its significance.
- Applications of sound processing with Raspberry Pi.

**2. Setting up the Raspberry Pi (10 minutes)**
- Booting up the Raspberry Pi.
- Setting up Wi-Fi/Ethernet.
- System updates:
  ```bash
  sudo apt update
  sudo apt upgrade
  ```
- Set up a [virtual environment](https://github.com/drfuzzi/INF2009_Setup) for this experiment (to avoid conflicts in libraries) using the details mentioned in Section 4.a
- Activate the virtual environment and complete the next steps within the environment

**3. Connecting and Testing the Microphone (15 minutes)**
- Physically connecting the microphone to the Raspberry Pi.
- Testing the microphone:
  ```bash
  arecord --duration=10 test.wav
  aplay test.wav
  ```

**4. Introduction to Sound Processing with Python (20 minutes)**
- Installing necessary Python libraries (may need additional libraries if not preinstalled with default python):
  ```bash
  sudo apt install portaudio19-dev
  pip3 install pyaudio
  pip3 install scipy matplotlib
  ```
- Capturing audio in Python.
- Fourier Transform: Understanding frequency components of sound.
- Visualizing sound waves (both the wave itself and the audio spectrum).
- For above tasks, you can use the [sample file](Codes/microphone_streaming_with_spectrum)
  

**5. Basic Sound Analytics (40 minutes)**
- Filtering: Removing noise or specific frequencies.
   ```
  For above task, you can use the sample file 'filtering_audio.py'. A sample bandpass filter (only passes audio within a certain frequencies as decided by the user are kept) is shown.
  ```
- Using the audio spectrum visualization, identify the frequency to be kept (e.g. tap sound or some particular sound) and change the above code accordingly.
- Feature extraction: MFCCs, chroma, and spectral contrast.
- Basic sound classification or sound event detection.

**6. Advanced Sound Analytics (20 minutes)**
- Introduction to machine learning with sound.
- Using pre-trained models or datasets for sound recognition.
- Real-time sound analytics: e.g., identifying specific sounds or words in a live audio stream.

---

**Homework/Extended Activities:**
1. Build a voice-activated command system.
2. Create a basic sound classifier using a dataset of various sounds.
3. Experiment with sound effects: reverb, echo, and pitch alteration.

---

**Resources:**
1. Raspberry Pi official documentation.
2. Python sound processing library documentation (e.g., `librosa`, `soundfile`).
3. Online communities and forums related to Raspberry Pi and sound analytics.

---
