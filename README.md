**Sound Analytics with Raspberry Pi 4 using Microphone**

**Objective:** By the end of this session, participants will understand how to set up a microphone with the Raspberry Pi 4, capture audio, and conduct basic sound analytics.

---

**Prerequisites:**
1. Raspberry Pi 4 with Raspbian OS installed.
2. MicroSD card (16GB or more recommended).
3. USB keyboard, mouse, and monitor. You can also use Real VNC to access the Rasperry Pi (Need to enable VNC configuration in your Pi)
4. USB microphone compatible with Raspberry Pi.
5. (Optional) A Speaker/USB headset to hear the playback
6. Internet connectivity (Wi-Fi or Ethernet).
7. Basic knowledge of Python and Linux commands.

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
  - For above tasks, you can use the [sample code](Codes/microphone_streaming_with_spectrum)
  

**5. Basic Sound Analytics (40 minutes)**
- Filtering: Removing noise or specific frequencies. The [sample code](Codes/filtering_audio) illustrates a bandpass filter (only passes audio within a certain frequencies as decided by the user are kept).
  - Using the audio spectrum visualization, identify the frequency to be kept (e.g. tap sound or some particular sound) and change the above code accordingly.
- Feature extraction: MFCCs, chroma, and spectral contrast.
- Basic sound classification or sound event detection.

**6. Advanced Sound Analytics (20 minutes)**
- Introduction to machine learning with sound through speech recognition task (through [CMU Sphinx](https://cmusphinx.github.io/wiki/) and [Google Speech Recognition](https://github.com/Uberi/speech_recognition/tree/master/third-party/Source%20code%20for%20Google%20API%20Client%20Library%20for%20Python%20and%20its%20dependencies).
- Installing the [speech_recognition library](https://github.com/Uberi/speech_recognition#readme) through following commands
  ```bash
  sudo apt-get install flac
  pip install pocketsphinx
  pip install SpeechRecognition
  ```
  - A FLAC encoder (installed through above command) is required to encode the audio data to send to the API. Similarly CM Sphinx is also installed through above commands
  - The [sample code](Codes/microphone_recognition.py) illustrates to record an audio, then use the Sphinx API and Google Speech Recognition APIs to predict the spoken text
- Employ other speech recognition APIs provided in the [speech_recognition library](https://github.com/Uberi/speech_recognition#readme) and compare the performance on Rasperry PI 4
- Modify the code to identify certain words in the generated (predicted text) which can form the basis for 'wake word' based system control (e.g. Ok Google, Alexa or Siri) 
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
