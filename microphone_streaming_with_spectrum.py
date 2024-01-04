#%% Import the required libraries
# Important: Close the figure to exit from the program
import pyaudio # Refer to https://people.csail.mit.edu/hubert/pyaudio/
import struct  # Refer to https://docs.python.org/3/library/struct.html (Used for converting audio read as bytes to int16)
import numpy as np 
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq # Refer to https://docs.scipy.org/doc/scipy/tutorial/fft.html (Used for Fourier Spectrum to display audio frequencies)
import time # In case time of execution is required (in this example, it is employed to check on frame rate)
from tkinter import TclError

#%% Parameters
BUFFER = 1024 * 16           # samples per frame (you can change the same to acquire more or less samples)
FORMAT = pyaudio.paInt16     # audio format (bytes per sample)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second


#%% create matplotlib figure and axes with initial random plots as placeholder
fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 7))
# create a line object with random data
x = np.arange(0, 2*BUFFER, 2)       # samples (waveform)
xf = fftfreq(BUFFER, (1/RATE))[:BUFFER//2]

line, = ax1.plot(x,np.random.rand(BUFFER), '-', lw=2)
line_fft, = ax2.plot(xf,np.random.rand(BUFFER//2), '-', lw=2)

# basic formatting for the axes
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')
ax1.set_ylim(-1000, 1000) # change this to see more amplitude values (when we speak)
ax1.set_xlim(0, BUFFER)

ax2.set_title('SPECTRUM')
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Log Magnitude')
ax2.set_ylim(0, 1000) 
ax2.set_xlim(0, RATE/2)

# show the plot
plt.show(block=False)

print('stream started')

#%% Initialize the pyaudio class instance
audio = pyaudio.PyAudio()

# stream object to get data from microphone
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=BUFFER
)

# for measuring frame rate
frame_count = 0
start_time = time.time()

while True:
   
    # binary data
    data = stream.read(BUFFER)  
   
    # convert data to 16bit integers
    data_int = struct.unpack(str(BUFFER) + 'h', data)    
   
   
    # compute FFT
    start_time=time.time()
    yf = fft(data_int)
   
    #update line plots for both axes
    line.set_ydata(data_int)
    line_fft.set_ydata(2.0/BUFFER * np.abs(yf[0:BUFFER//2]))
   
    # update figure canvas
    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
        frame_count += 1
       
    except TclError:
       
        # calculate average frame rate
        frame_rate = frame_count / (time.time() - start_time)
       
        print('stream stopped')
        print('average frame rate = {:.0f} FPS'.format(frame_rate))
        break

audio.terminate()