#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:55:12 2022

@author: garimasharma
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from librosa.feature import mfcc, spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_rolloff
from scipy import signal
import wave
import sys
from scipy.io import wavfile
from librosa import display
from librosa.feature import spectral_flatness, zero_crossing_rate



#%% import an audio signal

fs, x = wavfile.read(r'/Users/garimasharma/Downloads/Texture analysis/Sample audio texture files/McDermott_Simoncelli_2011_168_Sound_Textures/Natural/Fire1.wav')

size_x = np.size(x) # check the size of an audio file

x = wave.open('/Users/garimasharma/Downloads/Texture analysis/Sample audio texture files/McDermott_Simoncelli_2011_168_Sound_Textures/Natural/Fire1.wav')
length_file = x.getnframes()
no_of_channels = x.getnchannels()
sampling_rate = x.getframerate() 
# to get sample values or read frames

data = x.readframes(-1) 
wav_data = np.fromstring(data, 'int16')

# Another way of getting data is (I preffer)
import librosa

y, sr = librosa.load('/Users/garimasharma/Downloads/Texture analysis/Sample audio texture files/McDermott_Simoncelli_2011_168_Sound_Textures/Natural/Fire1.wav')
duration = librosa.get_duration(y,sr)  # durartion of the signal

#%% plot the original signal 

plt.figure(1)
plt.plot(y)
plt.xlabel('samples')

plt.title('original audio signal')
plt.show()

#%% plot original signal with time on x axis


Time = np.linspace(0, int(len(y)/sr), len(y))
plt.figure(1)
plt.plot(Time, y)
plt.xlabel('Time')
plt.ylabel('amplitude')
plt.title('Original audio signal')
plt.show()


#%% extract MFCC from an audio signal


M1 = mfcc(y,sr, n_mfcc=13, dct_type=2, norm='ortho')

#% Mel spectrogram 

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

# %plot MFCCs

fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max), ax = ax[0])

fig.colorbar(img, ax=[ax[0]])
ax[0].set(title='Mel spectrogram')
ax[0].label_outer()
img1 = librosa.display.specshow(M1, x_axis='time', ax = ax[1])
fig.colorbar(img1, ax=[ax[1]])
ax[1].set(title='MFCC')

#%% Spectral centroid from each frame (no of columns must be same as in mfcc, as these are frames)

cent = spectral_centroid(y, sr)  # from audio directly 

S, phase = librosa.magphase(librosa.stft(y=y))   # from spectrogram input
cent1 = spectral_centroid(S=S)     # both must be same 

#%% Spectral bandwidth

s_band = spectral_bandwidth(y, sr)

s_band1= spectral_bandwidth(S=S)  # both must be same and equal

#%% Spectral contrast 

s_contrast = spectral_contrast(y, sr)

#%% spectral rolloff

s_rolloff = spectral_rolloff(y, sr)

#%% spectral flatness

s_flatness = spectral_flatness(y)

#%% zero crossing rate

zero_rate = zero_crossing_rate(y)


#%% STFT on an audio signal 

y_freq_domain = librosa.stft(y,n_fft=512, hop_length=256)

y_fft_abs = np.abs(y_freq_domain)

# display a spectrogtam

fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(y_fft_abs,ref=np.max),y_axis='log', x_axis='time', ax=ax)
ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")

#%% constant Q transform

y_cqt = librosa.cqt(y, sr, hop_length=512, n_bins=35, bins_per_octave=10)
y_cqt_abs = np.abs(y_cqt)

fig, ax = plt.subplots()

img = librosa.display.specshow(librosa.amplitude_to_db(y_cqt_abs, ref=np.max),sr=sr, x_axis='time', y_axis='cqt_note', ax=ax)
ax.set_title('Constant Q transform')
fig.colorbar(img, ax=ax, format="%+2.0f dB")

#%% 