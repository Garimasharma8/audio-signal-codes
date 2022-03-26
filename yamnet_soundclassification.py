#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:56:40 2022

@author: garimasharma
Sound classification using YAMNet
YAMNet is used to check the classes of audio clips, has over 521 classes
"""

# steps to do sound classification on YAMNet
# 1. Import libraries
# 2. Import YAMnet model
# 3. Get class names from the model
# 4. Load the audio signal to be checked
# 5. Ensure its sr is 16k
# 6. Normalize the signal in +1 to -1 range
# 7. Execute the model and check the class
# 8. Visualize/ compare the predicted class with top 10 similar classes


#%% import useful libraries
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import csv
import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.io import wavfile
import scipy
from scipy import signal

#%% Import YAMNet model

model_yamnet = hub.load('https://tfhub.dev/google/yamnet/1')

#%% 

# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names

class_map_path = model_yamnet.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

#%% Add a method to verify and convert a loaded audio is on the proper sample_rate (16K), 
#otherwise it would affect the model's results.

def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform

#%% Load the audio you want to check on YAMNet, make sure the sampling rate is 16000. 

sample_rate, wavdata = wavfile.read('/Users/garimasharma/Downloads/covid cough/Analysis 1- covid cough vs healthy cough/covid with cough/covidandroidwithcough/cough/CC1.wav', 'rb')

sample_rate, wavdata = ensure_sample_rate(sample_rate, wavdata, desired_sample_rate=16000)

wavdata = wavdata[:,1]

# Show some basic information about the audio.
duration = len(wavdata)/sample_rate
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(wavdata)}')

#%% The wav_data needs to be normalized to values in [-1.0, 1.0] 

waveform = wavdata / tf.int16.max

#%% executing the model and check the class

# Run the model, check the output.
scores, embeddings, spectrogram = model_yamnet(waveform)

scores_np = scores.numpy()
spectrogram_np = spectrogram.numpy()
infered_class = class_names[scores_np.mean(axis=0).argmax()]
print(f'The main sound is: {infered_class}')

#%% Visulaization - see the top classes the waveform matches to 

plt.figure(figsize=(10, 6))

# Plot the waveform.
plt.subplot(3, 1, 1)
plt.plot(waveform)
plt.xlim([0, len(waveform)])

# Plot the log-mel spectrogram (returned by the model).
plt.subplot(3, 1, 2)
plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower')

# Plot and label the model output scores for the top-scoring classes.
mean_scores = np.mean(scores, axis=0)
top_n = 10   # no of best n classes
top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
plt.subplot(3, 1, 3)
plt.imshow(scores_np[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')

# patch_padding = (PATCH_WINDOW_SECONDS / 2) / PATCH_HOP_SECONDS
# values from the model documentation
patch_padding = (0.025 / 2) / 0.01
plt.xlim([-patch_padding-0.5, scores.shape[0] + patch_padding-0.5])
# Label the top_N classes.
yticks = range(0, top_n, 1)
plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
_ = plt.ylim(-0.5 + np.array([top_n, 0]))


#%% read all the files from a folder 
import librosa
import glob
import os.path

t=[]
sampling_rate=[]
path = '/Users/garimasharma/Downloads/data-to-share-covid-19-sounds/KDD_paper_data/covidandroidwithcough/cough/'
for filename in glob.glob(os.path.join(path, '*.wav')):
    y, sr = librosa.load(filename) 
    #y = y / tf.int16.max
    sampling_rate.append(sr)
    t.append(y)

for i in range(len(t)):
    scores,embeddings, spectrogram = model_yamnet(t[i])
    scores_np = scores.numpy()
    infered_class = class_names[scores_np.mean(axis=0).argmax()]
    print(f'The main sound is: {infered_class}')    
    
    
#%% 

