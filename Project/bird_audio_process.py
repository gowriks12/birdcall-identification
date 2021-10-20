import librosa, librosa.display
from matplotlib import pyplot as plt
import IPython.display as ipd
import numpy as np
# matplotlib inline
from pydub import AudioSegment
from os import path
import subprocess

# params
src = "XC16964.mp3"
dst = "test.wav"
sampling = 21952
hop_length = 245
n_mels = 224
n_fft = 892
win_length = n_fft

# convert wav to mp3
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")
signal, Fs = librosa.load(dst, sr=21952)

# Load signal and plot
signal,Fs = librosa.load(dst,sr=sampling,mono=True,res_type="kaiser_fast")
librosa.display.waveplot(signal,sr=sampling)
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.show()

# Create spectrogram and plot
# spectr = librosa.feature.melspectrogram(signal, sr=sampling, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
#                                         win_length=win_length, fmin=300)
stft = librosa.core.stft(signal,hop_length=hop_length,n_fft=n_fft)
spectr = np.abs(stft)
# print('%s Fs = %d, x.shape = %s, x.dtype = %s' % ('Bird-call plot', Fs, signal.shape, signal.dtype))
# print(spectr)
log_spectr = librosa.amplitude_to_db(spectr)
librosa.display.specshow(log_spectr, sr=sampling, hop_length=hop_length)
plt.xlabel('Time (samples)')
plt.ylabel('Frequency')
plt.colorbar()
plt.show()

ipd.display(ipd.Audio(data=signal, rate=Fs))
