import librosa, librosa.display
import os
from matplotlib import pyplot as plt
import IPython.display as ipd
import numpy as np
import torch
# matplotlib inline
from pydub import AudioSegment
# import FigureCanvasAgg as FigureCanvas
from os import path
import subprocess
directory = "aldfly"
new_dir = directory + "_spectr"
os.mkdir(new_dir)
for filename in os.listdir(directory):
    src = directory + "/" + filename
    save_path = new_dir + "/" + filename.split(".")[0] + ".png"
    print(src)

    dst = "test.wav"
    # params
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
    # librosa.display.waveplot(signal,sr=sampling)
    # plt.xlabel('Time (samples)')
    # plt.ylabel('Amplitude')
    # plt.show()

    # Create spectrogram and plot
    spectr = librosa.feature.melspectrogram(signal, sr=sampling, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
                                            win_length=win_length, fmin=300)
    # stft = librosa.core.stft(signal,hop_length=hop_length,n_fft=n_fft)
    # spectr = np.abs(stft)

    # print(spectr)
    log_spectr = librosa.amplitude_to_db(spectr)
    # print(type(log_spectr))
    librosa.display.specshow(log_spectr, sr=sampling, hop_length=hop_length)
    plt.savefig(save_path)
    # plt.xlabel('Time (samples)')
    # plt.ylabel('Frequency')
    # plt.colorbar()
    # plt.show()

    # log_spectr = torch.from_numpy(log_spectr)


    # torch.save(log_spectr,save_path)


    # ipd.display(ipd.Audio(data=signal, rate=Fs))
