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
import csv
from csv import writer

directory = "IndianaBirds"


def create_spectrogram():
    # imheight = 50
    # imwidth = 34
    train_dir = "Spectrograms/train_spectr"
    test_dir = "Spectrograms/test_spectr"
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    train_list = []
    test_list = []
    for bird in os.listdir(directory):
        dir_len = len(os.listdir(directory + "/" + bird)[:5])
        train_count = round(dir_len * 0.7)
        count = 1
        for filename in os.listdir(directory + "/" + bird)[:5]:
            if count <= train_count:
                new_dir = train_dir
                train_list.append([filename, bird])
            else:
                new_dir = test_dir
                test_list.append([filename, bird])
            src = directory + "/" + bird + "/" + filename
            save_path = new_dir + "/" + filename.split(".")[0] + ".png"
            # print(src)
            # print(save_path)

            dst = "test.wav"
            # params
            sampling = 21952
            hop_length = 245
            n_mels = 224
            n_fft = 892
            win_length = n_fft
            plt.rcParams["figure.figsize"] = [7.50, 5.00]
            plt.rcParams["figure.autolayout"] = True

            # convert mp3 to wav
            sound = AudioSegment.from_mp3(src)
            sound.export(dst, format="wav")
            signal, Fs = librosa.load(dst, sr=21952)

            # Load signal and plot
            signal,Fs = librosa.load(dst,sr=sampling,mono=True,res_type="kaiser_fast")
            plt.axis('off')

            # Create spectrogram and plot
            spectr = librosa.feature.melspectrogram(signal, sr=sampling, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
                                                    win_length=win_length, fmin=300)
            log_spectr = librosa.amplitude_to_db(spectr)

            librosa.display.specshow(log_spectr, sr=sampling, hop_length=hop_length)
            plt.savefig(save_path)

            count += 1
    return train_list, test_list


def list_to_csv(train_lis, test_lis):
    file1 = open('train.csv', 'w+', newline = '')
    file2 = open('test.csv', 'w+', newline = '')
    with file1:
        write = csv.writer(file1)
        write.writerows(train_lis)
    with file2:
        write = csv.writer(file2)
        write.writerows(test_lis)


trl, tel = create_spectrogram()
list_to_csv(trl, tel)
# print(trl)
# print(tel)

