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

#Keras imports
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import model_from_json
import pandas as pd

directory = "IndianaBirds"
num_classes = 10

def create_spectrogram():
    # imheight = 50
    # imwidth = 34
    train_dir = "Spectrograms/train_spectr"
    test_dir = "Spectrograms/test_spectr"
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    train_list = [["path", "label"]]
    test_list = [["path", "label"]]
    for bird in os.listdir(directory):
        dir_len = len(os.listdir(directory + "/" + bird)[:5])
        train_count = round(dir_len * 0.7)
        count = 1
        for filename in os.listdir(directory + "/" + bird)[:5]:
            if count <= train_count:
                new_dir = train_dir
                train_list.append([new_dir + "/" + filename.split(".")[0] + ".png", bird])
            else:
                new_dir = test_dir
                test_list.append([new_dir + "/" + filename.split(".")[0] + ".png", bird])
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


#Create Keras Model

def create_model(trl, tel):
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    x_train, y_train, x_test, y_test = train, train[-1], test, test[-1]

    print("Size of Training Data:", np.shape(x_train))
    print("Size of Training Labels:", np.shape(y_train))
    print("Size of Test Data:", np.shape(x_test))
    print("Size of Test Labels:", np.shape(y_test))

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.reshape(x_train.shape[0], 1)
    x_test = x_test.reshape(x_test.shape[0], 1)
    input_shape = (1)
    batch_size = 4
    epochs = 1

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(), metrics=['accuracy'])
    print(model.summary())

    model.fit(x_train, y_train, batch_size=4, epochs=10, verbose=1, validation_data=(x_test, y_test))
    return model

