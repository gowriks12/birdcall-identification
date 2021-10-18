import librosa
from matplotlib import pyplot as plt
import IPython.display as ipd
# matplotlib inline
from pydub import AudioSegment
from os import path
import subprocess


# files
src = "XC16964.mp3"
dst = "test.wav"

# convert wav to mp3
# sound = AudioSegment.from_mp3(src)
# sound.export(dst, format="wav")
subprocess.call(['ffmpeg', '-i', src, dst])

sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")
signal, Fs = librosa.load(dst, sr=21952)
print('%s Fs = %d, x.shape = %s, x.dtype = %s' % ('Bird-call plot', Fs, signal.shape, signal.dtype))
plt.figure(figsize=(8, 2))
plt.plot(signal, color='gray')
plt.xlim([0, signal.shape[0]])
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
ipd.display(ipd.Audio(data=signal, rate=Fs))
