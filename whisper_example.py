# Whisper Example

# For Mac
# brew install ffmpeg

# For Windows
# choco install ffmpeg
# https://chocolatey.org/

# For Linux
# sudo apt install ffmpeg

## Convert audio: ffmpeg -i audio_test.m4a audio_test.mp3

import os
os.system("ffmpeg -i audio_test.m4a audio_test.mp3")

os.system("ffmpeg -i bilingual_test.m4a bilingual_test.mp3")

import matplotlib.pyplot as plt

import whisper

base_model = whisper.load_model('base')

#base_result = base_model.transcribe('/Users/sberry5/bilingual_test.mp3')
base_result = base_model.transcribe('/Users/sberry5/Documents/audio_test.mp3')

print(base_result['text'])

big_model = whisper.load_model('medium')

# This is 1.42 gigs!
#result = big_model.transcribe('/Users/sberry5/bilingual_test.mp3')
result = big_model.transcribe('/Users/sberry5/Documents/audio_test.mp3')

print(result['text'])

#audio = whisper.load_audio("/Users/sethberry/bilingual_test.mp3")
audio = whisper.load_audio("/Users/sberry5/Documents/audio_test.mp3")
audio = whisper.pad_or_trim(audio)

mel = whisper.log_mel_spectrogram(audio).to(big_model.device)

mel_numpy = mel.numpy()

plt.specgram(mel_numpy, Fs=6, cmap="rainbow")

plt.show()

# Checking Language and Transcribing:
_, probs = big_model.detect_language(mel)

probs
max(probs, key=probs.get)

options = whisper.DecodingOptions(fp16 = False)

result = whisper.decode(big_model, mel, options)

result.text

os.system('whisper /Users/sethberry/bilingual_test.mp3 --language Spanish --task translate --model medium')

import matplotlib.pyplot as plt
import numpy as np
import wave
wav_obj = wave.open('/Users/sethberry/Documents/audio_test.wav', 'rb')
sample_freq = wav_obj.getframerate()
n_samples = wav_obj.getnframes()
t_audio = n_samples/sample_freq
n_channels = wav_obj.getnchannels()
signal_wave = wav_obj.readframes(n_samples)
signal_array = np.frombuffer(signal_wave, dtype=np.int16)
times = np.linspace(0, n_samples/sample_freq, num=n_samples)
plt.figure(figsize=(15, 5))
plt.plot(times, signal_array)
plt.xlim(0, t_audio)
plt.show()


plt.figure(figsize=(15, 5))
plt.specgram(signal_array, Fs=sample_freq, vmin=-20, vmax=50)
plt.title('Left Channel')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.xlim(0, t_audio)
plt.colorbar()
plt.show()
