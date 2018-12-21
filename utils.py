import librosa
import numpy as np


def melspectrogram(audio):
    spec = librosa.stft(audio, n_fft=512, window='hann',
                        hop_length=256, win_length=512, pad_mode='constant')
    mel_basis = librosa.filters.mel(sr=22050, n_fft=512, n_mels=80)
    mel_spec = np.dot(mel_basis, np.abs(spec))
    return np.log(mel_spec + 1e-6)

def time_stretch(sample, rate):
    stretched = librosa.effects.time_stretch(sample, rate)
    if len(stretched) >= 20462:
        return stretched[:20462]
    else:
        return np.pad(stretched, (0, 20462-len(stretched)), 'constant')

def pitch_shift(sample, half_steps):
    return librosa.effects.pitch_shift(sample, 22050, half_steps)
