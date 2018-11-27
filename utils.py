import librosa
import numpy as np
import pickle


def melspectrogram(audio):
    spec = librosa.stft(audio, n_fft=512, window='hann',
                        hop_length=256, win_length=512, pad_mode='constant')
    mel_basis = librosa.filters.mel(sr=22050, n_fft=512, n_mels=80)
    mel_spec = np.dot(mel_basis, np.abs(spec))
    return np.log(mel_spec + 1e-6).flatten().astype(np.float32, copy=False).tolist()


def load_music():
    print "Loading data..."
    with open('music_genres_dataset.pkl', 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
    print "Music Data Loaded."
    return train_set, test_set
