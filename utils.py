import librosa
import numpy as np
import pickle


def melspectrogram(audio):
    spec = librosa.stft(audio, n_fft=512, window='hann',
                        hop_length=256, win_length=512, pad_mode='constant')
    mel_basis = librosa.filters.mel(sr=22050, n_fft=512, n_mels=80)
    mel_spec = np.dot(mel_basis, np.abs(spec))
    return np.log(mel_spec + 1e-6)


def load_music():
    print("Loading data...")
    with open('music_genres_dataset.pkl', 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
    print("Music Data Loaded.")
    return train_set, test_set


def print_data_layout_and_types():
    a,b = load_music()
    audio = a['data'][0]
    mel =  melspectrogram(audio)

    print type(audio)
    print audio.shape
    print audio

    print mel
    print type(mel)
    print mel.shape


def test_melspectrogram_speeds():
    import timeit
    train_set, _ = load_music()
    train_data = np.array(train_set['data'])
    train_indices = range(len(train_set['data']))

    def f1():
            np.random.shuffle(train_indices)  # shuffle training every epoch
            for i in range(0, 1000, 16):
                a = map(melspectrogram, train_data[train_indices][i:i + 16])

    def f2():
        np.random.shuffle(train_indices)  # shuffle training every epoch
        m = np.array(map(melspectrogram, train_data))
        for i in range(0, 1000, 16):
            a = m[train_indices][i:i + 16]

    print timeit.timeit(f1, number=1) # > 129s
    print timeit.timeit(f2, number=1) # > 132s
