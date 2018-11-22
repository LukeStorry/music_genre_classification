import tensorflow as tf
import librosa

import pickle

load_music():
    with open('music_genres_dataset.pkl', 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
    return train_set, test_set
