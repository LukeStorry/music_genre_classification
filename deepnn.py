import tensorflow as tf
import librosa

import pickle
import utils


def load_music():
    with open('music_genres_dataset.pkl', 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
    return train_set, test_set

train_set, test_set = utils.load_music()
