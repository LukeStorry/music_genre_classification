import tensorflow as tf
import librosa

import pickle
import utils

<<<<<<< HEAD
def load_music():
    with open('music_genres_dataset.pkl', 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
    return train_set, test_set
=======
train_set, test_set = utils.load_music()
>>>>>>> 6a9fbcd9808f37b888fd68545db5eecb736d839f
