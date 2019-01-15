import librosa
import pickle
import numpy as np


def melspectrogram(audio):
    spec = librosa.stft(audio, n_fft=512, window='hann',
                        hop_length=256, win_length=512, pad_mode='constant')
    mel_basis = librosa.filters.mel(sr=22050, n_fft=512, n_mels=80)
    mel_spec = np.dot(mel_basis, np.abs(spec))
    return np.log(mel_spec + 1e-6)


def time_stretch(sample, rate):
    """Stretch audio sample, then either pad or concatenate it"""
    stretched = librosa.effects.time_stretch(sample, rate)
    if len(stretched) >= len(sample):
        return stretched[:len(sample)]
    else:
        return np.pad(stretched, (0, len(sample) - len(stretched)), 'constant')


def pitch_shift(sample, half_steps):
    return librosa.effects.pitch_shift(sample, 22050, half_steps)


def do_augmentation(data):
    """Append extra audio samples to given data, using both stretching and shifting.

    From Schindler's paper:
    For each deformation three segments have been randomly chosen from the audio content.
    The combinations of the two deformations with four different factors each
    resulted thus in 48 additional data instances per audio file."""

    print "Augmenting data..."
    n_tracks = len(set(data['track_id']))
    samples_per_track = data['track_id'].count(0)
    for track_id in range(n_tracks):
        # randomly choose three segments for each track
        for random_choice in np.random.choice(range(samples_per_track), 3):
            segment_index = random_choice + track_id * samples_per_track
            # Apply each time-stretch to the segment
            stretched_segments = [time_stretch(data['data'][segment_index], factor)
                                  for factor in [0.2, 0.5, 1.2, 1.5]]
            # Then apply pitch-shift to all of those
            augmentations = [pitch_shift(segment, factor)
                             for segment in stretched_segments
                             for factor in [-5, -2, 2, 5]]

            # Append the extra samples to the training set
            data['data'] += augmentations
            data['track_id'] += [data['track_id'][segment_index]] * len(augmentations)
            data['labels'] += [data['labels'][segment_index]] * len(augmentations)
    print "  done."


def get(samples, augment=False):
    """Import data, remove unwanted samples, convert to np.array,
        optionally augment data, and calculate melspectrograms"""

    print "Loading data..."
    with open('music_genres_dataset.pkl', 'rb') as f:
        train_set, test_set = pickle.load(f), pickle.load(f)
    train_set['data'] = train_set['data'][:samples]
    train_set['labels'] = train_set['labels'][:samples]
    train_set['track_id'] = train_set['track_id'][:samples]
    test_set['labels'] = np.array(test_set['labels'], copy=False)
    print "  done."

    # Optionally Augment the Data by appending to the training set's lists
    if augment:
        do_augmentation(train_set)

    # Calculate the Melsepctrograms from the audio files in the data lists
    print "Calculating melspectrograms..."
    train_set['melspectrograms'] = np.array(map(melspectrogram, train_set['data']))
    test_set['melspectrograms'] = np.array(map(melspectrogram, test_set['data']))
    print "  done."
    return train_set, test_set
