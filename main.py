import os
import tensorflow as tf
import numpy as np
import pickle
from time import strftime, localtime

import utils
import shallownn
import deepnn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'depth', 'shallow', 'Whether to run the "deep" or "shallow" network. (default: %(default)s)')
tf.app.flags.DEFINE_integer(
    'epochs', 100, 'Number of epochs to run. (default: %(default)d)')
tf.app.flags.DEFINE_integer(
    'samples', 11250, 'How many training samples to use (default: %(default)d)')
tf.app.flags.DEFINE_boolean(
    'augment', False, 'Whether to do Data-Augmentation. (default: %(default)d)')
tf.app.flags.DEFINE_integer(
    'log_frequency', 10, 'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_string('log_dir', '{cwd}/logs/'.format(cwd=os.getcwd(
)), 'Directory where to write event logs and checkpoint. (default: %(default)s)')

run_log_dir = os.path.join(FLAGS.log_dir, '{}_{}_{}ep'.format(
    strftime("%Y%d%m_%H%M%S", localtime()), FLAGS.depth, FLAGS.epochs,))


def main(_):
    print "Building", FLAGS.depth, "network, with", FLAGS.samples, "samples,",
    print("with" if FLAGS.augment else "without"), "augmentation"
    tf.reset_default_graph()

    # Set up the inputs and Dataset object
    training = tf.placeholder(tf.bool)
    spectrograms_placeholder = tf.placeholder(tf.float32, [None, 80, 80])
    labels_placeholder = tf.placeholder(tf.int64, [None, ])
    dataset = tf.data.Dataset.from_tensor_slices((spectrograms_placeholder, labels_placeholder))

    # Set up the two iterators to shuffle the data and split into batches
    train_iterator = dataset.shuffle(buffer_size=FLAGS.samples).batch(
        FLAGS.batch_size).make_initializable_iterator()
    test_iterator = dataset.batch(FLAGS.batch_size).make_initializable_iterator()

    # Set the graph up to use the correct iterator (we don't want to shuffle the test data)
    spectrograms, labels = tf.cond(training, train_iterator.get_next, test_iterator.get_next)

    # Build the graph for the shallow network
    if FLAGS.depth == 'shallow':
        graph = shallownn.graph
    elif FLAGS.depth == 'deep':
        graph = deepnn.graph
    logits = graph(tf.reshape(spectrograms, [-1, 80, 80, 1]), training)

    # Define loss function, optimiser and training step
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    adam = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0.9,
                                  beta2=0.999, epsilon=1e-08, name="adam")
    train_step = adam.minimize(cross_entropy)

    # Calculate max predictions and calculate the accuracy
    votes = tf.argmax(logits, 1)
    raw_accuracy = tf.reduce_mean(tf.cast(tf.equal(votes, labels), tf.float32))

    # summaries for TensorBoard visualisation
    loss_summary = tf.summary.scalar('Loss', cross_entropy)
    acc_summary = tf.summary.scalar('Raw Accuracy', raw_accuracy)
    la_summary = tf.summary.merge([loss_summary, acc_summary])

    # log to console if the graph has been set up without errors
    print "  done "

    # Import data, shorten if not all samples wanted, and convert to np.array
    print "Loading data..."
    with open('music_genres_dataset.pkl', 'rb') as f:
        train_set, test_set = pickle.load(f), pickle.load(f)
    train_set['data'] = train_set['data'][:FLAGS.samples]
    train_set['labels'] = train_set['labels'][:FLAGS.samples]
    train_set['track_id'] = train_set['track_id'][:FLAGS.samples]
    test_set['labels'] = np.array(test_set['labels'], copy=False)
    print "  done."

    # Optionally Augment the Data by appending to the training set's lists
    if FLAGS.augment:
        print "Augmenting data..."
        n_tracks = len(set(train_set['track_id']))
        samples_per_track=train_set['track_id'].count(0)
        for track_id in range(n_tracks):
            # randomly choose three segments for each track
            for random_choice in np.random.choice(range(samples_per_track), 3):
                segment_index = random_choice + track_id * samples_per_track
                # Apply each time-stretch to the segment
                stretched_segments = [utils.time_stretch(train_set['data'][segment_index], factor)
                                for factor in [0.2, 0.5, 1.2, 1.5]]
                # Then apply pitch-shift to all of those
                augmentations = [utils.pitch_shift(segment, factor)
                                    for segment in stretched_segments
                                    for factor in [-5, -2, 2, 5]]

                # Append the extra samples to the training set
                train_set['data'] += augmentations
                train_set['track_id'] += [train_set['track_id'][segment_index]] *  len(augmentations)
                train_set['labels'] += [train_set['labels'][segment_index]]*  len(augmentations)
        print "  done."

    # Calculate the Melsepctrograms from the audio files in the data lists
    print "Calculating melspectrograms..."
    train_set['melspectrograms'] = np.array(map(utils.melspectrogram, train_set['data']))
    test_set['melspectrograms'] = np.array(map(utils.melspectrogram, test_set['data']))
    print "  done."

    # Now everything has been set up, we can launch a Tensorflow Session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(run_log_dir + '_train', sess.graph)
        summary_writer_validation = tf.summary.FileWriter(run_log_dir + '_validate', sess.graph)

        # Create a shuffled list of indices to define the validation batch
        validation_indices = range(len(test_set['melspectrograms']))
        np.random.shuffle(validation_indices)
        validation_indices = validation_indices[:FLAGS.batch_size]

        # Train & Validate by epoch
        for epoch in range(FLAGS.epochs):
            sess.run(train_iterator.initializer, feed_dict={
                spectrograms_placeholder: train_set['melspectrograms'],
                labels_placeholder: train_set['labels']})

            # Repeat the Train Step until all Data used (end of epoch) - https://www.tensorflow.org/guide/datasets#consuming_values_from_an_iterator
            while True:
                try:
                    sess.run(train_step, feed_dict={training: True})
                except tf.errors.OutOfRangeError:
                    break

            # Validation with the pre-selected indices of train set
            sess.run(test_iterator.initializer, feed_dict={
                spectrograms_placeholder: test_set['melspectrograms'][validation_indices],
                labels_placeholder: test_set['labels'][validation_indices]})

            val_accuracy, val_summary = sess.run([raw_accuracy, la_summary],
                                                 feed_dict={training: False})

            summary_writer.add_summary(val_summary, epoch)
            print('epoch %d, accuracy on validation batch: %g' % (epoch, val_accuracy))


        # Testing after the Training has finished
        actual_track_genres = [x[1]
                               for x in sorted(set(zip(test_set['track_id'], test_set['labels'])))]
        n_tracks = len(actual_track_genres)
        n_labels = max(actual_track_genres) + 1

        test_raw_accuracy = 0.0
        test_probabilities = [np.array([0.0 for _ in range(n_labels)])
                              for _ in range(n_tracks)]
        test_votes = [[0 for _ in range(n_labels)] for _ in range(n_tracks)]

        # Set up the Training Iterator with the Training Set data
        sess.run(test_iterator.initializer, feed_dict={
            spectrograms_placeholder: test_set['melspectrograms'],
            labels_placeholder: test_set['labels']})

        batch_count = 0
        # Repeatedly run the Accuracy and Voting calculations, caching the values at the end of each batch
        while True:
            try:
                batch_accuracy, batch_probabilities, batch_vote = sess.run([raw_accuracy, logits, votes],
                                feed_dict={training: False})

            # If we have used all the data, exit the loop.
            except tf.errors.OutOfRangeError:
                break

            test_raw_accuracy += batch_accuracy
            # Loop through each sample in the batch and increment the corresponding probability and votes lists
            batch_index = FLAGS.batch_size * batch_count
            for sample, track_id in enumerate(test_set['track_id'][batch_index: batch_index + FLAGS.batch_size]):
                test_probabilities[track_id] += batch_probabilities[sample]
                test_votes[track_id][batch_vote[sample]] += 1
                if sample == len(batch_probabilities) - 1:
                    break
            batch_count += 1

        # Count the total amount of correct predictions
        correct_with_probs = np.sum(np.equal(actual_track_genres, np.argmax(test_probabilities, 1)))
        correct_with_votes = np.sum(np.equal(actual_track_genres, np.argmax(test_votes, 1)))

        print 'Raw Probability accuracy on test set: %0.3f' % ( float(test_raw_accuracy) / batch_count)
        print 'Maximum Probability accuracy on test set: %0.3f' % ( float(correct_with_probs) / n_tracks)
        print 'Majority Vote accuracy on test set: %0.3f' % ( float(correct_with_votes) / n_tracks)


if __name__ == '__main__':
    tf.app.run()
