import os
import time
import tensorflow as tf
import numpy as np

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
tf.app.flags.DEFINE_integer(
    'log_frequency', 10, 'Number of steps between logging results to the console and saving summaries (default: %(default)d)')


# Optimisation hyperparameters
tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_string('log_dir', '{cwd}/logs/'.format(cwd=os.getcwd(
)), 'Directory where to write event logs and checkpoint. (default: %(default)s)')

run_log_dir = os.path.join(FLAGS.log_dir, 'exp_{}_{}_{}'.format(
    FLAGS.depth, FLAGS.epochs, int(time.time())))


def main(_):
    print "Building ", FLAGS.depth, "network."
    tf.reset_default_graph()

    # Create the model
    with tf.variable_scope('inputs'):
        training = tf.placeholder(tf.bool)
        spectrograms = tf.placeholder(tf.float32, [None, 80, 80])
        spectro_reshaped = tf.reshape(spectrograms, [-1, 80, 80, 1])
        labels = tf.placeholder(tf.int64, [None, ])  # ten types of music

    # Build the graph for the shallow network
    with tf.variable_scope('model'):
        if FLAGS.depth == 'shallow':
            logits = shallownn.graph(spectro_reshaped, training)
        elif FLAGS.depth == 'deep':
            logits = deepnn.graph(spectro_reshaped, training)
        else:
            print "Unknown depth flag:", FLAGS.depth
            return -1

    # Define loss function, optimiser and training step
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    adam = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0.9,
                                  beta2=0.999, epsilon=1e-08, name="adam")
    train_step = adam.minimize(cross_entropy)

    # count correct predictions and calculate the accuracy
    votes = tf.argmax(logits, 1)
    raw_accuracy = tf.reduce_mean(tf.cast(tf.equal(votes, labels), tf.float32))

    # summaries for TensorBoard visualisation
    loss_summary = tf.summary.scalar('Loss', cross_entropy)
    acc_summary = tf.summary.scalar('Raw Accuracy', raw_accuracy)
    la_summary = tf.summary.merge([loss_summary, acc_summary])

    # saver for checkpoints
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    # Import data
    train_set, test_set = utils.load_music()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(run_log_dir + '_train', sess.graph)
        summary_writer_validation = tf.summary.FileWriter(run_log_dir + '_validate', sess.graph)

        sess.run(tf.global_variables_initializer())

        # Set up training lists to be selectable by shuffled list of indices
        train_labels = np.array(train_set['labels'])[:FLAGS.samples]
        train_data = np.array(train_set['data'])[:FLAGS.samples]
        train_indices = range(len(train_labels))
        # get shuffled list of indices for validation set
        np.random.shuffle(train_indices)
        validation_indices = train_indices[:FLAGS.batch_size]

        # Training & Validation by epoch
        for epoch in range(FLAGS.epochs):
            np.random.shuffle(train_indices)  # shuffle training every epoch

            # Training loop by batches
            for i in range(0, len(train_labels), FLAGS.batch_size):
                train_batch_spectrograms = map(utils.melspectrogram,
                                               train_data[train_indices][i:i + FLAGS.batch_size])
                train_batch_labels = train_labels[train_indices][i:i + FLAGS.batch_size]
                sess.run(train_step, feed_dict={training: True,
                                                spectrograms: train_batch_spectrograms,
                                                labels: train_batch_labels})

            # Validation with same pre-made selection of train set
            v_batch_spectrograms = map(utils.melspectrogram,
                                       train_data[validation_indices])
            v_batch_labels = train_labels[validation_indices]
            val_accuracy, val_summary = sess.run([raw_accuracy, la_summary], feed_dict={
                                                 training: False,
                                                 spectrograms: v_batch_spectrograms,
                                                 labels: v_batch_labels})

            print('epoch %d, accuracy on validation batch: %g' % (epoch, val_accuracy))
            summary_writer.add_summary(val_summary, epoch)

        # Testing
        n_tracks = max(test_set['track_id']) + 1
        actual_track_genres = [None for _ in range(n_tracks)]
        batch_count = 0
        test_raw_accuracy = 0.0
        test_sum_probabilities = [np.array([0.0 for _ in range(10)]) for _ in range(n_tracks)]
        test_votes = [[0 for _ in range(10)] for _ in range(n_tracks)]

        for i in range(0, len(test_set['labels']), FLAGS.batch_size):
            test_batch_spectrograms = map(utils.melspectrogram,
                                          test_set['data'][i:i + FLAGS.batch_size])
            test_batch_labels = test_set['labels'][i:i + FLAGS.batch_size]
            batch_accuracy, batch_probs, batch_vote = sess.run([raw_accuracy, logits, votes],
                                                               feed_dict={training: False,
                                                                          spectrograms: test_batch_spectrograms,
                                                                          labels: test_batch_labels})

            test_raw_accuracy += batch_accuracy
            batch_count += 1

            for x, track_id in enumerate(test_set['track_id'][i:i + FLAGS.batch_size]):
                test_sum_probabilities[track_id] += batch_probs[x]
                test_votes[track_id][batch_vote[x]] += 1
                actual_track_genres[track_id] = test_batch_labels[x]

        correct_with_probs = 0.0
        correct_with_votes = 0.0
        for genre, probabilites, votes in zip(actual_track_genres, test_sum_probabilities, test_votes):
            correct_with_probs += 1.0 if np.argmax(probabilites) == genre else 0.0
            correct_with_votes += 1.0 if np.argmax(votes) == genre else 0.0

        print 'Raw Probability accuracy on test set: %0.3f' % (test_raw_accuracy / batch_count)
        print 'Maximum Probability accuracy on test set: %0.3f' % (correct_with_probs / n_tracks)
        print 'Majority Vote accuracy on test set: %0.3f' % (correct_with_votes / n_tracks)


if __name__ == '__main__':
    tf.app.run()
