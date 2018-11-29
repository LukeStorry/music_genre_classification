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
tf.app.flags.DEFINE_integer(
    'log_frequency', 10, 'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_string('log_dir', '{cwd}/logs/'.format(cwd=os.getcwd(
)), 'Directory where to write event logs and checkpoint. (default: %(default)s)')

run_log_dir = os.path.join(FLAGS.log_dir, '{}_{}_{}ep'.format(
    strftime("%Y%d%m_%H%M%S", localtime()), FLAGS.depth, FLAGS.epochs,))


def main(_):
    print "Building", FLAGS.depth, "network..."
    tf.reset_default_graph()

    # Set up the inputs and dataset iterator
    training = tf.placeholder(tf.bool)
    spectrograms_placeholder = tf.placeholder(tf.float32, [None, 80, 80])
    labels_placeholder = tf.placeholder(tf.int64, [None, ])
    dataset = tf.data.Dataset.from_tensor_slices((spectrograms_placeholder, labels_placeholder))

    train_iterator = dataset.shuffle(buffer_size=FLAGS.samples).batch(
        FLAGS.batch_size).make_initializable_iterator()
    test_iterator = dataset.batch(FLAGS.batch_size).make_initializable_iterator()

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

    # count correct predictions and calculate the accuracy
    votes = tf.argmax(logits, 1)
    raw_accuracy = tf.reduce_mean(tf.cast(tf.equal(votes, labels), tf.float32))

    # summaries for TensorBoard visualisation
    loss_summary = tf.summary.scalar('Loss', cross_entropy)
    acc_summary = tf.summary.scalar('Raw Accuracy', raw_accuracy)
    la_summary = tf.summary.merge([loss_summary, acc_summary])

    print "  done "

    # Import data
    print "Loading data..."
    with open('music_genres_dataset.pkl', 'rb') as f:
        train_set, test_set = pickle.load(f), pickle.load(f)
    print "  done."

    # TODO augmentation here to extend t_s['data']

    print "Calculating melspectrograms..."
    train_set['melspectrograms'] = np.array(
        map(utils.melspectrogram, train_set['data'][:FLAGS.samples]), copy=False)
    train_set['labels'] = np.array(train_set['labels'][:FLAGS.samples], copy=False)
    test_set['melspectrograms'] = map(utils.melspectrogram, test_set['data'])
    print "  done."

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(run_log_dir + '_train', sess.graph)
        summary_writer_validation = tf.summary.FileWriter(run_log_dir + '_validate', sess.graph)

        sess.run(tf.global_variables_initializer())
        # get pre-shuffled list of indices for validation batch
        validation_indices = range(len(train_set['melspectrograms']))
        np.random.shuffle(validation_indices)
        validation_indices = validation_indices[:FLAGS.batch_size]

        # Training & Validation by epoch
        for epoch in range(FLAGS.epochs):
            sess.run(train_iterator.initializer, feed_dict={
                spectrograms_placeholder: train_set['melspectrograms'],
                labels_placeholder: train_set['labels']})

            # Training continues until end of epoch - https://www.tensorflow.org/guide/datasets#consuming_values_from_an_iterator
            while True:
                try:
                    sess.run(train_step, feed_dict={training: True})
                except tf.errors.OutOfRangeError:
                    break

            # Validation with same pre-made selection of train set
            sess.run(test_iterator.initializer, feed_dict={
                spectrograms_placeholder: train_set['melspectrograms'][validation_indices],
                labels_placeholder: train_set['labels'][validation_indices]})

            val_accuracy, val_summary = sess.run(
                [raw_accuracy, la_summary], feed_dict={training: False})

            print('epoch %d, accuracy on validation batch: %g' % (epoch, val_accuracy))
            summary_writer.add_summary(val_summary, epoch)

        # Testing
        n_tracks = 1000
        n_labels = 10
        actual_track_genres = [x[1] for x in
                               sorted(set(zip(test_set['track_id'], test_set['labels'])))]

        test_raw_accuracy = 0.0
        test_sum_probabilities = [np.array([0.0 for _ in range(n_labels)])
                                  for _ in range(n_tracks)]
        test_votes = [[0 for _ in range(n_labels)] for _ in range(n_tracks)]

        sess.run(test_iterator.initializer, feed_dict={
            spectrograms_placeholder: test_set['melspectrograms'],
            labels_placeholder: test_set['labels']})

        batch_count = 0
        while True:
            try:
                batch_accuracy, batch_probs, batch_vote = sess.run([raw_accuracy, logits, votes],
                                                                   feed_dict={training: False})
            except tf.errors.OutOfRangeError:
                break

            test_raw_accuracy += batch_accuracy
            for sample, track_id in enumerate(test_set['track_id'][batch_count:batch_count + FLAGS.batch_size]):
                test_sum_probabilities[track_id] += batch_probs[sample]
                test_votes[track_id][batch_vote[sample]] += 1
                if sample == len(batch_probs)-1:
                    break

            batch_count += 1

        correct_with_probs = 0.0
        correct_with_votes = 0.0
        for genre, probabilites, votes in zip(actual_track_genres, test_sum_probabilities, test_votes):
            correct_with_probs += 1.0 if np.argmax(probabilites) == genre else 0.0
            correct_with_votes += 1.0 if np.argmax(votes) == genre else 0.0

        # correct_with_probs_2 = np.sum(np.equal(
        #     actual_track_genres, np.argmax(test_sum_probabilities, 1)))
        # correct_with_votes_2 = np.sum(np.equal(
        #     actual_track_genres, np.argmax(test_votes, 1)))

        print 'Raw Probability accuracy on test set: %0.3f' % (test_raw_accuracy / batch_count)
        print 'Maximum Probability accuracy on test set: %0.3f' % (correct_with_probs / n_tracks)
        print 'Majority Vote accuracy on test set: %0.3f' % (correct_with_votes / n_tracks)
        print 'Maximum Probability2 accuracy on test set: %0.3f' % (correct_with_probs_2 / n_tracks)
        print 'Majority Vote accuracy2 on test set: %0.3f' % (correct_with_votes_2 / n_tracks)


if __name__ == '__main__':
    tf.app.run()
