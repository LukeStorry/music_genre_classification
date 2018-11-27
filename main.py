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
    'log_frequency', 10, 'Number of steps between logging results to the console and saving summaries (default: %(default)d)')


# Optimisation hyperparameters
tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_string('log_dir', '{cwd}/logs/'.format(cwd=os.getcwd(
)), 'Directory where to write event logs and checkpoint. (default: %(default)s)')

run_log_dir = os.path.join(FLAGS.log_dir, 'exp_{}_{}_{}'.format(
    FLAGS.depth, FLAGS.epochs, int(time.time())))


def main(_):
    tf.reset_default_graph()

    # Create the model
    with tf.variable_scope('inputs'):
        training = tf.placeholder(tf.bool)
        spectrograms = tf.reshape(tf.placeholder(tf.float32, [None, 80, 80]), [-1, 80, 80, 1])
        labels = tf.placeholder(tf.int64, [None, ])  # ten types of music

    # Build the graph for the shallow network
    with tf.variable_scope('model'):
        if FLAGS.depth == 'shallow':
            logits = shallownn.graph(spectrograms, training)
        elif FLAGS.depth == 'deep':
            logits = deepnn.graph(spectrograms, training)
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
    correct_predictions = tf.equal(tf.argmax(logits, 1), labels)
    raw_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    # TODO add the other accuracy metrics

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
        train_labels = np.array(train_set['labels'])
        train_data = np.array(train_set['data'])
        train_indices = range(len(train_set['data']))
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
            val_accuracy, val_summary = sess.run([accuracy, la_summary], feed_dict={
                                                 training: False,
                                                 spectrograms: train_data[validation_indices],
                                                 labels: train_labels[validation_indices]})

            print('epoch %d, accuracy on validation batch: %g' % (epoch, val_accuracy))
            summary_writer.add_summary(val_summary, epoch)

        # Testing
        batch_count = 0
        test_accuracy = 0
        for i in range(0, len(test_set['labels']), FLAGS.batch_size):
            test_batch_spectrograms = map(utils.melspectrogram,
                                             test_set['data'][i:i + FLAGS.batch_size])
            test_batch_labels = test_set['labels'][i:i + FLAGS.batch_size]
            test_batch_accuracy = sess.run(accuracy, feed_dict={training: False,
                                                                spectrograms: test_batch_spectrograms,
                                                                labels: test_batch_labels})
            batch_count += 1
            test_accuracy += test_batch_accuracy

        test_accuracy /= batch_count

        print('test set: accuracy on test set: %0.3f' % test_accuracy)


if __name__ == '__main__':
    tf.app.run()
