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
    'epochs', 10, 'Number of epochs to run. (default: %(default)d)')
tf.app.flags.DEFINE_integer(
    'log_frequency', 10, 'Number of steps between logging results to the console and saving summaries (default: %(default)d)')


# Optimisation hyperparameters
tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_string('log_dir', '{cwd}/logs/'.format(cwd=os.getcwd(
)), 'Directory where to write event logs and checkpoint. (default: %(default)s)')

# TODO add flags for architecture and accuracy types

run_log_dir = os.path.join(FLAGS.log_dir, 'exp_{}_{}_{}'.format(
    FLAGS.depth, FLAGS.epochs, int(time.time())))


def main(_):
    tf.reset_default_graph()

    # Create the model
    with tf.variable_scope('inputs'):
        training = tf.placeholder(tf.bool)
        x = tf.placeholder(tf.float32, [None, 80, 80])
        x_reshaped = tf.reshape(x, [-1, 80, 80, 1])
        y = tf.placeholder(tf.int64, [None, ])  # ten types of music

    # Build the graph for the shallow network
    with tf.variable_scope('model'):
        if FLAGS.depth == 'shallow':
            final_layer = shallownn.graph(x_reshaped, training)
        elif FLAGS.depth == 'deep':
            final_layer = deepnn.graph(x_reshaped, training)
        else:
            print "Unknown depth flag:", FLAGS.depth
            return

    # Define loss function - softmax_cross_entropy
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=final_layer))

    # Define the AdamOptimiser
    adam = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0.9,
                                  beta2=0.999, epsilon=1e-08, name="adam")
    train_step = adam.minimize(cross_entropy)

    # count correct predictions and calculate the accuracy
    correct_predictions = tf.equal(tf.argmax(final_layer, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # summaries for TensorBoard visualisation
    loss_summary = tf.summary.scalar('Loss', cross_entropy)
    acc_summary = tf.summary.scalar('Accuracy', accuracy)
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
        validation_indices = range(len(train_set['data']))
        np.random.shuffle(validation_indices)  # only shuffle validation once

        # Training & Validation by epoch
        for epoch in range(FLAGS.epochs):
            np.random.shuffle(train_indices)  # shuffle training every epoch

            # Training loop by batches
            for i in range(0, len(train_labels), FLAGS.batch_size):
                train_batch_labels = train_labels[train_indices][i:i + FLAGS.batch_size]
                train_batch_spectrograms = map(utils.melspectrogram,
                                               train_data[train_indices][i:i + FLAGS.batch_size])

                sess.run(train_step, feed_dict={training: True,
                                                x: train_batch_spectrograms,
                                                y: train_batch_labels})

            # Validation with same pre-made selection of train set
            v_batch_labels = train_labels[validation_indices][:FLAGS.batch_size]
            v_batch_spectrograms = map(utils.melspectrogram,
                                       train_data[validation_indices][:FLAGS.batch_size])
            val_accuracy, val_summary_str = sess.run([accuracy, la_summary],
                                                     feed_dict={training: False,
                                                                x:  v_batch_spectrograms,
                                                                y: v_batch_labels})
            print('epoch %d, accuracy on validation batch: %g' % (epoch, val_accuracy))
            summary_writer.add_summary(val_summary_str, epoch)

        # Testing
        batch_count = 0
        test_accuracy = 0
        for i in range(0, len(test_set['labels']), FLAGS.batch_size):
            test_batch_labels = test_set['labels'][i:i + FLAGS.batch_size]
            test_batch_spectrograms = map(utils.melspectrogram,
                                          test_set['data'][i:i + FLAGS.batch_size])
            test_batch_accuracy = sess.run(accuracy, feed_dict={training: False,
                                                                x: test_batch_spectrograms,
                                                                y: test_batch_labels})
            batch_count += 1
            test_accuracy += test_batch_accuracy

        test_accuracy /= batch_count

        print('test set: accuracy on test set: %0.3f' % test_accuracy)


if __name__ == '__main__':
    tf.app.run()
