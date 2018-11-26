import tensorflow as tf
import librosa
import pickle
import utils
import shallownn
import deepnn


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 200, 'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log_frequency', 10, 'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
                            

# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('batch_size', 32, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_string('log_dir', '{cwd}/logs/'.format(cwd=os.getcwd()),'Directory where to write event logs and checkpoint. (default: %(default)s)')

# TODO add flags for architecture and accuracy types

run_log_dir = os.path.join(FLAGS.log_dir, 'exp_'.format())



xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True)


def main(_):
    tf.reset_default_graph()

    # Import data
    train_set, test_set = utils.load_music()
    # TODO Pre-process?


    with tf.variable_scope('inputs'):
        # Create the model
        x = tf.placeholder(tf.float32, [None, 80*80]) # output from melspectrogram
        y_ = tf.placeholder(tf.float32, [None, 10]) # ten types of music


    # Build the graph for the deep net
    final_layer = shallownn.graph(x)

    # Define loss function - softmax_cross_entropy
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=final_layer))
    ## Define the AdamOptimiser
    adam = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0.9, beta2=0.999, epsilon=1e-08, name="adam")
    train_step = adam.minimize(cross_entropy)
        
        
    # calculate the prediction and the accuracy
    correct_predictions = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)) # TODO calculate predictions with data
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    loss_summary = tf.summary.scalar('Loss', cross_entropy)
    acc_summary = tf.summary.scalar('Accuracy', accuracy)

    
    # summaries for TensorBoard visualisation
    test_summary = tf.summary.merge([loss_summary, acc_summary])

    # saver for checkpoints
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(run_log_dir + '_train', sess.graph)
        summary_writer_validation = tf.summary.FileWriter(run_log_dir + '_validate', sess.graph)

        sess.run(tf.global_variables_initializer())

        # Training and validation
        for step in range(FLAGS.max_steps):
            # Training: Backpropagation using train set
            (trainImages, trainLabels) = cifar.getTrainBatch()
            (testImages, testLabels) = cifar.getTestBatch()
            
            _, summary_str = sess.run([train_step, training_summary], feed_dict={x: trainImages, y_: trainLabels})

            
            if step % (FLAGS.log_frequency + 1)== 0:
                summary_writer.add_summary(summary_str, step)

            # Validation: Monitoring accuracy using validation set
            if step % FLAGS.log_frequency == 0:
                validation_accuracy, summary_str = sess.run([accuracy, validation_summary], feed_dict={x: testImages, y_: testLabels})
                print('step %d, accuracy on validation batch: %g' % (step, validation_accuracy))
                summary_writer_validation.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % FLAGS.save_model == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(run_log_dir + '_train', 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        # Testing

        # resetting the internal batch indexes
        cifar.reset()
        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0

        # don't loop back when we reach the end of the test set
        while evaluated_images != cifar.nTestSamples:
            (testImages, testLabels) = cifar.getTestBatch(allowSmallerBatches=True)
            test_accuracy_temp, _ = sess.run([accuracy, test_summary], feed_dict={x: testImages, y_: testLabels})

            batch_count = batch_count + 1
            test_accuracy = test_accuracy + test_accuracy_temp
            evaluated_images = evaluated_images + testLabels.shape[0]

        test_accuracy = test_accuracy / batch_count
        print('test set: accuracy on test set: %0.3f' % test_accuracy)



if __name__ == '__main__':
    tf.app.run(main=main)
