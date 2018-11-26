import tensorflow as tf
import utils
import shallownn
import deepnn
import os


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_epochs', 200, 'Number of epochs to run. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log_frequency', 10, 'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
                            

# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('batch_size', 32, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_string('log_dir', '{cwd}/logs/'.format(cwd=os.getcwd()),'Directory where to write event logs and checkpoint. (default: %(default)s)')

# TODO add flags for architecture and accuracy types

run_log_dir = os.path.join(FLAGS.log_dir, 'exp_'.format())



xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True)


def main(_):
    tf.reset_default_graph()

    # Import & pre-process data
    train_set, test_set = utils.load_music()
    train_indices = np.arange(len(train_set['labels']))
    
    
    with tf.variable_scope('inputs'):
        # Create the model
        x = tf.placeholder(tf.float32, [None, 80*80]) # output from melspectrogram
        y_ = tf.placeholder(tf.float32, [None, 1]) # ten types of music

    # Build the graph for the shallow network
    final_layer = shallownn.graph(x)

    # Define loss function - softmax_cross_entropy
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=final_layer))
    
    # Define the AdamOptimiser
    adam = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0.9, beta2=0.999, epsilon=1e-08, name="adam")
    train_step = adam.minimize(cross_entropy)
        
    # count correct predictions and calculate the accuracy
    correct_predictions = tf.equal(tf.argmax(final_layer, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    # summaries for TensorBoard visualisation
    loss_summary = tf.summary.scalar('Loss', cross_entropy)
    acc_summary = tf.summary.scalar('Accuracy', accuracy)
    la_summary = tf.summary.merge([loss_summary, acc_summary])

    # saver for checkpoints
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(run_log_dir + '_train', sess.graph)
        summary_writer_validation = tf.summary.FileWriter(run_log_dir + '_validate', sess.graph)

        sess.run(tf.global_variables_initializer())

        # Training and validation
        for epoch in range(FLAGS.max_epochs):
            
            # shuffle data
            np.random.shuffle(train_indices)
            train_data = train_set['data'][train_indices]
            train_labels = train_set['labels'][train_indices]
            
            # training loop by batches
            for i in range(0, len(train_labels), batch_size):
                spectrograms = map(utils.melspectrogram, train_data[i:i + batch_size])
                labels = train_labels[i:i + batch_size]]
                sess.run(train_step, feed_dict={x: spectrograms,y_: labels]})
                
            # validation
            validation_accuracy, validation_summary_str = sess.run([accuracy, la_summary], feed_dict={
                x: train_spectrograms[:batch_size],
                y_: train_labels[:batch_size]})

            summary_writer.add_summary(validation_summary_str, epoch)
            print('epoch %d, accuracy on validation batch: %g' % (epoch, validation_accuracy))
                

        # Testing

        batch_count = 0
        test_accuracy = 0

        for i in range(0, len(test_set['labels']), batch_size):
            test_spectrograms = map(utils.melspectrogram, test_set['data'][i:i + batch_size])
            testLabels = test_set['labels'][i:i + batch_size]

            test_accuracy_temp = sess.run(accuracy, feed_dict={x: test_spectrograms, y_: testLabels})

            batch_count = batch_count + 1
            test_accuracy = test_accuracy + test_accuracy_temp
        test_accuracy = test_accuracy / batch_count
        
        print('test set: accuracy on test set: %0.3f' % test_accuracy)



if __name__ == '__main__':
    tf.app.run(main=main)
