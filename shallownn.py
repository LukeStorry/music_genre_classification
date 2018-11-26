import tensorflow as tf
import librosa
import pickle
import utils

train_set, test_set = utils.load_music()


'''
Hyperparameter flags taken from lab code.

'''

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', os.getcwd() + '/dataset/',
                            'Directory where the dataset will be stored and checkpoint. (default: %(default)s)')
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('save_model', 1000,
                            'Number of steps between model saves (default: %(default)d)')

# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('batch_size', 256, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float('learning_rate', 1e-5, 'Learning rate (default: %(default)d)')
tf.app.flags.DEFINE_integer('img_width', 32, 'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer('img_height', 32, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer('img_channels', 3, 'Image channels (default: %(default)d)')
tf.app.flags.DEFINE_integer('num_classes', 10, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_string('log_dir', '{cwd}/logs/'.format(cwd=os.getcwd()),'Directory where to write event logs and checkpoint. (default: %(default)s)')



xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True)

def shallownn(x_images):
    conv1_cf = tf.layers.conv2d(
        inputs=x_images,
        filters=16,
        kernel_size=[10, 23],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        name='conv1_cf'
    )
    conv1_cf_lr = tf.nn.leaky_relu(conv1_cf, alpha=0.3)
    pool1_cf = tf.layers.max_pooling2d(
            inputs=conv1_cf_lr,
            pool_size=[1, 20],
            strides=1,
            name='pool1_cf'
        )

    flat_pool_cf = tf.reshape(pool1, [-1, 5120])

    conv1_tc = tf.layers.conv2d(
        inputs=x_images,
        filters=16,
        kernel_size=[21, 20],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        name='conv1_tc'
    )
    conv1_tc_lr = tf.nn.leaky_relu(conv1_tc, alpha=0.3)
    pool1_tc = tf.layers.max_pooling2d(
            inputs=conv1_tc_lr,
            pool_size=[20, 1],
            strides=1,
            name='pool1_tc'
        )

    flat_pool_tc = tf.reshape(pool1, [-1, 5120])

    flat_pool_concat = tf.concat(pool1_cf, pool1_tc, 1)

    dropout = tf.layers.dropout(flat_pool_concat, 0.1)

    fc1 = tf.layers.dense(
        flat_pool_concat,
        200,
        activation=None,
        use_bias=True,
        trainable=True,
        kernel_initializer=xavier_initializer,
        bias_initializer=xavier_initializer,
        name='fc1'
    )
