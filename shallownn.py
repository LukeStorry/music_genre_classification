import tensorflow as tf
import librosa
import pickle
import utils

train_set, test_set = utils.load_music()

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
        bias_initializer=tf.zeros_initializer(),
        name='fc1'
    )
