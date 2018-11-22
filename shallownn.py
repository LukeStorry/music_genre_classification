import tensorflow as tf
import librosa
import pickle
import utils

train_set, test_set = utils.load_music()

xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True)

def shallownn(x_images):
    conv1 = tf.layers.conv2d(
        inputs=x_images,
        filters=16,
        kernel_size=[10, 23],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        name='conv1'
    )
    conv1_lr = tf.nn.leaky_relu(conv1, alpha=0.3)
    pool1 = tf.layers.max_pooling2d(
            inputs=conv1_lr,
            pool_size=[1, 20],
            strides=1,
            name='pool1'
        )

    flat_pool = tf.reshape(pool1, [-1, 5120])
