import tensorflow as tf


xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True)


def leaky_relu(x, alpha=0.3):
    return tf.maximum(x, alpha * x)


def graph(x, is_training):

    left_conv = tf.layers.conv2d(
        inputs=x,
        filters=16,
        kernel_size=[10, 23],
        padding='same',
        activation=leaky_relu,
        use_bias=False,  # TODO is this not meant to be True?
        kernel_initializer=xavier_initializer,
        name='left_conv'
    )
    right_conv = tf.layers.conv2d(
        inputs=x,
        filters=16,
        kernel_size=[21, 20],
        padding='same',
        activation=leaky_relu,
        use_bias=False,  # TODO is this not meant to be True?
        kernel_initializer=xavier_initializer,
        name='right_conv'
    )
    left_pooling = tf.layers.max_pooling2d(
        inputs=left_conv,
        pool_size=[1, 20],
        strides=[1, 20],  # is this meant to match pool_size?
        name='left_pooling'
    )
    right_pooling = tf.layers.max_pooling2d(
        inputs=right_conv,
        strides=[20, 1],
        pool_size=[20, 1],
        name='right_pooling'
    )

    left_flattened = tf.reshape(left_pooling, [-1, 5120])
    right_flattened = tf.reshape(right_pooling, [-1, 5120])

    merged = tf.concat([left_flattened, right_flattened], 1)

    dropout = tf.layers.dropout(merged, 0.1, training=is_training, name='dropout')

    fully_connected_layer = tf.layers.dense(
        inputs=dropout,
        units=200,
        activation=leaky_relu,
        use_bias=True,
        trainable=True,
        kernel_initializer=xavier_initializer,
        bias_initializer=xavier_initializer,
        name='fully_connected_layer'
    )

    final_layer = tf.layers.dense(
        inputs=fully_connected_layer,
        units=10,
        activation=None,
        use_bias=True,
        trainable=True,
        kernel_initializer=xavier_initializer,
        bias_initializer=xavier_initializer,
        name='final_layer'
    )
    return final_layer
