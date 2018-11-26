import tensorflow as tf

xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True)

def leaky_relu (x, alpha=0.3):
    if x > 0:
        return x
    else:
        return alpha * x
    
def graph(x_images):

    #layer 1
    left_conv_1 = tf.layers.conv2d(
        inputs=x_images,
        filters=16,
        kernel_size=[10, 23],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        name='left_conv'
    )
    right_conv_1 = tf.layers.conv2d(
        inputs=x_images,
        filters=16,
        kernel_size=[21, 20],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        name='right_conv'
    )

    left_conv_1_relu = leaky_relu(left_conv_1, alpha=0.3)
    right_conv_1_relu = leaky_relu(right_conv_1, alpha=0.3)

    left_pooling_1 = tf.layers.max_pooling2d(
            inputs=left_conv_1_relu,
            pool_size=[2, 2],
            strides=1,
            name='left_pooling'
    )

    right_pooling_1 = tf.layers.max_pooling2d(
            inputs=right_conv_1_relu,
            pool_size=[2, 2],
            strides=1,
            name='right_pooling'
    )

    #layer 2
    left_conv_2 = tf.layers.conv2d(
        inputs=left_pooling_1,
        filters=32,
        kernel_size=[5, 11],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        name='left_conv_2'
    )
    right_conv_2 = tf.layers.conv2d(
        inputs=right_pooling_1,
        filters=32,
        kernel_size=[10, 5],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        name='right_conv_2'
    )

    left_conv_2_relu = leaky_relu(left_conv_2, alpha=0.3)
    right_conv_2_relu = leaky_relu(right_conv_2, alpha=0.3)

    left_pooling_2 = tf.layers.max_pooling2d(
            inputs=left_conv_2_relu,
            pool_size=[2, 2],
            strides=1,
            name='left_pooling_2'
    )

    right_pooling_2 = tf.layers.max_pooling2d(
            inputs=right_conv_2_relu,
            pool_size=[2, 2],
            strides=1,
            name='right_pooling_2'
    )

    #layer 3
    left_conv_3 = tf.layers.conv2d(
        inputs=left_pooling_2,
        filters=64,
        kernel_size=[3, 5],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        name='left_conv_3'
    )
    right_conv_3 = tf.layers.conv2d(
        inputs=right_pooling_2,
        filters=64,
        kernel_size=[5, 3],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        name='right_conv_3'
    )

    left_conv_3_relu = leaky_relu(left_conv_3, alpha=0.3)
    right_conv_3_relu = leaky_relu(right_conv_3, alpha=0.3)

    left_pooling_3 = tf.layers.max_pooling2d(
            inputs=left_conv_3_relu,
            pool_size=[2, 2],
            strides=1,
            name='left_pooling_3'
    )

    right_pooling_3 = tf.layers.max_pooling2d(
            inputs=right_conv_3_relu,
            pool_size=[2, 2],
            strides=1,
            name='right_pooling_3'
    )

    #layer4
    left_conv_4 = tf.layers.conv2d(
        inputs=left_pooling_3,
        filters=128,
        kernel_size=[2, 4],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        name='left_conv_4'
    )
    right_conv_4 = tf.layers.conv2d(
        inputs=right_pooling_3,
        filters=128,
        kernel_size=[4, 2],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        name='right_conv_4'
    )

    left_conv_4_relu = leaky_relu(left_conv_4, alpha=0.3)
    right_conv_4_relu = leaky_relu(right_conv_4, alpha=0.3)

    #10x2x128 tensor
    left_pooling_4 = tf.layers.max_pooling2d(
            inputs=left_conv_4_relu,
            pool_size=[1, 5],
            strides=1,
            name='left_pooling_4'
    )

    #2x10x128 tensor
    right_pooling_4 = tf.layers.max_pooling2d(
            inputs=right_conv_4_relu,
            pool_size=[5, 1],
            strides=1,
            name='right_pooling_4'
    )

    left_flattened = tf.reshape(left_pooling_4, [-1, 2560])
    right_flattened = tf.reshape(right_pooling_4, [-1, 2560])

    merged = tf.concat(left_flattened, right_flattened, 1)

    dropout = tf.layers.dropout(merged, 0.25)

    fully_connected_layer = tf.layers.dense(
        inputs=dropout,
        units=200,
        activation=None,
        use_bias=True,
        trainable=True,
        kernel_initializer=xavier_initializer,
        bias_initializer=xavier_initializer,
        name='fully_connected_layer'
    )

    fully_connected_layer_2 = tf.layers.dense(
        inputs=fully_connected_layer,
        units=10,
        activation=None,
        use_bias=True,
        trainable=True,
        kernel_initializer=xavier_initializer,
        bias_initializer=xavier_initializer,
        name='fully_connected_layer_2'
    )
    return fully_connected_layer_2
