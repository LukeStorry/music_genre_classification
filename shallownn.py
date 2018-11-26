def graph(x_images):
    
    left_conv = tf.layers.conv2d(
        inputs=x_images,
        filters=16,
        kernel_size=[10, 23],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        name='left_conv'
    )
    right_conv = tf.layers.conv2d(
        inputs=x_images,
        filters=16,
        kernel_size=[21, 20],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        name='right_conv'
    )
    
    left_conv_relu = tf.nn.leaky_relu(left_conv, alpha=0.3)
    right_conv_relu = tf.nn.leaky_relu(right_conv, alpha=0.3)

    left_pooling = tf.layers.max_pooling2d(
            inputs=left_conv_1_relu,
            pool_size=[1, 20],
            strides=1,
            name='left_pooling'
    )

    right_pooling = tf.layers.max_pooling2d(
            inputs=conv1_tc_lr,
            pool_size=[20, 1],
            strides=1,
            name='right_pooling'
    )
        
    left_flattened = tf.reshape(left_pooling, [-1, 5120])
    right_flattened = tf.reshape(right_pooling, [-1, 5120])

    merged = tf.concat(left_flattened, right_flattened, 1)

    dropout = tf.layers.dropout(merged, 0.1)

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
