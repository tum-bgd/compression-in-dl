import tensorflow as tf


def bn_relu_conv(x,filters,kernel=1,strides=1):
    """Convolutional block

    Args:
        x (tf.Keras.Layer): Previous Keras layer
        filters (int): The dimensionality of the output space (i.e. the number of output filters in the convolution).
        kernel (int, optional): An integer, specifying the kernel of the convolutional layer.
        strides (int, optional): An integer, specifying the strides of the convolution along the height and width. Defaults to 1.

    Returns:
        tf.Keras.Layer: The model with a convolutional block at the top.
    """

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    x = tf.keras.layers.Conv2D(filters, kernel, strides=strides, padding='same')(x)
    return x


def dense_block(x,filters,strides=1, bottleneck=True):
    """Dense block

    Args:
        x (tf.Keras.Layer): Previous Keras layer
        filters (int): The dimensionality of the output space (i.e. the number of output filters in the convolution).
        strides (int, optional): An integer, specifying the strides of the dense along the height and width. Defaults to 1.
        bottleneck (boolean): An boolean to specify whether using a bottleneck.

    Returns:
        tf.Keras.Layer: The model with a dense block at the top.
    """

    if bottleneck is True:
        skip = tf.keras.layers.BatchNormalization()(x)
        skip = tf.keras.layers.Activation(tf.nn.relu)(skip)
        skip = tf.keras.layers.Conv2D(filters * 4, kernel_size=(1,1), strides=strides, padding='valid', use_bias=False)(skip)
    else:
        skip = x

    skip = tf.keras.layers.BatchNormalization()(skip)
    skip = tf.keras.layers.Activation(tf.nn.relu)(skip)
    skip = tf.keras.layers.Conv2D(filters, kernel_size=(3,3), strides=strides, padding='same', use_bias=False)(skip)

    out = tf.keras.layers.concatenate([skip,x])

    return out


def transition_layer(x, filters,strides=1, compression=0.5):
    """ Transition layer block.

    Args:
        x (tf.Keras.Layer): Previous Keras layer
        filters (int): The dimensionality of the output space (i.e. the number of output filters in the convolution).
        strides (int, optional): An integer, specifying the strides of the dense along the height and width. Defaults to 1.
        compression (float, optional): A floating point number, specifying the compression factor in this block.

    Returns:
        tf.Keras.Layer: The model with a transition layer block at the top.
    """

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    x = tf.keras.layers.Conv2D((x.shape[-1])*compression, kernel_size=(1,1), strides=strides, padding='same', use_bias=False)(x) 

    x = tf.keras.layers.AvgPool2D(2, strides = 2, padding = 'same')(x)

    return x


def densenet121_vitis(input_tensor=None, include_top=True, weight_path=None, return_tensor=False, classes=1000, classifier_activation="softmax", grow_rate=32, compression=0.5, kernel_size_first_layer=(7,7)):
    """Creates and returns the DenseNet121 CNN architecture.

    Args:
        input_tensor: optional keras layer, like an input tensor.
        include_top: whether to include the top layers or top.
        weight_path: If not none, these weights will be loaded.
        return_tensor: Whether to return the network as tensor or as `tf.keras.model` (if true, weights will not be loaded).
        classes: By default the number of classes are 1000 (ImageNet). Only important `include_top=True`.
        classifier_activation: By default softmax (ImageNet). Only important `include_top=True`.
        grow_rate: Graw rate of the network, by default 32.
        compression: Compression factor of the transition layer blocks, by default 0.5
        kernel_size_first_layer: Kernel size of the first layer, by default (7,7)

    Returns:
        The CNN architecture as `tf.keras.model` if `return_tensor=False`, otherwise as `tf.keras.layers`.
    """

    if input_tensor is None:
        input_tensor = tf.keras.layers.Input(shape=(224,224,3))
    
    x = tf.keras.layers.ZeroPadding2D()(input_tensor)
    x = tf.keras.layers.Conv2D(grow_rate*2, kernel_size=kernel_size_first_layer, strides=2, padding='valid', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)


    # Dense Block 1
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate) # 5
    x = dense_block(x, grow_rate)

    # Transition Layer 1
    x = transition_layer(x, grow_rate, compression=compression)

    # Dense Block 2
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate) # 5
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate) # 10
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)

    # Transition Layer 2
    x = transition_layer(x, grow_rate, compression=compression)

    # Dense Block 3
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate) # 5
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate) # 10
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate) # 15
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate) # 20
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)

    # Transition Layer 3
    x = transition_layer(x, grow_rate, compression=compression)

    # Dense Block 3
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate) # 5
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate) # 10
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate)
    x = dense_block(x, grow_rate) # 15
    x = dense_block(x, grow_rate)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    
    if include_top is True:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(classes, activation=classifier_activation, name="predictions")(x)

    if return_tensor:
        return x
    
    model = tf.keras.Model(input_tensor, x, name="densenet121")
    if weight_path is not None:
        model.load_weights(weight_path)

    return model