from ast import Str
import tensorflow as tf

def residual_block(x, filters, stride=1):
    """ResNet residual block / building block. Used for ResNet 18 and 34.

    Args:
        x (tf.Keras.Layer): Previous Keras layer
        filters (int): The dimensionality of the output space (i.e. the number of output filters in the convolution).
        stride (int, optional): An integer, specifying the strides of the convolution along the height and width. Defaults to 1.

    Returns:
        tf.Keras.Layer: The model with an residual block at the top. 
    """

    skip = tf.keras.layers.Conv2D(filters, kernel_size=(3,3), strides=(stride,stride), padding='same')(x)
    skip = tf.keras.layers.BatchNormalization()(skip)
    skip = tf.keras.layers.Activation(tf.nn.relu)(skip)

    skip = tf.keras.layers.Conv2D(filters, kernel_size=(3,3), padding='same')(skip)
    skip = tf.keras.layers.BatchNormalization()(skip)

    # Option B - See paper Deep Residual Learning for Image Recognition
    if stride != 1:
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size=(1,1), strides=(stride,stride), padding='valid')(x)
    
    out = tf.keras.layers.Add()([x,skip])
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation(tf.nn.relu)(out)

    return out

def bottleneck_block(x, filters, stride=1, option_b=False):
    """ResNet bottleneck building block. Used for ResNet 50, 101 and 152. 

    Args:
        x (tf.Keras.Layer): Previous Keras layer
        filters (int): The dimensionality of the output space (i.e. the number of output filters in the convolution).
        stride (int, optional): An integer, specifying the strides of the convolution along the height and width. Defaults to 1.

    Returns:
        tf.Keras.Layer: The model with an bottleneck building block at the top. 
    """

    skip = tf.keras.layers.Conv2D(filters, kernel_size=(1,1), strides=(stride,stride), padding='same')(x)
    skip = tf.keras.layers.BatchNormalization()(skip)
    skip = tf.keras.layers.Activation(tf.nn.relu)(skip)

    skip = tf.keras.layers.Conv2D(filters, kernel_size=(3,3), padding='same')(skip)
    skip = tf.keras.layers.BatchNormalization()(skip)
    skip = tf.keras.layers.Activation(tf.nn.relu)(skip)

    skip = tf.keras.layers.Conv2D(filters*4, kernel_size=(1,1), padding='same')(skip)
    skip = tf.keras.layers.BatchNormalization()(skip)

    # Option B - See paper Deep Residual Learning for Image Recognition
    if option_b:
        x = tf.keras.layers.Conv2D(filters*4, kernel_size=(1,1), strides=(stride,stride), padding='valid')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)

    out = tf.keras.layers.Add()([x,skip])
    out = tf.keras.layers.Activation(tf.nn.relu)(out)
    
    return out


def resnet152_vitis(input_tensor=None, include_top=True, weight_path=None, return_tensor=False, classes=1000, classifier_activation="softmax"):
    """Creates and returns the ResNet152 CNN architecture.

    Args:
        input_tensor: optional keras layer, like an input tensor.
        include_top: whether to include the top layers or top.
        weight_path: If not none, these weights will be loaded.
        return_tensor: Whether to return the network as tensor or as `tf.keras.model` (if true, weights will not be loaded).
        classes: By default the number of classes are 1000 (ImageNet). Only important `include_top=True`.
        classifier_activation: By default softmax (ImageNet). Only important if `include_top=True`.

    Returns:
        The CNN architecture as `tf.keras.model` if `return_tensor=False`, otherwise as `tf.keras.layers`.
    """

    if input_tensor is None:
        input_tensor = tf.keras.layers.Input(shape=(224,224,3))

    x = tf.keras.layers.ZeroPadding2D()(input_tensor)
    x = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)

    # conv2_x
    x = bottleneck_block(x, 64, option_b=True)
    x = bottleneck_block(x, 64)
    x = bottleneck_block(x, 64)

    # conv3_x
    x = bottleneck_block(x, 128, 2, option_b=True)
    x = bottleneck_block(x, 128)
    x = bottleneck_block(x, 128)
    x = bottleneck_block(x, 128)
    x = bottleneck_block(x, 128) # 5
    x = bottleneck_block(x, 128)
    x = bottleneck_block(x, 128)
    x = bottleneck_block(x, 128)

    # conv4_x
    x = bottleneck_block(x, 256, 2, option_b=True)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256) # 5
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256) # 10
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256) # 15
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256) # 20
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256) # 25
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256) # 30
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256) # 35
    x = bottleneck_block(x, 256)

    # conv5_x
    x = bottleneck_block(x, 512, 2, option_b=True)
    x = bottleneck_block(x, 512)
    x = bottleneck_block(x, 512)
    
    if include_top is True:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(classes, activation=classifier_activation, name="predictions")(x)

    if return_tensor:
        return x
    
    model = tf.keras.Model(input_tensor, x, name="resnet152")
    if weight_path is not None:
        model.load_weights(weight_path)

    return model
