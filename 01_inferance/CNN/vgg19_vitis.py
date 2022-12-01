import tensorflow as tf

def vgg19_vitis(input_tensor=None, include_top=True, weight_path=None, return_tensor=False, classes=1000, classifier_activation="softmax"):
    """Creates and returns the VGG19 CNN architecture.

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
    
    x = tf.keras.layers.Conv2D(filters =64, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(input_tensor)
    x = tf.keras.layers.Conv2D(filters =64, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    x = tf.keras.layers.Conv2D(filters =128, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters =128, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    x = tf.keras.layers.Conv2D(filters =256, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters =256, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters =256, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters =256, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    x = tf.keras.layers.Conv2D(filters =512, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters =512, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters =512, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters =512, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    x = tf.keras.layers.Conv2D(filters =512, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters =512, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters =512, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters =512, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    if include_top is True:
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        x = tf.keras.layers.Dense(classes, activation=classifier_activation, name="predictions")(x)

    if return_tensor:
        return x
    
    model = tf.keras.Model(input_tensor, x, name="vgg19")
    if weight_path is not None:
        model.load_weights(weight_path)

    return model
