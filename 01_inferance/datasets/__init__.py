import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import inspect
import importlib
import tensorflow as tf
from functools import partial

def fallback_data_generators(path, batch_size, img_dest_size):
    return fs_data_generators(path, batch_size, img_dest_size)


# +---------------------+
# |  FILESYSTEM         |
# +---------------------+
def fs_data_generators(path, batch_size, img_dest_size):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)

    generator_train = datagen.flow_from_directory(
        directory= "./%s/train/" %(path),
        batch_size=batch_size,
        target_size=img_dest_size,
        color_mode="rgb",  # number of channels ("grayscale", "rgb", "rgba". Default: "rgb")
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    generator_valid = datagen.flow_from_directory(
        directory= "./%s/val_0/" %(path),
        batch_size=batch_size,
        target_size=img_dest_size,
        color_mode="rgb",  # number of channels ("grayscale", "rgb", "rgba". Default: "rgb")
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    generator_test = datagen.flow_from_directory(
        directory= "./%s/val_1/" %(path),
        batch_size=batch_size,
        target_size=img_dest_size,
        color_mode="rgb",  # number of channels ("grayscale", "rgb", "rgba". Default: "rgb")
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    return generator_train, generator_valid, generator_test
