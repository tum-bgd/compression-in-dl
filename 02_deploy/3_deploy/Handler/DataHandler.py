import tensorflow as tf
import cv2
import numpy as np


def get_generator(path, batch_size, img_dest_size):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    generator_train = datagen.flow_from_directory(
        path,
        batch_size=batch_size,
        target_size=img_dest_size,
        class_mode='categorical',
    )

    return generator_train


def calib_input(path, calib_batch_size):
    images = []
    line = open(path).readlines()
    for index in range(0, calib_batch_size):
        curline = line[calib_batch_size + index]
        calib_image_name = curline.strip()

        # open image as BG
        image = cv2.imread(calib_image_name)

        # change to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # normalize
        image = image / 255.0
        images.append(image)
    images = np.array(images)
    return images  # ToDo: Hotfix, needs to be changed