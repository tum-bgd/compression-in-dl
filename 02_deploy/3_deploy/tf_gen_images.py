'''
 Copyright 2020 Xilinx Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

'''
Creates JPEG image files from Keras dataset in numpy array format
'''

import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from Handler.DataHandler import get_generator
import random
from pathlib import Path
import cv2
import numpy as np


import json
from glob import glob


def gen_images(image_dir, output_file, max_images, batchsize, data_path, image_dest):
#    generator_trainset = get_generator(data_path, batchsize, image_dest)
#    file_names = generator_trainset.filenames
#    random.shuffle(file_names)

    if output_file.split(".")[-1] != "json":
        print("\nOutput file must be a *.json!")
        exit()

    config = {}

    file_names = []
    for files in ('*.png', '*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.bmp'):
        file_names.extend(glob(data_path + "/*/" + files))
    random.shuffle(file_names)

    classes = [Path(item).name for item in glob(data_path + "/*")]
    
    print(f"Found {len(file_names)} images belonging to {len(classes)} classes.")

    
    config["class_labels"] = sorted(classes, key=str.lower)

    calib_list = {}

    tf.compat.v1.disable_eager_execution()
    array_img = tf.compat.v1.placeholder(tf.uint8)
    op = tf.io.encode_jpeg(array_img, format='rgb', quality=100)
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init)

        for i in range(len(file_names[:max_images])):
            path_src = Path(file_names[i])
            path_dest = Path(image_dir) / Path(file_names[i]).name

            f_name = Path(image_dir).name + "/" + path_src.name
            f_class = path_src.parent.name

            calib_list[str(f_name)] = f_class

            # Image preprocessing
            img = cv2.imread(str(path_src))
            img = cv2.resize(img, image_dest)

            jpg_tensor = sess.run(op, feed_dict={array_img: img})

            with open(path_dest, 'wb') as fd:
                fd.write(jpg_tensor)

    config["images"] = calib_list

    with open(Path(image_dir).parent / output_file, 'w') as f:
        json.dump(config, f)
            
    print('Calib images generated')
    return


# only used if script is run as 'main' from command line
def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-dir', '--image_dir',
                    type=str,
                    default='images',
                    help='Path to folder for saving images and images list file. Default is images')
    ap.add_argument('-l', '--calib_list',
                    type=str,
                    default='',
                    help='Name of images list file. Default is empty string so that no file is generated.')
    ap.add_argument('-m', '--max_images',
                    type=int,
                    default=1,
                    help='Number of images to generate. Default is 1')

    ap.add_argument('-b', '--batchsize',
                    type=int,
                    default=1,
                    help='Evaluation batchsize, must be integer value. Default is 1')
    ap.add_argument('-p', '--data_path',
                    type=str,
                    default='../../_data/eurosat-rgb_split/',
                    help='The path to the dataset.')
    ap.add_argument('-x', '--input_height',
                    type=int,
                    default=10,
                    help='Height of the image')
    ap.add_argument('-y', '--input_width',
                    type=int,
                    default=10,
                    help='width of the image.')
    args = ap.parse_args()

    print('Command line options:')
    print(' --image_dir    : ', args.image_dir)
    print(' --calib_list   : ', args.calib_list)
    print(' --max_images   : ', args.max_images)
    print(' --data_path  : ', args.data_path)
    print(' --input_height  : ', args.input_height)
    print(' --input_width  : ', args.input_width)

    gen_images(args.image_dir,
               args.calib_list,
               args.max_images,
               args.batchsize,
               args.data_path,
               (args.input_height, args.input_width)
               )


if __name__ == '__main__':
    main()
