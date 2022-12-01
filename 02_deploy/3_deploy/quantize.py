import sys
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from Handler.DataHandler import get_generator, calib_input
from tensorflow_model_optimization.quantization.keras import vitis_quantize
import json
import cv2
import numpy as np
from pathlib import Path



def calib_input2(path, calib_batch_size):
    images = []
    config = json.load(open(path))

    src_path = Path(path).parent

    for img_path in config["images"]:
        image = cv2.imread(str(src_path / img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image / 255.0
        images.append(image)
    images = np.array(images)
    return images


def quantiz_keras_model(model_dir, quant_model_dir, batchsize, data_dir, eval_data_path, evaluate, verbose=False):

    os.makedirs(os.path.split(quant_model_dir)[0], exist_ok=True)

    # load the floating point trained model
    float_model = tf.keras.models.load_model(model_dir)


    quant_dataset = calib_input2(data_dir, batchsize)

    config = float_model.get_config() # Returns pretty much every information about your model
    input_shape = (config["layers"][0]["config"]["batch_input_shape"][1], config["layers"][0]["config"]["batch_input_shape"][2]) # returns a tuple of width, height and channels

    width = float_model.input_shape[2]
    if verbose is True:
        print('Keras model information:')
        print('-------------------------------------')
        print(' Input names :', float_model.inputs)
        print(' Output names:', float_model.outputs)
        print(' Input shape: ', input_shape, '(automatically calculated!)')
        print(' Dataset shape: ', quant_dataset.shape)
        print('-------------------------------------')
        
        #float_model.summary()

    quantizer = vitis_quantize.VitisQuantizer(float_model)
#    quantizer = vitis_quantize.VitisQuantizer(float_model, quantize_strategy='8bit')
    quantized_model = quantizer.quantize_model(
        calib_dataset=quant_dataset,
#        include_cle=True,
#        cle_steps=10,
#        include_fast_ft=False,
#        fold_conv_bn=False,
#        calib_batch_size=1,
        verbose=1
        )

#    quantizer = vitis_quantize.VitisQuantizer(float_model, quantize_strategy='8bit')
#
#    qat_model = quantizer.get_qat_model(
#        init_quant=True,
#        # Do init PTQ quantization will help us to get a better initial state for the quantizers, especially for `8bit_tqt` strategy. Must be used together with calib_dataset
#        calib_dataset=generator_trainset)
#
#    quantizer.quantize_model(
#        calib_dataset=quant_dataset,
#        calib_batch_size=32,
#        #calib_steps=None,
#        verbose=1,
#        fold_bn=True,
#        replace_sigmoid=True,
#        replace_relu6=True,
#        include_cle=True,
#        cle_steps=10,
#        forced_cle=False,
#        include_fast_ft=False,
#        fast_ft_epochs=10)
#
#    #vitis_quantize.VitisQuantizer.quantize_model(
#    # calib_dataset=None,
#    # fold_conv_bn=True,
#    # fold_bn=True,
#    # replace_relu6=True,
#    # include_cle=True,
#    # cle_steps=10)

    quantized_model.save(quant_model_dir + '/q_model.h5')
    print('Saved quantized model to', quant_model_dir)


#    if verbose is True:
#        quantized_model.summary()

    if evaluate is True:
        print('\n-------------------------------------')
        print('Evaluating Quantized model..')
        print('-------------------------------------\n')
        quantized_model.compile(optimizer=tf.keras.optimizers.Adam(),
                                    loss='categorical_crossentropy',
                                    metrics=['acc'])
        generator_trainset = get_generator(eval_data_path, batchsize, input_shape)
        scores = quantized_model.evaluate(generator_trainset, verbose=1)
        print('Float model accuracy: {0:.4f}'.format(scores[1] * 100), '%')



def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument('--model',
                    type=str,
                    default='model.hdf5',
                    help='Model to be quantize.')
    ap.add_argument('--q_model',
                    type=str,
                    default='q_model.hdf5',
                    help='Path to save the quantize model-')
    ap.add_argument('-b', '--batchsize',
                    type=int,
                    default=32,
                    help='Evaluation batchsize, must be integer value. Default is 32')
    ap.add_argument('-d', '--datapath',
                    type=str,
                    default='data/',
                    help='The path to the data.')
    ap.add_argument('-eval', '--eval_data',
                    type=str,
                    default='data/',
                    help='The path to the evaluation data.')
    ap.add_argument('-v', '--verbose',
                    default=False,
                    action='store_true',
                    help='The path to the data.')
    ap.add_argument('-e', '--evaluate',
                    default=False,
                    action='store_true',
                    help='Evaluate the quanized model.')

    args = ap.parse_args()  

    print('\n------------------------------------')
    print('TensorFlow version : ', tf.__version__)
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --model     : ', args.model)
    print (' --q_model   : ', args.q_model)
    print (' --batchsize : ', args.batchsize)
    print (' --datapath  : ', args.datapath)
    print (' --eval_data : ', args.eval_data)
    print (' --verbose   : ', args.verbose)
    print (' --evaluate  : ', args.evaluate)
    print('------------------------------------\n')

    quantiz_keras_model(args.model, args.q_model, args.batchsize, args.datapath, args.eval_data, args.evaluate, args.verbose)

if __name__ == "__main__":
    main()
