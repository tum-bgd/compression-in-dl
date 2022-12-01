from sre_parse import Verbose
import sys
import os
import argparse
import tensorflow as tf
from Handler.DataHandler import get_generator
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def evaluate_graph(model_dir, data_dir, batchsize, verbose=False):
    # load the floating point trained model
    model = tf.keras.models.load_model(model_dir)

    config = model.get_config() # Returns pretty much every information about your model
    input_shape = (config["layers"][0]["config"]["batch_input_shape"][1], config["layers"][0]["config"]["batch_input_shape"][2]) # returns a tuple of width, height and channels

    if verbose is True:
        print('Keras model information:')
        print('-------------------------------------')
        print(' Input names :', model.inputs)
        print(' Output names:', model.outputs)
        print(' Input shape: ', input_shape, '(automatically calculated!)')
        print('-------------------------------------')

        # model.summary()

    generator_trainset = get_generator(data_dir, batchsize, input_shape)

    print('\n-------------------------------------')
    print('Evaluating model...')
    print('-------------------------------------\n')
    scores = model.evaluate(generator_trainset, verbose=1)
    print('Model accuracy: {0:.4f}'.format(scores[1] * 100), '%')





def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument('--model',
                    type=str,
                    default='model.hdf5',
                    help='Model to be evaluated.')
    ap.add_argument('-b', '--batchsize',
                    type=int,
                    default=32,
                    help='Evaluation batchsize, must be integer value. Default is 32')
    ap.add_argument('-d', '--datapath',
                    type=str,
                    default='data/',
                    help='The path to the data.')
    ap.add_argument('-v', '--verbose',
                    default=False,
                    action='store_true',
                    help='The path to the data.')

    args = ap.parse_args()  

    print('\n------------------------------------')
    print('TensorFlow version : ', tf.__version__)
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --model     : ', args.model)
    print (' --batchsize : ', args.batchsize)
    print (' --datapath  : ', args.datapath)
    print (' --verbose   : ', args.verbose)
    print('------------------------------------\n')

    evaluate_graph(args.model, args.datapath, args.batchsize, args.verbose)

if __name__ == "__main__":
    main()
